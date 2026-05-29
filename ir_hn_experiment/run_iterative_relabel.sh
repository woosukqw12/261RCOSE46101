#!/usr/bin/env bash
# ============================================================================
# Iterative self-relabeling experiment — fiqa_rlhn on local GPU (RTX 5090)
# ----------------------------------------------------------------------------
# Each round:
#   1. Train model on current train data (with epoch logging)
#   2. Compute dynamics signals from logs → mp3+ criterion
#   3. Build relabeled training data for next round
#   4. Evaluate checkpoint on BEIR (fiqa test)
#   5. Log: nDCG@10, |flags|, overlap with GT FN, overlap with prev flags
#
# Resumable: re-run skips rounds whose checkpoint already exists.
#
# Usage:
#   bash run_iterative_relabel.sh             # 4 rounds (default)
#   ROUNDS=6 bash run_iterative_relabel.sh    # 6 rounds
# ============================================================================

set -euo pipefail

REPO=${REPO:-$(pwd)}
cd "$REPO"

# ---- Config ----------------------------------------------------------------
DATASET=fiqa_rlhn
DATA=data/processed/$DATASET
BASE_OUT=experiments/$DATASET/iterative
LOGS_DIR=logs/iterative_relabel
mkdir -p "$BASE_OUT" "$LOGS_DIR"

ROUNDS=${ROUNDS:-10}
EPOCHS=${EPOCHS:-5}
BS=${BS:-16}
LR=${LR:-2e-5}
NUM_NEG=${NUM_NEG:-7}
SEED=${SEED:-42}
CRITERION=${CRITERION:-margin_persistent_3plus}

CONDA_ENV=${CONDA_ENV:-nlp}
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV"

BEIR_EVAL_DATASETS="fiqa"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOGS_DIR/master.log"; }

# ---- Setup round 0 (baseline) — use symlinks for ckpt/logs, COPY train ----
log "Setting up round 0 (baseline)"
R0="$BASE_OUT/round_0"
mkdir -p "$R0"
[ ! -e "$R0/ckpt" ] && ln -sfn "$(realpath experiments/$DATASET/ckpt_baseline)" "$R0/ckpt"
[ ! -e "$R0/logs" ] && ln -sfn "$(realpath experiments/$DATASET/logs_baseline)" "$R0/logs"
# DO NOT symlink train.jsonl — compute_flags_and_relabel writes to output_dir/train.jsonl
[ ! -f "$R0/train.jsonl" ] && cp "$DATA/train.jsonl" "$R0/train.jsonl"

# ---- Setup round 1 (existing mp3+ relabel) --------------------------------
log "Setting up round 1 (existing mp3+ relabel)"
R1="$BASE_OUT/round_1"
mkdir -p "$R1"
[ ! -e "$R1/ckpt" ] && ln -sfn "$(realpath experiments/$DATASET/ckpt_relabel_mp3)" "$R1/ckpt"
[ ! -e "$R1/logs" ] && ln -sfn "$(realpath experiments/$DATASET/logs_relabel_mp3)" "$R1/logs"
[ ! -f "$R1/train.jsonl" ] && cp "$DATA/train_relabeled_margin_persistent_3plus.jsonl" "$R1/train.jsonl"

# ---- Helper: compute flags from a log dir ---------------------------------
compute_flags_and_relabel() {
    local round_num="$1"
    local log_dir="$2"
    local output_dir="$3"

    log "  Computing signals for round $round_num from $log_dir"
    local sig_dir="$output_dir/signals"
    mkdir -p "$sig_dir"
    python src/compute_signals.py \
        --log_dir "$log_dir" \
        --output_dir "$sig_dir" \
        --source loss 2>&1 | tee "$LOGS_DIR/round${round_num}_signals.log"

    # Extract flags, build relabeled train, compute overlap stats
    python - "$sig_dir/criteria_loss.json" "$DATA/train.jsonl" "$output_dir" "$CRITERION" "$round_num" "$BASE_OUT" "$DATA/fn_ground_truth.json" <<'PYEOF'
import json, sys, os
from pathlib import Path

criteria_path = sys.argv[1]
orig_train_path = sys.argv[2]
output_dir = sys.argv[3]
criterion = sys.argv[4]
round_num = int(sys.argv[5])
base_out = sys.argv[6]
gt_path = sys.argv[7]

with open(criteria_path) as f:
    crit = json.load(f)
pairs = crit.get(criterion, [])
print(f"Round {round_num} flags ({criterion}): {len(pairs)} pairs")

# Save flags
with open(os.path.join(output_dir, "flags.json"), "w") as f:
    json.dump({"criterion": criterion, "pairs": pairs, "count": len(pairs)}, f, indent=2)

# Build relabel map
relabel_map = {}
for qid, ni in pairs:
    relabel_map.setdefault(qid, []).append(ni)

# Build relabeled training data from ORIGINAL train (not previous round's)
rows = []
with open(orig_train_path) as f:
    for line in f:
        rows.append(json.loads(line))

augmented = []
n_added = 0
for row in rows:
    augmented.append(row)
    qid = row["qid"]
    if qid not in relabel_map:
        continue
    negs = row["negatives"]
    neg_ids = row.get("neg_ids", list(range(len(negs))))
    for ni in relabel_map[qid]:
        if ni >= len(negs):
            continue
        new_row = {
            "qid": qid,
            "query": row["query"],
            "positives": [negs[ni]],
            "pos_ids": [neg_ids[ni]],
            "negatives": [n for i, n in enumerate(negs) if i != ni],
            "neg_ids": [n for i, n in enumerate(neg_ids) if i != ni],
        }
        augmented.append(new_row)
        n_added += 1

out_train = os.path.join(output_dir, "train.jsonl")
with open(out_train, "w") as f:
    for r in augmented:
        f.write(json.dumps(r) + "\n")
print(f"  Original: {len(rows)}, Added: {n_added}, Total: {len(augmented)}")

# Compute overlap stats
current_set = set((qid, ni) for qid, ni in pairs)

# vs GT FN
with open(gt_path) as f:
    gt = json.load(f)
idx_to_qid = {}
with open(orig_train_path) as f:
    for i, line in enumerate(f):
        idx_to_qid[i] = json.loads(line)["qid"]
fn_set = set()
for qi_str, neg_indices in gt["fn_pairs"].items():
    qid = idx_to_qid.get(int(qi_str))
    if qid:
        for ni in neg_indices:
            fn_set.add((qid, ni))

tp_gt = len(current_set & fn_set)
prec_gt = tp_gt / max(len(current_set), 1)
recall_gt = tp_gt / max(len(fn_set), 1)

# vs previous round flags
prev_set = set()
prev_flags_path = os.path.join(base_out, f"round_{round_num - 1}", "flags.json")
if os.path.exists(prev_flags_path):
    with open(prev_flags_path) as f:
        prev = json.load(f)
    prev_set = set(tuple(p) for p in prev["pairs"])

if prev_set:
    jaccard = len(current_set & prev_set) / max(len(current_set | prev_set), 1)
    overlap = len(current_set & prev_set)
    new_flags = len(current_set - prev_set)
    dropped = len(prev_set - current_set)
else:
    jaccard = 0.0
    overlap = 0
    new_flags = len(current_set)
    dropped = 0

stats = {
    "round": round_num,
    "n_flags": len(pairs),
    "n_augmented_instances": n_added,
    "total_train_rows": len(augmented),
    "gt_fn_overlap": {
        "tp": tp_gt, "precision": prec_gt, "recall": recall_gt,
        "gt_fn_total": len(fn_set),
    },
    "vs_prev_round": {
        "jaccard": jaccard, "overlap": overlap,
        "new_flags": new_flags, "dropped": dropped,
        "prev_n_flags": len(prev_set),
    },
}
with open(os.path.join(output_dir, "flag_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
for k, v in stats.items():
    print(f"  {k}: {v}")
PYEOF
}

# ---- Compute round-0 flags (for overlap tracking only, no relabeling) ------
if [ ! -f "$R0/flags.json" ]; then
    log "Computing round 0 flags (baseline dynamics)"
    mkdir -p "$R0/signals"
    python src/compute_signals.py \
        --log_dir "$R0/logs" \
        --output_dir "$R0/signals" \
        --source loss 2>&1 | tee "$LOGS_DIR/round0_signals.log"
    # Extract flags and stats only (no train.jsonl generation)
    python - "$R0/signals/criteria_loss.json" "$DATA/train.jsonl" "$R0" "$CRITERION" 0 "$BASE_OUT" "$DATA/fn_ground_truth.json" <<'PY_FLAGS_ONLY'
import json, sys, os
criteria_path, orig_train_path, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
criterion, round_num = sys.argv[4], int(sys.argv[5])
base_out, gt_path = sys.argv[6], sys.argv[7]

with open(criteria_path) as f:
    crit = json.load(f)
pairs = crit.get(criterion, [])
print(f"Round {round_num} flags ({criterion}): {len(pairs)} pairs")

with open(os.path.join(output_dir, "flags.json"), "w") as f:
    json.dump({"criterion": criterion, "pairs": pairs, "count": len(pairs)}, f, indent=2)

current_set = set((qid, ni) for qid, ni in pairs)

with open(gt_path) as f:
    gt = json.load(f)
idx_to_qid = {}
with open(orig_train_path) as f:
    for i, line in enumerate(f):
        idx_to_qid[i] = json.loads(line)["qid"]
fn_set = set()
for qi_str, neg_indices in gt["fn_pairs"].items():
    qid = idx_to_qid.get(int(qi_str))
    if qid:
        for ni in neg_indices:
            fn_set.add((qid, ni))

tp_gt = len(current_set & fn_set)
prec_gt = tp_gt / max(len(current_set), 1)
recall_gt = tp_gt / max(len(fn_set), 1)
stats = {
    "round": round_num, "n_flags": len(pairs),
    "gt_fn_overlap": {"tp": tp_gt, "precision": prec_gt, "recall": recall_gt, "gt_fn_total": len(fn_set)},
    "vs_prev_round": {"jaccard": 0, "overlap": 0, "new_flags": len(pairs), "dropped": 0, "prev_n_flags": 0},
}
with open(os.path.join(output_dir, "flag_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
for k, v in stats.items():
    print(f"  {k}: {v}")
PY_FLAGS_ONLY
fi

# ---- Compute round-1 flags from round-1 logs (for tracking) ---------------
if [ ! -f "$R1/flags.json" ]; then
    log "Computing round 1 flags (relabel_mp3 dynamics)"
    # Only compute signals+flags, don't overwrite train.jsonl
    local_sig="$R1/signals"
    mkdir -p "$local_sig"
    python src/compute_signals.py \
        --log_dir "$R1/logs" \
        --output_dir "$local_sig" \
        --source loss 2>&1 | tee "$LOGS_DIR/round1_signals.log"

    python - "$local_sig/criteria_loss.json" "$DATA/train.jsonl" "$R1" "$CRITERION" 1 "$BASE_OUT" "$DATA/fn_ground_truth.json" <<'PYEOF2'
import json, sys, os

criteria_path, orig_train_path, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
criterion, round_num = sys.argv[4], int(sys.argv[5])
base_out, gt_path = sys.argv[6], sys.argv[7]

with open(criteria_path) as f:
    crit = json.load(f)
pairs = crit.get(criterion, [])
print(f"Round {round_num} flags ({criterion}): {len(pairs)} pairs")

with open(os.path.join(output_dir, "flags.json"), "w") as f:
    json.dump({"criterion": criterion, "pairs": pairs, "count": len(pairs)}, f, indent=2)

current_set = set((qid, ni) for qid, ni in pairs)

with open(gt_path) as f:
    gt = json.load(f)
idx_to_qid = {}
with open(orig_train_path) as f:
    for i, line in enumerate(f):
        idx_to_qid[i] = json.loads(line)["qid"]
fn_set = set()
for qi_str, neg_indices in gt["fn_pairs"].items():
    qid = idx_to_qid.get(int(qi_str))
    if qid:
        for ni in neg_indices:
            fn_set.add((qid, ni))

tp_gt = len(current_set & fn_set)
prec_gt = tp_gt / max(len(current_set), 1)
recall_gt = tp_gt / max(len(fn_set), 1)

prev_flags_path = os.path.join(base_out, "round_0", "flags.json")
prev_set = set()
if os.path.exists(prev_flags_path):
    with open(prev_flags_path) as f:
        prev = json.load(f)
    prev_set = set(tuple(p) for p in prev["pairs"])

jaccard = len(current_set & prev_set) / max(len(current_set | prev_set), 1) if prev_set else 0
stats = {
    "round": round_num, "n_flags": len(pairs),
    "gt_fn_overlap": {"tp": tp_gt, "precision": prec_gt, "recall": recall_gt, "gt_fn_total": len(fn_set)},
    "vs_prev_round": {
        "jaccard": jaccard,
        "overlap": len(current_set & prev_set) if prev_set else 0,
        "new_flags": len(current_set - prev_set) if prev_set else len(current_set),
        "dropped": len(prev_set - current_set) if prev_set else 0,
        "prev_n_flags": len(prev_set),
    },
}
with open(os.path.join(output_dir, "flag_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
for k, v in stats.items():
    print(f"  {k}: {v}")
PYEOF2
fi

# ---- Main loop: rounds 2..ROUNDS ------------------------------------------
for (( r=2; r<=ROUNDS; r++ )); do
    RD="$BASE_OUT/round_$r"
    PREV_RD="$BASE_OUT/round_$((r-1))"
    mkdir -p "$RD"

    log "========== ROUND $r / $ROUNDS =========="

    # Step 1: Compute flags from previous round's logs → build this round's train
    if [ ! -f "$RD/train.jsonl" ]; then
        compute_flags_and_relabel "$r" "$PREV_RD/logs" "$RD"
    else
        log "  skip: $RD/train.jsonl exists"
    fi

    # Step 2: Train on this round's relabeled data
    LAST_EPOCH="$RD/ckpt/epoch_$((EPOCHS-1))/model.safetensors"
    if [ ! -f "$LAST_EPOCH" ]; then
        log "  Training round $r model..."
        echo '{}' > "$RD/adj_empty.json"
        python src/train_with_adj.py \
            --train_path "$RD/train.jsonl" \
            --adj_path "$RD/adj_empty.json" \
            --ckpt_dir "$RD/ckpt" \
            --log_dir "$RD/logs" \
            --num_neg $NUM_NEG --epochs $EPOCHS --batch_size $BS --lr $LR --seed $SEED \
            2>&1 | tee "$LOGS_DIR/round${r}_train.log"
    else
        log "  skip: round $r ckpt exists"
    fi

    # Step 3: Evaluate on BEIR (fiqa test)
    if [ ! -f "$RD/metrics.json" ]; then
        log "  Evaluating round $r..."
        EVAL_CKPT="$RD/ckpt/epoch_$((EPOCHS-1))"
        [ ! -d "$EVAL_CKPT" ] && EVAL_CKPT="$RD/ckpt"
        python src/evaluate_beir_raw.py \
            --checkpoints "$EVAL_CKPT" \
            --datasets $BEIR_EVAL_DATASETS \
            --split test \
            2>&1 | tee "$LOGS_DIR/round${r}_eval.log" && \
        cp results/beir_comparison_test.json "$RD/metrics.json" || log "  EVAL FAILED round $r"
    else
        log "  skip: round $r metrics exists"
    fi

    log "  Round $r done."
done

# ---- Also evaluate round 0 and 1 if not done ------------------------------
for r in 0 1; do
    RD="$BASE_OUT/round_$r"
    if [ ! -f "$RD/metrics.json" ]; then
        log "Evaluating round $r..."
        EVAL_CKPT="$RD/ckpt/epoch_$((EPOCHS-1))"
        [ ! -d "$EVAL_CKPT" ] && EVAL_CKPT="$RD/ckpt"
        python src/evaluate_beir_raw.py \
            --checkpoints "$EVAL_CKPT" \
            --datasets $BEIR_EVAL_DATASETS \
            --split test \
            2>&1 | tee "$LOGS_DIR/round${r}_eval.log" && \
        cp results/beir_comparison_test.json "$RD/metrics.json" || log "  EVAL FAILED round $r"
    fi
done

# ---- Aggregate results -----------------------------------------------------
log "Aggregating results"
python - "$BASE_OUT" "$ROUNDS" <<'AGGREGATE'
import json, sys, os, glob

base_out = sys.argv[1]
n_rounds = int(sys.argv[2])

summary = {"rounds": []}
for r in range(n_rounds + 1):
    rd = os.path.join(base_out, f"round_{r}")
    entry = {"round": r}

    # Metrics
    mp = os.path.join(rd, "metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            m = json.load(f)
        # extract nDCG@10 from nested structure
        def find_ndcg(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if kk.lower() == "ndcg@10":
                                return vv
                        result = find_ndcg(v)
                        if result is not None:
                            return result
            return None
        entry["ndcg10"] = find_ndcg(m)
        entry["metrics_raw"] = m

    # Flag stats
    fp = os.path.join(rd, "flag_stats.json")
    if os.path.exists(fp):
        with open(fp) as f:
            entry["flag_stats"] = json.load(f)

    summary["rounds"].append(entry)

# Print table
print("\n" + "="*80)
print(f"{'Round':<8} {'nDCG@10':<10} {'|Flags|':<10} {'P(GT)':<10} {'Jaccard(prev)':<15} {'New':<8} {'Dropped':<8}")
print("-"*80)
for e in summary["rounds"]:
    ndcg = f"{e.get('ndcg10', 'N/A'):.4f}" if isinstance(e.get('ndcg10'), float) else "N/A"
    fs = e.get("flag_stats", {})
    nf = fs.get("n_flags", "N/A")
    gt = fs.get("gt_fn_overlap", {})
    prec = f"{gt.get('precision', 0):.4f}" if gt else "N/A"
    vs = fs.get("vs_prev_round", {})
    jacc = f"{vs.get('jaccard', 0):.4f}" if vs else "N/A"
    new = vs.get("new_flags", "N/A") if vs else "N/A"
    drop = vs.get("dropped", "N/A") if vs else "N/A"
    print(f"{e['round']:<8} {ndcg:<10} {nf:<10} {prec:<10} {jacc:<15} {new:<8} {drop:<8}")
print("="*80)

out = os.path.join(base_out, "summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved → {out}")
AGGREGATE

log "ALL DONE. Summary: $BASE_OUT/summary.json"
