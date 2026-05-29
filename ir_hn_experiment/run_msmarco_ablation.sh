#!/usr/bin/env bash
# ============================================================================
# MS MARCO comprehensive ablation — run unattended on A100
# ----------------------------------------------------------------------------
# One-file orchestration of ALL essential ablations. Resumable: re-running
# skips stages whose output already exists.
#
# Stages:
#   0. Prepare data from rlhn/default-680K msmarco subset (+ FN ground truth)
#   1. Train baseline (produces epoch logs needed for dynamics)
#   2. Compute dynamics signals + criteria
#   3. Generate matched-budget adjustment files for all signals
#   4. Train all ablation variants (~10 runs)
#   5. Evaluate all checkpoints on MSMARCO dev + BEIR transfer
#   6. Aggregate → summary.json
#
# Logs: logs/msmarco_ablation_<stage>.log
# Checkpoints: experiments/msmarco_rlhn/ckpt_<variant>/
# Results: experiments/msmarco_rlhn/metrics_<variant>.json
# Summary: experiments/msmarco_rlhn/summary.json
# ============================================================================

set -euo pipefail

# ---- Config ----------------------------------------------------------------
REPO=${REPO:-$(pwd)}
cd "$REPO"

DATASET=msmarco_rlhn
DATA=data/processed/$DATASET
OUT=experiments/$DATASET
LOGS=logs/msmarco_ablation
mkdir -p "$OUT" "$LOGS"
export OUT

# Training hparams — tuned for A100 80GB, bf16-friendly
EPOCHS=${EPOCHS:-3}
BS=${BS:-64}
LR=${LR:-2e-5}
NUM_NEG=${NUM_NEG:-7}

# Matched budget as fraction of total pairs (≈ RLHN FN rate on msmarco)
# Multiple budgets = budget sweep on best signal
BUDGETS=${BUDGETS:-0.01,0.02,0.04}

# Signals to ablate at matched budget (P@K analysis on fiqa_rlhn suggested
# avg_margin > persistent_count > rank_top1_count)
SIGNALS=${SIGNALS:-avg_margin,persistent_count,rank_top1_count,persistent_x_margin}

# Random seeds for statistical control
RANDOM_SEEDS=${RANDOM_SEEDS:-42,123}

# Conda env (set to your env name)
CONDA_ENV=${CONDA_ENV:-nlp}
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV"

# BEIR evaluation datasets (transfer test — standard RLHN protocol)
BEIR_DATASETS="msmarco trec-covid nfcorpus nq hotpotqa fiqa scidocs scifact arguana"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOGS/master.log"; }
has_checkpoint() {
    local ckpt="$1"
    compgen -G "$ckpt/epoch_*/config.json" >/dev/null || [ -f "$ckpt/config.json" ]
}
has_complete_checkpoint() {
    local ckpt="$1"
    local last_epoch=$((EPOCHS - 1))
    [ -f "$ckpt/epoch_${last_epoch}/config.json" ] || [ -f "$ckpt/config.json" ]
}
latest_checkpoint() {
    local ckpt="$1"
    if [ -f "$ckpt/config.json" ]; then
        echo "$ckpt"
        return
    fi
    find "$ckpt" -maxdepth 2 -type f -path "$ckpt/epoch_*/config.json" 2>/dev/null \
        | sed 's#/config.json$##' \
        | sort -V \
        | tail -1
}

# ---- STAGE 0: Data prep ----------------------------------------------------
log "STAGE 0: Prepare data"
if [ ! -f "$DATA/train.jsonl" ] || [ ! -f "$DATA/fn_ground_truth.json" ]; then
    python src/prepare_rlhn_msmarco.py 2>&1 | tee "$LOGS/stage0_prepare.log"
else
    log "  skip: $DATA/train.jsonl exists"
fi
N_TRAIN=$(wc -l < "$DATA/train.jsonl")
log "  train.jsonl: $N_TRAIN queries"

# ---- STAGE 1: Baseline training --------------------------------------------
log "STAGE 1: Baseline training"
if ! has_complete_checkpoint "$OUT/ckpt_baseline"; then
    # Use train_with_adj.py with empty adjustments — it also logs per-epoch scores
    echo '{}' > "$OUT/adj_empty.json"
    python src/train_with_adj.py \
        --train_path "$DATA/train.jsonl" \
        --adj_path "$OUT/adj_empty.json" \
        --ckpt_dir "$OUT/ckpt_baseline" \
        --log_dir "$OUT/logs_baseline" \
        --num_neg $NUM_NEG --epochs $EPOCHS --batch_size $BS --lr $LR --seed 42 \
        2>&1 | tee "$LOGS/stage1_baseline.log"
else
    log "  skip: $OUT/ckpt_baseline exists"
fi
if [ ! -f "$OUT/adj_empty.json" ]; then
    echo '{}' > "$OUT/adj_empty.json"
fi

# ---- STAGE 2: Compute dynamics signals + criteria --------------------------
log "STAGE 2: Compute signals + criteria"
SIG_OUT=results/signals_$DATASET
if [ ! -f "$SIG_OUT/criteria_loss.json" ]; then
    mkdir -p "$SIG_OUT"
    python src/compute_signals.py \
        --log_dir "$OUT/logs_baseline" \
        --output_dir "$SIG_OUT" \
        --source loss 2>&1 | tee "$LOGS/stage2_signals.log"
else
    log "  skip: criteria_loss.json exists"
fi

# ---- STAGE 3: Generate matched-budget adj files ----------------------------
log "STAGE 3: Generate adjustment files (signals × budgets)"
if [ ! -f "$OUT/adj_generation_log.json" ]; then
    python src/generate_topk_adj.py \
        --log_dir "$OUT/logs_baseline" \
        --train_path "$DATA/train.jsonl" \
        --fn_ground_truth "$DATA/fn_ground_truth.json" \
        --output_dir "$OUT" \
        --budgets "$BUDGETS" \
        --signals "$SIGNALS" \
        --random_seeds "$RANDOM_SEEDS" 2>&1 | tee "$LOGS/stage3_adj.log"

else
    log "  skip: adj_generation_log.json exists"
fi

# Also materialize mp3+ (persistent>=3) criterion adj — "our original" baseline.
# Keep this outside the broad Stage 3 skip so interrupted runs can repair missing files.
if [ ! -f "$OUT/adj_mp3plus.json" ]; then
    python -c "
import json, numpy as np
with open('$SIG_OUT/criteria_loss.json') as f: crit = json.load(f)
pairs = crit.get('margin_persistent_3plus', [])
num_neg = $NUM_NEG
adj = {}
for qid, ni in pairs:
    adj.setdefault(qid, [0.0]*num_neg)
    adj[qid][ni] = -1e9
with open('$OUT/adj_mp3plus.json','w') as f: json.dump(adj, f)
print(f'mp3+ adj: {len(adj)} qids, {len(pairs)} pairs')
" 2>&1 | tee -a "$LOGS/stage3_adj.log"
else
    log "  skip: adj_mp3plus.json exists"
fi

# RLHN-style full relabel: treat all FN GT as positives via data augmentation.
if [ -f "$DATA/fn_ground_truth.json" ] && [ ! -f "$DATA/train_rlhn_relabel.jsonl" ]; then
    python src/prepare_relabeled_data.py \
        --train_path "$DATA/train.jsonl" \
        --fn_ground_truth "$DATA/fn_ground_truth.json" \
        --output_path "$DATA/train_rlhn_relabel.jsonl" 2>&1 | tee -a "$LOGS/stage3_adj.log"
else
    log "  skip: train_rlhn_relabel.jsonl exists or FN GT missing"
fi

# ---- STAGE 4: Train all ablation variants ----------------------------------
log "STAGE 4: Train ablation variants"

# Helper: train one variant (skip if checkpoint exists)
train_variant() {
    local name="$1" adj="$2" train="$3" seed="${4:-42}"
    local ckpt="$OUT/ckpt_$name" logd="$OUT/logs_$name"
    if has_complete_checkpoint "$ckpt"; then
        log "  skip train_variant $name: ckpt exists"
        return
    fi
    if [ ! -f "$adj" ]; then
        log "  FAILED $name: missing adj file $adj"
        return
    fi
    if [ ! -f "$train" ]; then
        log "  FAILED $name: missing train file $train"
        return
    fi
    log "  TRAIN $name  (adj=$adj, train=$train, seed=$seed)"
    python src/train_with_adj.py \
        --train_path "$train" \
        --adj_path "$adj" \
        --ckpt_dir "$ckpt" \
        --log_dir "$logd" \
        --num_neg $NUM_NEG --epochs $EPOCHS --batch_size $BS --lr $LR --seed "$seed" \
        2>&1 | tee "$LOGS/stage4_train_$name.log" || log "  FAILED $name (continuing)"
}

# Compute the primary budget K (first budget × total pairs)
PRIMARY_BUDGET=$(echo "$BUDGETS" | cut -d, -f1)
K_PRIMARY=$(python -c "
import json
n=sum(1 for _ in open('$DATA/train.jsonl'))
print(int(round(n * $NUM_NEG * $PRIMARY_BUDGET)))
")
log "  Primary matched budget K=$K_PRIMARY (budget=$PRIMARY_BUDGET)"

# --- Variants at PRIMARY budget (matched-budget ablation) ---
for sig in $(echo "$SIGNALS" | tr , ' '); do
    train_variant "${sig}_K${K_PRIMARY}" "$OUT/adj_${sig}_K${K_PRIMARY}.json" "$DATA/train.jsonl"
done
for seed in $(echo "$RANDOM_SEEDS" | tr , ' '); do
    train_variant "random_s${seed}_K${K_PRIMARY}" "$OUT/adj_random_s${seed}_K${K_PRIMARY}.json" "$DATA/train.jsonl" "$seed"
done

# --- mp3+ (our original threshold-based method) ---
train_variant "mp3plus" "$OUT/adj_mp3plus.json" "$DATA/train.jsonl"

# --- Oracle (RLHN FN as ground-truth mask) ---
ORACLE_FILE=$(ls "$OUT"/adj_oracle_K*.json 2>/dev/null | head -1 || true)
[ -n "$ORACLE_FILE" ] && train_variant "oracle" "$ORACLE_FILE" "$DATA/train.jsonl"

# --- RLHN full relabel (if prepared) ---
[ -f "$DATA/train_rlhn_relabel.jsonl" ] && \
    train_variant "rlhn_relabel" "$OUT/adj_empty.json" "$DATA/train_rlhn_relabel.jsonl"

# --- Budget sweep on best signal (avg_margin, assumed) ---
BEST_SIG=${BEST_SIG:-avg_margin}
for b in $(echo "$BUDGETS" | tr , ' '); do
    K=$(python -c "
n=sum(1 for _ in open('$DATA/train.jsonl'))
print(int(round(n * $NUM_NEG * $b)))
")
    [ "$K" = "$K_PRIMARY" ] && continue  # already trained at primary
    train_variant "${BEST_SIG}_K${K}" "$OUT/adj_${BEST_SIG}_K${K}.json" "$DATA/train.jsonl"
done

# ---- STAGE 5: Evaluate all checkpoints -------------------------------------
log "STAGE 5: Evaluate checkpoints on BEIR"

CHECKPOINTS=()
for d in "$OUT"/ckpt_*; do
    [ -d "$d" ] || continue
    ckpt=$(latest_checkpoint "$d")
    [ -n "$ckpt" ] || continue
    CHECKPOINTS+=("$ckpt")
done
log "  Found ${#CHECKPOINTS[@]} checkpoints"

for ckpt in "${CHECKPOINTS[@]}"; do
    parent=$(basename "$(dirname "$ckpt")")
    if [[ "$parent" == ckpt_* ]]; then
        name="${parent#ckpt_}"
    else
        name=$(basename "$ckpt" | sed 's/^ckpt_//')
    fi
    out="$OUT/metrics_${name}.json"
    if [ -f "$out" ]; then
        log "  skip eval $name: $out exists"
        continue
    fi
    log "  EVAL $name"
    python src/evaluate_beir_raw.py \
        --checkpoints "$ckpt" \
        --datasets $BEIR_DATASETS \
        --split test \
        --output "$out" \
        2>&1 | tee "$LOGS/stage5_eval_$name.log" || log "  FAILED eval $name"
done

# ---- STAGE 6: Aggregate summary --------------------------------------------
log "STAGE 6: Aggregate summary"
python - <<'PY' 2>&1 | tee "$LOGS/stage6_summary.log"
import glob, json, os, sys
OUT = os.environ.get("OUT", "experiments/msmarco_rlhn")
summary = {"runs": []}
for path in sorted(glob.glob(f"{OUT}/metrics_*.json")):
    name = os.path.basename(path)[len("metrics_"):-len(".json")]
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception as e:
        print(f"skip {name}: {e}"); continue
    summary["runs"].append({"name": name, "metrics": m})
with open(f"{OUT}/summary.json","w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote {OUT}/summary.json with {len(summary['runs'])} runs")
# Print quick table if nDCG@10 present
print("\n=== Quick summary (nDCG@10 on each dataset if available) ===")
for r in summary["runs"]:
    line = [r["name"]]
    m = r["metrics"]
    # evaluate_beir_raw typically returns nested {ckpt_path: {dataset: {metric: val}}}
    def find_ndcg(obj):
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict) and "ndcg@10" in {kk.lower(): None for kk in v}:
                    ndcg_key = next(kk for kk in v if kk.lower()=="ndcg@10")
                    out[k] = v[ndcg_key]
                elif isinstance(v, dict):
                    out.update(find_ndcg(v))
        return out
    ndcgs = find_ndcg(m)
    for ds, val in ndcgs.items():
        line.append(f"{ds}={val:.4f}")
    print("  " + " | ".join(line))
PY

log "ALL DONE. Summary: $OUT/summary.json"
