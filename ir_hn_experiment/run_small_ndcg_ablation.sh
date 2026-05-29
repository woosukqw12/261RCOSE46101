#!/usr/bin/env bash
# Small nDCG ablation for label-free harmful-negative scores.
#
# Default target is fiqa_rlhn because it already has baseline dynamics logs and
# RLHN FN GT for offline diagnostics. The downstream evaluation is on small BEIR
# datasets (fiqa/nfcorpus/scifact by default).

set -euo pipefail

REPO=${REPO:-$(pwd)}
cd "$REPO"

DATASET=${DATASET:-fiqa_rlhn}
DATA=${DATA:-data/processed/$DATASET}
OUT=${OUT:-experiments/$DATASET}
LOGS=${LOGS:-logs/small_ndcg_ablation}
RESULT=${RESULT:-results/pi_${DATASET}_mixture_pu.json}

K=${K:-597}
EPOCHS=${EPOCHS:-5}
BS=${BS:-16}
LR=${LR:-2e-5}
NUM_NEG=${NUM_NEG:-7}
SEED=${SEED:-42}

TAIL_PRIOR=${TAIL_PRIOR:-0.015}
PU_SEED_FRAC=${PU_SEED_FRAC:-0.01}
PU_PRIOR=${PU_PRIOR:-0.04}
PU_STEPS=${PU_STEPS:-3000}

EVAL_DATASETS=${EVAL_DATASETS:-"fiqa nfcorpus scifact"}
DATA_ROOT=${DATA_ROOT:-data/beir_raw}
CONDA_ENV=${CONDA_ENV:-nlp}

SEARCH_MODE=${SEARCH_MODE:-auto}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-256}
QUERY_BLOCK_SIZE=${QUERY_BLOCK_SIZE:-64}
DOC_BLOCK_SIZE=${DOC_BLOCK_SIZE:-65536}
SEARCH_DTYPE=${SEARCH_DTYPE:-float16}

mkdir -p "$OUT" "$LOGS"

source /home/yoonheon/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOGS/master.log"; }

complete_ckpt() {
    local ckpt="$1"
    local last_epoch=$((EPOCHS - 1))
    [ -f "$ckpt/epoch_${last_epoch}/config.json" ] || [ -f "$ckpt/config.json" ]
}

latest_ckpt() {
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

make_adj() {
    local score="$1"
    local path="$2"
    if [ -f "$path" ]; then
        log "skip adj $score: $path exists"
        return
    fi
    log "make adj $score -> $path"
    python src/estimate_pi.py \
        --mode both \
        --tail_prior "$TAIL_PRIOR" \
        --pu_seed_frac "$PU_SEED_FRAC" \
        --pu_prior "$PU_PRIOR" \
        --pu_steps "$PU_STEPS" \
        --log_dir "$OUT/logs_baseline" \
        --train_path "$DATA/train.jsonl" \
        --fn_ground_truth "$DATA/fn_ground_truth.json" \
        --output "$RESULT" \
        --adj_output "$path" \
        --adj_k "$K" \
        --adj_score "$score" \
        2>&1 | tee "$LOGS/adj_${score}.log"
}

train_variant() {
    local name="$1"
    local adj="$2"
    local ckpt="$OUT/ckpt_${name}"
    local logd="$OUT/logs_${name}"
    if complete_ckpt "$ckpt"; then
        log "skip train $name: complete checkpoint exists"
        return
    fi
    if [ ! -f "$adj" ]; then
        log "missing adj for $name: $adj"
        return 1
    fi
    log "train $name"
    python src/train_with_adj.py \
        --train_path "$DATA/train.jsonl" \
        --adj_path "$adj" \
        --ckpt_dir "$ckpt" \
        --log_dir "$logd" \
        --num_neg "$NUM_NEG" \
        --epochs "$EPOCHS" \
        --batch_size "$BS" \
        --lr "$LR" \
        --seed "$SEED" \
        2>&1 | tee "$LOGS/train_${name}.log"
}

eval_variant() {
    local name="$1"
    local ckpt_root="$2"
    local ckpt
    ckpt=$(latest_ckpt "$ckpt_root")
    if [ -z "$ckpt" ]; then
        log "skip eval $name: no checkpoint under $ckpt_root"
        return
    fi
    local out="$OUT/metrics_${name}.json"
    if [ -f "$out" ]; then
        log "skip eval $name: $out exists"
        return
    fi
    log "eval $name ($ckpt)"
    python src/evaluate_beir_raw.py \
        --checkpoints "$ckpt" \
        --datasets $EVAL_DATASETS \
        --split test \
        --data_root "$DATA_ROOT" \
        --search_mode "$SEARCH_MODE" \
        --encode_batch_size "$ENCODE_BATCH_SIZE" \
        --query_block_size "$QUERY_BLOCK_SIZE" \
        --doc_block_size "$DOC_BLOCK_SIZE" \
        --search_dtype "$SEARCH_DTYPE" \
        --output "$out" \
        2>&1 | tee "$LOGS/eval_${name}.log"
}

log "DATASET=$DATASET K=$K EPOCHS=$EPOCHS BS=$BS EVAL_DATASETS=[$EVAL_DATASETS]"

make_adj avg_margin "$OUT/adj_avg_margin_K${K}.json"
make_adj hardneg_prob "$OUT/adj_hardneg_prob_K${K}.json"
make_adj pi_pu "$OUT/adj_pi_pu_K${K}.json"
make_adj risk_pu "$OUT/adj_risk_pu_K${K}.json"
make_adj risk_unsup_tail "$OUT/adj_risk_unsup_tail_K${K}.json"

train_variant "avg_margin_K${K}" "$OUT/adj_avg_margin_K${K}.json"
train_variant "hardneg_prob_K${K}" "$OUT/adj_hardneg_prob_K${K}.json"
train_variant "pi_pu_K${K}" "$OUT/adj_pi_pu_K${K}.json"
train_variant "risk_pu_K${K}" "$OUT/adj_risk_pu_K${K}.json"
train_variant "risk_unsup_tail_K${K}" "$OUT/adj_risk_unsup_tail_K${K}.json"

# Evaluate existing and new checkpoints. Existing names are from prior fiqa_rlhn runs.
eval_variant baseline "$OUT/ckpt_baseline"
eval_variant random_s42 "$OUT/ckpt_random_s42"
eval_variant mp3plus "$OUT/ckpt_margin_persistent_3plus"
eval_variant oracle "$OUT/ckpt_oracle"
eval_variant avg_margin_K${K} "$OUT/ckpt_avg_margin_K${K}"
eval_variant hardneg_prob_K${K} "$OUT/ckpt_hardneg_prob_K${K}"
eval_variant pi_pu_K${K} "$OUT/ckpt_pi_pu_K${K}"
eval_variant risk_pu_K${K} "$OUT/ckpt_risk_pu_K${K}"
eval_variant risk_unsup_tail_K${K} "$OUT/ckpt_risk_unsup_tail_K${K}"

python - <<'PY' "$OUT" "$K" 2>&1 | tee "$LOGS/summary.log"
import glob, json, os, sys

out_dir = sys.argv[1]
k = sys.argv[2]
names = [
    "baseline",
    "random_s42",
    "mp3plus",
    "oracle",
    f"avg_margin_K{k}",
    f"hardneg_prob_K{k}",
    f"pi_pu_K{k}",
    f"risk_pu_K{k}",
    f"risk_unsup_tail_K{k}",
]

summary = {}
for name in names:
    path = os.path.join(out_dir, f"metrics_{name}.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    row = {}
    for ds, ckpt_map in data.items():
        vals = next(iter(ckpt_map.values())) if ckpt_map else {}
        if vals:
            row[ds] = vals
    summary[name] = row

summary_path = os.path.join(out_dir, f"small_ndcg_summary_K{k}.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved {summary_path}")
print("name\t" + "\t".join(["fiqa", "nfcorpus", "scifact"]))
for name in names:
    row = summary.get(name, {})
    vals = []
    for ds in ["fiqa", "nfcorpus", "scifact"]:
        val = row.get(ds, {}).get("nDCG@10")
        vals.append("NA" if val is None else f"{val:.4f}")
    print(name + "\t" + "\t".join(vals))
PY

log "done"
