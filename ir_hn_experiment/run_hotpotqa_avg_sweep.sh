#!/usr/bin/env bash
# Overnight HotpotQA avg-margin relabel K sweep.
#
# Assumes data/processed/hotpotqa/train.jsonl already exists. The baseline is
# trained once by run_small_relabel_pi_ablation.sh and skipped on later K values.

set -euo pipefail

REPO=${REPO:-$(pwd)}
cd "$REPO"

K_LIST=${K_LIST:-"1000 5000 10000"}
EPOCHS=${EPOCHS:-3}
BS=${BS:-16}
LR=${LR:-2e-5}
NUM_NEG=${NUM_NEG:-7}
SEED=${SEED:-42}
CONDA_ENV=${CONDA_ENV:-nlp}

DATASET=${DATASET:-hotpotqa}
DATA=${DATA:-data/processed/hotpotqa}
OUT=${OUT:-experiments/hotpotqa}
DATA_ROOT=${DATA_ROOT:-data/raw}
EVAL_DATASETS=${EVAL_DATASETS:-hotpotqa}
EVAL_SPLIT=${EVAL_SPLIT:-test}
SEARCH_MODE=${SEARCH_MODE:-torch_stream}
DOC_BLOCK_SIZE=${DOC_BLOCK_SIZE:-65536}
QUERY_BLOCK_SIZE=${QUERY_BLOCK_SIZE:-64}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-256}
SEARCH_DTYPE=${SEARCH_DTYPE:-float16}

LOG_ROOT=${LOG_ROOT:-logs/hotpotqa_avg_sweep}
mkdir -p "$LOG_ROOT" "$OUT"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_ROOT/master.log"; }

if [ ! -f "$DATA/train.jsonl" ]; then
    log "missing $DATA/train.jsonl; run prepare_beir.py first"
    exit 1
fi

log "HotpotQA avg-margin sweep K_LIST=[$K_LIST] EPOCHS=$EPOCHS BS=$BS"
for K in $K_LIST; do
    log "START K=$K"
    DATASET="$DATASET" \
    DATA="$DATA" \
    OUT="$OUT" \
    DATA_ROOT="$DATA_ROOT" \
    EPOCHS="$EPOCHS" \
    BS="$BS" \
    LR="$LR" \
    NUM_NEG="$NUM_NEG" \
    SEED="$SEED" \
    CONDA_ENV="$CONDA_ENV" \
    EVAL_DATASETS="$EVAL_DATASETS" \
    EVAL_SPLIT="$EVAL_SPLIT" \
    SCORES="avg_margin" \
    K="$K" \
    SEARCH_MODE="$SEARCH_MODE" \
    DOC_BLOCK_SIZE="$DOC_BLOCK_SIZE" \
    QUERY_BLOCK_SIZE="$QUERY_BLOCK_SIZE" \
    ENCODE_BATCH_SIZE="$ENCODE_BATCH_SIZE" \
    SEARCH_DTYPE="$SEARCH_DTYPE" \
    LOGS="$LOG_ROOT/K${K}" \
    ./run_small_relabel_pi_ablation.sh
    log "DONE K=$K"
done

python - "$OUT" $K_LIST <<'PY' 2>&1 | tee "$LOG_ROOT/summary.log"
import json
import os
import sys

out_dir = sys.argv[1]
ks = sys.argv[2:]
summary = {}
for k in ks:
    path = os.path.join(out_dir, f"small_relabel_ndcg_summary_K{k}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            summary[k] = json.load(f)

out_path = os.path.join(out_dir, "hotpotqa_avg_margin_sweep_summary.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Saved {out_path}")
print("K\tbaseline\trelabel_avg_margin\tdelta")
for k in ks:
    row = summary.get(k, {})
    base = row.get("baseline", {}).get("hotpotqa", {}).get("nDCG@10")
    rel = row.get(f"relabel_avg_margin_K{k}", {}).get("hotpotqa", {}).get("nDCG@10")
    if base is None or rel is None:
        print(f"{k}\tNA\tNA\tNA")
    else:
        print(f"{k}\t{base:.4f}\t{rel:.4f}\t{rel-base:+.4f}")
PY

log "ALL DONE"
