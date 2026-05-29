#!/usr/bin/env bash
# RunPod-oriented MS MARCO RLHN avg-margin relabel sweep.
#
# This intentionally runs a narrow experiment:
#   1. prepare rlhn/default-680K msmarco subset if missing
#   2. train baseline once and log dynamics
#   3. generate avg_margin top-K relabeled train files
#   4. train/evaluate avg_margin relabel variants on MS MARCO dev
#
# It reuses run_small_relabel_pi_ablation.sh for resumability.

set -euo pipefail

REPO=${REPO:-$(pwd)}
cd "$REPO"

DATASET=${DATASET:-msmarco_rlhn}
DATA=${DATA:-data/processed/$DATASET}
OUT=${OUT:-experiments/$DATASET}
LOG_ROOT=${LOG_ROOT:-logs/msmarco_avg_relabel_sweep}

K_LIST=${K_LIST:-"25000 50000"}
EPOCHS=${EPOCHS:-3}
BS=${BS:-64}
LR=${LR:-2e-5}
NUM_NEG=${NUM_NEG:-7}
SEED=${SEED:-42}
CONDA_ENV=${CONDA_ENV:-nlp}

DATA_ROOT=${DATA_ROOT:-data/raw}
EVAL_DATASETS=${EVAL_DATASETS:-msmarco}
EVAL_SPLIT=${EVAL_SPLIT:-dev}
SEARCH_MODE=${SEARCH_MODE:-torch_stream}
DOC_BLOCK_SIZE=${DOC_BLOCK_SIZE:-131072}
QUERY_BLOCK_SIZE=${QUERY_BLOCK_SIZE:-256}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-512}
SEARCH_DTYPE=${SEARCH_DTYPE:-float16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-2}
SAVE_EVERY_EPOCH=${SAVE_EVERY_EPOCH:-0}

mkdir -p "$OUT" "$LOG_ROOT"

CONDA_SH=${CONDA_SH:-/home/yoonheon/miniconda3/etc/profile.d/conda.sh}
if [ -z "${CONDA_ENV:-}" ]; then
    echo "CONDA_ENV empty; using current Python environment"
elif [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$CONDA_ENV"
else
    echo "conda not found; using current Python environment"
fi

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_ROOT/master.log"; }

log "MS MARCO avg-margin relabel sweep K_LIST=[$K_LIST] EPOCHS=$EPOCHS BS=$BS"

if [ ! -f "$DATA/train.jsonl" ]; then
    log "prepare $DATASET via src/prepare_rlhn_msmarco.py --skip_fn_gt"
    python src/prepare_rlhn_msmarco.py \
        --output_dir "$DATA" \
        --skip_fn_gt \
        2>&1 | tee "$LOG_ROOT/prepare_msmarco_rlhn.log"
else
    log "skip prepare: $DATA/train.jsonl exists"
fi

if [ ! -f "$DATA/train.jsonl" ]; then
    log "missing $DATA/train.jsonl after preparation"
    exit 1
fi

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
    GRADIENT_CHECKPOINTING="$GRADIENT_CHECKPOINTING" \
    DATALOADER_WORKERS="$DATALOADER_WORKERS" \
    SAVE_EVERY_EPOCH="$SAVE_EVERY_EPOCH" \
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

out_path = os.path.join(out_dir, "msmarco_avg_margin_sweep_summary.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Saved {out_path}")
print("K\tbaseline\trelabel_avg_margin\tdelta")
for k in ks:
    row = summary.get(k, {})
    base = row.get("baseline", {}).get("msmarco", {}).get("nDCG@10")
    rel = row.get(f"relabel_avg_margin_clean_K{k}", {}).get("msmarco", {}).get("nDCG@10")
    if base is None or rel is None:
        print(f"{k}\tNA\tNA\tNA")
    else:
        print(f"{k}\t{base:.4f}\t{rel:.4f}\t{rel-base:+.4f}")
PY

log "ALL DONE"
