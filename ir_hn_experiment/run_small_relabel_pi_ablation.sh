#!/usr/bin/env bash
# Relabel-only ablation for model-internal false-negative suspicion scores.
#
# Default target is fiqa_rlhn because it has baseline dynamics logs and RLHN
# FN labels for diagnostics. The training intervention is always relabeling:
# selected negatives are added back as positive training instances.
#
# Override examples:
#   K=1000 SCORES="pi_loss pi_dyn pi_bayes risk_bayes hardneg_prob" ./run_small_relabel_pi_ablation.sh
#   DATASET=nfcorpus EPOCHS=10 EVAL_DATASETS="nfcorpus" ./run_small_relabel_pi_ablation.sh

set -euo pipefail

REPO=${REPO:-$(pwd)}
cd "$REPO"

DATASET=${DATASET:-fiqa_rlhn}
DATA=${DATA:-data/processed/$DATASET}
OUT=${OUT:-experiments/$DATASET}
LOGS=${LOGS:-logs/small_relabel_pi_ablation}

K=${K:-597}
EPOCHS=${EPOCHS:-5}
BS=${BS:-16}
LR=${LR:-2e-5}
NUM_NEG=${NUM_NEG:-7}
SEED=${SEED:-42}

SCORES=${SCORES:-"pi_loss pi_dyn pi_bayes risk_bayes avg_margin hardneg_prob"}
EVAL_DATASETS=${EVAL_DATASETS:-"fiqa nfcorpus scifact"}
EVAL_SPLIT=${EVAL_SPLIT:-test}
DATA_ROOT=${DATA_ROOT:-data/beir_raw}
CONDA_ENV=${CONDA_ENV:-nlp}

SEARCH_MODE=${SEARCH_MODE:-auto}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-256}
QUERY_BLOCK_SIZE=${QUERY_BLOCK_SIZE:-64}
DOC_BLOCK_SIZE=${DOC_BLOCK_SIZE:-65536}
SEARCH_DTYPE=${SEARCH_DTYPE:-float16}

GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-2}
SAVE_EVERY_EPOCH=${SAVE_EVERY_EPOCH:-0}

LOSS_TAU=${LOSS_TAU:-1.0}
PI_TAU=${PI_TAU:-1.0}
EVIDENCE_TAU=${EVIDENCE_TAU:-1.0}
RELIABILITY_GAMMA=${RELIABILITY_GAMMA:-0.05}
BAYES_PRIOR=${BAYES_PRIOR:-0.04}
BAYES_STRENGTH=${BAYES_STRENGTH:-2.0}

mkdir -p "$OUT" "$LOGS"

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
log() { echo "[$(ts)] $*" | tee -a "$LOGS/master.log"; }

variant_key() {
    local score="$1"
    echo "${score}_clean"
}

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
    if [ ! -d "$ckpt" ]; then
        return
    fi
    find "$ckpt" -maxdepth 2 -type f -path "$ckpt/epoch_*/config.json" 2>/dev/null \
        | sed 's#/config.json$##' \
        | sort -V \
        | tail -1
}

EMPTY_ADJ="$OUT/adj_empty.json"
if [ ! -f "$EMPTY_ADJ" ]; then
    echo '{"_dummy_": [0,0,0,0,0,0,0]}' > "$EMPTY_ADJ"
fi

FN_ARGS=()
if [ -f "$DATA/fn_ground_truth.json" ]; then
    FN_ARGS=(--fn_ground_truth "$DATA/fn_ground_truth.json")
fi

make_relabel() {
    local score="$1"
    local key
    key=$(variant_key "$score")
    local out_train="$DATA/train_relabeled_${key}_K${K}.jsonl"
    local result="$OUT/pi_relabel_${key}_K${K}.json"
    local pairs="$OUT/selected_${key}_K${K}.json"
    if [ -f "$out_train" ] && [ -f "$pairs" ] && [ -f "$result" ]; then
        log "skip relabel $score: $out_train exists"
        return
    fi
    if [ -f "$out_train" ] || [ -f "$out_train.tmp" ] || [ -f "$result" ] || [ -f "$pairs" ]; then
        log "remove incomplete relabel artifacts for $score K=$K"
        rm -f "$out_train" "$out_train.tmp" "$result" "$pairs"
    fi
    log "make relabeled train $score K=$K -> $out_train"
    python src/estimate_pi.py \
        --mode none \
        --log_dir "$OUT/logs_baseline" \
        --train_path "$DATA/train.jsonl" \
        "${FN_ARGS[@]}" \
        --output "$result" \
        --loss_tau "$LOSS_TAU" \
        --pi_tau "$PI_TAU" \
        --evidence_tau "$EVIDENCE_TAU" \
        --reliability_gamma "$RELIABILITY_GAMMA" \
        --bayes_prior "$BAYES_PRIOR" \
        --bayes_strength "$BAYES_STRENGTH" \
        --relabel_output "$out_train" \
        --relabel_k "$K" \
        --relabel_score "$score" \
        --selected_pairs_output "$pairs" \
        2>&1 | tee "$LOGS/relabel_${key}_K${K}.log"
}

train_variant() {
    local score="$1"
    local key
    key=$(variant_key "$score")
    local name="relabel_${key}_K${K}"
    local train_path="$DATA/train_relabeled_${key}_K${K}.jsonl"
    local ckpt="$OUT/ckpt_${name}"
    local logd="$OUT/logs_${name}"
    if complete_ckpt "$ckpt"; then
        log "skip train $name: complete checkpoint exists"
        return
    fi
    if [ ! -f "$train_path" ]; then
        log "missing relabeled train for $name: $train_path"
        return 1
    fi
    log "train $name"
    local train_extra=()
    if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
        train_extra+=(--gradient_checkpointing)
    fi
    if [ "$SAVE_EVERY_EPOCH" = "1" ]; then
        train_extra+=(--save_every_epoch)
    fi
    python src/train_with_adj.py \
        --train_path "$train_path" \
        --adj_path "$EMPTY_ADJ" \
        --ckpt_dir "$ckpt" \
        --log_dir "$logd" \
        --num_neg "$NUM_NEG" \
        --epochs "$EPOCHS" \
        --batch_size "$BS" \
        --lr "$LR" \
        --seed "$SEED" \
        --dataloader_workers "$DATALOADER_WORKERS" \
        "${train_extra[@]}" \
        2>&1 | tee "$LOGS/train_${name}.log"
}

train_baseline() {
    local name="baseline"
    local ckpt="$OUT/ckpt_baseline"
    local logd="$OUT/logs_baseline"
    if complete_ckpt "$ckpt"; then
        log "skip train baseline: complete checkpoint exists"
        return
    fi
    log "train baseline"
    local train_extra=()
    if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
        train_extra+=(--gradient_checkpointing)
    fi
    if [ "$SAVE_EVERY_EPOCH" = "1" ]; then
        train_extra+=(--save_every_epoch)
    fi
    python src/train_with_adj.py \
        --train_path "$DATA/train.jsonl" \
        --adj_path "$EMPTY_ADJ" \
        --ckpt_dir "$ckpt" \
        --log_dir "$logd" \
        --num_neg "$NUM_NEG" \
        --epochs "$EPOCHS" \
        --batch_size "$BS" \
        --lr "$LR" \
        --seed "$SEED" \
        --dataloader_workers "$DATALOADER_WORKERS" \
        "${train_extra[@]}" \
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
        --split "$EVAL_SPLIT" \
        --data_root "$DATA_ROOT" \
        --search_mode "$SEARCH_MODE" \
        --encode_batch_size "$ENCODE_BATCH_SIZE" \
        --query_block_size "$QUERY_BLOCK_SIZE" \
        --doc_block_size "$DOC_BLOCK_SIZE" \
        --search_dtype "$SEARCH_DTYPE" \
        --output "$out" \
        2>&1 | tee "$LOGS/eval_${name}.log"
}

log "DATASET=$DATASET K=$K EPOCHS=$EPOCHS BS=$BS SCORES=[$SCORES] GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING SAVE_EVERY_EPOCH=$SAVE_EVERY_EPOCH EVAL_DATASETS=[$EVAL_DATASETS]"

train_baseline

for score in $SCORES; do
    make_relabel "$score"
done

for score in $SCORES; do
    train_variant "$score"
done

eval_variant baseline "$OUT/ckpt_baseline"
eval_variant mp3_relabel "$OUT/ckpt_relabel_mp3"
eval_variant rlhn_relabel "$OUT/ckpt_rlhn_relabel"

for score in $SCORES; do
    key=$(variant_key "$score")
    eval_variant "relabel_${key}_K${K}" "$OUT/ckpt_relabel_${key}_K${K}"
done

python - "$OUT" "$K" "$EVAL_DATASETS" $SCORES <<'PY' 2>&1 | tee "$LOGS/summary_K${K}.log"
import json
import os
import sys

out_dir = sys.argv[1]
k = sys.argv[2]
datasets = sys.argv[3].split()
scores = sys.argv[4:]

def variant_key(score):
    return f"{score}_clean"

names = ["baseline", "mp3_relabel", "rlhn_relabel"]
names += [f"relabel_{variant_key(score)}_K{k}" for score in scores]

summary = {}
for name in names:
    path = os.path.join(out_dir, f"metrics_{name}.json")
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    row = {}
    for ds, ckpt_map in data.items():
        vals = next(iter(ckpt_map.values())) if ckpt_map else {}
        if vals:
            row[ds] = vals
    summary[name] = row

summary_path = os.path.join(out_dir, f"small_relabel_ndcg_summary_K{k}.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Saved {summary_path}")
print("name\t" + "\t".join(datasets))
for name in names:
    row = summary.get(name, {})
    vals = []
    for ds in datasets:
        val = row.get(ds, {}).get("nDCG@10")
        vals.append("NA" if val is None else f"{val:.4f}")
    print(name + "\t" + "\t".join(vals))
PY

log "done"
