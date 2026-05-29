#!/bin/bash
# =============================================================================
# Experiment: GCG variants on SciFact
# =============================================================================

SCRIPT="python ./scripts/train_gcg.py"
DATA="./data/processed/scifact"
BASE_ARGS="--model_type bge --model_name BAAI/bge-base-en-v1.5 --epochs 5 --batch_size 16 --neg_per_query 8 --dynamic_refresh"

# =============================================================================
# 1. Baseline: no gating (warmup_epochs=999 keeps all weights at 1.0)
# =============================================================================
echo "===== 1. Baseline (no gating) ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/scifact_baseline \
  $BASE_ARGS \
  --warmup_epochs 999 \
  --sharpness 0.0

# =============================================================================
# 2. GCG original (closed-form, bare conflict, no pi_k, no tau_min)
# =============================================================================
echo "===== 2. GCG original ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/scifact_gcg_original \
  $BASE_ARGS \
  --gate_method adaptive \
  --percentile 80 \
  --sharpness 20 \
  --warmup_epochs 1

# =============================================================================
# 3. GCG + pi_k weighting only
# =============================================================================
echo "===== 3. GCG + pi_k ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/scifact_gcg_pi \
  $BASE_ARGS \
  --gate_method adaptive \
  --percentile 80 \
  --sharpness 20 \
  --warmup_epochs 1 \
  --use_pi_weighting

# =============================================================================
# 4. GCG + tau_min floor only (several values)
# =============================================================================
for tmin in 0.05 0.10 0.15 0.20; do
  echo "===== 4. GCG + tau_min=${tmin} ====="
  $SCRIPT \
    --dataset_dir $DATA \
    --output_dir  ./outputs/bge/scifact_gcg_taumin${tmin} \
    $BASE_ARGS \
    --gate_method adaptive \
    --percentile 80 \
    --sharpness 20 \
    --warmup_epochs 1 \
    --tau_min ${tmin}
done

# =============================================================================
# 5. GCG + pi_k + tau_min (combined)
# =============================================================================
for tmin in 0.01 0.02 0.05; do
  echo "===== 5. GCG + pi_k + tau_min=${tmin} ====="
  $SCRIPT \
    --dataset_dir $DATA \
    --output_dir  ./outputs/bge/scifact_gcg_pi_taumin${tmin} \
    $BASE_ARGS \
    --gate_method adaptive \
    --percentile 80 \
    --sharpness 20 \
    --warmup_epochs 1 \
    --use_pi_weighting \
    --tau_min ${tmin}
done

# =============================================================================
# 6. GCG autograd (sign-bug fixed, for validation vs closed-form)
# =============================================================================
echo "===== 6. GCG autograd (fixed) ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/scifact_gcg_autograd_fixed \
  $BASE_ARGS \
  --gate_method adaptive \
  --percentile 80 \
  --sharpness 20 \
  --warmup_epochs 1 \
  --conflict_mode autograd

echo "===== ALL DONE ====="