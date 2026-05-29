#!/bin/bash
# =============================================================================
# Experiment: Gradient Conflict Gating (GCG) vs Baselines
# =============================================================================

# ----- 1. Baseline: Standard InfoNCE (no gating) ----------------------------
#   GCG with warmup_epochs=999 and sharpness=0 effectively disables gating
#   (all weights stay at 1.0), giving a clean baseline.

python ./scripts/train_gcg.py \
  --dataset_dir ./data/processed/scifact \
  --output_dir  ./outputs/bge/scifact_baseline \
  --model_type  bge \
  --model_name  BAAI/bge-base-en-v1.5 \
  --epochs 5 \
  --batch_size 16 \
  --neg_per_query 8 \
  --dynamic_refresh \
  --warmup_epochs 999 \
  --sharpness 0.0

# ----- 2. GCG: Closed-form (default, recommended) ---------------------------

python ./scripts/train_gcg.py \
  --dataset_dir ./data/processed/scifact \
  --output_dir  ./outputs/bge/scifact_gcg_closed \
  --model_type  bge \
  --model_name  BAAI/bge-base-en-v1.5 \
  --epochs 5 \
  --batch_size 16 \
  --neg_per_query 8 \
  --dynamic_refresh \
  --gate_method adaptive \
  --percentile 80 \
  --sharpness 20 \
  --warmup_epochs 1

# ----- 3. GCG: Autograd mode (for ablation / validation) --------------------
#   Verifies that closed-form and autograd give equivalent results.

python ./scripts/train_gcg.py \
  --dataset_dir ./data/processed/scifact \
  --output_dir  ./outputs/bge/scifact_gcg_autograd \
  --model_type  bge \
  --model_name  BAAI/bge-base-en-v1.5 \
  --epochs 5 \
  --batch_size 16 \
  --neg_per_query 8 \
  --dynamic_refresh \
  --gate_method adaptive \
  --percentile 80 \
  --sharpness 20 \
  --warmup_epochs 1 \
  --conflict_mode autograd

# ----- 4. Ablation: Percentile sensitivity ----------------------------------

for pct in 60 70 80 90 95; do
  python ./scripts/train_gcg.py \
    --dataset_dir ./data/processed/scifact \
    --output_dir  ./outputs/bge/scifact_gcg_pct${pct} \
    --model_type  bge \
    --model_name  BAAI/bge-base-en-v1.5 \
    --epochs 5 \
    --batch_size 16 \
    --neg_per_query 8 \
    --dynamic_refresh \
    --gate_method adaptive \
    --percentile ${pct} \
    --sharpness 20 \
    --warmup_epochs 1
done

# ----- 5. Ablation: Sharpness sensitivity -----------------------------------

for sharp in 5 10 20 50 100; do
  python ./scripts/train_gcg.py \
    --dataset_dir ./data/processed/scifact \
    --output_dir  ./outputs/bge/scifact_gcg_sharp${sharp} \
    --model_type  bge \
    --model_name  BAAI/bge-base-en-v1.5 \
    --epochs 5 \
    --batch_size 16 \
    --neg_per_query 8 \
    --dynamic_refresh \
    --gate_method adaptive \
    --percentile 80 \
    --sharpness ${sharp} \
    --warmup_epochs 1
done

# ----- 6. Ablation: Warmup epochs -------------------------------------------

for wu in 0 1 2 3; do
  python ./scripts/train_gcg.py \
    --dataset_dir ./data/processed/scifact \
    --output_dir  ./outputs/bge/scifact_gcg_wu${wu} \
    --model_type  bge \
    --model_name  BAAI/bge-base-en-v1.5 \
    --epochs 5 \
    --batch_size 16 \
    --neg_per_query 8 \
    --dynamic_refresh \
    --gate_method adaptive \
    --percentile 80 \
    --sharpness 20 \
    --warmup_epochs ${wu}
done

# ----- 7. Compare with your previous score-based router ---------------------

python ./scripts/train_single_vector_router_sim.py \
  --dataset_dir ./data/processed/scifact \
  --output_dir  ./outputs/bge/scifact_router_sim \
  --model_type  bge \
  --model_name  BAAI/bge-base-en-v1.5 \
  --epochs 5 \
  --batch_size 16 \
  --neg_per_query 8 \
  --dynamic_refresh