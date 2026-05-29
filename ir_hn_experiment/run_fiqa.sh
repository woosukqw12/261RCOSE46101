#!/bin/bash
# =============================================================================
# FiQA Experiments: 2x2 ablation (refresh x GCG)
# =============================================================================
# Conditions:
#   A. no_refresh + no_gate  — pure vanilla InfoNCE (true baseline)
#   B. refresh_only          — ANCE-style hard neg mining, no gating
#   C. gcg_only              — GCG gating, static negatives (isolates GCG effect)
#   D. refresh_gcg           — full system: refresh + GCG
#
# Prerequisites:
#   1. Download FiQA:
#        wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
#        unzip fiqa.zip -d ./data/raw/
#
#   2. Preprocess:
#        python ./scripts/prepare_beir.py \
#          --beir_dir ./data/raw/fiqa \
#          --output_dir ./data/processed/fiqa \
#          --neg_per_query 8
#
#   3. Run this script:
#        chmod +x ./run_fiqa.sh && ./run_fiqa.sh
# =============================================================================

SCRIPT="python ./scripts/train_gcg.py"
DATA="./data/processed/fiqa"

# shared args (no refresh flag here — set per condition)
COMMON="--model_type bge --model_name BAAI/bge-base-en-v1.5 --epochs 10 --batch_size 16 --neg_per_query 8 --lr 2e-6 --temperature 0.05"

# GCG-off: warmup_epochs=999 → lam≈0 throughout → all neg weights=1.0 (standard InfoNCE)
NO_GATE="--warmup_epochs 999 --sharpness 0.0"

# GCG-on
GCG_ON="--gate_method adaptive --percentile 80 --sharpness 20 --warmup_epochs 1 --audit"

# =============================================================================
# A. no_refresh + no_gate  (true vanilla baseline)
# =============================================================================
echo "===== [FiQA] A. no_refresh + no_gate ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_A_norefresh_nogate \
  $COMMON \
  $NO_GATE

# =============================================================================
# B. refresh_only  (ANCE-style)
# =============================================================================
echo "===== [FiQA] B. refresh_only ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_B_refresh_nogate \
  $COMMON \
  --dynamic_refresh --refresh_every 3 \
  $NO_GATE

# =============================================================================
# C. gcg_only  (GCG gate, static negatives — isolates GCG effect)
# =============================================================================
echo "===== [FiQA] C. gcg_only ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_C_norefresh_gcg \
  $COMMON \
  $GCG_ON

# =============================================================================
# D. refresh + GCG  (full system)
# =============================================================================
echo "===== [FiQA] D. refresh + GCG ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_D_refresh_gcg \
  $COMMON \
  --dynamic_refresh --refresh_every 3 \
  $GCG_ON

echo "===== [FiQA] ALL DONE ====="