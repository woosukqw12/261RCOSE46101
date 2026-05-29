#!/bin/bash
# =============================================================================
# FiQA Experiments: HCG (Hard Conflict Gating) — 2x2 + floor ablation
# =============================================================================
# Conditions:
#   A. no_refresh + no_gate  — vanilla InfoNCE baseline  (same as run_fiqa.sh A)
#   B. refresh_only          — ANCE-style, no gating      (same as run_fiqa.sh B)
#   C. hcg_only              — HCG gate, static negatives (isolates HCG effect)
#   D. refresh_hcg           — full system: refresh + HCG
#
#   Extra ablation on sim_qn_floor (condition D only):
#   D1. floor=0.3  — wider suppression window
#   D2. floor=0.4  — default
#   D3. floor=0.5  — narrower (only very hard negatives gated)
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
#        chmod +x ./run_fiqa_hcg.sh && ./run_fiqa_hcg.sh
# =============================================================================

SCRIPT="python ./scripts/train_hcg.py"
DATA="./data/processed/fiqa"

COMMON="--model_type bge --model_name BAAI/bge-base-en-v1.5 \
  --epochs 10 --batch_size 16 --neg_per_query 8 \
  --lr 2e-6 --temperature 0.05"

# HCG-off: warmup_epochs=999 → threshold never reached → all weights=1.0
NO_GATE="--warmup_epochs 999 --sharpness 0.0"

# HCG-on (default floor=0.4)
HCG_ON="--gate_method adaptive --percentile 80 --sharpness 20 \
  --warmup_epochs 1 --sim_qn_floor 0.4 --audit"

# =============================================================================
# A. no_refresh + no_gate  (vanilla baseline)
# =============================================================================
echo "===== [FiQA-HCG] A. no_refresh + no_gate ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_hcg_A_norefresh_nogate \
  $COMMON \
  $NO_GATE

# =============================================================================
# B. refresh_only  (ANCE-style)
# =============================================================================
echo "===== [FiQA-HCG] B. refresh_only ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_hcg_B_refresh_nogate \
  $COMMON \
  --dynamic_refresh --refresh_every 3 \
  $NO_GATE

# =============================================================================
# C. hcg_only  (HCG gate, static negatives — isolates HCG effect)
# =============================================================================
echo "===== [FiQA-HCG] C. hcg_only ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_hcg_C_norefresh_hcg \
  $COMMON \
  $HCG_ON

# =============================================================================
# D. refresh + HCG  (full system, floor=0.4 default)
# =============================================================================
echo "===== [FiQA-HCG] D. refresh + HCG (floor=0.4) ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_hcg_D_refresh_hcg_f04 \
  $COMMON \
  --dynamic_refresh --refresh_every 3 \
  $HCG_ON

# =============================================================================
# Floor ablation on condition D
# =============================================================================
echo "===== [FiQA-HCG] D1. refresh + HCG (floor=0.3) ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_hcg_D1_refresh_hcg_f03 \
  $COMMON \
  --dynamic_refresh --refresh_every 3 \
  --gate_method adaptive --percentile 80 --sharpness 20 \
  --warmup_epochs 1 --sim_qn_floor 0.3 --audit

echo "===== [FiQA-HCG] D3. refresh + HCG (floor=0.5) ====="
$SCRIPT \
  --dataset_dir $DATA \
  --output_dir  ./outputs/bge/fiqa_hcg_D3_refresh_hcg_f05 \
  $COMMON \
  --dynamic_refresh --refresh_every 3 \
  --gate_method adaptive --percentile 80 --sharpness 20 \
  --warmup_epochs 1 --sim_qn_floor 0.5 --audit

echo "===== [FiQA-HCG] ALL DONE ====="
