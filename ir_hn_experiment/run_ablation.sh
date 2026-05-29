#!/bin/bash
set -e

BASE="experiments/fiqa_rlhn"
TRAIN="data/processed/fiqa_rlhn/train.jsonl"
COMMON="--num_neg 7 --epochs 5 --batch_size 16 --lr 2e-5"

# 1. Oracle (RLHN FN ground truth)
echo "=== [1/8] Oracle ==="
python src/train_with_adj.py --train_path $TRAIN --adj_path $BASE/adj_oracle.json \
    --ckpt_dir $BASE/ckpt_oracle --log_dir $BASE/logs_oracle $COMMON --seed 42

# 2. Random x3
for SEED in 42 123 456; do
    echo "=== [2-4] Random seed=$SEED ==="
    python src/train_with_adj.py --train_path $TRAIN --adj_path $BASE/adj_random_s${SEED}.json \
        --ckpt_dir $BASE/ckpt_random_s${SEED} --log_dir $BASE/logs_random_s${SEED} $COMMON --seed $SEED
done

# 3. margin_persistent_3plus x2 more seeds (seed 42 already done)
for SEED in 123 456; do
    echo "=== [5-6] margin_p3+ seed=$SEED ==="
    python src/train_with_adj.py --train_path $TRAIN --adj_path $BASE/adj_margin_persistent_3plus.json \
        --ckpt_dir $BASE/ckpt_mp3_s${SEED} --log_dir $BASE/logs_mp3_s${SEED} $COMMON --seed $SEED
done

# 4. Baseline x2 more seeds (no adjustments - use run_dataset_experiment train_model directly)
for SEED in 123 456; do
    echo "=== [7-8] baseline seed=$SEED ==="
    python -c "
import sys, torch, numpy as np; sys.path.insert(0,'.')
torch.manual_seed($SEED); np.random.seed($SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all($SEED)
from src.run_dataset_experiment import train_model
train_model('$TRAIN', '$BASE/logs_baseline_s${SEED}', '$BASE/ckpt_baseline_s${SEED}', 7, 5, 16, 2e-5)
"
done

echo "=== All training done ==="

# 5. Evaluate everything
python src/evaluate_beir_raw.py --checkpoints \
    $BASE/ckpt_baseline/epoch_4 \
    $BASE/ckpt_baseline_s123/epoch_4 \
    $BASE/ckpt_baseline_s456/epoch_4 \
    $BASE/ckpt_oracle/epoch_4 \
    $BASE/ckpt_random_s42/epoch_4 \
    $BASE/ckpt_random_s123/epoch_4 \
    $BASE/ckpt_random_s456/epoch_4 \
    $BASE/ckpt_margin_persistent_3plus/epoch_4 \
    $BASE/ckpt_mp3_s123/epoch_4 \
    $BASE/ckpt_mp3_s456/epoch_4 \
    --datasets fiqa --split test
