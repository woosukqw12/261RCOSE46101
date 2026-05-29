#!/bin/bash
set -e

COMMON="--num_neg 7 --epochs 5 --batch_size 16 --lr 2e-5 --seed 42"
EMPTY_ADJ='{"_dummy_": [0,0,0,0,0,0,0]}'

# fiqa_rlhn - relabeled (no adjustments needed, just different training data)
echo "=== fiqa_rlhn relabeled ==="
echo "$EMPTY_ADJ" > /tmp/empty_adj.json
python src/train_with_adj.py \
    --train_path data/processed/fiqa_rlhn/train_relabeled_margin_persistent_3plus.jsonl \
    --adj_path /tmp/empty_adj.json \
    --ckpt_dir experiments/fiqa_rlhn/ckpt_relabel_mp3 \
    --log_dir experiments/fiqa_rlhn/logs_relabel_mp3 \
    $COMMON

# fiqa
echo "=== fiqa relabeled ==="
python src/train_with_adj.py \
    --train_path data/processed/fiqa/train_relabeled_margin_persistent_3plus.jsonl \
    --adj_path /tmp/empty_adj.json \
    --ckpt_dir experiments/fiqa/ckpt_relabel_mp3 \
    --log_dir experiments/fiqa/logs_relabel_mp3 \
    $COMMON

# nfcorpus (10 epochs)
echo "=== nfcorpus relabeled ==="
python src/train_with_adj.py \
    --train_path data/processed/nfcorpus/train_relabeled_margin_persistent_3plus.jsonl \
    --adj_path /tmp/empty_adj.json \
    --ckpt_dir experiments/nfcorpus/ckpt_relabel_mp3 \
    --log_dir experiments/nfcorpus/logs_relabel_mp3 \
    --num_neg 7 --epochs 10 --batch_size 16 --lr 2e-5 --seed 42

# scifact (10 epochs)
echo "=== scifact relabeled ==="
python src/train_with_adj.py \
    --train_path data/processed/scifact/train_relabeled_margin_persistent_3plus.jsonl \
    --adj_path /tmp/empty_adj.json \
    --ckpt_dir experiments/scifact/ckpt_relabel_mp3 \
    --log_dir experiments/scifact/logs_relabel_mp3 \
    --num_neg 7 --epochs 10 --batch_size 16 --lr 2e-5 --seed 42

echo "=== All relabel training done ==="

# Evaluate fiqa_rlhn: baseline vs mask vs relabel
echo "=== Evaluating fiqa_rlhn ==="
python src/evaluate_beir_raw.py --checkpoints \
    experiments/fiqa_rlhn/ckpt_baseline/epoch_4 \
    experiments/fiqa_rlhn/ckpt_margin_persistent_3plus/epoch_4 \
    experiments/fiqa_rlhn/ckpt_relabel_mp3/epoch_4 \
    --datasets fiqa --split test

# Evaluate fiqa
echo "=== Evaluating fiqa ==="
python src/evaluate_beir_raw.py --checkpoints \
    experiments/fiqa/ckpt_baseline/epoch_4 \
    experiments/fiqa/ckpt_margin_persistent_3plus/epoch_4 \
    experiments/fiqa/ckpt_relabel_mp3/epoch_4 \
    --datasets fiqa --split test

# Evaluate nfcorpus
echo "=== Evaluating nfcorpus ==="
python src/evaluate_beir_raw.py --checkpoints \
    experiments/nfcorpus/ckpt_baseline/epoch_9 \
    experiments/nfcorpus/ckpt_mask_fn/epoch_9 \
    experiments/nfcorpus/ckpt_relabel_mp3/epoch_9 \
    --datasets nfcorpus --split test

# Evaluate scifact
echo "=== Evaluating scifact ==="
python src/evaluate_beir_raw.py --checkpoints \
    experiments/scifact/ckpt_baseline/epoch_9 \
    experiments/scifact/ckpt_mask_fn/epoch_9 \
    experiments/scifact/ckpt_relabel_mp3/epoch_9 \
    --datasets scifact --split test
