#!/bin/bash
set -e

echo "=============================================="
echo "H1 Full Verification Pipeline"
echo "=============================================="
echo ""
echo "Prerequisites:"
echo "  pip install torch transformers datasets tqdm scikit-learn matplotlib numpy"
echo ""
echo "Data: rlhn/default-680K and rlhn/rlhn-680K are loaded directly from HuggingFace."
echo "      No manual download needed. First run will cache automatically."
echo ""
echo "Signals to evaluate:"
echo "  Margin: final, avg, persistent, trend, nondecreasing"
echo "  Rank: final, avg, variance, persistence (top1/top2), low_var"
echo "  Forgetting: flip count, cartography (easy/medium/ambiguous/hard)"
echo "  Embedding: velocity, displacement"
echo "  Composite: margin+rank, hard+persistent, all_strict"
echo ""
echo "Sources: (A) Loss logs (stochastic) vs (B) Re-encoding (deterministic)"
echo ""

mkdir -p logs/training_scores logs/reencode_scores results/signals results/h1_comparison

# Step 1: Train + log scores
echo "[1/4] Training E5-base with score logging (~3h on RTX 5090)..."
python src/train_with_logging.py

# Step 2: Re-encode with each checkpoint
echo ""
echo "[2/4] Re-encoding with epoch checkpoints (~15min)..."
python src/reencode_checkpoints.py

# Step 3: Compute all signals
echo ""
echo "[3/4] Computing all training dynamics signals..."
python src/compute_signals.py

# Step 4: Compare with RLHN
echo ""
echo "[4/4] Comparing all signals against RLHN labels..."
python src/compare_all_signals.py

echo ""
echo "=============================================="
echo "Done! Check results/h1_comparison/full_comparison.json"
echo "=============================================="
