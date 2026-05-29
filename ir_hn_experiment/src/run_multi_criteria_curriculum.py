"""
Run curriculum learning with multiple criteria on fiqa_rlhn.
Reuses baseline from experiments/fiqa_rlhn, trains mask_fn for each criterion.
Evaluates all on BEIR fiqa test set.

python src/run_multi_criteria_curriculum.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

CRITERIA_TO_TEST = [
    "rank_persistent_top1",      # F1=0.163, 4527 pairs
    "rank_persistent_all_top1",  # F1=0.141, 1308 pairs
    "margin_persistent_3plus",   # F1=0.075, 597 pairs
    "cartography_hard",          # P=0.190, 84 pairs
    "rank_low_var_top",          # F1=0.160, 4388 pairs
]

LOG_DIR = "experiments/fiqa_rlhn/logs_baseline"
TRAIN_PATH = "data/processed/fiqa_rlhn/train.jsonl"
NUM_NEG = 7
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
MASK_VAL = -1e9


def get_criteria():
    """Compute all criteria from baseline logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "src/compute_signals.py",
            "--log_dir", LOG_DIR,
            "--output_dir", tmpdir,
            "--source", "loss",
        ]
        subprocess.run(cmd, check=True)
        with open(f"{tmpdir}/criteria_loss.json") as f:
            return json.load(f)


def build_adjustments(pairs, num_neg):
    """Build adjustment dict from (qid, neg_idx) pairs."""
    adjustments = {}
    for qid, neg_idx in pairs:
        if qid not in adjustments:
            adjustments[qid] = np.zeros(num_neg, dtype=np.float32)
        if neg_idx < num_neg:
            adjustments[qid][neg_idx] = MASK_VAL
    n_masked = sum(1 for a in adjustments.values() for v in a if v < -1e8)
    return adjustments, n_masked


def train_with_adj(name, adjustments):
    """Train model with adjustments, return checkpoint path."""
    ckpt_dir = f"experiments/fiqa_rlhn/ckpt_{name}"
    log_dir = f"experiments/fiqa_rlhn/logs_{name}"

    if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
        last = max(int(d.split("_")[1]) for d in os.listdir(ckpt_dir) if d.startswith("epoch_"))
        print(f"  [{name}] Skipping training (already exists)")
        return os.path.join(ckpt_dir, f"epoch_{last}")

    # Use run_dataset_experiment's train_model function via import
    # But simpler to just call inline since we need adjustments
    cmd = [
        sys.executable, "-c", f"""
import sys; sys.path.insert(0, '.')
from src.run_dataset_experiment import train_model
import json, numpy as np

# Load adjustments
adj = {json.dumps({k: v.tolist() for k, v in adjustments.items()})}
adj_np = {{k: np.array(v, dtype=np.float32) for k, v in adj.items()}}

ckpt = train_model(
    '{TRAIN_PATH}', '{log_dir}', '{ckpt_dir}',
    {NUM_NEG}, {EPOCHS}, {BATCH_SIZE}, {LR},
    adjustments=adj_np
)
print(f'DONE:{{ckpt}}')
"""
    ]
    # This won't work easily with inline code. Let's use a temp file instead.
    return None


def main():
    print("Computing criteria from baseline logs...")
    criteria = get_criteria()

    print(f"\n{'='*70}")
    print("Multi-criteria curriculum learning (fiqa_rlhn)")
    print(f"{'='*70}")

    # Save adjustments for each criterion, then train sequentially
    for crit_name in CRITERIA_TO_TEST:
        pairs = criteria.get(crit_name, [])
        if not pairs:
            print(f"\n  [{crit_name}] 0 pairs, skipping")
            continue

        adjustments, n_masked = build_adjustments(pairs, NUM_NEG)
        n_queries = len(adjustments)
        print(f"\n  [{crit_name}] {len(pairs)} pairs → {n_masked} masked ({n_queries} queries)")

        # Save adjustments to temp file
        adj_path = f"experiments/fiqa_rlhn/adj_{crit_name}.json"
        adj_json = {k: v.tolist() for k, v in adjustments.items()}
        with open(adj_path, "w") as f:
            json.dump(adj_json, f)

        ckpt_dir = f"experiments/fiqa_rlhn/ckpt_{crit_name}"
        log_dir = f"experiments/fiqa_rlhn/logs_{crit_name}"

        if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
            last = max(int(d.split("_")[1]) for d in os.listdir(ckpt_dir) if d.startswith("epoch_"))
            print(f"  Skipping training (checkpoint exists)")
            continue

        # Train via subprocess
        train_cmd = [
            sys.executable, "src/train_with_adj.py",
            "--train_path", TRAIN_PATH,
            "--adj_path", adj_path,
            "--ckpt_dir", ckpt_dir,
            "--log_dir", log_dir,
            "--num_neg", str(NUM_NEG),
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR),
        ]
        print(f"  Training...")
        subprocess.run(train_cmd, check=True)

    # Evaluate all checkpoints on BEIR fiqa
    print(f"\n{'='*70}")
    print("Evaluating all checkpoints on BEIR fiqa...")
    print(f"{'='*70}")

    ckpts = ["experiments/fiqa_rlhn/ckpt_baseline/epoch_4"]
    labels = ["baseline"]
    for crit_name in CRITERIA_TO_TEST:
        ckpt_dir = f"experiments/fiqa_rlhn/ckpt_{crit_name}"
        if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
            last = max(int(d.split("_")[1]) for d in os.listdir(ckpt_dir) if d.startswith("epoch_"))
            ckpts.append(os.path.join(ckpt_dir, f"epoch_{last}"))
            labels.append(crit_name)

    eval_cmd = [
        sys.executable, "src/evaluate_beir_raw.py",
        "--checkpoints", *ckpts,
        "--datasets", "fiqa",
        "--split", "test",
    ]
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
