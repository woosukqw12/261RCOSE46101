"""
Create relabeled training data: margin_persistent_3plus negatives → additional positives.

For each (query, negative) pair identified by margin_p3+,
create a new training instance where that negative becomes the positive.
The other negatives remain as negatives.

python src/prepare_relabeled_data.py --dataset fiqa_rlhn
python src/prepare_relabeled_data.py --dataset fiqa
python src/prepare_relabeled_data.py --dataset nfcorpus
python src/prepare_relabeled_data.py --dataset scifact
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--criterion", default="margin_persistent_3plus")
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--fn_ground_truth", default=None)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    if args.train_path and args.fn_ground_truth and args.output_path:
        create_from_fn_ground_truth(
            Path(args.train_path),
            Path(args.fn_ground_truth),
            Path(args.output_path),
        )
        return

    if not args.dataset:
        parser.error("Either --dataset or all of --train_path --fn_ground_truth --output_path is required")

    ds = args.dataset
    data_dir = Path(f"data/processed/{ds}")
    log_dir = Path(f"experiments/{ds}/logs_baseline")
    train_path = data_dir / "train.jsonl"

    # Compute criteria
    print(f"Computing signals for {ds}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run([
            sys.executable, "src/compute_signals.py",
            "--log_dir", str(log_dir),
            "--output_dir", tmpdir,
            "--source", "loss",
        ], check=True, capture_output=True)

        with open(f"{tmpdir}/criteria_loss.json") as f:
            criteria = json.load(f)

    pairs = criteria.get(args.criterion, [])
    print(f"  {args.criterion}: {len(pairs)} pairs")

    if not pairs:
        print("  No pairs found, skipping.")
        return

    # Build lookup: qid -> list of neg_indices to relabel
    relabel_map = {}
    for qid, neg_idx in pairs:
        relabel_map.setdefault(qid, []).append(neg_idx)

    # Load original training data
    rows = []
    with open(train_path) as f:
        for line in f:
            rows.append(json.loads(line))

    # Create augmented dataset
    augmented = []
    n_added = 0

    for row in rows:
        # Keep original instance as-is
        augmented.append(row)

        qid = row["qid"]
        if qid not in relabel_map:
            continue

        negs = row["negatives"]
        neg_ids = row.get("neg_ids", list(range(len(negs))))

        for neg_idx in relabel_map[qid]:
            if neg_idx >= len(negs):
                continue

            # New instance: FN negative becomes the positive
            new_row = {
                "qid": qid,
                "query": row["query"],
                "positives": [negs[neg_idx]],  # FN becomes positive
                "pos_ids": [neg_ids[neg_idx]],
                # Remove the relabeled neg from negatives
                "negatives": [n for i, n in enumerate(negs) if i != neg_idx],
                "neg_ids": [n for i, n in enumerate(neg_ids) if i != neg_idx],
            }
            augmented.append(new_row)
            n_added += 1

    # Save
    out_path = data_dir / f"train_relabeled_{args.criterion}.jsonl"
    with open(out_path, "w") as f:
        for row in augmented:
            f.write(json.dumps(row) + "\n")

    print(f"  Original: {len(rows)} instances")
    print(f"  Added: {n_added} relabeled instances")
    print(f"  Total: {len(augmented)} instances")
    print(f"  Saved → {out_path}")


def create_from_fn_ground_truth(train_path: Path, fn_ground_truth: Path, output_path: Path) -> None:
    """Create RLHN-style augmented data from indexed FN ground-truth pairs."""
    rows = []
    with train_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    with fn_ground_truth.open(encoding="utf-8") as f:
        gt = json.load(f)

    fn_pairs = {int(qidx): neg_indices for qidx, neg_indices in gt.get("fn_pairs", {}).items()}
    augmented = []
    n_added = 0

    for qidx, row in enumerate(rows):
        augmented.append(row)
        negs = row.get("negatives", [])
        neg_ids = row.get("neg_ids", list(range(len(negs))))

        for neg_idx in fn_pairs.get(qidx, []):
            if neg_idx >= len(negs):
                continue
            new_row = {
                "qid": row["qid"],
                "query": row["query"],
                "positives": [negs[neg_idx]],
                "pos_ids": [neg_ids[neg_idx]],
                "negatives": [n for i, n in enumerate(negs) if i != neg_idx],
                "neg_ids": [n for i, n in enumerate(neg_ids) if i != neg_idx],
            }
            augmented.append(new_row)
            n_added += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in augmented:
            f.write(json.dumps(row) + "\n")

    print(f"Original: {len(rows)} instances")
    print(f"Added: {n_added} relabeled instances")
    print(f"Total: {len(augmented)} instances")
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
