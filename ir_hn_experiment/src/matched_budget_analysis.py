"""
Matched-budget analysis: rank all signals by FN-ness, take top-K,
compare precision@K against FN ground truth.

Usage:
  python src/matched_budget_analysis.py \
    --log_dir experiments/fiqa_rlhn/logs_baseline \
    --fn_ground_truth results/fiqa_rlhn_fn_labels.json \
    --train_path data/processed/fiqa_rlhn/train.jsonl \
    --output results/matched_budget.json
"""

import argparse
import json
import os

import numpy as np


def ranks_from_scores(negs):
    order = np.argsort(-negs)
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[order] = np.arange(1, len(negs) + 1, dtype=np.int16)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--fn_ground_truth", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output", default="results/matched_budget.json")
    parser.add_argument("--budgets", default="100,300,597,1000,1500,3000,4500")
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]

    # Load FN ground truth
    with open(args.fn_ground_truth) as f:
        gt_data = json.load(f)

    # Build idx -> qid mapping
    idx_to_qid = {}
    with open(args.train_path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            idx_to_qid[i] = row["qid"]

    fn_set = set()
    for qidx_str, neg_indices in gt_data["fn_pairs"].items():
        qid = idx_to_qid.get(int(qidx_str))
        if qid:
            for ni in neg_indices:
                fn_set.add((qid, ni))
    print(f"FN ground truth: {len(fn_set)} pairs")

    # Load epoch files
    epoch_files = sorted(
        [os.path.join(args.log_dir, f) for f in os.listdir(args.log_dir)
         if f.startswith("epoch_") and f.endswith(".jsonl")]
    )
    n_epochs = len(epoch_files)
    print(f"Epochs: {n_epochs}")

    # First pass: count queries and negs
    n_queries = 0
    num_neg = None
    with open(epoch_files[0]) as f:
        for line in f:
            if not line.strip():
                continue
            n_queries += 1
            if num_neg is None:
                rec = json.loads(line)
                num_neg = len(rec["neg_scores"])
    print(f"Queries: {n_queries}, negs/query: {num_neg}")

    # Allocate arrays
    qids = [""] * n_queries
    qid_to_idx = {}

    persistent_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    flip_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    prev_exceed = np.zeros((n_queries, num_neg), dtype=bool)

    margin_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    rank_top1_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    rank_top2_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    sum_rank = np.zeros((n_queries, num_neg), dtype=np.float32)
    last_margin = np.zeros((n_queries, num_neg), dtype=np.float32)
    last_rank = np.zeros((n_queries, num_neg), dtype=np.int16)

    score_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    score_sq_sum = np.zeros((n_queries, num_neg), dtype=np.float32)

    # Process epochs
    for epoch_i, path in enumerate(epoch_files):
        print(f"  Processing {os.path.basename(path)}...")
        with open(path) as f:
            for line_i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = rec["query_id"]
                pos = float(rec["pos_score"])
                negs = np.asarray(rec["neg_scores"], dtype=np.float32)

                if epoch_i == 0:
                    qids[line_i] = qid
                    qid_to_idx[qid] = line_i

                idx = qid_to_idx.get(qid, line_i)
                margins = negs - pos
                exceeds = margins > 0
                ranks = ranks_from_scores(negs)

                persistent_count[idx] += exceeds.astype(np.uint8)
                if epoch_i > 0:
                    flip_count[idx] += (exceeds != prev_exceed[idx]).astype(np.uint8)
                prev_exceed[idx] = exceeds

                margin_sum[idx] += margins
                sum_rank[idx] += ranks.astype(np.float32)
                rank_top1_count[idx] += (ranks == 1).astype(np.uint8)
                rank_top2_count[idx] += (ranks <= 2).astype(np.uint8)
                last_margin[idx] = margins
                last_rank[idx] = ranks
                score_sum[idx] += negs
                score_sq_sum[idx] += negs ** 2

    avg_margin = margin_sum / n_epochs
    avg_rank = sum_rank / n_epochs
    avg_score = score_sum / n_epochs
    score_var = score_sq_sum / n_epochs - avg_score ** 2

    # Define ranking signals (higher = more FN-like)
    signals = {
        "persistent_count": persistent_count.astype(np.float32),
        "avg_margin": avg_margin,
        "final_margin": last_margin,
        "rank_top1_count": rank_top1_count.astype(np.float32),
        "neg_avg_rank": -avg_rank,  # negate: lower avg_rank = more FN-like
        "neg_final_rank": -last_rank.astype(np.float32),
        "rank_top2_count": rank_top2_count.astype(np.float32),
        "flip_count": flip_count.astype(np.float32),
        "score_variance": score_var,
        # Composite signals
        "persistent_x_margin": persistent_count.astype(np.float32) * np.maximum(avg_margin, 0),
        "top1_x_margin": rank_top1_count.astype(np.float32) * np.maximum(avg_margin, 0),
    }

    # Build flat (qid, neg_idx) pairs and check FN membership
    total_pairs = n_queries * num_neg
    is_fn = np.zeros(total_pairs, dtype=bool)
    pair_indices = []  # (query_idx, neg_idx)

    for qi in range(n_queries):
        for ni in range(num_neg):
            flat_idx = qi * num_neg + ni
            pair_indices.append((qi, ni))
            if (qids[qi], ni) in fn_set:
                is_fn[flat_idx] = True

    fn_total = int(is_fn.sum())
    fn_rate = fn_total / total_pairs
    print(f"\nFN in training data: {fn_total}/{total_pairs} ({fn_rate:.4f})")

    # For each signal, rank and compute precision@K
    results = {"fn_total": fn_total, "total_pairs": total_pairs, "fn_rate": fn_rate}
    results["budgets"] = budgets
    results["signals"] = {}

    for sig_name, sig_values in signals.items():
        flat_scores = sig_values.flatten()
        # Sort descending (higher = more FN-like)
        ranked_idx = np.argsort(-flat_scores)

        signal_results = {}
        for K in budgets:
            if K > total_pairs:
                K = total_pairs
            top_k = ranked_idx[:K]
            tp = int(is_fn[top_k].sum())
            precision = tp / K if K > 0 else 0
            recall = tp / fn_total if fn_total > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            signal_results[str(K)] = {
                "tp": tp, "precision": precision, "recall": recall, "f1": f1,
            }

        results["signals"][sig_name] = signal_results
        # Print summary at budget=597
        r597 = signal_results.get("597", {})
        print(f"  {sig_name:30s} P@597={r597.get('precision',0):.4f}  R={r597.get('recall',0):.4f}  F1={r597.get('f1',0):.4f}")

    # Random baseline
    rng = np.random.default_rng(42)
    random_results = {}
    n_trials = 100
    for K in budgets:
        if K > total_pairs:
            K = total_pairs
        tp_sum = 0
        for _ in range(n_trials):
            sampled = rng.choice(total_pairs, size=K, replace=False)
            tp_sum += int(is_fn[sampled].sum())
        avg_tp = tp_sum / n_trials
        precision = avg_tp / K
        recall = avg_tp / fn_total if fn_total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        random_results[str(K)] = {"tp": avg_tp, "precision": precision, "recall": recall, "f1": f1}
    results["signals"]["random"] = random_results
    r597 = random_results.get("597", {})
    print(f"  {'random':30s} P@597={r597.get('precision',0):.4f}  R={r597.get('recall',0):.4f}  F1={r597.get('f1',0):.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
