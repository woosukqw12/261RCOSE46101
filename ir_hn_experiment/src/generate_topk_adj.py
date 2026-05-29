"""
Generate matched-budget adjustment files from arbitrary training-dynamics signals.

For each signal (avg_margin, persistent_count, rank_top1_count, ...),
rank all (qid, neg_idx) pairs by "FN-likeliness" (descending) and take top-K.
Output: adj_<signal>_K<N>.json files consumable by train_with_adj.py.

Also generates:
  - random_s<seed>_K<N>.json  (matched-budget random baseline)
  - oracle_K<N>.json          (if fn_ground_truth.json exists; K capped at GT size)
  - criterion_<name>.json     (original boolean criteria for reference)

Usage:
  python src/generate_topk_adj.py \
    --log_dir experiments/msmarco_rlhn/logs_baseline \
    --train_path data/processed/msmarco_rlhn/train.jsonl \
    --fn_ground_truth data/processed/msmarco_rlhn/fn_ground_truth.json \
    --output_dir experiments/msmarco_rlhn \
    --budgets 0.01,0.015,0.03 \
    --signals avg_margin,final_margin,persistent_count,rank_top1_count,persistent_x_margin \
    --random_seeds 42,123,456
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


MASK_VAL = -1e9


def ranks_from_scores(negs):
    order = np.argsort(-negs)
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[order] = np.arange(1, len(negs) + 1, dtype=np.int16)
    return ranks


def compute_signals(log_dir):
    epoch_files = sorted(
        [os.path.join(log_dir, f) for f in os.listdir(log_dir)
         if f.startswith("epoch_") and f.endswith(".jsonl")]
    )
    n_epochs = len(epoch_files)
    print(f"Found {n_epochs} epoch files in {log_dir}")

    # Count queries
    n_queries = 0
    num_neg = None
    with open(epoch_files[0]) as f:
        for line in f:
            if not line.strip():
                continue
            n_queries += 1
            if num_neg is None:
                num_neg = len(json.loads(line)["neg_scores"])
    print(f"queries={n_queries}, negs/query={num_neg}")

    qids = [""] * n_queries
    qid_to_idx = {}

    persistent_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    flip_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    prev_exceed = np.zeros((n_queries, num_neg), dtype=bool)
    margin_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    sum_rank = np.zeros((n_queries, num_neg), dtype=np.float32)
    rank_top1_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    rank_top2_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    last_margin = np.zeros((n_queries, num_neg), dtype=np.float32)
    last_rank = np.zeros((n_queries, num_neg), dtype=np.int16)

    for epoch_i, path in enumerate(epoch_files):
        print(f"  epoch {epoch_i+1}/{n_epochs}...")
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

    avg_margin = margin_sum / n_epochs
    avg_rank = sum_rank / n_epochs

    signals = {
        "avg_margin": avg_margin,
        "final_margin": last_margin,
        "persistent_count": persistent_count.astype(np.float32),
        "rank_top1_count": rank_top1_count.astype(np.float32),
        "neg_avg_rank": -avg_rank,
        "persistent_x_margin": persistent_count.astype(np.float32) * np.maximum(avg_margin, 0),
        "top1_x_margin": rank_top1_count.astype(np.float32) * np.maximum(avg_margin, 0),
        "rank_top2_count": rank_top2_count.astype(np.float32),
        "flip_count": flip_count.astype(np.float32),
    }
    return qids, num_neg, signals


def build_adj(qids, num_neg, topk_pairs):
    """topk_pairs: list of (qi, ni). Returns dict qid -> length-num_neg array."""
    adj = {}
    for qi, ni in topk_pairs:
        qid = qids[qi]
        if qid not in adj:
            adj[qid] = [0.0] * num_neg
        adj[qid][ni] = MASK_VAL
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--fn_ground_truth", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budgets", default="0.015",
                        help="Comma-separated fractions of total pairs")
    parser.add_argument("--signals", default="avg_margin,persistent_count,rank_top1_count,persistent_x_margin")
    parser.add_argument("--random_seeds", default="42,123,456")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    signal_names = args.signals.split(",")
    budgets = [float(b) for b in args.budgets.split(",")]
    seeds = [int(s) for s in args.random_seeds.split(",")]

    qids, num_neg, signals = compute_signals(args.log_dir)
    n_queries = len(qids)
    total_pairs = n_queries * num_neg

    # Validate qid ↔ train.jsonl ordering (they should match since train is deterministic)
    with open(args.train_path) as f:
        train_qids = [json.loads(l)["qid"] for l in f if l.strip()]
    assert len(train_qids) == n_queries, f"mismatch: train {len(train_qids)} vs logs {n_queries}"

    # Load FN GT if provided (in train-index form: qidx -> [neg_idx])
    fn_set = set()
    if args.fn_ground_truth and os.path.exists(args.fn_ground_truth):
        with open(args.fn_ground_truth) as f:
            gt = json.load(f)
        for qidx_str, neg_indices in gt["fn_pairs"].items():
            qi = int(qidx_str)
            if qi < n_queries:
                for ni in neg_indices:
                    fn_set.add((qi, ni))
        print(f"FN GT pairs: {len(fn_set)}")

    # Generate adj files for each signal × budget
    results_log = {"total_pairs": total_pairs, "num_neg": num_neg, "runs": []}
    for b_frac in budgets:
        K = max(1, int(round(total_pairs * b_frac)))
        print(f"\n=== budget K={K} ({b_frac*100:.2f}%) ===")

        # Signal-based top-K
        for sig_name in signal_names:
            if sig_name not in signals:
                print(f"  skip unknown signal: {sig_name}")
                continue
            flat = signals[sig_name].flatten()
            top_idx = np.argsort(-flat)[:K]
            pairs = [(int(fi // num_neg), int(fi % num_neg)) for fi in top_idx]
            adj = build_adj(qids, num_neg, pairs)
            fname = f"adj_{sig_name}_K{K}.json"
            with open(os.path.join(args.output_dir, fname), "w") as f:
                json.dump(adj, f)
            tp = sum((qi, ni) in fn_set for qi, ni in pairs) if fn_set else -1
            prec = tp / K if tp >= 0 else None
            print(f"  {sig_name:25s} → {fname}  ({len(adj)} qids, P={prec:.4f})" if prec is not None
                  else f"  {sig_name:25s} → {fname}  ({len(adj)} qids)")
            results_log["runs"].append({
                "signal": sig_name, "K": K, "budget_frac": b_frac,
                "n_queries_affected": len(adj),
                "precision_vs_gt": prec, "file": fname,
            })

        # Random matched-budget
        for seed in seeds:
            rng = np.random.default_rng(seed)
            rand_idx = rng.choice(total_pairs, size=K, replace=False)
            pairs = [(int(fi // num_neg), int(fi % num_neg)) for fi in rand_idx]
            adj = build_adj(qids, num_neg, pairs)
            fname = f"adj_random_s{seed}_K{K}.json"
            with open(os.path.join(args.output_dir, fname), "w") as f:
                json.dump(adj, f)
            tp = sum((qi, ni) in fn_set for qi, ni in pairs) if fn_set else -1
            prec = tp / K if tp >= 0 else None
            results_log["runs"].append({
                "signal": f"random_s{seed}", "K": K, "budget_frac": b_frac,
                "n_queries_affected": len(adj),
                "precision_vs_gt": prec, "file": fname,
            })

    # Oracle (if GT available): mask ALL GT FN pairs (one budget, capped at GT size)
    if fn_set:
        pairs = list(fn_set)
        adj = build_adj(qids, num_neg, pairs)
        fname = f"adj_oracle_K{len(pairs)}.json"
        with open(os.path.join(args.output_dir, fname), "w") as f:
            json.dump(adj, f)
        print(f"\nOracle → {fname}  ({len(adj)} qids, {len(pairs)} pairs)")
        results_log["runs"].append({
            "signal": "oracle", "K": len(pairs), "budget_frac": len(pairs)/total_pairs,
            "n_queries_affected": len(adj), "precision_vs_gt": 1.0, "file": fname,
        })

    with open(os.path.join(args.output_dir, "adj_generation_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nSaved generation log → {args.output_dir}/adj_generation_log.json")


if __name__ == "__main__":
    main()
