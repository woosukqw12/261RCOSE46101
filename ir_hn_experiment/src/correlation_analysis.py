"""
(B) Correlation analysis: do training-dynamics signals actually correlate with
    false-negative-ness (as judged by RLHN LLM-judge GT)?

Three analyses:
  1. AUC(signal → P(is_FN))      — per-pair classification performance
  2. Calibration curve            — P(FN | signal bin) vs signal bin
  3. Spearman rank correlation    — signal rank vs FN membership
  4. Information gain             — how much each signal adds on top of random

Usage:
  python src/correlation_analysis.py \
    --log_dir    experiments/fiqa_rlhn/logs_baseline \
    --train_path data/processed/fiqa_rlhn/train.jsonl \
    --fn_ground_truth data/processed/fiqa_rlhn/fn_ground_truth.json \
    --output_dir results/fiqa_rlhn_correlation
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score


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

    n_queries = 0
    num_neg = None
    with open(epoch_files[0]) as f:
        for line in f:
            if not line.strip():
                continue
            n_queries += 1
            if num_neg is None:
                num_neg = len(json.loads(line)["neg_scores"])

    qids = [""] * n_queries
    qid_to_idx = {}

    persistent_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    flip_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    prev_exceed = np.zeros((n_queries, num_neg), dtype=bool)
    margin_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    sum_rank = np.zeros((n_queries, num_neg), dtype=np.float32)
    rank_top1_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    last_margin = np.zeros((n_queries, num_neg), dtype=np.float32)
    last_rank = np.zeros((n_queries, num_neg), dtype=np.int16)
    m2_margin = np.zeros((n_queries, num_neg), dtype=np.float32)
    seen = np.zeros(n_queries, dtype=np.uint8)

    for epoch_i, path in enumerate(epoch_files):
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

                c = int(seen[idx]); new_c = c + 1
                delta = margins - (margin_sum[idx] / max(c, 1) if c > 0 else 0)
                # Welford for variance
                if c == 0:
                    mean_prev = np.zeros(num_neg, dtype=np.float32)
                else:
                    mean_prev = margin_sum[idx] / c
                margin_sum[idx] += margins
                mean_new = margin_sum[idx] / new_c
                m2_margin[idx] += (margins - mean_prev) * (margins - mean_new)

                persistent_count[idx] += exceeds.astype(np.uint8)
                if epoch_i > 0:
                    flip_count[idx] += (exceeds != prev_exceed[idx]).astype(np.uint8)
                prev_exceed[idx] = exceeds
                sum_rank[idx] += ranks.astype(np.float32)
                rank_top1_count[idx] += (ranks == 1).astype(np.uint8)
                last_margin[idx] = margins
                last_rank[idx] = ranks
                seen[idx] = new_c

    avg_margin = margin_sum / n_epochs
    avg_rank = sum_rank / n_epochs
    margin_variance = m2_margin / n_epochs

    signals = {
        "avg_margin": avg_margin,
        "final_margin": last_margin,
        "persistent_count": persistent_count.astype(np.float32),
        "rank_top1_count": rank_top1_count.astype(np.float32),
        "neg_avg_rank": -avg_rank,
        "margin_variance": margin_variance,
        "flip_count": flip_count.astype(np.float32),
        "persistent_x_margin": persistent_count.astype(np.float32) * np.maximum(avg_margin, 0),
    }
    return qids, num_neg, signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--fn_ground_truth", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    qids, num_neg, signals = compute_signals(args.log_dir)
    n_queries = len(qids)

    with open(args.fn_ground_truth) as f:
        gt = json.load(f)
    fn_mat = np.zeros((n_queries, num_neg), dtype=bool)
    for qi_str, neg_indices in gt["fn_pairs"].items():
        qi = int(qi_str)
        if qi < n_queries:
            for ni in neg_indices:
                fn_mat[qi, ni] = True

    y = fn_mat.flatten()
    base_rate = y.mean()
    print(f"FN base rate: {base_rate:.4f} ({int(y.sum())}/{len(y)})")

    results = {"base_rate": float(base_rate), "n_pairs": int(len(y)), "signals": {}}

    # 1. AUC / AP / Spearman
    print("\n=== Signal → FN classification performance ===")
    print(f"{'signal':25s}  {'AUC':>7}  {'AP':>7}  {'AP/base':>8}  {'Spearman':>9}")
    for name, arr in signals.items():
        x = arr.flatten().astype(np.float64)
        try:
            auc = roc_auc_score(y, x)
        except Exception:
            auc = float("nan")
        ap_score = average_precision_score(y, x)
        rho, _ = spearmanr(x, y)
        lift = ap_score / base_rate
        results["signals"][name] = {
            "auc": float(auc), "ap": float(ap_score),
            "ap_lift_over_base": float(lift), "spearman": float(rho),
        }
        print(f"{name:25s}  {auc:7.4f}  {ap_score:7.4f}  {lift:8.2f}x  {rho:9.4f}")

    # 2. Calibration curves (deciles)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    for i, (name, arr) in enumerate(signals.items()):
        ax = axes.flat[i]
        x = arr.flatten()
        # Bin by deciles of signal; filter ties
        bin_edges = np.quantile(x, np.linspace(0, 1, 11))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            ax.set_title(f"{name}\n(constant)")
            continue
        bin_idx = np.digitize(x, bin_edges[1:-1])
        bin_rates = []
        bin_counts = []
        bin_centers = []
        for b in range(len(bin_edges) - 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            bin_rates.append(y[mask].mean())
            bin_counts.append(int(mask.sum()))
            bin_centers.append(float(x[mask].mean()))
        ax.bar(range(len(bin_rates)), bin_rates, color="steelblue")
        ax.axhline(base_rate, color="red", linestyle="--", label=f"base={base_rate:.3f}")
        ax.set_title(f"{name}\n(AP={results['signals'][name]['ap']:.3f}, "
                     f"{results['signals'][name]['ap_lift_over_base']:.1f}×)")
        ax.set_xlabel("signal decile (low → high)")
        ax.set_ylabel("P(FN | bin)")
        ax.legend(fontsize=8)
        results["signals"][name]["calibration"] = {
            "bin_centers": bin_centers,
            "bin_rates": [float(r) for r in bin_rates],
            "bin_counts": bin_counts,
        }
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "calibration_curves.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved calibration figure → {fig_path}")

    # 3. Combined summary figure: AP lift comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results["signals"].keys())
    lifts = [results["signals"][n]["ap_lift_over_base"] for n in names]
    order = np.argsort(lifts)[::-1]
    names_sorted = [names[i] for i in order]
    lifts_sorted = [lifts[i] for i in order]
    colors = ["#2e7d32" if l > 2 else "#fbc02d" if l > 1 else "#c62828" for l in lifts_sorted]
    ax.barh(names_sorted, lifts_sorted, color=colors)
    ax.axvline(1, color="black", linestyle="--", label="random (AP/base=1)")
    ax.set_xlabel("AP lift over base rate (higher = better FN detector)")
    ax.set_title("How well each training-dynamics signal predicts FN\n(fiqa_rlhn; GT from RLHN LLM-judge)")
    ax.legend()
    plt.tight_layout()
    lift_path = os.path.join(args.output_dir, "ap_lift.png")
    plt.savefig(lift_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved AP lift figure → {lift_path}")

    out_json = os.path.join(args.output_dir, "correlation_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics → {out_json}")


if __name__ == "__main__":
    main()
