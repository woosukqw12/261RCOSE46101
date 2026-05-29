"""
(C) Per-pair loss trajectory figure — Arpit+17 style.

Plots mean margin (neg_score - pos_score) across training epochs for four groups:
  (a) Clean negatives (GT says NOT FN)
  (b) Ground-truth FN (from RLHN LLM-judge)
  (c) Our TP: dynamics-flagged AND GT FN
  (d) Our FP: dynamics-flagged but GT says NOT FN

The theoretical prediction (Arpit+17, Liu+20):
  - Clean negs: margin starts near zero and drops strongly negative (model quickly
    learns to push them away).
  - FN: margin stays positive or only weakly negative — model CANNOT push them away
    without sacrificing the positive.
  - Our TP should mirror (b); our FP should be intermediate.

If these three curves are visibly separated, it validates that the signature
predicted by memorization theory appears in dense retrieval training dynamics.

Usage:
  python src/loss_trajectory_figure.py \
      --log_dir experiments/fiqa_rlhn/logs_baseline \
      --train_path data/processed/fiqa_rlhn/train.jsonl \
      --fn_ground_truth data/processed/fiqa_rlhn/fn_ground_truth.json \
      --criteria_path results/signals_fiqa_rlhn/criteria_loss.json \
      --output_dir results/fiqa_rlhn_trajectories
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_trajectories(log_dir):
    """Return (qids, qid_to_idx, num_neg, epoch_margins [n_epoch, n_q, n_neg])."""
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
    margins = np.zeros((n_epochs, n_queries, num_neg), dtype=np.float32)

    for ei, path in enumerate(epoch_files):
        with open(path) as f:
            for li, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = rec["query_id"]
                if ei == 0:
                    qids[li] = qid
                    qid_to_idx[qid] = li
                idx = qid_to_idx.get(qid, li)
                pos = float(rec["pos_score"])
                negs = np.asarray(rec["neg_scores"], dtype=np.float32)
                margins[ei, idx] = negs - pos

    return qids, qid_to_idx, num_neg, margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--fn_ground_truth", required=True)
    ap.add_argument("--criteria_path", default=None,
                    help="Optional: results/signals_<ds>/criteria_loss.json to show our flagged pairs")
    ap.add_argument("--criterion_name", default="margin_persistent_3plus")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    qids, qid_to_idx, num_neg, margins = load_trajectories(args.log_dir)
    n_epochs, n_queries, _ = margins.shape
    print(f"Loaded {n_epochs} epochs × {n_queries} queries × {num_neg} negs")

    # Load FN GT (indices are train-order query indices)
    with open(args.fn_ground_truth) as f:
        gt = json.load(f)
    fn_mat = np.zeros((n_queries, num_neg), dtype=bool)
    for qi_str, neg_indices in gt["fn_pairs"].items():
        qi = int(qi_str)
        if qi < n_queries:
            for ni in neg_indices:
                fn_mat[qi, ni] = True
    print(f"GT FN pairs: {int(fn_mat.sum())}")

    # Load our flagged pairs (criterion uses qid hash)
    flagged_mat = np.zeros((n_queries, num_neg), dtype=bool)
    crit_name = args.criterion_name
    if args.criteria_path and os.path.exists(args.criteria_path):
        with open(args.criteria_path) as f:
            crit = json.load(f)
        pairs = crit.get(crit_name, [])
        for qid, ni in pairs:
            if qid in qid_to_idx:
                flagged_mat[qid_to_idx[qid], ni] = True
        print(f"Our flagged ({crit_name}): {int(flagged_mat.sum())}")

    # Define four groups
    groups = {
        "Clean neg (GT)":       (~fn_mat) & (~flagged_mat),
        "GT FN (all)":          fn_mat,
        "Ours TP (flag ∩ GT)":  flagged_mat & fn_mat,
        "Ours FP (flag ∩ ¬GT)": flagged_mat & (~fn_mat),
    }

    # Compute per-epoch mean + bootstrap 95% CI for each group
    rng = np.random.default_rng(0)
    epochs = np.arange(n_epochs)
    group_stats = {}
    for name, mask in groups.items():
        data = margins[:, mask]  # [n_epoch, n_pairs_in_group]
        if data.shape[1] == 0:
            continue
        mean = data.mean(axis=1)
        # Bootstrap CI over pairs
        n_boot = 200
        boot_means = np.zeros((n_boot, n_epochs))
        for b in range(n_boot):
            idx = rng.integers(0, data.shape[1], size=data.shape[1])
            boot_means[b] = data[:, idx].mean(axis=1)
        low = np.percentile(boot_means, 2.5, axis=0)
        high = np.percentile(boot_means, 97.5, axis=0)
        group_stats[name] = {
            "count": int(data.shape[1]),
            "mean": mean.tolist(),
            "ci_low": low.tolist(),
            "ci_high": high.tolist(),
        }

    # Plot
    colors = {
        "Clean neg (GT)":       "#2e7d32",
        "GT FN (all)":          "#c62828",
        "Ours TP (flag ∩ GT)":  "#f57c00",
        "Ours FP (flag ∩ ¬GT)": "#1565c0",
    }
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, stats in group_stats.items():
        col = colors.get(name, "gray")
        mean = np.array(stats["mean"])
        low = np.array(stats["ci_low"])
        high = np.array(stats["ci_high"])
        ax.plot(epochs, mean, marker="o", color=col,
                label=f"{name}  (n={stats['count']})", linewidth=2)
        ax.fill_between(epochs, low, high, color=col, alpha=0.15)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean margin  (neg_score − pos_score)\n← negative = model correctly ranks pos above neg")
    ax.set_title("Per-pair margin trajectories during baseline training\n"
                 "(fiqa_rlhn;  memorization-effect prediction from Arpit+17, Liu+20)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "loss_trajectories.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory figure → {fig_path}")

    # Save raw numbers for paper
    out_json = os.path.join(args.output_dir, "trajectory_stats.json")
    with open(out_json, "w") as f:
        json.dump({
            "n_epochs": n_epochs, "n_queries": n_queries, "num_neg": num_neg,
            "criterion": crit_name,
            "groups": group_stats,
        }, f, indent=2)
    print(f"Saved stats → {out_json}")

    # Also plot per-group margin distributions at first and last epoch (histograms)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for col_i, ep_i in enumerate([0, n_epochs - 1]):
        ax = axes[col_i]
        for name, mask in groups.items():
            if mask.sum() == 0:
                continue
            vals = margins[ep_i, mask]
            ax.hist(vals, bins=50, alpha=0.4, label=f"{name} (n={int(mask.sum())})",
                    color=colors.get(name, "gray"), density=True)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"epoch {ep_i} margin distribution")
        ax.set_xlabel("margin")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    plt.tight_layout()
    hist_path = os.path.join(args.output_dir, "margin_distributions.png")
    plt.savefig(hist_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved distribution figure → {hist_path}")


if __name__ == "__main__":
    main()
