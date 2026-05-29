"""
Stability analysis: are the FN-candidate pairs identified by training dynamics
stable across random seeds? (Hyperparam variation handled by a wrapper script.)

For each signal:
  1. Compute signal values from each seed's epoch logs
  2. Extract top-K pairs (matched budget)
  3. Compute pairwise Jaccard overlap between seeds
  4. Compute Spearman rank correlation of signal values across seeds
  5. Compute per-seed P@K vs GT FN (does precision stay consistent?)

Interpretation:
  - High Jaccard (>0.5) + high Spearman (>0.6): signal captures data-intrinsic property
  - Low Jaccard (<0.2) + low Spearman: signal is optimization-noise dependent

Usage:
  python src/stability_analysis.py \
      --log_dirs experiments/fiqa_rlhn/logs_baseline \
                 experiments/fiqa_rlhn/logs_baseline_s123 \
                 experiments/fiqa_rlhn/logs_baseline_s456 \
      --seed_labels 42 123 456 \
      --train_path data/processed/fiqa_rlhn/train.jsonl \
      --fn_ground_truth data/processed/fiqa_rlhn/fn_ground_truth.json \
      --output_dir results/fiqa_rlhn_stability \
      --budget_frac 0.015
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
    margin_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    rank_top1_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    last_margin = np.zeros((n_queries, num_neg), dtype=np.float32)

    for ei, path in enumerate(epoch_files):
        with open(path) as f:
            for li, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = rec["query_id"]
                pos = float(rec["pos_score"])
                negs = np.asarray(rec["neg_scores"], dtype=np.float32)
                if ei == 0:
                    qids[li] = qid
                    qid_to_idx[qid] = li
                idx = qid_to_idx.get(qid, li)
                margins = negs - pos
                exceeds = margins > 0
                ranks = ranks_from_scores(negs)
                persistent_count[idx] += exceeds.astype(np.uint8)
                margin_sum[idx] += margins
                rank_top1_count[idx] += (ranks == 1).astype(np.uint8)
                last_margin[idx] = margins

    avg_margin = margin_sum / n_epochs
    signals = {
        "avg_margin": avg_margin,
        "final_margin": last_margin,
        "persistent_count": persistent_count.astype(np.float32),
        "rank_top1_count": rank_top1_count.astype(np.float32),
    }
    return qids, num_neg, signals


def topk_set_by_qid(signal_arr, qids, num_neg, K):
    """Return set of (qid_str, neg_idx) tuples for top-K pairs by signal value (desc)."""
    flat = signal_arr.flatten()
    top = np.argsort(-flat)[:K]
    out = set()
    for fi in top:
        qi, ni = int(fi // num_neg), int(fi % num_neg)
        out.add((qids[qi], ni))
    return out


def signal_by_qid(signal_arr, qids, num_neg):
    """Return dict (qid_str, neg_idx) -> score."""
    d = {}
    for qi, qid in enumerate(qids):
        for ni in range(num_neg):
            d[(qid, ni)] = float(signal_arr[qi, ni])
    return d


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dirs", nargs="+", required=True)
    ap.add_argument("--seed_labels", nargs="+", required=True)
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--fn_ground_truth", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--budget_frac", type=float, default=0.015)
    args = ap.parse_args()

    assert len(args.log_dirs) == len(args.seed_labels)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Compute signals per seed
    per_seed = {}
    for seed, ld in zip(args.seed_labels, args.log_dirs):
        print(f"\n[seed {seed}] computing signals from {ld}...")
        qids, num_neg, signals = compute_signals(ld)
        per_seed[seed] = {"qids": qids, "num_neg": num_neg, "signals": signals}

    # Build FN GT set keyed by (qid_str, neg_idx) — train.jsonl's int index → qid
    n_queries = len(per_seed[args.seed_labels[0]]["qids"])
    num_neg = per_seed[args.seed_labels[0]]["num_neg"]

    idx_to_qid = {}
    with open(args.train_path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            idx_to_qid[i] = row["qid"]
    with open(args.fn_ground_truth) as f:
        gt = json.load(f)
    fn_set_qid = set()
    for qi_str, neg_indices in gt["fn_pairs"].items():
        qid = idx_to_qid.get(int(qi_str))
        if qid is not None:
            for ni in neg_indices:
                fn_set_qid.add((qid, ni))
    print(f"GT FN pairs (qid-keyed): {len(fn_set_qid)}")

    total_pairs = n_queries * num_neg
    K = int(round(total_pairs * args.budget_frac))
    print(f"Budget K={K}  ({args.budget_frac*100:.2f}% of {total_pairs})")

    # Random expected Jaccard for sanity baseline
    # E[|A∩B|] = K*K/N, E[|A∪B|] = 2K - K*K/N
    rand_j = (K * K / total_pairs) / (2 * K - K * K / total_pairs)
    print(f"Random-expected Jaccard at K={K}: {rand_j:.4f}")

    signal_names = list(per_seed[args.seed_labels[0]]["signals"].keys())
    results = {
        "K": K, "budget_frac": args.budget_frac,
        "total_pairs": total_pairs,
        "random_expected_jaccard": rand_j,
        "per_signal": {},
    }

    for sig in signal_names:
        print(f"\n=== Signal: {sig} ===")
        topk_sets = {
            s: topk_set_by_qid(per_seed[s]["signals"][sig],
                               per_seed[s]["qids"], num_neg, K)
            for s in args.seed_labels
        }

        jaccard_pairs = {}
        for i, s1 in enumerate(args.seed_labels):
            for s2 in args.seed_labels[i+1:]:
                j = jaccard(topk_sets[s1], topk_sets[s2])
                jaccard_pairs[f"{s1}↔{s2}"] = j
                print(f"  Jaccard({s1} vs {s2}): {j:.4f}")
        mean_j = float(np.mean(list(jaccard_pairs.values())))

        core = set.intersection(*topk_sets.values())
        core_frac = len(core) / K
        print(f"  Core set (∩ all {len(args.seed_labels)} seeds): {len(core)} / {K}  ({core_frac:.2%})")

        # Spearman on signal values, aligned via (qid, ni) keys
        sig_dicts = {
            s: signal_by_qid(per_seed[s]["signals"][sig],
                             per_seed[s]["qids"], num_neg)
            for s in args.seed_labels
        }
        # Common keys (should match since all seeds see same train set)
        keys = sorted(set.intersection(*[set(d.keys()) for d in sig_dicts.values()]))
        spearman_pairs = {}
        for i, s1 in enumerate(args.seed_labels):
            for s2 in args.seed_labels[i+1:]:
                v1 = np.array([sig_dicts[s1][k] for k in keys])
                v2 = np.array([sig_dicts[s2][k] for k in keys])
                rho, _ = spearmanr(v1, v2)
                spearman_pairs[f"{s1}↔{s2}"] = float(rho)
                print(f"  Spearman({s1} vs {s2}): {rho:.4f}")
        mean_rho = float(np.mean(list(spearman_pairs.values())))

        per_seed_pk = {}
        for s in args.seed_labels:
            tp = sum(1 for pair in topk_sets[s] if pair in fn_set_qid)
            p = tp / max(K, 1)
            per_seed_pk[s] = p
            print(f"  P@K (seed {s}): {p:.4f}")
        pk_std = float(np.std(list(per_seed_pk.values())))

        core_tp = sum(1 for pair in core if pair in fn_set_qid)
        core_pk = core_tp / max(len(core), 1)
        print(f"  Core set P (intersection across seeds): {core_pk:.4f}  "
              f"(TP={core_tp}/{len(core)})")

        results["per_signal"][sig] = {
            "mean_jaccard": mean_j,
            "jaccard_pairs": jaccard_pairs,
            "mean_spearman": mean_rho,
            "spearman_pairs": spearman_pairs,
            "core_size": len(core),
            "core_fraction_of_K": core_frac,
            "per_seed_pk": per_seed_pk,
            "pk_std": pk_std,
            "core_precision": core_pk,
        }

    # Bar plot: mean Jaccard per signal, with random baseline line
    fig, ax = plt.subplots(figsize=(8, 5))
    sig_order = sorted(signal_names, key=lambda s: -results["per_signal"][s]["mean_jaccard"])
    mean_js = [results["per_signal"][s]["mean_jaccard"] for s in sig_order]
    colors = ["#2e7d32" if j > 0.5 else "#fbc02d" if j > 0.3 else "#c62828" for j in mean_js]
    ax.bar(sig_order, mean_js, color=colors)
    ax.axhline(rand_j, color="black", linestyle="--",
               label=f"random baseline ({rand_j:.3f})")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="0.5 threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean pairwise Jaccard overlap across seeds")
    ax.set_title(f"Seed-stability of top-K selections  (K={K}, {args.budget_frac*100:.1f}%, "
                 f"{len(args.seed_labels)} seeds)")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "stability_jaccard.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Jaccard bar → {fig_path}")

    # Per-seed P@K bar plot
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sig_order))
    w = 0.25
    for i, s in enumerate(args.seed_labels):
        vals = [results["per_signal"][sig]["per_seed_pk"][s] for sig in sig_order]
        ax.bar(x + (i - 1) * w, vals, w, label=f"seed {s}")
    # Core precision as black line
    core_p = [results["per_signal"][sig]["core_precision"] for sig in sig_order]
    ax.plot(x, core_p, "k^-", label="core-set precision", markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(sig_order, rotation=20, ha="right")
    ax.set_ylabel(f"P@K={K}  (vs RLHN GT FN)")
    ax.set_title("Per-seed precision + intersection-set precision")
    ax.legend()
    plt.tight_layout()
    fig2 = os.path.join(args.output_dir, "stability_pk.png")
    plt.savefig(fig2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved P@K bar → {fig2}")

    out_json = os.path.join(args.output_dir, "stability_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics → {out_json}")


if __name__ == "__main__":
    main()
