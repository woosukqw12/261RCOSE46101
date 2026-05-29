"""Exp 15 sequential diagnostics — (α) score-margin AUC + (γ) oracle conditional NDCG.

본 script 는 Exp 15 (Conditional LoRA) 의 *foundation 검증* 을 위한 cheap (학습 없음) diagnostic 2 개:

  (α) Score-margin AUC — frozen ColBERT 의 top-1/top-2 score margin 이 confused 를
      qrels-free 로 predict 할 수 있는가? Exp 15 의 router signal 의 *upper-bound predictability*.

  (γ) Oracle test-time conditional — Phase 2b LoRA × frozen 의 perfect-routing 조합 (gold label).
      Exp 15 의 *test-time ceiling* — perfect router 가 있으면 frontier 어디까지 가능?

Output:
  report/figures/_exp15_diagnostics/{
    diagnostic_alpha.json,        # (α) AUC + diagnostics
    diagnostic_gamma.json,        # (γ) oracle conditional NDCG
    diagnostic_alpha_gamma.{pdf,png}  # combined figure
  }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_beir  # noqa: E402
from src.metrics import paired_bootstrap_ci  # noqa: E402


DATASET = "scifact"
SEEDS = [42, 1337, 2024]
BOOT_N = 10000
ALPHA = 0.05


def auc_score(y_true, y_score):
    """Compute AUC manually (avoid sklearn dependency).

    y_true: binary array [0, 1]
    y_score: continuous array (higher = predict y=1)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank-based AUC
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    # for ties, average rank
    ranks = np.zeros_like(y_score, dtype=float)
    n = len(y_score)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # 1-indexed rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def ndcg_at_k(ranked_dids, qrels_q, k=10):
    """Compute NDCG@k for a single query."""
    rels = [qrels_q.get(d, 0) for d in ranked_dids[:k]]
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    sorted_rels = sorted(qrels_q.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(sorted_rels))
    return dcg / idcg if idcg > 0 else 0.0


def compute_alpha(split: str, runs_scored_path: Path, qrels: dict):
    """(α) Score-margin AUC diagnostic.

    For each query:
      - top-1 score (s1), top-2 score (s2), margin = s1 - s2
      - is_confused = (top-1 did is not in qrels[qid] with rel >= 1)
    AUC: predictor = -margin (low margin → confused), target = is_confused.
    """
    runs_scored = json.loads(runs_scored_path.read_text())
    margins = []
    is_confused = []
    top1_scores = []
    qids_processed = []
    for qid, scored in runs_scored.items():
        if qid not in qrels:
            continue
        rel_set = {d for d, r in qrels[qid].items() if r >= 1}
        # scored is dict {did: score}; sort by score desc
        sorted_pairs = sorted(scored.items(), key=lambda x: -x[1])
        if len(sorted_pairs) < 2:
            continue
        s1, s2 = sorted_pairs[0][1], sorted_pairs[1][1]
        top1_did = sorted_pairs[0][0]
        margins.append(s1 - s2)
        top1_scores.append(s1)
        is_confused.append(0 if top1_did in rel_set else 1)
        qids_processed.append(qid)
    margins = np.array(margins)
    is_confused = np.array(is_confused)
    top1_scores = np.array(top1_scores)

    # AUC: low margin → confused, so predictor for "confused" is -margin
    auc_margin = auc_score(is_confused, -margins)
    # alt predictor: top1 score (lower = less confident → confused)
    auc_top1 = auc_score(is_confused, -top1_scores)

    return {
        "split": split,
        "n_queries": len(margins),
        "n_confused": int(is_confused.sum()),
        "n_easy": int((is_confused == 0).sum()),
        "frac_confused": float(is_confused.mean()),
        "auc_margin_predicts_confused": auc_margin,
        "auc_top1_predicts_confused": auc_top1,
        "margin_stats": {
            "mean": float(margins.mean()),
            "std": float(margins.std()),
            "median": float(np.median(margins)),
            "min": float(margins.min()),
            "max": float(margins.max()),
        },
        "margin_by_class": {
            "confused_mean": float(margins[is_confused == 1].mean()) if is_confused.sum() > 0 else None,
            "confused_median": float(np.median(margins[is_confused == 1])) if is_confused.sum() > 0 else None,
            "easy_mean": float(margins[is_confused == 0].mean()) if (is_confused == 0).sum() > 0 else None,
            "easy_median": float(np.median(margins[is_confused == 0])) if (is_confused == 0).sum() > 0 else None,
        },
        # raw arrays for plotting (kept compact)
        "_margins_confused": margins[is_confused == 1].tolist()[:1000],
        "_margins_easy": margins[is_confused == 0].tolist()[:1000],
    }


def compute_gamma(seed: int, qrels: dict):
    """(γ) Oracle test-time conditional NDCG.

    For each test query:
      - if confused (gold label): use Phase 2b LoRA ranking
      - if easy (gold label): use frozen ranking
    Compute NDCG@10 per query, then paired bootstrap CI vs frozen baseline.
    """
    baseline_runs = json.loads(
        (PROJECT_ROOT / "outputs/00_baseline" / DATASET / f"seed_{seed}" / "runs.json").read_text()
    )
    lora_runs = json.loads(
        (PROJECT_ROOT / "outputs/10_lora_phi" / DATASET / f"seed_{seed}" / "qv_r8_l12" / "runs.json").read_text()
    )

    baseline_ndcg = {}
    lora_ndcg = {}
    oracle_ndcg = {}
    is_confused = {}

    for qid in baseline_runs:
        if qid not in qrels:
            continue
        rel_set = {d for d, r in qrels[qid].items() if r >= 1}
        if not rel_set:
            continue
        baseline_ranked = baseline_runs[qid]  # list of dids (already sorted)
        lora_ranked = lora_runs.get(qid, baseline_ranked)
        # confused = baseline top-1 not in rel_set
        top1_baseline = baseline_ranked[0] if baseline_ranked else None
        confused = top1_baseline not in rel_set
        is_confused[qid] = confused

        b_n = ndcg_at_k(baseline_ranked, qrels[qid], k=10)
        l_n = ndcg_at_k(lora_ranked, qrels[qid], k=10)
        # oracle conditional: use LoRA for confused, frozen for easy
        o_n = l_n if confused else b_n

        baseline_ndcg[qid] = b_n
        lora_ndcg[qid] = l_n
        oracle_ndcg[qid] = o_n

    qids = sorted(baseline_ndcg.keys())
    b_arr = np.array([baseline_ndcg[q] for q in qids])
    l_arr = np.array([lora_ndcg[q] for q in qids])
    o_arr = np.array([oracle_ndcg[q] for q in qids])
    confused_mask = np.array([is_confused[q] for q in qids], dtype=bool)
    easy_mask = ~confused_mask

    # paired bootstrap CIs
    def boot_ci(deltas, n=BOOT_N, alpha=ALPHA, seed_b=42):
        rng = np.random.default_rng(seed_b)
        n_q = len(deltas)
        boots = []
        for _ in range(n):
            idx = rng.integers(0, n_q, size=n_q)
            boots.append(deltas[idx].mean())
        boots = np.array(boots)
        return float(boots.mean()), float(np.percentile(boots, alpha/2*100)), float(np.percentile(boots, (1-alpha/2)*100))

    def slice_stats(slice_mask, label):
        d_oracle = (o_arr - b_arr)[slice_mask]
        d_lora = (l_arr - b_arr)[slice_mask]
        if len(d_oracle) == 0:
            return None
        m_o, lo_o, hi_o = boot_ci(d_oracle)
        m_l, lo_l, hi_l = boot_ci(d_lora)
        return {
            "n": int(slice_mask.sum()),
            "oracle_mean_delta": m_o, "oracle_ci": [lo_o, hi_o],
            "phase2b_mean_delta": m_l, "phase2b_ci": [lo_l, hi_l],
            "ndcg_baseline_mean": float(b_arr[slice_mask].mean()),
            "ndcg_lora_mean": float(l_arr[slice_mask].mean()),
            "ndcg_oracle_mean": float(o_arr[slice_mask].mean()),
        }

    return {
        "seed": seed,
        "n_queries": len(qids),
        "n_confused": int(confused_mask.sum()),
        "n_easy": int(easy_mask.sum()),
        "all": slice_stats(np.ones_like(confused_mask, dtype=bool), "all"),
        "confused": slice_stats(confused_mask, "confused"),
        "easy": slice_stats(easy_mask, "easy"),
    }


def main():
    out_dir = PROJECT_ROOT / "report/figures/_exp15_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============ (α) Score-margin AUC ============
    print("=" * 80)
    print("(α) Score-margin AUC diagnostic — frozen 의 top-1/top-2 margin 의 confused predictability")
    print("=" * 80)

    # Test split (300 queries) — uses 00_baseline/seed_42/runs_scored.json
    print("\nloading scifact test qrels...")
    _, _, qrels_test = load_beir(DATASET, split="test")
    alpha_test = compute_alpha(
        "test",
        PROJECT_ROOT / "outputs/00_baseline" / DATASET / "seed_42" / "runs_scored.json",
        qrels_test,
    )

    # Train split — need to encode train queries with frozen ColBERT (not pre-cached)
    # Skip train for now since it would require new encoding; test set is sufficient signal.
    # If needed, run separately with src.evaluate.

    alpha_results = {"test": alpha_test}
    (out_dir / "diagnostic_alpha.json").write_text(json.dumps(alpha_results, indent=2))

    print(f"\n[test] n={alpha_test['n_queries']}, confused={alpha_test['n_confused']} "
          f"({alpha_test['frac_confused']:.1%})")
    print(f"  AUC(margin → confused)    = {alpha_test['auc_margin_predicts_confused']:.4f}")
    print(f"  AUC(top1_score → confused) = {alpha_test['auc_top1_predicts_confused']:.4f}")
    print(f"  margin (confused queries): mean={alpha_test['margin_by_class']['confused_mean']:.4f}, "
          f"median={alpha_test['margin_by_class']['confused_median']:.4f}")
    print(f"  margin (easy queries):     mean={alpha_test['margin_by_class']['easy_mean']:.4f}, "
          f"median={alpha_test['margin_by_class']['easy_median']:.4f}")

    # ============ (γ) Oracle test-time conditional ============
    print("\n" + "=" * 80)
    print("(γ) Oracle test-time conditional NDCG — perfect routing ceiling")
    print("=" * 80)

    gamma_results = {"seeds": {}}
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        r = compute_gamma(seed, qrels_test)
        gamma_results["seeds"][seed] = r
        a = r["all"]
        c = r["confused"]
        e = r["easy"]
        print(f"  n_queries={r['n_queries']} (confused={r['n_confused']}, easy={r['n_easy']})")
        print(f"  ALL  : oracle Δ = {a['oracle_mean_delta']:+.4f} {a['oracle_ci']} | "
              f"Phase 2b Δ = {a['phase2b_mean_delta']:+.4f} {a['phase2b_ci']}")
        print(f"  CONF : oracle Δ = {c['oracle_mean_delta']:+.4f} {c['oracle_ci']} | "
              f"Phase 2b Δ = {c['phase2b_mean_delta']:+.4f} {c['phase2b_ci']}")
        print(f"  EASY : oracle Δ = {e['oracle_mean_delta']:+.4f} {e['oracle_ci']} | "
              f"Phase 2b Δ = {e['phase2b_mean_delta']:+.4f} {e['phase2b_ci']}")

    # 3-seed mean
    for slc in ("all", "confused", "easy"):
        vals_oracle = [gamma_results["seeds"][s][slc]["oracle_mean_delta"] for s in SEEDS]
        vals_phase2b = [gamma_results["seeds"][s][slc]["phase2b_mean_delta"] for s in SEEDS]
        gamma_results.setdefault("3seed_mean", {})[slc] = {
            "oracle_mean": float(np.mean(vals_oracle)),
            "oracle_std": float(np.std(vals_oracle, ddof=1)),
            "phase2b_mean": float(np.mean(vals_phase2b)),
            "phase2b_std": float(np.std(vals_phase2b, ddof=1)),
        }
    (out_dir / "diagnostic_gamma.json").write_text(json.dumps(gamma_results, indent=2))

    print("\n=== 3-seed mean ± std ===")
    for slc in ("all", "confused", "easy"):
        s = gamma_results["3seed_mean"][slc]
        print(f"Δ {slc}: oracle = {s['oracle_mean']:+.4f} ± {s['oracle_std']:.4f} | "
              f"Phase 2b = {s['phase2b_mean']:+.4f} ± {s['phase2b_std']:.4f}")

    # ============ Figure ============
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: margin distribution by class (test split)
    ax = axes[0]
    confused_margins = alpha_test["_margins_confused"]
    easy_margins = alpha_test["_margins_easy"]
    bins = np.linspace(0, max(max(confused_margins, default=0), max(easy_margins, default=0)) * 1.1, 40)
    ax.hist(easy_margins, bins=bins, alpha=0.6, color="#1f77b4", label=f"easy (n={len(easy_margins)})",
            density=True)
    ax.hist(confused_margins, bins=bins, alpha=0.6, color="#d62728", label=f"confused (n={len(confused_margins)})",
            density=True)
    ax.axvline(np.median(easy_margins), color="#1f77b4", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(np.median(confused_margins), color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Top-1 − Top-2 score margin (frozen ColBERT)")
    ax.set_ylabel("density")
    ax.set_title(f"(α) Score margin distribution by class (SciFact test)\n"
                 f"AUC(margin → confused) = {alpha_test['auc_margin_predicts_confused']:.3f}")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)

    # Panel B: Oracle vs Phase 2b NDCG Δ per slice (3-seed mean)
    ax = axes[1]
    slices = ["all", "confused", "easy"]
    oracle_means = [gamma_results["3seed_mean"][s]["oracle_mean"] for s in slices]
    oracle_stds = [gamma_results["3seed_mean"][s]["oracle_std"] for s in slices]
    phase2b_means = [gamma_results["3seed_mean"][s]["phase2b_mean"] for s in slices]
    phase2b_stds = [gamma_results["3seed_mean"][s]["phase2b_std"] for s in slices]

    x = np.arange(len(slices))
    width = 0.35
    bars_o = ax.bar(x - width/2, oracle_means, width, yerr=oracle_stds, capsize=4,
                    color="#2ca02c", alpha=0.85, label="Oracle conditional")
    bars_p = ax.bar(x + width/2, phase2b_means, width, yerr=phase2b_stds, capsize=4,
                    color="#cc4444", alpha=0.85, label="Phase 2b (full LoRA)")
    for b, v in zip(bars_o, oracle_means):
        ax.text(b.get_x() + b.get_width()/2, v + (0.003 if v >= 0 else -0.008),
                f"{v:+.3f}", ha="center", fontsize=8)
    for b, v in zip(bars_p, phase2b_means):
        ax.text(b.get_x() + b.get_width()/2, v + (0.003 if v >= 0 else -0.008),
                f"{v:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    # anchor-side reference line for Δ all
    ax.axhline(0.030, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.6,
               label="anchor-side family Δ all (+0.030)")
    ax.set_xticks(x)
    ax.set_xticklabels(slices)
    ax.set_ylabel("Δ NDCG@10 vs frozen baseline (3-seed mean ± std)")
    ax.set_title("(γ) Oracle test-time conditional NDCG\n(perfect gold-label routing ceiling)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: per-seed oracle scatter (oracle Δ all vs Phase 2b Δ all)
    ax = axes[2]
    oracle_per_seed = [gamma_results["seeds"][s]["all"]["oracle_mean_delta"] for s in SEEDS]
    phase2b_per_seed = [gamma_results["seeds"][s]["all"]["phase2b_mean_delta"] for s in SEEDS]
    for i, (s, o, p) in enumerate(zip(SEEDS, oracle_per_seed, phase2b_per_seed)):
        ax.scatter(p, o, s=120, color="#2ca02c", alpha=0.7, edgecolor="black", linewidth=1)
        ax.annotate(f"s{s}", (p, o), textcoords="offset points", xytext=(7, 7), fontsize=9)
    # diagonal y=x
    xs = np.linspace(min(phase2b_per_seed)-0.005, max(phase2b_per_seed)+0.01, 50)
    ax.plot(xs, xs, "--", color="grey", alpha=0.4, label="y = x (no oracle gain)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Phase 2b Δ all (full LoRA)")
    ax.set_ylabel("Oracle conditional Δ all")
    ax.set_title("(γ) Oracle gain per seed\n(distance above y=x = routing benefit)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("Exp 15 foundation diagnostics — (α) score-margin AUC + (γ) oracle conditional ceiling",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"diagnostic_alpha_gamma.{ext}")
    plt.close(fig)
    print(f"\nfigure → {out_dir}/diagnostic_alpha_gamma.{{pdf,png}}")
    print(f"data   → {out_dir}/diagnostic_alpha.json, diagnostic_gamma.json")


if __name__ == "__main__":
    main()
