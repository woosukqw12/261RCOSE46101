"""(δ) Margin-routed Phase 2b — *realistic* Exp 15 minimal realization.

본 script 는 Exp 15 의 *minimal realization* diagnostic:
  - Phase 2b LoRA 의 학습된 ranking 그대로 사용 (no retraining)
  - 각 test query 에 대해 frozen ColBERT 의 top-1/top-2 margin 측정
  - margin ≤ τ → LoRA ranking 사용 (confused 추정)
  - margin > τ → frozen ranking 사용 (easy 추정)
  - 다양한 τ 에 대한 NDCG@10 trajectory + paired bootstrap CI vs frozen baseline

(γ) oracle 와의 차이: gold confused/easy label 대신 *qrels-free* margin-predicted label.
AUC 0.836 (α 에서 측정) → realistic gain 은 oracle ceiling (+0.048) 과 0 (random routing) 사이.

Output:
  report/figures/_exp15_diagnostics/{
    diagnostic_delta.json,        # τ-sweep results + paired bootstrap CIs
    diagnostic_delta.{pdf,png}    # τ-sensitivity + 4-diagnostic summary figure
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

DATASET = "scifact"
SEEDS = [42, 1337, 2024]
BOOT_N = 10000
ALPHA = 0.05


def ndcg_at_k(ranked_dids, qrels_q, k=10):
    rels = [qrels_q.get(d, 0) for d in ranked_dids[:k]]
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    sorted_rels = sorted(qrels_q.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(sorted_rels))
    return dcg / idcg if idcg > 0 else 0.0


def boot_ci(deltas, n=BOOT_N, alpha=ALPHA, seed_b=42):
    rng = np.random.default_rng(seed_b)
    n_q = len(deltas)
    boots = np.array([deltas[rng.integers(0, n_q, size=n_q)].mean() for _ in range(n)])
    return float(boots.mean()), float(np.percentile(boots, alpha/2*100)), float(np.percentile(boots, (1-alpha/2)*100))


def evaluate_routed(seed, qrels, threshold=None, fraction_to_lora=None):
    """Compute margin-routed NDCG for given seed.

    Routing rule:
      - threshold mode: margin <= threshold → LoRA, else frozen
      - fraction mode: lowest fraction_to_lora of queries (by margin) → LoRA, rest frozen
    """
    baseline_scored = json.loads(
        (PROJECT_ROOT / "outputs/00_baseline" / DATASET / f"seed_{seed}" / "runs_scored.json").read_text()
    )
    baseline_runs = json.loads(
        (PROJECT_ROOT / "outputs/00_baseline" / DATASET / f"seed_{seed}" / "runs.json").read_text()
    )
    lora_runs = json.loads(
        (PROJECT_ROOT / "outputs/10_lora_phi" / DATASET / f"seed_{seed}" / "qv_r8_l12" / "runs.json").read_text()
    )

    qids = sorted(baseline_runs.keys())
    margins = {}
    is_confused = {}
    for qid in qids:
        if qid not in qrels:
            continue
        rel_set = {d for d, r in qrels[qid].items() if r >= 1}
        if not rel_set:
            continue
        scored = baseline_scored[qid]
        sorted_pairs = sorted(scored.items(), key=lambda x: -x[1])
        s1, s2 = sorted_pairs[0][1], sorted_pairs[1][1]
        margins[qid] = s1 - s2
        is_confused[qid] = sorted_pairs[0][0] not in rel_set

    qids = sorted(margins.keys())

    if threshold is not None:
        # threshold mode
        route_to_lora = {q: margins[q] <= threshold for q in qids}
    else:
        # fraction mode
        sorted_margins = sorted([margins[q] for q in qids])
        n_to_lora = int(len(qids) * fraction_to_lora)
        thresh = sorted_margins[n_to_lora - 1] if n_to_lora > 0 else float("-inf")
        route_to_lora = {q: margins[q] <= thresh for q in qids}

    routed_ndcg = {}
    baseline_ndcg = {}
    lora_ndcg = {}
    for qid in qids:
        rel_set_dict = qrels[qid]
        b_n = ndcg_at_k(baseline_runs[qid], rel_set_dict, k=10)
        l_n = ndcg_at_k(lora_runs.get(qid, baseline_runs[qid]), rel_set_dict, k=10)
        routed_ndcg[qid] = l_n if route_to_lora[qid] else b_n
        baseline_ndcg[qid] = b_n
        lora_ndcg[qid] = l_n

    qids_ord = sorted(routed_ndcg.keys())
    b_arr = np.array([baseline_ndcg[q] for q in qids_ord])
    l_arr = np.array([lora_ndcg[q] for q in qids_ord])
    r_arr = np.array([routed_ndcg[q] for q in qids_ord])
    confused_mask = np.array([is_confused[q] for q in qids_ord], dtype=bool)
    easy_mask = ~confused_mask
    routed_mask = np.array([route_to_lora[q] for q in qids_ord], dtype=bool)

    # confusion matrix
    tp = int((routed_mask & confused_mask).sum())  # confused correctly routed to LoRA
    fp = int((routed_mask & easy_mask).sum())      # easy incorrectly routed to LoRA
    fn = int((~routed_mask & confused_mask).sum()) # confused incorrectly bypassed
    tn = int((~routed_mask & easy_mask).sum())     # easy correctly bypassed

    def slice_delta(mask):
        d = (r_arr - b_arr)[mask]
        if len(d) == 0:
            return None
        m, lo, hi = boot_ci(d)
        return {"n": int(mask.sum()), "mean": m, "ci_lo": lo, "ci_hi": hi,
                "ndcg_routed_mean": float(r_arr[mask].mean()),
                "ndcg_baseline_mean": float(b_arr[mask].mean()),
                "ndcg_lora_mean": float(l_arr[mask].mean())}

    return {
        "seed": seed,
        "threshold_used": threshold,
        "fraction_to_lora": fraction_to_lora,
        "n_routed_to_lora": int(routed_mask.sum()),
        "n_total": len(qids_ord),
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                      "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                      "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                      "accuracy": (tp + tn) / len(qids_ord)},
        "all": slice_delta(np.ones_like(confused_mask, dtype=bool)),
        "confused": slice_delta(confused_mask),
        "easy": slice_delta(easy_mask),
    }


def main():
    out_dir = PROJECT_ROOT / "report/figures/_exp15_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("(δ) Margin-routed Phase 2b — realistic Exp 15 minimal realization")
    print("=" * 80)

    print("loading scifact test qrels...")
    _, _, qrels = load_beir(DATASET, split="test")

    # Sweep fractions for sensitivity analysis (NOT a hyperparameter selection — full curve reported)
    fractions = [0.10, 0.20, 0.30, 0.40, 0.46, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    # 0.46 = actual confused fraction (structural pre-commit point)

    results = {"sweep": {}, "natural_thresholds": {}}

    # ===== Fraction sweep =====
    for frac in fractions:
        per_seed = {}
        for seed in SEEDS:
            r = evaluate_routed(seed, qrels, fraction_to_lora=frac)
            per_seed[seed] = r
        agg = {}
        for slc in ("all", "confused", "easy"):
            vals = [per_seed[s][slc]["mean"] for s in SEEDS if per_seed[s][slc] is not None]
            agg[slc] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        results["sweep"][f"{frac:.2f}"] = {"per_seed": per_seed, "agg": agg}
        print(f"\nfraction={frac:.2f} (route {frac*100:.0f}% lowest-margin queries to LoRA):")
        for slc in ("all", "confused", "easy"):
            print(f"  Δ {slc}: {agg[slc]['mean']:+.4f} ± {agg[slc]['std']:.4f}")

    (out_dir / "diagnostic_delta.json").write_text(json.dumps(results, indent=2))

    # ===== Figure: 4-diagnostic summary + tau sensitivity =====
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # load (α) and (γ) results for context
    alpha_data = json.loads((out_dir / "diagnostic_alpha.json").read_text())
    gamma_data = json.loads((out_dir / "diagnostic_gamma.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # --- Panel A: τ-sensitivity curve (Δ all / confused / easy vs fraction routed)
    ax = axes[0]
    fracs = sorted(results["sweep"].keys(), key=float)
    fracs_f = [float(f) for f in fracs]
    for slc, color, label in [("all", "#444", "Δ all"),
                              ("confused", "#1f77b4", "Δ confused"),
                              ("easy", "#d62728", "Δ easy")]:
        means = [results["sweep"][f]["agg"][slc]["mean"] for f in fracs]
        stds = [results["sweep"][f]["agg"][slc]["std"] for f in fracs]
        ax.errorbar(fracs_f, means, yerr=stds, fmt="o-", color=color, label=label,
                    markersize=6, capsize=3, alpha=0.85)
    # oracle ceiling reference
    oracle_all = gamma_data["3seed_mean"]["all"]["oracle_mean"]
    ax.axhline(oracle_all, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"oracle ceiling Δ all (+{oracle_all:.3f})")
    # anchor-side family reference
    ax.axhline(0.030, color="#9467bd", linestyle=":", linewidth=1.2, alpha=0.7,
               label="anchor-side family Δ all (+0.030)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0.46, color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(0.46, ax.get_ylim()[1] * 0.92, "structural\npre-commit\n(46%)",
            ha="center", fontsize=8, color="grey")
    ax.set_xlabel("fraction of queries routed to LoRA (lowest-margin)")
    ax.set_ylabel("Δ NDCG@10 (3-seed mean ± std)")
    ax.set_title("(δ) τ-sensitivity — margin-routed Phase 2b")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(alpha=0.3)

    # --- Panel B: confusion matrix at structural threshold (frac=0.46)
    ax = axes[1]
    pre_commit_key = "0.46"
    per_seed_pc = results["sweep"][pre_commit_key]["per_seed"]
    # average confusion across seeds
    avg_cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for s in SEEDS:
        for k in avg_cm:
            avg_cm[k] += per_seed_pc[s]["confusion"][k]
    for k in avg_cm:
        avg_cm[k] /= len(SEEDS)
    cm_array = np.array([[avg_cm["TN"], avg_cm["FP"]],
                         [avg_cm["FN"], avg_cm["TP"]]])
    im = ax.imshow(cm_array, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_array[i, j]:.0f}", ha="center", va="center",
                    color="white" if cm_array[i, j] > cm_array.max()/2 else "black",
                    fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["frozen (predicted easy)", "LoRA (predicted confused)"])
    ax.set_yticklabels(["actual easy", "actual confused"])
    avg_acc = sum(per_seed_pc[s]["confusion"]["accuracy"] for s in SEEDS) / len(SEEDS)
    avg_prec = sum(per_seed_pc[s]["confusion"]["precision"] for s in SEEDS) / len(SEEDS)
    avg_rec = sum(per_seed_pc[s]["confusion"]["recall"] for s in SEEDS) / len(SEEDS)
    ax.set_title(f"Routing confusion matrix at frac=0.46\n"
                 f"acc={avg_acc:.3f}, precision={avg_prec:.3f}, recall={avg_rec:.3f}")

    # --- Panel C: full diagnostic comparison (α/γ/β/δ + anchor-side ref)
    ax = axes[2]
    labels = [
        "Phase 2b\n(no routing)",
        "(β) confused-only\ntraining",
        "(δ) margin-routed\nphase 2b\n(frac=0.46)",
        "anchor-side\n(Exp 13)",
        "(γ) oracle\nconditional\n(ceiling)",
    ]
    delta_all_vals = [
        0.001,  # Phase 2b 3-seed mean
        -0.387,  # (β) seed 42
        results["sweep"][pre_commit_key]["agg"]["all"]["mean"],  # (δ) frac=0.46 mean
        0.030,  # anchor-side
        gamma_data["3seed_mean"]["all"]["oracle_mean"],  # (γ) oracle
    ]
    delta_all_stds = [
        0.014,
        0.0,  # single seed
        results["sweep"][pre_commit_key]["agg"]["all"]["std"],
        0.002,
        gamma_data["3seed_mean"]["all"]["oracle_std"],
    ]
    colors = ["#cc4444", "#a30000", "#d62728", "#9467bd", "#2ca02c"]
    x = np.arange(len(labels))
    bars = ax.bar(x, delta_all_vals, yerr=delta_all_stds, capsize=4,
                  color=colors, alpha=0.85)
    for b, v in zip(bars, delta_all_vals):
        offset = 0.005 if v >= 0 else -0.015
        ax.text(b.get_x() + b.get_width()/2, v + offset, f"{v:+.3f}",
                ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ NDCG@10 all (3-seed mean ± std)")
    ax.set_title("Exp 15 diagnostic chain summary")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Exp 15 diagnostics — (α) AUC={alpha_data['test']['auc_margin_predicts_confused']:.3f} "
                 f"+ (γ) oracle +{oracle_all:.3f} + (β) train-filter catastrophic + (δ) margin-routed",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"diagnostic_delta.{ext}")
    plt.close(fig)

    # ===== Summary print =====
    print("\n" + "=" * 80)
    print("Summary (3-seed mean ± std at structural pre-commit frac=0.46):")
    print("=" * 80)
    pc = results["sweep"]["0.46"]["agg"]
    print(f"Δ all      = {pc['all']['mean']:+.4f} ± {pc['all']['std']:.4f}")
    print(f"Δ confused = {pc['confused']['mean']:+.4f} ± {pc['confused']['std']:.4f}")
    print(f"Δ easy     = {pc['easy']['mean']:+.4f} ± {pc['easy']['std']:.4f}")
    print(f"\nbest Δ all over sweep (post-hoc): "
          f"{max(results['sweep'][f]['agg']['all']['mean'] for f in fracs):+.4f}")
    print(f"oracle ceiling: +{oracle_all:.4f}")
    print(f"\nfigure → {out_dir}/diagnostic_delta.{{pdf,png}}")


if __name__ == "__main__":
    main()
