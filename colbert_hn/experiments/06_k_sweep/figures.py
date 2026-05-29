"""06_k_sweep figure 카탈로그.

생성:
  - ndcg_vs_k_bar.{pdf,png}              — K ∈ {2, 4, 8} 의 NDCG@10 + 02 K=1 + 01b α=10 + baseline
  - delta_ci_forest_kwise.{pdf,png}      — 각 K 의 paired bootstrap CI vs (baseline / 02 / 01b α=10)
  - routing_entropy_by_k.{pdf,png}       — K vs (entropy / max-entropy, effective K, π_max>0.6 fraction)
  - direction_redundancy_by_k.{pdf,png}  — K vs mean |cos(v_i, v_j)| + max |cos(v_k, v_md)| + ‖v_k‖ swarm
  - train_curve_kwise.{pdf,png}          — K ∈ {2, 4, 8} 의 train loss + val NDCG epoch curve overlay
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET = "scifact"
SEED = 42
K_VALUES = (2, 4, 8)


def set_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _save(fig, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def _load() -> dict:
    base = PROJECT_ROOT / "outputs"
    data = {"k_runs": {}}
    for k in K_VALUES:
        kd = base / "06_k_sweep" / DATASET / f"seed_{SEED}" / f"k_{k}"
        if not kd.exists():
            continue
        data["k_runs"][k] = {
            "agg": json.loads((kd / "metrics_aggregate.json").read_text()),
            "history": json.loads((kd / "train_history.json").read_text()),
            "routing": json.loads((kd / "routing_stats.json").read_text()),
            "diag": json.loads((kd / "cosine_v_pairs.json").read_text()),
            "delta_vs_baseline": json.loads((kd / "delta_vs_baseline.json").read_text())
            if (kd / "delta_vs_baseline.json").exists() else {},
            "delta_vs_alpha10": json.loads((kd / "delta_vs_mean_diff_alpha10.json").read_text())
            if (kd / "delta_vs_mean_diff_alpha10.json").exists() else {},
            "delta_vs_02": json.loads((kd / "delta_vs_02_learned.json").read_text())
            if (kd / "delta_vs_02_learned.json").exists() else {},
        }
    base_d = base / "00_baseline" / DATASET / f"seed_{SEED}"
    a10_d = base / "01b_mean_diff_scaled" / DATASET / f"seed_{SEED}" / "alpha_10p0"
    e2_d = base / "02_final_layer_vector" / DATASET / f"seed_{SEED}"
    data["baseline_agg"] = json.loads((base_d / "metrics_aggregate.json").read_text())
    data["alpha10_agg"] = json.loads((a10_d / "metrics_aggregate.json").read_text())
    data["e02_agg"] = json.loads((e2_d / "metrics_aggregate.json").read_text())
    return data


def fig_ndcg_vs_k_bar(d: dict, out_dir: Path) -> None:
    """K ∈ {2,4,8} 의 NDCG@10 (all / confused) + 02 K=1 + 01b α=10 + baseline."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = ["baseline\n(00)", "02 K=1\n(learned)", "01b α=10\n(mean-diff)"]
    all_vals = [
        d["baseline_agg"]["all"]["ndcg_cut_10"],
        d["e02_agg"]["all"]["ndcg_cut_10"],
        d["alpha10_agg"]["all"]["ndcg_cut_10"],
    ]
    conf_vals = [
        d["baseline_agg"]["confused"]["ndcg_cut_10"],
        d["e02_agg"]["confused"]["ndcg_cut_10"],
        d["alpha10_agg"]["confused"]["ndcg_cut_10"],
    ]
    for k in K_VALUES:
        if k not in d["k_runs"]:
            continue
        labels.append(f"06 K={k}\n(router)")
        all_vals.append(d["k_runs"][k]["agg"]["all"]["ndcg_cut_10"])
        conf_vals.append(d["k_runs"][k]["agg"]["confused"]["ndcg_cut_10"])
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, all_vals, w, color="#3b6e8f", label="all slice")
    ax.bar(x + w / 2, conf_vals, w, color="#d04545", label="confused slice")
    for i, (a, c) in enumerate(zip(all_vals, conf_vals)):
        ax.text(i - w / 2, a + 0.012, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + w / 2, c + 0.012, f"{c:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("NDCG@10")
    ax.set_title("K sweep: NDCG@10 across K (translation-family ceiling test)")
    ax.set_ylim(0, max(max(all_vals), max(conf_vals)) + 0.10)
    ax.axhline(d["e02_agg"]["all"]["ndcg_cut_10"], color="#888", linestyle="--",
               linewidth=0.8, label="02 K=1 all ceiling")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "ndcg_vs_k_bar")


def fig_delta_ci_forest_kwise(d: dict, out_dir: Path) -> None:
    """K=2,4,8 각각의 paired bootstrap CI 를 1 figure 에 통합."""
    rows = []
    for k in K_VALUES:
        if k not in d["k_runs"]:
            continue
        for ref_key, ref_label in (
            ("delta_vs_baseline", "vs baseline"),
            ("delta_vs_alpha10", "vs α=10 mean-diff"),
            ("delta_vs_02", "vs 02 K=1"),
        ):
            dd = d["k_runs"][k].get(ref_key, {})
            for sn in ("all", "confused"):
                r = dd.get(sn, {})
                if "mean_delta_ndcg10" not in r:
                    continue
                rows.append((k, ref_label, sn, r["mean_delta_ndcg10"],
                             r["ci_lo"], r["ci_hi"]))
    fig, ax = plt.subplots(figsize=(9.5, max(5, len(rows) * 0.36)))
    k_marker = {2: "o", 4: "s", 8: "^"}
    sn_color = {"all": "#3b6e8f", "confused": "#d04545"}
    y = np.arange(len(rows))
    for i, (k, ref, sn, m, lo, hi) in enumerate(rows):
        c = sn_color[sn]
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt=k_marker[k],
                    color=c, ecolor=c, capsize=4, markersize=6)
        if lo > 0:
            ax.text(hi + 0.005, i, "[+]", va="center", color="#2a8a3e",
                    fontweight="bold", fontsize=10)
        elif hi < 0:
            ax.text(lo - 0.005, i, "[-]", va="center", ha="right", color="#7a3a3a",
                    fontweight="bold", fontsize=10)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"K={k}, {ref}, {sn}" for k, ref, sn, *_ in rows])
    ax.set_xlabel("Δ NDCG@10")
    ax.set_title("K-sweep: paired bootstrap 95% CI on Δ NDCG@10")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    # legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markersize=8, label="K=2"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
               markersize=8, label="K=4"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#888",
               markersize=8, label="K=8"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sn_color["all"],
               markersize=8, label="all slice"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sn_color["confused"],
               markersize=8, label="confused slice"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest_kwise")


def fig_routing_entropy_by_k(d: dict, out_dir: Path) -> None:
    """K vs routing entropy / effective K / π_max>0.6 fraction."""
    ks_present = [k for k in K_VALUES if k in d["k_runs"]]
    if not ks_present:
        return
    ent = [d["k_runs"][k]["routing"]["entropy_mean"] for k in ks_present]
    max_ent = [d["k_runs"][k]["routing"]["max_entropy_uniform"] for k in ks_present]
    eff_K = [d["k_runs"][k]["routing"].get("effective_K_perplexity",
              float(np.exp(d["k_runs"][k]["routing"]["entropy_mean"]))) for k in ks_present]
    frac_06 = [d["k_runs"][k]["routing"]["frac_tokens_pi_max_above_0.6"] for k in ks_present]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(ks_present))
    # (1) entropy bar (vs max-entropy reference)
    w = 0.4
    axes[0].bar(x - w / 2, ent, w, color="#3b6e8f", label="actual entropy")
    axes[0].bar(x + w / 2, max_ent, w, color="#cbcbcb", label="uniform max (log K)")
    for i, (e, me) in enumerate(zip(ent, max_ent)):
        axes[0].text(i - w / 2, e + 0.03, f"{e:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, me + 0.03, f"{me:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"K={k}" for k in ks_present])
    axes[0].set_ylabel("entropy of routing")
    axes[0].set_title("Routing entropy vs uniform max")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.3)

    # (2) effective K (perplexity) — K 대비 얼마나 활용?
    axes[1].plot(ks_present, eff_K, "-o", color="#3b6e8f", label="effective K (exp entropy)")
    axes[1].plot(ks_present, ks_present, "--", color="#888", label="ideal: eff K = K")
    for k, ek in zip(ks_present, eff_K):
        axes[1].text(k, ek + 0.15, f"{ek:.2f}", ha="center", fontsize=8)
    axes[1].set_xticks(ks_present)
    axes[1].set_xlabel("K"); axes[1].set_ylabel("effective K (perplexity)")
    axes[1].set_title("Effective K — capacity utilization")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3)

    # (3) frac tokens with π_max > 0.6 (saturation indicator)
    axes[2].bar(x, frac_06, color="#d04545")
    for i, f in enumerate(frac_06):
        axes[2].text(i, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"K={k}" for k in ks_present])
    axes[2].set_ylabel("frac(π_max > 0.6)")
    axes[2].set_title("Router saturation (tokens with high-confidence routing)")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "routing_entropy_by_k")


def fig_direction_redundancy_by_k(d: dict, out_dir: Path) -> None:
    """K vs mean |cos(v_i, v_j)| + max |cos(v_k, v_md)| + 학습된 v 의 norm 분포."""
    ks_present = [k for k in K_VALUES if k in d["k_runs"]]
    if not ks_present:
        return
    mean_pairwise = []
    max_cos_md = []
    norms_per_k = []
    for k in ks_present:
        dg = d["k_runs"][k]["diag"]
        mean_pairwise.append(dg.get("mean_pairwise_abs_cosine", 0.0))
        max_cos_md.append(dg.get("max_cos_v_k_vs_mean_diff", 0.0))
        norms_per_k.append(dg.get("v_norms", []))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(ks_present))
    # (1) mean pairwise |cos|
    axes[0].bar(x, mean_pairwise, color="#3b6e8f")
    for i, v in enumerate(mean_pairwise):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"K={k}" for k in ks_present])
    axes[0].set_ylabel("mean |cos(v_i, v_j)|, i<j")
    axes[0].set_title("Direction redundancy (lower = better)")
    axes[0].set_ylim(0, max(mean_pairwise + [0.6]) * 1.2)
    axes[0].grid(axis="y", alpha=0.3)

    # (2) max |cos(v_k, v_md)|
    axes[1].bar(x, max_cos_md, color="#c47a2b")
    for i, v in enumerate(max_cos_md):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"K={k}" for k in ks_present])
    axes[1].set_ylabel("max_k |cos(v_k, v_mean_diff)|")
    axes[1].set_title("Best-alignment of any v_k with mean-diff direction")
    axes[1].set_ylim(0, max(max_cos_md + [0.6]) * 1.2)
    axes[1].grid(axis="y", alpha=0.3)

    # (3) per-k v_k norm distribution (jittered scatter)
    rng = np.random.default_rng(42)
    for i, (k, ns) in enumerate(zip(ks_present, norms_per_k)):
        if not ns:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(ns))
        axes[2].scatter([i] * len(ns) + jitter, ns, color="#3b6e8f",
                        alpha=0.7, s=40)
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"K={k}" for k in ks_present])
    axes[2].set_ylabel("‖v_k‖₂ per direction")
    axes[2].set_title("Direction magnitudes (per K)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "direction_redundancy_by_k")


def fig_train_curve_kwise(d: dict, out_dir: Path) -> None:
    """K=2,4,8 의 train loss + val NDCG overlay."""
    ks_present = [k for k in K_VALUES if k in d["k_runs"]]
    if not ks_present:
        return
    color_map = {2: "#3b6e8f", 4: "#c47a2b", 8: "#7a3a8e"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in ks_present:
        h = d["k_runs"][k]["history"]
        c = color_map.get(k, "#000000")
        axes[0].plot(h["steps"], h["rank_losses"], color=c, linewidth=1, label=f"K={k}")
        axes[1].plot(h["val_epochs"], h["val_ndcg_confused"], "-o", color=c,
                     label=f"K={k} confused", linewidth=1.5)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("pairwise margin loss")
    axes[0].set_title("Train loss (all K — overfitting pattern check)")
    axes[0].legend(frameon=False); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val NDCG@10 (confused)")
    axes[1].set_title("Val NDCG@10 confused — early-stop curve")
    axes[1].legend(frameon=False); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "train_curve_kwise")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "06_k_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load()
    if not d["k_runs"]:
        print("no K runs found yet — run experiments/06_k_sweep/run.py first")
        return
    fig_ndcg_vs_k_bar(d, out_dir)
    fig_delta_ci_forest_kwise(d, out_dir)
    fig_routing_entropy_by_k(d, out_dir)
    fig_direction_redundancy_by_k(d, out_dir)
    fig_train_curve_kwise(d, out_dir)
    print(f"figures → {out_dir}")


if __name__ == "__main__":
    main()
