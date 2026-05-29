"""08_bilinear_M_minimal figure 카탈로그.

생성:
  - ndcg_vs_baselines.{pdf,png}       — 08 의 NDCG@10 vs 6 anchor (baseline, 01b α=10, 02, 06 K=2/4/8) bar
  - delta_ci_forest.{pdf,png}         — paired bootstrap 95 % CI vs 5 anchor (baseline, α=10 mean-diff, 02, 06 K=2, 06 K=4)
  - M_spectrum.{pdf,png}              — 학습된 M = I + UV^T 의 singular value spectrum (top 10 + bottom 10)
  - train_curve.{pdf,png}             — train loss + ‖U;V‖ + val NDCG@10 (all / confused)
  - UV_inner.{pdf,png}                — U^T U / V^T V / U^T V 의 r×r heatmap (학습된 cross-feature structure)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET = "scifact"
SEED = 42
R = 8


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
    out08 = base / "08_bilinear_M_minimal" / DATASET / f"seed_{SEED}" / f"r_{R}"
    return {
        "module": torch.load(out08 / "module_final.pt", map_location="cpu"),
        "history": json.loads((out08 / "train_history.json").read_text()),
        "M_stats": json.loads((out08 / "M_stats.json").read_text()),
        "agg": json.loads((out08 / "metrics_aggregate.json").read_text()),
        "delta_vs_baseline": json.loads((out08 / "delta_vs_baseline.json").read_text()),
        "delta_vs_alpha10": json.loads((out08 / "delta_vs_mean_diff_alpha10.json").read_text()),
        "delta_vs_02": json.loads((out08 / "delta_vs_02_learned.json").read_text())
        if (out08 / "delta_vs_02_learned.json").exists() else {},
        "delta_vs_06_k2": json.loads((out08 / "delta_vs_06_k_sweep_k2.json").read_text())
        if (out08 / "delta_vs_06_k_sweep_k2.json").exists() else {},
        "delta_vs_06_k4": json.loads((out08 / "delta_vs_06_k_sweep_k4.json").read_text())
        if (out08 / "delta_vs_06_k_sweep_k4.json").exists() else {},
        "baseline_agg": json.loads((base / "00_baseline" / DATASET / f"seed_{SEED}"
                                    / "metrics_aggregate.json").read_text()),
        "alpha10_agg": json.loads((base / "01b_mean_diff_scaled" / DATASET / f"seed_{SEED}"
                                   / "alpha_10p0" / "metrics_aggregate.json").read_text()),
        "e02_agg": json.loads((base / "02_final_layer_vector" / DATASET / f"seed_{SEED}"
                               / "metrics_aggregate.json").read_text()),
        "k2_agg": json.loads((base / "06_k_sweep" / DATASET / f"seed_{SEED}" / "k_2"
                              / "metrics_aggregate.json").read_text()),
        "k4_agg": json.loads((base / "06_k_sweep" / DATASET / f"seed_{SEED}" / "k_4"
                              / "metrics_aggregate.json").read_text()),
        "k8_agg": json.loads((base / "06_k_sweep" / DATASET / f"seed_{SEED}" / "k_8"
                              / "metrics_aggregate.json").read_text()),
    }


def fig_ndcg_vs_baselines(d: dict, out_dir: Path) -> None:
    """08 vs 7 anchors NDCG@10."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = [
        "baseline (00)", "02 K=1\nlearned", "01b α=10\nmean-diff",
        "06 K=2\nrouter", "06 K=4\nrouter", "06 K=8\nrouter",
        "08 bilinear M\nr=8",
    ]
    all_vals = [
        d["baseline_agg"]["all"]["ndcg_cut_10"],
        d["e02_agg"]["all"]["ndcg_cut_10"],
        d["alpha10_agg"]["all"]["ndcg_cut_10"],
        d["k2_agg"]["all"]["ndcg_cut_10"],
        d["k4_agg"]["all"]["ndcg_cut_10"],
        d["k8_agg"]["all"]["ndcg_cut_10"],
        d["agg"]["all"]["ndcg_cut_10"],
    ]
    conf_vals = [
        d["baseline_agg"]["confused"]["ndcg_cut_10"],
        d["e02_agg"]["confused"]["ndcg_cut_10"],
        d["alpha10_agg"]["confused"]["ndcg_cut_10"],
        d["k2_agg"]["confused"]["ndcg_cut_10"],
        d["k4_agg"]["confused"]["ndcg_cut_10"],
        d["k8_agg"]["confused"]["ndcg_cut_10"],
        d["agg"]["confused"]["ndcg_cut_10"],
    ]
    x = np.arange(len(labels))
    w = 0.38
    colors = ["#888"] * 6 + ["#2a8a3e"]  # 08 in green
    ax.bar(x - w / 2, all_vals, w, color="#3b6e8f", label="all slice")
    ax.bar(x + w / 2, conf_vals, w, color="#d04545", label="confused slice")
    for i, (a, c) in enumerate(zip(all_vals, conf_vals)):
        ax.text(i - w / 2, a + 0.012, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + w / 2, c + 0.012, f"{c:.4f}", ha="center", fontsize=8)
    # ceiling line
    ceiling = max(d["e02_agg"]["all"]["ndcg_cut_10"], d["alpha10_agg"]["all"]["ndcg_cut_10"])
    ax.axhline(ceiling, color="#666", linestyle="--", linewidth=0.8,
               label=f"translation-family ceiling ({ceiling:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("NDCG@10")
    ax.set_title("08 bilinear M r=8 — translation-family ceiling test")
    ax.set_ylim(0, max(max(all_vals), max(conf_vals)) + 0.10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "ndcg_vs_baselines")


def fig_delta_ci_forest(d: dict, out_dir: Path) -> None:
    """vs 5 anchor의 paired bootstrap CI."""
    label_map = [
        ("delta_vs_baseline", "vs baseline (00)"),
        ("delta_vs_alpha10", "vs α=10 mean-diff"),
        ("delta_vs_02", "vs 02 K=1 learned"),
        ("delta_vs_06_k2", "vs 06 K=2 router"),
        ("delta_vs_06_k4", "vs 06 K=4 router"),
    ]
    rows = []
    for key, label in label_map:
        for sn in ("all", "confused"):
            r = d[key].get(sn, {}) if key in d else {}
            if "mean_delta_ndcg10" not in r:
                continue
            rows.append((label, sn, r["mean_delta_ndcg10"], r["ci_lo"], r["ci_hi"]))
    fig, ax = plt.subplots(figsize=(9, max(5, len(rows) * 0.4)))
    sn_color = {"all": "#3b6e8f", "confused": "#d04545"}
    y = np.arange(len(rows))
    for i, (ref, sn, m, lo, hi) in enumerate(rows):
        c = sn_color[sn]
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o" if sn == "all" else "s",
                    color=c, ecolor=c, capsize=4, markersize=6)
        if lo > 0:
            ax.text(hi + 0.005, i, "[+]", va="center", color="#2a8a3e",
                    fontweight="bold", fontsize=10)
        elif hi < 0:
            ax.text(lo - 0.005, i, "[-]", va="center", ha="right", color="#7a3a3a",
                    fontweight="bold", fontsize=10)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{ref}\n{sn}" for ref, sn, *_ in rows])
    ax.set_xlabel("Δ NDCG@10")
    ax.set_title("08 bilinear M r=8 — paired bootstrap 95% CI")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest")


def fig_M_spectrum(d: dict, out_dir: Path) -> None:
    """M = I + UV^T 의 SVD spectrum + UV^T 자체의 spectrum."""
    M_stats = d["M_stats"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # UV^T singular values (rank ≤ r)
    uv_s = M_stats["UV_singular_values"]
    axes[0].bar(np.arange(len(uv_s)) + 1, uv_s, color="#c47a2b")
    for i, v in enumerate(uv_s):
        axes[0].text(i + 1, v + max(uv_s) * 0.02, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].set_xlabel("singular value index"); axes[0].set_ylabel("σ_k(UV^T)")
    axes[0].set_title(f"UV^T singular values (rank ≤ r={M_stats['r']})")
    axes[0].set_xticks(np.arange(len(uv_s)) + 1)
    axes[0].grid(axis="y", alpha=0.3)

    # M = I + UV^T spectrum: top + bottom 10
    top = M_stats["M_singular_values_top10"]
    bot = M_stats["M_singular_values_bottom10"]
    x_pos = list(range(10)) + list(range(M_stats["dim"] - 10, M_stats["dim"]))
    vals = top + bot
    colors_b = ["#3b6e8f"] * 10 + ["#d04545"] * 10
    axes[1].bar(np.arange(len(vals)), vals, color=colors_b)
    axes[1].axhline(1.0, color="#666", linestyle="--", linewidth=0.8, label="identity σ=1")
    axes[1].set_xticks(np.arange(len(vals)))
    axes[1].set_xticklabels([str(i) for i in x_pos], rotation=45, fontsize=8)
    axes[1].set_xlabel("σ index (top 10 + bottom 10 of D=128)")
    axes[1].set_ylabel("σ_k(M)")
    axes[1].set_title(f"M = I + UV^T spectrum (cond = {M_stats['M_condition_number']:.3f})")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "M_spectrum")


def fig_train_curve(d: dict, out_dir: Path) -> None:
    h = d["history"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(h["steps"], h["rank_losses"], color="#3b6e8f", linewidth=1)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("pairwise margin loss")
    axes[0].set_title("Train loss"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(h["steps"], h["v_norms"], color="#c47a2b", linewidth=1)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("‖[U;V]‖₂")
    axes[1].set_title("Aggregate parameter magnitude"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(h["val_epochs"], h["val_ndcg_all"], "-o", color="#3b6e8f", label="all")
    axes[2].plot(h["val_epochs"], h["val_ndcg_confused"], "-s", color="#d04545", label="confused")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("val NDCG@10")
    axes[2].set_title("Validation"); axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "train_curve")


def fig_UV_inner(d: dict, out_dir: Path) -> None:
    """U^T U, V^T V, U^T V 의 r×r heatmap."""
    U = d["module"]["U"].float()
    V = d["module"]["V"].float()
    UtU = (U.T @ U).numpy()
    VtV = (V.T @ V).numpy()
    UtV = (U.T @ V).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, M, title in zip(
        axes, [UtU, VtV, UtV], ["U^T U", "V^T V", "U^T V"],
    ):
        vmax = max(abs(M).max(), 1e-6)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{title}  (‖·‖_F={np.linalg.norm(M):.3f})")
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_yticks(np.arange(M.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Bilinear correction internal structure (r=8)")
    fig.tight_layout()
    _save(fig, out_dir, "UV_inner")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "08_bilinear_M_minimal"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load()
    fig_ndcg_vs_baselines(d, out_dir)
    fig_delta_ci_forest(d, out_dir)
    fig_M_spectrum(d, out_dir)
    fig_train_curve(d, out_dir)
    fig_UV_inner(d, out_dir)
    print(f"figures → {out_dir}")


if __name__ == "__main__":
    main()
