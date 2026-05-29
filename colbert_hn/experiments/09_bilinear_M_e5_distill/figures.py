"""09_bilinear_M_e5_distill figure 카탈로그 (λ-sweep aware).

λ_distill ∈ {0.1, 0.5, 1.0} 의 3 run + 08 (λ=0, no distill) baseline 통합.

생성:
  - ndcg_vs_lambda.{pdf,png}            — λ vs NDCG@10 (all / confused) + 08 (λ=0) baseline
  - rank_collapse_by_lambda.{pdf,png}   — λ vs UV^T singular values (rank-collapse 해소 양상)
  - delta_ci_forest_kwise.{pdf,png}     — 3 λ × 5 anchor 의 paired bootstrap CI
  - train_curve_kwise.{pdf,png}         — 3 λ 의 rank / distill / total loss + val NDCG 비교
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
LAMBDAS = (0.1, 0.5, 1.0)


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


def _ld_tag(ld: float) -> str:
    return f"r_8_ld_{ld:.2f}".replace(".", "p")


def _load() -> dict:
    base = PROJECT_ROOT / "outputs"
    out: dict = {"runs": {}}
    for ld in LAMBDAS:
        rdir = base / "09_bilinear_M_e5_distill" / DATASET / f"seed_{SEED}" / _ld_tag(ld)
        if not (rdir / "metrics_aggregate.json").exists():
            continue
        out["runs"][ld] = {
            "agg": json.loads((rdir / "metrics_aggregate.json").read_text()),
            "history": json.loads((rdir / "train_history.json").read_text()),
            "M_stats": json.loads((rdir / "M_stats.json").read_text()),
            "delta_vs_baseline": json.loads((rdir / "delta_vs_baseline.json").read_text()),
            "delta_vs_alpha10": json.loads((rdir / "delta_vs_mean_diff_alpha10.json").read_text())
            if (rdir / "delta_vs_mean_diff_alpha10.json").exists() else {},
            "delta_vs_02": json.loads((rdir / "delta_vs_02_learned.json").read_text())
            if (rdir / "delta_vs_02_learned.json").exists() else {},
            "delta_vs_06_k4": json.loads((rdir / "delta_vs_06_k_sweep_k4.json").read_text())
            if (rdir / "delta_vs_06_k_sweep_k4.json").exists() else {},
            "delta_vs_08": json.loads((rdir / "delta_vs_08_r8.json").read_text())
            if (rdir / "delta_vs_08_r8.json").exists() else {},
        }
    # 08 reference (no distill)
    o8 = base / "08_bilinear_M_minimal" / DATASET / f"seed_{SEED}" / "r_8"
    out["08_agg"] = json.loads((o8 / "metrics_aggregate.json").read_text())
    out["08_M_stats"] = json.loads((o8 / "M_stats.json").read_text())
    out["08_delta_baseline"] = json.loads((o8 / "delta_vs_baseline.json").read_text())
    out["baseline_agg"] = json.loads(
        (base / "00_baseline" / DATASET / f"seed_{SEED}" / "metrics_aggregate.json").read_text()
    )
    return out


def fig_ndcg_vs_lambda(d: dict, out_dir: Path) -> None:
    """λ_distill 의 함수로 NDCG@10 (all + confused). 08 (λ=0) 도 점선으로."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [0.0] + list(LAMBDAS)
    all_vals = [d["08_agg"]["all"]["ndcg_cut_10"]] + [
        d["runs"][ld]["agg"]["all"]["ndcg_cut_10"] for ld in LAMBDAS if ld in d["runs"]
    ]
    conf_vals = [d["08_agg"]["confused"]["ndcg_cut_10"]] + [
        d["runs"][ld]["agg"]["confused"]["ndcg_cut_10"] for ld in LAMBDAS if ld in d["runs"]
    ]
    ax.plot(xs, all_vals, "-o", color="#3b6e8f", label="all slice")
    ax.plot(xs, conf_vals, "-s", color="#d04545", label="confused slice")
    baseline_all = d["baseline_agg"]["all"]["ndcg_cut_10"]
    ax.axhline(baseline_all, color="#888", linestyle="--", linewidth=0.8,
               label=f"baseline all ({baseline_all:.4f})")
    for x, a, c in zip(xs, all_vals, conf_vals):
        ax.text(x, a + 0.012, f"{a:.4f}", ha="center", fontsize=8, color="#3b6e8f")
        ax.text(x, c - 0.015, f"{c:.4f}", ha="center", fontsize=8, color="#d04545")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"λ=0\n(08)"] + [f"λ={ld}" for ld in LAMBDAS])
    ax.set_xlabel("λ_distill")
    ax.set_ylabel("NDCG@10")
    ax.set_title("09 E5 distillation sweep: NDCG@10 vs λ_distill")
    ax.set_ylim(0.15, max(all_vals) + 0.05)
    ax.legend(loc="center right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "ndcg_vs_lambda")


def fig_rank_collapse_by_lambda(d: dict, out_dir: Path) -> None:
    """λ vs UV^T 의 singular values + effective rank."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    colors = {0.0: "#888888", 0.1: "#3b6e8f", 0.5: "#c47a2b", 1.0: "#d04545"}
    # (1) singular values overlay
    for ld in [0.0] + list(LAMBDAS):
        if ld == 0.0:
            svs = d["08_M_stats"]["UV_singular_values"]
            label = "λ=0 (08, no distill)"
        elif ld not in d["runs"]:
            continue
        else:
            svs = d["runs"][ld]["M_stats"]["UV_singular_values"]
            label = f"λ={ld}"
        axes[0].plot(range(1, len(svs) + 1), svs, "-o", color=colors[ld], label=label)
    axes[0].set_xlabel("singular value index")
    axes[0].set_ylabel("σ_k(UV^T)")
    axes[0].set_title("UV^T spectrum by λ_distill (rank-collapse)")
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(1, 9))
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3, which="both")

    # (2) ‖UV^T‖_F vs λ + σ1/σ2 ratio
    xs = [0.0] + list(LAMBDAS)
    norm_fro = [d["08_M_stats"]["UV_norm_fro"]] + [
        d["runs"][ld]["M_stats"]["UV_norm_fro"] for ld in LAMBDAS if ld in d["runs"]
    ]
    ratios = []
    for ld in xs:
        m = d["08_M_stats"] if ld == 0.0 else d["runs"][ld]["M_stats"]
        svs = m["UV_singular_values"]
        ratios.append(svs[0] / svs[1] if len(svs) > 1 and svs[1] > 0 else float("inf"))

    ax_l = axes[1]
    ax_l.bar([x - 0.18 for x in range(len(xs))], norm_fro, width=0.36,
             color="#3b6e8f", label="‖UV^T‖_F")
    for i, n in enumerate(norm_fro):
        ax_l.text(i - 0.18, n + 0.05, f"{n:.2f}", ha="center", fontsize=8, color="#3b6e8f")
    ax_l.set_xticks(range(len(xs)))
    ax_l.set_xticklabels([f"λ={x}" for x in xs])
    ax_l.set_ylabel("‖UV^T‖_F", color="#3b6e8f")
    ax_l.tick_params(axis="y", labelcolor="#3b6e8f")
    ax_l.grid(axis="y", alpha=0.3)
    ax_r = ax_l.twinx()
    ax_r.bar([x + 0.18 for x in range(len(xs))], ratios, width=0.36,
             color="#d04545", label="σ₁/σ₂ ratio")
    for i, r in enumerate(ratios):
        ax_r.text(i + 0.18, r + 1.0, f"{r:.1f}", ha="center", fontsize=8, color="#d04545")
    ax_r.set_ylabel("σ₁ / σ₂  (rank-1 dominance)", color="#d04545")
    ax_r.tick_params(axis="y", labelcolor="#d04545")
    ax_l.set_title("Magnitude vs rank-1 dominance trade-off")
    fig.tight_layout()
    _save(fig, out_dir, "rank_collapse_by_lambda")


def fig_delta_ci_forest_kwise(d: dict, out_dir: Path) -> None:
    """3 λ × 4 anchor 의 paired bootstrap CI (vs baseline / α=10 / 02 / 08)."""
    rows = []
    label_map = [
        ("delta_vs_baseline", "vs baseline"),
        ("delta_vs_alpha10", "vs α=10 mean-diff"),
        ("delta_vs_02", "vs 02 K=1 learned"),
        ("delta_vs_08", "vs 08 r=8 no-distill"),
    ]
    for ld in LAMBDAS:
        if ld not in d["runs"]:
            continue
        for key, label in label_map:
            r_all = d["runs"][ld].get(key, {}).get("all", {})
            r_conf = d["runs"][ld].get(key, {}).get("confused", {})
            for sn, r in (("all", r_all), ("confused", r_conf)):
                if "mean_delta_ndcg10" not in r:
                    continue
                rows.append((ld, label, sn, r["mean_delta_ndcg10"],
                             r["ci_lo"], r["ci_hi"]))
    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.28)))
    ld_marker = {0.1: "o", 0.5: "s", 1.0: "^"}
    sn_color = {"all": "#3b6e8f", "confused": "#d04545"}
    y = np.arange(len(rows))
    for i, (ld, ref, sn, m, lo, hi) in enumerate(rows):
        c = sn_color[sn]
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt=ld_marker[ld],
                    color=c, ecolor=c, capsize=4, markersize=6)
        if lo > 0:
            ax.text(hi + 0.005, i, "[+]", va="center", color="#2a8a3e",
                    fontweight="bold", fontsize=10)
        elif hi < 0:
            ax.text(lo - 0.005, i, "[-]", va="center", ha="right", color="#7a3a3a",
                    fontweight="bold", fontsize=10)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"λ={ld}, {ref}, {sn}" for ld, ref, sn, *_ in rows])
    ax.set_xlabel("Δ NDCG@10")
    ax.set_title("09 λ-sweep: paired bootstrap 95% CI")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markersize=8, label="λ=0.1"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
               markersize=8, label="λ=0.5"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#888",
               markersize=8, label="λ=1.0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sn_color["all"],
               markersize=8, label="all slice"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sn_color["confused"],
               markersize=8, label="confused slice"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest_kwise")


def fig_train_curve_kwise(d: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    colors = {0.1: "#3b6e8f", 0.5: "#c47a2b", 1.0: "#d04545"}
    for ld in LAMBDAS:
        if ld not in d["runs"]:
            continue
        h = d["runs"][ld]["history"]
        c = colors[ld]
        axes[0].plot(h["steps"], h["rank_losses"], color=c, linewidth=1, label=f"λ={ld}")
        axes[1].plot(h["steps"], h["anchor_losses"], color=c, linewidth=1, label=f"λ={ld}")
        axes[2].plot(h["val_epochs"], h["val_ndcg_confused"], "-o", color=c,
                     label=f"λ={ld}")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("rank loss")
    axes[0].set_title("Pairwise margin loss"); axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("distill loss (raw MSE)")
    axes[1].set_title("Margin-MSE distill loss"); axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("val NDCG@10 (confused)")
    axes[2].set_title("Validation confused-slice"); axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "train_curve_kwise")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "09_bilinear_M_e5_distill"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load()
    if not d["runs"]:
        print("no runs found")
        return
    fig_ndcg_vs_lambda(d, out_dir)
    fig_rank_collapse_by_lambda(d, out_dir)
    fig_delta_ci_forest_kwise(d, out_dir)
    fig_train_curve_kwise(d, out_dir)
    print(f"figures → {out_dir}")


if __name__ == "__main__":
    main()
