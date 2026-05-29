"""Exp 17 figures — anchor family saturation visualization.

Reads artifacts from outputs/17_negative_side_anchor/scifact/seed_{42,1337,2024}/.../delta_vs_baseline.json
Output: report/figures/17_negative_side_anchor/saturation.{pdf,png}

본 figure 의 학술적 message: 세 anchor parameterization (H₁ relational, H₂ per-token, I symmetric/+d⁻)
모두 같은 +0.030 Δ all 에 수렴 — anchor family 의 *direct empirical saturation proof*.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "report" / "figures" / "17_negative_side_anchor"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_rc():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def load_exp17_seed(seed: int) -> dict:
    p = PROJECT_ROOT / "outputs" / "17_negative_side_anchor" / "scifact" / f"seed_{seed}" / "qv_r8_l12_dir1_neg1"
    d = json.loads((p / "delta_vs_baseline.json").read_text())
    ls = json.loads((p / "lora_stats.json").read_text())
    return {
        "all": d["all"]["mean_delta_ndcg10"],
        "hard": d["confused"]["mean_delta_ndcg10"],
        "easy": d["easy"]["mean_delta_ndcg10"],
        "B_norm": ls["B_norm_total"],
    }


def aggregate_exp17() -> dict:
    seeds = [42, 1337, 2024]
    rows = [load_exp17_seed(s) for s in seeds]
    out = {}
    for k in ("all", "hard", "easy", "B_norm"):
        vals = np.array([r[k] for r in rows])
        out[k] = (float(vals.mean()), float(vals.std()))
    return out


# Reference values from §4.4 (Exp 11 H₁ + Exp 13 H₂) — pinned per README §4.4 results table
H1_REL = {
    "all": (0.029, 0.005),
    "hard": (0.101, 0.010),
    "easy": (-0.031, 0.018),
    "B_norm": (1.80, 0.10),
}
H2_PERTOKEN = {
    "all": (0.030, 0.002),
    "hard": (0.092, 0.007),
    "easy": (-0.021, 0.003),
    "B_norm": (1.34, 0.05),
}


def make_saturation_figure():
    set_rc()
    exp17 = aggregate_exp17()

    forms = [
        ("H₁: relational\n(rotation-invariant)", H1_REL),
        ("H₂: per-token\n(rotation-sensitive)", H2_PERTOKEN),
        ("I: +d⁻ anchor\n(symmetric)", exp17),
    ]
    labels = [f for f, _ in forms]
    n = len(labels)
    x = np.arange(n)
    width = 0.26

    mean_all = np.array([f[1]["all"][0] for f in forms])
    std_all = np.array([f[1]["all"][1] for f in forms])
    mean_hard = np.array([f[1]["hard"][0] for f in forms])
    std_hard = np.array([f[1]["hard"][1] for f in forms])
    mean_easy = np.array([f[1]["easy"][0] for f in forms])
    std_easy = np.array([f[1]["easy"][1] for f in forms])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Panel A: Grouped bar (Δ all, Δ hard, Δ easy) × three anchor forms
    barA = axA.bar(x - width, mean_all, width, yerr=std_all, capsize=4,
                   color="#444", edgecolor="black", linewidth=0.5, label="Δ all")
    barH = axA.bar(x, mean_hard, width, yerr=std_hard, capsize=4,
                   color="#cc4444", edgecolor="black", linewidth=0.5, label="Δ hard")
    barE = axA.bar(x + width, mean_easy, width, yerr=std_easy, capsize=4,
                   color="#1f77b4", edgecolor="black", linewidth=0.5, label="Δ easy")

    for bars, mean_vals in [(barA, mean_all), (barH, mean_hard), (barE, mean_easy)]:
        for b, v in zip(bars, mean_vals):
            off = 0.005 if v >= 0 else -0.010
            axA.text(b.get_x() + b.get_width()/2, v + off, f"{v:+.3f}",
                     ha="center", fontsize=7.5)

    # Reference horizontal line at +0.030 (the ceiling)
    axA.axhline(0.030, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.6,
                label="anchor frontier (+0.030)")
    axA.axhline(0.0, color="black", linewidth=0.6)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=8.5)
    axA.set_ylabel(r"$\Delta\,$NDCG@10  vs frozen baseline")
    axA.set_title("(A) Three anchor parameterizations converge to +0.030 frontier\n— direct empirical saturation proof")
    axA.legend(loc="upper right", frameon=False, fontsize=8.5)
    axA.grid(axis="y", alpha=0.25)
    axA.set_ylim(-0.07, 0.15)

    # Panel B: trade-off scatter — (Δ hard, Δ easy) phase space
    colors = ["#888", "#000", "#cc4444"]
    markers = ["o", "s", "D"]
    short_labels = ["H₁ relational", "H₂ per-token", "I  +d⁻ anchor"]
    for i, (form_lbl, data) in enumerate(forms):
        x_h, e_h = data["hard"][0], data["hard"][1]
        x_e, e_e = data["easy"][0], data["easy"][1]
        axB.errorbar(x_h, x_e, xerr=e_h, yerr=e_e, fmt=markers[i], color=colors[i],
                     markersize=11, capsize=4, linewidth=1.5,
                     markeredgecolor="black", markeredgewidth=0.8,
                     label=short_labels[i])
    # Constant Δ all isocontour at +0.030 (approximated: Δ all = w_h Δ hard + w_e Δ easy)
    # SciFact: w_hard ≈ 0.457, w_easy ≈ 0.543 (per §4.1 redistribution accounting)
    w_h, w_e = 0.457, 0.543
    xs = np.linspace(0.05, 0.13, 50)
    ys = (0.030 - w_h * xs) / w_e
    axB.plot(xs, ys, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.7,
             label=r"isocontour: $\Delta_{\rm all} = +0.030$")
    axB.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    axB.set_xlabel(r"$\Delta\,$NDCG@10  (hard)")
    axB.set_ylabel(r"$\Delta\,$NDCG@10  (easy)")
    axB.set_title("(B) Trade-off geometry: all three forms\nlie on the same Δ all = +0.030 isocontour")
    axB.legend(loc="lower left", frameon=False, fontsize=8.5)
    axB.grid(alpha=0.25)
    axB.set_xlim(0.06, 0.13)
    axB.set_ylim(-0.060, -0.005)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"saturation.{ext}")
    plt.close(fig)
    print(f"saved → {OUT_DIR}/saturation.{{pdf,png}}")
    print(f"Exp 17 aggregate: Δ all = {exp17['all'][0]:+.4f} ± {exp17['all'][1]:.4f}, "
          f"Δ hard = {exp17['hard'][0]:+.4f} ± {exp17['hard'][1]:.4f}, "
          f"Δ easy = {exp17['easy'][0]:+.4f} ± {exp17['easy'][1]:.4f}")


if __name__ == "__main__":
    make_saturation_figure()
