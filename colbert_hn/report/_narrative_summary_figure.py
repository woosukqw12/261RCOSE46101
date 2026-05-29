"""NARRATIVE.md 의 모든 실험에 대한 Δ NDCG@10 종합 bar plot.

각 실험의 Δ all / Δ hard / Δ easy 를 grouped bar 로 시각화.
SciFact 의 self-baseline (frozen ColBERT) 대비.

Output: report/figures/narrative_summary/all_experiments_delta_ndcg.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "report" / "figures" / "narrative_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# (label, family, Δ all, Δ hard, Δ easy, note)
# None = 미보고 (해당 실험에서 측정/보고 안 됨)
EXPERIMENTS = [
    # §2 단순 개입
    ("Mean-diff α=10 (A)",            "non-learned",  None,    +0.064,  None,    "α-sweep best"),
    ("Single learned direction (B)",   "learned-dir",  None,    +0.044,  None,    "768 params"),
    ("Scalar gate (B)",                "gating",       -0.020,  None,    None,    "gradient bottleneck"),
    ("Per-token gate (B)",             "gating",       None,    -0.003,  None,    "saturate to 1.0"),
    ("Router K=2 (C)",                  "router",      +0.015,  +0.039,  None,    "eff K=1.41"),
    ("Router K=4 (C)",                  "router",      +0.015,  +0.045,  None,    "eff K=1.23"),
    ("Router K=8 (C)",                  "router",      -0.038,  +0.049,  None,    "anchor 손상"),
    ("Random direction α=10 (D)",       "control",      0.000,  +0.011,  None,    "magnitude-only falsified"),
    # §3 LoRA
    ("Plain LoRA r=8 (E)",              "lightweight", +0.001,  +0.104,  -0.085,  "24 intervention pts, 3-seed"),
    # §4 mechanism
    ("False negative removal (F)",      "training-data", -0.004, +0.080, -0.073,  "label noise mediation"),
    ("In-batch easy neg (G)",           "training-data", +0.021, +0.065, -0.017,  "first strict net+"),
    ("Continuous σ weighting (footnote)", "training-data", +0.006, +0.085, -0.060, "α_w=10 sigmoid"),
    ("Relational anchor (H₁)",          "anchoring",   +0.029,  +0.101,  -0.031,  "Sim Frobenius², 2/3 strict"),
    ("Per-token absolute anchor (H₂) *", "anchoring",  +0.030,  +0.092,  -0.021,  "3/3 strict, best lightweight"),
    ("Negative-side anchor (I)",         "anchoring",  +0.028,  +0.077,  -0.014,  "3/3 strict, saturation proof"),
]


FAMILY_COLOR = {
    "non-learned":   "#888888",
    "learned-dir":   "#777777",
    "gating":        "#999999",
    "router":        "#aaaaaa",
    "control":       "#cccccc",
    "lightweight":   "#cc4444",
    "training-data": "#ff7f0e",
    "anchoring":     "#2ca02c",
    "cross-domain":  "#9467bd",
}


def set_rc():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 9, "legend.fontsize": 8,
        "lines.linewidth": 1.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def make_figure():
    set_rc()

    n = len(EXPERIMENTS)
    labels = [e[0] for e in EXPERIMENTS]
    d_all_raw = [e[2] for e in EXPERIMENTS]
    d_err_raw = [e[3] for e in EXPERIMENTS]
    d_cor_raw = [e[4] for e in EXPERIMENTS]

    def fillna(v):
        return v if v is not None else 0.0

    d_all = [fillna(v) for v in d_all_raw]
    d_err = [fillna(v) for v in d_err_raw]
    d_cor = [fillna(v) for v in d_cor_raw]

    x = np.arange(n)
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 6.5))

    ax.bar(x - width, d_all, width, color="#444",
           edgecolor="black", linewidth=0.4, label="Δ all")
    ax.bar(x, d_err, width, color="#cc4444",
           edgecolor="black", linewidth=0.4, label="Δ hard")
    ax.bar(x + width, d_cor, width, color="#1f77b4",
           edgecolor="black", linewidth=0.4, label="Δ easy")

    # value labels
    for i, (v_all, v_err, v_cor) in enumerate(zip(d_all_raw, d_err_raw, d_cor_raw)):
        if v_all is not None:
            ax.text(i - width, v_all + (0.004 if v_all >= 0 else -0.008),
                    f"{v_all:+.3f}", ha="center", fontsize=6.5, rotation=90)
        else:
            ax.text(i - width, 0.003, "n/a", ha="center", fontsize=6, color="#999", rotation=90)
        if v_err is not None:
            ax.text(i, v_err + (0.004 if v_err >= 0 else -0.008),
                    f"{v_err:+.3f}", ha="center", fontsize=6.5, rotation=90)
        else:
            ax.text(i, 0.003, "n/a", ha="center", fontsize=6, color="#999", rotation=90)
        if v_cor is not None:
            ax.text(i + width, v_cor + (0.004 if v_cor >= 0 else -0.008),
                    f"{v_cor:+.3f}", ha="center", fontsize=6.5, rotation=90)
        else:
            ax.text(i + width, 0.003, "n/a", ha="center", fontsize=6, color="#999", rotation=90)

    # reference lines
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(0.030, color="#2ca02c", linestyle="--", linewidth=0.9, alpha=0.6,
               label="anchor frontier (+0.030) — best lightweight method")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=8)
    ax.set_ylabel("Δ NDCG@10 vs frozen baseline (SciFact)")
    ax.set_title("Δ NDCG@10 across all SciFact experiments (§2–§4.5) — anchor frontier saturates at +0.030")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.13, 0.16)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"all_experiments_delta_ndcg.{ext}")
    plt.close(fig)
    print(f"saved → {OUT_DIR}/all_experiments_delta_ndcg.{{pdf,png}}")


if __name__ == "__main__":
    make_figure()
