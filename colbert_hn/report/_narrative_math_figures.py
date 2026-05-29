"""NARRATIVE.md 의 후반 수학 보강용 CPU figures.

생성:
  - redistribution_identity.{pdf,png}  — §4.1 accounting identity 검증 scatter
                                          (predicted Δ_easy vs measured, 3 seed)
  - anchor_equilibrium.{pdf,png}       — §4.4 soft equilibrium 의 도식
                                          (cosine 0.824 의 force balance schematic + measured points)
  - fn_rate_vs_catastrophe.{pdf,png}   — §5 cross-domain
                                          (FN rate vs Δ all magnitude, SciFact / NF / FiQA)

모두 *기존 측정값* 으로 그리는 schematic / synthesis — 추가 학습 없음.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "report" / "figures" / "narrative_math"
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


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)


def fig_redistribution_identity():
    """§4.1 — accounting identity 검증.
    Δ_easy = (Δ_all - w_hard·Δ_hard) / w_easy 의 예측 vs 측정 (3 seeds).
    """
    set_rc()
    fig, ax = plt.subplots(figsize=(6.5, 6))

    w_hard, w_easy = 0.457, 0.543
    # 3 seeds of plain LoRA on SciFact
    seeds = [42, 1337, 2024]
    d_all = [-0.010, -0.004, +0.018]
    d_hard = [+0.091, +0.097, +0.123]
    d_easy_measured = [-0.095, -0.089, -0.072]

    d_easy_predicted = [
        (d_all[i] - w_hard * d_hard[i]) / w_easy for i in range(3)
    ]

    # scatter
    for s, p, m, c in zip(seeds, d_easy_predicted, d_easy_measured, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        ax.scatter(p, m, s=140, color=c, alpha=0.85, edgecolor="black", linewidth=0.8,
                   label=f"seed {s}: predicted={p:+.4f}, measured={m:+.4f}")
        ax.annotate(f"s{s}", (p, m), textcoords="offset points", xytext=(8, 6), fontsize=9)

    # y=x line
    xs = np.linspace(-0.15, -0.05, 50)
    ax.plot(xs, xs, "--", color="grey", alpha=0.5, label="$y = x$  (perfect identity)")

    # 3-seed mean point
    pm, mm = np.mean(d_easy_predicted), np.mean(d_easy_measured)
    ax.scatter([pm], [mm], s=280, color="black", marker="*", zorder=10,
               label=f"3-seed mean: predicted={pm:+.4f}, measured={mm:+.4f}")

    ax.set_xlabel(r"Predicted $\Delta_{\mathrm{easy}} = (\Delta_{\mathrm{all}} - w_{\mathrm{hard}}\,\Delta_{\mathrm{hard}}) / w_{\mathrm{easy}}$")
    ax.set_ylabel(r"Measured $\Delta_{\mathrm{easy}}$ (3-seed)")
    ax.set_title("§4.1 Accounting identity verification — 99 % match (predicted ≈ measured)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.12, -0.06)
    ax.set_ylim(-0.12, -0.06)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "redistribution_identity")
    print("→ redistribution_identity")


def fig_anchor_equilibrium():
    """§4.4 — soft equilibrium attractor 의 force-balance schematic + measured cosine.
    """
    set_rc()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5),
                                       gridspec_kw={"width_ratios": [1.2, 1.0]})

    # ===== Left: force-balance schematic =====
    # Conceptual: theta_frozen ↔ theta_LoRA, two forces meeting at equilibrium
    theta_frozen = np.array([0.0, 0.0])
    theta_eq = np.array([1.0, 0.0])  # equilibrium point
    theta_LoRA_only = np.array([2.2, 0.5])  # if no anchor

    # Draw axis
    ax_l.set_xlim(-0.5, 3.0)
    ax_l.set_ylim(-1.5, 1.5)
    ax_l.set_aspect("equal")
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax_l.spines[s].set_visible(False)

    # Frozen point
    ax_l.scatter(*theta_frozen, s=300, color="#888", zorder=5, marker="o")
    ax_l.annotate("$\\theta_{\\mathrm{frozen}}$\n(cos = 1)",
                  theta_frozen, textcoords="offset points", xytext=(-50, -10),
                  fontsize=11, ha="center")

    # Equilibrium point
    ax_l.scatter(*theta_eq, s=300, color="#2ca02c", zorder=5, marker="*", edgecolor="black")
    ax_l.annotate("$\\theta^{\\star}$  (cos ≈ 0.824)\nsoft equilibrium",
                  theta_eq, textcoords="offset points", xytext=(-10, -55),
                  fontsize=11, ha="center", color="#2ca02c", weight="bold")

    # LoRA-only point
    ax_l.scatter(*theta_LoRA_only, s=250, color="#cc4444", zorder=5, marker="o", alpha=0.5)
    ax_l.annotate("$\\theta_{\\mathrm{LoRA\\,only}}$\n(plain LoRA, cos $\\ll$ 1)",
                  theta_LoRA_only, textcoords="offset points", xytext=(10, 5),
                  fontsize=10, ha="left", color="#cc4444")

    # Margin push arrow (from frozen toward LoRA)
    ax_l.annotate("", xy=(0.85, 0.0), xytext=(0.05, 0.0),
                  arrowprops=dict(arrowstyle="->", color="#cc4444", lw=2.5))
    ax_l.text(0.45, 0.18, r"$g_{\mathrm{push}} = \nabla_\theta \mathcal{L}_{\mathrm{margin}}$",
              fontsize=10, color="#cc4444", ha="center")

    # Anchor pull arrow (from LoRA-only back toward frozen, at equilibrium)
    ax_l.annotate("", xy=(1.15, 0.0), xytext=(2.0, 0.4),
                  arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.5))
    ax_l.text(1.7, 0.5, r"$g_{\mathrm{pull}} = \lambda\,\nabla_\theta \mathcal{R}_{\mathrm{abs}}$",
              fontsize=10, color="#1f77b4", ha="center")

    # Title
    ax_l.set_title("§4.4 Soft equilibrium attractor (schematic)\n"
                   "$g_{\\mathrm{push}} + g_{\\mathrm{pull}} = 0$ at $\\theta^{\\star}$",
                   pad=15)

    # ===== Right: measured cosine 3 seeds =====
    seeds = [42, 1337, 2024]
    cos_measured = [0.820, 0.823, 0.830]  # Diagnostic B results
    cos_mean = np.mean(cos_measured)
    cos_std = np.std(cos_measured, ddof=1)

    ax_r.bar(range(len(seeds)), cos_measured, color="#2ca02c", alpha=0.85,
              edgecolor="black", linewidth=0.6, width=0.5)
    for i, c in enumerate(cos_measured):
        ax_r.text(i, c + 0.005, f"{c:.3f}", ha="center", fontsize=10)
    ax_r.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6,
                  label=r"$\hat h^{\mathrm{LoRA}}_t = \hat h^{\mathrm{frozen}}_t$  (hard constraint, cos = 1)")
    ax_r.axhline(cos_mean, color="#2ca02c", linestyle=":", linewidth=1.4, alpha=0.9,
                  label=f"3-seed mean = {cos_mean:.3f} ± {cos_std:.3f}")
    ax_r.set_xticks(range(len(seeds)))
    ax_r.set_xticklabels([f"seed {s}" for s in seeds])
    ax_r.set_ylabel(r"$\mathbb{E}_{x,t}\,[\,\langle \hat h^{\mathrm{LoRA}}_t,\,\hat h^{\mathrm{frozen}}_t\rangle\,|\,x \in \mathcal{Q}_{\mathrm{easy}}\,]$")
    ax_r.set_title("§4.4 Measured anchor cosine — partial satisfaction")
    ax_r.set_ylim(0.70, 1.05)
    ax_r.legend(loc="lower right", fontsize=8, frameon=False)
    ax_r.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "anchor_equilibrium")
    print("→ anchor_equilibrium")


def fig_fn_rate_vs_catastrophe():
    """§5 — Three-intervention 비교: hard contrast 의 유지/제거가 결정적, ρ_FN 은 dominant predictor 아님.

    데이터:
      - plain LoRA (hard 유지 + FN 유지) : SciFact +0.001, NF -0.320, FiQA -0.347
      - FN-removal (hard 유지 + FN 제거): NF -0.316 (Exp 12 × NF × 3-seed mean)
      - in-batch easy (hard 제거 + FN 제거): SciFact +0.021, NF -0.084, FiQA -0.020
      - ρ_FN: SciFact 0.14, NF 0.611 (직접 측정, Exp 12 filter), FiQA 0.45 (추정)
    """
    set_rc()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    datasets = ["SciFact", "NFCorpus", "FiQA"]
    rho_fn = [0.14, 0.611, 0.45]
    delta_plain_lora = [+0.001, -0.320, -0.347]
    delta_fn_only = [None, -0.316, None]
    delta_in_batch = [+0.021, -0.084, -0.020]

    x = np.arange(len(datasets))
    width = 0.27
    bars_plain = ax.bar(x - width, delta_plain_lora, width, color="#cc4444",
                        edgecolor="black", linewidth=0.6, alpha=0.85,
                        label="plain LoRA (hard kept + FN kept)")
    fn_vals = [v if v is not None else 0 for v in delta_fn_only]
    bars_fn = ax.bar(x, fn_vals, width, color="#ff7f0e",
                     edgecolor="black", linewidth=0.6, alpha=0.85,
                     label="FN-removal (hard kept + FN removed)")
    bars_in = ax.bar(x + width, delta_in_batch, width, color="#2ca02c",
                     edgecolor="black", linewidth=0.6, alpha=0.85,
                     label="in-batch easy (hard removed + FN removed)")

    for b, v in zip(bars_plain, delta_plain_lora):
        off = 0.012 if v >= 0 else -0.030
        ax.text(b.get_x() + b.get_width()/2, v + off, f"{v:+.3f}",
                ha="center", fontsize=8)
    for b, v, raw in zip(bars_fn, fn_vals, delta_fn_only):
        if raw is None:
            ax.text(b.get_x() + b.get_width()/2, 0.005, "n/a", ha="center",
                    fontsize=7, color="#888")
            continue
        off = 0.012 if v >= 0 else -0.030
        ax.text(b.get_x() + b.get_width()/2, v + off, f"{v:+.3f}",
                ha="center", fontsize=8)
    for b, v in zip(bars_in, delta_in_batch):
        off = 0.012 if v >= 0 else -0.030
        ax.text(b.get_x() + b.get_width()/2, v + off, f"{v:+.3f}",
                ha="center", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, rho_fn, "o--", color="#1f77b4", markersize=10, linewidth=1.5,
             label=r"$\rho_{\mathrm{FN}}$  (false-negative rate)")
    for i, r in enumerate(rho_fn):
        marker = " (measured)" if i == 1 else ""
        ax2.text(i, r + 0.03, f"{r:.2f}{marker}", ha="center",
                 color="#1f77b4", fontsize=8, weight="bold")
    ax2.set_ylim(0.0, 0.78)
    ax2.set_ylabel(r"$\rho_{\mathrm{FN}}$", color="#1f77b4")
    ax2.tick_params(axis="y", colors="#1f77b4")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#1f77b4")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(r"$\Delta\,$NDCG@10 (all) vs frozen baseline")
    ax.set_ylim(-0.42, 0.10)
    ax.set_title(r"§5 — hard contrast (keep/remove) is the decisive factor; $\rho_{\mathrm{FN}}$ is not the dominant predictor")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=7.5, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fn_rate_vs_catastrophe")
    print("→ fn_rate_vs_catastrophe")


def main():
    fig_redistribution_identity()
    fig_anchor_equilibrium()
    fig_fn_rate_vs_catastrophe()
    print(f"\nfigures → {OUT_DIR}")


if __name__ == "__main__":
    main()
