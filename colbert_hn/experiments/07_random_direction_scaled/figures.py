"""07_random_direction_scaled figure 카탈로그.

생성:
  - direction_compare.{pdf,png}   — 같은 magnitude (α=10), 다른 direction 의 효과 대비
  - delta_ci_forest.{pdf,png}     — paired bootstrap 95 % CI vs baseline + 01b α=10
  - ecdf_compare.{pdf,png}        — per-query NDCG@10 ECDF: baseline / random / mean-diff
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
    o7 = base / "07_random_direction_scaled" / DATASET / f"seed_{SEED}"
    a10 = base / "01b_mean_diff_scaled" / DATASET / f"seed_{SEED}" / "alpha_10p0"
    base_d = base / "00_baseline" / DATASET / f"seed_{SEED}"
    return {
        "07_per_q": json.loads((o7 / "metrics_per_query.json").read_text()),
        "07_agg": json.loads((o7 / "metrics_aggregate.json").read_text()),
        "delta_vs_baseline": json.loads((o7 / "delta_vs_baseline.json").read_text()),
        "delta_vs_alpha10": json.loads((o7 / "delta_vs_mean_diff_alpha10.json").read_text()),
        "alpha10_per_q": json.loads((a10 / "metrics_per_query.json").read_text()),
        "alpha10_agg": json.loads((a10 / "metrics_aggregate.json").read_text()),
        "baseline_per_q": json.loads((base_d / "metrics_per_query.json").read_text()),
        "baseline_agg": json.loads((base_d / "metrics_aggregate.json").read_text()),
    }


def fig_direction_compare(d: dict, out_dir: Path) -> None:
    """같은 magnitude, 다른 direction 의 효과 차이."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = ["baseline\n(no intervention)", "07 random\n(α=10)", "01b mean-diff\n(α=10)"]
    vals = [
        d["baseline_agg"]["all"]["ndcg_cut_10"],
        d["07_agg"]["all"]["ndcg_cut_10"],
        d["alpha10_agg"]["all"]["ndcg_cut_10"],
    ]
    confused = [
        d["baseline_agg"]["confused"]["ndcg_cut_10"],
        d["07_agg"]["confused"]["ndcg_cut_10"],
        d["alpha10_agg"]["confused"]["ndcg_cut_10"],
    ]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, vals, w, color="#3b6e8f", label="all slice")
    ax.bar(x + w / 2, confused, w, color="#d04545", label="confused slice")
    for i, (a, c) in enumerate(zip(vals, confused)):
        ax.text(i - w / 2, a + 0.012, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + w / 2, c + 0.012, f"{c:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Same magnitude (‖v‖=10), different direction → very different ranking")
    ax.set_ylim(0, max(max(vals), max(confused)) + 0.10)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "direction_compare")


def fig_delta_ci_forest(d: dict, out_dir: Path) -> None:
    rows = []
    for key, label in (
        ("delta_vs_baseline", "vs baseline (00)"),
        ("delta_vs_alpha10", "vs 01b α=10 (mean-diff)"),
    ):
        for sn in ("all", "confused"):
            r = d[key].get(sn, {})
            if "mean_delta_ndcg10" not in r:
                continue
            rows.append((label, sn, r["mean_delta_ndcg10"], r["ci_lo"], r["ci_hi"], r["n"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ds_color = {"all": "#3b6e8f", "confused": "#d04545"}
    y = np.arange(len(rows))
    for i, (ref, sn, m, lo, hi, n) in enumerate(rows):
        c = ds_color[sn]
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
    ax.set_title("07 random × α=10 — paired bootstrap 95% CI")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest")


def fig_ecdf_compare(d: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for per_q, color, label, lw in (
        (d["baseline_per_q"], "#888888", "baseline (00)", 1.5),
        (d["07_per_q"], "#c47a2b", "07 random × α=10", 1.7),
        (d["alpha10_per_q"], "#3b6e8f", "01b mean-diff × α=10", 1.7),
    ):
        vals = np.sort([v["ndcg_cut_10"] for v in per_q.values()])
        n = len(vals)
        ax.plot(vals, np.arange(1, n + 1) / n, color=color, label=label, linewidth=lw)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-query NDCG@10")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-query NDCG@10 ECDF: random vs mean-diff direction at α=10")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "ecdf_compare")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "07_random_direction_scaled"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load()
    fig_direction_compare(d, out_dir)
    fig_delta_ci_forest(d, out_dir)
    fig_ecdf_compare(d, out_dir)
    print(f"figures → {out_dir}")


if __name__ == "__main__":
    main()
