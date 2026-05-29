"""03_scalar_gate figure 카탈로그.

생성 figure:
  - train_curve.{pdf,png}        — train loss, ‖v‖, val NDCG by epoch
  - delta_ci_forest.{pdf,png}    — 3 anchor (baseline / α=10 / 02) 대비 CI forest
  - effective_magnitude.{pdf,png} — g, ‖v‖, g·‖v‖ 비교 + 01b α 와의 위치
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


def fig_train_curve(history: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(history["steps"], history["rank_losses"], color="#3b6e8f", linewidth=1)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("pairwise margin loss")
    axes[0].set_title("Train loss"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["steps"], history["v_norms"], color="#c47a2b", linewidth=1)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("‖v‖₂")
    axes[1].set_title("Direction magnitude (gated)"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history["val_epochs"], history["val_ndcg_all"], "-o",
                 color="#3b6e8f", label="val NDCG@10 (all)")
    axes[2].plot(history["val_epochs"], history["val_ndcg_confused"], "-s",
                 color="#d04545", label="val NDCG@10 (confused)")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("NDCG@10")
    axes[2].set_title("Val curves (early-stop)")
    axes[2].legend(loc="best", frameon=False); axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "train_curve")


def fig_delta_ci_forest(out_dir: Path, deltas: dict) -> None:
    rows = []
    label_map = {"vs_baseline": "vs baseline",
                 "vs_mean_diff_alpha10": "vs α=10 mean-diff",
                 "vs_02_learned": "vs 02 learned"}
    for key in ("vs_baseline", "vs_mean_diff_alpha10", "vs_02_learned"):
        for slice_name in ("all", "confused"):
            r = deltas[key].get(slice_name, {})
            if "mean_delta_ndcg10" not in r:
                continue
            rows.append((label_map[key], slice_name, r["mean_delta_ndcg10"],
                         r["ci_lo"], r["ci_hi"], r["n"]))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(rows))
    ds_color = {"all": "#3b6e8f", "confused": "#d04545"}
    for i, (ref, sn, m, lo, hi, n) in enumerate(rows):
        c = ds_color[sn]
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o" if sn == "all" else "s",
                    color=c, ecolor=c, capsize=4, markersize=6)
        if hi < 0:
            ax.text(lo - 0.005, i, "[−]", va="center", ha="right",
                    color="#7a3a3a", fontweight="bold", fontsize=10)
        elif lo > 0:
            ax.text(hi + 0.005, i, "[+]", va="center",
                    color="#2a8a3e", fontweight="bold", fontsize=10)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{ref}\n{sn}" for ref, sn, *_ in rows])
    ax.set_xlabel("Δ NDCG@10")
    ax.set_title("08 (scalar gate) — paired bootstrap CI vs three anchors")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest")


def fig_effective_magnitude(history: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    g = history["gate_final"]
    vn = history["v_norm_final"]
    eff = history["effective_magnitude"]

    # Compare to 01b alphas
    alpha_set = [0.5, 1.0, 2.0, 5.0, 10.0]
    bar_data = {"g (gate)": g, "‖v‖": vn, "g · ‖v‖": eff}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    names = list(bar_data.keys())
    vals = list(bar_data.values())
    bars = axes[0].bar(names, vals, color=["#5fa05c", "#c47a2b", "#3b6e8f"])
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     val + max(vals) * 0.02, f"{val:.3f}",
                     ha="center", fontsize=10)
    axes[0].set_ylabel("magnitude")
    axes[0].set_title("Final learned (g, ‖v‖, effective)")
    axes[0].grid(axis="y", alpha=0.3)

    # 01b alpha positions vs effective magnitude
    axes[1].bar([f"α={a:g}" for a in alpha_set], alpha_set,
                color="#cccccc", label="01b α (= effective magnitude)")
    axes[1].axhline(eff, color="#3b6e8f", linestyle="--", linewidth=2,
                    label=f"08 final g·‖v‖ = {eff:.3f}")
    axes[1].set_ylabel("effective magnitude")
    axes[1].set_title("08 의 effective magnitude vs 01b α sweep")
    axes[1].legend(loc="upper left", frameon=False)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "effective_magnitude")


def main() -> None:
    set_rc()
    base = PROJECT_ROOT / "outputs" / "03_scalar_gate" / DATASET / f"seed_{SEED}"
    out_dir = PROJECT_ROOT / "report" / "figures" / "03_scalar_gate"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = json.loads((base / "train_history.json").read_text())
    deltas = {
        "vs_baseline": json.loads((base / "delta_vs_baseline.json").read_text()),
        "vs_mean_diff_alpha10": json.loads((base / "delta_vs_mean_diff_alpha10.json").read_text()),
        "vs_02_learned": json.loads((base / "delta_vs_02_learned.json").read_text()),
    }
    fig_train_curve(history, out_dir)
    fig_delta_ci_forest(out_dir, deltas)
    fig_effective_magnitude(history, out_dir)
    print(f"figures → {out_dir}")


if __name__ == "__main__":
    main()
