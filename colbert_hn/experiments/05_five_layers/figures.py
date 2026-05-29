"""05_five_layers figure 카탈로그.

생성:
  - train_curve.{pdf,png}        — train loss, total ‖v‖, val NDCG by epoch
  - layer_norms_bar.{pdf,png}    — per-layer ‖v_ℓ‖ 비교 + cos(v_ℓ, v_meandiff_l12)
  - delta_ci_forest.{pdf,png}    — 5 anchor 대비 paired bootstrap CI forest
  - ecdf_compare.{pdf,png}       — per-query NDCG@10 ECDF (baseline / α=10 / 02 / 03 / 04 / 05)
  - single_direction_summary.{pdf,png}     — single-direction trio (02/03/04) + 05 비교 (architecture progression)
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
    o5 = base / "05_five_layers" / DATASET / f"seed_{SEED}"
    return {
        "05_per_q": json.loads((o5 / "metrics_per_query.json").read_text()),
        "05_agg": json.loads((o5 / "metrics_aggregate.json").read_text()),
        "history": json.loads((o5 / "train_history.json").read_text()),
        "layer_norms": json.loads((o5 / "layer_norms.json").read_text()),
        "cosine": json.loads((o5 / "cosine_with_mean_diff.json").read_text()),
        "delta_vs_baseline": json.loads((o5 / "delta_vs_baseline.json").read_text()),
        "delta_vs_alpha10": json.loads((o5 / "delta_vs_mean_diff_alpha10.json").read_text()),
        "delta_vs_02": json.loads((o5 / "delta_vs_02_learned.json").read_text()),
        "delta_vs_03": json.loads((o5 / "delta_vs_03_scalar_gate.json").read_text()),
        "delta_vs_04": json.loads((o5 / "delta_vs_04_per_token_gate.json").read_text()),
        "baseline_per_q": json.loads(
            (base / "00_baseline" / DATASET / f"seed_{SEED}"
             / "metrics_per_query.json").read_text()),
        "alpha10_per_q": json.loads(
            (base / "01b_mean_diff_scaled" / DATASET / f"seed_{SEED}"
             / "alpha_10p0" / "metrics_per_query.json").read_text()),
        "02_per_q": json.loads(
            (base / "02_final_layer_vector" / DATASET / f"seed_{SEED}"
             / "metrics_per_query.json").read_text()),
        "03_per_q": json.loads(
            (base / "03_scalar_gate" / DATASET / f"seed_{SEED}"
             / "metrics_per_query.json").read_text()),
        "04_per_q": json.loads(
            (base / "04_per_token_gate" / DATASET / f"seed_{SEED}"
             / "metrics_per_query.json").read_text()),
    }


def fig_train_curve(d: dict, out_dir: Path) -> None:
    h = d["history"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(h["steps"], h["rank_losses"], color="#3b6e8f", linewidth=1)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("pairwise margin loss")
    axes[0].set_title("Train loss (rank)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(h["steps"], h["v_norms"], color="#c47a2b", linewidth=1)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("total ‖v‖ over all layers")
    axes[1].set_title("Aggregate magnitude (5 layers)")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(h["val_epochs"], h["val_ndcg_all"], "-o", color="#3b6e8f", label="val NDCG@10 (all)")
    axes[2].plot(h["val_epochs"], h["val_ndcg_confused"], "-s", color="#d04545",
                 label="val NDCG@10 (confused)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("NDCG@10")
    axes[2].set_title("Validation curves")
    axes[2].legend(loc="best", frameon=False)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "train_curve")


def fig_layer_norms(d: dict, out_dir: Path) -> None:
    layer_norms = d["layer_norms"]
    cos = d["cosine"]["cosine_per_layer_vs_mean_diff_l12"]
    layers = sorted(int(k) for k in layer_norms.keys())
    norms = [layer_norms[str(l)] for l in layers]
    coses = [cos[str(l)] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(layers) - 1, 1)) for i in range(len(layers))]
    axes[0].bar([f"ℓ={l}" for l in layers], norms, color=colors)
    for i, val in enumerate(norms):
        axes[0].text(i, val + max(norms) * 0.02, f"{val:.2f}", ha="center", fontsize=9)
    axes[0].set_ylabel("‖v_ℓ‖₂")
    axes[0].set_title("Per-layer learned direction magnitudes")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar([f"ℓ={l}" for l in layers], coses, color=colors)
    for i, val in enumerate(coses):
        axes[1].text(i, val + 0.02 if val >= 0 else val - 0.05,
                     f"{val:.3f}", ha="center", fontsize=9)
    axes[1].axhline(0, color="#888", linestyle="--", linewidth=1)
    axes[1].set_ylim(min(-0.1, min(coses) - 0.1), max(0.1, max(coses) + 0.15))
    axes[1].set_ylabel("cos(v_ℓ, v_mean_diff at ℓ=12)")
    axes[1].set_title("Direction alignment with mean-diff (ref ℓ=12)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "layer_norms_bar")


def fig_delta_ci_forest(d: dict, out_dir: Path) -> None:
    rows = []
    label_map = {
        "delta_vs_baseline": "vs baseline",
        "delta_vs_alpha10": "vs α=10 mean-diff",
        "delta_vs_02": "vs 02 learned",
        "delta_vs_03": "vs 03 scalar gate",
        "delta_vs_04": "vs 04 per-token gate",
    }
    for key, label in label_map.items():
        for slice_name in ("all", "confused"):
            r = d[key].get(slice_name, {})
            if "mean_delta_ndcg10" not in r:
                continue
            rows.append((label, slice_name, r["mean_delta_ndcg10"],
                         r["ci_lo"], r["ci_hi"], r["n"]))
    fig, ax = plt.subplots(figsize=(8.5, 6))
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
            ax.text(lo - 0.005, i, "[−]", va="center", ha="right", color="#7a3a3a",
                    fontweight="bold", fontsize=10)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{ref}\n{sn}" for ref, sn, *_ in rows])
    ax.set_xlabel("Δ NDCG@10")
    ax.set_title("05 (5-layer learned LSR) — paired bootstrap 95 % CI vs 5 anchors")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest")


def fig_ecdf_compare(d: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for per_q, color, label, lw in (
        (d["baseline_per_q"], "#888888", "baseline (00)", 1.5),
        (d["alpha10_per_q"], "#c47a2b", "mean-diff α=10 (01b)", 1.5),
        (d["02_per_q"], "#3b6e8f", "02 single-layer learned", 1.5),
        (d["03_per_q"], "#a3a3c2", "03 scalar gate", 1.3),
        (d["04_per_q"], "#5fa05c", "04 per-token gate", 1.5),
        (d["05_per_q"], "#d04545", "05 5-layer learned", 2.0),
    ):
        vals = np.sort([v["ndcg_cut_10"] for v in per_q.values()])
        n = len(vals)
        ax.plot(vals, np.arange(1, n + 1) / n, color=color, label=label, linewidth=lw)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-query NDCG@10")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-query NDCG@10 ECDF: 6 conditions on SciFact")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "ecdf_compare")


def fig_single_direction_summary(d: dict, out_dir: Path) -> None:
    """single-direction 단계 의 architectural progression 한눈에."""
    names = ["baseline\n(00)", "mean-diff\nα=10 (01b)",
             "02 learned\n(no gate)", "03 scalar\ngate", "04 per-token\ngate",
             "05 5-layer\nlearned"]
    keys = ["baseline_per_q", "alpha10_per_q",
            "02_per_q", "03_per_q", "04_per_q", "05_per_q"]
    means = []
    confused_means = []

    # baseline 의 confused slice
    base_runs = json.loads(
        (PROJECT_ROOT / "outputs" / "00_baseline" / DATASET
         / f"seed_{SEED}" / "runs.json").read_text())
    # qrels via BEIR
    from src.data import load_beir
    _, _, qrels = load_beir(DATASET, split="test")
    from src.slices import confused_slice
    conf = confused_slice(base_runs, qrels, k=1)

    for key in keys:
        per_q = d[key]
        vals_all = [v["ndcg_cut_10"] for v in per_q.values()]
        vals_conf = [per_q[q]["ndcg_cut_10"] for q in conf if q in per_q]
        means.append(np.mean(vals_all))
        confused_means.append(np.mean(vals_conf) if vals_conf else float("nan"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(names))
    w = 0.4
    bars_all = ax.bar(x - w / 2, means, w, color="#3b6e8f", label="all")
    bars_conf = ax.bar(x + w / 2, confused_means, w, color="#d04545", label="confused")
    for bar, val in zip(bars_all, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"{val:.3f}", ha="center", fontsize=8)
    for bar, val in zip(bars_conf, confused_means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                    f"{val:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Architecture progression (SciFact, seed 42): all vs confused slice")
    ax.set_ylim(0, max(max(means), max(c for c in confused_means if not np.isnan(c))) + 0.08)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "single_direction_summary")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "05_five_layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load()
    fig_train_curve(d, out_dir)
    fig_layer_norms(d, out_dir)
    fig_delta_ci_forest(d, out_dir)
    fig_ecdf_compare(d, out_dir)
    fig_single_direction_summary(d, out_dir)
    print(f"figures → {out_dir}")


if __name__ == "__main__":
    main()
