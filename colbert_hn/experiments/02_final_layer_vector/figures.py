"""02_final_layer_vector figure 카탈로그.

생성 figure:
  - train_curve.{pdf,png}      — train loss, val NDCG, ‖v‖ trajectory
  - delta_ci_forest.{pdf,png}  — paired bootstrap CI vs baseline & vs α=10 anchor
  - delta_violin.{pdf,png}     — per-query Δ NDCG@10 violin
  - ecdf_compare.{pdf,png}     — per-query NDCG@10 ECDF (baseline vs α=10 vs 02)
  - direction_compare.{pdf,png} — ‖v_learned‖ vs ‖v_mean_diff‖ + cosine
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

from src.metrics import align_per_query  # noqa: E402

DATASET = "scifact"
SEED = 42
ALPHA10_NAME = "alpha_10p0"


def set_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif", "font.size": 10,
            "axes.titlesize": 11, "axes.labelsize": 10,
            "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "axes.spines.top": False, "axes.spines.right": False,
            "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        }
    )


def _save(fig, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def _load() -> dict:
    base = PROJECT_ROOT / "outputs"
    o2 = base / "02_final_layer_vector" / DATASET / f"seed_{SEED}"
    o1b = base / "01b_mean_diff_scaled" / DATASET / f"seed_{SEED}" / ALPHA10_NAME
    obase = base / "00_baseline" / DATASET / f"seed_{SEED}"
    return {
        "02_per_q": json.loads((o2 / "metrics_per_query.json").read_text()),
        "02_agg": json.loads((o2 / "metrics_aggregate.json").read_text()),
        "history": json.loads((o2 / "train_history.json").read_text()),
        "delta_vs_baseline": json.loads((o2 / "delta_vs_baseline.json").read_text()),
        "delta_vs_alpha10": json.loads((o2 / "delta_vs_mean_diff_alpha10.json").read_text()),
        "cosine": json.loads((o2 / "cosine_with_mean_diff.json").read_text()),
        "alpha10_per_q": json.loads((o1b / "metrics_per_query.json").read_text()),
        "baseline_per_q": json.loads((obase / "metrics_per_query.json").read_text()),
    }


def fig_train_curve(d: dict, out_dir: Path) -> None:
    h = d["history"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(h["steps"], h["rank_losses"], color="#3b6e8f", linewidth=1)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("pairwise margin loss")
    axes[0].set_title("Train loss (rank)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(h["steps"], h["v_norms"], color="#c47a2b", linewidth=1)
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("‖v‖₂")
    axes[1].set_title("Direction magnitude trajectory")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(h["val_epochs"], h["val_ndcg_all"], "-o", color="#3b6e8f", label="val NDCG@10 (all)")
    axes[2].plot(h["val_epochs"], h["val_ndcg_confused"], "-s", color="#d04545",
                 label="val NDCG@10 (confused)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("NDCG@10")
    axes[2].set_title("Validation curves (early-stop signal)")
    axes[2].legend(loc="best", frameon=False)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "train_curve")


def fig_delta_ci_forest(d: dict, out_dir: Path) -> None:
    rows = []
    for ref_name, key in (("vs baseline", "delta_vs_baseline"),
                          ("vs α=10 mean-diff", "delta_vs_alpha10")):
        for slice_name in ("all", "confused"):
            r = d[key].get(slice_name, {})
            if "mean_delta_ndcg10" not in r:
                continue
            rows.append((ref_name, slice_name, r["mean_delta_ndcg10"],
                         r["ci_lo"], r["ci_hi"], r["n"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(rows))
    ds_color = {"all": "#3b6e8f", "confused": "#d04545"}
    for i, (ref, sn, m, lo, hi, n) in enumerate(rows):
        c = ds_color[sn]
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o" if sn == "all" else "s",
                    color=c, ecolor=c, capsize=4, markersize=6)
        if lo > 0:
            ax.text(hi + 0.005, i, "[+]", va="center", color="#2a8a3e",
                    fontweight="bold", fontsize=10)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    labels = [f"{ref}\n{sn}" for ref, sn, *_ in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ NDCG@10")
    ax.set_title("02 (learned v) — paired bootstrap 95 % CI vs two anchors")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest")


def fig_delta_violin(d: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (ref_name, ref_per_q) in zip(
        axes,
        [("vs baseline", d["baseline_per_q"]), ("vs α=10 mean-diff", d["alpha10_per_q"])],
    ):
        data = []
        labels = []
        for slice_name, qids in (
            ("all", set(d["02_per_q"].keys())),
            ("confused", set(d["delta_vs_baseline"].get("confused", {}).get("n", 0)
                             and [q for q in d["02_per_q"].keys()
                                  if d["baseline_per_q"].get(q, {}).get("ndcg_cut_1", 0) == 0])),
        ):
            ours, base, _ = align_per_query(
                {q: vv for q, vv in d["02_per_q"].items() if q in qids},
                {q: vv for q, vv in ref_per_q.items() if q in qids},
                metric="ndcg_cut_10",
            )
            if len(ours) == 0:
                continue
            data.append(ours - base)
            labels.append(f"{slice_name}\n(n={len(ours)})")
        if not data:
            continue
        parts = ax.violinplot(data, showmeans=True, showmedians=False)
        for pc, c in zip(parts["bodies"], ["#3b6e8f", "#d04545"]):
            pc.set_facecolor(c)
            pc.set_alpha(0.55)
        ax.axhline(0, color="#888", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("per-query Δ NDCG@10")
        ax.set_title(ref_name)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("02 per-query Δ distributions vs two anchors", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "delta_violin")


def fig_ecdf_compare(d: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for per_q, color, label, lw in (
        (d["baseline_per_q"], "#888888", "baseline (00)", 1.5),
        (d["alpha10_per_q"], "#c47a2b", "mean-diff α=10 (01b)", 1.7),
        (d["02_per_q"], "#3b6e8f", "learned v (02)", 1.8),
    ):
        vals = np.sort([v["ndcg_cut_10"] for v in per_q.values()])
        n = len(vals)
        ax.plot(vals, np.arange(1, n + 1) / n, color=color, label=label, linewidth=lw)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-query NDCG@10")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-query NDCG@10 ECDF: baseline vs α=10 vs learned (SciFact)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "ecdf_compare")


def fig_direction_compare(d: dict, out_dir: Path) -> None:
    cos = d["cosine"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    names = ["mean-diff (01)", "learned (02)"]
    norms = [cos["v_mean_diff_norm"], cos["v_learned_norm"]]
    bars = ax1.bar(names, norms, color=["#c47a2b", "#3b6e8f"])
    for bar, val in zip(bars, norms):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + max(norms) * 0.02,
                 f"{val:.3f}", ha="center", fontsize=10)
    ax1.set_ylabel("‖v‖₂")
    ax1.set_title("Direction magnitude")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(["cos(v_learned, v_mean_diff)"], [cos["cosine_similarity"]],
            color="#5fa05c")
    ax2.text(0, cos["cosine_similarity"] + 0.03,
             f"{cos['cosine_similarity']:.3f}", ha="center", fontsize=11)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axhline(0, color="#888", linestyle="--", linewidth=1)
    ax2.axhline(0.9, color="#d04545", linestyle=":", linewidth=1,
                label="H5 threshold (0.9 = magnitude-only)")
    ax2.set_ylabel("cosine similarity")
    ax2.set_title("Direction alignment: learned vs mean-diff")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "direction_compare")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "02_final_layer_vector"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load()
    fig_train_curve(d, out_dir)
    fig_delta_ci_forest(d, out_dir)
    fig_delta_violin(d, out_dir)
    fig_ecdf_compare(d, out_dir)
    fig_direction_compare(d, out_dir)
    print(f"figures → {out_dir}")


if __name__ == "__main__":
    main()
