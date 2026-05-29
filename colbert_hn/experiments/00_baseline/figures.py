"""Generate the 00_baseline figure catalogue (CLAUDE.md §16.2).

Reads artifacts from `outputs/00_baseline/{dataset}/seed_{SEED}/` and writes
PDF + PNG figures to `report/figures/00_baseline/`. The figures are then
embedded into `experiments/00_baseline/REPORT.md` (CLAUDE.md §3.9, §16.6).

Run:
    .venv/bin/python experiments/00_baseline/figures.py
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

DATASETS = ["scifact", "nfcorpus", "scidocs", "trec-covid", "fiqa", "arguana"]
DATASET_DISPLAY = {
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "scidocs": "SciDocs",
    "trec-covid": "TREC-COVID",
    "fiqa": "FiQA-2018",
    "arguana": "ArguAna",
}
DOMAIN = {
    "scifact": "Science",
    "nfcorpus": "Medical",
    "scidocs": "Science",
    "trec-covid": "Medical",
    "fiqa": "Finance",
    "arguana": "Argument",
}
PAPER_NDCG10 = {
    "scifact": 0.693,
    "nfcorpus": 0.338,
    "scidocs": 0.154,
    "trec-covid": 0.738,
    "fiqa": 0.356,
    "arguana": 0.463,
}
SEED = 42


def set_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_results() -> tuple[dict, dict]:
    base = PROJECT_ROOT / "outputs" / "00_baseline"
    per_q, agg = {}, {}
    for ds in DATASETS:
        seed_dir = base / ds / f"seed_{SEED}"
        with (seed_dir / "metrics_per_query.json").open() as f:
            per_q[ds] = json.load(f)
        with (seed_dir / "metrics_aggregate.json").open() as f:
            agg[ds] = json.load(f)
    return per_q, agg


def _save(fig, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def fig_paper_overlay(agg: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    measured = np.array([agg[ds]["all"]["ndcg_cut_10"] for ds in DATASETS])
    paper = np.array([PAPER_NDCG10[ds] for ds in DATASETS])
    labels = [DATASET_DISPLAY[ds] for ds in DATASETS]
    x = np.arange(len(DATASETS))
    bar_w = 0.55
    ax.bar(x, measured, bar_w, color="#3b6e8f", label="Ours (frozen ColBERT v2 re-impl.)")
    for i, p in enumerate(paper):
        ax.hlines(
            p, x[i] - bar_w / 2, x[i] + bar_w / 2,
            colors="#d04545", linestyles="--", linewidth=2,
            label="ColBERT v2 paper" if i == 0 else None,
        )
    for i, (m, p) in enumerate(zip(measured, paper)):
        delta = m - p
        within = abs(delta) <= 0.005
        color = "#2a8a3e" if within else "#7a3a3a"
        ax.text(
            x[i], max(m, p) + 0.025, f"Δ={delta:+.3f}",
            ha="center", fontsize=8, color=color, fontweight="bold" if within else "normal",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Baseline NDCG@10 vs ColBERT v2 paper (zero-shot BEIR)")
    ax.set_ylim(0, float(max(measured.max(), paper.max())) + 0.12)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "metrics_paper_overlay")


def fig_metric_at_k_curves(agg: dict, out_dir: Path) -> None:
    ks = [1, 3, 5, 10, 20]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, ds in zip(axes.flat, DATASETS):
        ndcg = [agg[ds]["all"][f"ndcg_cut_{k}"] for k in ks]
        recall = [agg[ds]["all"][f"recall_{k}"] for k in ks]
        precision = [agg[ds]["all"][f"P_{k}"] for k in ks]
        ax.plot(ks, ndcg, marker="o", label="NDCG@k", color="#3b6e8f")
        ax.plot(ks, recall, marker="s", label="Recall@k", color="#5fa05c")
        ax.plot(ks, precision, marker="^", label="P@k", color="#c47a2b")
        ax.set_title(f"{DATASET_DISPLAY[ds]} ({DOMAIN[ds]})")
        ax.set_xticks(ks)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("k")
    for ax in axes[:, 0]:
        ax.set_ylabel("metric")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02)
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, "metric_at_k_curves")


def fig_per_query_dist(per_q: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap("viridis")
    n = len(DATASETS)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    for ds, color in zip(DATASETS, colors):
        vals = sorted(v["ndcg_cut_10"] for v in per_q[ds].values())
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, ecdf, label=DATASET_DISPLAY[ds], color=color, linewidth=1.7)
    ax.set_xlabel("per-query NDCG@10")
    ax.set_ylabel("ECDF (fraction of queries ≤ x)")
    ax.set_title("Per-query NDCG@10 distribution (ECDF) across BEIR datasets")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "per_query_metric_dist")


def fig_confused_slice_size(agg: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    n_total = np.array([agg[ds]["_meta"]["n_queries"] for ds in DATASETS])
    n_conf = np.array([agg[ds]["_meta"]["n_confused"] for ds in DATASETS])
    n_clean = n_total - n_conf
    frac_conf = n_conf / n_total
    x = np.arange(len(DATASETS))
    ax.bar(x, n_clean, color="#cccccc", label="Top-1 correct")
    ax.bar(x, n_conf, bottom=n_clean, color="#d04545", label="Confused (top-1 ≠ rel)")
    for i, (frac, total) in enumerate(zip(frac_conf, n_total)):
        ax.text(
            x[i], total + n_total.max() * 0.04,
            f"{frac * 100:.1f}%\n(n={total})", ha="center", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY[ds] for ds in DATASETS], rotation=20, ha="right")
    ax.set_ylabel("# queries")
    ax.set_title("Confused-slice base rate per dataset (top-1 ≠ relevant)")
    ax.set_ylim(0, n_total.max() * 1.25)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "confused_slice_size")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "00_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q, agg = load_results()
    fig_paper_overlay(agg, out_dir)
    fig_metric_at_k_curves(agg, out_dir)
    fig_per_query_dist(per_q, out_dir)
    fig_confused_slice_size(agg, out_dir)
    print(f"figures written → {out_dir}")


if __name__ == "__main__":
    main()
