"""Unified figures for 01 (mean-diff) — raw + magnitude sweep.

본 figures.py 는 `01_mean_diff` (raw v) 와 `01b_mean_diff_scaled` (α sweep) 의
artifact 를 모두 읽어 통합 보고서 `report/01_mean_diff_report.md` 의 figure
일체를 생성한다.

Run:
    .venv/bin/python experiments/01_mean_diff/figures.py
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

# Raw run datasets (01_mean_diff)
RAW_DATASETS = ["scifact", "nfcorpus", "fiqa"]
DATASET_DISPLAY = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA-2018"}

# Sweep config (01b_mean_diff_scaled)
SWEEP_DATASET = "scifact"
ALPHAS = (0.5, 1.0, 2.0, 5.0, 10.0)
SWEEP_FOCUS_ALPHA = 5.0  # for ECDF / detail panel

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


def _save(fig, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def _alpha_dir_name(alpha: float) -> str:
    return "alpha_" + f"{alpha:.1f}".replace(".", "p")


def load_raw() -> dict:
    """01_mean_diff (raw v) artifacts across RAW_DATASETS."""
    base = PROJECT_ROOT / "outputs"
    out = {}
    for ds in RAW_DATASETS:
        mdir = base / "01_mean_diff" / ds / f"seed_{SEED}"
        bdir = base / "00_baseline" / ds / f"seed_{SEED}"
        if not (mdir / "metrics_per_query.json").exists():
            continue
        out[ds] = {
            "baseline_per_q": json.loads((bdir / "metrics_per_query.json").read_text()),
            "mean_diff_per_q": json.loads((mdir / "metrics_per_query.json").read_text()),
            "delta": json.loads((mdir / "delta_vs_baseline.json").read_text()),
            "stats": json.loads((mdir / "triplet_stats.json").read_text()),
        }
    return out


def load_sweep() -> dict:
    """01b_mean_diff_scaled (α sweep) artifacts on SWEEP_DATASET."""
    base = PROJECT_ROOT / "outputs"
    root = base / "01b_mean_diff_scaled" / SWEEP_DATASET / f"seed_{SEED}"
    if not (root / "triplet_stats.json").exists():
        return {}
    out: dict = {
        "stats": json.loads((root / "triplet_stats.json").read_text()),
        "summary": json.loads((root / "sweep_summary.json").read_text()),
        "alphas": {},
    }
    bdir = base / "00_baseline" / SWEEP_DATASET / f"seed_{SEED}"
    out["baseline_per_q"] = json.loads((bdir / "metrics_per_query.json").read_text())
    for alpha in ALPHAS:
        adir = root / _alpha_dir_name(alpha)
        if not (adir / "metrics_per_query.json").exists():
            continue
        out["alphas"][alpha] = {
            "per_q": json.loads((adir / "metrics_per_query.json").read_text()),
            "delta": json.loads((adir / "delta_vs_baseline.json").read_text()),
        }
    return out


# ---------------------------------------------------------- raw figures (3-ds)


def fig_raw_delta_ci_forest(raw: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rows = []
    for ds in RAW_DATASETS:
        if ds not in raw:
            continue
        d = raw[ds]["delta"]
        for slice_name in ("all", "confused"):
            dd = d.get(slice_name, {})
            if "mean_delta_ndcg10" not in dd:
                continue
            rows.append(
                (
                    DATASET_DISPLAY[ds], slice_name,
                    dd["mean_delta_ndcg10"], dd["ci_lo"], dd["ci_hi"], dd["n"],
                )
            )
    if not rows:
        plt.close(fig)
        return
    y = np.arange(len(rows))
    ds_color = {"all": "#3b6e8f", "confused": "#d04545"}
    for i, (ds, slice_name, m, lo, hi, n) in enumerate(rows):
        c = ds_color[slice_name]
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, ecolor=c,
                    capsize=4, markersize=6)
        ax.text(hi + 0.0008, i, f"n={n}", va="center", fontsize=8, color="#555")
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    labels = [f"{ds}\n{slice_name}" for ds, slice_name, *_ in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ NDCG@10  (raw mean-diff − baseline)")
    ax.set_title("Raw unscaled mean-diff: paired bootstrap 95 % CI on Δ NDCG@10")
    ax.invert_yaxis()
    handles = [
        plt.Line2D([0], [0], color=ds_color[s], marker="o", linestyle="", label=s)
        for s in ("all", "confused")
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "raw_delta_ci_forest")


def fig_v_norm(raw: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [DATASET_DISPLAY[d] for d in RAW_DATASETS if d in raw]
    norms = [raw[d]["stats"]["v_norm"] for d in RAW_DATASETS if d in raw]
    mean_abs = [raw[d]["stats"]["v_mean_abs"] for d in RAW_DATASETS if d in raw]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, norms, w, color="#3b6e8f", label="‖v‖₂")
    ax.bar(x + w / 2, mean_abs, w, color="#c47a2b", label="mean |v_i|")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("magnitude")
    ax.set_title("Magnitude of raw mean-diff direction v per dataset")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "raw_v_norm")


# ---------------------------------------------------------- sweep figures (1-ds)


def fig_alpha_sweep_curve(sweep: dict, out_dir: Path) -> None:
    if not sweep:
        return
    alphas = sorted(sweep["alphas"].keys())
    delta_all = np.array(
        [sweep["alphas"][a]["delta"]["all"]["mean_delta_ndcg10"] for a in alphas]
    )
    ci_lo_all = np.array(
        [sweep["alphas"][a]["delta"]["all"]["ci_lo"] for a in alphas]
    )
    ci_hi_all = np.array(
        [sweep["alphas"][a]["delta"]["all"]["ci_hi"] for a in alphas]
    )
    delta_conf = np.array(
        [sweep["alphas"][a]["delta"]["confused"]["mean_delta_ndcg10"] for a in alphas]
    )
    ci_lo_conf = np.array(
        [sweep["alphas"][a]["delta"]["confused"]["ci_lo"] for a in alphas]
    )
    ci_hi_conf = np.array(
        [sweep["alphas"][a]["delta"]["confused"]["ci_hi"] for a in alphas]
    )
    x = np.array(alphas)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0, color="#888", linestyle="--", linewidth=1)
    ax.fill_between(x, ci_lo_all, ci_hi_all, color="#3b6e8f", alpha=0.15)
    ax.plot(x, delta_all, "-o", color="#3b6e8f", label="all queries")
    ax.fill_between(x, ci_lo_conf, ci_hi_conf, color="#d04545", alpha=0.15)
    ax.plot(x, delta_conf, "-s", color="#d04545", label="confused queries")
    ax.set_xlabel("scale α  (applied: $h - α \\cdot v / \\|v\\|$)")
    ax.set_ylabel("Δ NDCG@10  (mean_diff − baseline)")
    ax.set_title(
        f"Magnitude sweep on {DATASET_DISPLAY[SWEEP_DATASET]}: "
        f"95 % paired bootstrap CI vs α"
    )
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in x])
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "alpha_sweep_curve")


def fig_alpha_sweep_forest(sweep: dict, out_dir: Path) -> None:
    if not sweep:
        return
    alphas = sorted(sweep["alphas"].keys())
    rows = []
    for a in alphas:
        for slice_name in ("all", "confused"):
            d = sweep["alphas"][a]["delta"].get(slice_name, {})
            if "mean_delta_ndcg10" not in d:
                continue
            rows.append(
                (a, slice_name, d["mean_delta_ndcg10"], d["ci_lo"], d["ci_hi"], d["n"])
            )
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ds_color = {"all": "#3b6e8f", "confused": "#d04545"}
    y = np.arange(len(rows))
    for i, (a, sn, m, lo, hi, n) in enumerate(rows):
        c = ds_color[sn]
        marker = "o" if sn == "all" else "s"
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt=marker, color=c, ecolor=c,
                    capsize=4, markersize=6)
        if lo > 0:
            ax.text(hi + 0.003, i, "[+]", va="center", fontsize=10,
                    color="#2a8a3e", fontweight="bold")
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    labels = [f"α={a:g}\n{sn}" for a, sn, *_ in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ NDCG@10  (mean_diff − baseline)")
    ax.set_title(
        f"Magnitude sweep on {DATASET_DISPLAY[SWEEP_DATASET]}: forest CIs (95 %, paired bootstrap 10K)"
    )
    ax.invert_yaxis()
    handles = [
        plt.Line2D([0], [0], color=ds_color[s], marker=("o" if s == "all" else "s"),
                   linestyle="", label=s)
        for s in ("all", "confused")
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "alpha_sweep_forest")


def fig_alpha_sweep_ecdf(sweep: dict, out_dir: Path) -> None:
    if not sweep:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base = np.sort([v["ndcg_cut_10"] for v in sweep["baseline_per_q"].values()])
    n = len(base)
    ax.plot(base, np.arange(1, n + 1) / n, color="#888", label="baseline", linewidth=1.7)
    cmap = plt.get_cmap("viridis")
    alphas = sorted(sweep["alphas"].keys())
    for i, a in enumerate(alphas):
        per_q = sweep["alphas"][a]["per_q"]
        vals = np.sort([v["ndcg_cut_10"] for v in per_q.values()])
        ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals),
                color=cmap(i / max(len(alphas) - 1, 1)),
                label=f"α={a:g}", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-query NDCG@10")
    ax.set_ylabel("ECDF")
    ax.set_title(
        f"Per-query NDCG@10 ECDF: baseline vs mean_diff sweep ({DATASET_DISPLAY[SWEEP_DATASET]})"
    )
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "alpha_sweep_ecdf")


def fig_alpha_sweep_violin(sweep: dict, out_dir: Path) -> None:
    if not sweep:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    alphas = sorted(sweep["alphas"].keys())
    per_q_base = sweep["baseline_per_q"]
    data = []
    labels = []
    for a in alphas:
        per_q = sweep["alphas"][a]["per_q"]
        ours, base, _ = align_per_query(per_q, per_q_base, "ndcg_cut_10")
        deltas = ours - base
        data.append(deltas)
        labels.append(f"α={a:g}\n(n={len(deltas)})")
    parts = ax.violinplot(data, showmeans=True, showmedians=False)
    cmap = plt.get_cmap("viridis")
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(cmap(i / max(len(alphas) - 1, 1)))
        pc.set_alpha(0.55)
    ax.axhline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("per-query Δ NDCG@10")
    ax.set_title(
        f"Per-query Δ distribution by α  ({DATASET_DISPLAY[SWEEP_DATASET]}, all queries)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "alpha_sweep_violin")


# ---------------------------------------------------------- main


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "01_mean_diff"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    sweep = load_sweep()

    # Remove stale figures from earlier non-unified runs
    for old in (
        "delta_metric_ci_forest", "delta_metric_violin",
        "delta_metric_ecdf", "v_norm_per_dataset",
    ):
        for ext in ("pdf", "png"):
            p = out_dir / f"{old}.{ext}"
            if p.exists():
                p.unlink()

    fig_raw_delta_ci_forest(raw, out_dir)
    fig_v_norm(raw, out_dir)
    fig_alpha_sweep_curve(sweep, out_dir)
    fig_alpha_sweep_forest(sweep, out_dir)
    fig_alpha_sweep_ecdf(sweep, out_dir)
    fig_alpha_sweep_violin(sweep, out_dir)

    print(f"figures → {out_dir}")
    print(f"  raw datasets: {list(raw.keys())}")
    print(f"  sweep dataset: {SWEEP_DATASET}, alphas: {sorted(sweep.get('alphas', {}).keys())}")


if __name__ == "__main__":
    main()
