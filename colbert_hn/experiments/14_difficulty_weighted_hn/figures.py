"""14_difficulty_weighted_hn figure 카탈로그 (3 seeds × SciFact).

생성:
  - delta_ci_forest.{pdf,png}          — Exp 14 (3 seeds + mean) vs Exp 12 (binary FN cut) vs Phase 2b paired bootstrap 95 % CI
  - six_lever_scatter.{pdf,png}        — Δ confused × Δ easy plane 의 6 lever (Phase 2b, Exp 11, 12, 13, 14, M1b) seed-level + mean
  - weight_distribution.{pdf,png}      — triplet weight histogram + e5 margin × weight scatter (seed 42)
  - train_curves.{pdf,png}             — Exp 14 seed 2024 의 weighted_loss + val NDCG@10 epoch curves
  - ndcg_slice_grid.{pdf,png}          — 3 seeds × 3 slices NDCG@10 bar grid
  - family_frontier_overview.{pdf,png} — 6 lever 의 3-seed mean Δ all/conf/easy grouped bars + family annotation
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
SEEDS = [42, 1337, 2024]

LEVERS = {
    "phase2b":   {"dir": "10_lora_phi",                 "tag": "qv_r8_l12",          "label": "Phase 2b",        "family": "baseline", "color": "#444"},
    "exp11":     {"dir": "11_easy_preservation",        "tag": "qv_r8_l12_le1",      "label": "Exp 11 (rel.)",   "family": "anchor",   "color": "#1f77b4"},
    "exp12":     {"dir": "12_fn_denoised_hn",           "tag": "qv_r8_l12_thresh0",  "label": "Exp 12 (binary)", "family": "data-w",   "color": "#ff7f0e"},
    "exp13":     {"dir": "13_frozen_direction_anchor",  "tag": "qv_r8_l12_dir1",     "label": "Exp 13 (abs.)",   "family": "anchor",   "color": "#2ca02c"},
    "exp14":     {"dir": "14_difficulty_weighted_hn",   "tag": "qv_r8_l12_diffw10",  "label": "Exp 14 (cont.)",  "family": "data-w",   "color": "#d62728"},
}


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


def _load_seed(exp_dir: str, tag: str, seed: int) -> dict | None:
    d = PROJECT_ROOT / "outputs" / exp_dir / DATASET / f"seed_{seed}" / tag
    if not (d / "delta_vs_baseline.json").exists():
        return None
    out = {
        "delta": json.loads((d / "delta_vs_baseline.json").read_text()),
        "agg": json.loads((d / "metrics_aggregate.json").read_text()),
    }
    for fn in ("train_history.json", "lora_stats.json", "weight_stats.json"):
        p = d / fn
        if p.exists():
            out[fn.replace(".json", "")] = json.loads(p.read_text())
    return out


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out = {}
    for slc in ("all", "confused", "easy"):
        vals = [r["delta"][slc]["mean_delta_ndcg10"] for r in rows if slc in r["delta"]]
        if vals:
            out[slc] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "vals": vals,
            }
    return out


def _load_all() -> dict:
    data = {}
    for key, info in LEVERS.items():
        rows = [r for seed in SEEDS if (r := _load_seed(info["dir"], info["tag"], seed))]
        data[key] = {"seeds": rows, "agg": _aggregate(rows), "info": info}
    # M1b — separate path
    m1b_dir = PROJECT_ROOT / "outputs" / "10_lora_phi" / DATASET
    m1b_rows = []
    for seed in SEEDS:
        # M1b artifact tag varies — check both possible locations
        for tag in ("qv_r8_l12_inbatch", "qv_r8_l12_m1b"):
            p = m1b_dir / f"seed_{seed}" / tag / "delta_vs_baseline.json"
            if p.exists():
                m1b_rows.append({"delta": json.loads(p.read_text())})
                break
    data["m1b"] = {"seeds": m1b_rows, "agg": _aggregate(m1b_rows),
                   "info": {"label": "M1b (in-batch)", "family": "data-sub", "color": "#9467bd"}}
    return data


def fig_delta_ci_forest(d: dict, out_dir: Path) -> None:
    """3 slices × 3 levers (Exp 14, Exp 12, Phase 2b)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    slices = ["all", "confused", "easy"]
    targets = ["exp14", "exp12", "phase2b"]
    for ax, slc in zip(axes, slices):
        y_pos = 0
        ylabels = []
        for key in targets:
            color = LEVERS[key]["color"]
            label = LEVERS[key]["label"]
            for i, r in enumerate(d[key]["seeds"]):
                if slc not in r["delta"]:
                    continue
                rd = r["delta"][slc]
                ax.errorbar(rd["mean_delta_ndcg10"], y_pos,
                            xerr=[[rd["mean_delta_ndcg10"] - rd["ci_lo"]],
                                  [rd["ci_hi"] - rd["mean_delta_ndcg10"]]],
                            fmt="o", color=color, ecolor=color, alpha=0.5, capsize=3)
                ylabels.append(f"{label} s={SEEDS[i]}")
                y_pos += 1
            agg = d[key]["agg"].get(slc, {})
            if agg:
                ax.errorbar(agg["mean"], y_pos,
                            xerr=[[agg["std"]], [agg["std"]]],
                            fmt="s", color=color, markersize=11,
                            ecolor=color, capsize=5, linewidth=2,
                            markerfacecolor="white", markeredgewidth=2)
                ylabels.append(f"{label} mean ± std")
                y_pos += 1
            y_pos += 0.4
        ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"Δ NDCG@10 ({slc})")
        ax.set_xlabel("Δ NDCG@10")
        ax.grid(axis="x", alpha=0.3)
        # branch (a) threshold
        if slc == "all":
            ax.axvline(0.025, color="green", linestyle=":", linewidth=0.8, alpha=0.6)
        elif slc == "easy":
            ax.axvline(-0.040, color="green", linestyle=":", linewidth=0.8, alpha=0.6)

    fig.suptitle("Exp 14 (continuous sigmoid) vs Exp 12 (binary FN cut) vs Phase 2b — SciFact (3 seeds)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest")


def fig_six_lever_scatter(d: dict, out_dir: Path) -> None:
    """Δ confused × Δ easy plane scatter for 6 levers."""
    fig, ax = plt.subplots(figsize=(8, 7))
    families_seen = set()
    for key in ["phase2b", "exp11", "exp12", "exp13", "exp14", "m1b"]:
        if key not in d or not d[key]["seeds"]:
            continue
        info = d[key]["info"]
        marker_by_family = {"baseline": "x", "anchor": "o", "data-w": "s", "data-sub": "^"}
        marker = marker_by_family.get(info["family"], "o")
        xs, ys = [], []
        for r in d[key]["seeds"]:
            if "confused" in r["delta"] and "easy" in r["delta"]:
                xs.append(r["delta"]["confused"]["mean_delta_ndcg10"])
                ys.append(r["delta"]["easy"]["mean_delta_ndcg10"])
        ax.scatter(xs, ys, color=info["color"], marker=marker, s=70, alpha=0.5)
        agg = d[key]["agg"]
        if agg.get("confused") and agg.get("easy"):
            ax.errorbar(agg["confused"]["mean"], agg["easy"]["mean"],
                        xerr=agg["confused"]["std"], yerr=agg["easy"]["std"],
                        color=info["color"], marker=marker,
                        markersize=14, capsize=5, linewidth=2,
                        markerfacecolor="white", markeredgewidth=2,
                        label=info["label"])
    # 1:1 line
    xs = np.linspace(0, 0.13, 50)
    ax.plot(xs, -xs, "--", color="grey", alpha=0.4, label="1:1 trade-off")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    # family annotations
    ax.annotate("anchor-side (upper)\nExp 11, Exp 13",
                xy=(0.097, -0.026), xytext=(0.110, -0.005),
                fontsize=9, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", alpha=0.5))
    ax.annotate("data-side weighting\n(lower)\nExp 12, Exp 14",
                xy=(0.082, -0.067), xytext=(0.020, -0.090),
                fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.5))
    ax.set_xlabel("Δ NDCG@10 (confused)")
    ax.set_ylabel("Δ NDCG@10 (easy)")
    ax.set_title("Six-lever trade-off frontier — family-separated structure (SciFact, 3 seeds)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "six_lever_scatter")


def fig_weight_distribution(d: dict, out_dir: Path) -> None:
    """Triplet weight histogram + e5_margin × weight scatter from seed 42 weight_stats."""
    if not d["exp14"]["seeds"]:
        return
    ws = d["exp14"]["seeds"][0].get("weight_stats")
    if not ws:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_h, ax_s = axes

    # Histogram of weights
    w_stats = ws["weights"]
    # Reconstruct an approximate distribution from stats — since raw weights are not saved,
    # we visualize the sigmoid curve over the e5_margin range
    margins = np.linspace(ws["margins"]["min"], ws["margins"]["max"], 1000)
    weights_curve = 1.0 / (1.0 + np.exp(-10.0 * margins))
    ax_s.plot(margins, weights_curve, color="#d62728", linewidth=2, label="σ(α_w · margin), α_w=10")
    ax_s.axhline(0.5, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_s.axvline(0, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_s.axhline(w_stats["mean"], color="#1f77b4", linestyle=":", linewidth=1.2,
                 label=f"mean={w_stats['mean']:.3f}")
    ax_s.axhline(w_stats["median"], color="#2ca02c", linestyle=":", linewidth=1.2,
                 label=f"median={w_stats['median']:.3f}")
    ax_s.set_xlabel("e5_margin = cos(eq, epos) − cos(eq, ehn)")
    ax_s.set_ylabel("Triplet weight = σ(α_w · margin)")
    ax_s.set_title("Sigmoid weight vs e5_margin (α_w=10)")
    ax_s.legend(fontsize=8, frameon=False)
    ax_s.grid(alpha=0.3)

    # Bar chart of weight statistics
    stats_names = ["min", "mean", "median", "max"]
    w_vals = [w_stats[k] for k in stats_names]
    bars = ax_h.bar(stats_names, w_vals, color="#d62728", alpha=0.7)
    for b, v in zip(bars, w_vals):
        ax_h.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                  ha="center", fontsize=9)
    ax_h.set_ylabel("Triplet weight")
    ax_h.set_ylim(0, 1.1)
    ax_h.set_title(f"Triplet weight statistics (9190 triplets, std={w_stats['std']:.3f})")
    ax_h.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "weight_distribution")


def fig_train_curves(d: dict, out_dir: Path) -> None:
    """seed 2024 의 weighted_loss + val NDCG@10 epoch curves."""
    if not d["exp14"]["seeds"] or len(d["exp14"]["seeds"]) < 3:
        return
    h = d["exp14"]["seeds"][2].get("train_history")  # seed 2024 = index 2
    if not h:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_l, ax_v = axes

    if "steps" in h and "losses" in h:
        ax_l.plot(h["steps"], h["losses"], color="#d62728", alpha=0.5, label="weighted_loss (per step)")
        # epoch boundaries
        if "epoch" in h:
            for i in range(len(h["steps"]) - 1):
                if h["epoch"][i] != h["epoch"][i + 1]:
                    ax_l.axvline(h["steps"][i], color="grey", linestyle=":", linewidth=0.6)
    ax_l.set_xlabel("Step")
    ax_l.set_ylabel("Loss")
    ax_l.set_title("Exp 14 seed 2024 — weighted margin loss")
    ax_l.legend(loc="upper right", fontsize=8, frameon=False)
    ax_l.grid(alpha=0.3)

    val_eps = h["val_epochs"]
    val_all = h["val_ndcg_all"]
    val_conf = h["val_ndcg_confused"]
    ax_v.plot(val_eps, val_all, "o-", color="#1f77b4", label="val NDCG@10 (all)")
    ax_v.plot(val_eps, val_conf, "o-", color="#d62728", label="val NDCG@10 (confused)")
    base_p = PROJECT_ROOT / "outputs" / "00_baseline" / DATASET / "seed_42" / "metrics_aggregate.json"
    if base_p.exists():
        b = json.loads(base_p.read_text())["all"]["ndcg_cut_10"]
        ax_v.axhline(b, color="grey", linestyle="--", linewidth=0.8,
                     label=f"baseline all={b:.4f}")
    ax_v.set_xlabel("Epoch")
    ax_v.set_ylabel("Val NDCG@10")
    ax_v.set_title("Exp 14 seed 2024 — val NDCG@10 (monotone increase, late best)")
    ax_v.set_xticks(val_eps)
    ax_v.legend(loc="lower right", fontsize=8, frameon=False)
    ax_v.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "train_curves")


def fig_ndcg_slice_grid(d: dict, out_dir: Path) -> None:
    """3 seeds × 3 slices Δ NDCG bar grid."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    slices = ["all", "confused", "easy"]
    seed_labels = [str(s) for s in SEEDS]
    for ax, slc in zip(axes, slices):
        x = np.arange(len(SEEDS))
        deltas = []
        ci_los = []
        ci_his = []
        for r in d["exp14"]["seeds"]:
            if slc in r["delta"]:
                deltas.append(r["delta"][slc]["mean_delta_ndcg10"])
                ci_los.append(r["delta"][slc]["ci_lo"])
                ci_his.append(r["delta"][slc]["ci_hi"])
            else:
                deltas.append(0)
                ci_los.append(0)
                ci_his.append(0)
        yerr_lo = [d - lo for d, lo in zip(deltas, ci_los)]
        yerr_hi = [hi - d for d, hi in zip(deltas, ci_his)]
        bars = ax.bar(x, deltas, 0.6, color="#d62728", alpha=0.7,
                      yerr=[yerr_lo, yerr_hi], capsize=5, ecolor="black")
        for i, dv in enumerate(deltas):
            ax.text(i, dv + (0.005 if dv >= 0 else -0.008), f"{dv:+.3f}",
                    ha="center", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels)
        ax.set_xlabel("Seed")
        ax.set_ylabel(f"Δ NDCG@10 ({slc})")
        ax.set_title(f"slice = {slc}")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Exp 14 — Δ NDCG@10 per slice (3 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "ndcg_slice_grid")


def fig_family_frontier_overview(d: dict, out_dir: Path) -> None:
    """6 lever 의 3-seed mean Δ all / Δ confused / Δ easy grouped bars."""
    fig, ax = plt.subplots(figsize=(13, 6))
    levers_order = ["phase2b", "exp12", "exp14", "m1b", "exp11", "exp13"]
    keep = [k for k in levers_order if k in d and d[k]["agg"].get("all")
            and d[k]["agg"].get("confused") and d[k]["agg"].get("easy")]
    labels = [d[k]["info"]["label"] for k in keep]
    delta_all = [d[k]["agg"]["all"]["mean"] for k in keep]
    delta_all_std = [d[k]["agg"]["all"]["std"] for k in keep]
    delta_conf = [d[k]["agg"]["confused"]["mean"] for k in keep]
    delta_conf_std = [d[k]["agg"]["confused"]["std"] for k in keep]
    delta_easy = [d[k]["agg"]["easy"]["mean"] for k in keep]
    delta_easy_std = [d[k]["agg"]["easy"]["std"] for k in keep]

    n = len(labels)
    x = np.arange(n)
    width = 0.27
    ax.bar(x - width, delta_all, width, yerr=delta_all_std, capsize=3,
           color="#444", alpha=0.8, label="Δ all")
    ax.bar(x, delta_conf, width, yerr=delta_conf_std, capsize=3,
           color="#1f77b4", alpha=0.8, label="Δ confused")
    ax.bar(x + width, delta_easy, width, yerr=delta_easy_std, capsize=3,
           color="#d62728", alpha=0.8, label="Δ easy")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Δ NDCG@10 (3-seed mean ± std)")
    ax.set_title("6-lever framework — 3-seed mean Δ NDCG@10 (SciFact)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # family bands — index by lever family in the actual `keep` order
    family_indices = {}
    for i, k in enumerate(keep):
        fam = d[k]["info"]["family"]
        family_indices.setdefault(fam, []).append(i)
    family_labels_above = {
        "data-w": "data-side weighting",
        "data-sub": "data-side substitution",
        "anchor": "anchor-side (upper frontier)",
    }
    family_color = {"data-w": "#d62728", "data-sub": "#9467bd", "anchor": "#2ca02c"}
    for fam, idxs in family_indices.items():
        if fam == "baseline":
            continue
        x_lo = min(idxs) - 0.4
        x_hi = max(idxs) + 0.4
        ax.axvspan(x_lo, x_hi, ymin=0.95, ymax=1.0, alpha=0.15, color=family_color.get(fam, "#888"))
        ax.text((x_lo + x_hi) / 2, max(delta_conf) + 0.015, family_labels_above.get(fam, fam),
                ha="center", fontsize=8, color=family_color.get(fam, "#888"))
    fig.tight_layout()
    _save(fig, out_dir, "family_frontier_overview")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "14_difficulty_weighted_hn"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load_all()
    fig_delta_ci_forest(d, out_dir)
    fig_six_lever_scatter(d, out_dir)
    fig_weight_distribution(d, out_dir)
    fig_train_curves(d, out_dir)
    fig_ndcg_slice_grid(d, out_dir)
    fig_family_frontier_overview(d, out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
