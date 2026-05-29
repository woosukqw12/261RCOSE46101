"""16_multilayer_anchor figure 카탈로그 (3 seeds × SciFact).

생성:
  - delta_ci_forest.{pdf,png}        — Exp 16 (3 seeds + mean) vs Exp 13 vs Phase 2b paired bootstrap CI
  - layer_count_comparison.{pdf,png} — Exp 13 (single layer) vs Exp 16 (5-layer): Δ all/conf/easy + 'anchor scope' visual
  - train_curves.{pdf,png}           — Exp 16 seed 42 의 rank + multi-layer dir loss + val NDCG
  - lora_AB_norms.{pdf,png}          — 24 adapters 의 ‖A‖, ‖B‖ vs Exp 13 비교
  - ndcg_slice_grid.{pdf,png}        — 3 seeds × 3 slices NDCG@10
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
EXP16_TAG = "qv_r8_l12_dir1_multilayer"
EXP13_TAG = "qv_r8_l12_dir1"
PHASE2B_TAG = "qv_r8_l12"


def set_rc():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def _load_seed(exp_dir, tag, seed):
    d = PROJECT_ROOT / "outputs" / exp_dir / DATASET / f"seed_{seed}" / tag
    if not (d / "delta_vs_baseline.json").exists():
        return None
    out = {"delta": json.loads((d / "delta_vs_baseline.json").read_text()),
           "agg": json.loads((d / "metrics_aggregate.json").read_text())}
    for fn in ("train_history.json", "lora_stats.json"):
        p = d / fn
        if p.exists():
            out[fn.replace(".json", "")] = json.loads(p.read_text())
    return out


def _aggregate(rows):
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


def _load_all():
    d = {}
    for key, exp_dir, tag in [
        ("exp16", "16_multilayer_anchor", EXP16_TAG),
        ("exp13", "13_frozen_direction_anchor", EXP13_TAG),
        ("phase2b", "10_lora_phi", PHASE2B_TAG),
    ]:
        rows = [r for seed in SEEDS if (r := _load_seed(exp_dir, tag, seed))]
        d[key] = {"seeds": rows, "agg": _aggregate(rows)}
    base_p = PROJECT_ROOT / "outputs" / "00_baseline" / DATASET / "seed_42" / "metrics_aggregate.json"
    d["baseline_ndcg_all"] = json.loads(base_p.read_text())["all"]["ndcg_cut_10"] if base_p.exists() else None
    return d


def fig_delta_ci_forest(d, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    slices = ["all", "confused", "easy"]
    method_colors = {"exp16": "#2ca02c", "exp13": "#1f77b4", "phase2b": "#cc4444"}
    method_labels = {"exp16": "Exp 16 (5-layer)", "exp13": "Exp 13 (final-only)", "phase2b": "Phase 2b"}

    for ax, slc in zip(axes, slices):
        y_pos = 0
        ylabels = []
        for key in ("phase2b", "exp13", "exp16"):
            color = method_colors[key]
            label = method_labels[key]
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
                            fmt="s", color=color, markersize=10,
                            ecolor=color, capsize=5, linewidth=2,
                            markerfacecolor="white", markeredgewidth=2)
                ylabels.append(f"{label} mean ± std")
                y_pos += 1
            y_pos += 0.4
        ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
        # branch (a) threshold
        if slc == "all":
            ax.axvline(0.040, color="green", linestyle=":", linewidth=1, alpha=0.6)
            ax.text(0.040, -0.5, "branch (a)\n>+0.040", color="green", fontsize=7, ha="center")
        elif slc == "easy":
            ax.axvline(-0.015, color="green", linestyle=":", linewidth=1, alpha=0.6)
            ax.text(-0.015, -0.5, "branch (a)\n>-0.015", color="green", fontsize=7, ha="center")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"Δ NDCG@10 ({slc})")
        ax.set_xlabel("Δ NDCG@10")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Exp 16 (multi-layer) vs Exp 13 (final-only) vs Phase 2b — SciFact (3 seeds)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest")


def fig_layer_count_comparison(d, out_dir):
    """3-seed mean Δ comparison: Exp 13 (1 layer) vs Exp 16 (5 layers) — anchor scope ablation."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    slices = ["all", "confused", "easy"]
    x = np.arange(len(slices))
    width = 0.35
    exp13_means = [d["exp13"]["agg"][s]["mean"] for s in slices]
    exp13_stds = [d["exp13"]["agg"][s]["std"] for s in slices]
    exp16_means = [d["exp16"]["agg"][s]["mean"] for s in slices]
    exp16_stds = [d["exp16"]["agg"][s]["std"] for s in slices]
    b13 = ax.bar(x - width/2, exp13_means, width, yerr=exp13_stds, capsize=4,
                 color="#1f77b4", alpha=0.85, label="Exp 13 (1 layer: final 128-dim)")
    b16 = ax.bar(x + width/2, exp16_means, width, yerr=exp16_stds, capsize=4,
                 color="#2ca02c", alpha=0.85, label="Exp 16 (5 layers: {0,3,6,9,12} 768-dim)")
    for b, v in zip(b13, exp13_means):
        ax.text(b.get_x() + b.get_width()/2, v + (0.005 if v >= 0 else -0.012),
                f"{v:+.3f}", ha="center", fontsize=8)
    for b, v in zip(b16, exp16_means):
        ax.text(b.get_x() + b.get_width()/2, v + (0.005 if v >= 0 else -0.012),
                f"{v:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(slices)
    ax.set_ylabel("Δ NDCG@10 vs frozen baseline (3-seed mean ± std)")
    ax.set_title("Anchor scope ablation — single-layer (Exp 13) vs multi-layer (Exp 16)\n"
                 "CLAUDE.md §1.3 prior diagnostic finding direct test")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "layer_count_comparison")


def fig_train_curves(d, out_dir):
    h = d["exp16"]["seeds"][0].get("train_history") if d["exp16"]["seeds"] else None
    if not h:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_l, ax_v = axes
    steps = h["steps"]
    epoch = h["epoch"]
    ax_l.plot(steps, h["rank_losses"], color="#1f77b4", alpha=0.5, label="rank_loss")
    ax_l.plot(steps, h["anchor_losses"], color="#d62728", alpha=0.5, label="multi-layer dir_loss")
    for i in range(len(steps) - 1):
        if epoch[i] != epoch[i + 1]:
            ax_l.axvline(steps[i], color="grey", linestyle=":", linewidth=0.6)
    ax_l.set_xlabel("Step")
    ax_l.set_ylabel("Loss")
    ax_l.set_title("Exp 16 seed 42 — rank + multi-layer dir loss")
    ax_l.legend(loc="upper right", fontsize=8, frameon=False)
    ax_l.grid(alpha=0.3)

    val_eps = h["val_epochs"]
    ax_v.plot(val_eps, h["val_ndcg_all"], "o-", color="#1f77b4", label="val NDCG@10 (all)")
    ax_v.plot(val_eps, h["val_ndcg_confused"], "o-", color="#d62728", label="val NDCG@10 (confused)")
    if d.get("baseline_ndcg_all"):
        ax_v.axhline(d["baseline_ndcg_all"], color="grey", linestyle="--", linewidth=0.8,
                     label=f"baseline all={d['baseline_ndcg_all']:.4f}")
    ax_v.set_xlabel("Epoch")
    ax_v.set_ylabel("Val NDCG@10")
    ax_v.set_title("Exp 16 seed 42 — val NDCG@10")
    ax_v.set_xticks(val_eps)
    ax_v.legend(loc="lower left", fontsize=8, frameon=False)
    ax_v.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "train_curves")


def fig_lora_AB_norms(d, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_a, ax_b = axes
    width = 0.4
    x = np.arange(24)
    for method, color, offset, label in [("exp13", "#1f77b4", -width/2, "Exp 13"),
                                          ("exp16", "#2ca02c", +width/2, "Exp 16")]:
        if not d[method]["seeds"]:
            continue
        ls = d[method]["seeds"][0].get("lora_stats")
        if not ls:
            continue
        A_norms = ls.get("A_norms_per_adapter", [])
        B_norms = ls.get("B_norms_per_adapter", [])
        if A_norms:
            ax_a.bar(x + offset, A_norms, width, color=color, alpha=0.75,
                     label=f"{label} (‖A‖_total={ls.get('A_norm_total', 0):.2f})")
        if B_norms:
            ax_b.bar(x + offset, B_norms, width, color=color, alpha=0.75,
                     label=f"{label} (‖B‖_total={ls.get('B_norm_total', 0):.2f})")
    ax_a.set_xlabel("Adapter index (12 layers × q,v)")
    ax_a.set_ylabel("‖A‖")
    ax_a.set_title("LoRA A norms per adapter (seed 42)")
    ax_a.legend(fontsize=8, frameon=False)
    ax_a.grid(axis="y", alpha=0.3)
    ax_b.set_xlabel("Adapter index (12 layers × q,v)")
    ax_b.set_ylabel("‖B‖")
    ax_b.set_title("LoRA B norms per adapter (seed 42)")
    ax_b.legend(fontsize=8, frameon=False)
    ax_b.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "lora_AB_norms")


def fig_ndcg_slice_grid(d, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    slices = ["all", "confused", "easy"]
    seed_labels = [str(s) for s in SEEDS]
    for ax, slc in zip(axes, slices):
        x = np.arange(len(SEEDS))
        deltas = [r["delta"][slc]["mean_delta_ndcg10"] for r in d["exp16"]["seeds"]]
        ci_los = [r["delta"][slc]["ci_lo"] for r in d["exp16"]["seeds"]]
        ci_his = [r["delta"][slc]["ci_hi"] for r in d["exp16"]["seeds"]]
        yerr_lo = [dv - lo for dv, lo in zip(deltas, ci_los)]
        yerr_hi = [hi - dv for dv, hi in zip(deltas, ci_his)]
        ax.bar(x, deltas, 0.6, color="#2ca02c", alpha=0.8,
               yerr=[yerr_lo, yerr_hi], capsize=5, ecolor="black")
        for i, dv in enumerate(deltas):
            ax.text(i, dv + (0.005 if dv >= 0 else -0.010), f"{dv:+.3f}",
                    ha="center", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels)
        ax.set_xlabel("Seed")
        ax.set_ylabel(f"Δ NDCG@10 ({slc})")
        ax.set_title(f"slice = {slc}")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Exp 16 — Δ NDCG@10 per slice (3 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "ndcg_slice_grid")


def main():
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "16_multilayer_anchor"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load_all()
    fig_delta_ci_forest(d, out_dir)
    fig_layer_count_comparison(d, out_dir)
    fig_train_curves(d, out_dir)
    fig_lora_AB_norms(d, out_dir)
    fig_ndcg_slice_grid(d, out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
