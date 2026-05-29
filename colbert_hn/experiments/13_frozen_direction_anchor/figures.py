"""13_frozen_direction_anchor figure 카탈로그 (3 seeds × SciFact).

생성:
  - delta_ci_forest.{pdf,png}        — Exp 13 (3 seeds + mean) vs Exp 11 (3 seeds + mean) of 95 % CI on Δ NDCG@10
  - anchor_family_scatter.{pdf,png}  — Δ confused × Δ easy 평면 상 anchor-side family (Exp 11, 13) + Phase 2b scatter
  - train_curves.{pdf,png}           — Exp 13 seed 42 의 rank/anchor loss + val NDCG@10 epoch curves
  - lora_AB_norms.{pdf,png}          — 24 adapters (12 layers × q,v) 의 ‖A‖, ‖B‖ 분포 (Exp 11 비교)
  - ndcg_slice_grid.{pdf,png}        — 3 seeds × 3 slices (all/confused/easy) NDCG@10 bar grid
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
EXP13_TAG = "qv_r8_l12_dir1"
EXP11_TAG = "qv_r8_l12_le1"
PHASE2B_TAG = "qv_r8_l12"


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
    for fn in ("train_history.json", "lora_stats.json"):
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
        out[slc] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "vals": vals,
        }
    return out


def _load_all() -> dict:
    data = {}
    for exp_dir, tag, key in [
        ("13_frozen_direction_anchor", EXP13_TAG, "exp13"),
        ("11_easy_preservation", EXP11_TAG, "exp11"),
        ("10_lora_phi", PHASE2B_TAG, "phase2b"),
    ]:
        rows = [s for seed in SEEDS if (s := _load_seed(exp_dir, tag, seed))]
        data[key] = {"seeds": rows, "agg": _aggregate(rows)}
    # baseline
    base_p = PROJECT_ROOT / "outputs" / "00_baseline" / DATASET / "seed_42" / "metrics_aggregate.json"
    data["baseline_ndcg_all"] = json.loads(base_p.read_text())["all"]["ndcg_cut_10"] if base_p.exists() else None
    return data


def fig_delta_ci_forest(d: dict, out_dir: Path) -> None:
    """3 slices (all / confused / easy) × 2 methods (Exp 11, Exp 13) × 3 seeds + mean."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    slices = ["all", "confused", "easy"]
    titles = ["Δ NDCG@10 (all)", "Δ NDCG@10 (confused)", "Δ NDCG@10 (easy)"]

    method_colors = {"exp11": "#3b6e8f", "exp13": "#d04545"}
    method_labels = {"exp11": "Exp 11 (relational)", "exp13": "Exp 13 (per-token cos)"}

    for ax, slc, title in zip(axes, slices, titles):
        rows = []  # (label, mean, lo, hi, color, is_mean)
        for method in ("exp11", "exp13"):
            for r in d[method]["seeds"]:
                if slc not in r["delta"]:
                    continue
                rd = r["delta"][slc]
                rows.append((f"{method_labels[method]} s={r['agg'].get('_meta', {}).get('seed', '?')}",
                             rd["mean_delta_ndcg10"], rd["ci_lo"], rd["ci_hi"],
                             method_colors[method], False))
            agg = d[method]["agg"].get(slc, {})
            if agg:
                rows.append((f"{method_labels[method]} mean ± std",
                             agg["mean"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                             method_colors[method], True))
        # We don't actually need seed in label; rebuild without _meta dependency
        rows = []
        y_pos = 0
        ylabels = []
        for method in ("exp11", "exp13"):
            for i, r in enumerate(d[method]["seeds"]):
                if slc not in r["delta"]:
                    continue
                rd = r["delta"][slc]
                ax.errorbar(rd["mean_delta_ndcg10"], y_pos,
                            xerr=[[rd["mean_delta_ndcg10"] - rd["ci_lo"]],
                                  [rd["ci_hi"] - rd["mean_delta_ndcg10"]]],
                            fmt="o", color=method_colors[method],
                            ecolor=method_colors[method], alpha=0.5, capsize=3)
                ylabels.append(f"{method_labels[method]} s={SEEDS[i]}")
                y_pos += 1
            agg = d[method]["agg"].get(slc, {})
            if agg:
                ax.errorbar(agg["mean"], y_pos,
                            xerr=[[agg["std"]], [agg["std"]]],
                            fmt="s", color=method_colors[method], markersize=10,
                            ecolor=method_colors[method], capsize=5, linewidth=2)
                ylabels.append(f"{method_labels[method]} mean ± std")
                y_pos += 1
            y_pos += 0.4  # gap between methods
        ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Δ NDCG@10")
        ax.grid(axis="x", alpha=0.3)
        # mark pre-commit thresholds for branch (a)
        if slc == "all":
            ax.axvline(0.025, color="green", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.text(0.025, -0.5, "branch (a)\n>+0.025", color="green", fontsize=7, ha="center")
        elif slc == "confused":
            ax.axvline(0.08, color="green", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.text(0.08, -0.5, "branch (a)\n>+0.08", color="green", fontsize=7, ha="center")
        elif slc == "easy":
            ax.axvline(-0.020, color="green", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.text(-0.020, -0.5, "branch (a)\n>-0.020", color="green", fontsize=7, ha="center")

    fig.suptitle("Exp 13 (per-token cosine) vs Exp 11 (Sim Frobenius²) — anchor-side family on SciFact (3 seeds)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest")


def fig_anchor_family_scatter(d: dict, out_dir: Path) -> None:
    """Δ confused × Δ easy plane, with three methods 의 seed-level points."""
    fig, ax = plt.subplots(figsize=(7, 6))
    method_styles = {
        "phase2b": {"color": "#444", "marker": "x", "label": "Phase 2b (baseline)"},
        "exp11":   {"color": "#3b6e8f", "marker": "o", "label": "Exp 11 (relational)"},
        "exp13":   {"color": "#d04545", "marker": "s", "label": "Exp 13 (per-token cos)"},
    }
    for method, style in method_styles.items():
        seed_xs, seed_ys = [], []
        for r in d[method]["seeds"]:
            if "confused" in r["delta"] and "easy" in r["delta"]:
                seed_xs.append(r["delta"]["confused"]["mean_delta_ndcg10"])
                seed_ys.append(r["delta"]["easy"]["mean_delta_ndcg10"])
        ax.scatter(seed_xs, seed_ys, color=style["color"], marker=style["marker"],
                   s=70, alpha=0.5, label=f"{style['label']} (seeds)")
        agg = d[method]["agg"]
        if agg.get("confused") and agg.get("easy"):
            ax.errorbar(agg["confused"]["mean"], agg["easy"]["mean"],
                        xerr=agg["confused"]["std"], yerr=agg["easy"]["std"],
                        color=style["color"], marker=style["marker"],
                        markersize=14, capsize=5, linewidth=2,
                        markerfacecolor="white", markeredgewidth=2,
                        label=f"{style['label']} mean ± std")
    # 1:1 trade-off line
    xs = np.linspace(0, 0.13, 50)
    ax.plot(xs, -xs, "--", color="grey", alpha=0.4, label="1:1 trade-off (Δ confused = −Δ easy)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ NDCG@10 (confused)")
    ax.set_ylabel("Δ NDCG@10 (easy)")
    ax.set_title("Anchor-side family — Δ confused × Δ easy trade-off (SciFact, 3 seeds)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "anchor_family_scatter")


def fig_train_curves(d: dict, out_dir: Path) -> None:
    """seed 42 의 step × (rank_loss, anchor_loss) + epoch × val NDCG."""
    h = d["exp13"]["seeds"][0].get("train_history") if d["exp13"]["seeds"] else None
    if not h:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_l, ax_v = axes

    steps = h["steps"]
    epoch = h["epoch"]
    rank_losses = h["rank_losses"]
    anchor_losses = h["anchor_losses"]

    ax_l.plot(steps, rank_losses, color="#3b6e8f", alpha=0.5, label="rank_loss (per step)")
    ax_l.plot(steps, anchor_losses, color="#d04545", alpha=0.5, label="anchor_loss (per step)")
    # epoch boundary
    for ep_end in [steps[i] for i in range(len(steps) - 1) if epoch[i] != epoch[i + 1]]:
        ax_l.axvline(ep_end, color="grey", linestyle=":", linewidth=0.6)
    ax_l.set_xlabel("Step")
    ax_l.set_ylabel("Loss")
    ax_l.set_title("Exp 13 seed 42 — rank + anchor loss (per step)")
    ax_l.legend(loc="upper right", fontsize=8, frameon=False)
    ax_l.grid(alpha=0.3)

    val_eps = h["val_epochs"]
    val_all = h["val_ndcg_all"]
    val_conf = h["val_ndcg_confused"]
    ax_v.plot(val_eps, val_all, "o-", color="#3b6e8f", label="val NDCG@10 (all)")
    ax_v.plot(val_eps, val_conf, "o-", color="#d04545", label="val NDCG@10 (confused)")
    if d.get("baseline_ndcg_all"):
        ax_v.axhline(d["baseline_ndcg_all"], color="grey", linestyle="--", linewidth=0.8,
                     label=f"baseline all={d['baseline_ndcg_all']:.4f}")
    ax_v.set_xlabel("Epoch")
    ax_v.set_ylabel("Val NDCG@10")
    ax_v.set_title("Exp 13 seed 42 — validation NDCG@10")
    ax_v.set_xticks(val_eps)
    ax_v.legend(loc="lower left", fontsize=8, frameon=False)
    ax_v.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "train_curves")


def fig_lora_AB_norms(d: dict, out_dir: Path) -> None:
    """24 adapters 의 ‖A‖, ‖B‖ 분포 — Exp 11 vs Exp 13 비교 (seed 42)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_a, ax_b = axes
    width = 0.4
    x = np.arange(24)
    for method, color, offset in [("exp11", "#3b6e8f", -width/2),
                                  ("exp13", "#d04545", +width/2)]:
        if not d[method]["seeds"]:
            continue
        ls = d[method]["seeds"][0].get("lora_stats")
        if not ls:
            continue
        A_norms = ls.get("A_norms_per_adapter", [])
        B_norms = ls.get("B_norms_per_adapter", [])
        if A_norms:
            ax_a.bar(x + offset, A_norms, width, color=color, alpha=0.7,
                     label=f"{method} (‖A‖_total={ls.get('A_norm_total', 0):.2f})")
        if B_norms:
            ax_b.bar(x + offset, B_norms, width, color=color, alpha=0.7,
                     label=f"{method} (‖B‖_total={ls.get('B_norm_total', 0):.2f})")
    ax_a.set_xlabel("Adapter index (12 layers × q,v)")
    ax_a.set_ylabel("‖A‖")
    ax_a.set_title("LoRA A norms per adapter (seed 42)")
    ax_a.legend(fontsize=8, frameon=False)
    ax_a.grid(axis="y", alpha=0.3)

    ax_b.set_xlabel("Adapter index (12 layers × q,v)")
    ax_b.set_ylabel("‖B‖")
    ax_b.set_title("LoRA B norms per adapter (seed 42) — direct measure of anchor preservation")
    ax_b.legend(fontsize=8, frameon=False)
    ax_b.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "lora_AB_norms")


def fig_ndcg_slice_grid(d: dict, out_dir: Path) -> None:
    """3 seeds × 3 slices NDCG@10 bar grid."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    slices = ["all", "confused", "easy"]
    seed_labels = [str(s) for s in SEEDS]
    baseline_per_seed = {}
    for s in SEEDS:
        bp = PROJECT_ROOT / "outputs" / "00_baseline" / DATASET / f"seed_{s}" / "metrics_aggregate.json"
        if bp.exists():
            baseline_per_seed[s] = json.loads(bp.read_text())
    for ax, slc in zip(axes, slices):
        x = np.arange(len(SEEDS))
        width = 0.35
        if slc != "easy":
            base_vals = [baseline_per_seed.get(s, {}).get(slc, {}).get("ndcg_cut_10", 0) for s in SEEDS]
        else:
            base_vals = [None] * len(SEEDS)  # easy NDCG 는 metrics_aggregate 에 없을 수 있음
        exp13_vals = []
        for r in d["exp13"]["seeds"]:
            if slc in r["agg"]:
                exp13_vals.append(r["agg"][slc].get("ndcg_cut_10", 0))
            else:
                # easy slice 는 metrics_aggregate 에 없을 수 있음 — delta 로 추정
                exp13_vals.append(None)
        if all(v is not None for v in base_vals + exp13_vals):
            ax.bar(x - width/2, base_vals, width, color="#888", label="baseline (frozen)")
            ax.bar(x + width/2, exp13_vals, width, color="#d04545", label="Exp 13")
            for i, (b, e) in enumerate(zip(base_vals, exp13_vals)):
                if b is not None:
                    ax.text(i - width/2, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)
                if e is not None:
                    ax.text(i + width/2, e + 0.005, f"{e:.3f}", ha="center", fontsize=7)
        else:
            # show Δ instead
            deltas = [r["delta"][slc]["mean_delta_ndcg10"] for r in d["exp13"]["seeds"]]
            ax.bar(x, deltas, width*2, color="#d04545")
            for i, dv in enumerate(deltas):
                ax.text(i, dv + (0.002 if dv >= 0 else -0.005), f"{dv:+.3f}",
                        ha="center", fontsize=8, color="black" if abs(dv) > 0.005 else "grey")
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_ylabel(f"Δ NDCG@10 ({slc})")
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels)
        ax.set_xlabel("Seed")
        ax.set_title(f"slice = {slc}")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Exp 13 — NDCG@10 per slice (3 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "ndcg_slice_grid")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "13_frozen_direction_anchor"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load_all()
    fig_delta_ci_forest(d, out_dir)
    fig_anchor_family_scatter(d, out_dir)
    fig_train_curves(d, out_dir)
    fig_lora_AB_norms(d, out_dir)
    fig_ndcg_slice_grid(d, out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
