"""10_lora_phi figure 카탈로그 (Phase 1 + 2a + 2b).

생성:
  - ndcg_vs_configs.{pdf,png}        — 3 LoRA configs + 5 anchor 의 NDCG@10 bar
  - delta_ci_forest.{pdf,png}        — paired bootstrap 95 % CI on Δ NDCG@10 vs baseline / 02 frozen / 02 unfrozen
  - lora_progression.{pdf,png}       — Phase 1 → 2a → 2b 의 *anchor preservation* 개선 trajectory
  - lora_AB_norms.{pdf,png}          — 3 configs 의 LoRA A/B norm per adapter
  - train_curve_3configs.{pdf,png}   — 3 configs 의 train loss + val NDCG@10 (all + confused)
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

CONFIGS = [
    {"tag": "qv_r1_l12", "label": "Phase 1\nr=1, LR=5e-5, α=r", "params": 36_864},
    {"tag": "qv_r8_l12", "label": "Phase 2a/2b\nr=8 (last run = 2b)", "params": 294_912},
]
# Note: Phase 2a and 2b share the same artifact subdir (qv_r8_l12) since the
# (B) run overwrote. We rely on the (B) result being the "final" for r=8 row.
# Phase 2a 의 결과는 보고서 본문 표로만 인용 (artifact 없음).
PHASE2A_LITERAL = {
    "ndcg_all": 0.5879,
    "delta_all_vs_baseline": (-0.0585, -0.1014, -0.0166),  # mean, ci_lo, ci_hi
    "delta_conf_vs_baseline": (+0.0804, +0.0208, +0.1395),
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


def _load() -> dict:
    base = PROJECT_ROOT / "outputs"
    data = {"configs": {}}
    for cfg in CONFIGS:
        d = base / "10_lora_phi" / DATASET / f"seed_{SEED}" / cfg["tag"]
        if not (d / "metrics_aggregate.json").exists():
            continue
        data["configs"][cfg["tag"]] = {
            "agg": json.loads((d / "metrics_aggregate.json").read_text()),
            "history": json.loads((d / "train_history.json").read_text()),
            "lora_stats": json.loads((d / "lora_stats.json").read_text()),
            "delta_vs_baseline": json.loads((d / "delta_vs_baseline.json").read_text()),
            "delta_vs_alpha10": json.loads((d / "delta_vs_mean_diff_alpha10.json").read_text())
            if (d / "delta_vs_mean_diff_alpha10.json").exists() else {},
            "delta_vs_02": json.loads((d / "delta_vs_02_learned.json").read_text())
            if (d / "delta_vs_02_learned.json").exists() else {},
            "delta_vs_02_unfrozen": json.loads((d / "delta_vs_02_unfrozen.json").read_text())
            if (d / "delta_vs_02_unfrozen.json").exists() else {},
        }
    # 5 anchors
    data["baseline_agg"] = json.loads(
        (base / "00_baseline" / DATASET / f"seed_{SEED}" / "metrics_aggregate.json").read_text()
    )
    data["alpha10_agg"] = json.loads(
        (base / "01b_mean_diff_scaled" / DATASET / f"seed_{SEED}" / "alpha_10p0"
         / "metrics_aggregate.json").read_text()
    )
    data["e02_agg"] = json.loads(
        (base / "02_final_layer_vector" / DATASET / f"seed_{SEED}" / "metrics_aggregate.json").read_text()
    )
    unf = base / "02_final_layer_vector" / DATASET / f"seed_{SEED}" / "unfrozen" / "metrics_aggregate.json"
    data["unfrozen_agg"] = json.loads(unf.read_text()) if unf.exists() else None
    data["unfrozen_delta_vs_baseline"] = json.loads(
        (unf.parent / "delta_vs_baseline.json").read_text()
    ) if unf.exists() and (unf.parent / "delta_vs_baseline.json").exists() else None
    return data


def fig_ndcg_vs_configs(d: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = [
        "baseline\n(00)", "02 frozen\n(768)",
        "01b α=10\n(0)", "02 unfrozen\n(110 M)",
        "10 r=1\n(37 K)", "10 r=8 (B)\n(295 K)",
    ]
    all_vals = [
        d["baseline_agg"]["all"]["ndcg_cut_10"],
        d["e02_agg"]["all"]["ndcg_cut_10"],
        d["alpha10_agg"]["all"]["ndcg_cut_10"],
        d["unfrozen_agg"]["all"]["ndcg_cut_10"] if d["unfrozen_agg"] else 0.0,
        d["configs"]["qv_r1_l12"]["agg"]["all"]["ndcg_cut_10"],
        d["configs"]["qv_r8_l12"]["agg"]["all"]["ndcg_cut_10"],
    ]
    conf_vals = [
        d["baseline_agg"]["confused"]["ndcg_cut_10"],
        d["e02_agg"]["confused"]["ndcg_cut_10"],
        d["alpha10_agg"]["confused"]["ndcg_cut_10"],
        d["unfrozen_agg"]["confused"]["ndcg_cut_10"] if d["unfrozen_agg"] else 0.0,
        d["configs"]["qv_r1_l12"]["agg"]["confused"]["ndcg_cut_10"],
        d["configs"]["qv_r8_l12"]["agg"]["confused"]["ndcg_cut_10"],
    ]
    x = np.arange(len(labels))
    w = 0.38
    bars1 = ax.bar(x - w / 2, all_vals, w, color="#3b6e8f", label="all slice")
    bars2 = ax.bar(x + w / 2, conf_vals, w, color="#d04545", label="confused slice")
    for i, (a, c) in enumerate(zip(all_vals, conf_vals)):
        ax.text(i - w / 2, a + 0.012, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + w / 2, c + 0.012, f"{c:.4f}", ha="center", fontsize=8)
    baseline_all = d["baseline_agg"]["all"]["ndcg_cut_10"]
    ax.axhline(baseline_all, color="#666", linestyle="--", linewidth=0.8,
               label=f"baseline all ({baseline_all:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("NDCG@10")
    ax.set_title("10 LoRA on Φ — SciFact NDCG@10 vs anchors")
    ax.set_ylim(0, max(max(all_vals), max(conf_vals)) + 0.10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "ndcg_vs_configs")


def fig_delta_ci_forest(d: dict, out_dir: Path) -> None:
    """3 anchors × 2 LoRA configs (+ Phase 2a 인용) 의 paired bootstrap CI."""
    rows = []
    config_label = {
        "qv_r1_l12": "10 r=1 (37 K)",
        "qv_r8_l12": "10 r=8 (B, 295 K)",
    }
    for tag, label in config_label.items():
        if tag not in d["configs"]:
            continue
        r = d["configs"][tag]
        for ref_key, ref_label in (
            ("delta_vs_baseline", "vs baseline"),
            ("delta_vs_02", "vs 02 frozen"),
            ("delta_vs_02_unfrozen", "vs 02 unfrozen"),
        ):
            dd = r.get(ref_key, {})
            for sn in ("all", "confused"):
                rr = dd.get(sn, {}) if dd else {}
                if "mean_delta_ndcg10" not in rr:
                    continue
                rows.append((label, ref_label, sn,
                             rr["mean_delta_ndcg10"], rr["ci_lo"], rr["ci_hi"]))
    # Phase 2a literal insert (for visualization context only)
    rows.append(("10 r=8 (2a, 295 K)\nLR=1e-4 α=16", "vs baseline", "all",
                 *PHASE2A_LITERAL["delta_all_vs_baseline"]))
    rows.append(("10 r=8 (2a, 295 K)\nLR=1e-4 α=16", "vs baseline", "confused",
                 *PHASE2A_LITERAL["delta_conf_vs_baseline"]))

    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.32)))
    sn_color = {"all": "#3b6e8f", "confused": "#d04545"}
    y = np.arange(len(rows))
    for i, (cfg_lbl, ref, sn, m, lo, hi) in enumerate(rows):
        c = sn_color[sn]
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
    ax.set_yticklabels([f"{cfg_lbl}\n{ref}, {sn}" for cfg_lbl, ref, sn, *_ in rows],
                       fontsize=8)
    ax.set_xlabel("Δ NDCG@10")
    ax.set_title("10 LoRA — paired bootstrap 95% CI")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "delta_ci_forest")


def fig_lora_progression(d: dict, out_dir: Path) -> None:
    """Phase 1 → 2a → 2b 의 anchor preservation + confused lift trajectory."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    phases = ["Phase 1\nr=1\nLR=5e-5\nα=r", "Phase 2a\nr=8\nLR=1e-4\nα=2r",
              "Phase 2b (B)\nr=8\nLR=5e-5\nα=r"]
    # Δ all vs baseline (anchor preservation)
    delta_all_means = [
        d["configs"]["qv_r1_l12"]["delta_vs_baseline"]["all"]["mean_delta_ndcg10"],
        PHASE2A_LITERAL["delta_all_vs_baseline"][0],
        d["configs"]["qv_r8_l12"]["delta_vs_baseline"]["all"]["mean_delta_ndcg10"],
    ]
    delta_all_los = [
        d["configs"]["qv_r1_l12"]["delta_vs_baseline"]["all"]["ci_lo"],
        PHASE2A_LITERAL["delta_all_vs_baseline"][1],
        d["configs"]["qv_r8_l12"]["delta_vs_baseline"]["all"]["ci_lo"],
    ]
    delta_all_his = [
        d["configs"]["qv_r1_l12"]["delta_vs_baseline"]["all"]["ci_hi"],
        PHASE2A_LITERAL["delta_all_vs_baseline"][2],
        d["configs"]["qv_r8_l12"]["delta_vs_baseline"]["all"]["ci_hi"],
    ]
    delta_conf_means = [
        d["configs"]["qv_r1_l12"]["delta_vs_baseline"]["confused"]["mean_delta_ndcg10"],
        PHASE2A_LITERAL["delta_conf_vs_baseline"][0],
        d["configs"]["qv_r8_l12"]["delta_vs_baseline"]["confused"]["mean_delta_ndcg10"],
    ]
    delta_conf_los = [
        d["configs"]["qv_r1_l12"]["delta_vs_baseline"]["confused"]["ci_lo"],
        PHASE2A_LITERAL["delta_conf_vs_baseline"][1],
        d["configs"]["qv_r8_l12"]["delta_vs_baseline"]["confused"]["ci_lo"],
    ]
    delta_conf_his = [
        d["configs"]["qv_r1_l12"]["delta_vs_baseline"]["confused"]["ci_hi"],
        PHASE2A_LITERAL["delta_conf_vs_baseline"][2],
        d["configs"]["qv_r8_l12"]["delta_vs_baseline"]["confused"]["ci_hi"],
    ]
    x = np.arange(len(phases))
    w = 0.38
    ax.errorbar(x - w / 2, delta_all_means,
                yerr=[[m - lo for m, lo in zip(delta_all_means, delta_all_los)],
                      [hi - m for m, hi in zip(delta_all_means, delta_all_his)]],
                fmt="o", color="#3b6e8f", label="Δ all vs baseline (CI)",
                capsize=4, markersize=8)
    ax.errorbar(x + w / 2, delta_conf_means,
                yerr=[[m - lo for m, lo in zip(delta_conf_means, delta_conf_los)],
                      [hi - m for m, hi in zip(delta_conf_means, delta_conf_his)]],
                fmt="s", color="#d04545", label="Δ confused vs baseline (CI)",
                capsize=4, markersize=8)
    for i, (a, c) in enumerate(zip(delta_all_means, delta_conf_means)):
        ax.text(i - w / 2, a + 0.012, f"{a:+.3f}", ha="center", fontsize=9,
                color="#3b6e8f")
        ax.text(i + w / 2, c + 0.012, f"{c:+.3f}", ha="center", fontsize=9,
                color="#d04545")
    ax.axhline(0, color="#666", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylabel("Δ NDCG@10 vs baseline")
    ax.set_title("10 LoRA progression: 1 → 2a → 2b (anchor preservation 회복)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "lora_progression")


def fig_lora_AB_norms(d: dict, out_dir: Path) -> None:
    """2 configs (r=1, r=8 (B)) 의 LoRA A/B norm distribution per adapter."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, cfg_tag, cfg_label in zip(
        axes, ["qv_r1_l12", "qv_r8_l12"], ["r=1 (Phase 1)", "r=8 (Phase 2b)"],
    ):
        if cfg_tag not in d["configs"]:
            continue
        stats = d["configs"][cfg_tag]["lora_stats"]
        A_norms = stats["A_norms_per_adapter"]
        B_norms = stats["B_norms_per_adapter"]
        # 24 adapters (2 components × 12 layers) — index by adapter
        x = np.arange(len(A_norms))
        ax.bar(x - 0.2, A_norms, width=0.4, color="#3b6e8f", label="‖A‖")
        ax.bar(x + 0.2, B_norms, width=0.4, color="#c47a2b", label="‖B‖")
        ax.set_xlabel("adapter index (q/v × layer)")
        ax.set_ylabel("norm")
        ax.set_title(f"{cfg_label} per-adapter A/B norm "
                     f"(‖A‖_tot={stats['A_norm_total']:.2f}, "
                     f"‖B‖_tot={stats['B_norm_total']:.2f})")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "lora_AB_norms")


def fig_train_curve_3configs(d: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    color_map = {"qv_r1_l12": "#3b6e8f", "qv_r8_l12": "#d04545"}
    label_map = {"qv_r1_l12": "r=1 (Phase 1)", "qv_r8_l12": "r=8 (Phase 2b)"}
    for tag, color in color_map.items():
        if tag not in d["configs"]:
            continue
        h = d["configs"][tag]["history"]
        axes[0].plot(h["steps"], h["rank_losses"], color=color, linewidth=1,
                     label=label_map[tag])
        axes[1].plot(h["val_epochs"], h["val_ndcg_all"], "-o", color=color,
                     label=label_map[tag])
        axes[2].plot(h["val_epochs"], h["val_ndcg_confused"], "-s", color=color,
                     label=label_map[tag])
    baseline_all = d["baseline_agg"]["all"]["ndcg_cut_10"]
    baseline_conf = d["baseline_agg"]["confused"]["ndcg_cut_10"]
    axes[1].axhline(baseline_all, color="#666", linestyle="--", linewidth=0.8,
                    label="baseline all")
    axes[2].axhline(baseline_conf, color="#666", linestyle="--", linewidth=0.8,
                    label="baseline confused")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("pairwise margin loss")
    axes[0].set_title("Train loss"); axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val NDCG@10 (all)")
    axes[1].set_title("Val all-slice"); axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("val NDCG@10 (confused)")
    axes[2].set_title("Val confused-slice"); axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "train_curve_3configs")


def main() -> None:
    set_rc()
    out_dir = PROJECT_ROOT / "report" / "figures" / "10_lora_phi"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load()
    fig_ndcg_vs_configs(d, out_dir)
    fig_delta_ci_forest(d, out_dir)
    fig_lora_progression(d, out_dir)
    fig_lora_AB_norms(d, out_dir)
    fig_train_curve_3configs(d, out_dir)
    print(f"figures → {out_dir}")


if __name__ == "__main__":
    main()
