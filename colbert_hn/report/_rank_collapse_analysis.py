"""Cross-method *capacity utilization* (effective rank/K) analysis — paper punchline.

비교 method:
  - 06 K-router: K=2, 4, 8  (routing_stats.json 의 effective_K_perplexity)
  - 08 bilinear M r=8 (seed 42)  (UV^T 의 SVD spectrum)
  - 10 LoRA q,v r=8 (Phase 2b, seed 42) (24 adapters 의 평균 effective rank)
  - 10 LoRA q,v r=1 (Phase 1, seed 42, 비교)

Effective rank/K 정의 (모두 perplexity-based, unified):
    p_i = σ_i² / Σ σ²
    H = -Σ p_i log p_i
    effective = exp(H)

→ "nominal" capacity (K or r) 대비 "effective utilization" 비율.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def effective_rank(singular_values: list) -> float:
    """exp(entropy of normalized squared singular values)."""
    s = np.asarray(singular_values, dtype=np.float64)
    s2 = s ** 2
    if s2.sum() <= 0:
        return 0.0
    p = s2 / s2.sum()
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum()
    return float(np.exp(ent))


def load_06_routing(k: int) -> dict:
    p = (PROJECT_ROOT / "outputs/06_k_sweep/scifact/seed_42" / f"k_{k}"
         / "routing_stats.json")
    return json.loads(p.read_text())


def load_08_M_stats() -> dict:
    p = (PROJECT_ROOT / "outputs/08_bilinear_M_minimal/scifact/seed_42/r_8"
         / "M_stats.json")
    return json.loads(p.read_text())


def load_10_lora(tag: str) -> dict:
    p = (PROJECT_ROOT / "outputs/10_lora_phi/scifact/seed_42" / tag
         / "module_final.pt")
    return torch.load(p, map_location="cpu", weights_only=False)


def lora_adapter_effective_ranks(module_pt: dict) -> list:
    """For each adapter pair (A, B), compute effective rank of ΔW = B @ A.

    module_pt['lora'] contains adapter_0=A0, adapter_1=B0, adapter_2=A1, ...
    pairs are (2i, 2i+1) i.e., (A_i, B_i).
    """
    lora = module_pt["lora"]
    keys = sorted(lora.keys(), key=lambda k: int(k.split("_")[-1]))
    eff_ranks = []
    for i in range(0, len(keys), 2):
        A = lora[keys[i]].float()    # (r, in_d)
        B = lora[keys[i + 1]].float()  # (out_d, r)
        # ΔW = B @ A, shape (out_d, in_d)
        dW = B @ A
        if dW.norm() < 1e-8:  # adapter not learned (B=0 initially)
            eff_ranks.append(0.0)
            continue
        sv = torch.linalg.svdvals(dW).numpy()
        eff_ranks.append(effective_rank(sv))
    return eff_ranks


def main():
    rows = []  # (method_label, nominal_capacity, effective, utilization_ratio)

    # 06 K-router (use exp(routing_entropy_mean) as effective K, already saved)
    for k in [2, 4, 8]:
        r = load_06_routing(k)
        eff = r["effective_K_perplexity"]
        rows.append((f"06 K-router K={k}", k, eff, eff / k))

    # 08 bilinear M r=8 (UV^T spectrum)
    m08 = load_08_M_stats()
    eff_08 = effective_rank(m08["UV_singular_values"])
    rows.append(("08 bilinear M r=8", 8, eff_08, eff_08 / 8))

    # 10 LoRA q,v r=1 (Phase 1)
    pt = load_10_lora("qv_r1_l12")
    eff_ranks_lora1 = lora_adapter_effective_ranks(pt)
    nonzero1 = [e for e in eff_ranks_lora1 if e > 0]
    eff_lora1 = float(np.mean(nonzero1)) if nonzero1 else 0.0
    rows.append(("10 LoRA r=1 (Phase 1)", 1, eff_lora1, eff_lora1 / 1))

    # 10 LoRA q,v r=8 Phase 2b
    pt = load_10_lora("qv_r8_l12")
    eff_ranks_lora8 = lora_adapter_effective_ranks(pt)
    nonzero8 = [e for e in eff_ranks_lora8 if e > 0]
    eff_lora8 = float(np.mean(nonzero8)) if nonzero8 else 0.0
    rows.append(("10 LoRA r=8 (Phase 2b)", 8, eff_lora8, eff_lora8 / 8))

    # Per-adapter distribution for LoRA (paper-grade: rank-collapse는 *per-position* 도 발생,
    # 그러나 LoRA 는 *position 수* 가 24 개).
    lora_distributions = {
        "Phase 1 (r=1, 24 adapters)": eff_ranks_lora1,
        "Phase 2b (r=8, 24 adapters)": eff_ranks_lora8,
    }

    # Print table
    print("=" * 76)
    print(f"{'Method':<30s} {'Nominal':>10s} {'Effective':>12s} {'Util ratio':>12s}")
    print("-" * 76)
    for lbl, nom, eff, util in rows:
        print(f"{lbl:<30s} {nom:>10d} {eff:>12.3f} {util:>12.3%}")
    print("=" * 76)

    # ----------------------- figure
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # (1) absolute effective vs nominal
    labels = [r[0] for r in rows]
    nominals = [r[1] for r in rows]
    effectives = [r[2] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    colors_n = ["#cbcbcb"] * 4 + ["#9bd49b"] * 2  # 06/08 grey, 10 green
    colors_e = ["#3b6e8f"] * 4 + ["#2a8a3e"] * 2  # 06/08 blue, 10 dark green
    axes[0].bar(x - w / 2, nominals, w, color=colors_n, label="nominal (K or r)")
    axes[0].bar(x + w / 2, effectives, w, color=colors_e, label="effective (perplexity)")
    for i, (n, e) in enumerate(zip(nominals, effectives)):
        axes[0].text(i - w / 2, n + 0.2, str(n), ha="center", fontsize=9)
        axes[0].text(i + w / 2, e + 0.2, f"{e:.2f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("rank / K")
    axes[0].set_title("Capacity *nominal vs effective* — rank-collapse vs uniform")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(axis="y", alpha=0.3)

    # (2) utilization ratio (effective / nominal) — the contrast
    utils = [r[3] for r in rows]
    bars = axes[1].bar(x, utils, color=colors_e)
    axes[1].axhline(1.0, color="#666", linestyle="--", linewidth=0.8,
                    label="ideal (effective = nominal)")
    for i, u in enumerate(utils):
        axes[1].text(i, u + 0.02, f"{u:.0%}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("utilization ratio (effective / nominal)")
    axes[1].set_title("*Per-axis* utilization: 06/08 collapse, LoRA *per-adapter* same collapse")
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(axis="y", alpha=0.3)

    # (3) LoRA's *spatial distribution* — 24 adapters' effective ranks
    # (06/08 are single-position single-axis, no spatial distribution to show)
    for label, eff_ranks in lora_distributions.items():
        positions = np.arange(len(eff_ranks))
        axes[2].plot(positions, eff_ranks, "-o", linewidth=1.5,
                     markersize=4, label=label, alpha=0.85)
    axes[2].axhline(np.mean(lora_distributions["Phase 2b (r=8, 24 adapters)"]),
                    color="#2a8a3e", linestyle="--", linewidth=0.8,
                    label=f"Phase 2b mean = {np.mean(lora_distributions['Phase 2b (r=8, 24 adapters)']):.2f}")
    axes[2].set_xlabel("adapter index (q/v × 12 layers = 24)")
    axes[2].set_ylabel("per-adapter effective rank")
    axes[2].set_title("LoRA *spatial uniformity*: 24 adapters 모두 *유사 rank* 활용")
    axes[2].set_xticks(np.arange(0, 24, 2))
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = PROJECT_ROOT / "report/figures/_cross_method"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"rank_collapse_contrast.{ext}")
    plt.close(fig)
    print(f"\nfigure → {out_dir}/rank_collapse_contrast.{{pdf,png}}")

    # Also save raw data
    raw = {
        "methods": [
            {"label": r[0], "nominal": r[1], "effective": r[2], "utilization_ratio": r[3]}
            for r in rows
        ],
        "lora_per_adapter": {
            label: {
                "ranks": list(eff_ranks),
                "mean": float(np.mean([e for e in eff_ranks if e > 0])) if any(e > 0 for e in eff_ranks) else 0.0,
                "std": float(np.std([e for e in eff_ranks if e > 0])) if any(e > 0 for e in eff_ranks) else 0.0,
                "n_active": int(sum(1 for e in eff_ranks if e > 0)),
            }
            for label, eff_ranks in lora_distributions.items()
        },
        "definition": "effective = exp(-Σ p_i log p_i) where p_i = σ_i²/Σσ²",
        "notes": (
            "06 effective K from routing entropy perplexity (saved). "
            "08 effective rank from UV^T singular value spectrum. "
            "10 LoRA per-adapter effective rank of ΔW=B@A, mean over 24 adapters. "
            "LoRA spatial uniformity: 24 adapters 모두 유사 rank 활용 (std 측정)."
        ),
    }
    with (out_dir / "rank_collapse_data.json").open("w") as f:
        json.dump(raw, f, indent=2)
    print(f"data → {out_dir}/rank_collapse_data.json")


if __name__ == "__main__":
    main()
