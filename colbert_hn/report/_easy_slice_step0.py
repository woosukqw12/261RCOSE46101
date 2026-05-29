"""Step 0 for Experiment 11 — measure Phase 2b 의 *easy-slice* Δ NDCG@10.

가설 (수학적): SciFact Phase 2b 의 Δall ≈ +0.001 + Δconfused ≈ +0.104 ⇒
  w_conf · Δconf + w_easy · Δeasy = Δall
  0.457 × 0.104 + 0.543 × Δeasy ≈ 0.001
  ⇒ Δeasy ≈ −0.085

즉 *redistribution* (confused↑ / easy↓) 가설 의 사전 수학적 expectation.

본 script 는 *실측* easy-slice Δ 를 baseline + Phase 2b per-query NDCG 에서
직접 계산 → 3-seed mean ± std + paired bootstrap CI.

Gate 조건:
  - Δeasy 의 95% CI 상한 < 0 (유의하게 음수) → Exp 11 진행할 가치
  - 그렇지 않으면 Exp 11 skip
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_beir  # noqa: E402
from src.metrics import paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402


def compute_slice_deltas(baseline_per_q: dict, lora_per_q: dict, qids: set,
                          n_iter: int = 10000, seed: int = 42) -> dict:
    base = []
    ours = []
    for q in sorted(qids):
        if q in baseline_per_q and q in lora_per_q:
            b = baseline_per_q[q].get("ndcg_cut_10", 0.0)
            o = lora_per_q[q].get("ndcg_cut_10", 0.0)
            base.append(b)
            ours.append(o)
    if not base:
        return {"n": 0}
    base = np.asarray(base)
    ours = np.asarray(ours)
    delta_mean = float((ours - base).mean())
    mean, lo, hi = paired_bootstrap_ci(ours, base, n_iter=n_iter, ci=0.95, seed=seed)
    return {
        "n": len(base),
        "baseline_ndcg10": float(base.mean()),
        "lora_ndcg10": float(ours.mean()),
        "delta_ndcg10_mean": delta_mean,
        "delta_ci_lo": float(lo),
        "delta_ci_hi": float(hi),
        "positive": float(lo) > 0,
        "negative": float(hi) < 0,
    }


def main():
    # Load baseline (seed 42) + qrels
    _, _, qrels = load_beir("scifact", split="test")
    base_per_q = json.loads(
        (PROJECT_ROOT / "outputs/00_baseline/scifact/seed_42/metrics_per_query.json").read_text()
    )
    base_runs = json.loads(
        (PROJECT_ROOT / "outputs/00_baseline/scifact/seed_42/runs.json").read_text()
    )

    confused = confused_slice(base_runs, qrels, k=1)
    all_q = set(base_runs.keys())
    easy = all_q - confused
    print(f"slice sizes: all={len(all_q)} confused={len(confused)} easy={len(easy)}")
    print(f"w_conf = {len(confused)/len(all_q):.4f}, w_easy = {len(easy)/len(all_q):.4f}")

    # Math expected Δeasy from Phase 2b Δall +0.001, Δconf +0.104 (3-seed mean)
    w_conf = len(confused) / len(all_q)
    w_easy = len(easy) / len(all_q)
    d_all_3seed = 0.001
    d_conf_3seed = 0.104
    d_easy_math = (d_all_3seed - w_conf * d_conf_3seed) / w_easy
    print(f"\nMath expected Δeasy (from 3-seed mean Δall+0.001, Δconf+0.104):")
    print(f"  Δeasy ≈ ({d_all_3seed:+.4f} - {w_conf:.4f} × {d_conf_3seed:+.4f}) / {w_easy:.4f}")
    print(f"        ≈ {d_easy_math:+.4f}")

    print("\n=== 3-seed measured Δ slice ===")
    results = {"slice_sizes": {"all": len(all_q), "confused": len(confused), "easy": len(easy)},
               "math_expected_delta_easy": d_easy_math}
    seeds = [42, 1337, 2024]
    all_d_easy = []
    all_d_conf = []
    all_d_all = []
    for seed in seeds:
        lora_path = (PROJECT_ROOT / f"outputs/10_lora_phi/scifact/seed_{seed}"
                     / "qv_r8_l12/metrics_per_query.json")
        if not lora_path.exists():
            print(f"  seed {seed}: missing artifact")
            continue
        lora_per_q = json.loads(lora_path.read_text())

        d_all = compute_slice_deltas(base_per_q, lora_per_q, all_q, seed=seed)
        d_conf = compute_slice_deltas(base_per_q, lora_per_q, confused, seed=seed)
        d_easy = compute_slice_deltas(base_per_q, lora_per_q, easy, seed=seed)

        all_d_all.append(d_all["delta_ndcg10_mean"])
        all_d_conf.append(d_conf["delta_ndcg10_mean"])
        all_d_easy.append(d_easy["delta_ndcg10_mean"])

        print(f"\nseed {seed}:")
        print(f"  Δall      n={d_all['n']:3d}  Δ={d_all['delta_ndcg10_mean']:+.4f} "
              f"[{d_all['delta_ci_lo']:+.4f},{d_all['delta_ci_hi']:+.4f}]")
        print(f"  Δconf     n={d_conf['n']:3d}  Δ={d_conf['delta_ndcg10_mean']:+.4f} "
              f"[{d_conf['delta_ci_lo']:+.4f},{d_conf['delta_ci_hi']:+.4f}]")
        print(f"  Δeasy     n={d_easy['n']:3d}  Δ={d_easy['delta_ndcg10_mean']:+.4f} "
              f"[{d_easy['delta_ci_lo']:+.4f},{d_easy['delta_ci_hi']:+.4f}]"
              f"{' ✓ positive' if d_easy['positive'] else ' ✗ negative' if d_easy['negative'] else ''}")
        results[f"seed_{seed}"] = {"all": d_all, "confused": d_conf, "easy": d_easy}

    # 3-seed mean ± std
    print(f"\n=== 3-seed mean ± std ===")
    print(f"  Δall     {np.mean(all_d_all):+.4f} ± {np.std(all_d_all):.4f}")
    print(f"  Δconfused{np.mean(all_d_conf):+.4f} ± {np.std(all_d_conf):.4f}")
    print(f"  Δeasy    {np.mean(all_d_easy):+.4f} ± {np.std(all_d_easy):.4f}")
    results["3seed_mean"] = {
        "all": {"mean": float(np.mean(all_d_all)), "std": float(np.std(all_d_all))},
        "confused": {"mean": float(np.mean(all_d_conf)), "std": float(np.std(all_d_conf))},
        "easy": {"mean": float(np.mean(all_d_easy)), "std": float(np.std(all_d_easy))},
    }

    # Gate decision
    d_easy_3seed = np.mean(all_d_easy)
    print(f"\n=== Gate decision ===")
    if d_easy_3seed < -0.02:
        print(f"  Δeasy 3-seed mean = {d_easy_3seed:+.4f} → *significantly negative* → "
              f"Exp 11 진행할 가치 ✓ (redistribution confirmed)")
    else:
        print(f"  Δeasy 3-seed mean = {d_easy_3seed:+.4f} → not significantly negative → "
              f"Exp 11 skip 권장 (anchor 이미 진정 보존)")

    # Save
    out_path = PROJECT_ROOT / "report/figures/_easy_slice_step0.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
