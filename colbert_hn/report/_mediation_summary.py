"""Summarize mediation 1 + 1b results once all 6 runs complete.

Reads:
  outputs/10_lora_phi/{dataset}/seed_42/qv_r8_l12{_m1,_m1b}/{
    metrics_aggregate.json, delta_vs_baseline.json, train_history.json,
  }

Prints comparison table:
  Phase 2b baseline vs Mediation 1 vs Mediation 1b, per dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["scifact", "nfcorpus", "fiqa"]
TAGS = {
    "Phase 2b (baseline)": "qv_r8_l12",
    "Mediation 1 (warmup+clip)": "qv_r8_l12_m1",
    "Mediation 1b (in-batch neg)": "qv_r8_l12_m1b",
}


def load_run(dataset: str, tag: str) -> dict:
    d = PROJECT_ROOT / "outputs/10_lora_phi" / dataset / "seed_42" / tag
    if not (d / "metrics_aggregate.json").exists():
        return None
    agg = json.loads((d / "metrics_aggregate.json").read_text())
    dlt_path = d / "delta_vs_baseline.json"
    dlt = json.loads(dlt_path.read_text()) if dlt_path.exists() else None
    hist_path = d / "train_history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else None
    return {
        "ndcg10_all": agg["all"].get("ndcg_cut_10"),
        "ndcg10_confused": agg.get("confused", {}).get("ndcg_cut_10"),
        "n_queries": agg["_meta"].get("n_queries"),
        "delta_all": dlt["all"] if dlt else None,
        "delta_confused": dlt.get("confused", {}) if dlt else None,
        "val_history_all": hist.get("val_ndcg_all") if hist else None,
        "val_history_confused": hist.get("val_ndcg_confused") if hist else None,
    }


def main():
    rows = []
    for dataset in DATASETS:
        for cond_name, tag in TAGS.items():
            r = load_run(dataset, tag)
            if r is None:
                rows.append((dataset, cond_name, None))
            else:
                rows.append((dataset, cond_name, r))

    print("=" * 110)
    print(f"{'Dataset':<10s} {'Condition':<28s} {'NDCG@10 all':>12s} {'Δ all (CI)':>22s} "
          f"{'Δ conf (CI)':>22s} {'Judge':>8s}")
    print("-" * 110)
    for dataset, cond, r in rows:
        if r is None:
            print(f"{dataset:<10s} {cond:<28s} {'(not yet)':>12s}")
            continue
        ndcg = r["ndcg10_all"]
        d_all = r["delta_all"]
        d_conf = r["delta_confused"]
        d_all_str = (f"{d_all['mean_delta_ndcg10']:+.4f} [{d_all['ci_lo']:+.3f},{d_all['ci_hi']:+.3f}]"
                     if d_all and "mean_delta_ndcg10" in d_all else "—")
        d_conf_str = (f"{d_conf['mean_delta_ndcg10']:+.4f} [{d_conf['ci_lo']:+.3f},{d_conf['ci_hi']:+.3f}]"
                      if d_conf and "mean_delta_ndcg10" in d_conf else "—")
        # judgement: per pre-commit §2.3 (Δ all > -0.10 = recovered, Δ all > 0 + Δ conf > 0 = strict)
        if d_all and "mean_delta_ndcg10" in d_all:
            d_all_v = d_all["mean_delta_ndcg10"]
            if d_all.get("positive", False) and d_conf.get("positive", False):
                judge = "strict ✓"
            elif d_all_v > -0.10:
                judge = "recovered"
            elif d_all_v > -0.20:
                judge = "partial"
            else:
                judge = "✗ cat"
        else:
            judge = "—"
        print(f"{dataset:<10s} {cond:<28s} {ndcg:>12.4f} {d_all_str:>22s} {d_conf_str:>22s} {judge:>8s}")
    print("=" * 110)


if __name__ == "__main__":
    main()
