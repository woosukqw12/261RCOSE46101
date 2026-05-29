"""Sanity check for Diagnostic B — re-evaluate NDCG@10 on the diagnostic-loaded model.

Reviewer 의 critical catch: "rank-1 embedding 으로 NDCG 0.65 불가". 만약 diagnostic
이 로드한 model 의 NDCG 가 보고된 SciFact 0.6367 과 일치하면 → collapse 가 진짜
+ rank-1 residual 이 task ranking 을 보존한다는 *놀라운* 발견. 만약 NDCG 가
random 수준 (~0.01) 이면 → checkpoint / injection 불일치 → 모든 collapse 수치
의심 → reload + 재실행.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colbert_hook import ColBERTConfig, ColBERTv2  # noqa: E402
from src.configs import BASELINE  # noqa: E402
from src.data import load_beir  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_aggregate, compute_metrics_trec, encode_corpus, score_queries,
)
from src.lora import inject_lora_into_bert  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402


def load_lora_into_model(model: ColBERTv2, ckpt_path: Path, components, r):
    lora_params = inject_lora_into_bert(
        model.bert, target_components=components, layers=None,
        r=r, alpha=None, init_std=0.02,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_state = state["lora"]
    keys = sorted(lora_state.keys(), key=lambda k: int(k.split("_")[-1]))
    assert len(keys) == len(lora_params), (
        f"adapter count mismatch: ckpt={len(keys)} expected={len(lora_params)}"
    )
    with torch.no_grad():
        for k, p in zip(keys, lora_params):
            p.data.copy_(lora_state[k].to(p.device))
    return lora_params


def main():
    cfg = BASELINE
    set_seed(42)
    device = get_device(None)
    print(f"device = {device}")

    results = {}
    for dataset in ["scifact", "nfcorpus", "fiqa"]:
        ckpt = (PROJECT_ROOT / "outputs/10_lora_phi" / dataset / "seed_42"
                / "qv_r8_l12/module_final.pt")
        print(f"\n=== {dataset}: loading {ckpt}")

        model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
        load_lora_into_model(model, ckpt, components=["q", "v"], r=8)
        model.to(device)
        model.eval()

        corpus, queries, qrels = load_beir(dataset, split="test")
        print(f"  test: corpus={len(corpus)} queries={len(queries)}")

        with torch.no_grad():
            dids, d_emb, d_mask = encode_corpus(model, corpus, device, batch_size=64)
            topk = score_queries(
                model, queries, dids, d_emb, d_mask, device,
                query_batch=16, doc_chunk=512, top_k=100,
            )
        runs_scored = {q: dict(lst) for q, lst in topk.items()}
        runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
        per_q = compute_metrics_trec(runs_scored, qrels, metrics_k=(10,))
        agg = build_aggregate(per_q, runs_ranked, qrels, cfg.eval.confused_slice_def)

        ndcg10_all = agg["all"]["ndcg_cut_10"]
        ndcg10_confused = (agg.get("confused", {}).get("ndcg_cut_10", float("nan")))
        reported = {
            "scifact": 0.6367,
            "nfcorpus": 0.0094,
            "fiqa": 0.0005,
        }[dataset]

        results[dataset] = {
            "diagnostic_NDCG10_all": ndcg10_all,
            "diagnostic_NDCG10_confused": ndcg10_confused,
            "reported_NDCG10_all": reported,
            "match": abs(ndcg10_all - reported) < 0.01,
            "n_queries": len(per_q),
        }
        print(f"  diagnostic-loaded NDCG@10 all = {ndcg10_all:.4f}")
        print(f"  reported NDCG@10 all          = {reported:.4f}")
        print(f"  match within 0.01: {results[dataset]['match']}")

        del model

    out_path = (PROJECT_ROOT / "report/figures/_repr_collapse"
                / "sanity_check_ndcg.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {out_path}")

    print("\n" + "=" * 70)
    print(f"{'dataset':<12s} {'diagnostic':>14s} {'reported':>12s} {'match':>8s}")
    print("-" * 70)
    for ds, r in results.items():
        print(f"{ds:<12s} {r['diagnostic_NDCG10_all']:>14.4f} "
              f"{r['reported_NDCG10_all']:>12.4f} {'✓' if r['match'] else '✗':>8s}")


if __name__ == "__main__":
    main()
