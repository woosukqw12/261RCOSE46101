"""Mediation sanity check — verify diagnostic-loaded model 의 NDCG = reported NDCG.

Reviewer agent 의 catch: Claim A (same eff_rank, different NDCG = direction matters) 가
*eff_rank ↔ NDCG pairing* 의 정확성에 걸려 있음. §7.3.c.i 의 sanity check 는
*Phase 2b 만* 검증, *M1 / M1b 는 미검증*. *LoRA best-state 미snapshot* 한계 + ep3-final
사용 환경에서 NDCG 재현이 정확한지 확인 필요. 특히 *NFCorpus M1b 의 0.246 재현*
이 Claim A 의 직접 단단함 증거.

CPU 강제 (GPU 점유 회피).

각 (dataset, condition) 에 대해 module_final.pt 로드 → test corpus encode + NDCG@10
재현 → reported 와 비교.
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
from src.utils.repro import set_seed  # noqa: E402

# CPU 강제
DEVICE = torch.device("cpu")


def load_lora_into_model(model, ckpt_path, r=8):
    lora_params = inject_lora_into_bert(
        model.bert, target_components=["q", "v"], layers=None,
        r=r, alpha=None, init_std=0.02,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_state = state["lora"]
    keys = sorted(lora_state.keys(), key=lambda k: int(k.split("_")[-1]))
    assert len(keys) == len(lora_params)
    with torch.no_grad():
        for k, p in zip(keys, lora_params):
            p.data.copy_(lora_state[k].to(p.device))
    return lora_params


REPORTED = {
    ("scifact", "qv_r8_l12"): 0.6367,
    ("scifact", "qv_r8_l12_m1"): 0.6342,
    ("scifact", "qv_r8_l12_m1b"): 0.6613,
    ("nfcorpus", "qv_r8_l12"): 0.0094,
    ("nfcorpus", "qv_r8_l12_m1"): 0.0113,
    ("nfcorpus", "qv_r8_l12_m1b"): 0.2459,
    ("fiqa", "qv_r8_l12"): 0.0005,
    ("fiqa", "qv_r8_l12_m1"): 0.0009,
    # ("fiqa", "qv_r8_l12_m1b"): (queue 종료 후 추가)
}


def main():
    cfg = BASELINE
    set_seed(42)
    print(f"device = {DEVICE} (CPU forced)")

    out_path = (PROJECT_ROOT / "report/figures/_repr_collapse"
                / "mediation_sanity_ndcg.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())
        print(f"resumed: {sorted(results.keys())}")

    for (dataset, tag), reported in REPORTED.items():
        key = f"{dataset}/{tag}"
        if key in results:
            print(f"skip {key} (cached)")
            continue

        ckpt = (PROJECT_ROOT / "outputs/10_lora_phi" / dataset / "seed_42"
                / tag / "module_final.pt")
        if not ckpt.exists():
            print(f"WARN: missing {ckpt}")
            continue

        print(f"\n=== {dataset} / {tag} ===")
        model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(DEVICE)
        load_lora_into_model(model, ckpt, r=8)
        model.to(DEVICE)
        model.eval()

        corpus, queries, qrels = load_beir(dataset, split="test")
        print(f"  test: corpus={len(corpus)} queries={len(queries)}")

        with torch.no_grad():
            dids, d_emb, d_mask = encode_corpus(model, corpus, DEVICE, batch_size=32)
            topk = score_queries(model, queries, dids, d_emb, d_mask, DEVICE,
                                  query_batch=16, doc_chunk=512, top_k=100)
        runs_scored = {q: dict(lst) for q, lst in topk.items()}
        runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
        per_q = compute_metrics_trec(runs_scored, qrels, metrics_k=(10,))
        agg = build_aggregate(per_q, runs_ranked, qrels, cfg.eval.confused_slice_def)

        ndcg10_all = agg["all"]["ndcg_cut_10"]
        results[key] = {
            "diagnostic_NDCG10_all": ndcg10_all,
            "reported_NDCG10_all": reported,
            "match_within_0.01": abs(ndcg10_all - reported) < 0.01,
            "abs_diff": abs(ndcg10_all - reported),
        }
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  diagnostic NDCG@10 all = {ndcg10_all:.4f}")
        print(f"  reported  NDCG@10 all = {reported:.4f}")
        print(f"  abs_diff = {abs(ndcg10_all - reported):.4f}, match={results[key]['match_within_0.01']}")
        del model

    # summary
    print("\n" + "=" * 80)
    print(f"{'condition':<30s} {'diagnostic':>14s} {'reported':>12s} {'match':>8s}")
    print("-" * 80)
    for key, r in results.items():
        print(f"{key:<30s} {r['diagnostic_NDCG10_all']:>14.4f} "
              f"{r['reported_NDCG10_all']:>12.4f} {'✓' if r['match_within_0.01'] else '✗':>8s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
