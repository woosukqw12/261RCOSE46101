"""00_baseline — Frozen ColBERT v2 재현 실험.

본 스크립트는 모든 후속 steered configuration (T1.01+) 의 anchor (CLAUDE.md
§3.5) 를 산출한다. anchor 는 paired bootstrap on per-query Δ-metric
(DESIGN.md §5.3) 의 비교 기준이며, 본 실험이 paper 보고치를 재현하지 못하면
이후 LSR 실험의 개선 주장은 해석 불가능해진다.

성공 기준 (DESIGN.md §8): 데이터셋 별 NDCG@10 이 ColBERT v2 paper
(Santhanam et al., 2022, Table 3) 의 보고치와 ±0.005 이내 일치.

    SciFact 0.693  NFCorpus 0.338  SciDocs 0.154
    TREC-COVID 0.738  FiQA 0.356  ArguAna 0.463

사용:
    .venv/bin/python experiments/00_baseline/run.py --dataset scifact   --seed 42
    .venv/bin/python experiments/00_baseline/run.py --dataset nfcorpus  --seed 42
    .venv/bin/python experiments/00_baseline/run.py --dataset scidocs   --seed 42
    .venv/bin/python experiments/00_baseline/run.py --dataset trec-covid --seed 42
    .venv/bin/python experiments/00_baseline/run.py --dataset fiqa      --seed 42
    .venv/bin/python experiments/00_baseline/run.py --dataset arguana   --seed 42

Artifact 출력 경로:
    outputs/00_baseline/{dataset}/seed_{seed}/
        config.json
        env.json
        runs.json              # {qid: [did_top1, ...]}
        runs_scored.json       # {qid: {did: maxsim_score, ...}}
        metrics_per_query.json # pytrec_eval per-query metrics
        metrics_aggregate.json # {all / confused / _meta}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make project root importable regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colbert_hook import ColBERTConfig, ColBERTv2  # noqa: E402
from src.configs import BASELINE  # noqa: E402
from src.data import load_beir  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_aggregate,
    compute_metrics_trec,
    encode_corpus,
    save_env,
    score_queries,
)
from src.utils.io import artifact_dir, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("00_baseline")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        required=True,
        choices=("scifact", "nfcorpus", "scidocs", "trec-covid", "fiqa", "arguana"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default=None)
    p.add_argument("--doc-batch", type=int, default=64)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--doc-chunk", type=int, default=512)
    args = p.parse_args()

    cfg = BASELINE
    if cfg.steering.enabled:
        raise AssertionError(
            "BASELINE config has steering enabled — expected disabled for 00_baseline."
        )

    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(
        "config=%s dataset=%s seed=%d device=%s",
        cfg.config_id,
        args.dataset,
        args.seed,
        device,
    )

    out = artifact_dir(
        exp_name="00_baseline",
        dataset=args.dataset,
        seed=args.seed,
    )
    cfg.save(out / "config.json")
    save_env(out, args.seed, device)

    corpus, queries, qrels = load_beir(args.dataset, split=args.split)
    logger.info(
        "loaded %s: corpus=%d queries=%d qrels=%d",
        args.dataset,
        len(corpus),
        len(queries),
        len(qrels),
    )

    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)

    t0 = time.time()
    dids, d_emb, d_mask = encode_corpus(
        model, corpus, device, batch_size=args.doc_batch
    )
    logger.info(
        "corpus encoded in %.1fs: shape=%s", time.time() - t0, tuple(d_emb.shape)
    )

    # ArguAna places each query as a corpus doc under the same id (counter-
    # argument task); ColBERT-style BEIR evaluation excludes the query's own
    # doc from retrieval. Other datasets do not need this filter.
    exclude_self = args.dataset == "arguana"

    t0 = time.time()
    topk = score_queries(
        model,
        queries,
        dids,
        d_emb,
        d_mask,
        device,
        query_batch=args.query_batch,
        doc_chunk=args.doc_chunk,
        top_k=cfg.eval.retrieval_top_k,
        exclude_self=exclude_self,
    )
    logger.info("queries scored in %.1fs", time.time() - t0)

    runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
    runs_scored = {q: dict(lst) for q, lst in topk.items()}
    per_q = compute_metrics_trec(runs_scored, qrels, metrics_k=cfg.eval.metrics_k)
    agg = build_aggregate(per_q, runs_ranked, qrels, cfg.eval.confused_slice_def)

    save_json(runs_ranked, out / "runs.json")
    save_json(runs_scored, out / "runs_scored.json")
    save_json(per_q, out / "metrics_per_query.json")
    save_json(agg, out / "metrics_aggregate.json")

    logger.info("=== aggregate (all slice) ===")
    for k in sorted(agg["all"]):
        logger.info("  %-20s %.4f", k, agg["all"][k])
    logger.info(
        "confused: %d / %d (%.1f%%)",
        agg["_meta"]["n_confused"],
        agg["_meta"]["n_queries"],
        100 * agg["_meta"]["frac_confused"],
    )
    logger.info("artifacts → %s", out)


if __name__ == "__main__":
    main()
