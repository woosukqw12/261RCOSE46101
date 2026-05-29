"""01_mean_diff — non-learned mean-difference direction baseline.

전체 절차 (README.md 참조):
  1. Train split 로드 + train 시점 baseline ColBERT v2 retrieval (in-memory).
  2. HN-pos triplet mining (top-100 pool, top-10 HN per query).
  3. Unique HN / positive doc 들의 layer-12 hidden state 평균으로 v 계산.
  4. v 를 layer 12 hook 으로 주입, test 시점 retrieval 재실행.
  5. Per-query NDCG@10 와 baseline 의 per-query NDCG@10 을 paired bootstrap.
  6. Artifact 저장.

사용:
    .venv/bin/python experiments/01_mean_diff/run.py --dataset scifact  --seed 42
    .venv/bin/python experiments/01_mean_diff/run.py --dataset nfcorpus --seed 42
    .venv/bin/python experiments/01_mean_diff/run.py --dataset fiqa     --seed 42

데이터셋 제약: train split 보유한 SciFact / NFCorpus / FiQA 한정 (README §"데이터셋 범위" 참조).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colbert_hook import ColBERTConfig, ColBERTv2  # noqa: E402
from src.configs import BASELINE  # noqa: E402
from src.data import doc_text, load_beir  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_aggregate,
    compute_metrics_trec,
    encode_corpus,
    save_env,
    score_queries,
)
from src.hn_mining import mine_triplets  # noqa: E402
from src.mean_diff import HOOK_LAYER, compute_v  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("01_mean_diff")

TRAIN_AVAILABLE = ("scifact", "nfcorpus", "fiqa")
N_HNS_PER_Q = 10
HN_POOL = 100


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset", required=True, choices=TRAIN_AVAILABLE,
        help="train split 보유 dataset 만 허용",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--doc-batch", type=int, default=64)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--doc-chunk", type=int, default=512)
    args = p.parse_args()

    cfg = BASELINE  # encoder + eval config 는 baseline 과 동일 (개입만 추가)

    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(
        "dataset=%s seed=%d device=%s hook_layer=%d",
        args.dataset, args.seed, device, HOOK_LAYER,
    )

    out = artifact_dir(exp_name="01_mean_diff", dataset=args.dataset, seed=args.seed)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    # ---------------------------------------------------------------- model
    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)

    # ----------------------------------------------------- train-side mining
    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info(
        "train: corpus=%d queries=%d qrels=%d",
        len(train_corpus), len(train_queries), len(train_qrels),
    )

    t0 = time.time()
    train_dids, td_emb, td_mask = encode_corpus(
        model, train_corpus, device, batch_size=args.doc_batch,
    )
    logger.info("train corpus encoded in %.1fs", time.time() - t0)

    t0 = time.time()
    train_topk = score_queries(
        model, train_queries, train_dids, td_emb, td_mask, device,
        query_batch=args.query_batch, doc_chunk=args.doc_chunk, top_k=HN_POOL,
    )
    logger.info("train queries scored in %.1fs", time.time() - t0)
    del td_emb, td_mask  # free memory

    train_runs = {q: [d for d, _ in lst] for q, lst in train_topk.items()}
    triplets = mine_triplets(
        train_runs, train_qrels, n_hns_per_q=N_HNS_PER_Q, pool=HN_POOL,
    )

    # ------------------------------------------------------ v computation
    v, stats = compute_v(model, train_corpus, triplets, device, batch_size=args.doc_batch)
    save_json(stats, out / "triplet_stats.json")
    torch.save(v, out / "v.pt")
    logger.info("v computed: norm=%.4f mean|v|=%.4f", stats["v_norm"], stats["v_mean_abs"])
    del train_corpus, train_queries, train_qrels, train_topk, train_runs, triplets

    # ------------------------------------------------ test-side hook + eval
    v_device = v.to(device)

    def steer(h: torch.Tensor) -> torch.Tensor:
        return h - v_device.to(h.dtype)

    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steer)

    test_corpus, test_queries, test_qrels = load_beir(args.dataset, split="test")
    logger.info(
        "test: corpus=%d queries=%d qrels=%d",
        len(test_corpus), len(test_queries), len(test_qrels),
    )

    t0 = time.time()
    test_dids, d_emb, d_mask = encode_corpus(
        model, test_corpus, device, batch_size=args.doc_batch,
    )
    logger.info("test corpus encoded (steered) in %.1fs", time.time() - t0)

    t0 = time.time()
    topk = score_queries(
        model, test_queries, test_dids, d_emb, d_mask, device,
        query_batch=args.query_batch, doc_chunk=args.doc_chunk,
        top_k=cfg.eval.retrieval_top_k,
    )
    logger.info("test queries scored (steered) in %.1fs", time.time() - t0)

    runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
    runs_scored = {q: dict(lst) for q, lst in topk.items()}
    per_q = compute_metrics_trec(runs_scored, test_qrels, metrics_k=cfg.eval.metrics_k)
    agg = build_aggregate(per_q, runs_ranked, test_qrels, cfg.eval.confused_slice_def)
    save_json(runs_ranked, out / "runs.json")
    save_json(runs_scored, out / "runs_scored.json")
    save_json(per_q, out / "metrics_per_query.json")
    save_json(agg, out / "metrics_aggregate.json")

    # ----------------------------------------------- paired bootstrap vs baseline
    baseline_per_q_path = (
        PROJECT_ROOT / "outputs" / "00_baseline" / args.dataset
        / f"seed_{args.seed}" / "metrics_per_query.json"
    )
    if not baseline_per_q_path.exists():
        logger.warning("baseline per-query metrics missing: %s", baseline_per_q_path)
        baseline_per_q = None
    else:
        baseline_per_q = load_json(baseline_per_q_path)

    delta_report: Dict[str, dict] = {}
    if baseline_per_q is not None:
        # confused slice (from baseline's runs, per ROADMAP convention)
        baseline_runs_path = baseline_per_q_path.parent / "runs.json"
        baseline_runs = load_json(baseline_runs_path)
        from src.slices import confused_slice  # local import to avoid head-of-file noise

        conf_k = 1 if cfg.eval.confused_slice_def == "top1_ne_rel" else 3
        conf = confused_slice(baseline_runs, test_qrels, k=conf_k)

        for slice_name, qid_set in (("all", set(per_q.keys())), ("confused", conf)):
            ours, base, qids = align_per_query(
                {q: v for q, v in per_q.items() if q in qid_set},
                {q: v for q, v in baseline_per_q.items() if q in qid_set},
                metric="ndcg_cut_10",
            )
            if len(qids) == 0:
                delta_report[slice_name] = {"n": 0, "skipped": "empty slice"}
                continue
            mean, lo, hi = paired_bootstrap_ci(
                ours, base, n_iter=cfg.eval.bootstrap_iter,
                ci=cfg.eval.bootstrap_ci, seed=args.seed,
            )
            delta_report[slice_name] = {
                "n": len(qids),
                "mean_delta_ndcg10": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_excludes_zero_positive": lo > 0,
                "ci_excludes_zero_negative": hi < 0,
            }

    save_json(delta_report, out / "delta_vs_baseline.json")

    # ------------------------------------------------------------------ log
    logger.info("=== aggregate (all slice) ===")
    for k in sorted(agg["all"]):
        logger.info("  %-20s %.4f", k, agg["all"][k])
    logger.info("confused: %d / %d (%.1f%%)",
                agg["_meta"]["n_confused"], agg["_meta"]["n_queries"],
                100 * agg["_meta"]["frac_confused"])
    if delta_report:
        logger.info("=== paired bootstrap CI vs baseline ===")
        for slice_name, dr in delta_report.items():
            if "mean_delta_ndcg10" in dr:
                logger.info(
                    "  %-9s Δ=%+.4f [%+.4f, %+.4f] (n=%d)%s",
                    slice_name, dr["mean_delta_ndcg10"], dr["ci_lo"], dr["ci_hi"], dr["n"],
                    " ✓ positive" if dr["ci_excludes_zero_positive"]
                    else " ✗ negative" if dr["ci_excludes_zero_negative"]
                    else "",
                )
    logger.info("artifacts → %s", out)


if __name__ == "__main__":
    main()
