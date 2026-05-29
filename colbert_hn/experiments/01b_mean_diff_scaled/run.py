"""01b_mean_diff_scaled — magnitude sweep over the non-learned mean-diff direction.

`v = mean(h_HN) - mean(h_pos)` 를 unit-normalize 후 α 로 scale:
    \\tilde{h}^{(12)} = h^{(12)} - α · v / ‖v‖,   α ∈ {0.5, 1, 2, 5, 10}

Train v 계산은 한 번만 (01_mean_diff 의 로직 재현), 각 α 별로 test corpus 를
재인코딩 + 평가 + paired bootstrap (baseline 대비). 결과는
`outputs/01b_mean_diff_scaled/{dataset}/seed_{seed}/alpha_{α}/` 에 분리 저장.

사용:
    .venv/bin/python experiments/01b_mean_diff_scaled/run.py --dataset scifact --seed 42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

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
from src.hn_mining import mine_triplets  # noqa: E402
from src.mean_diff import HOOK_LAYER, compute_v  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("01b_mean_diff_scaled")

ALPHAS = (0.5, 1.0, 2.0, 5.0, 10.0)
N_HNS_PER_Q = 10
HN_POOL = 100
TRAIN_AVAILABLE = ("scifact", "nfcorpus", "fiqa")


def _alpha_dir_name(alpha: float) -> str:
    s = f"{alpha:.1f}"
    return "alpha_" + s.replace(".", "p")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset", required=True, choices=TRAIN_AVAILABLE,
        help="train split 보유 dataset 만 허용 (현재 sub-experiment 는 scifact 권장)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--doc-batch", type=int, default=64)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--doc-chunk", type=int, default=512)
    args = p.parse_args()

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("dataset=%s seed=%d device=%s alphas=%s", args.dataset, args.seed, device, ALPHAS)

    root = artifact_dir(exp_name="01b_mean_diff_scaled", dataset=args.dataset, seed=args.seed)
    save_env(root, args.seed, device)
    cfg.save(root / "config.json")

    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)

    # ----------------------------------------------------- train-side: compute v
    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info("train: corpus=%d queries=%d", len(train_corpus), len(train_queries))

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
    del td_emb, td_mask

    train_runs = {q: [d for d, _ in lst] for q, lst in train_topk.items()}
    triplets = mine_triplets(train_runs, train_qrels, n_hns_per_q=N_HNS_PER_Q, pool=HN_POOL)

    v, stats = compute_v(model, train_corpus, triplets, device, batch_size=args.doc_batch)
    torch.save(v, root / "v.pt")
    save_json(stats, root / "triplet_stats.json")
    logger.info("v computed: norm=%.4f", stats["v_norm"])
    del train_corpus, train_queries, train_qrels, train_topk, train_runs, triplets

    v_unit = v / max(v.norm().item(), 1e-12)

    # ---------------------------------------------- test data + baseline metrics
    test_corpus, test_queries, test_qrels = load_beir(args.dataset, split="test")
    baseline_per_q_path = (
        PROJECT_ROOT / "outputs" / "00_baseline" / args.dataset
        / f"seed_{args.seed}" / "metrics_per_query.json"
    )
    baseline_runs_path = baseline_per_q_path.parent / "runs.json"
    baseline_per_q = load_json(baseline_per_q_path) if baseline_per_q_path.exists() else None
    baseline_runs = load_json(baseline_runs_path) if baseline_runs_path.exists() else None

    sweep_summary: List[dict] = []

    for alpha in ALPHAS:
        adir = root / _alpha_dir_name(alpha)
        adir.mkdir(parents=True, exist_ok=True)
        scaled = (alpha * v_unit).to(device)

        def steer(h: torch.Tensor, _scaled=scaled) -> torch.Tensor:
            return h - _scaled.to(h.dtype)

        model.clear_hooks()
        model.register_layer_hook(HOOK_LAYER, steer)

        logger.info("--- alpha=%.1f ---", alpha)
        t0 = time.time()
        test_dids, d_emb, d_mask = encode_corpus(
            model, test_corpus, device, batch_size=args.doc_batch,
        )
        logger.info("test corpus encoded (alpha=%.1f) in %.1fs", alpha, time.time() - t0)

        t0 = time.time()
        topk = score_queries(
            model, test_queries, test_dids, d_emb, d_mask, device,
            query_batch=args.query_batch, doc_chunk=args.doc_chunk,
            top_k=cfg.eval.retrieval_top_k,
        )
        logger.info("queries scored (alpha=%.1f) in %.1fs", alpha, time.time() - t0)

        runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
        runs_scored = {q: dict(lst) for q, lst in topk.items()}
        per_q = compute_metrics_trec(runs_scored, test_qrels, metrics_k=cfg.eval.metrics_k)
        agg = build_aggregate(per_q, runs_ranked, test_qrels, cfg.eval.confused_slice_def)
        save_json(runs_ranked, adir / "runs.json")
        save_json(runs_scored, adir / "runs_scored.json")
        save_json(per_q, adir / "metrics_per_query.json")
        save_json(agg, adir / "metrics_aggregate.json")

        delta_report: Dict[str, dict] = {}
        if baseline_per_q is not None and baseline_runs is not None:
            conf = confused_slice(baseline_runs, test_qrels, k=1)
            for slice_name, qid_set in (("all", set(per_q.keys())), ("confused", conf)):
                ours, base, qids = align_per_query(
                    {q: vv for q, vv in per_q.items() if q in qid_set},
                    {q: vv for q, vv in baseline_per_q.items() if q in qid_set},
                    metric="ndcg_cut_10",
                )
                if len(qids) == 0:
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
                    "positive": lo > 0,
                    "negative": hi < 0,
                }
        save_json(delta_report, adir / "delta_vs_baseline.json")
        sweep_summary.append(
            {
                "alpha": alpha,
                "ndcg_cut_10_all": agg["all"]["ndcg_cut_10"],
                "delta": delta_report,
            }
        )

        logger.info(
            "  alpha=%.1f NDCG@10=%.4f  Δall=%+.4f [%+.4f,%+.4f]  Δconf=%+.4f [%+.4f,%+.4f]",
            alpha, agg["all"]["ndcg_cut_10"],
            delta_report.get("all", {}).get("mean_delta_ndcg10", float("nan")),
            delta_report.get("all", {}).get("ci_lo", float("nan")),
            delta_report.get("all", {}).get("ci_hi", float("nan")),
            delta_report.get("confused", {}).get("mean_delta_ndcg10", float("nan")),
            delta_report.get("confused", {}).get("ci_lo", float("nan")),
            delta_report.get("confused", {}).get("ci_hi", float("nan")),
        )

    save_json(sweep_summary, root / "sweep_summary.json")
    logger.info("=== sweep summary ===")
    logger.info("alpha   NDCG@10  Δ_all(CI)             Δ_conf(CI)")
    for s in sweep_summary:
        da = s["delta"].get("all", {})
        dc = s["delta"].get("confused", {})
        logger.info(
            "%.1f     %.4f   %+.4f [%+.4f,%+.4f]   %+.4f [%+.4f,%+.4f]",
            s["alpha"], s["ndcg_cut_10_all"],
            da.get("mean_delta_ndcg10", float("nan")),
            da.get("ci_lo", float("nan")), da.get("ci_hi", float("nan")),
            dc.get("mean_delta_ndcg10", float("nan")),
            dc.get("ci_lo", float("nan")), dc.get("ci_hi", float("nan")),
        )
    logger.info("artifacts → %s", root)


if __name__ == "__main__":
    main()
