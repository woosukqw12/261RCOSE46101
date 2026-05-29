"""08_bilinear_M_minimal — Translation family 밖, Stage 2 main novelty 의 *critical*
falsification.

ColBERT v2 의 *MaxSim 의 inner product 자체* 를 일반화. Hidden state 는 frozen
모델의 출력 그대로 (hook 없음); 학습은 *bilinear correction* $M = I + UV^\top$
($U, V \in \mathbb{R}^{128 \times r}$) 의 U, V 만.

수식 (single layer 의 hook 변경 없이 retrieval 시점에 적용):
    s_M(q, d) = sum_i max_j q_i^T M d_j
             = sum_i max_j ( <q_i, d_j> + (U^T q_i)^T (V^T d_j) )

Anchor preservation: U=V=0 init → M=I → baseline 과 *정확히* 동일 retrieval.

학습 파라미터: 2 * 128 * r. r=8 시 2,048 (≪ 50K).

사용:
    .venv/bin/python experiments/08_bilinear_M_minimal/run.py --dataset scifact --seed 42 --r 8

Artifact:
    outputs/08_bilinear_M_minimal/{dataset}/seed_{seed}/r_{r}/
        ├── config / env / train_config / module_final.pt / train_history.json
        ├── runs / runs_scored / metrics_per_query / metrics_aggregate.json
        ├── M_stats.json (‖U‖, ‖V‖, ‖UV^T‖, M 의 eigenvalue spectrum 일부)
        └── delta_vs_{baseline, mean_diff_alpha10, 02_learned, 06_k_sweep_k2}.json
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bilinear import BilinearMetric  # noqa: E402
from src.colbert_hook import ColBERTConfig, ColBERTv2  # noqa: E402
from src.configs import BASELINE  # noqa: E402
from src.data import load_beir  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_aggregate,
    compute_metrics_trec,
    encode_corpus,
    save_env,
)
from src.hn_mining import mine_triplets  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.train import (  # noqa: E402
    TrainConfigLite,
    _bilinear_score_queries,
    train_bilinear_metric,
)
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("08_bilinear_M_minimal")

TRAIN_AVAILABLE = ("scifact", "nfcorpus", "fiqa")
N_HNS_PER_Q = 10
HN_POOL = 100


def _paired_ci_vs(per_q, ref_per_q, qids, metric="ndcg_cut_10",
                  n_iter=10000, ci=0.95, seed=42):
    ours, base, q = align_per_query(
        {q: v for q, v in per_q.items() if q in qids},
        {q: v for q, v in ref_per_q.items() if q in qids},
        metric=metric,
    )
    if len(q) == 0:
        return {"n": 0, "skipped": "empty"}
    mean, lo, hi = paired_bootstrap_ci(ours, base, n_iter=n_iter, ci=ci, seed=seed)
    return {"n": len(q), "mean_delta_ndcg10": mean, "ci_lo": lo, "ci_hi": hi,
            "positive": lo > 0, "negative": hi < 0}


def _m_diagnostics(metric: BilinearMetric) -> dict:
    """학습된 M = I + UV^T 의 spectrum / norm / rank 진단."""
    U = metric.U.detach().to("cpu").to(torch.float32)
    V = metric.V.detach().to("cpu").to(torch.float32)
    UV = U @ V.T  # (D, D)
    M = torch.eye(metric.dim) + UV
    # eigenvalues of UV^T (rank ≤ r) — note UV^T not symmetric, use svd
    s_uv = torch.linalg.svdvals(UV)  # singular values
    s_m = torch.linalg.svdvals(M)
    return {
        "dim": int(metric.dim),
        "r": int(metric.r),
        "U_norm": float(U.norm().item()),
        "V_norm": float(V.norm().item()),
        "UV_norm_fro": float(UV.norm().item()),
        "UV_singular_values": [float(x) for x in s_uv[:int(metric.r)].tolist()],
        "M_singular_values_top10": [float(x) for x in s_m[:10].tolist()],
        "M_singular_values_bottom10": [float(x) for x in s_m[-10:].tolist()],
        "M_condition_number": float(s_m[0].item() / s_m[-1].item()) if s_m[-1] > 0 else float("inf"),
        "M_deviation_from_I_fro": float((M - torch.eye(metric.dim)).norm().item()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=TRAIN_AVAILABLE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--r", type=int, default=8, help="rank of UV^T")
    p.add_argument("--init", default="zero", choices=["zero", "small_random"])
    p.add_argument("--device", default=None)
    p.add_argument("--doc-batch", type=int, default=64)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--doc-chunk", type=int, default=512)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lambda-anchor", type=float, default=0.0)
    args = p.parse_args()

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(
        "dataset=%s seed=%d device=%s r=%d init=%s",
        args.dataset, args.seed, device, args.r, args.init,
    )

    base_out = artifact_dir(
        exp_name="08_bilinear_M_minimal", dataset=args.dataset, seed=args.seed,
    )
    out = base_out / f"r_{args.r}"
    out.mkdir(parents=True, exist_ok=True)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    train_cfg = TrainConfigLite(
        hook_layer=12,  # ignored for bilinear, but kept for consistency
        margin=args.margin, lr=args.lr, weight_decay=1e-4,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        val_split=0.1, lambda_anchor=args.lambda_anchor, seed=args.seed,
    )
    save_json({**dataclasses.asdict(train_cfg), "r": args.r, "init": args.init},
              out / "train_config.json")

    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
    # NO hook — frozen ColBERT 의 vanilla output 사용
    model.clear_hooks()
    metric = BilinearMetric(dim=128, r=args.r, init=args.init).to(device)
    logger.info(
        "metric: dim=128 r=%d params=%d (init=%s)",
        args.r, metric.num_params(), args.init,
    )
    if args.init == "zero":
        assert metric.U.detach().norm().item() == 0.0
        assert metric.V.detach().norm().item() == 0.0

    # ------------------------------------------------------------------- train
    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info("train: corpus=%d queries=%d", len(train_corpus), len(train_queries))

    t0 = time.time()
    with torch.no_grad():
        train_dids, td_emb, td_mask = encode_corpus(
            model, train_corpus, device, batch_size=args.doc_batch,
        )
        # baseline pass (M=I) for HN mining — use vanilla model.maxsim
        train_topk = _bilinear_score_queries(
            model, metric, train_queries, train_dids, td_emb, td_mask, device,
            query_batch=args.query_batch, doc_chunk=args.doc_chunk, top_k=HN_POOL,
        )
    logger.info("train baseline pass in %.1fs", time.time() - t0)
    del td_emb, td_mask

    train_runs = {q: [d for d, _ in lst] for q, lst in train_topk.items()}
    triplets = mine_triplets(train_runs, train_qrels, n_hns_per_q=N_HNS_PER_Q, pool=HN_POOL)
    logger.info("mined %d triplets", len(triplets))

    history = train_bilinear_metric(
        model=model, metric=metric, train_triplets=triplets,
        queries=train_queries, corpus=train_corpus, qrels=train_qrels,
        device=device, cfg=train_cfg,
        val_eval_kwargs={"doc_batch": args.doc_batch,
                         "query_batch": args.query_batch,
                         "doc_chunk": args.doc_chunk},
    )

    diagnostics = _m_diagnostics(metric)
    save_json(diagnostics, out / "M_stats.json")
    save_json(dataclasses.asdict(history), out / "train_history.json")
    torch.save(metric.state_dict(), out / "module_final.pt")
    logger.info(
        "train done: ‖U‖=%.4f ‖V‖=%.4f ‖UV^T‖_F=%.4f M_cond=%.4f deviation_from_I=%.4f",
        diagnostics["U_norm"], diagnostics["V_norm"], diagnostics["UV_norm_fro"],
        diagnostics["M_condition_number"], diagnostics["M_deviation_from_I_fro"],
    )

    del train_corpus, train_queries, train_qrels, triplets, train_topk, train_runs

    # -------------------------------------------------------------------- test
    test_corpus, test_queries, test_qrels = load_beir(args.dataset, split="test")
    logger.info("test: corpus=%d queries=%d", len(test_corpus), len(test_queries))
    metric.eval()
    t0 = time.time()
    with torch.no_grad():
        test_dids, d_emb, d_mask = encode_corpus(
            model, test_corpus, device, batch_size=args.doc_batch,
        )
        topk = _bilinear_score_queries(
            model, metric, test_queries, test_dids, d_emb, d_mask, device,
            query_batch=args.query_batch, doc_chunk=args.doc_chunk,
            top_k=cfg.eval.retrieval_top_k,
        )
    logger.info("test eval pass in %.1fs", time.time() - t0)

    runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
    runs_scored = {q: dict(lst) for q, lst in topk.items()}
    per_q = compute_metrics_trec(runs_scored, test_qrels, metrics_k=cfg.eval.metrics_k)
    agg = build_aggregate(per_q, runs_ranked, test_qrels, cfg.eval.confused_slice_def)
    save_json(runs_ranked, out / "runs.json")
    save_json(runs_scored, out / "runs_scored.json")
    save_json(per_q, out / "metrics_per_query.json")
    save_json(agg, out / "metrics_aggregate.json")

    baseline_per_q_path = (
        PROJECT_ROOT / "outputs" / "00_baseline" / args.dataset
        / f"seed_{args.seed}" / "metrics_per_query.json"
    )
    baseline_runs_path = baseline_per_q_path.parent / "runs.json"
    baseline_per_q = load_json(baseline_per_q_path) if baseline_per_q_path.exists() else None
    baseline_runs = load_json(baseline_runs_path) if baseline_runs_path.exists() else None
    conf = confused_slice(baseline_runs, test_qrels, k=1) if baseline_runs else set()

    deltas = {}

    def _record(name: str, ref_path: Path) -> None:
        if ref_path.exists():
            ref = load_json(ref_path)
            deltas[name] = {
                "all": _paired_ci_vs(per_q, ref, set(per_q.keys()), seed=args.seed),
                "confused": _paired_ci_vs(per_q, ref, conf, seed=args.seed) if conf else {},
            }
            save_json(deltas[name], out / f"delta_{name}.json")

    if baseline_per_q is not None:
        deltas["vs_baseline"] = {
            "all": _paired_ci_vs(per_q, baseline_per_q, set(per_q.keys()), seed=args.seed),
            "confused": _paired_ci_vs(per_q, baseline_per_q, conf, seed=args.seed) if conf else {},
        }
        save_json(deltas["vs_baseline"], out / "delta_vs_baseline.json")

    _record(
        "vs_mean_diff_alpha10",
        PROJECT_ROOT / "outputs" / "01b_mean_diff_scaled" / args.dataset
        / f"seed_{args.seed}" / "alpha_10p0" / "metrics_per_query.json",
    )
    _record(
        "vs_02_learned",
        PROJECT_ROOT / "outputs" / "02_final_layer_vector" / args.dataset
        / f"seed_{args.seed}" / "metrics_per_query.json",
    )
    _record(
        "vs_06_k_sweep_k2",
        PROJECT_ROOT / "outputs" / "06_k_sweep" / args.dataset
        / f"seed_{args.seed}" / "k_2" / "metrics_per_query.json",
    )
    _record(
        "vs_06_k_sweep_k4",
        PROJECT_ROOT / "outputs" / "06_k_sweep" / args.dataset
        / f"seed_{args.seed}" / "k_4" / "metrics_per_query.json",
    )

    logger.info("=== aggregate (all slice, r=%d) ===", args.r)
    for k in sorted(agg["all"]):
        logger.info("  %-20s %.4f", k, agg["all"][k])
    logger.info("confused: %d / %d (%.1f%%)",
                agg["_meta"]["n_confused"], agg["_meta"]["n_queries"],
                100 * agg["_meta"]["frac_confused"])
    for name, dd in deltas.items():
        logger.info("=== Δ vs %s ===", name)
        for sn, dr in dd.items():
            if "mean_delta_ndcg10" in dr:
                logger.info(
                    "  %-9s Δ=%+.4f [%+.4f,%+.4f] (n=%d)%s",
                    sn, dr["mean_delta_ndcg10"], dr["ci_lo"], dr["ci_hi"], dr["n"],
                    " ✓ positive" if dr.get("positive")
                    else " ✗ negative" if dr.get("negative") else "",
                )
    logger.info("artifacts → %s", out)


if __name__ == "__main__":
    main()
