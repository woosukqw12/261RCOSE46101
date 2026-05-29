"""09_bilinear_M_e5_distill — Bilinear M + E5-Mistral margin distillation.

08 의 *pairwise margin only* 학습에서 발견된 rank-1 collapse 의 해소 시도 —
E5-Mistral-7B-Instruct 의 cross-encoder-quality margin 을 *teacher* 로
distillation:

Loss = clamp(margin - (s_pos - s_hn), 0).mean()
       + lambda_distill * MSE(s_student_margin, teacher_scale * e5_margin)

E5 margin 은 사전 추출된 train query embedding (data/e5_teacher/
e5_train_q_emb_scifact.pt) + corpus embedding (data/e5_teacher/
e5_topk_scifact.pt) 의 cosine 으로 batch 단위 lookup.

사용:
    .venv/bin/python experiments/09_bilinear_M_e5_distill/run.py \\
        --dataset scifact --seed 42 --r 8 --lambda-distill 1.0
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
    train_bilinear_metric_distill,
)
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("09_bilinear_M_e5_distill")

TRAIN_AVAILABLE = ("scifact",)  # E5 train q emb 추출된 dataset 만
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
    U = metric.U.detach().to("cpu").to(torch.float32)
    V = metric.V.detach().to("cpu").to(torch.float32)
    UV = U @ V.T
    M = torch.eye(metric.dim) + UV
    s_uv = torch.linalg.svdvals(UV)
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


def _load_e5_teacher(dataset: str):
    """E5 train query emb + corpus emb 로드 + qid/did → idx 매핑 빌드."""
    e5_dir = PROJECT_ROOT / "data" / "e5_teacher"
    q_path = e5_dir / f"e5_train_q_emb_{dataset}.pt"
    d_path = e5_dir / f"e5_topk_{dataset}.pt"
    if not q_path.exists():
        raise FileNotFoundError(
            f"E5 train query emb 부재: {q_path}. "
            f"먼저 .venv/bin/python data/e5_teacher/extract_train_queries.py --dataset {dataset}"
        )
    if not d_path.exists():
        raise FileNotFoundError(f"E5 corpus emb 부재: {d_path}.")
    qd = torch.load(q_path, weights_only=False)
    dd = torch.load(d_path, weights_only=False)
    e5_q_emb = qd["query_emb"]  # (N_q_train, D_e5) fp16
    e5_d_emb = dd["doc_emb"]    # (N_d, D_e5) fp16
    qid_to_idx = {q: i for i, q in enumerate(qd["qids"])}
    did_to_idx = {d: i for i, d in enumerate(dd["doc_ids"])}
    logger.info(
        "loaded E5 teacher: train_q=%d (%s), corpus=%d (%s)",
        e5_q_emb.shape[0], e5_q_emb.dtype,
        e5_d_emb.shape[0], e5_d_emb.dtype,
    )
    return e5_q_emb, e5_d_emb, qid_to_idx, did_to_idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=TRAIN_AVAILABLE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--init", default="small_random", choices=["zero", "small_random"])
    p.add_argument("--device", default=None)
    p.add_argument("--doc-batch", type=int, default=64)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--doc-chunk", type=int, default=512)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lambda-distill", type=float, default=1.0,
                   help="Margin-MSE distill weight (vs pairwise rank loss)")
    p.add_argument("--teacher-scale", type=float, default=8.0,
                   help="Multiplier for E5 cosine margin → ColBERT-scale margin")
    args = p.parse_args()

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(
        "dataset=%s seed=%d device=%s r=%d init=%s λ_distill=%.3f teacher_scale=%.3f",
        args.dataset, args.seed, device, args.r, args.init,
        args.lambda_distill, args.teacher_scale,
    )

    base_out = artifact_dir(
        exp_name="09_bilinear_M_e5_distill", dataset=args.dataset, seed=args.seed,
    )
    tag = f"r_{args.r}_ld_{args.lambda_distill:.2f}".replace(".", "p")
    out = base_out / tag
    out.mkdir(parents=True, exist_ok=True)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    train_cfg = TrainConfigLite(
        hook_layer=12, margin=args.margin, lr=args.lr, weight_decay=1e-4,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        val_split=0.1, lambda_anchor=0.0, seed=args.seed,
    )
    save_json(
        {**dataclasses.asdict(train_cfg), "r": args.r, "init": args.init,
         "lambda_distill": args.lambda_distill, "teacher_scale": args.teacher_scale},
        out / "train_config.json",
    )

    # E5 teacher load
    e5_q_emb, e5_d_emb, e5_qid_idx, e5_did_idx = _load_e5_teacher(args.dataset)

    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
    model.clear_hooks()
    metric = BilinearMetric(dim=128, r=args.r, init=args.init).to(device)
    logger.info(
        "metric: dim=128 r=%d params=%d (init=%s)",
        args.r, metric.num_params(), args.init,
    )

    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info("train: corpus=%d queries=%d", len(train_corpus), len(train_queries))

    t0 = time.time()
    with torch.no_grad():
        train_dids, td_emb, td_mask = encode_corpus(
            model, train_corpus, device, batch_size=args.doc_batch,
        )
        train_topk = _bilinear_score_queries(
            model, metric, train_queries, train_dids, td_emb, td_mask, device,
            query_batch=args.query_batch, doc_chunk=args.doc_chunk, top_k=HN_POOL,
        )
    logger.info("train baseline pass in %.1fs", time.time() - t0)
    del td_emb, td_mask

    train_runs = {q: [d for d, _ in lst] for q, lst in train_topk.items()}
    triplets_raw = mine_triplets(train_runs, train_qrels, n_hns_per_q=N_HNS_PER_Q, pool=HN_POOL)

    # Filter triplets to those with E5 teacher coverage
    triplets = [
        t for t in triplets_raw
        if t[0] in e5_qid_idx and t[1] in e5_did_idx and t[2] in e5_did_idx
    ]
    skipped = len(triplets_raw) - len(triplets)
    logger.info(
        "mined %d triplets, %d kept after E5-teacher coverage filter (%d skipped)",
        len(triplets_raw), len(triplets), skipped,
    )
    if skipped > 0:
        logger.warning(
            "%d triplets dropped — E5 teacher coverage gap (확인 필요)", skipped,
        )

    history = train_bilinear_metric_distill(
        model=model, metric=metric, train_triplets=triplets,
        queries=train_queries, corpus=train_corpus, qrels=train_qrels,
        device=device, cfg=train_cfg,
        e5_qid_to_idx=e5_qid_idx, e5_did_to_idx=e5_did_idx,
        e5_q_emb=e5_q_emb, e5_d_emb=e5_d_emb,
        lambda_distill=args.lambda_distill,
        teacher_scale=args.teacher_scale,
        val_eval_kwargs={"doc_batch": args.doc_batch,
                         "query_batch": args.query_batch,
                         "doc_chunk": args.doc_chunk},
    )

    diagnostics = _m_diagnostics(metric)
    save_json(diagnostics, out / "M_stats.json")
    save_json(dataclasses.asdict(history), out / "train_history.json")
    torch.save(metric.state_dict(), out / "module_final.pt")
    logger.info(
        "train done: ‖U‖=%.4f ‖V‖=%.4f ‖UV^T‖_F=%.4f UV_sing_top3=%s",
        diagnostics["U_norm"], diagnostics["V_norm"], diagnostics["UV_norm_fro"],
        [f"{x:.3f}" for x in diagnostics["UV_singular_values"][:3]],
    )

    del train_corpus, train_queries, train_qrels, triplets, triplets_raw
    del train_topk, train_runs

    # ----------------------------------------------------------------- test
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
        "vs_06_k_sweep_k4",
        PROJECT_ROOT / "outputs" / "06_k_sweep" / args.dataset
        / f"seed_{args.seed}" / "k_4" / "metrics_per_query.json",
    )
    _record(
        "vs_08_r8",
        PROJECT_ROOT / "outputs" / "08_bilinear_M_minimal" / args.dataset
        / f"seed_{args.seed}" / "r_8" / "metrics_per_query.json",
    )

    logger.info("=== aggregate (all slice, r=%d λ=%.2f) ===",
                args.r, args.lambda_distill)
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
