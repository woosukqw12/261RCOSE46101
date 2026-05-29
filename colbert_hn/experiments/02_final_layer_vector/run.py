"""02_final_layer_vector — single learned direction at layer 12.

single-direction 단계 의 첫 학습 실험. 절차:
  1. Train split 의 baseline ColBERT retrieval (in-memory) → HN-pos triplets.
  2. SteeringModule (zero-init v ∈ ℝ^768) 를 layer 12 에 hook 등록.
  3. Pairwise margin loss 로 v 학습 (λ_anc = 0, DESIGN.md §11 deviation).
  4. Best-val state 복원, test 시점 재평가.
  5. Baseline + 01b α=10 anchor 대비 paired bootstrap CI 보고.
  6. cos(v_learned, v_mean_diff) 계산 (H5 qualitative).

사용:
    .venv/bin/python experiments/02_final_layer_vector/run.py --dataset scifact --seed 42
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Dict

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
from src.lsr import SteeringModule  # noqa: E402
from src.mean_diff import HOOK_LAYER, compute_v  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.train import TrainConfigLite, train_steering  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("02_final_layer_vector")

TRAIN_AVAILABLE = ("scifact", "nfcorpus", "fiqa")
N_HNS_PER_Q = 10
HN_POOL = 100


def _paired_ci_vs(
    per_q: Dict[str, dict],
    ref_per_q: Dict[str, dict],
    qids: set,
    metric: str = "ndcg_cut_10",
    n_iter: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    ours, base, q = align_per_query(
        {q: v for q, v in per_q.items() if q in qids},
        {q: v for q, v in ref_per_q.items() if q in qids},
        metric=metric,
    )
    if len(q) == 0:
        return {"n": 0, "skipped": "empty"}
    mean, lo, hi = paired_bootstrap_ci(ours, base, n_iter=n_iter, ci=ci, seed=seed)
    return {
        "n": len(q),
        "mean_delta_ndcg10": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "positive": lo > 0,
        "negative": hi < 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=TRAIN_AVAILABLE)
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument(
        "--unfreeze-encoder", action="store_true",
        help=("Encoder (bert + projection) 도 학습. Vanilla ColBERT fine-tune "
              "baseline 으로의 robustness check. artifact 는 별도 subdir 'unfrozen/' 에 저장."),
    )
    p.add_argument(
        "--encoder-lr", type=float, default=5e-5,
        help="Unfrozen encoder 의 별도 LR (steering 의 cfg.lr 와 다름). BERT finetune typical.",
    )
    p.add_argument(
        "--no-steering", action="store_true",
        help=("SteeringModule 의 v 를 0 으로 frozen (no-grad, no-op hook). Pure "
              "encoder finetune baseline 용 — 'v=0 hook 이 학습 신호 추가했냐' 의 "
              "reviewer 공격 회피. artifact subdir 에 '_no_steering' 접미사."),
    )
    args = p.parse_args()

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(
        "dataset=%s seed=%d device=%s unfreeze_encoder=%s",
        args.dataset, args.seed, device, args.unfreeze_encoder,
    )

    out_root = artifact_dir(exp_name="02_final_layer_vector", dataset=args.dataset, seed=args.seed)
    if args.unfreeze_encoder:
        sub = "unfrozen_no_steering" if args.no_steering else "unfrozen"
        out = out_root / sub
    else:
        out = out_root
    out.mkdir(parents=True, exist_ok=True)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    train_cfg = TrainConfigLite(
        hook_layer=HOOK_LAYER,
        margin=args.margin,
        lr=args.lr,
        weight_decay=1e-4,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        val_split=0.1,
        lambda_anchor=args.lambda_anchor,
        seed=args.seed,
    )
    save_json(
        {**dataclasses.asdict(train_cfg),
         "unfreeze_encoder": args.unfreeze_encoder,
         "encoder_lr": args.encoder_lr if args.unfreeze_encoder else None},
        out / "train_config.json",
    )

    # -------------------------------------------------- model + steering module
    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
    if args.unfreeze_encoder:
        for p_ in model.parameters():
            p_.requires_grad_(True)
        n_enc = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
        logger.info("unfrozen encoder params: %d (~%.1fM)", n_enc, n_enc / 1e6)
    steering = SteeringModule(hidden_dim=768, init="zero").to(device)
    if args.no_steering:
        for sp in steering.parameters():
            sp.requires_grad_(False)
        logger.info("steering frozen (v=0 no-grad) — pure encoder finetune baseline")
    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    # ------------------------------------------------------ train-side mining
    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info(
        "train: corpus=%d queries=%d qrels=%d",
        len(train_corpus), len(train_queries), len(train_qrels),
    )

    # Important: when mining HNs, we want the *baseline* (no-steer) ranking.
    # Temporarily disable the hook by setting v to zero -> identical to baseline.
    # We rely on the zero-init: at this point v == 0 so the hook is no-op.
    assert steering.v.detach().norm().item() == 0.0, "expected zero init for mining"

    t0 = time.time()
    with torch.no_grad():
        train_dids, td_emb, td_mask = encode_corpus(
            model, train_corpus, device, batch_size=args.doc_batch,
        )
        train_topk = score_queries(
            model, train_queries, train_dids, td_emb, td_mask, device,
            query_batch=args.query_batch, doc_chunk=args.doc_chunk, top_k=HN_POOL,
        )
    logger.info("train baseline pass in %.1fs", time.time() - t0)
    del td_emb, td_mask

    train_runs = {q: [d for d, _ in lst] for q, lst in train_topk.items()}
    triplets = mine_triplets(train_runs, train_qrels, n_hns_per_q=N_HNS_PER_Q, pool=HN_POOL)
    logger.info("mined %d triplets", len(triplets))

    # Also compute v_mean_diff for H5 qualitative cosine analysis.
    v_mean_diff, mean_diff_stats = compute_v(
        model, train_corpus, triplets, device, batch_size=args.doc_batch,
    )
    save_json(mean_diff_stats, out / "mean_diff_stats.json")
    logger.info("v_mean_diff norm=%.4f (for cosine analysis)", mean_diff_stats["v_norm"])

    # Re-register the steering hook (compute_v cleared hooks internally)
    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    # ------------------------------------------------------------------ train
    extra_groups = None
    if args.unfreeze_encoder:
        encoder_params = [p for p in model.parameters() if p.requires_grad]
        extra_groups = [{"params": encoder_params,
                         "lr": args.encoder_lr, "weight_decay": 1e-4}]
        logger.info(
            "extra optimizer group: %d encoder params @ lr=%.1e",
            sum(p.numel() for p in encoder_params), args.encoder_lr,
        )

    history = train_steering(
        model=model,
        steering=steering,
        train_triplets=triplets,
        queries=train_queries,
        corpus=train_corpus,
        qrels=train_qrels,
        device=device,
        cfg=train_cfg,
        extra_param_groups=extra_groups,
        train_encoder=args.unfreeze_encoder,
        val_eval_kwargs={
            "doc_batch": args.doc_batch,
            "query_batch": args.query_batch,
            "doc_chunk": args.doc_chunk,
        },
    )
    save_json(dataclasses.asdict(history), out / "train_history.json")
    torch.save(steering.state_dict(), out / "v_final.pt")
    v_learned = steering.v.detach().to("cpu")
    logger.info(
        "training complete. ‖v_learned‖=%.4f (mean-diff ‖v‖=%.4f)",
        v_learned.norm().item(), v_mean_diff.norm().item(),
    )

    # ---------------------------------------------- H5 qualitative: cosine
    cos = torch.nn.functional.cosine_similarity(
        v_learned.flatten().unsqueeze(0),
        v_mean_diff.flatten().to(torch.float32).unsqueeze(0),
    ).item()
    save_json(
        {
            "v_learned_norm": float(v_learned.norm().item()),
            "v_mean_diff_norm": float(v_mean_diff.norm().item()),
            "cosine_similarity": float(cos),
            "interpretation": (
                "high (>0.9): learned 가 magnitude 만 조정; "
                "low (<0.5): qualitatively 다른 방향 학습"
            ),
        },
        out / "cosine_with_mean_diff.json",
    )
    logger.info("cos(v_learned, v_mean_diff) = %.4f", cos)

    del train_corpus, train_queries, train_qrels, triplets, train_topk, train_runs

    # ---------------------------------------------------- test side: encode + eval
    test_corpus, test_queries, test_qrels = load_beir(args.dataset, split="test")
    logger.info(
        "test: corpus=%d queries=%d qrels=%d",
        len(test_corpus), len(test_queries), len(test_qrels),
    )

    steering.eval()
    t0 = time.time()
    with torch.no_grad():
        test_dids, d_emb, d_mask = encode_corpus(
            model, test_corpus, device, batch_size=args.doc_batch,
        )
        topk = score_queries(
            model, test_queries, test_dids, d_emb, d_mask, device,
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

    # -------------------------------------------------- paired bootstrap CIs
    baseline_per_q_path = (
        PROJECT_ROOT / "outputs" / "00_baseline" / args.dataset
        / f"seed_{args.seed}" / "metrics_per_query.json"
    )
    baseline_runs_path = baseline_per_q_path.parent / "runs.json"
    baseline_per_q = load_json(baseline_per_q_path) if baseline_per_q_path.exists() else None
    baseline_runs = load_json(baseline_runs_path) if baseline_runs_path.exists() else None
    conf = confused_slice(baseline_runs, test_qrels, k=1) if baseline_runs else set()

    deltas: Dict[str, Dict[str, dict]] = {}
    if baseline_per_q is not None:
        deltas["vs_baseline"] = {
            "all": _paired_ci_vs(per_q, baseline_per_q, set(per_q.keys()), seed=args.seed),
            "confused": _paired_ci_vs(per_q, baseline_per_q, conf, seed=args.seed) if conf else {},
        }
    save_json(deltas.get("vs_baseline", {}), out / "delta_vs_baseline.json")

    # Sharpened anchor: 01b α=10 (if available, SciFact 만 현재 보유)
    alpha10_per_q_path = (
        PROJECT_ROOT / "outputs" / "01b_mean_diff_scaled" / args.dataset
        / f"seed_{args.seed}" / "alpha_10p0" / "metrics_per_query.json"
    )
    if alpha10_per_q_path.exists():
        alpha10_per_q = load_json(alpha10_per_q_path)
        deltas["vs_mean_diff_alpha10"] = {
            "all": _paired_ci_vs(per_q, alpha10_per_q, set(per_q.keys()), seed=args.seed),
            "confused": (
                _paired_ci_vs(per_q, alpha10_per_q, conf, seed=args.seed) if conf else {}
            ),
        }
        save_json(
            deltas["vs_mean_diff_alpha10"], out / "delta_vs_mean_diff_alpha10.json"
        )
    else:
        logger.warning("no 01b alpha=10 result at %s — sharpened anchor skipped", alpha10_per_q_path)

    # ----------------------------------------------------------------- log
    logger.info("=== aggregate (all slice) ===")
    for k in sorted(agg["all"]):
        logger.info("  %-20s %.4f", k, agg["all"][k])
    logger.info(
        "confused: %d / %d (%.1f%%)",
        agg["_meta"]["n_confused"], agg["_meta"]["n_queries"],
        100 * agg["_meta"]["frac_confused"],
    )
    for anchor_name, dd in deltas.items():
        logger.info("=== Δ vs %s ===", anchor_name)
        for slice_name, dr in dd.items():
            if "mean_delta_ndcg10" in dr:
                logger.info(
                    "  %-9s Δ=%+.4f [%+.4f,%+.4f] (n=%d)%s",
                    slice_name, dr["mean_delta_ndcg10"],
                    dr["ci_lo"], dr["ci_hi"], dr["n"],
                    " ✓ positive" if dr.get("positive")
                    else " ✗ negative" if dr.get("negative")
                    else "",
                )
    logger.info("artifacts → %s", out)


if __name__ == "__main__":
    main()
