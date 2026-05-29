"""12_fn_denoised_hn — *FN-denoised mined-HN* (캐비엇 1 disambiguator).

Phase 2b 와 *negative quality 만* 다름. mined HN 의 e5_margin ≤ 0 (likely FN) 제거 →
*hard 유지 + noise 제거* 의 깨끗한 mined-HN 으로 재학습. (나-1 noise) vs (나-2 difficulty)
의 결정적 disambiguator.

Implementation:
1. Load E5 train q emb (809) + E5 train doc emb (5183).
2. Mine triplets with baseline ColBERT (same as Phase 2b).
3. For each (q, pos, hn), compute e5_margin = cos(eq, ep) - cos(eq, eh).
4. Filter triplets with e5_margin <= threshold (default 0.0).
5. Train Phase 2b config on filtered triplets.

사용:
    .venv/bin/python experiments/12_fn_denoised_hn/run.py \\
        --dataset scifact --seed 42 \\
        --margin-threshold 0.0 \\
        --r 8 --alpha 8.0 --lora-lr 5e-5 --max-triplets 9190
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

from src.colbert_hook import ColBERTConfig, ColBERTv2  # noqa: E402
from src.configs import BASELINE  # noqa: E402
from src.data import load_beir  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_aggregate, compute_metrics_trec, encode_corpus, save_env, score_queries,
)
from src.hn_mining import mine_triplets  # noqa: E402
from src.lora import inject_lora_into_bert, lora_param_count  # noqa: E402
from src.lsr import SteeringModule  # noqa: E402
from src.mean_diff import HOOK_LAYER  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.train import TrainConfigLite, train_steering  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("12_fn_denoised_hn")

TRAIN_AVAILABLE = ("scifact", "nfcorpus")
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


def load_e5_train(dataset: str):
    """Load E5 train query + corpus doc embeddings."""
    e5_dir = PROJECT_ROOT / "data" / "e5_teacher"
    q_path = e5_dir / f"e5_train_q_emb_{dataset}.pt"
    d_path = e5_dir / f"e5_train_doc_emb_{dataset}.pt"
    if not q_path.exists():
        raise FileNotFoundError(
            f"E5 train query emb 부재: {q_path}. "
            f"실행: .venv/bin/python data/e5_teacher/extract_train_queries.py --dataset {dataset}"
        )
    if not d_path.exists():
        raise FileNotFoundError(
            f"E5 train doc emb 부재: {d_path}. "
            f"실행: .venv/bin/python data/e5_teacher/extract_train_docs.py --dataset {dataset}"
        )
    q = torch.load(q_path, weights_only=False)
    d = torch.load(d_path, weights_only=False)
    qid_to_idx = {qid: i for i, qid in enumerate(q["qids"])}
    did_to_idx = {did: i for i, did in enumerate(d["dids"])}
    return q["query_emb"], d["doc_emb"], qid_to_idx, did_to_idx


def filter_triplets_by_e5_margin(
    triplets, e5_q_emb, e5_d_emb, qid_to_idx, did_to_idx, threshold=0.0,
):
    """Remove triplets where e5_margin = cos(eq, epos) - cos(eq, ehn) <= threshold.

    threshold=0.0 → remove all where E5 thinks hn ≥ pos (likely FN).

    Returns (filtered_triplets, margins, stats_dict).
    """
    kept = []
    removed = []
    margins = []
    missing = 0
    for qid, pos_did, hn_did in triplets:
        if qid not in qid_to_idx or pos_did not in did_to_idx or hn_did not in did_to_idx:
            missing += 1
            continue
        eq = e5_q_emb[qid_to_idx[qid]].float()
        ep = e5_d_emb[did_to_idx[pos_did]].float()
        eh = e5_d_emb[did_to_idx[hn_did]].float()
        m = float((eq @ ep).item() - (eq @ eh).item())
        margins.append(m)
        if m > threshold:
            kept.append((qid, pos_did, hn_did))
        else:
            removed.append((qid, pos_did, hn_did, m))
    return kept, removed, margins, missing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=TRAIN_AVAILABLE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--init-std", type=float, default=0.02)
    p.add_argument("--lora-lr", type=float, default=5e-5)
    p.add_argument("--margin-threshold", type=float, default=0.0,
                   help="Remove triplets with e5_margin <= threshold (default 0.0)")
    p.add_argument("--device", default=None)
    p.add_argument("--doc-batch", type=int, default=64)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--doc-chunk", type=int, default=512)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-triplets", type=int, default=9190)
    p.add_argument("--early-stop-metric", default="all", choices=["all", "confused"])
    args = p.parse_args()

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("dataset=%s seed=%d margin_threshold=%.3f", args.dataset, args.seed, args.margin_threshold)

    tag = f"qv_r{args.r}_l12_thresh{args.margin_threshold:g}"
    base_out = artifact_dir(exp_name="12_fn_denoised_hn", dataset=args.dataset, seed=args.seed)
    out = base_out / tag
    out.mkdir(parents=True, exist_ok=True)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    train_cfg = TrainConfigLite(
        hook_layer=HOOK_LAYER, margin=args.margin,
        lr=1e-3, weight_decay=1e-4,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        val_split=0.1, lambda_anchor=0.0, seed=args.seed,
    )
    save_json({**dataclasses.asdict(train_cfg),
               "components": ["q", "v"], "r": args.r, "alpha": args.alpha,
               "init_std": args.init_std, "lora_lr": args.lora_lr,
               "margin_threshold": args.margin_threshold,
               "max_triplets": args.max_triplets},
              out / "train_config.json")

    # 1. Model + LoRA injection
    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
    expected = lora_param_count(["q", "v"], 12, hidden_dim=768, r=args.r)
    lora_params = inject_lora_into_bert(
        model.bert, target_components=["q", "v"], layers=None,
        r=args.r, alpha=args.alpha, init_std=args.init_std,
    )
    model.to(device)
    assert sum(p.numel() for p in lora_params) == expected
    logger.info("LoRA injected: %d params", expected)

    steering = SteeringModule(hidden_dim=768, init="zero").to(device)
    for p_ in steering.parameters():
        p_.requires_grad_(False)
    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    # 2. Load E5 teacher (queries + docs)
    e5_q_emb, e5_d_emb, qid_to_idx, did_to_idx = load_e5_train(args.dataset)
    logger.info("E5 teacher: %d queries, %d docs", e5_q_emb.shape[0], e5_d_emb.shape[0])

    # 3. Mine baseline ColBERT HN (same as Phase 2b)
    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info("train: corpus=%d queries=%d", len(train_corpus), len(train_queries))

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
    triplets_raw = mine_triplets(train_runs, train_qrels,
                                  n_hns_per_q=N_HNS_PER_Q, pool=HN_POOL)
    logger.info("mined %d raw triplets (before FN-denoising)", len(triplets_raw))

    # 4. Filter by e5_margin > threshold
    triplets_filt, removed, margins, missing = filter_triplets_by_e5_margin(
        triplets_raw, e5_q_emb, e5_d_emb, qid_to_idx, did_to_idx,
        threshold=args.margin_threshold,
    )
    import numpy as np
    margins = np.array(margins)
    logger.info(
        "FN-denoising: kept=%d, removed=%d (%.1f%% likely FN), missing=%d",
        len(triplets_filt), len(removed),
        100 * len(removed) / max(1, len(triplets_filt) + len(removed)),
        missing,
    )
    logger.info(
        "  margin stats: mean=%+.4f, median=%+.4f, min=%+.4f, max=%+.4f",
        margins.mean(), float(np.median(margins)), margins.min(), margins.max(),
    )
    save_json({
        "n_raw": len(triplets_raw),
        "n_kept": len(triplets_filt),
        "n_removed": len(removed),
        "n_missing_emb": missing,
        "fn_rate_per_e5": float(len(removed) / max(1, len(triplets_filt) + len(removed))),
        "margin_mean": float(margins.mean()),
        "margin_median": float(np.median(margins)),
        "margin_min": float(margins.min()),
        "margin_max": float(margins.max()),
        "margin_threshold": args.margin_threshold,
    }, out / "denoising_stats.json")

    # Subsample if needed (post-filter)
    if args.max_triplets and len(triplets_filt) > args.max_triplets:
        import random
        _rng = random.Random(args.seed)
        _rng.shuffle(triplets_filt)
        triplets_filt = triplets_filt[: args.max_triplets]
        logger.info("subsampled to %d filtered triplets (seed=%d)", len(triplets_filt), args.seed)

    # 5. Train Phase 2b config on filtered triplets
    extra_groups = [{"params": lora_params, "lr": args.lora_lr, "weight_decay": 1e-4}]
    history = train_steering(
        model=model, steering=steering, train_triplets=triplets_filt,
        queries=train_queries, corpus=train_corpus, qrels=train_qrels,
        device=device, cfg=train_cfg,
        extra_param_groups=extra_groups,
        train_encoder=True,
        early_stop_metric=args.early_stop_metric,
        val_eval_kwargs={"doc_batch": args.doc_batch,
                         "query_batch": args.query_batch,
                         "doc_chunk": args.doc_chunk},
    )

    A_norms, B_norms = [], []
    for i in range(0, len(lora_params), 2):
        A_norms.append(float(lora_params[i].detach().norm().item()))
        B_norms.append(float(lora_params[i + 1].detach().norm().item()))
    save_json({
        "components": ["q", "v"], "n_layers": 12, "rank": args.r,
        "alpha": args.alpha if args.alpha is not None else args.r,
        "total_params": expected,
        "A_norms_per_adapter": A_norms, "B_norms_per_adapter": B_norms,
        "A_norm_total": float(sum(a * a for a in A_norms) ** 0.5),
        "B_norm_total": float(sum(b * b for b in B_norms) ** 0.5),
    }, out / "lora_stats.json")
    save_json(dataclasses.asdict(history), out / "train_history.json")
    torch.save({
        "steering": steering.state_dict(),
        "lora": {f"adapter_{i}": p.detach().cpu() for i, p in enumerate(lora_params)},
    }, out / "module_final.pt")

    del train_corpus, train_queries, train_qrels, triplets_raw, triplets_filt
    del train_topk, train_runs

    # 6. Test eval
    test_corpus, test_queries, test_qrels = load_beir(args.dataset, split="test")
    logger.info("test: corpus=%d queries=%d", len(test_corpus), len(test_queries))
    model.eval(); steering.eval()
    with torch.no_grad():
        test_dids, d_emb, d_mask = encode_corpus(
            model, test_corpus, device, batch_size=args.doc_batch,
        )
        topk = score_queries(
            model, test_queries, test_dids, d_emb, d_mask, device,
            query_batch=args.query_batch, doc_chunk=args.doc_chunk,
            top_k=cfg.eval.retrieval_top_k,
        )
    runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
    runs_scored = {q: dict(lst) for q, lst in topk.items()}
    per_q = compute_metrics_trec(runs_scored, test_qrels, metrics_k=cfg.eval.metrics_k)
    agg = build_aggregate(per_q, runs_ranked, test_qrels, cfg.eval.confused_slice_def)
    save_json(runs_ranked, out / "runs.json")
    save_json(runs_scored, out / "runs_scored.json")
    save_json(per_q, out / "metrics_per_query.json")
    save_json(agg, out / "metrics_aggregate.json")

    baseline_per_q_path = (PROJECT_ROOT / "outputs" / "00_baseline" / args.dataset
                           / f"seed_{args.seed}" / "metrics_per_query.json")
    baseline_runs_path = baseline_per_q_path.parent / "runs.json"
    baseline_per_q = load_json(baseline_per_q_path)
    baseline_runs = load_json(baseline_runs_path)
    confused = confused_slice(baseline_runs, test_qrels, k=1)
    easy = set(per_q.keys()) - confused

    deltas = {
        "all": _paired_ci_vs(per_q, baseline_per_q, set(per_q.keys()), seed=args.seed),
        "confused": _paired_ci_vs(per_q, baseline_per_q, confused, seed=args.seed),
        "easy": _paired_ci_vs(per_q, baseline_per_q, easy, seed=args.seed),
    }
    save_json(deltas, out / "delta_vs_baseline.json")
    logger.info("=== Δ vs baseline (FN-denoised) ===")
    for sn, dr in deltas.items():
        if "mean_delta_ndcg10" in dr:
            logger.info("  %-9s n=%d Δ=%+.4f [%+.4f,%+.4f]%s",
                        sn, dr["n"], dr["mean_delta_ndcg10"], dr["ci_lo"], dr["ci_hi"],
                        " ✓ positive" if dr.get("positive")
                        else " ✗ negative" if dr.get("negative") else "")
    logger.info("artifacts → %s", out)


if __name__ == "__main__":
    main()
