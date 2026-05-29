"""04_per_token_gate — direction $v$ + per-token gate $g(h)$ at layer 12.

08 (scalar gate) 의 후속. SteeringModule = PerTokenGatedSteeringModule.
다른 학습 절차 / 평가 / artifact 는 02 / 08 와 동일 구조.

사용:
    .venv/bin/python experiments/04_per_token_gate/run.py --dataset scifact --seed 42
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

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
from src.lsr import PerTokenGatedSteeringModule  # noqa: E402
from src.mean_diff import HOOK_LAYER, compute_v  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.train import TrainConfigLite, train_steering  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("04_per_token_gate")

TRAIN_AVAILABLE = ("scifact", "nfcorpus", "fiqa")
N_HNS_PER_Q = 10
HN_POOL = 100


def _paired_ci_vs(
    per_q, ref_per_q, qids: set, metric="ndcg_cut_10",
    n_iter=10000, ci=0.95, seed=42,
):
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


@torch.no_grad()
def _gate_distribution_stats(
    model: ColBERTv2,
    steering: PerTokenGatedSteeringModule,
    corpus: Dict[str, dict],
    dids,
    device: torch.device,
    n_samples: int = 256,
    batch_size: int = 32,
) -> dict:
    """Sample a few docs, encode, capture per-token gate values."""
    if len(dids) > n_samples:
        idx = np.random.default_rng(42).choice(len(dids), n_samples, replace=False)
        sample_dids = [dids[i] for i in idx]
    else:
        sample_dids = dids
    gates = []

    # Temporarily replace hook with one that captures gate values.
    captured = []

    def capture(h: torch.Tensor) -> torch.Tensor:
        g = steering.gate(h)  # (B, T, 1)
        captured.append(g.squeeze(-1).detach().to("cpu"))
        return steering.forward(h)

    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, capture)

    for start in range(0, len(sample_dids), batch_size):
        texts = [doc_text(corpus[d]) for d in sample_dids[start:start + batch_size]]
        captured.clear()
        _emb, score_mask = model.encode_docs(texts, device=device)
        if captured:
            g = captured[0]  # (B, T)
            mask = score_mask.to("cpu").bool()
            valid = g[mask]
            gates.append(valid.numpy())

    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())
    if not gates:
        return {}
    all_g = np.concatenate(gates)
    return {
        "n_tokens_sampled": int(all_g.size),
        "mean": float(all_g.mean()),
        "std": float(all_g.std()),
        "min": float(all_g.min()),
        "p25": float(np.percentile(all_g, 25)),
        "p50": float(np.percentile(all_g, 50)),
        "p75": float(np.percentile(all_g, 75)),
        "max": float(all_g.max()),
        "frac_above_0.5": float((all_g > 0.5).mean()),
        "frac_above_0.1": float((all_g > 0.1).mean()),
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
    p.add_argument("--gate-bias-init", type=float, default=-3.0)
    args = p.parse_args()

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("dataset=%s seed=%d device=%s gate_bias_init=%.1f",
                args.dataset, args.seed, device, args.gate_bias_init)

    out = artifact_dir(exp_name="04_per_token_gate", dataset=args.dataset, seed=args.seed)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    train_cfg = TrainConfigLite(
        hook_layer=HOOK_LAYER, margin=args.margin, lr=args.lr,
        weight_decay=1e-4, batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, val_split=0.1,
        lambda_anchor=args.lambda_anchor, seed=args.seed,
    )
    save_json(dataclasses.asdict(train_cfg), out / "train_config.json")

    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
    steering = PerTokenGatedSteeringModule(
        hidden_dim=768, init="zero", gate_bias_init=args.gate_bias_init,
    ).to(device)
    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info("train: corpus=%d queries=%d", len(train_corpus), len(train_queries))
    assert steering.v.detach().norm().item() == 0.0

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

    v_mean_diff, _ = compute_v(model, train_corpus, triplets, device, batch_size=args.doc_batch)
    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    history = train_steering(
        model=model, steering=steering, train_triplets=triplets,
        queries=train_queries, corpus=train_corpus, qrels=train_qrels,
        device=device, cfg=train_cfg,
        val_eval_kwargs={"doc_batch": args.doc_batch,
                         "query_batch": args.query_batch,
                         "doc_chunk": args.doc_chunk},
    )

    v_norm_final = float(steering.v.detach().norm().item())
    W_norm_final = float(steering.gate_weight.detach().norm().item())
    b_final = float(steering.gate_bias.detach().item())
    save_json(
        {
            **dataclasses.asdict(history),
            "v_norm_final": v_norm_final,
            "W_norm_final": W_norm_final,
            "gate_bias_final": b_final,
        },
        out / "train_history.json",
    )
    torch.save(steering.state_dict(), out / "module_final.pt")

    v_learned = steering.v.detach().to("cpu")
    cos = torch.nn.functional.cosine_similarity(
        v_learned.flatten().unsqueeze(0),
        v_mean_diff.flatten().to(torch.float32).unsqueeze(0),
    ).item()
    save_json(
        {
            "v_learned_norm": v_norm_final,
            "v_mean_diff_norm": float(v_mean_diff.norm().item()),
            "gate_W_norm_final": W_norm_final,
            "gate_bias_final": b_final,
            "cosine_v_with_mean_diff": float(cos),
        },
        out / "cosine_with_mean_diff.json",
    )
    logger.info(
        "train done: ‖v‖=%.4f ‖W‖=%.4f b=%.4f cos=%.4f",
        v_norm_final, W_norm_final, b_final, cos,
    )

    del train_corpus, train_queries, train_qrels, triplets, train_topk, train_runs

    test_corpus, test_queries, test_qrels = load_beir(args.dataset, split="test")
    logger.info("test: corpus=%d queries=%d", len(test_corpus), len(test_queries))
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

    gate_stats = _gate_distribution_stats(model, steering, test_corpus, test_dids, device)
    save_json(gate_stats, out / "gate_distribution.json")
    logger.info(
        "gate dist on test: mean=%.3f std=%.3f frac>0.5=%.3f frac>0.1=%.3f",
        gate_stats.get("mean", float("nan")),
        gate_stats.get("std", float("nan")),
        gate_stats.get("frac_above_0.5", float("nan")),
        gate_stats.get("frac_above_0.1", float("nan")),
    )

    baseline_per_q_path = (
        PROJECT_ROOT / "outputs" / "00_baseline" / args.dataset
        / f"seed_{args.seed}" / "metrics_per_query.json"
    )
    baseline_runs_path = baseline_per_q_path.parent / "runs.json"
    baseline_per_q = load_json(baseline_per_q_path) if baseline_per_q_path.exists() else None
    baseline_runs = load_json(baseline_runs_path) if baseline_runs_path.exists() else None
    conf = confused_slice(baseline_runs, test_qrels, k=1) if baseline_runs else set()

    deltas = {}
    if baseline_per_q is not None:
        deltas["vs_baseline"] = {
            "all": _paired_ci_vs(per_q, baseline_per_q, set(per_q.keys()), seed=args.seed),
            "confused": _paired_ci_vs(per_q, baseline_per_q, conf, seed=args.seed) if conf else {},
        }
        save_json(deltas["vs_baseline"], out / "delta_vs_baseline.json")

    alpha10 = (
        PROJECT_ROOT / "outputs" / "01b_mean_diff_scaled" / args.dataset
        / f"seed_{args.seed}" / "alpha_10p0" / "metrics_per_query.json"
    )
    if alpha10.exists():
        deltas["vs_mean_diff_alpha10"] = {
            "all": _paired_ci_vs(per_q, load_json(alpha10), set(per_q.keys()), seed=args.seed),
            "confused": _paired_ci_vs(per_q, load_json(alpha10), conf, seed=args.seed) if conf else {},
        }
        save_json(deltas["vs_mean_diff_alpha10"], out / "delta_vs_mean_diff_alpha10.json")

    o2 = (PROJECT_ROOT / "outputs" / "02_final_layer_vector" / args.dataset
          / f"seed_{args.seed}" / "metrics_per_query.json")
    if o2.exists():
        deltas["vs_02_learned"] = {
            "all": _paired_ci_vs(per_q, load_json(o2), set(per_q.keys()), seed=args.seed),
            "confused": _paired_ci_vs(per_q, load_json(o2), conf, seed=args.seed) if conf else {},
        }
        save_json(deltas["vs_02_learned"], out / "delta_vs_02_learned.json")

    o8 = (PROJECT_ROOT / "outputs" / "03_scalar_gate" / args.dataset
          / f"seed_{args.seed}" / "metrics_per_query.json")
    if o8.exists():
        deltas["vs_03_scalar_gate"] = {
            "all": _paired_ci_vs(per_q, load_json(o8), set(per_q.keys()), seed=args.seed),
            "confused": _paired_ci_vs(per_q, load_json(o8), conf, seed=args.seed) if conf else {},
        }
        save_json(deltas["vs_03_scalar_gate"], out / "delta_vs_03_scalar_gate.json")

    logger.info("=== aggregate (all slice) ===")
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
