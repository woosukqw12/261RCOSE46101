"""06_k_sweep — Multi-direction + softmax router 의 K sweep.

원래의 06_two_directions (K=2 단일 proof-of-concept) 를 K ∈ {2, 4, 8} sweep
으로 확장. *Translation family ceiling 의 multi-direction 차원에서의
robustness* 를 검정 — *informed direction subspace* 의 ceiling 이 K 와 무관
하게 유지되는가?

사용:
    .venv/bin/python experiments/06_k_sweep/run.py --dataset scifact --seed 42 --k 2
    .venv/bin/python experiments/06_k_sweep/run.py --dataset scifact --seed 42 --k 4
    .venv/bin/python experiments/06_k_sweep/run.py --dataset scifact --seed 42 --k 8

Artifact 구조:
    outputs/06_k_sweep/{dataset}/seed_{seed}/k_{K}/
        ├── config / env / train_config / module_final.pt / train_history.json
        ├── cosine_v_pairs.json (모든 pair-wise cos + cos vs mean-diff)
        ├── routing_stats.json
        ├── runs / runs_scored / metrics_per_query / metrics_aggregate.json
        └── delta_vs_{baseline, mean_diff_alpha10, 02_learned}.json
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
from src.lsr import MultiDirectionSteeringModule  # noqa: E402
from src.mean_diff import HOOK_LAYER, compute_v  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.train import TrainConfigLite, train_steering  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("06_k_sweep")

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


@torch.no_grad()
def _routing_distribution(
    model: ColBERTv2,
    steering: MultiDirectionSteeringModule,
    corpus: Dict[str, dict],
    dids,
    device: torch.device,
    n_samples: int = 256,
    batch_size: int = 32,
) -> dict:
    """Sample docs, capture per-token routing distributions, summarize.

    K-agnostic: works for any K. Returns per-component mean / std + entropy
    + saturation fraction + effective K (perplexity = exp(entropy)).
    """
    if len(dids) > n_samples:
        idx = np.random.default_rng(42).choice(len(dids), n_samples, replace=False)
        sample_dids = [dids[i] for i in idx]
    else:
        sample_dids = dids

    captured = []

    def capture(h: torch.Tensor) -> torch.Tensor:
        pi = steering.routing(h)
        captured.append(pi.detach().to("cpu"))
        return steering.forward(h)

    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, capture)

    pis = []
    for start in range(0, len(sample_dids), batch_size):
        texts = [doc_text(corpus[d]) for d in sample_dids[start:start + batch_size]]
        captured.clear()
        _emb, score_mask = model.encode_docs(texts, device=device)
        if captured:
            pi = captured[0]  # (B, T, K)
            mask = score_mask.to("cpu").bool()
            valid = pi[mask]  # (M, K) — only non-mask tokens
            pis.append(valid.numpy())

    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    if not pis:
        return {}
    all_pi = np.concatenate(pis)  # (M_total, K)
    entropy = (-all_pi * np.log(np.clip(all_pi, 1e-12, None))).sum(axis=-1)  # (M,)
    pi_max = all_pi.max(axis=-1)
    K = all_pi.shape[1]
    eff_K = float(np.exp(entropy.mean()))
    stats = {
        "n_tokens_sampled": int(all_pi.shape[0]),
        "K": K,
        "pi_mean_over_tokens": all_pi.mean(axis=0).tolist(),
        "pi_std_over_tokens": all_pi.std(axis=0).tolist(),
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        "max_entropy_uniform": float(np.log(K)),
        "effective_K_perplexity": eff_K,
        "pi_max_mean": float(pi_max.mean()),
        "frac_tokens_pi_max_above_0.5": float((pi_max > 0.5).mean()),
        "frac_tokens_pi_max_above_0.6": float((pi_max > 0.6).mean()),
        "frac_tokens_pi_max_above_0.8": float((pi_max > 0.8).mean()),
    }
    return stats


def _direction_diagnostics(
    steering: MultiDirectionSteeringModule,
    v_mean_diff: torch.Tensor,
) -> dict:
    """K-agnostic direction analysis. Returns norms + pairwise cosines + cos
    vs mean-diff."""
    vs = [steering.v.detach()[k].to("cpu").to(torch.float32) for k in range(steering.K)]
    v_md = v_mean_diff.to(torch.float32)
    norms = [float(v.norm().item()) for v in vs]
    pairwise = {}
    K = steering.K
    for i in range(K):
        for j in range(i + 1, K):
            c = torch.nn.functional.cosine_similarity(
                vs[i].unsqueeze(0), vs[j].unsqueeze(0)
            ).item()
            pairwise[f"cos_v{i}_v{j}"] = float(c)
    cos_v_md = [
        float(torch.nn.functional.cosine_similarity(
            v.unsqueeze(0), v_md.unsqueeze(0)
        ).item())
        for v in vs
    ]
    abs_pairs = [abs(v) for v in pairwise.values()] if pairwise else [0.0]
    return {
        "K": K,
        "v_norms": norms,
        "pairwise_cosines": pairwise,
        "mean_pairwise_abs_cosine": float(np.mean(abs_pairs)),
        "cos_v_k_vs_mean_diff_l12": cos_v_md,
        "max_cos_v_k_vs_mean_diff": float(np.max(np.abs(cos_v_md))),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=TRAIN_AVAILABLE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, required=True,
                   help="K — number of directions (2 / 4 / 8 etc.)")
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
        "--max-triplets", type=int, default=None,
        help=("Cap on mined triplets (deterministic random subsample). "
              "Dense-qrels datasets (e.g., NFCorpus) produce > 1M triplets; "
              "set to ~9000 to match SciFact scale."),
    )
    args = p.parse_args()

    K = args.k
    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("dataset=%s seed=%d device=%s K=%d", args.dataset, args.seed, device, K)

    base_out = artifact_dir(exp_name="06_k_sweep", dataset=args.dataset, seed=args.seed)
    out = base_out / f"k_{K}"
    out.mkdir(parents=True, exist_ok=True)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    train_cfg = TrainConfigLite(
        hook_layer=HOOK_LAYER, margin=args.margin, lr=args.lr,
        weight_decay=1e-4, batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, val_split=0.1,
        lambda_anchor=args.lambda_anchor, seed=args.seed,
    )
    save_json({**dataclasses.asdict(train_cfg), "K": K},
              out / "train_config.json")

    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
    steering = MultiDirectionSteeringModule(
        hidden_dim=768, n_directions=K, init="zero",
        router_bias_init=0.0, router_weight_std=0.02,
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
    if args.max_triplets is not None and len(triplets) > args.max_triplets:
        import random as _r
        _rng = _r.Random(args.seed)
        _rng.shuffle(triplets)
        triplets = triplets[: args.max_triplets]
        logger.info("subsampled to %d triplets (deterministic, seed=%d)",
                    len(triplets), args.seed)

    v_mean_diff_12, _ = compute_v(model, train_corpus, triplets, device, batch_size=args.doc_batch)
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

    diagnostics = _direction_diagnostics(steering, v_mean_diff_12)
    diagnostics["router_weight_norm"] = float(steering.router_weight.detach().norm().item())
    diagnostics["router_bias"] = steering.router_bias.detach().tolist()
    save_json(diagnostics, out / "cosine_v_pairs.json")
    save_json(dataclasses.asdict(history), out / "train_history.json")
    torch.save(steering.state_dict(), out / "module_final.pt")
    logger.info(
        "train done: K=%d ‖v_k‖=%s mean|cos(v_i,v_j)|=%.4f max|cos(v_k,v_md)|=%.4f",
        K,
        [f"{n:.2f}" for n in diagnostics["v_norms"]],
        diagnostics["mean_pairwise_abs_cosine"],
        diagnostics["max_cos_v_k_vs_mean_diff"],
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

    routing_stats = _routing_distribution(model, steering, test_corpus, test_dids, device)
    save_json(routing_stats, out / "routing_stats.json")
    logger.info(
        "routing on test: K=%d eff_K_perp=%.3f entropy=%.3f (max=%.3f) "
        "frac_pi_max>0.6=%.3f pi_max_mean=%.3f",
        routing_stats.get("K", K),
        routing_stats.get("effective_K_perplexity", float("nan")),
        routing_stats.get("entropy_mean", float("nan")),
        routing_stats.get("max_entropy_uniform", float("nan")),
        routing_stats.get("frac_tokens_pi_max_above_0.6", float("nan")),
        routing_stats.get("pi_max_mean", float("nan")),
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

    logger.info("=== aggregate (all slice, K=%d) ===", K)
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
