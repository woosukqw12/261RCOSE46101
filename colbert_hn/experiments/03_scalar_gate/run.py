"""03_scalar_gate — direction $v$ + scalar gate $g = \\sigma(b)$ at layer 12.

02 와의 차이는 단 하나: SteeringModule 대신 ScalarGatedSteeringModule 사용.
다른 학습 hyperparameter / 평가 절차 / artifact 형식은 02 와 동일.

사용:
    .venv/bin/python experiments/03_scalar_gate/run.py --dataset scifact --seed 42
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
from src.lsr import ScalarGatedSteeringModule  # noqa: E402
from src.mean_diff import HOOK_LAYER, compute_v  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.train import TrainConfigLite, train_steering  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("03_scalar_gate")

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
        "n": len(q), "mean_delta_ndcg10": mean, "ci_lo": lo, "ci_hi": hi,
        "positive": lo > 0, "negative": hi < 0,
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

    out = artifact_dir(exp_name="03_scalar_gate", dataset=args.dataset, seed=args.seed)
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
    steering = ScalarGatedSteeringModule(
        hidden_dim=768, init="zero", gate_bias_init=args.gate_bias_init,
    ).to(device)
    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    # ------------------------------------------------------ train-side mining
    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info("train: corpus=%d queries=%d", len(train_corpus), len(train_queries))

    # Initial state: v=0, gate=sigmoid(-3)≈0.047 → intervention ≈ 0 → ranking ≈ baseline.
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

    # mean-diff for cosine reference (uses its own hook → clears)
    v_mean_diff, _ = compute_v(
        model, train_corpus, triplets, device, batch_size=args.doc_batch,
    )
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

    # Augment history with gate-specific trace (recompute from steps)
    gate_final = float(steering.gate.detach().item())
    v_norm_final = float(steering.v.detach().norm().item())
    effective_mag = gate_final * v_norm_final
    save_json(
        {
            **dataclasses.asdict(history),
            "gate_final": gate_final,
            "v_norm_final": v_norm_final,
            "effective_magnitude": effective_mag,
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
            "gate_final": gate_final,
            "effective_magnitude_g_times_norm_v": effective_mag,
            "cosine_v_with_mean_diff": float(cos),
        },
        out / "cosine_with_mean_diff.json",
    )
    logger.info(
        "train done: g=%.4f ‖v‖=%.4f g·‖v‖=%.4f cos=%.4f",
        gate_final, v_norm_final, effective_mag, cos,
    )

    del train_corpus, train_queries, train_qrels, triplets, train_topk, train_runs

    # ---------------------------------------------------- test side: encode + eval
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

    # -------------------------------------------------- paired bootstrap CIs
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
        alpha10_per_q = load_json(alpha10)
        deltas["vs_mean_diff_alpha10"] = {
            "all": _paired_ci_vs(per_q, alpha10_per_q, set(per_q.keys()), seed=args.seed),
            "confused": _paired_ci_vs(per_q, alpha10_per_q, conf, seed=args.seed) if conf else {},
        }
        save_json(deltas["vs_mean_diff_alpha10"], out / "delta_vs_mean_diff_alpha10.json")

    o2 = (PROJECT_ROOT / "outputs" / "02_final_layer_vector" / args.dataset
          / f"seed_{args.seed}" / "metrics_per_query.json")
    if o2.exists():
        o2_per_q = load_json(o2)
        deltas["vs_02_learned"] = {
            "all": _paired_ci_vs(per_q, o2_per_q, set(per_q.keys()), seed=args.seed),
            "confused": _paired_ci_vs(per_q, o2_per_q, conf, seed=args.seed) if conf else {},
        }
        save_json(deltas["vs_02_learned"], out / "delta_vs_02_learned.json")

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
