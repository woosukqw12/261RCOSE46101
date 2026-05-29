"""05_five_layers — learned direction $v_\\ell$ at 5 layers (multi-layer LSR).

02 의 single-layer 확장. SteeringModule → MultiLayerSteeringModule.
나머지 학습 / 평가 절차는 02 와 동일 구조.

사용:
    .venv/bin/python experiments/05_five_layers/run.py --dataset scifact --seed 42
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
from src.lsr import MultiLayerSteeringModule  # noqa: E402
from src.mean_diff import compute_v  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.train import TrainConfigLite, train_steering  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("05_five_layers")

TRAIN_AVAILABLE = ("scifact", "nfcorpus", "fiqa")
LAYERS = (0, 3, 6, 9, 12)
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
    args = p.parse_args()

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("dataset=%s seed=%d device=%s layers=%s", args.dataset, args.seed, device, LAYERS)

    out = artifact_dir(exp_name="05_five_layers", dataset=args.dataset, seed=args.seed)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    train_cfg = TrainConfigLite(
        hook_layer=LAYERS[-1],  # for legacy compatibility — actual hooks are multi-layer
        margin=args.margin, lr=args.lr, weight_decay=1e-4,
        batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, val_split=0.1,
        lambda_anchor=args.lambda_anchor, seed=args.seed,
    )
    save_json({**dataclasses.asdict(train_cfg), "layers": list(LAYERS)},
              out / "train_config.json")

    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
    steering = MultiLayerSteeringModule(
        hidden_dim=768, layers=LAYERS, init="zero",
    ).to(device)
    steering.register_all(model)

    # ------------------------------------------------------ train-side mining
    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info("train: corpus=%d queries=%d", len(train_corpus), len(train_queries))

    # Initial: all v_l = 0 → all 5 hooks are no-op → ranking = baseline.
    for layer in LAYERS:
        assert steering.v(layer).detach().norm().item() == 0.0

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

    # mean-diff at layer 12 only (compute_v uses fixed HOOK_LAYER=12) for cosine reference
    v_mean_diff_12, _ = compute_v(
        model, train_corpus, triplets, device, batch_size=args.doc_batch,
    )
    steering.register_all(model)  # re-register after compute_v cleared hooks

    history = train_steering(
        model=model, steering=steering, train_triplets=triplets,
        queries=train_queries, corpus=train_corpus, qrels=train_qrels,
        device=device, cfg=train_cfg,
        val_eval_kwargs={"doc_batch": args.doc_batch,
                         "query_batch": args.query_batch,
                         "doc_chunk": args.doc_chunk},
    )

    # ---------------------------------------------------- per-layer diagnostics
    layer_norms: Dict[int, float] = {}
    layer_cos: Dict[int, float] = {}
    for layer in LAYERS:
        v_l = steering.v(layer).detach().to("cpu").to(torch.float32)
        layer_norms[layer] = float(v_l.norm().item())
        cos = torch.nn.functional.cosine_similarity(
            v_l.flatten().unsqueeze(0),
            v_mean_diff_12.flatten().to(torch.float32).unsqueeze(0),
        ).item()
        layer_cos[layer] = float(cos)
    total_v_norm = float(
        torch.sqrt(sum(steering.v(l).detach() ** 2 for l in LAYERS).sum()).item()
    )

    save_json(
        {**dataclasses.asdict(history), "layer_norms": layer_norms,
         "total_v_norm": total_v_norm},
        out / "train_history.json",
    )
    save_json(layer_norms, out / "layer_norms.json")
    save_json(
        {
            "v_mean_diff_l12_norm": float(v_mean_diff_12.norm().item()),
            "cosine_per_layer_vs_mean_diff_l12": layer_cos,
            "interpretation": (
                "cos(v_ℓ, v_mean_diff_l12) per layer. high (>0.9): "
                "layer ℓ 가 mean-diff 와 같은 방향 학습. low: 다른 정보."
            ),
        },
        out / "cosine_with_mean_diff.json",
    )
    torch.save(steering.state_dict(), out / "module_final.pt")
    logger.info("train done. total ‖v‖=%.4f", total_v_norm)
    for layer in LAYERS:
        logger.info(
            "  layer %2d: ‖v_ℓ‖=%.4f  cos(v_ℓ, v_meandiff_l12)=%.4f",
            layer, layer_norms[layer], layer_cos[layer],
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
        "vs_03_scalar_gate",
        PROJECT_ROOT / "outputs" / "03_scalar_gate" / args.dataset
        / f"seed_{args.seed}" / "metrics_per_query.json",
    )
    _record(
        "vs_04_per_token_gate",
        PROJECT_ROOT / "outputs" / "04_per_token_gate" / args.dataset
        / f"seed_{args.seed}" / "metrics_per_query.json",
    )

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
