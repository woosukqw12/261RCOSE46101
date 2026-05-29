"""07_random_direction_scaled — translation-trap falsification.

Random Gaussian unit vector × α 를 layer 12 에 적용. 01b 의 α=10 (mean-diff)
와 paired bootstrap 비교. 학습 무필요.

사용:
    .venv/bin/python experiments/07_random_direction_scaled/run.py \\
        --dataset scifact --seed 42 --alpha 10
"""
from __future__ import annotations

import argparse
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
from src.mean_diff import HOOK_LAYER  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("07_random_direction_scaled")


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=("scifact", "nfcorpus", "fiqa"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--device", default=None)
    p.add_argument("--doc-batch", type=int, default=64)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--doc-chunk", type=int, default=512)
    args = p.parse_args()

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("dataset=%s seed=%d alpha=%.1f device=%s",
                args.dataset, args.seed, args.alpha, device)

    out = artifact_dir(exp_name="07_random_direction_scaled", dataset=args.dataset, seed=args.seed)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    # ------------------------------- random direction (seed-fixed, unit-normalized)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    v_raw = torch.randn(768, generator=generator)
    v_unit = v_raw / v_raw.norm()
    v_applied = (args.alpha * v_unit).to(device)
    torch.save(
        {"v_raw": v_raw, "v_unit": v_unit, "alpha": args.alpha,
         "v_applied_norm": float(v_applied.norm().item())},
        out / "v_random.pt",
    )
    logger.info(
        "random v generated: ‖v_raw‖=%.4f ‖v_unit‖=%.4f ‖α·v_unit‖=%.4f",
        v_raw.norm().item(), v_unit.norm().item(), v_applied.norm().item(),
    )

    # ------------------------------------------------------------------ model
    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)

    def steer(h: torch.Tensor) -> torch.Tensor:
        return h - v_applied.to(h.dtype)

    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steer)

    # ---------------------------------------------------- test side: encode + eval
    test_corpus, test_queries, test_qrels = load_beir(args.dataset, split="test")
    logger.info("test: corpus=%d queries=%d", len(test_corpus), len(test_queries))

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

    # ----------------------------------------------------- paired bootstrap CIs
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
        save_json(deltas["vs_baseline"], out / "delta_vs_baseline.json")

    # 01b α=10 — the KEY comparison for translation-trap falsification
    alpha10_path = (
        PROJECT_ROOT / "outputs" / "01b_mean_diff_scaled" / args.dataset
        / f"seed_{args.seed}" / "alpha_10p0" / "metrics_per_query.json"
    )
    if alpha10_path.exists():
        alpha10_per_q = load_json(alpha10_path)
        deltas["vs_mean_diff_alpha10"] = {
            "all": _paired_ci_vs(per_q, alpha10_per_q, set(per_q.keys()), seed=args.seed),
            "confused": _paired_ci_vs(per_q, alpha10_per_q, conf, seed=args.seed) if conf else {},
        }
        save_json(deltas["vs_mean_diff_alpha10"], out / "delta_vs_mean_diff_alpha10.json")

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
