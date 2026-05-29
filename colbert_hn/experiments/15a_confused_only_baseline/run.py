"""15a_confused_only_baseline — Phase 2b LoRA + *confused-only triplet filter*.

본 실험은 Exp 15 (Conditional LoRA) 의 *training-side ceiling diagnostic*.
*Perfect routing* 의 *training 시점 realization* — confused-query 의 triplet 만 학습.
Single seed × single config, *diagnostic only* (no sweep).

기존 10_lora_phi 와의 변경:
  - Triplet mining 후 confused_slice(train_runs, train_qrels, k=1) 로 confused
    query 의 triplet 만 keep
  - Artifact dir: outputs/15a_confused_only_baseline/{ds}/seed_{seed}/{tag}/
  - tag suffix '_confonly' 자동 부착

원본 motivation (참고):
Robustness audit (2026-05-23) 의 unfrozen 02 실험에서 ColBERT 의 *encoder 전체*
unfreeze (110M params) 가 Δ confused +0.252 의 5× lift 확인 → frozen-encoder
가 진짜 bottleneck. 본 실험은 **50K param budget 안** LoRA (Low-Rank Adaptation)
adapter 로 그 lift 의 *얼마나* 회복 가능한지 검정.

LoRA design (Phase 1 default, q+v r=1 all 12 layers = 36,864 params):
    각 attention 의 q, v Linear(768, 768) 에 LoRA rank-1 부착.
    Forward: y = W x + (α/r) B A x ,  A ∈ ℝ^{r×768}, B ∈ ℝ^{768×r}
    Init: A ~ N(0, 0.02²), B = 0 → ΔW = 0 at t=0 → 정확히 baseline.

사용 (Phase 1):
    .venv/bin/python experiments/10_lora_phi/run.py --dataset scifact --seed 42 \\
        --components q,v --r 1

Artifact: outputs/10_lora_phi/{ds}/seed_{seed}/{tag}/...
    tag = "qv_r1_l12"  (components, rank, n_layers)
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
    build_aggregate,
    compute_metrics_trec,
    encode_corpus,
    save_env,
    score_queries,
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

logger = get_logger("15a_confused_only_baseline")

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=TRAIN_AVAILABLE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--components", default="q,v",
                   help="Comma-separated attention components: subset of q,k,v,o")
    p.add_argument("--r", type=int, default=1, help="LoRA rank")
    p.add_argument("--alpha", type=float, default=None,
                   help="LoRA scaling (default = r ⇒ scaling=1)")
    p.add_argument("--init-std", type=float, default=0.02,
                   help="stdev of LoRA A init (B=0)")
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer indices (0..11), default all")
    p.add_argument("--lora-lr", type=float, default=5e-5,
                   help="LoRA param LR (typical BERT finetune scale)")
    p.add_argument("--device", default=None)
    p.add_argument("--doc-batch", type=int, default=64)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--doc-chunk", type=int, default=512)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-triplets", type=int, default=None,
                   help="Cap on triplets (subsample) for dense-qrels datasets")
    p.add_argument(
        "--early-stop-metric", default="confused", choices=["all", "confused"],
        help=("Val NDCG@10 slice for early-stop / best-state. 'all' = "
              "anchor-preservation focus (post-hoc cherry-picking 회피 위해 "
              "pre-commit). Default 'confused' = 기존 02-09 의 historical "
              "behavior."),
    )
    # Mediation 1: warmup + grad-clip (optimization-root disentangling)
    p.add_argument("--warmup-frac", type=float, default=0.0,
                   help="Linear warmup over first warmup_frac × total_steps (0 = disabled)")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="Max grad norm for LoRA + steering params (0 = no clip)")
    # Mediation 1b: in-batch negative (supervision-root disentangling)
    p.add_argument("--in-batch-neg", action="store_true",
                   help="Replace mined HN with in-batch negative (다른 query 의 positive)")
    p.add_argument("--tag-suffix", default="",
                   help="Suffix appended to artifact tag (e.g. 'm1' for mediation 1)")
    args = p.parse_args()

    components = [c.strip() for c in args.components.split(",") if c.strip()]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info(
        "dataset=%s seed=%d device=%s components=%s r=%d layers=%s",
        args.dataset, args.seed, device, components, args.r, layers or "all(12)",
    )

    tag = f"{''.join(components)}_r{args.r}_l{len(layers) if layers else 12}_confonly"
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
    base_out = artifact_dir(exp_name="15a_confused_only_baseline", dataset=args.dataset, seed=args.seed)
    out = base_out / tag
    out.mkdir(parents=True, exist_ok=True)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    train_cfg = TrainConfigLite(
        hook_layer=HOOK_LAYER, margin=args.margin,
        lr=1e-3,  # steering LR (steering is frozen no-op, this doesn't matter)
        weight_decay=1e-4,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        val_split=0.1, lambda_anchor=0.0, seed=args.seed,
        warmup_frac=args.warmup_frac, grad_clip_max_norm=args.grad_clip,
        in_batch_neg=args.in_batch_neg,
    )
    save_json(
        {**dataclasses.asdict(train_cfg),
         "components": components, "r": args.r, "alpha": args.alpha,
         "init_std": args.init_std,
         "layers": layers if layers else list(range(12)),
         "lora_lr": args.lora_lr,
         "max_triplets": args.max_triplets},
        out / "train_config.json",
    )

    # ------------------------------------------------- model + LoRA injection
    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
    n_layers = len(layers) if layers else 12
    expected_params = lora_param_count(components, n_layers, hidden_dim=768, r=args.r)
    logger.info(
        "LoRA budget: 2 × r × d × |comp| × |layers| = 2×%d×768×%d×%d = %d (≤ 50K? %s)",
        args.r, len(components), n_layers, expected_params,
        expected_params <= 50_000,
    )
    if expected_params > 50_000:
        logger.warning("LoRA params %d exceeds 50K budget!", expected_params)

    lora_params = inject_lora_into_bert(
        model.bert, target_components=components, layers=layers,
        r=args.r, alpha=args.alpha, init_std=args.init_std,
    )
    # Move LoRA params to device
    model.to(device)
    actual_params = sum(p.numel() for p in lora_params)
    logger.info("injected %d LoRA params (expected %d)", actual_params, expected_params)
    assert actual_params == expected_params

    # Steering module = frozen v=0 (no-op hook, train_steering 호환 위함)
    steering = SteeringModule(hidden_dim=768, init="zero").to(device)
    for p_ in steering.parameters():
        p_.requires_grad_(False)
    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    # ------------------------------------------------------ train-side mining
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
    triplets_full = mine_triplets(train_runs, train_qrels, n_hns_per_q=N_HNS_PER_Q, pool=HN_POOL)
    logger.info("mined %d triplets (full)", len(triplets_full))

    # === EXP 15a: confused-only triplet filter ===
    confused_train_qids = confused_slice(train_runs, train_qrels, k=1)
    triplets = [t for t in triplets_full if t[0] in confused_train_qids]
    logger.info(
        "confused-only filter: kept %d / %d triplets (%.1f%%) for %d / %d confused queries",
        len(triplets), len(triplets_full), 100 * len(triplets) / len(triplets_full),
        len(confused_train_qids), len(train_runs),
    )

    if args.max_triplets is not None and len(triplets) > args.max_triplets:
        import random as _r
        _rng = _r.Random(args.seed)
        _rng.shuffle(triplets)
        triplets = triplets[: args.max_triplets]
        logger.info("subsampled to %d triplets (seed=%d)", len(triplets), args.seed)

    # ----------------------------------------------------------------- train
    extra_groups = [{"params": lora_params, "lr": args.lora_lr, "weight_decay": 1e-4}]

    history = train_steering(
        model=model, steering=steering, train_triplets=triplets,
        queries=train_queries, corpus=train_corpus, qrels=train_qrels,
        device=device, cfg=train_cfg,
        extra_param_groups=extra_groups,
        train_encoder=True,   # LoRA adapter active in train mode (dropout 등)
        early_stop_metric=args.early_stop_metric,
        val_eval_kwargs={"doc_batch": args.doc_batch,
                         "query_batch": args.query_batch,
                         "doc_chunk": args.doc_chunk},
    )

    # LoRA diagnostics
    A_norms, B_norms = [], []
    for i in range(0, len(lora_params), 2):
        A_norms.append(float(lora_params[i].detach().norm().item()))
        B_norms.append(float(lora_params[i + 1].detach().norm().item()))
    diagnostics = {
        "components": components,
        "n_layers": n_layers,
        "rank": args.r,
        "alpha": args.alpha if args.alpha is not None else args.r,
        "total_params": actual_params,
        "A_norms_per_adapter": A_norms,
        "B_norms_per_adapter": B_norms,
        "A_norm_total": float(sum(a * a for a in A_norms) ** 0.5),
        "B_norm_total": float(sum(b * b for b in B_norms) ** 0.5),
    }
    save_json(diagnostics, out / "lora_stats.json")
    save_json(dataclasses.asdict(history), out / "train_history.json")
    torch.save({
        "steering": steering.state_dict(),
        "lora": {f"adapter_{i}": p.detach().cpu() for i, p in enumerate(lora_params)},
    }, out / "module_final.pt")
    logger.info(
        "train done: total LoRA params=%d ‖A‖_total=%.4f ‖B‖_total=%.4f",
        actual_params, diagnostics["A_norm_total"], diagnostics["B_norm_total"],
    )

    del train_corpus, train_queries, train_qrels, triplets, train_topk, train_runs

    # ----------------------------------------------------------------- test
    test_corpus, test_queries, test_qrels = load_beir(args.dataset, split="test")
    logger.info("test: corpus=%d queries=%d", len(test_corpus), len(test_queries))
    model.eval()  # freeze adapter (still gradient-trackable, but eval mode)
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
        "vs_02_unfrozen",
        PROJECT_ROOT / "outputs" / "02_final_layer_vector" / args.dataset
        / f"seed_{args.seed}" / "unfrozen" / "metrics_per_query.json",
    )
    _record(
        "vs_08_r8",
        PROJECT_ROOT / "outputs" / "08_bilinear_M_minimal" / args.dataset
        / f"seed_{args.seed}" / "r_8" / "metrics_per_query.json",
    )

    logger.info("=== aggregate (all slice, tag=%s) ===", tag)
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
