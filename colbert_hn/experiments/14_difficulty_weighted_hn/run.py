"""14_difficulty_weighted_hn — Continuous HN difficulty weighting via E5 margin sigmoid.

Theory-driven *continuous control* of hard contrast (M1b binary 0 % ↔ Phase 2b binary 100 %).
Pre-committed: `report/_exp13_14_pre_commit.md` (α_w = 10 single value).

Per-triplet weight: w_i = sigmoid(α_w * e5_margin_i)
    e5_margin = cos(eq, epos) − cos(eq, ehn) using E5-Mistral cached embeddings.

Loss: weighted mean of clamp(margin - s_pos + s_hn, 0).

Implementation:
1. Build frozen ColBERT + load E5 train q/d embeddings
2. Mine triplets + compute e5_margin per triplet
3. Inject LoRA
4. Train with per-triplet sigmoid-weighted margin loss
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colbert_hook import ColBERTConfig, ColBERTv2  # noqa: E402
from src.configs import BASELINE  # noqa: E402
from src.data import doc_text, load_beir  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_aggregate, compute_metrics_trec, encode_corpus, save_env, score_queries,
)
from src.hn_mining import mine_triplets  # noqa: E402
from src.lora import inject_lora_into_bert, lora_param_count  # noqa: E402
from src.lsr import SteeringModule  # noqa: E402
from src.mean_diff import HOOK_LAYER  # noqa: E402
from src.metrics import align_per_query, paired_bootstrap_ci  # noqa: E402
from src.slices import confused_slice  # noqa: E402
from src.train import TrainConfigLite, TrainHistory  # noqa: E402
from src.utils.io import artifact_dir, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.repro import get_device, set_seed  # noqa: E402

logger = get_logger("14_difficulty_weighted_hn")

TRAIN_AVAILABLE = ("scifact",)
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
    """Load E5 train query + corpus doc embeddings (cached)."""
    e5_dir = PROJECT_ROOT / "data" / "e5_teacher"
    q_path = e5_dir / f"e5_train_q_emb_{dataset}.pt"
    d_path = e5_dir / f"e5_train_doc_emb_{dataset}.pt"
    if not q_path.exists():
        raise FileNotFoundError(f"E5 train query emb 부재: {q_path}")
    if not d_path.exists():
        raise FileNotFoundError(f"E5 train doc emb 부재: {d_path}")
    q = torch.load(q_path, weights_only=False)
    d = torch.load(d_path, weights_only=False)
    qid_to_idx = {qid: i for i, qid in enumerate(q["qids"])}
    did_to_idx = {did: i for i, did in enumerate(d["dids"])}
    return q["query_emb"], d["doc_emb"], qid_to_idx, did_to_idx


def compute_triplet_weights(
    triplets, e5_q_emb, e5_d_emb, qid_to_idx, did_to_idx, alpha_w: float,
):
    """Compute sigmoid(α_w * e5_margin) weight per triplet."""
    weights = []
    margins = []
    missing = 0
    for qid, pos_did, hn_did in triplets:
        if qid not in qid_to_idx or pos_did not in did_to_idx or hn_did not in did_to_idx:
            missing += 1
            weights.append(0.5)  # default neutral weight if embedding missing
            margins.append(0.0)
            continue
        eq = e5_q_emb[qid_to_idx[qid]].float()
        ep = e5_d_emb[did_to_idx[pos_did]].float()
        eh = e5_d_emb[did_to_idx[hn_did]].float()
        m = float((eq @ ep).item() - (eq @ eh).item())
        margins.append(m)
        w = 1.0 / (1.0 + math.exp(-alpha_w * m))
        weights.append(w)
    return weights, margins, missing


def _split_triplets_local(triplets, val_frac: float, seed: int):
    by_q: Dict[str, list] = {}
    for t in triplets:
        by_q.setdefault(t[0], []).append(t)
    qids = sorted(by_q.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * val_frac))
    val_qids = set(qids[:n_val])
    train_triplets_w_idx = [
        (i, t) for i, t in enumerate(triplets) if t[0] not in val_qids
    ]
    return train_triplets_w_idx, sorted(val_qids)


def _make_batches_local(items, batch_size, seed, epoch):
    rng = random.Random(seed * 100003 + epoch)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]


def _val_pass_local(model, val_qids, queries, qrels, corpus, device,
                     doc_batch=64, query_batch=16, doc_chunk=512, top_k=100):
    sub_queries = {q: queries[q] for q in val_qids if q in queries}
    sub_qrels = {q: qrels[q] for q in val_qids if q in qrels}
    with torch.no_grad():
        dids, d_emb, d_mask = encode_corpus(model, corpus, device, batch_size=doc_batch)
        topk = score_queries(model, sub_queries, dids, d_emb, d_mask, device,
                             query_batch=query_batch, doc_chunk=doc_chunk, top_k=top_k)
    runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
    runs_scored = {q: dict(lst) for q, lst in topk.items()}
    per_q = compute_metrics_trec(runs_scored, sub_qrels, metrics_k=(10,))
    if not per_q:
        return float("nan"), float("nan")
    all_avg = sum(v["ndcg_cut_10"] for v in per_q.values()) / len(per_q)
    conf = confused_slice(runs_ranked, sub_qrels, k=1)
    if not conf:
        return all_avg, float("nan")
    conf_avg = sum(per_q[q]["ndcg_cut_10"] for q in conf if q in per_q) / len(conf)
    return all_avg, conf_avg


def train_difficulty_weighted(
    model: ColBERTv2,
    lora_params: List[torch.nn.Parameter],
    triplets: List[Tuple[str, str, str]],
    triplet_weights: List[float],
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    qrels: Dict[str, Dict[str, int]],
    device: torch.device,
    cfg: TrainConfigLite,
    val_eval_kwargs: dict,
) -> TrainHistory:
    """Training with per-triplet sigmoid-weighted margin loss.

    LoRA best-state snapshot (M1 lesson).
    """
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    )
    train_items, val_qids = _split_triplets_local(triplets, cfg.val_split, cfg.seed)
    logger.info("train: %d triplets (%d after val split), val_qids=%d",
                len(triplets), len(train_items), len(val_qids))

    history = TrainHistory()
    step = 0
    best_val = -math.inf
    best_lora_state = None
    epochs_since_best = 0

    for epoch in range(cfg.epochs):
        model.bert.train(); model.linear.train()
        batches = _make_batches_local(train_items, cfg.batch_size, cfg.seed, epoch)
        t_epoch = time.time()
        epoch_loss = 0.0
        epoch_w_sum = 0.0
        n_batches = 0
        for batch in batches:
            batch_idxs = [item[0] for item in batch]
            batch_triplets = [item[1] for item in batch]
            qids = [t[0] for t in batch_triplets]
            pos_dids = [t[1] for t in batch_triplets]
            hn_dids = [t[2] for t in batch_triplets]
            weights = torch.tensor([triplet_weights[i] for i in batch_idxs],
                                    device=device, dtype=torch.float32)

            q_texts = [queries[q] for q in qids]
            pos_texts = [doc_text(corpus[d]) for d in pos_dids]
            hn_texts = [doc_text(corpus[d]) for d in hn_dids]

            q_emb, _ = model.encode_queries(q_texts, device=device)
            pos_emb, pos_mask = model.encode_docs(pos_texts, device=device)
            hn_emb, hn_mask = model.encode_docs(hn_texts, device=device)

            s_pos = model.diagonal_maxsim(q_emb, pos_emb, pos_mask)
            s_hn = model.diagonal_maxsim(q_emb, hn_emb, hn_mask)
            per_triplet_loss = torch.clamp(cfg.margin - s_pos + s_hn, min=0)  # (B,)
            # Weighted average: sum(w_i * loss_i) / sum(w_i)
            w_sum = weights.sum().clamp(min=1e-6)
            rank_loss = (weights * per_triplet_loss).sum() / w_sum
            epoch_loss += float(rank_loss.detach().item())
            epoch_w_sum += float(w_sum.item())
            n_batches += 1

            optimizer.zero_grad(set_to_none=True)
            rank_loss.backward()
            optimizer.step()
            step += 1

            v_norm = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in lora_params)).item())
            history.steps.append(step)
            history.losses.append(float(rank_loss.detach().item()))
            history.rank_losses.append(float(rank_loss.detach().item()))
            history.anchor_losses.append(0.0)
            history.v_norms.append(v_norm)
            history.epoch.append(epoch)

        logger.info("epoch %d/%d: weighted_loss=%.4f mean_weight=%.4f time=%.1fs",
                    epoch + 1, cfg.epochs,
                    epoch_loss / max(1, n_batches),
                    epoch_w_sum / max(1, n_batches * cfg.batch_size),
                    time.time() - t_epoch)

        model.bert.eval(); model.linear.eval()
        t_val = time.time()
        val_all, val_conf = _val_pass_local(model, val_qids, queries, qrels, corpus,
                                             device, **val_eval_kwargs)
        logger.info("  val: NDCG@10 all=%.4f confused=%.4f (time=%.1fs)",
                    val_all, val_conf, time.time() - t_val)
        history.val_epochs.append(epoch + 1)
        history.val_ndcg_all.append(val_all)
        history.val_ndcg_confused.append(val_conf)

        score = val_all if not math.isnan(val_all) else val_conf
        if score > best_val:
            best_val = score
            best_lora_state = [p.detach().clone().cpu() for p in lora_params]
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, cfg.patience)
                break

    if best_lora_state is not None:
        for p, best in zip(lora_params, best_lora_state):
            p.data.copy_(best.to(p.device))
        logger.info("restored LoRA to best state (val=%.4f)", best_val)
    return history


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="scifact", choices=TRAIN_AVAILABLE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha-w", type=float, default=10.0,
                   help="Sigmoid sharpness for triplet weight (pre-commit 10 single value)")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--init-std", type=float, default=0.02)
    p.add_argument("--lora-lr", type=float, default=5e-5)
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
    logger.info("dataset=%s seed=%d α_w=%.3f", args.dataset, args.seed, args.alpha_w)

    tag = f"qv_r{args.r}_l12_diffw{args.alpha_w:g}"
    base_out = artifact_dir(exp_name="14_difficulty_weighted_hn",
                            dataset=args.dataset, seed=args.seed)
    out = base_out / tag
    out.mkdir(parents=True, exist_ok=True)
    save_env(out, args.seed, device)
    cfg.save(out / "config.json")

    train_cfg = TrainConfigLite(
        hook_layer=HOOK_LAYER, margin=args.margin,
        lr=args.lora_lr, weight_decay=1e-4,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        val_split=0.1, lambda_anchor=0.0, seed=args.seed,
    )
    save_json({**dataclasses.asdict(train_cfg),
               "alpha_w": args.alpha_w, "r": args.r, "alpha": args.alpha,
               "max_triplets": args.max_triplets},
              out / "train_config.json")

    # 1. Build frozen ColBERT + LoRA injection
    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
    expected = lora_param_count(["q", "v"], 12, hidden_dim=768, r=args.r)
    lora_params = inject_lora_into_bert(model.bert, target_components=["q", "v"],
                                         layers=None, r=args.r, alpha=args.alpha,
                                         init_std=args.init_std)
    model.to(device)
    assert sum(p.numel() for p in lora_params) == expected
    logger.info("LoRA injected: %d params", expected)

    steering = SteeringModule(hidden_dim=768, init="zero").to(device)
    for p_ in steering.parameters():
        p_.requires_grad_(False)
    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    # 2. Load E5 teacher
    e5_q_emb, e5_d_emb, qid_to_idx, did_to_idx = load_e5_train(args.dataset)
    logger.info("E5 teacher: %d queries, %d docs", e5_q_emb.shape[0], e5_d_emb.shape[0])

    # 3. Mine HN
    train_corpus, train_queries, train_qrels = load_beir(args.dataset, split="train")
    logger.info("train: corpus=%d queries=%d", len(train_corpus), len(train_queries))

    t0 = time.time()
    with torch.no_grad():
        train_dids, td_emb, td_mask = encode_corpus(model, train_corpus, device,
                                                    batch_size=args.doc_batch)
        train_topk = score_queries(model, train_queries, train_dids, td_emb, td_mask,
                                    device, query_batch=args.query_batch,
                                    doc_chunk=args.doc_chunk, top_k=HN_POOL)
    logger.info("train baseline pass in %.1fs", time.time() - t0)
    del td_emb, td_mask

    train_runs = {q: [d for d, _ in lst] for q, lst in train_topk.items()}
    triplets = mine_triplets(train_runs, train_qrels, n_hns_per_q=N_HNS_PER_Q, pool=HN_POOL)
    logger.info("mined %d triplets", len(triplets))

    if args.max_triplets and len(triplets) > args.max_triplets:
        _rng = random.Random(args.seed); _rng.shuffle(triplets)
        triplets = triplets[: args.max_triplets]
        logger.info("subsampled to %d triplets (seed=%d)", len(triplets), args.seed)

    # 4. Compute per-triplet weights using E5 margin sigmoid
    weights, margins, missing = compute_triplet_weights(
        triplets, e5_q_emb, e5_d_emb, qid_to_idx, did_to_idx, args.alpha_w,
    )
    import numpy as np
    weights_np = np.array(weights); margins_np = np.array(margins)
    logger.info("triplet weights: mean=%.4f, median=%.4f, std=%.4f (missing emb=%d)",
                float(weights_np.mean()), float(np.median(weights_np)),
                float(weights_np.std()), missing)
    logger.info("e5 margins: mean=%+.4f, median=%+.4f, range=[%+.4f, %+.4f]",
                float(margins_np.mean()), float(np.median(margins_np)),
                float(margins_np.min()), float(margins_np.max()))
    save_json({
        "n_triplets": len(triplets),
        "n_missing_emb": missing,
        "alpha_w": args.alpha_w,
        "weights": {"mean": float(weights_np.mean()),
                    "median": float(np.median(weights_np)),
                    "std": float(weights_np.std()),
                    "min": float(weights_np.min()),
                    "max": float(weights_np.max())},
        "margins": {"mean": float(margins_np.mean()),
                    "median": float(np.median(margins_np)),
                    "min": float(margins_np.min()),
                    "max": float(margins_np.max())},
    }, out / "weight_stats.json")

    # 5. Train
    history = train_difficulty_weighted(
        model, lora_params, triplets, weights,
        train_queries, train_corpus, train_qrels,
        device=device, cfg=train_cfg,
        val_eval_kwargs={"doc_batch": args.doc_batch,
                         "query_batch": args.query_batch,
                         "doc_chunk": args.doc_chunk},
    )

    A_norms, B_norms = [], []
    for i in range(0, len(lora_params), 2):
        A_norms.append(float(lora_params[i].detach().norm().item()))
        B_norms.append(float(lora_params[i + 1].detach().norm().item()))
    save_json({"A_norms_per_adapter": A_norms, "B_norms_per_adapter": B_norms,
               "A_norm_total": float(sum(a * a for a in A_norms) ** 0.5),
               "B_norm_total": float(sum(b * b for b in B_norms) ** 0.5),
               "alpha_w": args.alpha_w},
              out / "lora_stats.json")
    save_json(dataclasses.asdict(history), out / "train_history.json")
    torch.save({"steering": steering.state_dict(),
                "lora": {f"adapter_{i}": p.detach().cpu() for i, p in enumerate(lora_params)}},
               out / "module_final.pt")

    del train_corpus, train_queries, train_qrels, triplets, train_topk, train_runs

    # 6. Test eval
    test_corpus, test_queries, test_qrels = load_beir(args.dataset, split="test")
    logger.info("test: corpus=%d queries=%d", len(test_corpus), len(test_queries))
    model.eval(); steering.eval()
    with torch.no_grad():
        test_dids, d_emb, d_mask = encode_corpus(model, test_corpus, device, batch_size=args.doc_batch)
        topk = score_queries(model, test_queries, test_dids, d_emb, d_mask, device,
                              query_batch=args.query_batch, doc_chunk=args.doc_chunk,
                              top_k=cfg.eval.retrieval_top_k)
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
    logger.info("=== Δ vs baseline (Exp 14 difficulty-weighted HN) ===")
    for sn, dr in deltas.items():
        if "mean_delta_ndcg10" in dr:
            logger.info("  %-9s n=%d Δ=%+.4f [%+.4f,%+.4f]%s",
                        sn, dr["n"], dr["mean_delta_ndcg10"], dr["ci_lo"], dr["ci_hi"],
                        " ✓ positive" if dr.get("positive")
                        else " ✗ negative" if dr.get("negative") else "")
    logger.info("artifacts → %s", out)


if __name__ == "__main__":
    main()
