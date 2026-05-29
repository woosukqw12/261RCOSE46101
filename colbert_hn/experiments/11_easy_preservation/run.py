"""11_easy_preservation — *Explicit easy-slice preservation* (λ > 0 relational loss).

외부 reviewer agent 의 제안 (Experiment 11 brief, 2026-05-24).
Step 0 (`report/_easy_slice_step0.py`) PASSED — Phase 2b 의 Δeasy = −0.085 ± 0.010
(math −0.086, 99 % match) → redistribution 확정 → 본 실험 진행 가치.

가설 (pre-committed):
  (a) Δeasy → ~0 + Δconfused ≈ +0.10 유지 → *first net improvement* on frozen encoder.
  (b) Δeasy → ~0 + Δconfused → ~0 → confused-easy *inherent entanglement* (encoder bottleneck 강화).
  (c) inconclusive → STOP, future work.

Loss 형식:
  L = (margin loss on *confused* queries) + λ × (relational self-sim 보존 on *easy* queries).
  Sim(H) = H̃ H̃^T (per-token L2-normed embedding 의 cosine matrix).
  Easy 의 Sim 은 *frozen* (LoRA 미적용) 의 reference 와 Frobenius² diff.

사용:
  .venv/bin/python experiments/11_easy_preservation/run.py \
    --dataset scifact --seed 42 --lambda-easy 1.0
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

logger = get_logger("11_easy_preservation")

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


# ============================================================================
# Frozen self-sim cache (precomputed BEFORE LoRA injection)
# ============================================================================


def precompute_frozen_self_sim(
    model: ColBERTv2,
    easy_qids_in_train: Set[str],
    triplets: List[Tuple[str, str, str]],
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    device: torch.device,
) -> Dict[Tuple[str, str], Dict[str, torch.Tensor]]:
    """Encode q + pos_doc with frozen model (LoRA not yet injected), cache
    embeddings for relational loss computation in training loop.

    Returns: dict keyed by (qid, pos_did) → {
        "q_emb": (T_q, 128) tensor on CPU,
        "d_emb": (T_d_valid, 128) tensor on CPU,  # only valid (non-pad) tokens
    }
    """
    unique_pairs = set()
    for qid, pos_did, _ in triplets:
        if qid in easy_qids_in_train:
            unique_pairs.add((qid, pos_did))
    logger.info("precompute frozen self-sim cache for %d unique (qid, pos_did) pairs (easy)",
                len(unique_pairs))

    cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
    model.bert.eval(); model.linear.eval()
    t0 = time.time()
    with torch.no_grad():
        for qid, pos_did in unique_pairs:
            q_emb, _ = model.encode_queries([queries[qid]], device=device)
            pos_emb, pos_mask = model.encode_docs(
                [doc_text(corpus[pos_did])], device=device
            )
            # q_emb: (1, T_q, 128), L2-normed, masked positions = 0
            # extract: for query, all tokens are valid (MASK-padded but per ColBERT score_mask=1)
            H_q = q_emb[0].cpu()  # (T_q, 128)
            T_d_valid = int(pos_mask[0].sum().item())
            H_d = pos_emb[0, :T_d_valid].cpu()  # (T_d_valid, 128)
            cache[(qid, pos_did)] = {"q_emb": H_q, "d_emb": H_d}
    logger.info("frozen self-sim cache done in %.1fs (%.2fs / pair)",
                time.time() - t0, (time.time() - t0) / max(1, len(unique_pairs)))
    return cache


# ============================================================================
# Relational loss (Frobenius² of self-sim diff)
# ============================================================================


def relational_self_sim_loss(
    q_emb_batch: torch.Tensor,        # (B, T_q, 128)
    pos_emb_batch: torch.Tensor,      # (B, T_d, 128)
    pos_mask_batch: torch.Tensor,     # (B, T_d) bool
    batch_qids: List[str],
    batch_pos_dids: List[str],
    easy_indices_in_batch: List[int],
    frozen_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Compute ||Sim(H_LoRA) - Sim(H_frozen)||_F² averaged over easy queries.

    Sim(H) = H @ H.T for L2-normalised H (i.e., per-token cosine matrix).
    Padding tokens in pos_doc are excluded via pos_mask.
    Query: all positions are valid per ColBERT (MASK-padding included).
    """
    if not easy_indices_in_batch:
        return torch.zeros((), device=q_emb_batch.device)

    losses = []
    for i in easy_indices_in_batch:
        qid = batch_qids[i]
        pos_did = batch_pos_dids[i]
        key = (qid, pos_did)
        if key not in frozen_cache:
            continue
        ref = frozen_cache[key]
        H_q_frozen = ref["q_emb"].to(q_emb_batch.device)   # (T_q, 128)
        H_d_frozen = ref["d_emb"].to(q_emb_batch.device)   # (T_d_valid, 128)

        # LoRA-side
        H_q_lora = q_emb_batch[i]    # (T_q, 128), L2-normed, no mask needed for query
        T_d_valid = int(pos_mask_batch[i].sum().item())
        H_d_lora = pos_emb_batch[i, :T_d_valid]  # (T_d_valid, 128)

        # Self-sim matrices
        sim_q_lora = H_q_lora @ H_q_lora.t()      # (T_q, T_q)
        sim_q_frozen = H_q_frozen @ H_q_frozen.t()
        sim_d_lora = H_d_lora @ H_d_lora.t()
        sim_d_frozen = H_d_frozen @ H_d_frozen.t()

        # Frobenius² (sum of squared element diffs) — normalised by Sim 의 element count
        T_q = H_q_lora.shape[0]
        loss_q = (sim_q_lora - sim_q_frozen).pow(2).sum() / (T_q * T_q)
        loss_d = (sim_d_lora - sim_d_frozen).pow(2).sum() / max(1, T_d_valid * T_d_valid)
        losses.append(loss_q + loss_d)

    if not losses:
        return torch.zeros((), device=q_emb_batch.device)
    return torch.stack(losses).mean()


# ============================================================================
# Custom training loop — split confused/easy + relational loss
# ============================================================================


def _split_triplets_local(triplets, val_frac: float, seed: int):
    by_q: Dict[str, list] = {}
    for t in triplets:
        by_q.setdefault(t[0], []).append(t)
    qids = sorted(by_q.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * val_frac))
    val_qids = set(qids[:n_val])
    train_triplets = [t for t in triplets if t[0] not in val_qids]
    return train_triplets, sorted(val_qids)


def _make_batches_local(triplets, batch_size, seed, epoch):
    rng = random.Random(seed * 100003 + epoch)
    shuffled = list(triplets)
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


def train_easy_preserve(
    model: ColBERTv2,
    lora_params: List[torch.nn.Parameter],
    train_triplets: List[Tuple[str, str, str]],
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    qrels: Dict[str, Dict[str, int]],
    confused_qids: Set[str],     # baseline-derived confused queries
    easy_qids: Set[str],         # baseline-derived easy queries
    frozen_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]],
    lambda_easy: float,
    device: torch.device,
    cfg: TrainConfigLite,
    val_eval_kwargs: dict,
    in_batch_neg: bool = False,  # combined M1b + Exp 11 if True
) -> TrainHistory:
    """Easy-preserving training: confused queries → margin loss, easy queries → relational loss.

    *Critical fix vs train_steering*: snapshot **LoRA + steering** both at best epoch (Phase 2b /
    M1 의 LoRA-not-snapshotted 한계 해소).
    """
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    )
    train_subset, val_qids = _split_triplets_local(train_triplets, cfg.val_split, cfg.seed)
    logger.info("train: %d triplets (%d after val held-out), val_qids=%d",
                len(train_triplets), len(train_subset), len(val_qids))

    # warmup / grad-clip support
    total_steps = max(1, math.ceil(len(train_subset) / cfg.batch_size)) * cfg.epochs
    warmup_steps = int(cfg.warmup_frac * total_steps)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    if warmup_steps > 0:
        logger.info("warmup: 0→LR over %d / %d steps", warmup_steps, total_steps)
    if cfg.grad_clip_max_norm > 0:
        logger.info("grad-clip: max_norm = %.4f", cfg.grad_clip_max_norm)
    logger.info("lambda_easy = %.4f, |confused|=%d, |easy|=%d in train queries",
                lambda_easy,
                sum(1 for t in train_subset if t[0] in confused_qids),
                sum(1 for t in train_subset if t[0] in easy_qids))

    history = TrainHistory()
    step = 0
    best_val = -math.inf
    best_lora_state = None
    epochs_since_best = 0

    for epoch in range(cfg.epochs):
        model.bert.train(); model.linear.train()
        batches = _make_batches_local(train_subset, cfg.batch_size, cfg.seed, epoch)
        t_epoch = time.time()
        epoch_rank = 0.0
        epoch_rel = 0.0
        n_rank_batches = 0
        n_rel_batches = 0
        for batch in batches:
            qids = [t[0] for t in batch]
            pos_dids = [t[1] for t in batch]
            hn_dids = [t[2] for t in batch]

            q_texts = [queries[q] for q in qids]
            pos_texts = [doc_text(corpus[d]) for d in pos_dids]

            q_emb, _ = model.encode_queries(q_texts, device=device)
            pos_emb, pos_mask = model.encode_docs(pos_texts, device=device)

            confused_idx = [i for i, q in enumerate(qids) if q in confused_qids]
            easy_idx = [i for i, q in enumerate(qids) if q in easy_qids]

            # Confused: margin loss
            if confused_idx:
                if in_batch_neg:
                    # M1b combined: 다른 query 의 positive (whole batch roll) 를 negative 로
                    if pos_emb.size(0) < 2:
                        rank_loss = torch.zeros((), device=device)
                    else:
                        hn_emb_all = pos_emb.roll(1, dims=0)
                        hn_mask_all = pos_mask.roll(1, dims=0)
                        hn_emb = hn_emb_all[confused_idx]
                        hn_mask = hn_mask_all[confused_idx]
                        q_emb_c = q_emb[confused_idx]
                        pos_emb_c = pos_emb[confused_idx]
                        pos_mask_c = pos_mask[confused_idx]
                        s_pos = model.diagonal_maxsim(q_emb_c, pos_emb_c, pos_mask_c)
                        s_hn = model.diagonal_maxsim(q_emb_c, hn_emb, hn_mask)
                        rank_loss = torch.clamp(cfg.margin - s_pos + s_hn, min=0).mean()
                        epoch_rank += float(rank_loss.detach().item())
                        n_rank_batches += 1
                else:
                    hn_texts = [doc_text(corpus[hn_dids[i]]) for i in confused_idx]
                    hn_emb, hn_mask = model.encode_docs(hn_texts, device=device)
                    q_emb_c = q_emb[confused_idx]
                    pos_emb_c = pos_emb[confused_idx]
                    pos_mask_c = pos_mask[confused_idx]
                    s_pos = model.diagonal_maxsim(q_emb_c, pos_emb_c, pos_mask_c)
                    s_hn = model.diagonal_maxsim(q_emb_c, hn_emb, hn_mask)
                    rank_loss = torch.clamp(cfg.margin - s_pos + s_hn, min=0).mean()
                    epoch_rank += float(rank_loss.detach().item())
                    n_rank_batches += 1
            else:
                rank_loss = torch.zeros((), device=device)

            # Easy: relational self-sim 보존 loss
            if easy_idx and lambda_easy > 0:
                rel_loss = relational_self_sim_loss(
                    q_emb, pos_emb, pos_mask, qids, pos_dids, easy_idx, frozen_cache,
                )
                epoch_rel += float(rel_loss.detach().item())
                n_rel_batches += 1
            else:
                rel_loss = torch.zeros((), device=device)

            loss = rank_loss + lambda_easy * rel_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip_max_norm)
            if warmup_steps > 0 and step < warmup_steps:
                mult = (step + 1) / warmup_steps
                for pg, base in zip(optimizer.param_groups, base_lrs):
                    pg["lr"] = base * mult
            elif warmup_steps > 0 and step == warmup_steps:
                for pg, base in zip(optimizer.param_groups, base_lrs):
                    pg["lr"] = base
            optimizer.step()
            step += 1

            v_norm = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in lora_params)).item())
            history.steps.append(step)
            history.losses.append(float(loss.detach().item()))
            history.rank_losses.append(float(rank_loss.detach().item()))
            history.anchor_losses.append(float(rel_loss.detach().item()))  # repurpose for relational
            history.v_norms.append(v_norm)
            history.epoch.append(epoch)

        logger.info("epoch %d/%d: rank_loss=%.4f rel_loss=%.4f time=%.1fs",
                    epoch + 1, cfg.epochs,
                    epoch_rank / max(1, n_rank_batches),
                    epoch_rel / max(1, n_rel_batches),
                    time.time() - t_epoch)

        # Validation
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
            # **Critical fix**: snapshot LoRA params (not just steering)
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
    p.add_argument("--dataset", default="scifact", choices=["scifact"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-easy", type=float, default=1.0,
                   help="Easy-preservation relational loss weight (default 1.0, single value pre-commit)")
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
    p.add_argument("--warmup-frac", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--early-stop-metric", default="all", choices=["all", "confused"],
                   help="(unused in Exp 11 — always val_all; flag accepted for compatibility)")
    p.add_argument("--in-batch-neg", action="store_true",
                   help="Combined M1b + Exp 11: replace mined HN with in-batch neg for confused queries")
    p.add_argument("--fn-denoise", action="store_true",
                   help="Exp 13: filter mined HN by E5 margin > threshold (remove FN). Requires e5_train_doc_emb_{dataset}.pt")
    p.add_argument("--fn-threshold", type=float, default=0.0,
                   help="E5 margin threshold for FN denoising (default 0.0)")
    p.add_argument("--tag-suffix", default="",
                   help="Suffix appended to artifact tag (e.g. 'm1b' for combined)")
    args = p.parse_args()

    cfg = BASELINE
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("dataset=%s seed=%d λ_easy=%.3f", args.dataset, args.seed, args.lambda_easy)

    tag = f"qv_r{args.r}_l12_le{args.lambda_easy:g}"
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
    base_out = artifact_dir(exp_name="11_easy_preservation",
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
        warmup_frac=args.warmup_frac, grad_clip_max_norm=args.grad_clip,
    )
    save_json({**dataclasses.asdict(train_cfg),
               "lambda_easy": args.lambda_easy, "r": args.r, "alpha": args.alpha,
               "max_triplets": args.max_triplets},
              out / "train_config.json")

    # 1. Build frozen ColBERT
    model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)

    # 2. Train side: mine HN + identify easy/confused queries
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
    confused_train_qids = confused_slice(train_runs, train_qrels, k=1)
    easy_train_qids = set(train_runs.keys()) - confused_train_qids
    logger.info("train slices: confused=%d easy=%d (total=%d)",
                len(confused_train_qids), len(easy_train_qids), len(train_runs))

    triplets = mine_triplets(train_runs, train_qrels, n_hns_per_q=N_HNS_PER_Q, pool=HN_POOL)
    logger.info("mined %d triplets", len(triplets))

    # Exp 13: FN-denoising (Exp 12 logic re-applied AFTER mining, BEFORE subsample)
    if args.fn_denoise:
        sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "12_fn_denoised_hn"))
        from run import load_e5_train, filter_triplets_by_e5_margin  # type: ignore
        e5_q, e5_d, q2i, d2i = load_e5_train(args.dataset)
        logger.info("E5 teacher: %d queries, %d docs", e5_q.shape[0], e5_d.shape[0])
        triplets_raw_n = len(triplets)
        triplets, removed, margins, missing = filter_triplets_by_e5_margin(
            triplets, e5_q, e5_d, q2i, d2i, threshold=args.fn_threshold,
        )
        import numpy as np
        margins_np = np.array(margins) if margins else np.array([0.0])
        logger.info(
            "FN-denoising: kept=%d / %d (%.1f%% removed as likely FN, threshold=%.2f)",
            len(triplets), triplets_raw_n,
            100 * len(removed) / max(1, triplets_raw_n), args.fn_threshold,
        )
        logger.info(
            "  margin stats: mean=%+.4f, median=%+.4f",
            margins_np.mean(), float(np.median(margins_np)),
        )
        save_json({
            "n_raw": triplets_raw_n, "n_kept": len(triplets),
            "n_removed": len(removed), "n_missing_emb": missing,
            "fn_rate_per_e5": float(len(removed) / max(1, triplets_raw_n)),
            "margin_mean": float(margins_np.mean()),
            "margin_median": float(np.median(margins_np)),
            "margin_threshold": args.fn_threshold,
        }, out / "denoising_stats.json")

    if args.max_triplets and len(triplets) > args.max_triplets:
        _rng = random.Random(args.seed); _rng.shuffle(triplets)
        triplets = triplets[: args.max_triplets]
        logger.info("subsampled to %d triplets (seed=%d)", len(triplets), args.seed)

    # 3. Precompute frozen self-sim cache (BEFORE LoRA injection)
    frozen_cache = precompute_frozen_self_sim(model, easy_train_qids, triplets,
                                               train_queries, train_corpus, device)

    # 4. Inject LoRA (q, v, r=args.r)
    expected = lora_param_count(["q", "v"], 12, hidden_dim=768, r=args.r)
    lora_params = inject_lora_into_bert(model.bert, target_components=["q", "v"],
                                         layers=None, r=args.r, alpha=args.alpha,
                                         init_std=args.init_std)
    model.to(device)
    assert sum(p.numel() for p in lora_params) == expected

    # 5. Setup steering (frozen v=0 hook, for hook infrastructure)
    steering = SteeringModule(hidden_dim=768, init="zero").to(device)
    for p_ in steering.parameters():
        p_.requires_grad_(False)
    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, steering.hook_fn())

    # 6. Train with easy preservation
    history = train_easy_preserve(
        model, lora_params, triplets, train_queries, train_corpus, train_qrels,
        confused_train_qids, easy_train_qids, frozen_cache,
        lambda_easy=args.lambda_easy, device=device, cfg=train_cfg,
        val_eval_kwargs={"doc_batch": args.doc_batch,
                         "query_batch": args.query_batch,
                         "doc_chunk": args.doc_chunk},
        in_batch_neg=args.in_batch_neg,
    )

    # Diagnostics
    A_norms, B_norms = [], []
    for i in range(0, len(lora_params), 2):
        A_norms.append(float(lora_params[i].detach().norm().item()))
        B_norms.append(float(lora_params[i + 1].detach().norm().item()))
    save_json({"A_norms_per_adapter": A_norms, "B_norms_per_adapter": B_norms,
               "A_norm_total": float(sum(a * a for a in A_norms) ** 0.5),
               "B_norm_total": float(sum(b * b for b in B_norms) ** 0.5),
               "lambda_easy": args.lambda_easy},
              out / "lora_stats.json")
    save_json(dataclasses.asdict(history), out / "train_history.json")
    torch.save({"steering": steering.state_dict(),
                "lora": {f"adapter_{i}": p.detach().cpu() for i, p in enumerate(lora_params)}},
               out / "module_final.pt")

    del train_corpus, train_queries, train_qrels, triplets, train_topk, train_runs, frozen_cache

    # 7. Test eval
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

    # Δall, Δconfused, Δeasy
    deltas = {
        "all": _paired_ci_vs(per_q, baseline_per_q, set(per_q.keys()), seed=args.seed),
        "confused": _paired_ci_vs(per_q, baseline_per_q, confused, seed=args.seed),
        "easy": _paired_ci_vs(per_q, baseline_per_q, easy, seed=args.seed),
    }
    save_json(deltas, out / "delta_vs_baseline.json")
    logger.info("=== Δ vs baseline ===")
    for sn, dr in deltas.items():
        if "mean_delta_ndcg10" in dr:
            logger.info("  %-9s n=%d Δ=%+.4f [%+.4f,%+.4f]%s",
                        sn, dr["n"], dr["mean_delta_ndcg10"], dr["ci_lo"], dr["ci_hi"],
                        " ✓ positive" if dr.get("positive")
                        else " ✗ negative" if dr.get("negative") else "")
    logger.info("artifacts → %s", out)


if __name__ == "__main__":
    main()
