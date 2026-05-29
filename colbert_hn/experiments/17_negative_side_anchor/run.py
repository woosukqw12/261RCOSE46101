"""17_negative_side_anchor — Per-token cosine anchor extended to mined HN docs.

Forks experiments/13_frozen_direction_anchor with a single principled extension:
the anchor target set is augmented from {q, d^+} to {q, d^+, d^-} on easy queries.
Pre-committed: `report/_exp17_negative_anchor_pre_commit.md`.

Loss:
    L = L_margin(confused)
        + λ_dir * mean_{easy x} [ cos_dev(H_q) + cos_dev(H_d^+) ]
        + λ_neg * mean_{easy x} [ cos_dev(H_d^-) ]
    cos_dev(H) = mean_t (1 - cos(h_t_LoRA, h_t_frozen)),  embeddings L2-normed.

Single config, no sweep: λ_dir = 1.0, λ_neg = 1.0. 3 seeds {42, 1337, 2024} on SciFact only.
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

logger = get_logger("17_negative_side_anchor")

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


def precompute_frozen_embeddings(
    model: ColBERTv2,
    easy_qids: Set[str],
    triplets: List[Tuple[str, str, str]],
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    device: torch.device,
) -> Tuple[Dict[Tuple[str, str], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Encode q + pos_doc + hn_doc with frozen model, cache L2-normed embeddings.

    Returns:
      pair_cache: dict[(qid, pos_did)] -> {"q_emb": (T_q, 128), "d_emb": (T_d, 128)}
      hn_cache: dict[hn_did] -> (T_d, 128)   # keyed by doc id only (HN frozen rep is doc-only)
    """
    unique_pairs = set()
    unique_hns = set()
    for qid, pos_did, hn_did in triplets:
        if qid in easy_qids:
            unique_pairs.add((qid, pos_did))
            unique_hns.add(hn_did)
    logger.info("precompute frozen: %d (qid, pos_did) pairs + %d unique HN docs",
                len(unique_pairs), len(unique_hns))

    pair_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
    hn_cache: Dict[str, torch.Tensor] = {}
    model.bert.eval(); model.linear.eval()
    t0 = time.time()
    with torch.no_grad():
        for qid, pos_did in unique_pairs:
            q_emb, _ = model.encode_queries([queries[qid]], device=device)
            pos_emb, pos_mask = model.encode_docs([doc_text(corpus[pos_did])], device=device)
            H_q = q_emb[0].cpu()
            T_d_valid = int(pos_mask[0].sum().item())
            H_d = pos_emb[0, :T_d_valid].cpu()
            pair_cache[(qid, pos_did)] = {"q_emb": H_q, "d_emb": H_d}
        for hn_did in unique_hns:
            hn_emb, hn_mask = model.encode_docs([doc_text(corpus[hn_did])], device=device)
            T_hn_valid = int(hn_mask[0].sum().item())
            hn_cache[hn_did] = hn_emb[0, :T_hn_valid].cpu()
    logger.info("frozen embedding cache done in %.1fs", time.time() - t0)
    return pair_cache, hn_cache


def cosine_deviation_loss_with_hn(
    q_emb_batch: torch.Tensor,        # (B, T_q, 128) LoRA
    pos_emb_batch: torch.Tensor,      # (B, T_d, 128) LoRA
    pos_mask_batch: torch.Tensor,
    hn_emb_batch: torch.Tensor,       # (B, T_hn, 128) LoRA (full batch — caller passes full hn fwd)
    hn_mask_batch: torch.Tensor,
    batch_qids: List[str],
    batch_pos_dids: List[str],
    batch_hn_dids: List[str],
    easy_indices: List[int],
    pair_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]],
    hn_cache: Dict[str, torch.Tensor],
    lambda_neg: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token cosine deviation:
       returns (dir_loss, neg_loss) — separate so caller can apply (λ_dir, λ_neg) explicitly.
       dir_loss = mean_x [ mean_t(1-cos_q) + mean_t(1-cos_d+) ]
       neg_loss = mean_x [ mean_t(1-cos_d-) ]
    """
    if not easy_indices:
        zero = torch.zeros((), device=q_emb_batch.device)
        return zero, zero

    dir_losses = []
    neg_losses = []
    for i in easy_indices:
        qid = batch_qids[i]
        pos_did = batch_pos_dids[i]
        hn_did = batch_hn_dids[i]
        key = (qid, pos_did)
        if key not in pair_cache:
            continue
        ref = pair_cache[key]
        H_q_frozen = ref["q_emb"].to(q_emb_batch.device)
        H_dp_frozen = ref["d_emb"].to(q_emb_batch.device)
        H_q_lora = q_emb_batch[i]
        T_dp_valid = int(pos_mask_batch[i].sum().item())
        H_dp_lora = pos_emb_batch[i, :T_dp_valid]

        cos_q = (H_q_lora * H_q_frozen).sum(dim=-1)
        cos_dp = (H_dp_lora * H_dp_frozen).sum(dim=-1)
        loss_q = (1.0 - cos_q).mean()
        loss_dp = (1.0 - cos_dp).mean()
        dir_losses.append(loss_q + loss_dp)

        if lambda_neg > 0 and hn_did in hn_cache:
            H_dn_frozen = hn_cache[hn_did].to(q_emb_batch.device)
            T_dn_valid = int(hn_mask_batch[i].sum().item())
            H_dn_lora = hn_emb_batch[i, :T_dn_valid]
            # Guard: frozen HN may have different valid length than LoRA HN
            T_use = min(T_dn_valid, H_dn_frozen.shape[0])
            if T_use > 0:
                cos_dn = (H_dn_lora[:T_use] * H_dn_frozen[:T_use]).sum(dim=-1)
                loss_dn = (1.0 - cos_dn).mean()
                neg_losses.append(loss_dn)

    if not dir_losses:
        zero = torch.zeros((), device=q_emb_batch.device)
        return zero, zero
    dir_loss = torch.stack(dir_losses).mean()
    if neg_losses:
        neg_loss = torch.stack(neg_losses).mean()
    else:
        neg_loss = torch.zeros((), device=q_emb_batch.device)
    return dir_loss, neg_loss


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


def train_with_neg_anchor(
    model: ColBERTv2,
    lora_params: List[torch.nn.Parameter],
    train_triplets: List[Tuple[str, str, str]],
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    qrels: Dict[str, Dict[str, int]],
    confused_qids: Set[str],
    easy_qids: Set[str],
    pair_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]],
    hn_cache: Dict[str, torch.Tensor],
    lambda_dir: float,
    lambda_neg: float,
    device: torch.device,
    cfg: TrainConfigLite,
    val_eval_kwargs: dict,
) -> TrainHistory:
    """Training with per-token cosine anchor extended to negative docs on easy queries."""
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    )
    train_subset, val_qids = _split_triplets_local(train_triplets, cfg.val_split, cfg.seed)
    logger.info("train: %d triplets (%d after val split), val_qids=%d",
                len(train_triplets), len(train_subset), len(val_qids))
    logger.info("λ_dir=%.3f λ_neg=%.3f |confused|=%d |easy|=%d in train",
                lambda_dir, lambda_neg,
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
        epoch_rank = epoch_dir = epoch_neg = 0.0
        n_rank_batches = n_anchor_batches = 0
        for batch in batches:
            qids = [t[0] for t in batch]
            pos_dids = [t[1] for t in batch]
            hn_dids = [t[2] for t in batch]
            q_texts = [queries[q] for q in qids]
            pos_texts = [doc_text(corpus[d]) for d in pos_dids]
            hn_texts = [doc_text(corpus[d]) for d in hn_dids]

            q_emb, _ = model.encode_queries(q_texts, device=device)
            pos_emb, pos_mask = model.encode_docs(pos_texts, device=device)
            # All HNs forwarded (needed for both confused rank loss and easy d^- anchor)
            hn_emb, hn_mask = model.encode_docs(hn_texts, device=device)

            confused_idx = [i for i, q in enumerate(qids) if q in confused_qids]
            easy_idx = [i for i, q in enumerate(qids) if q in easy_qids]

            # Confused queries: margin loss with mined HN
            if confused_idx:
                q_emb_c = q_emb[confused_idx]
                pos_emb_c = pos_emb[confused_idx]
                pos_mask_c = pos_mask[confused_idx]
                hn_emb_c = hn_emb[confused_idx]
                hn_mask_c = hn_mask[confused_idx]
                s_pos = model.diagonal_maxsim(q_emb_c, pos_emb_c, pos_mask_c)
                s_hn = model.diagonal_maxsim(q_emb_c, hn_emb_c, hn_mask_c)
                rank_loss = torch.clamp(cfg.margin - s_pos + s_hn, min=0).mean()
                epoch_rank += float(rank_loss.detach().item())
                n_rank_batches += 1
            else:
                rank_loss = torch.zeros((), device=device)

            # Easy queries: cosine direction anchor on {q, d^+, d^-}
            if easy_idx and (lambda_dir > 0 or lambda_neg > 0):
                dir_loss, neg_loss = cosine_deviation_loss_with_hn(
                    q_emb, pos_emb, pos_mask, hn_emb, hn_mask,
                    qids, pos_dids, hn_dids, easy_idx,
                    pair_cache, hn_cache, lambda_neg,
                )
                epoch_dir += float(dir_loss.detach().item())
                epoch_neg += float(neg_loss.detach().item())
                n_anchor_batches += 1
            else:
                dir_loss = torch.zeros((), device=device)
                neg_loss = torch.zeros((), device=device)

            loss = rank_loss + lambda_dir * dir_loss + lambda_neg * neg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            v_norm = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in lora_params)).item())
            history.steps.append(step)
            history.losses.append(float(loss.detach().item()))
            history.rank_losses.append(float(rank_loss.detach().item()))
            history.anchor_losses.append(float(dir_loss.detach().item() + neg_loss.detach().item()))
            history.v_norms.append(v_norm)
            history.epoch.append(epoch)

        logger.info("epoch %d/%d: rank=%.4f dir=%.4f neg=%.4f time=%.1fs",
                    epoch + 1, cfg.epochs,
                    epoch_rank / max(1, n_rank_batches),
                    epoch_dir / max(1, n_anchor_batches),
                    epoch_neg / max(1, n_anchor_batches),
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
    p.add_argument("--lambda-dir", type=float, default=1.0,
                   help="q + d^+ anchor strength (pre-commit 1.0 single value)")
    p.add_argument("--lambda-neg", type=float, default=1.0,
                   help="d^- anchor strength (pre-commit 1.0 single value, this experiment's contribution)")
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
    logger.info("dataset=%s seed=%d λ_dir=%.3f λ_neg=%.3f",
                args.dataset, args.seed, args.lambda_dir, args.lambda_neg)

    tag = f"qv_r{args.r}_l12_dir{args.lambda_dir:g}_neg{args.lambda_neg:g}"
    base_out = artifact_dir(exp_name="17_negative_side_anchor",
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
               "lambda_dir": args.lambda_dir, "lambda_neg": args.lambda_neg,
               "r": args.r, "alpha": args.alpha,
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
    if args.max_triplets and len(triplets) > args.max_triplets:
        _rng = random.Random(args.seed); _rng.shuffle(triplets)
        triplets = triplets[: args.max_triplets]
        logger.info("subsampled to %d triplets (seed=%d)", len(triplets), args.seed)

    # 3. Precompute frozen embeddings (BEFORE LoRA injection) — q, d^+, AND d^-
    pair_cache, hn_cache = precompute_frozen_embeddings(
        model, easy_train_qids, triplets, train_queries, train_corpus, device,
    )

    # 4. Inject LoRA
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

    # 5. Train
    history = train_with_neg_anchor(
        model, lora_params, triplets, train_queries, train_corpus, train_qrels,
        confused_train_qids, easy_train_qids, pair_cache, hn_cache,
        lambda_dir=args.lambda_dir, lambda_neg=args.lambda_neg,
        device=device, cfg=train_cfg,
        val_eval_kwargs={"doc_batch": args.doc_batch,
                         "query_batch": args.query_batch,
                         "doc_chunk": args.doc_chunk},
    )

    # Diagnostics
    A_norms, B_norms = [], []
    for i in range(0, len(lora_params), 2):
        A_norms.append(float(lora_params[i].detach().norm().item()))
        B_norms.append(float(lora_params[i + 1].detach().norm().item()))
    save_json({"A_norms_per_adapter": A_norms, "B_norms_per_adapter": B_norms,
               "A_norm_total": float(sum(a * a for a in A_norms) ** 0.5),
               "B_norm_total": float(sum(b * b for b in B_norms) ** 0.5),
               "lambda_dir": args.lambda_dir, "lambda_neg": args.lambda_neg},
              out / "lora_stats.json")
    save_json(dataclasses.asdict(history), out / "train_history.json")
    torch.save({"steering": steering.state_dict(),
                "lora": {f"adapter_{i}": p.detach().cpu() for i, p in enumerate(lora_params)}},
               out / "module_final.pt")

    del train_corpus, train_queries, train_qrels, triplets, train_topk, train_runs
    del pair_cache, hn_cache

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
    logger.info("=== Δ vs baseline (Exp 17 negative-side anchor) ===")
    for sn, dr in deltas.items():
        if "mean_delta_ndcg10" in dr:
            logger.info("  %-9s n=%d Δ=%+.4f [%+.4f,%+.4f]%s",
                        sn, dr["n"], dr["mean_delta_ndcg10"], dr["ci_lo"], dr["ci_hi"],
                        " ✓ positive" if dr.get("positive")
                        else " ✗ negative" if dr.get("negative") else "")
    logger.info("artifacts → %s", out)


if __name__ == "__main__":
    main()
