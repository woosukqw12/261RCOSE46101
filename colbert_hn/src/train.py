"""Training primitives for learnable steering modules.

ROADMAP.md 의 single-direction 단계+ 모든 학습 실험이 본 모듈을 공유한다. 02_final_layer_vector
이후 gate / multi-direction 확장 시 본 `train_steering` 의 hook 만 교체.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.colbert_hook import ColBERTv2
from src.data import doc_text
from src.evaluate import compute_metrics_trec, encode_corpus, score_queries
from src.hn_mining import Triplet
from src.slices import confused_slice
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfigLite:
    """02 의 minimal training config. DESIGN.md §4 의 풀 ExpConfig 의 학습 부분
    subset. λ_anc = 0 default — 02 가 gate 없는 single-direction 인 점 반영
    (DESIGN.md §11 참조)."""

    hook_layer: int = 12
    margin: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 5
    patience: int = 2
    val_split: float = 0.1
    lambda_anchor: float = 0.0
    seed: int = 42
    # Mediation 1 (warmup + grad_clip) — optional, default 0 = disabled (옛 동일 동작)
    warmup_frac: float = 0.0       # linear warmup over first warmup_frac × total_steps
    grad_clip_max_norm: float = 0.0  # max grad norm; 0 = no clip
    # Mediation 1b (in-batch negative) — optional, default False = mined HN (옛 동일 동작)
    in_batch_neg: bool = False


@dataclass
class TrainHistory:
    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    rank_losses: List[float] = field(default_factory=list)
    anchor_losses: List[float] = field(default_factory=list)
    v_norms: List[float] = field(default_factory=list)
    epoch: List[int] = field(default_factory=list)
    val_epochs: List[int] = field(default_factory=list)
    val_ndcg_all: List[float] = field(default_factory=list)
    val_ndcg_confused: List[float] = field(default_factory=list)


def _split_triplets(
    triplets: List[Triplet], val_frac: float, seed: int
) -> Tuple[List[Triplet], List[str]]:
    """Hold out `val_frac` of UNIQUE queries for validation. Returns
    (train_triplets, val_qids).

    Splitting by *query* (not triplet) avoids leakage where a val query has
    its own triplets in train."""
    by_q: Dict[str, List[Triplet]] = {}
    for t in triplets:
        by_q.setdefault(t[0], []).append(t)
    qids = sorted(by_q.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * val_frac))
    val_qids = set(qids[:n_val])
    train_triplets = [t for t in triplets if t[0] not in val_qids]
    return train_triplets, sorted(val_qids)


def _make_batches(
    triplets: List[Triplet], batch_size: int, seed: int, epoch: int
) -> List[List[Triplet]]:
    rng = random.Random(seed * 100003 + epoch)
    shuffled = list(triplets)
    rng.shuffle(shuffled)
    return [
        shuffled[i:i + batch_size]
        for i in range(0, len(shuffled), batch_size)
    ]


def _val_pass(
    model: ColBERTv2,
    val_qids: List[str],
    val_queries: Dict[str, str],
    val_qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, dict],
    device: torch.device,
    metrics_k: Tuple[int, ...] = (10,),
    doc_batch: int = 64,
    query_batch: int = 16,
    doc_chunk: int = 512,
    top_k: int = 100,
) -> Tuple[float, float]:
    """Run retrieval over `corpus` for `val_qids` with the current steering
    hook installed. Returns (NDCG@10 all, NDCG@10 confused). The hook MUST
    already be registered on `model` by the caller."""
    sub_queries = {q: val_queries[q] for q in val_qids if q in val_queries}
    sub_qrels = {q: val_qrels[q] for q in val_qids if q in val_qrels}

    with torch.no_grad():
        dids, d_emb, d_mask = encode_corpus(model, corpus, device, batch_size=doc_batch)
        topk = score_queries(
            model, sub_queries, dids, d_emb, d_mask, device,
            query_batch=query_batch, doc_chunk=doc_chunk, top_k=top_k,
        )
    runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
    runs_scored = {q: dict(lst) for q, lst in topk.items()}
    per_q = compute_metrics_trec(runs_scored, sub_qrels, metrics_k=metrics_k)
    if not per_q:
        return float("nan"), float("nan")
    all_avg = sum(v["ndcg_cut_10"] for v in per_q.values()) / len(per_q)
    conf = confused_slice(runs_ranked, sub_qrels, k=1)
    if not conf:
        return all_avg, float("nan")
    conf_avg = sum(per_q[q]["ndcg_cut_10"] for q in conf if q in per_q) / len(conf)
    return all_avg, conf_avg


def train_steering(
    model: ColBERTv2,
    steering: nn.Module,
    train_triplets: List[Triplet],
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    qrels: Dict[str, Dict[str, int]],
    device: torch.device,
    cfg: TrainConfigLite,
    val_eval_kwargs: Optional[dict] = None,
    extra_param_groups: Optional[List[dict]] = None,
    train_encoder: bool = False,
    early_stop_metric: str = "confused",
) -> TrainHistory:
    """Train `steering` via pairwise margin loss on `train_triplets`.

    `steering` must be already registered on `model` at `cfg.hook_layer` —
    the caller is responsible for that. We do not unregister at the end (the
    caller can decide whether to use the trained module for downstream eval).

    Anchor regularizer: `cfg.lambda_anchor` × ‖v‖². For 02 default
    (`lambda_anchor=0`), this term is skipped — see DESIGN.md §11.

    Args:
        extra_param_groups: optional additional optimizer param groups, e.g.
            for unfrozen ColBERT encoder fine-tune ([{'params': model.parameters(),
            'lr': 5e-5}]). Default None = train only steering params.
        train_encoder: if True, `model.bert.train()` + `model.linear.train()`
            inside each epoch (dropout active). Default False (eval mode,
            ColBERT 의 frozen 동작 유지).
    """
    param_groups: List[dict] = [{"params": list(steering.parameters()),
                                  "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    if extra_param_groups:
        param_groups.extend(extra_param_groups)
    optimizer = torch.optim.AdamW(param_groups)

    train_subset, val_qids = _split_triplets(
        train_triplets, val_frac=cfg.val_split, seed=cfg.seed
    )
    logger.info(
        "train: %d triplets (%d after val held-out), val_qids=%d",
        len(train_triplets), len(train_subset), len(val_qids),
    )

    # Mediation 1 — warmup scheduler 설정 (linear from 0 → 1 over warmup_steps)
    total_steps_estimate = max(1, math.ceil(len(train_subset) / cfg.batch_size)) * cfg.epochs
    warmup_steps = int(cfg.warmup_frac * total_steps_estimate)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    if warmup_steps > 0:
        logger.info(
            "warmup: linear 0 → LR over first %d / %d steps (%.1f%%)",
            warmup_steps, total_steps_estimate, 100 * cfg.warmup_frac,
        )
    if cfg.grad_clip_max_norm > 0:
        logger.info("grad-clip: max_norm = %.4f", cfg.grad_clip_max_norm)
    all_trainable: List[nn.Parameter] = []
    for pg in optimizer.param_groups:
        all_trainable.extend([p for p in pg["params"] if p.requires_grad])

    val_eval_kwargs = val_eval_kwargs or {}
    history = TrainHistory()
    step = 0
    best_val = -math.inf
    best_state: Optional[dict] = None
    epochs_since_best = 0

    for epoch in range(cfg.epochs):
        steering.train()
        if train_encoder:
            model.bert.train()
            model.linear.train()
        else:
            model.bert.eval()
            model.linear.eval()
        batches = _make_batches(train_subset, cfg.batch_size, cfg.seed, epoch)
        t_epoch = time.time()
        epoch_loss = 0.0
        epoch_rank = 0.0
        epoch_anc = 0.0
        for batch in batches:
            qids = [t[0] for t in batch]
            pos_dids = [t[1] for t in batch]
            hn_dids = [t[2] for t in batch]

            q_texts = [queries[q] for q in qids]
            pos_texts = [doc_text(corpus[d]) for d in pos_dids]

            q_emb, _ = model.encode_queries(q_texts, device=device)
            pos_emb, pos_mask = model.encode_docs(pos_texts, device=device)

            # Mediation 1b — in-batch negative: 다른 query 의 positive 를 negative 로
            # roll(1, dims=0) → 이웃 query 의 positive. shuffle 된 batches 라 random.
            # mined HN 의 noise (~50% irrelevant) 회피, supervision-quality root 검정.
            if cfg.in_batch_neg:
                if pos_emb.size(0) < 2:
                    continue  # batch_size=1 일 때 skip (다른 query 없음)
                hn_emb = pos_emb.roll(1, dims=0)
                hn_mask = pos_mask.roll(1, dims=0)
            else:
                hn_texts = [doc_text(corpus[d]) for d in hn_dids]
                hn_emb, hn_mask = model.encode_docs(hn_texts, device=device)

            s_pos = model.diagonal_maxsim(q_emb, pos_emb, pos_mask)
            s_hn = model.diagonal_maxsim(q_emb, hn_emb, hn_mask)

            rank_loss = torch.clamp(cfg.margin - s_pos + s_hn, min=0).mean()
            if cfg.lambda_anchor > 0:
                # multi-parameter aware: sum of L2 squared over all trainable params
                anc_loss = cfg.lambda_anchor * sum(
                    (p ** 2).sum() for p in steering.parameters()
                )
                loss = rank_loss + anc_loss
            else:
                anc_loss = torch.zeros((), device=device)
                loss = rank_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Mediation 1 — grad clipping (backward 후, step 전)
            if cfg.grad_clip_max_norm > 0 and all_trainable:
                torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip_max_norm)

            # Mediation 1 — warmup LR multiplier (step < warmup_steps 일 때 < 1)
            if warmup_steps > 0 and step < warmup_steps:
                warmup_mult = (step + 1) / warmup_steps
                for pg, base in zip(optimizer.param_groups, base_lrs):
                    pg["lr"] = base * warmup_mult
            elif warmup_steps > 0 and step == warmup_steps:
                for pg, base in zip(optimizer.param_groups, base_lrs):
                    pg["lr"] = base

            optimizer.step()

            step += 1
            # Track *total* ‖params‖ across all trainable parameters of the module;
            # for single-direction modules this equals ‖v‖, for multi-layer modules
            # this is √(Σ_ℓ ‖v_ℓ‖²).
            v_norm = float(
                torch.sqrt(sum((p.detach() ** 2).sum() for p in steering.parameters())).item()
            )
            history.steps.append(step)
            history.losses.append(float(loss.detach().item()))
            history.rank_losses.append(float(rank_loss.detach().item()))
            history.anchor_losses.append(float(anc_loss.detach().item()))
            history.v_norms.append(v_norm)
            history.epoch.append(epoch)
            epoch_loss += float(loss.detach().item())
            epoch_rank += float(rank_loss.detach().item())
            epoch_anc += float(anc_loss.detach().item())

        n_batches = max(len(batches), 1)
        logger.info(
            "epoch %d/%d: loss=%.4f (rank=%.4f anc=%.4f) ‖v‖=%.4f time=%.1fs",
            epoch + 1, cfg.epochs,
            epoch_loss / n_batches, epoch_rank / n_batches, epoch_anc / n_batches,
            v_norm, time.time() - t_epoch,
        )

        steering.eval()
        t_val = time.time()
        val_all, val_conf = _val_pass(
            model, val_qids, queries, qrels, corpus, device, **val_eval_kwargs,
        )
        logger.info(
            "  val: NDCG@10 all=%.4f confused=%.4f (time=%.1fs)",
            val_all, val_conf, time.time() - t_val,
        )
        history.val_epochs.append(epoch + 1)
        history.val_ndcg_all.append(val_all)
        history.val_ndcg_confused.append(val_conf)

        if early_stop_metric == "all":
            score = val_all if not math.isnan(val_all) else val_conf
        else:  # default "confused" — historical behavior
            score = val_conf if not math.isnan(val_conf) else val_all
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().clone().cpu() for k, v in steering.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                logger.info("early stop at epoch %d (patience %d, metric=%s)",
                            epoch + 1, cfg.patience, early_stop_metric)
                break

    if best_state is not None:
        steering.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info("restored best state (val score=%.4f)", best_val)
    return history


# ============================================================================
# Bilinear metric training — Stage 2 (08_bilinear_M_minimal+) main novelty
# ============================================================================
# 별도 함수로 분리 (translation-family 의 layer-wise hook 대신 *MaxSim 의 metric*
# 을 직접 학습). 동일한 pairwise margin loss + val/early stop 인프라 공유.


def _bilinear_val_pass(
    model: ColBERTv2,
    metric,  # BilinearMetric (avoid circular import)
    val_qids: List[str],
    val_queries: Dict[str, str],
    val_qrels: Dict[str, Dict[str, int]],
    corpus: Dict[str, dict],
    device: torch.device,
    metrics_k: Tuple[int, ...] = (10,),
    doc_batch: int = 64,
    query_batch: int = 16,
    doc_chunk: int = 512,
    top_k: int = 100,
) -> Tuple[float, float]:
    """Bilinear-metric variant of `_val_pass`. Computes retrieval scores using
    `metric.maxsim` instead of `model.maxsim`. Returns (val_all, val_conf)."""
    sub_queries = {q: val_queries[q] for q in val_qids if q in val_queries}
    sub_qrels = {q: val_qrels[q] for q in val_qids if q in val_qrels}

    with torch.no_grad():
        dids, d_emb, d_mask = encode_corpus(model, corpus, device, batch_size=doc_batch)
        topk = _bilinear_score_queries(
            model, metric, sub_queries, dids, d_emb, d_mask, device,
            query_batch=query_batch, doc_chunk=doc_chunk, top_k=top_k,
        )
    runs_ranked = {q: [d for d, _ in lst] for q, lst in topk.items()}
    runs_scored = {q: dict(lst) for q, lst in topk.items()}
    per_q = compute_metrics_trec(runs_scored, sub_qrels, metrics_k=metrics_k)
    if not per_q:
        return float("nan"), float("nan")
    all_avg = sum(v["ndcg_cut_10"] for v in per_q.values()) / len(per_q)
    conf = confused_slice(runs_ranked, sub_qrels, k=1)
    if not conf:
        return all_avg, float("nan")
    conf_avg = sum(per_q[q]["ndcg_cut_10"] for q in conf if q in per_q) / len(conf)
    return all_avg, conf_avg


def _bilinear_score_queries(
    model: ColBERTv2,
    metric,  # BilinearMetric
    queries: Dict[str, str],
    dids: List[str],
    d_emb: torch.Tensor,
    d_mask: torch.Tensor,
    device: torch.device,
    query_batch: int = 16,
    doc_chunk: int = 512,
    top_k: int = 100,
    exclude_self: bool = False,
):
    """Bilinear MaxSim variant of `evaluate.score_queries`. Uses
    `metric.maxsim` to apply M = I + U V^T."""
    from tqdm.auto import tqdm
    qids = list(queries.keys())
    q_texts = [queries[q] for q in qids]
    n = d_emb.size(0)
    did_to_idx = {d: i for i, d in enumerate(dids)} if exclude_self else None
    out: Dict[str, List[Tuple[str, float]]] = {}
    for q_start in tqdm(range(0, len(qids), query_batch), desc="bilinear_score_queries"):
        batch_qids = qids[q_start:q_start + query_batch]
        batch_texts = q_texts[q_start:q_start + query_batch]
        q_emb, _ = model.encode_queries(batch_texts, device=device)
        scores = torch.zeros(q_emb.size(0), n)
        for d_start in range(0, n, doc_chunk):
            d_end = min(d_start + doc_chunk, n)
            s = metric.maxsim(
                q_emb,
                d_emb[d_start:d_end].to(device),
                d_mask[d_start:d_end].to(device),
            )
            scores[:, d_start:d_end] = s.cpu()
        if exclude_self and did_to_idx is not None:
            for i, qid in enumerate(batch_qids):
                self_idx = did_to_idx.get(qid)
                if self_idx is not None:
                    scores[i, self_idx] = float("-inf")
        top_vals, top_idx = scores.topk(min(top_k, n), dim=-1)
        for i, qid in enumerate(batch_qids):
            out[qid] = [
                (dids[j], float(v))
                for j, v in zip(top_idx[i].tolist(), top_vals[i].tolist())
            ]
    return out


def train_bilinear_metric(
    model: ColBERTv2,
    metric,  # BilinearMetric
    train_triplets: List[Triplet],
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    qrels: Dict[str, Dict[str, int]],
    device: torch.device,
    cfg: TrainConfigLite,
    val_eval_kwargs: Optional[dict] = None,
) -> TrainHistory:
    """Train `metric` (BilinearMetric) via pairwise margin loss.

    Frozen ColBERT 의 forward 는 그대로 (no hook). 각 step 에서 q / pos / hn
    의 token embedding 을 frozen 모델로 추출 후 `metric.diagonal_maxsim` 으로
    pairwise score 계산. metric.U, metric.V 만 학습.

    `cfg.hook_layer` 는 *무시* (bilinear M 은 hook 기반 아님). 다른 hyper-
    parameter (margin / lr / epochs / patience / val_split / λ_anc) 는 동일."""
    optimizer = torch.optim.AdamW(
        metric.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_subset, val_qids = _split_triplets(
        train_triplets, val_frac=cfg.val_split, seed=cfg.seed
    )
    logger.info(
        "train: %d triplets (%d after val held-out), val_qids=%d",
        len(train_triplets), len(train_subset), len(val_qids),
    )

    val_eval_kwargs = val_eval_kwargs or {}
    history = TrainHistory()
    step = 0
    best_val = -math.inf
    best_state: Optional[dict] = None
    epochs_since_best = 0

    for epoch in range(cfg.epochs):
        metric.train()
        model.bert.eval()
        model.linear.eval()
        batches = _make_batches(train_subset, cfg.batch_size, cfg.seed, epoch)
        t_epoch = time.time()
        epoch_loss = 0.0
        epoch_rank = 0.0
        epoch_anc = 0.0
        v_norm = 0.0
        for batch in batches:
            qids = [t[0] for t in batch]
            pos_dids = [t[1] for t in batch]
            hn_dids = [t[2] for t in batch]

            q_texts = [queries[q] for q in qids]
            pos_texts = [doc_text(corpus[d]) for d in pos_dids]
            hn_texts = [doc_text(corpus[d]) for d in hn_dids]

            q_emb, _ = model.encode_queries(q_texts, device=device)
            pos_emb, pos_mask = model.encode_docs(pos_texts, device=device)
            hn_emb, hn_mask = model.encode_docs(hn_texts, device=device)

            s_pos = metric.diagonal_maxsim(q_emb, pos_emb, pos_mask)
            s_hn = metric.diagonal_maxsim(q_emb, hn_emb, hn_mask)

            rank_loss = torch.clamp(cfg.margin - s_pos + s_hn, min=0).mean()
            if cfg.lambda_anchor > 0:
                anc_loss = cfg.lambda_anchor * sum(
                    (p ** 2).sum() for p in metric.parameters()
                )
                loss = rank_loss + anc_loss
            else:
                anc_loss = torch.zeros((), device=device)
                loss = rank_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            v_norm = float(
                torch.sqrt(sum((p.detach() ** 2).sum() for p in metric.parameters())).item()
            )
            history.steps.append(step)
            history.losses.append(float(loss.detach().item()))
            history.rank_losses.append(float(rank_loss.detach().item()))
            history.anchor_losses.append(float(anc_loss.detach().item()))
            history.v_norms.append(v_norm)
            history.epoch.append(epoch)
            epoch_loss += float(loss.detach().item())
            epoch_rank += float(rank_loss.detach().item())
            epoch_anc += float(anc_loss.detach().item())

        n_batches = max(len(batches), 1)
        logger.info(
            "epoch %d/%d: loss=%.4f (rank=%.4f anc=%.4f) ‖[U;V]‖=%.4f time=%.1fs",
            epoch + 1, cfg.epochs,
            epoch_loss / n_batches, epoch_rank / n_batches, epoch_anc / n_batches,
            v_norm, time.time() - t_epoch,
        )

        metric.eval()
        t_val = time.time()
        val_all, val_conf = _bilinear_val_pass(
            model, metric, val_qids, queries, qrels, corpus, device,
            **val_eval_kwargs,
        )
        logger.info(
            "  val: NDCG@10 all=%.4f confused=%.4f (time=%.1fs)",
            val_all, val_conf, time.time() - t_val,
        )
        history.val_epochs.append(epoch + 1)
        history.val_ndcg_all.append(val_all)
        history.val_ndcg_confused.append(val_conf)

        score = val_conf if not math.isnan(val_conf) else val_all
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().clone().cpu() for k, v in metric.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, cfg.patience)
                break

    if best_state is not None:
        metric.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info("restored best state (val score=%.4f)", best_val)
    return history


# ============================================================================
# Bilinear metric training with E5-Mistral margin distillation — 09 +
# ============================================================================
# 09_bilinear_M_e5_distill 등. train_bilinear_metric 의 super-set —
# pairwise margin loss 에 *teacher margin* (E5-Mistral cosine margin) 의
# Margin-MSE distillation term 추가.


def train_bilinear_metric_distill(
    model: ColBERTv2,
    metric,  # BilinearMetric
    train_triplets: List[Triplet],
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    qrels: Dict[str, Dict[str, int]],
    device: torch.device,
    cfg: TrainConfigLite,
    e5_qid_to_idx: Dict[str, int],
    e5_did_to_idx: Dict[str, int],
    e5_q_emb: torch.Tensor,   # (N_q, D_e5) fp16 on CPU, L2-normalized
    e5_d_emb: torch.Tensor,   # (N_d, D_e5) fp16 on CPU, L2-normalized
    lambda_distill: float = 1.0,
    teacher_scale: float = 8.0,
    val_eval_kwargs: Optional[dict] = None,
) -> TrainHistory:
    """train_bilinear_metric + Margin-MSE distillation.

    Loss per batch:
        L = clamp(margin - (s_pos - s_hn), 0).mean()
            + lambda_distill * MSE(s_pos - s_hn, teacher_scale * e5_margin)

    `e5_margin` = e5_q · e5_d_pos - e5_q · e5_d_hn  (cosine since L2-norm).
    `teacher_scale` brings E5 cosine margin (~0.03 magnitude) up to ColBERT
    pairwise margin (~0.2 magnitude) scale — fixed value tunable per dataset.

    e5_q_emb, e5_d_emb 는 fp16 on CPU (메모리 절약). 매 batch 에서 lookup +
    GPU 로 transfer + cosine 계산.
    """
    optimizer = torch.optim.AdamW(
        metric.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    train_subset, val_qids = _split_triplets(
        train_triplets, val_frac=cfg.val_split, seed=cfg.seed,
    )
    logger.info(
        "train: %d triplets (%d after val held-out), val_qids=%d, "
        "lambda_distill=%.3f, teacher_scale=%.3f",
        len(train_triplets), len(train_subset), len(val_qids),
        lambda_distill, teacher_scale,
    )

    val_eval_kwargs = val_eval_kwargs or {}
    history = TrainHistory()
    step = 0
    best_val = -math.inf
    best_state: Optional[dict] = None
    epochs_since_best = 0

    for epoch in range(cfg.epochs):
        metric.train()
        model.bert.eval()
        model.linear.eval()
        batches = _make_batches(train_subset, cfg.batch_size, cfg.seed, epoch)
        t_epoch = time.time()
        epoch_loss = 0.0
        epoch_rank = 0.0
        epoch_distill = 0.0
        v_norm = 0.0
        for batch in batches:
            qids = [t[0] for t in batch]
            pos_dids = [t[1] for t in batch]
            hn_dids = [t[2] for t in batch]

            # Teacher margin per triplet (no grad, fp16 → fp32)
            try:
                qi = torch.tensor(
                    [e5_qid_to_idx[q] for q in qids], dtype=torch.long,
                )
                pi = torch.tensor(
                    [e5_did_to_idx[d] for d in pos_dids], dtype=torch.long,
                )
                hi = torch.tensor(
                    [e5_did_to_idx[d] for d in hn_dids], dtype=torch.long,
                )
            except KeyError as e:
                # Skip batch if some id missing from E5 cache (shouldn't happen
                # after pre-mining filter)
                logger.warning("skip batch — E5 id missing: %s", e)
                continue
            with torch.no_grad():
                eq = e5_q_emb[qi].to(device).to(torch.float32)
                ep = e5_d_emb[pi].to(device).to(torch.float32)
                eh = e5_d_emb[hi].to(device).to(torch.float32)
                t_pos = (eq * ep).sum(dim=-1)   # (B,) cosine since L2-normed
                t_hn = (eq * eh).sum(dim=-1)
                teacher_margin = (t_pos - t_hn) * teacher_scale  # (B,)

            # Student
            q_texts = [queries[q] for q in qids]
            pos_texts = [doc_text(corpus[d]) for d in pos_dids]
            hn_texts = [doc_text(corpus[d]) for d in hn_dids]

            q_emb, _ = model.encode_queries(q_texts, device=device)
            pos_emb, pos_mask = model.encode_docs(pos_texts, device=device)
            hn_emb, hn_mask = model.encode_docs(hn_texts, device=device)

            s_pos = metric.diagonal_maxsim(q_emb, pos_emb, pos_mask)
            s_hn = metric.diagonal_maxsim(q_emb, hn_emb, hn_mask)
            student_margin = s_pos - s_hn  # (B,)

            rank_loss = torch.clamp(cfg.margin - student_margin, min=0).mean()
            distill_loss = (
                ((student_margin - teacher_margin) ** 2).mean()
                if lambda_distill > 0
                else torch.zeros((), device=device)
            )

            loss = rank_loss + lambda_distill * distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            v_norm = float(
                torch.sqrt(sum((p.detach() ** 2).sum() for p in metric.parameters())).item()
            )
            history.steps.append(step)
            history.losses.append(float(loss.detach().item()))
            history.rank_losses.append(float(rank_loss.detach().item()))
            history.anchor_losses.append(float(distill_loss.detach().item()))  # repurpose
            history.v_norms.append(v_norm)
            history.epoch.append(epoch)
            epoch_loss += float(loss.detach().item())
            epoch_rank += float(rank_loss.detach().item())
            epoch_distill += float(distill_loss.detach().item())

        n_batches = max(len(batches), 1)
        logger.info(
            "epoch %d/%d: loss=%.4f (rank=%.4f distill=%.4f) ‖[U;V]‖=%.4f time=%.1fs",
            epoch + 1, cfg.epochs,
            epoch_loss / n_batches, epoch_rank / n_batches, epoch_distill / n_batches,
            v_norm, time.time() - t_epoch,
        )

        metric.eval()
        t_val = time.time()
        val_all, val_conf = _bilinear_val_pass(
            model, metric, val_qids, queries, qrels, corpus, device,
            **val_eval_kwargs,
        )
        logger.info(
            "  val: NDCG@10 all=%.4f confused=%.4f (time=%.1fs)",
            val_all, val_conf, time.time() - t_val,
        )
        history.val_epochs.append(epoch + 1)
        history.val_ndcg_all.append(val_all)
        history.val_ndcg_confused.append(val_conf)

        score = val_conf if not math.isnan(val_conf) else val_all
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().clone().cpu() for k, v in metric.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, cfg.patience)
                break

    if best_state is not None:
        metric.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info("restored best state (val score=%.4f)", best_val)
    return history
