"""
Gradient Conflict Gating (GCG) for Dense Retriever Training
============================================================

Core idea:
  In contrastive learning, the loss gradient w.r.t. the query representation
  decomposes into a "pull-positive" component and per-negative "push-negative"
  components. For a *true* hard negative, its push direction should be
  orthogonal or opposed to the pull direction. For a *false* negative (actually
  relevant), the push direction aligns with the pull direction — a gradient
  conflict. We detect this conflict and gate (suppress) the offending negative's
  contribution to the loss.

  Mathematically, for L2-normalised embeddings q, p, n_k the pull/push
  directions projected into the tangent plane at q are:

      g_pos   = p  - (q·p)  q        (direction positive wants to pull q)
      g_neg_k = n_k - (q·n_k) q      (direction negative k pushes q)

  conflict_k = cos(g_pos, g_neg_k)

  High conflict → likely false negative → downweight in loss.

Usage:
  python train_gcg.py \
    --dataset_dir ./data/processed/scifact \
    --output_dir  ./outputs/bge/scifact_gcg \
    --model_type  bge \
    --model_name  BAAI/bge-base-en-v1.5 \
    --epochs 5 \
    --batch_size 16 \
    --neg_per_query 8 \
    --dynamic_refresh \
    --gate_method adaptive \
    --percentile 80 \
    --sharpness 20 \
    --warmup_epochs 1
"""

import argparse
import heapq
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    DPRContextEncoder,
    DPRQuestionEncoder,
)


# =============================================================================
# Utilities
# =============================================================================

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_queries(path: Path):
    rows = load_jsonl(path)
    return {str(x["qid"]): x["query"] for x in rows}


def load_qrels(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_collection(path: Path):
    rows = load_jsonl(path)
    out = []
    for row in rows:
        pid = str(row["pid"])
        text = ((row.get("title", "") or "") + " " + (row.get("text", "") or "")).strip()
        out.append((pid, text))
    return out


def dcg(rels):
    s = 0.0
    for i, rel in enumerate(rels):
        s += (2 ** rel - 1) / math.log2(i + 2)
    return s


def evaluate_run(qrels, results, k=10):
    ndcg_scores, mrr_scores, recall_scores = [], [], []

    for qid, rel_docs in qrels.items():
        positives = {pid: rel for pid, rel in rel_docs.items() if rel > 0}
        if not positives:
            continue

        ranked = sorted(results.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        ranked_ids = [pid for pid, _ in ranked]

        topk = ranked_ids[:k]
        rels = [positives.get(pid, 0) for pid in topk]
        ideal = sorted(positives.values(), reverse=True)[:k]
        ndcg_scores.append(dcg(rels) / max(dcg(ideal), 1e-12))

        rr = 0.0
        for i, pid in enumerate(topk):
            if positives.get(pid, 0) > 0:
                rr = 1.0 / (i + 1)
                break
        mrr_scores.append(rr)

        top100 = set(ranked_ids[:100])
        recall_scores.append(sum(1 for pid in positives if pid in top100) / max(len(positives), 1))

    return {
        "nDCG@10": sum(ndcg_scores) / max(len(ndcg_scores), 1),
        "MRR@10": sum(mrr_scores) / max(len(mrr_scores), 1),
        "Recall@100": sum(recall_scores) / max(len(recall_scores), 1),
    }


# =============================================================================
# Dataset
# =============================================================================

class TrainDataset(Dataset):
    def __init__(self, path: Path, neg_per_query: int = 8, seed: int = 42):
        self.rows = load_jsonl(path)
        self.neg_per_query = neg_per_query
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def refresh_rows(self, rows):
        self.rows = rows

    def __getitem__(self, idx):
        row = self.rows[idx]
        pos_idx = self.rng.randrange(len(row["positives"]))
        negs = row["negatives"]
        neg_ids = row["neg_ids"]

        if len(negs) > self.neg_per_query:
            chosen = self.rng.sample(range(len(negs)), self.neg_per_query)
            negs = [negs[i] for i in chosen]
            neg_ids = [neg_ids[i] for i in chosen]

        return {
            "qid": str(row["qid"]),
            "query": row["query"],
            "pos_id": str(row["pos_ids"][pos_idx]),
            "positive": row["positives"][pos_idx],
            "neg_ids": [str(x) for x in neg_ids],
            "negatives": negs,
        }


def collate_fn(batch):
    qids = [x["qid"] for x in batch]
    queries = [x["query"] for x in batch]
    pos_ids = [x["pos_id"] for x in batch]
    positives = [x["positive"] for x in batch]
    neg_ids = [x["neg_ids"] for x in batch]
    negatives = [x["negatives"] for x in batch]

    K = max(len(x) for x in neg_ids)
    for i in range(len(batch)):
        if len(neg_ids[i]) < K:
            pad_n = K - len(neg_ids[i])
            neg_ids[i] = neg_ids[i] + neg_ids[i][-1:] * pad_n
            negatives[i] = negatives[i] + negatives[i][-1:] * pad_n

    return {
        "qids": qids,
        "queries": queries,
        "pos_ids": pos_ids,
        "positives": positives,
        "neg_ids": neg_ids,
        "negatives": negatives,
        "K": K,
    }


# =============================================================================
# Model wrappers
# =============================================================================

class BGEBiEncoder(nn.Module):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.q_tok = AutoTokenizer.from_pretrained(model_name)
        self.d_tok = AutoTokenizer.from_pretrained(model_name)

    def _encode(self, tokenizer, texts, device, max_length):
        toks = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.model(**toks)
        hidden = out.last_hidden_state[:, 0]          # (B, D) pre-norm
        emb = F.normalize(hidden, dim=-1)              # (B, D) L2 normalised
        return emb, hidden

    def encode_query(self, texts, device, max_length=64):
        emb, _ = self._encode(self.q_tok, texts, device, max_length)
        return emb

    def encode_doc(self, texts, device, max_length=180):
        emb, _ = self._encode(self.d_tok, texts, device, max_length)
        return emb

    def encode_query_with_hidden(self, texts, device, max_length=64):
        return self._encode(self.q_tok, texts, device, max_length)

    def encode_doc_with_hidden(self, texts, device, max_length=180):
        return self._encode(self.d_tok, texts, device, max_length)

    def parameters_to_optimize(self):
        return self.model.parameters()

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.q_tok.save_pretrained(out_dir)

    @property
    def train_module(self):
        return self.model


class DPRBiEncoder(nn.Module):
    def __init__(
        self,
        q_model_name="facebook/dpr-question_encoder-single-nq-base",
        d_model_name="facebook/dpr-ctx_encoder-single-nq-base",
    ):
        super().__init__()
        self.q_model = DPRQuestionEncoder.from_pretrained(q_model_name)
        self.d_model = DPRContextEncoder.from_pretrained(d_model_name)
        self.q_tok = AutoTokenizer.from_pretrained(q_model_name)
        self.d_tok = AutoTokenizer.from_pretrained(d_model_name)

    def encode_query(self, texts, device, max_length=64):
        toks = self.q_tok(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        hidden = self.q_model(**toks).pooler_output
        return F.normalize(hidden, dim=-1)

    def encode_doc(self, texts, device, max_length=180):
        toks = self.d_tok(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        hidden = self.d_model(**toks).pooler_output
        return F.normalize(hidden, dim=-1)

    def encode_query_with_hidden(self, texts, device, max_length=64):
        toks = self.q_tok(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        hidden = self.q_model(**toks).pooler_output
        return F.normalize(hidden, dim=-1), hidden

    def encode_doc_with_hidden(self, texts, device, max_length=180):
        toks = self.d_tok(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        hidden = self.d_model(**toks).pooler_output
        return F.normalize(hidden, dim=-1), hidden

    def parameters_to_optimize(self):
        return list(self.q_model.parameters()) + list(self.d_model.parameters())

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "query_encoder").mkdir(parents=True, exist_ok=True)
        (out_dir / "doc_encoder").mkdir(parents=True, exist_ok=True)
        self.q_model.save_pretrained(out_dir / "query_encoder")
        self.q_tok.save_pretrained(out_dir / "query_encoder")
        self.d_model.save_pretrained(out_dir / "doc_encoder")
        self.d_tok.save_pretrained(out_dir / "doc_encoder")

    @property
    def train_module(self):
        return self.q_model


# =============================================================================
# Gradient Conflict Gater (GCG)  — the core contribution
# =============================================================================

@dataclass
class GCGConfig:
    temperature: float = 0.02

    # --- conflict → weight mapping -------------------------------------------
    gate_method: str = "adaptive"     # "adaptive" (percentile) or "fixed"
    fixed_threshold: float = 0.3      # used when gate_method == "fixed"
    percentile: float = 80.0          # suppress top-X% conflicting negatives
    sharpness: float = 20.0           # sigmoid steepness (α)

    # --- warmup --------------------------------------------------------------
    warmup_epochs: int = 1            # ramp gating from 0→1 over N epochs

    # --- conflict mode -------------------------------------------------------
    #   "closed_form"  — analytic tangent-plane projection (efficient, default)
    #   "autograd"     — torch.autograd.grad through encoder (general, slower)
    conflict_mode: str = "closed_form"

    # --- EMA for adaptive threshold ------------------------------------------
    threshold_ema: float = 0.95

    # --- π_k weighting (effective conflict = π_k · ReLU(conflict)) -----------
    use_pi_weighting: bool = False

    # --- absolute floor for adaptive threshold --------------------------------
    tau_min: float = -1.0             # < 0 means disabled


class GradientConflictGater:
    """
    Detects false negatives by measuring gradient conflict between the
    positive-pull and negative-push directions in embedding space, then
    gates (downweights) suspected false negatives in the contrastive loss.
    """

    def __init__(self, cfg: GCGConfig):
        self.cfg = cfg
        self.epoch = 0
        self.global_step = 0
        self.running_threshold = 0.0
        self._threshold_initialised = False

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    # --------------------------------------------------------------------- #
    #  Conflict score computation                                             #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def _conflict_closed_form(
        self,
        q_emb: torch.Tensor,   # (B, D) normalised
        p_emb: torch.Tensor,   # (B, D) normalised
        n_emb: torch.Tensor,   # (B, K, D) normalised
    ) -> torch.Tensor:
        """
        Closed-form conflict score.

        The gradient of InfoNCE w.r.t. the pre-normalisation query hidden
        state h_q decomposes into pull/push components that, after the
        Jacobian of L2 normalisation (I − qqᵀ)/‖h‖, become:

            g_pos   ∝ (I − qqᵀ) p   = p  − (q·p)  q
            g_neg_k ∝ (I − qqᵀ) n_k = n_k − (q·n_k) q

        conflict_k = cos(g_pos, g_neg_k)
        """
        # tangent-plane projections
        qp = (q_emb * p_emb).sum(dim=-1, keepdim=True)          # (B, 1)
        g_pos = p_emb - qp * q_emb                              # (B, D)

        qn = torch.einsum("bd,bkd->bk", q_emb, n_emb)          # (B, K)
        g_neg = n_emb - qn.unsqueeze(-1) * q_emb.unsqueeze(1)   # (B, K, D)

        # cosine similarity
        g_pos_n = F.normalize(g_pos, dim=-1).unsqueeze(1)        # (B, 1, D)
        g_neg_n = F.normalize(g_neg, dim=-1)                     # (B, K, D)

        conflict = (g_pos_n * g_neg_n).sum(dim=-1)               # (B, K)
        return conflict

    def _conflict_autograd(
        self,
        q_hidden: torch.Tensor,   # (B, D) pre-norm, requires_grad
        q_emb: torch.Tensor,      # (B, D) normalised (from q_hidden)
        p_emb: torch.Tensor,      # (B, D) normalised
        n_emb: torch.Tensor,      # (B, K, D) normalised
        temperature: float,
    ) -> torch.Tensor:
        """
        Autograd-based conflict score.
        Computes actual gradients ∂L_pos/∂h_q and ∂L_neg_k/∂h_q via autograd.
        More general but slower.
        """
        B, K, D = n_emb.shape

        # positive pull direction: ∂(q·p)/∂h_q  (no negation — matches closed-form)
        # closed-form: g_pos = p - (q·p)q ∝ ∂(q·p)/∂h via normalisation Jacobian
        # the old code used ∂(−q·p)/∂h which flipped the sign → conflict went negative
        pos_score = (q_emb * p_emb).sum(dim=-1) / temperature   # (B,)
        g_pos = torch.autograd.grad(
            pos_score.sum(), q_hidden,
            retain_graph=True, create_graph=False,
        )[0].detach()                                            # (B, D)

        # per-negative loss components: (q·n_k)/τ
        conflicts = []
        for k in range(K):
            neg_score_k = (q_emb * n_emb[:, k]).sum(dim=-1) / temperature
            g_neg_k = torch.autograd.grad(
                neg_score_k.sum(), q_hidden,
                retain_graph=True, create_graph=False,
            )[0].detach()                                        # (B, D)

            cos = F.cosine_similarity(g_pos, g_neg_k, dim=-1)   # (B,)
            conflicts.append(cos)

        return torch.stack(conflicts, dim=1)                     # (B, K)

    def compute_conflict(self, q_emb, p_emb, n_emb,
                         q_hidden=None, temperature=None):
        if self.cfg.conflict_mode == "autograd":
            assert q_hidden is not None, "autograd mode needs q_hidden"
            return self._conflict_autograd(
                q_hidden, q_emb, p_emb, n_emb,
                temperature or self.cfg.temperature,
            )
        else:
            return self._conflict_closed_form(q_emb, p_emb, n_emb)

    # --------------------------------------------------------------------- #
    #  Gating weights                                                         #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def compute_pi(
        self,
        q_emb: torch.Tensor,   # (B, D)
        p_emb: torch.Tensor,   # (B, D)
        n_emb: torch.Tensor,   # (B, K, D)
        temperature: float,
    ) -> torch.Tensor:
        """
        Softmax posterior π_k = exp(s(q,n_k)/τ) / [exp(s(q,p)/τ) + Σ exp(s(q,n_j)/τ)]
        This is the fraction of the loss gradient attributable to negative k.
        """
        pos_logit = (q_emb * p_emb).sum(dim=-1, keepdim=True) / temperature  # (B, 1)
        neg_logits = torch.einsum("bd,bkd->bk", q_emb, n_emb) / temperature  # (B, K)
        all_logits = torch.cat([pos_logit, neg_logits], dim=-1)               # (B, K+1)
        all_probs = F.softmax(all_logits, dim=-1)                              # (B, K+1)
        pi_k = all_probs[:, 1:]                                               # (B, K)
        return pi_k

    def compute_weights(
        self,
        conflict: torch.Tensor,
        pi_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Maps conflict scores → per-negative loss weights in [0, 1].

        If pi_k is provided (use_pi_weighting=True):
            effective_conflict = pi_k * ReLU(conflict)
        Else:
            effective_conflict = conflict  (original GCG)

        Returns:
            weights: (B, K)
            stats:   dict with monitoring info
        """
        B, K = conflict.shape

        # ---- effective conflict (optionally π-weighted) ---------------------- #
        if self.cfg.use_pi_weighting and pi_k is not None:
            effective = pi_k * torch.relu(conflict)
        else:
            effective = conflict

        # ---- adaptive or fixed threshold --------------------------------- #
        with torch.no_grad():
            if self.cfg.gate_method == "adaptive":
                batch_threshold = torch.quantile(
                    effective.flatten(),
                    self.cfg.percentile / 100.0,
                ).item()
                if not self._threshold_initialised:
                    self.running_threshold = batch_threshold
                    self._threshold_initialised = True
                else:
                    self.running_threshold = (
                        self.cfg.threshold_ema * self.running_threshold
                        + (1 - self.cfg.threshold_ema) * batch_threshold
                    )
                threshold = self.running_threshold

                # ---- absolute floor τ_min -------------------------------- #
                if self.cfg.tau_min >= 0:
                    threshold = max(threshold, self.cfg.tau_min)
            else:
                threshold = self.cfg.fixed_threshold

        # ---- sigmoid gating ---------------------------------------------- #
        gate = torch.sigmoid(
            -self.cfg.sharpness * (effective - threshold)
        )  # (B, K)

        # ---- warmup blend ------------------------------------------------ #
        if self.cfg.warmup_epochs > 0 and self.epoch < self.cfg.warmup_epochs:
            lam = self.epoch / self.cfg.warmup_epochs
        else:
            lam = 1.0

        weights = (1.0 - lam) + lam * gate   # (B, K)

        # ---- monitoring stats -------------------------------------------- #
        with torch.no_grad():
            suppressed = (weights < 0.5).float().mean().item()
            stats = {
                "conflict_mean": conflict.mean().item(),
                "conflict_std":  conflict.std().item(),
                "conflict_min":  conflict.min().item(),
                "conflict_max":  conflict.max().item(),
                "eff_conflict_mean": effective.mean().item(),
                "eff_conflict_max":  effective.max().item(),
                "threshold":     threshold,
                "weight_mean":   weights.mean().item(),
                "suppressed":    suppressed,
                "warmup_lam":    lam,
            }
            if pi_k is not None:
                stats["pi_mean"] = pi_k.mean().item()
                stats["pi_max"]  = pi_k.max().item()

        self.global_step += 1
        return weights, stats


# =============================================================================
# FN/THN Audit logger
# =============================================================================

class AuditLogger:
    """
    Per-epoch logger that tracks suspected FN and THN samples.

    For each (query, negative) pair we record:
      - conflict score  : cos(g_pos, g_neg) — high → suspected FN
      - weight          : gating weight     — low  → suppressed (treated as FN)
      - sim_q_n         : cos(q, n)         — how hard this negative is
      - sim_p_n         : cos(p, n)         — how similar neg is to positive
      - is_fn_gt        : whether neg_id appears in qrels (ground-truth FN)

    At the end of each epoch, saves top-K suspected FN (highest conflict) and
    top-K suspected THN (lowest conflict among high-sim negatives) to JSON.
    """

    def __init__(self, topk: int = 30, hard_sim_threshold: float = 0.5):
        self.topk = topk
        self.hard_sim_threshold = hard_sim_threshold
        self._reset()

    def _reset(self):
        # max-heap for top-K FN candidates  (conflict DESC  → store -conflict)
        # min-heap for top-K THN candidates (conflict ASC, among sim_q_n > thr)
        self._fn_heap: List  = []   # (-conflict, record)
        self._thn_heap: List = []   # (conflict,  record)

    @torch.no_grad()
    def collect(
        self,
        qids:       List[str],
        queries:    List[str],
        pos_ids:    List[str],
        positives:  List[str],
        neg_ids:    List[List[str]],
        negatives:  List[List[str]],
        conflict:   torch.Tensor,   # (B, K)
        weights:    torch.Tensor,   # (B, K)
        q_emb:      torch.Tensor,   # (B, D)
        p_emb:      torch.Tensor,   # (B, D)
        n_emb:      torch.Tensor,   # (B, K, D)
        qrels:      Dict,
    ):
        sim_q_n = torch.einsum("bd,bkd->bk", q_emb, n_emb).cpu()  # (B, K)
        sim_p_n = torch.einsum("bd,bkd->bk", p_emb, n_emb).cpu()  # (B, K)
        conflict_cpu = conflict.cpu()
        weights_cpu  = weights.cpu()

        B, K = conflict_cpu.shape
        for b in range(B):
            qid = qids[b]
            rel_ids = {str(pid) for pid, rel in qrels.get(qid, {}).items() if rel > 0}
            rel_ids.add(str(pos_ids[b]))

            for k in range(K):
                c   = float(conflict_cpu[b, k])
                w   = float(weights_cpu[b, k])
                sqn = float(sim_q_n[b, k])
                spn = float(sim_p_n[b, k])
                nid = neg_ids[b][k]
                is_fn_gt = nid in rel_ids

                record = {
                    "qid":       qid,
                    "query":     queries[b],
                    "pos_id":    str(pos_ids[b]),
                    "positive":  positives[b][:120],
                    "neg_id":    nid,
                    "negative":  negatives[b][k][:120],
                    "conflict":  round(c,   4),
                    "weight":    round(w,   4),
                    "sim_q_n":   round(sqn, 4),
                    "sim_p_n":   round(spn, 4),
                    "is_fn_gt":  is_fn_gt,
                }

                # top-K suspected FN: highest conflict
                heapq.heappush(self._fn_heap, (-c, id(record), record))
                if len(self._fn_heap) > self.topk:
                    heapq.heappop(self._fn_heap)

                # top-K suspected THN: lowest conflict among hard negatives
                if sqn >= self.hard_sim_threshold:
                    heapq.heappush(self._thn_heap, (c, id(record), record))
                    if len(self._thn_heap) > self.topk:
                        heapq.heappop(self._thn_heap)

    def save(self, path: Path):
        fn_samples  = [r for _, _, r in sorted(self._fn_heap,  key=lambda x: x[0])]
        thn_samples = [r for _, _, r in sorted(self._thn_heap, key=lambda x: x[0], reverse=True)]

        out = {
            "suspected_fn":  fn_samples,   # high conflict → GCG suppresses
            "suspected_thn": thn_samples,  # low conflict, high sim(q,n) → kept
            "fn_gt_rate":  sum(r["is_fn_gt"] for r in fn_samples)  / max(len(fn_samples),  1),
            "thn_gt_rate": sum(r["is_fn_gt"] for r in thn_samples) / max(len(thn_samples), 1),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        self._reset()


# =============================================================================
# Weighted InfoNCE loss
# =============================================================================

def weighted_infonce(q_emb, p_emb, n_emb, neg_weights, temperature=0.02, eps=1e-8):
    """
    InfoNCE with per-negative weights.
    neg_weights: (B, K) — 1.0 = keep fully, 0.0 = suppress completely.
    """
    pos_scores = (q_emb * p_emb).sum(dim=-1)                   # (B,)
    neg_scores = torch.einsum("bd,bkd->bk", q_emb, n_emb)      # (B, K)

    pos_logits = pos_scores / temperature
    neg_logits = neg_scores / temperature

    pos_exp = torch.exp(pos_logits)                             # (B,)
    neg_exp = torch.exp(neg_logits) * neg_weights               # (B, K)

    denom = pos_exp + neg_exp.sum(dim=-1) + eps                 # (B,)
    loss = -torch.log(pos_exp / denom)                          # (B,)

    return loss.mean(), pos_scores, neg_scores


# =============================================================================
# Retrieval helpers (eval / negative refresh)
# =============================================================================

@torch.no_grad()
def encode_corpus(model, collection, device, d_maxlen=180, batch_size=64):
    doc_ids, embs = [], []
    texts = [text for _, text in collection]
    ids = [pid for pid, _ in collection]

    for i in tqdm(range(0, len(texts), batch_size), desc="encode corpus"):
        batch = texts[i : i + batch_size]
        emb = model.encode_doc(batch, device=device, max_length=d_maxlen)
        embs.append(emb.cpu())
        doc_ids.extend(ids[i : i + batch_size])

    return doc_ids, torch.cat(embs, dim=0)


@torch.no_grad()
def retrieve(model, queries, collection, device,
             q_maxlen=64, d_maxlen=180, topk=100, batch_size=64):
    doc_ids, doc_embs = encode_corpus(
        model, collection, device=device, d_maxlen=d_maxlen, batch_size=batch_size,
    )
    doc_embs = doc_embs.to(device)
    q_items = list(queries.items())
    results = {}

    for i in tqdm(range(0, len(q_items), batch_size), desc="retrieve"):
        batch = q_items[i : i + batch_size]
        qids = [qid for qid, _ in batch]
        qtexts = [q for _, q in batch]
        q_emb = model.encode_query(qtexts, device=device, max_length=q_maxlen)
        scores = q_emb @ doc_embs.T
        k = min(topk, scores.size(1))
        vals, inds = torch.topk(scores, k=k, dim=1)

        for b in range(len(qids)):
            results[qids[b]] = {
                doc_ids[int(inds[b, j].item())]: float(vals[b, j].item())
                for j in range(k)
            }
    return results


@torch.no_grad()
def refresh_train_rows(model, train_rows, collection, qrels_train, device,
                       topk=50, keep_negs=8,
                       q_maxlen=64, d_maxlen=180, batch_size=64):
    doc_ids, doc_embs = encode_corpus(
        model, collection, device=device, d_maxlen=d_maxlen, batch_size=batch_size,
    )
    doc_embs = doc_embs.to(device)
    pid_to_text = {pid: text for pid, text in collection}
    qid_to_row = {str(row["qid"]): row for row in train_rows}
    queries = {str(row["qid"]): row["query"] for row in train_rows}
    q_items = list(queries.items())
    refreshed = []

    for i in tqdm(range(0, len(q_items), batch_size), desc="refresh negatives"):
        batch = q_items[i : i + batch_size]
        qids = [qid for qid, _ in batch]
        qtexts = [q for _, q in batch]
        q_emb = model.encode_query(qtexts, device=device, max_length=q_maxlen)
        scores = q_emb @ doc_embs.T
        k = min(topk, scores.size(1))
        _, inds = torch.topk(scores, k=k, dim=1)

        for b, qid in enumerate(qids):
            row = qid_to_row[qid]
            pos_ids = {str(x) for x in row["pos_ids"]}
            rel_docs = {str(pid) for pid, rel in qrels_train.get(qid, {}).items() if rel > 0}

            neg_ids = []
            for j in range(k):
                pid = doc_ids[int(inds[b, j].item())]
                if pid in rel_docs or pid in pos_ids:
                    continue
                neg_ids.append(pid)
                if len(neg_ids) >= keep_negs:
                    break

            if not neg_ids:
                continue

            new_row = dict(row)
            new_row["neg_ids"] = neg_ids
            new_row["negatives"] = [pid_to_text[nid] for nid in neg_ids]
            refreshed.append(new_row)

    return refreshed


# =============================================================================
# Training loop
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----------------------------------------------------------------
    collection = load_collection(dataset_dir / "collection.jsonl")
    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    train_qrels = load_qrels(dataset_dir / "qrels_train.json")

    train_ds = TrainDataset(
        dataset_dir / "train.jsonl",
        neg_per_query=args.neg_per_query,
        seed=args.seed,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )

    # ---- model ---------------------------------------------------------------
    if args.model_type == "bge":
        model = BGEBiEncoder(args.model_name)
        model.train_module.to(device)
    elif args.model_type == "dpr":
        model = DPRBiEncoder(args.q_model_name, args.d_model_name)
        model.q_model.to(device)
        model.d_model.to(device)
    else:
        raise ValueError("model_type must be bge or dpr")

    optimizer = torch.optim.AdamW(model.parameters_to_optimize(), lr=args.lr)

    # ---- Audit logger --------------------------------------------------------
    auditor = AuditLogger(
        topk=args.audit_topk,
        hard_sim_threshold=args.audit_hard_sim_threshold,
    ) if args.audit else None

    # ---- GCG -----------------------------------------------------------------
    gcg = GradientConflictGater(
        GCGConfig(
            temperature=args.temperature,
            gate_method=args.gate_method,
            fixed_threshold=args.fixed_threshold,
            percentile=args.percentile,
            sharpness=args.sharpness,
            warmup_epochs=args.warmup_epochs,
            conflict_mode=args.conflict_mode,
            threshold_ema=args.threshold_ema,
            use_pi_weighting=args.use_pi_weighting,
            tau_min=args.tau_min,
        )
    )

    best_dev = -1.0
    best_state_dir = output_dir / "best"
    all_history = []

    # ---- epoch loop ----------------------------------------------------------
    for epoch in range(args.epochs):
        gcg.set_epoch(epoch)
        model.train_module.train()

        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        epoch_stats = []

        for batch in pbar:
            queries = batch["queries"]
            positives = batch["positives"]
            neg_ids = batch["neg_ids"]
            negatives = batch["negatives"]
            B, K = len(queries), batch["K"]

            flat_negs = [x for row in negatives for x in row]

            # ---------- forward pass ---------------------------------------- #
            use_autograd = (args.conflict_mode == "autograd")

            if use_autograd:
                q_emb, q_hidden = model.encode_query_with_hidden(
                    queries, device=device, max_length=args.q_maxlen,
                )
                q_hidden.requires_grad_(True)
                # re-normalise from hidden so graph connects
                q_emb_ag = F.normalize(q_hidden, dim=-1)
            else:
                q_emb = model.encode_query(queries, device=device, max_length=args.q_maxlen)
                q_hidden = None
                q_emb_ag = None

            p_emb = model.encode_doc(positives, device=device, max_length=args.d_maxlen)
            n_emb = model.encode_doc(
                flat_negs, device=device, max_length=args.d_maxlen,
            ).view(B, K, -1)

            # ---------- conflict scores & gating weights -------------------- #
            if use_autograd:
                conflict = gcg.compute_conflict(
                    q_emb_ag, p_emb.detach(), n_emb.detach(),
                    q_hidden=q_hidden,
                    temperature=args.temperature,
                )
            else:
                conflict = gcg.compute_conflict(q_emb, p_emb, n_emb)

            # π_k weighting (if enabled)
            pi_k = None
            if args.use_pi_weighting:
                with torch.no_grad():
                    pi_k = gcg.compute_pi(
                        q_emb, p_emb, n_emb, temperature=args.temperature,
                    )

            neg_weights, stats = gcg.compute_weights(conflict, pi_k=pi_k)

            # ---------- weighted InfoNCE loss ------------------------------- #
            loss, pos_scores, neg_scores = weighted_infonce(
                q_emb, p_emb, n_emb, neg_weights.detach(),
                temperature=args.temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # ---------- audit ------------------------------------------------ #
            if auditor is not None:
                auditor.collect(
                    qids=batch["qids"],
                    queries=queries,
                    pos_ids=batch["pos_ids"],
                    positives=positives,
                    neg_ids=neg_ids,
                    negatives=negatives,
                    conflict=conflict,
                    weights=neg_weights.detach(),
                    q_emb=q_emb.detach(),
                    p_emb=p_emb.detach(),
                    n_emb=n_emb.detach(),
                    qrels=train_qrels,
                )

            # ---------- logging --------------------------------------------- #
            stats["loss"] = loss.item()
            with torch.no_grad():
                pos_neg_sim = torch.einsum("bd,bkd->bk", p_emb, n_emb)
                stats["pos_neg_sim_mean"] = pos_neg_sim.mean().item()

            epoch_stats.append(stats)

            pbar.set_postfix({
                "loss":     f"{stats['loss']:.4f}",
                "c_mean":   f"{stats['conflict_mean']:.3f}",
                "ec_mean":  f"{stats['eff_conflict_mean']:.4f}",
                "ec_max":   f"{stats['eff_conflict_max']:.4f}",
                "thr":      f"{stats['threshold']:.4f}",
                "w_mean":   f"{stats['weight_mean']:.3f}",
                "suppr":    f"{stats['suppressed']:.2f}",
                "pi":       f"{stats.get('pi_mean', 0):.3f}",
                "pn_sim":   f"{stats['pos_neg_sim_mean']:.3f}",
            })

        # ---- dynamic negative refresh --------------------------------------- #
        if args.dynamic_refresh and (epoch + 1) % args.refresh_every == 0:
            model.train_module.eval()
            train_rows = refresh_train_rows(
                model=model,
                train_rows=train_rows,
                collection=collection,
                qrels_train=train_qrels,
                device=device,
                topk=args.refresh_topk,
                keep_negs=args.neg_per_query,
                q_maxlen=args.q_maxlen,
                d_maxlen=args.d_maxlen,
                batch_size=args.eval_batch_size,
            )
            train_ds.refresh_rows(train_rows)
            train_dl = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
            )

        # ---- audit save ------------------------------------------------------ #
        if auditor is not None:
            audit_path = output_dir / f"audit_epoch{epoch}.json"
            auditor.save(audit_path)
            # quick summary print
            with audit_path.open() as _f:
                _a = json.load(_f)
            print(f"  [audit] fn_gt_rate={_a['fn_gt_rate']:.3f} "
                  f"thn_gt_rate={_a['thn_gt_rate']:.3f} "
                  f"(suspected_fn={len(_a['suspected_fn'])}, "
                  f"suspected_thn={len(_a['suspected_thn'])})")

        # ---- evaluation ------------------------------------------------------ #
        model.train_module.eval()
        metrics_all = {}

        for split in ["dev", "test"]:
            q_path = dataset_dir / f"queries_{split}.jsonl"
            r_path = dataset_dir / f"qrels_{split}.json"
            if not q_path.exists() or not r_path.exists():
                continue

            queries_eval = load_queries(q_path)
            qrels_eval = load_qrels(r_path)
            results = retrieve(
                model=model,
                queries=queries_eval,
                collection=collection,
                device=device,
                q_maxlen=args.q_maxlen,
                d_maxlen=args.d_maxlen,
                topk=args.eval_topk,
                batch_size=args.eval_batch_size,
            )
            metrics = evaluate_run(qrels_eval, results, k=10)
            metrics_all[split] = metrics
            print(f"  [{split}] {metrics}")

            with (output_dir / f"run_{split}_epoch{epoch}.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

        # aggregate epoch-level GCG stats
        if epoch_stats:
            agg = {}
            keys = [
                "loss", "conflict_mean", "conflict_std", "conflict_min",
                "conflict_max", "eff_conflict_mean", "eff_conflict_max",
                "threshold", "weight_mean", "suppressed",
                "pos_neg_sim_mean", "pi_mean", "pi_max",
            ]
            for key in keys:
                vals = [s[key] for s in epoch_stats if key in s]
                if vals:
                    agg[f"avg_{key}"] = sum(vals) / len(vals)
            agg["min_conflict_min"] = min(s["conflict_min"] for s in epoch_stats)
            agg["max_conflict_max"] = max(s["conflict_max"] for s in epoch_stats)
            metrics_all["gcg"] = agg

        with (output_dir / f"metrics_epoch{epoch}.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_all, f, ensure_ascii=False, indent=2)

        all_history.append(metrics_all)

        # ---- best checkpoint ------------------------------------------------- #
        dev_score = metrics_all.get("dev", {}).get("nDCG@10", -1.0)
        if dev_score > best_dev:
            best_dev = dev_score
            model.save(best_state_dir)
            with (best_state_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    # ---- save full history --------------------------------------------------- #
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(all_history, f, ensure_ascii=False, indent=2)

    print(f"\nbest dev nDCG@10 = {best_dev:.4f}")


# =============================================================================
# CLI
# =============================================================================

def build_argparser():
    ap = argparse.ArgumentParser(
        description="Train dense retriever with Gradient Conflict Gating (GCG)",
    )

    # data / output
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)

    # model
    ap.add_argument("--model_type", choices=["bge", "dpr"], required=True)
    ap.add_argument("--model_name", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--q_model_name", default="facebook/dpr-question_encoder-single-nq-base")
    ap.add_argument("--d_model_name", default="facebook/dpr-ctx_encoder-single-nq-base")

    # training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)

    # GCG hyperparameters
    ap.add_argument("--gate_method", choices=["adaptive", "fixed"], default="adaptive",
                    help="adaptive = percentile-based threshold, fixed = manual threshold")
    ap.add_argument("--fixed_threshold", type=float, default=0.3,
                    help="conflict threshold (only for gate_method=fixed)")
    ap.add_argument("--percentile", type=float, default=80.0,
                    help="suppress negatives above this conflict percentile (adaptive)")
    ap.add_argument("--sharpness", type=float, default=20.0,
                    help="sigmoid steepness for gating")
    ap.add_argument("--warmup_epochs", type=int, default=1,
                    help="linearly ramp gating strength over N epochs")
    ap.add_argument("--conflict_mode", choices=["closed_form", "autograd"],
                    default="closed_form",
                    help="closed_form = analytic projection (fast), autograd = via torch.autograd.grad")
    ap.add_argument("--threshold_ema", type=float, default=0.95,
                    help="EMA decay for adaptive threshold smoothing")
    ap.add_argument("--use_pi_weighting", action="store_true",
                    help="weight conflict by softmax posterior π_k (effective_conflict = π_k * ReLU(cos))")
    ap.add_argument("--tau_min", type=float, default=-1.0,
                    help="absolute floor for adaptive threshold (< 0 = disabled)")

    # tokenisation
    ap.add_argument("--q_maxlen", type=int, default=64)
    ap.add_argument("--d_maxlen", type=int, default=180)

    # negatives
    ap.add_argument("--neg_per_query", type=int, default=8)
    ap.add_argument("--dynamic_refresh", action="store_true")
    ap.add_argument("--refresh_every", type=int, default=1,
                    help="refresh hard negatives every N epochs (default: 1 = every epoch)")
    ap.add_argument("--refresh_topk", type=int, default=50)

    # eval
    ap.add_argument("--eval_topk", type=int, default=100)

    # audit: FN/THN inspection
    ap.add_argument("--audit", action="store_true",
                    help="save per-epoch FN/THN audit files (audit_epoch{N}.json)")
    ap.add_argument("--audit_topk", type=int, default=30,
                    help="number of top suspected FN and THN samples to save per epoch")
    ap.add_argument("--audit_hard_sim_threshold", type=float, default=0.5,
                    help="min sim(q,n) to be considered a hard negative in THN audit")

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)