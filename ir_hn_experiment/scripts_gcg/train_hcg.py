"""
Hard Conflict Gating (HCG) for Dense Retriever Training
========================================================

Core idea:
  Gradient conflict between the positive-pull and negative-push directions is
  harmful only when the conflicting negative is *also hard* (i.e., genuinely
  high-scoring for the query).  A high-conflict easy negative barely contributes
  to the loss and can be ignored safely; a high-conflict hard negative creates
  a contradictory gradient that destabilises training.

  We suppress a negative only when BOTH conditions hold:
    (1) Hard:       sim(q, n_k)  > sim_qn_floor   (the negative is hard)
    (2) Conflicting: cos(g_pos, g_neg_k) > 0       (its gradient opposes the pull)

  Effective conflict score:
    effective_k = ReLU(conflict_k) * ReLU(sim(q, n_k) - sim_qn_floor)

  This is a 2-D gating:
    - easy + conflicting  → effective ≈ 0  → kept  (wasted suppression budget avoided)
    - hard  + non-conflict → effective ≈ 0  → kept  (useful hard neg preserved)
    - hard  + conflicting  → effective high  → suppressed  ✓

  Note on framing:
    Unlike the original GCG which framed suppression as "false negative detection",
    HCG targets *conflicting hard true negatives* — negatives that are genuinely
    irrelevant but whose push gradient interferes with the pull gradient, causing
    contradictory weight updates.  Experimental evidence (C > A in 2×2 ablation)
    shows that suppressing these stabilises training even when the dataset
    contains zero labeled false negatives.

Usage:
  python train_hcg.py \\
    --dataset_dir ./data/processed/fiqa \\
    --output_dir  ./outputs/bge/fiqa_hcg \\
    --model_type  bge \\
    --model_name  BAAI/bge-base-en-v1.5 \\
    --epochs 10 \\
    --batch_size 16 \\
    --neg_per_query 8 \\
    --lr 2e-6 \\
    --temperature 0.05 \\
    --gate_method adaptive \\
    --percentile 80 \\
    --sharpness 20 \\
    --sim_qn_floor 0.4 \\
    --warmup_epochs 1 \\
    --audit
"""

import argparse
import heapq
import json
import math
import random
from dataclasses import dataclass
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
        hidden = out.last_hidden_state[:, 0]
        emb = F.normalize(hidden, dim=-1)
        return emb

    def encode_query(self, texts, device, max_length=64):
        return self._encode(self.q_tok, texts, device, max_length)

    def encode_doc(self, texts, device, max_length=180):
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
# Hard Conflict Gater (HCG)
# =============================================================================

@dataclass
class HCGConfig:
    temperature: float = 0.02

    # --- hardness floor -------------------------------------------------------
    # Negatives with sim(q, n) < sim_qn_floor are considered easy and are
    # never suppressed regardless of their conflict score.
    sim_qn_floor: float = 0.4

    # --- conflict → weight mapping -------------------------------------------
    gate_method: str = "adaptive"   # "adaptive" (percentile) or "fixed"
    fixed_threshold: float = 0.1    # used when gate_method == "fixed"
    percentile: float = 80.0        # suppress top-X% by effective conflict
    sharpness: float = 20.0         # sigmoid steepness

    # --- warmup --------------------------------------------------------------
    warmup_epochs: int = 1

    # --- EMA for adaptive threshold ------------------------------------------
    threshold_ema: float = 0.95


class HardConflictGater:
    """
    Suppresses hard negatives whose push gradient conflicts with the positive
    pull gradient.  Unlike the original GCG which applies conflict gating to
    all negatives, HCG restricts suppression to negatives that are:

      (1) Hard:        sim(q, n_k) > sim_qn_floor
      (2) Conflicting: cos(g_pos, g_neg_k) > 0

    effective_k = ReLU(conflict_k) * ReLU(sim(q, n_k) - sim_qn_floor)

    This avoids wasting the suppression budget on easy negatives and preserves
    hard non-conflicting negatives as valuable training signal.
    """

    def __init__(self, cfg: HCGConfig):
        self.cfg = cfg
        self.epoch = 0
        self.global_step = 0
        self.running_threshold = 0.0
        self._threshold_initialised = False

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    # --------------------------------------------------------------------- #
    #  Conflict score (closed-form tangent-plane projection)                  #
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def compute_conflict(
        self,
        q_emb: torch.Tensor,   # (B, D) L2-normalised
        p_emb: torch.Tensor,   # (B, D) L2-normalised
        n_emb: torch.Tensor,   # (B, K, D) L2-normalised
    ) -> torch.Tensor:         # (B, K)
        """
        Closed-form gradient conflict on the unit hypersphere.

          g_pos   = p  - (q·p) q      (tangent component toward positive)
          g_neg_k = n_k - (q·n_k) q   (tangent component toward negative k)
          conflict_k = cos(g_pos, g_neg_k)
        """
        qp = (q_emb * p_emb).sum(dim=-1, keepdim=True)           # (B, 1)
        g_pos = p_emb - qp * q_emb                               # (B, D)

        qn = torch.einsum("bd,bkd->bk", q_emb, n_emb)           # (B, K)
        g_neg = n_emb - qn.unsqueeze(-1) * q_emb.unsqueeze(1)    # (B, K, D)

        g_pos_n = F.normalize(g_pos, dim=-1).unsqueeze(1)         # (B, 1, D)
        g_neg_n = F.normalize(g_neg, dim=-1)                      # (B, K, D)

        return (g_pos_n * g_neg_n).sum(dim=-1)                    # (B, K)

    # --------------------------------------------------------------------- #
    #  2-D effective conflict and gating weights                              #
    # --------------------------------------------------------------------- #

    def compute_weights(
        self,
        conflict: torch.Tensor,   # (B, K)  cos(g_pos, g_neg)
        sim_q_n:  torch.Tensor,   # (B, K)  cos(q, n)
    ) -> Tuple[torch.Tensor, dict]:
        """
        effective_k = ReLU(conflict_k) * ReLU(sim(q,n_k) - sim_qn_floor)

        A negative is suppressed only if it is both hard AND conflicting.
        Returns (weights, stats).
        """
        # ---- 2-D effective conflict ---------------------------------------- #
        hardness = torch.relu(sim_q_n - self.cfg.sim_qn_floor)   # (B, K)
        effective = torch.relu(conflict) * hardness               # (B, K)

        # ---- threshold (adaptive or fixed) --------------------------------- #
        with torch.no_grad():
            if self.cfg.gate_method == "adaptive":
                batch_thr = torch.quantile(
                    effective.flatten(),
                    self.cfg.percentile / 100.0,
                ).item()
                if not self._threshold_initialised:
                    self.running_threshold = batch_thr
                    self._threshold_initialised = True
                else:
                    self.running_threshold = (
                        self.cfg.threshold_ema * self.running_threshold
                        + (1 - self.cfg.threshold_ema) * batch_thr
                    )
                threshold = self.running_threshold
            else:
                threshold = self.cfg.fixed_threshold

        # ---- sigmoid gating ------------------------------------------------ #
        gate = torch.sigmoid(-self.cfg.sharpness * (effective - threshold))

        # ---- warmup blend -------------------------------------------------- #
        if self.cfg.warmup_epochs > 0 and self.epoch < self.cfg.warmup_epochs:
            lam = self.epoch / self.cfg.warmup_epochs
        else:
            lam = 1.0
        weights = (1.0 - lam) + lam * gate   # (B, K)

        # ---- stats --------------------------------------------------------- #
        with torch.no_grad():
            hard_mask = (sim_q_n > self.cfg.sim_qn_floor)
            suppressed_all  = (weights < 0.5).float().mean().item()
            suppressed_hard = (
                (weights < 0.5)[hard_mask].float().mean().item()
                if hard_mask.any() else 0.0
            )
            stats = {
                "conflict_mean":    conflict.mean().item(),
                "conflict_max":     conflict.max().item(),
                "sim_qn_mean":      sim_q_n.mean().item(),
                "hard_ratio":       hard_mask.float().mean().item(),
                "eff_mean":         effective.mean().item(),
                "eff_max":          effective.max().item(),
                "threshold":        threshold,
                "weight_mean":      weights.mean().item(),
                "suppressed":       suppressed_all,
                "suppressed_hard":  suppressed_hard,
                "warmup_lam":       lam,
            }

        self.global_step += 1
        return weights, stats


# =============================================================================
# Audit logger
# =============================================================================

class AuditLogger:
    """
    Per-epoch snapshot of the top-K suppressed (high effective conflict) and
    top-K preserved hard negatives (high sim_q_n, low effective conflict).

    Saves audit_epoch{N}.json with:
      suppressed   : top-K by effective_conflict  (hard + conflicting → suppressed)
      preserved    : top-K hard negs with lowest effective conflict (kept as signal)
      sim_pn stats : distribution of sim(p,n) in both groups
    """

    def __init__(self, topk: int = 30):
        self.topk = topk
        self._reset()

    def _reset(self):
        self._supp_heap: List = []   # (-eff, id, record)
        self._pres_heap: List = []   # (eff,  id, record)  among hard negs

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
        effective:  torch.Tensor,   # (B, K)  ReLU(conflict)*ReLU(sim_qn-floor)
        weights:    torch.Tensor,   # (B, K)
        sim_q_n:    torch.Tensor,   # (B, K)
        p_emb:      torch.Tensor,   # (B, D)
        n_emb:      torch.Tensor,   # (B, K, D)
        sim_qn_floor: float,
    ):
        sim_p_n = torch.einsum("bd,bkd->bk", p_emb, n_emb).cpu()
        conflict_cpu  = conflict.cpu()
        effective_cpu = effective.cpu()
        weights_cpu   = weights.cpu()
        sim_qn_cpu    = sim_q_n.cpu()

        B, K = conflict_cpu.shape
        for b in range(B):
            for k in range(K):
                eff = float(effective_cpu[b, k])
                rec = {
                    "qid":      qids[b],
                    "query":    queries[b],
                    "pos_id":   str(pos_ids[b]),
                    "positive": positives[b][:120],
                    "neg_id":   neg_ids[b][k],
                    "negative": negatives[b][k][:120],
                    "conflict": round(float(conflict_cpu[b, k]),  4),
                    "effective":round(eff,                         4),
                    "weight":   round(float(weights_cpu[b, k]),   4),
                    "sim_q_n":  round(float(sim_qn_cpu[b, k]),    4),
                    "sim_p_n":  round(float(sim_p_n[b, k]),       4),
                }
                heapq.heappush(self._supp_heap, (-eff, id(rec), rec))
                if len(self._supp_heap) > self.topk:
                    heapq.heappop(self._supp_heap)

                if float(sim_qn_cpu[b, k]) > sim_qn_floor:
                    heapq.heappush(self._pres_heap, (eff, id(rec), rec))
                    if len(self._pres_heap) > self.topk:
                        heapq.heappop(self._pres_heap)

    def save(self, path: Path):
        suppressed = [r for _, _, r in sorted(self._supp_heap, key=lambda x: x[0])]
        preserved  = [r for _, _, r in sorted(self._pres_heap, key=lambda x: x[0], reverse=True)]

        def avg(lst, key):
            vals = [r[key] for r in lst]
            return round(sum(vals) / max(len(vals), 1), 4)

        out = {
            "suppressed": suppressed,
            "preserved":  preserved,
            "suppressed_sim_pn_mean": avg(suppressed, "sim_p_n"),
            "preserved_sim_pn_mean":  avg(preserved,  "sim_p_n"),
            "suppressed_sim_qn_mean": avg(suppressed, "sim_q_n"),
            "preserved_sim_qn_mean":  avg(preserved,  "sim_q_n"),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        self._reset()


# =============================================================================
# Weighted InfoNCE loss
# =============================================================================

def weighted_infonce(q_emb, p_emb, n_emb, neg_weights, temperature=0.02, eps=1e-8):
    pos_scores = (q_emb * p_emb).sum(dim=-1)
    neg_scores = torch.einsum("bd,bkd->bk", q_emb, n_emb)

    pos_exp = torch.exp(pos_scores / temperature)
    neg_exp = torch.exp(neg_scores / temperature) * neg_weights

    denom = pos_exp + neg_exp.sum(dim=-1) + eps
    loss = -torch.log(pos_exp / denom)
    return loss.mean(), pos_scores, neg_scores


# =============================================================================
# Retrieval helpers
# =============================================================================

@torch.no_grad()
def encode_corpus(model, collection, device, d_maxlen=180, batch_size=64):
    doc_ids, embs = [], []
    texts = [text for _, text in collection]
    ids   = [pid  for pid,  _ in collection]
    for i in tqdm(range(0, len(texts), batch_size), desc="encode corpus"):
        emb = model.encode_doc(texts[i:i+batch_size], device=device, max_length=d_maxlen)
        embs.append(emb.cpu())
        doc_ids.extend(ids[i:i+batch_size])
    return doc_ids, torch.cat(embs, dim=0)


@torch.no_grad()
def retrieve(model, queries, collection, device,
             q_maxlen=64, d_maxlen=180, topk=100, batch_size=64):
    doc_ids, doc_embs = encode_corpus(model, collection, device=device,
                                      d_maxlen=d_maxlen, batch_size=batch_size)
    doc_embs = doc_embs.to(device)
    q_items  = list(queries.items())
    results  = {}
    for i in tqdm(range(0, len(q_items), batch_size), desc="retrieve"):
        batch  = q_items[i:i+batch_size]
        qids   = [qid for qid, _ in batch]
        qtexts = [q   for _,   q in batch]
        q_emb  = model.encode_query(qtexts, device=device, max_length=q_maxlen)
        scores = q_emb @ doc_embs.T
        k = min(topk, scores.size(1))
        vals, inds = torch.topk(scores, k=k, dim=1)
        for b in range(len(qids)):
            results[qids[b]] = {
                doc_ids[int(inds[b, j])]: float(vals[b, j]) for j in range(k)
            }
    return results


@torch.no_grad()
def refresh_train_rows(model, train_rows, collection, qrels_train, device,
                       topk=50, keep_negs=8, q_maxlen=64, d_maxlen=180, batch_size=64):
    doc_ids, doc_embs = encode_corpus(model, collection, device=device,
                                      d_maxlen=d_maxlen, batch_size=batch_size)
    doc_embs   = doc_embs.to(device)
    pid_to_text = {pid: text for pid, text in collection}
    qid_to_row  = {str(row["qid"]): row for row in train_rows}
    q_items     = list({str(row["qid"]): row["query"] for row in train_rows}.items())
    refreshed   = []

    for i in tqdm(range(0, len(q_items), batch_size), desc="refresh negatives"):
        batch  = q_items[i:i+batch_size]
        qids   = [qid for qid, _ in batch]
        qtexts = [q   for _,   q in batch]
        q_emb  = model.encode_query(qtexts, device=device, max_length=q_maxlen)
        scores = q_emb @ doc_embs.T
        k = min(topk, scores.size(1))
        _, inds = torch.topk(scores, k=k, dim=1)

        for b, qid in enumerate(qids):
            row      = qid_to_row[qid]
            pos_ids  = {str(x) for x in row["pos_ids"]}
            rel_docs = {str(pid) for pid, rel in qrels_train.get(qid, {}).items() if rel > 0}
            neg_ids  = []
            for j in range(k):
                pid = doc_ids[int(inds[b, j])]
                if pid in rel_docs or pid in pos_ids:
                    continue
                neg_ids.append(pid)
                if len(neg_ids) >= keep_negs:
                    break
            if not neg_ids:
                continue
            new_row = dict(row)
            new_row["neg_ids"]   = neg_ids
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
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----------------------------------------------------------------
    collection   = load_collection(dataset_dir / "collection.jsonl")
    train_rows   = load_jsonl(dataset_dir / "train.jsonl")
    train_qrels  = load_qrels(dataset_dir / "qrels_train.json")

    train_ds = TrainDataset(dataset_dir / "train.jsonl",
                            neg_per_query=args.neg_per_query, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate_fn)

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

    # ---- HCG -----------------------------------------------------------------
    hcg = HardConflictGater(HCGConfig(
        temperature   = args.temperature,
        sim_qn_floor  = args.sim_qn_floor,
        gate_method   = args.gate_method,
        fixed_threshold = args.fixed_threshold,
        percentile    = args.percentile,
        sharpness     = args.sharpness,
        warmup_epochs = args.warmup_epochs,
        threshold_ema = args.threshold_ema,
    ))

    # ---- audit ---------------------------------------------------------------
    auditor = AuditLogger(topk=args.audit_topk) if args.audit else None

    best_dev       = -1.0
    best_state_dir = output_dir / "best"
    all_history    = []

    # ---- epoch loop ----------------------------------------------------------
    for epoch in range(args.epochs):
        hcg.set_epoch(epoch)
        model.train_module.train()

        pbar        = tqdm(train_dl, desc=f"epoch {epoch}")
        epoch_stats = []

        for batch in pbar:
            queries   = batch["queries"]
            positives = batch["positives"]
            neg_ids   = batch["neg_ids"]
            negatives = batch["negatives"]
            B, K      = len(queries), batch["K"]
            flat_negs = [x for row in negatives for x in row]

            # ---------- forward pass ---------------------------------------- #
            q_emb = model.encode_query(queries,   device=device, max_length=args.q_maxlen)
            p_emb = model.encode_doc(positives,   device=device, max_length=args.d_maxlen)
            n_emb = model.encode_doc(flat_negs,   device=device, max_length=args.d_maxlen).view(B, K, -1)

            # ---------- HCG: conflict + hardness ---------------------------- #
            with torch.no_grad():
                sim_q_n  = torch.einsum("bd,bkd->bk", q_emb, n_emb)   # (B, K)
                conflict = hcg.compute_conflict(q_emb, p_emb, n_emb)   # (B, K)

            neg_weights, stats = hcg.compute_weights(conflict, sim_q_n)

            # ---------- loss ------------------------------------------------ #
            loss, pos_scores, neg_scores = weighted_infonce(
                q_emb, p_emb, n_emb, neg_weights.detach(),
                temperature=args.temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # ---------- audit ----------------------------------------------- #
            if auditor is not None:
                with torch.no_grad():
                    hardness  = torch.relu(sim_q_n - args.sim_qn_floor)
                    effective = torch.relu(conflict) * hardness
                auditor.collect(
                    qids=batch["qids"], queries=queries,
                    pos_ids=batch["pos_ids"], positives=positives,
                    neg_ids=neg_ids, negatives=negatives,
                    conflict=conflict, effective=effective,
                    weights=neg_weights.detach(),
                    sim_q_n=sim_q_n,
                    p_emb=p_emb.detach(), n_emb=n_emb.detach(),
                    sim_qn_floor=args.sim_qn_floor,
                )

            # ---------- logging --------------------------------------------- #
            stats["loss"] = loss.item()
            with torch.no_grad():
                stats["pos_neg_sim_mean"] = torch.einsum(
                    "bd,bkd->bk", p_emb, n_emb).mean().item()
            epoch_stats.append(stats)

            pbar.set_postfix({
                "loss":    f"{stats['loss']:.4f}",
                "c_mean":  f"{stats['conflict_mean']:.3f}",
                "sqn":     f"{stats['sim_qn_mean']:.3f}",
                "hard":    f"{stats['hard_ratio']:.2f}",
                "eff":     f"{stats['eff_mean']:.4f}",
                "thr":     f"{stats['threshold']:.4f}",
                "w":       f"{stats['weight_mean']:.3f}",
                "s_all":   f"{stats['suppressed']:.2f}",
                "s_hard":  f"{stats['suppressed_hard']:.2f}",
            })

        # ---- dynamic negative refresh --------------------------------------- #
        if args.dynamic_refresh and (epoch + 1) % args.refresh_every == 0:
            model.train_module.eval()
            train_rows = refresh_train_rows(
                model=model, train_rows=train_rows, collection=collection,
                qrels_train=train_qrels, device=device,
                topk=args.refresh_topk, keep_negs=args.neg_per_query,
                q_maxlen=args.q_maxlen, d_maxlen=args.d_maxlen,
                batch_size=args.eval_batch_size,
            )
            train_ds.refresh_rows(train_rows)
            train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_fn)

        # ---- audit save ------------------------------------------------------ #
        if auditor is not None:
            audit_path = output_dir / f"audit_epoch{epoch}.json"
            auditor.save(audit_path)
            with audit_path.open() as _f:
                _a = json.load(_f)
            print(f"  [audit] "
                  f"suppressed sim_pn={_a['suppressed_sim_pn_mean']:.3f} "
                  f"preserved sim_pn={_a['preserved_sim_pn_mean']:.3f} "
                  f"(suppressed sim_qn={_a['suppressed_sim_qn_mean']:.3f}, "
                  f"preserved sim_qn={_a['preserved_sim_qn_mean']:.3f})")

        # ---- evaluation ------------------------------------------------------ #
        model.train_module.eval()
        metrics_all = {}

        for split in ["dev", "test"]:
            q_path = dataset_dir / f"queries_{split}.jsonl"
            r_path = dataset_dir / f"qrels_{split}.json"
            if not q_path.exists() or not r_path.exists():
                continue
            queries_eval = load_queries(q_path)
            qrels_eval   = load_qrels(r_path)
            results = retrieve(model=model, queries=queries_eval,
                               collection=collection, device=device,
                               q_maxlen=args.q_maxlen, d_maxlen=args.d_maxlen,
                               topk=args.eval_topk, batch_size=args.eval_batch_size)
            metrics = evaluate_run(qrels_eval, results, k=10)
            metrics_all[split] = metrics
            print(f"  [{split}] {metrics}")
            with (output_dir / f"run_{split}_epoch{epoch}.json").open("w") as f:
                json.dump(results, f, ensure_ascii=False)

        # aggregate HCG stats
        if epoch_stats:
            agg = {}
            for key in ["loss", "conflict_mean", "conflict_max", "sim_qn_mean",
                        "hard_ratio", "eff_mean", "eff_max", "threshold",
                        "weight_mean", "suppressed", "suppressed_hard",
                        "pos_neg_sim_mean"]:
                vals = [s[key] for s in epoch_stats if key in s]
                if vals:
                    agg[f"avg_{key}"] = sum(vals) / len(vals)
            metrics_all["hcg"] = agg

        with (output_dir / f"metrics_epoch{epoch}.json").open("w") as f:
            json.dump(metrics_all, f, ensure_ascii=False, indent=2)
        all_history.append(metrics_all)

        dev_score = metrics_all.get("dev", {}).get("nDCG@10", -1.0)
        if dev_score > best_dev:
            best_dev = dev_score
            model.save(best_state_dir)
            with (best_state_dir / "metrics.json").open("w") as f:
                json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    with (output_dir / "history.json").open("w") as f:
        json.dump(all_history, f, ensure_ascii=False, indent=2)
    print(f"\nbest dev nDCG@10 = {best_dev:.4f}")


# =============================================================================
# CLI
# =============================================================================

def build_argparser():
    ap = argparse.ArgumentParser(
        description="Train dense retriever with Hard Conflict Gating (HCG)"
    )

    # data / output
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir",  required=True)

    # model
    ap.add_argument("--model_type",   choices=["bge", "dpr"], required=True)
    ap.add_argument("--model_name",   default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--q_model_name", default="facebook/dpr-question_encoder-single-nq-base")
    ap.add_argument("--d_model_name", default="facebook/dpr-ctx_encoder-single-nq-base")

    # training
    ap.add_argument("--epochs",          type=int,   default=10)
    ap.add_argument("--batch_size",      type=int,   default=16)
    ap.add_argument("--eval_batch_size", type=int,   default=64)
    ap.add_argument("--lr",              type=float, default=2e-6)
    ap.add_argument("--temperature",     type=float, default=0.05)
    ap.add_argument("--seed",            type=int,   default=42)

    # HCG hyperparameters
    ap.add_argument("--sim_qn_floor", type=float, default=0.4,
                    help="min sim(q,n) to qualify for suppression (hardness gate)")
    ap.add_argument("--gate_method",  choices=["adaptive", "fixed"], default="adaptive")
    ap.add_argument("--fixed_threshold", type=float, default=0.1,
                    help="effective conflict threshold (gate_method=fixed only)")
    ap.add_argument("--percentile",   type=float, default=80.0,
                    help="suppress top-X%% by effective conflict (adaptive)")
    ap.add_argument("--sharpness",    type=float, default=20.0,
                    help="sigmoid steepness")
    ap.add_argument("--warmup_epochs",type=int,   default=1)
    ap.add_argument("--threshold_ema",type=float, default=0.95)

    # tokenisation
    ap.add_argument("--q_maxlen", type=int, default=64)
    ap.add_argument("--d_maxlen", type=int, default=180)

    # negatives
    ap.add_argument("--neg_per_query",   type=int,  default=8)
    ap.add_argument("--dynamic_refresh", action="store_true")
    ap.add_argument("--refresh_every",   type=int,  default=3)
    ap.add_argument("--refresh_topk",    type=int,  default=50)

    # eval
    ap.add_argument("--eval_topk", type=int, default=100)

    # audit
    ap.add_argument("--audit",     action="store_true",
                    help="save per-epoch audit_epoch{N}.json")
    ap.add_argument("--audit_topk", type=int, default=30)

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
