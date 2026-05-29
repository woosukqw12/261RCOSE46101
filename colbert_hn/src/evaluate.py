"""Evaluation primitives — used by per-experiment `run.py` scripts.

This module is a **library**, not an entry point. Each
`experiments/{NN}_*/run.py` composes the functions here into an
experiment-specific orchestrator. CLI parsing and config selection are the
responsibility of the experiment script, not this file.

Public API:
    encode_corpus(model, corpus, device, batch_size)
        → (dids, d_emb [N, T_max, D], d_mask [N, T_max])
    score_queries(model, queries, dids, d_emb, d_mask, device, ...)
        → {qid: [(did, score), ...]}  (length top_k)
    compute_metrics_trec(runs_scored, qrels, metrics_k)
        → {qid: {ndcg_cut_10, recip_rank, ...}}
    build_aggregate(per_q, runs_ranked, qrels, confused_slice_def)
        → {all: {...}, confused: {...}, _meta: {...}}
    save_env(out_dir, seed, device)
        → writes env.json
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytrec_eval
import torch
from tqdm import tqdm

from src.colbert_hook import ColBERTv2
from src.data import doc_text
from src.metrics import aggregate_mean
from src.slices import confused_slice, restrict_per_query
from src.utils.io import PathLike, save_json


@torch.no_grad()
def encode_corpus(
    model: ColBERTv2,
    corpus: Dict[str, dict],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    dids = list(corpus.keys())
    texts = [doc_text(corpus[d]) for d in dids]
    embs: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode_docs"):
        emb, mask = model.encode_docs(texts[start:start + batch_size], device=device)
        embs.append(emb.cpu())
        masks.append(mask.cpu())
    t_max = max(e.shape[1] for e in embs)
    n = sum(e.shape[0] for e in embs)
    dim = embs[0].shape[-1]
    d_emb = torch.zeros(n, t_max, dim)
    d_mask = torch.zeros(n, t_max, dtype=torch.bool)
    off = 0
    for emb, mask in zip(embs, masks):
        b, t = emb.shape[:2]
        d_emb[off:off + b, :t] = emb
        d_mask[off:off + b, :t] = mask
        off += b
    return dids, d_emb, d_mask


@torch.no_grad()
def score_queries(
    model: ColBERTv2,
    queries: Dict[str, str],
    dids: List[str],
    d_emb: torch.Tensor,
    d_mask: torch.Tensor,
    device: torch.device,
    query_batch: int = 16,
    doc_chunk: int = 512,
    top_k: int = 100,
    exclude_self: bool = False,
) -> Dict[str, List[Tuple[str, float]]]:
    """Brute-force MaxSim scoring.

    `exclude_self=True` removes any doc whose id equals the query id from the
    retrieved list — required for datasets where queries are also corpus docs
    (ArguAna's counter-argument task: each query is an argument that itself
    appears as a doc; the relevant doc is its counter-argument, never itself).
    Without this filter, MaxSim self-similarity always wins top-1.
    """
    qids = list(queries.keys())
    q_texts = [queries[q] for q in qids]
    n = d_emb.size(0)
    did_to_idx = {d: i for i, d in enumerate(dids)} if exclude_self else None
    out: Dict[str, List[Tuple[str, float]]] = {}
    for q_start in tqdm(range(0, len(qids), query_batch), desc="score_queries"):
        batch_qids = qids[q_start:q_start + query_batch]
        batch_texts = q_texts[q_start:q_start + query_batch]
        q_emb, _ = model.encode_queries(batch_texts, device=device)
        scores = torch.zeros(q_emb.size(0), n)
        for d_start in range(0, n, doc_chunk):
            d_end = min(d_start + doc_chunk, n)
            s = model.maxsim(
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


def _trec_measures(metrics_k: Tuple[int, ...]) -> set[str]:
    m = {"recip_rank", "map"}
    for k in metrics_k:
        m.add(f"ndcg_cut.{k}")
        m.add(f"recall.{k}")
        m.add(f"P.{k}")
    return m


def compute_metrics_trec(
    runs_scored: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    metrics_k: Tuple[int, ...] = (1, 3, 5, 10, 20),
) -> Dict[str, Dict[str, float]]:
    qrels_int = {q: {d: int(r) for d, r in rels.items()} for q, rels in qrels.items()}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_int, _trec_measures(metrics_k))
    return evaluator.evaluate(runs_scored)


def build_aggregate(
    per_q: Dict[str, Dict[str, float]],
    runs_ranked: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    confused_slice_def: str = "top1_ne_rel",
) -> Dict[str, Dict[str, float]]:
    conf_k = 1 if confused_slice_def == "top1_ne_rel" else 3
    conf = confused_slice(runs_ranked, qrels, k=conf_k)
    agg = {
        "all": aggregate_mean(per_q),
        "confused": aggregate_mean(restrict_per_query(per_q, conf)),
    }
    agg["_meta"] = {
        "n_queries": len(per_q),
        "n_confused": len(conf),
        "frac_confused": len(conf) / max(len(per_q), 1),
        "confused_slice_def": confused_slice_def,
    }
    return agg


def save_env(out_dir: PathLike, seed: int, device: torch.device) -> None:
    save_json(
        {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "platform": platform.platform(),
            "seed": seed,
        },
        Path(out_dir) / "env.json",
    )
