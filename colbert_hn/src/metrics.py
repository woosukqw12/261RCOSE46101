"""Retrieval metrics and paired-bootstrap statistics.

Metrics (DESIGN.md §5.2): NDCG@k, MRR@k, Recall@k, MAP@k.
Statistics (DESIGN.md §5.3, CLAUDE.md §3.7): paired bootstrap on per-query
Δ-metric, 10,000 iterations, 95 % CI by default.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def _dcg(rels: List[int], k: int) -> float:
    rels_arr = np.asarray(rels[:k], dtype=float)
    if rels_arr.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels_arr.size + 2))
    return float((rels_arr * discounts).sum())


def ndcg_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    if not retrieved:
        return 0.0
    rel_at_pos = [int(qrels.get(d, 0)) for d in retrieved[:k]]
    ideal = sorted((int(v) for v in qrels.values()), reverse=True)
    idcg = _dcg(ideal, k)
    return _dcg(rel_at_pos, k) / idcg if idcg > 0 else 0.0


def mrr_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    for i, d in enumerate(retrieved[:k]):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    rel_set = {d for d, r in qrels.items() if r > 0}
    if not rel_set:
        return 0.0
    return len(set(retrieved[:k]) & rel_set) / len(rel_set)


def map_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    rel_set = {d for d, r in qrels.items() if r > 0}
    if not rel_set:
        return 0.0
    hits = 0
    score = 0.0
    for i, d in enumerate(retrieved[:k]):
        if d in rel_set:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(rel_set), k)


def compute_per_query_metrics(
    runs: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    ks: Iterable[int] = (1, 3, 5, 10, 20),
) -> Dict[str, Dict[str, float]]:
    """{qid → {metric@k: value}} for the metrics in DESIGN.md §5.2.

    `runs[qid]` is the ranked list of `did` from highest score to lowest.
    """
    out: Dict[str, Dict[str, float]] = {}
    ks = tuple(sorted(set(int(k) for k in ks)))
    for qid, ranked in runs.items():
        rel = qrels.get(qid, {})
        row: Dict[str, float] = {}
        for k in ks:
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
            row[f"mrr@{k}"] = mrr_at_k(ranked, rel, k)
            row[f"recall@{k}"] = recall_at_k(ranked, rel, k)
            row[f"map@{k}"] = map_at_k(ranked, rel, k)
        out[qid] = row
    return out


def aggregate_mean(per_q: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not per_q:
        return {}
    keys = next(iter(per_q.values())).keys()
    return {k: float(np.mean([row[k] for row in per_q.values()])) for k in keys}


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_iter: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired bootstrap CI on (a - b).

    Returns (mean_delta, lo, hi). CI excluding 0 → statistically distinguishable
    (DESIGN.md §5.3).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n = a.size
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    means = diffs[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(diffs.mean()), float(lo), float(hi)


def align_per_query(
    per_q_a: Dict[str, Dict[str, float]],
    per_q_b: Dict[str, Dict[str, float]],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return paired (a, b, qids) over the intersection of qids — guards against
    bootstrap on misaligned query sets."""
    qids = sorted(set(per_q_a) & set(per_q_b))
    a = np.array([per_q_a[q][metric] for q in qids], dtype=float)
    b = np.array([per_q_b[q][metric] for q in qids], dtype=float)
    return a, b, qids
