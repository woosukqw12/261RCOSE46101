"""Hard-negative mining utilities.

Used by `01_mean_diff` and downstream learned experiments. Mining strategy:
given a baseline ColBERT v2 ranked list for each query, take the top-K
positions whose docs are *not* in `qrels[q]` as hard negatives. Positives
come directly from `qrels[q]` (relevance ≥ 1).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

Triplet = Tuple[str, str, str]  # (qid, pos_did, hn_did)


def positives(qrels_q: Dict[str, int], threshold: int = 1) -> List[str]:
    return [d for d, r in qrels_q.items() if r >= threshold]


def hard_negatives(
    ranked_q: List[str],
    qrels_q: Dict[str, int],
    n: int = 10,
    pool: int = 100,
    threshold: int = 1,
) -> List[str]:
    """Return up to `n` HNs taken from the top-`pool` ranked docs, excluding
    docs with relevance ≥ `threshold` in `qrels_q`."""
    rel_set = {d for d, r in qrels_q.items() if r >= threshold}
    out: List[str] = []
    for d in ranked_q[:pool]:
        if d in rel_set:
            continue
        out.append(d)
        if len(out) >= n:
            break
    return out


def mine_triplets(
    runs: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    n_hns_per_q: int = 10,
    pool: int = 100,
    threshold: int = 1,
) -> List[Triplet]:
    """Returns list of (qid, pos_did, hn_did) triplets — every positive × every
    sampled HN for each query that has both ≥ 1 positive and ≥ 1 HN."""
    out: List[Triplet] = []
    for qid, ranked in runs.items():
        rels = qrels.get(qid, {})
        pos = positives(rels, threshold=threshold)
        if not pos:
            continue
        hns = hard_negatives(ranked, rels, n=n_hns_per_q, pool=pool, threshold=threshold)
        if not hns:
            continue
        for p in pos:
            for h in hns:
                out.append((qid, p, h))
    return out


def unique_dids(triplets: Iterable[Triplet]) -> Tuple[List[str], List[str]]:
    """Returns (sorted unique positive dids, sorted unique HN dids)."""
    pos_set: set[str] = set()
    hn_set: set[str] = set()
    for _, p, h in triplets:
        pos_set.add(p)
        hn_set.add(h)
    return sorted(pos_set), sorted(hn_set)
