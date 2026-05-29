"""Query slice definitions (DESIGN.md §5.1).

`confused` only depends on the baseline retrieval + qrels and is implemented
here. `lexical-HN` and `hard-HN` depend on the prior diagnostic study's
operational definitions (DESIGN.md §10 references) and are left unimplemented
until those definitions are confirmed in this repo — see TODO below.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Set


def confused_slice(
    runs: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 1,
) -> Set[str]:
    """Queries whose baseline top-k retrieval contains no relevant doc.

    Default `k=1` realises the `top1_ne_rel` definition (DESIGN.md §5.1).
    `k=3` realises the `top3_ne_rel` ablation (T2B.14).
    """
    out: Set[str] = set()
    for qid, ranked in runs.items():
        rel_set = {d for d, r in qrels.get(qid, {}).items() if r > 0}
        if not rel_set:
            continue
        top = set(ranked[:k])
        if not (top & rel_set):
            out.add(qid)
    return out


def all_slice(runs: Dict[str, List[str]]) -> Set[str]:
    return set(runs.keys())


def restrict_per_query(
    per_q: Dict[str, Dict[str, float]],
    qids: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    keep = set(qids)
    return {q: v for q, v in per_q.items() if q in keep}


# TODO(slices): lexical_hn / hard_hn definitions to be implemented after the
# prior diagnostic study's operational definitions are confirmed and cited.
# Until then, they are intentionally absent rather than guessed — DESIGN.md
# §3.8 (ablation completeness) requires their definition to be documented
# before any comparison is reported.
