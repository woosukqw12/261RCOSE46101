"""BEIR dataset loading (SciFact / NFCorpus / SciDocs).

Downloads via the BEIR utility if not present on disk; thereafter loads
locally. Triplet construction for training is intentionally minimal here —
hard-negative mining (DESIGN.md §4.1, T2B.09/T2B.10 ablations) lives in
training-side scripts that depend on a baseline ColBERT pass.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

from src.utils.logging import get_logger

logger = get_logger(__name__)

BEIR_DATASETS: Tuple[str, ...] = (
    "scifact",
    "nfcorpus",
    "scidocs",
    "trec-covid",
    "fiqa",
    "arguana",
)
BEIR_URL_FMT = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)


def ensure_dataset(name: str, data_root: str = "data") -> Path:
    """Download a BEIR dataset if missing; return its on-disk directory."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists() and any(target.iterdir()):
        return target
    url = BEIR_URL_FMT.format(name=name)
    logger.info("Downloading BEIR dataset: %s → %s", url, target)
    beir_util.download_and_unzip(url, str(root))
    return target


def load_beir(
    name: str,
    split: str = "test",
    data_root: str = "data",
) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Returns (corpus, queries, qrels) for the requested split.

    `corpus[did] = {"title": str, "text": str}` (BEIR convention).
    `queries[qid] = str`.
    `qrels[qid][did] = relevance` (int, ≥ 1 means relevant).
    """
    path = ensure_dataset(name, data_root=data_root)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(path)).load(split=split)
    return corpus, queries, qrels


def doc_text(d: dict) -> str:
    """Concatenate title + body per ColBERT v2 BEIR convention.

    Uses a single space separator (matches the official ColBERT v2 BEIR eval
    pipeline). Many SciFact titles already end with a period, so an explicit
    ". " separator would create a double-period that perturbs WordPiece.
    """
    title = (d.get("title") or "").strip()
    body = (d.get("text") or "").strip()
    return (title + " " + body) if title else body


def build_pos_pairs(
    qrels: Dict[str, Dict[str, int]],
) -> List[Tuple[str, str]]:
    """Flatten qrels into (qid, positive_did) pairs (relevance ≥ 1)."""
    pairs: List[Tuple[str, str]] = []
    for qid, rels in qrels.items():
        for did, r in rels.items():
            if r >= 1:
                pairs.append((qid, did))
    return pairs


def _cli() -> None:
    parser = argparse.ArgumentParser(description="BEIR dataset preparation.")
    parser.add_argument("--extract", action="store_true", help="Download all canonical BEIR datasets.")
    parser.add_argument("--datasets", nargs="+", default=list(BEIR_DATASETS))
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    if args.extract:
        for name in args.datasets:
            path = ensure_dataset(name, data_root=args.data_root)
            try:
                corpus, queries, qrels = load_beir(name, split="test", data_root=args.data_root)
                logger.info(
                    "%s: corpus=%d queries=%d qrels=%d (path=%s)",
                    name, len(corpus), len(queries), len(qrels), path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: failed to load split=test (%s)", name, exc)


if __name__ == "__main__":
    _cli()
