"""
Convert BEIR dataset format → training format for train_gcg.py

Supports memory-aware BM25 hard negative mining (default), full rank_bm25
mining (--neg_mode bm25_full), or random negatives (--neg_mode random).

Usage:
  # Memory-aware BM25 hard negatives (recommended for NQ/HotpotQA):
  python scripts/prepare_beir.py \
    --beir_dir ./data/raw/fiqa \
    --output_dir ./data/processed/fiqa \
    --neg_per_query 8 \
    --neg_mode bm25_lite \
    --bm25_topk 50

  # Random negatives (fast fallback):
  python scripts/prepare_beir.py \
    --beir_dir ./data/raw/fiqa \
    --output_dir ./data/processed/fiqa \
    --neg_per_query 8 \
    --neg_mode random
"""

import argparse
import heapq
import json
import math
import shutil
import random
import re
import zipfile
from array import array
from multiprocessing import get_context
from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter, defaultdict


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
}

_LITE_INDEX = None
_LITE_ALL_PIDS = None
_LITE_NEG_PER_QUERY = None
_LITE_BM25_TOPK = None


def load_beir_corpus(beir_dir: Path):
    corpus = {}
    with (beir_dir / "corpus.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = str(row["_id"])
            corpus[pid] = {
                "title": row.get("title", "") or "",
                "text": row.get("text", "") or "",
            }
    return corpus


def ensure_beir_dataset(beir_dir: Path, dataset: str | None = None, download_root: Path | None = None):
    if (beir_dir / "corpus.jsonl").exists():
        return beir_dir
    if dataset is None:
        dataset = beir_dir.name
    download_root = download_root or beir_dir.parent
    download_root.mkdir(parents=True, exist_ok=True)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    zip_path = download_root / f"{dataset}.zip"
    print(f"BEIR dataset not found at {beir_dir}/corpus.jsonl")
    print(f"Downloading {dataset} from {url}...")
    urlretrieve(url, zip_path)
    print(f"Extracting {zip_path} -> {download_root}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(download_root)
    extracted = download_root / dataset
    if not extracted.exists():
        candidates = [
            p for p in download_root.iterdir()
            if p.is_dir() and (p / "corpus.jsonl").exists()
        ]
        if candidates:
            extracted = candidates[0]
    if extracted != beir_dir and extracted.exists():
        if beir_dir.exists():
            shutil.rmtree(beir_dir)
        shutil.move(str(extracted), str(beir_dir))
    if not (beir_dir / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Downloaded {dataset}, but corpus.jsonl was not found under {beir_dir}")
    return beir_dir


def load_beir_queries(beir_dir: Path):
    queries = {}
    with (beir_dir / "queries.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    return queries


def load_beir_qrels(tsv_path: Path):
    qrels = defaultdict(dict)
    with tsv_path.open("r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid, rel = str(parts[0]), str(parts[1]), int(parts[2])
            qrels[qid][pid] = rel
    return dict(qrels)


def simple_tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def query_terms(text):
    return [
        tok for tok in simple_tokenize(text)
        if len(tok) > 1 and tok not in STOPWORDS
    ]


def corpus_text(corpus, pid):
    doc = corpus[pid]
    return (doc["title"] + " " + doc["text"]).strip()


def sample_non_relevant(all_pids, excluded, k, rng):
    """Sample up to k pids without constructing a full per-query candidate list."""
    if k <= 0:
        return []
    excluded = set(excluded)
    selected = []
    blocked = set(excluded)
    n_docs = len(all_pids)

    # Rejection sampling is cheap when the relevant/excluded set is tiny, which
    # is the usual BEIR case.
    max_attempts = max(100, k * 100)
    attempts = 0
    while len(selected) < k and attempts < max_attempts and len(blocked) < n_docs:
        pid = rng.choice(all_pids)
        attempts += 1
        if pid in blocked:
            continue
        selected.append(pid)
        blocked.add(pid)

    if len(selected) < k:
        for pid in all_pids:
            if pid in blocked:
                continue
            selected.append(pid)
            blocked.add(pid)
            if len(selected) >= k:
                break

    return selected


def build_bm25_index(corpus, all_pids):
    from rank_bm25 import BM25Okapi

    print("  Tokenizing corpus for BM25...")
    tokenized = []
    for pid in all_pids:
        tokenized.append(simple_tokenize(corpus_text(corpus, pid)))

    print("  Building BM25 index...")
    bm25 = BM25Okapi(tokenized)
    return bm25


def mine_bm25_negatives_full(bm25, all_pids, queries, qrels, neg_per_query=8, bm25_topk=50):
    print(f"  Mining BM25 negatives (topk={bm25_topk}, neg_per_query={neg_per_query})...")
    query_negatives = {}

    for i, (qid, query_text) in enumerate(queries.items()):
        if qid not in qrels:
            continue

        tokenized_query = simple_tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:bm25_topk]

        rel_pids = set(qrels[qid].keys())
        neg_ids = []
        for idx in top_indices:
            pid = all_pids[idx]
            if pid not in rel_pids:
                neg_ids.append(pid)
                if len(neg_ids) >= neg_per_query:
                    break

        query_negatives[qid] = neg_ids

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(queries)} queries processed")

    print(f"  Done: {len(query_negatives)} queries with BM25 negatives")
    return query_negatives


def build_query_vocab(queries):
    vocab = set()
    for text in queries.values():
        vocab.update(query_terms(text))
    return vocab


def build_lite_bm25_index(
    corpus,
    all_pids,
    queries,
    max_df_ratio=0.15,
    min_df=2,
    max_postings_per_term=0,
):
    """Build BM25 postings only for terms that appear in train queries.

    Full rank_bm25 stores tokenized text for every document. That is simple but
    expensive for NQ/HotpotQA-sized corpora. This lite index stores postings for
    train-query terms only and drops very common terms whose BM25 IDF is tiny.
    """
    query_vocab = build_query_vocab(queries)
    n_docs = len(all_pids)
    max_df = max(int(max_df_ratio * n_docs), 1)
    doc_lens = array("I")
    df = Counter()

    print(
        f"  Building lite BM25 stats: {len(query_vocab):,} query terms, "
        f"max_df={max_df:,} ({max_df_ratio:.3f} of corpus)"
    )
    for i, pid in enumerate(all_pids):
        toks = simple_tokenize(corpus_text(corpus, pid))
        doc_lens.append(len(toks))
        df.update({tok for tok in toks if tok in query_vocab})
        if (i + 1) % 250000 == 0:
            print(f"    df pass: {i + 1:,}/{n_docs:,} docs")

    kept_terms = {
        term for term, freq in df.items()
        if freq >= min_df and freq <= max_df
    }
    avgdl = sum(doc_lens) / max(len(doc_lens), 1)
    print(
        f"  Kept {len(kept_terms):,}/{len(query_vocab):,} query terms "
        f"(min_df={min_df}, max_df_ratio={max_df_ratio})"
    )

    postings = defaultdict(list)
    for i, pid in enumerate(all_pids):
        counts = {}
        for tok in simple_tokenize(corpus_text(corpus, pid)):
            if tok in kept_terms:
                counts[tok] = counts.get(tok, 0) + 1
        for term, tf in counts.items():
            postings[term].append((i, tf))
        if (i + 1) % 250000 == 0:
            print(f"    postings pass: {i + 1:,}/{n_docs:,} docs")

    idf = {
        term: math.log(1.0 + (n_docs - len(term_postings) + 0.5) / (len(term_postings) + 0.5))
        for term, term_postings in postings.items()
    }
    if max_postings_per_term and max_postings_per_term > 0:
        print(f"  Pruning postings to top {max_postings_per_term:,} docs per term")
        n_pruned_terms = 0
        n_pruned_postings = 0
        for term, term_postings in list(postings.items()):
            if len(term_postings) <= max_postings_per_term:
                continue
            term_idf = idf[term]

            def approx_contribution(item):
                doc_idx, tf = item
                dl = doc_lens[doc_idx]
                # Same BM25 term contribution as scoring; this makes pruning
                # deterministic and biased toward docs where the term matters.
                denom = tf + 1.2 * (1.0 - 0.75 + 0.75 * dl / max(avgdl, 1e-6))
                return term_idf * (tf * (1.2 + 1.0)) / denom

            old_len = len(term_postings)
            postings[term] = heapq.nlargest(max_postings_per_term, term_postings, key=approx_contribution)
            n_pruned_terms += 1
            n_pruned_postings += old_len - len(postings[term])
        print(f"  Pruned {n_pruned_postings:,} postings from {n_pruned_terms:,} terms")
    return {
        "postings": dict(postings),
        "idf": idf,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
    }


def score_lite_bm25(query_text, index, k1=1.2, b=0.75):
    scores = defaultdict(float)
    avgdl = index["avgdl"]
    doc_lens = index["doc_lens"]
    postings = index["postings"]
    idf = index["idf"]

    for term in set(query_terms(query_text)):
        term_postings = postings.get(term)
        if not term_postings:
            continue
        term_idf = idf[term]
        for doc_idx, tf in term_postings:
            dl = doc_lens[doc_idx]
            denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1e-6))
            scores[doc_idx] += term_idf * (tf * (k1 + 1.0)) / denom
    return scores


def mine_lite_one(task):
    qid, query_text, rels, seed = task
    rng = random.Random(f"{seed}:{qid}")
    rel_pids = set(rels.keys())
    scores = score_lite_bm25(query_text, _LITE_INDEX)
    top_items = heapq.nlargest(
        _LITE_BM25_TOPK + len(rel_pids) + _LITE_NEG_PER_QUERY,
        scores.items(),
        key=lambda item: item[1],
    )

    neg_ids = []
    for doc_idx, _ in top_items:
        pid = _LITE_ALL_PIDS[doc_idx]
        if pid in rel_pids:
            continue
        neg_ids.append(pid)
        if len(neg_ids) >= _LITE_NEG_PER_QUERY:
            break

    if len(neg_ids) < _LITE_NEG_PER_QUERY:
        rel_set = rel_pids | set(neg_ids)
        need = _LITE_NEG_PER_QUERY - len(neg_ids)
        neg_ids.extend(sample_non_relevant(_LITE_ALL_PIDS, rel_set, need, rng))

    return qid, neg_ids


def mine_bm25_negatives_lite(
    index,
    all_pids,
    queries,
    qrels,
    neg_per_query=8,
    bm25_topk=50,
    rng=None,
    workers=1,
    seed=42,
):
    print(
        f"  Mining lite BM25 negatives "
        f"(topk={bm25_topk}, neg_per_query={neg_per_query}, workers={workers})..."
    )
    rng = rng or random.Random(42)
    query_negatives = {}
    global _LITE_INDEX, _LITE_ALL_PIDS, _LITE_NEG_PER_QUERY, _LITE_BM25_TOPK
    _LITE_INDEX = index
    _LITE_ALL_PIDS = all_pids
    _LITE_NEG_PER_QUERY = neg_per_query
    _LITE_BM25_TOPK = bm25_topk

    tasks = [
        (qid, query_text, qrels[qid], seed)
        for qid, query_text in queries.items()
        if qid in qrels
    ]

    if workers and workers > 1:
        try:
            ctx = get_context("fork")
            with ctx.Pool(processes=workers) as pool:
                for i, (qid, neg_ids) in enumerate(pool.imap_unordered(mine_lite_one, tasks, chunksize=64), start=1):
                    query_negatives[qid] = neg_ids
                    if i % 500 == 0:
                        print(f"    {i:,}/{len(tasks):,} queries processed")
            print(f"  Done: {len(query_negatives)} queries with lite BM25 negatives")
            return query_negatives
        except ValueError:
            print("  multiprocessing fork context is unavailable; falling back to one worker")

    for i, task in enumerate(tasks, start=1):
        qid, neg_ids = mine_lite_one(task)
        query_negatives[qid] = neg_ids
        if i % 500 == 0:
            print(f"    {i:,}/{len(tasks):,} queries processed")

    print(f"  Done: {len(query_negatives)} queries with lite BM25 negatives")
    return query_negatives


def build_train_rows(queries, qrels, corpus, all_pids,
                     neg_per_query=8, seed=42,
                     neg_mode="bm25_lite", bm25_topk=50,
                     bm25_max_df_ratio=0.15,
                     bm25_min_df=2,
                     bm25_max_postings_per_term=0,
                     bm25_workers=1):
    rng = random.Random(seed)
    rows = []

    bm25_negatives = None
    if neg_mode == "bm25":
        neg_mode = "bm25_lite"

    if neg_mode == "bm25_full":
        bm25 = build_bm25_index(corpus, all_pids)
        bm25_negatives = mine_bm25_negatives_full(
            bm25, all_pids, queries, qrels,
            neg_per_query=neg_per_query, bm25_topk=bm25_topk,
        )
    elif neg_mode == "bm25_lite":
        lite_index = build_lite_bm25_index(
            corpus,
            all_pids,
            queries,
            max_df_ratio=bm25_max_df_ratio,
            min_df=bm25_min_df,
            max_postings_per_term=bm25_max_postings_per_term,
        )
        bm25_negatives = mine_bm25_negatives_lite(
            lite_index,
            all_pids,
            queries,
            qrels,
            neg_per_query=neg_per_query,
            bm25_topk=bm25_topk,
            rng=rng,
            workers=bm25_workers,
            seed=seed,
        )

    for qid, rels in qrels.items():
        if qid not in queries:
            continue

        pos_ids = [pid for pid, rel in rels.items() if rel > 0 and pid in corpus]
        if not pos_ids:
            continue

        positives = [corpus_text(corpus, pid) for pid in pos_ids]

        if neg_mode in ("bm25_lite", "bm25_full") and bm25_negatives and qid in bm25_negatives:
            neg_ids = bm25_negatives[qid]
            if len(neg_ids) < neg_per_query:
                rel_set = set(rels.keys()) | set(neg_ids)
                need = neg_per_query - len(neg_ids)
                neg_ids += sample_non_relevant(all_pids, rel_set, need, rng)
        else:
            neg_ids = sample_non_relevant(all_pids, set(rels.keys()), neg_per_query, rng)

        if not neg_ids:
            continue

        negatives = [corpus_text(corpus, nid) for nid in neg_ids]

        rows.append({
            "qid": qid,
            "query": queries[qid],
            "pos_ids": pos_ids,
            "positives": positives,
            "neg_ids": neg_ids,
            "negatives": negatives,
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beir_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset", default=None,
                    help="BEIR dataset name for auto-download; defaults to the basename of --beir_dir")
    ap.add_argument("--download", action="store_true",
                    help="Download and unzip the BEIR dataset if --beir_dir is missing corpus.jsonl")
    ap.add_argument("--download_root", default=None,
                    help="Where to place downloaded BEIR zips; defaults to parent of --beir_dir")
    ap.add_argument("--neg_per_query", type=int, default=8)
    ap.add_argument("--neg_mode", choices=["bm25", "bm25_lite", "bm25_full", "random"], default="bm25_lite",
                    help="bm25/bm25_lite = query-term postings BM25, bm25_full = rank_bm25 full corpus, random = random negatives")
    ap.add_argument("--bm25_topk", type=int, default=50,
                    help="BM25 retrieval depth to mine negatives from")
    ap.add_argument("--bm25_max_df_ratio", type=float, default=0.15,
                    help="For bm25_lite, discard query terms appearing in more than this corpus fraction")
    ap.add_argument("--bm25_min_df", type=int, default=2,
                    help="For bm25_lite, discard query terms appearing in fewer docs than this")
    ap.add_argument("--bm25_max_postings_per_term", type=int, default=0,
                    help="For bm25_lite, keep only this many highest-contribution postings per term; 0 disables pruning")
    ap.add_argument("--bm25_workers", type=int, default=1,
                    help="For bm25_lite, number of forked workers for query mining")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    beir_dir = Path(args.beir_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        beir_dir = ensure_beir_dataset(
            beir_dir,
            dataset=args.dataset or beir_dir.name,
            download_root=Path(args.download_root) if args.download_root else beir_dir.parent,
        )

    print(f"Loading corpus from {beir_dir}...")
    corpus = load_beir_corpus(beir_dir)
    all_pids = list(corpus.keys())
    print(f"  {len(corpus)} documents")

    print("Loading queries...")
    queries = load_beir_queries(beir_dir)
    print(f"  {len(queries)} queries")

    # ---- collection.jsonl ------------------------------------------------
    print("Writing collection.jsonl...")
    with (output_dir / "collection.jsonl").open("w", encoding="utf-8") as f:
        for pid, doc in corpus.items():
            json.dump({"pid": pid, "title": doc["title"], "text": doc["text"]}, f, ensure_ascii=False)
            f.write("\n")

    # ---- process each split ----------------------------------------------
    qrels_dir = beir_dir / "qrels"

    for split in ["train", "dev", "test"]:
        tsv_path = qrels_dir / f"{split}.tsv"
        if not tsv_path.exists():
            print(f"  [{split}] qrels not found, skipping")
            continue

        qrels = load_beir_qrels(tsv_path)
        split_qids = set(qrels.keys())
        split_queries = {qid: queries[qid] for qid in split_qids if qid in queries}
        print(f"  [{split}] {len(split_queries)} queries, {sum(len(v) for v in qrels.values())} judgments")

        with (output_dir / f"qrels_{split}.json").open("w", encoding="utf-8") as f:
            json.dump(qrels, f, ensure_ascii=False)

        with (output_dir / f"queries_{split}.jsonl").open("w", encoding="utf-8") as f:
            for qid, text in split_queries.items():
                json.dump({"qid": qid, "query": text}, f, ensure_ascii=False)
                f.write("\n")

        if split == "train":
            print(f"  Building train rows with {args.neg_mode} negatives...")
            rows = build_train_rows(
                split_queries, qrels, corpus, all_pids,
                neg_per_query=args.neg_per_query,
                seed=args.seed,
                neg_mode=args.neg_mode,
                bm25_topk=args.bm25_topk,
                bm25_max_df_ratio=args.bm25_max_df_ratio,
                bm25_min_df=args.bm25_min_df,
                bm25_max_postings_per_term=args.bm25_max_postings_per_term,
                bm25_workers=args.bm25_workers,
            )
            with (output_dir / "train.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")
            print(f"  {len(rows)} training examples written")

    print(f"\nDone! Output in {output_dir}")


if __name__ == "__main__":
    main()
