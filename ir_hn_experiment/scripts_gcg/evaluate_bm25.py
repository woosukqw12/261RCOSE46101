import argparse
import json
import math
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm


TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())


def load_collection(path: Path):
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = str(row["pid"])
            text = ((row.get("title", "") or "") + " " + (row.get("text", "") or "")).strip()
            docs.append((pid, text))
    return docs


def load_queries(path: Path):
    queries = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["qid"])] = row["query"]
    return queries


def load_qrels(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bm25_retrieve(docs, queries, topk=100):
    doc_ids = [pid for pid, _ in docs]
    bm25 = BM25Okapi([tokenize(text) for _, text in docs])

    results = {}
    for qid, query in tqdm(queries.items(), desc="bm25"):
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        results[qid] = {doc_ids[i]: float(scores[i]) for i in ranked}
    return results


def dcg(rels):
    s = 0.0
    for i, rel in enumerate(rels):
        s += (2 ** rel - 1) / math.log2(i + 2)
    return s


def evaluate(qrels, results, k=10):
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_collection(dataset_dir / "collection.jsonl")

    final_metrics = {}
    for split in ["dev", "test"]:
        q_path = dataset_dir / f"queries_{split}.jsonl"
        r_path = dataset_dir / f"qrels_{split}.json"
        if not q_path.exists() or not r_path.exists():
            continue

        queries = load_queries(q_path)
        qrels = load_qrels(r_path)
        results = bm25_retrieve(docs, queries, topk=args.topk)
        metrics = evaluate(qrels, results, k=10)
        final_metrics[split] = metrics

        with (output_dir / f"run_{split}.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
