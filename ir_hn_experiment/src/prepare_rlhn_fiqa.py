"""
Convert RLHN default-680K fiqa subset to our training format.
Also extract FN ground truth by comparing default vs rlhn.
"""

import json
from datasets import load_dataset
from pathlib import Path

out_dir = Path("data/processed/fiqa_rlhn")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")
default_ds = load_dataset("rlhn/default-680K", split="train")
rlhn_ds = load_dataset("rlhn/rlhn-680K", split="train")

# Index rlhn by query_id
rlhn_by_qid = {}
for r in rlhn_ds:
    if r["subset"] == "fiqa":
        rlhn_by_qid[r["query_id"]] = r

# Process default fiqa
rows = []
fn_ground_truth = {}  # query_idx -> list of neg_idx that are FN

for r in default_ds:
    if r["subset"] != "fiqa":
        continue

    qid = r["query_id"]
    query = r["query"]
    positives = [p["text"] for p in r["positive_passages"]]
    negatives = [p["text"] for p in r["negative_passages"]]
    neg_docids = [p["docid"] for p in r["negative_passages"]]

    # Limit to 7 negatives (like our other experiments)
    negatives = negatives[:7]
    neg_docids = neg_docids[:7]

    if not positives or not negatives:
        continue

    row = {
        "qid": qid,
        "query": query,
        "positives": positives,
        "pos_ids": [p["docid"] for p in r["positive_passages"]],
        "negatives": negatives,
        "neg_ids": neg_docids,
    }
    query_idx = len(rows)
    rows.append(row)

    # Find FN: negatives in default that became positives in rlhn
    if qid in rlhn_by_qid:
        rlhn_row = rlhn_by_qid[qid]
        rlhn_pos_ids = {p["docid"] for p in rlhn_row["positive_passages"]}
        default_pos_ids = {p["docid"] for p in r["positive_passages"]}
        fn_indices = []
        for ni, docid in enumerate(neg_docids):
            if docid in rlhn_pos_ids and docid not in default_pos_ids:
                fn_indices.append(ni)
        if fn_indices:
            fn_ground_truth[query_idx] = fn_indices

# Save training data
with open(out_dir / "train.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(f"Saved {len(rows)} queries → {out_dir / 'train.jsonl'}")

# Save FN ground truth
total_fn = sum(len(v) for v in fn_ground_truth.values())
print(f"FN ground truth: {len(fn_ground_truth)} queries, {total_fn} pairs")
print(f"FN rate: {total_fn}/{len(rows)*7} = {100*total_fn/(len(rows)*7):.2f}%")

with open(out_dir / "fn_ground_truth.json", "w") as f:
    json.dump({
        "stats": {
            "total_queries": len(rows),
            "queries_with_fn": len(fn_ground_truth),
            "total_fn_pairs": total_fn,
            "fn_rate": round(total_fn / (len(rows) * 7), 4),
        },
        "fn_pairs": {str(k): v for k, v in fn_ground_truth.items()},
    }, f, indent=2)
print(f"Saved → {out_dir / 'fn_ground_truth.json'}")

# Also copy test data from existing fiqa (same evaluation set)
import shutil
for fname in ["corpus.jsonl", "queries.jsonl", "qrels.tsv"]:
    src = Path("data/processed/fiqa") / fname
    if src.exists():
        shutil.copy(src, out_dir / fname)
        print(f"Copied {fname}")
    else:
        # Try beir_raw
        src2 = Path("data/beir_raw/fiqa") / fname
        if src2.exists():
            shutil.copy(src2, out_dir / fname)
            print(f"Copied {fname} from beir_raw")
