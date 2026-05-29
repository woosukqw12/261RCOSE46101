"""
Convert RLHN default-680K msmarco subset to our training format.
Also extract FN ground truth by comparing default vs rlhn.

Mirror of prepare_rlhn_fiqa.py. Run once before ablation.
"""

import argparse
import json
from datasets import load_dataset
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="data/processed/msmarco_rlhn")
parser.add_argument(
    "--skip_fn_gt",
    action="store_true",
    help="Only create train.jsonl from rlhn/default-680K. This is enough for label-free avg-margin relabel experiments.",
)
args = parser.parse_args()

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
MSMARCO_SUBSETS = {"msmarco", "msmarco_passage"}

print("Loading rlhn/default-680K ...")
default_ds = load_dataset("rlhn/default-680K", split="train")
rlhn_by_qid = {}
if not args.skip_fn_gt:
    print("Loading rlhn/rlhn-680K ...")
    rlhn_ds = load_dataset("rlhn/rlhn-680K", split="train")

    # Index rlhn by query_id for fast lookup
    for r in rlhn_ds:
        if r["subset"] in MSMARCO_SUBSETS:
            rlhn_by_qid[r["query_id"]] = r
    print(f"rlhn msmarco queries: {len(rlhn_by_qid)}")
else:
    print("Skipping rlhn/rlhn-680K; FN GT will not be generated.")

rows = []
fn_ground_truth = {}

for r in default_ds:
    if r["subset"] not in MSMARCO_SUBSETS:
        continue
    qid = r["query_id"]
    positives = [p["text"] for p in r["positive_passages"]]
    negatives = [p["text"] for p in r["negative_passages"]][:7]
    neg_docids = [p["docid"] for p in r["negative_passages"]][:7]

    if not positives or not negatives:
        continue

    row = {
        "qid": qid,
        "query": r["query"],
        "positives": positives,
        "pos_ids": [p["docid"] for p in r["positive_passages"]],
        "negatives": negatives,
        "neg_ids": neg_docids,
    }
    query_idx = len(rows)
    rows.append(row)

    if rlhn_by_qid and qid in rlhn_by_qid:
        rlhn_row = rlhn_by_qid[qid]
        rlhn_pos_ids = {p["docid"] for p in rlhn_row["positive_passages"]}
        default_pos_ids = {p["docid"] for p in r["positive_passages"]}
        fn_indices = [ni for ni, d in enumerate(neg_docids)
                      if d in rlhn_pos_ids and d not in default_pos_ids]
        if fn_indices:
            fn_ground_truth[query_idx] = fn_indices

with open(out_dir / "train.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(f"Saved {len(rows)} queries → {out_dir / 'train.jsonl'}")

if not args.skip_fn_gt:
    total_fn = sum(len(v) for v in fn_ground_truth.values())
    print(f"FN ground truth: {len(fn_ground_truth)} queries, {total_fn} pairs")
    print(f"FN rate: {100*total_fn/(len(rows)*7):.2f}%")

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
