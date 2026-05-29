"""
Convert RLHN's relabeled fiqa data (rlhn/rlhn-680K) to our training format.
Expand multi-positive rows into separate training instances,
matching our relabeling approach (one positive per instance).
"""

import json
from datasets import load_dataset
from pathlib import Path

out_dir = Path("data/processed/fiqa_rlhn")

print("Loading rlhn/rlhn-680K (fiqa subset)...")
ds = load_dataset("rlhn/rlhn-680K", split="train")
rlhn_fiqa = [r for r in ds if r["subset"] == "fiqa"]
print(f"  {len(rlhn_fiqa)} queries")

# Also load default to identify which positives are "original" vs "relabeled"
print("Loading rlhn/default-680K (fiqa subset)...")
default_ds = load_dataset("rlhn/default-680K", split="train")
default_by_qid = {}
for r in default_ds:
    if r["subset"] == "fiqa":
        default_by_qid[r["query_id"]] = {p["docid"] for p in r["positive_passages"]}

# Expand: each positive gets its own training instance
rows = []
n_original = 0
n_relabeled = 0

for r in rlhn_fiqa:
    qid = r["query_id"]
    query = r["query"]
    negatives = [p["text"] for p in r["negative_passages"]][:7]
    neg_ids = [p["docid"] for p in r["negative_passages"]][:7]

    if not negatives:
        continue

    original_pos_ids = default_by_qid.get(qid, set())

    for p in r["positive_passages"]:
        rows.append({
            "qid": qid,
            "query": query,
            "positives": [p["text"]],
            "pos_ids": [p["docid"]],
            "negatives": negatives,
            "neg_ids": neg_ids,
        })
        if p["docid"] in original_pos_ids:
            n_original += 1
        else:
            n_relabeled += 1

out_path = out_dir / "train_rlhn_relabeled.jsonl"
with open(out_path, "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

print(f"  Total instances: {len(rows)}")
print(f"  Original positive instances: {n_original}")
print(f"  Relabeled FN instances: {n_relabeled}")
print(f"  Saved → {out_path}")
