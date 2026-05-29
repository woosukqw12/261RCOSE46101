"""
Compare RLHN default vs rlhn datasets for FiQA subset.
Extract which negatives were relabeled as positives (= LLM judge FN labels).
Then compare with our training dynamics signals.
"""

import json
from collections import defaultdict
from datasets import load_dataset

print("Loading rlhn/default-680K (fiqa subset)...")
default_ds = load_dataset("rlhn/default-680K", split="train")
default_fiqa = [r for r in default_ds if r["subset"] == "fiqa"]
print(f"  default fiqa: {len(default_fiqa)} queries")

print("Loading rlhn/rlhn-680K (fiqa subset)...")
rlhn_ds = load_dataset("rlhn/rlhn-680K", split="train")
rlhn_fiqa = [r for r in rlhn_ds if r["subset"] == "fiqa"]
print(f"  rlhn fiqa: {len(rlhn_fiqa)} queries")

# Index by query_id
default_by_qid = {r["query_id"]: r for r in default_fiqa}
rlhn_by_qid = {r["query_id"]: r for r in rlhn_fiqa}

common_qids = set(default_by_qid) & set(rlhn_by_qid)
print(f"  Common query_ids: {len(common_qids)}")

# Find negatives that moved to positives
fn_labels = {}  # query_id -> set of docids that are FN
stats = {"total_neg": 0, "moved_to_pos": 0, "removed_from_neg": 0, "queries_with_fn": 0}

for qid in common_qids:
    default_row = default_by_qid[qid]
    rlhn_row = rlhn_by_qid[qid]

    default_neg_ids = {p["docid"] for p in default_row["negative_passages"]}
    rlhn_neg_ids = {p["docid"] for p in rlhn_row["negative_passages"]}
    rlhn_pos_ids = {p["docid"] for p in rlhn_row["positive_passages"]}
    default_pos_ids = {p["docid"] for p in default_row["positive_passages"]}

    stats["total_neg"] += len(default_neg_ids)

    # Negatives in default that became positives in rlhn = FN
    moved = default_neg_ids & rlhn_pos_ids - default_pos_ids
    # Negatives in default that were removed (not in rlhn neg or pos)
    removed = default_neg_ids - rlhn_neg_ids - rlhn_pos_ids

    if moved:
        fn_labels[qid] = moved
        stats["moved_to_pos"] += len(moved)
        stats["queries_with_fn"] += 1
    stats["removed_from_neg"] += len(removed)

print(f"\n{'='*60}")
print(f"FiQA FN Analysis (RLHN LLM Judge Ground Truth)")
print(f"{'='*60}")
print(f"  Total queries: {len(common_qids)}")
print(f"  Total negatives: {stats['total_neg']}")
print(f"  Moved neg→pos (FN): {stats['moved_to_pos']} ({100*stats['moved_to_pos']/stats['total_neg']:.2f}%)")
print(f"  Removed from neg: {stats['removed_from_neg']}")
print(f"  Queries with FN: {stats['queries_with_fn']} ({100*stats['queries_with_fn']/len(common_qids):.2f}%)")

# Save FN labels
output = {
    "stats": stats,
    "fn_by_query": {qid: list(docids) for qid, docids in fn_labels.items()},
}
with open("results/fiqa_rlhn_fn_labels.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved → results/fiqa_rlhn_fn_labels.json")

# Now try to map to our training data indices
# Our training data uses index-based negatives (0-6), not docids
# We need to load our processed fiqa data and match
print(f"\n{'='*60}")
print("Mapping RLHN FN labels to our training data indices...")

our_data = []
with open("data/processed/fiqa/train.jsonl") as f:
    for line in f:
        our_data.append(json.loads(line))
print(f"  Our training data: {len(our_data)} queries")

# Build mapping: our query text -> index
# RLHN uses query_id (md5 hash), we need to match by query text
our_by_query_text = {}
for i, row in enumerate(our_data):
    q = row["query"]
    our_by_query_text[q] = (i, row)

# Map RLHN FN to our neg indices
mapped_fn = []  # (query_idx, neg_idx) pairs matching our format
mapped_count = 0
unmapped_count = 0

for qid in fn_labels:
    rlhn_row = rlhn_by_qid[qid]
    default_row = default_by_qid[qid]
    query_text = default_row["query"]

    if query_text not in our_by_query_text:
        unmapped_count += 1
        continue

    our_idx, our_row = our_by_query_text[query_text]
    fn_docids = fn_labels[qid]

    # Match by negative passage text (our data might have different docids)
    default_negs = default_row["negative_passages"]
    our_negs = our_row.get("negatives", our_row.get("hard_negatives", []))

    # Try matching by text content
    for fn_docid in fn_docids:
        # Find which default negative this is
        fn_text = None
        for neg in default_negs:
            if neg["docid"] == fn_docid:
                fn_text = neg["text"]
                break

        if fn_text is None:
            continue

        # Find matching neg in our data
        for neg_idx, our_neg in enumerate(our_negs):
            our_neg_text = our_neg if isinstance(our_neg, str) else our_neg.get("text", "")
            if our_neg_text == fn_text or fn_text in our_neg_text or our_neg_text in fn_text:
                mapped_fn.append((our_idx, neg_idx))
                mapped_count += 1
                break

print(f"  Mapped FN pairs: {mapped_count}")
print(f"  Unmapped queries: {unmapped_count}")

# Now compare with our training dynamics
print(f"\n{'='*60}")
print("Comparing with training dynamics signals...")

# Load our criteria from fiqa experiment
import subprocess, tempfile, sys
with tempfile.TemporaryDirectory() as tmpdir:
    cmd = [
        sys.executable, "src/compute_signals.py",
        "--log_dir", "experiments/fiqa/logs_baseline",
        "--output_dir", tmpdir,
        "--source", "loss",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    with open(f"{tmpdir}/criteria_loss.json") as f:
        criteria = json.load(f)

# Convert our criteria to sets of (query_idx, neg_idx)
criteria_sets = {}
for name, pairs in criteria.items():
    criteria_sets[name] = set(tuple(p) for p in pairs)

# RLHN FN ground truth set
rlhn_fn_set = set(tuple(p) for p in mapped_fn)
print(f"\n  RLHN FN ground truth: {len(rlhn_fn_set)} pairs")

# Compare each criterion
print(f"\n  {'Criterion':<30} {'Count':>7} {'∩ FN':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
print(f"  {'-'*70}")

for name in ["all_strict", "margin_persistent_3plus", "margin_avg_positive",
             "rank_persistent_all_top1", "rank_persistent_top1", "rank_final_top1",
             "cartography_hard", "cartography_ambiguous"]:
    s = criteria_sets.get(name, set())
    if not s and not rlhn_fn_set:
        continue
    overlap = s & rlhn_fn_set
    prec = len(overlap) / len(s) if s else 0
    rec = len(overlap) / len(rlhn_fn_set) if rlhn_fn_set else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  {name:<30} {len(s):>7} {len(overlap):>7} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

# Save full comparison
comparison = {
    "rlhn_fn_pairs": mapped_fn,
    "rlhn_fn_count": len(rlhn_fn_set),
    "criteria_comparison": {}
}
for name, s in criteria_sets.items():
    overlap = s & rlhn_fn_set
    prec = len(overlap) / len(s) if s else 0
    rec = len(overlap) / len(rlhn_fn_set) if rlhn_fn_set else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    comparison["criteria_comparison"][name] = {
        "count": len(s), "overlap": len(overlap),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)
    }

with open("results/fiqa_rlhn_vs_dynamics.json", "w") as f:
    json.dump(comparison, f, indent=2)
print(f"\nSaved → results/fiqa_rlhn_vs_dynamics.json")
