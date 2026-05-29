"""
Prepare adjustment files for ablation experiments on fiqa_rlhn:
1. Oracle: RLHN FN ground truth as mask
2. Random (x3 seeds): same count as margin_persistent_3plus, random pairs
"""

import json
import random
import numpy as np

NUM_NEG = 7
MASK_VAL = -1e9
N_QUERIES = 5500

# Load query IDs from training data
qids = []
with open("data/processed/fiqa_rlhn/train.jsonl") as f:
    for line in f:
        row = json.loads(line)
        qids.append(row["qid"])

# 1. Oracle: RLHN FN ground truth
with open("data/processed/fiqa_rlhn/fn_ground_truth.json") as f:
    gt = json.load(f)

oracle_adj = {}
for qidx_str, neg_indices in gt["fn_pairs"].items():
    qid = qids[int(qidx_str)]
    if qid not in oracle_adj:
        oracle_adj[qid] = np.zeros(NUM_NEG, dtype=np.float32)
    for ni in neg_indices:
        if ni < NUM_NEG:
            oracle_adj[qid][ni] = MASK_VAL

oracle_count = sum(1 for a in oracle_adj.values() for v in a if v < -1e8)
print(f"Oracle: {oracle_count} masked pairs, {len(oracle_adj)} queries")

with open("experiments/fiqa_rlhn/adj_oracle.json", "w") as f:
    json.dump({k: v.tolist() for k, v in oracle_adj.items()}, f)

# 2. Random baselines (match margin_persistent_3plus count = 597)
TARGET_COUNT = 597

for seed in [42, 123, 456]:
    random.seed(seed)
    random_pairs = set()
    while len(random_pairs) < TARGET_COUNT:
        qi = random.randint(0, N_QUERIES - 1)
        ni = random.randint(0, NUM_NEG - 1)
        random_pairs.add((qids[qi], ni))

    random_adj = {}
    for qid, ni in random_pairs:
        if qid not in random_adj:
            random_adj[qid] = np.zeros(NUM_NEG, dtype=np.float32)
        random_adj[qid][ni] = MASK_VAL

    n_masked = sum(1 for a in random_adj.values() for v in a if v < -1e8)
    print(f"Random seed={seed}: {n_masked} masked pairs, {len(random_adj)} queries")

    with open(f"experiments/fiqa_rlhn/adj_random_s{seed}.json", "w") as f:
        json.dump({k: v.tolist() for k, v in random_adj.items()}, f)

# 3. Also save margin_persistent_3plus for seed runs (need to regenerate criteria)
# The adj already exists from the multi-criteria run, just copy for clarity
print("\nAll adjustment files saved.")
