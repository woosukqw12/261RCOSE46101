"""
Compare training dynamics criteria with RLHN FN ground truth on fiqa_rlhn.

python src/eval_dynamics_vs_rlhn.py
"""

import json
import subprocess
import sys
import tempfile

# Load FN ground truth
with open("data/processed/fiqa_rlhn/fn_ground_truth.json") as f:
    gt_data = json.load(f)

# Build index->qid mapping from training data
idx_to_qid = {}
with open("data/processed/fiqa_rlhn/train.jsonl") as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        idx_to_qid[i] = row["qid"]

fn_set = set()
for qidx_str, neg_indices in gt_data["fn_pairs"].items():
    qid = idx_to_qid[int(qidx_str)]
    for ni in neg_indices:
        fn_set.add((qid, ni))

print(f"RLHN FN ground truth: {len(fn_set)} pairs")
print(f"  {gt_data['stats']['queries_with_fn']} queries, FN rate {gt_data['stats']['fn_rate']*100:.2f}%")

# Compute training dynamics
print("\nComputing training dynamics signals...")
with tempfile.TemporaryDirectory() as tmpdir:
    cmd = [
        sys.executable, "src/compute_signals.py",
        "--log_dir", "experiments/fiqa_rlhn/logs_baseline",
        "--output_dir", tmpdir,
        "--source", "loss",
    ]
    subprocess.run(cmd, check=True)

    with open(f"{tmpdir}/criteria_loss.json") as f:
        criteria = json.load(f)

# Compare
print(f"\n{'='*80}")
print(f"Training Dynamics vs RLHN FN Ground Truth (FiQA, {len(fn_set)} FN pairs)")
print(f"{'='*80}")
print(f"  {'Criterion':<35} {'Count':>6} {'TP':>6} {'FP':>6} {'Prec':>7} {'Recall':>7} {'F1':>7}")
print(f"  {'-'*80}")

results = {}
for name in sorted(criteria.keys()):
    pairs = criteria[name]
    pred_set = set(tuple(p) for p in pairs)
    tp = len(pred_set & fn_set)
    fp = len(pred_set) - tp
    fn_missed = len(fn_set) - tp
    prec = tp / len(pred_set) if pred_set else 0
    rec = tp / len(fn_set) if fn_set else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    results[name] = {"count": len(pred_set), "tp": tp, "fp": fp, "precision": prec, "recall": rec, "f1": f1}
    if len(pred_set) > 0:
        print(f"  {name:<35} {len(pred_set):>6} {tp:>6} {fp:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

# Also show top criteria by F1
print(f"\n  Top criteria by F1:")
for name, r in sorted(results.items(), key=lambda x: -x[1]["f1"])[:10]:
    if r["count"] > 0:
        print(f"    {name:<35} F1={r['f1']:.3f} (P={r['precision']:.3f} R={r['recall']:.3f}) [{r['count']} predicted]")

# Random baseline
import random
random.seed(42)
all_qids = list(idx_to_qid.values())
total_pairs = 5500 * 7
for target_count in [len(fn_set), 597, 1308]:  # match sizes of key criteria
    random_pred = set()
    while len(random_pred) < target_count:
        random_pred.add((all_qids[random.randint(0, 5499)], random.randint(0, 6)))
    random_tp = len(random_pred & fn_set)
    random_prec = random_tp / len(random_pred) if random_pred else 0
    random_rec = random_tp / len(fn_set) if fn_set else 0
    random_f1 = 2 * random_prec * random_rec / (random_prec + random_rec) if (random_prec + random_rec) > 0 else 0
    print(f"\n  Random baseline ({target_count} pairs): P={random_prec:.3f} R={random_rec:.3f} F1={random_f1:.3f}")

with open("results/fiqa_rlhn_dynamics_vs_gt.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → results/fiqa_rlhn_dynamics_vs_gt.json")
