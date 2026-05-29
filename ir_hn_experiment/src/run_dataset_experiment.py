"""
Dataset-specific curriculum learning pipeline.

Full pipeline for a single BEIR dataset:
  1. Train baseline → training logs
  2. Compute training dynamics signals
  3. Train with mask_fn curriculum
  4. Evaluate both on test set

python src/run_dataset_experiment.py --dataset scifact
python src/run_dataset_experiment.py --dataset scidocs
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

faiss = None
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "scifact": {"epochs": 10, "batch_size": 16, "lr": 2e-5, "num_neg": 7},
    "scidocs": {"epochs": 10, "batch_size": 16, "lr": 2e-5, "num_neg": 7},
    "nfcorpus": {"epochs": 10, "batch_size": 16, "lr": 2e-5, "num_neg": 7},
    "fiqa":    {"epochs": 5,  "batch_size": 16, "lr": 2e-5, "num_neg": 7},
    "fiqa_rlhn": {"epochs": 5, "batch_size": 16, "lr": 2e-5, "num_neg": 7},
}

MODEL_NAME = "intfloat/e5-base-unsupervised"
MAX_LEN = 350
TEMPERATURE = 0.05
WARMUP_RATIO = 0.1
MASK_VAL = -1e9


# ── Model ─────────────────────────────────────────────────────────────────────

class E5BiEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(emb).to(emb.dtype)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1)


# ── Dataset (BEIR processed format) ──────────────────────────────────────────

class BeirTrainDataset(Dataset):
    def __init__(self, train_path, num_neg=7, adjustments=None):
        self.num_neg = num_neg
        self.adjustments = adjustments or {}
        self.instances = []

        with open(train_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                negs = item["negatives"][:num_neg]
                if not item["positives"] or not negs:
                    continue
                self.instances.append({
                    "qid": str(item["qid"]),
                    "query": item["query"],
                    "pos": item["positives"][0],
                    "negs": negs,
                })

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        negs = list(inst["negs"])
        while len(negs) < self.num_neg:
            negs.append(negs[-1])
        adj = self.adjustments.get(inst["qid"], np.zeros(self.num_neg, dtype=np.float32))
        return {
            "qid": inst["qid"],
            "query_text": "query: " + inst["query"],
            "pos_text": "passage: " + inst["pos"],
            "neg_texts": ["passage: " + n for n in negs],
            "neg_adj": adj,
        }


def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        qids = [b["qid"] for b in batch]
        queries = tokenizer([b["query_text"] for b in batch],
                            max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        positives = tokenizer([b["pos_text"] for b in batch],
                              max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        negatives = tokenizer([t for b in batch for t in b["neg_texts"]],
                              max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        neg_adj = torch.tensor(np.stack([b["neg_adj"] for b in batch]), dtype=torch.float32)
        return {"qids": qids, "query": queries, "positive": positives,
                "negatives": negatives, "num_neg": len(batch[0]["neg_texts"]), "neg_adj": neg_adj}
    return collate_fn


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    train_path,
    log_dir,
    ckpt_dir,
    num_neg,
    epochs,
    batch_size,
    lr,
    adjustments=None,
    gradient_checkpointing=False,
    dataloader_workers=2,
    save_every_epoch=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = E5BiEncoder(MODEL_NAME).to(device)
    if gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    else:
        model.encoder.gradient_checkpointing_disable()

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = BeirTrainDataset(train_path, num_neg, adjustments)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=make_collate_fn(tokenizer, MAX_LEN),
                        num_workers=dataloader_workers, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    use_adj = adjustments is not None and len(adjustments) > 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        log_path = os.path.join(log_dir, f"epoch_{epoch}.jsonl")

        with open(log_path, "w", encoding="utf-8") as log_file:
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                B = len(batch["qids"])
                num_neg_b = batch["num_neg"]
                optimizer.zero_grad(set_to_none=True)

                q_enc = {k: v.to(device, non_blocking=True) for k, v in batch["query"].items()}
                p_enc = {k: v.to(device, non_blocking=True) for k, v in batch["positive"].items()}
                n_enc = {k: v.to(device, non_blocking=True) for k, v in batch["negatives"].items()}

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    q_emb = model.encode(**q_enc)
                    p_emb = model.encode(**p_enc)
                    n_emb = model.encode(**n_enc).view(B, num_neg_b, -1)

                    in_batch = q_emb @ p_emb.T
                    hard_neg = torch.bmm(q_emb.unsqueeze(1), n_emb.transpose(1, 2)).squeeze(1)

                    if use_adj:
                        neg_adj = batch["neg_adj"].to(device, non_blocking=True)
                        adjusted = hard_neg + neg_adj
                        all_scores = torch.cat([in_batch, adjusted], dim=1) / TEMPERATURE
                    else:
                        all_scores = torch.cat([in_batch, hard_neg], dim=1) / TEMPERATURE

                    loss = F.cross_entropy(all_scores, torch.arange(B, device=device))

                loss.backward()
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    ps = in_batch.diagonal().float().cpu().tolist()
                    ns = hard_neg.float().cpu().tolist()
                    for i, qid in enumerate(batch["qids"]):
                        log_file.write(json.dumps({
                            "step": global_step, "query_id": qid,
                            "pos_score": round(ps[i], 6),
                            "neg_scores": [round(s, 6) for s in ns[i]],
                        }) + "\n")

                epoch_loss += loss.item()
                global_step += 1

        avg = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")

        if save_every_epoch or epoch == epochs - 1:
            ckpt = os.path.join(ckpt_dir, f"epoch_{epoch}")
            os.makedirs(ckpt, exist_ok=True)
            model.encoder.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)

    del model, optimizer
    torch.cuda.empty_cache()
    return ckpt  # last checkpoint path


# ── Compute signals (inline, lightweight) ─────────────────────────────────────

def compute_criteria(log_dir, num_neg):
    """Compute all_strict criteria from training logs."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "src/compute_signals.py",
            "--log_dir", log_dir,
            "--output_dir", tmpdir,
            "--source", "loss",
        ]
        subprocess.run(cmd, check=True)

        criteria_path = os.path.join(tmpdir, "criteria_loss.json")
        with open(criteria_path, encoding="utf-8") as f:
            criteria = json.load(f)

        counts_path = os.path.join(tmpdir, "criteria_loss_counts.json")
        with open(counts_path, encoding="utf-8") as f:
            counts_data = json.load(f)
        n_queries = counts_data["total_pairs"] // num_neg

    # Pick best available criterion: all_strict > margin_persistent_3plus > rank_persistent_all_top1
    FALLBACK_ORDER = ["all_strict", "margin_persistent_3plus", "rank_persistent_all_top1"]
    selected_pairs = []
    selected_name = None
    for crit_name in FALLBACK_ORDER:
        pairs = criteria.get(crit_name, [])
        if pairs:
            selected_pairs = pairs
            selected_name = crit_name
            break

    if not selected_pairs:
        print("  No FN candidates found under any criterion.")
        return {}

    # Build adjustments: selected criterion → MASK_VAL
    adjustments = {}
    for qid, neg_idx in selected_pairs:
        if qid not in adjustments:
            adjustments[qid] = np.zeros(num_neg, dtype=np.float32)
        if neg_idx < num_neg:
            adjustments[qid][neg_idx] = MASK_VAL

    n_masked = sum(1 for a in adjustments.values() for v in a if v < -1e8)
    print(f"  {selected_name}: {len(selected_pairs)} pairs → {n_masked} masked ({len(adjustments)} queries)")

    # Print summary
    total = n_queries * num_neg
    for name in FALLBACK_ORDER + ["cartography_hard", "easy_cartography"]:
        count = len(criteria.get(name, []))
        print(f"    {name}: {count} ({100*count/total:.1f}%)")

    return adjustments


# ── Evaluate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, prefix="passage: ", batch_size=256):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.encoder(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.float().cpu().numpy())
    return np.vstack(all_embs)


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def evaluate_checkpoint(ckpt_path, data_dir, split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = E5BiEncoder(ckpt_path).to(device).eval()

    # Load collection
    pids, doc_texts = [], []
    with open(data_dir / "collection.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pids.append(str(row["pid"]))
            doc_texts.append(((row.get("title", "") or "") + " " + (row.get("text", "") or "")).strip())

    # Load queries
    qids, q_texts = [], []
    with open(data_dir / f"queries_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qids.append(str(row["qid"]))
            q_texts.append(row["query"])

    # Load qrels
    with open(data_dir / f"qrels_{split}.json", encoding="utf-8") as f:
        qrels = json.load(f)

    doc_embs = encode_texts(doc_texts, tokenizer, model, device, prefix="passage: ")
    q_embs = encode_texts(q_texts, tokenizer, model, device, prefix="query: ")

    sims = q_embs @ doc_embs.T
    indices = np.argpartition(-sims, kth=min(100, sims.shape[1] - 1), axis=1)[:, :100]
    part_scores = np.take_along_axis(sims, indices, axis=1)
    order = np.argsort(-part_scores, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    scores = np.take_along_axis(part_scores, order, axis=1)

    results = {}
    for qi, qid in enumerate(qids):
        results[qid] = {pids[idx]: float(scores[qi][r]) for r, idx in enumerate(indices[qi])}

    ndcg_scores, recall_scores = [], []
    for qid, pos in qrels.items():
        pos = {pid: rel for pid, rel in pos.items() if rel > 0}
        if not pos:
            continue
        ranked = sorted(results.get(qid, {}).items(), key=lambda x: -x[1])
        ranked_ids = [pid for pid, _ in ranked]
        topk = ranked_ids[:10]
        rels = [pos.get(pid, 0) for pid in topk]
        ideal = sorted(pos.values(), reverse=True)[:10]
        ndcg_scores.append(dcg(rels) / max(dcg(ideal), 1e-12))
        recall_scores.append(len(set(ranked_ids[:100]) & set(pos)) / len(pos))

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "nDCG@10": round(float(np.mean(ndcg_scores)), 4),
        "Recall@100": round(float(np.mean(recall_scores)), 4),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DEFAULTS.keys()))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_root", default="experiments")
    args = parser.parse_args()

    ds = args.dataset
    cfg = DEFAULTS[ds]
    data_dir = Path(args.data_dir) if args.data_dir else Path(f"data/processed/{ds}")
    out = Path(args.output_root) / ds
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {ds}")
    print(f"Config: {cfg}")
    print(f"Data: {data_dir}")
    print(f"Output: {out}")
    print(f"{'='*60}")

    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run prepare_beir.py first.")
        return

    # Step 1: Train baseline
    print(f"\n[1/4] Training baseline...")
    baseline_log = str(out / "logs_baseline")
    baseline_ckpt = str(out / "ckpt_baseline")
    if Path(baseline_ckpt).exists() and any(Path(baseline_ckpt).iterdir()):
        print("  Skipping (already exists)")
        last_epoch = max(int(d.name.split("_")[1]) for d in Path(baseline_ckpt).iterdir() if d.name.startswith("epoch_"))
        baseline_last = str(Path(baseline_ckpt) / f"epoch_{last_epoch}")
    else:
        baseline_last = train_model(train_path, baseline_log, baseline_ckpt,
                                     cfg["num_neg"], cfg["epochs"], cfg["batch_size"], cfg["lr"])

    # Step 2: Compute signals
    print(f"\n[2/4] Computing training dynamics signals...")
    adjustments = compute_criteria(baseline_log, cfg["num_neg"])

    if not adjustments:
        print("  No FN candidates found. Skipping curriculum training.")
        print("  Evaluating baseline only...")
        baseline_metrics = evaluate_checkpoint(baseline_last, data_dir)
        print(f"  Baseline: {baseline_metrics}")
        return

    # Step 3: Train with mask_fn
    print(f"\n[3/4] Training with mask_fn curriculum...")
    maskfn_log = str(out / "logs_mask_fn")
    maskfn_ckpt = str(out / "ckpt_mask_fn")
    if Path(maskfn_ckpt).exists() and any(Path(maskfn_ckpt).iterdir()):
        print("  Skipping (already exists)")
        last_epoch = max(int(d.name.split("_")[1]) for d in Path(maskfn_ckpt).iterdir() if d.name.startswith("epoch_"))
        maskfn_last = str(Path(maskfn_ckpt) / f"epoch_{last_epoch}")
    else:
        maskfn_last = train_model(train_path, maskfn_log, maskfn_ckpt,
                                   cfg["num_neg"], cfg["epochs"], cfg["batch_size"], cfg["lr"],
                                   adjustments=adjustments)

    # Step 4: Evaluate
    print(f"\n[4/4] Evaluating...")
    baseline_metrics = evaluate_checkpoint(baseline_last, data_dir)
    maskfn_metrics = evaluate_checkpoint(maskfn_last, data_dir)

    print(f"\n{'='*60}")
    print(f"RESULTS: {ds}")
    print(f"{'='*60}")
    print(f"  {'':>15} {'nDCG@10':>10} {'R@100':>10}")
    print(f"  {'Baseline':>15} {baseline_metrics['nDCG@10']:>10.4f} {baseline_metrics['Recall@100']:>10.4f}")
    print(f"  {'mask_fn':>15} {maskfn_metrics['nDCG@10']:>10.4f} {maskfn_metrics['Recall@100']:>10.4f}")

    delta_ndcg = maskfn_metrics['nDCG@10'] - baseline_metrics['nDCG@10']
    delta_r100 = maskfn_metrics['Recall@100'] - baseline_metrics['Recall@100']
    print(f"  {'Δ':>15} {delta_ndcg:>+10.4f} {delta_r100:>+10.4f}")

    # Save
    results = {
        "dataset": ds, "config": cfg,
        "baseline": baseline_metrics, "mask_fn": maskfn_metrics,
        "delta": {"nDCG@10": delta_ndcg, "Recall@100": delta_r100},
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out / 'results.json'}")


if __name__ == "__main__":
    main()
