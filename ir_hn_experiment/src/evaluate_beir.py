"""
BEIR evaluation for E5-base checkpoints.

python src/evaluate_beir.py \
    --checkpoint checkpoints/epoch_3 \
    --dataset_dir data/processed/scifact \
    --split test

python src/evaluate_beir.py \
    --checkpoint checkpoints_mask_fn/epoch_3 \
    --dataset_dir data/processed/fiqa \
    --split test
"""

import argparse
import json
import math
import os
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ── model ──────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def encode(texts, tokenizer, model, device, batch_size=256, max_length=350, prefix="passage: "):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch = [prefix + t for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, max_length=max_length, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.float().cpu().numpy())
    return np.vstack(all_embs)


# ── data loading ────────────────────────────────────────────────────────────

def load_collection(path: Path):
    pids, texts = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = str(row["pid"])
            text = ((row.get("title", "") or "") + " " + (row.get("text", "") or "")).strip()
            pids.append(pid)
            texts.append(text)
    return pids, texts


def load_queries(path: Path):
    qids, texts = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qids.append(str(row["qid"]))
            texts.append(row["query"])
    return qids, texts


def load_qrels(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)  # {qid: {pid: rel}}


# ── metrics ──────────────────────────────────────────────────────────────────

def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def evaluate(qrels, results, k=10):
    ndcg, mrr, recall = [], [], []
    for qid, positives in qrels.items():
        pos = {pid: rel for pid, rel in positives.items() if rel > 0}
        if not pos:
            continue
        ranked = sorted(results.get(qid, {}).items(), key=lambda x: -x[1])
        ranked_ids = [pid for pid, _ in ranked]

        topk = ranked_ids[:k]
        rels = [pos.get(pid, 0) for pid in topk]
        ideal = sorted(pos.values(), reverse=True)[:k]
        ndcg.append(dcg(rels) / max(dcg(ideal), 1e-12))

        rr = 0.0
        for i, pid in enumerate(topk):
            if pid in pos:
                rr = 1 / (i + 1)
                break
        mrr.append(rr)

        top100 = set(ranked_ids[:100])
        recall.append(len(top100 & set(pos)) / len(pos))

    return {
        f"nDCG@{k}": round(float(np.mean(ndcg)), 4),
        f"MRR@{k}": round(float(np.mean(mrr)), 4),
        "Recall@100": round(float(np.mean(recall)), 4),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.dataset_dir)
    dataset_name = data_dir.name

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {dataset_name} ({args.split})")
    print(f"Device     : {device}")

    tokenizer, model = load_model(args.checkpoint, device)

    pids, doc_texts = load_collection(data_dir / "collection.jsonl")
    qids, q_texts = load_queries(data_dir / f"queries_{args.split}.jsonl")
    qrels = load_qrels(data_dir / f"qrels_{args.split}.json")

    print(f"Docs: {len(pids):,}  Queries: {len(qids):,}")

    doc_embs = encode(doc_texts, tokenizer, model, device, args.batch_size, prefix="passage: ")
    q_embs   = encode(q_texts,   tokenizer, model, device, args.batch_size, prefix="query: ")

    # FAISS exact search
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs)
    scores, indices = index.search(q_embs, args.topk)

    results = {}
    for qi, qid in enumerate(qids):
        results[qid] = {pids[idx]: float(scores[qi][r])
                        for r, idx in enumerate(indices[qi])}

    metrics = evaluate(qrels, results)
    print(f"\n{'='*40}")
    print(f"  nDCG@10   : {metrics['nDCG@10']:.4f}")
    print(f"  MRR@10    : {metrics['MRR@10']:.4f}")
    print(f"  Recall@100: {metrics['Recall@100']:.4f}")
    print(f"{'='*40}\n")

    ckpt_slug = args.checkpoint.replace("/", "_")
    out_path = Path("results") / f"metrics_{ckpt_slug}_{dataset_name}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"checkpoint": args.checkpoint, "dataset": dataset_name,
                   "split": args.split, **metrics}, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
