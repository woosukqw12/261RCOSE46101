"""
Curriculum learning using training dynamics from compute_signals.py.

Three modes
-----------
normal       : baseline, identical to train_with_logging.py
mask_fn      : all_strict pairs are masked out of the negative loss
               (model not penalised for ranking potential FNs near the positive)
weight_hard  : rank_persistent_all_top1 \ all_strict pairs are upweighted
               (confirmed hard & not potential FN → strong gradient signal)
staged       : epoch < stage_epoch → normal; epoch >= stage_epoch → mask_fn
               lets the model stabilise before applying the mask

Usage
-----
python src/train_curriculum.py --mode mask_fn
python src/train_curriculum.py --mode weight_hard --hard_weight 3.0
python src/train_curriculum.py --mode staged --stage_epoch 2
python src/train_curriculum.py --mode normal   # baseline re-run
"""

import os
import json
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

MASK_VAL = -1e9   # effective -inf before /temperature


# ──────────────────────────────────────────────
# Model (identical to train_with_logging.py)
# ──────────────────────────────────────────────

class E5BiEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(emb).to(emb.dtype)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1)


# ──────────────────────────────────────────────
# Hardness weights
# ──────────────────────────────────────────────

def load_neg_adjustments(
    criteria_path: str,
    mode: str,
    num_neg: int = 7,
    hard_weight: float = 3.0,
) -> dict:
    """Return {qid: np.ndarray(num_neg,) of score adjustments}.

    mask_fn mode  : all_strict → MASK_VAL, else 0
    weight_hard   : rank_persistent_all_top1 \ all_strict → log(hard_weight)*T added
                    at loss time; stored as raw weight here, applied in training loop
    staged        : same lookup as mask_fn; caller decides when to apply
    """
    with open(criteria_path, encoding="utf-8") as f:
        criteria = json.load(f)

    adjustments: dict[str, np.ndarray] = {}

    def _set(qid, neg_idx, val):
        if qid not in adjustments:
            adjustments[qid] = np.zeros(num_neg, dtype=np.float32)
        if neg_idx < num_neg:
            adjustments[qid][neg_idx] = val

    if mode in ("mask_fn", "staged"):
        for qid, neg_idx in criteria.get("all_strict", []):
            _set(qid, neg_idx, MASK_VAL)

    elif mode == "weight_hard":
        strict_set = {(q, n) for q, n in criteria.get("all_strict", [])}
        for qid, neg_idx in criteria.get("rank_persistent_all_top1", []):
            if (qid, neg_idx) not in strict_set:
                # store the raw multiplier; convert to log-space in training loop
                _set(qid, neg_idx, hard_weight)
        # all_strict still masked (potential FN — don't upweight)
        for qid, neg_idx in criteria.get("all_strict", []):
            _set(qid, neg_idx, MASK_VAL)

    n_affected = sum(1 for arr in adjustments.values() for v in arr if v != 0)
    logger.info(f"Loaded adjustments for {len(adjustments):,} queries, "
                f"{n_affected:,} (query, neg) pairs affected [{mode}]")
    return adjustments


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class CurriculumDataset(Dataset):
    def __init__(self, tokenizer, max_len=350, num_neg=7, adjustments=None):
        self.num_neg = num_neg
        self.adjustments = adjustments or {}
        self.instances = []

        from datasets import load_dataset
        logger.info("Loading rlhn/default-680K from HuggingFace...")
        hf_ds = load_dataset("rlhn/default-680K", split="train")

        for item in hf_ds:
            pos_list = item.get("positive_passages", [])
            neg_list = item.get("negative_passages", [])
            query = item.get("query", "")
            if not query or not pos_list or not neg_list:
                continue
            self.instances.append({
                "qid": item["query_id"],
                "query": query,
                "pos": pos_list[0]["text"],
                "negs": [n["text"] for n in neg_list[:num_neg]],
            })

        logger.info(f"Loaded {len(self.instances):,} instances")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        qid = inst["qid"]
        negs = list(inst["negs"])
        while len(negs) < self.num_neg:
            negs.append(negs[-1])

        adj = self.adjustments.get(qid, None)
        if adj is None:
            adj = np.zeros(self.num_neg, dtype=np.float32)

        return {
            "qid": qid,
            "query_text": "query: " + inst["query"],
            "pos_text": "passage: " + inst["pos"],
            "neg_texts": ["passage: " + n for n in negs],
            "neg_adj": adj,
        }


def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        qids = [b["qid"] for b in batch]
        num_neg = len(batch[0]["neg_texts"])

        queries = tokenizer(
            [b["query_text"] for b in batch],
            max_length=max_length, truncation=True, padding=True, return_tensors="pt",
        )
        positives = tokenizer(
            [b["pos_text"] for b in batch],
            max_length=max_length, truncation=True, padding=True, return_tensors="pt",
        )
        negatives = tokenizer(
            [t for b in batch for t in b["neg_texts"]],
            max_length=max_length, truncation=True, padding=True, return_tensors="pt",
        )
        neg_adj = torch.tensor(
            np.stack([b["neg_adj"] for b in batch]), dtype=torch.float32
        )  # (B, num_neg)

        return {
            "qids": qids,
            "query": queries,
            "positive": positives,
            "negatives": negatives,
            "num_neg": num_neg,
            "neg_adj": neg_adj,
        }
    return collate_fn


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────

def curriculum_loss(
    in_batch_scores: torch.Tensor,   # (B, B)
    hard_neg_scores: torch.Tensor,   # (B, num_neg)
    neg_adj: torch.Tensor,           # (B, num_neg)  raw adjustments
    temperature: float,
    mode: str,
    apply_adj: bool = True,
) -> torch.Tensor:
    B = in_batch_scores.size(0)
    labels = torch.arange(B, device=in_batch_scores.device)

    if not apply_adj:
        all_scores = torch.cat([in_batch_scores, hard_neg_scores], dim=1) / temperature
        return F.cross_entropy(all_scores, labels)

    if mode == "weight_hard":
        # neg_adj stores raw multiplier (e.g. 3.0) or MASK_VAL
        # log-trick: add log(w)*temperature to score before dividing
        # MASK_VAL entries stay as MASK_VAL (they dominate)
        is_mask = neg_adj < -1e8
        log_adj = torch.where(
            is_mask,
            torch.full_like(neg_adj, MASK_VAL),
            torch.where(neg_adj > 0, torch.log(neg_adj.clamp(min=1e-9)) * temperature,
                        torch.zeros_like(neg_adj)),
        )
        adjusted = hard_neg_scores + log_adj
    else:
        # mask_fn / staged: neg_adj is 0 or MASK_VAL
        adjusted = hard_neg_scores + neg_adj.to(hard_neg_scores.device)

    all_scores = torch.cat([in_batch_scores, adjusted], dim=1) / temperature
    return F.cross_entropy(all_scores, labels)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train(cfg: Config, mode: str, criteria_path: str,
          hard_weight: float, stage_epoch: int, out_suffix: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info(f"Mode: {mode}  device: {device}")

    adjustments = {}
    if mode != "normal":
        adjustments = load_neg_adjustments(
            criteria_path, mode, cfg.num_hard_negatives, hard_weight
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = E5BiEncoder(cfg.model_name).to(device)

    if cfg.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()

    ckpt_dir = cfg.checkpoint_dir + out_suffix
    log_dir = cfg.log_dir + out_suffix
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "init_weights.pt"))

    dataset = CurriculumDataset(tokenizer, cfg.max_seq_length,
                                cfg.num_hard_negatives, adjustments)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, cfg.max_seq_length),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )

    global_step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0

        # staged: apply mask only from stage_epoch onwards
        apply_adj = True
        if mode == "staged":
            apply_adj = (epoch >= stage_epoch)
            logger.info(f"Epoch {epoch+1}: adj={'ON' if apply_adj else 'OFF'}")

        log_path = os.path.join(log_dir, f"epoch_{epoch}.jsonl")
        with open(log_path, "w", encoding="utf-8") as log_file:
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):
                B = len(batch["qids"])
                num_neg = batch["num_neg"]

                optimizer.zero_grad(set_to_none=True)

                q_enc = {k: v.to(device, non_blocking=True) for k, v in batch["query"].items()}
                p_enc = {k: v.to(device, non_blocking=True) for k, v in batch["positive"].items()}
                n_enc = {k: v.to(device, non_blocking=True) for k, v in batch["negatives"].items()}
                neg_adj = batch["neg_adj"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                    q_emb = model.encode(**q_enc)
                    p_emb = model.encode(**p_enc)
                    n_emb = model.encode(**n_enc).view(B, num_neg, -1)

                    in_batch_scores = q_emb @ p_emb.T
                    hard_neg_scores = torch.bmm(
                        q_emb.unsqueeze(1), n_emb.transpose(1, 2)
                    ).squeeze(1)

                    loss = curriculum_loss(
                        in_batch_scores, hard_neg_scores, neg_adj,
                        cfg.temperature, mode, apply_adj,
                    )

                loss.backward()
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    pos_scores = in_batch_scores.diagonal().float().cpu().tolist()
                    neg_scores = hard_neg_scores.float().cpu().tolist()
                    for i, qid in enumerate(batch["qids"]):
                        log_file.write(json.dumps({
                            "step": global_step,
                            "query_id": qid,
                            "pos_score": round(pos_scores[i], 6),
                            "neg_scores": [round(s, 6) for s in neg_scores[i]],
                        }) + "\n")

                epoch_loss += loss.item()
                global_step += 1

        logger.info(f"Epoch {epoch+1} avg loss: {epoch_loss/len(loader):.4f}")
        ckpt = os.path.join(ckpt_dir, f"epoch_{epoch}")
        os.makedirs(ckpt, exist_ok=True)
        model.encoder.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    logger.info("Done.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "mask_fn", "weight_hard", "staged"],
                        default="mask_fn")
    parser.add_argument("--criteria", default="results/signals/criteria_reencode.json")
    parser.add_argument("--hard_weight", type=float, default=3.0,
                        help="Upweight multiplier for weight_hard mode")
    parser.add_argument("--stage_epoch", type=int, default=2,
                        help="Epoch from which mask is applied (staged mode)")
    parser.add_argument("--suffix", default=None,
                        help="Suffix for checkpoint/log dirs (default: _<mode>)")
    args = parser.parse_args()

    suffix = args.suffix if args.suffix is not None else f"_{args.mode}"
    train(Config(), args.mode, args.criteria, args.hard_weight, args.stage_epoch, suffix)
