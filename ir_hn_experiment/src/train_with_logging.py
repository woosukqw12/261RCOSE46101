import os
import json
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from config import Config

logger = logging.getLogger(__name__)


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


class SimpleRetrievalDataset(Dataset):
    """rlhn/default-680K를 HuggingFace에서 직접 로드.
    __getitem__은 raw text만 반환하고, collate에서 batch tokenize한다.
    """

    def __init__(self, data_dir, datasets, tokenizer, max_len=350, num_neg=7):
        self.num_neg = num_neg
        self.instances = []

        try:
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

            logger.info(f"Loaded {len(self.instances)} instances from HuggingFace")

        except Exception as e:
            logger.warning(f"HuggingFace load failed ({e}), falling back to local files...")
            for ds in datasets:
                path = os.path.join(data_dir, ds)
                if not os.path.isdir(path):
                    continue
                for fname in os.listdir(path):
                    if not fname.endswith((".jsonl", ".json")):
                        continue
                    with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                        for idx, line in enumerate(f):
                            item = json.loads(line)
                            q = item.get("query", item.get("question", ""))
                            pos = item.get("pos", item.get("positive", []))
                            neg = item.get("neg", item.get("negative", []))
                            if isinstance(pos, str):
                                pos = [pos]
                            if isinstance(neg, str):
                                neg = [neg]
                            if q and pos and neg:
                                self.instances.append({
                                    "qid": f"{ds}_{idx}",
                                    "query": q,
                                    "pos": pos[0],
                                    "negs": neg[:num_neg],
                                })
            logger.info(f"Loaded {len(self.instances)} instances from local files")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        negs = list(inst["negs"])
        while len(negs) < self.num_neg:
            negs.append(negs[-1])

        return {
            "qid": inst["qid"],
            "query_text": "query: " + inst["query"],
            "pos_text": "passage: " + inst["pos"],
            "neg_texts": ["passage: " + n for n in negs],
        }


def make_collate_fn(tokenizer, max_length):
    """Fast tokenizer의 __call__로 batch tokenize하는 collate function"""

    def collate_fn(batch):
        qids = [b["qid"] for b in batch]
        num_neg = len(batch[0]["neg_texts"])

        query_texts = [b["query_text"] for b in batch]
        pos_texts = [b["pos_text"] for b in batch]
        neg_texts = [b["neg_texts"][ni] for b in batch for ni in range(num_neg)]

        queries = tokenizer(
            query_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        positives = tokenizer(
            pos_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        negatives = tokenizer(
            neg_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return {
            "qids": qids,
            "query": queries,
            "positive": positives,
            "negatives": negatives,
            "num_neg": num_neg,
        }

    return collate_fn


def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info(f"Device: {device}")
    logger.info(
        f"Batch size: {cfg.batch_size}, BF16: {cfg.bf16}, Grad ckpt: {cfg.gradient_checkpointing}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = E5BiEncoder(cfg.model_name).to(device)

    if cfg.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "init_weights.pt"))

    dataset = SimpleRetrievalDataset(
        cfg.data_dir,
        cfg.datasets,
        tokenizer,
        cfg.max_seq_length,
        cfg.num_hard_negatives,
    )

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "collate_fn": make_collate_fn(tokenizer, cfg.max_seq_length),
        "num_workers": 4,
        "pin_memory": True,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(dataset, **loader_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * cfg.warmup_ratio),
        total_steps,
    )

    os.makedirs(cfg.log_dir, exist_ok=True)
    global_step = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        log_path = os.path.join(cfg.log_dir, f"epoch_{epoch}.jsonl")

        with open(log_path, "w", encoding="utf-8") as log_file:
            for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}"):
                B = len(batch["qids"])
                num_neg = batch["num_neg"]

                optimizer.zero_grad(set_to_none=True)

                query_enc = {k: v.to(device, non_blocking=True) for k, v in batch["query"].items()}
                pos_enc = {k: v.to(device, non_blocking=True) for k, v in batch["positive"].items()}
                neg_enc = {k: v.to(device, non_blocking=True) for k, v in batch["negatives"].items()}

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                    q_emb = model.encode(**query_enc)                   # (B, d)
                    p_emb = model.encode(**pos_enc)                     # (B, d)
                    n_emb = model.encode(**neg_enc)                     # (B*num_neg, d)
                    n_emb = n_emb.view(B, num_neg, -1)                  # (B, num_neg, d)

                    # local in-batch negatives
                    in_batch_scores = q_emb @ p_emb.T                   # (B, B)
                    hard_neg_scores = torch.bmm(
                        q_emb.unsqueeze(1),
                        n_emb.transpose(1, 2)
                    ).squeeze(1)                                        # (B, num_neg)

                    all_scores = torch.cat([in_batch_scores, hard_neg_scores], dim=1) / cfg.temperature
                    labels = torch.arange(B, device=device)
                    loss = F.cross_entropy(all_scores, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    pos_scores_log = in_batch_scores.diagonal().float().cpu().tolist()
                    hn_scores_log = hard_neg_scores.float().cpu().tolist()

                    for i, qid in enumerate(batch["qids"]):
                        log_file.write(json.dumps({
                            "step": global_step,
                            "query_id": qid,
                            "pos_score": round(pos_scores_log[i], 6),
                            "neg_scores": [round(s, 6) for s in hn_scores_log[i]],
                        }) + "\n")

                epoch_loss += loss.item()
                global_step += 1

        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        ckpt = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch}")
        os.makedirs(ckpt, exist_ok=True)
        model.encoder.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    logger.info("Training complete.")


if __name__ == "__main__":
    train(Config())
