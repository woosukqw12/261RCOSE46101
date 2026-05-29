import os
import json
import torch
import logging
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from train_with_logging import E5BiEncoder, SimpleRetrievalDataset, make_collate_fn
from config import Config

logger = logging.getLogger(__name__)


@torch.no_grad()
def reencode_epoch_to_file(model, loader, device, out_path, save_embeddings=False, use_bf16=True):
    model.eval()

    with open(out_path, "w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="Re-encoding"):
            B = len(batch["qids"])
            num_neg = batch["num_neg"]

            query_enc = {k: v.to(device, non_blocking=True) for k, v in batch["query"].items()}
            pos_enc = {k: v.to(device, non_blocking=True) for k, v in batch["positive"].items()}
            neg_enc = {k: v.to(device, non_blocking=True) for k, v in batch["negatives"].items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                q_emb = model.encode(**query_enc)       # (B, d)
                p_emb = model.encode(**pos_enc)         # (B, d)
                n_emb = model.encode(**neg_enc)         # (B*num_neg, d)
                n_emb = n_emb.view(B, num_neg, -1)      # (B, num_neg, d)

            pos_scores = (q_emb * p_emb).sum(dim=1).float().cpu().tolist()
            neg_scores = torch.bmm(
                q_emb.unsqueeze(1),
                n_emb.transpose(1, 2)
            ).squeeze(1).float().cpu().tolist()

            pos_emb_cpu = None
            neg_emb_cpu = None
            if save_embeddings:
                pos_emb_cpu = p_emb.cpu().to(torch.float16)
                neg_emb_cpu = n_emb.cpu().to(torch.float16)

            for i, qid in enumerate(batch["qids"]):
                rec = {
                    "query_id": qid,
                    "pos_score": round(pos_scores[i], 6),
                    "neg_scores": [round(s, 6) for s in neg_scores[i]],
                }
                if save_embeddings:
                    rec["pos_emb"] = pos_emb_cpu[i].tolist()
                    rec["neg_embs"] = [neg_emb_cpu[i, ni].tolist() for ni in range(num_neg)]

                f.write(json.dumps(rec) + "\n")


def run_reencoding(cfg: Config, save_embeddings: bool = True, reencode_batch_size: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = SimpleRetrievalDataset(
        cfg.data_dir,
        cfg.datasets,
        tokenizer,
        cfg.max_seq_length,
        cfg.num_hard_negatives,
    )

    batch_size = reencode_batch_size if reencode_batch_size is not None else cfg.batch_size
    logger.info(f"Re-encode batch size: {batch_size}, save_embeddings={save_embeddings}, bf16={cfg.bf16}")

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": make_collate_fn(tokenizer, cfg.max_seq_length),
        "num_workers": 4,
        "pin_memory": True,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(dataset, **loader_kwargs)

    os.makedirs(cfg.reencode_dir, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch}")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}, skipping")
            continue

        logger.info(f"Re-encoding with epoch {epoch} checkpoint...")
        model = E5BiEncoder(ckpt_path).to(device)

        out_path = os.path.join(cfg.reencode_dir, f"epoch_{epoch}.jsonl")
        reencode_epoch_to_file(
            model,
            loader,
            device,
            out_path,
            save_embeddings=save_embeddings,
            use_bf16=cfg.bf16,
        )

        logger.info(f"Saved re-encoded records to {out_path}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Re-encoding complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_embeddings", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None, help="Override re-encode batch size")
    args = parser.parse_args()
    run_reencoding(
        Config(),
        save_embeddings=not args.no_embeddings,
        reencode_batch_size=args.batch_size,
    )
