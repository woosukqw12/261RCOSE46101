"""
Train E5 bi-encoder with precomputed adjustments (mask_fn).
Called as subprocess by run_multi_criteria_curriculum.py.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.run_dataset_experiment import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--adj_path", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--num_neg", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--save_every_epoch", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.adj_path) as f:
        adj_raw = json.load(f)
    adjustments = {k: np.array(v, dtype=np.float32) for k, v in adj_raw.items()}
    print(f"Loaded adjustments: {len(adjustments)} queries")

    ckpt = train_model(
        args.train_path, args.log_dir, args.ckpt_dir,
        args.num_neg, args.epochs, args.batch_size, args.lr,
        adjustments=adjustments,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_workers=args.dataloader_workers,
        save_every_epoch=args.save_every_epoch,
    )
    print(f"Done: {ckpt}")


if __name__ == "__main__":
    main()
