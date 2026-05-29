import os
import json
import math
import logging
import argparse
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


CRITERIA_ORDER = [
    "margin_final_positive",
    "margin_avg_positive",
    "margin_persistent_3plus",
    "margin_persistent_all",
    "margin_nondecreasing",
    "rank_final_top1",
    "rank_final_top2",
    "rank_persistent_top1",
    "rank_persistent_all_top1",
    "rank_avg_top2",
    "rank_low_var_top",
    "cartography_hard",
    "cartography_ambiguous",
    "zero_flips_exceeding",
    "low_velocity_exceeding",
    "low_displacement",
    "margin_and_rank",
    "hard_and_persistent",
    "all_strict",
    "easy_cartography",
    "easy_low_margin",
    "easy_always_bottom",
]


def list_epoch_files(log_dir: str) -> List[str]:
    files = []
    for fname in sorted(os.listdir(log_dir)):
        if fname.startswith("epoch_") and fname.endswith(".jsonl"):
            files.append(os.path.join(log_dir, fname))
    return files


def inspect_epoch_file(path: str) -> Tuple[int, int]:
    n = 0
    num_neg = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if num_neg is None:
                rec = json.loads(line)
                num_neg = len(rec.get("neg_scores", []))
    if num_neg is None:
        raise ValueError(f"No valid records found in {path}")
    return n, num_neg


def ranks_from_scores(negs: np.ndarray) -> np.ndarray:
    order = np.argsort(-negs)
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[order] = np.arange(1, len(negs) + 1, dtype=np.int16)
    return ranks


def process_source(
    source_name: str,
    log_dir: str,
    output_dir: str,
    skip_if_exists: bool = True,
) -> None:
    criteria_path = os.path.join(output_dir, f"criteria_{source_name}.json")
    counts_path = os.path.join(output_dir, f"criteria_{source_name}_counts.json")

    if skip_if_exists and os.path.exists(criteria_path):
        logger.info(f"Skipping {source_name}: {criteria_path} already exists")
        return

    epoch_files = list_epoch_files(log_dir)
    if len(epoch_files) < 2:
        logger.warning(f"Need at least 2 epoch files in {log_dir}; found {len(epoch_files)}")
        return

    n_queries, num_neg = inspect_epoch_file(epoch_files[0])
    n_epochs_total = len(epoch_files)
    logger.info(
        f"[{source_name}] queries={n_queries:,}, negs/query={num_neg}, epochs={n_epochs_total}"
    )

    # Compact global arrays
    first_pos = np.empty(n_queries, dtype=np.float32)
    last_pos = np.empty(n_queries, dtype=np.float32)
    seen = np.zeros(n_queries, dtype=np.uint8)

    first_neg = np.empty((n_queries, num_neg), dtype=np.float32)
    last_neg = np.empty((n_queries, num_neg), dtype=np.float32)

    persistent_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    flip_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    prev_exceed = np.zeros((n_queries, num_neg), dtype=bool)

    mean_margin = np.zeros((n_queries, num_neg), dtype=np.float32)
    m2_margin = np.zeros((n_queries, num_neg), dtype=np.float32)
    mean_score = np.zeros((n_queries, num_neg), dtype=np.float32)
    m2_score = np.zeros((n_queries, num_neg), dtype=np.float32)

    sum_rank = np.zeros((n_queries, num_neg), dtype=np.float32)
    sumsq_rank = np.zeros((n_queries, num_neg), dtype=np.float32)
    rank_top1_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    rank_top2_count = np.zeros((n_queries, num_neg), dtype=np.uint8)
    last_rank = np.zeros((n_queries, num_neg), dtype=np.int16)

    qids: List[str] = [""] * n_queries
    qid_to_idx: Dict[str, int] = {}

    # Stream all epoch files. Assumes each qid appears once per epoch (true for current pipeline).
    for epoch_i, path in enumerate(epoch_files):
        logger.info(f"[{source_name}] processing {os.path.basename(path)} ({epoch_i+1}/{n_epochs_total})")
        with open(path, "r", encoding="utf-8") as f:
            if epoch_i == 0:
                idx = 0
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    qid = rec["query_id"]
                    pos = float(rec["pos_score"])
                    negs = np.asarray(rec["neg_scores"], dtype=np.float32)
                    if len(negs) != num_neg:
                        raise ValueError(f"Inconsistent neg count for {qid}: expected {num_neg}, got {len(negs)}")

                    qids[idx] = qid
                    qid_to_idx[qid] = idx

                    margins = negs - pos
                    exceeds = margins > 0
                    ranks = ranks_from_scores(negs)

                    first_pos[idx] = pos
                    last_pos[idx] = pos
                    first_neg[idx] = negs
                    last_neg[idx] = negs
                    persistent_count[idx] = exceeds.astype(np.uint8)
                    prev_exceed[idx] = exceeds
                    mean_margin[idx] = margins
                    mean_score[idx] = negs
                    sum_rank[idx] = ranks
                    sumsq_rank[idx] = ranks.astype(np.float32) ** 2
                    rank_top1_count[idx] = (ranks == 1).astype(np.uint8)
                    rank_top2_count[idx] = (ranks <= 2).astype(np.uint8)
                    last_rank[idx] = ranks
                    seen[idx] = 1
                    idx += 1

                if idx != n_queries:
                    raise ValueError(f"Count mismatch in first epoch: expected {n_queries}, loaded {idx}")
            else:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    qid = rec["query_id"]
                    idx = qid_to_idx.get(qid)
                    if idx is None:
                        continue

                    pos = float(rec["pos_score"])
                    negs = np.asarray(rec["neg_scores"], dtype=np.float32)
                    if len(negs) != num_neg:
                        continue

                    margins = negs - pos
                    exceeds = margins > 0
                    ranks = ranks_from_scores(negs)

                    c = int(seen[idx])
                    new_c = c + 1

                    # Welford update for margin mean/variance
                    delta_m = margins - mean_margin[idx]
                    mean_margin[idx] += delta_m / new_c
                    m2_margin[idx] += delta_m * (margins - mean_margin[idx])

                    # Welford update for score mean/variance
                    delta_s = negs - mean_score[idx]
                    mean_score[idx] += delta_s / new_c
                    m2_score[idx] += delta_s * (negs - mean_score[idx])

                    persistent_count[idx] += exceeds.astype(np.uint8)
                    flip_count[idx] += (exceeds != prev_exceed[idx]).astype(np.uint8)
                    prev_exceed[idx] = exceeds

                    sum_rank[idx] += ranks
                    sumsq_rank[idx] += ranks.astype(np.float32) ** 2
                    rank_top1_count[idx] += (ranks == 1).astype(np.uint8)
                    rank_top2_count[idx] += (ranks <= 2).astype(np.uint8)
                    last_rank[idx] = ranks

                    last_pos[idx] = pos
                    last_neg[idx] = negs
                    seen[idx] = new_c

    seen_f = seen.astype(np.float32)[:, None]
    valid_query_mask = seen >= 2
    if not np.all(valid_query_mask):
        missing = int((~valid_query_mask).sum())
        logger.warning(f"[{source_name}] {missing} queries have <2 epochs; they will still be written if masks match")

    first_margin = first_neg - first_pos[:, None]
    final_margin = last_neg - last_pos[:, None]
    avg_margin = mean_margin
    margin_trend = final_margin - first_margin
    margin_variance = np.divide(
        m2_margin,
        np.maximum(seen_f, 1.0),
        out=np.zeros_like(m2_margin),
        where=seen_f > 0,
    )

    final_rank = last_rank
    avg_rank = np.divide(
        sum_rank,
        np.maximum(seen_f, 1.0),
        out=np.zeros_like(sum_rank),
        where=seen_f > 0,
    )
    rank_variance = np.divide(
        sumsq_rank,
        np.maximum(seen_f, 1.0),
        out=np.zeros_like(sumsq_rank),
        where=seen_f > 0,
    ) - avg_rank ** 2
    score_variance = np.divide(
        m2_score,
        np.maximum(seen_f, 1.0),
        out=np.zeros_like(m2_score),
        where=seen_f > 0,
    )

    # Cartography categories
    cart_easy = persistent_count == 0
    cart_hard = (persistent_count == seen[:, None]) | (persistent_count >= (seen[:, None] - 1))
    cart_amb = (flip_count >= 2) & (~cart_easy) & (~cart_hard)

    # Embedding-based criteria: skipped in 32GB-safe version unless later extended.
    low_velocity_exceeding = np.zeros((n_queries, num_neg), dtype=bool)
    low_displacement = np.zeros((n_queries, num_neg), dtype=bool)

    criteria_masks = {
        "margin_final_positive": final_margin > 0,
        "margin_avg_positive": avg_margin > 0,
        "margin_persistent_3plus": persistent_count >= 3,
        "margin_persistent_all": persistent_count == seen[:, None],
        "margin_nondecreasing": (margin_trend >= 0) & (final_margin > 0),
        "rank_final_top1": final_rank == 1,
        "rank_final_top2": final_rank <= 2,
        "rank_persistent_top1": rank_top1_count >= 3,
        "rank_persistent_all_top1": rank_top1_count == seen[:, None],
        "rank_avg_top2": avg_rank <= 2.0,
        "rank_low_var_top": (avg_rank <= 2.0) & (rank_variance < 0.5),
        "cartography_hard": cart_hard,
        "cartography_ambiguous": cart_amb,
        "zero_flips_exceeding": (flip_count == 0) & (final_margin > 0),
        "low_velocity_exceeding": low_velocity_exceeding,
        "low_displacement": low_displacement,
        "margin_and_rank": (final_margin > 0) & (final_rank == 1),
        "hard_and_persistent": cart_hard & (rank_top1_count >= 3),
        "all_strict": (final_margin > 0) & (rank_top1_count >= 3) & cart_hard,
        "easy_cartography": cart_easy,
        "easy_low_margin": avg_margin < -0.3,
        "easy_always_bottom": avg_rank >= 6.0,
    }

    counts = {name: int(criteria_masks[name].sum()) for name in CRITERIA_ORDER}
    total_pairs = int(n_queries * num_neg)

    logger.info(f"\n--- {source_name.upper()} criteria counts ---")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {name:35s}: {count:>8d} ({100*count/total_pairs:.1f}%)")

    # Stream-write JSON so we never materialize all pair lists at once.
    with open(criteria_path, "w", encoding="utf-8") as f:
        f.write("{")
        first_crit = True
        for name in CRITERIA_ORDER:
            mask = criteria_masks[name]
            rows, cols = np.where(mask)
            if not first_crit:
                f.write(",")
            first_crit = False
            f.write(json.dumps(name, ensure_ascii=False))
            f.write(":")
            f.write("[")
            first_pair = True
            for qi, ni in zip(rows.tolist(), cols.tolist()):
                if not first_pair:
                    f.write(",")
                first_pair = False
                f.write(json.dumps([qids[qi], int(ni)], ensure_ascii=False))
            f.write("]")
            # free temporary arrays early
            del rows, cols
        f.write("}")

    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump({"total_pairs": total_pairs, "counts": counts}, f, indent=2)

    logger.info(f"Saved {source_name} criteria to {criteria_path}")
    logger.info(f"Saved {source_name} counts to {counts_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs/training_scores")
    parser.add_argument("--reencode_dir", default="logs/reencode_scores")
    parser.add_argument("--output_dir", default="results/signals")
    parser.add_argument("--source", choices=["loss", "reencode", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Recompute even if criteria file already exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.source in ("loss", "all"):
        process_source("loss", args.log_dir, args.output_dir, skip_if_exists=not args.force)

    if args.source in ("reencode", "all"):
        process_source("reencode", args.reencode_dir, args.output_dir, skip_if_exists=not args.force)


if __name__ == "__main__":
    main()
