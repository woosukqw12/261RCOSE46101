import os
import json
import sqlite3
import logging
import argparse
import tempfile
from typing import Dict, Set, Tuple, Iterable, Optional

logger = logging.getLogger(__name__)


def _iter_hf_dataset(name: str, split: str = "train", cache_dir: Optional[str] = None, streaming: bool = True):
    from datasets import load_dataset

    kwargs = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if streaming:
        kwargs["streaming"] = True
    return load_dataset(name, **kwargs)


def _open_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")  # ~200MB page cache upper bound
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS default_index (
            qid TEXT PRIMARY KEY,
            orig_pos_json TEXT NOT NULL,
            orig_neg_json TEXT NOT NULL
        )
        """
    )
    return conn


def build_default_index_sqlite(
    db_path: str,
    rlhn_cache_dir: Optional[str] = None,
    streaming: bool = True,
    commit_every: int = 5000,
) -> int:
    """Store only minimal fields from rlhn/default-680K in SQLite.

    This avoids materializing the full Hugging Face dataset items and two large
    Python dicts in RAM.
    """
    logger.info("Loading rlhn/default-680K (original)...")
    default_ds = _iter_hf_dataset("rlhn/default-680K", split="train", cache_dir=rlhn_cache_dir, streaming=streaming)

    conn = _open_sqlite(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM default_index")

    insert_sql = (
        "INSERT OR REPLACE INTO default_index (qid, orig_pos_json, orig_neg_json) "
        "VALUES (?, ?, ?)"
    )

    count = 0
    batch = []
    for item in default_ds:
        qid = item["query_id"]
        orig_pos_ids = sorted({str(p["docid"]) for p in item.get("positive_passages", [])})
        orig_neg_docids = [str(p["docid"]) for p in item.get("negative_passages", [])]
        batch.append((qid, json.dumps(orig_pos_ids), json.dumps(orig_neg_docids)))
        count += 1

        if len(batch) >= commit_every:
            cur.executemany(insert_sql, batch)
            conn.commit()
            batch.clear()
            if count % 50000 == 0:
                logger.info(f"Indexed {count:,} original queries...")

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()

    conn.close()
    logger.info(f"Built SQLite index for {count:,} original queries: {db_path}")
    return count


def load_rlhn_labels(
    rlhn_dir: str = None,
    sqlite_index_path: Optional[str] = None,
    streaming: bool = True,
) -> Set[Tuple[str, int]]:
    """Load RLHN false negative labels without blowing up RAM.

    Strategy:
    1) Stream rlhn/default-680K and store minimal per-query info in SQLite.
    2) Stream rlhn/rlhn-680K and compare against the SQLite index.

    Returns:
        set of (query_id, neg_idx)
    """
    if sqlite_index_path is None:
        tmp_dir = rlhn_dir if rlhn_dir else tempfile.gettempdir()
        sqlite_index_path = os.path.join(tmp_dir, "rlhn_default_index.sqlite")

    # Rebuild each run to avoid stale partial indices.
    total_default = build_default_index_sqlite(
        sqlite_index_path,
        rlhn_cache_dir=rlhn_dir,
        streaming=streaming,
    )

    logger.info("Loading rlhn/rlhn-680K (relabeled)...")
    rlhn_ds = _iter_hf_dataset("rlhn/rlhn-680K", split="train", cache_dir=rlhn_dir, streaming=streaming)

    conn = _open_sqlite(sqlite_index_path)
    cur = conn.cursor()

    fn_set: Set[Tuple[str, int]] = set()
    matched = 0
    processed = 0

    for item in rlhn_ds:
        processed += 1
        qid = item["query_id"]
        row = cur.execute(
            "SELECT orig_pos_json, orig_neg_json FROM default_index WHERE qid = ?",
            (qid,),
        ).fetchone()
        if row is None:
            continue

        matched += 1
        orig_pos_ids = set(json.loads(row[0]))
        orig_neg_docids = json.loads(row[1])
        orig_neg_set = set(orig_neg_docids)
        rlhn_pos_ids = {str(p["docid"]) for p in item.get("positive_passages", [])}

        false_neg_docids = (rlhn_pos_ids - orig_pos_ids) & orig_neg_set
        if false_neg_docids:
            for neg_idx, docid in enumerate(orig_neg_docids):
                if docid in false_neg_docids:
                    fn_set.add((qid, neg_idx))

        if processed % 50000 == 0:
            logger.info(
                f"Compared {processed:,} relabeled queries... "
                f"matched={matched:,}, fn_pairs={len(fn_set):,}"
            )

    conn.close()

    logger.info(f"Matched {matched:,}/{total_default:,} queries between default and rlhn")
    logger.info(f"Found {len(fn_set):,} false negative (query_id, neg_idx) pairs")

    fn_queries = len({qid for qid, _ in fn_set})
    logger.info(f"  Queries with at least 1 FN: {fn_queries:,}")
    if fn_queries:
        logger.info(f"  Avg FN per affected query: {len(fn_set) / fn_queries:.1f}")

    return fn_set


def load_criteria_results(criteria_path: str) -> Dict[str, Set[Tuple[str, int]]]:
    """Load compute_signals.py criteria results.

    Assumes the file is a JSON dict where each value is a list of [query_id, neg_idx]
    pairs. These files are usually much smaller than the RLHN datasets themselves, so
    keeping them in memory is acceptable.
    """
    with open(criteria_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: {tuple(k) for k in keys} for name, keys in raw.items()}


def precision_recall_f1(pred: set, gt: set):
    tp = len(pred & gt)
    if not pred:
        return 0, 0, 0, 0
    prec = tp / len(pred)
    rec = tp / len(gt) if gt else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    jacc = tp / len(pred | gt) if (pred | gt) else 0
    return round(prec, 4), round(rec, 4), round(f1, 4), round(jacc, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals_dir", default="results/signals")
    parser.add_argument("--rlhn_dir", default=None, help="Optional Hugging Face cache_dir / temp dir")
    parser.add_argument("--output_dir", default="results/h1_comparison")
    parser.add_argument(
        "--sqlite_index_path",
        default=None,
        help="Optional path for the temporary SQLite index. Defaults to <rlhn_dir>/rlhn_default_index.sqlite or /tmp.",
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable Hugging Face streaming. Not recommended on 32GB RAM.",
    )
    parser.add_argument(
        "--keep_sqlite_index",
        action="store_true",
        help="Keep the SQLite index file after the script finishes.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.sqlite_index_path is None:
        tmp_dir = args.rlhn_dir if args.rlhn_dir else tempfile.gettempdir()
        args.sqlite_index_path = os.path.join(tmp_dir, "rlhn_default_index.sqlite")

    rlhn_gt = load_rlhn_labels(
        rlhn_dir=args.rlhn_dir,
        sqlite_index_path=args.sqlite_index_path,
        streaming=not args.no_streaming,
    )
    if not rlhn_gt:
        logger.error("No RLHN labels loaded. Check your datasets installation/cache.")
        return

    all_comparisons = []

    for source in ["loss", "reencode"]:
        crit_path = os.path.join(args.signals_dir, f"criteria_{source}.json")
        if not os.path.exists(crit_path):
            logger.warning(f"Not found: {crit_path}")
            continue

        criteria = load_criteria_results(crit_path)

        for crit_name, pred_set in criteria.items():
            prec, rec, f1, jacc = precision_recall_f1(pred_set, rlhn_gt)
            all_comparisons.append({
                "source": source,
                "criterion": crit_name,
                "predicted": len(pred_set),
                "rlhn_gt": len(rlhn_gt),
                "overlap": len(pred_set & rlhn_gt),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "jaccard": jacc,
            })

    all_comparisons.sort(key=lambda x: -x["f1"])

    print("\n" + "=" * 100)
    print("H1 VERIFICATION: All Training Dynamics Signals vs RLHN (LLM Judge)")
    print("=" * 100)
    print(f"\nRLHN ground truth: {len(rlhn_gt)} false negatives\n")

    header = f"{'Source':<10} {'Criterion':<35} {'Pred':>7} {'Overlap':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Jacc':>7}"
    print(header)
    print("-" * len(header))

    for r in all_comparisons:
        if "easy" in r["criterion"]:
            continue
        print(
            f"{r['source']:<10} {r['criterion']:<35} {r['predicted']:>7} {r['overlap']:>7} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {r['jaccard']:>7.4f}"
        )

    top5 = [r for r in all_comparisons if "easy" not in r["criterion"]][:5]
    print(f"\n{'='*100}")
    print("TOP 5 SIGNALS (by F1)")
    print(f"{'='*100}")
    for i, r in enumerate(top5):
        print(
            f"  #{i+1}: [{r['source']}] {r['criterion']} — "
            f"F1={r['f1']:.4f} (P={r['precision']:.4f}, R={r['recall']:.4f})"
        )

    print(f"\n{'='*100}")
    print("LOSS vs RE-ENCODING (same criterion)")
    print(f"{'='*100}")

    loss_results = {r["criterion"]: r for r in all_comparisons if r["source"] == "loss"}
    reenc_results = {r["criterion"]: r for r in all_comparisons if r["source"] == "reencode"}

    shared_criteria = set(loss_results.keys()) & set(reenc_results.keys())
    for crit in sorted(shared_criteria):
        if "easy" in crit or "velocity" in crit or "displacement" in crit:
            continue
        l, e = loss_results[crit], reenc_results[crit]
        winner = "LOSS" if l["f1"] > e["f1"] else "REENC" if e["f1"] > l["f1"] else "TIE"
        print(f"  {crit:<35} Loss F1={l['f1']:.4f}  Reenc F1={e['f1']:.4f}  → {winner}")

    print(f"\n{'='*100}")
    print("SIGNAL FAMILY COMPARISON (best of each family, re-encoding)")
    print(f"{'='*100}")

    families = {
        "Margin": [r for r in all_comparisons if r["source"] == "reencode" and "margin" in r["criterion"] and "easy" not in r["criterion"]],
        "Rank": [r for r in all_comparisons if r["source"] == "reencode" and "rank" in r["criterion"]],
        "Forgetting": [r for r in all_comparisons if r["source"] == "reencode" and ("cartography" in r["criterion"] or "flip" in r["criterion"])],
        "Velocity": [r for r in all_comparisons if r["source"] == "reencode" and ("velocity" in r["criterion"] or "displacement" in r["criterion"])],
        "Composite": [r for r in all_comparisons if r["source"] == "reencode" and ("_and_" in r["criterion"] or "all_strict" in r["criterion"])],
    }

    for family, results in families.items():
        if not results:
            print(f"  {family:<15} — no results")
            continue
        best = max(results, key=lambda x: x["f1"])
        print(
            f"  {family:<15} best: {best['criterion']:<30} "
            f"F1={best['f1']:.4f} (P={best['precision']:.4f}, R={best['recall']:.4f})"
        )

    print(f"\n{'='*100}")
    print("EASY NEGATIVE DETECTION")
    print(f"{'='*100}")
    for r in all_comparisons:
        if "easy" in r["criterion"]:
            print(f"  [{r['source']}] {r['criterion']}: {r['predicted']} detected")

    print(f"\n{'='*100}")
    print("CONCLUSION")
    print(f"{'='*100}")

    if top5:
        best = top5[0]
        if best["f1"] > 0.3:
            print("  ✅ H1 STRONGLY SUPPORTED")
            print(f"     Best signal: [{best['source']}] {best['criterion']}")
            print(f"     F1={best['f1']:.4f} — training dynamics are a STRONG proxy for FN detection")
        elif best["f1"] > 0.15:
            print("  ⚠️  H1 PARTIALLY SUPPORTED")
            print(f"     Best signal: [{best['source']}] {best['criterion']}")
            print(f"     F1={best['f1']:.4f} — some overlap, but limited")
        else:
            print("  ❌ H1 NOT SUPPORTED")
            print(f"     Best F1={best['f1']:.4f} — training dynamics alone insufficient")

    out_json = os.path.join(args.output_dir, "full_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, indent=2)
    logger.info(f"Saved full results to {out_json}")

    if not args.keep_sqlite_index and os.path.exists(args.sqlite_index_path):
        try:
            os.remove(args.sqlite_index_path)
            wal_path = args.sqlite_index_path + "-wal"
            shm_path = args.sqlite_index_path + "-shm"
            if os.path.exists(wal_path):
                os.remove(wal_path)
            if os.path.exists(shm_path):
                os.remove(shm_path)
        except OSError as e:
            logger.warning(f"Failed to remove SQLite temp files: {e}")


if __name__ == "__main__":
    main()
