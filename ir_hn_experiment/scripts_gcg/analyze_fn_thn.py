"""
FN vs THN Diagnostic Analysis
==============================

세 가지 분석:

A. Frozen Encoder Separation
   - pseudo-FN label: sim(p,n) > pn_threshold
   - pseudo-THN label: sim(q,n) > qn_threshold AND sim(p,n) < pn_threshold
   - predictor: conflict score
   - metric: AUROC (FN vs THN 구별력), 평균 conflict 차이

B. Hard-Subset Enrichment
   - 조건: sim(q,n) 상위 20% (hard negatives만)
   - 비교: high-conflict vs low-conflict
   - metric: judged positive rate (qrels에 있는 비율)

C. Qualitative Examples
   - high-conflict 상위 20개
   - low-conflict 하위 20개
   - 텍스트 출력 → 정성 평가

Usage:
  python analyze_fn_thn.py \\
    --dataset_dir ./data/processed/fiqa \\
    --model_name BAAI/bge-base-en-v1.5 \\
    --split dev \\
    --pn_threshold 0.5 \\
    --qn_threshold 0.4 \\
    --topk 20 \\
    --output_dir ./outputs/analysis
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Data loading
# =============================================================================

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def load_qrels(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_collection(path):
    rows = load_jsonl(path)
    pid2text = {}
    for row in rows:
        pid = str(row["pid"])
        text = ((row.get("title", "") or "") + " " + (row.get("text", "") or "")).strip()
        pid2text[pid] = text
    return pid2text


# =============================================================================
# Encoder
# =============================================================================

class BGEEncoder:
    def __init__(self, model_name, device, max_len=180):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, texts, batch_size=64, show_progress=False):
        all_embs = []
        it = range(0, len(texts), batch_size)
        if show_progress:
            it = tqdm(it, desc="encode", leave=False)
        for i in it:
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt"
            ).to(self.device)
            out = self.model(**enc)
            # CLS token
            emb = out.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)  # (N, D)


# =============================================================================
# Conflict computation (closed-form tangent-plane)
# =============================================================================

def compute_conflict(q, p, n):
    """
    q: (D,)  p: (D,)  n: (K, D)
    returns conflict: (K,)
    """
    # tangent-plane projections on unit sphere
    g_pos = p - (q @ p) * q        # (D,)
    g_neg = n - (n @ q).unsqueeze(1) * q.unsqueeze(0)  # (K, D)

    g_pos_norm = F.normalize(g_pos.unsqueeze(0), dim=-1)  # (1, D)
    g_neg_norm = F.normalize(g_neg, dim=-1)               # (K, D)

    conflict = (g_pos_norm * g_neg_norm).sum(dim=-1)      # (K,)
    return conflict


# =============================================================================
# AUROC (simple, no sklearn dependency)
# =============================================================================

def auroc(scores, labels):
    """
    scores: list of float
    labels: list of 0/1  (1 = positive class)
    """
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp  # rectangle area
    auc /= (n_pos * n_neg)
    return auc


# =============================================================================
# Main analysis
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",  required=True)
    ap.add_argument("--model_name",   default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--checkpoint",   default=None,
                    help="path to fine-tuned checkpoint dir (optional, uses pretrained if not given)")
    ap.add_argument("--split",        default="dev", choices=["dev", "test", "train"])
    ap.add_argument("--pn_threshold", type=float, default=0.5,
                    help="sim(p,n) > this → pseudo-FN")
    ap.add_argument("--qn_threshold", type=float, default=0.4,
                    help="sim(q,n) > this → hard neg (used in analysis B & C)")
    ap.add_argument("--hard_top_pct", type=float, default=0.20,
                    help="top X fraction of sim(q,n) = 'hard' for analysis B")
    ap.add_argument("--topk",         type=int,   default=20,
                    help="qualitative examples per category (analysis C)")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--output_dir",   default="./outputs/analysis")
    ap.add_argument("--max_queries",  type=int,   default=None,
                    help="limit number of queries (for quick testing)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data_dir = Path(args.dataset_dir)
    print("Loading data...")

    # train.jsonl has (qid, query, pos_ids, positives, neg_ids, negatives)
    train_rows = load_jsonl(data_dir / "train.jsonl")
    qrels_dev  = load_qrels(data_dir / f"qrels_{args.split}.json")
    pid2text   = load_collection(data_dir / "collection.jsonl")

    # Build qid→query from train rows
    qid2query = {str(r["qid"]): r["query"] for r in train_rows}
    # Build qid→pos_texts, neg_ids, neg_texts from train rows
    # (we use training negatives — BM25 hard negatives)
    qid2row = {str(r["qid"]): r for r in train_rows}

    if args.max_queries:
        all_qids = list(qid2row.keys())[:args.max_queries]
    else:
        all_qids = list(qid2row.keys())

    print(f"  queries: {len(all_qids)}, split qrels: {len(qrels_dev)}")

    # ------------------------------------------------------------------
    # Load encoder
    # ------------------------------------------------------------------
    model_name = args.checkpoint if args.checkpoint else args.model_name
    print(f"Loading encoder: {model_name}")
    encoder = BGEEncoder(model_name, device)

    # ------------------------------------------------------------------
    # Collect all (q, p, n) tuples and compute scores
    # ------------------------------------------------------------------
    print("Computing embeddings and scores...")

    records = []  # each record: dict with scores + text

    for qid in tqdm(all_qids, desc="queries"):
        row = qid2row[qid]
        query_text = row["query"]
        pos_texts  = row["positives"][:1]   # use first positive
        neg_texts  = row["negatives"]
        neg_ids    = [str(x) for x in row["neg_ids"]]

        if not pos_texts or not neg_texts:
            continue

        q_emb = encoder.encode([query_text])[0]   # (D,)
        p_emb = encoder.encode(pos_texts)[0]       # (D,)
        n_emb = encoder.encode(neg_texts)          # (K, D)

        sim_qn = (n_emb @ q_emb)                  # (K,)
        sim_pn = (n_emb @ p_emb)                  # (K,)
        sim_qp = float(q_emb @ p_emb)

        conflict = compute_conflict(q_emb, p_emb, n_emb)  # (K,)

        # qrels lookup for judged positives
        qrels_for_q = qrels_dev.get(str(qid), {})

        for i, (nid, ntxt) in enumerate(zip(neg_ids, neg_texts)):
            is_judged_pos = int(str(nid) in qrels_for_q and qrels_for_q[str(nid)] > 0)

            records.append({
                "qid":           qid,
                "neg_id":        nid,
                "sim_qn":        float(sim_qn[i]),
                "sim_pn":        float(sim_pn[i]),
                "sim_qp":        sim_qp,
                "conflict":      float(conflict[i]),
                "is_judged_pos": is_judged_pos,
                "query_text":    query_text,
                "pos_text":      pos_texts[0],
                "neg_text":      ntxt,
            })

    print(f"  total (q,n) pairs: {len(records)}")

    # Assign pseudo labels
    for r in records:
        r["pseudo_fn"]  = int(r["sim_pn"] >  args.pn_threshold)
        r["pseudo_thn"] = int(r["sim_qn"] >  args.qn_threshold and
                              r["sim_pn"] <= args.pn_threshold)

    # ------------------------------------------------------------------
    # Analysis A: Frozen Encoder Separation
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analysis A: Frozen Encoder Separation")
    print("="*60)

    fn_records  = [r for r in records if r["pseudo_fn"]  == 1]
    thn_records = [r for r in records if r["pseudo_thn"] == 1]
    other       = [r for r in records if r["pseudo_fn"] == 0 and r["pseudo_thn"] == 0]

    print(f"  pseudo-FN  (sim_pn > {args.pn_threshold}):                   {len(fn_records):5d}")
    print(f"  pseudo-THN (sim_qn > {args.qn_threshold} AND sim_pn <= {args.pn_threshold}): {len(thn_records):5d}")
    print(f"  other:                                               {len(other):5d}")

    if fn_records and thn_records:
        fn_conflict_mean  = np.mean([r["conflict"] for r in fn_records])
        thn_conflict_mean = np.mean([r["conflict"] for r in thn_records])
        fn_simqn_mean     = np.mean([r["sim_qn"]   for r in fn_records])
        thn_simqn_mean    = np.mean([r["sim_qn"]   for r in thn_records])
        fn_simpn_mean     = np.mean([r["sim_pn"]   for r in fn_records])
        thn_simpn_mean    = np.mean([r["sim_pn"]   for r in thn_records])

        print(f"\n  {'metric':<25} {'FN':>10} {'THN':>10} {'diff':>10}")
        print(f"  {'-'*55}")
        print(f"  {'conflict mean':<25} {fn_conflict_mean:>10.4f} {thn_conflict_mean:>10.4f} {fn_conflict_mean - thn_conflict_mean:>+10.4f}")
        print(f"  {'sim(q,n) mean':<25} {fn_simqn_mean:>10.4f} {thn_simqn_mean:>10.4f} {fn_simqn_mean - thn_simqn_mean:>+10.4f}")
        print(f"  {'sim(p,n) mean':<25} {fn_simpn_mean:>10.4f} {thn_simpn_mean:>10.4f} {fn_simpn_mean - thn_simpn_mean:>+10.4f}")

        # AUROC: conflict as predictor of pseudo-FN label
        # Among FN + THN only
        subset = fn_records + thn_records
        scores = [r["conflict"] for r in subset]
        labels = [r["pseudo_fn"] for r in subset]
        auc = auroc(scores, labels)
        print(f"\n  AUROC (conflict → pseudo-FN label): {auc:.4f}")
        print(f"  (0.5 = random, 1.0 = perfect separation)")
        if auc > 0.5:
            print(f"  → conflict HIGHER in FN than THN (FN has more conflict)")
        elif auc < 0.5:
            print(f"  → conflict LOWER in FN than THN (THN has more conflict)")
        else:
            print(f"  → no separation")

        # Also check sim(p,n) AUROC as sanity check (should be ~1.0 by construction)
        scores_pn = [r["sim_pn"] for r in subset]
        auc_pn = auroc(scores_pn, labels)
        print(f"  AUROC (sim_pn → pseudo-FN label, sanity): {auc_pn:.4f}  [expected ~1.0]")

    # ------------------------------------------------------------------
    # Analysis B: Hard-Subset Enrichment
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analysis B: Hard-Subset Enrichment")
    print("="*60)

    sim_qn_vals = np.array([r["sim_qn"] for r in records])
    hard_floor  = np.percentile(sim_qn_vals, (1 - args.hard_top_pct) * 100)
    hard_subset = [r for r in records if r["sim_qn"] >= hard_floor]
    print(f"  hard floor (top {args.hard_top_pct*100:.0f}% sim_qn): {hard_floor:.4f}")
    print(f"  hard subset size: {len(hard_subset)}")

    if hard_subset:
        conflict_vals = np.array([r["conflict"] for r in hard_subset])
        median_conflict = np.median(conflict_vals)
        high_conf = [r for r in hard_subset if r["conflict"] >= median_conflict]
        low_conf  = [r for r in hard_subset if r["conflict"] <  median_conflict]

        def judged_pos_rate(recs):
            if not recs:
                return 0.0, 0
            rate = np.mean([r["is_judged_pos"] for r in recs])
            count = sum(r["is_judged_pos"] for r in recs)
            return float(rate), count

        hc_rate, hc_count = judged_pos_rate(high_conf)
        lc_rate, lc_count = judged_pos_rate(low_conf)

        print(f"\n  median conflict threshold: {median_conflict:.4f}")
        print(f"\n  {'group':<25} {'n':>6} {'judged_pos':>12} {'rate':>8}")
        print(f"  {'-'*55}")
        print(f"  {'high conflict':<25} {len(high_conf):>6} {hc_count:>12} {hc_rate:>8.4f}")
        print(f"  {'low  conflict':<25} {len(low_conf):>6}  {lc_count:>12} {lc_rate:>8.4f}")
        print(f"\n  ratio high/low judged_pos rate: {hc_rate/(lc_rate+1e-9):.2f}x")
        if hc_rate > lc_rate:
            print("  → high-conflict hard negs are MORE likely to be judged positives (FN proxy ✓)")
        else:
            print("  → high-conflict hard negs are NOT more likely to be judged positives (FN proxy ✗)")

        # Also compare sim(p,n) between groups
        hc_pn = np.mean([r["sim_pn"] for r in high_conf])
        lc_pn = np.mean([r["sim_pn"] for r in low_conf])
        hc_qn = np.mean([r["sim_qn"] for r in high_conf])
        lc_qn = np.mean([r["sim_qn"] for r in low_conf])
        print(f"\n  {'metric':<25} {'high_conflict':>15} {'low_conflict':>15}")
        print(f"  {'-'*55}")
        print(f"  {'sim(p,n) mean':<25} {hc_pn:>15.4f} {lc_pn:>15.4f}")
        print(f"  {'sim(q,n) mean':<25} {hc_qn:>15.4f} {lc_qn:>15.4f}")

    # ------------------------------------------------------------------
    # Analysis C: Qualitative Examples
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analysis C: Qualitative Examples")
    print("="*60)

    # Sort all records by conflict
    sorted_by_conflict = sorted(records, key=lambda r: r["conflict"], reverse=True)
    high_examples = sorted_by_conflict[:args.topk]
    low_examples  = sorted_by_conflict[-args.topk:]

    def print_examples(examples, label):
        print(f"\n{'─'*60}")
        print(f"  {label} ({len(examples)} examples)")
        print(f"{'─'*60}")
        for i, r in enumerate(examples):
            judged = "★JUDGED_POS" if r["is_judged_pos"] else ""
            fn_tag = "→pseudo-FN"  if r["pseudo_fn"]    else ""
            thn_tag= "→pseudo-THN" if r["pseudo_thn"]   else ""
            tag = " ".join(filter(None, [judged, fn_tag, thn_tag]))
            print(f"\n  [{i+1}] conflict={r['conflict']:+.4f}  sim_qn={r['sim_qn']:.4f}  sim_pn={r['sim_pn']:.4f}  {tag}")
            print(f"  Q: {r['query_text'][:120]}")
            print(f"  P: {r['pos_text'][:120]}")
            print(f"  N: {r['neg_text'][:120]}")

    print_examples(high_examples, f"HIGH conflict (top {args.topk})")
    print_examples(low_examples,  f"LOW  conflict (bottom {args.topk})")

    # ------------------------------------------------------------------
    # Save full records to JSON
    # ------------------------------------------------------------------
    out_path = output_dir / "fn_thn_analysis.json"
    # Save without full texts to keep file small; texts available from collection
    slim_records = []
    for r in records:
        slim_records.append({k: v for k, v in r.items()
                              if k not in ("query_text", "pos_text", "neg_text")})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slim_records, f, indent=2, ensure_ascii=False)
    print(f"\n\nFull records saved to: {out_path}")

    # Save qualitative examples
    qual_path = output_dir / "qualitative_examples.txt"
    import sys
    import io
    # Redirect stdout to capture print_examples output
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_examples(high_examples, f"HIGH conflict (top {args.topk})")
    print_examples(low_examples,  f"LOW  conflict (bottom {args.topk})")
    sys.stdout = old_stdout
    with open(qual_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Qualitative examples saved to: {qual_path}")


if __name__ == "__main__":
    main()
