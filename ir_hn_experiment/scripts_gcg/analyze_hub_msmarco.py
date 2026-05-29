"""
HNSW Hub Analysis — MS MARCO scale
=====================================

MS MARCO:
  corpus:  8.8M passages
  queries: 502,939 train queries (sample 50K)
  qrels:   dev qrels for judged_pos check

전처리 없이 BEIR 포맷 MS MARCO를 직접 읽음.

Prerequisites:
  # MS MARCO (BEIR format)
  wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
  unzip msmarco.zip -d ./data/raw/

Usage:
  python analyze_hub_msmarco.py \\
    --msmarco_dir  ./data/raw/msmarco \\
    --model_name   BAAI/bge-base-en-v1.5 \\
    --hnsw_m       32 \\
    --hub_top_pct  0.05 \\
    --n_queries    50000 \\
    --topk_search  50 \\
    --qual_topk    15 \\
    --output_dir   ./outputs/hub_msmarco \\
    --batch_size   512

예상 소요 시간 (RTX 5090):
  corpus 인코딩:  ~30분 (8.8M × 768)
  HNSW 구축:      ~10분
  hub degree:     ~1분  (vectorized)
  query 검색:     ~15분 (50K queries × top50)
  총:             ~1시간
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Data loading (BEIR format)
# =============================================================================

def load_jsonl(path, max_rows=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows

def load_qrels(path):
    """BEIR tsv qrels: 3-col (qid\tdid\trel) or 4-col (qid\t0\tdid\trel)"""
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
            elif len(parts) == 4:
                qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
            else:
                continue
            if rel > 0:
                qrels[qid][did] = rel
    return dict(qrels)

def load_corpus(corpus_path):
    """Returns (pids, texts) lists and pid2idx dict."""
    print("Loading corpus (this may take a while)...")
    pids, texts = [], []
    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="corpus", mininterval=5.0):
            row = json.loads(line)
            pid  = str(row["_id"])
            title= (row.get("title") or "").strip()
            text = (row.get("text")  or "").strip()
            full = (title + " " + text).strip() if title else text
            pids.append(pid)
            texts.append(full)
    pid2idx = {pid: i for i, pid in enumerate(pids)}
    return pids, texts, pid2idx

def load_queries(queries_path, max_queries=None, seed=42):
    """Returns sampled {qid: query_text} dict."""
    rows = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append((str(row["_id"]), row["text"]))
    if max_queries and len(rows) > max_queries:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_queries)
    return {qid: text for qid, text in rows}


# =============================================================================
# Encoder
# =============================================================================

class Encoder:
    def __init__(self, model_name, device, max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, texts, batch_size=512, desc=None):
        all_embs = []
        it = range(0, len(texts), batch_size)
        if desc:
            it = tqdm(it, desc=desc, mininterval=10.0)
        for i in it:
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt"
            ).to(self.device)
            out = self.model(**enc)
            emb = out.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().half().numpy())  # float16: 절반 메모리
        return np.vstack(all_embs)  # (N, D) float16


# =============================================================================
# HNSW hub degree — vectorized
# =============================================================================

def compute_hub_degrees_fast(index_hnsw, n):
    """
    Vectorized: offsets 배열로 neighbor range 추출 후 bincount.
    8.8M 규모에서도 수십 초 이내.
    """
    hnsw      = index_hnsw.hnsw
    nb_array  = faiss.vector_to_array(hnsw.neighbors).copy()  # (total_edges,)
    off_array = faiss.vector_to_array(hnsw.offsets).copy()    # (N+1,)

    # level-0 only: offsets의 stride = 2*M (level-0 + level-1+ ...)
    # off_array[i] ~ off_array[i+1] = level-0 neighbors of node i
    # (HNSW stores all levels contiguously, level-0 is always included)

    # valid neighbors only (negative ids = empty slots)
    valid_mask = nb_array >= 0
    valid_nbs  = nb_array[valid_mask].astype(np.int64)

    # clip to valid range
    valid_nbs = valid_nbs[valid_nbs < n]

    degree = np.bincount(valid_nbs, minlength=n).astype(np.int32)
    return degree


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msmarco_dir",  required=True,
                    help="BEIR-format msmarco dir (contains corpus/, queries/, qrels/)")
    ap.add_argument("--model_name",   default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--checkpoint",   default=None)
    ap.add_argument("--hnsw_m",       type=int,   default=32)
    ap.add_argument("--hub_top_pct",  type=float, default=0.05)
    ap.add_argument("--n_queries",    type=int,   default=50000,
                    help="train 쿼리에서 샘플링할 수")
    ap.add_argument("--topk_search",  type=int,   default=50)
    ap.add_argument("--qual_topk",    type=int,   default=15)
    ap.add_argument("--batch_size",   type=int,   default=512)
    ap.add_argument("--output_dir",   default="./outputs/hub_msmarco")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--emb_cache",    default=None,
                    help="corpus embedding cache path (.npy). "
                         "지정하면 인코딩 skip하고 로드.")
    args = ap.parse_args()

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    msmarco = Path(args.msmarco_dir)

    # ------------------------------------------------------------------
    # Load corpus (support both flat and nested BEIR layouts)
    # ------------------------------------------------------------------
    corpus_path = msmarco / "corpus" / "corpus.jsonl"
    if not corpus_path.exists():
        corpus_path = msmarco / "corpus.jsonl"
    all_pids, all_texts, pid2idx = load_corpus(corpus_path)
    n_docs = len(all_pids)
    print(f"corpus: {n_docs:,} docs  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # Load queries (train) + dev qrels
    # ------------------------------------------------------------------
    # queries: try nested then flat
    train_q_path = msmarco / "queries" / "queries.train.jsonl"
    if not train_q_path.exists():
        train_q_path = msmarco / "queries" / "queries.jsonl"
    if not train_q_path.exists():
        train_q_path = msmarco / "queries.jsonl"
    qid2query = load_queries(train_q_path, max_queries=args.n_queries, seed=args.seed)

    # qrels: try nested then flat
    dev_qrels_path = msmarco / "qrels" / "dev.tsv"
    if not dev_qrels_path.exists():
        dev_qrels_path = msmarco / "qrels" / "test.tsv"
    if not dev_qrels_path.exists():
        dev_qrels_path = msmarco / "dev.tsv"
    qrels_dev = load_qrels(dev_qrels_path) if dev_qrels_path.exists() else {}

    all_qids   = list(qid2query.keys())
    query_texts= [qid2query[q] for q in all_qids]
    print(f"queries sampled: {len(all_qids):,}  "
          f"dev qrels: {len(qrels_dev):,}  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # Encode corpus (or load cache)
    # ------------------------------------------------------------------
    model_name = args.checkpoint if args.checkpoint else args.model_name
    encoder    = Encoder(model_name, device, max_len=128)

    cache_path = out_dir / "corpus_embs.npy"
    emb_cache = args.emb_cache or (str(cache_path) if cache_path.exists() else None)

    if emb_cache and Path(emb_cache).exists():
        print(f"Loading corpus embeddings (mmap) from: {emb_cache}")
        # mmap_mode='r': OS pages in only what's accessed → peak RAM = index + one chunk
        corpus_embs = np.load(emb_cache, mmap_mode='r')
        assert corpus_embs.shape[0] == n_docs, "cache size mismatch"
    else:
        print(f"Encoding {n_docs:,} corpus docs (batch={args.batch_size})...")
        corpus_embs = encoder.encode(all_texts, batch_size=args.batch_size,
                                     desc="corpus")
        np.save(cache_path, corpus_embs)
        print(f"  saved to {cache_path}  ({time.time()-t0:.0f}s)")
        # reload as mmap so we don't hold 12.8 GB in RAM during index build
        del corpus_embs
        corpus_embs = np.load(cache_path, mmap_mode='r')

    dim = corpus_embs.shape[1]
    print(f"corpus_embs: {corpus_embs.shape}  ({time.time()-t0:.0f}s)")

    # free text list to save RAM
    del all_texts

    # ------------------------------------------------------------------
    # Build HNSW index  (IndexHNSWSQ, SQ8 — 4× smaller than HNSWFlat)
    # ------------------------------------------------------------------
    # Memory: 8.8M × 768 × 1 byte (SQ8) + graph ≈ 6.7 + 2.3 = 9 GB
    # corpus_embs loaded via mmap → only accessed pages stay in RAM
    # Peak ≈ 9 GB (index) + 0.6 GB (one chunk float32) + overhead ≈ 11 GB
    print(f"Building HNSW+SQ8 index (M={args.hnsw_m}, dim={dim})...")
    index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit,
                              args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch       = 128

    # SQ8 training: learn per-dim [min, max] from 50K sample
    print("  training SQ8 quantiser on 50K sample...")
    train_sample = corpus_embs[:50_000].astype(np.float32)
    index.train(train_sample)
    del train_sample

    # Add in chunks; corpus_embs is mmap'd float16 → cast per chunk
    chunk = 200_000
    for i in tqdm(range(0, n_docs, chunk), desc="index.add", mininterval=10.0):
        chunk_f32 = corpus_embs[i:i+chunk].astype(np.float32)
        index.add(chunk_f32)
        del chunk_f32
    print(f"  index ntotal: {index.ntotal:,}  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # Compute hub degrees (vectorized)
    # ------------------------------------------------------------------
    print("Computing hub degrees (vectorized)...")
    degree = compute_hub_degrees_fast(index, n_docs)
    hub_threshold = np.percentile(degree, (1 - args.hub_top_pct) * 100)
    is_hub = degree >= hub_threshold

    print(f"  degree: min={degree.min()} max={degree.max()} "
          f"mean={degree.mean():.1f} std={degree.std():.1f}")
    print(f"  hub threshold (top {args.hub_top_pct*100:.0f}%): {hub_threshold:.0f}")
    print(f"  hubs: {is_hub.sum():,} / {n_docs:,} "
          f"({is_hub.mean()*100:.1f}%)  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # Encode queries
    # ------------------------------------------------------------------
    print(f"Encoding {len(query_texts):,} queries...")
    query_embs = encoder.encode(query_texts, batch_size=args.batch_size,
                                desc="queries")
    print(f"  query_embs: {query_embs.shape}  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # HNSW search: all queries → topk neg
    # ------------------------------------------------------------------
    print(f"Searching top-{args.topk_search} for {len(all_qids):,} queries...")
    scores_mat, ids_mat = index.search(
        query_embs.astype(np.float32), args.topk_search + 5)
    print(f"  done  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # Build per-neg records
    # ------------------------------------------------------------------
    print("Building per-neg records...")

    neg_query_sims  = defaultdict(list)   # neg_idx → [sim(q,n)]
    neg_appearance  = np.zeros(n_docs, dtype=np.int32)

    records_sim_qn     = []
    records_sim_pn     = []
    records_is_hub     = []
    records_is_judged  = []
    records_hub_degree = []

    # encode positives in bulk per query would be expensive;
    # instead sample a subset for sim(p,n) computation
    POS_SAMPLE = 5000   # queries to compute sim(p,n) for
    pos_sample_qids = set(random.sample(all_qids, min(POS_SAMPLE, len(all_qids))))

    pos_texts_sampled = []
    pos_qid_order     = []
    for qid in all_qids:
        if qid in pos_sample_qids:
            # MS MARCO train qrels for positive
            # use dev qrels as proxy (train qrels too large)
            pos_did = None
            if qid in qrels_dev:
                pos_did = next(iter(qrels_dev[qid]))
            if pos_did and pos_did in pid2idx:
                pos_texts_sampled.append(all_pids[pid2idx[pos_did]])
                pos_qid_order.append(qid)

    # encode positives
    pos_emb_map = {}
    if pos_texts_sampled:
        # get embeddings from corpus_embs (already encoded)
        for qid, pos_pid in zip(pos_qid_order, pos_texts_sampled):
            pidx = pid2idx.get(pos_pid)
            if pidx is not None:
                pos_emb_map[qid] = corpus_embs[pidx]

    for qi, qid in enumerate(tqdm(all_qids, desc="records", mininterval=5.0)):
        qrels_for_q = qrels_dev.get(str(qid), {})
        p_emb = pos_emb_map.get(qid, None)

        cand_ids    = ids_mat[qi]
        cand_scores = scores_mat[qi]

        collected = 0
        for nidx, score in zip(cand_ids, cand_scores):
            nidx = int(nidx)
            if nidx < 0 or nidx >= n_docs:
                continue

            nid = all_pids[nidx]
            is_judged = int(str(nid) in qrels_for_q and qrels_for_q[str(nid)] > 0)
            sim_qn    = float(score)
            hub_flag  = bool(is_hub[nidx])
            hub_deg   = int(degree[nidx])

            sim_pn = float(np.dot(p_emb.astype(np.float32),
                                   corpus_embs[nidx].astype(np.float32))) \
                     if p_emb is not None else float("nan")

            neg_appearance[nidx]     += 1
            neg_query_sims[nidx].append(sim_qn)

            records_sim_qn.append(sim_qn)
            records_sim_pn.append(sim_pn)
            records_is_hub.append(int(hub_flag))
            records_is_judged.append(is_judged)
            records_hub_degree.append(hub_deg)

            collected += 1
            if collected >= args.topk_search:
                break

    records_sim_qn    = np.array(records_sim_qn,    dtype=np.float32)
    records_sim_pn    = np.array(records_sim_pn,    dtype=np.float32)
    records_is_hub    = np.array(records_is_hub,    dtype=np.int8)
    records_is_judged = np.array(records_is_judged, dtype=np.int8)
    records_hub_degree= np.array(records_hub_degree,dtype=np.int32)

    total_appearances = len(records_sim_qn)
    print(f"  total (q,n) pairs: {total_appearances:,}  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # Analysis A: Hub 분포
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analysis A: Hub Degree Distribution")
    print("="*60)

    hub_mask    = records_is_hub == 1
    nonhub_mask = records_is_hub == 0

    hub_appearances    = hub_mask.sum()
    nonhub_appearances = nonhub_mask.sum()

    print(f"  total appearances:  {total_appearances:,}")
    print(f"  hub appearances:    {hub_appearances:,} "
          f"({hub_appearances/total_appearances*100:.1f}%)")
    print(f"  non-hub appearances:{nonhub_appearances:,} "
          f"({nonhub_appearances/total_appearances*100:.1f}%)")
    print(f"\n  degree percentiles:")
    for pct in [50, 75, 90, 95, 99, 99.9]:
        print(f"    p{pct:5.1f}: {np.percentile(degree, pct):.0f}")

    # ------------------------------------------------------------------
    # Analysis B: Hub vs Non-Hub
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analysis B: Hub vs Non-Hub Comparison")
    print("="*60)

    def mean_nonnan(arr, mask):
        vals = arr[mask]
        vals = vals[~np.isnan(vals)]
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    metrics = [
        ("sim(q,n) mean",    records_sim_qn),
        ("sim(p,n) mean",    records_sim_pn),
        ("judged_pos rate",  records_is_judged.astype(np.float32)),
    ]

    print(f"\n  {'metric':<28} {'hub':>10} {'non-hub':>10} {'diff':>10}")
    print(f"  {'-'*60}")
    for label, arr in metrics:
        hm = mean_nonnan(arr, hub_mask)
        nm = mean_nonnan(arr, nonhub_mask)
        diff = hm - nm if not (np.isnan(hm) or np.isnan(nm)) else float("nan")
        print(f"  {label:<28} {hm:>10.4f} {nm:>10.4f} {diff:>+10.4f}")

    print(f"\n  n(hub pairs):    {hub_mask.sum():,}")
    print(f"  n(non-hub pairs):{nonhub_mask.sum():,}")

    # H2: query-specificity
    print(f"\n  H2. Query-Specificity (variance of sim_qn across queries)")
    hub_vars, nonhub_vars = [], []
    for nidx, sim_list in neg_query_sims.items():
        if len(sim_list) < 5:
            continue
        v = float(np.var(sim_list))
        if is_hub[nidx]:
            hub_vars.append(v)
        else:
            nonhub_vars.append(v)

    if hub_vars and nonhub_vars:
        hv_mean = np.mean(hub_vars)
        nv_mean = np.mean(nonhub_vars)
        ratio   = hv_mean / (nv_mean + 1e-9)
        print(f"  hub    sim_qn variance: {hv_mean:.6f}  (n={len(hub_vars):,})")
        print(f"  nonhub sim_qn variance: {nv_mean:.6f}  (n={len(nonhub_vars):,})")
        print(f"  ratio hub/nonhub: {ratio:.3f}")
        if ratio < 0.9:
            print("  → hub neg는 query 간 sim 분산이 작음 (query-agnostic) ✓ H2 강하게 지지")
        elif ratio < 1.0:
            print("  → hub neg가 약간 query-agnostic ✓ H2 약하게 지지")
        else:
            print("  → H2 기각 (hub가 오히려 query-specific)")

    # H1: top-5% hub의 appearance 집중도
    print(f"\n  H1. Appearance 집중도")
    top5_hub_mask = degree >= np.percentile(degree, 95)
    top5_appearances = neg_appearance[top5_hub_mask].sum()
    print(f"  상위 5% hub ({top5_hub_mask.sum():,}개) → "
          f"전체 appearance의 {top5_appearances/total_appearances*100:.1f}%")
    top1_hub_mask = degree >= np.percentile(degree, 99)
    top1_appearances = neg_appearance[top1_hub_mask].sum()
    print(f"  상위 1% hub ({top1_hub_mask.sum():,}개) → "
          f"전체 appearance의 {top1_appearances/total_appearances*100:.1f}%")

    # ------------------------------------------------------------------
    # Analysis C: Qualitative
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analysis C: Top Hub Negatives (Qualitative)")
    print("="*60)

    top_hub_idxs = np.argsort(degree)[::-1][:args.qual_topk]
    hub_query_map = defaultdict(list)
    for qi, qid in enumerate(all_qids):
        for nidx in ids_mat[qi]:
            nidx = int(nidx)
            if nidx >= 0 and nidx < n_docs:
                hub_query_map[nidx].append(qid)

    qual_lines = []
    pid2text_partial = {}
    # reload just the top hub texts
    print("  loading top hub texts...")
    top_hub_set = set(int(i) for i in top_hub_idxs)
    with open(corpus_path, encoding="utf-8") as f:
        for li, line in enumerate(tqdm(f, desc="reload top hubs", total=n_docs, mininterval=5.0)):
            row = json.loads(line)
            pid = str(row["_id"])
            idx = pid2idx.get(pid)
            if idx in top_hub_set:
                title = (row.get("title") or "").strip()
                text  = (row.get("text")  or "").strip()
                pid2text_partial[idx] = (title + " " + text).strip()[:300]
            if len(pid2text_partial) == len(top_hub_set):
                break

    for rank_i, nidx in enumerate(top_hub_idxs):
        nidx = int(nidx)
        nid  = all_pids[nidx]
        deg  = int(degree[nidx])
        text = pid2text_partial.get(nidx, "[not loaded]")
        appeared_in = hub_query_map[nidx]
        sample_qs   = random.sample(appeared_in, min(5, len(appeared_in)))

        line = (f"\n[Hub #{rank_i+1}] neg_id={nid}  degree={deg}  "
                f"appeared_in={len(appeared_in)} queries\n"
                f"  TEXT: {text}\n"
                f"  sample queries:\n")
        for qid in sample_qs:
            qt = qid2query.get(qid, "?")[:100]
            line += f"    [{qid}] {qt}\n"
        print(line)
        qual_lines.append(line)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    summary = {
        "model":        model_name,
        "n_docs":       int(n_docs),
        "n_queries":    len(all_qids),
        "hnsw_m":       args.hnsw_m,
        "hub_top_pct":  args.hub_top_pct,
        "hub_threshold":float(hub_threshold),
        "hub_count":    int(is_hub.sum()),
        "degree_stats": {
            "min":  int(degree.min()),
            "max":  int(degree.max()),
            "mean": float(degree.mean()),
            "std":  float(degree.std()),
            **{f"p{p}": float(np.percentile(degree, p))
               for p in [50, 75, 90, 95, 99]},
        },
        "H1_hub_appearance_pct": float(hub_appearances / total_appearances * 100),
        "H1_top1pct_appearance_pct": float(top1_appearances / total_appearances * 100),
        "H2_hub_var":    float(np.mean(hub_vars))    if hub_vars    else None,
        "H2_nonhub_var": float(np.mean(nonhub_vars)) if nonhub_vars else None,
        "H2_ratio":      float(np.mean(hub_vars) / (np.mean(nonhub_vars) + 1e-9))
                         if hub_vars and nonhub_vars else None,
        "hub_vs_nonhub": {
            "hub_sim_qn":      float(mean_nonnan(records_sim_qn, hub_mask)),
            "nonhub_sim_qn":   float(mean_nonnan(records_sim_qn, nonhub_mask)),
            "hub_sim_pn":      float(mean_nonnan(records_sim_pn, hub_mask)),
            "nonhub_sim_pn":   float(mean_nonnan(records_sim_pn, nonhub_mask)),
            "hub_judged_pos":  float(mean_nonnan(records_is_judged.astype(np.float32), hub_mask)),
            "nonhub_judged_pos":float(mean_nonnan(records_is_judged.astype(np.float32), nonhub_mask)),
        },
        "elapsed_sec": int(time.time() - t0),
    }

    with open(out_dir / "hub_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "qualitative_hubs.txt", "w") as f:
        f.write("\n".join(qual_lines))
    np.save(out_dir / "hub_degrees.npy", degree)

    print(f"\n{'='*60}")
    print(f"Done in {time.time()-t0:.0f}s")
    print(f"Results: {out_dir}")
    print(f"  hub_summary.json")
    print(f"  qualitative_hubs.txt")
    print(f"  hub_degrees.npy")
    print(f"  corpus_embs.npy  (재사용 가능: --emb_cache {out_dir}/corpus_embs.npy)")


if __name__ == "__main__":
    main()
