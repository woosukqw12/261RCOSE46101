"""
HNSW Hub Analysis for Dense Retrieval
======================================

가설 검증:
  H1. Hub neg는 학습에서 불균형하게 많이 등장한다
      - 상위 5% hub가 전체 neg 등장 횟수의 X% 차지
  H2. Hub neg는 query-specific하지 않다
      - hub neg의 sim(q,n) 분포가 query 내용과 무관
      - non-hub neg는 query-specific sim 분포
  H3. (추후 학습 실험에서 검증)
      - hub down-weight → retrieval 성능 향상

출력:
  A. Hub 분포 분석
     - degree 분포 (mean, std, 상위 1%/5%/10%)
     - 상위 5% hub가 전체 neg appearance에서 차지하는 비율
  B. Hub vs Non-Hub 비교
     - sim(q,n), sim(p,n), conflict 평균 비교
     - judged positive rate (qrels 기준)
     - query-specificity (query별 sim(q,n) variance)
  C. Qualitative
     - 상위 hub neg 텍스트 + 어떤 query들에 등장하는지

Usage:
  python analyze_hub.py \\
    --dataset_dir ./data/processed/fiqa \\
    --model_name  BAAI/bge-base-en-v1.5 \\
    --hnsw_m      32 \\
    --hub_top_pct 0.05 \\
    --output_dir  ./outputs/hub_analysis
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import faiss
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
        text = ((row.get("title","") or "") + " " + (row.get("text","") or "")).strip()
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
    def encode(self, texts, batch_size=128, desc=None):
        all_embs = []
        it = range(0, len(texts), batch_size)
        if desc:
            it = tqdm(it, desc=desc, leave=False)
        for i in it:
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt"
            ).to(self.device)
            out = self.model(**enc)
            emb = out.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
        return np.vstack(all_embs)  # (N, D)


# =============================================================================
# HNSW hub degree computation
# =============================================================================

def compute_hub_degrees(index_hnsw, n):
    """
    HNSW level-0 graph에서 각 노드의 incoming degree 계산.
    offsets 배열로 neighbor range를 찾음.
    """
    hnsw     = index_hnsw.hnsw
    nb_array = faiss.vector_to_array(hnsw.neighbors)   # flat neighbor ids
    off_array= faiss.vector_to_array(hnsw.offsets)     # offsets[i] = start of node i

    degree = np.zeros(n, dtype=np.int32)

    for i in range(n):
        begin = int(off_array[i])
        end   = int(off_array[i+1])
        neighbors_of_i = nb_array[begin:end]
        for nb_id in neighbors_of_i:
            nb_id = int(nb_id)
            if 0 <= nb_id < n:
                degree[nb_id] += 1

    return degree  # (N,)


# =============================================================================
# Conflict
# =============================================================================

def compute_conflict_batch(q_emb, p_emb, n_embs):
    """
    q_emb:  (D,)
    p_emb:  (D,)
    n_embs: (K, D)
    returns conflict: (K,)
    """
    q = torch.from_numpy(q_emb)
    p = torch.from_numpy(p_emb)
    n = torch.from_numpy(n_embs)

    g_pos = p - (q @ p) * q
    g_neg = n - (n @ q).unsqueeze(1) * q.unsqueeze(0)

    g_pos_norm = F.normalize(g_pos.unsqueeze(0), dim=-1)
    g_neg_norm = F.normalize(g_neg, dim=-1)
    conflict   = (g_pos_norm * g_neg_norm).sum(dim=-1)
    return conflict.numpy()


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",  required=True)
    ap.add_argument("--model_name",   default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--checkpoint",   default=None)
    ap.add_argument("--hnsw_m",       type=int,   default=32,
                    help="HNSW M parameter (neighbors per node)")
    ap.add_argument("--hub_top_pct",  type=float, default=0.05,
                    help="상위 X 비율을 hub로 정의")
    ap.add_argument("--topk_search",  type=int,   default=50,
                    help="각 query에서 HNSW로 검색할 neg 수")
    ap.add_argument("--qual_topk",    type=int,   default=10,
                    help="qualitative 출력 hub 수")
    ap.add_argument("--output_dir",   default="./outputs/hub_analysis")
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.dataset_dir)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    train_rows = load_jsonl(data_dir / "train.jsonl")
    qrels_dev  = load_qrels(data_dir / "qrels_dev.json")
    pid2text   = load_collection(data_dir / "collection.jsonl")

    qid2row  = {str(r["qid"]): r for r in train_rows}
    all_qids = list(qid2row.keys())

    # corpus: all pids in collection
    all_pids  = list(pid2text.keys())
    all_texts = [pid2text[p] for p in all_pids]
    pid2idx   = {pid: i for i, pid in enumerate(all_pids)}
    print(f"  corpus: {len(all_pids)} docs, queries: {len(all_qids)}")

    # ------------------------------------------------------------------
    # Encode corpus
    # ------------------------------------------------------------------
    model_name = args.checkpoint if args.checkpoint else args.model_name
    print(f"Encoding corpus with: {model_name}")
    encoder = BGEEncoder(model_name, device)
    corpus_embs = encoder.encode(all_texts, batch_size=128, desc="corpus")
    print(f"  corpus_embs: {corpus_embs.shape}")

    # ------------------------------------------------------------------
    # Build HNSW index
    # ------------------------------------------------------------------
    print(f"Building HNSW index (M={args.hnsw_m})...")
    dim   = corpus_embs.shape[1]
    index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch       = 128
    # normalize for cosine (already normalized)
    index.add(corpus_embs)
    print(f"  index ntotal: {index.ntotal}")

    # ------------------------------------------------------------------
    # Compute hub degrees
    # ------------------------------------------------------------------
    print("Computing hub degrees...")
    degree = compute_hub_degrees(index, index.ntotal)
    hub_threshold = np.percentile(degree, (1 - args.hub_top_pct) * 100)
    print(f"  degree: min={degree.min()} max={degree.max()} "
          f"mean={degree.mean():.1f} std={degree.std():.1f}")
    print(f"  hub threshold (top {args.hub_top_pct*100:.0f}%): {hub_threshold:.0f}")

    is_hub = degree >= hub_threshold  # (N,)
    print(f"  hubs: {is_hub.sum()} / {len(is_hub)} "
          f"({is_hub.mean()*100:.1f}%)")

    # ------------------------------------------------------------------
    # Search: each query → topk_search neg via HNSW
    # ------------------------------------------------------------------
    print("Searching negatives for all queries...")
    query_texts = [qid2row[qid]["query"] for qid in all_qids]
    query_embs  = encoder.encode(query_texts, batch_size=128, desc="queries")

    # HNSW search
    scores_mat, ids_mat = index.search(query_embs, args.topk_search + 10)
    # ids_mat: (Q, topk_search+10) — add buffer for qrels filtering

    # ------------------------------------------------------------------
    # Build per-neg records
    # ------------------------------------------------------------------
    print("Building per-neg records...")

    # neg_id → list of queries it appeared in (for H2)
    neg_query_sims = defaultdict(list)  # neg_idx → [sim(q,n), ...]

    records = []
    pos_emb_cache = {}

    for qi, qid in enumerate(tqdm(all_qids, desc="records")):
        row = qid2row[qid]
        qrels_for_q = qrels_dev.get(str(qid), {})

        q_emb = query_embs[qi]

        # positive embedding (first positive)
        pos_texts = row["positives"][:1]
        if not pos_texts:
            continue
        if qid not in pos_emb_cache:
            pos_emb_cache[qid] = encoder.encode(pos_texts)[0]
        p_emb = pos_emb_cache[qid]

        # retrieve neg candidates from HNSW
        cand_ids   = ids_mat[qi]
        cand_scores= scores_mat[qi]

        collected = 0
        for rank, (nidx, score) in enumerate(zip(cand_ids, cand_scores)):
            if nidx < 0 or nidx >= len(all_pids):
                continue
            nid = all_pids[nidx]

            # skip if this is the positive
            if nid in row.get("pos_ids", []):
                continue
            if str(nid) in (row.get("pos_ids") or []):
                continue

            is_judged = int(str(nid) in qrels_for_q and qrels_for_q[str(nid)] > 0)
            sim_qn    = float(score)
            sim_pn    = float(np.dot(p_emb, corpus_embs[nidx]))
            hub_deg   = int(degree[nidx])
            hub_flag  = bool(is_hub[nidx])

            neg_query_sims[nidx].append(sim_qn)

            records.append({
                "qid":        qid,
                "neg_idx":    int(nidx),
                "neg_id":     nid,
                "rank":       rank,
                "sim_qn":     sim_qn,
                "sim_pn":     sim_pn,
                "hub_degree": hub_deg,
                "is_hub":     hub_flag,
                "is_judged_pos": is_judged,
                "query_text": row["query"],
                "pos_text":   pos_texts[0],
                "neg_text":   pid2text[nid][:200],
            })
            collected += 1
            if collected >= args.topk_search:
                break

    print(f"  total (q,n) pairs: {len(records)}")

    # ------------------------------------------------------------------
    # Analysis A: Hub 분포
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analysis A: Hub Degree Distribution")
    print("="*60)

    # appearance count: how many times each neg appeared across queries
    neg_appearance = defaultdict(int)
    for r in records:
        neg_appearance[r["neg_idx"]] += 1

    total_appearances = len(records)
    hub_indices = set(i for i, h in enumerate(is_hub) if h)

    hub_appearances    = sum(v for k, v in neg_appearance.items() if k in hub_indices)
    nonhub_appearances = total_appearances - hub_appearances

    print(f"  total neg appearances: {total_appearances}")
    print(f"  hub appearances:    {hub_appearances} "
          f"({hub_appearances/total_appearances*100:.1f}%)")
    print(f"  non-hub appearances:{nonhub_appearances} "
          f"({nonhub_appearances/total_appearances*100:.1f}%)")

    # degree percentiles
    for pct in [50, 75, 90, 95, 99]:
        print(f"  degree p{pct}: {np.percentile(degree, pct):.0f}")

    # ------------------------------------------------------------------
    # Analysis B: Hub vs Non-Hub comparison
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analysis B: Hub vs Non-Hub Comparison")
    print("="*60)

    hub_recs    = [r for r in records if r["is_hub"]]
    nonhub_recs = [r for r in records if not r["is_hub"]]

    def stats(recs, key):
        vals = [r[key] for r in recs]
        return np.mean(vals), np.std(vals)

    print(f"\n  {'metric':<30} {'hub':>12} {'non-hub':>12} {'diff':>10}")
    print(f"  {'-'*66}")
    for key, label in [
        ("sim_qn",        "sim(q,n) mean"),
        ("sim_pn",        "sim(p,n) mean"),
        ("is_judged_pos", "judged_pos rate"),
    ]:
        hm, hs = stats(hub_recs,    key)
        nm, ns = stats(nonhub_recs, key)
        print(f"  {label:<30} {hm:>12.4f} {nm:>12.4f} {hm-nm:>+10.4f}")

    print(f"\n  n(hub pairs):    {len(hub_recs)}")
    print(f"  n(non-hub pairs):{len(nonhub_recs)}")

    # H2: query-specificity
    # query-specificity = variance of sim(q,n) across different queries
    print(f"\n  H2. Query-Specificity (variance of sim_qn across queries)")
    hub_vars    = []
    nonhub_vars = []
    for nidx, sim_list in neg_query_sims.items():
        if len(sim_list) < 3:
            continue
        v = float(np.var(sim_list))
        if is_hub[nidx]:
            hub_vars.append(v)
        else:
            nonhub_vars.append(v)

    if hub_vars and nonhub_vars:
        print(f"  hub    sim_qn variance: mean={np.mean(hub_vars):.6f}  "
              f"(n={len(hub_vars)})")
        print(f"  nonhub sim_qn variance: mean={np.mean(nonhub_vars):.6f}  "
              f"(n={len(nonhub_vars)})")
        ratio = np.mean(hub_vars) / (np.mean(nonhub_vars) + 1e-9)
        print(f"  ratio hub/nonhub variance: {ratio:.3f}")
        if ratio < 1.0:
            print("  → hub neg는 query 간 sim 분산이 작음 (query-agnostic) ✓ H2 지지")
        else:
            print("  → hub neg가 오히려 variance 큼 (H2 기각)")

    # ------------------------------------------------------------------
    # Analysis C: Qualitative — top hub negatives
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Analysis C: Qualitative — Top Hub Negatives")
    print("="*60)

    # top hubs by degree
    top_hub_idxs = np.argsort(degree)[::-1][:args.qual_topk]

    # gather queries that each hub appeared in
    hub_query_map = defaultdict(list)
    for r in records:
        hub_query_map[r["neg_idx"]].append(r["qid"])

    qual_lines = []
    for rank_i, nidx in enumerate(top_hub_idxs):
        nid    = all_pids[nidx]
        deg    = int(degree[nidx])
        text   = pid2text[nid][:300]
        q_list = list(set(hub_query_map[nidx]))[:5]

        line = (f"\n[Hub #{rank_i+1}] neg_id={nid}  degree={deg}\n"
                f"  TEXT: {text}\n"
                f"  appeared in queries ({len(hub_query_map[nidx])}):\n")
        for qid in q_list:
            qt = qid2row[qid]["query"][:100]
            line += f"    - [{qid}] {qt}\n"
        print(line)
        qual_lines.append(line)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    summary = {
        "degree_stats": {
            "min": int(degree.min()), "max": int(degree.max()),
            "mean": float(degree.mean()), "std": float(degree.std()),
            "p50": float(np.percentile(degree, 50)),
            "p90": float(np.percentile(degree, 90)),
            "p95": float(np.percentile(degree, 95)),
            "p99": float(np.percentile(degree, 99)),
        },
        "hub_threshold": float(hub_threshold),
        "hub_count": int(is_hub.sum()),
        "total_docs": int(len(is_hub)),
        "hub_appearance_pct": float(hub_appearances / total_appearances * 100),
        "hub_vs_nonhub": {
            "hub_sim_qn_mean":       float(np.mean([r["sim_qn"] for r in hub_recs])),
            "nonhub_sim_qn_mean":    float(np.mean([r["sim_qn"] for r in nonhub_recs])),
            "hub_sim_pn_mean":       float(np.mean([r["sim_pn"] for r in hub_recs])),
            "nonhub_sim_pn_mean":    float(np.mean([r["sim_pn"] for r in nonhub_recs])),
            "hub_judged_pos_rate":   float(np.mean([r["is_judged_pos"] for r in hub_recs])),
            "nonhub_judged_pos_rate":float(np.mean([r["is_judged_pos"] for r in nonhub_recs])),
            "hub_sim_qn_var":        float(np.mean(hub_vars)) if hub_vars else None,
            "nonhub_sim_qn_var":     float(np.mean(nonhub_vars)) if nonhub_vars else None,
        },
    }

    with open(out_dir / "hub_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "qualitative_hubs.txt", "w") as f:
        f.write("\n".join(qual_lines))

    # degree array
    np.save(out_dir / "hub_degrees.npy", degree)

    print(f"\nResults saved to: {out_dir}")
    print(f"  hub_summary.json")
    print(f"  qualitative_hubs.txt")
    print(f"  hub_degrees.npy")


if __name__ == "__main__":
    main()
