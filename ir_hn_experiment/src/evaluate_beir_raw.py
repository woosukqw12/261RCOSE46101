"""
Evaluate checkpoints on BEIR datasets using the beir package.

python src/evaluate_beir_raw.py \
    --checkpoints checkpoints/epoch_3 checkpoints_mask_fn/epoch_3 \
    --datasets nfcorpus trec-covid arguana scidocs scifact fiqa \
    --data_root data/beir_raw
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
faiss = None
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


BEIR_DATASETS = [
    "nfcorpus", "trec-covid", "arguana", "scidocs",
    "scifact", "fiqa", "hotpotqa", "fever",
    "dbpedia-entity", "quora", "nq",
]


def load_model(checkpoint: str, device):
    checkpoint = resolve_checkpoint(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()
    return tokenizer, model


def resolve_checkpoint(checkpoint: str) -> str:
    """Accept either a HF checkpoint dir or a parent dir containing epoch_* dirs."""
    ckpt = Path(checkpoint)
    if (ckpt / "config.json").exists():
        return str(ckpt)

    epoch_dirs = []
    for child in ckpt.glob("epoch_*"):
        if child.is_dir() and (child / "config.json").exists():
            try:
                epoch_num = int(child.name.split("_", 1)[1])
            except (IndexError, ValueError):
                epoch_num = -1
            epoch_dirs.append((epoch_num, child))

    if epoch_dirs:
        return str(sorted(epoch_dirs, key=lambda x: x[0])[-1][1])

    return str(ckpt)


@torch.no_grad()
def encode(texts, tokenizer, model, device, batch_size=256, max_length=350, prefix="passage: "):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), leave=False):
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


@torch.no_grad()
def encode_tensor(
    texts,
    tokenizer,
    model,
    device,
    batch_size=512,
    max_length=350,
    prefix="passage: ",
    output_dtype=torch.float16,
):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), leave=False):
        batch = [prefix + t for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, max_length=max_length, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.to(dtype=output_dtype))
    return torch.cat(all_embs, dim=0)


def retrieve_numpy(corpus, queries, tokenizer, model, device, topk=100, encode_batch_size=256):
    pids = list(corpus.keys())
    doc_texts = [(corpus[p].get("title", "") + " " + corpus[p].get("text", "")).strip()
                 for p in pids]
    q_ids = list(queries.keys())
    q_texts = [queries[q] for q in q_ids]

    print(f"    Encoding {len(doc_texts):,} docs...")
    doc_embs = encode(doc_texts, tokenizer, model, device, batch_size=encode_batch_size, prefix="passage: ")
    print(f"    Encoding {len(q_texts):,} queries...")
    q_embs = encode(q_texts, tokenizer, model, device, batch_size=encode_batch_size, prefix="query: ")

    if faiss is not None:
        index = faiss.IndexFlatIP(doc_embs.shape[1])
        index.add(doc_embs)
        scores, indices = index.search(q_embs, topk)
    else:
        print("    faiss unavailable; using numpy inner-product search")
        score_chunks, index_chunks = [], []
        for start in range(0, len(q_embs), 128):
            sims = q_embs[start:start + 128] @ doc_embs.T
            part = np.argpartition(-sims, kth=min(topk, sims.shape[1] - 1), axis=1)[:, :topk]
            part_scores = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            index_chunks.append(np.take_along_axis(part, order, axis=1))
            score_chunks.append(np.take_along_axis(part_scores, order, axis=1))
        indices = np.vstack(index_chunks)
        scores = np.vstack(score_chunks)

    results = {}
    for qi, qid in enumerate(q_ids):
        results[qid] = {pids[idx]: float(scores[qi][r]) for r, idx in enumerate(indices[qi])}
    return results


def iter_corpus_blocks(corpus, pids, block_size):
    for start in range(0, len(pids), block_size):
        end = min(start + block_size, len(pids))
        block_pids = pids[start:end]
        texts = [
            (corpus[p].get("title", "") + " " + corpus[p].get("text", "")).strip()
            for p in block_pids
        ]
        yield start, block_pids, texts


@torch.no_grad()
def retrieve_torch_stream(
    corpus,
    queries,
    tokenizer,
    model,
    device,
    topk=100,
    encode_batch_size=256,
    query_block_size=64,
    doc_block_size=65536,
    search_dtype="float16",
):
    """Memory-bounded retrieval by streaming corpus blocks through GPU top-k.

    This avoids materializing the full doc embedding matrix and the full
    query-doc similarity matrix at once, which is important for NQ/HotpotQA.
    """
    pids = list(corpus.keys())
    q_ids = list(queries.keys())
    q_texts = [queries[q] for q in q_ids]
    n_queries = len(q_ids)

    torch_dtype = torch.float16 if search_dtype == "float16" and device.type == "cuda" else torch.float32
    print(f"    Encoding {n_queries:,} queries...")
    q_embs = encode_tensor(
        q_texts,
        tokenizer,
        model,
        device,
        batch_size=encode_batch_size,
        prefix="query: ",
        output_dtype=torch_dtype,
    )

    top_scores = torch.full((n_queries, topk), -torch.inf, dtype=torch_dtype, device=device)
    top_indices = torch.full((n_queries, topk), -1, dtype=torch.int64, device=device)

    print(
        f"    Streaming {len(pids):,} docs "
        f"(doc_block={doc_block_size:,}, query_block={query_block_size}, dtype={torch_dtype})..."
    )
    for doc_start, block_pids, doc_texts in tqdm(
        iter_corpus_blocks(corpus, pids, doc_block_size),
        total=(len(pids) + doc_block_size - 1) // doc_block_size,
        leave=False,
    ):
        doc_t = encode_tensor(
            doc_texts,
            tokenizer,
            model,
            device,
            batch_size=encode_batch_size,
            prefix="passage: ",
            output_dtype=torch_dtype,
        )
        block_topk = min(topk, doc_t.shape[0])

        for q_start in range(0, n_queries, query_block_size):
            q_end = min(q_start + query_block_size, n_queries)
            q_t = q_embs[q_start:q_end]
            sims = q_t @ doc_t.T
            block_scores, block_idx = torch.topk(sims, block_topk, dim=1)
            block_idx = block_idx.to(torch.int64) + doc_start

            merged_scores = torch.cat([top_scores[q_start:q_end], block_scores], dim=1)
            merged_idx = torch.cat([top_indices[q_start:q_end], block_idx], dim=1)
            new_scores, pos = torch.topk(merged_scores, topk, dim=1)
            new_idx = torch.gather(merged_idx, 1, pos)

            top_scores[q_start:q_end] = new_scores
            top_indices[q_start:q_end] = new_idx

        del doc_t

    top_scores_np = top_scores.float().cpu().numpy()
    top_indices_np = top_indices.cpu().numpy()
    results = {}
    for qi, qid in enumerate(q_ids):
        row = {}
        for score, idx in zip(top_scores_np[qi], top_indices_np[qi]):
            if idx >= 0:
                row[pids[int(idx)]] = float(score)
        results[qid] = row
    return results


def retrieve(
    corpus,
    queries,
    tokenizer,
    model,
    device,
    topk=100,
    search_mode="auto",
    encode_batch_size=256,
    query_block_size=64,
    doc_block_size=65536,
    search_dtype="float16",
):
    if search_mode == "auto":
        search_mode = "torch_stream" if len(corpus) >= 500_000 else "numpy"
    if search_mode == "torch_stream":
        return retrieve_torch_stream(
            corpus,
            queries,
            tokenizer,
            model,
            device,
            topk=topk,
            encode_batch_size=encode_batch_size,
            query_block_size=query_block_size,
            doc_block_size=doc_block_size,
            search_dtype=search_dtype,
        )
    if search_mode == "numpy":
        return retrieve_numpy(
            corpus,
            queries,
            tokenizer,
            model,
            device,
            topk=topk,
            encode_batch_size=encode_batch_size,
        )
    raise ValueError(f"Unknown search_mode: {search_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=["nfcorpus", "trec-covid", "arguana", "scidocs", "scifact", "fiqa"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_root", default="data/beir_raw")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--search_mode", choices=["auto", "numpy", "torch_stream"], default="auto")
    parser.add_argument("--encode_batch_size", type=int, default=256)
    parser.add_argument("--query_block_size", type=int, default=64)
    parser.add_argument("--doc_block_size", type=int, default=65536)
    parser.add_argument("--search_dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    all_results = {}  # {ds: {ckpt: {metric: val}}}

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  split={args.split}")

        # download if needed
        ds_dir = data_root / ds_name
        if not (ds_dir / "corpus.jsonl").exists():
            print(f"  Downloading {ds_name}...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
            zip_path = util.download_and_unzip(url, str(data_root))

        split_used = args.split
        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=str(ds_dir)).load(split=split_used)
            print(f"  Docs: {len(corpus):,}  Queries: {len(queries):,}")
        except Exception as e:
            if ds_name == "msmarco" and args.split == "test":
                split_used = "dev"
                print(f"  test split unavailable for msmarco; trying split=dev")
                try:
                    corpus, queries, qrels = GenericDataLoader(data_folder=str(ds_dir)).load(split=split_used)
                    print(f"  Docs: {len(corpus):,}  Queries: {len(queries):,}")
                except Exception as e2:
                    print(f"  SKIP: {e2}")
                    continue
            else:
                print(f"  SKIP: {e}")
                continue

        all_results[ds_name] = {}

        for ckpt in args.checkpoints:
            print(f"\n  [{ckpt}]")
            try:
                resolved_ckpt = resolve_checkpoint(ckpt)
                if resolved_ckpt != ckpt:
                    print(f"    resolved checkpoint: {resolved_ckpt}")
                tokenizer, model = load_model(resolved_ckpt, device)
                results = retrieve(
                    corpus,
                    queries,
                    tokenizer,
                    model,
                    device,
                    args.topk,
                    search_mode=args.search_mode,
                    encode_batch_size=args.encode_batch_size,
                    query_block_size=args.query_block_size,
                    doc_block_size=args.doc_block_size,
                    search_dtype=args.search_dtype,
                )

                ndcg, map_score, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [10, 100])
                metrics = {
                    "nDCG@10": round(ndcg["NDCG@10"], 4),
                    "MAP@10":  round(map_score["MAP@10"], 4),
                    "Recall@10": round(recall.get("Recall@10", 0), 4),
                    "Recall@100": round(recall.get("Recall@100", 0), 4),
                    "split": split_used,
                }
                all_results[ds_name][ckpt] = metrics
                print(f"    nDCG@10={metrics['nDCG@10']:.4f}  MAP@10={metrics['MAP@10']:.4f}  R@100={metrics['Recall@100']:.4f}")

                del model, tokenizer
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback; traceback.print_exc()

    # 비교 테이블
    print(f"\n\n{'='*80}")
    print("FINAL — nDCG@10")
    print(f"{'='*80}")
    labels = [c.replace("checkpoints", "ckpt") for c in args.checkpoints]
    header = f"{'Dataset':<18}" + "".join(f"{l:>25}" for l in labels)
    print(header)
    print("-" * len(header))
    for ds, ds_res in all_results.items():
        row = f"{ds:<18}"
        for ckpt in args.checkpoints:
            m = ds_res.get(ckpt)
            row += f"{m['nDCG@10']:>25.4f}" if m else f"{'N/A':>25}"
        print(row)

    out_path = Path(args.output) if args.output else Path("results") / f"beir_comparison_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
