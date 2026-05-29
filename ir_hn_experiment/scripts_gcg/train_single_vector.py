import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)


# -----------------------------
# util
# -----------------------------
def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_queries(path: Path):
    rows = load_jsonl(path)
    return {str(x["qid"]): x["query"] for x in rows}


def load_qrels(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_collection(path: Path):
    rows = load_jsonl(path)
    out = []
    for row in rows:
        pid = str(row["pid"])
        text = ((row.get("title", "") or "") + " " + (row.get("text", "") or "")).strip()
        out.append((pid, text))
    return out


def dcg(rels):
    s = 0.0
    for i, rel in enumerate(rels):
        s += (2 ** rel - 1) / math.log2(i + 2)
    return s


def evaluate_run(qrels, results, k=10):
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for qid, rel_docs in qrels.items():
        positives = {pid: rel for pid, rel in rel_docs.items() if rel > 0}
        if not positives:
            continue

        ranked = sorted(results.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        ranked_ids = [pid for pid, _ in ranked]

        topk = ranked_ids[:k]
        rels = [positives.get(pid, 0) for pid in topk]
        ideal = sorted(positives.values(), reverse=True)[:k]
        ndcg_scores.append(dcg(rels) / max(dcg(ideal), 1e-12))

        rr = 0.0
        for i, pid in enumerate(topk):
            if positives.get(pid, 0) > 0:
                rr = 1.0 / (i + 1)
                break
        mrr_scores.append(rr)

        top100 = set(ranked_ids[:100])
        recall_scores.append(sum(1 for pid in positives if pid in top100) / max(len(positives), 1))

    return {
        "nDCG@10": sum(ndcg_scores) / max(len(ndcg_scores), 1),
        "MRR@10": sum(mrr_scores) / max(len(mrr_scores), 1),
        "Recall@100": sum(recall_scores) / max(len(recall_scores), 1),
    }


# -----------------------------
# dataset
# -----------------------------
class TrainDataset(Dataset):
    def __init__(self, path: Path, neg_per_query: int = 8, seed: int = 42):
        self.rows = load_jsonl(path)
        self.neg_per_query = neg_per_query
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def refresh_rows(self, rows):
        self.rows = rows

    def __getitem__(self, idx):
        row = self.rows[idx]
        pos_idx = self.rng.randrange(len(row["positives"]))
        negs = row["negatives"]
        neg_ids = row["neg_ids"]

        if len(negs) > self.neg_per_query:
            chosen = self.rng.sample(range(len(negs)), self.neg_per_query)
            negs = [negs[i] for i in chosen]
            neg_ids = [neg_ids[i] for i in chosen]

        return {
            "qid": int(row["qid"]) if str(row["qid"]).isdigit() else str(row["qid"]),
            "query": row["query"],
            "pos_id": row["pos_ids"][pos_idx],
            "positive": row["positives"][pos_idx],
            "neg_ids": neg_ids,
            "negatives": negs,
        }


def collate_fn(batch):
    qids = [x["qid"] for x in batch]
    queries = [x["query"] for x in batch]
    pos_ids = [x["pos_id"] for x in batch]
    positives = [x["positive"] for x in batch]
    neg_ids = [x["neg_ids"] for x in batch]
    negatives = [x["negatives"] for x in batch]

    K = max(len(x) for x in neg_ids)
    # pad by repeating last negative if needed
    for i in range(len(batch)):
        if len(neg_ids[i]) < K:
            pad_n = K - len(neg_ids[i])
            neg_ids[i] = neg_ids[i] + neg_ids[i][-1:] * pad_n
            negatives[i] = negatives[i] + negatives[i][-1:] * pad_n

    return {
        "qids": qids,
        "queries": queries,
        "pos_ids": pos_ids,
        "positives": positives,
        "neg_ids": neg_ids,
        "negatives": negatives,
        "K": K,
    }


# -----------------------------
# model wrappers
# -----------------------------
class BGEBiEncoder(nn.Module):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.q_tok = AutoTokenizer.from_pretrained(model_name)
        self.d_tok = AutoTokenizer.from_pretrained(model_name)

    def encode_query(self, texts, device, max_length=64):
        toks = self.q_tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.model(**toks)
        emb = F.normalize(out.last_hidden_state[:, 0], dim=-1)
        return emb

    def encode_doc(self, texts, device, max_length=180):
        toks = self.d_tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.model(**toks)
        emb = F.normalize(out.last_hidden_state[:, 0], dim=-1)
        return emb

    def parameters_to_optimize(self):
        return self.model.parameters()

    def save(self, out_dir: Path):
        self.model.save_pretrained(out_dir)
        self.q_tok.save_pretrained(out_dir)

    @property
    def train_module(self):
        return self.model


class DPRBiEncoder(nn.Module):
    def __init__(
        self,
        q_model_name="facebook/dpr-question_encoder-single-nq-base",
        d_model_name="facebook/dpr-ctx_encoder-single-nq-base",
    ):
        super().__init__()
        self.q_model = DPRQuestionEncoder.from_pretrained(q_model_name)
        self.d_model = DPRContextEncoder.from_pretrained(d_model_name)
        self.q_tok = DPRQuestionEncoderTokenizer.from_pretrained(q_model_name)
        self.d_tok = DPRContextEncoderTokenizer.from_pretrained(d_model_name)

    def encode_query(self, texts, device, max_length=64):
        toks = self.q_tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        emb = F.normalize(self.q_model(**toks).pooler_output, dim=-1)
        return emb

    def encode_doc(self, texts, device, max_length=180):
        toks = self.d_tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        emb = F.normalize(self.d_model(**toks).pooler_output, dim=-1)
        return emb

    def parameters_to_optimize(self):
        return list(self.q_model.parameters()) + list(self.d_model.parameters())

    def save(self, out_dir: Path):
        (out_dir / "query_encoder").mkdir(parents=True, exist_ok=True)
        (out_dir / "doc_encoder").mkdir(parents=True, exist_ok=True)
        self.q_model.save_pretrained(out_dir / "query_encoder")
        self.q_tok.save_pretrained(out_dir / "query_encoder")
        self.d_model.save_pretrained(out_dir / "doc_encoder")
        self.d_tok.save_pretrained(out_dir / "doc_encoder")

    @property
    def train_module(self):
        return self.q_model


# -----------------------------
# dynamic router
# -----------------------------
@dataclass
class RouterConfig:
    temperature: float = 0.02
    ema_decay: float = 0.9

    # harmfulness = alpha * margin + beta * volatility
    alpha: float = 1.0
    beta: float = 0.2

    # 좀 더 공격적으로 보이도록 기본값 완화
    tau_safe: float = -0.05
    tau_amb: float = 0.00
    tau_susp: float = 0.05

    w_safe: float = 1.0
    w_amb: float = 0.5
    w_susp: float = 0.1
    w_harm: float = 0.0

    cooldown_steps: int = 1
    eps: float = 1e-8


class DynamicRouter:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.t = 0
        self.state = {}  # (qid, nid) -> prev_margin, ema_h, cooldown_until

    def set_time(self, t: int):
        self.t = int(t)

    def _get(self, qid: str, nid: str):
        key = (str(qid), str(nid))
        if key not in self.state:
            self.state[key] = {"prev_margin": 0.0, "ema_h": 0.0, "cooldown_until": -1}
        return self.state[key]

    @torch.no_grad()
    def update(self, qids, neg_ids, pos_scores, neg_scores):
        B, K = neg_scores.shape
        for i in range(B):
            qid = str(qids[i])
            pos_s = float(pos_scores[i].item())
            for j in range(K):
                nid = str(neg_ids[i][j])
                st = self._get(qid, nid)
                margin = float(neg_scores[i, j].item()) - pos_s
                volatility = abs(margin - st["prev_margin"])
                raw_h = self.cfg.alpha * margin + self.cfg.beta * volatility
                ema_h = self.cfg.ema_decay * st["ema_h"] + (1 - self.cfg.ema_decay) * raw_h
                st["prev_margin"] = margin
                st["ema_h"] = ema_h
                if ema_h >= self.cfg.tau_susp:
                    st["cooldown_until"] = max(st["cooldown_until"], self.t + self.cfg.cooldown_steps)

    def build_weights(self, qids, neg_ids, device):
        B, K = len(qids), len(neg_ids[0])
        w = torch.zeros(B, K, dtype=torch.float32, device=device)
        hvals = torch.zeros(B, K, dtype=torch.float32, device=device)
        names = []

        for i in range(B):
            row = []
            for j in range(K):
                st = self._get(qids[i], neg_ids[i][j])
                ema_h = st["ema_h"]
                cd = st["cooldown_until"]

                hvals[i, j] = float(ema_h)

                if self.t <= cd:
                    if ema_h >= self.cfg.tau_susp:
                        ww, name = self.cfg.w_harm, "harmful"
                    else:
                        ww, name = self.cfg.w_susp, "suspicious"
                else:
                    if ema_h < self.cfg.tau_safe:
                        ww, name = self.cfg.w_safe, "safe"
                    elif ema_h < self.cfg.tau_amb:
                        ww, name = self.cfg.w_amb, "ambiguous"
                    elif ema_h < self.cfg.tau_susp:
                        ww, name = self.cfg.w_susp, "suspicious"
                    else:
                        ww, name = self.cfg.w_harm, "harmful"

                w[i, j] = ww
                row.append(name)
            names.append(row)

        return w, names, hvals


# -----------------------------
# retrieval for eval / refresh
# -----------------------------
@torch.no_grad()
def encode_corpus(model, collection, device, d_maxlen=180, batch_size=64):
    doc_ids = []
    embs = []
    texts = [text for _, text in collection]
    ids = [pid for pid, _ in collection]

    for i in tqdm(range(0, len(texts), batch_size), desc="encode corpus"):
        batch = texts[i:i + batch_size]
        emb = model.encode_doc(batch, device=device, max_length=d_maxlen)
        embs.append(emb.cpu())
        doc_ids.extend(ids[i:i + batch_size])

    embs = torch.cat(embs, dim=0)
    return doc_ids, embs


@torch.no_grad()
def retrieve(model, queries, collection, device, q_maxlen=64, d_maxlen=180, topk=100, batch_size=64):
    doc_ids, doc_embs = encode_corpus(model, collection, device=device, d_maxlen=d_maxlen, batch_size=batch_size)
    doc_embs = doc_embs.to(device)

    q_items = list(queries.items())
    results = {}

    for i in tqdm(range(0, len(q_items), batch_size), desc="retrieve"):
        batch = q_items[i:i + batch_size]
        qids = [qid for qid, _ in batch]
        qtexts = [q for _, q in batch]
        q_emb = model.encode_query(qtexts, device=device, max_length=q_maxlen)
        scores = q_emb @ doc_embs.T
        k = min(topk, scores.size(1))
        vals, inds = torch.topk(scores, k=k, dim=1)

        for b in range(len(qids)):
            results[qids[b]] = {
                doc_ids[int(inds[b, j].item())]: float(vals[b, j].item())
                for j in range(k)
            }

    return results


@torch.no_grad()
def refresh_train_rows(model, train_rows, collection, qrels_train, device, topk=50, keep_negs=8, q_maxlen=64, d_maxlen=180, batch_size=64):
    # encode corpus once
    doc_ids, doc_embs = encode_corpus(model, collection, device=device, d_maxlen=d_maxlen, batch_size=batch_size)
    doc_embs = doc_embs.to(device)
    pid_to_text = {pid: text for pid, text in collection}

    qid_to_row = {str(row["qid"]): row for row in train_rows}
    queries = {str(row["qid"]): row["query"] for row in train_rows}
    q_items = list(queries.items())
    refreshed = []

    for i in tqdm(range(0, len(q_items), batch_size), desc="refresh negatives"):
        batch = q_items[i:i + batch_size]
        qids = [qid for qid, _ in batch]
        qtexts = [q for _, q in batch]
        q_emb = model.encode_query(qtexts, device=device, max_length=q_maxlen)
        scores = q_emb @ doc_embs.T
        k = min(topk, scores.size(1))
        vals, inds = torch.topk(scores, k=k, dim=1)

        for b, qid in enumerate(qids):
            row = qid_to_row[qid]
            pos_ids = [str(x) for x in row["pos_ids"]]
            rel_docs = set(str(pid) for pid, rel in qrels_train.get(qid, {}).items() if rel > 0)

            neg_ids = []
            for j in range(k):
                pid = doc_ids[int(inds[b, j].item())]
                if pid in rel_docs or pid in pos_ids:
                    continue
                neg_ids.append(pid)
                if len(neg_ids) >= keep_negs:
                    break

            if not neg_ids:
                continue

            new_row = dict(row)
            new_row["neg_ids"] = neg_ids
            new_row["negatives"] = [pid_to_text[nid] for nid in neg_ids]
            refreshed.append(new_row)

    return refreshed


# -----------------------------
# train
# -----------------------------
def weighted_infonce(q_emb, p_emb, n_emb, neg_weights, temperature=0.02, eps=1e-8):
    pos_scores = torch.sum(q_emb * p_emb, dim=-1)
    neg_scores = torch.einsum("bd,bkd->bk", q_emb, n_emb)
    pos_logits = pos_scores / temperature
    neg_logits = neg_scores / temperature
    pos_exp = torch.exp(pos_logits)
    neg_exp = torch.exp(neg_logits) * neg_weights
    denom = pos_exp + neg_exp.sum(dim=-1) + eps
    loss = -torch.log(pos_exp / denom)
    return loss.mean(), pos_scores, neg_scores


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collection = load_collection(dataset_dir / "collection.jsonl")
    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    train_qrels = load_qrels(dataset_dir / "qrels_train.json")

    train_ds = TrainDataset(dataset_dir / "train.jsonl", neg_per_query=args.neg_per_query, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.model_type == "bge":
        model = BGEBiEncoder(args.model_name)
        model.train_module.to(device)
    elif args.model_type == "dpr":
        model = DPRBiEncoder(args.q_model_name, args.d_model_name)
        model.q_model.to(device)
        model.d_model.to(device)
    else:
        raise ValueError("model_type must be bge or dpr")

    optimizer = torch.optim.AdamW(model.parameters_to_optimize(), lr=args.lr)
    router = DynamicRouter(
        RouterConfig(
            temperature=args.temperature,
            ema_decay=args.ema_decay,
            alpha=args.alpha,
            beta=args.beta,
            tau_safe=args.tau_safe,
            tau_amb=args.tau_amb,
            tau_susp=args.tau_susp,
            w_safe=args.w_safe,
            w_amb=args.w_amb,
            w_susp=args.w_susp,
            w_harm=args.w_harm,
            cooldown_steps=args.cooldown_steps,
        )
    )

    best_dev = -1.0
    best_state_dir = output_dir / "best"
    best_state_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        router.set_time(epoch)
        model.train_module.train()
        pbar = tqdm(train_dl, desc=f"epoch={epoch}")
        epoch_router_stats = []
        for batch in pbar:
            qids = [str(x) for x in batch["qids"]]
            queries = batch["queries"]
            positives = batch["positives"]
            neg_ids = batch["neg_ids"]
            negatives = batch["negatives"]
            B, K = len(queries), batch["K"]

            flat_negatives = [x for row in negatives for x in row]
            q_emb = model.encode_query(queries, device=device, max_length=args.q_maxlen)
            p_emb = model.encode_doc(positives, device=device, max_length=args.d_maxlen)
            n_emb = model.encode_doc(flat_negatives, device=device, max_length=args.d_maxlen).view(B, K, -1)

            with torch.no_grad():
                pos_scores = torch.sum(q_emb * p_emb, dim=-1)
                neg_scores = torch.einsum("bd,bkd->bk", q_emb, n_emb)
                router.update(qids, neg_ids, pos_scores, neg_scores)
                neg_weights, route_names, hvals = router.build_weights(qids, neg_ids, q_emb.device)

            loss, _, _ = weighted_infonce(
                q_emb, p_emb, n_emb, neg_weights,
                temperature=args.temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            flat_names = [x for row in route_names for x in row]
            total = max(len(flat_names), 1)

            safe_cnt = sum(x == "safe" for x in flat_names)
            amb_cnt = sum(x == "ambiguous" for x in flat_names)
            susp_cnt = sum(x == "suspicious" for x in flat_names)
            harm_cnt = sum(x == "harmful" for x in flat_names)

            h_mean = float(hvals.mean().item())
            h_min = float(hvals.min().item())
            h_max = float(hvals.max().item())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "safe": f"{safe_cnt/total:.2f}",
                "amb": f"{amb_cnt/total:.2f}",
                "susp": f"{susp_cnt/total:.2f}",
                "harm": f"{harm_cnt/total:.2f}",
                "hmean": f"{h_mean:.3f}",
                "hmin": f"{h_min:.3f}",
                "hmax": f"{h_max:.3f}",
            })
            epoch_router_stats.append({
                "safe_ratio": safe_cnt / total,
                "amb_ratio": amb_cnt / total,
                "susp_ratio": susp_cnt / total,
                "harm_ratio": harm_cnt / total,
                "h_mean": h_mean,
                "h_min": h_min,
                "h_max": h_max,
            })

        # optional refresh
        if args.dynamic_refresh:
            train_rows = refresh_train_rows(
                model=model,
                train_rows=train_rows,
                collection=collection,
                qrels_train=train_qrels,
                device=device,
                topk=args.refresh_topk,
                keep_negs=args.neg_per_query,
                q_maxlen=args.q_maxlen,
                d_maxlen=args.d_maxlen,
                batch_size=args.eval_batch_size,
            )
            train_ds.refresh_rows(train_rows)
            train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        # eval
        metrics_all = {}
        for split in ["dev", "test"]:
            q_path = dataset_dir / f"queries_{split}.jsonl"
            r_path = dataset_dir / f"qrels_{split}.json"
            if not q_path.exists() or not r_path.exists():
                continue

            queries = load_queries(q_path)
            qrels = load_qrels(r_path)
            results = retrieve(
                model=model,
                queries=queries,
                collection=collection,
                device=device,
                q_maxlen=args.q_maxlen,
                d_maxlen=args.d_maxlen,
                topk=args.eval_topk,
                batch_size=args.eval_batch_size,
            )
            metrics = evaluate_run(qrels, results, k=10)
            metrics_all[split] = metrics

            if epoch_router_stats:
                metrics_all["router"] = {
                    "safe_ratio": sum(x["safe_ratio"] for x in epoch_router_stats) / len(epoch_router_stats),
                    "amb_ratio": sum(x["amb_ratio"] for x in epoch_router_stats) / len(epoch_router_stats),
                    "susp_ratio": sum(x["susp_ratio"] for x in epoch_router_stats) / len(epoch_router_stats),
                    "harm_ratio": sum(x["harm_ratio"] for x in epoch_router_stats) / len(epoch_router_stats),
                    "h_mean": sum(x["h_mean"] for x in epoch_router_stats) / len(epoch_router_stats),
                    "h_min": min(x["h_min"] for x in epoch_router_stats),
                    "h_max": max(x["h_max"] for x in epoch_router_stats),
                }

            with (output_dir / f"run_{split}_epoch{epoch}.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

        with (output_dir / f"metrics_epoch{epoch}.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_all, f, ensure_ascii=False, indent=2)

        dev_score = metrics_all.get("dev", {}).get("nDCG@10", -1.0)
        if dev_score > best_dev:
            best_dev = dev_score
            model.save(best_state_dir)
            with (best_state_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    print(f"best dev nDCG@10 = {best_dev:.4f}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_type", choices=["bge", "dpr"], required=True)

    ap.add_argument("--model_name", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--q_model_name", default="facebook/dpr-question_encoder-single-nq-base")
    ap.add_argument("--d_model_name", default="facebook/dpr-ctx_encoder-single-nq-base")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--q_maxlen", type=int, default=64)
    ap.add_argument("--d_maxlen", type=int, default=180)
    ap.add_argument("--neg_per_query", type=int, default=8)
    ap.add_argument("--eval_topk", type=int, default=100)
    ap.add_argument("--refresh_topk", type=int, default=50)
    ap.add_argument("--dynamic_refresh", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ema_decay", type=float, default=0.9)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.2)

    ap.add_argument("--tau_safe", type=float, default=-0.05)
    ap.add_argument("--tau_amb", type=float, default=0.00)
    ap.add_argument("--tau_susp", type=float, default=0.05)

    ap.add_argument("--w_safe", type=float, default=1.0)
    ap.add_argument("--w_amb", type=float, default=0.5)
    ap.add_argument("--w_susp", type=float, default=0.1)
    ap.add_argument("--w_harm", type=float, default=0.0)

    ap.add_argument("--cooldown_steps", type=int, default=1)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
