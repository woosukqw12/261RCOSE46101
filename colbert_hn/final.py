"""
final.py
======================

모델개입팀 (anchor regularizer + LoRA) ⨯ 데이터팀 (HN 전처리) 합동 실험을 위한
단일 standalone 스크립트. `src/` 의존성 없이 본 파일 하나만으로 모든 코드 (
ColBERT v2 wrapper, LoRA injection, anchor 손실, 학습 루프, 평가 + paired
bootstrap) 가 inline 되어 있어, 다른 팀이 그대로 본인 codebase 에 붙여넣고
실행할 수 있다.

>>> 데이터팀과의 통합 지점 (3 가지 방법 중 선택) <<<

    1) 가장 간단 — JSON 으로 triplet 사전 mining 후 전달:
       데이터팀이 본인 method 로 만든 triplet 리스트를 다음 형식의 JSON 으로 저장:

           [
               {"qid": "Q1", "pos_did": "D5", "hn_did": "D9"},
               {"qid": "Q1", "pos_did": "D5", "hn_did": "D17"},
               ...
           ]

       그리고 본 스크립트 실행:

           python final.py --dataset scifact --seed 42 \\
               --triplets-json path/to/data_team_triplets.json

    2) Python API 로 직접 호출 — 함수 import:

           from final import run
           triplets = my_data_team_method(...)  # List[Tuple[str, str, str]]
           result = run(triplets, dataset="scifact", seed=42)

    3) 데이터팀 mining 함수 inline 삽입 — `data_team_mine_triplets()` 함수를
       본 파일 안에서 직접 교체. 함수의 입력 (frozen ColBERT 의 top-100 ranked
       list, qrels) 과 출력 (List[(qid, pos_did, hn_did)]) signature 만 지키면 됨.

>>> 모델개입팀 best 방법론 (그대로 inline) <<<

    - Frozen ColBERT v2 (encoder + projection 모두 frozen)
    - LoRA(r=8, α=r) on q, v projections of all 12 transformer layers
        → 학습 파라미터 ≈ 295K (전체 110M 중 0.27 %)
    - Hard/easy query partition: frozen top-1 이 relevant 이면 easy
    - Hard query 측: 표준 pairwise margin loss
    - Easy query 측: per-token cosine anchor R_abs^q + R_abs^{d+}
        L = L_margin(Q_hard) + λ_dir · (R_abs^q + R_abs^{d+})
        λ_dir = 1.0 (단일 값, sweep 없음)
    - Optimizer: AdamW, LR 5e-5, weight decay 1e-4
    - Batch 32, 3 epochs, patience-2 early stop on val NDCG@10 (10 % held-out)
    - 3 seeds × 1 config × paired bootstrap (10 k iter) 95 % CI

본 코드 그대로 돌리면 모델개입팀 단독 best 결과 (Δ all = +0.030 ± 0.002,
3/3 strict net+) 를 재현한다.

>>> 의존성 <<<
    torch, transformers, numpy, tqdm, beir (BEIR loader 만), safetensors,
    huggingface_hub
    GPU/MPS 권장 (CPU 도 동작하나 매우 느림).

>>> 실행 예시 <<<

    # (모델개입팀 단독) reproduce — 자체 default mining 사용:
    python final.py --dataset scifact --seed 42

    # (합동 실험) 데이터팀 triplet 사용:
    python final.py --dataset scifact --seed 42 \\
        --triplets-json data_team/triplets_clean.json \\
        --output-dir outputs/combined/scifact_seed42

    # 3-seed 일괄:
    for seed in 42 1337 2024; do
        python final.py --dataset scifact --seed $seed \\
            --triplets-json data_team/triplets_clean.json \\
            --output-dir outputs/combined/scifact_seed$seed
    done
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# =============================================================================
#  Section 0 — Configuration
# =============================================================================

@dataclass
class Config:
    """All hyperparameters. Defaults reproduce the model-team best configuration."""

    # Dataset
    dataset: str = "scifact"           # BEIR dataset name
    data_root: str = "data"             # BEIR cache directory

    # ColBERT v2
    encoder_id: str = "colbert-ir/colbertv2.0"
    query_max_len: int = 32
    doc_max_len: int = 180
    embedding_dim: int = 128
    mask_punctuation: bool = True

    # LoRA
    lora_r: int = 8
    lora_alpha: float = 8.0             # = r (scaling = 1)
    lora_targets: Tuple[str, ...] = ("query", "value")  # BERT attention projections
    lora_init_std: float = 0.02

    # Anchor regularizer (model-team best)
    lambda_dir: float = 1.0             # weight on R_abs^q + R_abs^{d+}

    # Optimization
    lr: float = 5e-5
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 3
    patience: int = 2
    margin: float = 0.2
    val_split: float = 0.1

    # Triplet stream (only used by default miner; data-team JSON overrides)
    max_triplets: int = 9190
    hn_pool: int = 100
    n_hns_per_q: int = 10

    # Evaluation
    eval_doc_batch: int = 64
    eval_query_batch: int = 16
    eval_top_k: int = 100
    bootstrap_iter: int = 10_000
    bootstrap_ci: float = 0.95

    # Reproducibility / I/O
    seed: int = 42
    device: Optional[str] = None        # auto: mps → cuda → cpu


# =============================================================================
#  Section 1 — Utilities (seed, device, I/O)
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: Optional[str] = None) -> torch.device:
    if name is not None:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


# =============================================================================
#  Section 2 — BEIR loader
# =============================================================================

def load_beir(name: str, split: str, data_root: str = "data"
              ) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load (corpus, queries, qrels). Auto-downloads dataset if not present.
    Returns:
        corpus[did]  = {"title": str, "text": str}
        queries[qid] = str
        qrels[qid][did] = int (relevance ≥ 1 means relevant)
    """
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader
    root = Path(data_root); root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if not (target.exists() and any(target.iterdir())):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        beir_util.download_and_unzip(url, str(root))
    return GenericDataLoader(data_folder=str(target)).load(split=split)


def doc_text(d: dict) -> str:
    """Concatenate title + body with single space (ColBERT v2 BEIR convention)."""
    title = (d.get("title") or "").strip()
    body = (d.get("text") or "").strip()
    return f"{title} {body}".strip() if title else body


# =============================================================================
#  Section 3 — Frozen ColBERT v2 wrapper
# =============================================================================

class ColBERTv2(nn.Module):
    """Frozen ColBERT v2 = BERT-base + 768 → 128 linear projection + L2 norm.

    Loads the official `colbert-ir/colbertv2.0` checkpoint.  Both `bert.*` and
    the `linear.weight` projection are loaded from the safetensors file.
    All parameters set to requires_grad_(False) by default.
    """

    NUM_LAYERS = 12

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_id)
        self.bert = AutoModel.from_pretrained(cfg.encoder_id)
        hidden = self.bert.config.hidden_size  # 768
        self.linear = nn.Linear(hidden, cfg.embedding_dim, bias=False)

        # `linear.weight` lives in the ColBERT v2 safetensors but is NOT loaded
        # by AutoModel — we must copy it in explicitly or retrieval is random.
        state = self._load_full_state(cfg.encoder_id)
        with torch.no_grad():
            self.linear.weight.copy_(state["linear.weight"])

        # Query / doc markers — ColBERT v2 reuses [unused0]/[unused1]
        self.q_marker_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.d_marker_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.mask_id = self.tokenizer.mask_token_id
        self.pad_id = self.tokenizer.pad_token_id

        # Punctuation mask (applied to doc score positions only)
        if cfg.mask_punctuation:
            punct_ids: Set[int] = set()
            for ch in string.punctuation:
                for i in self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ch)):
                    if i != self.tokenizer.unk_token_id:
                        punct_ids.add(i)
            self.register_buffer("punctuation_ids",
                                  torch.tensor(sorted(punct_ids), dtype=torch.long),
                                  persistent=False)
        else:
            self.register_buffer("punctuation_ids",
                                  torch.empty(0, dtype=torch.long), persistent=False)

        # Freeze everything by default
        for p in self.parameters():
            p.requires_grad_(False)
        self.bert.eval()
        self.linear.eval()

    @staticmethod
    def _load_full_state(model_id: str) -> Dict[str, torch.Tensor]:
        p = Path(model_id)
        if p.exists() and (p / "model.safetensors").exists():
            path = p / "model.safetensors"
        else:
            path = Path(hf_hub_download(model_id, "model.safetensors"))
        return load_file(str(path))

    # ----- tokenization with [Q] / [D] markers ------------------------------

    def _inject_marker(self, ids: torch.Tensor, attn: torch.Tensor, marker_id: int
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = ids.size(0)
        marker = torch.full((B, 1), marker_id, dtype=ids.dtype, device=ids.device)
        ids = torch.cat([ids[:, :1], marker, ids[:, 1:]], dim=1)
        attn_marker = torch.ones((B, 1), dtype=attn.dtype, device=attn.device)
        attn = torch.cat([attn[:, :1], attn_marker, attn[:, 1:]], dim=1)
        return ids, attn

    def _prep_query(self, queries: List[str]):
        enc = self.tokenizer(queries, padding="max_length", truncation=True,
                             max_length=self.cfg.query_max_len - 1, return_tensors="pt")
        ids, attn = self._inject_marker(enc["input_ids"], enc["attention_mask"], self.q_marker_id)
        # ColBERT v2 query expansion: pad tokens replaced by [MASK]; BERT does NOT
        # attend to them, but MaxSim DOES use them.
        pad_mask = ids == self.pad_id
        ids = ids.masked_fill(pad_mask, self.mask_id)
        score_mask = torch.ones_like(attn, dtype=torch.bool)
        return ids, attn, score_mask

    def _prep_doc(self, docs: List[str]):
        enc = self.tokenizer(docs, padding="longest", truncation=True,
                             max_length=self.cfg.doc_max_len - 1, return_tensors="pt")
        ids, attn = self._inject_marker(enc["input_ids"], enc["attention_mask"], self.d_marker_id)
        return ids, attn, attn.bool()

    def _encode(self, ids, attn, score_mask, is_doc: bool):
        out = self.bert(input_ids=ids, attention_mask=attn).last_hidden_state
        emb = F.normalize(self.linear(out), p=2, dim=-1)
        mask = score_mask.bool()
        if is_doc and self.punctuation_ids.numel() > 0:
            mask = mask & ~torch.isin(ids, self.punctuation_ids)
        emb = emb * mask.unsqueeze(-1).to(emb.dtype)
        return emb, mask

    def encode_queries(self, queries: List[str], device: torch.device):
        ids, attn, sm = self._prep_query(queries)
        return self._encode(ids.to(device), attn.to(device), sm.to(device), is_doc=False)

    def encode_docs(self, docs: List[str], device: torch.device):
        ids, attn, sm = self._prep_doc(docs)
        return self._encode(ids.to(device), attn.to(device), sm.to(device), is_doc=True)

    # ----- MaxSim scoring ---------------------------------------------------

    @staticmethod
    def maxsim(q_emb: torch.Tensor, d_emb: torch.Tensor, d_mask: torch.Tensor
               ) -> torch.Tensor:
        """Pairwise MaxSim. Shapes:
            q_emb : (B_q, T_q, D), d_emb : (B_d, T_d, D), d_mask : (B_d, T_d)
        Returns (B_q, B_d).
        """
        sim = torch.einsum("qid,kjd->qkij", q_emb, d_emb)
        sim = sim.masked_fill(~d_mask[None, :, None, :], float("-inf"))
        maxed = sim.max(dim=-1).values
        maxed = torch.where(torch.isinf(maxed), torch.zeros_like(maxed), maxed)
        return maxed.sum(dim=-1)

    @staticmethod
    def diagonal_maxsim(q_emb, d_emb, d_mask):
        """Per-row MaxSim s_i = Σ_t max_j ⟨q[i,t], d[i,j]⟩. Used during training."""
        sim = torch.einsum("bid,bjd->bij", q_emb, d_emb)
        sim = sim.masked_fill(~d_mask.unsqueeze(1), float("-inf"))
        maxed = sim.max(dim=-1).values
        maxed = torch.where(torch.isinf(maxed), torch.zeros_like(maxed), maxed)
        return maxed.sum(dim=-1)


# =============================================================================
#  Section 4 — LoRA injection
# =============================================================================

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with rank-r additive adapter.
    y = base(x) + (α/r) · x A^T B^T
    A ~ N(0, σ²), B = 0  ⇒  BA = 0 at step 0 ⇒  identical to frozen baseline.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, init_std: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * init_std)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

    def lora_parameters(self) -> List[nn.Parameter]:
        return [self.A, self.B]


def inject_lora(bert, target_components: Tuple[str, ...], r: int, alpha: float,
                init_std: float) -> List[nn.Parameter]:
    """Replace BERT attention.self.{query|key|value} with LoRA-wrapped versions.
    Returns the list of trainable LoRA parameters."""
    params: List[nn.Parameter] = []
    for layer in bert.encoder.layer:
        for name in target_components:
            sub = layer.attention.self
            base = getattr(sub, name)
            wrapped = LoRALinear(base, r=r, alpha=alpha, init_std=init_std)
            setattr(sub, name, wrapped)
            params.extend(wrapped.lora_parameters())
    return params


def lora_param_count(target_components: Tuple[str, ...], n_layers: int = 12,
                     hidden_dim: int = 768, r: int = 8) -> int:
    return 2 * r * hidden_dim * len(target_components) * n_layers


# =============================================================================
#  Section 5 — Default hard-negative miner (for data-team override)
# =============================================================================

Triplet = Tuple[str, str, str]


def default_triplet_miner(model: ColBERTv2, queries: Dict[str, str],
                          corpus: Dict[str, dict],
                          qrels: Dict[str, Dict[str, int]],
                          cfg: Config, device: torch.device
                          ) -> List[Triplet]:
    """모델개입팀의 default mining (자체 baseline).

    프로세스:
      1) frozen ColBERT v2 로 전체 corpus 에 대해 top-K (default 100) 검색
      2) 각 query 마다 (pos × hn) cross product 로 triplet 생성
         pos: qrels 의 relevance ≥ 1 인 doc
         hn:  top-K 중 qrels positive 아닌 doc (n=10 sampled)
      3) cfg.max_triplets 로 subsample

    데이터팀: 본 함수를 본인 method 로 교체하면 됩니다. Signature 만 동일하면
    학습 코드 변경 없이 그대로 작동합니다.
    """
    print(f"  [default miner] encoding {len(corpus)} train docs ...")
    dids, d_emb, d_mask = encode_corpus(model, corpus, device, batch_size=cfg.eval_doc_batch)
    print(f"  [default miner] scoring {len(queries)} train queries (top-{cfg.hn_pool}) ...")
    runs = score_queries(model, queries, dids, d_emb, d_mask, device,
                         query_batch=cfg.eval_query_batch, top_k=cfg.hn_pool)
    runs_ranked = {q: [d for d, _ in lst] for q, lst in runs.items()}
    triplets: List[Triplet] = []
    for qid, ranked in runs_ranked.items():
        rels = qrels.get(qid, {})
        positives = [d for d, r in rels.items() if r >= 1]
        if not positives:
            continue
        rel_set = set(positives)
        hns = [d for d in ranked[:cfg.hn_pool] if d not in rel_set][:cfg.n_hns_per_q]
        if not hns:
            continue
        for p in positives:
            for h in hns:
                triplets.append((qid, p, h))
    print(f"  [default miner] mined {len(triplets)} triplets")
    if cfg.max_triplets and len(triplets) > cfg.max_triplets:
        rng = random.Random(cfg.seed)
        rng.shuffle(triplets)
        triplets = triplets[:cfg.max_triplets]
        print(f"  [default miner] subsampled to {len(triplets)} (seed={cfg.seed})")
    return triplets


def load_triplets_from_json(path: Path) -> List[Triplet]:
    """Load triplets from a JSON file.

    JSON 형식:
        [{"qid": "Q1", "pos_did": "D5", "hn_did": "D9"}, ...]
    """
    raw = json.loads(Path(path).read_text())
    triplets: List[Triplet] = []
    for row in raw:
        triplets.append((row["qid"], row["pos_did"], row["hn_did"]))
    print(f"  [data-team JSON] loaded {len(triplets)} triplets from {path}")
    return triplets


# =============================================================================
#  Section 6 — Encode corpus / score queries (used for both mining and eval)
# =============================================================================

@torch.no_grad()
def encode_corpus(model: ColBERTv2, corpus: Dict[str, dict], device: torch.device,
                  batch_size: int = 64) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    dids = list(corpus.keys())
    texts = [doc_text(corpus[d]) for d in dids]
    all_emb, all_mask = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode_docs", leave=False):
        chunk = texts[i:i + batch_size]
        emb, mask = model.encode_docs(chunk, device=device)
        all_emb.append(emb.cpu()); all_mask.append(mask.cpu())
    # pad to common doc length
    T_max = max(e.shape[1] for e in all_emb)
    padded = []
    masks = []
    for emb, m in zip(all_emb, all_mask):
        if emb.shape[1] < T_max:
            pad = torch.zeros(emb.shape[0], T_max - emb.shape[1], emb.shape[2])
            pad_m = torch.zeros(m.shape[0], T_max - m.shape[1], dtype=torch.bool)
            emb = torch.cat([emb, pad], dim=1)
            m = torch.cat([m, pad_m], dim=1)
        padded.append(emb); masks.append(m)
    return dids, torch.cat(padded, dim=0), torch.cat(masks, dim=0)


@torch.no_grad()
def score_queries(model: ColBERTv2, queries: Dict[str, str], dids: List[str],
                  d_emb: torch.Tensor, d_mask: torch.Tensor, device: torch.device,
                  query_batch: int = 16, doc_chunk: int = 512, top_k: int = 100
                  ) -> Dict[str, List[Tuple[str, float]]]:
    qids = list(queries.keys())
    q_texts = [queries[q] for q in qids]
    out: Dict[str, List[Tuple[str, float]]] = {}
    n_q = len(qids)
    n_d = d_emb.shape[0]
    for qi in tqdm(range(0, n_q, query_batch), desc="score_queries", leave=False):
        q_chunk_qids = qids[qi:qi + query_batch]
        q_chunk_texts = q_texts[qi:qi + query_batch]
        q_emb, _ = model.encode_queries(q_chunk_texts, device=device)
        q_emb = q_emb.cpu()
        scores = torch.empty(len(q_chunk_qids), n_d)
        for di in range(0, n_d, doc_chunk):
            d_chunk_emb = d_emb[di:di + doc_chunk]
            d_chunk_mask = d_mask[di:di + doc_chunk]
            s = ColBERTv2.maxsim(q_emb, d_chunk_emb, d_chunk_mask)
            scores[:, di:di + s.shape[1]] = s
        topv, topi = scores.topk(min(top_k, n_d), dim=1)
        for k, q in enumerate(q_chunk_qids):
            out[q] = [(dids[int(topi[k, j])], float(topv[k, j])) for j in range(topv.shape[1])]
    return out


# =============================================================================
#  Section 7 — Hard/easy partition
# =============================================================================

def confused_slice(runs: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]],
                   k: int = 1) -> Set[str]:
    """A query is 'confused' (= hard) if its top-k frozen results contain no
    relevant doc.  This is the slice that the margin loss targets.
    """
    out: Set[str] = set()
    for qid, ranked in runs.items():
        rels = qrels.get(qid, {})
        if not any(rels.get(d, 0) >= 1 for d in ranked[:k]):
            out.add(qid)
    return out


# =============================================================================
#  Section 8 — Per-token cosine anchor (model-team contribution)
# =============================================================================

def precompute_frozen_embeddings(model: ColBERTv2, easy_qids: Set[str],
                                  triplets: List[Triplet],
                                  queries: Dict[str, str],
                                  corpus: Dict[str, dict],
                                  device: torch.device
                                  ) -> Dict[Tuple[str, str], Dict[str, torch.Tensor]]:
    """Cache frozen-encoder outputs for (qid, pos_did) pairs in easy queries.
    Used by the anchor loss during training (before LoRA is injected).
    """
    pairs: Set[Tuple[str, str]] = set()
    for qid, pos_did, _ in triplets:
        if qid in easy_qids:
            pairs.add((qid, pos_did))
    cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
    model.bert.eval(); model.linear.eval()
    with torch.no_grad():
        for qid, pos_did in tqdm(pairs, desc="precompute_frozen", leave=False):
            q_emb, _ = model.encode_queries([queries[qid]], device=device)
            d_emb, d_mask = model.encode_docs([doc_text(corpus[pos_did])], device=device)
            T_d_valid = int(d_mask[0].sum().item())
            cache[(qid, pos_did)] = {
                "q_emb": q_emb[0].cpu(),
                "d_emb": d_emb[0, :T_d_valid].cpu(),
            }
    return cache


def cosine_anchor_loss(q_emb_batch, pos_emb_batch, pos_mask_batch,
                         batch_qids, batch_pos_dids, easy_indices,
                         frozen_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]],
                         device: torch.device) -> torch.Tensor:
    """R_abs^q + R_abs^{d+} on easy queries.

    For each easy-query triplet, compute the per-token cosine deviation between
    the LoRA-adapted and the frozen encoder, on both query and positive doc tokens.
    Embeddings are already L2-normalized → cos = dot product.
    """
    if not easy_indices:
        return torch.zeros((), device=device)
    losses: List[torch.Tensor] = []
    for i in easy_indices:
        key = (batch_qids[i], batch_pos_dids[i])
        if key not in frozen_cache:
            continue
        ref = frozen_cache[key]
        H_q_frozen = ref["q_emb"].to(device)
        H_d_frozen = ref["d_emb"].to(device)
        H_q_lora = q_emb_batch[i]
        T_d_valid = int(pos_mask_batch[i].sum().item())
        H_d_lora = pos_emb_batch[i, :T_d_valid]
        cos_q = (H_q_lora * H_q_frozen).sum(dim=-1)
        cos_d = (H_d_lora * H_d_frozen).sum(dim=-1)
        losses.append((1.0 - cos_q).mean() + (1.0 - cos_d).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


# =============================================================================
#  Section 9 — Training loop
# =============================================================================

@dataclass
class TrainHistory:
    losses: List[float] = field(default_factory=list)
    rank_losses: List[float] = field(default_factory=list)
    anchor_losses: List[float] = field(default_factory=list)
    val_epochs: List[int] = field(default_factory=list)
    val_ndcg_all: List[float] = field(default_factory=list)
    val_ndcg_confused: List[float] = field(default_factory=list)


def train_with_anchor(model: ColBERTv2, lora_params: List[nn.Parameter],
                       triplets: List[Triplet], queries: Dict[str, str],
                       corpus: Dict[str, dict],
                       qrels: Dict[str, Dict[str, int]],
                       confused_qids: Set[str], easy_qids: Set[str],
                       frozen_cache,
                       cfg: Config, device: torch.device) -> TrainHistory:
    """Joint training: margin loss on hard triplets + anchor on easy triplets.

    Returns training history; the model's LoRA weights are restored to the
    best-validation snapshot at exit.
    """
    optim = torch.optim.AdamW([{"params": lora_params,
                                 "lr": cfg.lr, "weight_decay": cfg.weight_decay}])

    # Held-out validation set (10 % of queries)
    by_q: Dict[str, list] = {}
    for t in triplets:
        by_q.setdefault(t[0], []).append(t)
    qids_sorted = sorted(by_q.keys())
    rng = random.Random(cfg.seed); rng.shuffle(qids_sorted)
    n_val = max(1, int(len(qids_sorted) * cfg.val_split))
    val_qids = set(qids_sorted[:n_val])
    train_subset = [t for t in triplets if t[0] not in val_qids]

    print(f"  [train] {len(train_subset)} triplets after val held-out (val_qids={len(val_qids)})")
    n_hard = sum(1 for t in train_subset if t[0] in confused_qids)
    n_easy = sum(1 for t in train_subset if t[0] in easy_qids)
    print(f"  [train] in train: hard={n_hard}, easy={n_easy}; λ_dir={cfg.lambda_dir}")

    history = TrainHistory()
    best_val = -math.inf
    best_state: Optional[List[torch.Tensor]] = None
    epochs_since_best = 0
    step = 0

    for epoch in range(cfg.epochs):
        model.bert.train(); model.linear.train()
        rng_e = random.Random(cfg.seed * 100003 + epoch)
        shuffled = list(train_subset); rng_e.shuffle(shuffled)
        batches = [shuffled[i:i + cfg.batch_size]
                    for i in range(0, len(shuffled), cfg.batch_size)]
        t0 = time.time()
        ep_rank = ep_anchor = 0.0
        n_rank = n_anchor = 0
        for batch in tqdm(batches, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False):
            qids = [t[0] for t in batch]
            pos_dids = [t[1] for t in batch]
            hn_dids = [t[2] for t in batch]
            q_texts = [queries[q] for q in qids]
            pos_texts = [doc_text(corpus[d]) for d in pos_dids]

            q_emb, _ = model.encode_queries(q_texts, device=device)
            pos_emb, pos_mask = model.encode_docs(pos_texts, device=device)

            confused_idx = [i for i, q in enumerate(qids) if q in confused_qids]
            easy_idx = [i for i, q in enumerate(qids) if q in easy_qids]

            if confused_idx:
                hn_texts = [doc_text(corpus[hn_dids[i]]) for i in confused_idx]
                hn_emb, hn_mask = model.encode_docs(hn_texts, device=device)
                q_emb_c = q_emb[confused_idx]
                pos_emb_c = pos_emb[confused_idx]
                pos_mask_c = pos_mask[confused_idx]
                s_pos = ColBERTv2.diagonal_maxsim(q_emb_c, pos_emb_c, pos_mask_c)
                s_hn = ColBERTv2.diagonal_maxsim(q_emb_c, hn_emb, hn_mask)
                rank_loss = torch.clamp(cfg.margin - s_pos + s_hn, min=0).mean()
                ep_rank += float(rank_loss.detach().item()); n_rank += 1
            else:
                rank_loss = torch.zeros((), device=device)

            if easy_idx and cfg.lambda_dir > 0:
                anchor_loss = cosine_anchor_loss(
                    q_emb, pos_emb, pos_mask, qids, pos_dids, easy_idx,
                    frozen_cache, device)
                ep_anchor += float(anchor_loss.detach().item()); n_anchor += 1
            else:
                anchor_loss = torch.zeros((), device=device)

            loss = rank_loss + cfg.lambda_dir * anchor_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            step += 1
            history.losses.append(float(loss.detach().item()))
            history.rank_losses.append(float(rank_loss.detach().item()))
            history.anchor_losses.append(float(anchor_loss.detach().item()))

        print(f"  [epoch {epoch+1}/{cfg.epochs}] "
              f"rank={ep_rank/max(1,n_rank):.4f} anchor={ep_anchor/max(1,n_anchor):.4f} "
              f"time={time.time()-t0:.1f}s")

        # Validation
        model.bert.eval(); model.linear.eval()
        sub_queries = {q: queries[q] for q in val_qids if q in queries}
        sub_qrels = {q: qrels[q] for q in val_qids if q in qrels}
        val_all, val_conf = _val_eval(model, sub_queries, corpus, sub_qrels, cfg, device)
        history.val_epochs.append(epoch + 1)
        history.val_ndcg_all.append(val_all); history.val_ndcg_confused.append(val_conf)
        print(f"  [epoch {epoch+1}/{cfg.epochs}] val NDCG@10: all={val_all:.4f} confused={val_conf:.4f}")

        score = val_all if not math.isnan(val_all) else val_conf
        if score > best_val:
            best_val = score
            best_state = [p.detach().clone().cpu() for p in lora_params]
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                print(f"  [early stop] epoch {epoch+1} (patience {cfg.patience})")
                break

    if best_state is not None:
        for p, b in zip(lora_params, best_state):
            p.data.copy_(b.to(p.device))
        print(f"  [restore] best LoRA state (val={best_val:.4f})")
    return history


def _val_eval(model: ColBERTv2, queries: Dict[str, str], corpus: Dict[str, dict],
              qrels: Dict[str, Dict[str, int]], cfg: Config, device: torch.device
              ) -> Tuple[float, float]:
    if not queries:
        return float("nan"), float("nan")
    with torch.no_grad():
        dids, d_emb, d_mask = encode_corpus(model, corpus, device,
                                              batch_size=cfg.eval_doc_batch)
        runs = score_queries(model, queries, dids, d_emb, d_mask, device,
                              query_batch=cfg.eval_query_batch, top_k=cfg.eval_top_k)
    ranked = {q: [d for d, _ in lst] for q, lst in runs.items()}
    per_q = {q: ndcg_at_k(ranked[q], qrels.get(q, {}), 10) for q in ranked}
    if not per_q:
        return float("nan"), float("nan")
    all_avg = float(np.mean(list(per_q.values())))
    conf = confused_slice(ranked, qrels, k=1)
    if not conf:
        return all_avg, float("nan")
    conf_avg = float(np.mean([per_q[q] for q in conf if q in per_q]))
    return all_avg, conf_avg


# =============================================================================
#  Section 10 — Evaluation + paired bootstrap CI
# =============================================================================

def ndcg_at_k(retrieved: List[str], qrels_q: Dict[str, int], k: int) -> float:
    if not retrieved:
        return 0.0
    rel_pos = np.array([int(qrels_q.get(d, 0)) for d in retrieved[:k]], dtype=float)
    if rel_pos.size == 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, rel_pos.size + 2))
    dcg = float((rel_pos * disc).sum())
    ideal = sorted((int(v) for v in qrels_q.values()), reverse=True)[:k]
    ideal = np.array(ideal, dtype=float)
    if ideal.size == 0:
        return 0.0
    idisc = 1.0 / np.log2(np.arange(2, ideal.size + 2))
    idcg = float((ideal * idisc).sum())
    return dcg / idcg if idcg > 0 else 0.0


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_iter: int = 10000,
                         ci: float = 0.95, seed: int = 42
                         ) -> Tuple[float, float, float]:
    diffs = a - b
    n = diffs.size
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    means = diffs[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(diffs.mean()), float(lo), float(hi)


def evaluate_and_compare(model: ColBERTv2, test_corpus: Dict[str, dict],
                          test_queries: Dict[str, str],
                          test_qrels: Dict[str, Dict[str, int]],
                          baseline_per_q: Dict[str, float],
                          baseline_runs: Dict[str, List[str]],
                          cfg: Config, device: torch.device) -> Dict:
    """Run trained model on test set, slice by frozen-baseline hard/easy, paired
    bootstrap CI vs baseline_per_q. Returns dict of summary statistics."""
    print("  [eval] encoding test corpus ...")
    dids, d_emb, d_mask = encode_corpus(model, test_corpus, device,
                                          batch_size=cfg.eval_doc_batch)
    print(f"  [eval] scoring {len(test_queries)} test queries ...")
    runs = score_queries(model, test_queries, dids, d_emb, d_mask, device,
                          query_batch=cfg.eval_query_batch, top_k=cfg.eval_top_k)
    ranked = {q: [d for d, _ in lst] for q, lst in runs.items()}
    per_q = {q: ndcg_at_k(ranked[q], test_qrels.get(q, {}), 10) for q in ranked}

    # Slice by frozen-baseline behaviour
    confused = confused_slice(baseline_runs, test_qrels, k=1)
    easy = set(per_q.keys()) - confused

    deltas: Dict[str, dict] = {}
    for slc_name, slc in [("all", set(per_q.keys())), ("confused", confused), ("easy", easy)]:
        common = sorted(slc & per_q.keys() & baseline_per_q.keys())
        if not common:
            deltas[slc_name] = {"n": 0, "skipped": "empty"}
            continue
        ours = np.array([per_q[q] for q in common])
        base = np.array([baseline_per_q[q] for q in common])
        mean, lo, hi = paired_bootstrap_ci(ours, base, n_iter=cfg.bootstrap_iter,
                                             ci=cfg.bootstrap_ci, seed=cfg.seed)
        deltas[slc_name] = {
            "n": len(common), "mean_delta_ndcg10": mean,
            "ci_lo": lo, "ci_hi": hi,
            "positive": lo > 0, "negative": hi < 0,
        }

    aggregate = {
        "ndcg_cut_10_all": float(np.mean(list(per_q.values()))),
        "ndcg_cut_10_confused": (float(np.mean([per_q[q] for q in confused if q in per_q]))
                                  if confused else None),
        "ndcg_cut_10_easy": (float(np.mean([per_q[q] for q in easy if q in per_q]))
                              if easy else None),
    }
    return {"delta_vs_baseline": deltas, "metrics_aggregate": aggregate,
            "per_query_ndcg10": per_q, "runs_ranked": ranked}


def compute_frozen_baseline(model: ColBERTv2, test_corpus, test_queries, test_qrels,
                            cfg: Config, device: torch.device
                            ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """Run the *frozen* (= pre-LoRA-injection) model on the test set to get the
    paired baseline per-query NDCG@10 + ranked runs. Must be called BEFORE
    inject_lora()."""
    dids, d_emb, d_mask = encode_corpus(model, test_corpus, device,
                                          batch_size=cfg.eval_doc_batch)
    runs = score_queries(model, test_queries, dids, d_emb, d_mask, device,
                          query_batch=cfg.eval_query_batch, top_k=cfg.eval_top_k)
    ranked = {q: [d for d, _ in lst] for q, lst in runs.items()}
    per_q = {q: ndcg_at_k(ranked[q], test_qrels.get(q, {}), 10) for q in ranked}
    return per_q, ranked


# =============================================================================
#  Section 11 — High-level entry point (programmatic API)
# =============================================================================

def run(
    triplets: Optional[List[Triplet]] = None,
    dataset: str = "scifact",
    seed: int = 42,
    output_dir: Optional[Path] = None,
    cfg_overrides: Optional[Dict] = None,
) -> Dict:
    """Train ColBERT v2 + LoRA + anchor on the given triplets, then evaluate.

    Args:
        triplets: Pre-mined triplets. If None, the default miner is used to
            reproduce the model-team baseline. 데이터팀: 본인 method 의 출력을
            여기에 넘기면 됩니다.
        dataset: BEIR dataset (scifact / nfcorpus / fiqa).
        seed: Random seed.
        output_dir: If given, save all artifacts (per-query metrics, deltas,
            LoRA state) here.
        cfg_overrides: Dict of Config field overrides (e.g. {"lr": 1e-4}).

    Returns:
        Dict with keys 'delta_vs_baseline', 'metrics_aggregate', etc.
    """
    cfg = Config(dataset=dataset, seed=seed)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"[final] dataset={cfg.dataset} seed={cfg.seed} device={device}")
    print(f"[final] λ_dir={cfg.lambda_dir} lora_r={cfg.lora_r} epochs={cfg.epochs}")

    # 1. Load data
    train_corpus, train_queries, train_qrels = load_beir(cfg.dataset, split="train",
                                                          data_root=cfg.data_root)
    test_corpus, test_queries, test_qrels = load_beir(cfg.dataset, split="test",
                                                        data_root=cfg.data_root)
    print(f"[data] train: corpus={len(train_corpus)} queries={len(train_queries)}")
    print(f"[data] test:  corpus={len(test_corpus)} queries={len(test_queries)}")

    # 2. Build frozen ColBERT
    print("[model] loading ColBERT v2 ...")
    model = ColBERTv2(cfg).to(device)

    # 3. Frozen baseline on test set (paired comparison anchor)
    print("[baseline] computing frozen baseline on test set ...")
    baseline_per_q, baseline_runs = compute_frozen_baseline(
        model, test_corpus, test_queries, test_qrels, cfg, device)

    # 4. Train-side: identify hard / easy queries from frozen behaviour
    print("[partition] computing frozen runs on TRAIN side ...")
    train_dids, td_emb, td_mask = encode_corpus(model, train_corpus, device,
                                                  batch_size=cfg.eval_doc_batch)
    train_runs = score_queries(model, train_queries, train_dids, td_emb, td_mask,
                                 device, query_batch=cfg.eval_query_batch,
                                 top_k=cfg.hn_pool)
    train_runs_ranked = {q: [d for d, _ in lst] for q, lst in train_runs.items()}
    confused_train = confused_slice(train_runs_ranked, train_qrels, k=1)
    easy_train = set(train_runs_ranked.keys()) - confused_train
    print(f"[partition] hard={len(confused_train)} easy={len(easy_train)}")
    del td_emb, td_mask  # free memory

    # 5. Triplets — either supplied by data team or default miner
    if triplets is None:
        print("[triplets] no external triplets supplied → using default miner")
        triplets = default_triplet_miner(model, train_queries, train_corpus,
                                           train_qrels, cfg, device)
    else:
        print(f"[triplets] using externally-supplied triplets: {len(triplets)}")
        if cfg.max_triplets and len(triplets) > cfg.max_triplets:
            rng = random.Random(cfg.seed); rng.shuffle(triplets)
            triplets = triplets[:cfg.max_triplets]
            print(f"[triplets] subsampled to {len(triplets)} (seed={cfg.seed})")

    # 6. Cache frozen embeddings for easy queries (for the anchor loss)
    print("[anchor] caching frozen embeddings for easy-query pairs ...")
    frozen_cache = precompute_frozen_embeddings(
        model, easy_train, triplets, train_queries, train_corpus, device)

    # 7. Inject LoRA (AFTER frozen baselines + caches are recorded)
    expected = lora_param_count(cfg.lora_targets, n_layers=12,
                                  hidden_dim=768, r=cfg.lora_r)
    lora_params = inject_lora(model.bert, cfg.lora_targets, r=cfg.lora_r,
                                alpha=cfg.lora_alpha, init_std=cfg.lora_init_std)
    actual = sum(p.numel() for p in lora_params)
    assert actual == expected, (actual, expected)
    print(f"[lora] injected: {actual} trainable params ({100*actual/110_000_000:.2f}% of 110M)")
    model.to(device)

    # 8. Train
    print("[train] starting ...")
    t0 = time.time()
    history = train_with_anchor(model, lora_params, triplets, train_queries,
                                  train_corpus, train_qrels, confused_train,
                                  easy_train, frozen_cache, cfg, device)
    print(f"[train] done in {time.time()-t0:.1f}s")

    # 9. Test eval with paired-bootstrap CI against frozen baseline
    print("[eval] running test eval + paired bootstrap ...")
    result = evaluate_and_compare(model, test_corpus, test_queries, test_qrels,
                                    baseline_per_q, baseline_runs, cfg, device)
    deltas = result["delta_vs_baseline"]

    print("\n========= results =========")
    for slc, d in deltas.items():
        if "mean_delta_ndcg10" in d:
            sign = ("✓ positive" if d["positive"]
                    else "✗ negative" if d["negative"]
                    else "(CI contains 0)")
            print(f"  Δ {slc:9s}  n={d['n']:4d}  "
                  f"mean={d['mean_delta_ndcg10']:+.4f}  "
                  f"CI=[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  {sign}")

    # 10. Save artifacts
    if output_dir is not None:
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        save_json({"config": dataclasses.asdict(cfg)}, output_dir / "config.json")
        save_json(result["delta_vs_baseline"], output_dir / "delta_vs_baseline.json")
        save_json(result["metrics_aggregate"], output_dir / "metrics_aggregate.json")
        save_json(result["per_query_ndcg10"], output_dir / "per_query_ndcg10.json")
        save_json(result["runs_ranked"], output_dir / "runs.json")
        save_json(dataclasses.asdict(history), output_dir / "train_history.json")
        torch.save(
            {f"adapter_{i}": p.detach().cpu() for i, p in enumerate(lora_params)},
            output_dir / "lora_state.pt")
        print(f"[save] artifacts → {output_dir}")

    return result


# =============================================================================
#  Section 12 — CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="scifact",
                    choices=("scifact", "nfcorpus", "fiqa"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--triplets-json", type=str, default=None,
                    help="Path to pre-mined triplets JSON (data-team method 적용 결과). "
                         "지정하지 않으면 default miner 사용.")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory for artifacts (delta, per-query metrics, LoRA state).")
    p.add_argument("--lambda-dir", type=float, default=1.0,
                    help="Anchor weight (default 1.0 = model-team best).")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-triplets", type=int, default=9190)
    p.add_argument("--device", type=str, default=None,
                    help="cpu / mps / cuda. Default = auto.")
    args = p.parse_args()

    triplets = (load_triplets_from_json(Path(args.triplets_json))
                 if args.triplets_json else None)

    cfg_overrides = {
        "lambda_dir": args.lambda_dir,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_triplets": args.max_triplets,
        "device": args.device,
    }
    out = Path(args.output_dir) if args.output_dir else None

    run(
        triplets=triplets,
        dataset=args.dataset,
        seed=args.seed,
        output_dir=out,
        cfg_overrides=cfg_overrides,
    )


if __name__ == "__main__":
    main()
