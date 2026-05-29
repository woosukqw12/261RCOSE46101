"""Frozen ColBERT v2 with layer-wise hook infrastructure.

This module is the foundation that every experiment — baseline and steered —
builds on. It loads `colbert-ir/colbertv2.0` (BERT-base + a 768→128 linear
projection), freezes all parameters, and exposes `register_layer_hook` so that
a `SteeringModule` (see `src/lsr.py`, future) can be plugged in at any of the
canonical layer indices ℓ ∈ {0, 3, 6, 9, 12} (DESIGN.md §3.1).

Hook contract
-------------
A hook callable receives the post-layer hidden state `h_ℓ` of shape
`(batch, T, 768)` and returns its replacement `h_ℓ̃` of the same shape. The
caller (SteeringModule) is responsible for any gating / direction-vector
arithmetic per DESIGN.md §3.2.

Layer indexing convention
-------------------------
- `ℓ = 0`  → output of the embedding module (token + position, post-LN, pre
            first transformer layer).
- `ℓ = k`  for `k ∈ {1, …, 12}` → output of the k-th transformer layer.

Scoring
-------
ColBERT MaxSim on L2-normalised 128-d token embeddings:
    s(q, d) = Σ_i max_j ⟨E_q[i], E_d[j]⟩
with [Q]/[D] markers prepended (queries padded with [MASK] to a fixed length
per the original ColBERT v2 design; document punctuation tokens masked out).

Baseline reproduction
---------------------
The first run of this wrapper on SciFact / NFCorpus / SciDocs must reproduce
ColBERT v2's paper-reported NDCG to within ±0.005 (DESIGN.md §8). If it does
not, the projection-layer or special-token loading from the HF checkpoint
needs investigation before any steering experiment is interpreted.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

HookFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ColBERTConfig:
    model_name: str = "colbert-ir/colbertv2.0"
    # ColBERT v2 reuses [unused0]/[unused1] as query/doc markers — never add new tokens.
    query_marker_token: str = "[unused0]"
    doc_marker_token: str = "[unused1]"
    query_max_len: int = 32
    doc_max_len: int = 180
    embedding_dim: int = 128
    mask_punctuation: bool = True
    similarity: str = "cosine"  # cosine (L2-normed dot) is ColBERT v2 default


def _load_colbert_state(model_name: str) -> Dict[str, torch.Tensor]:
    """Locate the ColBERT v2 state dict (safetensors) given an HF id or local path."""
    p = Path(model_name)
    if p.exists() and (p / "model.safetensors").exists():
        path = p / "model.safetensors"
    else:
        path = Path(hf_hub_download(model_name, "model.safetensors"))
    return load_file(str(path))


class ColBERTv2(nn.Module):
    """Frozen ColBERT v2 with hook-able layer outputs.

    All parameters are placed in `requires_grad_(False)`. The encoder runs in
    eval mode by default (DESIGN.md §3.4); flip via `set_encoder_train_mode`.
    """

    NUM_LAYERS = 12

    def __init__(self, cfg: Optional[ColBERTConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ColBERTConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.bert = AutoModel.from_pretrained(self.cfg.model_name)
        hidden = self.bert.config.hidden_size
        self.linear = nn.Linear(hidden, self.cfg.embedding_dim, bias=False)

        # The HF AutoModel only deserialises bert.* keys; the 768→128 projection
        # lives under `linear.weight` in the same checkpoint and must be loaded
        # explicitly. Without this, projection weights are random Gaussian and
        # retrieval reduces to noise.
        state = _load_colbert_state(self.cfg.model_name)
        with torch.no_grad():
            self.linear.weight.copy_(state["linear.weight"])

        self.q_marker_id = self.tokenizer.convert_tokens_to_ids(self.cfg.query_marker_token)
        self.d_marker_id = self.tokenizer.convert_tokens_to_ids(self.cfg.doc_marker_token)
        self.mask_id = self.tokenizer.mask_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id

        if self.cfg.mask_punctuation:
            punct_ids = set()
            for ch in string.punctuation:
                ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(ch)
                )
                for i in ids:
                    if i != self.tokenizer.unk_token_id:
                        punct_ids.add(i)
            self.register_buffer(
                "punctuation_ids",
                torch.tensor(sorted(punct_ids), dtype=torch.long),
                persistent=False,
            )
        else:
            self.register_buffer(
                "punctuation_ids", torch.empty(0, dtype=torch.long), persistent=False
            )

        for p in self.parameters():
            p.requires_grad_(False)
        self.bert.eval()
        self.linear.eval()

        self._hooks: Dict[int, HookFn] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._install_hooks()

    # ------------------------------------------------------------------ hooks

    def _install_hooks(self) -> None:
        def make_layer_hook(idx: int):
            def _h(_module, _inp, out):
                hook = self._hooks.get(idx)
                if hook is None:
                    return out
                if isinstance(out, tuple):
                    h = out[0]
                    h = hook(h)
                    return (h, *out[1:])
                return hook(out)
            return _h

        def emb_hook(_module, _inp, out):
            hook = self._hooks.get(0)
            if hook is None:
                return out
            return hook(out)

        self._handles.append(self.bert.embeddings.register_forward_hook(emb_hook))
        for i, layer in enumerate(self.bert.encoder.layer, start=1):
            self._handles.append(layer.register_forward_hook(make_layer_hook(i)))

    def register_layer_hook(self, layer_idx: int, hook: HookFn) -> None:
        """Plug `hook` at layer `ℓ ∈ {0, …, 12}` (see module docstring)."""
        if not 0 <= layer_idx <= self.NUM_LAYERS:
            raise ValueError(f"layer_idx must be in [0, {self.NUM_LAYERS}], got {layer_idx}")
        self._hooks[layer_idx] = hook

    def clear_hooks(self) -> None:
        self._hooks.clear()

    def set_encoder_train_mode(self, train: bool) -> None:
        """Toggle encoder train/eval — only T2A.15 ablation should flip this."""
        self.bert.train(train)

    # ------------------------------------------------------------------ encode

    def _inject_marker(
        self, ids: torch.Tensor, attn: torch.Tensor, marker_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Insert `marker_id` right after [CLS]. Inputs of shape (B, T-1) → (B, T)."""
        B = ids.size(0)
        marker = torch.full((B, 1), marker_id, dtype=ids.dtype, device=ids.device)
        ids_out = torch.cat([ids[:, :1], marker, ids[:, 1:]], dim=1)
        attn_marker = torch.ones((B, 1), dtype=attn.dtype, device=attn.device)
        attn_out = torch.cat([attn[:, :1], attn_marker, attn[:, 1:]], dim=1)
        return ids_out, attn_out

    def _prep_query_inputs(
        self, queries: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (ids, attn, score_mask).

        Format: `[CLS] [Q] tok_1 ... tok_N [SEP] [MASK] ... [MASK]` padded to
        `query_max_len`. Per ColBERT v2's `attend_to_mask_tokens=False`:
        `attn` is 0 at the [MASK]-expansion positions (BERT does not attend
        to them), while `score_mask` is 1 everywhere so MaxSim still uses them.
        """
        enc = self.tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.query_max_len - 1,
            return_tensors="pt",
        )
        ids, attn = self._inject_marker(enc["input_ids"], enc["attention_mask"], self.q_marker_id)
        pad_mask = ids == self.pad_id
        ids = ids.masked_fill(pad_mask, self.mask_id)
        score_mask = torch.ones_like(attn, dtype=torch.bool)
        return ids, attn, score_mask

    def _prep_doc_inputs(
        self, docs: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (ids, attn, score_mask).

        `score_mask = attn` (excludes pad); punctuation masking is applied
        downstream inside `_encode` after the punctuation-id buffer is checked.
        """
        enc = self.tokenizer(
            docs,
            padding="longest",
            truncation=True,
            max_length=self.cfg.doc_max_len - 1,
            return_tensors="pt",
        )
        ids, attn = self._inject_marker(enc["input_ids"], enc["attention_mask"], self.d_marker_id)
        return ids, attn, attn.bool()

    def _encode(
        self,
        ids: torch.Tensor,
        attn: torch.Tensor,
        score_mask: torch.Tensor,
        is_doc: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`attn` controls BERT attention; `score_mask` selects MaxSim-contributing
        positions. The two differ for queries: attend_to_mask_tokens=False means
        BERT does not attend to the [MASK]-padded positions, but those positions
        are still used in MaxSim as query expansion (ColBERT v2 convention)."""
        out = self.bert(input_ids=ids, attention_mask=attn).last_hidden_state
        emb = self.linear(out)
        emb = F.normalize(emb, p=2, dim=-1)
        mask = score_mask.bool()
        if is_doc and self.punctuation_ids.numel() > 0:
            punct_mask = ~torch.isin(ids, self.punctuation_ids)
            mask = mask & punct_mask
        emb = emb * mask.unsqueeze(-1).to(emb.dtype)
        return emb, mask

    def encode_queries(self, queries: List[str], device: Optional[torch.device] = None):
        """Note: callers are responsible for the surrounding `torch.no_grad()`
        context when only inference is needed. We do not wrap this method so
        that training-time callers (e.g. `experiments/02_*/run.py`) can let
        gradients flow through the steering hook back to `SteeringModule.v`."""
        ids, attn, score_mask = self._prep_query_inputs(queries)
        if device is not None:
            ids = ids.to(device)
            attn = attn.to(device)
            score_mask = score_mask.to(device)
        return self._encode(ids, attn, score_mask, is_doc=False)

    def encode_docs(self, docs: List[str], device: Optional[torch.device] = None):
        ids, attn, score_mask = self._prep_doc_inputs(docs)
        if device is not None:
            ids = ids.to(device)
            attn = attn.to(device)
            score_mask = score_mask.to(device)
        return self._encode(ids, attn, score_mask, is_doc=True)

    # ------------------------------------------------------------------ score

    @staticmethod
    def maxsim(
        q_emb: torch.Tensor,
        d_emb: torch.Tensor,
        d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """MaxSim score s(q, d) = Σ_i max_j ⟨q_i, d_j⟩.

        Shapes:
            q_emb : (B_q, T_q, D)
            d_emb : (B_d, T_d, D)
            d_mask: (B_d, T_d)
        Returns: (B_q, B_d) — pairwise MaxSim scores.
        """
        # (B_q, T_q, D) · (B_d, T_d, D) → (B_q, B_d, T_q, T_d)
        sim = torch.einsum("qid,kjd->qkij", q_emb, d_emb)
        d_mask_b = d_mask.unsqueeze(0).unsqueeze(2)  # (1, B_d, 1, T_d)
        sim = sim.masked_fill(~d_mask_b, float("-inf"))
        maxed = sim.max(dim=-1).values  # (B_q, B_d, T_q)
        # Replace any -inf rows (entire doc masked) by 0 to keep finite scores
        maxed = torch.where(torch.isinf(maxed), torch.zeros_like(maxed), maxed)
        return maxed.sum(dim=-1)  # (B_q, B_d)

    @staticmethod
    def diagonal_maxsim(
        q_emb: torch.Tensor,
        d_emb: torch.Tensor,
        d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row MaxSim: s_i = Σ_t max_j ⟨q_emb[i,t], d_emb[i,j]⟩.

        Used in training where each query is paired with exactly one doc.
        Shapes:
            q_emb : (B, T_q, D)
            d_emb : (B, T_d, D)
            d_mask: (B, T_d)
        Returns: (B,)
        """
        sim = torch.einsum("bid,bjd->bij", q_emb, d_emb)  # (B, T_q, T_d)
        sim = sim.masked_fill(~d_mask.unsqueeze(1), float("-inf"))
        maxed = sim.max(dim=-1).values  # (B, T_q)
        maxed = torch.where(torch.isinf(maxed), torch.zeros_like(maxed), maxed)
        return maxed.sum(dim=-1)  # (B,)
