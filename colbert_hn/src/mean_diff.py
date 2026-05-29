"""Mean-difference direction computation utilities.

Shared between `01_mean_diff` (raw v) and `01b_mean_diff_scaled` (α-sweep).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.colbert_hook import ColBERTv2
from src.data import doc_text
from src.hn_mining import Triplet, unique_dids
from src.utils.logging import get_logger

logger = get_logger(__name__)

HOOK_LAYER = 12


@torch.no_grad()
def encode_doc_layer12_means(
    model: ColBERTv2,
    corpus: Dict[str, dict],
    dids: List[str],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """For each did in `dids`, encode the doc and return the masked mean of
    the layer-12 hidden state — shape (len(dids), 768).

    Uses a capture hook at layer 12. The captured tensor is the *unmodified*
    layer-12 output (no steering applied; this is the source of truth for v
    computation)."""
    captured: List[torch.Tensor] = []

    def capture(h: torch.Tensor) -> torch.Tensor:
        captured.append(h.detach().to("cpu"))
        return h

    model.clear_hooks()
    model.register_layer_hook(HOOK_LAYER, capture)

    means: List[torch.Tensor] = []
    for start in tqdm(range(0, len(dids), batch_size), desc=f"encode_l{HOOK_LAYER}"):
        batch_dids = dids[start:start + batch_size]
        texts = [doc_text(corpus[d]) for d in batch_dids]
        captured.clear()
        _emb, score_mask = model.encode_docs(texts, device=device)
        if not captured:
            raise RuntimeError("hook did not fire")
        h12 = captured[0].to(torch.float32)
        mask = score_mask.to("cpu").bool()
        h_masked = h12 * mask.unsqueeze(-1)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h_masked.dtype)
        per_doc_mean = h_masked.sum(dim=1) / counts
        means.append(per_doc_mean)
    model.clear_hooks()
    return torch.cat(means, dim=0)


def compute_v(
    model: ColBERTv2,
    corpus: Dict[str, dict],
    triplets: List[Triplet],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, dict]:
    """Returns (v ∈ ℝ^768, stats dict)."""
    pos_dids, hn_dids = unique_dids(triplets)
    logger.info(
        "mining: %d triplets, %d unique pos, %d unique HN",
        len(triplets), len(pos_dids), len(hn_dids),
    )
    pos_means = encode_doc_layer12_means(model, corpus, pos_dids, device, batch_size)
    hn_means = encode_doc_layer12_means(model, corpus, hn_dids, device, batch_size)
    v_pos = pos_means.mean(dim=0)
    v_hn = hn_means.mean(dim=0)
    v = (v_hn - v_pos).to(torch.float32)
    stats = {
        "n_triplets": len(triplets),
        "n_unique_pos_docs": len(pos_dids),
        "n_unique_hn_docs": len(hn_dids),
        "v_norm": float(v.norm().item()),
        "v_mean_abs": float(v.abs().mean().item()),
        "hook_layer": HOOK_LAYER,
    }
    return v, stats
