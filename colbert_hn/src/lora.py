"""LoRA (Low-Rank Adaptation) wrappers for ColBERT v2 encoder finetune.

ROADMAP §"Stage 3" 의 main novelty — frozen-encoder limit (02 unfrozen 의 Δ
conf +0.252) 의 *50 K param budget 안* 회복.

핵심 원리 (Hu et al. 2021):
    Linear(W ∈ ℝ^{d_out × d_in}) → Linear + LoRA :
        y = W x + (α/r) B A x ,   A ∈ ℝ^{r × d_in}, B ∈ ℝ^{d_out × r}
    Init: A ~ N(0, σ²), B = 0 → ΔW = 0 at t=0 → *exactly baseline retrieval*.

수식 (학습 가능):
    A ∈ ℝ^{r × d}: down-projection (r << d)
    B ∈ ℝ^{d × r}: up-projection
    학습 파라미터 per Linear = 2 r d

For BERT-base (d = 768, 12 layers, 4 attn linears each):
    q, k, v, o each: 2 r × 768 = 1,536 r params/Linear
    q+v on all 12 layers, r=1: 36,864 params (50 K 안)
    q+k+v+o on 6 layers, r=2: 73,728 params (50 K 초과)
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a rank-r additive adapter.

    Forward: y = base(x) + scaling * (x @ A^T @ B^T)
        base(x): frozen original linear (parameters set to requires_grad=False)
        A ∈ ℝ^{r × in_features}: down-proj, init small_random.
        B ∈ ℝ^{out_features × r}: up-proj, init zero → BA = 0 at t=0.

    scaling = alpha / r (per Hu et al. 2021 convention). Default alpha = r ⇒
    scaling = 1 (no rescale).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 1,
        alpha: Optional[float] = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_linear)}")
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        in_d = base_linear.in_features
        out_d = base_linear.out_features
        self.alpha = float(alpha if alpha is not None else r)
        self.scaling = self.alpha / self.r
        # A ~ N(0, init_std²), B = 0  →  ΔW = BA = 0 at t=0
        self.A = nn.Parameter(torch.randn(self.r, in_d) * init_std)
        self.B = nn.Parameter(torch.zeros(out_d, self.r))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # x: (..., in_d). LoRA path: x @ A^T → (..., r) → @ B^T → (..., out_d)
        lora_out = torch.matmul(torch.matmul(x, self.A.T), self.B.T) * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.r}, "
            f"α={self.alpha}, ‖A‖={self.A.norm().item():.4f}, "
            f"‖B‖={self.B.norm().item():.4f}"
        )

    def lora_parameters(self) -> List[nn.Parameter]:
        return [self.A, self.B]


# ============================================================================
# BERT-specific LoRA injection
# ============================================================================
# ColBERT v2 의 BERT-base 위 attention 의 q / k / v / o 4 개 Linear 에 LoRA 부착.


_LAYER_LINEAR_PATHS = {
    "q": ("attention", "self", "query"),
    "k": ("attention", "self", "key"),
    "v": ("attention", "self", "value"),
    "o": ("attention", "output", "dense"),
}


def inject_lora_into_bert(
    bert_model,
    target_components: List[str],
    layers: Optional[List[int]] = None,
    r: int = 1,
    alpha: Optional[float] = None,
    init_std: float = 0.02,
) -> List[nn.Parameter]:
    """Replace specified attention linears with LoRA-wrapped versions in-place.

    Args:
        bert_model: HuggingFace BERT model (`model.bert` of ColBERTv2).
        target_components: subset of {'q','k','v','o'}, e.g. ['q','v'].
        layers: subset of layer indices (0..11); None = all 12 layers.
        r: LoRA rank.
        alpha: LoRA scaling (default r ⇒ scaling=1).
        init_std: stdev for A init (B=0).

    Returns:
        List of LoRA parameters (A, B for each injected adapter) for the
        optimizer.
    """
    valid = {"q", "k", "v", "o"}
    for c in target_components:
        if c not in valid:
            raise ValueError(f"unknown target component: {c!r}")
    if layers is None:
        layers = list(range(len(bert_model.encoder.layer)))
    params: List[nn.Parameter] = []
    for li in layers:
        bert_layer = bert_model.encoder.layer[li]
        for comp in target_components:
            path = _LAYER_LINEAR_PATHS[comp]
            obj = bert_layer
            for attr in path[:-1]:
                obj = getattr(obj, attr)
            last_attr = path[-1]
            base_lin = getattr(obj, last_attr)
            wrapped = LoRALinear(base_lin, r=r, alpha=alpha, init_std=init_std)
            setattr(obj, last_attr, wrapped)
            params.extend(wrapped.lora_parameters())
    return params


def lora_param_count(
    target_components: List[str],
    n_layers: int,
    hidden_dim: int = 768,
    r: int = 1,
) -> int:
    """Total LoRA params for given config."""
    per_linear = 2 * r * hidden_dim
    return per_linear * len(target_components) * n_layers
