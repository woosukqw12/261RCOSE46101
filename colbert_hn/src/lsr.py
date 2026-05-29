"""Layer-wise Steering Representation (LSR) module.

본 module 은 ROADMAP.md 의 single-direction 단계 (02_final_layer_vector 부터) 의 학습 가능
intervention 의 *최소* 형태. 단일 layer 에 적용되는 단일 direction vector v.
Gate / multi-direction / multi-layer 등 후속 확장은 본 module 의 *옆에* 새
class 로 추가 (single-variable principle, CLAUDE.md §3.8).

수식 (02 default):
    \\tilde{h}^{(ℓ)} = h^{(ℓ)} - v,   v ∈ ℝ^768,  v|_{t=0} = 0
"""
from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn

InitMode = Literal["zero", "small_random"]


class SteeringModule(nn.Module):
    """Single learnable direction vector applied at one layer.

    안된 형태로 broadcasting subtract: `h - self.v`. h shape `(B, T, D)`,
    `self.v` shape `(D,)` → 각 토큰에서 동일 v 가 차감된다.

    Anchor preservation init: v = 0 으로 시작 → t=0 에서 개입은 정확히 no-op,
    즉 baseline ColBERT 의 동작을 그대로 보존. 학습이 진행되며 pairwise margin
    loss 가 v 를 의미 있는 방향으로 움직임.
    """

    def __init__(self, hidden_dim: int = 768, init: InitMode = "zero") -> None:
        super().__init__()
        if init == "zero":
            v = torch.zeros(hidden_dim)
        elif init == "small_random":
            v = torch.randn(hidden_dim) * 1e-3
        else:
            raise ValueError(f"unknown init mode: {init!r}")
        self.v = nn.Parameter(v)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h - self.v.to(h.dtype)

    def hook_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Returns a closure compatible with `ColBERTv2.register_layer_hook`."""

        def _hook(h: torch.Tensor) -> torch.Tensor:
            return self.forward(h)

        return _hook

    def extra_repr(self) -> str:
        return f"hidden_dim={self.v.numel()}, ‖v‖={self.v.norm().item():.4f}"


class ScalarGatedSteeringModule(nn.Module):
    """Single learnable direction + single scalar gate.

    수식:  \\tilde{h} = h - σ(b) · v,   v ∈ ℝ^768,  b ∈ ℝ.

    Anchor preservation: b init = -3 → σ(b) ≈ 0.047, v init = 0 → 개입 ≈ 0.
    Gate σ(b) ∈ [0,1] 가 *모든* 토큰에 동일 적용 (per-layer scalar, not per-token).
    학습 파라미터: 768 + 1 = 769.
    """

    def __init__(
        self, hidden_dim: int = 768,
        init: InitMode = "zero",
        gate_bias_init: float = -3.0,
    ) -> None:
        super().__init__()
        if init == "zero":
            v = torch.zeros(hidden_dim)
        elif init == "small_random":
            v = torch.randn(hidden_dim) * 1e-3
        else:
            raise ValueError(f"unknown init mode: {init!r}")
        self.v = nn.Parameter(v)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_bias_init)))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h - self.gate * self.v.to(h.dtype)

    def hook_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _hook(h: torch.Tensor) -> torch.Tensor:
            return self.forward(h)

        return _hook

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.v.numel()}, "
            f"gate={self.gate.item():.4f}, ‖v‖={self.v.norm().item():.4f}"
        )


class PerTokenGatedSteeringModule(nn.Module):
    """Direction + per-token gate.

    수식:  \\tilde{h}_t = h_t - g(h_t) · v,   g(h_t) = σ(W h_t + b)

    Per-token gate: each token computes its own gate value from its own
    hidden state. This breaks the multiplicative saturation problem observed
    in `ScalarGatedSteeringModule` — for tokens where v is unhelpful, the
    learned W can amplify the gate; for tokens where v is harmful, the gate
    closes. Capacity: per-token bit of selectivity.

    Parameters:
      v ∈ ℝ^768                  (direction)
      W ∈ ℝ^{1 × 768}            (gate logit linear)
      b ∈ ℝ                      (gate bias)
    학습 파라미터: 768 + 768 + 1 = 1537.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        init: InitMode = "zero",
        gate_bias_init: float = -3.0,
        gate_weight_std: float = 0.02,
    ) -> None:
        super().__init__()
        if init == "zero":
            v = torch.zeros(hidden_dim)
        elif init == "small_random":
            v = torch.randn(hidden_dim) * 1e-3
        else:
            raise ValueError(f"unknown init mode: {init!r}")
        self.v = nn.Parameter(v)
        self.gate_weight = nn.Parameter(torch.randn(hidden_dim) * gate_weight_std)
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))

    def gate(self, h: torch.Tensor) -> torch.Tensor:
        """Returns gate ∈ [0,1] per token. Shape: (..., 1) (broadcastable)."""
        logits = (h * self.gate_weight.to(h.dtype)).sum(dim=-1, keepdim=True)
        return torch.sigmoid(logits + self.gate_bias.to(h.dtype))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        g = self.gate(h)  # (..., 1)
        return h - g * self.v.to(h.dtype)

    def hook_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _hook(h: torch.Tensor) -> torch.Tensor:
            return self.forward(h)

        return _hook

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.v.numel()}, "
            f"‖v‖={self.v.norm().item():.4f}, "
            f"‖W‖={self.gate_weight.norm().item():.4f}, "
            f"b={self.gate_bias.item():.4f}"
        )


class MultiLayerSteeringModule(nn.Module):
    """학습 가능 direction $v_\\ell \\in \\mathbb{R}^{768}$ 를 *다수의 layer* 에 동시 적용.

    수식 (각 layer ℓ ∈ \\mathcal{L} 에 독립적):
        \\tilde{h}^{(\\ell)} = h^{(\\ell)} - v_\\ell

    Gate 없음 (single-direction 단계 incremental: 02 의 multi-layer 확장). 각 $v_\\ell$ 는
    독립 학습. Default $\\mathcal{L} = \\{0, 3, 6, 9, 12\\}$ (ColBERT v2 의
    embedding + 4 transformer layer 의 마지막).

    학습 파라미터: $|\\mathcal{L}| \\times 768$. Default 5 × 768 = 3,840.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        layers: tuple[int, ...] = (0, 3, 6, 9, 12),
        init: InitMode = "zero",
    ) -> None:
        super().__init__()
        self.layers = tuple(layers)
        self.hidden_dim = hidden_dim
        for layer in self.layers:
            if init == "zero":
                v = torch.zeros(hidden_dim)
            elif init == "small_random":
                v = torch.randn(hidden_dim) * 1e-3
            else:
                raise ValueError(f"unknown init mode: {init!r}")
            # nn.Parameter 의 이름 규약 — getattr 로 접근
            self.register_parameter(f"v_l{layer}", nn.Parameter(v))

    def v(self, layer: int) -> torch.Tensor:
        """Returns the learnable direction at `layer`. Raises if `layer` not in
        `self.layers`."""
        if layer not in self.layers:
            raise KeyError(f"layer {layer} not in {self.layers}")
        return getattr(self, f"v_l{layer}")

    def hook_fn(self, layer: int) -> Callable[[torch.Tensor], torch.Tensor]:
        """Returns the per-layer closure compatible with
        `ColBERTv2.register_layer_hook(layer, ...)`."""

        def _hook(h: torch.Tensor) -> torch.Tensor:
            return h - self.v(layer).to(h.dtype)

        return _hook

    def register_all(self, colbert_model) -> None:
        """ColBERTv2 의 모든 target layer 에 본 module 의 hook 들을 일괄 등록."""
        colbert_model.clear_hooks()
        for layer in self.layers:
            colbert_model.register_layer_hook(layer, self.hook_fn(layer))

    def extra_repr(self) -> str:
        norms = ", ".join(
            f"ℓ{l}: ‖v‖={self.v(l).norm().item():.3f}" for l in self.layers
        )
        return f"layers={self.layers}, {norms}"


class MultiDirectionSteeringModule(nn.Module):
    """K learnable directions $v_k$ + per-token softmax router at one layer.

    수식 (single layer ℓ):
        \\tilde{h}_t = h_t - \\sum_{k=1}^{K} \\pi_k(h_t) \\cdot v_k
        \\pi(h_t) = \\softmax(W h_t + b) \\in \\Delta^K

    multi-direction 단계 main novelty. Single-direction subspace 의 capacity 한계를 *K 개*
    direction + per-token *routing* 으로 우회한다. 각 token 이 K 개 direction
    중 어느 mixture 로 보정될지 router 가 결정.

    Anchor preservation: $v_k|_{t=0} = \\mathbf{0}$, router bias=0 → 균등
    routing (π_k = 1/K), 그러나 모든 $v_k = 0$ 이므로 \\sum_k π_k v_k = 0 →
    개입 = 0.

    학습 파라미터: $K \\cdot D + K \\cdot D + K = 2 K D + K$.
    For K=2, D=768: 2 × 2 × 768 + 2 = 3,074.
    For K=4: 6,148.  For K=8: 12,296.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        n_directions: int = 2,
        init: InitMode = "zero",
        router_bias_init: float = 0.0,
        router_weight_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.K = int(n_directions)
        self.hidden_dim = hidden_dim
        if init == "zero":
            v_init = torch.zeros(self.K, hidden_dim)
        elif init == "small_random":
            v_init = torch.randn(self.K, hidden_dim) * 1e-3
        else:
            raise ValueError(f"unknown init mode: {init!r}")
        self.v = nn.Parameter(v_init)  # (K, D)
        self.router_weight = nn.Parameter(
            torch.randn(self.K, hidden_dim) * router_weight_std
        )  # (K, D)
        self.router_bias = nn.Parameter(
            torch.full((self.K,), float(router_bias_init))
        )  # (K,)

    def routing(self, h: torch.Tensor) -> torch.Tensor:
        """Returns routing distribution π ∈ ℝ^(..., K)."""
        logits = torch.einsum(
            "...d,kd->...k", h, self.router_weight.to(h.dtype)
        ) + self.router_bias.to(h.dtype)
        return torch.softmax(logits, dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pi = self.routing(h)  # (..., K)
        weighted_v = torch.einsum("...k,kd->...d", pi, self.v.to(h.dtype))
        return h - weighted_v

    def hook_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _hook(h: torch.Tensor) -> torch.Tensor:
            return self.forward(h)

        return _hook

    def extra_repr(self) -> str:
        norms = ", ".join(
            f"v_{k}:‖v‖={self.v[k].norm().item():.3f}" for k in range(self.K)
        )
        return (
            f"K={self.K}, hidden_dim={self.hidden_dim}, {norms}, "
            f"‖W‖={self.router_weight.norm().item():.3f}, "
            f"b={self.router_bias.detach().tolist()}"
        )
