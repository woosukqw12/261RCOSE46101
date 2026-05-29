"""Bilinear interaction metric for MaxSim correction.

ROADMAP §"Stage 2" 의 main novelty. Translation family ($\\tilde h = h - u(h)$)
의 algebraic ceiling 을 *우회* 하는 minimal form 변경. Hidden state 자체는
frozen ColBERT 의 출력 그대로 두고, *MaxSim 의 inner product 형식* 을
$\\langle q_i, d_j \\rangle \\to q_i^\\top M d_j$ 로 일반화.

$$M = I + U V^\\top, \\quad U, V \\in \\mathbb{R}^{D \\times r}$$

Anchor preservation: $U = V = \\mathbf{0}$ init → $M = I$ → baseline 과 정확히
동일. 학습이 진행되며 $UV^\\top$ 의 *q-d cross-feature* 가 활성화.

핵심 차이 (translation family vs bilinear):
- Translation: $\\tilde h_t = h_t - u(h_t)$ → MaxSim 에서 $-q_i^\\top u(d_j)$
  항이 *d 의 함수만* → query-conditional re-ranking 정보 표현 불가.
- Bilinear: $q_i^\\top M d_j = \\langle q_i, d_j \\rangle + (U^\\top q_i)^\\top
  (V^\\top d_j)$ — *q 와 d 의 cross-feature 의 곱셈적 결합* → q-d 상호작용
  차원의 새 expressivity.

학습 파라미터: $2 D r$. ColBERT 의 projected dim $D = 128$ 기준:
- r=8: 2,048
- r=16: 4,096
- r=32: 8,192
- r=64: 16,384

모두 50K 상한 (CLAUDE.md §3.2) 의 ≪ 50% 안.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

InitMode = Literal["zero", "small_random"]


class BilinearMetric(nn.Module):
    """Low-rank bilinear correction of MaxSim. M = I + U V^T.

    Forward 형식 (token-level maxsim):
        s_M(q, d) = sum_i max_j (q_i^T d_j + (U^T q_i)^T (V^T d_j))

    효율적 계산: 두 항 분리 후 element-wise 합. (B_q, B_d, T_q, T_d) 텐서를
    중간에 만드는 점은 vanilla MaxSim 과 동일 — bilinear 부분은 동일 shape 의
    추가 텐서만 더해짐.

    Anchor: U=V=0 init → 모든 token-pair 의 bilinear 항 = 0 → 정확히 baseline.
    """

    def __init__(
        self,
        dim: int = 128,
        r: int = 8,
        init: InitMode = "zero",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.r = int(r)
        if init == "zero":
            u_init = torch.zeros(self.dim, self.r)
            v_init = torch.zeros(self.dim, self.r)
        elif init == "small_random":
            u_init = torch.randn(self.dim, self.r) * 1e-2
            v_init = torch.randn(self.dim, self.r) * 1e-2
        else:
            raise ValueError(f"unknown init mode: {init!r}")
        self.U = nn.Parameter(u_init)
        self.V = nn.Parameter(v_init)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, r={self.r}, "
            f"‖U‖={self.U.norm().item():.4f}, ‖V‖={self.V.norm().item():.4f}"
        )

    def num_params(self) -> int:
        return self.U.numel() + self.V.numel()

    # ---------------------------------------------------------------- maxsim

    def maxsim(
        self,
        q_emb: torch.Tensor,
        d_emb: torch.Tensor,
        d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise bilinear MaxSim. Shapes match `ColBERTv2.maxsim`.

        Returns: (B_q, B_d).
        """
        # vanilla similarity: (B_q, B_d, T_q, T_d)
        sim_iden = torch.einsum("qid,kjd->qkij", q_emb, d_emb)
        # bilinear correction
        Uq = torch.einsum("qid,dr->qir", q_emb, self.U.to(q_emb.dtype))   # (B_q, T_q, r)
        Vd = torch.einsum("kjd,dr->kjr", d_emb, self.V.to(d_emb.dtype))   # (B_d, T_d, r)
        sim_bilin = torch.einsum("qir,kjr->qkij", Uq, Vd)
        sim = sim_iden + sim_bilin

        d_mask_b = d_mask.unsqueeze(0).unsqueeze(2)  # (1, B_d, 1, T_d)
        sim = sim.masked_fill(~d_mask_b, float("-inf"))
        maxed = sim.max(dim=-1).values  # (B_q, B_d, T_q)
        maxed = torch.where(torch.isinf(maxed), torch.zeros_like(maxed), maxed)
        return maxed.sum(dim=-1)  # (B_q, B_d)

    def diagonal_maxsim(
        self,
        q_emb: torch.Tensor,
        d_emb: torch.Tensor,
        d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row bilinear MaxSim. Shapes match `ColBERTv2.diagonal_maxsim`.

        Returns: (B,).
        """
        sim_iden = torch.einsum("bid,bjd->bij", q_emb, d_emb)
        Uq = torch.einsum("bid,dr->bir", q_emb, self.U.to(q_emb.dtype))
        Vd = torch.einsum("bjd,dr->bjr", d_emb, self.V.to(d_emb.dtype))
        sim_bilin = torch.einsum("bir,bjr->bij", Uq, Vd)
        sim = sim_iden + sim_bilin

        sim = sim.masked_fill(~d_mask.unsqueeze(1), float("-inf"))
        maxed = sim.max(dim=-1).values  # (B, T_q)
        maxed = torch.where(torch.isinf(maxed), torch.zeros_like(maxed), maxed)
        return maxed.sum(dim=-1)  # (B,)
