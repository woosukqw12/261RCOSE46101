# 03_scalar_gate — direction + scalar gate at layer 12

## 목적

ROADMAP.md *single-direction 단계* 의 8 번째 실험. 02 의 학습된 $v$ 위에 **scalar gate** $g_\ell \in [0,1]$ 을 추가. 02 에서 관찰된 train-overfitting (‖v‖ 폭주, val NDCG 조기 plateau) 을 *bounded magnitude* 로 완화하고자 함.

수식:
$$\tilde{h}^{(12)} = h^{(12)} - g \cdot v, \quad g = \sigma(b), \quad v|_{t=0} = \mathbf{0}, \; b|_{t=0} = -3$$

학습 가능 파라미터: **769** (= 768 for $v$ + 1 for $b$).

Anchor preservation: t=0 에서 $g \approx 0.047$, $v=\mathbf{0}$ → 개입 = 0. v 가 크게 자라도 g 가 그 effective magnitude 를 통제.

## 가설

| 가설 | 기준 | 근거 |
|---|---|---|
| H08a | 08 > 02 (vs baseline 의 Δ NDCG@10 confused 가 더 큼) | gate 의 anchor preservation 효과로 overfitting 완화 → 더 큰 net improvement |
| H08b | 08 ≥ 01b α=10 anchor (CI 0 초과 또는 동등) | informed non-learned baseline 통과 — H5 학습 의미성 강화 |
| H08c | 학습 종료 시 $g \cdot \|v\|$ 가 01b 의 α=10 부근에 수렴 | 학습이 *학습된 magnitude* 를 적절히 찾아냄 |

## 학습 design (02 와 동일, gate 만 추가)

| 항목 | 값 |
|---|---|
| Loss | pairwise margin $m=0.2$ |
| Optimizer | AdamW (LR=$10^{-3}$, WD=$10^{-4}$) |
| λ_anc | 0 (DESIGN.md §11 deviation) |
| Batch / Epochs / Patience | 32 / 5 / 2 |
| Init | $v=\mathbf{0}$, $b=-3$ (즉 gate σ(b)≈0.047) |
| Hook layer | 12 |
| Dataset | SciFact |

## 실행

```bash
.venv/bin/python experiments/03_scalar_gate/run.py --dataset scifact --seed 42
```

Artifact: `outputs/03_scalar_gate/{dataset}/seed_{seed}/` — 02 와 동일 구조 + `gate_trace.json` (epoch 별 g 값) + 학습 종료 시 effective magnitude $g \cdot \|v\|$.

## 비판적 review

1. **Multiplicative parametrization 의 gradient 문제**: $\partial L / \partial v = g \cdot \partial L / \partial (gv)$. $g$ 가 작으면 (≈0.05) gradient 가 v 에 약하게 흐름 → 학습 늦어질 수 있음. ‖v‖ trace 으로 모니터링.
2. **g 와 ‖v‖ 가 *둘 중 하나만* 자라도 같은 g·v 가능**: 둘이 동시에 자라도 OK. 학습 history 에서 두 값의 trajectory 시각화 필요.
3. **02 의 best epoch (epoch 2) 가 g 없이 적정 magnitude 였음** → 08 도 비슷한 시점에 best 일 가능성. 학습 곡선 형태로 검증.

## 상세 보고서

[`report/03_scalar_gate_report.md`](../../report/03_scalar_gate_report.md) — 실행 후 작성.
