# 04_per_token_gate — direction + *per-token* gate at layer 12

## 목적

ROADMAP *single-direction 단계* 의 11 번째 실험. 08 (scalar gate) 의 multiplicative gradient saturation 한계를 극복하고자 gate 를 **per-token 으로 확장**:

$$\tilde{h}_t = h_t - g(h_t) \cdot v, \quad g(h_t) = \sigma(W h_t + b)$$

여기서:
- $v \in \mathbb{R}^{768}$ (single direction, 02 / 08 와 동일 형식)
- $W \in \mathbb{R}^{1 \times 768}$ (gate logit linear)
- $b \in \mathbb{R}$ (gate bias)
- $g(h_t) \in [0, 1]$ (per-token gate value)

**핵심 차이점 vs 08**: token 마다 gate 값이 다름. 어떤 token 에서는 gate ≈ 1 (강한 개입), 어떤 token 에서는 gate ≈ 0 (개입 없음) — 학습이 *어디에 개입할지* 를 token-level 로 결정. 01b Figure 6 의 query-heterogeneous 효과 + 02 의 single-direction redundancy 모두 *per-token selectivity 의 필요성* 시사.

학습 가능 파라미터: **1537** = 768 (v) + 768 (W) + 1 (b).

## 가설

| 가설 | 기준 | 근거 |
|---|---|---|
| H11a | 11 > 08 (의미 있게 개선) | per-token gate 가 08 의 gradient saturation 해소 |
| H11b | 11 ≥ 02 (CI 0 초과 또는 동등) | 02 의 redundancy 를 per-token selectivity 가 해결 |
| H11c | 11 ≥ 01b α=10 anchor | informed non-learned 통과 — 본 paper 의 *학습된 representation-level intervention* 의 가치 입증 |

## 학습 design

| 항목 | 값 |
|---|---|
| Loss | pairwise margin $m=0.2$ |
| Optimizer | AdamW (LR=$10^{-3}$, WD=$10^{-4}$) |
| λ_anc | 0 (DESIGN.md §11) |
| Init | $v=\mathbf{0}$, $W \sim \mathcal{N}(0, 0.02^2)$, $b=-3$ |
| Batch / Epochs / Patience | 32 / 5 / 2 |
| Hook layer | 12 |
| Dataset | SciFact |

## 실행

```bash
.venv/bin/python experiments/04_per_token_gate/run.py --dataset scifact --seed 42
```

Artifact: `outputs/04_per_token_gate/{dataset}/seed_{seed}/` — 02 / 08 와 동일 구조 + gate 분포 통계 (mean / std / quantile of per-token gate values on test set).

## 비판적 review

1. **02 가 이미 strong anchor (α=10 과 동등)** — 11 이 02 를 *유의하게* 능가해야 본 architectural step 의 가치 입증. 그렇지 않으면 단일 direction 의 representation 한계가 본질.
2. **Gate 가 모든 token 에 동일한 값 (∼ scalar) 으로 수렴할 위험**: training data 가 token 의 *어디에* 개입할 가치를 distinguish 할 만큼 정보 있는가? Gate value 의 *분산* (per-token spread) 으로 진단.
3. **gate_bias_init = -3 의 효과 변화**: 08 의 scalar 와 달리 W 가 학습되어 token-specific signal 을 amplify 할 수 있어 saturation 위험 작음. 다만 b=-3 의 보수적 시작은 동일.
4. **잠재적 over-fitting**: 768 → 1537 parameters 증가 (2x). SciFact 의 9,190 triplets 가 충분한가? Val curve monitoring.

## 상세 보고서

[`report/04_per_token_gate_report.md`](../../report/04_per_token_gate_report.md) — 실행 후 작성.
