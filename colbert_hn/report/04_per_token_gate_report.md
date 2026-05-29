# 04_per_token_gate — direction + per-token gate (saturation result)

ROADMAP *single-direction 단계* 11 번째 실험. 08 (scalar gate) 의 실패 후 per-token gate $g(h_t) = \sigma(W h_t + b)$ 로 token-level selectivity 부여 시도.

## 1. 결과 요약

| Metric | 11 | 02 | 08 | 01b α=10 | baseline |
|---|---|---|---|---|---|
| NDCG@10 | **0.6641** | 0.6651 | 0.6448 | 0.6690 | 0.6464 |
| ‖v‖ 학습 종료 | 8.07 | 7.08 | 3.35 | 10.0 | — |
| ‖W‖ 학습 종료 | 2.72 | — | — | — | — |
| gate bias 학습 종료 | -2.84 | — | -2.65 (8 epoch) | — | — |
| **gate 평균 (test)** | **1.000** | — | 0.068 | — | — |
| **gate std (test)** | **0.001** | — | (scalar) | — | — |
| cos(v_learned, v_mean_diff) | 0.32 | 0.32 | 0.29 | — | — |

### Paired bootstrap CI

| Anchor | Slice | Δ NDCG@10 (95 % CI) | 유의 |
|---|---|---|---|
| baseline | all | +0.0177 [+0.0070, +0.0294] | ✓ positive |
| baseline | confused | +0.0403 [+0.0191, +0.0627] | ✓ positive |
| 02 (no gate) | all | -0.0009 [-0.0035, +0.0011] | — (0 포함) |
| 02 (no gate) | confused | -0.0034 [-0.0085, -0.0004] | marginally negative |
| 08 (scalar gate) | all | +0.0193 [+0.0084, +0.0313] | ✓ positive |
| 08 (scalar gate) | confused | +0.0438 [+0.0222, +0.0666] | ✓ positive |
| 01b α=10 | all | -0.0049 [-0.0177, +0.0073] | — (0 포함) |
| 01b α=10 | confused | -0.0241 [-0.0501, -0.0006] | marginally negative |

## 2. 결정적 진단 — gate saturation at 1

학습 종료 시점의 test set 의 per-token gate 분포:

```
mean: 1.000
std : 0.001
min : ~0.999
max : ~1.000
frac > 0.5: 100 %
frac > 0.1: 100 %
```

**Per-token gate 가 모든 token 에서 ≈ 1 로 수렴**. 즉 학습된 gate function $g(h_t)$ 가 *all tokens 에 대해 saturated* → 효과적으로 *gate 없는 02 와 동일*.

수치적 일관성:
- 11 vs 02 (no gate) 의 Δ ≈ 0 (CI 포함 0): 통계적 동등.
- 11 vs 08 의 Δ > 0 명확: 11 의 gate 가 *효과적* gate=1 → 02 수준 성능 회복.

## 3. 메커니즘 — 왜 gate saturated 됐나

학습 dynamics 분석:
- $b$ init = -3, $W \sim \mathcal{N}(0, 0.02^2)$ (작은 시작) → gate $\approx \sigma(-3) \approx 0.047$ 으로 *anchor preservation* 시작.
- Pairwise margin loss 가 $v$ 의 magnitude 를 키우려 함. $g \cdot v$ 가 효과적이려면 gate 가 자라야 함.
- $W$ 가 hidden state 의 어떤 방향과 정렬되면 $W \cdot h$ 가 *모든 token 에서* 큰 양수 → gate ≈ 1.
- Sigmoid 의 gradient $\sigma'(z) = \sigma(z)(1-\sigma(z))$ 이 saturation 시 0 → 추가 학습 정지.
- **결과**: gate 가 fully open 상태로 수렴, 사실상 02 의 *no gate* 와 등가.

**Anchor preservation 의 *학습된 부재***: 02 의 all-slice Δ 가 양수 (+0.019) → 학습된 v 가 anchor 를 손상시키지 *않음* → gate 가 anchor preserve 할 *필요 없음* → gate 가 closed 될 motivation 없음 → 학습된 gate = always-on.

## 4. 본 프로젝트 narrative 측면 함의

### single-direction 단계 의 핵심 발견 (02-11)

| Step | 형식 | NDCG@10 | 결과 |
|---|---|---|---|
| baseline | $h$ (no intervention) | 0.6464 | — |
| 02 | $h - v$ (learned, no gate) | **0.6651** | baseline 통과 |
| 08 | $h - g v$ (learned + scalar gate) | 0.6448 | gradient saturation at 0 → 실패 |
| 11 | $h - g(h) v$ (learned + per-token gate) | 0.6641 | gate saturation at 1 → 02 와 동등 |

**single-direction 단계 (single direction at single layer 12) 의 ceiling 도달**: ≈ 0.665. 01b 의 α=10 (0.6690) 가 sharpened anchor 인데 02/11 모두 이 anchor 와 *통계적 동등* (CI 0 포함).

### 단일 direction 의 한계

세 학습된 single-direction 변형 (02, 08, 11) 의 결과를 종합:
1. 학습된 $v$ 는 mean-diff $v$ 와 *qualitatively 다른* (cos ≈ 0.3) 방향임에도 *retrieval 성능은 동등*. 즉 **single direction subspace 내에 *서로 다른 방향들* 이 비슷한 효과** → 단일 direction 의 *redundancy* + *capacity 부족*.
2. Gate (scalar / per-token) 의 도입이 *학습된 lever* 가 되지 못함 — anchor 손상이 없을 때 gate 는 학습 의무가 없음.

→ ROADMAP 의 다음 lever 선택:
- **13_five_layers** (multi-layer): 단일 layer 의 capacity 한계를 layer 수로 보완. 각 layer 가 *다른 측면* 의 confusion 을 잡을 수 있는지 검정.
- **17_projection_out** (alternative form): subtract vs orthogonal projection — form 자체의 효과 비교 (LEACE family).
- **multi-direction 단계 multi-direction router**: 본 paper 의 *main novelty*. Single direction 의 redundancy 를 multiple direction + routing 으로 해결.

## 5. ROADMAP 영향

| 항목 | 영향 |
|---|---|
| 11 narrative | "per-token gate 의 비활성 학습" — gate 형식의 한계를 데이터로 입증 |
| 12_gate_capacity | 우선순위 ↓ — gate 형식 자체가 약 |
| 13_five_layers | 우선순위 ↑ — single-layer 의 ceiling 우회 시도 |
| 17_projection_out | 우선순위 ↑ — alternative form 비교 |
| multi-direction 단계 (multi-direction router) | **우선순위 매우 ↑** — 본 결과들이 multi-direction 의 *empirical 필요성* 강력히 시사. paper main contribution 으로 더욱 sharpening. |
| ROADMAP changelog | "single-direction 단계 의 gate 실험 (08, 11) 결과 — single-direction-at-single-layer 의 ceiling 0.66x; multi-direction (multi-direction 단계) 의 priority 격상." |

## 6. Artifact 위치

```
outputs/04_per_token_gate/scifact/seed_42/
├── config / env / train_config.json
├── module_final.pt (v, W, b)
├── train_history.json (‖v‖, ‖W‖, b trace)
├── cosine_with_mean_diff.json
├── gate_distribution.json (mean=1.0, std=0.001 ← saturation)
├── runs / runs_scored / metrics_*
└── delta_vs_{baseline, mean_diff_alpha10, 02_learned, 03_scalar_gate}.json
```

Figures 는 후속 통합 single-direction 단계 보고서 (TBD) 에서 02 / 08 / 11 비교 charts 로 통합 제시.
