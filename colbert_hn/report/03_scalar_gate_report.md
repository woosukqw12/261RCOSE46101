# 03_scalar_gate — direction + scalar gate (negative result)

본 보고서는 ROADMAP *single-direction 단계* 의 8 번째 실험. **결론 — scalar gate 의 multiplicative parametrization 이 학습 lever 가 되지 못함**.

## 1. 결과 요약

| Metric | 08 | 02 (이전 단계) | 01b α=10 | baseline |
|---|---|---|---|---|
| NDCG@10 | **0.6448** | 0.6651 | 0.6690 | 0.6464 |
| 학습 종료 g | 0.0683 | — | — | — |
| ‖v‖ 학습 종료 | 3.35 | 7.08 | 10.0 (unit·α) | — |
| effective g·‖v‖ | **0.23** | 7.08 | 10.0 | — |

### Paired bootstrap CI

| Anchor | Slice | Δ NDCG@10 (95% CI) | 유의 |
|---|---|---|---|
| baseline | all | -0.0016 [-0.0047, +0.0006] | — (0 포함) |
| baseline | confused | -0.0035 [-0.0102, +0.0013] | — (0 포함) |
| α=10 mean-diff | all | -0.0242 [-0.0416, -0.0081] | **negative** ✗ |
| α=10 mean-diff | confused | -0.0679 [-0.1025, -0.0371] | **negative** ✗ |
| 02 learned | all | -0.0202 [-0.0319, -0.0096] | **negative** ✗ |
| 02 learned | confused | -0.0471 [-0.0697, -0.0264] | **negative** ✗ |

**08 가 02 보다 통계적으로 유의하게 worse**. 모든 ablation 에서 *후퇴*.

![Delta CI forest](figures/03_scalar_gate/delta_ci_forest.png)

*Figure 1. 08 vs 3 anchor (baseline / α=10 mean-diff / 02 learned). vs baseline 은 통계적 동등, vs α=10 / 02 는 명확히 negative.*

## 2. 진단 — 무엇이 잘못됐나

![Effective magnitude](figures/03_scalar_gate/effective_magnitude.png)

*Figure 2. (왼쪽) 학습 종료 시점의 $g$, $\|v\|$, $g \cdot \|v\|$. (오른쪽) 08 의 effective magnitude (0.23) 가 01b 의 α sweep 위에서 어느 위치인지 — α=0.5 보다도 작음 → 01b 에서 *효과 없음* 으로 분류된 magnitude.*

**메커니즘 분석** — multiplicative parametrization $h - g \cdot v$ 의 gradient flow:

- $\partial L / \partial v = g \cdot \partial L / \partial (gv)$ — $g$ 가 작으면 $v$ 의 gradient 도 작음 (factor $g$ ≈ 0.07).
- $\partial L / \partial b = (\partial L / \partial (gv)) \cdot v \cdot g(1-g)$ — $\|v\|$ 가 작으면 (초기 0) $b$ 의 gradient 도 작음.

**결과**: $g$ 와 $\|v\|$ 모두 *느리게* 자람. 학습 종료 시점에 $g \cdot \|v\| = 0.23$ — 01b 의 α=0.5 와 유사 → 효과 미발현.

## 3. 학습 동학

![Train curve](figures/03_scalar_gate/train_curve.png)

*Figure 3. (왼쪽) Train loss 는 단조 감소 (0.90 → 0.79) 이지만 02 (0.74 → 0.24) 보다 *훨씬 느림* — multiplicative gradient saturation 의 직접 증거. (가운데) ‖v‖ 는 epoch 1 에서 3.35, epoch 3 에서 13.5 까지 자람 — 그러나 best state 는 epoch 1 (val confused peak). (오른쪽) Val NDCG@10 confused 는 epoch 1 peak 후 단조 감소 — best 가 매우 일찍 발견됨.*

## 4. 해석 — 본 프로젝트 narrative 에의 함의

| 발견 | 의미 |
|---|---|
| 08 < 02 (학습 동학 모두) | scalar gate 의 *anchor preservation init* (b=-3) 이 학습 lever 를 차단 |
| effective magnitude 0.23 만 학습 | 01b 가 입증한 "effective magnitude 가 큰 영역" 에 도달 못 함 |
| ‖v‖ 만큼은 자라기 시작 (3.35 → 13.5 단 3 epoch) | 학습이 *결국* 작동할 수 있으나 patience 안에서 못 끝남 |

**Empirical conclusion**: 본 multiplicative scalar gate 형식은 본 setting 에서 *실패*. 가능 처방:
1. **Gate init 변경**: $b = 0$ (g=0.5) 또는 $b = 2$ (g≈0.88) — anchor preservation 포기, 학습 freedom ↑
2. **Decoupled parametrization**: $h - \alpha v$ where $\alpha$ unbounded (no sigmoid) — 02 의 unconstrained 와 사실상 같음
3. **Per-token gate (11)**: gate 를 token-dependent 으로. 일부 token 에서 gate 가 자라고 다른 곳에서 작음 → 본질적으로 다른 메커니즘
4. **Longer training**: patience 늘려서 g, ‖v‖ 가 효과적 영역까지 자라기를 기다림

**결정** (autonomous progression): 옵션 3 (04_per_token_gate) 로 진행. 근거:
- 01b Figure 6 (per-query heterogeneous effect) 가 *per-token selectivity* 의 필요성 시사
- 옵션 1/2 는 *gate 의 존재 의의* 자체를 약화 (anchor preservation 무력화)
- 옵션 4 는 8 의 변형일 뿐 (longer training of same architecture)
- 11 은 본질적으로 *다른 메커니즘* — token 마다 다른 gate 값을 학습

## 5. ROADMAP 영향

| 항목 | 영향 |
|---|---|
| 08 의 *narrative* 위치 | "scalar gate 의 한계" 를 입증한 negative result. Paper 의 *어떤 gate 가 필요한가* 의 동기 |
| 09_bias_init_sweep | 우선순위 ↑ — gate init 의 영향 정량 (generalization 단계 의 robustness 와 결합 가능) |
| 10_gate_off | 우선순위 ↓ — 02 가 이미 "no gate" 의 결과. Redundant |
| 04_per_token_gate | **우선순위 최고** — 본 결과의 자연 follow-up |
| 12_gate_capacity | 11 의 결과 보고 결정 |

## 6. Artifact 위치

```
outputs/03_scalar_gate/scifact/seed_42/
├── config.json, env.json, train_config.json
├── module_final.pt
├── train_history.json (gate trace 포함)
├── cosine_with_mean_diff.json (cos = 0.29 → 02 와 유사한 방향)
├── runs / runs_scored / metrics_*
└── delta_vs_{baseline, mean_diff_alpha10, 02_learned}.json

report/figures/03_scalar_gate/{train_curve, delta_ci_forest, effective_magnitude}.{pdf,png}
```
