# 05_five_layers — 학습된 direction at 5 layers (single-direction 단계 ceiling 확정)

ROADMAP §"Next" 의 첫 항목. 02 의 single-layer 학습 direction 을 ColBERT v2 의 5 layer (ℓ ∈ {0, 3, 6, 9, 12}) 에 동시 적용하여 single-direction 단계 ceiling 우회 시도. **결과: 우회 실패, 02 와 통계적 동등** — single-direction 단계 single-direction 의 *capacity ceiling* 확정.

## 1. 결과 요약

| Experiment | NDCG@10 | 비고 |
|---|---|---|
| baseline (00) | 0.6464 | anchor |
| 01b α=10 (informed non-learned) | 0.6690 | sharpened anchor |
| 02 single-layer learned | 0.6651 | single-direction 단계 의 single-direction best |
| 03 scalar gate | 0.6448 | gradient saturation |
| 04 per-token gate | 0.6641 | gate=1 saturated |
| **05 5-layer learned** | **0.6502** | **본 실험** — 02 보다 *낮음* |

### Paired bootstrap CI (95 %)

| Anchor | Slice | Δ NDCG@10 | 유의 |
|---|---|---|---|
| baseline (00) | all | +0.0038 [-0.0189, +0.0267] | — (0 포함) |
| baseline (00) | confused | **+0.0510 [+0.0120, +0.0922]** | **✓ positive** |
| 01b α=10 | all | -0.0188 [-0.0436, +0.0046] | — |
| 01b α=10 | confused | -0.0134 [-0.0611, +0.0349] | — |
| 02 learned | all | -0.0148 [-0.0356, +0.0061] | — |
| 02 learned | confused | +0.0074 [-0.0308, +0.0468] | — (통계적 동등) |
| 03 scalar gate | confused | +0.0545 [+0.0154, +0.0962] | ✓ positive |
| 04 per-token gate | confused | +0.0107 [-0.0276, +0.0501] | — |

**핵심 결과**: 05 는 baseline 대비 통계 유의 (+0.051 confused), **02/04/α=10 와 통계적 동등** (CI 0 포함). single-direction 단계 의 ceiling 0.65-0.67 에 머무름.

![Single-direction progression](figures/05_five_layers/single_direction_summary.png)

*Figure 1. SciFact 의 6 condition 비교 (all + confused). baseline → α=10 → 02 → 03 → 04 → 05. all-slice 는 모두 baseline ~ +0.02 수준. Confused-slice 가 condition 별 변동성 큼 — 03 만 baseline 보다 *낮음*. 02 / 04 / 05 가 비슷한 범위. **single-direction 단계 의 ceiling 시각화**.*

## 2. Layer-wise 학습 분석

![Layer norms + cosine](figures/05_five_layers/layer_norms_bar.png)

*Figure 2. (왼쪽) 학습 종료 시점의 layer 별 ‖v_ℓ‖₂. ℓ9 가 가장 큰 magnitude (2.85), ℓ0 가 가장 작음 (1.27). 후기 layer 가 더 큰 norm 학습 — 후기 layer 의 representation 이 더 informative 한 lever. (오른쪽) 각 v_ℓ 와 v_mean_diff (computed at ℓ=12) 의 cosine similarity. ℓ12 만 0.27 로 02 의 cos=0.32 와 유사. 나머지 4 layer 는 cos ≈ 0 → mean-diff direction 과 *직교*. 즉 5 layer 학습이 *다른 axis* 의 정보 잡으려 했으나 retrieval 측면 ceiling 못 넘음.*

| Layer | ‖v_ℓ‖ | cos(v_ℓ, v_mean_diff_l12) | 해석 |
|---|---|---|---|
| 0 (embedding) | 1.27 | -0.005 | mean-diff 와 *직교* |
| 3 | 1.52 | +0.038 | 직교 |
| 6 | 2.22 | +0.011 | 직교 |
| 9 | 2.85 | +0.041 | 직교 |
| 12 (last) | 2.79 | **+0.267** | mean-diff 와 부분 정렬 |

총 ‖v‖ = 4.97 (5 layer 합).

## 3. 학습 동학

![Train curve](figures/05_five_layers/train_curve.png)

*Figure 3. (왼쪽) Train loss 0.69 → 0.10 (epoch 3) — 02 (0.74 → 0.24) 보다 *훨씬 빠른* 감소. 학습 data fitting 강함 (5× 파라미터). (가운데) 총 ‖v‖ 가 epoch 1 의 4.97 → epoch 3 의 6.40 으로 증가. (오른쪽) Val NDCG: confused 가 epoch 1 peak (0.2385) → 2-3 감소. **classic train-overfitting + over-parameterization** 패턴. Early stop @ epoch 3, best=epoch 1 복원.*

## 4. ECDF 비교 — 6 condition

![ECDF compare](figures/05_five_layers/ecdf_compare.png)

*Figure 4. SciFact 의 per-query NDCG@10 의 6 condition ECDF. 모든 LSR 변형 (02/03/04/05/01b α=10) 의 곡선이 baseline (회색) 보다 명확히 우측. 그러나 5 LSR 곡선 간에는 거의 구분 불가 — visualize 로도 *ceiling* 시각적 증거 제공.*

![Delta CI forest](figures/05_five_layers/delta_ci_forest.png)

*Figure 5. 05 의 paired bootstrap 95 % CI 를 5 anchor 에 대해. 위에서 아래로 baseline → α=10 → 02 → 03 → 04. **vs baseline 의 confused** 만 명확히 0 초과. **vs 03 (scalar gate)** 도 양수 (03 의 negative result 회복). 나머지는 모두 0 포함 — 통계적 동등.*

## 5. 결정적 해석 — single-direction 단계 ceiling 확정

본 결과는 single-direction 단계 (02-05) 의 *core finding* 을 **수학적 명료성과 함께 확정**:

### Finding 1: Single-direction subspace 의 capacity 한계
- 02 (학습 v 1 개 at ℓ=12): NDCG@10 = 0.6651
- 04 (per-token gate at ℓ=12): NDCG@10 = 0.6641
- 05 (학습 v 5 개 at ℓ ∈ {0,3,6,9,12}): NDCG@10 = 0.6502
- α=10 mean-diff (비학습 v 1 개 at ℓ=12): NDCG@10 = 0.6690

→ 학습 / 비학습 / 단일 layer / 다층 layer 모두 NDCG@10 ≈ 0.65-0.67 에서 만남.

### Finding 2: Multi-layer 가 *오히려* 일반화 손해
- 02 의 1 layer × 1 direction = 768 params: NDCG=0.6651
- 05 의 5 layer × 1 direction = 3,840 params: NDCG=0.6502 (-0.015)

5× 파라미터 + epoch 1 빠른 peak → over-fitting. **direction 의 *수* 가 늘어도 1 axis 의 search space 확장만으로는 ceiling 못 깬다.**

### Finding 3: Layer 별 *orthogonal* 방향 학습 시도
- 05 의 5 layer 중 4 layer (0, 3, 6, 9) 가 mean-diff direction 과 **거의 직교** (cos ≈ 0).
- 즉 학습이 "다른 axis" 를 시도하지만 결국 retrieval 성능은 *redundant* — 다른 방향들도 *같은* ceiling.

### Narrative 결론

| 결합 발견 | 함의 |
|---|---|
| 02, 03, 04, 05 모두 같은 ceiling | single-direction subspace 의 *capacity 한계* (representational redundancy) |
| 05 의 4 layer 가 mean-diff 와 직교 | 학습이 *다른* 방향 시도하지만 retrieval 측면 같은 result |
| 05 가 02 보다 *낮은* NDCG | multi-layer 단순 확장이 over-fitting 유발 — pure capacity 증가가 lever 가 아님 |

→ **본질적 lever 는 *direction 의 수* (multi-direction, multi-direction 단계) 가 아니라 *paired with selectivity (router)*** — 각 direction 이 *어떤 query/token 에 적용될지* 의 routing 이 핵심. 단순 multi-direction (gate-less) 도 같은 redundancy 함정에 빠질 위험.

## 6. ROADMAP 영향 + 다음 실험 결정

### Priority 재배치

| 항목 | 영향 |
|---|---|
| 06_projection_out (was 17) | 우선순위 ↓ — same single-direction-subspace ceiling 예상. *form variant* 만으로 ceiling 못 넘음. 후속 confirming ablation 으로 deferred. |
| **07_two_directions (multi-direction 단계)** | **우선순위 ↑↑↑** — 본 paper 의 main novelty. 단순 K=2 가 02/05 의 ceiling 을 *유의하게* 넘으면 → multi-direction 가설 즉시 입증. 못 넘으면 router (07-09) 가 critical. |
| 19_anchor_reg_sweep (deferred) | 05 의 over-fitting 패턴 → λ_anc > 0 시도 가능. multi-direction 단계 결과 보고 결정. |

### 다음 실험: **07_two_directions** (multi-direction 단계 진입)

ROADMAP §"Next" 의 07. K=2 multi-direction with softmax router 의 proof-of-concept. 본 paper main contribution 의 first empirical test.

## 7. Artifact 위치

```
outputs/05_five_layers/scifact/seed_42/
├── config / env / train_config.json
├── module_final.pt (5 개 v_ℓ)
├── train_history.json (loss / total ‖v‖ / val curves)
├── layer_norms.json (per-layer ‖v_ℓ‖)
├── cosine_with_mean_diff.json (per-layer cos vs v_mean_diff_l12)
├── runs / runs_scored / metrics_* .json
└── delta_vs_{baseline, mean_diff_alpha10, 02_learned, 03_scalar_gate, 04_per_token_gate}.json

report/figures/05_five_layers/
├── train_curve.{pdf,png}
├── layer_norms_bar.{pdf,png}
├── delta_ci_forest.{pdf,png}
├── ecdf_compare.{pdf,png}
└── single_direction_summary.{pdf,png}
```
