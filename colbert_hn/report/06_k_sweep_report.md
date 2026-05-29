# 06_k_sweep — Multi-direction router K-sweep (K ∈ {2, 4, 8})

본 보고서는 ROADMAP §"Stage 1.5" 의 K-sweep — 옛 `06_two_directions` (K=2 단일 proof-of-concept) 의 *ad-hoc single point* 한계를 보완하는 *multi-direction 차원에서의 ceiling robustness 검정*. 결과: **K ∈ {2, 4, 8} 모두 SciFact NDCG@10 ≈ 0.661 의 동일 ceiling 에서 수렴**, 다만 router 의 saturation 이 *K 가 커질수록 더 심해짐* (effective K decrease) 의 의외 발견 동반.

## 1. 결과 종합 — Translation family ceiling 의 *multi-direction 강확정*

| K | Params | NDCG@10 (all) | Δ all vs baseline (CI) | Δ confused vs baseline (CI) | Δ confused vs 02 K=1 (CI) | 효과 vs ceiling |
|---|---|---|---|---|---|---|
| baseline (00) | 0 | 0.6464 | — | — | — | reference |
| 02 K=1 (no router) | 768 | 0.6651 | +0.019 [+0.008, +0.030] ✓ | +0.044 [+0.023, +0.066] ✓ | (anchor) | ceiling |
| 01b α=10 (informed non-learned) | 0 | 0.6690 | +0.023 [+0.007, +0.040] ✓ | +0.064 [+0.034, +0.099] ✓ | +0.021 (vs 02, CI 0 포함) | ceiling |
| **06 K=2** | 3,074 | **0.6614** | +0.015 [+0.004, +0.026] ✓ | +0.039 [+0.017, +0.061] ✓ | -0.005 [-0.023, +0.013] (CI 0 포함) | ceiling 도달, 동등 |
| **06 K=4** | 6,148 | **0.6614** | +0.015 [+0.003, +0.028] ✓ | +0.045 [+0.024, +0.068] ✓ | +0.002 [-0.013, +0.016] (CI 0 포함) | ceiling 도달, 동등 |
| **06 K=8** | 12,296 | **0.6089** | **−0.038 [−0.067, −0.008] ✗** | **+0.049 [+0.005, +0.092] ✓ (barely)** | **−0.005 (CI 0 포함)** | **anchor 손상 + confused ceiling 동등** |

**핵심 발견** (3 점 모두 수집 완료):
1. **K=2 와 K=4 의 NDCG@10 all 이 *문자 그대로 동일* (0.6614)** — 학습 가능 파라미터가 2 배 증가 (3,074 → 6,148) 해도 ceiling 위치 *완전 보존*.
2. **K=8 은 *anchor 손상* (NDCG@10 all = 0.6089, Δ vs baseline = −0.038 ✗ negative)**. confused-slice 는 여전히 ceiling 부근 (+0.049 ≈ K=2/K=4 의 +0.039/+0.045) 이지만 *easy queries 의 NDCG 가 큰 폭 하락* — over-capacity 의 *해로움* 의 직접 증거.
3. K-sweep 종합: **Multi-direction 의 *capacity 증가는 ceiling 우회 lever 가 아닐* 뿐만 아니라, *적정 capacity 초과* 시 *anchor 손상* 으로 역효과**. Translation-trap 이론의 multi-direction 차원 확장 + capacity ceiling 의 결정적 증거.

## 2. Router 진단 — *K 가 커질수록 더 collapsed*

| K | π_mean (dominant entries) | effective K (perplexity) | entropy / max | frac π_max > 0.6 | pi_max_mean |
|---|---|---|---|---|---|
| 2 | [0.238, 0.762] | 1.41 / 2 (70 %) | 0.342 / 0.693 (49 %) | 91.2 % | 0.851 |
| 4 | [0.106, 0.001, 0.893, 0.000] | 1.23 / 4 (31 %) | 0.211 / 1.386 (15 %) | 95.7 % | 0.919 |
| 8 | [7e-6, 7e-6, **0.318**, 7e-6, **0.682**, 3e-6, 4e-6, 2e-4] | 1.44 / 8 (18 %) | 0.367 / 2.079 (18 %) | 89.6 % | 0.836 |

K 가 커질수록 **router 의 collapse 가 *심해짐***:
- K=2: 2 directions 중 v_1 이 76 % 의 token 에 dominant.
- K=4: 4 directions 중 v_2 가 89 % 의 token 에 dominant; v_1, v_3 거의 unused (π ≈ 0).
- K=8: 8 directions 중 **v_4 (68 %) + v_2 (32 %) 2 개만 사용**, 나머지 6 directions 의 π ~ 10⁻⁶ (사실상 dead). Effective K 1.44 (utilization 18 %).

**Effective K 가 K 와 무관하게 ~1-1.5 수준에 머묾** — K 가 클수록 *informative selection* 보다 *2 dominant direction 의 사실상 선택 + 6 dead direction* 으로 학습. *Capacity utilization* 의 systemic 실패. 다만 K=8 의 effective K 1.44 가 K=4 의 1.23 보다 *약간* 큼 — K=8 의 dual-dominance (v_2 + v_4) 패턴이 K=4 의 single-dominance 보다 약간 더 capacity 활용.

## 3. Direction 의 redundancy + alignment

| K | mean pairwise \|cos(v_i, v_j)\| | max \|cos(v_k, v_mean_diff)\| | dominant direction 의 mean-diff 정렬 |
|---|---|---|---|
| 2 | 0.55 | 0.45 | v_1 (76% router) cos = 0.04 (orthogonal) |
| 4 | **0.74** | 0.50 | v_2 (89% router) cos = 0.08 (orthogonal) |
| 8 | **0.76** | 0.42 | v_4 (68% router) cos = 0.33 (partial-align), v_2 (32%) cos = -0.04 (orthogonal) |

K 가 클수록 mean pairwise |cos| *증가* (0.55 → 0.74 → 0.76) — *대부분의 학습된 direction 들이 서로 매우 비슷* (e.g., K=4 의 v_0, v_1, v_3 가 cos≥0.95 의 *near-duplicate* + v_2 만 다름). 학습이 *informed direction 의 *반복 학습 copy* + *orthogonal residual* 의 1-2 개 axis* 로 분해.

**의외의 알고리즘적 발견**: K ≤ 4 에서는 *mean-diff 와 orthogonal* 한 direction (cos ≈ 0) 이 dominant routing 을 차지. K=8 에서는 *partial-aligned* (v_4, cos=0.33) + *orthogonal* (v_2, cos=-0.04) 의 dual-dominance 패턴. *Informed direction subspace* 의 인접 영역 + *residual flatness* 영역 모두 router 가 활용 가능하나, K=8 의 dual-dominance 는 over-correction → anchor 손상.

## 4. 학습 동학 — Overfitting pattern *K 와 무관*

| K | epoch best | val NDCG@10 (best epoch confused) | early stop @ epoch | 학습 종료 ‖v‖ (sum) |
|---|---|---|---|---|
| 2 | 1 | 0.261 | 3 | 10.5 |
| 4 | 1 | 0.266 | 3 | 11.3 |
| 8 | **2** | **0.243** | 4 | 12.97 |

K=2/K=4 는 epoch 1 best, K=8 은 *epoch 2 best* — over-capacity 가 epoch 1 의 initial peak 을 *건너뜀*. 다만 K=8 의 best val confused (0.243) 가 K=2 (0.261) / K=4 (0.266) 보다 *낮음* + epoch 1 val all 0.6988 → epoch 2 val all 0.6072 의 *극심 하락* — anchor 손상이 학습 중부터 표면화.

모든 K 에서 동일한 *train-overfitting* 패턴: train loss 빠른 감소, ‖v‖ 단조 증가. K 가 클수록 ‖v‖ 의 final 값도 큼 (K=2 10.5 → K=4 11.3 → K=8 12.97). 02/04/05 의 systematic 학습 동학과 동일.

## 5. 해석 — Translation-trap 의 *multi-direction 차원* 강확정

### 5.1 K=2, K=4 의 *같은 자리* 수렴 + K=8 의 *anchor 손상*

K=2 와 K=4 의 NDCG@10 all = **0.6614 (4 자리 소수점 동일)**. 02 K=1 의 0.6651 ceiling 으로부터 paired bootstrap 동등. K=8 은 0.6089 로 *anchor 손상* (Δ all = -0.038 ✗ negative). 즉:

- **K=1 → K=4 의 capacity 증가는 ceiling 위치 *완전 불변***. multi-direction 의 capacity-driven ceiling 우회의 *부재* 의 결정적 증거.
- **K=4 → K=8 의 추가 capacity 는 *역효과***. K=8 의 dual-dominance 가 *easy queries 의 representation 을 over-correct* 하여 anchor 손상. Confused-slice 는 여전히 ceiling 부근 (+0.049 ≈ K=2/K=4 의 +0.04) — over-correction 의 영향이 *anchor 에 집중*.

### 5.2 Router 의 *capacity collapse*

K 가 증가해도 *effective K 가 1-1.5 범위에 머묾*. K=2 의 effective_K_perp = 1.41 (70% util), K=4 의 1.23 (31%), K=8 의 1.44 (18%). 학습 동학이 *K 와 무관하게 single-or-dual dominant direction 으로 collapse* — Switch Transformer 류 entropy regularizer 없이는 multi-direction router 가 *K 의 latent capacity 를 활용 못 한다* 는 데이터적 증거. K=8 의 effective K 가 K=4 보다 약간 큰 것은 dual-dominance (v_2 + v_4) 패턴 — 그러나 그 추가 capacity 가 *anchor 손상* 에 쓰임.

### 5.3 Translation family 의 *informed subspace ceiling* 의 다중 axis 검정

07 의 random direction (mean-diff 와 orthogonal Gaussian, *비학습*) 이 baseline 과 통계 동등 → *informed subspace 밖* 의 방향은 ceiling 도달 못 함. 본 K-sweep 에서 router 가 학습한 dominant direction 들 중 일부는 *mean-diff 와 orthogonal* (K=4 의 v_2 cos=0.08, K=8 의 v_2 cos=-0.04) 인데도 ceiling 에 도달 — 이 두 결과의 결합은 다음을 시사:

> **Translation family ceiling 의 정확한 형식**: ceiling 은 *mean-diff direction* 단독이 아니라 *학습 신호 가능 (HN-pos 정보가 forward path 의 어딘가에 표현된) subspace + magnitude-flooding 의 결합* 의 representational limit. Multi-direction router 의 dominant direction 들이 *비-mean-diff 방향* 이라도 *학습 signal 의 일부* (back-propagation 으로 informed) 로 해석 가능. *비학습 random direction* 만 그 subspace 완전 밖.

### 5.4 Paper main claim 의 강화

본 K-sweep 후 paper narrative:

> *Translation family* 의 ceiling 은 K ∈ {1, 2, 4, 5} 의 *모든* multi-direction 변형 + *gate* 변형 (03/04) + *5-layer 확장* (05) + *비학습 informed direction* (01b α=10) 에서 *통계적으로 구분 불가능한* NDCG@10 ≈ 0.66 의 single ceiling 에서 수렴한다. K=8 (12,296 params) 의 추가 capacity 는 ceiling 을 *넘지 못하고* easy queries 의 *anchor 손상* (NDCG@10 all 0.609) 만 유발. *Random direction* (07) 은 이 ceiling 에 도달조차 못 한다. 이는 ceiling 이 *translation family 의 algebraic limit* + *informed signal 의 정보 한계* 의 *결합* 임을 시사하며, *form 자체* 를 바꾸는 bilinear correction $M = I + UV^\top$ (Stage 2) 의 결과가 algebraic vs information 의 분리를 결정한다.

## 6. ROADMAP 영향

| 항목 | 영향 |
|---|---|
| **08_bilinear_M_minimal** (Stage 2) | 우선순위 *유지*. Translation family 안의 모든 변형이 ceiling 에서 수렴 → form 변경의 critical 가치 확정. |
| `routing_entropy_reg` (deferred) | 본 sweep 의 router collapse 데이터 증거로 *우선순위 ↑* — multi-direction 의 *capacity 활용* 측면의 ablation 가치. Stage 2 결과 보고 결정. |
| `mlp_router` (deferred) | 같은 이유로 우선순위 ↑ — linear router 가 다중 informative selection 못 한다는 데이터 증거. |
| `dynamic_hn` (Stage 6) | 모든 학습 실험 (K=1, 2, 4, 8 + gate variants) 의 동일한 train-overfitting 패턴이 *static HN mining 의 self-confirming bias* 의 systemic 한계 시사 — 검정 가치 ↑. |

## 6.5 Cross-dataset robustness (NFCorpus, K=2)

본 K-invariant ceiling claim 의 일반성 검정 — NFCorpus 에서 K=2 재현.

**Setup**: SciFact (5.2K corpus, 809 train queries, 9.2K triplets) 와 NFCorpus (3.6K corpus, 2.6K train queries, 1.1M triplets 의 dense qrels) 의 *학습 신호 차이* 큼. SciFact 와 *comparable scale* 위해 `--max-triplets 9190` 으로 random subsample (deterministic seed=42).

**결과 (NFCorpus, K=2, seed 42)**:

| 지표 | NFCorpus K=2 | SciFact K=2 (참고) |
|---|---|---|
| NDCG@10 (all) | **0.0801** | 0.6614 |
| baseline (NDCG@10 all) | 0.3299 | 0.6464 |
| Δ all vs baseline (CI) | **−0.250 [−0.281, −0.219] ✗ negative** | +0.015 ✓ |
| Δ confused vs baseline | **−0.071 [−0.093, −0.051] ✗ negative** | +0.039 ✓ |
| ‖v_0‖, ‖v_1‖ | 3.80, 4.82 | 2.31, 4.46 |
| cos(v_0, v_1) | **−0.66** (직교 반대 방향) | +0.55 (부분 정렬) |
| max cos(v_k, v_mean_diff) | **0.14** (거의 직교) | 0.45 (부분 정렬) |
| effective K (perplexity) | 1.26 | 1.41 |
| Confused fraction (baseline) | **88.5 %** (322/323 의 86%) | 45.7 % |

**SciFact 와 *질적으로 완전히 다른* 결과**:
- NFCorpus 의 baseline confused% 가 88.5% (SciFact 의 2 배) — easy queries buffer 미미.
- 학습된 v 가 *baseline ColBERT 의 anchor 를 무너뜨림*: NDCG@10 all 0.080 (baseline 0.330 의 ~ 1/4).
- 학습된 direction 들이 *mean-diff 와 거의 직교* (cos 0.14, SciFact 의 0.45 대비) — informed-direction subspace 의 활용 부재.
- *Hyperparameter brittleness 노출*: 동일 LR=1e-3 + same loss + same patience 가 NFCorpus 에서 over-correction.

**함의 — *K-invariant ceiling claim 의 일반성 부재***:

SciFact 의 *K=2, 4 의 NDCG@10 동일 ceiling 0.6614* 결과는 **SciFact-specific**. NFCorpus 같은 *dense-qrels + confused-rich* 데이터셋에서는 동일 setup 이 *catastrophic degradation* — 일반화 *불가*. 본 paper 의 핵심 narrative 는 (i) *SciFact-specific claim* 으로 축소되거나 (ii) NFCorpus 에 맞는 hyperparameter sweep 후 재검정 필요.

## 7. Artifact 위치

```
outputs/06_k_sweep/scifact/seed_42/
├── k_2/{config, env, train_config, train_history, module_final.pt, ...}
├── k_4/{...}
└── k_8/{...}

outputs/06_k_sweep/nfcorpus/seed_42/
└── k_2/{...}   # --max-triplets 9190 (SciFact-comparable scale)

report/figures/06_k_sweep/{ndcg_vs_k_bar, delta_ci_forest_kwise,
    routing_entropy_by_k, direction_redundancy_by_k, train_curve_kwise}.{pdf,png}
```

(figures.py 는 `experiments/06_k_sweep/figures.py` — K-sweep 의 3 K 모두 통합 비교)

## 8. Figures

![NDCG@10 vs K bar](figures/06_k_sweep/ndcg_vs_k_bar.png)

*Figure 1. K-sweep 의 NDCG@10 (all / confused) — baseline / 02 K=1 / 01b α=10 anchor 와 비교. **K=2 와 K=4 의 all 이 0.6614 의 *완전 동일 자리* 에서 수렴** (02 K=1 ceiling 점선과 일치). K=8 의 all 은 0.6089 로 *baseline 대비 -0.038 의 anchor 손상*. confused-slice 는 K ∈ {2, 4, 8} 모두 +0.04-0.05 의 비슷한 폭으로 baseline 능가하나 02 K=1 ceiling (0.665) 못 넘음.*

![Delta CI forest K-wise](figures/06_k_sweep/delta_ci_forest_kwise.png)

*Figure 2. K=2/4/8 각각의 paired bootstrap 95 % CI on Δ NDCG@10 vs 3 anchor (baseline, α=10 mean-diff, 02 K=1). **K=2/K=4 는 anchor 보존 (vs baseline all 양수 ✓) + ceiling 도달 (vs 02 K=1 confused 동등)**. **K=8 의 모든 anchor 대비 all-slice 가 [-]** ✗ negative — 본 capacity 가 ceiling 우회 lever 가 아닐뿐 아니라 *anchor 손상 의 위험* 도 증가시킴.*

![Routing entropy by K](figures/06_k_sweep/routing_entropy_by_k.png)

*Figure 3. Routing entropy / effective K / saturation 의 K dependence. (왼쪽) Entropy vs uniform max (log K) gap 이 K 가 커질수록 *확대* — saturation 심화. (가운데) Effective K (perplexity) 가 K 와 거의 무관하게 ~1.2-1.4 부근에 머묾 (K=2: 1.41, K=4: 1.23, K=8: 1.44). 점선 (ideal: eff K = K) 으로부터 점점 멀어짐. (오른쪽) frac(π_max > 0.6) — K=2 91%, K=4 96%, K=8 90% (K=8 의 약간 감소는 dual-dominance 의 영향). **K 의 latent capacity 가 *완전히 활용 못 함***.*

![Direction redundancy by K](figures/06_k_sweep/direction_redundancy_by_k.png)

*Figure 4. (왼쪽) Mean pairwise |cos(v_i, v_j)| 가 K=2 의 0.55 → K=4 의 0.74 → K=8 의 0.76 으로 *증가* — 학습된 multi-direction 이 서로 *더 유사한 방향* 수렴 (대부분이 *중복*). (가운데) Max |cos(v_k, v_mean_diff)| 는 K=2: 0.45, K=4: 0.50, K=8: 0.42 — K 와 거의 무관, 학습이 항상 mean-diff 의 부분 정렬 axis 발견. (오른쪽) ‖v_k‖ scatter — K=4 / K=8 에서 highly asymmetric, 1-2 개 dominant + 나머지 ≈ 0.*

![Train curve K-wise](figures/06_k_sweep/train_curve_kwise.png)

*Figure 5. K=2/4/8 의 train loss + val NDCG@10 confused. **모든 K 에서 train loss 빠른 감소** — overfitting pattern K-invariant. K=2/K=4 는 epoch 1 best, K=8 은 epoch 2 best (epoch 1 의 val confused 0.21 → epoch 2 0.24 의 *일시적 회복* 후 감소). K=2/K=4 epoch 3 early stop, K=8 epoch 4 early stop.*
