# 08_bilinear_M_minimal — Stage 2 critical falsification

본 보고서는 ROADMAP §"Stage 2" 의 **main novelty 의 *critical falsification* 결과**. *Bilinear interaction metric* $M = I + UV^\top$ (r=8, 2,048 학습 가능 파라미터) 의 frozen ColBERT v2 + SciFact 실험. **결론: r=8 + pairwise margin only 에서 *Stage 2 통과 실패* — NDCG@10 all = 0.6439 (≈ baseline 0.6464), translation family ceiling 위로 가지 못 함**. Confused +0.054 ≈ K-sweep 의 +0.04 와 통계 동등.

다만 *흥미로운 부산물 발견*: 학습된 M 의 effective rank 가 **사실상 1** (UV^T 의 singular values 가 [2.60, 0.06, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01] 으로 rank-1 dominant). r=8 의 latent capacity 가 router K-collapse 와 동일한 *optimization-driven collapse* 양상.

## 1. 결과 종합

| Comparison | NDCG@10 (all) | NDCG@10 (confused, self) | Δ vs anchor confused (CI) | 통과? |
|---|---|---|---|---|
| baseline (00) | 0.6464 | 0.2377 | — | reference |
| 02 K=1 (learned) | 0.6651 | 0.2388 | +0.044 ✓ (vs baseline) | ceiling |
| 01b α=10 mean-diff | 0.6690 | 0.2377 | +0.064 ✓ (vs baseline) | ceiling |
| 06 K=2 router | 0.6614 | 0.2476 | +0.039 ✓ (vs baseline) | ceiling |
| 06 K=4 router | 0.6614 | 0.2521 | +0.045 ✓ (vs baseline) | ceiling |
| 06 K=8 router | 0.6089 | 0.2281 | +0.049 ✓ (vs baseline), but **anchor 손상** | over-cap |
| **08 r=8 bilinear M** | **0.6439** | 0.2358 | **+0.054 ✓ (vs baseline)** | **ceiling 동등 + anchor 미세 손상** |

### Paired bootstrap 95 % CI (vs 4 anchor, baseline-defined confused slice)

| Anchor | Slice | Δ NDCG@10 | 유의 |
|---|---|---|---|
| baseline (00) | all | -0.003 [-0.027, +0.022] | — (CI 0 포함) |
| baseline (00) | confused | **+0.054 [+0.013, +0.097]** | **✓ positive** |
| 01b α=10 mean-diff | all | -0.025 [-0.046, -0.005] | ✗ negative |
| 01b α=10 mean-diff | confused | -0.010 [-0.048, +0.026] | — (CI 0 포함) |
| 02 K=1 learned | all | -0.021 [-0.045, +0.002] | — (CI 0 포함) |
| 02 K=1 learned | confused | +0.011 [-0.031, +0.052] | — (CI 0 포함) |
| 06 K=2 router | all | -0.018 [-0.041, +0.006] | — (CI 0 포함) |
| 06 K=2 router | confused | +0.015 [-0.027, +0.058] | — (CI 0 포함) |
| 06 K=4 router | all | -0.018 [-0.042, +0.007] | — (CI 0 포함) |
| 06 K=4 router | confused | +0.009 [-0.036, +0.054] | — (CI 0 포함) |

**Critical 진단**:
- vs baseline confused **+0.054** — *baseline 대비 confused 개선*. ceiling 부근 도달.
- vs 01b α=10 all **-0.025 ✗ negative** — *informed non-learned anchor 도 못 넘음*.
- vs translation family (02 / 06 K=2 / 06 K=4): all CI 0 포함, confused CI 0 포함. **통계적 동등** — *form 변경의 추가 lever 없음* 의 직접 증거.

## 2. 학습 동학 + M 의 spectrum 진단

![Train curve](figures/08_bilinear_M_minimal/train_curve.png)

*Figure 1. Train loss 0.91 → 0.56 (epoch 1-5, 단조 감소). ‖[U; V]‖₂ 0.45 → 2.83 (안정적 성장). Val NDCG@10: all 곡선이 epoch 3 (0.6924) peak 후 epoch 5 의 0.5928 로 큰 폭 하락 — *over-fitting* 패턴. confused 곡선은 epoch 4 (0.2516) peak 후 epoch 5 (0.1747) 큰 폭 하락. Best state = epoch 4 복원.*

![M spectrum](figures/08_bilinear_M_minimal/M_spectrum.png)

*Figure 2. (왼쪽) UV^T 의 singular values: **σ_1 = 2.60 (dominant) >> σ_2 = 0.062 (rest)**. r=8 의 latent capacity 중 *사실상 1 차원* 만 활용. (오른쪽) M = I + UV^T 의 spectrum: σ_1 = 2.64 (강하게 amplify 된 한 방향), σ_2 ≈ 1.02, σ_3-8 ≈ 1.01 (사실상 identity); σ_119-128 ≈ 1.000 (정확히 identity). M condition number = 81.14.*

![UV inner structure](figures/08_bilinear_M_minimal/UV_inner.png)

*Figure 3. U^T U / V^T V / U^T V 의 r×r heatmap. U, V 모두 strong dominant 방향 + rest 작은 잡음. **U 와 V 가 *같은* dominant rank-1 axis 에 정렬** — 학습이 rank-1 효과적 사용에 수렴.*

**핵심 진단**:
- **Rank-1 collapse**: r=8 의 capacity 8 차원 중 ~99% 의 spectral mass 가 dominant direction 에 집중. K-sweep 의 effective K collapse 와 *완전히 동일* 패턴 — *capacity utilization* 의 systemic failure.
- **M condition = 81**: M 이 *극단적* 으로 condition 됨 — 어떤 방향 (singular vector 1) 은 ×2.64 scale, 다른 모든 방향은 ×1 (identity). 이는 *single bilinear direction 의 강한 활성* 학습이지만 *retrieval 측면에서는 ceiling 위로 못 감*.

## 3. 비교 — translation family vs bilinear M 의 effective lever

| 형식 | Algebraic family | Effective capacity (학습 후) | Retrieval ceiling |
|---|---|---|---|
| 02 (single direction) | translation | 1 direction | 0.665 |
| 06 K=2/4/8 router | translation (convex combo) | 1-2 effective directions (K-collapse) | 0.661 (K=2/4), 0.609 (K=8 over-cap) |
| **08 bilinear M r=8** | bilinear (NEW) | **1 effective rank** (rank-collapse) | **0.644** |

전체적으로 학습이 *항상 1 차원의 dominant lever 로 collapse* — 가능한 explanation:
- Pairwise margin loss 가 *single dominant direction* 만 강하게 신호 줌.
- AdamW + 1e-4 LR + 5 epochs 가 *higher-order capacity* 활용에 부족한 optimization budget.
- SciFact 의 9K triplet 의 *training signal* 한계 (모든 학습 실험의 systematic train-overfitting).

## 4. 해석 — Stage 2 *critical falsification* 결과

### 4.1 표면 결론: r=8 bilinear M *unaided* 으로 ceiling 못 넘음

본 실험만으로는 *form 자체* 의 변경 (translation → bilinear) 의 lever 가 **insufficient**. Translation family ceiling (0.665) 위로 가지 못함. paper main contribution 의 *naive minimal* 형식 부족.

### 4.2 심층 해석: *optimization-driven rank collapse*

학습된 UV^T 의 effective rank 가 1 — r=8 의 capacity 가 활용 안 됨. *Algebraic form change* 의 가능성 자체는 검정되지 못함 (capacity 는 있는데 학습 동학이 못 끌어내는 상황).

### 4.3 Stage 2 critical 의 *재해석*

본 결과는 **Stage 2 의 *부분 falsification* + open question**:
- Translation family ceiling 의 *algebraic limit* 진단은 *아직 확정 안 됨* — r=8 의 학습이 rank-1 으로 collapse 했기 때문에 *r=8 의 full capacity* 를 사용한 ceiling 우회의 가능성은 검정 안 된 상태.
- 즉 *form 자체* 의 lever 부재가 아니라 *optimization 의 lever 활용 부재* 가능성.

### 4.4 다음 단계 후보

| 후보 | 이유 | 우선 |
|---|---|---|
| **09 E5 distillation** | richer ranking 신호로 rank-collapse 해소 가능성. cross-encoder margin 이 multi-axis bilinear interaction 을 더 informative 하게 학습 가능. | **HIGH** |
| **10 r sweep** (r=1, 16, 32) | r=1 이 정말 ceiling 도달 못 하는지 (effective rank=1 인 r=8 과 동일 예상). r=16/32 의 추가 capacity 가 학습 dynamics 변화시키는지. | **HIGH** |
| `bilinear LR sweep` (e.g., 5e-5, 5e-4) | LR=1e-4 가 너무 보수적일 수도. 단 LR=1e-3 의 val crash 패턴은 너무 공격적. | medium |
| **E5 distillation + r sweep 동시** | 둘 다 활성 lever 일 수 있음 — 한 단계 실험. | medium |
| **bilinear projection 위치 변경** | M 이 128-d projected space 가 아닌 768-d hidden 에 적용 → 더 풍부한 정보. 구현 복잡, 비용 ↑. | low |
| **18 LoRA on Φ** | bilinear M 이 ceiling 못 넘으면 frozen-encoder representational limit 의 가능성 — encoder 자체 lightweight finetune. paper 의 *upper bound* 검정. | low |

## 5. ROADMAP 영향

- **Stage 2 진입은 *유지*** — 09 E5 distillation + 10 r sweep 의 결과로 form-change 의 진짜 lever 검정 필요.
- **r=8 + pairwise margin only 의 *불충분성* 데이터 증거** — paper main novelty 는 *(bilinear M) + (E5 distillation)* 의 *결합* 으로 narrative.
- **새 ablation 추가** (autonomous, DESIGN.md mirror 필요): *rank-collapse* 의 routing K-collapse 와의 평행 관찰 + entropy reg / nuclear norm penalty 의 *rank activation* 가치.

## 5.5 Seed × 3 robustness check — *seed 42 의 +0.054 가 artifact*

SciFact + r=8 + LR=1e-4 + small_random init 의 *seed-dependency* 점검. seed ∈ {42, 1337, 2024} 각각 동일 hyperparameter 로 학습.

| 지표 | Seed 42 | Seed 1337 | Seed 2024 |
|---|---|---|---|
| NDCG@10 all | 0.6439 | 0.6446 | 0.6446 |
| **Δ confused vs baseline** | **+0.054 [+0.013, +0.097] ✓** | **−0.001 [−0.006, +0.002] (≈ baseline)** | **−0.001 [−0.005, +0.001] (≈ baseline)** |
| ‖UV^T‖_F | **2.61** | 0.085 | 0.100 |
| UV^T σ₁ (dominant) | **2.60** | 0.068 | 0.089 |
| σ₁/σ₂ ratio | **42** (rank-1 dominant) | 3.4 | 3.4 |
| M condition number | **81.14** | 1.08 (≈ identity) | 1.09 (≈ identity) |

**핵심 발견**:
- **Seed 42 만 *rank-1 collapse + σ₁=2.60 강한 학습* 발생**. Seed 1337, 2024 는 **M ≈ identity** (사실상 학습 안 됨).
- **Δ confused +0.054 의 SciFact-specific 학습 lever 도 *seed 42 artifact***. 두 다른 seed 에서는 *baseline 동등*.
- 3-seed 평균 Δ confused ≈ **+0.017** — 무시 가능 magnitude.

**진단** — *왜 seed 가 학습을 결정하는가*:

Small_random init (U, V ~ N(0, 10⁻⁴)) 의 *초기 UV^T 방향* 이 LR=1e-4 + Adam 의 수렴 basin 을 결정. Seed 42 의 init 이 *우연히* 어떤 informed direction subspace 의 axis 쪽으로 향했고 → bilinear lever 활성. Seed 1337, 2024 의 init 은 그 방향 못 잡고 → loss landscape 의 *near-identity* basin 에 갇힘. Validation set 80 queries 의 noise 가 best-epoch 선택을 *seed 마다 다른 방향* 으로 유도.

**함의** — 본 paper 의 핵심 claim *대대적 재검토 필요*:
- "Bilinear M form 변경이 confused +0.054 의 small but significant lever" 의 claim 은 **seed 42 단일 관찰** 에 의존.
- 정직한 paper-grade narrative: "*Bilinear M r=8 + pairwise margin 의 seed-mean confused 효과는 ≈ 0; seed 42 만 특이 학습 trajectory*. 본 setup 의 학습 신호 가 *seed-noise dominated*."
- 후속 정직한 검정: (a) LR / init scale / batch size 의 sweep × 3 seed, (b) val set 확대 (현 80 queries → 160 queries) 의 best-state selection robustness ↑.

## 6. Artifact 위치

```
outputs/08_bilinear_M_minimal/scifact/seed_{42,1337,2024}/r_8/
├── config / env / train_config / module_final.pt / train_history.json
├── M_stats.json (‖U‖, ‖V‖, ‖UV^T‖_F, SVD spectrum, cond num)
├── runs / runs_scored / metrics_per_query / metrics_aggregate.json
└── delta_vs_{baseline, mean_diff_alpha10, 02_learned, 06_k_sweep_k2, 06_k_sweep_k4}.json
   (seed 1337/2024 의 deltas 는 baseline 의 seed_42 symlink 로 paired bootstrap)

report/figures/08_bilinear_M_minimal/{ndcg_vs_baselines, delta_ci_forest,
    M_spectrum, train_curve, UV_inner}.{pdf,png}
```

## 7. Figures

![NDCG vs baselines](figures/08_bilinear_M_minimal/ndcg_vs_baselines.png)

*Figure 4. 08 bilinear M r=8 vs 6 baseline/anchor 의 NDCG@10 (all / confused). 점선 = translation-family ceiling (0.669 of α=10 mean-diff). **08 의 all-slice (0.6439) 가 baseline (0.6464) 보다 약간 낮음 — ceiling 위로 못 감**. confused-slice (0.236, self-defined) 는 K-sweep 과 유사.*

![Delta CI forest](figures/08_bilinear_M_minimal/delta_ci_forest.png)

*Figure 5. 08 의 5 anchor 대비 paired bootstrap CI. **vs baseline confused 만 [+] positive (+0.054)**. 나머지 (vs α=10, vs 02, vs 06 K=2, vs 06 K=4) 의 CI 가 모두 0 포함 — translation family 와 *통계적 동등*. form 변경의 추가 lever 부재.*
