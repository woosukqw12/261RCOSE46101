# 09_bilinear_M_e5_distill — Bilinear M + E5-Mistral Margin-MSE distillation (λ sweep)

본 보고서는 ROADMAP §"Stage 2" 의 *partial-fail* 분기 후속 — 08 의 *rank-1 collapse* 해소 시도. E5-Mistral-7B-Instruct 의 cross-encoder-quality margin 을 *teacher* 로 Margin-MSE distillation. λ_distill ∈ {0.1, 0.5, 1.0} sweep. **결론: distillation 이 *rank-collapse 해소 lever 가 아닌 anchor regularizer* 로 잘못 작동 — ceiling 우회 실패 + confused-slice 의 lever 도 약화**.

## 1. 결과 종합 — λ-sweep

| λ_distill | Params | NDCG@10 (all) | Δ all vs baseline | Δ confused vs baseline | Δ confused vs 08 r=8 |
|---|---|---|---|---|---|
| **0 (08, no distill)** | 2,048 | 0.6439 | -0.003 (CI 0 포함) | **+0.054 [+0.013, +0.097]** ✓ | (anchor) |
| **0.1** | 2,048 | **0.6509** | +0.005 (CI 0 포함) | **+0.019 [+0.006, +0.033]** ✓ | -0.036 (CI 0 포함) |
| **0.5** | 2,048 | 0.6451 | -0.001 (CI 0 포함) | -0.002 (CI 0 포함, ≈ baseline) | -0.056 ✗ negative |
| **1.0** | 2,048 | 0.6453 | -0.001 (CI 0 포함) | -0.002 (CI 0 포함, ≈ baseline) | -0.056 ✗ negative |

**Key observation**: λ ↑ 면 *anchor preservation* 개선 (all-slice 점점 baseline 에 가까워짐) 이지만 **confused-slice 의 lever 가 점점 죽음** (+0.054 → +0.019 → ~0 → ~0). λ=0.5, 1.0 는 사실상 *학습 안 한 것과 동등*.

![NDCG vs lambda](figures/09_bilinear_M_e5_distill/ndcg_vs_lambda.png)

*Figure 1. λ_distill 의 함수로 SciFact NDCG@10. all-slice (파랑) 는 λ=0.1 에서 최고 (0.6509, baseline 보다 ↑), λ ↑ 면 baseline 으로 수렴. confused-slice (빨강) 는 λ=0 (08) 에서 최고, λ ↑ 면 baseline 수준으로 하락. **Distillation 이 anchor 보호하면서도 confused lever 죽임** — paper main contribution 의 *opposite* 방향.*

## 2. M structure 의 *rank-collapse 해소 양상* (λ-dependent)

| λ | ‖U‖ | ‖V‖ | ‖UV^T‖_F | σ₁/σ₂ ratio | 해석 |
|---|---|---|---|---|---|
| **0 (08)** | 2.04 | 1.38 | **2.61** | **42** | rank-1 강한 dominant (effective rank ≈ 1) |
| **0.1** | 0.99 | 0.82 | 0.49 | **4.1** | partial diversification (rank ≈ 2) |
| **0.5** | 0.51 | 0.48 | 0.10 | 1.5 | uniform, magnitude *너무 작음* (M ≈ I) |
| **1.0** | 0.51 | 0.48 | 0.10 | 1.5 | uniform, M ≈ I |

![Rank collapse by lambda](figures/09_bilinear_M_e5_distill/rank_collapse_by_lambda.png)

*Figure 2. (왼쪽) UV^T 의 singular values, log scale. 08 (λ=0, 회색) 의 σ₁ = 2.6 dominant. λ=0.1 (파랑) 의 σ₁ = 0.46 보다 5× 작지만 σ₂-σ₈ 가 더 평평. λ=0.5/1.0 (주황/빨강) 는 *모든* σ 가 작고 거의 uniform. (오른쪽) ‖UV^T‖_F (파랑 막대) 는 λ ↑ 면 *기하급수적 감소* (2.6 → 0.49 → 0.10). σ₁/σ₂ ratio (빨강 막대, rank-1 dominance) 도 동시 감소 — *rank-1 collapse 는 해소되지만 동시에 M 의 의미 있는 학습도 봉쇄됨*.*

## 3. 학습 동학

![Train curves k-wise](figures/09_bilinear_M_e5_distill/train_curve_kwise.png)

*Figure 3. (왼쪽) Pairwise margin loss — λ 와 거의 무관하게 비슷한 곡선 (∼0.91 → ∼0.7). (가운데) Margin-MSE distill loss (raw, λ 미적용) — 시작점 25.3 으로 모두 동일, λ ↑ 면 학습 후 감소율 거의 같음. (오른쪽) Val NDCG@10 confused — λ=0.1 (파랑) 만 epoch 2 에서 best (0.246) 도달, λ=0.5/1.0 은 모두 epoch 1 best (0.248) 후 dip. 모두 patience 2 로 epoch 3-4 early stop.*

**진단 (학습 동학)**:
- λ 와 무관하게 initial pairwise loss ≈ 0.91 (mined HN 의 정의로 student_margin ≈ -0.7).
- λ ↑ 면 *effective* gradient 가 distill 로 dominate → student_margin → teacher_margin × τ. Teacher × 8 ≈ -0.24 (E5 도 HN 을 약간 더 선호) → student_margin 도 -0.24 근처에 묶임 → 학습 차단.

## 4. 해석 — *Distillation 이 잘못된 lever 인 이유*

### 4.1 Teacher signal 의 *질* 자체 문제

phase_02 의 E5 soft labels sample (45 SciFact qids) 분석:
- e5_margin 의 부호 분포: 약 50% 가 음수 (E5 도 HN 을 더 높게 평가).
- E5-Mistral 이 *strong retriever* 지만 *ColBERT 가 mining 한 HN 에 대해선* 정확한 ranking 못 줌.
- **결과**: distillation 이 *noise teacher* 를 가르침 — student 가 teacher 의 *불확실한 margin* 을 모방하느라 *진짜 ranking lever* 학습 차단.

### 4.2 Margin-MSE 의 *scale mismatch*

- Student margin scale: ColBERT MaxSim sum 의 차이 ~ -0.7 magnitude (mined HN 의 정의).
- Teacher margin × τ=8: E5 cos margin × 8 ~ -0.24 magnitude.
- 두 값이 다른 scale 인 데다 noise 가 큼 → MSE loss = 25 의 huge magnitude → λ=0.1 만 해도 effective gradient 가 distill dominate.
- *Fixed teacher_scale 의 한계*: τ 를 조정해도 *teacher 의 noisy signal* 은 변경 안 됨.

### 4.3 Lever 의 *opposite direction*

Distillation 은 *M 을 identity 근처에 잡아두는 strong regularizer* — 08 의 학습된 rank-1 큰 σ₁ ≈ 2.6 deviation 을 *축소* 시킴. 그러나 본 paper 의 main objective 는 *M ≠ I 로 학습되는 것* (form 변경). Distillation 이 정확히 *반대 방향* 으로 작동.

비유: 08 의 rank-1 dominance 는 *학습이 *informed direction subspace* 의 단일 axis 를 강하게 활용하는 결과 — 비록 r=8 의 capacity 미활용이지만 *retrieval 측면에서는 ceiling 부근 도달*. Distillation 이 이 single dominant lever 도 *눌러버림*.

### 4.4 Stage 2 진행 의 *재해석*

08 + 09 의 종합 발견:
| 구분 | 결과 | 함의 |
|---|---|---|
| 08 (no distill) | confused +0.054, M rank-1 dominant, all-slice 약간 dip | *form 자체* 의 lever 는 *효과 있음* (translation family 보다 confused 약간 우수) 하지만 *ceiling 위로 못 감* |
| 09 (distill) | confused 약화 또는 죽음, M ≈ I | *Margin-MSE distillation 은 잘못된 lever* |

**Stage 2 (bilinear M) 의 form-change lever 자체는 *부분 유효* (08), 그러나 ceiling 우회 미달**. paper main novelty 는 *form 변경* 이 *informed subspace ceiling 위로 가지 못한다* 의 *negative result* 로 정리 가능 — 이는 *frozen-encoder representational limit* 의 정황 증거.

## 5. ROADMAP 영향 + 다음 단계

### 5.1 Distillation pivot 의 *기각*

E5-Mistral Margin-MSE 형식의 distillation 은 *본 setup 에서 효과적이지 않음*. 후속 distillation 시도 시:
- Teacher 변경: MonoT5 / monoBERT cross-encoder soft labels (full cross-attention 이라 ColBERT mined HN 에 대해서도 명확한 margin 줄 가능성).
- Loss 변경: KL divergence on listwise score distribution (multiple HN per q).
- Warmup → distill schedule.

본 시점에서는 우선순위 ↓.

### 5.2 다음 critical 검정

| 후보 | 이유 | 우선 |
|---|---|---|
| **10 r sweep** (r ∈ {1, 4, 16, 32, 64}) | 08 의 r=8 의 effective rank 1 collapse 가 r ↑ 시 변화하는지 직접 측정 — 진정한 rank limit 검정 | **HIGH** |
| **08 + nuclear norm penalty** | distillation 대신 *직접* rank diversity 강제 (low-rank but balanced). 08 의 rank-1 학습 dynamics 변경 가능성 | medium |
| **18 LoRA on Φ** | 08, 09 모두 ceiling 못 넘었으니 *frozen-encoder 자체 한계* 가능성. encoder lightweight finetune 의 *upper bound* 검정 | medium-high |
| 11, 12 cross-dataset | 본 발견의 dataset 의존성 확인 (paper-grade evidence) | 본 main 검정 후 |

### 5.3 Paper narrative 의 *재정렬*

본 시점의 paper main contribution candidate:

> Frozen ColBERT v2 의 *MaxSim form 자체* 의 일반화 ($M = I + UV^\top$) 는 *translation family ceiling* 우회 못 함. r=8 의 학습이 effective rank 1 collapse 로 수렴, E5-Mistral Margin-MSE distillation 은 *over-regularize* 로 form lever 자체를 죽임. 종합 결과는 *informed direction subspace 의 ceiling 이 frozen-encoder representational limit* 임을 시사 — *form 변경만으로는 부족, encoder-level finetune (LoRA on Φ) 가 critical*.

## 6. Artifact 위치

```
outputs/09_bilinear_M_e5_distill/scifact/seed_42/
├── r_8_ld_0p10/{config, env, train_config, module_final.pt, train_history, M_stats, runs, metrics_per_query, metrics_aggregate, delta_vs_*.json}
├── r_8_ld_0p50/{...}
└── r_8_ld_1p00/{...}

report/figures/09_bilinear_M_e5_distill/{ndcg_vs_lambda, rank_collapse_by_lambda, delta_ci_forest_kwise, train_curve_kwise}.{pdf,png}

data/e5_teacher/{e5_train_q_emb_scifact.pt, e5_topk_scifact.pt, e5_soft_labels.json}
```

## 7. 추가 figure

![Delta CI forest k-wise](figures/09_bilinear_M_e5_distill/delta_ci_forest_kwise.png)

*Figure 4. λ ∈ {0.1, 0.5, 1.0} × 4 anchor (baseline / α=10 / 02 / 08) 의 paired bootstrap 95 % CI. λ=0.1 (○) 만 vs baseline confused 가 [+] positive. λ=0.5 (■) 와 λ=1.0 (▲) 는 거의 모든 비교에서 [-] negative (Translation family anchor 보다 worse). **λ=0.1 의 confused +0.019 도 08 의 +0.054 보다 작음** — distillation 은 8 의 lever 를 *약화*.*
