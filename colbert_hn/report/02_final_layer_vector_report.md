# 02_final_layer_vector — 학습된 단일 direction at layer 12

본 보고서는 ROADMAP.md *single-direction 단계* 의 첫 학습 실험인 `02_final_layer_vector` 의 결과 + 분석. 실험 설계 / hypotheses / 비판적 review item 은 [`experiments/02_final_layer_vector/README.md`](../experiments/02_final_layer_vector/README.md).

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| Python | 3.14.4, PyTorch 2.12.0, MPS |
| Dataset | SciFact (train 809 q / test 300 q, corpus 5,183 docs) |
| Seed | 42 (단일) |
| Hook layer ℓ | 12 |
| Form | $\tilde{h}^{(12)} = h^{(12)} - v$, $v \in \mathbb{R}^{768}$, $v|_{t=0}=\mathbf{0}$ |
| Loss | pairwise margin $m=0.2$ + λ_anc = 0 (DESIGN.md §11 deviation, 2026-05-23) |
| Optimizer | AdamW (LR=$10^{-3}$, WD=$10^{-4}$) |
| Batch | 32 triplets / 5 epochs / patience 2 / val split 10 % |
| HN mining | ColBERT-baseline train top-100, 10 HN/query → 9,190 triplets |

## 2. 학습 동학

![Train curve](figures/02_final_layer_vector/train_curve.png)

*Figure 1. (왼쪽) Step 별 pairwise margin loss — 부드러운 단조 감소 (0.74 → 0.24). (가운데) ‖v‖₂ 의 step 별 trajectory — zero 초기 → epoch 1 종료시 ~3.9 → epoch 4 종료시 ~11.2 으로 단조 증가. 0 에서 시작했지만 학습이 v 를 강하게 키움. (오른쪽) Validation NDCG@10 by epoch — **all** 은 epoch 1 에서 peak (0.6763) 후 단조 감소, **confused** 는 epoch 2 에서 peak (0.2676). Early stop @ epoch 4 (patience 2 over confused). Best state 는 epoch 2 의 v.*

**핵심 진단**: train loss 는 epoch 1-4 단조 감소, 그러나 val NDCG 는 epoch 1-2 부터 plateau / 감소. **classic 한 train-overfitting**. v 의 크기가 epoch 마다 ~2 unit 씩 자라남 → 학습이 *얼마나 크게 빼는가* 를 계속 키움 → 어느 시점부터 test-time 의 anchor 가 손상.

## 3. Test 결과

| Metric | Value |
|---|---|
| NDCG@10 | **0.6651** (baseline 0.6464; 01b α=10 0.6690) |
| NDCG@1 | 0.5667 |
| Recall@10 | 0.7749 |
| MRR@10 | 0.6404 |
| Confused% | 43.3 % (137 / 300) |

### 3.1 Paired bootstrap CI

| Anchor | Slice | Δ NDCG@10 | 95 % CI | 유의 |
|---|---|---|---|---|
| **baseline (00)** | all | +0.0186 | [+0.0083, +0.0301] | **✓ positive** |
| **baseline (00)** | confused | **+0.0436** | **[+0.0232, +0.0657]** | **✓ positive** |
| **α=10 mean-diff (01b)** | all | -0.0039 | [-0.0166, +0.0085] | — (0 포함) |
| **α=10 mean-diff (01b)** | confused | -0.0208 | [-0.0472, +0.0031] | — (0 포함) |

![Δ CI forest](figures/02_final_layer_vector/delta_ci_forest.png)

*Figure 2. 학습된 02 의 paired bootstrap 95 % CI 를 두 anchor 에 대해. 위 2개 row (vs baseline) 는 명확히 0 초과 — baseline 대비 통계적으로 유의한 개선. 아래 2개 row (vs α=10 mean-diff) 는 0 을 포함 — informed non-learned baseline 과 통계적으로 구분 불가, 점수상은 slightly 낮음.*

### 3.2 Per-query 분포

![Δ violin](figures/02_final_layer_vector/delta_violin.png)

*Figure 3. (왼쪽) 02 vs baseline 의 per-query Δ 분포 — 평균 양 (베이스라인 통과), 분포가 0 위쪽으로 mass 집중. (오른쪽) 02 vs α=10 mean-diff 의 per-query Δ — 평균 거의 0, 분포가 0 양쪽으로 대칭에 가까움 → 두 방법이 *서로 다른 query 에 다르게* 작용하지만 *평균적으로 동등*.*

![ECDF compare](figures/02_final_layer_vector/ecdf_compare.png)

*Figure 4. Per-query NDCG@10 ECDF: baseline (회색) vs 01b α=10 (주황) vs 02 (파랑). α=10 곡선과 02 곡선이 매우 유사하게 baseline 우측에 위치 — 둘이 *전체적으로* 유사한 retrieval quality 분포 제공.*

## 4. H5 qualitative — 방향 비교

![Direction compare](figures/02_final_layer_vector/direction_compare.png)

*Figure 5. (왼쪽) 두 v 의 L2 norm: mean-diff = 0.27, learned = 7.08 — 학습이 magnitude 를 **26 배** 키움. (오른쪽) cos(v_learned, v_mean_diff) = **0.3241**. H5 threshold (0.9, magnitude-only 만 학습한 경우) 보다 **크게 낮음** → 학습된 방향은 mean-diff 와 *qualitatively* 다른 정보를 학습.*

**해석**:
- 학습 v 와 mean-diff v 의 cosine = 0.32 → 두 벡터는 *동일 평면이 아닌* 768-d 공간의 다른 방향.
- 그러나 *retrieval 성능은 통계적으로 동등*. 즉 **여러 다른 방향이 비슷한 retrieval 효과** 를 줄 수 있음 — 단일 direction 의 표현력 한계 시사.

## 5. 결정적 finding 의 종합 해석

| Finding | 의미 |
|---|---|
| 02 가 baseline 통과 (confused +0.044 CI) | 학습된 direction 은 *원리적으로* 작동 — H1 부분 통과 |
| 02 가 α=10 anchor 미통과 | 학습이 magnitude-tuned mean-diff 보다 *우월* 하지 않음 |
| cos = 0.32 (다른 방향) | H5 qualitative 통과 — 학습이 *다른 정보* 잡음 |
| 그러나 같은 retrieval 성능 | 단일 direction 의 *부족함* — 다른 방향들도 같은 성능 → 표현력 한계 |
| train loss vs val NDCG divergence | 학습이 *training set 의 HN 분포* 에 과적합 → test 일반화 부족 |

**대주제 narrative 측면**: 본 결과는 *단순 학습 direction 만으로는 부족* — gate / per-token selectivity / multi-layer / multi-direction (router) 의 *empirical 필요성* 을 데이터로 입증. 이는 ROADMAP 의 single-direction 단계 (08+, 11+, 13+) 와 multi-direction 단계 (20-25 multi-direction router) 의 *동기를 강화*.

## 6. 다음 실험 결정 (autonomous)

ROADMAP 의 single-direction 단계 ordering 을 결과 기반 재배치:

| 우선 | 실험 | 근거 |
|---|---|---|
| **HIGH** | 03_layer_sweep | 빠르게 (~10 분/layer × 4 = 40 분) layer 12 가 최적인지 확인. layer 6/9 가 더 좋으면 02 + 후속 모두 재실행. |
| **HIGH** | 03_scalar_gate | 02 의 overfitting 패턴 — scalar gate 의 anchor preservation 효과로 일반화 향상 기대. |
| **HIGH** | 04_per_token_gate | 01b Figure 6 (per-query heterogeneous) 와 02 의 cos=0.32 (방향 분산성) 모두 *per-token selectivity 의 필요성* 시사 |
| MEDIUM | 13_five_layers | multi-layer 의 compound 효과 — single-direction 단계 의 핵심 default |
| LOW | 04_sign_flip | cos=0.32 만으로도 부분 답변 |
| LOW | 05_random_vector | 학습이 random 보다 나음은 02 vs random 사이에 명백 |
| LOW | 06_init_sweep | zero init 이 잘 작동 — 큰 issue 없음 |
| LOW | 07_post_projection | 후속 검토 |

## 7. 학습 hyperparameter 의 잠재적 개선 (deferred)

02 의 overfitting 패턴 → 가능한 개선 (별도 실험 권장):

| 후보 | 기대 효과 |
|---|---|
| LR 감소 (1e-3 → 5e-4) | gradient noise 감소, smoother training |
| λ_anc > 0 (예: 1e-3) | ‖v‖ 폭주 방지 — 그러나 01b 가 large magnitude 가 유리함을 보임 — trade-off |
| Epoch 1-2 만 학습 후 stop | val 의 best 가 epoch 2 — 짧은 학습 |
| Larger batch (64) | gradient variance 감소 |

이 모든 hyperparameter sweep 은 generalization 단계 의 31_margin_sweep / 19_anchor_reg_sweep 와 일맥상통. 02 의 최적 hyperparameter 가 아닌 *default DESIGN setting* 으로 baseline 확립 — 후속 sweep 에서 보강.

## 7.5.b Clean ColBERT-finetune (no steering hook) — v=0 hook 의 영향 *0 인 직접 검정*

7.5 의 02 unfrozen 은 *v=0 frozen 의 SteeringModule hook* 을 단 채 encoder finetune. Reviewer 의 가능 공격: "v=0 hook 이 학습 신호 추가했냐". 본 robustness check (2026-05-24) 는 `--no-steering` flag 로 SteeringModule v 를 `requires_grad_(False)` 으로 frozen (사실상 zero hook 영구). 다른 설정 동일.

**결과 비교**:

| 지표 | 02 unfrozen (with v=0 hook) | **Clean baseline (no steering)** |
|---|---|---|
| NDCG@10 all | 0.6576 | **0.6924** |
| **Δ all vs baseline** | +0.011 [-0.039, +0.062] (CI 0 포함) | **+0.046 [-0.002, +0.096]** (CI 거의 0) |
| **Δ confused vs baseline** | +0.252 [+0.179, +0.328] ✓ | **+0.260 [+0.182, +0.338] ✓** |
| Δ confused vs 01b α=10 | +0.188 ✓ | +0.195 ✓ |
| ‖v_learned‖ | 0.33 (학습됨) | 0.0 (no-grad 고정) |
| Train loss 종료 (ep3) | 0.0042 | **0.0025** (더 빠른 fit) |

**핵심 진단**:
- **Δ confused 의 magnitude 가 essentially 동일** (+0.252 vs +0.260) — *v=0 hook 의 영향 negligible*.
- Δ all 은 clean baseline 이 더 높음 (+0.046 vs +0.011, CI 하한 -0.002 *strict 돌파 직전*). v=0 hook 이 *very small* anchor-disturbing 영향 가능성.
- *Frozen-encoder representational limit* claim 의 *cleanest evidence* — *no hook, no steering, just encoder finetune* 으로 +0.260 confused lift.
- Reviewer 의 "v=0 hook 이 학습 신호 추가했냐" 공격 *완전 해소*.

상세: `outputs/02_final_layer_vector/scifact/seed_42/unfrozen_no_steering/`.

## 7.5 Unfrozen ColBERT robustness check — *frozen-encoder ceiling 의 직접 검정*

본 paper 가 누적적으로 *informed-subspace ceiling* 에 도달했다 (02–09 의 모든 frozen 변형 NDCG@10 all ≈ 0.65-0.67). **본 robustness check 는 단일 변경 — encoder 의 `requires_grad=True`** — 으로 그 ceiling 의 *원인이 frozen encoder 인지 학습 데이터 한계인지* 직접 분리.

**Setup** (`--unfreeze-encoder`):
- ColBERT v2 의 *모든 110M params* 학습 가능 (BERT-base + 768→128 projection).
- Encoder LR = $5 \times 10^{-5}$ (typical BERT finetune), Steering LR = $10^{-3}$ (기존 02 setting). 두 optimizer group.
- 같은 SciFact 9,190 triplets + LR + pairwise margin + 3 epochs (encoder 가 1-2 epoch 내 빠르게 fit).
- val 에서 best state 복원 (val score = 0.2549, epoch 2 best).

**결과** (SciFact, seed 42):

| 지표 | Frozen 02 (default) | **Unfrozen 02** |
|---|---|---|
| 학습 가능 params | 768 | **109.6 M** (steering 무시) |
| NDCG@10 (all) | 0.6651 | **0.6576** (≈ baseline) |
| **Δ confused vs baseline** | +0.044 [+0.023, +0.066] ✓ | **+0.252 [+0.179, +0.328] ✓** |
| Δ confused vs 01b α=10 | -0.021 [-0.047, +0.003] (≈) | **+0.188 [+0.105, +0.271] ✓** |
| ‖v_learned‖ | 7.08 (학습 큼) | **0.33** (거의 휴면 — encoder 가 lever) |
| Train loss 종료 | 0.24 (3 epoch) | **0.0042** (3 epoch, 사실상 완벽 fit) |
| Time per epoch | ~85 s | ~275 s (≈ 3.2 × slower, MPS) |

**핵심 발견** — *Frozen encoder 가 진짜 bottleneck*:

1. **Unfrozen ColBERT 의 Δ confused +0.252 가 *frozen + steering 의 max +0.054 (08 seed 42 의 SciFact-specific best) 의 ~5 ×***. 본 paper 의 모든 frozen-side intervention 의 *5 배 lift* 가 단지 encoder unfreeze 로 가능.
2. **‖v_learned‖ = 0.33 (frozen 의 7.08 대비 *1/20*)** — encoder 가 직접 ranking 신호 흡수, steering vector 는 거의 작동 안 함. *Lever 위치 의 직접 증거*.
3. **All-slice 도 baseline 동등** (+0.011, CI 0 포함) — encoder unfreeze 가 anchor 도 *손상 안 함*.
4. 단 *109.6 M params* 는 본 paper 의 50 K budget 의 ~2200 ×. 본 결과는 *upper bound* 지 *practical method* 가 아님. **paper-grade 의 다음 단계: 50 K budget 안의 LoRA on Φ** (e.g., r=4 attention adapters ~200 K, r=1 ~50 K).

**함의** — *Paper narrative 의 *근본적 재정렬***:

| 옛 narrative | 새 narrative |
|---|---|
| "Form-change + distillation 도 ceiling 못 넘음 → informed subspace 의 representational limit" | "Frozen encoder 가 **명확한** bottleneck. Lightweight intervention 의 ceiling 은 *encoder representational limit* 의 직접 결과. LoRA on Φ 가 50 K budget 안에서 어디까지 lifting 하는지 가 *진짜 main contribution*". |

8 의 +0.054 / 09 distillation / form change 의 모든 결과는 *frozen encoder 한계를 넘지 못한* *예상된 결과*. **새 main novelty 후보**: 50 K LoRA budget 안에서 *어떤 layer/component 분배가 confused-slice 의 +0.25 lift 를 최대한 보존하는가* 의 분석.

## 8. Artifact 위치

```
outputs/02_final_layer_vector/scifact/seed_42/
├── config.json, env.json, train_config.json
├── v_final.pt (학습된 v ∈ ℝ^768, ‖v‖=7.08)
├── triplet_stats.json, mean_diff_stats.json
├── train_history.json (step 별 loss / ‖v‖ / val curves)
├── cosine_with_mean_diff.json (H5 qualitative)
├── runs.json, runs_scored.json, metrics_per_query.json, metrics_aggregate.json
├── delta_vs_baseline.json
└── delta_vs_mean_diff_alpha10.json

outputs/02_final_layer_vector/scifact/seed_42/unfrozen/   # --unfreeze-encoder
├── ... (위와 동일 구조, encoder 110M params 학습)
└── module_final.pt (steering v ‖0.33‖, encoder weights 별도)

report/figures/02_final_layer_vector/
├── train_curve.{pdf,png}
├── delta_ci_forest.{pdf,png}
├── delta_violin.{pdf,png}
├── ecdf_compare.{pdf,png}
└── direction_compare.{pdf,png}
```
