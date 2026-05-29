# ROADMAP — 실험 master sequence

본 문서는 본 프로젝트의 실험 master plan 의 single source of truth. CLAUDE.md / DESIGN.md / 각 `experiments/{NN}_*/README.md` 가 본 문서를 *원본* 으로 참조.

**중대 수정 (2026-05-23)**: 실험 00–06 의 결과를 종합한 algebraic 진단 (*translation-trap*) 에 따라 ROADMAP 전면 개편. 옛 master plan (gate / multi-direction / routing 의 incremental ablation) 은 *모두 translation family 안의 재배치* 였음을 사후 확인 → 같은 ceiling (NDCG@10 ≈ 0.665) 도달 불가피. 새 ROADMAP 은 *translation family 밖으로의 minimal 우회* (bilinear interaction metric) 를 main novelty 로 재정렬.

**구성 원칙**:
- 폴더 / 보고서 / output 디렉토리 번호 는 **실행 순서 (sequential)** 로 부여.
- 신규 main novelty (bilinear M) 의 검정은 *conditional* — falsification (07) 통과 후 본격 진입.
- 옛 master plan 의 미실행 실험은 *translation family 의 confirmatory ablation* 으로 강등 (supplementary appendix 등급).

---

## 📐 새 thesis (revised)

> Frozen retriever 의 hidden state 에 대한 **데이터 의존적 translation** (additive subtraction $\tilde{h} = h - u(h)$) 은 MaxSim 의 bilinear form 을 변경하지 못해 *query-conditional ranking* 정보를 구조적으로 표현할 수 없다. 본 paper 는 (a) 이 *translation-trap* 의 algebraic ceiling 을 실증하고, (b) q–d interaction form 자체를 바꾸는 *minimal bilinear correction* $M = I + UV^\top$ 이 그 ceiling 을 *유의하게* 넘는 frozen-encoder representation-level intervention 임을 입증한다.

기존 thesis (lightweight steering module 로 HN 완화) 의 *upgrade*. 학습 가능 파라미터 ≤ 50K, 추론 시 외부 LLM 호출 불필요 등의 제약은 그대로 유지.

---

## ✅ 완료 (00–07) — translation family ceiling 의 증명 + direction-agnostic 가설 falsification

옛 ROADMAP 의 "Phase 1 / 3 incremental ablation" 으로 분류되었던 실험들이, 새 narrative 에서는 *translation family 안에서의 capacity / form 변경 모두 같은 ceiling 에 수렴* 함을 *증명* 하는 역할. 07 은 그 ceiling 의 *direction-agnostic* 측면을 *falsify* 하는 결정적 실험.

| # | Dir | NDCG@10 | $u(h)$ 형식 | 역할 |
|---|---|---|---|---|
| 00 | `00_baseline` | 0.6464 | — | reference |
| 01 | `01_mean_diff` | 0.6459 | $u = v_{\text{md}}$ (raw) | translation w/ insufficient magnitude → ≈ 0 |
| 01b | `01b_mean_diff_scaled` | α=10: 0.6690 | $u = \alpha \hat{v}_{\text{md}}$ | *magnitude lever 발견* — direction-agnostic 의심 |
| 02 | `02_final_layer_vector` | 0.6651 | $u = v$ (learned) | learned 가 α=10 과 통계 동등 → *direction 의 학습이 큰 의미 없음* |
| 03 | `03_scalar_gate` | 0.6448 | $u = \sigma(b) v$ | translation, multiplicative gradient saturation |
| 04 | `04_per_token_gate` | 0.6641 | $u = \sigma(W h + b) v$ | translation, gate always-on collapse |
| 05 | `05_five_layers` | 0.6502 | $u_\ell = v_\ell$, 5 layer | 5 translation의 합성 = translation, 같은 ceiling |
| 06 K∈{2,4,8} | `06_k_sweep` | K=2,4: **0.6614**, K=8: 0.6089 | $u = \sum_k \pi_k(h) v_k$, K-sweep | convex combination of translations = translation, *K-invariant* ceiling; K=8 over-capacity → anchor 손상 |
| **07** | `07_random_direction_scaled` | **0.6485** | $u = \alpha \hat{v}_{\text{random}}$, α=10 | *direction-agnostic 가설 falsification* — random 은 baseline 과 통계 동등, mean-diff 만 ceiling 도달 |

**Cumulative 결론 (translation-trap 의 *재정렬*)**: $u(h)$ 의 *informed direction* 변형들 (01b mean-diff / 02 learned / 03-06 의 모든 학습 변형) 이 SciFact NDCG@10 ≈ 0.65–0.67 의 **동일 ceiling 에 수렴**. *Random direction* (07) 은 그 ceiling 에 도달조차 못 함. 즉 ceiling 은 *informed direction subspace 의 representational limit* — *informed* 한 방향이면 학습 / 비학습 / 단층 / 다층 / multi-direction 변형 모두 redundant 하게 같은 ceiling 에 닿는다. 그 ceiling 의 본질이 *algebraic family 한계* (form 의 한계) 인지 *information 한계* (frozen-encoder 의 representational limit) 인지의 분리는 *form 자체 를 바꾸는* bilinear M (08) 의 결과로만 결정 가능.

상세: [`REPORT.md`](REPORT.md), 각 [`report/{NN}_*_report.md`](report/).

---

## 🔬 Stage 1 — Translation-trap falsification (07, **완료**)

### 가설 (사후 결과 반영)

이론이 옳다면 *어떤* large-magnitude direction 이든 confused-slice 의 비슷한 magnitude-driven 개선 (≈ +0.064) 을 보여야 한다. mean-diff direction 의 *내용* 은 무관해야 한다.

**결과**: 같은 magnitude (α=10) 의 random direction 의 confused Δ vs baseline 은 +0.011 [-0.006, +0.029] (CI 0 포함, 실용적 0). 01b α=10 의 +0.064 [+0.034, +0.099] 와 비교 시 **paired bootstrap Δ confused -0.0533 [-0.0905, -0.0201] ✗ negative** — 통계적으로 유의하게 worse. **Direction-agnostic 가설 명확히 기각**.

| # | Dir | 검정 | 결과 |
|---|---|---|---|
| 07 | `07_random_direction_scaled` | $v_{\text{random}} \sim \mathcal{N}(0, I/\sqrt{D})$, unit-normalize 후 α=10. SciFact 에서 측정. *01b α=10* (mean-diff 방향) 과 paired bootstrap. | **Outcome 2 (partial fail)**: direction matters. |

**Conditional graph 의 *partial fail* 분기로 진행 결정**:
- Stage 2 (08 bilinear M) 의 critical 검정 *유지* — *form 자체* 의 algebraic 한계 검정은 별도 의의.
- 옛 deferred (mean_diff_pca / projection_out) 의 *informed direction subspace 의 다른 element 로 같은 ceiling 에 닿는지* 의 confirmatory ablation 가치 *부분 회복*.

상세: [`report/07_random_direction_scaled_report.md`](report/07_random_direction_scaled_report.md).

---

## 🧪 Stage 1.5 — Multi-direction router K-sweep (06_k_sweep, **완료**)

### 가설 (사후 결과 반영)

옛 `06_two_directions` (K=2 단일) 의 *ad-hoc single point* 한계 해소. K ∈ {2, 4, 8} sweep 으로 *translation-trap 의 multi-direction 차원에서의 robustness* 검정.

### 결과 (SciFact, seed 42)

| K | Params | NDCG@10 (all) | Δ all vs baseline | Δ confused vs baseline | 효과 |
|---|---|---|---|---|---|
| 2 | 3,074 | **0.6614** | +0.015 ✓ | +0.039 ✓ | ceiling 도달, 통계 동등 |
| 4 | 6,148 | **0.6614** | +0.015 ✓ | +0.045 ✓ | ceiling 도달, 통계 동등 |
| 8 | 12,296 | 0.6089 | **−0.038 ✗ negative** | +0.049 ✓ | **anchor 손상** + confused ceiling 동등 |

**결과 종합** (Outcome: translation-family ceiling 의 *K-invariant* 확정 + over-capacity 의 *해로움*):
- **K=2 와 K=4 의 NDCG@10 all 이 *문자 그대로 동일* (0.6614, 소수점 4 자리)** — 학습 가능 capacity 2 배 증가에도 ceiling 위치 *완전 보존*. capacity-driven ceiling 우회의 부재.
- **K=8 의 anchor 손상** — over-capacity 가 ceiling 우회 못 하고 *easy queries* 손상 (-0.038 ✗).
- **Effective K (perplexity) 가 K 와 무관하게 1.2-1.5 범위** — linear router 의 systematic capacity collapse. K=4 의 dominant 1 개 / K=8 의 dominant 2 개 (rest dead π~10⁻⁶).

**Conditional graph 의 *pass* 분기로 진행 결정** (translation-trap 강확정):
- Stage 2 (08 bilinear M) critical 검정 *진입* — *form 변경* 만이 가능한 lever.
- 옛 ROADMAP 의 "K↑ + router 표현력 + entropy reg" *capacity-only* 가설은 본 sweep 으로 *직접 falsify*. routing_entropy_reg / mlp_router 는 *supplementary ablation* (capacity activation 측면) 으로 강등.

상세: [`report/06_k_sweep_report.md`](report/06_k_sweep_report.md).

---

## 🚀 Stage 2 — Bilinear interaction metric (main novelty, *partial fail* 분기 진행 중)

### 진행 상태 (08 완료, partial fail)

**08 결과 (SciFact, seed 42, r=8, LR=1e-4)**: NDCG@10 all = **0.6439** (≈ baseline 0.6464). Δ vs baseline confused +0.054 [+0.013, +0.097] ✓ positive (≈ K-sweep 의 +0.04). Δ vs 01b α=10 all -0.025 ✗ negative. **Translation family ceiling 위로 못 감**. UV^T 의 *effective rank 1 collapse* 발견 (singular values [2.60, 0.06, 0.035, ...]). *Form 자체 의 lever 부재인지 optimization-driven rank-collapse 인지의 분리* 가 09 (E5 distillation) + 10 (r sweep) 의 critical 질문.

**09 결과 (E5-Mistral Margin-MSE distillation, λ-sweep)**: NDCG@10 all 최고 0.6509 (λ=0.1) 이지만 *Δ confused +0.019 (≪ 08 의 +0.054)*. λ ↑ 면 M ≈ I 로 수렴 (anchor regularizer 로 잘못 작동). E5 의 *noise teacher* (mined HN 의 ~50% 에서 e5_margin < 0) + Margin-MSE 의 *scale mismatch*. **Distillation 이 잘못된 lever** 확정.

**Stage 2 종합 진단** (08 + 09): *form 자체* 의 변경 lever + distillation 모두 *informed subspace ceiling 위로 못 감*. *Frozen-encoder representational limit* 의 정황 증거 → 다음 critical 검정 (1) 10 r sweep 의 effective rank vs r 직접 측정, (2) 18 LoRA on Φ 의 encoder-level upper bound.

상세: [`report/08_bilinear_M_minimal_report.md`](report/08_bilinear_M_minimal_report.md), [`report/09_bilinear_M_e5_distill_report.md`](report/09_bilinear_M_e5_distill_report.md).

### 신규 형식

$$s_M(q, d) = \sum_i \max_j q_i^\top M d_j, \quad M = I + UV^\top, \quad U, V \in \mathbb{R}^{768 \times r}$$

학습 파라미터: $2 \cdot 768 \cdot r$. r = 8 시 12,288 (≪ 50K), r = 16 시 24,576, r = 32 시 49,152 (≤ 50K 상한).

핵심: $q_i^\top M d_j = \langle q_i, d_j \rangle + (U^\top q_i)^\top (V^\top d_j)$. **두 번째 항은 q 와 d 의 *cross-feature* 를 곱셈적으로 결합** — translation family 가 *절대* 못 접근하는 q–d 상호작용 차원.

### Supervision: E5 margin distillation

frozen ColBERT 의 bilinear 상한 안에서 $M$ 의 *어디로 휘어야 할지* 를 가르치는 teacher 가 필요. E5-base 의 train query 별 (q, d⁺, d⁻) margin 을 distillation target 으로:

$$\mathcal{L} = \big\| [s_M(q, d^+) - s_M(q, d^-)] - [m_{\text{E5}}(q, d^+) - m_{\text{E5}}(q, d^-)] \big\|^2 + \text{pairwise margin}$$

E5 의 cross-encoder-quality margin 을 ColBERT 의 frozen bilinear 상한 안에서 *흉내내도록* M 을 학습.

### 실험 sequence

| # | Dir | 검정 | 우선 |
|---|---|---|---|
| 08 ✓ | `08_bilinear_M_minimal` | $M = I + UV^\top$ (r=8, 2,048 params), pairwise margin only. NDCG@10 all = 0.6439 (baseline 0.6464 ≈ 동등), confused +0.054 ≈ K-sweep 의 +0.04. **Ceiling 못 넘음**. UV^T effective rank 1 collapse 발견. | **DONE — partial fail** |
| 09 ✓ | `09_bilinear_M_e5_distill` | E5-Mistral Margin-MSE distill λ ∈ {0.1, 0.5, 1.0}. 결과: NDCG@10 all 의 최고 0.6509 (λ=0.1) 이지만 confused +0.019 (08 의 +0.054 보다 약함). λ ↑ 면 M ≈ I (anchor regularizer). **Distillation 이 잘못된 lever**. | **DONE — partial fail** |
| Robust ✓ | 06 K=2 NFCorpus, 08 r=8 seed×3, 02 unfrozen | 세 가지 robustness audit (자세한 결과는 REPORT.md §7 + 위 changelog 참조). 핵심: *frozen encoder 가 진짜 bottleneck, K-invariant ceiling 은 SciFact-specific, 08 rank-1 collapse 는 seed artifact*. | **DONE — paper-narrative pivot** |
| **10 ✓** | `10_lora_phi` (was 18) | **Stage 3 main novelty** — LoRA adapter (q,v r=8, 295K params, 50K budget 완화 후) 의 *bounded improvement* 검정. Phase 1 (r=1) → 2a (r=8 LR=1e-4 α=2r) → 2b (r=8 LR=5e-5 α=r). **Phase 2b: Δ confused +0.091 ✓ + anchor preserved**. Pre-committed strict 돌파 (CI(Δ all)>0) 미달 — *9K data bottleneck*. | **DONE — bounded improvement** |
| (deprecated) | `10_bilinear_rank_sweep` | 옛 10 후보 — bilinear rank sweep. 본 audit + LoRA 결과 후 *우선순위 ↓ supplementary* (bilinear form 자체는 frozen 한계 안). | deprecated |
| Robust v2 | 10 Phase 2b seed × 3 + NFCorpus | Paper deliverable 의 마지막 robustness check (08 seed-artifact + 06 NFCorpus catastrophic 교훈). pre-commit 따라 *hyperparameter sweep 금지*. | **HIGHEST (next)** |

**Critical falsification (08, 완료)**: r=8 bilinear M 이 ceiling 도달 못 함 (0.6439, baseline 동등). UV^T effective rank 1 collapse 발견 → *form 자체* 의 lever 부재인지 *optimization-driven rank collapse* 인지 분리 필요. 09 (E5 distill) + 10 (r sweep) 의 결과로 결정.

---

## 🌐 Stage 3 — 일반화: cross-dataset 확인 (Stage 2 통과 후)

본 시점의 *single-dataset (SciFact) 한계* 를 직접 해소.

| # | Dir | 검정 | 우선 |
|---|---|---|---|
| 11 | `11_bilinear_nfcorpus` | Stage 2 best config (r 결정) 를 NFCorpus 에서 재현. NFCorpus 의 110K qrels 환경에서도 ceiling 우회 유지하는가? | **HIGH** |
| 12 | `12_bilinear_fiqa` | FiQA 에서 재현. 금융 도메인 의 cross-dataset robustness. | **HIGH** |
| 13 | `13_translation_trap_nfcorpus` | NFCorpus 에서 02 / 06 도 ceiling 가지는지 — translation-trap 이 SciFact-specific 인지 universal 인지 검정. | medium |

---

## 🤖 Stage 4 — Cross-model: translation-trap 의 일반성

본 paper 의 *대주제 의 일반화* 측면 — translation-trap 이 ColBERT 만의 현상이 아니라 *dense retriever 전반* 의 algebraic 한계임을 입증.

| # | Dir | 검정 | 우선 |
|---|---|---|---|
| 14 | `14_translation_trap_e5` | E5-base 에서 02 형식 (single direction) 의 translation 적용. ColBERT 와 같은 ceiling 패턴 보이는가? | core |
| 15 | `15_bilinear_M_e5_encoder` | E5 위에 bilinear M 적용. cross-encoder 재현. | core |
| 16 | `16_translation_trap_bge` | BGE-small 에서 같은 검정. cross-model 의 second data point. | extended |

---

## 🎯 Stage 5 — 더 위로 (Stage 2-3 의 결과 따라 분기)

bilinear M 이 *부분* 성공이면 (예: +0.02 정도 개선이지만 cross-encoder 와는 여전히 큰 gap), expressivity 사다리의 다음 칸:

| # | Dir | 형식 | 비용 |
|---|---|---|---|
| 17 | `17_nonlinear_interaction` | $s(q, d) = \sum_i \max_j \text{MLP}([q_i; d_j; q_i \odot d_j])$ — bilinear 보다 1 단계 더 표현력 ↑ | medium |
| 18 | `18_lora_phi` | LoRA on Φ (ColBERT encoder 의 lightweight finetune). *frozen* 제약 완화 — DPI 상한 자체 ↑. paper 의 *최종* upper bound 검정. | high |

---

## 🛡 Stage 6 — Statistical robustness (Stage 2-5 후)

| # | Dir | 검정 | 우선 |
|---|---|---|---|
| 19 | `19_seed_robustness` | seed × 3 (42 / 1337 / 2024) × Stage 2 best. CLAUDE.md §3.7 만족. | core |
| 20 | `20_loocv_held_out` | 3 dataset (SciFact / NFCorpus / FiQA) 의 LOOCV. domain transfer. | core |
| 21 | `21_dynamic_hn` | ANCE 류 dynamic mining. 모든 학습 실험의 train-overfitting 패턴 해소 후보. | extended |

---

## 🗂 Deferred (translation family 의 confirmatory ablations)

옛 master plan 의 미실행 실험들. 새 narrative 에서는 *translation-trap 이론의 supplementary verification* 으로 강등. paper 의 main 결과 아니며, *시간 여유 시 supplementary appendix* 로 보강.

| 후보 | 옛 위치 | 새 역할 |
|---|---|---|
| `layer_sweep` | architectural | translation 의 다른 layer 도 같은 ceiling 검증 |
| `sign_flip` | architectural | translation 의 sign 반전도 같은 family → 같은 ceiling |
| `random_vector` (학습) | architectural | learned 와 random 의 차이 — 07 가 *비학습* random 검정 후 학습 random 도 비교 |
| `init_sweep` | architectural | translation 이라 init 무관 예상 |
| `post_projection` | architectural | 128-d translation 도 같은 family |
| `bias_init_sweep` | architectural | 03 의 변형, translation 안 |
| `gate_off` | architectural | 02 와 동치, redundant |
| `gate_capacity` (MLP gate) | architectural | translation 안 |
| `dense_layers` | architectural | 5+ layer 의 translation, family 안 |
| `layer_subset` | architectural | translation 의 다른 부분집합 |
| `mean_diff_pca` | form variant | translation w/ PCA direction — direction-agnostic 가설 추가 검정 |
| `projection_out` | form variant | $h - \text{proj}_v(h)$ 도 translation. LEACE 와의 form-level 비교 |
| `combined_form` | form variant | projection + gate 결합 — translation 안 |
| `anchor_reg_sweep` | training | 02 의 anchor 자연 보존이라 가치 ↓ |
| `routing_entropy` | router | 06 의 router collapse 해소 시도 — translation family 안 |
| `mlp_router` | router | translation 안 |
| `kmeans_init` | initialization | translation 안 |

---

## 📊 통과 기준 (paper-grade)

| 영역 | 통과 기준 | 현 상태 |
|---|---|---|
| Translation-trap ceiling 증명 | 02–06 가 같은 ceiling 보임 + 07 의 random 검정 | 02–06 ✓, 07 ✓ *partial fail* (direction matters → ceiling 은 *informed subspace 의 representational limit*) |
| Bilinear M 의 ceiling 우회 | r=8 단독으로 0.665 *유의 초과* (paired bootstrap CI 하한 > 0.665) | 08 미실시 |
| E5 distillation 의 가치 | 09 가 08 보다 *유의* 개선 + bilinear vs cross-encoder gap quantification | 09 미실시 |
| Cross-dataset 일반화 | 11, 12 에서 SciFact 와 정성 일치 (informed subspace ceiling 발생 + bilinear M 우회) | 미실시 |
| Cross-model 일반화 | E5 / BGE 에서도 같은 패턴 + bilinear M 우회 | 미실시 |
| Statistical robustness | seed × 3, LOOCV 모두 stable | 미실시 |

---

## 🔁 Conditional execution graph

```
                      ┌──────────────┐
                      │ 07 falsify   │  ✓ DONE
                      └──────┬───────┘
                             │
              ┌──────────────┼──────────────┐
       pass (direction-agnostic 확정)   fail (direction matters)
                                              │
                                              ▼
                              ┌──────────────────────────────┐
                              │ Stage 1.5 (06_k_sweep K∈2,4,8)│  ✓ DONE
                              │ K-invariant ceiling 확정    │
                              └─────────┬────────────────────┘
                                        ▼
                              ┌──────────────────────────────┐
                              │ Stage 2 (08 bilinear M, r=8) │  ✓ DONE
                              │ ceiling 못 넘음 + rank-1 collapse │
                              └─────────┬────────────────────┘
                                        │  ← 현 분기 (partial fail)
                                        ▼
                  ┌────────────────────────────────────┐
                  │ 09 E5 distill (rank-collapse 해소?)  │
                  │ 10 r sweep (r ∈ {1, 4, 16, 32, 64}) │
                  └─────────┬──────────────────────────┘
                            │
                ┌───────────┴───────────┐
            pass (ceiling 초과)     fail (모두 ceiling)
                │                       │
                ▼                       ▼
    ┌──────────────────┐    ┌──────────────┐
    │ 11-12 cross-ds   │    │ 18 LoRA on Φ │
    │ 14-15 cross-model│    │ (encoder lim.)│
    │ 19-20 stat robust│    └──────────────┘
    └──────────────────┘
```

---

## 🔄 현 진행 중 + 예약된 실험 (2026-05-24 evening)

> **Note**: Exp 13/14 의 *number assignment* 정정 — 옛 queue 의 "Exp 13 = (f) Difficulty-aware" / "Exp 14 = (e) Frozen-direction anchor" 는 *사용자 confirm 후* 순서가 swap 되어 **Exp 13 = Frozen-direction anchor**, **Exp 14 = Difficulty-weighted HN** 으로 최종 확정 (`report/_exp13_14_pre_commit.md`). 본 row 가 권위 있는 numbering.

### Queue 상태 (auto, GPU)

| Phase | 실험 | 상태 | ETA |
|---|---|---|---|
| 1 | Combined M1b + Exp 11 SciFact × 3 seeds | ✅ 완료 (3-seed mean: Δ all +0.015 ± 0.002, mildly antagonistic vs M1b alone +0.021) | done |
| 2 | FN+EP variant SciFact × 3 seeds (Exp 11 의 `--fn-denoise` flag) | ✅ 완료 (3-seed mean: Δ all +0.027 ± 0.009, redundant ≈ Exp 11 λ=1) | done |
| 3 | **Exp 13** (frozen-direction anchor, per-token cosine, λ_dir=1.0) SciFact × 3 seeds | ✅ 완료 (3-seed mean: Δ all +0.030 ± 0.002 **3/3 strict**, Δ confused +0.092, Δ easy −0.021 — branch (b) frontier-shared with Exp 11) | done @ 22:10:24 |
| 4 | **Exp 14** (difficulty-weighted HN, α_w=10) SciFact × 3 seeds | ✅ 완료 (3-seed mean: Δ all +0.006 ± 0.003 **3/3 CI 0 포함**, Δ confused +0.085, Δ easy −0.060 — branch (c) 변형, data-side family equivalence with Exp 12) | done @ 22:50:09 |
| 5 | **🏁 Exp 13+14 queue 종료** + 5 root docs final update (REPORT §6.1 + §7.3.g/h + §7.4.1 6-lever, RESEARCH #6/#7, CHANGELOG #12/#13, ROADMAP) | ✅ 완료 | done |
| 6 | **Diagnostic B on Exp 13** (per-token cosine anchor mechanism verification) | ✅ 완료 — cos(LoRA, frozen) = 0.824, *soft equilibrium attractor* 입증 | done |
| 7 | **Diagnostic B on Exp 14** (data-side family internal representation) | ✅ 완료 — bimodal seed pattern + 6-lever × internal/external mapping 완성 | done |
| 8 | **Exp 15 (Conditional LoRA) 4-diagnostic chain** — (α)/(β)/(γ)/(δ) sequential falsification | ✅ 완료 — *frontier-breaking minimal realization empirically falsified*, future work F1/F2/F3 으로 정리 | done @ 2026-05-25 |
| 9 | **🏁 모든 sub-experiment + diagnostic chain 종료** | ✅ STOP rule 준수, paper writing phase 진입 | done |
| 10 | **Exp 16 (multi-layer per-token anchor)** — anchor scope ablation, layers {0,3,6,9,12}, 3 seeds | ✅ 완료 (3-seed mean Δ all +0.004 ± 0.006, 3/3 CI 0, Δ easy −0.052 — **branch (c) over-restriction confirmed**) | done @ 2026-05-25 01:39 |
| 11 | **Diagnostic B on Exp 16** (per-layer anchor proximity + eff_rank) | ✅ 완료 — *loss budget dilution + intermediate redundancy* mechanism direct evidence | done |
| 12 | **Spine ablations (reviewer Tier 1 + B1 + C1)** — M1b Δeasy 실측 / anchor incremental / sanity / split consistency | ✅ 완료 — M1b Δ easy 정정 (−0.017 vs 이전 ~−0.05), anchor mechanism interpretation 정정 | done |
| 13 | **🏁 anchor scope ablation + spine ablations 종료** | ✅ §3.8 strict 충족, paper writing 진입 가능 | done |

### 예약 안 된 잠재 추가 실험 (post-Exp 14, STOP rule 준수 시 *전부 deferred*)

| Priority | 실험 | 이유 | Cost |
|---|---|---|---|
| 1 | **Higher λ Exp 11 의 cross-dataset (NFCorpus / FiQA)** | Current strict net+ 가 SciFact-only. *Cross-dataset robustness* 검정 — *paper-grade cross-domain claim*. | ~60 min × 2 datasets |
| 2 | **Exp 13 cross-dataset (NFCorpus / FiQA)** | NFCorpus puzzle 의 *cross-regime 전이성* 검정 — direction mechanism 이 catastrophic regime 특이인지. *STOP rule 따라 본 paper 미실시*, future work. | ~60 min × 2 datasets |
| 3 | **(a) Initial-loss-aware LR** for cross-dataset | M1 의 *zero contribution* lesson; *cross-domain* 에서 *trajectory + final-state* 모두 효과 가능성. | ~45 min |
| 4 | **FN-denoised NFCorpus / FiQA** | Exp 12 의 SciFact 단독 (나-2 difficulty dominant) 결과 의 cross-dataset 확인. E5 doc emb 추가 추출 (~10 min × 2 datasets) + training (45 min × 2). | ~120 min |
| 5 | **Higher principle (measure → adapt)** single-rule proof of concept | Methodology contribution; initial-loss measure → auto LR scaling + dynamic anchor strength. | ~120 min |

### Paper-writing 단계 (after current queue 종료)

| Task | 비고 |
|---|---|
| §5f update (Higher λ=5 의 *best lever* status + 4-lever ranking) | 진행 중 (이번 turn) |
| §5f.7 "Cross-domain robustness gap" subsection | NFCorpus/FiQA 의 *no strict net+* 명시 + future work |
| 캐비엇 1/2 final closure 명시 | 5 root docs 의 consistent narrative |
| Future-work proposals 의 (e)/(f)/(a) inclusion | paper future work section |

---

## Changelog

| Date | Change |
|---|---|
| 2026-05-25#2 | **Exp 16 (multi-layer per-token anchor, layers {0,3,6,9,12}) 3-seed complete — *branch (c) over-restriction confirmed***. 3-seed mean Δ all = **+0.004 ± 0.006** (3/3 CI 0, NOT strict), Δ confused = +0.071 ± 0.004 ✓, Δ easy = **−0.052 ± 0.008** ✗. *Anchor 의 5-layer 확장 (CLAUDE.md §1.3 prior diagnostic finding 의 direct architectural translation) 실패* — 모든 metric 에서 Exp 13 single-layer 대비 *명백 열등* (Δ all 1/8, Δ easy damage 2.5×). Diagnostic B 의 *loss budget dilution + intermediate-layer redundancy* mechanism direct evidence: L0-L6 redundant constraint (cos ≥ 0.99), L12 insufficient constraint (cos 0.697 vs Exp 13 0.824, token eff_rank 4.6 vs 9.01). **Anchor-side family 의 optimal scope = final layer only**. **Spine ablations (reviewer Tier 1 + B1 + C1)**: A1 M1b Δ easy 실측 −0.017 ± 0.003 (이전 추정 ~−0.05 정정), A2 anchor incremental Δ over Phase 2b LoRA = *전적으로 easy preservation, no incremental confused gain* (anchor mechanism interpretation 정정), B1/C1 sanity ✓. §3.8 ablation completeness strict 충족. Pre-commit `report/_exp16_pre_commit.md`, 상세: `report/16_multilayer_anchor_report.md`, `report/_spine_ablations.py`. STOP rule 준수, 추가 실험 미실시. |
| 2026-05-25 | **Exp 15 (Conditional LoRA) 4-diagnostic chain complete — *frontier-breaking hypothesis empirically falsified***. Sequential cheap diagnostics (~30 min total): (α) score-margin AUC = **0.836** (router signal 강함), (γ) oracle test-time conditional Δ all = **+0.048 ± 0.008** ✓ (perfect routing ceiling real, anchor-side +0.030 의 1.58×), (β) confused-only triplet training Δ all = **−0.387** ✗ catastrophic (training distribution dependency), (δ) margin-routed Phase 2b Δ all = **+0.011 ± 0.007** (realistic < anchor-side, *frontier-breaking minimal realization falsified*). **6-lever framework 유지** — Exp 15 의 realistic 형태가 framework 의 inferior 구성원, frontier 추가 lever 아님. *Frontier 가 inference-time conditional routing (AUC 0.84) 에도 robust* = paper main contribution 강화. Elaborate Exp 15 (learned router F1 / end-to-end joint F2 / reranker 형태 F3) 는 §9.3 future work 로 정리. 상세: `report/15_exp15_diagnostics_report.md` + 2 figures. STOP rule 준수, 추가 실험 미실시. |
| 2026-05-24#13 | **Exp 14 (continuous sigmoid weighting, α_w=10) 3-seed complete** — SciFact 3 seeds 모두 Δ all CI 0 포함, 3-seed mean Δ all **+0.006 ± 0.003 (NOT strict)**, Δ confused +0.085 ± 0.022 ✓, Δ easy −0.060 ± 0.020 ✗. **Branch (c) 변형 — softer Phase 2b, sub-binary on Δ all, but Δ confused not attenuated**. *Data-side family 의 binary ≈ continuous equivalence* 확정 — Exp 12 (binary FN cut) 와 statistically equivalent frontier. **Three-frontier structure** (anchor-side upper / data-side weighting lower / data-side substitution unique) → paper §7.4.1 **6-lever framework** 으로 완성. α_w=10 의 unstable variance (Δ confused std 0.022, anchor-side 의 3-5×) + val NDCG late-best (ep3) 관찰. STOP rule 준수 (α_w sweep / variant / cross-dataset 금지). Pre-commit `report/_exp13_14_pre_commit.md`, 상세: `report/14_difficulty_weighted_hn_report.md` + 6 figures. |
| 2026-05-24#12 | **Exp 13 (frozen-direction anchor, per-token cosine, λ_dir=1.0) 3-seed complete** — SciFact 3 seeds 모두 strict, 3-seed mean Δ all **+0.030 ± 0.002 ✓ (3/3 strict, Exp 11 의 2/3 보다 robust)**, Δ confused +0.092 ± 0.007 ✓, Δ easy −0.021 ± 0.003 (branch (a) 임계 −0.020 을 0.001 차이로 miss). **Branch (b) — Exp 11 과 frontier 공유** 확정. *Anchor-side family 의 frontier 강건성* paper-grade 증거 — Sim Frobenius² (rotation-invariant) 와 per-token cosine (rotation-sensitive) 가 *수학적으로 다른* constraint 임에도 *통계적으로 구분 안 되는* trade-off frontier. **5-lever framework** (data-side: Exp 12, M1b / anchor-side: Exp 11, Exp 13) 으로 §7.4.1 확장. Pre-commit `report/_exp13_14_pre_commit.md`, 상세: `report/13_frozen_direction_anchor_report.md` + figures. STOP rule 준수 (λ_dir sweep / variant 금지). |
| 2026-05-23 | 초안 작성 (32 실험, 5 그룹 구조). |
| 2026-05-23 | Sequential renumbering, "Phase" 용어 제거. |
| 2026-05-23 | **Translation-trap pivot (대대적 수정)**: 실험 00–06 의 algebraic 진단 후 *전면 개편*. 옛 master plan 의 translation family ablation 들을 *confirmatory* 등급으로 강등; bilinear interaction metric $M = I + UV^\top$ + E5 distillation 을 main novelty 로 격상. Conditional execution graph 도입 (07 falsification → 08+ pivot). 옛 thesis 도 *upgrade* — "lightweight steering" 에서 "translation-trap algebraic 진단 + bilinear 우회". 근거: 02–06 의 모든 학습 실험이 같은 ceiling (0.665) 에 수렴한 사실이 capacity/training-signal 한계가 아닌 *algebraic form* 한계임을 시사. |
| 2026-05-23 | **07 완료 → conditional graph 의 *partial fail* 분기 확정**. random direction × α=10 (NDCG@10 0.6485, Δ vs 01b α=10 confused -0.0533 ✗) 으로 *direction-agnostic 가설* 명확히 기각. 새 narrative: ceiling 은 *informed direction subspace 의 representational limit*. Stage 2 (08 bilinear M) critical 검정 유지 + 옛 deferred (mean_diff_pca / projection_out) 의 *informed subspace 의 다른 element* 로서의 confirmatory 가치 부분 회복. |
| 2026-05-23 | **Stage 1.5 (06_k_sweep) 완료** — `06_two_directions` 의 *ad-hoc single point* 한계 해소 위해 K ∈ {2, 4, 8} sweep. **K=2 와 K=4 의 NDCG@10 all 이 *문자 그대로 동일* (0.6614)** — translation family ceiling 의 *K-invariant* 강확정. **K=8 의 anchor 손상** (Δ all -0.038 ✗) — over-capacity 의 *역효과* 첫 데이터 증거. Effective K 가 K 와 무관하게 1.2-1.5 — linear router 의 systemic capacity collapse. 옛 ROADMAP 의 "K↑ + router 표현력 + entropy reg" *capacity-only* 가설 직접 falsify. **Stage 2 (08 bilinear M) 즉시 진입**. routing_entropy_reg / mlp_router 는 supplementary 등급. |
| 2026-05-23 | **Stage 2 (08 bilinear M r=8) 완료 → partial fail** — NDCG@10 all = 0.6439 (baseline 0.6464 ≈ 동등), Δ confused +0.054 ≈ K-sweep 의 +0.04. **Translation family ceiling 위로 못 감**. 새 발견: UV^T 의 effective rank 1 collapse (singular values [2.60, 0.06, 0.03, ...]) — r=8 의 latent capacity 미활용. K-router 의 effective K collapse 와 평행 패턴. **Form 자체 의 lever 부재인지 optimization-driven collapse 인지** 의 분리는 09 (E5 distill) + 10 (r sweep) 결과로. *Zero-init pathology + LR sensitivity* 도 기술적 noteworthy. |
| 2026-05-23 | **09 (E5-Mistral Margin-MSE distill) λ-sweep 완료 → distillation 이 잘못된 lever** — λ ∈ {0.1, 0.5, 1.0} 모두 ceiling 우회 미달. λ=0.1 best NDCG@10 all = 0.6509 (baseline 보다 약간 ↑) 이지만 Δ confused +0.019 (≪ 08 의 +0.054). λ ↑ 면 M ≈ I 로 수렴 (anchor regularizer). Rank-collapse 부분 해소 (σ₁/σ₂ 42 → 4.1) 와 magnitude 감소 (‖UV^T‖ 2.6 → 0.49) 의 trade-off. E5 의 *noise teacher* (mined HN 의 ~50% 가 e5_margin < 0) + *scale mismatch* 가 원인. **Stage 2 form-change + distillation 두 lever 모두 ceiling 우회 미달 → *frozen-encoder representational limit* 의 정황 증거**. 다음 critical: 10 r sweep + 18 LoRA on Φ. |
| 2026-05-23 | **Robustness audit 3 가지 (NFCorpus / seed×3 / unfrozen) 완료 → *Paper narrative 의 근본적 재정렬***. (1) **06 K=2 NFCorpus** Δ all **−0.250 ✗ catastrophic** (SciFact 의 +0.015 ✓ 와 반대) — *K-invariant ceiling claim 의 SciFact-specific 한계*. (2) **08 seed × 3**: seed 42 만 rank-1 collapse + Δ conf +0.054, seed 1337/2024 는 M≈I + Δ conf ≈0 → *seed-specific artifact*. (3) **02 unfrozen ColBERT (110M params)** Δ conf **+0.252 ✓** (5× lift over frozen 의 max +0.054) — *Frozen-encoder 가 진짜 bottleneck 직접 증거*. **새 main contribution 후보**: 50 K LoRA budget 안에서 encoder representational limit 의 *어디까지 회복* 가능한지의 정밀 분석. *Stage 3 (18 LoRA on Φ) 직행*. |
| 2026-05-23 | **10 LoRA on Φ (was 18, Stage 3) 3-phase sweep 완료 → bounded improvement framing**. 50K budget 완화 (사용자 결정). Phase 1 (r=1, 36K): all -0.052 ✗ / conf +0.038 (≈). Phase 2a (r=8, 295K, LR=1e-4 α=2r): all -0.059 ✗ / conf +0.080 ✓. **Phase 2b (r=8, 295K, LR=5e-5 α=r, pre-committed early-stop=val_all): all -0.010 (CI 0 포함, anchor preserved) + conf +0.091 ✓** (frozen-side max 의 1.7×, 02 unfrozen 의 36% 회복). Strict 돌파 (CI(Δ all) > 0) 미달 — *9K SciFact triplet data bottleneck*. Pre-commit 따라 hyperparameter sweep 중단, *bounded improvement* narrative 채택. *LoRA capacity utilization 균등* (rank-1 collapse 반대 양상). 다음 robust check: 10 Phase 2b seed × 3 + NFCorpus. |
| 2026-05-24 | **10 Phase 2b seed × 3 + cross-method universal rank-collapse 통합 → Paper main contribution 의 *최종 재정렬***. (1) **3-seed sweep (42/1337/2024)**: Δ conf **+0.091 / +0.097 / +0.123** ✓ — 3 seeds 모두 통계 유의 + anchor 보존. *3-seed mean Δ conf +0.104 ± 0.017* (02 unfrozen 의 41% 회복). *08 의 seed-artifact 와 완전 반대 — robust*. (2) **Cross-method universal rank-collapse 발견**: 06 K-router (eff K 1.4), 08 bilinear M (eff rank 1.01), 10 LoRA (per-adapter mean 1.71) 모두 *per-position rank ~1-2* 의 universal collapse. (3) **Spatial multiplicity 가 진정한 lever**: position 수 (06/08: 1, 10 LoRA: 24, 02 unfrozen: full) log-scale 의 monotonic correlation with confused lift (+0.045 → +0.104 → +0.252). **새 paper main contribution: *universal per-position rank-collapse + LoRA's spatial multiplicity escape***. (4) **Clean ColBERT-finetune baseline (02 --no-steering)**: Δ conf **+0.260 ✓** (02 unfrozen 의 +0.252 와 essentially 동일) — *v=0 hook 영향 negligible*, reviewer 의 "v=0 hook 이 학습 신호 추가했냐" 공격 해소. NDCG@10 all 0.6924 (+0.046 vs baseline, CI 하한 −0.002 — *strict 돌파 직전*). (5) **10 Phase 2b on NFCorpus** (same config + --max-triplets 9,190): Δ all **−0.320 ✗** catastrophic (06 K=2 NFCorpus 의 −0.250 보다 더 심함). NDCG@10 all 0.0094, ep1 rank loss 4.47 (SciFact 의 7×). *Same SciFact-tuned hyperparameter 가 NFCorpus 의 strong-HN regime 에 incompatible*. **Paper limitations 명시**: hyperparameter sensitivity 가 dataset-specific, numerical lift 는 SciFact-specific. Pre-commit 의 *시간 여유에도 불변* (외부 reviewer agent 채택) — hyperparameter sweep 금지. |
| 🚫 2026-05-24#9 (POST-HOC EXCLUDED) | **Exp 11 extensions + FN+EP variant complete (post-hoc exploratory, NOT main paper)** — 본 row 의 3 묶음 (Higher λ=5, Combined M1b+Exp 11, FN+EP variant) 은 *test 결과 본 후 generative question 으로 발의된 post-hoc exploratory*. **Pre-commit timing**: Exp 11 (λ=1) 3-seed 결과 본 *후* `report/_exp11_extensions_pre_commit.md` 작성. **Main paper claim base 에서 제외** (9 runs). Reviewer recommendation 따라 *Exp 11 (λ=1) 의 2/3 strict partial 로 honest 종착*. Raw data 보존 (`outputs/...`), chronological record (`report/_overnight_results.md`). 결과 자체는 *historical record* — 인용 안 함. |
| 🚫 2026-05-24#8 (POST-HOC EXCLUDED) | **Higher λ Exp 11 (λ=5) 3-seed complete** — 본 row 는 *post-hoc exploratory* (Exp 11 λ=1 결과 본 후 추가). Numerical claim 인용 금지. |
| 🚫 2026-05-24#7 (POST-HOC EXCLUDED) | **Exp 11 extensions launched** — *post-hoc exploratory* 묶음 launch record. 본 row 의 모든 lever / variant / claim 은 main paper 에서 제외. |
| 2026-05-24#6 | **Diagnostic B on new checkpoints — mechanism direct verification** (`report/_repr_collapse_new_ckpts.py`, 10 conditions CPU). **4 findings**: (1) **Exp 12 ≈ Phase 2b at collapse** (eff_rank doc 1.22 ≈ 1.14, 3-seed) — *FN removal 만 collapse zero change*, (나-2) difficulty dominant collapse-level 추가 증거. (2) **M1+M1b ≡ M1b alone at collapse** (SciFact 7.05 ≈ 7.12, NFCorpus 1.05 ≈ 1.06) — *M1 추가 기여 zero* 둘 다 NDCG + collapse 확정. (3) **Exp 11 의 selective token-level preservation 직접 확인**: token eff_rank 1.58 → ~9.6 (**6× recovery**), doc 1.14 → ~1.9 — *loss = token sim matrix 직접 규제 = 직접 보존*, direct mechanism evidence. (4) **M1b collapse 감소 3-seed robust** (eff_rank doc 6.73/7.29/7.33). NFCorpus *direction matters* puzzle 강화 (M1+M1b 도 eff_rank 1.05 임에도 NDCG 74 % recovery → direction > magnitude 의 추가 evidence). |
| 2026-05-24#5 | **🎯 Exp 12 (FN-denoised mined-HN) → 캐비엇 1 결정적 disambiguation**. (1) `data/e5_teacher/extract_train_docs.py` 신규 + E5-Mistral encode 5183 SciFact train docs (41 MB cached, 3660 sec). (2) `experiments/12_fn_denoised_hn/` + pre-commit `report/_exp12_pre_commit.md` (3 branches: 나-1 noise / 나-2 difficulty / both). (3) **SciFact × 3 seeds**: 36.5 % mined HN 이 likely FN (e5_margin ≤ 0) 제거, cleaned 5832 triplets 학습. **3-seed mean**: Δ all −0.004 ± 0.005 (CI 0 all 3), Δ confused +0.080 ± 0.004 ✓, Δ easy −0.073 ± 0.005 ✗. **🎯 (나-2) Difficulty dominant + (나-1) noise minor 확정**: redistribution 거의 동일 (Δ easy −0.073 ≈ Phase 2b −0.085, FN removal 만으로 14 % 만 recovery), Δ all CI 0. ⇒ **Hard-contrast over-correction 이 catastrophic / redistribution 의 *주요* mechanism**. M1b net+ 는 *easy contrast 효과* (hard 회피, noise 제거 부수적). **Paper narrative 근본 정정**: sole mechanism = hard-contrast supervision over-correction → 4-lever framework (Phase 2b baseline / Exp 12 hard+clean / M1b easy+clean / Exp 11 hard+selective preservation). |
| 2026-05-24#4 | **Overnight autonomous experiments → sole-mechanism (supervision root) + two-lever partial resolution**. (1) **🎯 M1b SciFact 3-seed strict robust**: Δ all +0.015 / +0.022 / +0.026 ✓ (모두 strict), 3-seed mean +0.021 ± 0.005 ✓. **캐비엇 2 (08-style seed-artifact) fully 해소**. (2) **🎯 M1b NFCorpus 3-seed gap recovery robust**: 74 % ± 7, 3-seed mean Δ all −0.086 ± 0.020 ✗ (NOT net+). (3) **M1+M1b combined → M1 contribution = ZERO** (SciFact +0.020 = M1b alone, NFCorpus −0.083 = M1b alone). **Optimization root = red herring**. (4) **🎯 Exp 11 (relational easy preservation, λ=1.0) branch (a) partial**: 3-seed mean Δ all +0.029 ± 0.005 (2/3 strict), Δ confused +0.101 ± 0.010 ✓ (Phase 2b 의 +0.104 fully preserved!), Δ easy −0.031 ± 0.018 (63 % 감소 vs Phase 2b −0.085). **Two levers** for partial redistribution resolution: M1b (general clean, strict robust) 와 Exp 11 (selective preservation, higher confused). **Paper sole-mechanism**: catastrophic = mined HN noise (supervision root only). 캐비엇 1 (clean ≠ easy) *여전히* unresolved → FN-denoised mined-HN full-strength replication 필요. |
| 2026-05-24#3 | **Mediation 1 (warmup+clip) + Mediation 1b (in-batch neg) → optimization vs supervision root disentangling + 🎯 첫 strict net 향상 시그널 (single seed) + 🎯 cross-dataset 74 % catastrophic-gap 회복 (NOT net+)**. ⚠️ 캐비엇 1 (clean ≠ easy) *해소 안 됨* — easy in-batch 도 방향 교정 가능, FN-denoised mined-HN full-strength 여전히 필요. ⚠️ NFCorpus M1b 는 *net+ 아님* (Δ all 여전히 −0.084). ⚠️ Mediation sanity check (M1/M1b NDCG 재현) 미실시 — claim A (same eff_rank, different NDCG) 단단함의 근거. (1) **M1 on 3 datasets**: SciFact Δ all −0.012 (≈ Phase 2b), NFCorpus Δ all −0.319 ✗, FiQA Δ all −0.346 ✗ — *test NDCG 측면 catastrophic 그대로*. (2) **M1 train trajectory 명확한 효과**: NFCorpus ep1 val_all 0.073 → 0.140 (1.9× ↑), FiQA 0.090 → 0.257 (2.86× ↑). *Warmup 가 collapse 지연*, 단 post-warmup full LR 가 ep2/3 재현. *Optimization root 부분 지지*. (3) **M1b SciFact (signal, 확정 아님)**: NDCG@10 all = **0.6613**, Δ all = **+0.015 [+0.001, +0.029] ✓ strict positive *시그널*** (pre-committed CI(Δ all) > 0 기준 첫 충족 — single seed razor-thin), Δ confused +0.055 ✓. **Phase 2b redistribution 깨뜨림** — zero-sum → non-zero net 향상. **⚠️ 두 캐비엇**: (i) *clean ≠ easy* — in-batch 는 noise 제거 + difficulty 감소 *동시* → mechanism confounded (FN-denoised mined-HN replication 가 disambiguate). (ii) *seed 42 단독* + CI razor-thin → 08 seed-artifact 시나리오, 3-seed 전 *확정 아님*. (4) **🎯 M1b NFCorpus**: NDCG@10 all = **0.2459 (baseline 0.330 의 74.5%)**, Δ all = −0.084 [−0.105, −0.064] ✗ (catastrophic 의 74% 회복), Δ confused = −0.013 (CI 0 포함, *baseline 회복*). **ep1 val_all = 0.376 > baseline** — strict positive 가능 했음, LoRA snapshot 한계로 ep3=0.259 사용. *Mined HN noise 가 cross-dataset 의 universal 원인 부분 지지*. (5) **Step 0 Δeasy 측정**: 수학 예측 −0.086 ↔ 실측 −0.085 ± 0.010 (99 % match) — Phase 2b 의 "anchor preserved" = *redistribution* 확정. (6) **Exp 11 (easy preservation, λ_anc>0)** code + pre-commit prediction 준비 완료, 사용자 confirm 후 queue 종료 시점 launch. (7) FiQA M1b 진행 중 — cross-dataset universality 최종 확정. (8) Paper narrative 정밀화: *Catastrophic = mined HN noise (supervision root 주요) + optimization 폭주 (optimization root 부분), additive*. |
| 2026-05-24#2 | **FiQA cross-dataset + Diagnostic B (representation collapse) + Sanity check → Catastrophic mechanism 의 *direction misalignment* 정밀화**. (1) **10 Phase 2b on FiQA** (same config + --max-triplets 9,190): NDCG@10 all = **0.0005** (baseline 0.347 의 0.15 %, *literal 0% retrieval*), Δ all **−0.347 [−0.374, −0.319] ✗** catastrophic (NFCorpus 의 0.0094 보다 더 심함), Δ conf −0.147 ✗. **2 / 2 cross-dataset catastrophic 확정** — single-dataset artifact 가설 명확히 기각. (2) **Diagnostic B representation collapse** (`report/_repr_collapse_diagnostic.py`, n=500 docs per corpus, random pair cosine + singular-spectrum effective rank): 3 dataset 모두 LoRA Phase 2b 에서 doc-pair cos ≈ 0.99, eff_rank ≈ 1 (NFCorpus: 11.7 → 1.1, FiQA: 23.6 → 1.1, SciFact: 10.6 → 1.2) — **universal extreme collapse**. (3) **Sanity check** (`_repr_collapse_sanity.py`, reviewer 의 *"rank-1 → random retrieval"* puzzle 검정): 진단이 로드한 `module_final.pt` 의 test NDCG@10 재현 — *3 / 3 match* (SciFact 0.6367 = 0.6367, NFCorpus 0.0094 = 0.0094, FiQA 0.0005 = 0.0005). 가설 A (best vs final 불일치) / B (LoRA α scaling 불일치) 모두 기각. **Collapse 가 진짜** + **rank-1 puzzle 도 진짜** — *SciFact 의 eff_rank ≈ 1.15 *상태에서* NDCG 0.6367 실제 발생*. (4) **그러나 SciFact 는 catastrophic *아님*** (Δ all = −0.010 ≈ baseline) — *collapse* ↔ *catastrophic NDCG* 의 1:1 대응 없음. ⇒ **Catastrophic mechanism = universal collapse + *direction* misalignment**. *Collapse magnitude* 는 necessary but not sufficient; *task-alignment* 가 sufficient 조건의 추가 요인. SciFact 의 collapse direction 은 task ranking 신호와 align (1.15-dim residual 이 MaxSim 의 per-token max 로 amplify → sufficient); NFCorpus/FiQA 는 baseline 약 → ep0 loss 7× → supervision distortion → collapse direction wrong. (5) **Paper section 골격** `report/_catastrophic_failure_section_draft.md` (pre-commit): "Catastrophic Failure as Representation Collapse: Disentangling Optimization vs Supervision". *Disentangling experiment* — Mediation 1 (warmup + grad_clip, optimization root) vs Mediation 1b (in-batch negative, supervision root), 각 single-rule + result-blind 1 run/dataset. |
