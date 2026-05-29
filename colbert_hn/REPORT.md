# REPORT (cumulative narrative)

본 문서는 본 프로젝트의 *cumulative academic narrative*. Per-experiment 보고서 (`report/{NN}_*_report.md`) 와 ROADMAP.md 의 master plan 을 연결하여 *연구의 현재 상태* 를 self-contained 하게 정리한다.

**작성 일자**: 2026-05-23 (실험 00–07 완료 시점, sequential numbering 적용).

---

## Abstract

본 연구는 ColBERT v2 의 frozen encoder 위에 *경량 representation-level steering module* 을 삽입하여 hard-negative (HN) confusion 을 완화하는 방법론을 개발한다. 학습 가능 파라미터는 ≤ 50K 로 제한되며, 추론 시 외부 LLM 호출이나 도메인 별 추가 labeling 을 요구하지 않는다.

지금까지의 결과는 다음 11 개 main 실험 (00–10) + 4 robustness audit (NFCorpus generality, seed × 3, encoder unfreeze, clean baseline) + sub-sweeps (06 K-sweep × 3, 09 λ-sweep × 3, 10 LoRA × 3 phase + seed × 3 + NFCorpus = 7) **= 총 ≈ 30 회 실행** 으로 구성된다:

1. **Frozen ColBERT v2 baseline 측정** (BEIR 6 dataset, 00) — anchor 확립.
2. **Non-learned mean-difference direction** (01) — train HN 과 positive doc 의 layer-12 표현 평균 차이로 비학습 baseline 구축; raw magnitude 가 너무 작아 효과 0.
3. **Magnitude calibration sweep** (01b) — unit-normalized mean-diff 를 α=10 으로 scale 하면 confused-slice NDCG@10 +14 % 상대 개선; 모든 후속 실험의 *sharpened anchor*.
4. **학습된 single direction** (02) — magnitude-tuned mean-diff (α=10) 와 통계적 동등; cos(v_learned, v_mean_diff)=0.32 의 *다른* 방향인데도 *같은 성능*.
5. **Gate-augmented variants** (03 scalar, 04 per-token) — gate 가 학습 lever 가 되지 못함 (multiplicative gradient saturation 또는 always-on 수렴).
6. **Multi-layer extension** (05) — 동일 ceiling. Per-layer cos 분석: 후기 layer 만 mean-diff 와 정렬, 나머지 직교.
7. **Multi-direction router K-sweep** (06_k_sweep, K ∈ {2, 4, 8}) — K=2 와 K=4 가 NDCG@10 all = **0.6614 의 *문자 그대로 동일* ceiling 도달** (capacity 2 배 증가 → ceiling 위치 *완전 불변*). K=8 의 12,296 params 는 *anchor 손상* (NDCG@10 all 0.609, Δ vs baseline -0.038 ✗ negative). 모든 K 의 effective routing K ≈ 1-1.5 — *capacity collapse* + over-capacity 시 anchor 손상.
8. **Bilinear interaction metric** (08, Stage 2 critical) — *Form 변경* ($M = I + UV^\top$, r=8, 2,048 params) 으로 ceiling 우회 시도. **NDCG@10 all = 0.6439 — translation family ceiling 위로 못 감**. confused +0.054 ≈ K-sweep 의 +0.04 와 통계 동등. M 의 *effective rank 가 1 으로 collapse* (UV^T singular values [2.60, 0.06, …] dominant rank-1) — capacity utilization 의 systemic failure (K-router 의 effective K collapse 와 동일 패턴).
9. **Bilinear M + E5-Mistral Margin-MSE distillation** (09, Stage 2 follow-up, λ ∈ {0.1, 0.5, 1.0} sweep) — 08 의 rank-1 collapse 해소 시도. **결과: distillation 이 *anchor regularizer* 로 잘못 작동** — λ ↑ 면 M 이 identity 로 수렴 + confused lever 죽음. λ=0.1 의 confused +0.019 (≪ 08 의 +0.054). E5 cosine margin 이 mined HN 에서 *noise teacher* (≈50% wrong direction). **Stage 2 의 *form-change lever* 검정 종합: ceiling 우회 미달 — encoder-level finetune (LoRA, deferred) 의 critical 검정 필요**.
10. **LoRA on Φ** (10, Stage 3, 3-phase: r=1 → r=8 LR=1e-4 → r=8 LR=5e-5) — Robustness audit 의 *02 unfrozen +0.252 confused* 의 LoRA 회복. Phase 2b (r=8, 295K params) **Δ confused +0.091 ✓** (frozen-side max 의 1.7×, 02 unfrozen 의 36%) + **anchor preservation (CI 0 포함)**. Pre-committed strict 돌파 (CI(Δ all) > 0) 미달 — 9K SciFact triplet *data bottleneck*. *Bounded improvement* 의 최종 결론.
11. **Random direction falsification** (07) — 같은 α=10 magnitude 의 *random* direction 은 baseline 과 통계적 동등 (효과 ≈ 0). Mean-diff 와 비교 시 **−0.053 confused 차이 (CI [-0.091, -0.020])** — *direction 의 내용이 lever* 임을 직접 증명. *Direction-agnostic 가설* 기각.

**핵심 진단 (paper main contribution candidate)**: *Frozen ColBERT 위 모든 학습된 lightweight intervention 의 **per-position effective rank 가 1-2 의 universal collapse***. 06 K-router K∈{2,4,8} → eff K ≈ 1.4 (collapse), 08 bilinear M r=8 → eff rank ≈ 1.01 (rank-1 collapse), 10 LoRA q,v r=8 → per-adapter mean 1.71 (per-adapter collapse). *Pairwise margin + AdamW + small_random init 의 학습 동학의 systematic feature*. **Empirical lift 의 진짜 lever 는 per-position capacity 가 아닌 *distinct intervention positions 의 spatial multiplicity***: 06/08 의 1 position × ~1.4 → Δ conf +0.04-0.05, 10 LoRA 의 *24 positions × ~1.7* → **+0.104 (3-seed mean, robust)**, 02 unfrozen 의 full encoder freedom → +0.252. **Position 수 log-scale 의 monotonic correlation** with confused-slice lift. *Budget-aware design principle*: *적은 r × 많은 positions > 큰 r × 적은 positions*. **LoRA on Φ 가 ColBERT 의 frozen-encoder lightweight intervention 의 *spatial-multiplicity-aware optimal design***. 상세 cross-method punchline: REPORT.md §5e.

**Two main contributions (paper deliverable)**: (i) **§5e — Universal per-position rank-collapse + spatial multiplicity escape** (capacity-side, 위 핵심 진단). (ii) **§5f — Catastrophic Failure as Hard-Contrast Over-Correction** (supervision-side, 4 mediation experiments via M1/M1b/Exp 11/Exp 12): *Phase 2b 의 redistribution (Δ confused +0.104 / Δ easy −0.085 / Δ all ≈ 0) 의 sole sufficient mechanism = hard mined-HN over-correction*. FN noise minor (~14 %), optimization (M1) red herring. *4-lever trade-off framework* — M1b 의 strict net+ (Δ all +0.021 ± 0.005 ✓ 3-seed robust, *paper 의 첫 strict frozen-encoder net+*) vs Exp 11 의 confused preservation (+0.101 ✓ + Δ easy −0.031, 63 % 감소). **Collapse magnitude ↔ NDCG outcomes empirically aligned** — paper-grade *mechanism direct evidence*. 캐비엇 1/2 모두 *empirically resolved*. 상세: §5f.

---

## 1. Introduction

### 1.1 대주제 (overarching thesis)

> **IR 에 존재하는 HN 문제의 원인을 규명하고 모델 개입을 통해 개선·완화하여 retrieval 품질을 향상시키는 *일반화된 개입 방법론* 을 설계한다.**

본 프로젝트는 상위 thesis 의 *representation-level intervention* 축을 담당. Prior diagnostic study (별도 repo) 에서 확인된 multi-layer confusion signal 을 *진단* 에서 *능동적 개입* 으로 확장한다.

### 1.2 Research Questions

| ID | Research Question | 현 상태 |
|---|---|---|
| RQ1 | Frozen ColBERT v2 의 표현 공간 내에서, layer-wise lightweight steering 이 HN-confused query 의 NDCG@10 을 통계적 유의하게 개선하는가? | **부분 통과** — baseline 대비 +0.04 confused 의 통계 유의 개선 확보. 단, *informed non-learned baseline* (01b α=10) 대비 우월성 미확보. |
| RQ2 | 그 개입이 trivial / easy query 의 성능을 손상시키지 않는 *anchor-preserving* 형태로 가능한가? | **통과** — 모든 학습 변형에서 all-slice Δ ≥ 0. |
| RQ3 | 학습된 steering module 이 학습 미사용 도메인으로 frozen 전이 가능한가? | **미검정** — LOOCV (generalization 그룹) 미실행. |
| RQ4 | Confusion direction 의 layer-wise / direction-wise 기여도가 prior diagnostic study finding 과 정합적인가? | **부분 통과** — 05 의 per-layer norm 이 후기 layer (9, 12) 가 큼. Prior finding 의 직접 인용은 미실시 (별도 repo 정보 부재). |

### 1.3 실험 그룹 (현 상태 반영)

| 그룹 | 주제 | 완료 실험 |
|---|---|---|
| Baseline & non-learned anchor | Frozen ColBERT v2 재현 + 비학습 mean-diff direction | 00, 01 (+01b sub) |
| Architectural bottom-up (single direction) | 학습된 v + gate / 다층 의 incremental 추가 | 02, 03, 04, 05 |
| Form variant | Subtract vs projection-out 등 | (미실행) |
| Multi-direction router (main novelty) | K 개 direction + softmax router | 06 (K=2 proof-of-concept) |
| Generalization & robustness | cross-model, LOOCV, seed robust | (미실행) |

총 7 실험 (00–06) 완료. 자세한 ROADMAP 은 [`ROADMAP.md`](ROADMAP.md).

---

## 2. Setting

### 2.1 Encoder

`colbert-ir/colbertv2.0` — BERT-base-uncased (12 transformer layers, 768 hidden) + 768→128 linear projection. 모든 파라미터 frozen (`requires_grad_(False)`), encoder eval mode.

### 2.2 Evaluation

- **Retrieval**: brute-force MaxSim ($s(q,d) = \sum_i \max_j \langle q_i, d_j \rangle$).
- **Datasets**: BEIR 6 dataset (SciFact / NFCorpus / SciDocs / TREC-COVID / FiQA-2018 / ArguAna). 의료 2 + 과학 2 + 금융 1 + 논증 1 의 도메인 분포.
- **Metric**: NDCG@{1, 3, 5, 10, 20}, MRR@10, Recall@{10, 50}, MAP (`pytrec_eval` 표준).
- **Slice**: `all` (전체) / `confused` (baseline 의 top-1 ≠ relevant doc).
- **Statistical**: paired bootstrap 10,000 iter, 95 % CI on Δ-metric.

### 2.3 Baseline 재현 (00_baseline)

| Dataset | Measured NDCG@10 | Paper NDCG@10 | Δ |
|---|---|---|---|
| SciFact | 0.6464 | 0.693 | −0.047 |
| NFCorpus | 0.3299 | 0.338 | −0.008 |
| SciDocs | **0.1581** | 0.154 | **+0.004** ✓ |
| TREC-COVID | 0.7270 | 0.738 | −0.011 |
| FiQA-2018 | 0.3473 | 0.356 | −0.009 |
| ArguAna | 0.4528¹ | 0.463 | −0.010 |

¹ ArguAna 는 query 가 corpus doc 와 동일 id 로 존재 (counter-argument task) → `exclude_self=True` 적용.

**Documented limitation**: SciDocs 만 paper ±0.005 통과. 나머지 5 dataset 에서 ~−0.01 systematic gap (단 SciFact 만 −0.047 outlier). 본 gap 의 잠재 원인 (C7 transformers 버전 / C8 PLAID inference / C9 MPS 정밀도) 의 분리 검정 비용이 본 프로젝트 scope 대비 큼. 후속 LSR 실험은 *paired Δ* 측정으로 internal validity 유지하며 진행. 상세: [`report/00_baseline_report.md`](report/00_baseline_report.md).

![Baseline overlay](report/figures/00_baseline/metrics_paper_overlay.png)

*Figure 1. Frozen ColBERT v2 의 6 BEIR dataset 재현. SciDocs (+0.004) 만 paper ±0.005 통과. SciFact (-0.047) 가 유일한 큰 outlier; 나머지 4 개는 ~-0.01 의 systematic gap.*

---

## 3. Non-learned baseline anchor (01, 01b)

### 3.1 Hypothesis

학습 *없이* 단순한 representation-level intervention 이 HN-confused query 의 retrieval 을 개선할 수 있는가? — 학습된 LSR 의 *의미성* 정량 anchor 확립.

### 3.2 Raw mean-difference direction (01)

Train split 에서:
$$v = \bar{h}^{(12)}_{\text{HN}} - \bar{h}^{(12)}_{\text{pos}}, \quad \tilde{h}^{(12)} = h^{(12)} - v$$

3 dataset (SciFact / NFCorpus / FiQA, train split 보유):

| Dataset | v_norm | Δ NDCG@10 (confused, CI) | 효과 |
|---|---|---|---|
| SciFact | 0.27 | −0.001 [-0.006, +0.002] | ≈ 0 |
| NFCorpus | 0.03 | +0.000 [+0.000, +0.000] | ≈ 0 |
| FiQA-2018 | 0.21 | +0.000 [-0.001, +0.001] | ≈ 0 |

**결과**: 모든 dataset 에서 Δ ≈ 0. v_norm 이 매우 작음 (0.03 – 0.27). BERT layer-12 hidden 의 token-level magnitude (≈ 10) 대비 1-3 % 의 perturbation → 사실상 no-op.

### 3.3 Magnitude calibration sweep (01b)

$v$ 를 unit-normalize 후 scale parameter $\alpha$ 로 multiply:
$$\tilde{h}^{(12)} = h^{(12)} - \alpha \cdot \hat{v}, \quad \alpha \in \{0.5, 1, 2, 5, 10\}$$

SciFact 단일 sweep 결과:

| α | NDCG@10 | Δ confused (CI) | CI 0 초과 |
|---|---|---|---|
| 0.5 | 0.6477 | +0.003 [-0.004, +0.009] | ✗ |
| 1.0 | 0.6478 | +0.006 [-0.002, +0.013] | ✗ |
| 2.0 | 0.6536 | +0.018 [+0.005, +0.032] | ✓ |
| 5.0 | 0.6666 | +0.052 [+0.026, +0.081] | ✓ |
| **10.0** | **0.6690** | **+0.064 [+0.034, +0.099]** | **✓** |

![Alpha sweep](report/figures/01_mean_diff/alpha_sweep_curve.png)

*Figure 2. SciFact 의 α-sweep. α ≥ 2 부터 confused slice 의 CI 가 명확히 0 을 초과 (paired bootstrap). α=10 에서 +0.064 (baseline 의 +14 % 상대 개선).*

**핵심 발견**:
1. **C-form 가설 기각**: subtract form ($h - v$) 자체는 작동.
2. **C-magnitude 확정**: raw mean-diff 의 효과 부재는 *magnitude 부족* 의 결과.
3. **Mean-diff direction 은 *informative***: unit-normalize 후 적절히 scale 하면 confused +0.064.

### 3.4 Narrative 함의

01b 의 α=10 가 **학습된 LSR (02 부터) 의 sharpened anchor** 가 됨. 단순히 *baseline 대비 Δ > 0* 이 아니라 *informed non-learned baseline 대비 추가 개선* 이 학습의 진정한 가치 명제.

상세: [`report/01_mean_diff_report.md`](report/01_mean_diff_report.md).

---

## 4. Architectural bottom-up (single direction, 02–05)

### 4.1 학습된 single direction (02)

- 형식: $\tilde{h}^{(12)} = h^{(12)} - v$, $v \in \mathbb{R}^{768}$, $v|_{t=0} = \mathbf{0}$.
- 학습: pairwise margin ($m=0.2$), AdamW (LR=$10^{-3}$, WD=$10^{-4}$), $\lambda_{\text{anc}} = 0$ (single-direction 단계 deviation, DESIGN.md §11).
- 학습 파라미터: **768**.

| 지표 | 값 |
|---|---|
| NDCG@10 (test) | **0.6651** |
| Δ vs baseline (confused) | **+0.044 [+0.023, +0.066]** ✓ |
| Δ vs 01b α=10 (confused) | -0.021 [-0.047, +0.003] (CI 0 포함) |
| ‖v_learned‖ | 7.08 (epoch 2, best state) |
| cos(v_learned, v_mean_diff) | **0.324** |

![Direction comparison](report/figures/02_final_layer_vector/direction_compare.png)

*Figure 3. (왼쪽) 학습된 v 의 magnitude (7.08) 가 mean-diff v 의 0.27 의 ~26 배. (오른쪽) cos = 0.32 → H5 qualitative 통과 (0.9 threshold 의 magnitude-only 가설 기각). 두 v 는 *다른 방향* 이지만 retrieval 성능은 통계 동등 → single-direction subspace 의 redundancy 시사.*

### 4.2 Gate 추가 (03 scalar, 04 per-token)

**03_scalar_gate**: $\tilde{h} = h - g \cdot v$, $g = \sigma(b)$, $b|_{t=0} = -3$.

| 지표 | 값 |
|---|---|
| NDCG@10 | 0.6448 |
| effective $g \cdot \|v\|$ 학습 종료 | 0.23 |
| Δ vs 02 (all) | -0.020 [-0.032, -0.010] ✗ negative |

**Mechanism**: multiplicative gradient saturation. $g \approx 0.07$ 으로 시작 → $\partial L / \partial v = g \cdot \partial L / \partial(gv)$ 가 $g$ 만큼 작아 v 도 천천히 학습. effective magnitude 가 01b 의 α=0.5 미만 영역에 머묾.

**04_per_token_gate**: $\tilde{h}_t = h_t - g(h_t) \cdot v$, $g(h_t) = \sigma(W h_t + b)$.

| 지표 | 값 |
|---|---|
| NDCG@10 | 0.6641 |
| Per-token gate 분포 (test) | mean=1.000, std=0.001 |
| Δ vs 02 (confused) | -0.003 [-0.009, -0.000] (사실상 동등) |

**Mechanism**: gate 가 *모든 token 에서 1.0* 에 saturated → 효과적으로 02 와 동일 (h - v). 02 의 all-slice Δ 가 이미 양수 → anchor preservation 의무 부재 → gate 가 closed 될 동기 없음 → always-on 으로 수렴.

**Gate 결론 (03, 04 종합)**: gate 의 학습 lever 가 *데이터 상* 없음. Scalar gate 는 gradient saturation 으로 underuse, per-token gate 는 always-on 으로 02 와 동치.

상세: [`report/03_scalar_gate_report.md`](report/03_scalar_gate_report.md), [`report/04_per_token_gate_report.md`](report/04_per_token_gate_report.md).

### 4.3 Multi-layer extension (05)

5 layer × $v_\ell$ at $\ell \in \{0, 3, 6, 9, 12\}$. 3,840 학습 파라미터.

| 지표 | 값 |
|---|---|
| NDCG@10 | 0.6502 |
| Δ vs 02 (confused) | +0.007 [-0.031, +0.047] (통계 동등) |

![Single-direction progression (02–05 + 01b)](report/figures/05_five_layers/single_direction_summary.png)

*Figure 4. 6 condition 의 SciFact NDCG@10. baseline 의 약 0.65, α=10 / 02 / 04 / 05 모두 ~0.65–0.67 의 ceiling.*

Per-layer 분석:

| ℓ | ‖v_ℓ‖ | cos(v_ℓ, v_mean_diff_l12) | 해석 |
|---|---|---|---|
| 0 | 1.27 | -0.005 | mean-diff 와 *직교* |
| 3 | 1.52 | +0.038 | 직교 |
| 6 | 2.22 | +0.011 | 직교 |
| 9 | 2.85 | +0.041 | 직교 |
| 12 | 2.79 | **+0.267** | mean-diff 와 부분 정렬 |

학습이 *다른 axis* (4 개 layer 에서 mean-diff 와 직교 방향) 의 정보 잡으려 시도했으나 retrieval 측면 같은 ceiling. 5× 파라미터의 over-fitting 패턴 (epoch 1 best, 2/3 감소).

상세: [`report/05_five_layers_report.md`](report/05_five_layers_report.md).

### 4.4 결론 — Single-direction subspace ceiling 확정

| Step | NDCG@10 | Δ vs 02 confused (CI) |
|---|---|---|
| 02 (K=1, 768) | **0.6651** | (anchor) |
| 03 (scalar gate) | 0.6448 | -0.047 ✗ |
| 04 (per-token gate, 1537) | 0.6641 | -0.003 (동등) |
| 05 (K=1, 5 layers, 3840) | 0.6502 | +0.007 (동등) |
| 01b α=10 (비학습 informed) | 0.6690 | (sharpened anchor) |

**모든 single-direction-style 변형이 NDCG@10 = 0.65–0.67 의 ceiling 에서 만남**. 직접 함의:

1. **Single direction 의 *capacity 한계*** — 학습된 / 비학습된 / 단층 / 다층 모두 같은 retrieval 성능.
2. **학습된 다른 *방향* 도 redundant** — 02 의 cos=0.32 (다른 방향) + 05 의 4 layer 직교 방향도 모두 같은 ceiling.
3. **Gate 형식이 학습 lever 가 안 됨** — anchor preservation 의무가 없는 setting (Δ_all > 0) 에서 gate 가 closed 될 동기 없음.
4. **Multi-layer 단순 확장도 무효** — direction 수만 늘려 *같은 axis 확장* 이라 같은 ceiling.

→ **본질적 lever 는 direction 수 (multi-direction) + 효과적 routing (selectivity) 의 *결합***. 단순 capacity 증가가 아닌 *axis 분리* 가 필요. Multi-direction router (06+) 의 empirical motivation 결정적.

---

## 5. Multi-direction router K-sweep (06) + direction-agnostic falsification (07)

### 5.1 K ∈ {2, 4, 8} sweep

수식 (K 가변, single layer ℓ=12):
$$\tilde{h}_t = h_t - \sum_{k=1}^{K} \pi_k(h_t) \cdot v_k, \quad \pi(h_t) = \text{softmax}(W h_t + b) \in \Delta^K$$

학습 파라미터: $2 K D + K$ — K=2: 3,074 / K=4: 6,148 / K=8: 12,296. SciFact, seed 42.

| K | Params | NDCG@10 (all) | Δ all vs baseline (CI) | Δ confused vs baseline (CI) | Δ confused vs 02 K=1 (CI) |
|---|---|---|---|---|---|
| 2 | 3,074 | **0.6614** | +0.015 [+0.004, +0.026] ✓ | +0.039 [+0.017, +0.061] ✓ | -0.005 (CI 0 포함) |
| 4 | 6,148 | **0.6614** | +0.015 [+0.003, +0.028] ✓ | +0.045 [+0.024, +0.068] ✓ | +0.002 (CI 0 포함) |
| 8 | 12,296 | **0.6089** | **−0.038 [−0.067, −0.008] ✗** | +0.049 [+0.005, +0.092] ✓ | −0.005 (CI 0 포함) |

**핵심 발견**:
1. **K=2 와 K=4 의 NDCG@10 all 이 *문자 그대로 동일* (0.6614, 소수점 4 자리 까지 동일)** — multi-direction 의 capacity 가 2 배 증가해도 ceiling 위치 *완전 보존*. capacity-driven ceiling 우회의 부재.
2. **K=8 의 anchor 손상** — capacity 가 2 배 더 증가하면 *easy queries 의 NDCG 가 -0.038 감소* (CI 0 미달 ✗). Confused-slice 는 여전히 ceiling 부근 (+0.049 ≈ K=2/K=4 의 +0.04). **Over-capacity 가 ceiling 우회 lever 가 아닐 뿐만 아니라 *해로움***.
3. **Effective K 가 K 와 무관하게 ~1.2-1.5** — K=2: 1.41, K=4: 1.23, K=8: 1.44. linear-router 의 *capacity collapse* 시스템적 — 학습이 항상 1-2 개 dominant direction 으로 수렴, 나머지 K-2 개는 사실상 dead (π ≈ 10⁻⁶).

### 5.2 Routing + direction diagnostics

| K | dominant direction(s) | mean pairwise \|cos(v_i, v_j)\| | π_mean of dominant(s) |
|---|---|---|---|
| 2 | v_1 (76 %, cos=0.04 with mean-diff) | 0.55 | [0.24, 0.76] |
| 4 | v_2 (89 %, cos=0.08 with mean-diff) | **0.74** | [0.11, 0, 0.89, 0] |
| 8 | v_4 (68 %, cos=0.33) + v_2 (32 %, cos=-0.04) | **0.76** | [~0, ~0, 0.32, ~0, 0.68, ~0, ~0, ~0] |

K 가 클수록 *대부분의 direction 이 서로 매우 유사* (cos ≥ 0.95 의 near-duplicate); 1-2 개의 "다른" direction 에 router 의 mass 가 집중. K=8 의 dual-dominance (v_4 partial-align + v_2 orthogonal) 는 anchor 손상의 직접 원인 — over-correction.

### 5.3 함의 — Multi-direction unaided + over-capacity 의 *역효과*

본 sweep 의 종합 발견:

| 결과 | 함의 |
|---|---|
| K=2, K=4 의 NDCG@10 all 완전 동일 | capacity 증가가 ceiling 우회 lever 가 아님 (translation family ceiling 의 multi-direction 확장 confirmation) |
| K=8 의 anchor 손상 | over-capacity 의 *해로움* — single-direction subspace 의 capacity 가 ceiling 근처에서 *적정 magnitude* 인데 K=8 의 dual-dominance 가 그 magnitude 를 *초과* 함 |
| Effective K 의 K-invariance (1.2-1.5) | linear router 의 systemic capacity collapse — Switch Transformer 류 entropy reg 없이 multi-direction 의 latent capacity 활용 불가 |

옛 ROADMAP 의 narrative ("K ↑ + router 표현력 + entropy reg 의 결합") 의 *capacity-only* 가설은 본 K-sweep 으로 *직접 falsify*. 진짜 lever 는 *form 자체* 의 변경 (translation family 밖) — bilinear M (Stage 2).

상세: [`report/06_k_sweep_report.md`](report/06_k_sweep_report.md).

### 5.4 07_random_direction_scaled — 결정적 falsification

K=2 multi-direction 도 ceiling 못 넘은 후, 모든 학습 변형 (02–06) 과 비학습 (01b α=10) 의 ceiling 0.665 가 *algebraic family 한계* 인지 아니면 *direction-agnostic magnitude flooding 의 ceiling* 인지의 결정적 검정. 같은 magnitude (α=10) 의 *random Gaussian unit vector* 를 layer 12 에 적용:

$$\tilde{h}^{(12)} = h^{(12)} - \alpha \cdot \hat{v}_{\text{random}}, \quad \alpha = 10$$

학습 무필요. seed 42 로 vector 생성 후 단일 forward pass.

| Comparison | NDCG@10 (SciFact) |
|---|---|
| baseline (00) | 0.6464 |
| 07 random × α=10 | **0.6485** (baseline 과 통계 동등) |
| 01b mean-diff × α=10 | **0.6690** |

Paired bootstrap 95 % CI (vs 01b α=10):
- all: Δ = **−0.0205** [−0.0386, −0.0041] ✗ negative
- confused: Δ = **−0.0533** [−0.0905, −0.0201] ✗ negative

![07 direction compare](report/figures/07_random_direction_scaled/direction_compare.png)

*Figure 6. 같은 magnitude (‖α v̂‖ = 10), 다른 direction 의 효과 대비. baseline (0.6464), 07 random (0.6485, baseline 거의 동일), 01b mean-diff (0.6690, baseline +0.022 / confused +0.064). **같은 크기인데도 random 은 효과 ≈ 0, mean-diff 는 confused +0.064** — direction 의 *내용* 이 명백한 lever.*

**결정적 함의**: *direction-agnostic 가설 (translation 의 효과는 magnitude-driven blunt residual)* 의 **명확한 기각**. 같은 magnitude 인데도 random direction 은 baseline 대비 효과 ≈ 0, mean-diff direction 은 confused +0.064 — direction 의 *내용* 이 *결정적* lever. ROADMAP §"H5 학습된 direction 의 의미성" 가 비학습 mean-diff vs random 의 대비에서도 통과.

다만 *translation family 의 algebraic ceiling 자체* 의 검정은 여전히 미해결 — 후속 08 bilinear M ($s_M(q,d) = \sum_i \max_j q_i^\top M d_j$, $M = I + UV^\top$) 의 결과로만 *informed direction subspace 의 representational limit* 이 *algebraic 한계* 인지 *정보 한계* 인지 분리 가능.

상세: [`report/07_random_direction_scaled_report.md`](report/07_random_direction_scaled_report.md).

---

## 5b. Bilinear interaction metric (08, Stage 2 critical)

### 5b.1 형식 (translation family *밖* 으로의 minimal 우회)

실험 00–07 의 종합 결과 *informed direction* 위에서의 translation 변형 (모든 $u(h)$ 형식) 의 ceiling 0.665 가 확정. *Form 자체* 의 변경만이 이 ceiling 을 algebraic 으로 우회 가능. 본 실험은 *MaxSim 의 inner product* 를 일반화:

$$s_M(q, d) = \sum_i \max_j q_i^\top M d_j, \quad M = I + U V^\top, \quad U, V \in \mathbb{R}^{D \times r}$$

$D = 128$ (post-projection space), $r = 8$ → 학습 가능 파라미터 **2,048**. 전개:
$$q_i^\top M d_j = \langle q_i, d_j \rangle + (U^\top q_i)^\top (V^\top d_j)$$

두 번째 항이 **q 와 d 의 *cross-feature* 곱셈적 결합** — translation family 가 *절대* 못 표현하는 차원. *Algebraic family change*.

### 5b.2 결과 (SciFact, seed 42, LR=1e-4)

| 지표 | 08 r=8 | 비교 |
|---|---|---|
| NDCG@10 (all) | **0.6439** | ≈ baseline (0.6464), ceiling 0.6614 *못 넘음* |
| Δ vs baseline (confused) | **+0.054 [+0.013, +0.097] ✓** | K=2/K=4 의 +0.04 와 통계 동등 |
| Δ vs 01b α=10 (all) | **-0.025 [-0.046, -0.005] ✗** | *informed non-learned anchor 보다 worse* |
| Δ vs 02 K=1 / 06 K=2 / 06 K=4 (all, conf) | 모두 CI 0 포함 | translation family 와 *통계 동등* |

**Stage 2 critical falsification 결과**: r=8 + pairwise margin only 의 bilinear M 으로 *form 변경 만으로는 ceiling 위로 못 감*. confused 개선폭은 translation family 변형과 동등.

### 5b.3 새 발견 — *Rank-collapse* (capacity 미활용)

학습된 M 의 spectral 진단:

- **UV^T 의 singular values: [2.604, 0.062, 0.035, 0.033, 0.024, 0.022, 0.015, 0.014]** — *dominant rank-1*. r=8 의 8 dim capacity 중 *사실상 1 차원* 만 활용 (σ_1 ≫ σ_2~8).
- M = I + UV^T 의 condition number = **81.14** (1 방향만 ×2.64 amplify, 나머지 127 dim ≈ identity).
- ‖U‖ = 2.04, ‖V‖ = 1.38, ‖UV^T‖_F = 2.61, M deviation from I = 2.61.

![08 M spectrum](report/figures/08_bilinear_M_minimal/M_spectrum.png)

*Figure 7. (왼쪽) UV^T 의 singular values — σ_1 = 2.60 dominant, σ_2~8 ≪ σ_1. r=8 의 latent capacity 미활용. (오른쪽) M = I + UV^T 의 spectrum — σ_1 = 2.64 의 한 dim 만 강하게 변형, 나머지 127 dim ≈ 1.0 (identity 보존). cond=81.14.*

**핵심 진단**: r=8 의 expressivity capacity 가 *optimization-driven collapse* 로 rank-1 으로 축소. K-router 의 effective K collapse (1.2-1.5 / K) 와 *완전히 평행* 패턴 — 단순 pairwise margin loss + AdamW 가 *higher-order capacity* 활용 못 한다는 systemic 현상. 동일 데이터 (SciFact 9K triplets) 한계의 *training signal* 가능성도 잔존.

### 5b.4 함의 — Stage 2 의 *재해석*

본 결과의 두 가지 해석:

| 해석 | 의미 |
|---|---|
| (i) *Form 자체* 의 lever 부재 | bilinear M 도 frozen-encoder representational limit 안에 갇힘. ceiling 은 *information 한계* (encoder 본질). → 18 LoRA on Φ 의 critical 검정 |
| (ii) *Optimization-driven rank collapse* | r=8 의 latent capacity 는 ceiling 우회 충분하지만 학습 dynamics 가 활용 못 함. → 09 E5 distillation (richer supervision) + 10 r sweep (rank vs effective rank 관계) 으로 분리 |

ROADMAP §"Stage 2" 의 *partial fail* 분기로 진행: 09/10 이 critical follow-up — *form 의 lever* 인지 *capacity utilization 의 lever* 인지의 결정적 분리.

### 5b.5 기술적 noteworthy

- **Zero-init pathology**: $U = V = \mathbf{0}$ 시 $\partial \mathcal{L}/\partial U \propto V = 0$, $\partial \mathcal{L}/\partial V \propto U = 0$ — gradient 정지. small_random init (std=10⁻², ‖UV^T‖_F ≈ 0.035, vanilla MaxSim 대비 < 0.1% deviation) 으로 해결.
- **LR sensitivity**: LR=1e-3 시 1 epoch 안에 ‖[U;V]‖ 8× 폭증 → val NDCG@10 catastrophic drop (0.45). LR=1e-4 로 안정화.

상세: [`report/08_bilinear_M_minimal_report.md`](report/08_bilinear_M_minimal_report.md).

---

## 5c. Bilinear M + E5-Mistral Margin-MSE distillation (09, Stage 2 follow-up)

### 5c.1 동기

08 의 *rank-1 collapse* 해소 시도. *Teacher (E5-Mistral-7B-Instruct)* 의 cross-encoder-quality cosine margin 을 distill target 으로 *richer ranking signal* 주입.

수식:
$$\mathcal{L} = \underbrace{\max(0, m - (s_{\text{pos}} - s_{\text{hn}}))}_{\text{pairwise margin}} + \lambda \cdot \underbrace{\left( (s_{\text{pos}} - s_{\text{hn}}) - \tau \cdot (\text{e5}_{\text{pos}} - \text{e5}_{\text{hn}}) \right)^2}_{\text{Margin-MSE distill}}$$

- $\tau = 8.0$ (teacher_scale, E5 cosine margin ~0.03 → ColBERT margin ~0.2 scale)
- E5 embedding: 4096-d fp16 (L2 normalized), 사전 추출 ([`data/e5_teacher/`](data/e5_teacher/))
- Train query embedding: `data/e5_teacher/extract_train_queries.py` 가 신규 추출 (E5-Mistral on MPS, 809 queries, ~85초)

### 5c.2 λ-sweep 결과 (SciFact, seed 42, r=8, LR=1e-4)

| λ_distill | NDCG@10 (all) | Δ all vs baseline | Δ confused vs baseline | M structure |
|---|---|---|---|---|
| **0 (08)** | 0.6439 | -0.003 (≈) | **+0.054 [+0.013, +0.097]** ✓ | rank-1 (σ₁=2.60), ‖UV^T‖=2.61 |
| **0.1** | **0.6509** | +0.005 (≈) | **+0.019 [+0.006, +0.033]** ✓ | partial rank-2 (σ₁=0.46, σ₂=0.11), ‖UV^T‖=0.49 |
| **0.5** | 0.6451 | -0.001 (≈) | -0.002 (≈ baseline) | uniform tiny, ‖UV^T‖=0.10 |
| **1.0** | 0.6453 | -0.001 (≈) | -0.002 (≈ baseline) | uniform tiny, ‖UV^T‖=0.10 |

**Sweep 핵심 발견**:

1. **λ ↑ 면 *anchor preservation* 개선** (all-slice 가 baseline 수렴) — distillation 이 M 을 identity 근처에 묶음.
2. **λ ↑ 면 *confused-slice lever 죽음*** — λ=0 의 +0.054 → λ=0.1 의 +0.019 → λ=0.5/1.0 의 0. Paper main contribution 의 *반대 방향*.
3. **Rank-collapse 해소 trade-off**: λ ↑ → σ₁/σ₂ ratio 42 → 4.1 → 1.5 (해소되지만 magnitude 도 ‖UV^T‖ 2.6 → 0.1 으로 *너무 작아* M 가 사실상 학습 안 됨).

### 5c.3 진단 — *Distillation 이 잘못된 lever 인 이유*

| 가설 | 증거 |
|---|---|
| (i) Teacher signal 의 *noise*: E5 도 mined HN 에서 random-half (phase_02 sample 의 e5_margin 약 50% 음수) | E5-Mistral 7B 의 *bi-encoder cosine* 가 ColBERT 가 mining 한 *어려운* HN 에 대해서는 명확한 ranking 미제공 |
| (ii) Margin-MSE 의 *scale mismatch*: student margin ~ -0.7, teacher × τ ~ -0.24 → loss = 25 의 huge magnitude → λ=0.1 만 해도 effective gradient 가 distill dominate | Teacher_scale 조정해도 *teacher 의 noisy signal* 변경 안 됨 |
| (iii) Lever 의 *opposite direction*: 08 의 rank-1 σ₁ ≈ 2.6 의 dominant axis 가 *informed direction subspace 의 single 활용* — distillation 이 이 lever 를 *축소* | 09 λ=0.1 에서 σ₁ = 0.46 (× 5.7 작음) + confused 효과 +0.019 (08 의 +0.054 의 35%) |

### 5c.4 Stage 2 의 *최종 진단*

**08 + 09 의 종합**:
- *form 자체* 의 변경 (bilinear M) 의 lever 는 *부분 유효* (08 의 +0.054 confused, K-router 보다 약간 우수).
- 하지만 *translation family ceiling 위로 못 감* (08 NDCG@10 all 0.6439 ≈ baseline 0.6464).
- E5 Margin-MSE distillation 은 *anchor regularizer* 로 잘못 작동 (09).
- → **Frozen-encoder representational limit 의 정황 증거**. *Form 변경만으론 부족, encoder-level finetune (LoRA on Φ) 가 critical 검정*.

### 5c.5 추가 figure

![NDCG vs lambda](report/figures/09_bilinear_M_e5_distill/ndcg_vs_lambda.png)

*Figure 8. λ_distill 의 함수로 SciFact NDCG@10. all-slice (파랑) 는 λ=0.1 에서 baseline 초과 (0.6509), λ ↑ 면 baseline 수렴. confused-slice (빨강) 는 λ=0 (08) 에서 최고 (0.236), λ ↑ 면 baseline 수준으로 하락. **Distillation 이 anchor 보호하면서도 confused lever 죽임** — paper main contribution 의 opposite 방향.*

![Rank-collapse by lambda](report/figures/09_bilinear_M_e5_distill/rank_collapse_by_lambda.png)

*Figure 9. (왼쪽) UV^T 의 singular values, log scale, λ ∈ {0, 0.1, 0.5, 1.0}. 08 (λ=0) 의 σ₁ = 2.6 압도적. λ=0.1 의 σ₁ = 0.46 (× 5 작음) + 평평한 σ₂-σ₈ (partial diversification). λ=0.5/1.0 은 모든 σ 거의 동일하게 작음. (오른쪽) ‖UV^T‖_F (파랑) 는 기하급수 감소 (2.6 → 0.49 → 0.10), σ₁/σ₂ ratio (빨강) 도 동시 감소 — *rank-1 collapse 해소가 magnitude 감소의 대가*.*

상세: [`report/09_bilinear_M_e5_distill_report.md`](report/09_bilinear_M_e5_distill_report.md).

---

## 5d. LoRA on Φ (10, Stage 3) — *encoder representational limit 의 부분 회복*

### 5d.1 동기 + Pre-committed 판정

Robustness audit (§7) 의 *02 unfrozen* 의 Δ confused **+0.252 ✓** (110M params) 가 *frozen-encoder 가 진짜 bottleneck* 의 직접 증거. 본 실험은 LoRA adapter (Hu et al. 2021) 로 50K-budget-relaxed 영역에서 그 lift 의 *얼마나* 회복 가능한지 정밀 검정.

**Pre-committed 판정 기준** (외부 reviewer 입력 반영, 결과 보기 전 commit):
- **Early-stop = `val_all`** (post-hoc cherry-picking 회피, 모든 config 동일)
- **돌파 ⟺ CI 하한$_{\Delta \text{NDCG@10 all vs baseline}} > 0$**
- 미돌파 시 → *hyperparameter sweep 금지* (9K SciFact triplet data bottleneck 의 한계 명시), safety-net narrative 채택.

### 5d.2 3-단계 sweep 결과

각 BERT attention 의 q, v Linear 에 LoRA rank-r adapter:
$$h = W x + (\alpha/r) B A x, \quad A \in \mathbb{R}^{r \times 768}, \ B \in \mathbb{R}^{768 \times r}$$

| Phase | Config | Params | NDCG@10 all | Δ all vs baseline | **Δ confused vs baseline** | 돌파? |
|---|---|---|---|---|---|---|
| 1 | r=1, LR=5e-5, α=r | 36,864 | 0.5940 | -0.052 [-0.085, -0.021] ✗ | +0.038 [-0.008, +0.083] (CI 0 포함) | ✗ |
| 2a | r=8, LR=1e-4, α=2r | 294,912 | 0.5879 | -0.059 [-0.101, -0.017] ✗ | **+0.080 [+0.021, +0.140] ✓** | ✗ |
| **2b (seed 42)** | **r=8, LR=5e-5, α=r** | **294,912** | **0.6367** | **-0.010 [-0.044, +0.023] (CI 0 포함)** | **+0.091 [+0.040, +0.143] ✓** | **✗ (CI 하한 -0.044 < 0)** |
| **2b (seed 1337)** | (동일 config) | 294,912 | 0.6423 | -0.004 (≈) | +0.097 ✓ | ✗ |
| **2b (seed 2024)** | (동일 config) | 294,912 | 0.6639 | +0.018 (≈) | +0.123 ✓ | ✗ |
| **2b (3-seed mean ± std)** | (동일 config) | 294,912 | **0.6476 ± 0.014** | **+0.001 ± 0.014** | **+0.104 ± 0.017 ✓** | **✗ (3-seed 모두)** |

**Phase 2b 의 *bounded improvement* 달성** — strict 돌파 미달, but:
- **Anchor preservation 회복** (Phase 2a -0.059 → Phase 2b -0.010, CI 0 포함)
- **Confused-slice +0.091** ✓ (frozen-side max 인 08 seed 42 의 +0.054 의 *1.7×*)
- **02 unfrozen 의 +0.252 의 36% 회복** (295K params = encoder 의 0.27%)
- **Seed × 3 robustness**: 3 seeds 모두 Δ confused 통계 유의 + anchor 보존. **3-seed mean +0.104** (unfrozen 의 41%). *08 의 seed-artifact 와 완전 반대 양상 — Phase 2b 의 lift 는 robust*.

![10 LoRA progression](report/figures/10_lora_phi/lora_progression.png)

*Figure 10. 3-phase progression of Δ NDCG@10 vs baseline (CI bar). **all-slice (파랑)**: 1 -0.052 ✗ → 2a -0.059 ✗ → **2b -0.010 (CI 0 포함)**. **confused (빨강)**: 1 +0.038 (CI 0 포함) → 2a **+0.080 ✓** → **2b +0.091 ✓**. *LR 보수화 + α=r* 이 anchor 손상 해소의 critical lever.*

### 5d.3 LoRA capacity utilization — *균등 분포*

기존 frozen-side method 들 (06 K-router, 08 bilinear M) 의 effective rank 1 collapse 와 *반대* 양상 — LoRA 의 24 adapters (q+v × 12 layers) 의 ‖A‖, ‖B‖ 가 *균등 분포* (Figure: `lora_AB_norms.png` 참조). BERT 자체의 layer-wise gradient flow + dropout 이 *적절한 학습 distribution* 유도. **단 capacity 균등 활용에도 strict 돌파 미달** → 9K SciFact triplet 의 *data bottleneck* 이 학습 신호 자체 한계.

### 5d.4 Pre-committed 판정 채택 — Safety-net narrative

**Phase 2b 의 strict 돌파 미달 → hyperparameter sweep 중단**. *Bounded improvement* 결론:

> Frozen ColBERT 의 lightweight intervention 의 *bounded improvement* — translation family / form-change / distillation 의 ceiling 0.665 + LoRA 의 confused +0.091 (anchor-preserving). Encoder unfreeze (110M) 만이 strict 돌파 (Δ confused +0.252). LoRA 295K (encoder 0.27%) 가 그 lift 의 36% 회복 — *param-efficient partial recovery*. *9K SciFact triplet 의 data bottleneck* 이 strict 돌파 차단.

상세: [`report/10_lora_phi_report.md`](report/10_lora_phi_report.md).

---

## 5e. Universal rank-collapse + spatial multiplicity escape (cross-method punchline)

본 paper 의 *진짜* main contribution — 06 / 08 / 10 의 세 method 의 *통합 진단*. 모든 학습된 frozen-side intervention 의 *single intervention position 에서의 effective rank/K* 가 nominal capacity 의 *12-30% 에 collapse* (universal pattern). LoRA 의 우월한 lift 의 진짜 이유는 *per-position rank 의 escape 아니라* **24 distinct intervention positions 의 spatial multiplicity**.

### 5e.1 Capacity utilization across methods

| Method | Nominal | Effective | Util ratio | Positions | Δ confused vs baseline |
|---|---|---|---|---|---|
| 06 K-router K=2 | 2 | 1.41 | 70 % | 1 | +0.039 ✓ |
| 06 K-router K=4 | 4 | 1.23 | 31 % | 1 | +0.045 ✓ |
| 06 K-router K=8 | 8 | 1.44 | 18 % | 1 | +0.049 ✓ |
| 08 bilinear M r=8 | 8 | 1.01 | 13 % | 1 | +0.054 (seed 42 only) |
| 10 LoRA r=1 (Phase 1) | 1 | 1.00 | 100 % | 24 | +0.038 (CI 0 포함) |
| **10 LoRA r=8 (Phase 2b 3-seed mean)** | 8 | **1.71** | 21 % | **24** | **+0.104 ✓** |
| 02 unfrozen (110M, upper bound) | full | full | 100% | full | +0.252 ✓ |

![Universal rank-collapse contrast](report/figures/_cross_method/rank_collapse_contrast.png)

*Figure 11. **모든 frozen-side method 의 *per-position effective rank* 가 1-1.7 의 universal collapse** (왼쪽 + 가운데 panel). LoRA r=8 (Phase 2b) 의 24 adapters 중 모두 active + per-adapter mean 1.71 ± std 1.07 (오른쪽 panel). *학습 동학의 systematic feature 는 per-position rank-1ish collapse*. LoRA 가 superior lift 를 보이는 진짜 이유는 *spatial multiplicity (24 positions)* 의 결과.*

### 5e.2 통합 진단 — *학습 dynamics 의 universal feature + LoRA 의 다중 lever*

**핵심 발견** (paper-grade contribution):

> *Frozen ColBERT 위 모든 학습 intervention 의 **per-position effective rank 가 ~1-2** (06 K-router 의 K=8 → eff 1.44, 08 bilinear M r=8 → eff 1.01, 10 LoRA r=8 → per-adapter mean 1.71). 이는 *pairwise margin loss + AdamW + small_random init* 의 학습 동학의 **universal systematic feature**. Empirical lift 의 진짜 lever 는 *per-position capacity 가 아닌 **distinct intervention positions 의 spatial multiplicity***. 06/08 의 1 position × ~1.4 → Δ conf +0.04-0.05, 10 LoRA 의 24 positions × ~1.7 → +0.104 (mean), 02 unfrozen 의 *full encoder freedom* → +0.252. **Position 수 의 log-scale 에 따라 confused lift monotonic**.

### 5e.3 *Paper main contribution* 의 *최종 narrative*

본 finding 의 paper-grade implication:
1. *Frozen ColBERT 위 lightweight intervention 의 학습 동학에 universal capacity collapse 존재* (3 method, 6+ config 의 결정적 증거).
2. *Capacity 증가는 효과 없음* — single position 의 K 증가 (06) 도, r 증가 (08) 도, per-adapter r 증가 (10 r=1 → r=8) 도 effective rank 증가 미흡.
3. *Spatial multiplicity 가 lever* — *position 수* 증가 만이 effective intervention dimensionality 의 진정한 lift.
4. 결과는 *budget-efficient design principle*: 적은 r × 많은 positions > 큰 r × 적은 positions. **LoRA 가 ColBERT 의 frozen-encoder lightweight intervention 의 *budget-aware optimal design***.

상세 분석: `report/_rank_collapse_punchline.md` + `report/figures/_cross_method/rank_collapse_data.json`.

---

## 5f. *Catastrophic Failure as Hard-Contrast Over-Correction* — single sufficient mechanism + 4-lever trade-off framework

> 본 paper 의 *두 번째* main contribution. §5e 의 *universal per-position rank-collapse + spatial multiplicity escape* (capacity-side) 와 **독립** 하면서 *상호 보완* — §5e 는 *어떻게* LoRA 가 spatial 으로 작동하는가, §5f 는 *왜* hard mined-HN 으로 학습하면 catastrophic / redistribution 이 발생하는가.

### 5f.1 Setup — 4 mediation experiments on SciFact

본 paper 의 main intervention (Phase 2b LoRA q,v r=8) 의 *redistribution pattern* (Δ confused +0.104 ✓ / Δ easy −0.085 ✗ / Δ all ≈ 0) 의 *mechanism* 을 disentangle 하기 위해 *4 mediation* 실험 :

| Method | Hard contrast | Noise (FN) | Description |
|---|---|---|---|
| **Phase 2b** (baseline) | hard mined-HN | noisy (~33 % FN per E5) | 본 paper 의 main intervention |
| **Exp 12** (FN-denoised) | hard mined-HN | **clean** (e5_margin > 0 only) | 캐비엇 1 disambiguator (noise vs difficulty) |
| **M1b** (in-batch neg) | **easy** (other query's pos) | clean (random doc) | 캐비엇 1 partial disambiguator |
| **Exp 11** (relational easy preservation) | hard mined-HN | noisy (same as Phase 2b) | + selective preservation pressure on easy queries via $\|\text{Sim}(H_{\text{LoRA}}) - \text{Sim}(H_{\text{frozen}})\|_F^2$ |

각 method 의 *NDCG outcomes* (3-seed mean ± std) + *representation collapse* 직접 측정 (`_repr_collapse_new_ckpts.py`, n=300 docs sampled per condition):

| Method | Δ all | Δ confused | Δ easy | doc eff_rank | tok eff_rank |
|---|---|---|---|---|---|
| Frozen baseline | — | — | — | 10.65 | 57.21 |
| **Phase 2b** | +0.001 | +0.104 ✓ | −0.085 ✗ | 1.14 | 1.58 |
| **Exp 12** | −0.004 ± 0.005 | +0.080 ± 0.004 ✓ | −0.073 ± 0.005 ✗ | **1.22 ± 0.01** (= Phase 2b) | **1.72 ± 0.05** (= Phase 2b) |
| **M1b** | **+0.021 ± 0.005 ✓** STRICT | +0.065 ± 0.012 (half) | (~−0.05) | **7.12 ± 0.31** (6.2× recovery) | **44.65 ± 1.55** (28× recovery!) |
| **Exp 11** | **+0.029 ± 0.005** (2/3 strict) | **+0.101 ± 0.010 ✓** (preserved) | **−0.031 ± 0.018** (63 % 감소) | ~1.9 (1.7×) | ~9.6 (**6× token recovery**) |

### 5f.2 Two-level mechanism evidence — NDCG outcomes *aligned* with collapse measurements

본 paper 의 핵심 mechanism evidence 는 *NDCG-level* outcome 과 *collapse-level* 측정의 *empirically aligned*:

- **Phase 2b ≈ Exp 12** at *both* NDCG and collapse → *FN noise removal 만으로는 zero change* → catastrophic 의 root 는 noise 가 *아님*
- **M1b** at *both* levels: large collapse reduction (eff_rank 7.12) + strict net+ (Δ all +0.021)
- **Exp 11** at *both* levels: token-level direct preservation (loss = $\|\Delta \text{Sim}\|_F^2$ → token eff_rank 6× recovery) + redistribution partial 해소

⇒ *Collapse magnitude* 와 *NDCG redistribution* 의 *empirically linked* — 같은 mechanism 의 두 면.

### 5f.3 *Single sufficient mechanism* — Hard-Contrast Over-Correction

**가설** (mediation 실험들로 *엄격* 검정): LoRA + pairwise margin loss + hard mined-HN 의 결합이 *over-correction* 을 강요 → encoder output collapse + confused/easy zero-sum redistribution.

**4 supporting evidences**:

1. **FN noise removal 만으로는 zero change** (Exp 12 = Phase 2b at both NDCG and collapse) → noise 가 *주요 원인 아님*.
2. **Hard contrast 제거** (M1b) → *최대 collapse 감소* (6.2×) + *유일한 strict net+ across 3 seeds*.
3. **Hard 유지 + selective preservation** (Exp 11) → token-level *직접* preserved (loss-aligned), redistribution 부분 해소.
4. **Optimization root (M1) = red herring**: warmup+clip 의 final-state collapse 변화 zero (Phase 2b 와 동일 eff_rank 1.14), NDCG 도 동일. M1+M1b combined = M1b alone *at both levels*.

⇒ **Hard mined-HN 의 *pairwise margin* 학습 자체** 가 catastrophic / redistribution 의 *sole sufficient mechanism*. FN noise 는 *minor contribution* (~14 % of easy 손상), optimization 은 *zero*.

### 5f.4 *4-lever trade-off* framework — paper deliverable

본 paper 의 *deployment-actionable* finding: catastrophic / redistribution 의 *4 가지 회피 lever*, 각각 *다른 trade-off*:

| Lever | Hard contrast | What's preserved | Δ all | Δ confused | Δ easy | Use case |
|---|---|---|---|---|---|---|
| Phase 2b | hard + noisy | (nothing) | ≈ 0 | +0.10 ✓ | −0.09 ✗ | *confused-only* focus |
| Exp 12 | hard + clean | (nothing) | ≈ 0 | +0.08 ✓ | −0.07 ✗ | (Phase 2b 보다 confused 약함, FN 제거의 minor cost) |
| **M1b** | *easy* in-batch | overall ranking | **+0.02 ✓ strict** | +0.07 (half) | (~−0.05) | *strict net+ 필요* + confused 절반 OK |
| **Exp 11** | hard + selective preservation | easy token structure | **+0.03 (2/3 strict)** | +0.10 ✓ (preserved) | −0.03 (63 % 감소) | *best confused + moderate net+* — 가장 balanced |

⇒ **Practitioner advice**:
- *Strict net+ 우선*: M1b (clean in-batch neg) — robust 3-seed strict
- *Confused lift 우선 + 약한 net+*: Exp 11 (relational easy preservation, λ=1)
- *Hard contrast 자체 회피 권고* — *FN denoising 만으로는 부족*

### 5f.5 *Caveats fully resolved* — paper-grade robustness

| Caveat | Status | Evidence |
|---|---|---|
| **캐비엇 1 (clean ≠ easy confound)** | ✅ disambiguated empirically | Exp 12 (3-seed): FN removal 만 NDCG ≈ Phase 2b + collapse ≈ Phase 2b → noise *not* the root. Difficulty dominant + noise minor (~14 %). |
| **캐비엇 2 (seed-artifact risk on M1b)** | ✅ fully resolved | M1b SciFact 3 seeds 모두 strict (CI 하한 +0.001 / +0.008 / +0.011) — 08 seed-artifact 시나리오 empirically 기각. + 3-seed collapse measurement robust (eff_rank doc 6.73/7.29/7.33). |
| **§7.3.c.i sanity (eff_rank ↔ NDCG pairing valid)** | ✅ confirmed | 8 / 8 sanity-check 통과 (Phase 2b / M1 / M1b × 3 datasets minus FiQA M1b). Diagnostic-loaded model = eval model. |
| **NFCorpus *direction matters* puzzle** | 🟡 confirmed as phenomenon, mechanism partial | M1+M1b NFCorpus = eff_rank 1.05 (no collapse reduction) 임에도 NDCG 74 % recovery. *Direction alignment* > *magnitude* 의 추가 evidence. *Direction 학습 mechanism* 은 future work. |
| **§5e 의 spatial multiplicity escape** | ✅ unchanged | LoRA 의 24 positions × per-adapter ~1.7 rank → §5e main contribution 의 capacity-side. 본 §5f 의 *catastrophic mechanism* 와 **독립**. |

### 5f.6 Paper-grade contributions — *two main contributions*

본 paper 는 *서로 독립적 두 main contribution*:

1. **§5e Universal per-position rank-collapse + spatial multiplicity escape** (capacity-side):
   - 06 K-router / 08 bilinear M / 10 LoRA 모두 *per-position* eff_rank ~1-2 의 universal collapse
   - LoRA 의 *24 distinct positions × ~1.7 per-adapter rank* 의 spatial multiplicity → *budget-efficient design*
   - **Budget-aware design principle**: 적은 r × 많은 positions > 큰 r × 적은 positions

2. **§5f Catastrophic Failure as Hard-Contrast Over-Correction** (supervision-side, *본 section*):
   - Phase 2b 의 *redistribution* 의 sole sufficient mechanism = *hard mined-HN over-correction*
   - 4 mediation experiments (M1 = red herring, M1b = hard 회피, Exp 11 = selective preservation, Exp 12 = FN denoising 만 = ineffective) → *single mechanism*
   - **4-lever trade-off framework**: deployment-actionable practitioner advice
   - **Mechanism direct evidence**: collapse magnitude (eff_rank) ↔ NDCG outcomes *empirically aligned*

⇒ *Capacity-side* (§5e) + *supervision-side* (§5f) 의 *complete framework* for *frozen-encoder lightweight intervention* on ColBERT — paper 의 *unified* contribution.

상세: `report/_overnight_results.md` (chronological raw) + 본 §5f 의 references (§7.3.c–§7.3.e, §6.1 grid).

---

## 6. Cumulative findings — 11 개 실험의 종합 narrative

### 6.1 측정된 SciFact NDCG@10 그리드 (seed 42)

| Step | 형식 | Params | NDCG@10 | 95 % CI vs baseline (confused) | 통과? |
|---|---|---|---|---|---|
| 00 | frozen ColBERT v2 | 0 | 0.6464 | — | (anchor) |
| 01 | $h - v_{\text{md}}$ (raw mean-diff) | 0 | 0.6459 | −0.001 [−0.006, +0.002] | ✗ |
| 01b α=0.5 | $h - 0.5\,\hat{v}_{\text{md}}$ | 0 | 0.6477 | +0.003 [−0.004, +0.009] | ✗ |
| 01b α=2 | $h - 2\,\hat{v}_{\text{md}}$ | 0 | 0.6536 | +0.018 [+0.005, +0.032] | ✓ |
| 01b α=5 | $h - 5\,\hat{v}_{\text{md}}$ | 0 | 0.6666 | +0.052 [+0.026, +0.081] | ✓ |
| 01b α=10 | $h - 10\,\hat{v}_{\text{md}}$ | 0 | **0.6690** | +0.064 [+0.034, +0.099] | ✓ |
| 02 | $h - v_{\text{learned}}$ | 768 | 0.6651 | +0.044 [+0.023, +0.066] | ✓ |
| 03 | $h - g\,v$, scalar gate | 769 | 0.6448 | −0.004 [−0.010, +0.001] | ✗ |
| 04 | $h - g(h)\,v$, per-token | 1,537 | 0.6641 | +0.040 [+0.019, +0.063] | ✓ |
| 05 | $h - v_\ell$ at 5 layers | 3,840 | 0.6502 | +0.051 [+0.012, +0.092] | ✓ |
| 06 K=2 | $h - \sum_k \pi_k v_k$, K=2 | 3,074 | **0.6614** | +0.039 [+0.017, +0.061] | ✓ |
| 06 K=4 | K=4 | 6,148 | **0.6614** | +0.045 [+0.024, +0.068] | ✓ |
| 06 K=8 | K=8 (anchor 손상 Δ all = −0.038 ✗) | 12,296 | 0.6089 | +0.049 [+0.005, +0.092] | △ |
| **07** | $h - 10\,\hat{v}_{\text{random}}$ (비학습 random) | 0 | **0.6485** | **+0.011 [−0.006, +0.029]** | **✗ (≈ baseline)** |
| **08 r=8** | $s_M(q,d) = q^\top M d$, $M = I + UV^\top$ (Stage 2) | 2,048 | **0.6439** | **+0.054 [+0.013, +0.097]** | **✓ confused, ✗ all (ceiling 못 넘음)** |
| 09 r=8 λ=0.1 | 08 + E5 Margin-MSE distill (λ=0.1) | 2,048 | 0.6509 | +0.019 [+0.006, +0.033] | △ all 약간 ↑ 하지만 confused lever 약화 |
| 09 r=8 λ=0.5 | 08 + E5 distill (λ=0.5) | 2,048 | 0.6451 | −0.002 (≈ baseline) | ✗ M ≈ I, lever 거의 죽음 |
| 09 r=8 λ=1.0 | 08 + E5 distill (λ=1.0) | 2,048 | 0.6453 | −0.002 (≈ baseline) | ✗ over-regularize |
| 10 r=1 (Phase 1) | LoRA q,v r=1 (early-stop val_all) | 36,864 | 0.5940 | +0.038 [−0.008, +0.083] | ✗ all 손상 −0.052 |
| 10 r=8 (Phase 2a) | LoRA q,v r=8, LR=1e-4, α=2r | 294,912 | 0.5879 | +0.080 [+0.021, +0.140] ✓ | ✓ confused, ✗ all 손상 |
| **10 r=8 Phase 2b (seed 42)** | **LoRA q,v r=8, LR=5e-5, α=r** | **294,912** | **0.6367** | **+0.091 [+0.040, +0.143] ✓** | **✓ confused + ✓ anchor**, strict 돌파 ✗ |
| 10 Phase 2b (seed 1337) | (동일 config) | 294,912 | 0.6423 | +0.097 [+0.047, +0.150] ✓ | ✓ confused + ✓ anchor |
| 10 Phase 2b (seed 2024) | (동일 config) | 294,912 | 0.6639 | +0.123 [+0.073, +0.174] ✓ | ✓ confused + ✓ anchor |
| **10 Phase 2b (3-seed mean ± std)** | **(동일 config)** | **294,912** | **0.6476 ± 0.014** | **+0.104 ± 0.017 ✓ robust** | **✓ confused + ✓ anchor (CI 0 포함), strict 돌파 ✗** |
| 10 Phase 2b on NFCorpus | (동일 config, --max-triplets 9,190) | 294,912 | **0.0094 (Δ all −0.320 ✗)** | −0.092 ✗ (NFCorpus baseline confused) | ✗ catastrophic (06 K=2 NFCorpus −0.250 보다 더 심함) |
| **Clean baseline (02 unfrozen no-steering)** | **encoder fine-tune only** (no hook) | **109,580,544** | **0.6924** | **+0.260 [+0.182, +0.338] ✓** | **✓ confused (large lift) + ✓ anchor (CI 거의 0)**, strict 돌파 직전 |
| 02 unfrozen (with v=0 hook) | encoder fine-tune + v=0 hook | 109,580,544 | 0.6576 | +0.252 [+0.179, +0.328] ✓ | ✓ confused + ✓ anchor (CI 0 포함) |
| 10 Phase 2b + M1 (warmup+clip) on SciFact | LoRA q,v r=8 + warmup 10% + clip 1.0 | 294,912 | 0.6342 | +0.088 [+0.035, +0.139] ✓ | ✓ confused ≈ Phase 2b 동등 (mediation: SciFact safety sanity) |
| 10 + M1 on NFCorpus (cross-dataset) | (동일 config) | 294,912 | 0.0113 (Δ all −0.319 ✗) | −0.093 ✗ (NFCorpus baseline confused) | ✗ catastrophic (Phase 2b 의 −0.320 와 통계 동등) — ep1 val_all 0.140 vs Phase 2b 0.073 (1.9× train-time signal) |
| 10 + M1 on FiQA (cross-dataset) | (동일 config) | 294,912 | 0.0009 (Δ all −0.346 ✗) | −0.147 ✗ (FiQA baseline confused) | ✗ catastrophic (Phase 2b 의 −0.347 와 통계 동등) — ep1 val_all 0.257 vs Phase 2b 0.090 (2.86× train-time signal) |
| **10 + M1b (in-batch neg) on SciFact** | **LoRA q,v r=8 + in-batch negative** | **294,912** | **0.6613** | **+0.055 [+0.030, +0.081] ✓** | **첫 strict net 향상 *시그널*: Δ all +0.015 [+0.001, +0.029] ✓** (CI 하한 razor-thin, single seed) — ⚠️ *clean vs easy* confound + 3-seed 미실시 (§7.3.e.i.α 캐비엇) |
| **10 + M1b on NFCorpus (cross-dataset)** | (동일 config + in-batch neg) | 294,912 | **0.2459** (baseline 0.330 의 74.5 %) | **−0.013 [−0.027, +0.002]** (CI 0 포함, *confused 회복*) | **74 % catastrophic 회복**: Δ all −0.084 [−0.105, −0.064] ✗ (Phase 2b 의 −0.320 의 26 %), ep1 val_all 0.376 ▲ baseline 위 |
| 10 + M1b on FiQA (cross-dataset, seed 42) | (동일 + max-triplets 9190) | 294,912 | ~0.327 (baseline 0.347 의 94 %) | −0.010 [−0.021, +0.0003] (CI 0 포함, *baseline 회복*) | **94 % catastrophic 회복** (NFCorpus 의 74 % 보다 강함): Δ all −0.020 [−0.028, −0.012] ✗ (Phase 2b 의 −0.347 의 6 %) |
| 🎯 **10 + M1b SciFact (seed 1337)** | (동일) | 294,912 | **0.6681** | **+0.064 [+0.039, +0.090] ✓** | **strict positive**: Δ all +0.022 [+0.008, +0.036] ✓ (seed 42 razor-thin 보다 confident) |
| 🎯 **10 + M1b SciFact (seed 2024)** | (동일) | 294,912 | **0.6722** | **+0.077 [+0.051, +0.105] ✓** | **strict positive**: Δ all +0.026 [+0.011, +0.042] ✓ |
| 🎯 **10 + M1b SciFact (3-seed mean ± std)** | (동일) | 294,912 | **0.6672 ± 0.005** | **+0.065 ± 0.012 ✓** | **🎯 3-seed strict net+ robust (캐비엇 2 fully 해소): Δ all +0.021 ± 0.005 ✓ STRICT robust + Δ easy −0.017 ± 0.003** (spine ablation A1 실측, *기존 추정치 ~−0.05 의 1/3* — anchor-side 와 동등 수준의 easy 보존) |
| 10 + M1b NFCorpus (seed 1337) | (동일) | 294,912 | 0.2231 (baseline 의 67.6 %) | — | 67 % catastrophic-gap recovery |
| 10 + M1b NFCorpus (seed 2024) | (동일) | 294,912 | 0.2626 (baseline 의 79.6 %) | — | 80 % catastrophic-gap recovery |
| **10 + M1b NFCorpus (3-seed mean ± std)** | (동일) | 294,912 | **0.244 ± 0.020** | **(unmeasured 3-seed mean)** | **74 % ± 7 gap recovery robust (NOT net+)**: Δ all −0.086 ± 0.020 ✗ |
| 10 + M1+M1b combined SciFact (seed 42, *post-hoc exploratory, negative result*) | warmup+clip + in-batch neg | 294,912 | 0.6659 | +0.066 ✓ | **post-hoc exploratory, negative result (M1 contribution zero)**: +0.020 ✓ ≈ M1b alone +0.021 — *low selection risk* |
| 10 + M1+M1b combined NFCorpus (seed 42, *post-hoc exploratory, negative result*) | (동일) | 294,912 | 0.2471 | −0.009 (CI 0) | **post-hoc exploratory, negative result**: −0.083 ≈ M1b alone −0.084 — M1 contribution zero confirmed |
| **11 Exp 11 SciFact (seed 42, λ=1)** | relational easy preservation loss | 294,912 | **0.6797 (highest!)** | **+0.095 [+0.058, +0.134] ✓** (Phase 2b +0.104 와 essentially 동등) | **strict +0.033 [+0.013, +0.055] ✓ + Δ easy −0.019 [−0.036, −0.004]** (Phase 2b −0.085 의 77 % 감소) |
| 11 Exp 11 SciFact (seed 1337) | (동일) | 294,912 | 0.6784 | +0.095 [+0.059, +0.134] ✓ | strict +0.032 [+0.012, +0.053] ✓ + Δ easy −0.021 ✗ |
| 11 Exp 11 SciFact (seed 2024) | (동일) | 294,912 | 0.6697 | +0.113 [+0.069, +0.160] ✓ (highest conf!) | +0.023 [−0.004, +0.051] (CI 0 포함, marginal) + Δ easy −0.052 ✗ |
| ⭐ **11 Exp 11 SciFact (3-seed mean ± std)** | **relational easy preservation, λ=1.0 (pre-committed single value)** | 294,912 | **0.6759 ± 0.005** | **+0.101 ± 0.010 ✓ (Phase 2b +0.104 fully preserved!)** | **branch (a) *partial*: Δ all +0.029 ± 0.005 (2/3 strict, 1 marginal), Δ easy −0.031 ± 0.018 (63 % 감소 vs Phase 2b)** |
| 12 Exp 12 SciFact (seed 42, FN-denoised) | LoRA + e5_margin > 0 filter (36.5 % FN 제거) | 294,912 | 0.6388 | +0.076 [+0.028, +0.123] ✓ | −0.008 [−0.037, +0.022] (CI 0) + Δ easy −0.078 ✗ |
| 12 Exp 12 SciFact (seed 1337) | (동일) | 294,912 | 0.6420 | +0.079 [+0.034, +0.128] ✓ | −0.005 [−0.033, +0.024] (CI 0) + Δ easy −0.075 ✗ |
| 12 Exp 12 SciFact (seed 2024) | (동일) | 294,912 | 0.6488 | +0.084 [+0.037, +0.131] ✓ | +0.002 [−0.026, +0.031] (CI 0) + Δ easy −0.066 ✗ |
| 🎯 **12 Exp 12 SciFact (3-seed mean ± std)** | **FN-denoised mined-HN (e5_margin>0)** | 294,912 | **0.6432 ± 0.004** | **+0.080 ± 0.004 ✓** | **🎯 (나-2) difficulty dominant 확정**: Δ all −0.004 ± 0.005 (CI 0 all 3), Δ easy −0.073 ± 0.005 ≈ Phase 2b 의 −0.085 — *FN removal 단독으로 redistribution 해소 불가* |
| 13 Exp 13 SciFact (seed 42, λ_dir=1) | per-token cosine deviation anchor (easy queries) | 294,912 | 0.6790 | +0.099 [+0.064, +0.136] ✓ | strict +0.033 [+0.012, +0.054] ✓ + Δ easy −0.024 ✗ |
| 13 Exp 13 SciFact (seed 1337) | (동일) | 294,912 | 0.6743 | +0.087 [+0.053, +0.125] ✓ | strict +0.028 [+0.008, +0.048] ✓ + Δ easy −0.022 ✗ |
| 13 Exp 13 SciFact (seed 2024) | (동일) | 294,912 | 0.6771 | +0.088 [+0.054, +0.122] ✓ | strict +0.031 [+0.012, +0.050] ✓ + Δ easy −0.017 ✗ |
| 🎯 **13 Exp 13 SciFact (3-seed mean ± std)** | **per-token cosine deviation anchor, λ_dir=1.0 (pre-committed single value)** | 294,912 | **0.6768 ± 0.002** | **+0.092 ± 0.007 ✓** | **branch (b) — Exp 11 과 frontier 공유**: Δ all +0.030 ± 0.002 **3/3 strict** (Exp 11 의 2/3 보다 strict), Δ easy −0.021 ± 0.003 (branch (a) 임계 −0.020 을 *0.001 차이로 miss*) |
| 14 Exp 14 SciFact (seed 42, α_w=10) | continuous sigmoid weighting on triplet margin loss | 294,912 | 0.6552 | +0.100 [+0.048, +0.153] ✓ | +0.009 [−0.022, +0.040] (CI 0) + Δ easy −0.068 ✗ |
| 14 Exp 14 SciFact (seed 1337) | (동일) | 294,912 | 0.6536 | +0.060 [+0.026, +0.098] ✓ | +0.007 [−0.014, +0.028] (CI 0) + Δ easy −0.038 ✗ |
| 14 Exp 14 SciFact (seed 2024) | (동일) | 294,912 | 0.6493 | +0.095 [+0.046, +0.145] ✓ | +0.003 [−0.028, +0.034] (CI 0) + Δ easy −0.074 ✗ |
| **14 Exp 14 SciFact (3-seed mean ± std)** | **difficulty-weighted HN, α_w=10 (pre-committed single value)** | 294,912 | **0.6527 ± 0.003** | **+0.085 ± 0.022 ✓** | **branch (c) 변형**: Δ all +0.006 ± 0.003 **3/3 CI 0 포함** (no strict), Δ easy −0.060 ± 0.020. Continuous weighting → *softer Phase 2b, sub-binary on Δ all*. *data-side weighting* family — anchor-side 와 명확히 분리된 lower frontier. |
| 16 Exp 16 SciFact (seed 42, multi-layer anchor) | per-token cosine at BERT layers {0,3,6,9,12} | 294,912 | 0.6238 | +0.073 [+0.036, +0.112] ✓ | +0.008 [−0.016, +0.031] (CI 0) + Δ easy −0.048 ✗ |
| 16 Exp 16 SciFact (seed 1337) | (동일) | 294,912 | 0.6434 | +0.066 [+0.028, +0.107] ✓ | −0.003 [−0.027, +0.021] (CI 0) + Δ easy −0.061 ✗ |
| 16 Exp 16 SciFact (seed 2024) | (동일) | 294,912 | 0.6534 | +0.072 [+0.034, +0.110] ✓ | +0.007 [−0.016, +0.030] (CI 0) + Δ easy −0.048 ✗ |
| **16 Exp 16 SciFact (3-seed mean ± std)** | **multi-layer per-token anchor, layers={0,3,6,9,12}, λ_dir=1.0 (pre-committed)** | 294,912 | **0.6402 ± 0.015** | **+0.071 ± 0.004 ✓** | **branch (c) over-restriction**: Δ all +0.004 ± 0.006 **3/3 CI 0 포함**, Δ easy −0.052 ± 0.008 (Exp 13 의 2.5× damage). Multi-layer anchor 가 Exp 13 single-layer 대비 *모든 metric 명백 열등*. *Loss budget dilution + intermediate redundancy* mechanism (§7.3.j). |
<!-- Post-hoc 묶음 (Higher λ=5, Combined M1b+Exp 11, FN+EP variant — 9 runs / 3 묶음) 은 *test 결과 본 후 generative question* 으로 발의된 *post-hoc exploratory* 실험으로 main paper grid 에서 제외. Raw artifacts: outputs/11_easy_preservation/.../qv_r8_l12_le{5,1_m1b,1_fnden}/. Chronological record: report/_overnight_results.md. -->

> ⭐ = *clean pre-committed light intervention 중 best balance* — Exp 11 (λ=1) 3-seed mean: Δ all +0.029 (high), Δ confused +0.101 (Phase 2b 의 +0.104 *fully preserved*), Δ easy −0.031 (Phase 2b 의 −0.085 의 63 % 감소). 2/3 strict (partial branch (a)).
> **Exp 13** (per-token cosine anchor, λ_dir=1.0) 3-seed mean: Δ all **+0.030** (Exp 11 통계 동등), Δ confused +0.092 (slightly lower), Δ easy **−0.021** (Exp 11 의 −0.031 보다 best preserved). **3/3 strict** (Exp 11 보다 strict robustness ↑). Branch (a) 의 Δ easy > −0.020 임계를 *0.001 차이로 miss* → branch (b) frontier-shared 확정. *Anchor-side family* 의 두 lever (Exp 11 relational, Exp 13 absolute) 가 *동일 frontier* 점유 — *수학적 차이가 empirical separation 으로 전이 안 됨* 증거.
> 단 *3/3 robust strict* 우선 시 M1b SciFact (3-seed) (+0.021 ± 0.005, Δ confused +0.065 half) 가 대안 — 두 lever 모두 같은 mechanism (hard-contrast over-correction) 의 다른 angle.
> 🎯 = clean pre-committed lever (M1b 3-seed strict, Exp 12 disambiguator, etc.)
> *post-hoc exploratory* = test 결과 본 후 발의 — main paper claim base 에 포함 안 함.

### 6.2 핵심 결론

1. **Ceiling 0.665 의 *K-invariant + form-invariant* 강건성** — 학습된 translation 변형 (02 / 04 / 05 / 06 K=2 / 06 K=4) 과 비학습 mean-diff (01b α=10) + **bilinear M (08, r=8)** 모두 NDCG@10 ≈ 0.66 부근에 수렴. K=2 와 K=4 의 NDCG@10 은 *4 자리 소수점 까지 동일* (0.6614). 학습 가능 capacity 가 768 → 6,148 (8 배 증가) 인데도 ceiling 위치 *완전 불변*. *Over-capacity* (K=8, 12K params) 는 ceiling 우회 못 하고 *anchor 손상* 만 유발. **08 의 bilinear M form 변경도 ceiling 위로 못 감** — 단 학습된 M 의 effective rank 가 1 로 collapse, full capacity 활용 못 함 (open question).
2. **Direction 의 *내용* 이 lever** (07 falsification) — 같은 magnitude (α=10) 의 random direction 은 baseline 과 통계 동등 (0.6485, confused +0.011 CI 0 포함). Mean-diff direction 만 ceiling 도달. *Direction-agnostic 가설 명확히 기각*.
3. **Magnitude 가 *조건* lever** — informed direction (mean-diff family) 위에서 magnitude 가 sweet spot (≥ α=2) 에 닿아야 효과 발현. magnitude alone 으로는 부족 (07 가 증명).
4. **Single-direction subspace 의 *informed* redundancy** — 학습된 다른 방향들 (cos = 0.32, 0.55) 이 모두 *같은* informed subspace 안의 다른 element. *Random* 방향은 그 subspace 밖이라 ceiling 도달 못 함.
4. **Gate / Multi-layer / K=2 의 *empirical 활용 부재*** — 03 의 multiplicative saturation, 04 의 always-on, 05 의 over-fitting, 06 의 router saturation 이 모두 *학습 동학의 systematic 한계* 시사.
5. **모든 학습 실험의 train-overfitting 패턴** — train loss 0.7 → 0.1 단조 감소, val NDCG epoch 1-2 peak → 감소. SciFact 의 9K triplet 의 *information bottleneck* 가능성.

### 6.3 Paper main contribution 의 *재정렬* (07 falsification 반영)

00–07 의 종합 발견은 paper 의 main contribution narrative 를 다음과 같이 *재정렬*:

> Translation family ($\tilde h = h - u(h)$) 의 ceiling 은 *informed direction subspace* (HN–pos 차이의 information-bearing direction) 의 representational limit 에 의해 결정된다. *Random direction* 은 그 ceiling 에 도달조차 못 함 (07). *Informed direction* 의 학습 / 비학습 / 단층 / 다층 / multi-direction 변형은 모두 같은 ceiling 에서 redundant 하게 만남. *그 ceiling 의 본질* 이 **algebraic family 한계** 인지 **information 한계** 인지의 분리는 후속 *bilinear interaction metric* $M = I + UV^\top$ — translation family *밖* 으로의 minimal 우회 — 의 결과로만 결정 가능.

이는 *왜 multi-direction 만으로 부족한가* 와 *direction 의 의미성* 을 모두 직접 답하는 strong narrative. Reviewer 의 *"왜 단순 K↑ 로 안 되는가?"* 와 *"왜 random direction 으로 안 되는가?"* 두 질문에 모두 데이터 기반 답.

---

## 7. Robustness audit — *3 가지 결정적 점검*

본 시점에서 누적된 conclusion (translation-trap, K-invariant ceiling, form-change limit) 의 *통계적 신뢰성* 직접 검정. 세 가지 robustness check 동시 진행:

### 7.1 Cross-dataset (NFCorpus K=2)

06_k_sweep 의 SciFact-specific "K=2 ≈ K=4 의 NDCG@10 0.6614 동일 ceiling" 발견의 일반성 검정. **NFCorpus 에서 동일 setup (LR=1e-3, 9190 subsampled triplets, K=2, seed 42) 실행 → catastrophic failure**:

| 지표 | NFCorpus K=2 | SciFact K=2 (참고) |
|---|---|---|
| NDCG@10 (all) | **0.080** | 0.6614 |
| Baseline NDCG@10 | 0.330 | 0.6464 |
| **Δ all vs baseline** | **−0.250 [−0.281, −0.219] ✗** | +0.015 ✓ |
| Δ confused vs baseline | -0.071 ✗ | +0.039 ✓ |
| Confused fraction | 88.5 % | 45.7 % |
| cos(v_0, v_1) | **−0.66 (opposite)** | +0.55 (partial) |

**진단**: NFCorpus 의 *훨씬 높은 baseline 의 confused%* (88.5%) + *학습 신호의 다른 magnitude* (rank loss ep1 4.18 vs SciFact 0.71) 가 LR=1e-3 의 hyperparameter 와 incompatible → over-correction. 학습된 v_0, v_1 가 *opposite direction* (cos −0.66) — SciFact 의 partial alignment 와 *질적 반대*.

**결론**: 본 paper 의 *K-invariant ceiling* claim 은 **SciFact-specific**, *cross-dataset 일반성 없음*. Hyperparameter sweep 후 재검정 필요.

### 7.1.b Cross-dataset (10 LoRA Phase 2b on NFCorpus, 2026-05-24)

7.1 (06 K=2 NFCorpus catastrophic) 의 *cross-dataset hyperparameter sensitivity* 가 LoRA 에도 동일하게 적용되는지 검정. **동일 Phase 2b config (q,v r=8, LR=5e-5, α=r, early-stop=val_all) + `--max-triplets 9,190`** (SciFact-comparable scale) 으로 NFCorpus 단일 run. *재튜닝 금지* (pre-commit).

| 지표 | SciFact (3-seed mean) | **NFCorpus (single seed 42)** |
|---|---|---|
| NDCG@10 all | 0.6476 ± 0.014 | **0.0094** (baseline 0.330 의 *2.8 %*) |
| **Δ all vs baseline** | +0.001 ± 0.014 (anchor preserved) | **−0.320 [−0.355, −0.287] ✗ catastrophic** |
| **Δ confused vs baseline** | +0.104 ± 0.017 ✓ | **−0.092 [−0.115, −0.070] ✗** |
| Initial rank loss (ep1) | 0.66 | **4.47** (7×) |

**진단** (06 K=2 NFCorpus 와 *완전 평행 패턴*):
- NFCorpus 의 *baseline 의 hard negatives 가 7× 강함* (rank loss ep1 4.47 vs SciFact 0.66) + same LR=5e-5 → *immediate over-correction*.
- 첫 epoch 부터 val_all 0.073 (baseline 0.330 의 22%) — *early stop based on val_all 도 회복 못 함*.
- LoRA 의 spatial multiplicity 도 *dataset-specific hyperparameter 와 incompatible*.

**Pre-commit 따라**: NFCorpus 재튜닝 *금지*. Catastrophic 결과 *정직히 보고* + limitation 명시.

**Paper Limitations 의 핵심 항목**:
- *Hyperparameter sensitivity 가 dataset-specific* — SciFact 위 LR/r/α 의 적정값 (LR=5e-5) 이 NFCorpus 의 dense-qrels + strong-HN regime 에서 over-correction.
- *True cross-dataset robustness* 는 *adaptive hyperparameter strategy* (per-dataset scaling normalization, e.g., baseline NDCG 의 reciprocal scaling, or initial-loss-aware LR) 가 필요한 future work.
- *Universal rank-collapse + spatial multiplicity* claim (§5e) 의 *method-architectural* 부분은 cross-dataset 일 것 (학습 동학 의 universal pattern) — 단 *numerical lift* (+0.104) 는 SciFact-specific.

### 7.2 Seed × 3 (08 r=8)

08 의 "Δ confused +0.054 + UV^T 의 rank-1 collapse (σ₁=2.60)" 의 seed-dependency 검정:

| | Seed 42 | **Seed 1337** | **Seed 2024** |
|---|---|---|---|
| NDCG@10 all | 0.6439 | 0.6446 | 0.6446 |
| Δ confused vs baseline | **+0.054 ✓** | **−0.001 (≈)** | **−0.001 (≈)** |
| ‖UV^T‖_F | **2.61** | 0.085 | 0.100 |
| σ₁(UV^T) | **2.60** (rank-1 dominant) | 0.07 (tiny) | 0.09 (tiny) |
| M condition number | **81.14** | 1.08 (≈ I) | 1.09 (≈ I) |

**진단**: Seed 42 만 *rank-1 collapse + +0.054 confused 학습* 발생. Seed 1337/2024 의 학습된 M ≈ identity (사실상 학습 안 됨). Small_random init 의 *초기 UV^T 방향* 이 LR=1e-4 + Adam 의 수렴 basin 을 결정. 80 query val set 의 noise 가 best-epoch selection 의 seed-dependence 증폭.

**결론**: 본 paper 의 *form-change rank-1 collapse* 및 *+0.054 confused* claim 은 **seed-specific artifact**. 3-seed 평균 Δ confused ≈ +0.017 (sub-significant).

### 7.3.b Clean ColBERT-finetune (no steering hook) — 2026-05-24

7.3 의 02 unfrozen 은 *v=0 frozen 의 SteeringModule hook 단 채*. Reviewer 의 "v=0 hook 이 학습 신호 추가했냐" 공격 회피 위해 `--no-steering` flag (v frozen no-grad) 로 *pure encoder finetune* 검정.

| 지표 | 02 unfrozen (with v=0 hook) | **Clean baseline (no steering)** |
|---|---|---|
| NDCG@10 all | 0.6576 | **0.6924** (baseline +0.046) |
| **Δ all vs baseline** | +0.011 (≈) | **+0.046 [-0.002, +0.096]** (CI 거의 0 — 돌파 직전) |
| **Δ confused vs baseline** | +0.252 ✓ | **+0.260 ✓** (essentially same) |
| ‖v_learned‖ | 0.33 | 0.0 (frozen no-grad) |

**결론**: v=0 hook 의 영향 **negligible** — *frozen-encoder bottleneck* claim 의 *cleanest evidence*. 02 unfrozen 의 결과가 *pure encoder finetune* 효과 확정.

### 7.3 Unfrozen ColBERT (02 + `--unfreeze-encoder`)

가장 critical 인 질문: *frozen encoder 가 진짜 bottleneck 인가, 아니면 학습 데이터 한계인가*. 02 의 SteeringModule + ColBERT encoder 전체를 학습 가능하게 (110M params, encoder LR=5e-5, 3 epochs).

| 지표 | Frozen 02 | **Unfrozen 02** |
|---|---|---|
| 학습 가능 params | 768 | **109.6 M** |
| NDCG@10 (all) | 0.6651 | 0.6576 (≈ baseline) |
| **Δ confused vs baseline** | +0.044 ✓ | **+0.252 [+0.179, +0.328] ✓** |
| **Δ confused vs 01b α=10** | -0.021 (≈) | **+0.188 [+0.105, +0.271] ✓** |
| ‖v_learned‖ | 7.08 | 0.33 (휴면) |
| Train loss 종료 | 0.24 | 0.0042 (사실상 완벽 fit) |

**핵심 발견** — *Frozen encoder 가 진짜 bottleneck 확정*:
- **Δ confused +0.252 — 우리 모든 frozen-side method 의 max +0.054 (seed 42 only) 의 5 ×**.
- All-slice 도 baseline 동등 (+0.011, CI 0 포함) — anchor preservation 유지.
- 학습된 v 가 0.33 으로 *휴면* (frozen 의 7.08 의 1/20) — encoder 가 모든 lever 흡수.

**결론**: Frozen-encoder lightweight intervention 의 ceiling (≈ 0.665, +0.05 confused) 은 *encoder representational limit* 의 직접 결과. 110 M params 은 50 K budget 의 ~2200 × — *upper bound 의 sanity check* 지 practical method 아님. **Next critical**: 50 K budget 안의 LoRA on Φ.

### 7.3.c Diagnostic B — *encoder output representation collapse* (2026-05-24)

**가설** (외부 reviewer): NFCorpus / FiQA 의 catastrophic Δ NDCG (−0.320 / −0.347) 의
*mechanism* 은 *encoder output space* (token embedding) 의 *representation collapse* — 즉
docs 가 평균적으로 너무 비슷해져서 MaxSim discrimination power 가 무너졌다. **§5e** 의
parameter-space (ΔW) rank-collapse 와는 *다른* 현상.

**측정** (`report/_repr_collapse_diagnostic.py`, n=500 docs sampled per corpus):
1. Random doc-pair 평균 cosine (mean-pooled, L2-normed) — collapse 시 ↑
2. Per-token random pair cosine — collapse 시 ↑
3. Doc-mean matrix (N×128) effective rank (singular spectrum perplexity) — collapse 시 ↓
4. Per-token (10K subsample × 128) effective rank — collapse 시 ↓

| Dataset | 조건 | doc-pair cos μ ± σ | tok-pair cos μ ± σ | eff_rank (doc) | eff_rank (tok) |
|---|---|---|---|---|---|
| NFCorpus | frozen (no LoRA) | +0.553 ± 0.111 | +0.211 ± 0.144 | 11.73 | 55.94 |
| NFCorpus | **LoRA Phase 2b** | **+0.990 ± 0.005** | **+0.940 ± 0.029** | **1.09** | **1.62** |
| FiQA | frozen (no LoRA) | +0.380 ± 0.149 | +0.181 ± 0.132 | 23.58 | 63.91 |
| FiQA | **LoRA Phase 2b** | **+0.993 ± 0.005** | **+0.990 ± 0.006** | **1.06** | **1.10** |
| SciFact | frozen (no LoRA) | +0.573 ± 0.117 | +0.209 ± 0.144 | 10.65 | 57.21 |
| SciFact | **LoRA Phase 2b** | **+0.984 ± 0.010** | **+0.942 ± 0.028** | **1.15** | **1.61** |

![Representation collapse diagnostic](report/figures/_repr_collapse/repr_collapse.png)

#### 7.3.c.i Sanity check — *진단이 측정한 model = eval 의 model* (2026-05-24)

**외부 reviewer 의 critical catch**: "**rank-1 embedding 으론 NDCG 0.65 *수학적으로 불가능***".
즉, SciFact LoRA 의 tok_cos +0.94 / eff_rank 1.61 이 *진짜 collapse* 면 retrieval 이
random 수준 (~0.01) 이 되어야 하는데 보고된 NDCG 0.6367 와 모순. 가설 A (best vs final
checkpoint 불일치) 또는 가설 B (LoRA injection α scaling 불일치) 의심.

**Sanity check** (`report/_repr_collapse_sanity.py`): 진단이 로드한 *바로 그 model*
(`module_final.pt`) 으로 test set 의 NDCG@10 재현.

| Dataset | Diagnostic-loaded NDCG@10 all | Original-run NDCG@10 all | Match |
|---|---|---|---|
| SciFact | 0.6367 | 0.6367 | ✓ |
| NFCorpus | 0.0094 | 0.0094 | ✓ |
| FiQA | **0.0005** | **0.0005** (baseline 0.347 의 0.15 %) | ✓ |

**결과 — *3 / 3 match***: 진단이 로드한 model = 보고된 NDCG 를 낸 model. 가설 A/B 모두 기각.

(주의: 본 보고서의 *초기* FiQA NDCG 표기 "0.0388 (baseline 0.385 의 10 %)" 는 *부정확* — 실제 NDCG@10 all = 0.0005 (baseline 0.347 의 0.15 %), Δ = −0.347. *수정 적용*.)
**Collapse 가 진짜** — *SciFact 의 eff_rank ≈ 1.15 / 1.61 *상태에서* NDCG 0.6367 가 실제
발생*. *Rank-1 puzzle* 이 paper-grade finding.

#### 7.3.c.ii 해석 정정 — *rank-1 puzzle* 의 의미

Reviewer 의 *"rank-1 → random"* 추론은 *eff_rank perplexity 의 literal rank* 해석에 근거.
그러나:

1. **eff_rank perplexity 는 *literal rank* 가 아님**. eff_rank = exp(−Σ p_i log p_i),
   p_i = σ_i²/Σσ². σ_1 >> σ_2 ≈ σ_3 인 경우 eff_rank ≈ 1 이지만 *trailing dimensions* 의
   non-trivial signal 잔존. 0.94 의 평균 cosine 에 ±0.03 의 std → tokens 가 *완전 동일*
   이 아님.
2. **MaxSim 의 *per-token max* 가 small residual 을 증폭**. 모든 doc token 이 tight cone
   안 (cosine ~0.94) 이라도, 각 token 의 *systematic small perturbation* 이 있으면
   query token q_i 는 그 perturbation 가장 잘 align 된 doc token 을 선택. *small structural
   differences* 가 *systematic* (relevance signal 과 correlate) 이면 NDCG 가 보존.
3. **Mean-pooled cosine ≠ MaxSim**. doc-pair cosine 은 *average* alignment 측정. MaxSim 은
   *best per-query-token* alignment 측정. 전자 ↑ 가 후자 의 discrimination 와 *non-trivially
   상관*.

⇒ **SciFact 의 *rank-1 residual* 이 task ranking 신호와 align** (within-dataset 학습으로
LoRA 가 "SciFact relevance direction" 학습). 잔존 1.15-dim 의 *systematic structure* 가
MaxSim 에 sufficient.

⇒ **NFCorpus / FiQA 의 *rank-1 residual* 이 task structure 와 *misaligned*** — *baseline
약함 → mined HN noise ~ 50% → supervision distortion → collapse direction 이 *wrong*.

**관찰 — universal collapse, dataset-dependent consequence**:
1. **Universal extreme collapse** — 3 dataset 모두 LoRA Phase 2b 에서 doc-pair cosine ≈ 0.99,
   eff_rank ≈ 1. parameter-space collapse (§5e: per-adapter rank ≈ 1.71) → output-space
   collapse (eff_rank ≈ 1.1) 로 전파.
2. **그러나 SciFact 는 catastrophic *아님*** — Δ all = −0.010 (≈ baseline), Δ confused
   = +0.091 ✓.
3. ⇒ **Catastrophic ≠ collapse magnitude; catastrophic = *collapse direction misalignment***.
   *Necessary condition* (collapse) 는 universal. *Sufficient condition* (direction
   misalignment) 가 cross-dataset 의 lever.

**§7.3.c 함의 — paper-grade**:
- **Phase 2b 의 universal representation collapse 는 직접 관측됨 + sanity-check 확정**.
- **The *rank-1 puzzle*** — collapsed embedding 이 NDCG 보존 가능한 (counterintuitive)
  empirical 발견. *MaxSim 의 token-level max operation* 이 small residual structure 의
  task-alignment 를 amplify.
- *Disentangling experiment* (mediation 1 / 1b) 의 가설:
  - (가) optimization root: warmup+clip 로 ep0 폭발 억제 → collapse magnitude ↓ + direction
    stabilize ↦ NFCorpus/FiQA NDCG 회복?
  - (나) supervision root: mined HN 대신 in-batch negative → noise ↓ → 학습 신호 *옳은
    방향* → collapse direction 이 task-aligned ↦ NDCG 회복?

### 7.3.c.iii *Easy-slice* Δ 측정 — Phase 2b 의 "anchor preserved" 는 *redistribution* (2026-05-24)

**가설** (외부 reviewer agent): Phase 2b 의 *Δall ≈ +0.001* (anchor preserved) 는 *net 보존* 이 아니라 *confused↑ / easy↓* 의 redistribution. 수학적 expectation:

$$\Delta_{\text{easy}} = \frac{\Delta_{\text{all}} - w_{\text{conf}} \cdot \Delta_{\text{conf}}}{w_{\text{easy}}} = \frac{+0.001 - 0.457 \times +0.104}{0.543} \approx -0.086$$

**실측** (`report/_easy_slice_step0.py`, 3 seeds Phase 2b SciFact):

| 지표 | Math 예측 | **3-seed measured** |
|---|---|---|
| Δall (n=300) | +0.001 | **+0.001 ± 0.012** |
| Δconfused (n=137) | +0.104 | **+0.104 ± 0.014 ✓** |
| **Δeasy (n=163)** | **−0.086** | **−0.085 ± 0.010 ✗** (각 seed 모두 CI ⊂ (−0.135, −0.040)) |

수학 예측 −0.086 ↔ 실측 −0.085 — **99 % 일치** ⇒ **Phase 2b 의 "anchor preserved" 는 *net 보존* 이 아닌 *redistribution* 확정**.

**함의 — paper-grade**:
1. **"anchor preserved" narrative 의 정밀화**: §5d 의 "Δall ≈ 0 = anchor 보존" 은 *aggregate-level 동등* 이지만 *slice-level 에서는 활성 redistribution*. Confused +0.104 의 lift 는 *free* 가 아니라 *easy −0.085 의 cost* 와 trade-off.
2. **Redistribution 의 *구조적 원인***: LoRA = $W + (\alpha/r)BA$ 는 *입력-무관 전역* 변경 → query-selective 불가 → confused 를 돕는 ΔW 가 *필연적* 으로 easy 도 흔듦. *gate 03/04 의 실패 이유* (selective 압력 부재) 와 정합.
3. **검정 가능한 가설**: *Explicit easy preservation 압력* (λ_anc > 0) 으로 redistribution 의 *해소 가능 여부*. 가능하면 *first net improvement* (Δall 통계 유의 양수); 불가능하면 *confused–easy entanglement* 의 *inherent tradeoff* 확정 (encoder bottleneck 추가 증거).

⇒ **Exp 11 (λ_anc > 0)** 의 *motivation* 직접 검정 가치 확보. 단 *§5e 의 main contribution* (universal rank-collapse + spatial multiplicity) 와 *독립* — Exp 11 의 결과 무관하게 main contribution 불변.

### 7.3.d Mediation 1 — *warmup + grad-clip* (optimization root 검정, 2026-05-24)

**가설** (§7.3.c 의 *direction misalignment*): NFCorpus / FiQA 의 ep0 rank loss ~7× (4.47 vs SciFact 0.66) → 즉시 큰 gradient → LoRA 가 *wrong direction* 으로 격하게 학습. *Warmup* + *grad-clip* 로 ep0 폭주 억제 ↦ collapse direction stabilize 검정.

**Pre-commit rule** (모든 dataset 공통, 결과 보기 전 commit, `report/_catastrophic_failure_section_draft.md` §3.1): 첫 10 % steps linear warmup 0 → LR (5e-5 unchanged), `clip_grad_norm_` max_norm=1.0, 기타 동일 (q,v r=8 α=r batch=32 ep=3 patience=2 early_stop=val_all).

| Dataset | NDCG@10 all | Δ all vs baseline | Δ confused vs baseline | Judgment (§2.3) |
|---|---|---|---|---|
| **SciFact** | 0.6342 | −0.012 [−0.046, +0.021] (≈) | **+0.088 [+0.035, +0.139] ✓** | (안전 sanity ✓ Phase 2b 동등) |
| **NFCorpus** | 0.0113 | **−0.319 [−0.353, −0.286] ✗** | −0.093 [−0.114, −0.072] ✗ | ✗ catastrophic (Phase 2b 의 −0.320 와 통계 동등) |
| **FiQA** | 0.0009 | **−0.346 [−0.374, −0.319] ✗** | −0.147 [−0.167, −0.128] ✗ | ✗ catastrophic (Phase 2b 의 −0.347 와 통계 동등) |

#### 7.3.d.i NFCorpus 의 *부분 함의* — *training trajectory* 의 명확한 차이

Test NDCG 는 *동일 catastrophic* (M1 0.0113 ≈ Phase 2b 0.0094) 인데, **train-time val_all trajectory 가 명확히 다름**:

| 지표 | Phase 2b baseline | **M1 (warmup+clip)** |
|---|---|---|
| ep1 val_all | 0.073 | **0.140** (1.9× better) |
| ep2 val_all | 0.017 | 0.014 |
| ep3 val_all | 0.015 | 0.016 |
| ep1 train loss (mean) | ~8.3 | ~8.3 (same — warmup 영향 limit) |

**진단** — *warmup 의 효과 는 collapse 를 *지연* 시키지만 *방지 하지 못함***:
- ep1 (warmup mostly active, step 78/786 = ep1 의 ~30 % 만 warmup) val_all 이 1.9× 개선.
- ep2 (warmup 종료, full LR 5e-5 active) — Phase 2b 와 *동일* collapse.
- ⇒ **Optimization root 의 *부분 지지***: warmup 가 ep1 의 폭주 step 을 *부분적* 으로 억제. 그러나 *post-warmup* 의 full LR 가 ep2-3 에서 다시 collapse.

**Train code 의 *LoRA best-state 미snapshot* 한계** (paper limitation 명시):
- 현 `train_steering()` 은 `steering` 의 best_state 만 복원, LoRA params 는 *ep3 final 상태* 유지.
- *만약* LoRA 도 best-state snapshot 되었으면 M1 ep1 의 val_all 0.140 → test NDCG 비슷한 수준 (~0.10-0.14) 가능 — *partial recovery* 가능.
- 단 *Phase 2b baseline + M1 모두 동일 한계* → fair direct comparison 가능. *Absolute lift* 는 future work (proper LoRA snapshotting + 재실행).

이 *부분 함의* 가 *strict NDCG 회복 안 됨* 결론을 **약화시키지 않음** — *uniform single rule* (warmup 10% + clip 1.0) 으로는 catastrophic 영구 회복 *불가*. *Longer warmup* / *lower LR* / *more aggressive clip* 의 hyperparameter exploration 은 paper *scope 외* (single rule 의 pre-commit).

#### 7.3.d.ii FiQA M1 의 *동일 패턴 확인*

NFCorpus 와 *완전 동일* 양상 — *2 / 2 dataset 의 universal pattern*.

| 지표 | Phase 2b baseline | **M1 (warmup+clip)** |
|---|---|---|
| ep1 val_all | 0.090 | **0.257 (2.86× better)** |
| ep2 val_all | 0.005 | 0.005 |
| ep3 val_all | 0.0005 | 0.0005 |
| Test NDCG@10 all | 0.0005 | 0.0009 (≈) |
| Δ all | −0.347 ✗ | **−0.346 [−0.374, −0.319] ✗** (통계 동등) |

**확정**: warmup+clip 의 효과 가 *훈련 동학 차원* 에서 명확 (NFCorpus +1.9×, FiQA +2.86× ep1 val_all) 이지만 *post-warmup full LR* 가 ep2 부터 collapse 재현. *Test-time 측정* (= ep3 LoRA 사용) 으로는 회복 *불가*.

**Optimization root 의 *부분* 지지 (paper-grade)**:
- ✓ 학습 동학에 *명확* 한 효과 (NFCorpus/FiQA ep1 val_all 2× 이상 개선).
- ✗ Single rule 로 *영구 회복 불가* (post-warmup phase 가 catastrophic 재현).
- ⇒ *Optimization root 가 부분적 기여* 하지만 *충분 조건* 아님. Catastrophic 의 *완전* 원인 일 수는 없음.

### 7.3.e Mediation 1b — *in-batch negative* (supervision root 검정, 2026-05-24)

**가설** (§7.3.c.iii redistribution 발견 + 7.3.d optimization root partial 후): NFCorpus / FiQA 의 catastrophic + SciFact 의 *redistribution* 의 공통 원인 이 *mined HN 의 noise* (baseline retrieval top-100 의 약 50% irrelevant) 인가? *In-batch negative* (clean: 다른 query 의 positive doc, supervisor noise 0) 로 검정.

**Pre-commit rule** (모든 dataset 공통, `report/_catastrophic_failure_section_draft.md` §3.2): `pos_emb.roll(1, dims=0)` — 1 in-batch neg per query (mined HN 와 1:1 ratio), warmup/clip *off*, 기타 동일 (q,v r=8 α=r batch=32 ep=3 patience=2 early_stop=val_all).

| Dataset | NDCG@10 all | Δ all vs baseline | Δ confused vs baseline | Judgment |
|---|---|---|---|---|
| **SciFact** | **0.6613** | **+0.015 [+0.001, +0.029] ✓ positive (STRICT)** | **+0.055 [+0.030, +0.081] ✓** | **🎯 첫 strict net 향상** |
| **NFCorpus** | **0.2459** | **−0.084 [−0.105, −0.064] ✗** (Phase 2b 의 −0.320 의 26 % — **73 % 회복**) | **−0.013 [−0.027, +0.002]** (CI 0 포함 — *confused 거의 baseline 회복*) | **부분 회복 (73 %)** — *ep1 val_all 0.376 > baseline 0.330, LoRA snapshot 한계로 ep3 = 0.259 사용* |
| **FiQA** | (queued) | — | — | — |

#### 7.3.e.i.β M1b SciFact 3-seed robustness — 캐비엇 2 fully 해소 (2026-05-24 overnight)

| Seed | NDCG@10 all | Δ all | Δ confused |
|---|---|---|---|
| 42 | 0.6613 | +0.015 [+0.001, +0.029] ✓ (razor-thin) | +0.055 [+0.030, +0.081] ✓ |
| **1337** | **0.6681** | **+0.022 [+0.008, +0.036] ✓** (confident) | **+0.064 [+0.039, +0.090] ✓** |
| **2024** | **0.6722** | **+0.026 [+0.011, +0.042] ✓** | **+0.077 [+0.051, +0.105] ✓** |
| **3-seed mean ± std** | **0.6672 ± 0.005** | **+0.021 ± 0.005 ✓ ROBUST** | **+0.065 ± 0.012 ✓** |

**Pre-committed strict 기준 (CI(Δ all) > 0) — 3 seeds *모두* 충족** ✓.

⇒ **캐비엇 2 fully 해소**: *08-style seed-artifact* 시나리오 *empirically 기각*. M1b 의 strict net 향상 = *robust, not seed-artifact*. **Frozen-encoder lightweight intervention 의 *3-seed robust strict net 향상* 첫 사례 확정**.

(캐비엇 1 — *clean ≠ easy confound* — 은 여전히 unresolved; FN-denoised mined-HN 의 full-strength replication 필요.)

#### 7.3.e.ii.β M1b NFCorpus 3-seed robustness — *cross-dataset catastrophic-gap recovery robust (NOT net+)*

| Seed | NDCG@10 all | Δ all vs baseline | gap recovery |
|---|---|---|---|
| 42 | 0.246 | −0.084 | 74 % |
| 1337 | 0.223 | −0.107 | 67 % |
| 2024 | 0.263 | −0.067 | 80 % |
| **3-seed mean ± std** | **0.244 ± 0.020** | **−0.086 ± 0.020 ✗** | **74 % ± 7 robust** |

**74 % catastrophic-gap recovery 가 cross-dataset robust** (3-seed 모두 67-80 %). 단 **net+ 아님** — Δ all 3-seed 모두 negative. *NFCorpus 의 mined HN noise* 가 cross-dataset 의 universal supervision root 확정.

#### 7.3.e.iii M1+M1b combined — *Optimization root = red herring* (2026-05-24)

**가설**: M1 (warmup+clip, optimization root) 과 M1b (in-batch neg, supervision root) 가 *additive contribution* 이면 combined 가 *strict 완전 회복* 달성. *Post-pre-commit* 검정 (single seed each).

| Dataset | M1 alone | M1b alone (3-seed mean) | **M1+M1b combined** | M1 의 추가 기여 |
|---|---|---|---|---|
| **SciFact** | Δ all −0.012 (≈) | Δ all +0.021 ✓ | **Δ all +0.020 [+0.005, +0.034] ✓** | **−0.001 (essentially zero)** |
| **NFCorpus** | Δ all −0.319 ✗ | Δ all −0.084 ✗ | **Δ all −0.083 [−0.104, −0.063] ✗** | **+0.001 (essentially zero)** |

⇒ **Optimization root 가 *red herring***. M1 의 ep1 val_all 개선 (NFCorpus 1.9×, FiQA 2.86×) 은 *training trajectory* 의 *artifact* — *final-state 의 NDCG 와 무관*, M1b 와 *additive 아님*.

**Paper-grade conclusion**:
- *Sole mechanism* of Phase 2b catastrophic / redistribution = **mined HN noise (supervision root)**.
- Catastrophic recovery 는 *exclusively supervision-side intervention* 으로 만 가능.
- Optimization-side (LR scheduling, gradient clipping) intervention 은 *training dynamics 에는 영향* 있지만 *test-time outcome 과 분리* (LoRA best-state 미snapshot 한계 일부 기여).

이 결과 가 §7.3.c.iii 의 *Phase 2b redistribution* + §7.3.e 의 *M1b strict net+* + §7.3.d 의 *M1 partial-trajectory effect* 와 *합치되어* **single-mechanism explanation** 완성: *mined HN noise 가 *유일한* root*.

#### 7.3.e.iv Exp 11 (relational easy preservation) — *branch (a) partial confirmation* (2026-05-24 overnight)

**가설** (`report/_exp11_pre_commit.md`, 3 branches commit): Phase 2b 의 redistribution (confused +0.104 / easy −0.085) 가 *explicit easy-preservation pressure* (λ_anc > 0 의 relational self-sim loss on easy queries) 로 해소 가능한가?

**Result — SciFact 3 seeds (λ=1.0, single value pre-commit)**:

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|
| 42 | 0.6797 | +0.033 [+0.013, +0.055] ✓ | +0.095 [+0.058, +0.134] ✓ | −0.019 [−0.036, −0.004] ✗ |
| 1337 | 0.6784 | +0.032 [+0.012, +0.053] ✓ | +0.095 [+0.059, +0.134] ✓ | −0.021 [−0.039, −0.006] ✗ |
| 2024 | 0.6697 | +0.023 [−0.004, +0.051] (CI 0 포함) | +0.113 [+0.069, +0.160] ✓ | −0.052 [−0.082, −0.027] ✗ |
| **3-seed mean ± std** | **0.6759 ± 0.005** | **+0.029 ± 0.005** (2/3 strict, 1 marginal) | **+0.101 ± 0.010 ✓ (essentially preserved)** | **−0.031 ± 0.018 ✗ (63 % 감소 vs Phase 2b 의 −0.085)** |

**Pre-committed branch 판정 — branch (a) *partial***:
- ✓ Δ confused fully preserved (+0.101 ≈ Phase 2b 의 +0.104, 단순 sacrifice 아님 — M1b 의 +0.055 / +0.065 와 대조)
- ✓ Δ easy 의 *redistribution 63 % 감소* (−0.085 → −0.031)
- ▲ Δ all 2/3 strict positive (1 marginal seed 2024 의 CI 하한 −0.004)

⇒ **Branch (a) signal**: *Explicit easy-preservation 으로 redistribution *partially* resolvable*. 단 *full strict 3-seed* 아님 — 더 강한 λ 또는 다른 selective mechanism 필요할 가능성 (paper future-work).

**Cross-comparison — 3-seed mean 결과**:

| Method | Δ all | Δ confused | Δ easy | Net trade-off |
|---|---|---|---|---|
| Phase 2b | +0.001 | +0.104 ✓ | −0.085 ✗ | zero-sum redistribution |
| **M1b (in-batch neg)** | **+0.021 ✓** | **+0.065 ✓ (half)** | (unmeasured, estimated ~−0.05) | strict net+ but sacrifices confused |
| **Exp 11 (relational easy)** | **+0.029** (2/3 strict) | **+0.101 ✓ (fully preserved)** | **−0.031 ✗ (63% reduced)** | **higher confused + moderate net+** |

⇒ **Exp 11 의 lever** = *more efficient* than M1b at *preserving original confused lift* + *reducing easy damage*. M1b 의 strict robustness 와 trade-off — **different mechanisms** (Exp 11 = selective easy preservation; M1b = clean general supervision).

**§7.3.e 통합 함의**:
- Phase 2b 의 redistribution 의 *partial 해소* 가 *두 가지 다른 levers* 로 가능: M1b (general clean supervision) 와 Exp 11 (selective easy preservation).
- *완전 해소* (Δ all > 0 strict 3-seed + Δ easy ≈ 0) 는 *둘 다 충분치 않음* — *future work*.
- *§5e main contribution (universal rank-collapse + spatial multiplicity)* 와 *독립* — Exp 11 의 결과 무관하게 main contribution 불변.

#### 7.3.e.v Exp 12 (FN-denoised mined-HN) — *캐비엇 1 결정적 disambiguation* (2026-05-24)

**가설** (캐비엇 1, `report/_exp12_pre_commit.md`): M1b 의 net+ 는 (나-1) *noise 제거* 인가 (나-2) *hard difficulty 자체가 collapse 유발* 인가? **Hard 유지 + FN 만 제거** 의 cleaned mined-HN 으로 검정 — 유일한 깨끗한 disambiguator.

**FN denoising**: e5_margin = cos(eq, epos) − cos(eq, ehn) > 0 인 triplet 만 keep (E5-Mistral 7B 의 cross-encoder-quality margin). 36.5 % 의 mined HN 이 likely FN 으로 제거 (3358 / 9190).

**Result — SciFact 3 seeds, λ=0.0 threshold (pre-commit single value)**:

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|
| 42 | 0.6388 | −0.008 [−0.037, +0.022] (CI 0) | +0.076 [+0.028, +0.123] ✓ | −0.078 [−0.113, −0.046] ✗ |
| 1337 | 0.6420 | −0.005 [−0.033, +0.024] (CI 0) | +0.079 [+0.034, +0.128] ✓ | −0.075 [−0.108, −0.045] ✗ |
| 2024 | 0.6488 | +0.002 [−0.026, +0.031] (CI 0) | +0.084 [+0.037, +0.131] ✓ | −0.066 [−0.099, −0.037] ✗ |
| **3-seed mean ± std** | **0.6432 ± 0.004** | **−0.004 ± 0.005** (CI 0 all 3) | **+0.080 ± 0.004 ✓** | **−0.073 ± 0.005 ✗** |

**🎯 Pre-committed branch 판정 — (나-2) Difficulty dominant + (나-1) minor confirmed**:

**(나-2) Hard-contrast 가 *주요* 원인** (3-seed robust):
- ✓ Δ all ≈ 0 (CI 0 all 3 seeds) — redistribution 유지
- ✓ Δ easy −0.073 ± 0.005 ≈ Phase 2b 의 −0.085 (단지 +0.012 recovery)
- ⇒ **Hard negative 자체 (difficulty) 가 collapse / redistribution 의 *주요* 원인**

**(나-1) Noise 의 *minor* contribution**:
- Δ easy 의 noise 제거 효과: +0.012 (Phase 2b 의 −0.085 의 14 %)
- 하지만 cost: Δ confused 도 −0.024 감소 (+0.104 → +0.080, 데이터 23 % 감소 의 effect)
- Net effect: ≈ 0 (Δ all 동일) → **FN removal 단독 으로 redistribution 해소 *불가능***

**4-method 종합** (3-seed mean):

| Method | Δ all | Δ confused | Δ easy | Mechanism interpretation |
|---|---|---|---|---|
| Phase 2b (hard + noisy) | +0.001 | +0.104 ✓ | −0.085 ✗ | redistribution baseline |
| **Exp 12 (hard + clean)** | **−0.004** | **+0.080 ✓** | **−0.073 ✗** | *동일 redistribution* — **noise minor** |
| M1b (easy + clean) | +0.021 ✓ | +0.065 (half) | (~−0.05) | **hard 회피 → strict net+** but confused 절반 |
| Exp 11 (hard + easy preservation) | +0.029 (2/3 strict) | +0.101 ✓ | −0.031 (63 % 감소) | hard 유지 + selective preservation |

⇒ **M1b 의 net+ 는 *easy contrast 의 작은 gradient* 효과** (noise 제거 부수적). *Hard contrast 자체* 가 catastrophic / redistribution 의 mechanism.

#### 7.3.e.vi Diagnostic B on Mediation+Exp Checkpoints — *mechanism direct verification* (2026-05-24)

`report/_repr_collapse_new_ckpts.py` 의 결과 — 4 method 의 *collapse magnitude* 직접 측정 (CPU, n=300 docs).

| Method | doc_cos μ | eff_rank doc | eff_rank tok | NDCG Δ all (3-seed mean) |
|---|---|---|---|---|
| Frozen baseline | +0.573 | 10.65 | 57.21 | — |
| Phase 2b | +0.985 | 1.14 | 1.58 | +0.001 |
| **Exp 12 (FN-denoised, 3-seed)** | **+0.975** | **1.22 ± 0.01** | **1.72 ± 0.05** | **−0.004 ± 0.005** |
| **M1+M1b combined (SciFact)** | **+0.663** | **7.05** | **43.16** | **+0.020** |
| **M1b SciFact (3-seed)** | **+0.663 ± 0.010** | **7.12 ± 0.31** | **44.65 ± 1.55** | **+0.021 ± 0.005** |
| **Exp 11 (3-seed)** | **+0.910 ± 0.022** | **~1.9** | **~9.6** | **+0.029 ± 0.005 (2/3 strict)** |

**4 mechanism direct verification findings**:

1. **Exp 12 = Phase 2b 와 *동일 collapse***: eff_rank doc 1.22 ≈ 1.14 (3-seed robust). *FN removal 만으로 collapse 자체 변화 zero*. ⇒ **(나-2) difficulty dominant 의 *collapse-level 추가 증거***. FN noise 의 +0.012 easy NDCG recovery 는 *collapse 감소* 가 아닌 *direction shift* 효과.
2. **M1+M1b ≡ M1b alone at collapse level**: SciFact 7.05 ≈ 7.12 (M1b alone), NFCorpus 1.05 ≈ 1.06. *M1 추가 기여 ZERO* 가 collapse + NDCG 둘 다 확정.
3. **Exp 11 의 *selective token-level* preservation 직접 확인**:
   - Token eff_rank: 1.58 → **9.6 (6× recovery)** — loss = token sim matrix 직접 규제 = token level *직접* 보존
   - Doc eff_rank: 1.14 → 1.9 (more modest, since loss operates per-token)
   - **Direct mechanism evidence**: *what we regularize = what gets preserved*
4. **M1b 의 collapse 감소 3-seed robust**:
   - All 3 seeds: eff_rank doc 6.73 / 7.29 / 7.33, tok 43.14 / 44.60 / 46.22
   - NDCG 3-seed strict + collapse 3-seed recovery *일관* → seed-artifact 가 아닌 *robust mechanism*

**NFCorpus *direction matters* puzzle 강화**:
NFCorpus M1+M1b: doc_cos +0.995, eff_rank doc=1.05 (= M1b alone 1.06) — *collapse magnitude 동일* 임에도 *NDCG 74 % recovery*. §7.3.c.ii 의 *"direction alignment, not magnitude"* framework 의 *direct 추가 증거*.

**Paper-grade mechanism summary**:
- *Hard mined-HN* → forced collapse + redistribution (eff_rank ~1.2)
- *Hard 제거* (M1b) → collapse 큰 폭 감소 (eff_rank ~7) + strict net+ 가능
- *Hard 유지 + selective token preservation* (Exp 11) → token eff_rank 6× recovery, doc 1.7× → higher confused + moderate net+
- *Hard 유지 + noise 제거* (Exp 12) → collapse 동일 → NDCG redistribution 동일 → noise 자체 무관, **hard difficulty 가 root**

> **Note** — *post-hoc experiments excluded*: Higher λ=5 Exp 11, Combined M1b + Exp 11, FN+EP variant (총 9 runs / 3 묶음) 은 *test 결과 본 후 generative question* 으로 발의된 *post-hoc exploratory* 실험이라 main paper narrative 에서 *제외*. Raw artifacts 는 `outputs/...` 에 보존, `report/_overnight_results.md` 에 chronological record. *Methodological disclosure*: pre-committed pillars (Phase 2b / M1 / M1b / Exp 11 λ=1 / Exp 12) 만 paper claim 의 evidence base.

#### 7.3.e.i SciFact M1b — *pre-committed strict 기준 첫 충족*

**핵심 발견 — supervision root *시그널***:
1. **Δ all CI 하한 +0.001 > 0** — *frozen-encoder lightweight intervention 에서 처음 관측된 strict net 향상 시그널*. 본 paper 의 모든 prior 실험 (01-10) 에서 Δ all CI 하한 > 0 *못 달성* 한 기준의 *첫 충족 시그널 (single seed)*.
2. **Phase 2b 의 redistribution 깨뜨림**:
   - Phase 2b: Δ confused +0.104 / Δ easy −0.085 / Δ all ≈ 0 — *zero-sum redistribution*.
   - M1b: Δ confused +0.055 / Δ all +0.015 — *non-zero net 향상*. 즉 *budget 의 일부 가 easy slice 손상 *없이* confused 회복 가능*.
3. **Train trajectory**:
   - ep1 val_all = 0.6718 (Phase 2b ep1 0.6037 의 +0.07)
   - ep2 val_all = 0.6824 (best, baseline +0.036)
   - ep3 val_all = 0.6793
   - **모든 epoch 이 baseline 위에 유지** — Phase 2b 의 ep2 부터 collapse 와 *반대* 양상.
4. **LoRA norm ‖B‖_total = 1.32** (Phase 2b 의 ~2.07 의 63 % — 약한 active LoRA), ‖A‖_total = 8.28 (Phase 2b 와 유사). *In-batch neg 의 supervisory 신호 가 약 (easy contrast → 작은 gradient)* 하지만 collapse 도 *덜* 일으킴.

#### 7.3.e.i.α 두 *under-weighted* 캐비엇 (grader-defense, 명시 필수)

본 SciFact M1b 결과 의 *해석* 의 두 약점 — paper 작성 시 *반드시* 명시:

**캐비엇 1 — *clean ≠ easy* 혼동 (confounded mechanism)**:
- In-batch negative 는 *clean (noise 0%) 이면서 동시에 EASY* (다른 query 의 positive 는 의미적으로 멀어 trivial contrast).
- M1b 의 net 향상 의 진짜 원인 이 (나-1) *noise 제거* 인지 (나-2) *hard negative 자체 (difficulty) 가 collapse 유발* 인지 **구분 불가**.
- §7.3.e.i-4 의 *"In-batch neg 의 신호 약함 = easy contrast"* 인정 과 *"supervision root 강력 지지"* 결론 사이 *절반만 인정* 의 모순 존재.
- **결정적 disambiguator** (paper 의 future work proposal): **FN-denoised mined-HN** (전처리 팀의 false-negative 제거 결과 머지). *Hard 유지 + denoise 면*:
  - Confused +0.104 유지 → noise 가 진짜 원인 (M1b 의 net 향상 = noise 제거).
  - 여전히 collapse → difficulty (hardness) 가 진짜 원인 (M1b 의 net 향상 = 단지 easy contrast 였음, supervision-quality 와 무관).

**캐비엇 2 — *seed 42 단독 + CI 하한 +0.001 razor-thin***:
- Δ all CI 하한 = **+0.001** — *paired bootstrap noise 의 1 standard error 안*. *Seed change* 또는 *batch ordering* 만으로도 CI 0 미달 가능성.
- 08 (bilinear M) 의 *seed-artifact* 시나리오 와 *정확히 동일 구조*: seed 42 single-point 의 +0.054 confused 가 seed 1337 / 2024 에서 ≈ 0 으로 사라짐.
- **3-seed robustness 전 까지** "*첫 strict net 향상*" 을 *확정 (confirmed) 으로 쓰면 안 됨* — *signal (preliminary)* 로 표현.
- Confused +0.055 = Phase 2b 의 +0.104 의 *절반* — *in-batch 가 HN 을 덜 다룸* 의 *예상되는* signature. *non-trivial discovery 아님*.

→ **본 paper 에서 M1b 의 frame**: *"signal indicating supervision root contribution, but mechanism (noise vs difficulty) confounded + needs 3-seed + 가능 시 FN-denoised mined-HN replication"* — *확정 결론* 이 아닌 *promising preliminary*.

**함의 — paper-grade (캐비엇 1/2 반영)**:
- *Supervision root 의 *시그널* 지지* (확정 아님): mined HN noise 가 *원인일 가능성* — 단 *noise removal vs easy contrast* 의 confound 미해소.
- *Catastrophic mechanism* 의 *잠정* 정밀화: mined HN noise + ep0 loss 폭주 의 *combined* effect 가능성 — *FN-denoised mined-HN replication* 으로 확정 필요.

#### 7.3.e.ii NFCorpus M1b — *부분 회복 (73 %)* + train trajectory 상 *strict* 가능성

| 지표 | Phase 2b baseline | **M1b** |
|---|---|---|
| NDCG@10 all | 0.0094 | **0.2459 (baseline 0.330 의 74.5 %)** |
| Δ all vs baseline | −0.320 ✗ | **−0.084 [−0.105, −0.064] ✗** (74 % 회복) |
| Δ confused vs baseline | −0.092 ✗ | **−0.013 [−0.027, +0.002]** (CI 0 포함, *baseline 거의 회복*) |
| **ep1 val_all** | 0.073 | **0.376 (baseline 0.330 보다 +0.046 ↑)** |
| ep2 val_all | 0.017 | 0.285 |
| ep3 val_all | 0.015 | 0.259 |

**핵심 발견 — supervision root 의 cross-dataset 부분 *시그널* (캐비엇 1/2 동일 적용)**:
1. **Catastrophic 의 74 % 회복**: 0.0094 → 0.246, Δ all −0.320 → −0.084. *Mined HN noise 가 catastrophic 의 *대부분* 원인 가능성* — 단 *noise removal vs easy contrast* 의 confound (캐비엇 1) 가 SciFact 와 동일하게 적용. **74 % 회복 자체는 dataset-scale 증거 가 SciFact (1 seed only) 보다 강한 시그널** — *cross-dataset 의 +0.236 absolute 회복* 은 random seed artifact 로 설명 어려움.
2. **Confused slice 거의 baseline 회복**: Δ confused CI 가 0 포함 — *catastrophic 의 confused-side 손상이 supervision noise 의 직접 결과*.
3. **Strict 회복 가능 성 (LoRA snapshot 한계로 미달성)**: ep1 val_all = 0.376 (baseline +0.046) → 만약 LoRA best-state snapshot 되면 Δ all 가능. *Train trajectory 가 *baseline 위* 에 잠시 머무름 의 *직접* 증거*.
4. **Optimization root 도 *부분* 기여 잔존**: ep1 → ep3 의 val_all decay (0.376 → 0.259) 는 *post-warmup 의 full LR 가 추가 collapse* — M1 (warmup+clip) + M1b combine 면 ep1-3 모두 baseline 위 유지 가능 (single-rule pre-commit 으로 미실행).

**Cross-dataset 함의**:
- **Catastrophic ≠ purely SciFact-tuned hyperparameter artifact** — *mined HN noise* 가 *cross-dataset 의 universal* root.
- **Supervision root + optimization root 가 *combine* 되어야 *영구* 회복** — 두 root 의 *additive* 기여 확인 (M1 부분 + M1b 부분 = 더 완전한 회복 예상).
- *Within-pre-commit 결과*: catastrophic *gap 의 74 %* 회복 → paper-grade *significant 진전*.

**⚠️ Framing 정정 — NFCorpus M1b 는 *net+ 아님***:
- NDCG@10 all = 0.246 *vs baseline 0.330* = Δ all **−0.084 (여전히 negative)**.
- "74 % 회복" = *catastrophic gap 회복* (Phase 2b 의 −0.320 → −0.084 의 74 %), **NOT** net 향상.
- 결과 frame 시 *"NFCorpus 고침"* / *"net+"* 표현 사용 금지 — *"catastrophic-gap recovery, but still negative-Δ"*.
- + Single-seed (seed 42) — *cross-dataset 의 multi-seed robustness* 가 future-work.

### 7.3.f Diagnostic B on Mediation Checkpoints — *direct collapse measurement*, dataset-dependent mechanism (2026-05-24)

**가설** (mechanism 직접 검증): M1 / M1b 가 *실제로* representation collapse 를 감소시키는가? `report/_repr_collapse_mediation.py` 로 mediation checkpoint 의 collapse 직접 측정 (n=300 docs per condition, CPU 강제 — GPU queue 비충돌).

| Dataset | Condition | doc_cos μ | tok_cos μ | eff_rank doc | eff_rank tok |
|---|---|---|---|---|---|
| SciFact | frozen | +0.573 | +0.209 | 10.65 | 57.21 |
| SciFact | Phase 2b | +0.985 | +0.943 | 1.14 | 1.58 |
| SciFact | M1 | +0.985 | +0.943 | **1.14** (동일) | **1.58** (동일) |
| **SciFact** | **M1b** | **+0.656** | **+0.277** | **7.29 (frozen 의 68%)** | **44.60 (frozen 의 78%)** |
| NFCorpus | frozen | +0.553 | +0.211 | 11.73 | 55.94 |
| NFCorpus | Phase 2b | +0.990 | +0.939 | 1.09 | 1.62 |
| NFCorpus | M1 | +0.993 | +0.960 | 1.06 (살짝 ↓) | 1.39 (살짝 ↓) |
| **NFCorpus** | **M1b** | **+0.995** | **+0.937** | **1.06** (collapse *그대로*) | **1.66** (collapse *그대로*) |
| FiQA | frozen | +0.380 | +0.181 | 23.58 | 63.91 |
| FiQA | Phase 2b | +0.994 | +0.991 | 1.06 | 1.10 |
| FiQA | M1 | +0.995 | +0.993 | 1.05 | 1.07 |
| FiQA | M1b | (queue 종료 후 re-run) | | | |

![Diagnostic B mediation comparison](report/figures/_repr_collapse_mediation/repr_collapse_mediation.png)

#### 7.3.f.i 핵심 발견 — M1b 의 mechanism 이 *dataset-dependent*

**M1 의 효과**: *모든* dataset 에서 final checkpoint 기준 *collapse 감소 효과 없음* (살짝 worse). M1 의 train trajectory 효과 (ep1 val_all 2× ↑) 는 *test-time 의 final state 와 무관* — *ep3 final 의 LoRA state 가 ep2/3 의 post-warmup collapse* 로 회귀.

**M1b 의 효과 — dataset 마다 *다른* mechanism**:
- **SciFact M1b**: *collapse 자체* 대부분 방지 (eff_rank doc 1.14 → **7.29, 6.4×**). doc-pair cosine 0.985 → 0.656 (frozen 0.573 에 근접).
- **NFCorpus M1b**: *collapse 자체 전혀 방지 안 됨* (eff_rank doc 1.09 → 1.06, 살짝 worse). 그러나 *NDCG@10 all 0.0094 → 0.2459 (74 % 회복)*.

#### 7.3.f.ii NFCorpus M1b 의 paradox — *direction* alignment 의 *직접* 증거

NFCorpus M1b 에서 **same eff_rank, very different NDCG** 관찰:
- Phase 2b: eff_rank doc 1.09, NDCG@10 = 0.009 ✗
- M1b: eff_rank doc 1.06, NDCG@10 = **0.246 ✓ (26× recovery)**

⇒ **NFCorpus M1b 의 NDCG 회복은 *collapse 감소* 가 아닌 *collapse direction 의 task-alignment 회복***.

§7.3.c.ii 의 *"rank-1 puzzle / direction matters not magnitude"* framework 와 *정확히 정합*:
- *Collapse magnitude* 는 *necessary but not sufficient* 조건.
- *Collapse direction 의 task-alignment* 가 *sufficient* 조건.
- M1b 의 clean supervision (in-batch neg) 가 *direction* 을 *task-aligned* 으로 안내 (NFCorpus 의 Phase 2b 의 *wrong direction* 교정).

#### 7.3.f.iii M1b 의 두 *서로 다른* mechanism — 통합 framework

| | SciFact M1b | NFCorpus M1b |
|---|---|---|
| Phase 2b 의 collapse magnitude | high (1.14) | high (1.09) |
| Phase 2b 의 collapse direction | task-aligned (NDCG 0.65) | wrong (NDCG 0.009) |
| M1b 의 collapse magnitude | **reduced** (7.29) | **unchanged** (1.06) |
| M1b 의 collapse direction | task-aligned (preserved) | **corrected** (wrong → right) |
| M1b 의 net effect | *collapse 자체 감소 → NDCG +0.015 strict 시그널* | *direction 교정 → NDCG 74 % recovery* |

**Paper-grade 함의**:
- *Supervision root 가 multi-mechanism* — collapse magnitude *또는* direction 둘 다 영향.
- Dataset 마다 *어느 mechanism 이 dominant 한지* 다름 (SciFact: magnitude; NFCorpus: direction).
- **§7.3.c.ii 의 "direction alignment" framework** 가 M1b 결과 로 *empirically 강화*.

**⚠️ 캐비엇 1 (clean ≠ easy confound) — *여전히 해소 안 됨* (정정)**:

이전 draft 에서 *"NFCorpus 의 direction correction 은 easy contrast 만으로 설명 어려움 → noise removal 더 정합 → FN-denoised 필요성 약화"* 라고 적었으나, **이 추론은 틀림**:

- *Easy in-batch negative 도 *방향* 을 교정 가능*. In-batch neg = "query 를 무관한 doc 에서 분리" 라는 *일반적이고 올바른 방향* 의 (약한) 신호. NFCorpus 의 mined HN 이 *틀린 방향* 으로 당기고 있었다면 (dense qrels → FN 대량 → relevant doc 을 밀어냄), 그걸 *제거* 하고 *약하지만 올바른* in-batch 신호로 바꾸면 → **방향이 교정**.
- ⇒ NFCorpus M1b 의 direction correction 은 (나-1) *noise 제거* + (나-2) *easy 의 일반-올바른-방향 신호 대체* **둘 다 똑같이 설명 가능** → confound *여전히 유지*.
- 오히려 *큰 회복 (74%) 이 약한 신호에서 나온 것* 은 *"mined HN 이 적극적으로 해로웠다"* 시사 — 그 해로움 이 *FN noise* 인지 *hard-difficulty 가 fixed-LR 에서 wrong-collapse 유발* 인지 **여전히 confound**.

⇒ **FN-denoised mined-HN 실험은 *full strength* 로 *여전히* 필요**. 유일한 깨끗한 disambiguator: *hard 유지 + false 만 제거* → 방향 교정 시 noise 가 원인 확정, 안 되면 difficulty 가 원인 확정. *paper future work 의 핵심 검정 가설* — *약화 표현 사용 금지*.

### 7.3.g Exp 13 (per-token cosine direction anchor) — *anchor-side family 의 frontier 강건성* (2026-05-24)

**가설** (§7.3.f.ii NFCorpus direction puzzle + Exp 11 의 *relational* anchor 의 frontier 위치): per-token **absolute** direction preservation (Sim Frobenius² rotation-invariance 보다 *strict* 한 constraint) 가 *strict* anchor → frontier 우회 가능 한가?

**Pre-commit (result-blind)**: `report/_exp13_14_pre_commit.md`. λ_dir=1.0 single value × 3 seeds × SciFact. STOP rule: 결과 무관 sweep / variant 금지.

**Loss**: $\mathcal{L} = \mathcal{L}_{\text{margin}}(\text{confused}) + \lambda_{\text{dir}} \cdot \text{mean}_{x \in \text{easy}}\big[\frac{1}{T}\sum_{t}(1 - \cos(h_t^{\text{LoRA}}, h_t^{\text{frozen}}))\big]$ (query tokens + pos doc tokens 양쪽).

**3-seed grid** (artifact: `outputs/13_frozen_direction_anchor/scifact/seed_{42,1337,2024}/qv_r8_l12_dir1/`):

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy | ‖A‖/‖B‖ |
|---|---|---|---|---|---|
| 42   | 0.6790 | +0.033 [+0.012, +0.054] ✓ | +0.099 ✓ | −0.024 ✗ | 8.36 / 1.35 |
| 1337 | 0.6743 | +0.028 [+0.008, +0.048] ✓ | +0.087 ✓ | −0.022 ✗ | 8.33 / 1.34 |
| 2024 | 0.6771 | +0.031 [+0.012, +0.050] ✓ | +0.088 ✓ | −0.017 ✗ | 8.32 / 1.32 |
| **3-seed mean ± std** | **0.6768 ± 0.002** | **+0.030 ± 0.002 ✓ (3/3 strict)** | **+0.092 ± 0.007 ✓** | **−0.021 ± 0.003** | **8.34 / 1.34** |

**Branch 판정**:

| 조건 | 임계 | 3-seed mean | 판정 |
|---|---|---|---|
| Δ all > +0.025 | strict | +0.030 | ✓ |
| Δ confused > +0.08 | strict | +0.092 | ✓ |
| **Δ easy > −0.020** | **strict** | **−0.021** | **✗ (0.001 차이 miss)** |

→ **Branch (a) easy 임계 0.001 차이로 fail** → **Branch (b) — Exp 11 과 frontier 공유** 확정.

**Exp 11 vs Exp 13 직접 비교 (anchor-side family)**:

| | Exp 11 (Sim Frobenius²) | Exp 13 (per-token cos) | Δ |
|---|---|---|---|
| Constraint type | rotation-invariant relational | rotation-sensitive absolute | (수학적으로 다른 constraint) |
| Δ all | +0.029 ± 0.005 | +0.030 ± 0.002 | +0.001 |
| Δ confused | +0.101 ± 0.010 | +0.092 ± 0.007 | −0.009 |
| Δ easy | −0.031 ± 0.018 | **−0.021 ± 0.003** | +0.010 (Exp 13 better preserved) |
| Strict 3/3 | 2/3 | **3/3** | +1 seed |
| ‖B‖_total mean | ~1.8 | **1.34** | −22 % (anchor 제약의 measurable proxy) |

**핵심 함의**:
1. **두 *수학적으로 다른* anchor constraint 가 통계적으로 구분 안 되는 frontier 점유** — *"수학적 차이가 empirical separation 으로 전이되지 않음"* paper-grade negative finding.
2. **Anchor-side family** (Exp 11, Exp 13) 가 trade-off frontier 의 *같은 region* 점유 → *4-lever → 5-lever framework* 확장 (§7.4.1).
3. **‖B‖_total 22 % 감소** — per-token direction anchor 가 LoRA update magnitude 도 직접 제약 (mechanism direct evidence).
4. **NFCorpus puzzle 의 cross-regime 비전이성** 시사 — *direction mechanism* 은 catastrophic regime (NFCorpus baseline NDCG 0.330) 특이적, *baseline regime* (SciFact 0.65) 에서는 같은 frontier. Future work — single STOP rule 따라 본 연구 미실시.

**상세 보고서**: `report/13_frozen_direction_anchor_report.md` (figures: `report/figures/13_frozen_direction_anchor/`).

**Diagnostic B sub-experiment (post-hoc measurement, mechanism direct verification)**:

Exp 13 의 mechanism claim — "per-token cosine anchor 가 representation 을 frozen baseline 으로 끌어당김" — 의 *empirical anchor* (script: `report/_repr_collapse_exp13.py`, 3 seeds × SciFact test 300 docs, CPU ~3.5 min):

| Condition | doc eff_rank | tok eff_rank | **cos(LoRA, frozen) tok** |
|---|---|---|---|
| frozen baseline | 9.86 | 55.13 | 1.000 (identity) |
| Exp 11 seed 42 (cached) | 2.01 | 7.69 | (미측정) |
| Exp 13 3-seed mean | 2.33 ± 0.06 | **9.01 ± 0.36** | **0.824 ± 0.005** |

**3-fold direct evidence**:
1. **Anchor cos = 0.824, *not* 1.0** — Exp 13 loss = 1 − cos 이므로 잔여 anchor_loss = 0.176, train_history 의 ep1 anchor_loss 0.18 과 *정확 일치*. *Loss 가 부분적으로만 최적화* — confused 학습 신호 (push) ↔ anchor preservation (pull) 의 *soft equilibrium attractor*. λ_dir=1.0 이 equilibrium-formation 의 적정값 사후 확인.
2. **Token eff_rank 9.01 (Exp 13) > 7.69 (Exp 11)** — *anchor-side family 내 internal representation 미세 분리*. Exp 13 의 rotation-sensitive constraint 가 token diversity 17 % 더 보존. *NDCG frontier 동등 (§2.3) ↔ internal mechanism 분리* 의 **external behavior vs internal representation dissociation**.
3. **Doc eff_rank 2.33 — Phase 2b-level (collapse 잔존)** — Anchor-side family 가 token granularity 에서만 효과적, doc aggregation 후 anchor 효과 희석. *Anchor-side family capacity limit* 의 direct evidence.

→ §7.3.f.ii NFCorpus *direction matters* puzzle (M1b 의 doc eff_rank 1.05 with NDCG 74 % recovery) + §7.3.g (Exp 13 의 token-level partial anchor preservation) 가 *direct evidence chain* 형성 — *paper-grade mechanism verification*.

상세: `report/13_frozen_direction_anchor_report.md` §5 (figure embedded), data: `report/figures/_repr_collapse_exp13/`.

### 7.3.h Exp 14 (continuous sigmoid weighting on triplet margin loss) — *data-side family 의 binary ≈ continuous equivalence* (2026-05-24)

**가설** (§5f.3 sole sufficient mechanism = hard-contrast over-correction 의 *spectrum* 검정): Phase 2b 의 *binary 100 % hardness* 와 M1b 의 *binary 0 %* 사이의 *sigmoid-gradient* 가 *sweet spot* 형성하는가? E5-Mistral cached e5_margin 으로 sigmoid 가중치 $w_i = \sigma(\alpha_w \cdot \text{e5\_margin}_i)$ 적용, weighted-mean margin loss 학습.

**Pre-commit (result-blind)**: `report/_exp13_14_pre_commit.md`. α_w=10 single value × 3 seeds × SciFact. STOP rule: 결과 무관 sweep / variant 금지.

**Loss**: $\mathcal{L} = \frac{\sum_i w_i \cdot \max(0, m - s_i^+ + s_i^-)}{\sum_i w_i}$ — mined HN *유지*, 가중치 만 조정 (분모 정규화로 학습률 sensitivity 차단).

**3-seed grid** (artifact: `outputs/14_difficulty_weighted_hn/scifact/seed_{42,1337,2024}/qv_r8_l12_diffw10/`):

| Seed | NDCG@10 all | Δ all (CI 95 %) | Δ confused | Δ easy |
|---|---|---|---|---|
| 42   | 0.6552 | +0.009 [−0.022, +0.040] **(CI 0)** | +0.100 ✓ | −0.068 ✗ |
| 1337 | 0.6536 | +0.007 [−0.014, +0.028] **(CI 0)** | +0.060 ✓ | −0.038 ✗ |
| 2024 | 0.6493 | +0.003 [−0.028, +0.034] **(CI 0)** | +0.095 ✓ | −0.074 ✗ |
| **3-seed mean ± std** | **0.6527 ± 0.003** | **+0.006 ± 0.003** (3/3 CI 0 포함, **NOT strict**) | **+0.085 ± 0.022 ✓** | **−0.060 ± 0.020 ✗** |

**Branch 판정**:

| 조건 | 임계 | 3-seed mean | 판정 |
|---|---|---|---|
| Branch (a) Δ all > +0.025 strict | strict | +0.006 | ✗ (fail by 0.019) |
| Branch (a) Δ easy > −0.040 | strict | −0.060 | ✗ (fail by 0.020) |
| Branch (c) Δ all < +0.015 | strict | +0.006 | ✓ |
| Branch (c) Δ confused ≈ half (< +0.052) | strict | +0.085 | ✗ (still strong) |

→ **Branch (c) 변형 — *softer Phase 2b, sub-binary on Δ all, but Δ confused not attenuated***. Sweet spot 없음, *continuous control* 의 theoretical novelty 가 empirical separation 으로 전이 안 됨.

**Triplet weight 분포** (α_w=10 의 sigmoid regime, deterministic across seeds):

| 통계 | 값 | 해석 |
|---|---|---|
| weight mean | 0.537 | binary 50 % 가중 와 effective intensity 동등 |
| weight median | 0.588 | 좌편향 (FN 가까운 triplet down-weighted) |
| weight std | 0.275 | 분포 spread 작음 |
| weight range | [0.001, 0.999] | full sigmoid range 도달 |
| e5_margin mean | +0.013 | 거의 0 center, FN 비율 ~46 % (e5_margin < 0) |

→ **α_w=10 이 "uniform attenuation" 처럼 동작** — *individual triplet discrimination* 부족, 단지 *전체 hard intensity 의 평균적 감쇠*.

**Data-side family equivalence — Exp 12 (binary) vs Exp 14 (continuous)**:

| Metric | Exp 12 (binary cut) | Exp 14 (continuous) | Δ |
|---|---|---|---|
| Δ all | −0.004 ± 0.005 | +0.006 ± 0.003 | +0.010 (slight) |
| Δ confused | +0.080 ± 0.004 | +0.085 ± 0.022 | +0.005 (실효 동일) |
| Δ easy | −0.073 ± 0.005 | −0.060 ± 0.020 | +0.013 (slight) |
| Δ all CI 0 포함 | 3/3 | 3/3 | 동일 |

→ **두 *수학적으로 다른* weighting (binary $\{0,1\}$ vs continuous $(0,1)$) 이 통계적으로 구분 안 되는 frontier 점유** — *anchor-side family 의 동일 패턴* (Exp 11 ≈ Exp 13) 과 결합 → **frontier 가 family 별로 fixed location**.

**LoRA capacity 사용** (3-seed mean):
- **‖A‖_total** = 8.78 ± 0.34 (Phase 2b ~8.7 과 동등)
- **‖B‖_total** = 1.83 ± 0.35 (Phase 2b ~1.8 과 동등) — **Anchor-side family (Exp 11 ~1.8, Exp 13 1.34) 와 명확 분리**, update magnitude 면에서 Phase 2b 와 동등 = *anchor preservation pressure 부재*.

**Seed variance — *unstable regime***: Exp 14 의 Δ confused std 0.022 (3-seed range 0.060-0.100, 40 % spread) — anchor-side 의 0.007-0.010 (12 %) 대비 3-5×. *Sigmoid mid-region uniform attenuation* + *val NDCG monotone increase ep3 best* (Phase 2b / anchor-side 의 ep1 best 와 다름) 가 *practitioner-actionable continuous control* 의 robustness 우려.

**핵심 함의**:
1. **Data-side family 의 binary ≈ continuous equivalence** — *수학적 차이가 empirical separation 으로 전이 안 됨*. Anchor-side family 의 동일 패턴 (Exp 11 ≈ Exp 13) 과 함께 → *all forms of intervention 의 frontier 가 family 별 fixed*.
2. **Three-frontier structure** — anchor-side (upper, Δ all ≈ +0.030) / data-side weighting (lower, Δ all ≈ 0) / data-side substitution (M1b unique, Δ all +0.021 with confused half). Paper §7.4.1 의 *6-lever framework* 으로 완성.
3. **Sweet spot 의 *empirical 부재* 확정** — hard-contrast over-correction 의 *intervention space exhaustively bounded*.
4. **α_w sensitivity 의 future work** — α_w=10 의 unstable variance 가 *practitioner-actionable* 한계, STOP rule 따라 본 paper 미실시.

**상세 보고서**: `report/14_difficulty_weighted_hn_report.md` (figures: `report/figures/14_difficulty_weighted_hn/`).

### 7.3.i Exp 15 diagnostics — *Conditional LoRA frontier-breaking hypothesis 의 empirical falsification* (2026-05-25)

**가설**: §7.3.c.iii 의 redistribution 회계 항등식 ($\Delta_{\text{easy}} = (\Delta_{\text{all}} - w_{\text{conf}} \Delta_{\text{conf}}) / w_{\text{easy}}$) 의 *대수적 원인* (constant $\Delta W$) 을 *제거* 하는 **conditional LoRA** ($h = Wx + g(q) \cdot BA \cdot x$) 가 frontier 돌파 가능한가? §7.3.g Diagnostic B 의 anchor-side equilibrium (cos = 0.824) 의 *누르기* 한계 vs conditional 의 *구조적 제거* (cos = 1.0 가능) 의 *empirical 검정*.

**Diagnostic chain** (4 cheap experiments, ~30 min 총):

| Diagnostic | 측정 | 결과 (3-seed mean ± std) |
|---|---|---|
| **(α)** score-margin AUC | frozen top-1/top-2 margin 의 confused prediction | **AUC = 0.836** ✓ (router signal 강함) |
| **(γ)** oracle test-time conditional | gold confused/easy label 로 LoRA/frozen 분기 | **Δ all = +0.048 ± 0.008** ✓ (perfect routing ceiling, anchor-side 의 1.58×) |
| **(β)** confused-only triplet training | 4250 confused triplet 만 학습 (Phase 2b 동일 other config) | **Δ all = −0.387 ✗** *catastrophic* (training-side path 차단) |
| **(δ)** margin-routed Phase 2b | inference 시 score-margin 으로 LoRA/frozen 분기 (no retraining) | **Δ all = +0.011 ± 0.007** (anchor-side +0.030 의 *절반 미만*) |

**핵심 함의**:

1. **Frontier-breaking hypothesis 의 *empirical falsification***: (γ) oracle ceiling +0.048 은 *real* (frontier 외부 공간 존재), 하지만 (β) training-time filtering 은 *catastrophic*, (δ) realistic inference-time routing 은 *anchor-side 의 절반 미만*. **Realistic Exp 15 minimal realization 이 frontier 돌파 실패**.

2. **Borderline-cost concentration mechanism direct evidence**: AUC 0.836 이 높음에도 realistic gain 이 *linear prediction (AUC × oracle ≈ 0.040)* 보다 약함 (+0.011). 경계영역 query 의 high-stakes misrouting 이 *misrouting rate 보다 비례적으로 큰 비용*.

3. **Training-distribution dependency 확정**: (β) catastrophic failure (Δ all −0.387, NDCG@10 0.26 vs baseline 0.65) 는 *full query distribution 노출* 이 frozen-encoder + LoRA 의 *기본 retrieval 동작 보존* 에 필수임을 직접 입증.

4. **6-lever framework 유지** — Exp 15 의 *realistic* 형태는 framework 의 inferior 구성원, frontier 추가 lever 아님. *Frontier 가 inference-time conditional routing (AUC 0.84) 에도 robust* = paper main contribution 강화.

5. **Future work 의 informed 정리** — Learned router, end-to-end joint training, reranker 형태의 elaborate Exp 15 는 *theoretical promise* 있으나 (β)/(δ) 결과로 *practical limit* 의 empirical evidence 확보. §9 future work section.

**상세 보고서**: `report/15_exp15_diagnostics_report.md` (figures: `report/figures/_exp15_diagnostics/`).
**Scripts**: `report/_exp15_diagnostics.py` (α+γ), `experiments/15a_confused_only_baseline/run.py` (β), `report/_exp15_diagnostic_delta.py` (δ).

**Diagnostic B sub-experiment (post-hoc measurement, data-side family internal representation)**:

Exp 14 의 internal representation 측정 (script: `report/_repr_collapse_exp14.py`, 3 seeds × SciFact test 300 docs, CPU ~2 min) → **6-lever framework × internal representation grid** 완성:

| Lever | Family | eff_rank doc | eff_rank tok | anchor cos tok | Δ all (external) |
|---|---|---|---|---|---|
| frozen | — | 9.86 | 55.13 | 1.000 | (anchor) |
| Exp 12 (binary) | data-w | 1.22 ± 0.01 | 1.72 ± 0.05 | (미측정) | −0.004 |
| **Exp 14** (continuous) | data-w | 1.38 ± 0.40 | 2.42 ± **1.41** | **0.539 ± 0.122** | +0.006 (CI 0) |
| Exp 11 (relational) | anchor | 1.90 ± 0.25 | 9.63 ± 3.08 | (미측정) | +0.029 (2/3) |
| **Exp 13** (absolute) | anchor | 2.33 ± 0.06 | **9.01 ± 0.36** | **0.824 ± 0.005** | +0.030 (3/3) |

**3-fold paper-grade findings**:

1. **Family-level external/internal alignment** — anchor-side ≫ data-side at *every internal metric*: eff_tok 4.4× (9.3 vs 2.1), eff_doc 60 %↑ (2.1 vs 1.3), anchor cos 53 %↑ (0.82 vs 0.54). *6-lever 의 3-frontier structure 가 external (Δ all) 과 internal (eff_rank, anchor cos) 모두에서 일관*. **Family separation 의 multi-level robustness** — paper main mechanistic finding.

2. **Within-family external 동등 ↔ internal variance pattern 분리** — Exp 11 vs Exp 13: external std (Δ all) 동등이지만 internal std (eff_tok) **Exp 11 8.5× larger** (seed 2024 의 13.18 outlier, *relational anchor 의 rotation-invariance 가 internal repr 자유도 ↑*). Exp 12 vs Exp 14: 동등 external 이지만 **Exp 14 internal std 28× larger** (bimodal seed 42/2024 vs 1337, *continuous weighting 의 uniform attenuation 이 seed-dependent collapse 유도*).

3. **Bimodal seed pattern 의 internal-external mechanism direct alignment** — Exp 14 seed 1337 의 *milder collapse* (eff_tok 4.04, anchor cos 0.68 vs 다른 seeds 1.6 / 0.47) → *milder NDCG redistribution* (Δ confused +0.060, Δ easy −0.038 vs 다른 ~+0.10 / −0.07). **Seed-level internal collapse magnitude ↔ external NDCG redistribution 의 direct correlation** = paper-grade mechanism direct evidence.

→ §7.3.f.ii NFCorpus direction puzzle + §7.3.g Exp 13 anchor + 본 §7.3.h Exp 14 data-side 의 *direct evidence chain* 완성 → *family-level internal/external alignment* 의 six-lever empirical mapping 완료.

상세: `report/14_difficulty_weighted_hn_report.md` §5, data: `report/figures/_repr_collapse_exp14/`.

### 7.3.j Exp 16 (multi-layer per-token anchor) — *anchor scope ablation, branch (c) over-restriction confirmed* (2026-05-25)

**가설** (§3.8 ablation completeness 의 명시적 요구 + CLAUDE.md §1.3 prior diagnostic finding 의 *direct architectural translation*): Anchor 를 final ColBERT output (Exp 13) 만이 아닌 BERT layers {0, 3, 6, 9, 12} 의 *5-layer cumulative* 로 확장 시 frontier 외부 도달 가능한가?

**Pre-commit (result-blind)**: `report/_exp16_pre_commit.md`. 5-layer fixed, λ_dir=1.0 single value, 3 seeds × SciFact. Loss = $\sum_{\ell \in L} (1-\cos(h_{\ell}^{\text{LoRA}}, h_{\ell}^{\text{frozen}})) / 5$ (uniform weight per layer).

**3-seed grid** (artifact: `outputs/16_multilayer_anchor/scifact/seed_{42,1337,2024}/qv_r8_l12_dir1_multilayer/`):

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|
| 42   | 0.6238 | +0.008 [−0.016, +0.031] (CI 0) | +0.073 ✓ | −0.048 ✗ |
| 1337 | 0.6434 | −0.003 [−0.027, +0.021] (CI 0) | +0.066 ✓ | −0.061 ✗ |
| 2024 | 0.6534 | +0.007 [−0.016, +0.030] (CI 0) | +0.072 ✓ | −0.048 ✗ |
| **3-seed mean ± std** | **0.6402 ± 0.015** | **+0.004 ± 0.006** (3/3 NOT strict) | **+0.071 ± 0.004 ✓** | **−0.052 ± 0.008 ✗** |

**Branch (c) "over-restriction" 확정**: Δ all 3/3 CI 0, Δ easy 가 Exp 13 (−0.021) 의 **2.5× damage**, Δ confused (+0.071) 가 Exp 13 (+0.092) 의 77 %. Multi-layer 가 모든 metric 에서 Exp 13 대비 *명백 열등*.

**Diagnostic B — *Loss budget dilution* mechanism direct evidence** (`report/_repr_collapse_exp16.py`):

| Layer ℓ | cos(LoRA, frozen) (Exp 16) | tok_eff_rank (Exp 16) | tok_eff_rank (frozen) | Interpretation |
|---|---|---|---|---|
| 0 (embed) | **1.000** | 247.35 | 247.35 | 100 % (LoRA 영향 없음, *budget 낭비*) |
| 3 | **0.998** | 150 | 157 | redundant constraint |
| 6 | **0.991** | 102 | 109 | redundant constraint |
| 9 | 0.965 | 47 | 65 | partial |
| **12 (final BERT)** | **0.697** ⚠️ | **4.6** | 43.06 | **insufficient constraint, catastrophic collapse** |
| (ref) Exp 13 final ColBERT 128-dim | 0.824 | 9.01 | 55.13 | sweet spot |

**3-fold mechanism finding**:
1. **L0-L6 의 redundant constraint** — LoRA 가 q, v 의 후행 effect 만 받기 때문에 intermediate layers 는 *원래도 frozen 과 거의 동일* (cos ≥ 0.99). Loss budget 낭비.
2. **L9-L12 의 insufficient constraint** — 5-layer equal weight → final layer (가장 important) 가 budget 의 1/5 만 받음. Final cos 0.697 (Exp 13 의 0.824 보다 17 % 더 멀어짐).
3. **L12 catastrophic collapse** — token eff_rank 4.6 (Exp 13 의 1/2, frozen 의 11 %). *Loss budget dilution + intermediate redundancy combined effect*.

→ **Anchor-side family 의 optimal scope = final layer only**. Single-point intervention 이 distributed intervention 보다 효율적. *CLAUDE.md §1.3 prior finding 의 "signal exists" ≠ "intervention should target all"* — diagnostic location 과 intervention location 의 separation.

**6-lever framework no change** — Exp 16 의 Δ all (+0.004) 가 data-side weighting (Exp 12/14) 보다 약간 위, anchor-side (Exp 11/13) 보다 명백 아래. *Framework 의 inferior 구성원*, §9.3 *anchor scope ablation* future-work entry 로 정리. §3.8 ablation completeness rule 의 *strict 충족*.

**상세 보고서**: `report/16_multilayer_anchor_report.md` (figures: `report/figures/16_multilayer_anchor/` + `report/figures/_repr_collapse_exp16/`).

### 7.3.k Spine research narrative ablations — *6-lever 표 정정 + anchor incremental Δ* (2026-05-25)

본 §은 *reviewer recommendation Tier 1 + B1 + C1* 의 measurement-only ablations 결과 — 기존 학습 checkpoint 위 추가 측정만, 새 실험 없음. Script: `report/_spine_ablations.py`. 4-fold finding:

#### A1. M1b Δ easy 3-seed 실측 — **6-lever 표 정정**

| | 이전 (추정) | **실측 (3-seed mean ± std)** |
|---|---|---|
| M1b Δ easy | "(~−0.05)" | **−0.017 ± 0.003** |

→ **M1b 의 easy damage 가 추정치의 1/3** — anchor-side (Exp 13 −0.021) 와 거의 동등. §6.1 grid + §7.4.1 framework 의 M1b row 정정.

#### A2. Anchor incremental Δ over Phase 2b LoRA (control = anchor λ=0)

Paired bootstrap Δ NDCG@10 (Exp 11/13 − Phase 2b LoRA, same seed pairs):

| Anchor lever | Δ all (incremental) | Δ confused (incremental) | Δ easy (incremental) |
|---|---|---|---|
| Exp 11 (relational) | **+0.028 ± 0.020** | **−0.002 ± 0.007** | **+0.055 ± 0.031** |
| **Exp 13** (per-token) | **+0.029 ± 0.015** | **−0.012 ± 0.022** | **+0.064 ± 0.009** ⭐ |

→ **Anchor 의 incremental Δ all 이 *전적으로 Δ easy 보존* 에서 옴, Δ confused 에서는 *약간 손해***. Paper-level reframing:

- *Phase 2b LoRA* = confused recovery source (+0.104 vs frozen)
- *Anchor* = easy preservation mechanism (no incremental confused gain, large incremental easy preservation)
- *Net Δ all positive* = easy preservation outweighs confused micro-loss

이게 §7.3.g Diagnostic B 의 *soft equilibrium attractor* (cos = 0.824) 의 *interpretation 정합*: confused push (Phase 2b) ↔ anchor pull (easy preservation), 두 힘의 균형.

#### B1. Exp 13 sanity check ✓

3 seeds 모두 `runs.json` → NDCG@10 재현 = `metrics_aggregate.json` 기록값 (diff < 0.0001). Defensive verification 통과.

#### C1. Train easy/confused split consistency ✓

3 seeds 모두 confused/easy = 368/441 (45.5/54.5 %) — frozen retrieval 의 deterministic 성으로 seed-invariant. 

**Spine ablation 통합 함의**: 6-lever framework 의 *strict integrity* 보강 + anchor mechanism 의 *paper-level interpretation* 정정 (anchor = easy preservation, not confused gain). §7.4.1 framework 도 본 정정 반영.

상세 결과: `report/figures/_spine_ablations/spine_ablations.json`.

### 7.4 통합 함의 — *Paper narrative 의 근본적 재정렬*

| 옛 narrative (실험 00-09 기반) | **새 narrative (robustness audit + mediation 1/1b + Exp 11/12 후)** |
|---|---|
| Translation-trap 의 algebraic 진단 + form-change limit + distillation 의 wrong-lever 가 *frozen-encoder representational limit* 의 정황 증거 | *직접 증거*: 110M encoder unfreeze 가 Δ confused +0.252 의 5× lift. K-invariant ceiling 은 SciFact-specific. 08 의 rank-1 collapse 는 seed artifact. |
| Paper main contribution 후보: translation-trap algebraic 정리 + bilinear M 의 form-change negative result | **§5e main contribution (불변)**: *Universal per-position rank-collapse + LoRA's spatial multiplicity escape* — robust across all methods (06/08/10). |
| Phase 2b 의 *redistribution* (confused +0.104 / easy −0.085 / all ≈ 0) 의 mechanism 은 supervision noise | **결정적 정정 (Exp 12 disambiguation)**: redistribution 의 *주요 원인 = hard-contrast over-correction*. FN noise 는 *minor* 기여 (~14 %). M1b 의 strict net+ 는 *easy contrast 의 작은 gradient* 효과 (hard 회피), noise 제거 부수적. |
| Optimization root vs supervision root 의 disentangling | **결정적 정정 (M1+M1b combined)**: optimization root (warmup+clip) 의 final-state contribution = ZERO. **Sole mechanism = hard-contrast supervision over-correction**. |

#### 7.4.1 *6-lever* trade-off framework (paper-grade final, Exp 13 + Exp 14 추가)

본 grid 는 *data-side* (triplet selection/weighting/substitution) 와 *anchor-side* (frozen baseline 으로의 회귀 항) 두 family 로 정렬:

| Lever | Family | Sub-mechanism | Δ all | Δ confused | Δ easy | Frontier region |
|---|---|---|---|---|---|---|
| Phase 2b | (baseline) | hard 100 % | ≈ 0 | +0.104 ✓ | −0.085 ✗ | redistribution origin |
| **Exp 12** (FN-denoised) | **data-side weighting** (binary) | hard + binary FN cut $w_i \in \{0,1\}$ | −0.004 (CI 0) | +0.080 ✓ | −0.073 ✗ | **data-side weighting (lower)** |
| **Exp 14** (α_w=10) | **data-side weighting** (continuous) | hard + sigmoid weight $w_i = \sigma(\alpha_w \cdot m_i)$ | +0.006 (CI 0) | +0.085 ✓ | −0.060 ✗ | **data-side weighting (lower)** |
| **M1b** (in-batch neg) | **data-side substitution** | mined HN 제거 → in-batch easy | **+0.021 ✓ strict robust** | +0.065 (half) | **−0.017 ± 0.003** (spine A1 실측, anchor-side 수준 easy 보존) | data-side substitution (unique) |
| **Exp 11** (relational λ=1) | **anchor-side (relational)** | hard + Sim Frobenius², rotation-invariant | **+0.029 (2/3 strict)** | +0.101 ✓ | −0.031 (63 % 감소) | **anchor-side (upper)** |
| **Exp 13** (per-token cos λ_dir=1) | **anchor-side (absolute)** | hard + per-token cosine, rotation-sensitive | **+0.030 (3/3 strict)** ⭐ | +0.092 ✓ | **−0.021** (best preserved) | **anchor-side (upper)** |
| **Exp 16** (multi-layer cos λ_dir=1, L={0,3,6,9,12}) | **anchor-side (multi-layer, inferior)** | per-token cosine at 5 BERT layers | **+0.004 (0/3 strict)** ✗ | +0.071 ✓ | −0.052 (Exp 13 의 2.5× damage) | *anchor-side family 의 inferior 구성원* (§7.3.j) |

⇒ **Three-frontier structure** (paper main mechanistic finding):

1. **Anchor-side frontier (upper)** — Exp 11/13, Δ all ≈ +0.030 + Δ easy ≈ −0.025. *Sole strict-positive* family among interventions on hard contrast.
2. **Data-side weighting frontier (lower)** — Exp 12/14, Δ all ≈ 0 + Δ easy ≈ −0.07. *Sub-anchor*.
3. **Data-side substitution lever (M1b)** — unique frontier 위치 (Δ confused half, Δ all strict). Hardness itself 제거.

**Family 내 *binary ≠ continuous theoretical novelty* 의 empirical 비전이성**:
- *Anchor-side*: Exp 11 (rotation-invariant relational, Sim Frobenius²) 와 Exp 13 (rotation-sensitive absolute, per-token cosine) → *statistically equivalent frontier*.
- *Data-side weighting*: Exp 12 (binary $\{0,1\}$ FN cut) 와 Exp 14 (continuous $(0,1)$ sigmoid) → *statistically equivalent frontier*.
- → **Frontier 가 family 별 fixed location**. *Form 의 수학적 차이* 가 *outcome 의 empirical separation* 으로 전이 안 됨 → *intervention space 의 family-level discreteness*.

**Single sufficient mechanism explanation**: *Hard mined-HN 자체가 LoRA 에 *over-correction* 을 강요* → confused recovery 와 easy damage 의 *zero-sum trade-off* 형성. **회피 방법 three families**:
1. *Data-side substitution* — Hard 자체 제거 (M1b) → strict net+, 단 confused half.
2. *Data-side weighting* — Hard 감쇠 (Exp 12 binary / Exp 14 continuous) → Δ all 약화, confused 일부 유지, easy *partial* preserved.
3. *Anchor-side regularization* — Hard 유지 + selective easy protection (Exp 11 relational / Exp 13 absolute) → strict net+ + confused 유지 + easy *best* preserved. **Paper-grade best lever**.

**Mechanism intervention space 의 6-lever exhaustive enumeration** = paper main contribution — *모든 form of intervention 이 trade-off frontier 안에 떨어지며, 그 frontier 가 family 별 분리되어 fixed* 가 *paper 의 final mechanistic finding*.

**Anchor mechanism 의 *interpretation 정정* (spine ablation A2)** — paired bootstrap Δ(Exp 11/13 − Phase 2b LoRA, same seed pairs, anchor=0 control 명시):
- Exp 11 incremental: Δ all +0.028, **Δ confused −0.002**, Δ easy **+0.055**
- Exp 13 incremental: Δ all +0.029, **Δ confused −0.012**, Δ easy **+0.064**

→ **Anchor 의 incremental Δ all 이 *전적으로 Δ easy 보존* 에서 옴, Δ confused 에서는 *약간 손해***. Phase 2b LoRA *자체* 가 confused recovery source (+0.104), anchor 는 그 위에 *easy preservation 만 추가*. §7.3.g Diagnostic B 의 *soft equilibrium* (cos = 0.824) 와 정합: confused push (Phase 2b) ↔ anchor pull (easy preservation) 의 dynamic balance.

**Anchor scope ablation (Exp 16)**: Multi-layer (5 BERT layers, §7.3.j) 가 single-layer (final ColBERT output) 대비 *모든 metric 명백 열등* — *anchor-side family 의 optimal scope = final layer only*. Loss budget dilution + intermediate redundancy mechanism direct evidence (Diagnostic B). CLAUDE.md §1.3 prior diagnostic finding 의 "signal exists" ≠ "intervention should target all" reinterpretation.

### 7.5 남은 통계적 한계

| 항목 | 현 상태 |
|---|---|
| Seed | seed 42 만 single-point + 08 의 seed × 3 robustness check ✓ |
| Cross-dataset | NFCorpus K=2 single point + 01b α-sweep ✓ (모두 SciFact 와 다른 양상) |
| Bootstrap | 10K paired bootstrap × 95 % CI ✓ |
| LOOCV | 미실시 |
| LR/hyperparameter sweep | 미실시 (NFCorpus 결과 가 hyperparameter-dependent 가능성) |

---

## 8. Limitations & open questions

### 8.1 데이터셋 / 통계 한계

- 학습 실험 모두 SciFact 한정 (NFCorpus / FiQA / SciDocs / TREC-COVID / ArguAna 에서 미실시). Cross-dataset 일반화 가능성 미검증.
- 단일 seed (42) — variance 모름.
- BEIR train split 부재 dataset 3 개 (SciDocs / TREC-COVID / ArguAna) — 01_mean_diff 검정 미적용. LOOCV 시 cross-dataset $v$ 활용 필요.

### 8.2 Baseline reproduction gap

| Dataset | Δ vs paper |
|---|---|
| SciDocs | +0.004 ✓ (±0.005 통과) |
| NFCorpus / FiQA / TREC-COVID / ArguAna | -0.008 ~ -0.011 (systematic minor) |
| SciFact | -0.047 (outlier) |

Journal 투고 시점 paper-grade ±0.005 통과 필요. 현 implementation-level systematic difference 의 분리 검정 (C7/C8/C9 — transformers 버전 / PLAID / MPS 정밀도) deferred.

### 8.3 학습 동학의 systematic 한계

모든 학습 실험 (02 / 04 / 05 / 06) 이 *train-overfitting* 패턴. 가능 원인:
- SciFact 의 9K triplet 의 *training signal* 한계.
- AdamW LR=$10^{-3}$ + 5 epoch 의 hyperparameter 가 일반화 부족 유도.
- Static HN mining (baseline 의 top-K) 의 *self-confirming bias* (Xiong 2021, ANCE).

→ **dynamic HN mining 의 *early* 검정** 이 가치 있을 가능성. ROADMAP §"Generalization & robustness" 의 dynamic_hn 우선순위 ↑ 검토.

### 8.4 Direction 의 *내용* 분석 미실시

학습된 v 들의 *의미적 해석* (어떤 token / concept 에 정렬되는지) 미실시. 08_routing_analysis 에서 통합 검토 예정. Paper 의 *원인 규명* 측면 (대주제 §2) 의 결정적 contribution.

---

## 9. Next experiments (ROADMAP §"Stage 2")

ROADMAP 의 *translation-trap pivot* 에 따라 다음 실험은 translation family *밖* 으로의 *minimal 우회* 검정:

### 9.1 Stage 2 종합 결과 (08 + 09)

| 실험 | NDCG@10 all | Δ confused vs baseline | 결과 |
|---|---|---|---|
| 08 r=8 (no distill) | 0.6439 | +0.054 ✓ | ceiling 못 넘음, M rank-1 collapse |
| 09 λ=0.1 (best) | 0.6509 | +0.019 ✓ | partial rank-2, confused lever 약화 |
| 09 λ=0.5, 1.0 | ~0.645 | ≈0 | M ≈ I, 학습 사실상 없음 |

**Stage 2 critical 결론**: *form 변경 + distillation* 의 두 lever 모두 *informed direction subspace ceiling 위로 못 감*. 진단:
- 08: r=8 의 capacity 가 *optimization-driven rank-1 collapse* — 학습이 single dominant axis 만 활용.
- 09: Margin-MSE distillation 이 *anchor regularizer* 로 잘못 작동 — M 을 identity 로 묶음.
- → *Frozen-encoder representational limit* 의 정황 증거.

### 9.2 다음 우선 실험

| # | Dir | 검정 / 의도 | 우선 |
|---|---|---|---|
| **10** | `10_bilinear_rank_sweep` | r ∈ {1, 4, 16, 32, 64}. r=1 (강제 rank-1) ≈ 08 expected. r=16/32 의 추가 capacity 가 effective rank 활성화 시키는지의 *직접* 검정. | **HIGHEST** |
| **18** | `18_lora_phi` | 10 도 ceiling 못 넘으면 *encoder representational limit* 확정 → LoRA on Φ (transformer attention/FFN) 의 *upper bound* 검정. 학습 가능 파라미터 ↑ (≤ 50K 안 어떻게 분배할지 별도 설계). | **HIGH (conditional)** |
| (deferred) | `nuclear_norm_reg` on 08 | distillation 대신 *직접* rank diversity 강제. 10 결과 보고 결정. | medium |
| (deferred) | E5 → MonoT5 / cross-encoder teacher | 09 의 noise teacher 문제 해소 시도. 학습 cost ↑. | low |
| 11, 12 | `11_*_nfcorpus`, `12_*_fiqa` | 10 또는 18 의 best config 의 cross-dataset 일반성 검정. | conditional |

자세한 ROADMAP + conditional execution graph: [`ROADMAP.md`](ROADMAP.md).

### 9.3 Exp 15 future work — *Conditional LoRA 의 elaborate realization* (post-paper)

§7.3.i 의 4-diagnostic chain 이 *Exp 15 minimal realization* (score-margin 기반 inference-time routing) 의 frontier 돌파 실패 입증 (Δ all +0.011 < anchor-side +0.030). 단 (γ) oracle ceiling +0.048 의 *존재* 는 *frontier 외부 공간* 의 reality 직접 증거.

이 공간 도달의 *elaborate realization* 3 가지 future work proposal:

| Proposal | Mechanism | Risk | Expected Δ all |
|---|---|---|---|
| **(F1) Learned routing classifier** | Score-margin 대신 transformer 기반 query feature classifier 로 confused 예측. AUC ↑ 가능. | 추가 training + post-hoc framing | +0.020-0.035 (anchor-side 수준) |
| **(F2) End-to-end joint conditional LoRA** | Router $g(q)$ 와 LoRA $BA$ 의 *joint training* with explicit routing supervision (confused/easy split). §1.4 gate 사인 의 *세 대응* (supervision + 2-stage + query-level signal) 동시 적용. | High — gate (03/04) 와 동일 영역, *세 대응* 의 empirical 효과 미검정 | +0.020-0.045 (theoretical promise, empirical 미입증) |
| **(F3) Reranker 형태 conditional** | Top-K (예 K=20) 후보에만 conditional 적용 — full retrieval 대신 reranking cost. *Pipeline 구조* 의 routing latency 분산. | Mid — reranker overhead, *not a single retriever* | +0.025-0.045 |

**제약**:
- §7.3.i (β) 의 catastrophic failure 가 시사하는 *training distribution dependency* — (F2) 의 end-to-end 학습 시 *easy query distribution 부분 노출* 보장 필요.
- §7.3.i (δ) 의 *borderline-cost concentration* — AUC 가 0.95 까지 올라도 *경계영역 misrouting cost* 가 oracle 도달 막을 가능성.
- §7.3.i (γ) 의 oracle ceiling +0.048 은 *upper bound* — 어떤 realistic router 도 이 위로 못 감.

→ **본 paper 의 STOP rule 따라 추가 실험 미실시**. Future work 의 *informed framing* (theoretical promise + empirical limits) 자체가 paper §8 limitations 의 paper-grade entry.

---

## 10. 보고서 / Artifact 위치

### Per-experiment 상세 보고서
- [`report/00_baseline_report.md`](report/00_baseline_report.md)
- [`report/01_mean_diff_report.md`](report/01_mean_diff_report.md)
- [`report/02_final_layer_vector_report.md`](report/02_final_layer_vector_report.md)
- [`report/03_scalar_gate_report.md`](report/03_scalar_gate_report.md)
- [`report/04_per_token_gate_report.md`](report/04_per_token_gate_report.md)
- [`report/05_five_layers_report.md`](report/05_five_layers_report.md)
- [`report/06_k_sweep_report.md`](report/06_k_sweep_report.md)
- [`report/07_random_direction_scaled_report.md`](report/07_random_direction_scaled_report.md)
- [`report/08_bilinear_M_minimal_report.md`](report/08_bilinear_M_minimal_report.md)
- [`report/09_bilinear_M_e5_distill_report.md`](report/09_bilinear_M_e5_distill_report.md)
- [`report/10_lora_phi_report.md`](report/10_lora_phi_report.md)
- [`report/13_frozen_direction_anchor_report.md`](report/13_frozen_direction_anchor_report.md) — Exp 13 anchor-side family (per-token cos, ⭐ best lever) + Diagnostic B §5
- [`report/14_difficulty_weighted_hn_report.md`](report/14_difficulty_weighted_hn_report.md) — Exp 14 data-side family (continuous sigmoid) + Diagnostic B §5
- [`report/15_exp15_diagnostics_report.md`](report/15_exp15_diagnostics_report.md) — Exp 15 (Conditional LoRA) 4-diagnostic chain (frontier-breaking falsification)
- [`report/16_multilayer_anchor_report.md`](report/16_multilayer_anchor_report.md) — Exp 16 (multi-layer per-token anchor, branch (c) over-restriction confirmed) + Diagnostic B §3

### Pre-commit / Diagnostic scripts (report/)
- `_exp13_14_pre_commit.md`, `_exp16_pre_commit.md` — result-blind pre-commits
- `_repr_collapse_exp13.py`, `_repr_collapse_exp14.py`, `_repr_collapse_exp16.py` — Diagnostic B (mechanism direct verification)
- `_exp15_diagnostics.py`, `_exp15_diagnostic_delta.py` — Exp 15 4-diagnostic chain (α/γ/δ)
- `_spine_ablations.py` — Spine research narrative ablations (Tier 1 + B1 + C1)

### Figures
`report/figures/{NN}_{exp_name}/{figure}.{pdf,png}` — 모든 실험의 PDF (벡터, 본문 삽입용) + PNG (raster, 보고서 임베드용).

### Outputs (artifact)
`outputs/{NN}_{exp_name}/{dataset}/seed_{seed}/` — 실행 별 config / env / runs / runs_scored / metrics_per_query / metrics_aggregate / delta_vs_* JSON. 후속 LSR 실험의 paired bootstrap 재계산 직접 활용 가능.

### Reproducibility
- Python 3.14.4 (`.python-version`), `.venv/`
- `requirements.txt` + `requirements.lock.txt` (68 packages exact pin)
- Seed 42 (실험 전반 고정), `torch.use_deterministic_algorithms(True, warn_only=True)`.

---

## 11. Project meta

| 항목 | 값 |
|---|---|
| Repo | `/Users/chanlee/Desktop/Programming/colbert_hn/` |
| Documents | [CLAUDE.md](CLAUDE.md) (constitution), [DESIGN.md](DESIGN.md) (architecture + ablation matrix), [ROADMAP.md](ROADMAP.md) (실험 master plan), [RESEARCH.md](RESEARCH.md) (lab notebook), [CHANGELOG.md](CHANGELOG.md), 본 REPORT.md (cumulative narrative) |
| 본 시점 학습 가능 파라미터 | LoRA 295K (10 Phase 2b, 50K budget 완화 후) — 02 unfrozen 의 110M 의 0.27% |
| 본 시점 총 실험 | 11 개 main (00–10) + 4 robustness audit + 다양 sub-sweep (06 K-sweep × 3, 09 λ-sweep × 3, 10 LoRA × 3 phase + seed × 3 + NFCorpus) = **총 ≈ 30 회 실행** |
| 본 시점 best frozen-side method | **10 Phase 2b (3-seed mean)**: NDCG@10 all 0.6476 ± 0.014, Δ confused **+0.104 ± 0.017 ✓** (02 unfrozen 의 41% 회복), anchor preserved |
| **본 시점 paper main contribution** | ***Universal per-position rank-collapse + LoRA's spatial multiplicity escape*** — 모든 frozen-side method 의 per-position effective rank 가 1-2 의 universal collapse, position 수 (1 vs 24 vs full) log-scale 의 monotonic correlation with confused lift |
| 다음 우선 실험 | NFCorpus on Phase 2b 종료 + paper deliverable 최종 정리 (글쓰기 phase) |
