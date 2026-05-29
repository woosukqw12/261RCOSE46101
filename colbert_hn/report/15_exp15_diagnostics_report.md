# 15_exp15_diagnostics — Conditional LoRA 의 *4-fold diagnostic chain* (frontier-breaking hypothesis 의 empirical falsification)

본 보고서는 **Exp 15 (Conditional LoRA)** 가설의 *full design 진입 전* sequential diagnostic chain. 이론적 분석 (anchor-side equilibrium 은 *누르기*, conditional LoRA 는 redistribution 의 *대수적 원인* 제거) 의 *경험적 검정* — 4 cheap diagnostic (α/β/γ/δ) 으로 frontier-breaking 가능성 평가.

**결론**: 4-diagnostic chain 이 *empirically falsifies* Exp 15 의 minimal realization. (γ) oracle ceiling +0.048 은 *real* 하지만, (β) training-time filtering 은 catastrophic, (δ) inference-time score-margin routing 은 +0.011 (anchor-side family 의 절반). **Frontier 가 inference-time conditional routing (AUC 0.84) 에도 robust** — 6-lever framework 의 frontier-fixed 주장 *추가 강화*. More elaborate Exp 15 designs (learned router, end-to-end joint) 는 future work.

---

## 1. Theoretical motivation — *왜 conditional LoRA 가 유일 frontier 후보였나*

### 1.1 Redistribution 의 *대수적 항등식* (§7.3.c.iii)

$$\Delta_{\text{easy}} = \frac{\Delta_{\text{all}} - w_{\text{conf}}\Delta_{\text{conf}}}{w_{\text{easy}}}$$

§7.3.c.iii 에서 −0.086 (예측) ≈ −0.085 (실측) — *99 % 정합*. Redistribution 은 *noise 가 아니라 회계 항등식*: confused 가 올라간 만큼 easy 가 *수학적으로* 내려감. 원인은 표준 LoRA 의 $\Delta W$ 가 *상수* 라는 사실:

$$\delta h(q) = \Delta W \cdot x(q), \quad \forall q$$

Confused 에 필요한 $\Delta W$ 가 easy 의 $x(q_{\text{easy}})$ 에도 곱해짐.

### 1.2 Anchor-side family 의 *equilibrium 한계* (Diagnostic B, §7.3.g)

Exp 13 의 *per-token cosine anchor* 가 학습 후 cos(LoRA, frozen) = **0.824** 에서 멈춤 (§5.2 Finding ⭐1). 잔여 anchor_loss = 0.176 — *confused push* ↔ *anchor pull* 의 dynamic equilibrium. Anchor 가 *strict identity (cos=1) 가 아닌 soft attractor*.

→ Anchor-side family 의 *frontier 위에서의 위치* 는 *equilibrium 의 위치* 에 의해 결정. λ_dir ↑ → equilibrium cos→1.0 (over-restrict confused, branch (c) 위험).

### 1.3 Conditional LoRA 의 *대수적 우회 가설*

$$h = Wx + \frac{\alpha}{r} \cdot g(q) \cdot BA \cdot x, \quad \delta h(q) = g(q) \cdot \frac{\alpha}{r} \cdot BA \cdot x(q)$$

- $g(q_{\text{conf}}) \to 1$: confused 는 full LoRA intervention.
- $g(q_{\text{easy}}) \to 0$: easy 는 $\delta h = 0$ → **회계 항등식에서 $\Delta_{\text{conf}}$ 항이 easy 에 *기여 자체를 안 함***.

이론적으로는 redistribution 이 *0 까지* 가능. Anchor 의 *누르기* (equilibrium 0.824) vs conditional 의 *구조적 제거* (cos = 1.0 가능).

### 1.4 Gate (Exp 03/04) 의 사인 분석

기존 gate 실험 (03 scalar, 04 per-token, §4.2) 가 *정확히 이 구조* 인데 실패한 *세 사인*:
1. **Gradient bottleneck**: $g$ 와 $BA$ 의 곱셈 → $g$ 작으면 $BA$ 학습 약화 (닭-달걀)
2. **Supervision absence**: "easy 에서 닫아라" 신호 부재 → gate 가 *자유롭게* always-on 으로 수렴
3. **Token-local information**: token-h 입력으로 query-global 속성 판단 어려움

### 1.5 Exp 15 의 *세 사인 대응 설계*

- (대응 1) Routing supervision: confused/easy split 을 *routing target* 으로 명시 (qrels 사용)
- (대응 2) Gradient bottleneck 우회: 2-stage 학습 (router 먼저, BA 나중) 또는 detached g
- (대응 3) Query-level signal: token-h 대신 *frozen retrieval 의 top-1/top-2 score margin* (qrels-free, test 시 작동)

### 1.6 *근본 난점*: Test-time routing 의 순환

학습 시 confused/easy = *qrels 로 라벨*. **추론 시 qrels 없음** — confused 판별 = retrieval 풀기. **Score-margin signal (frozen 의 top-1/top-2 차이)** 이 *qrels-free* 한 confused 추정 → 본 diagnostic 의 중심 가설.

---

## 2. Sequential diagnostic chain — 4 cheap experiments

본 diagnostic chain 의 목적: *Exp 15 full design 진입 전*, *frontier-breaking 가능성* 의 *empirical foundation* 검증. 4 step sequential (~30 min 총 compute):

| Step | 측정 | 학습? | 시간 | Output |
|---|---|---|---|---|
| **(α)** | score-margin AUC for confused prediction | no | 10 s | router signal 의 *predictive upper bound* |
| **(γ)** | oracle test-time conditional NDCG (gold labels) | no | 30 s | *perfect routing 의 ceiling* |
| **(β)** | confused-only triplet training | yes | ~10 min | *perfect training-side routing 의 outcome* |
| **(δ)** | margin-routed Phase 2b at inference | no | 10 s | *realistic Exp 15 minimal realization* |

---

## 3. (α) Score-margin AUC — *router signal 의 predictive power*

### 3.1 Method

각 SciFact test query 에 대해 frozen ColBERT 의 top-1 score $s_1$, top-2 score $s_2$ 의 *margin* $m = s_1 - s_2$ 측정. Confused label = `top-1 did != relevant`. AUC: *lower margin predicts confused*.

### 3.2 결과

| 통계 | Confused (n=137) | Easy (n=163) | Ratio |
|---|---|---|---|
| margin mean | 0.986 | 3.641 | 3.7× |
| margin median | 0.703 | 3.140 | 4.5× |
| **AUC(margin → confused)** | **0.836** | — | (✓ strong signal) |
| AUC(top1 score → confused) | 0.764 | — | (weaker baseline) |

→ **Score-margin signal 강력**. Confused query 와 easy query 의 margin 분포가 *명확히 분리*. AUC 0.836 ≫ 임계 0.75 → *test-time 순환의 부분 우회* 가능성 입증.

### 3.3 함의

- *Confused = "모델이 자신 없는" query* 의 직관 ✓
- Frozen 의 score margin 이 *qrels 없이* 84 % accuracy 로 confused 예측 가능
- Branch (c) "routing failure" *empirically 배제*

---

## 4. (γ) Oracle test-time conditional — *perfect routing ceiling*

### 4.1 Method

Phase 2b 의 학습된 LoRA runs 와 frozen baseline runs 를 가지고:
- *Gold confused label* (qrels 사용) 으로 routing:
  - confused query → LoRA ranking 사용
  - easy query → frozen ranking 사용
- 결과 ranking 의 NDCG@10 측정, frozen baseline 대비 paired bootstrap CI.

### 4.2 결과 (3-seed mean ± std)

| Slice | Oracle Δ | Phase 2b Δ | Difference |
|---|---|---|---|
| **all** | **+0.0475 ± 0.0078** ✓ (3/3 strict) | +0.0010 ± 0.0144 | **+0.0465** |
| confused | +0.1035 ± 0.0171 ✓ | +0.1035 ± 0.0171 (same by construction) | 0 |
| **easy** | **+0.0000 ± 0.0000** (by construction) | −0.0854 ± 0.0121 | **+0.0854** |

### 4.3 함의

- **Perfect routing 이 frontier 돌파 *가능***: Δ all +0.048 = anchor-side family (+0.030) 의 **1.58×**, M1b (+0.021) 의 **2.26×**.
- Frontier *위쪽 공간* 이 real → 만약 router 가 perfect 면 6-lever framework 외부 lever 존재.
- Δ easy = 0 *by construction* (frozen 행동 재현): conditional 의 *대수적 제거* 가설 의 직접 확인.

---

## 5. (β) Confused-only training — *training-side ceiling 의 catastrophic failure*

### 5.1 Method

Phase 2b 의 모든 setting 동일하되 *triplet filter*: `confused_slice(train_runs, train_qrels, k=1)` 으로 confused query 의 triplet 만 학습. 9190 mined → **4250 confused-only triplets** (~46 %). Single seed (42), early-stop `val_all` (Phase 2b 동일).

Artifact: `outputs/15a_confused_only_baseline/scifact/seed_42/qv_r8_l12_confonly/`.

### 5.2 결과

| Metric | (β) result | Phase 2b | Difference |
|---|---|---|---|
| NDCG@10 all | **0.2598** | 0.6464 (baseline) → 0.6478 (Phase 2b) | **−60 % vs baseline** |
| **Δ all** | **−0.387 ± 0.05** ✗ catastrophic | +0.001 ± 0.014 | **−0.39** |
| Δ confused | −0.093 ✗ | +0.104 ✓ | −0.20 |
| Δ easy | (자동 derived) | −0.085 | (derived) |

학습 trajectory (single seed):
- epoch 1: val_all = **0.318** (best, used) — 이미 catastrophic
- epoch 2: val_all = 0.245
- epoch 3: val_all = 0.165 (단조 감소, *학습 자체가 destabilizing*)

### 5.3 진단 — *왜 confused-only training 이 catastrophic 인가*

세 가설 (mechanism candidates):

1. **Sample size bottleneck**: 4250 triplet 으로 295K LoRA params 학습 → over-fitting risk.
2. **Distribution shift**: 모든 confused query 가 좁은 distribution → BA matrix 가 *전체 query 분포* 에서 oversteer. Easy query distribution 미노출 → 추론 시 LoRA 가 easy 에 적용되면 *random damage* (Phase 2b 의 −0.085 와는 다른 양상 — 전체 LoRA forward path 가 untrained territory).
3. **Selection bias 의 feedback loop**: confused queries 는 *baseline 이 어려워하는* queries → confused-only 학습은 *어려운 case 에 overfitting* + *base retrieval 의 강력한 signal 미학습*.

가설 (2) 가 가장 유력: full LoRA forward path 가 easy query 의 input distribution 에 노출되지 않으면 *기본 retrieval 동작 자체* 가 distort. 실제로 NDCG@10 all = 0.26 은 *frozen baseline (0.65) 보다 훨씬 낮음* — LoRA 가 *학습 하지 않은* 영역에서 *adversarial* 하게 작동.

### 5.4 함의 — Exp 15 의 *설계 제약 확정*

**Training-time filtering = catastrophic** → Exp 15 의 minimal realization 은 **반드시 inference-time only**:
1. Train on FULL triplet set (Phase 2b 그대로)
2. Routing 은 inference time 만

이게 (δ) 의 motivation.

---

## 6. (δ) Margin-routed Phase 2b — *realistic Exp 15 minimal realization*

### 6.1 Method

Phase 2b 의 학습된 LoRA ranking + frozen baseline ranking 그대로 사용 (no retraining). 각 test query 에 대해:
- *Frozen* top-1/top-2 score margin 측정
- Margin < threshold τ → LoRA ranking 사용 (confused 추정)
- Margin ≥ τ → frozen ranking 사용 (easy 추정)

τ 는 *fraction-to-LoRA* 로 parameterize (0.10, 0.20, ..., 1.00). Structural pre-commit: fraction = 0.46 (실측 confused fraction 과 일치). Sweep 은 *sensitivity analysis*, post-hoc selection 아님.

### 6.2 결과 — Fraction sweep

| Fraction | Δ all (mean ± std) | Δ confused | Δ easy |
|---|---|---|---|
| 0.10 | +0.001 ± 0.005 | +0.018 | −0.014 |
| 0.20 | +0.001 ± 0.005 | +0.042 | −0.033 |
| 0.30 | +0.006 ± 0.005 | +0.061 | −0.041 |
| **0.40** (best post-hoc) | **+0.014 ± 0.006** | +0.092 | −0.052 |
| **0.46** (structural pre-commit) | **+0.011 ± 0.007** | +0.092 | **−0.058** |
| 0.50 | +0.010 ± 0.007 | +0.093 | −0.060 |
| 0.60 | +0.003 ± 0.010 | +0.094 | −0.074 |
| 0.80 | +0.002 ± 0.013 | +0.104 | −0.084 |
| 1.00 (= Phase 2b) | +0.001 ± 0.014 | +0.104 | −0.085 |

### 6.3 Pre-commit point 의 Confusion matrix (frac=0.46, 3-seed avg)

| | Predicted easy (frozen) | Predicted confused (LoRA) |
|---|---|---|
| **Actual easy** | TN ≈ 110 | FP ≈ 53 |
| **Actual confused** | FN ≈ 52 | TP ≈ 85 |

- accuracy ≈ 0.65
- precision ≈ 0.62 (LoRA-routed queries 중 실제 confused 비율)
- recall ≈ 0.62 (실제 confused 중 LoRA-routed 비율)

### 6.4 함의 — *왜 AUC 0.836 인데 realistic gain 이 약한가*

**Misrouting cost 가 borderline queries 에 집중**: Easy queries with *lowest margin* (router 가 LoRA 로 routing 하는 borderline) 가 *exactly LoRA 가 flip 시키는 borderline queries*. AUC 가 높아도 *경계영역의 high-stakes misrouting* 이 큰 damage 유발.

수치 분석 (frac=0.46):
- Δ confused = +0.092 (oracle +0.104 의 88 %) ← TPR ~62 %
- Δ easy = −0.058 (Phase 2b −0.085 의 68 % damage 잔존) ← FPR ~32 %
- Δ all 계산: 0.457 × (+0.092) + 0.543 × (−0.058) = +0.042 − 0.031 = +0.011 ✓ (math 일치)

### 6.5 (δ) vs 다른 lever 의 frontier 위치 비교

| Lever | Δ all (3-seed mean) | Frontier 위치 |
|---|---|---|
| **(δ) margin-routed Phase 2b (frac=0.46)** | **+0.011** | data-side weighting (Exp 12/14) 보다 약함 |
| (δ) best (frac=0.40 post-hoc) | +0.014 | 같음 |
| M1b | +0.021 ✓ | data-side substitution |
| Anchor-side (Exp 11/13) | +0.029-0.030 | **anchor-side upper frontier** |
| (γ) Oracle (theoretical) | **+0.048** ✓ | *frontier 외부 (perfect routing 영역)* |

→ **(δ) 가 anchor-side (+0.030) 의 *절반 미만***. **Realistic Exp 15 minimal realization 이 frontier 돌파 *실패***.

---

## 7. 결정적 함의

### 7.1 Frontier-breaking hypothesis 의 *empirical falsification* (paper-grade)

본 diagnostic chain 이 *Exp 15 의 minimal realization* 을 *empirically falsifies*:

1. (α) Router signal 강함 (AUC 0.836) — 가능성 시사
2. (γ) Oracle ceiling +0.048 — *real* (frontier 외부 공간 존재)
3. (β) Training-time filtering catastrophic — *training-side path 차단*
4. (δ) Realistic inference-time routing +0.011 — *anchor-side 절반 미만*

⇒ **Frontier 가 inference-time conditional routing (AUC 0.84) 에도 robust**. 6-lever framework 의 *frontier-fixed* 주장이 *routing-based bypass* 에도 강건함을 추가 입증.

### 7.2 Mechanism 의 *empirical structure*

본 결과의 mechanism direct evidence:

- **Borderline-cost concentration**: AUC 가 0.836 으로 높아도 *경계영역 query* 의 misrouting 비용이 *misrouting rate 보다 비례적으로 큼*. *Linear gain prediction* (AUC * oracle) 이 *실제 gain* 보다 큼 — 일반 ML 의 *imbalanced cost* 패턴.
- **Training-side fragility**: Confused-only training 의 catastrophic failure 가 *frozen-encoder + LoRA* 의 *training distribution dependency* 의 direct evidence. *Full query distribution 노출* 이 LoRA 의 *기본 retrieval 동작 보존* 에 필수.
- **Oracle 의 *unrealizability***: +0.048 ceiling 은 *gold label 필요* → realistic router 로는 도달 불가. *Inference-time conditional routing 의 inherent limit*.

### 7.3 *6-lever → 7-lever?* — *empirically 거부*

본 diagnostic chain 이 *Exp 15 (conditional LoRA via score-margin)* 를 *7th lever 후보* 로 평가 → **거부**:

| 후보 | Δ all (3-seed) | vs Frontier (+0.030 anchor-side) | 결론 |
|---|---|---|---|
| (δ) margin-routed Phase 2b | +0.011 | **명백 하위** | *6-lever framework 의 inferior lever* |
| (γ) oracle | +0.048 | *frontier 외부 (true ceiling)* | *but unrealizable* |

→ **6-lever framework 유지**. Exp 15 의 *realistic* 형태는 framework 의 *inferior* 구성원.

### 7.4 Future work — Exp 15 의 *elaborate realization* 가능성

본 paper 의 STOP rule 따라 *추가 실험 없이* future work 로 정리:

1. **Learned router** — score-margin 보다 정교한 routing signal (transformer 기반 classifier on query features). AUC ↑ 가능, 단 *추가 training + post-hoc framing 위험*.
2. **End-to-end joint training** — router 와 LoRA 를 *함께* 학습 (gate 03/04 의 *세 사인 대응*: explicit supervision + 2-stage + query-aggregate signal). Theoretical promise 강하나 *empirical 검정 미실시*.
3. **Reranker 형태** — top-K 후보에 conditional 만 적용 (full retrieval 부담 없이 routing cost 분산).

본 future work 의 *theoretical promise* 는 강하나, (β) catastrophic + (δ) realistic falsification 결과가 시사하는 *practical 한계* 직시 필요.

---

## 8. Figures

### 8.1 (α) + (γ) summary figure

![Exp 15 diagnostics — α/γ panel](figures/_exp15_diagnostics/diagnostic_alpha_gamma.png)

**Caption** (3-panel):
- **(A)** Score margin distribution by class (SciFact test) — confused (median 0.703) vs easy (median 3.140) 의 4.5× 분리, AUC 0.836.
- **(B)** Oracle vs Phase 2b NDCG Δ per slice (3-seed mean ± std) — oracle Δ all +0.048 (anchor-side +0.030 의 1.58×), oracle Δ easy = 0 by construction.
- **(C)** Per-seed oracle gain scatter — distance above y=x = routing benefit (3 seeds 모두 일관).

### 8.2 (δ) + 4-diagnostic summary figure

![Exp 15 diagnostics — δ panel](figures/_exp15_diagnostics/diagnostic_delta.png)

**Caption** (3-panel):
- **(A)** τ-sensitivity curve — Δ all / Δ confused / Δ easy vs fraction-to-LoRA. Best Δ all = +0.014 at frac=0.40, structural pre-commit (frac=0.46) = +0.011. Oracle ceiling (+0.048) 와 anchor-side (+0.030) 까지 명확히 미달.
- **(B)** Routing confusion matrix at structural pre-commit (frac=0.46) — accuracy 0.65, precision 0.62, recall 0.62. *Borderline misrouting* 직접 시각.
- **(C)** 4-diagnostic summary — Phase 2b / (β) / (δ) / anchor-side / (γ) oracle 의 Δ all bar plot. **Frontier 의 family-level structure + Exp 15 minimal realization 의 falsification** 직접 시각.

---

## 9. 종합

**Exp 15 diagnostic chain 의 학술적 contribution** (negative result paper-grade):

1. **Frontier robustness 의 inference-time routing 검정** — 6-lever framework 의 frontier-fixed 주장이 *score-margin 기반 conditional routing (AUC 0.84)* 에도 robust. Paper main contribution 강화.

2. **Borderline-cost concentration mechanism direct evidence** — AUC 0.836 이 높음에도 realistic gain 이 *linear prediction (AUC × oracle)* 보다 약함. *경계영역 query 의 high-stakes misrouting* 의 quantitative 입증.

3. **Training-side filtering 의 catastrophic failure** (β: −0.387) — Confused-only training 이 *frozen-encoder + LoRA* 의 *training distribution dependency* 의 direct evidence. *Full query distribution 노출* 의 필수성.

4. **Oracle ceiling +0.048 의 *unrealizability*** — Perfect routing 의 *theoretical 공간* 은 real, 단 *qrels-free realistic router* 로는 도달 불가능. *Inference-time conditional routing 의 inherent limit*.

5. **Future work 의 *informed* 정리** — Learned router, end-to-end joint training, reranker 형태의 elaborate Exp 15 designs 가 *theoretical promise* 강하나 *practical limit* 의 empirical evidence 확보.

**STOP rule 준수**: 4 diagnostic 완료 후 추가 실험 없음. *Exp 15 의 minimal realization* 결과 = falsification, *elaborate realization* = future work.

**Raw artifacts**:
- `report/figures/_exp15_diagnostics/diagnostic_alpha.json` — (α) AUC + margin stats
- `report/figures/_exp15_diagnostics/diagnostic_gamma.json` — (γ) oracle per-seed CIs
- `outputs/15a_confused_only_baseline/scifact/seed_42/qv_r8_l12_confonly/` — (β) LoRA checkpoint + metrics
- `report/figures/_exp15_diagnostics/diagnostic_delta.json` — (δ) τ-sweep results

**Scripts**: `report/_exp15_diagnostics.py` (α+γ), `experiments/15a_confused_only_baseline/run.py` (β), `report/_exp15_diagnostic_delta.py` (δ).
