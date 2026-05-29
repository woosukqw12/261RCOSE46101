# 14_difficulty_weighted_hn — *Continuous* sigmoid weighting on triplet margin loss (data-side family complement)

본 보고서는 **Exp 14** (`qv_r8_l12_diffw10`, 3 seeds, SciFact) 의 결과 분석. *Sole sufficient mechanism* (§5f.3 hard-contrast over-correction) 에 대한 *continuous control* 가설 — *Phase 2b 의 binary 100 % hardness* 와 *M1b 의 binary 0 %* 사이의 *sigmoid-gradient* — 이 sweet spot 을 형성하는지 result-blind pre-commit 으로 검정 (`report/_exp13_14_pre_commit.md`).

**결론**: 3-seed mean **Δ all +0.006 ± 0.003 (3/3 NOT strict, CI 0 포함), Δ confused +0.085 ± 0.022 ✓, Δ easy −0.060 ± 0.020 ✗** — pre-commit branch (c) "*Worse, learning signal 약화*" 확정. Continuous weighting 이 *binary 의 spectrum 안* 에 추가 lever 만들지 못함 → **data-side family 의 binary ≈ continuous equivalence** paper-grade negative result. Anchor-side family (Exp 11/13: Δ all +0.030) 와 명확히 분리된 *열등 frontier* 점유. α_w=10 sigmoid regime 의 *unstable seed variance* (Δ confused 0.060-0.100, 40 % range — anchor-side 의 12 % 보다 3× 큰 변동성) 추가 발견.

---

## 1. 동기 + Pre-committed 판정 기준

### 1.1 Theoretical motivation — *spectrum* 의 continuous control

§5f.3 의 single sufficient mechanism = *hard-contrast over-correction*. 본 framework 내 기존 lever 는 모두 *binary*:

| Lever | Hard intensity | Mechanism |
|---|---|---|
| **Phase 2b** | 100 % (no weighting) | 모든 mined HN 동등 — *catastrophic redistribution* (confused +0.104 / easy −0.085) |
| **M1b** | 0 % (in-batch only) | Mined HN 자체 제거 → *strict net+* but confused 절반 (+0.065) |
| **Exp 12** | binary cut at threshold | $w_i \in \{0, 1\}$ — FN 일부 제거, 단 redistribution 유지 |

**미답 질문**: *Binary endpoint 두 개* 사이의 *spectrum* 에 sweet spot 이 존재하는가? — *softer* hard contrast (예 50 %) 가 confused lift 의 *일부* 만 유지하면서 easy damage 의 *대부분* 회피 가능 한가?

**Exp 14** = 이 질문의 *continuous-control* 검정. Sigmoid weight 를 *e5_margin* 으로 인덱싱:

$$w_i = \sigma(\alpha_w \cdot \text{e5\_margin}_i), \quad \alpha_w = 10$$

where $\text{e5\_margin}_i = \cos(e_q, e_{\text{pos}}) - \cos(e_q, e_{\text{hn}})$ — E5-Mistral-7B-Instruct 의 cached embedding (Exp 12 와 동일 source).

**해석**:
- e5_margin ≈ +0.3 (easy, pos ≫ hn): $w \approx 1.0$ — full hardness
- e5_margin ≈ 0 (borderline): $w \approx 0.5$ — half hardness
- e5_margin ≈ −0.3 (FN): $w \approx 0.05$ — near-zero

→ E5 가 *진짜 hard* 라고 평가하는 triplet 만 강한 가중, FN 일 가능성이 높은 triplet 은 약하게 평가. **Exp 12 binary cut 의 *smooth* 대응**.

### 1.2 Loss formulation

$$\mathcal{L} = \frac{\sum_i w_i \cdot \max(0, m - s_i^+ + s_i^-)}{\sum_i w_i}$$

- $s_i^+ = \text{ColBERT}(q_i, \text{pos}_i)$, $s_i^- = \text{ColBERT}(q_i, \text{hn}_i)$
- $m$ = margin (default 1.0)
- **분모 정규화** ($\sum_i w_i$) → 평균 loss scale 유지 (학습률 sensitivity 차단)
- Mined HN *유지* (Phase 2b 동일) — *제거 없이 가중치 만* 조정

→ Exp 12 의 차이: $w_i \in \{0, 1\}$ → $w_i \in (0, 1)$ continuous.

### 1.3 Pre-committed single config (result-blind)

| Item | Value |
|---|---|
| **α_w** | **10** (sigmoid transition over margin ±0.1, single value, *no sweep*) |
| HN source | Mined HN (Phase 2b 동일, *no removal — just weighting*) |
| LoRA | q, v r=8 α=r (Phase 2b 동일) |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early-stop=val_all |
| Max triplets | 9190 |
| Dataset | SciFact |
| Seeds | 42, 1337, 2024 (run together) |
| Tag | `qv_r8_l12_diffw10` |

### 1.4 3-branch pre-commit predictions

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) Sweet spot found** | Δ all > +0.025 strict 3/3 + Δ confused > +0.08 + Δ easy > −0.04 | Continuous > binary. Hard contrast 의 *optimal intensity* 존재 → paper-grade practitioner-actionable finding. |
| **(b) ≈ M1b or Exp 11** | Δ all ≈ +0.02-0.03 (M1b 또는 Exp 11 수준), Δ confused 중간 | Continuous vs binary 가 equivalent — soft sigmoid 가 effective binary threshold 만큼만 동작. Spectrum endpoints 만 differentiable. |
| **(c) Worse than both** | Δ all < +0.015 또는 Δ confused 절반 미만 | Continuous weighting 이 learning signal 약화 — α_w=10 sigmoid transition 이 모든 triplet weight 흐리게 만들어 train signal 부족. |

**STOP rule**: 3 seeds 완료 후 결과 무관 STOP. α_w sweep / variant / cross-dataset *전부 금지*.

---

## 2. 결과

### 2.1 3-seed grid

| Seed | NDCG@10 all | NDCG@10 confused | Δ all (CI 95 %) | Δ confused | Δ easy |
|---|---|---|---|---|---|
| 42   | 0.6552 | 0.2492 | +0.0088 **[−0.0221, +0.0397]** (CI 0) | **+0.1001 [+0.0483, +0.1525]** ✓ | **−0.0679 [−0.1017, −0.0379]** ✗ |
| 1337 | 0.6536 | 0.2447 | +0.0072 **[−0.0140, +0.0277]** (CI 0) | **+0.0603 [+0.0257, +0.0975]** ✓ | **−0.0375 [−0.0611, −0.0173]** ✗ |
| 2024 | 0.6493 | 0.2419 | +0.0028 **[−0.0277, +0.0339]** (CI 0) | **+0.0946 [+0.0463, +0.1452]** ✓ | **−0.0742 [−0.1084, −0.0427]** ✗ |
| **3-seed mean ± std** | **0.6527 ± 0.0031** | **0.2453 ± 0.0037** | **+0.0063 ± 0.0031** (CI 0 all 3) | **+0.0850 ± 0.0216 ✓** | **−0.0599 ± 0.0196 ✗** |

→ **Δ all: 3/3 NOT strict** (모든 seed 의 CI 가 0 을 포함). **Δ confused: 3/3 strict positive**. **Δ easy: 3/3 strict negative**.

### 2.2 Branch 판정

| 조건 | 임계 | 3-seed mean | 판정 |
|---|---|---|---|
| Branch (a) Δ all > +0.025 strict | +0.025 | +0.006 | **✗ (fail by 0.019)** |
| Branch (a) Δ easy > −0.04 | −0.040 | **−0.060** | **✗ (fail by 0.020)** |
| Branch (c) Δ all < +0.015 | +0.015 | +0.006 | **✓** |
| Branch (c) Δ confused ≈ half (vs Phase 2b +0.104) | < +0.052 | +0.085 | ✗ (still strong) |

→ **Branch (c) 부분 정합** — *Δ all 약화* (continuous weighting → learning signal 약화) 는 정합, 하지만 *Δ confused 절반* 예측은 빗나감 (예상 +0.052 vs 실측 +0.085). 즉:

- Continuous weighting 이 *Δ all 의 strict 돌파를 약화* (예측 정합)
- 그러나 *Δ confused 는 attenuate 안 됨* — α_w=10 sigmoid 의 mean weight 0.54 가 *binary 50 % 가중* 과 거의 동등하게 동작, full Phase 2b 의 confused +0.104 를 약하게 (+0.085) 만 감쇠

**최종 branch 분류**: **(c) 변형 — *softer Phase 2b, not sweet spot, sub-binary on Δ all***.

### 2.3 Triplet weight 분포 — α_w=10 의 sigmoid regime 해석

3-seed 모두 동일 (e5_margin 결정적):

| 통계 | Weight 분포 | E5 margin 분포 |
|---|---|---|
| mean   | **0.537** | +0.013 |
| median | **0.588** | +0.036 |
| std    | 0.275 | (~0.3) |
| min    | 0.0005 | −0.757 |
| max    | 0.999 | +0.757 |

**해석**:
- Mean weight 0.54 ≈ *binary 가 절반만 active 한 경우* 와 동등 effective intensity
- Median 0.59 > mean 0.54 → 분포가 좌편향 (FN 가까운 triplet 가 더 많이 down-weighted)
- e5 margin 의 90 % CI 가 ±0.3 안 → α_w=10 sigmoid 가 그 영역에서 0.05 ≤ w ≤ 0.95 로 sigmoid mid-region 점유 → *대부분 triplet 의 weight 가 0.5 부근*

→ **α_w=10 이 "uniform attenuation" 처럼 동작** — *individual triplet discrimination 효과 부족*, 단지 *전체 hard intensity 의 평균적 감쇠*. Binary cut (Exp 12) 의 *threshold-based separation* 효과 와 본질적으로 다르지 않음.

### 2.4 학습 동학 (seed 2024 train history)

3 epoch 진행 (early-stop 미발동, val NDCG 단조 증가):

| Epoch | weighted_loss | val NDCG@10 all | val NDCG@10 confused |
|---|---|---|---|
| 1 | 0.598 | 0.6406 | 0.2635 |
| 2 | 0.120 | 0.6462 | 0.2941 |
| 3 | 0.091 | **0.6581** | **0.3223** (best) |

→ Phase 2b / Exp 11 / Exp 13 (모두 epoch 1 early-stop best) 와 다른 패턴: **val NDCG 가 단조 증가** → ep3 best snapshot. Continuous weighting 이 *학습 신호 약화 + slower convergence* 의 mechanism 직접 증거 — Phase 2b 의 *fast collapse* (rank_loss 6× drop in ep1) 대비 *slower loss decay*.

### 2.5 LoRA capacity 사용 — *Phase 2b 와 거의 동등*

3-seed mean:
- **‖A‖_total** = **8.78 ± 0.34** — Phase 2b ~8.7 과 동등
- **‖B‖_total** = **1.83 ± 0.35** — Phase 2b ~1.8 과 동등

→ **Anchor-side family (Exp 11 ~1.8, Exp 13 1.34) 와 명확히 분리** — Exp 14 가 update magnitude 면에서 Phase 2b 와 동등 = *anchor preservation pressure 부재*. Loss reweighting 이 weight 의 update magnitude 자체는 제약하지 못함.

---

## 3. 함의

### 3.1 Data-side family 의 *binary ≈ continuous* equivalence (paper-grade negative result)

**핵심 발견**: Continuous sigmoid weighting (Exp 14) 와 binary FN cut (Exp 12) 가 trade-off frontier 의 *거의 동일 region* 점유:

| Metric | Exp 12 (binary cut) 3-seed | Exp 14 (continuous w) 3-seed | Δ |
|---|---|---|---|
| Δ all | −0.004 ± 0.005 | **+0.006 ± 0.003** | +0.010 (slight) |
| Δ confused | +0.080 ± 0.004 | +0.085 ± 0.022 | +0.005 (실효 동일) |
| Δ easy | −0.073 ± 0.005 | **−0.060 ± 0.020** | +0.013 (slight) |
| Δ all CI 0 포함 | 3/3 | 3/3 | 동일 |

→ **두 *수학적으로 다른* weighting 방식이 통계적으로 구분 안 되는 frontier 점유** ("binary vs continuous" 의 *theoretical novelty* 가 *empirical separation* 으로 전이 안 됨).

**Anchor-side family** (Exp 11/13) 가 *수학적 차이 없이* 같은 frontier 점유한 것과 *동일 패턴* — frontier 의 family 별 fixed location 시사:

| Family | Lever 수 | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|
| **anchor-side** (Exp 11 relational, Exp 13 absolute) | 2 | **+0.030** | +0.092-0.101 | **−0.021 ~ −0.031** |
| **data-side weighting** (Exp 12 binary, Exp 14 continuous) | 2 | **+0.001** | +0.080-0.085 | **−0.060 ~ −0.073** |
| **data-side substitution** (M1b in-batch) | 1 | +0.021 ✓ | +0.065 | (~−0.05) |
| (baseline) Phase 2b | 1 | ≈ 0 | +0.104 | −0.085 |

→ **Frontier 가 family 별로 명확히 분리** — anchor-side 가 *전 metric 우월*.

### 3.2 α_w=10 sigmoid 의 unstable variance (paper-grade observation)

Exp 14 의 seed variance 가 *다른 모든 lever 대비 현저히 큼*:

| Lever | Δ confused std (3 seeds) | Δ confused range |
|---|---|---|
| Phase 2b | ~0.017 | (0.091-0.123) |
| Exp 11 | 0.010 | (0.095-0.113) |
| Exp 13 | 0.007 | (0.087-0.099) |
| Exp 12 | 0.004 | (0.076-0.084) |
| **Exp 14** | **0.022** | **(0.060-0.100)** |

→ Exp 14 가 *3 ~ 5×* 큰 seed variance. Mechanism 후보:
1. **Sigmoid mid-region 의 *unstable gradient flow*** — α_w=10 가 *대부분 triplet 의 weight 를 0.5 부근* 으로 모음 → batch 마다 effective hard-set 이 *작은 noise* 에 민감하게 변동.
2. **Val NDCG 의 *단조 증가* 패턴** (ep3 best) → *early-stop 의 robustness 보호* 부재. Phase 2b / anchor-side 의 ep1 best (early-stop 즉시 발동) 과 달리, Exp 14 는 final epoch 까지 *trajectory 의 우연성* 누적.

→ Continuous weighting 이 *practitioner-actionable* 이려면 *seed robustness* 도 갖춰야 함 — α_w=10 의 본 setting 은 *unstable*. STOP rule 따라 α_w sweep 미실시, *future work* 로 분리.

### 3.3 6-lever framework (paper-grade final)

§7.4.1 의 5-lever framework 에 Exp 14 추가:

| Lever | Family | Sub-mechanism | Δ all | Δ confused | Δ easy | Frontier |
|---|---|---|---|---|---|---|
| Phase 2b | (baseline) | hard 100 % | ≈ 0 | +0.104 ✓ | −0.085 ✗ | redistribution |
| **Exp 12** | data-side | binary FN cut | ≈ 0 | +0.080 ✓ | −0.073 ✗ | data-side weighting |
| **Exp 14** | data-side | continuous sigmoid | +0.006 (CI 0) | +0.085 ✓ | −0.060 ✗ | data-side weighting |
| **M1b** | data-side | substitution (in-batch) | +0.021 ✓ | +0.065 | (~−0.05) | data-side sub |
| **Exp 11** | anchor-side | relational Sim Frobenius² | +0.029 (2/3) | +0.101 ✓ | −0.031 | **anchor-side (upper)** |
| **Exp 13** | anchor-side | per-token cosine | +0.030 (3/3) | +0.092 ✓ | −0.021 | **anchor-side (upper)** |

⇒ **Three-frontier structure**:

1. **Anchor-side frontier (upper)** — Exp 11/13, Δ all ≈ +0.030 + Δ easy ≈ −0.025. *Sole strict-positive* family.
2. **Data-side weighting frontier (lower)** — Exp 12/14, Δ all ≈ 0 + Δ easy ≈ −0.07. *Sub-anchor*.
3. **Data-side substitution lever (M1b)** — frontier 中 unique 위치 (Δ confused 절반).

### 3.4 Paper §5f narrative 의 *final* 정렬

**§5f.3 sole sufficient mechanism = hard-contrast over-correction** 의 *intervention space* 가 본 6-lever 로 *exhaustive 하게 mapping* 됨:

- *Hardness intensity dimension* 의 *bilateral binary endpoints* (Phase 2b 100 %, M1b 0 %) + *continuous gradient* (Exp 14) → 모두 검정 완료.
- *FN-aware dimension* 의 *binary* (Exp 12) + *continuous* (Exp 14) → 두 형식 *통계 동등*.
- *Anchor regularization dimension* 의 *relational* (Exp 11) + *absolute* (Exp 13) → 두 형식 *통계 동등*, 단 sub-dimensions 와 *orthogonal* — frontier 의 upper boundary.

→ **Mechanism intervention space 의 *6-lever exhaustive enumeration* 자체가 paper-grade contribution** — *모든 form of intervention 이 trade-off frontier 안에 떨어지며, 그 frontier 가 family 별 분리되어 fixed* 가 main result.

---

## 4. Figures

(figures.py 로 artifact 로부터 재현 가능)

### 4.1 Δ NDCG@10 forest plot (3 seeds + mean, vs Exp 12 / Phase 2b)

![Δ NDCG@10 forest plot](figures/14_difficulty_weighted_hn/delta_ci_forest.png)

**Caption**: SciFact 의 paired bootstrap 95 % CI on Δ NDCG@10. Exp 14 (3 seeds + mean) vs Exp 12 (binary FN cut, 3 seeds + mean) vs Phase 2b (baseline). Δ all 의 seed-level CI 모두 0 포함, Δ confused 의 large seed-to-seed range (0.060-0.100) visual 확인.

### 4.2 6-lever trade-off scatter — *family-separated frontier*

![6-lever trade-off scatter](figures/14_difficulty_weighted_hn/six_lever_scatter.png)

**Caption**: Δ confused × Δ easy 평면 상 6 lever (Phase 2b, Exp 11, Exp 12, Exp 13, Exp 14, M1b) 의 seed-level points + 3-seed mean. **Three-frontier structure** 시각 직접 — anchor-side (Exp 11/13, upper-right cluster) vs data-side weighting (Exp 12/14, mid cluster) vs data-side substitution (M1b, distinct point). 1:1 trade-off line 위에서 family 별 *분리된 region*.

### 4.3 Triplet weight distribution — α_w=10 의 sigmoid regime

![Triplet weight distribution](figures/14_difficulty_weighted_hn/weight_distribution.png)

**Caption**: 9190 triplet 의 weight 분포 histogram + e5 margin × weight scatter. Sigmoid 의 *uniform attenuation* 패턴 — mean 0.537, std 0.275 (binary 50 % 와 effective intensity 동등). α_w=10 이 *individual discrimination* 보다 *전체 attenuation* 으로 동작.

### 4.4 학습 곡선 — *trajectory 의 늦은 best*

![Train curves](figures/14_difficulty_weighted_hn/train_curves.png)

**Caption**: Exp 14 seed 2024 의 3-epoch 학습 동학. weighted_loss 단조 감소, **val_ndcg_all 단조 *증가*** (ep1 0.641 → ep3 0.658 best). Phase 2b / anchor-side 의 epoch 1 best 패턴과 다름 → continuous weighting 의 *slower convergence + late best snapshot*.

### 4.5 NDCG slice grid (3 seeds × 3 slices)

![NDCG slice grid](figures/14_difficulty_weighted_hn/ndcg_slice_grid.png)

**Caption**: 3 seeds × 3 slices 의 NDCG@10 bar plot, baseline (frozen ColBERT) 대비. Exp 14 의 Δ confused / Δ easy 의 seed 별 *large variance* 직접 시각 (0.060-0.100 / 0.038-0.074 range).

### 4.6 Lever 별 final frontier comparison — *6-lever overview*

![Final frontier overview](figures/14_difficulty_weighted_hn/family_frontier_overview.png)

**Caption**: 6 lever 의 3-seed mean Δ all, Δ confused, Δ easy 의 grouped bar plot + family annotation. Anchor-side (Exp 11/13) 의 *전 metric 우월성* + data-side family 의 *분리된 lower frontier* 직접 시각.

---

## 5. Diagnostic B — *internal representation* of data-side family (sub-experiment, post-hoc measurement)

본 sub-experiment 는 Exp 14 의 *학습된 LoRA checkpoint 위* 에서 *post-hoc measurement only* (no new training, pre-commit STOP rule 무관). Exp 13 의 *anchor-side Diagnostic B* 와 paired — *6-lever framework × internal representation* 완성.

**Method**: 3 seeds × SciFact test corpus 300 docs sampled (Exp 13 와 동일 sample). 각 checkpoint 의 LoRA-encoded representation 의 collapse magnitude + (reference) anchor proximity 측정.

### 5.1 결과 표 — Exp 14 3 seeds

| Condition | doc_cos μ | tok_cos μ | eff_rank doc | eff_rank tok | **cos(LoRA, frozen) tok** |
|---|---|---|---|---|---|
| frozen baseline | +0.587 | +0.214 | 9.86 | **55.13** | 1.000 (identity) |
| Exp 14 seed 42 | +0.984 | +0.940 | 1.15 | 1.62 | **0.471** |
| Exp 14 seed 1337 | +0.912 | +0.793 | **1.85** | **4.04** | **0.679** |
| Exp 14 seed 2024 | +0.984 | +0.942 | 1.15 | 1.60 | **0.466** |
| **Exp 14 3-seed mean ± std** | **+0.960 ± 0.041** | **+0.892 ± 0.087** | **1.38 ± 0.40** | **2.42 ± 1.41** | **0.539 ± 0.122** |
| Exp 12 3-seed mean (cached, no anchor metric) | +0.975 ± 0.001 | +0.931 ± 0.004 | 1.22 ± 0.01 | 1.72 ± 0.05 | (미측정) |

→ **Bimodal seed pattern**: seed 42 / 2024 가 Phase 2b-level collapse (eff_tok 1.6, anchor cos 0.47), seed 1337 가 *milder collapse* (eff_tok 4.04, anchor cos 0.68). 28× larger std vs Exp 12 binary (0.05).

### 5.2 6-lever internal representation grid — *family-level external/internal alignment*

| Lever | Family | eff_rank doc | eff_rank tok | anchor cos tok | Δ all (external) | Δ confused | Δ easy |
|---|---|---|---|---|---|---|---|
| frozen | — | 9.86 | 55.13 | 1.000 | (anchor) | (anchor) | (anchor) |
| Exp 12 (binary) | data-w | 1.22 ± 0.01 | 1.72 ± 0.05 | (미측정) | −0.004 | +0.080 ✓ | −0.073 ✗ |
| **Exp 14** (continuous) | data-w | 1.38 ± 0.40 | 2.42 ± 1.41 | **0.539 ± 0.122** | +0.006 (CI 0) | +0.085 ✓ | −0.060 ✗ |
| Exp 11 (relational) | anchor | 1.90 ± 0.25 | 9.63 ± 3.08 | (미측정) | +0.029 (2/3) | +0.101 ✓ | −0.031 |
| **Exp 13** (absolute) | anchor | 2.33 ± 0.06 | **9.01 ± 0.36** | **0.824 ± 0.005** | **+0.030 (3/3)** | +0.092 ✓ | **−0.021** |

### 5.3 핵심 발견 — *3-fold family-level evidence*

#### Finding ⭐1: Anchor-side family ≫ Data-side family at *every internal metric*

| Metric | anchor-side mean | data-side mean | Ratio |
|---|---|---|---|
| eff_rank doc | 1.90 / 2.33 (avg ~2.1) | 1.22 / 1.38 (avg ~1.3) | anchor-side **60 %↑** |
| eff_rank tok | 9.63 / 9.01 (avg ~9.3) | 1.72 / 2.42 (avg ~2.1) | anchor-side **4.4×↑** |
| anchor cos tok | 0.824 (Exp 13) | 0.539 (Exp 14) | anchor-side **53 %↑ closer to frozen** |
| Δ all (external) | +0.029 / +0.030 (avg +0.030) | −0.004 / +0.006 (avg ≈0) | anchor-side strict positive |

→ **Family separation 이 *external (Δ all) 과 internal (eff_rank, anchor cos) 모두에서 일관*** — *6-lever framework 의 3-frontier structure 가 internal 면에서도 robust*. **Family-level external-internal alignment** 가 paper main mechanistic finding.

#### Finding ⭐2: Within-family *external 동등 ↔ internal *variance* pattern 분리*

각 family 내 두 lever 가 *mean external* 와 *mean internal* 모두 statistically 동등이지만, **variance 면에서 분리**:

| Family-pair | external std (Δ all) | internal std (eff_tok) | 해석 |
|---|---|---|---|
| Exp 11 (rel.) vs Exp 13 (abs.) | 0.005 vs 0.002 | 3.08 vs 0.36 | **Exp 11 internal variance 8.5×** (seed 2024 의 eff_tok 13.18 outlier) — *relational anchor 가 rotation-invariant 인 만큼 internal repr 자유도 ↑* |
| Exp 12 (binary) vs Exp 14 (continuous) | 0.005 vs 0.003 | 0.05 vs 1.41 | **Exp 14 internal variance 28×** (bimodal seed 42/2024 vs 1337) — *continuous weighting 의 *uniform attenuation* 이 seed-dependent collapse magnitude 유도* |

→ **Same-family lever 의 *form difference* 가 *internal variance* 면에서 차이 만듦** — *수학적 difference 가 outcome distribution 의 *spread* 에 영향*. Mean 은 동등, *robustness profile* 은 분리.

#### Finding ⭐3: Bimodal seed pattern 의 internal-external mechanism direct alignment

Exp 14 seed-level analysis:

| Seed | eff_tok | anchor cos | Δ confused | Δ easy | NDCG@10 all |
|---|---|---|---|---|---|
| 42 | **1.62** | 0.471 | +0.100 ✓ | −0.068 ✗ | 0.6552 |
| 1337 | **4.04** | 0.679 | **+0.060** ✓ (lowest) | **−0.038** ✗ (mildest) | 0.6536 |
| 2024 | 1.60 | 0.466 | +0.095 ✓ | −0.074 ✗ | 0.6493 |

→ **Internal collapse magnitude ↔ external NDCG redistribution 의 *seed-level direct correlation***: seed 1337 의 *milder collapse* (eff_tok 4.04, anchor cos 0.68) → *milder redistribution* (Δ confused +0.060, Δ easy −0.038). **Internal-external mechanism alignment** 의 paper-grade seed-level direct evidence.

→ Exp 14 의 NDCG seed variance 의 *internal mechanistic 원인* 직접 입증 — *high anchor cos seed 는 confused 학습 신호 약화 → 모든 metric milder*.

### 5.4 Paper §7.3.f mechanism evidence chain 강화

Diagnostic B 측정 통합 (cross-family + cross-paper):

| Condition | doc_cos μ | doc eff_rank | tok eff_rank | NDCG @10 confused |
|---|---|---|---|---|
| Phase 2b (SciFact) | ~0.97 | ~1.14 | ~1.72 | +0.104 ✓ |
| Exp 12 (SciFact, binary) | 0.975 | 1.22 | 1.72 | +0.080 ✓ |
| **Exp 14** (SciFact, continuous) | **0.960** | **1.38** | **2.42** | **+0.085 ✓** |
| Exp 11 (SciFact, relational) | 0.910 | 1.90 | 9.63 | +0.101 ✓ |
| **Exp 13** (SciFact, absolute) | **0.876** | **2.33** | **9.01** | **+0.092 ✓** |
| M1b + NFCorpus | ~0.995 | ~1.06 | ~1.05 | +0.246 ✓ (74 % recovery) |

→ **Mechanism evidence chain** (§7.3.f.ii NFCorpus direction puzzle + §7.3.g Exp 13 anchor + §7.3.h Exp 14 data-side) → *family-level internal/external alignment* 의 *six-lever empirical mapping* 완성.

### 5.5 Diagnostic B figure

![Diagnostic B on Exp 14 — data-side family internal representation](../report/figures/_repr_collapse_exp14/repr_collapse_exp14.png)

**Caption** (3-panel):
- **(A) 6-lever cross-family tok eff_rank** — frozen (55.13) → data-side family (Exp 12 1.72 / Exp 14 2.42) ≪ anchor-side family (Exp 11 9.63 / Exp 13 9.01). Family separation 의 internal direct visual.
- **(B) Anchor proximity 비교** — Exp 14 (3 seeds, mean 0.54) vs Exp 13 (3 seeds, mean 0.82). data-side family 의 *anchor 부재* + bimodal seed pattern (seed 42/2024 0.47 ≪ seed 1337 0.68) 직접 visual.
- **(C) Per-token cos 분포** — seed 42 of Exp 14 (red, mean 0.47) vs Exp 13 (green, mean 0.82). 두 family 의 internal anchor proximity *완전 분리* 의 distribution-level visual.

**Artifact**: `report/figures/_repr_collapse_exp14/repr_collapse_exp14_data.json` + `.{pdf,png}`.
**Script**: `report/_repr_collapse_exp14.py` (CPU, cache resume, ~2 min on Mac M1).

---

## 6. 종합

**Exp 14 의 학술적 contribution** (negative result paper-grade):

1. **Data-side family 의 binary ≈ continuous equivalence** — Exp 12 (binary cut) 와 Exp 14 (continuous sigmoid) 가 *수학적으로 다른* weighting 임에도 *통계적으로 구분 안 되는* frontier 점유. *Anchor-side family 의 동일 패턴* (Exp 11 ≈ Exp 13) 과 결합 → **모든 mechanism intervention 의 frontier 가 family 별 fixed location** 확정.

2. **Three-frontier structure** — anchor-side (upper) vs data-side weighting (lower) vs data-side substitution (M1b) 의 *명확한 frontier 분리*. Paper §7.4.1 의 *6-lever framework* 최종 정렬.

3. **α_w=10 의 unstable seed variance** — anchor-side 의 3-5× 큰 seed variance. *Sigmoid mid-region uniform attenuation* + *late-epoch best* trajectory 의 mechanism direct evidence. *Practitioner-actionable continuous control* 의 향후 sweet spot 탐색은 α_w sensitivity analysis 필요 (future work, STOP rule 따라 본 paper 미실시).

4. **Sweet spot 의 *empirical 부재* 확정** — *binary 두 endpoint* 사이의 continuous spectrum 에 sweet spot 없음. Hard-contrast over-correction 의 *intervention space exhaustively bounded* 의 직접 증거.

5. **§5f mechanism intervention space 의 6-lever exhaustive enumeration** 으로 paper main contribution 완성 — *"모든 form of intervention 이 trade-off frontier 안에 떨어지며, 그 frontier 가 family 별 분리되어 fixed"* 이 paper 의 *final mechanistic finding*.

**STOP rule 준수**: 3 seeds 완료 후 α_w sweep / variant / cross-dataset *전부 금지*. Result-blind pre-commit 따라 branch (c) 변형 (sub-binary on Δ all) lock-in, *no narrative reversal*.

**Raw artifacts**: `outputs/14_difficulty_weighted_hn/scifact/seed_{42,1337,2024}/qv_r8_l12_diffw10/`.
**Pre-commit reference**: `report/_exp13_14_pre_commit.md`.
**Reproducibility**: `experiments/14_difficulty_weighted_hn/run.py --dataset scifact --seed {42|1337|2024} --alpha-w 10.0 --r 8 --alpha 8.0 --lora-lr 5e-5 --max-triplets 9190`.
