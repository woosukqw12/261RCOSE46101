# Exp 13 + Exp 14 Pre-Commit (BEFORE training)

**작성 시점**: 2026-05-24 evening (학습 시작 *전*).

**Methodological commitments** (절대 위반 금지):
- Pre-commit single config per experiment (no sweep)
- 3 seeds {42, 1337, 2024} run *together*, no seed-by-seed iteration
- Result-blind: 결과 보기 *전* commit, 결과 후 *수정 금지*
- STOP rule: 결과 무관 single-config single-rule, *no follow-up variant / sweep*
- Negative result 도 *equal honest weight* 보고

---

## Exp 13 = Frozen-Direction Anchor

### Theoretical motivation

§7.3.f.ii NFCorpus M1+M1b *direction matters* puzzle (3-seed robust): eff_rank doc 1.05 ≈ 1.06 (no collapse magnitude change) 임에도 NDCG@10 0.0094 → 0.246 (74 % gap recovery). **Direction alignment** 이 sufficient lever, magnitude 가 아님.

Exp 11 (relational self-sim preservation) = *magnitude-side* preservation. **Exp 13** = *direction-side complement*.

### Hypothesis

Frozen encoder 의 output direction 을 *trust anchor* 로 두고, easy queries 에서 LoRA-output direction 이 frozen 에서 *과도하게 벗어나지* 않도록 cosine-deviation penalty.

### Formulation

$$\mathcal{L} = \mathcal{L}_{\text{margin}}(\text{confused queries}) + \lambda_{\text{dir}} \cdot \frac{1}{|E|}\sum_{x\in E}\bigg[\frac{1}{T_q^x}\sum_{t=1}^{T_q^x}\big(1 - \cos(h_{q,t}^{\text{LoRA}}, h_{q,t}^{\text{frozen}})\big) + \frac{1}{T_d^x}\sum_{t=1}^{T_d^x}\big(1 - \cos(h_{d,t}^{\text{LoRA}}, h_{d,t}^{\text{frozen}})\big)\bigg]$$

- Easy queries (baseline top-1 = relevant) 만 — Exp 11 와 동일 selective scope
- *Per-token* cosine deviation (L2-normed embeddings 이라 cos = dot product)
- Query tokens + pos doc tokens 둘 다

### Pre-commit single config

| Item | Value |
|---|---|
| **λ_dir** | **1.0** (Exp 11 λ=1 과 동일 scale, single value) |
| LoRA | q, v r=8 α=r (Phase 2b 동일) |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early_stop=val_all |
| Dataset | SciFact |
| Seeds | 42, 1337, 2024 |
| Tag | `qv_r8_l12_dir1` |

### 3 branches (pre-committed prediction, result-blind)

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) Direction lever works** | Δ all > +0.025 (3-seed mean) 강력 + Δ easy ≈ 0 (CI 0 포함) | *Direction alignment 가 sufficient lever* 직접 증거. NFCorpus puzzle 의 *mechanism translation* — paper-grade. *§5f.4 의 5th lever* 로 framework 확장. |
| **(b) Comparable to Exp 11** | Δ all ≈ +0.029 ± 0.01 (Exp 11 λ=1 수준), Δ easy ≈ −0.03 (preserved partial) | *Direction-side preservation* 와 *magnitude-side preservation* (Exp 11) 가 *equivalent lever* — relational sim 이 이미 direction 도 implicit 포함. |
| **(c) Worse than Exp 11** | Δ all < +0.020 (Exp 11 λ=1 보다 낮음), Δ confused drop | *Per-token direction* 가 *over-restrictive* — confused lever 도 죽임. Direction-side intervention 의 *natural limit*. |

세 분기 모두 paper-grade 가치 (no result-dependent narrative reversal).

### Mechanism readout (Diagnostic B 적용)

학습 후 easy-doc 의 *direction* 측정:
- Per-token cos(LoRA, frozen) 평균 — should be close to 1.0 if loss working
- Doc-level direction cos — overall direction preservation
- eff_rank (Diagnostic B 와 동일) — magnitude-side 도 변하는지 cross-check

---

## Exp 14 = Difficulty-Aware HN Weighting

### Theoretical motivation

§5f.3 sole sufficient mechanism = *hard-contrast over-correction* (Exp 12 confirmed). Existing interventions:
- **Phase 2b**: hard 100 % (no weight) → catastrophic redistribution
- **M1b**: hard 0 % (in-batch easy) → strict net+ but confused 절반
- **Exp 12**: binary "FN 제거" (hard ≤ threshold removed) → ineffective on redistribution

**Exp 14** = *continuous control* of hard contrast intensity per triplet — *gradient between M1b (binary 0%) and Phase 2b (binary 100%)*.

### Formulation

$$w_i = \sigma(\alpha_w \cdot \text{e5\_margin}_i), \quad \mathcal{L} = \frac{\sum_i w_i \cdot \max(0, m - s_i^+ + s_i^-)}{\sum_i w_i}$$

- e5_margin = cos(eq, epos) − cos(eq, ehn) using cached E5-Mistral embeddings
- Weight ↑ for "real hard" (E5 thinks pos >> hn slightly, margin small positive)
- Weight → 0 for FN (margin negative)
- Weight → 1 for "easy" (margin large positive)

### Pre-commit single config

| Item | Value |
|---|---|
| **α_w** | **10** (sigmoid transition over margin ±0.1, single value) |
| HN source | Mined HN (Phase 2b 동일, *no removal — just weighting*) |
| LoRA | q, v r=8 α=r |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early_stop=val_all max-triplets 9190 |
| Dataset | SciFact |
| Seeds | 42, 1337, 2024 |
| Tag | `qv_r8_l12_diffw10` |

### 3 branches (pre-committed prediction, result-blind)

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) Sweet spot found** | Δ all > +0.025 strict 3/3 + Δ confused > +0.08 + Δ easy > −0.04 | *Continuous control* 이 *binary 보다 better balance*. Hard contrast 의 *optimal intensity* 존재. Paper-grade *practitioner-actionable* finding. |
| **(b) Comparable to M1b or Exp 11 λ=1** | Δ all ≈ +0.02-0.03 (M1b 또는 Exp 11 λ=1 수준), Δ confused 중간 | *Continuous vs binary 가 equivalent* — soft sigmoid 가 effective hard threshold 만큼만 행동. Sweet spot 존재 안 함, *spectrum endpoints 만 differentiable*. |
| **(c) Worse than both** | Δ all < +0.015 또는 Δ confused 절반 미만 | Continuous weighting 이 *learning signal 약화* — α=10 의 sigmoid transition 이 *모든 triplet 의 weight 흐릿하게* 만들어 train signal 부족. |

### Mechanism readout

학습 후:
- Effective HN intensity = mean weight 분포
- Collapse magnitude (eff_rank) — Phase 2b (~1.1) ↔ M1b (~7.2) 의 *continuous* 어디 위치?
- Δ all / confused / easy trade-off shape

---

## Combined STOP rule

**각 실험 3 seeds 완료 후 결과 무관 STOP**:
- λ_dir / α_w 의 다른 값 sweep *금지* (e.g., λ_dir=5 또는 α_w=20 추가 시도 금지)
- Cross-dataset 확장 *금지* (이 또한 post-hoc 일반화 위험)
- Future work proposal 로 *글로만* 가능 — *추가 실행 X*

만약 (a) branch 가 나와도 *post-hoc trail 재진입 금지* — 결과 *honest report* + paper write.

---

**Commit timestamp**: 2026-05-24 evening.
**Training start**: 즉시 (code 완성 후).
**Result reveal**: training complete 후 honest analysis.
