# Exp 16 Pre-Commit (BEFORE training)

**작성 시점**: 2026-05-25 (학습 시작 *전*).

**Methodological commitments** (절대 위반 금지):
- Pre-commit single config (no sweep)
- 3 seeds {42, 1337, 2024} run *together*, no seed-by-seed iteration
- Result-blind: 결과 보기 *전* commit, 결과 후 *수정 금지*
- STOP rule: 결과 무관 single-config single-rule, *no follow-up variant / sweep*
- Negative result 도 *equal honest weight* 보고

---

## Exp 16 = Multi-Layer Per-Token Direction Anchor

### Theoretical motivation

Exp 13 (final-layer only per-token cosine anchor) 는 *anchor-side family 의 best lever* (Δ all +0.030, 3/3 strict). 그러나 `report/_repr_collapse_exp13.py` 의 Diagnostic B 가 *3-fold mechanism evidence* 발견:
1. Anchor cos = 0.824 (잔여 0.176, *soft equilibrium attractor*)
2. Token eff_rank 9.01 (frozen 55.13 의 16 % 만 회복)
3. Doc eff_rank 2.33 (Phase 2b-level collapse 잔존)

**CLAUDE.md §1.3 prior diagnostic study finding**: *"layer-wise confusion signal exists at layers [0, 3, 6, 9, 12]"*. 현 Exp 13 은 layer 12 (=BERT 의 마지막 transformer output) **단일 지점** 만 anchor — *5 layer 의 prior finding 을 활용 안 함*. CLAUDE.md §3.8 (ablation completeness 철칙) 이 *명시적으로* 본 ablation 을 요구.

### Hypothesis

5 layer 의 *multi-layer* anchor 가:
- (i) Anchor cos → 1.0 에 더 가까이 (equilibrium 강화) — *intermediate layer 별 soft attractor 가 *cumulative restraint* 작용*
- (ii) Token eff_rank 가 9.01 보다 ↑ (frozen 에 더 가까운 internal representation)
- (iii) Δ all 이 +0.030 보다 ↑ (frontier 외부 도달) OR Δ confused 감소 (over-restriction)

세 outcome 모두 *paper-grade direct evidence*.

### Formulation

$$\mathcal{L} = \mathcal{L}_{\text{margin}}(\text{confused queries}) + \lambda_{\text{dir}} \cdot \frac{1}{|L|} \sum_{\ell \in L} \mathcal{R}_{\text{dir}}^{(\ell)}(\text{easy queries})$$

where $L = \{0, 3, 6, 9, 12\}$ (5 layers, CLAUDE.md §1.3 prior finding), and

$$\mathcal{R}_{\text{dir}}^{(\ell)} = \frac{1}{|E|}\sum_{x \in E}\bigg[\frac{1}{T_q^x}\sum_{t=1}^{T_q^x}\big(1 - \cos(h_{q,t,\ell}^{\text{LoRA}}, h_{q,t,\ell}^{\text{frozen}})\big) + \frac{1}{T_d^x}\sum_{t=1}^{T_d^x}\big(1 - \cos(h_{d,t,\ell}^{\text{LoRA}}, h_{d,t,\ell}^{\text{frozen}})\big)\bigg]$$

- $h_{*,*,\ell}^{\text{LoRA}/\text{frozen}}$ = BERT layer $\ell$ 의 hidden state (768-dim, L2-normed)
- $\ell = 0$ → embedding output (before any transformer)
- $\ell = k$ for $k \in \{3, 6, 9, 12\}$ → output of $k$-th transformer layer
- Uniform weighting across 5 layers (no per-layer weight)

**중요 차이점 (Exp 13 vs Exp 16)**:
- Exp 13: 128-dim projected ColBERT output (final), single layer
- Exp 16: 768-dim BERT hidden states, 5 layers (intermediate + final BERT layer 12)

### Pre-commit single config

| Item | Value |
|---|---|
| **Layer set** $L$ | **{0, 3, 6, 9, 12}** (CLAUDE.md §1.3 prior diagnostic, *no sweep*) |
| **λ_dir** | **1.0** (Exp 13 동일 scale, single value) |
| LoRA | q, v r=8 α=r (Phase 2b 동일) |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early_stop=val_all |
| Triplets | mined HN, n_hns_per_q=10, pool=100, cap=9190 |
| Dataset | SciFact |
| Seeds | 42, 1337, 2024 (run together) |
| Tag | `qv_r8_l12_dir1_multilayer` |

### 3-branch pre-commit predictions (result-blind)

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) Multi-layer 이 frontier 외부 도달** | Δ all > +0.040 strict 3/3 + Δ easy > −0.015 + Δ confused > +0.085 | *Intermediate layer anchor 가 single-layer 보다 substantively 우월* — CLAUDE.md §1.3 prior diagnostic finding 의 *direct empirical confirmation*, paper main contribution **+1 upgrade** |
| **(b) Exp 13 과 frontier 공유** | Δ all ≈ +0.030 ± 0.005, Δ easy ≈ −0.020 ± 0.005 | *Layer-invariance* — anchor-side family frontier 의 *layer scope 에 robust*, 6-lever framework 의 frontier-fixed 추가 입증 (negative result paper-grade) |
| **(c) Multi-layer over-restriction** | Δ all < +0.020 또는 Δ confused < +0.075 | *Intermediate layer anchor 가 confused 학습 신호 죽임* — 5-layer cumulative restraint 의 *over-constraint mechanism direct evidence* |

세 분기 모두 paper-grade 가치 (no result-dependent narrative reversal).

### Diagnostic B (post-hoc) — sub-experiment

학습 후 동일 측정 (Exp 13 의 measurement-only sub-experiment 와 paired):
- Per-token cos(LoRA, frozen) at each of 5 layers — *anchor proximity at each layer*
- Doc / tok eff_rank — Exp 13 (9.01 / 2.33) 과 비교
- *Soft equilibrium 의 layer-wise pattern* 직접 측정

### Computational cost

| Phase | 시간 (M1, ~) |
|---|---|
| Frozen cache (5 layers × ~441 easy pairs × ~230 tokens × 768 × float16) | ~3 min |
| Training (3 seeds × 3 epochs × ~10 min) | ~30 min |
| Eval + Diagnostic B (3 seeds, ~3 min each) | ~10 min |
| **Total** | **~45 min** |

### STOP rule

**3 seeds 완료 후 결과 무관 STOP**:
- λ_dir 의 다른 값 sweep *금지*
- Layer set 변경 *금지* (e.g., {0, 6, 12} 또는 dense {0..12} 추가 시도 금지)
- Cross-dataset 확장 *금지*
- Future work proposal 로 *글로만* 가능

만약 (a) branch 가 나와도 *post-hoc trail 재진입 금지* — 결과 *honest report* + paper write.

---

**Commit timestamp**: 2026-05-25.
**Training start**: 즉시 (code 완성 후).
**Result reveal**: training complete 후 honest analysis.
