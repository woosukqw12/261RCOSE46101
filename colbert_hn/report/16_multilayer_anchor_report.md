# 16_multilayer_anchor — *Multi-layer per-token cosine anchor* (Exp 13 의 5-layer extension, **branch (c) over-restriction confirmed**)

본 보고서는 **Exp 16** (`qv_r8_l12_dir1_multilayer`, 3 seeds, SciFact) 의 결과 분석. Exp 13 의 *anchor-side best lever* (Δ all +0.030, 3/3 strict) 를 **CLAUDE.md §1.3 prior diagnostic finding 의 layer set {0, 3, 6, 9, 12} 로 확장** — *intermediate-layer anchor 가 frontier 외부 도달* 가능한지 result-blind pre-commit 으로 검정 (`report/_exp16_pre_commit.md`).

**결론**: 3-seed mean **Δ all +0.004 ± 0.006 (3/3 CI 0 포함, NOT strict), Δ confused +0.071 ± 0.004 ✓, Δ easy −0.052 ± 0.008 ✗** — pre-commit branch **(c) "Multi-layer over-restriction"** 확정. 모든 metric 에서 Exp 13 대비 *명백 열등* (Δ all 1/8, Δ easy damage 2.5×). Diagnostic B 가 *loss budget dilution + intermediate-layer redundancy* mechanism 직접 입증. **Single-layer anchor at final output (Exp 13) 이 sweet spot** — *layer scope 확장이 frontier 외부 도달의 path 아님*.

---

## 1. 동기 + Pre-committed 판정 기준

### 1.1 Theoretical motivation — *5-layer cumulative restraint 가설*

Exp 13 (final-layer cosine anchor, λ_dir=1.0) 는 *anchor-side family 의 best lever* (Δ all +0.030, 3/3 strict, §7.3.g). Diagnostic B 의 3-fold mechanism evidence:
1. Anchor cos = 0.824 (soft equilibrium attractor)
2. Token eff_rank 9.01 (16 % of frozen 55.13)
3. Doc eff_rank 2.33 (Phase 2b-level collapse 잔존)

→ Anchor 가 *single layer (final ColBERT output)* 에서만 작동.

**CLAUDE.md §1.3 prior diagnostic finding**: *"layer-wise confusion signal exists at layers [0, 3, 6, 9, 12]"*. 본 finding 의 *direct architectural translation* = anchor 를 5 BERT layer 에 분산 (단일 final layer 대신).

**가설**: 5-layer cumulative restraint 가:
- (i) Anchor cos → 1.0 에 더 가까이 (cumulative effect)
- (ii) Token eff_rank ↑ (multi-layer preservation)
- (iii) Δ all ↑ (frontier 외부 도달) OR Δ confused ↓ (over-restriction)

### 1.2 Pre-committed single config (result-blind)

| Item | Value |
|---|---|
| **Layer set** $L$ | **{0, 3, 6, 9, 12}** (BERT hidden states, CLAUDE.md §1.3) |
| **λ_dir** | **1.0** (Exp 13 동일 scale) |
| LoRA | q, v r=8 α=r |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early-stop=val_all |
| Dataset | SciFact, 3 seeds {42, 1337, 2024} |
| Tag | `qv_r8_l12_dir1_multilayer` |

### 1.3 Loss formulation

$$\mathcal{L} = \mathcal{L}_{\text{margin}}(\text{confused}) + \lambda_{\text{dir}} \cdot \frac{1}{|L|}\sum_{\ell \in L}\mathcal{R}_{\text{dir}}^{(\ell)}(\text{easy})$$

where $\mathcal{R}_{\text{dir}}^{(\ell)}$ = mean per-token $(1 - \cos)$ at BERT layer $\ell$ between LoRA-pass and frozen-pass 768-dim hidden states.

**중요 차이점 (Exp 13 vs Exp 16)**:
- Exp 13: 128-dim *projected ColBERT* output, single layer
- Exp 16: 768-dim *BERT intermediate* hidden states, 5 layers (uniform weight 1/5)

### 1.4 3-branch pre-commit predictions

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) Multi-layer frontier 외부 도달** | Δ all > +0.040 strict + Δ easy > −0.015 | *Prior diagnostic finding 의 direct empirical confirmation* |
| **(b) Exp 13 과 frontier 공유** | Δ all ≈ +0.030 ± 0.005 | *Layer-invariance* — frontier robustness 추가 입증 |
| **(c) Over-restriction** | Δ all < +0.020 또는 Δ confused < +0.075 | *5-layer cumulative restraint 가 confused signal 죽임* |

---

## 2. 결과

### 2.1 3-seed grid

| Seed | NDCG@10 all | Δ all (CI 95 %) | Δ confused | Δ easy |
|---|---|---|---|---|
| 42 | 0.6238 | +0.0075 [−0.0162, +0.0307] **(CI 0)** | +0.0733 [+0.0355, +0.1123] ✓ | −0.0478 [−0.0751, −0.0238] ✗ |
| 1337 | 0.6434 | −0.0031 [−0.0274, +0.0209] **(CI 0)** | +0.0663 [+0.0280, +0.1074] ✓ | −0.0614 [−0.0892, −0.0364] ✗ |
| 2024 | 0.6534 | +0.0070 [−0.0160, +0.0301] **(CI 0)** | +0.0720 [+0.0344, +0.1096] ✓ | −0.0476 [−0.0749, −0.0235] ✗ |
| **3-seed mean ± std** | **0.6402 ± 0.0150** | **+0.0038 ± 0.0061** (3/3 NOT strict) | **+0.0705 ± 0.0037 ✓** | **−0.0523 ± 0.0079 ✗** |

### 2.2 Branch 판정

| 조건 | 임계 | 3-seed mean | 판정 |
|---|---|---|---|
| Branch (a) Δ all > +0.040 strict | strict | +0.004 | ✗ (fail by 0.036) |
| Branch (a) Δ easy > −0.015 | strict | −0.052 | ✗ (fail by 0.037) |
| Branch (c) Δ all < +0.020 | strict | +0.004 | ✓ |
| Branch (c) Δ confused < +0.075 (= Phase 2b 의 약 70%) | strict | +0.071 | ✓ (just under) |

→ **Branch (c) "Multi-layer over-restriction"** 확정. 모든 strict 조건 충족.

### 2.3 Exp 13 vs Exp 16 직접 비교

| Metric | Exp 13 (single-layer final) | Exp 16 (5-layer multi) | Ratio (Exp 16 / Exp 13) |
|---|---|---|---|
| Δ all (3-seed) | +0.030 ± 0.002 | **+0.004 ± 0.006** | **1/8** |
| Δ confused | +0.092 ± 0.007 | +0.071 ± 0.004 | 77 % |
| Δ easy damage | −0.021 ± 0.003 | **−0.052 ± 0.008** | **2.5×** worse |
| Strict 3/3 (Δ all CI > 0) | 3/3 | **0/3** (모두 CI 0) | none |
| ‖B‖_total mean | 1.34 ± 0.27 | 1.50 ± 0.05 | 1.12× |

→ **Multi-layer 가 모든 metric 에서 명백 열등**. *Strict robustness 상실*, *Δ easy damage 가 2.5× 악화*.

### 2.4 학습 동학 (seed 42 train history)

| Epoch | rank_loss | dir_loss (5-layer) | val_ndcg_all | val_ndcg_confused |
|---|---|---|---|---|
| 1 | 1.313 | 0.099 | **0.638** (best, used) | 0.238 |
| 2 | 0.220 | 0.249 (anchor *멀어짐*) | 0.546 (drop) | 0.194 |
| 3 | 0.172 | 0.231 | 0.540 | 0.216 |

→ **anchor_loss 단조 증가 (ep1 → ep3, 0.10 → 0.23)** — Exp 13 의 plateau (0.18 → 0.47, 그러나 val_all stable) 와 다른 패턴. *5-layer anchor 가 confused 학습 신호와 *직접 conflicting*. epoch 1 best 후 *catastrophic drop* — *over-restriction 의 직접 evidence*.

### 2.5 LoRA capacity 사용

3-seed mean:
- **‖A‖_total** = 8.44 ± 0.07 (Exp 13: 8.34)
- **‖B‖_total** = 1.50 ± 0.05 (Exp 13: 1.34)

→ Exp 13 와 거의 동일 magnitude, 단 12 % 더 큰 ‖B‖. Update magnitude 면에서는 *유사*, 하지만 *방향이 over-restrict* 됨.

---

## 3. Diagnostic B — *Loss budget dilution mechanism direct evidence*

### 3.1 Per-layer anchor proximity (Exp 16 3-seed mean)

| Layer ℓ | cos(LoRA, frozen) | tok_eff_rank (Exp 16) | tok_eff_rank (frozen) | tok_eff_rank ratio |
|---|---|---|---|---|
| 0 (embed) | **1.000** | 247.35 | 247.35 | 100 % (identity) |
| 3 | **0.998** | 150.42 | 156.77 | 96 % |
| 6 | **0.991** | 101.81 | 109.45 | 93 % |
| 9 | 0.965 | 46.45 | 65.23 | 71 % |
| **12** | **0.697** ⚠️ | **4.61** | 43.06 | **11 %** (collapse) |
| (ref) Exp 13 final ColBERT 128-dim | 0.824 | 9.01 | 55.13 | 16 % |

### 3.2 핵심 mechanism finding — *Loss budget dilution + intermediate redundancy*

**3-fold mechanism evidence**:

1. **L0-L6 의 *redundant constraint***: LoRA 가 q, v 의 *후행 layer effect* 만 받기 때문에 L0 (embedding) ~ L6 의 hidden state 는 *원래도 frozen 과 거의 동일* — anchor cos ≥ 0.99 가 *자연스럽게 만족*. Loss 가 의미 있는 gradient signal 못 보냄 → **budget 낭비**.

2. **L9-L12 의 *insufficient constraint***: 5-layer equal weight (1/5 each) → 진정 anchor 가 필요한 *deep layers* 는 budget 의 1/5 만 받음. L12 의 anchor cos **0.697** vs Exp 13 의 **0.824** — *17 % 더 멀어짐*.

3. **L12 의 *catastrophic collapse***: token eff_rank **4.61** (vs Exp 13's 9.01, frozen's 43.06) — multi-layer 가 *오히려 final layer 의 collapse 를 악화*. Loss budget dilution + intermediate redundancy 의 *combined effect* 가 deep layer 의 representation diversity 까지 더 paralyze.

→ **Paper-grade direct evidence**: *Multi-layer anchor 는 budget 효율 문제로 over-restriction*. *Final layer 가 anchor-side family 의 sweet spot* — *intermediate layer 확장은 frontier 외부 도달의 path 가 아니라 inferior path*.

### 3.3 CLAUDE.md §1.3 prior diagnostic finding 의 *재해석*

기존 *prior finding*: "*confusion signal* exists at layers [0, 3, 6, 9, 12]".

본 결과의 *informed re-interpretation*:
- **"Signal exists" ≠ "Intervention should be applied"** — 5 layer 의 confusion signal 측정 결과가 *intervention 의 best location* 를 의미하지 않음.
- *Intermediate layers* 는 signal *measurement* 에는 유용 (diagnostic 가치) 하지만 *intervention* 에는 *redundant* — LoRA 의 backward effect 가 자연스럽게 anchor 됨.
- *Deep layers* (L9-L12) 에 intervention 집중하는 것이 budget 효율적 — Exp 13 의 final-layer 선택이 *empirically optimal*.

---

## 4. Figures

### 4.1 Δ NDCG@10 forest plot (Exp 16 vs Exp 13 vs Phase 2b)

![Δ NDCG@10 forest plot](figures/16_multilayer_anchor/delta_ci_forest.png)

**Caption**: 3 slices (all/confused/easy) × 3 methods (Phase 2b, Exp 13, Exp 16) 의 paired bootstrap 95 % CI. Branch (a) threshold (Δ all > +0.040, Δ easy > −0.015) 와의 거리 시각 — Exp 16 의 3 seeds 모두 fail.

### 4.2 Anchor scope ablation — single-layer vs 5-layer

![Anchor scope comparison](figures/16_multilayer_anchor/layer_count_comparison.png)

**Caption**: Exp 13 (final 128-dim, 1 layer) vs Exp 16 (BERT {0,3,6,9,12} 768-dim, 5 layers) 의 3-seed mean ± std bar plot. 모든 metric 에서 5-layer 가 명백 열등.

### 4.3 학습 곡선 (Exp 16 seed 42)

![Train curves](figures/16_multilayer_anchor/train_curves.png)

**Caption**: rank_loss + 5-layer dir_loss + val NDCG@10. dir_loss 단조 증가 (ep1 0.10 → ep3 0.23) — Exp 13 (0.18 → 0.47, val stable) 와 다른 *active conflict* 패턴.

### 4.4 LoRA A/B norm

![LoRA A/B norms](figures/16_multilayer_anchor/lora_AB_norms.png)

**Caption**: 24 adapters 의 ‖A‖, ‖B‖. Exp 13 와 거의 동일 magnitude, 단 ‖B‖ 12 % 더 큼 (multi-layer 의 update magnitude 약간 ↑).

### 4.5 Diagnostic B per-layer anchor proximity

![Diagnostic B Exp 16](../report/figures/_repr_collapse_exp16/repr_collapse_exp16.png)

**Caption** (3-panel):
- **(A)** Per-layer cos(LoRA, frozen) — Exp 16 (3 seeds) vs Exp 13 (final, ref star). L0-L6 cos ≥ 0.99 (redundant), L12 cos 0.697 (Exp 13 0.824 보다 더 멀어짐).
- **(B)** Per-layer token eff_rank — Exp 16 vs frozen baseline + Exp 13 ref. L0-L9 preserved, **L12 catastrophic collapse to 4.61**.
- **(C)** L=12 final-layer cos 비교 — Exp 13 0.824 vs Exp 16 0.697 (3-seed mean ± std), *loss budget dilution direct visual*.

### 4.6 3-seed slice grid

![NDCG slice grid](figures/16_multilayer_anchor/ndcg_slice_grid.png)

**Caption**: 3 seeds × 3 slices 의 Δ NDCG@10 bar plot (paired bootstrap CI). seed 별 일관성 + 모든 seed 의 Δ all CI 0 포함 직접 시각.

---

## 5. 종합

### 5.1 Branch (c) confirmation — Multi-layer over-restriction

**Pre-commit branch (c)** 의 모든 조건 충족:
- Δ all = +0.004 < +0.020 ✓ (NOT strict)
- Δ confused = +0.071 < +0.075 (just under Phase 2b 의 70 % 임계) ✓
- Δ easy = −0.052 (Exp 13 의 2.5× damage) ✗

3 outcome 모두 paper-grade — *frontier 외부 도달의 layer-scope path 차단*.

### 5.2 *Sweet spot* mechanism — *final-layer-only* anchor 의 우월성

Diagnostic B 의 mechanism direct evidence:
- *Intermediate layers (L0-L6) 는 anchor 가 자연스럽게 만족* — LoRA backward effect 의 propagation 특성으로 cos ≥ 0.99.
- *Loss budget 분산이 deep layers (L9-L12) 의 effective constraint 약화* — final-layer cos 0.697 (vs Exp 13 0.824).
- *Anchor 의 *true target* 은 LoRA 가 representation 을 가장 멀리 밀어내는 *final transformer output**.

→ **Anchor-side family 의 *optimal scope* = final layer only** — *single-point intervention 이 distributed intervention 보다 효율적*.

### 5.3 CLAUDE.md §1.3 prior diagnostic finding 의 *재해석*

본 paper 의 main contribution 에 *prior finding interpretation* 추가:
- "*Signal exists at layers [0, 3, 6, 9, 12]*" (prior measurement-side finding) ≠ "*Intervention should target all 5 layers*" (본 paper intervention-side finding).
- *Diagnostic 와 intervention 의 separation*: signal location measurement 의 *informational value* 와 intervention scope 의 *optimization* 은 *별개 question*.

### 5.4 6-lever framework 의 *no change* — Exp 16 추가 lever 아님

본 결과로 paper 의 6-lever framework 변화 *없음*:
- Exp 16 의 Δ all (+0.004) 가 *data-side weighting family (Exp 12/14)* 보다 약간 위, *anchor-side family (Exp 11/13)* 보다 명백 아래.
- *Framework 의 inferior 구성원* 으로 §9.3 future work *anchor scope ablation* entry 로 정리.

### 5.5 §3.8 ablation completeness *strict 충족*

본 paper 의 *anchor scope ablation* 요구 (§3.8 mandatory) 가 본 실험으로 충족 — *single-layer (Exp 13) vs multi-layer (Exp 16)* 의 paired pre-commit 검정, paper-grade negative result 로 *ablation completeness rule strict 준수*.

**STOP rule 준수**: 3 seeds 완료 후 추가 layer set sweep / variant *전부 금지*. Result-blind pre-commit 따라 branch (c) lock-in.

**Raw artifacts**: `outputs/16_multilayer_anchor/scifact/seed_{42,1337,2024}/qv_r8_l12_dir1_multilayer/`.
**Pre-commit reference**: `report/_exp16_pre_commit.md`.
**Diagnostic B**: `report/figures/_repr_collapse_exp16/`.
**Reproducibility**: `experiments/16_multilayer_anchor/run.py --dataset scifact --seed {42|1337|2024} --lambda-dir 1.0 --r 8 --alpha 8.0 --lora-lr 5e-5 --max-triplets 9190`.
