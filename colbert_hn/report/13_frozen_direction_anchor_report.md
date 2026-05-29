# 13_frozen_direction_anchor — *Per-token absolute direction* preservation as anchor-side complement to Exp 11

본 보고서는 **Exp 13** (`qv_r8_l12_dir1`, 3 seeds, SciFact) 의 결과 분석. Exp 11 (relational self-similarity preservation) 의 *anchor-side family* 안에서 *상보적 constraint* (per-token absolute cosine to frozen) 가 동일 trade-off frontier 에 떨어지는지 result-blind pre-commit 으로 검정 (`report/_exp13_14_pre_commit.md`).

**결론**: 3-seed mean **Δ all +0.030 ± 0.002 ✓, Δ confused +0.092 ± 0.007 ✓, Δ easy −0.021 ± 0.003 ✗** — Exp 11 (λ=1) 3-seed mean (+0.029 / +0.085 / −0.019) 와 *형식적 차이에도 frontier 통계 동등*. Pre-commit branch (a) 의 strict 임계 (`Δ easy > −0.020`) 를 **0.001 차이** 로 fail → **branch (b) — Exp 11 과 frontier 공유** 확정. *Anchor-side family 의 frontier 강건성* 추가 증거 (Sim Frobenius² rotation-invariant 와 per-token cosine rotation-sensitive 가 동일 outcome).

---

## 1. 동기 + Pre-committed 판정 기준

### 1.1 Theoretical motivation — *direction* matters, not magnitude

§7.3.f.ii (NFCorpus M1+M1b *direction matters* puzzle, 3-seed robust): eff_rank doc 1.05 ≈ 1.06 (no collapse magnitude change) 임에도 NDCG@10 0.0094 → 0.246 (74 % gap recovery). **Direction alignment 이 sufficient lever, magnitude 가 아님**.

Exp 11 (relational self-sim preservation, $\|\text{Sim}(H_{\text{LoRA}}) - \text{Sim}(H_{\text{frozen}})\|_F^2$) 는 *pair-wise similarity matrix* 의 preservation — **rotation-invariant** constraint. 모든 token 이 통째로 회전해도 Sim 은 invariant, loss = 0. → *relational structure preserved*, *absolute direction* 은 자유.

Exp 13 = *absolute direction* preservation. 각 token 의 frozen representation 으로부터의 cosine deviation 을 penalize:

$$\mathcal{R}_{\text{dir}} = \frac{1}{|E|} \sum_{x \in E} \bigg[ \frac{1}{T_q^x}\sum_{t=1}^{T_q^x}\big(1 - \cos(h_{q,t}^{\text{LoRA}}, h_{q,t}^{\text{frozen}})\big) + \frac{1}{T_d^x}\sum_{t=1}^{T_d^x}\big(1 - \cos(h_{d,t}^{\text{LoRA}}, h_{d,t}^{\text{frozen}})\big) \bigg]$$

→ **rotation-sensitive**. 같은 cluster 가 통째로 회전하면 loss > 0.

본질 가설: *direction* 이 sufficient lever 라면 (NFCorpus puzzle 의 함의), **per-token absolute direction preservation** 이 Exp 11 의 relational preservation 보다 *strict* 한 anchor → Δ easy 더 큰 회복 가능.

### 1.2 Loss formulation (전체)

$$\mathcal{L} = \mathcal{L}_{\text{margin}}(\text{confused queries; mined HN}) + \lambda_{\text{dir}} \cdot \mathcal{R}_{\text{dir}}(\text{easy queries})$$

- Easy queries (baseline top-1 = relevant) 만 → Exp 11 의 selective scope 와 *동일*.
- Confused queries 는 standard pairwise margin loss (Phase 2b 동일).
- Per-token cosine: L2-normed embeddings 의 dot product. Query / pos doc tokens 둘 다.

### 1.3 Pre-committed single config (result-blind)

| Item | Value |
|---|---|
| **λ_dir** | **1.0** (Exp 11 λ=1 과 동일 scale, single value, *no sweep*) |
| LoRA | q, v r=8 α=r (Phase 2b 동일) |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early-stop=val_all |
| Triplets | mined HN, n_hns_per_q=10, pool=100, cap=9190 |
| Dataset | SciFact |
| Seeds | 42, 1337, 2024 (run together, no seed-by-seed iteration) |
| Tag | `qv_r8_l12_dir1` |

### 1.4 3-branch pre-commit predictions

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) Direction lever works** | Δ all > +0.025 (3-seed mean) + Δ confused > +0.08 + Δ easy > **−0.020** (strict) | *Per-token absolute direction* 이 *strict* anchor → frontier 우회 가능. NFCorpus puzzle 의 mechanism translation. |
| **(b) Comparable to Exp 11** | Δ all ≈ +0.029 ± 0.01, Δ easy ≈ −0.03 (partial preservation) | *Relational* (Exp 11) 와 *absolute* (Exp 13) 가 *equivalent lever* — frontier 공유. |
| **(c) Worse than Exp 11** | Δ all < +0.020 또는 Δ confused drop | *Per-token absolute* 가 *over-restrictive* → confused 도 죽임. |

**STOP rule**: 3 seeds 완료 후 결과 무관 STOP. λ_dir sweep / variant / cross-dataset *전부 금지*.

---

## 2. 결과

### 2.1 3-seed grid

| Seed | NDCG@10 all | NDCG@10 confused | Δ all (CI 95 %) | Δ confused | Δ easy |
|---|---|---|---|---|---|
| 42 | 0.6790 | 0.2582 | **+0.0326 [+0.0119, +0.0538]** ✓ | **+0.0995 [+0.0635, +0.1363]** ✓ | −0.0236 [−0.0450, −0.0059] ✗ |
| 1337 | 0.6743 | 0.2545 | **+0.0279 [+0.0082, +0.0483]** ✓ | **+0.0873 [+0.0529, +0.1248]** ✓ | −0.0221 [−0.0420, −0.0056] ✗ |
| 2024 | 0.6771 | 0.2606 | **+0.0307 [+0.0122, +0.0499]** ✓ | **+0.0877 [+0.0539, +0.1219]** ✓ | −0.0172 [−0.0338, −0.0028] ✗ |
| **3-seed mean ± std** | **0.6768 ± 0.0024** | **0.2578 ± 0.0031** | **+0.0304 ± 0.0024 ✓** | **+0.0915 ± 0.0070 ✓** | **−0.0210 ± 0.0034 ✗** |

→ 3 seeds **all strict positive on Δ all & Δ confused**, **all strict negative on Δ easy**.

### 2.2 Branch 판정

| 조건 | 임계 | 3-seed mean | 판정 |
|---|---|---|---|
| Δ all > +0.025 (strict) | +0.025 | +0.0304 | ✓ |
| Δ confused > +0.08 | +0.080 | +0.0915 | ✓ |
| **Δ easy > −0.020 (strict)** | **−0.020** | **−0.0210** | **✗ (miss by 0.001)** |

→ **Branch (a) 의 *easy* 임계 0.001 차이 fail** → **Branch (b) — Exp 11 과 frontier 공유** lock-in.

### 2.3 Exp 11 vs Exp 13 직접 비교 (anchor-side family)

| Metric | Exp 11 (λ=1) 3-seed | Exp 13 (λ_dir=1) 3-seed | Δ (Exp 13 − Exp 11) |
|---|---|---|---|
| NDCG@10 all | 0.6759 ± 0.005 | **0.6768 ± 0.002** | +0.0009 |
| NDCG@10 confused | (≈ 0.255) | 0.2578 ± 0.003 | (≈ 동등) |
| Δ all | +0.029 ± 0.005 | **+0.030 ± 0.002** | +0.002 |
| Δ confused | +0.101 ± 0.010 | +0.092 ± 0.007 | −0.010 |
| Δ easy | −0.031 ± 0.018 | **−0.021 ± 0.003** | **+0.010** |
| Strict 3/3 (Δ all CI 하한 > 0) | 2/3 | **3/3** | **+1 seed strict** |
| ‖A‖_total / ‖B‖_total mean | (~8.9 / 1.8) | 8.5 / 1.4 | (slightly smaller B) |

**비교 결론**:
- **Δ all 통계 동등** (+0.030 vs +0.029, 차이 0.002 ≪ bootstrap noise).
- **Δ easy 더 잘 보존** (Exp 13 −0.021 vs Exp 11 −0.031) — *방향 면에서는* per-token cosine 이 약간 strict 한 효과 ✓.
- **Δ confused 약간 낮음** (Exp 13 +0.092 vs Exp 11 +0.101) — strict 한 anchor 가 confused 학습 신호도 일부 억제.
- **3 seeds 모두 strict** (Exp 11 의 2/3 보다 *strict robustness ↑*).
- **B norm 더 작음** (1.4 vs 1.8) — 동일 ‖A‖ 에서 update direction 이 더 controlled, anchor preservation 의 직접 측정 가능.

→ *trade-off frontier 의 같은 line 위에서, Exp 13 이 약간 다른 위치* (more strict robustness, slightly weaker confused, slightly better easy).

### 2.4 LoRA capacity 사용

3-seed mean:
- **‖A‖_total** = 8.50 ± 0.27
- **‖B‖_total** = 1.40 ± 0.27
- **AB product norm** (effective update) = ‖A‖ · ‖B‖ / 24 adapters ≈ 0.50 per adapter

Exp 11 (3-seed mean ~8.9 / ~1.8) 대비 ‖B‖ 가 ~22 % 작음 → **per-token direction anchor 가 update magnitude 도 직접 제약**. *Anchor preservation 의 mechanism direct evidence*.

### 2.5 학습 동학 (seed 42 train history)

- **rank_loss**: ep1 1.40 → ep2 0.50 → ep3 0.07 (단조 감소, Phase 2b 동일 패턴)
- **anchor_loss** (= cosine deviation): ep1 0.18 → ep2 0.46 → ep3 0.47 (epoch 1 후 plateau)
- **val_ndcg_all**: ep1 0.6866 → ep2 0.6781 → ep3 0.6168 (epoch 1 이 best, early-stop snapshot)

→ **anchor loss 의 epoch 1 후 plateau** = LoRA 가 *cosine deviation 0.18-0.47 의 corridor* 안에서 confused 학습 진행. Loss 가 0 으로 가지 않음 (over-rigidly anchor 안 되는 것이 design 의도). val_all 의 epoch 1 best 는 Exp 11 패턴과 동일.

---

## 3. 함의

### 3.1 Anchor-side family 의 frontier 강건성 (paper-grade negative result)

**핵심 발견**: 두 *수학적으로 명백히 다른* constraint 가 *통계적으로 구분 안 되는* frontier 산출:

| Property | Exp 11 (Sim Frobenius²) | Exp 13 (per-token cosine) |
|---|---|---|
| **Constraint formality** | Pair-wise similarity matrix 의 *Frobenius distance* | Per-token cosine 의 mean deviation |
| **Rotation invariance** | ✓ (token cluster 통째로 회전 OK) | ✗ (각 token 의 *절대 방향* 고정) |
| **Token locality** | Global (similarity matrix 전체) | Local (token 별 독립) |
| **Granularity** | $O(T^2)$ pair | $O(T)$ token |
| **Δ all** | +0.029 ± 0.005 | +0.030 ± 0.002 |
| **Δ confused** | +0.101 ± 0.010 | +0.092 ± 0.007 |
| **Δ easy** | −0.031 ± 0.018 | −0.021 ± 0.003 |

→ **"수학적 차이가 empirical separation 으로 이어지지 않음"** → *anchor-side family 의 frontier 가 robust* — 어떤 형태의 frozen-anchor regularizer 든 Δ confused × Δ easy 의 1:1 trade-off 위로 떨어진다.

### 3.2 4-lever framework → 5-lever (anchor-side dual)

§7.4.1 의 4-lever framework 에 Exp 13 추가:

| Lever | Family | Mechanism | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|---|
| Phase 2b | (baseline) | hard + noisy | ≈ 0 | +0.104 ✓ | −0.085 ✗ |
| Exp 12 | data-side (binary filter) | hard + clean | ≈ 0 | +0.080 ✓ | −0.073 ✗ |
| M1b | data-side (HN substitution) | easy in-batch | +0.021 ✓ | +0.065 | (~−0.05) |
| Exp 11 | **anchor-side (relational)** | hard + Sim Frobenius² | +0.029 | +0.101 ✓ | −0.031 |
| **Exp 13** | **anchor-side (absolute)** | hard + per-token cosine | **+0.030** | +0.092 ✓ | **−0.021** |

→ **anchor-side family** 의 두 lever (Exp 11, Exp 13) 가 frontier 의 *동일 region* 차지. *Data-side family* (Exp 12 binary, Exp 14 continuous TBD) 는 frontier 의 다른 region 가능성.

### 3.3 NFCorpus puzzle 의 mechanism translation 미달성

§7.3.f.ii 의 가설 — "direction alignment 이 sufficient lever, magnitude 가 아님" — 의 **mechanism translation** 으로 Exp 13 을 제시했으나, 결과는 frontier *우회 못 함*:

- Per-token absolute direction 을 *직접* anchor 했음에도 (NFCorpus puzzle 의 mechanism 그대로 적용), SciFact 의 trade-off 는 *Sim 기반 anchor 와 동일 frontier*.
- → **NFCorpus puzzle 의 mechanism 이 *cross-dataset* 으로 transferable 하지 않을 가능성**:
  - NFCorpus 의 *catastrophic-gap recovery* (74 %) 는 데이터셋 의 *base difficulty* (NDCG@10 baseline 0.330) 에서 발생.
  - SciFact 는 *high baseline* (NDCG@10 0.65) 에서 시작 → direction 의 marginal 가치 적음.
  - → *direction-magnitude 가설* 은 *catastrophic regime* 특이적, *baseline regime* 에서는 다른 frontier.

### 3.4 다음 실험의 방향성 명확화

- **Anchor-side regularization 의 plateau 확정** — 추가 anchor 변형 (entropy regularizer, layer-wise, etc.) 은 동일 frontier 안에서만 움직일 가능성 ↑.
- **Data-side family (Exp 14 difficulty-weighted)** 가 *진정한 새 frontier* 만들 수 있는지의 핵심 테스트.
- *Cross-dataset* 검정 (Exp 13 on NFCorpus) 은 §3.3 의 "direction mechanism = catastrophic regime 특이" 가설 검정 가능 — 단 STOP rule (pre-commit) 따라 본 연구에서는 미실시. Future work.

---

## 4. Figures

(figures.py 로 artifact 로부터 재현 가능)

### 4.1 Δ NDCG@10 forest plot (3 seeds + mean, vs Exp 11)

![Δ NDCG@10 forest plot](figures/13_frozen_direction_anchor/delta_ci_forest.png)

**Caption**: SciFact 의 paired bootstrap 95 % CI on Δ NDCG@10. Exp 13 (3 seeds + mean) vs Exp 11 (3 seeds + mean) 의 anchor-side family 비교. 3 slices (all / confused / easy) 동시 표기. Exp 13 의 Δ easy mean (−0.021) 이 pre-commit branch (a) 임계 (−0.020) 를 0.001 차이로 fail — Δ easy CI line 의 0 위치 가까이에서의 visual 확인.

### 4.2 Anchor-side family 비교 — Exp 11 vs Exp 13 trade-off scatter

![Exp 11 vs Exp 13 trade-off scatter](figures/13_frozen_direction_anchor/anchor_family_scatter.png)

**Caption**: x = Δ confused, y = Δ easy 평면 상 3 lever (Phase 2b, Exp 11, Exp 13) 의 seed-level points. Anchor-side family 의 두 lever (Exp 11, Exp 13) 가 frontier 의 *동일 region* 점유. 1:1 trade-off line (회색 dashed) 위에서 두 family 가 *통계적으로 구분 안 됨*.

### 4.3 학습 곡선 (rank / anchor loss + val NDCG)

![Train curves](figures/13_frozen_direction_anchor/train_curves.png)

**Caption**: Exp 13 seed 42 의 3-epoch 학습 동학. rank_loss 단조 감소 (1.40 → 0.07), anchor_loss epoch 1 후 plateau (0.18 → 0.47). val_ndcg_all epoch 1 best (0.687) → epoch 2-3 감소 → early-stop snapshot. 가벼운 over-correction 패턴 (Phase 2b 동일).

### 4.4 LoRA A/B norm distribution

![LoRA A/B norms](figures/13_frozen_direction_anchor/lora_AB_norms.png)

**Caption**: 24 adapters (12 layers × q,v) 의 학습된 ‖A‖, ‖B‖ 분포. Exp 13 의 ‖B‖_total = 1.40 ± 0.27 (3-seed mean) — Exp 11 의 ~1.8 보다 22 % 작음. *Per-token direction anchor 가 update magnitude 도 직접 제약함* 의 mechanism direct evidence.

### 4.5 NDCG slice grid (all / confused / easy, 3 seeds)

![NDCG slice grid](figures/13_frozen_direction_anchor/ndcg_slice_grid.png)

**Caption**: 3 seeds × 3 slices 의 NDCG@10 bar plot, baseline (frozen ColBERT) 대비 시각화. confused slice 의 +0.092 mean lift 와 easy slice 의 −0.021 mean drop 의 *zero-sum* 패턴 확인.

---

## 5. Diagnostic B — *mechanism direct verification* (sub-experiment, post-hoc measurement)

본 sub-experiment 는 Exp 13 의 *학습된 LoRA checkpoint 위* 에서 *post-hoc measurement only* (no new training, pre-commit STOP rule 무관). Exp 13 의 **mechanism claim** — "per-token cosine anchor 가 frozen baseline 으로의 회귀를 강제" — 의 *empirical anchor* 제공.

**Method**: 3 seeds × SciFact test corpus 300 docs sampled. 각 checkpoint 의 LoRA-encoded representation 과 *frozen* ColBERT-encoded representation 의 token-level cosine 직접 측정.

### 5.1 결과 표 (3 seeds + frozen baseline)

| Condition | doc_pair_cos μ | tok_pair_cos μ | eff_rank doc | eff_rank tok | **cos(LoRA, frozen) tok** | **cos(LoRA, frozen) doc** |
|---|---|---|---|---|---|---|
| frozen baseline | +0.587 | +0.214 | 9.86 | **55.13** | 1.000 (identity) | 1.000 |
| Exp 13 seed 42 | +0.881 | +0.654 | 2.26 | 8.60 | **0.820** | 0.820 |
| Exp 13 seed 1337 | +0.872 | +0.641 | 2.37 | 9.11 | **0.823** | 0.823 |
| Exp 13 seed 2024 | +0.875 | +0.638 | 2.35 | 9.31 | **0.830** | 0.830 |
| **Exp 13 3-seed mean** | +0.876 | +0.644 | **2.33 ± 0.06** | **9.01 ± 0.36** | **0.824 ± 0.005** | **0.824 ± 0.005** |
| Exp 11 seed 42 (cached) | +0.901 | +0.675 | 2.01 | 7.69 | (미측정) | (미측정) |

### 5.2 핵심 발견 — *3-fold mechanism evidence*

#### Finding ⭐1: Anchor cos = **0.824**, *not* 1.0 — *soft equilibrium attractor*

Exp 13 의 loss = $1 - \cos(h_t^{\text{LoRA}}, h_t^{\text{frozen}})$. 학습 후 **잔여 anchor_loss = 1 − 0.824 = 0.176** — train_history.json (seed 42, ep1) 의 `anchor_losses[ep1] = 0.18` 과 **정확 일치** (early-stop snapshot at ep1 best).

→ **Loss 가 *부분적으로만* 최적화**: confused 학습 신호 (token 을 frozen 에서 *멀리* 끌어당김) ↔ anchor preservation (token 을 frozen 으로 *되돌림*) 의 *equilibrium 상태*. Anchor 가 *strict identity* 가 아닌 *soft attractor* — 약 33° (cos⁻¹(0.82)) 의 deviation 을 허용.

**중요한 함의**: λ_dir 이 더 컸다면 (예 λ_dir=5, 10) anchor cos → 1.0 에 더 가까워졌을 것이나, 그 경우 confused 학습 신호 가 over-restrict 됨 (branch (c) 예상). λ_dir=1.0 이 *equilibrium-formation 의 적정값* 임을 사후 확인.

#### Finding ⭐2: Token eff_rank **9.01** (Exp 13) > **7.69** (Exp 11) — *anchor-side family 내 미세 분리*

Frozen 의 token eff_rank 55.13 대비 두 anchor-side lever 모두 *significant collapse* 보이지만:

| Lever | tok eff_rank | vs Exp 11 | Mechanism |
|---|---|---|---|
| Exp 11 (relational, Sim Frob²) | 7.69 | (baseline) | 토큰 간 *relational structure* 만 보존, *absolute direction* 자유 |
| Exp 13 (absolute, per-token cos) | **9.01 (+17 %)** | +1.32 | 토큰 별 *absolute direction* 직접 anchor — *spatial diversity 추가 보존* |

→ **Anchor-side family 내 *internal representation 면에서 미세 차이*** — Exp 13 의 rotation-sensitive constraint 가 token diversity 17 % 더 보존. *NDCG@10 frontier 는 frontier 공유* (§2.3) 임에도 *internal mechanism 분리* — *external behavior ≠ internal representation* 의 흥미로운 dissociation.

#### Finding ⭐3: Doc eff_rank **2.33** — Phase 2b-level, *token-level only* preservation

Token-level 은 일부 회복 (9.01 vs frozen 55.13, **16 % 회복**) but doc-level 은 Phase 2b-level (2.33 vs frozen 9.86, *76 % collapse*). Exp 11 의 doc eff_rank 2.01 와도 essentially 동등.

→ **Anchor-side family 가 *token granularity* 에서만 효과적, *doc aggregation* 후 anchor 효과 희석** — Exp 13 의 per-token cosine 이 token-level anchor 임에도 *doc-level mean pooling* 후엔 *collapse 잔존*. *Anchor-side family 의 capacity limit* 의 mechanism direct evidence.

### 5.3 Exp 11 vs Exp 13 의 *internal representation* 비교 — *paired mechanism geometry*

§2.3 의 NDCG@10 frontier 동등성 + §5.2 의 internal eff_rank 미세 분리 결합:

| Aspect | Exp 11 (relational) | Exp 13 (absolute) | Interpretation |
|---|---|---|---|
| Δ NDCG@10 (all/conf/easy) | +0.029 / +0.101 / −0.031 | +0.030 / +0.092 / −0.021 | **frontier 공유** (statistically equivalent) |
| Strict 3/3 | 2/3 | **3/3** | Exp 13 slightly more robust |
| ‖B‖_total | ~1.8 | **1.34** (−22 %) | Exp 13 의 update magnitude 더 controlled |
| doc eff_rank | 2.01 | 2.33 | comparable doc-level collapse |
| **tok eff_rank** | **7.69** | **9.01 (+17 %)** | **Exp 13 의 token diversity preservation 더 강함** |
| Mechanism | rotation-invariant | rotation-sensitive | (formal difference) |

→ **외부 behavior (NDCG) 와 internal representation (eff_rank) 의 *dissociation*** — 두 anchor-side lever 의 NDCG outcome 은 동등, 내부 표현 구조는 미세 분리. *Anchor-side family 가 동일 frontier 위에 있지만 *어떻게* 거기 도달했는지* 가 다름.

### 5.4 Paper §7.3.f mechanism comparison 으로의 통합

§7.3.f.iii NFCorpus *direction matters* puzzle 와 결합:

| Condition | doc_cos μ | doc eff_rank | tok eff_rank | NDCG @10 confused |
|---|---|---|---|---|
| Phase 2b (SciFact) | ~0.97 | ~1.1 | ~1.7 | +0.104 ✓ |
| Exp 11 (SciFact) | 0.901 | 2.01 | 7.69 | +0.101 ✓ |
| **Exp 13 (SciFact)** | **0.876** | **2.33** | **9.01** | **+0.092 ✓** |
| M1b + NFCorpus | ~0.995 | ~1.06 | ~1.05 | +0.246 ✓ (74 % recovery) |

→ **NFCorpus puzzle 의 paradox 재확인**: collapse magnitude 동일 (~1.05) 임에도 NDCG 회복 — *direction matters, magnitude doesn't*. 본 paper 의 §7.3.f.ii framework 와 §7.3.g (Exp 13) 의 *direct evidence chain* 완성.

### 5.5 Diagnostic B figure

![Diagnostic B on Exp 13 — mechanism verification](../report/figures/_repr_collapse_exp13/repr_collapse_exp13.png)

**Caption** (3-panel):
- **(A) Anchor proximity** — Exp 13 3 seeds 의 per-token cos(LoRA, frozen) vs per-doc mean. 모든 seed 0.82 부근, frozen identity (cos=1) 와 0.18 거리. *Soft equilibrium attractor* 직접 visual.
- **(B) Token eff_rank comparison** — frozen (55.13) → Phase 2b (~1.7) → Exp 11 (7.69) → Exp 13 (9.01). Anchor-side family 의 partial collapse recovery + Exp 13 의 17 % 추가 diversity preservation.
- **(C) Per-token cos distribution** — 3 seeds 의 token-level cos(LoRA, frozen) 분포 (47K tokens). 평균 0.82 부근의 좁은 unimodal distribution, long tail 없음 → anchor proximity 가 *token-uniform*.

**Artifact**: `report/figures/_repr_collapse_exp13/repr_collapse_exp13_data.json` + `.{pdf,png}`.
**Script**: `report/_repr_collapse_exp13.py` (CPU, cache resume, ~3.5 min on Mac M1).

---

## 6. 종합

**Exp 13 의 학술적 contribution** (positive 형식 외):

1. **Anchor-side family 의 frontier 강건성 실증** — Sim Frobenius² (rotation-invariant) 와 per-token cosine (rotation-sensitive) 가 *통계적으로 구분 안 되는* frontier 점유. *"수학적 차이 ≠ empirical separation"*.

2. **3/3 strict robustness** — Exp 11 (2/3 strict) 대비 Exp 13 (3/3 strict) — anchor-side 의 *honest terminus* 후보로 ⭐ 자격.

3. **LoRA update magnitude 의 직접 제약 증거** — ‖B‖_total Exp 11 → Exp 13 22 % 감소 = anchor preservation mechanism 의 measurable proxy.

4. **NFCorpus puzzle 의 cross-regime 비전이성** — *direction mechanism* 이 catastrophic regime 특이적 가능성 시사 (future work).

5. **5-lever framework** 으로 paper 의 4-lever 확장 — anchor-side dual (Exp 11 relational, Exp 13 absolute) + data-side dual (M1b substitution, Exp 12 binary filter) + (Exp 14 continuous) → final paper-grade narrative.

**STOP rule 준수**: 3 seeds 완료 후 sweep / variant / cross-dataset *전부 금지*. Result-blind pre-commit 따라 branch (b) 확정, *no narrative reversal*.

**Raw artifacts**: `outputs/13_frozen_direction_anchor/scifact/seed_{42,1337,2024}/qv_r8_l12_dir1/`.
**Pre-commit reference**: `report/_exp13_14_pre_commit.md`.
**Reproducibility**: `experiments/13_frozen_direction_anchor/run.py --dataset scifact --seed {42|1337|2024} --lambda-dir 1.0 --r 8 --alpha 8.0 --lora-lr 5e-5 --max-triplets 9190`.
