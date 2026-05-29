# Exp 11 Pre-Commit Prediction (BEFORE training)

**작성 일시**: 2026-05-24 04:05
**상태**: M1b queue 진행 중, Exp 11 학습 *시작 전*. 본 prediction 은 결과 후 *수정 금지*.

---

## 1. Experimental setup (single rule, no sweep)

| 항목 | 값 |
|---|---|
| Dataset | SciFact (redistribution 측정된 유일 dataset) |
| Loss | L = margin_loss(confused 만) + λ × relational_self_sim_loss(easy 만) |
| **λ_easy** | **1.0** (single value commit, sweep 금지) |
| LoRA | q, v on all 12 layers, r=8, α=r (scaling 1.0) |
| LR | 5e-5 (Phase 2b 동일) |
| Optimizer | AdamW, weight_decay=1e-4 |
| Warmup / Clip | **off** (Exp 11 의 *순수* 효과 측정 — mediation 1 의 entanglement 회피) |
| Batch / Epochs / Patience | 32 / 3 / 2 (Phase 2b 동일) |
| Early-stop metric | val_all |
| Seeds | 42, 1337, 2024 (3-seed mean ± std) |
| **Train code fix** | LoRA params **best-state snapshot** 추가 (M1 의 ep3-only 한계 해소) |

## 2. Step 0 결과 (commit 의 motivation, 이미 확인)

`report/_easy_slice_step0.py` 결과:
- 수학 예측 Δeasy = (Δall − w_c × Δconf) / w_e ≈ **−0.086**.
- 실측 3-seed mean Δeasy = **−0.085 ± 0.010**.
- 99 % match → Phase 2b 의 "anchor preserved" = *redistribution* 확정.

→ Exp 11 gate PASSED.

## 3. Pre-committed prediction (3 분기, 결과 보기 *전* commit)

세 분기 모두 paper 강화 (대칭 risk, "폭탄" 결과 없음).

### (a) **Positive — net improvement** (확률 추정: ~30 %)
- 결과: Δeasy → ~0 (CI 0 포함, 보존) + Δconfused ≈ +0.10 유지 ✓.
- 함의: **Frozen-encoder intervention 의 첫 *net* 향상 사례**. *entanglement 해소 가능* — confused 와 easy 의 표현 방향이 *분리* → LoRA 의 selective 압력 응답 가능.
- Paper 통합: 새 소절 §6.x "Explicit easy-preservation enables net improvement" (redistribution 의 *해소 가능성* 직접 증명).
- 단 *unfrozen +0.260 가 상한* 임은 변하지 않음 — bounded improvement 의 *upper 위 한 단계*.

### (b) **Diagnostic — confused–easy entanglement** (확률 추정: ~50 %)
- 결과: Δeasy → ~0 (보존됨) **+** Δconfused → ~0 (lever 사라짐, ≈ random 07 / ≈ 09 distill λ=0.5).
- 함의: *confused 를 돕는 표현 방향* 과 *easy 가 점유한 표현 방향* 이 *공통 subspace* — easy 보존 압력 이 confused lever 도 죽임. **§5e main contribution (universal rank-collapse + spatial multiplicity) 와 정합** — LoRA 의 *spatial multiplicity escape* 가 *task-specific* 인 이유 의 추가 증거.
- Paper 통합: §7 "Confused–easy entanglement: an inherent frozen-budget tradeoff" — 09 distill 의 "anchor 보호 ↔ lever 죽음" 의 *2 번째 독립 증거*. encoder-bottleneck 서사 강화.

### (c) **Inconclusive — preservation term 약함** (확률 추정: ~20 %)
- 결과: Δeasy 여전히 음수 (CI 상한 < 0).
- 함의: λ=1.0 의 압력 부족 또는 *relational self-sim ≠ MaxSim discrimination* 의 misalignment.
- Paper 통합: Limitations 에 정직 기록 + future work (λ tuning / per-token weighting / different sim metric). **STOP — λ sweep 금지 (pre-commit)**.

## 4. Mechanism readout (필수)

학습 후 *easy doc* 의 effective rank (`report/_repr_collapse_diagnostic.py` 재사용):
- *frozen baseline ~10* 수준 보존 (preservation 작동) ↔ Phase 2b *collapse ~1*.
- (a) 분기 면 frozen 수준 (~5-10), (b) 분기 면 약간 collapse 완화 (~3-5), (c) 분기 면 ~1 (collapse 유지).
- Self-sim loss 가 *바로 그 양* 을 직접 규제 → readout 와 loss 가 *같은 quantity* (mechanism 직접 확증).

## 5. 분석 결과 표 (학습 후 채움)

| Slice | Phase 2b (3-seed) | **Exp 11 λ=1 (3-seed)** | Branch |
|---|---|---|---|
| Δall | +0.001 ± 0.012 | ? | — |
| Δconfused | +0.104 ± 0.014 | ? | — |
| Δeasy | −0.085 ± 0.010 | ? | — |
| easy doc eff_rank (mean) | ~1.15 (collapse) | ? | — |

## 6. STOP rule

본 실험 한 번 끝나면 결과 무관 STOP:
- 결과 (a) → "positive 발견, follow-up 은 future work" — 추가 실험 금지.
- 결과 (b) → diagnostic 의미 확인, paper 강화. 추가 실험 금지.
- 결과 (c) → 한계 정직 기록. λ sweep / orthogonality reg / RS-LoRA / 다른 dataset 으로 확장 *전부 금지*.

## 7. Implementation 확인 (verify list)

- [x] `experiments/11_easy_preservation/run.py` — frozen self-sim cache + relational loss + LoRA snapshot 보장.
- [x] cache: encode 전 LoRA injection 전. `with torch.no_grad()`.
- [x] easy queries 만 cache. Mask 처리 (doc 의 valid token).
- [x] 학습 loop: confused → margin, easy → relational, 동시 처리 (mixed batch).
- [x] **LoRA best-state snapshot** (M1 의 ep3-only 한계 해소).
- [x] Doc 출력: Δall, Δconfused, Δeasy 동시 보고.

---

**Commit 시점**: 2026-05-24 04:05 (M1b queue 진행 중, Exp 11 training *시작 전*).
**Training 시작**: M1b queue 종료 후 (사용자 confirm 필요).
