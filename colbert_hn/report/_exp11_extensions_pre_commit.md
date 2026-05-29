# Exp 11 Extensions Pre-Commit — Higher λ + M1b combined

**작성**: 2026-05-24 evening (학습 시작 *전*).

## Motivation

§5f 의 *4-lever framework* 의 *완전 해소* 가능 여부 검정. Exp 11 (λ=1, 3-seed) 의 partial branch (a) 결과:
- Δ all +0.029 ± 0.005 (2/3 strict)
- Δ confused +0.101 ± 0.010 ✓ preserved
- Δ easy −0.031 ± 0.018 (63 % 감소 vs Phase 2b −0.085)

**두 가지 미해소**:
1. **Higher λ 가 *full strict* + *Δ easy ≈ 0* 달성 가능?** (preservation pressure 강화)
2. **M1b + Exp 11 combined 가 *additive full resolution* 달성 가능?** (hard 회피 + selective preservation)

## Experiment 1 — Higher λ Exp 11

| Item | Value |
|---|---|
| **λ_easy** | **5.0** (5× current, single value pre-commit) |
| Negatives | Mined HN (Phase 2b default) |
| Other | Phase 2b 동일 (q,v r=8 LR=5e-5 α=r ep=3 patience=2 early-stop=val_all) |
| Dataset | SciFact |
| Seeds | 42, 1337, 2024 |
| Tag | `qv_r8_l12_le5` |

## Experiment 2 — M1b + Exp 11 combined

| Item | Value |
|---|---|
| **λ_easy** | **1.0** (Exp 11 와 *fair comparison* 위해 동일) |
| Negatives | **In-batch negative** (confused queries 만, `pos_emb.roll(1, dims=0)`) |
| Easy queries | Relational preservation loss (Exp 11 와 동일) |
| Other | 동일 |
| Dataset | SciFact |
| Seeds | 42, 1337, 2024 |
| Tag | `qv_r8_l12_le1_m1b` |

## Pre-committed predictions (3 branches each, *결과 보기 전*)

### Higher λ Exp 11 (λ=5)

| Branch | 결과 | 함의 |
|---|---|---|
| **(a) Full resolution** | Δ all > 0.03 strict 3-seed + Δ easy ≈ 0 + Δ confused ≈ +0.10 | Higher pressure → entanglement fully solvable. Paper *최강* finding. |
| **(b) Diminishing returns** | Δ all ≈ Exp 11 (+0.029) 또는 *조금 떨어짐* + Δ easy 약간 개선 | λ 의 marginal benefit 감소 — partial resolution 의 *natural ceiling* 확인. |
| **(c) Over-regularization** | Δ confused 감소 (+0.05 미만) + Δ all 감소 | Higher λ 가 confused lever 도 죽임 — preservation pressure 가 너무 강함. |

### M1b + Exp 11 combined (in-batch + λ=1)

| Branch | 결과 | 함의 |
|---|---|---|
| **(a) Additive full** | Δ all > 0.04 strict 3-seed + Δ confused +0.10 ✓ + Δ easy ≈ 0 | Two levers *fully additive* — both *hard 회피* + *easy preservation* 동시 필요. |
| **(b) Sub-additive** | Δ all ≈ +0.03 (M1b 의 +0.021 + Exp 11 의 +0.029 보다 작음) | Two mechanisms overlap — single lever 로 enough. |
| **(c) Antagonistic** | Δ all < M1b (+0.021) | 두 lever 가 *충돌* — in-batch 의 easy contrast 와 relational preservation 이 incompatible. |

## STOP rule

각 실험 한 번 끝나면 결과 무관 STOP. Further λ sweep / lever combine 금지 → paper future work 명시.

---

**Commit 시점**: 2026-05-24 evening, training *시작 전*.
**Training**: SciFact × 3 seeds for each (total 6 runs, ~90 min).
