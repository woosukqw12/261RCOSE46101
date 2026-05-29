# 11_easy_preservation — Explicit easy-slice preservation loss (λ_anc > 0)

> **본 실험은 외부 reviewer agent 의 제안**. 본 paper 의 §5e main contribution (universal rank-collapse + spatial multiplicity escape) 와 *독립*. 결과 무관하게 *§6 / §7 의 supplementary diagnostic* 으로 통합.

## 가설

Phase 2b 의 *Δall ≈ +0.001* (anchor preserved) 가 *net 보존* 인가 *redistribution* 인가? **Step 0 (`report/_easy_slice_step0.py`) gate PASSED** — 실측 Δeasy = −0.085 (math expected −0.086, 99 % match) → *redistribution* 확정.

**Exp 11**: *explicit easy-preservation 압력* (λ > 0 의 relational self-sim 보존 loss on easy queries) 으로 *redistribution 해소 가능 여부* 검정.

세 분기 (pre-committed prediction, `report/_exp11_pre_commit.md`):
- **(a) net improvement**: Δeasy → ~0 + Δconfused ≈ +0.10 유지 → entanglement *해소 가능* → *first net improvement* on frozen-encoder.
- **(b) diagnostic**: Δeasy → ~0 + Δconfused → ~0 → confused–easy *inherent entanglement* → 09 distill 의 "anchor 보호 ↔ lever 죽음" 의 2 번째 독립 증거 (encoder bottleneck 강화).
- **(c) inconclusive**: Δeasy 여전히 negative → preservation term 약 / 구현 issue → 정직 STOP.

세 분기 모두 paper 강화 (대칭 risk, 폭탄 결과 없음).

## Loss 설계

batch 의 query 를 *baseline 의 confused (top1 ≠ rel)* vs *easy (top1 = rel)* 로 분리:

$$
\mathcal{L} = \underbrace{\frac{1}{|T_C|}\sum_{(q,d^+,d^-) \in T_C} \max(0, m - s(q,d^+) + s(q,d^-))}_{\text{confused: margin (lever 자유)}} + \lambda \cdot \underbrace{\frac{1}{|E|}\sum_{x \in E} \big\|\text{Sim}(H_{\text{LoRA}}^x) - \text{Sim}(H_{\text{frozen}}^x)\big\|_F^2}_{\text{easy: relational(self-sim) 보존}}
$$

- $\text{Sim}(H) = \hat H \hat H^\top$ (token × token cosine matrix, $\hat H$ = per-token L2-norm).
- $x$ = easy query + 그 *positive doc*.
- $H_{\text{frozen}}$ = LoRA 미적용 frozen encoder 출력. **precompute, no_grad, 캐시** (easy queries + pos docs 만).
- **scoping**: easy-selective preservation → confused 는 자유 deviate; confused/easy 가 같은 표현 방향 공유 시 (entanglement) confused lever 도 죽음 = 분기 (b).

## Config

- Inherits Phase 2b: q,v r=8 α=r LR=5e-5, batch=32 ep=3 patience=2 early_stop=val_all.
- λ = **1.0** (single value, no sweep — pre-commit binding).
- Seeds: 42, 1337, 2024.
- Dataset: **SciFact only** (redistribution 이 측정된 유일 dataset; NFCorpus/FiQA 의 catastrophic 은 별개 문제).

## *Train code 의 수정점* (M1 의 LoRA snapshotting 한계 해소)

`train_steering()` 의 best-state 가 `steering` 만 snapshot → LoRA params 는 ep3 final. M1 의 trajectory 가 ep1 best 였는데 test 는 ep3 사용했다. Exp 11 의 새 training function 은 LoRA params 도 best-state snapshot 적용 — *fair* mediation 검정.

## Pre-commit binding

`report/_exp11_pre_commit.md` 에 prediction 작성 후 *학습 시작*. 결과 후 prediction 수정 금지. seed × 3 robustness.

## 사용자 confirm 필요

- Step 0 gate 결과 (이미 PASS, 확인 완료)
- λ commit (=1.0)
- 학습 시작 (queue 종료 후)
- *결과 무관 STOP* — 추가 실험 / λ sweep / multi-dataset 확장 금지.
