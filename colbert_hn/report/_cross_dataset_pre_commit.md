# Cross-dataset Ablations Pre-Commit (BEFORE training)

**작성 시점**: 2026-05-25 (학습 시작 *전*).

**Methodological commitments** (절대 위반 금지):

- Pre-commit single config per (실험 × 데이터셋) — *SciFact 에서 검증된 동일 config 그대로 적용*. HP 재튜닝 금지.
- 3 seeds {42, 1337, 2024} 각 (실험 × 데이터셋) 쌍에 대해 실행.
- Result-blind: 결과 보기 *전* commit, 결과 후 수정 금지.
- STOP rule: 본 cross-dataset 묶음 (Tier 1 + 2 = 9 runs) 종료 후 *추가 실험 금지*. 결과가 catastrophic 이어도 *다른 데이터셋 / config sweep / variant* 진행 금지.
- Negative result 도 *equal honest weight* 보고.

---

## Motivation

본 cross-dataset 묶음은 **NARRATIVE.md 의 비일관성 빈칸 메우기** — 새 design 탐색이 아닌 *기존 best 방법론 의 cross-domain 검증*:

- **Tier 1**: anchor (실험 H, per-token cosine, λ_dir=1) 가 *SciFact 만* 검증됨. *best method* 의 cross-domain 빈칸 → reviewer 가 가장 의문 가질 지점.
- **Tier 2**: §5 (cross-domain) 의 핵심 주장 ("NFCorpus catastrophic 의 원인 = false-negative 오염") 이 *간접 증거* (in-batch easy negative 의 74 % 회복) 에 의존. NFCorpus 에서 FN 직접 제거 (Exp 12 mechanism) 가 catastrophic 을 푸는지의 *직접 검정*.

---

## Tier 1 — anchor × {NFCorpus, FiQA} × 3 seeds

### Config (SciFact 와 동일, 재튜닝 금지)

| Item | Value |
|---|---|
| Loss | $\mathcal{L} = \mathcal{L}_{\text{margin}}(\text{confused}) + \lambda_{\text{dir}} \cdot \text{mean}_{x \in \text{easy}}[\text{mean}_t(1 - \cos(h_t^{\text{LoRA}}, h_t^{\text{frozen}}))]$ |
| **λ_dir** | **1.0** (SciFact 와 동일 single value, *no sweep*) |
| LoRA | q, v r=8 α=r |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early-stop=val_all |
| Triplet cap | 9190 (SciFact 와 동일) |
| Datasets | **NFCorpus + FiQA** |
| Seeds | 42, 1337, 2024 (per dataset) |
| Total runs | **2 datasets × 3 seeds = 6** |
| Tag | `qv_r8_l12_dir1` (SciFact 와 동일) |

### 3-branch predictions

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) Cross-domain robust** | 두 데이터셋 모두 Δ all > 0 with CI 0 unincluded (anchor preserved) 또는 strict net+ | *Anchor 가 model-side intervention 의 일반적 lever* — paper main contribution **+1 strength**. SciFact-only 의 비일관성 해소. |
| **(b) Plain LoRA 대비 부분 회복** | catastrophic 이지만 plain LoRA 의 NFCorpus −0.320, FiQA −0.347 보다 *덜 심함* | *Anchor 가 cross-domain 손상 완화 효과는 있으나 model-side 단독으론 부족* — §5 의 *2-축 분업* (anchor + FN 정제) 주장 강화. |
| **(c) Catastrophic 동등 또는 더 심함** | plain LoRA NFCorpus −0.320, FiQA −0.347 와 통계 동등 또는 더 나쁨 | *Anchor 가 SciFact-specific 처방* — paper §5 의 *sparse-judgment 한정* 주장 직접 입증. |

세 branch 모두 paper-grade.

---

## Tier 2 — Exp 12 (FN-denoised) × NFCorpus × 3 seeds

### Config (SciFact 와 동일, 재튜닝 금지)

| Item | Value |
|---|---|
| Method | FN-denoised mined-HN (E5-Mistral cosine margin 으로 e5_margin > 0 인 triplet 만 keep) |
| **Threshold** | **0** (e5_margin > 0, SciFact 동일 single value) |
| LoRA | q, v r=8 α=r |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early-stop=val_all |
| Triplet cap | 9190 |
| Dataset | **NFCorpus** |
| Seeds | 42, 1337, 2024 |
| Total runs | **1 dataset × 3 seeds = 3** |
| Tag | `qv_r8_l12_thresh0` (SciFact 동일) |

### Prerequisite

E5-Mistral-7B-Instruct embeddings 추출:
- `data/e5_teacher/e5_train_q_emb_nfcorpus.pt` (~10 min)
- `data/e5_teacher/e5_train_doc_emb_nfcorpus.pt` (~5 min)

### 3-branch predictions

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) FN 제거로 NFCorpus 회복** | NFCorpus 에서 Exp 12 Δ all > −0.10 (plain LoRA −0.320 의 거의 회복) | §5 의 *"NFCorpus catastrophic = FN 원인"* 주장 **직접 입증**. Paper-grade 강한 evidence. |
| **(b) 부분 회복 (M1b 수준)** | Δ all ≈ −0.08 (M1b NFCorpus 74 % recovery 와 유사) | *FN 제거 가 in-batch easy negative 대체와 effective equivalent* — 두 방법 의 mechanism 일치 증거. §5 narrative 보강. |
| **(c) Catastrophic 그대로** | Δ all ≈ −0.30 (plain LoRA 와 동등) | *FN 제거 만으로 부족, hard contrast 자체가 catastrophic* — SciFact 의 Exp 12 결과 (FN noise minor) 와 일치, NFCorpus 의 catastrophic mechanism 이 *FN 외 추가 요인* 임을 시사. §5 narrative *재해석* 필요. |

세 branch 모두 paper-grade.

---

## Engineering required

### Tier 1
- `experiments/13_frozen_direction_anchor/run.py`:
  - `TRAIN_AVAILABLE = ("scifact",)` → `("scifact", "nfcorpus", "fiqa")`
  - Argparse choices 확장

### Tier 2
- `data/e5_teacher/extract_train_docs.py`:
  - `choices=("scifact",)` → `("scifact", "nfcorpus", "fiqa")`
- `experiments/12_fn_denoised_hn/run.py`:
  - `TRAIN_AVAILABLE = ("scifact",)` → `("scifact", "nfcorpus")`

### Smoke test 순서
1. Tier 1: NFCorpus seed 42 (~15 min) — code 검증
2. (OK 시) Tier 1 remaining 5 + Tier 2 launch 백그라운드

---

## STOP rule (강조)

**Tier 1 + 2 = 9 runs 후 결과 무관 STOP**:
- 추가 데이터셋 (SciDocs, TREC-COVID, ArguAna) 진행 *금지*.
- HP 재튜닝 (예 LR 조정, λ_dir 조정) *금지*.
- Variant (예 anchor + FN 결합) *금지*.
- *Future work* 글로만 가능.

본 묶음은 *narrative consistency 빈칸 메우기* 이지 *새 탐색* 아니다. 결과가 어떻든 paper writing 진입.

---

**Commit timestamp**: 2026-05-25.
**Training start**: 즉시 (engineering 완성 후).
