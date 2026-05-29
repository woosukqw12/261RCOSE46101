# 12_fn_denoised_hn — *FN-denoised mined-HN* (캐비엇 1 disambiguator)

> **목적**: 캐비엇 1 (clean ≠ easy confound) 의 *유일한 깨끗한 disambiguator*. M1b (in-batch neg) 의 net+ 가 (나-1) *noise 제거* 인지 (나-2) *hard-difficulty 자체가 collapse 유발* 인지 구분.

## 가설 (pre-committed)

**Phase 2b 가 cleaned mined-HN (hard 유지 + FN 만 제거) 으로 학습 시**:
- (나-1) **Noise 가 원인** ⟹ Δ confused +0.104 (Phase 2b level) *유지* + Δ all *strict positive* (캐비엇 1 부분 해소 + supervision root 가 *진짜* noise 때문).
- (나-2) **Difficulty 가 원인** ⟹ Phase 2b 처럼 *redistribution 유지* (Δ confused +0.10 / Δ easy −0.08 / Δ all ≈ 0) — *cleaned hard 도 collapse 유발*. M1b 의 net+ 는 *easy contrast (작은 gradient)* 의 결과.

세 분기 모두 paper 강화 (대칭 risk).

## 방법 — *negative quality* 만 교체 (그 외 Phase 2b 와 100 % 동일)

| | Phase 2b (원본) | **Exp 12 (FN-denoised)** |
|---|---|---|
| HN source | baseline ColBERT top-100 의 non-relevant doc | 동일 (그 위에 *filter* 추가) |
| HN difficulty | hard (ColBERT 가 confused) | **hard 유지** (ColBERT 가 confused) |
| Noise (FN rate) | ~33 % (per E5 margin < 0) | **~0 % (FN 명시 제거)** |
| Loss / optimizer / LR / epochs / patience | (Phase 2b 동일) | (동일) |

## E5 margin denoising

`data/e5_teacher/e5_train_q_emb_scifact.pt` (809 queries) + `e5_train_doc_emb_scifact.pt` (5183 docs, 본 실험 위해 새로 추출) 의 cosine similarity 로 margin 계산:

$$\text{e5\_margin}(q, \text{pos}, \text{hn}) = \cos(\vec{e_q}, \vec{e_{\text{pos}}}) - \cos(\vec{e_q}, \vec{e_{\text{hn}}})$$

- `e5_margin > 0`: E5 가 pos 를 hn 보다 *더* relevant 로 본 (정상 hard negative)
- **`e5_margin ≤ 0`: E5 가 hn 을 pos *≥* 로 본 (likely false negative) → *제거***

## Pre-commit binding (`report/_exp12_pre_commit.md`)

- **Threshold = 0** (e5_margin ≤ 0 의 triplet 제거). single value, no sweep.
- **Dataset**: SciFact (M1b/Phase 2b 의 redistribution 측정된 유일 dataset).
- **Seeds**: 42, 1337, 2024 (3-seed robustness).
- **결과 보기 전 판정 조건**:
  - Δ confused 가 *Phase 2b 의 +0.104* 와 통계 동등 (CI 겹침) + Δ all *strict positive* (3-seed 모두 CI 하한 > 0) ⟹ **(나-1) noise 원인 확정**.
  - Phase 2b 와 동일 redistribution 패턴 (Δ confused ≈ +0.10 / Δ easy ≈ −0.08 / Δ all ≈ 0) ⟹ **(나-2) difficulty 원인 확정**.
  - 중간 (Δ confused 부분 감소 + 부분 strict net+) ⟹ *두 mechanism additive* — paper-grade nuanced.

## 사용 (queue 종료 후)

```bash
.venv/bin/python experiments/12_fn_denoised_hn/run.py \
  --dataset scifact --seed {42|1337|2024} \
  --margin-threshold 0.0 \
  --r 8 --alpha 8.0 --lora-lr 5e-5 \
  --max-triplets 9190
```

## 한계 + future work

- SciFact 만 (cross-dataset 의 FN denoising 은 NFCorpus / FiQA 의 E5 train corpus encoding 필요 → future).
- E5-Mistral 의 *teacher quality* 자체의 한계 — 09 의 distillation lever 가 약했던 reason. 단 *FN 식별* 은 *full distillation* 보다 약한 요구 (margin sign 만 보면 됨).
- Threshold sensitivity 미검증 (pre-commit single value).

## STOP rule

본 실험 한 번 끝나면 결과 무관 STOP. λ / threshold sweep / 다른 dataset 확장 금지 → paper future work 명시.
