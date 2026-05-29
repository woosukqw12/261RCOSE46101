# Exp 12 Pre-Commit Prediction — FN-denoised mined-HN

**작성**: 2026-05-24 morning (E5 doc encoding 중, 학습 시작 *전*).

## 1. 가설 (캐비엇 1 의 결정적 disambiguator)

| Branch | 조건 | 함의 |
|---|---|---|
| **(나-1) Noise 가 원인** | FN-denoised mined-HN 학습 시 **Δ confused +0.10 (Phase 2b level) 유지** + **Δ all strict positive 3-seed robust** | M1b 의 net+ = *noise 제거 효과*. *Supervision noise* 가 catastrophic / redistribution 의 *진짜 원인* 확정. |
| **(나-2) Difficulty 가 원인** | FN-denoised mined-HN 학습 시 **Phase 2b 와 동일 redistribution** (Δ conf +0.10 / Δ easy −0.08 / Δ all ≈ 0) | M1b 의 net+ = *easy contrast (작은 gradient)* 의 결과. *Hard negative 자체* 가 collapse 유발 — *noise 제거* 만으로 부족. |
| **(둘 다)** | 중간 결과 (Δ confused 부분 ↓ + 부분 strict net+) | *두 mechanism additive* — paper-grade nuanced narrative. |

## 2. 학습 config (Phase 2b 와 *negative source 만* 다름)

| 항목 | Phase 2b | **Exp 12 (FN-denoised)** |
|---|---|---|
| HN source | baseline ColBERT top-100 | **동일** (위에 filter) |
| HN difficulty | hard | **hard 유지** |
| **FN noise** | ~33 % | **~0 %** (e5_margin ≤ 0 제거) |
| LR, α, r, batch, ep, patience | (Phase 2b 동일) | (동일) |
| early-stop | val_all | val_all |
| Dataset | SciFact | SciFact (M1b/Phase 2b redistribution 측정된 유일 dataset) |
| Seeds | 42 / 1337 / 2024 | 42 / 1337 / 2024 |

## 3. E5 margin denoising 의 정확한 정의

- e5_q = `data/e5_teacher/e5_train_q_emb_scifact.pt` 의 query emb (809 queries × 4096)
- e5_d = `data/e5_teacher/e5_train_doc_emb_scifact.pt` 의 doc emb (5183 docs × 4096, 본 실험 위해 새로 추출)
- e5_margin (q, pos, hn) = cos(eq, epos) − cos(eq, ehn)
- 둘 다 L2-normed → cosine 은 dot product
- **Threshold = 0.0**: e5_margin ≤ 0 인 triplet 제거. Single value, no sweep (pre-commit).

## 4. 예측 데이터 — 09 cache 기준 prior (45 queries × ~56 triplets = 2526)

- e5_margin < 0: 32.5 % of triplets (likely FN per E5)
- e5_margin < −0.05: 27.3 %
- e5_margin < −0.1: 23.7 %

**Phase 2b 의 9190 train triplets 에서 ~32 % FN 제거 시 ~6100 cleaned triplets** 예상.

## 5. 결과 보기 전 commit

**3-seed mean 결과 의 *기각 vs 확정* 조건**:

| Metric | (나-1) noise 확정 조건 | (나-2) difficulty 확정 조건 |
|---|---|---|
| Δ confused | +0.09 ~ +0.11 (Phase 2b 동등) | +0.09 ~ +0.11 (Phase 2b 동등) |
| Δ all | **+0.02 이상 strict (CI 하한 > 0) — 3 seeds 모두** | ≈ 0 (CI 0 포함, redistribution) |
| Δ easy | −0.04 이상 (절반 이상 회복) | −0.07 ~ −0.09 (Phase 2b 와 동일) |

두 가설 모두 *유사 confused lift* 예측 — *Δ all + Δ easy* 가 분기 결정.

## 6. STOP rule

본 실험 한 번 끝나면 결과 무관 STOP. λ / threshold sweep / 다른 dataset 확장 금지. NFCorpus / FiQA 의 FN-denoising = paper future work (E5 train doc emb 미캐시).

---

**Commit 시점**: 2026-05-24 morning. **Training 시작 전 commit, 결과 후 수정 금지**.
