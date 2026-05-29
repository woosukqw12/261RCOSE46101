# 01_mean_diff — 비학습 mean-difference direction baseline

## 목적

본 실험은 ROADMAP.md 의 *baseline 단계* 의 두 번째 단계로, **학습 없이** 단순한 representation-level intervention 이 *원리적으로* HN-confused query 의 retrieval 을 개선하는지 검정한다.

구체적으로, train split 에서 추출한 hard-negative / positive 문서들의 layer-12 BERT 표현을 평균하여 *비학습* direction vector 를 계산한다:

$$v = \bar{h}^{(12)}_{\text{HN}} - \bar{h}^{(12)}_{\text{pos}}$$

이후 test 시점에 모든 query / doc 의 layer-12 hidden state 에서 $v$ 를 그대로 뺀다:

$$\tilde{h}^{(12)} = h^{(12)} - v$$

학습 가능 파라미터: **0 개**.

## 가설

- **Primary**: paired bootstrap 95 % CI on Δ NDCG@10 (confused slice, mean_diff − baseline) 의 하한이 *0 을 초과* (mean-diff 가 baseline 보다 confused query 를 유의하게 개선).
- **Anchor**: paired bootstrap 95 % CI on Δ NDCG@10 (all slice) 의 하한이 −0.005 이상 (anchor preservation; non-confused query 가 손상되지 않음).
- **다음 실험 의 anchor**: 02_final_layer_vector (학습된 single-direction LSR) 가 본 실험의 mean-diff 보다 *유의하게 우월* 해야 H5 (학습의 의미성, ROADMAP.md / DESIGN.md §2) 의 primary 증거 확보.

## 데이터셋 범위 (제약)

본 실험은 **BEIR train split 이 존재하는 dataset 에 한정**:

| Dataset | Train queries | Train qrels | 본 실험 대상 |
|---|---|---|---|
| SciFact | 809 | 919 | ✓ |
| NFCorpus | 2,590 | 110,575 | ✓ |
| FiQA-2018 | 5,500 | 14,166 | ✓ |
| SciDocs | — | — | ✗ (no train split) |
| TREC-COVID | — | — | ✗ |
| ArguAna | — | — | ✗ |

3 dataset 제외는 *temporary documented limitation*. 후속 cross-dataset transfer (ROADMAP `29_loocv_held_out`) 에서 train 보유 dataset 의 $v$ 를 held-out dataset 에 적용하여 보강 검정.

## 방법

### 1) HN-pos triplet mining

- Baseline ColBERT v2 를 **train split 의 query** 에 대해 실행 → ranked list 얻음 (in-memory, artifact 비저장).
- 각 query 에 대해:
  - **Positives**: `qrels[q]` 의 relevance ≥ 1 모든 doc.
  - **Hard negatives**: ranked top-100 중 positive 가 아닌 doc 의 top-10.
- 모든 (q, pos, hn) 쌍을 triplet 으로 unfold.

### 2) Direction vector $v$ 계산

- Triplet 의 unique HN doc set + unique positive doc set 추출.
- 각 doc 을 frozen ColBERT v2 의 `[D]`-marked encode 로 forward.
- Layer 12 의 last hidden state `h^(12) ∈ ℝ^(T × 768)` 캡처 (`register_layer_hook(12, capture_fn)` 활용).
- Per-doc 평균: `h_d = mean_t h^(12)_d[t]` (punctuation / [PAD] mask 적용 토큰만).
- 전체 평균: $\bar{h}_{\text{HN}} = \text{mean}_d h_d$ (HN docs), $\bar{h}_{\text{pos}} = \text{mean}_d h_d$ (pos docs).
- $v = \bar{h}_{\text{HN}} - \bar{h}_{\text{pos}} \in \mathbb{R}^{768}$.

### 3) Test 시점 hook 주입 + 평가

- `register_layer_hook(12, lambda h: h - v)` 로 layer 12 출력에서 $v$ subtract.
- 본 hook 은 query / doc 양측에 동일 적용 (ColBERT LSR 컨벤션).
- Test corpus + test queries 를 hook 적용 상태로 *재-encode* (baseline encoding 재사용 불가 — representation 변형됨).
- 평가는 `src/evaluate.py` 의 `score_queries` + `compute_metrics_trec` + `build_aggregate` 재사용.

### 4) Paired bootstrap CI

- 본 실험의 per-query NDCG@10 와 `outputs/00_baseline/{ds}/seed_42/metrics_per_query.json` 의 per-query NDCG@10 을 paired.
- Bootstrap 10K iter × 95 % CI on Δ (mean_diff − baseline). All slice + confused slice 각각.

## 실행 방법

```bash
.venv/bin/python experiments/01_mean_diff/run.py --dataset scifact  --seed 42
.venv/bin/python experiments/01_mean_diff/run.py --dataset nfcorpus --seed 42
.venv/bin/python experiments/01_mean_diff/run.py --dataset fiqa     --seed 42
```

Artifact 출력 경로: `outputs/01_mean_diff/{dataset}/seed_{seed}/`
- `config.json`, `env.json` (실행 환경)
- `v.pt` (계산된 direction vector, 768-d)
- `triplet_stats.json` (mining 통계: # triplets, # unique pos/hn docs)
- `runs.json`, `runs_scored.json` (test 시점 retrieval 결과)
- `metrics_per_query.json`, `metrics_aggregate.json` (mean_diff 의 metric)
- `delta_vs_baseline.json` (per-query Δ + paired bootstrap CI on all/confused slice)

## 성공 기준

| 슬라이스 | Δ NDCG@10 CI 하한 | 통과 조건 |
|---|---|---|
| confused | > 0 | mean_diff 가 baseline 보다 유의하게 개선 |
| all | ≥ −0.005 | anchor preservation (손상 없음) |

두 조건 모두 통과 → **다음 실험 02_final_layer_vector** (학습 infrastructure 구축) 진행.
한쪽이라도 실패 → 원인 분석 (다른 layer 시도, projection-out form, 또는 다른 mining 전략) 후 진행 결정.

## 상세 보고서

[`report/01_mean_diff_report.md`](../../report/01_mean_diff_report.md) — 본 raw 실험 (3 dataset) + `01b_mean_diff_scaled` 의 magnitude sweep (SciFact) 의 **통합 결과** 및 시각화.
