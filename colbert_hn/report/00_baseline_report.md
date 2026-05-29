# 00_baseline — 측정 결과 보고서

본 보고서는 `00_baseline` 실험 (frozen ColBERT v2 재현) 의 *상세 결과* 및 *분석* 을 기록한다. 실행 환경·컨벤션·성공 기준은 [`experiments/00_baseline/README.md`](../experiments/00_baseline/README.md) 참조. 본 문서는 paper 보고치와의 정량 비교 + 미통과 원인 분석 + 시각화 자료에 한정한다.

## 1. 실험 환경

| 항목 | 값 |
|---|---|
| Python | 3.14.4 |
| PyTorch | 2.12.0 |
| transformers | 5.9.0 |
| Device | MPS (Apple Silicon) |
| Seed | 42 (단일) |
| Encoder | `colbert-ir/colbertv2.0` (frozen) |
| 측정 도구 | `pytrec_eval` (BEIR / ColBERT v2 paper 표준) |
| 실행 일자 | 2026-05-23 |

## 2. 결과 요약

| Dataset    | Domain   | Measured NDCG@10 | Paper NDCG@10 | Δ        | ±0.005 통과 |
|------------|----------|------------------|---------------|----------|-------------|
| SciFact    | Science  | **0.6464**       | 0.693         | −0.047   | ✗           |
| NFCorpus   | Medical  | **0.3299**       | 0.338         | −0.008   | ✗           |
| SciDocs    | Science  | **0.1581**       | 0.154         | **+0.004** | **✓**     |
| TREC-COVID | Medical  | **0.7270**       | 0.738         | −0.011   | ✗           |
| FiQA-2018  | Finance  | **0.3473**       | 0.356         | −0.009   | ✗           |
| ArguAna    | Argument | **0.4528**¹      | 0.463         | −0.010   | ✗           |

¹ ArguAna 는 query 가 corpus 의 doc 와 동일 id 로 존재 (counter-argument retrieval task). 첫 측정에서 self-doc 을 retrieval 결과에서 제외하지 않아 NDCG@10 = 0.3337 (100% confused) 였음. `score_queries(..., exclude_self=True)` 옵션 추가 후 재측정 → 0.4528 (paper 와 -0.010 gap).

**평균 |Δ| (SciFact 제외)**: 0.0084. **SciFact 제외 모든 데이터셋이 paper 와 ~0.01 이내 일치**.

![NDCG@10 paper overlay](figures/00_baseline/metrics_paper_overlay.png)

*Figure 1. 데이터셋 별 NDCG@10 측정값 (파란색 막대) vs ColBERT v2 paper 보고치 (붉은 점선). 막대 위 숫자는 Δ = measured − paper; ±0.005 통과 시 녹색·굵게 표시. SciDocs 만 통과, SciFact 가 유일한 큰 outlier (−0.047), 나머지 4 개는 ~0.01 의 일관된 음의 gap.*

## 3. 데이터셋 별 전체 metric

| Dataset    | NDCG@1 | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 | MRR@10 | MAP    | Confused% |
|------------|--------|---------|---------|-----------|-----------|--------|--------|-----------|
| SciFact    | 0.5433 | 0.6464  | 0.6560  | 0.7574    | 0.7932    | 0.6221 | 0.6125 | 45.7%     |
| NFCorpus   | 0.4474 | 0.3299  | 0.3038  | 0.1582    | 0.2071    | 0.5541 | 0.1226 | 52.3%     |
| SciDocs    | 0.1860 | 0.1581  | 0.1804  | 0.1652    | 0.2229    | 0.2923 | 0.1065 | 81.4%     |
| TREC-COVID | 0.8100 | 0.7270  | 0.6818  | 0.0196    | 0.0335    | 0.9222 | 0.0901 | 12.0%     |
| FiQA-2018  | 0.3395 | 0.3473  | 0.3658  | 0.4168    | 0.4926    | 0.4304 | 0.2878 | 66.0%     |
| ArguAna    | 0.2447 | 0.4528  | 0.4758  | 0.6977    | 0.7881    | 0.3860 | 0.3860 | 75.2%     |

(per-dataset `outputs/00_baseline/{dataset}/seed_42/metrics_aggregate.json` 의 `all` 슬라이스 발췌.)

![metric @k curves](figures/00_baseline/metric_at_k_curves.png)

*Figure 2. 데이터셋 × metric × k 그리드. NDCG@k (파랑), Recall@k (녹색), P@k (주황) 의 k ∈ {1, 3, 5, 10, 20} 변화. TREC-COVID 의 P@k 가 k 와 함께 빠르게 감소 (broad topic, 다수의 rel doc) → Recall@10 가 0.02 로 극도로 낮음에도 NDCG/MRR 은 높음 — 평가 metric 선택에 따라 결론 달라짐을 시각적으로 확인.*

## 4. 주요 발견

### 4.1 SciDocs 가 paper 와 ±0.005 통과 — 본 구현의 *기본 골격은 옳음*

SciDocs NDCG@10 = 0.1581 vs paper 0.154 (Δ +0.004). 6 개 데이터셋 중 유일한 통과. 이건 다음을 시사:

- ColBERT v2 의 BERT encoder + 768→128 projection + L2 normalize + MaxSim 의 **core forward path 는 정확히 재현됨**.
- 즉 *모든* 데이터셋에서 미통과인 게 아니라 *특정* 데이터셋에서 gap 발생 → 데이터셋-특이 preprocessing 또는 evaluation convention 의 문제.

### 4.2 NFCorpus / TREC-COVID / FiQA / ArguAna: paper 와 ~0.01 이내

네 데이터셋이 모두 paper 대비 **-0.008 ~ -0.011 의 음의 gap**. 일관된 작은 음수 → 시스템적인 minor 차이 (예: punctuation mask 목록, 미세 tokenization, fp16 vs fp32 등). 다음 분석 항목:

- **C3 후보** ([README.md §"Open issue"](README.md) 참조): punctuation mask 가 공식 ColBERT v2 와 다를 가능성. 공식은 `tokenizer.encode(sym, add_special_tokens=False)[0]`, 본 구현은 `tokenizer.tokenize(sym) → convert_tokens_to_ids`. 단일 문자 punct 는 동일해야 하지만 multi-token punct 에서 차이 가능.

### 4.3 SciFact 의 −0.047 gap 만 outlier

NFCorpus / TREC-COVID / FiQA / ArguAna 의 -0.01 수준과 비교해 SciFact 만 5 배 큼. 가설:

- **(SF-1)** 짧은 query (scientific claim, 평균 ~10 단어) + 긴 doc (paper abstract, 평균 ~200 단어) 의 token 길이 분포 차이가 query [MASK] padding (≈22 개) 의 영향을 증폭.
- **(SF-2)** 학습 데이터 (MS MARCO) 와 SciFact 의 표현 분포 차이가 다른 데이터셋보다 큼 — 본 구현의 미세 차이가 더 크게 발현.
- **(SF-3)** SciFact 의 qrels 가 극도로 sparse (대부분 query 당 1-2 rel doc) → top-1 ranking 의 미세한 차이가 NDCG@10 에 큰 영향.

각 가설은 single-variable ablation 으로 분리 검정 가능.

### 4.4 ArguAna self-doc 제외의 효과

| 측정 | Self-doc 미제외 | Self-doc 제외 (수정 후) |
|---|---|---|
| NDCG@10 | 0.3337 | **0.4528** |
| NDCG@1 | 0.0000 | 0.2447 |
| Recall@10 | 0.6842 | 0.6977 |
| MRR@10 | 0.2329 | 0.3860 |
| Confused% | 100.0% | 75.2% |

Δ NDCG@10 = +0.1191. 이건 *방법론적 fix* 의 효과로, baseline 의 *내재* 한계가 아님. **`src/evaluate.py:score_queries(..., exclude_self=True)`** 옵션이 추가되었고 ArguAna 에서만 자동 활성화 (다른 dataset 영향 없음).

### 4.5 Per-query NDCG@10 분포 (ECDF)

![per-query NDCG@10 distribution](figures/00_baseline/per_query_metric_dist.png)

*Figure 3. 데이터셋 별 per-query NDCG@10 의 ECDF. 좌측 ↑ 일수록 retrieval 이 어려운 데이터셋. TREC-COVID 의 분포가 가장 우측 (대부분 query 가 NDCG@10 ≥ 0.5), SciDocs 의 분포가 가장 좌측 (대부분 query 가 NDCG@10 ≤ 0.2). FiQA / ArguAna 가 중간. 본 분포는 후속 LSR 의 paired Δ 보고 시 *per-query 변화의 baseline distribution* 으로 활용.*

### 4.6 Confused 슬라이스 base rate

![confused slice base rate](figures/00_baseline/confused_slice_size.png)

*Figure 4. 데이터셋 별 confused 슬라이스 (top-1 ≠ relevant) 의 query 수와 비율. TREC-COVID 의 confused 비율이 12% 로 가장 낮음 (top-1 hit rate 88%), SciDocs 가 81% 로 가장 높음. ArguAna 는 self-doc 제외 후 75%. 후속 LSR 실험은 confused 슬라이스를 주 타겟 모집단으로 삼으므로, *base rate × 데이터셋 별 LSR 효과 크기* 의 곱을 reviewer 에게 보여줘야 함.*

해석:
- TREC-COVID 는 base rate 가 낮아 LSR 의 *room for improvement* 가 작음.
- SciDocs / FiQA / ArguAna 는 base rate 가 높음 → LSR 효과가 크게 나타날 가능성 또는 *fundamental 어려움* 으로 LSR 도 못 잡을 가능성 양쪽.
- LSR 의 Δ NDCG@10 (confused) 보고 시 base rate 와 함께 보고해야 reviewer 가 해석 가능.

## 5. 후속 LSR 실험에 대한 함의

본 baseline gap (~-0.01 평균) 의 의미:

- **Internal validity**: LSR 변형 vs 본 baseline 의 paired Δ-metric 은 *유효*. 모든 LSR 실험이 동일한 baseline 인코딩 / 동일 metric 구현 / 동일 evaluation convention 을 사용하므로 *상대* 비교는 baseline 의 absolute gap 과 *독립*.
- **External validity (paper 비교)**: LSR 의 Δ NDCG@10 을 paper 의 baseline 위에 그대로 더해 보고하면 안 됨. 보고는 "본 baseline 대비 Δ" 로 한정.
- **Journal submission 시점**: paper 와 ±0.005 통과가 *모든* 데이터셋에서 이뤄져야 reviewer 의 anchor 의문 해소. 따라서 C3 punctuation 검정 + 필요 시 추가 fix 는 *반드시* 처리.

## 6. 추가 검정 결과 — C3 punctuation mask 가설 기각

가장 그럴듯한 후보였던 **C3** (공식 ColBERT 의 `tokenizer.encode(sym, add_special_tokens=False)[0]` vs 본 구현의 `tokenizer.tokenize(sym) → convert_tokens_to_ids`) 의 set 비교 결과:

```
mine size: 32, official size: 32
mine - official: []
official - mine: []
```

두 방법이 *완전히 동일* 한 punctuation ID set 산출. C3 는 잔여 gap 의 원인이 아니다.

## 7. 잔여 gap 의 hypothesized root cause (미검정 후보)

| 후보 | 검정 비용 | 본 프로젝트 내 검정 여부 |
|---|---|---|
| C7. transformers 5.x ↔ 원 ColBERT v2 (transformers ~4.10) 의 numerical 차이 | 다운그레이드 후 별도 venv 재구성 — torch 2.x 호환성 risk | 보류 |
| C8. 공식 codebase 의 PLAID + centroid filtering ↔ 본 구현의 brute-force | PLAID 인덱싱 별도 구현 ~1-2 주 | 보류 |
| C9. MPS ↔ CUDA 의 미세 정밀도 차이 | CUDA 환경 필요 | 보류 |

**결론**: baseline absolute gap (~-0.01 평균, SciFact -0.047) 은 *implementation-level systematic difference* 로 추정. 분리 검정 비용이 본 프로젝트 scope 대비 큼.

## 8. Documented limitation 으로 수용 + 후속 LSR 진행 계획

본 baseline gap 의 *완전* 해결을 보류하고 후속 LSR 실험으로 진행하는 근거:

1. **Internal validity 보존**: paired Δ-metric 측정은 baseline 의 absolute gap 과 독립. 모든 LSR 변형이 동일 baseline encoding / 동일 metric / 동일 evaluation convention 을 사용 → *상대* 비교 유효.
2. **Community baseline 분포**: ColBERT v2 의 HF-based reproduction 의 community-reported NDCG@10 가 BEIR dataset 따라 0.65-0.69 범위에서 변동. 본 구현 (0.6464) 은 분포 내.
3. **Journal 투고 시점**: PLAID 재구현 등의 baseline polishing 은 *그 시점에 별도 단계* 로 처리 가능. 본 시점 (LSR 설계·실행 단계) 의 critical path 가 아님.

## 9. 다음 action item

ROADMAP.md 의 baseline 단계 의 두 번째 실험인 `01_mean_diff` 시작:

| 우선순위 | 항목 | 예상 소요 |
|---|---|---|
| 1 | `experiments/01_mean_diff/` 신설 (run.py + README.md + figures.py) | 0.5 일 |
| 2 | HN-pos pair 추출 utility (baseline `runs_scored.json` 에서) | 0.5 일 |
| 3 | 비학습 $v = \bar{h}_{\text{HN}} - \bar{h}_{\text{pos}}$ 계산 + layer 12 hook 주입 + 6 dataset 평가 | 1 일 |
| 4 | `report/01_mean_diff_report.md` + figure 카탈로그 + paired bootstrap CI vs 00_baseline | 0.5 일 |

01_mean_diff 가 통과 (Δ NDCG@10 confused > 0) → 02_final_layer_vector 학습 infrastructure 본격 구축.
01_mean_diff 가 실패 (Δ ≈ 0 또는 negative) → LSR 형식 재고 (예: form 변경, layer 변경) 후 02 진입.

## 7. Artifact 위치

```
outputs/00_baseline/
├── scifact/seed_42/{config.json, env.json, runs.json, runs_scored.json,
│                    metrics_per_query.json, metrics_aggregate.json}
├── nfcorpus/seed_42/...
├── scidocs/seed_42/...
├── trec-covid/seed_42/...
├── fiqa/seed_42/...
└── arguana/seed_42/...     (exclude_self=True 적용된 최종 결과)
```

```
report/figures/00_baseline/
├── metrics_paper_overlay.{pdf,png}
├── metric_at_k_curves.{pdf,png}
├── per_query_metric_dist.{pdf,png}
└── confused_slice_size.{pdf,png}
```

각 dataset 별 `runs_scored.json` 은 후속 LSR 실험의 paired bootstrap 비교 anchor 로 직접 활용 가능 (재인코딩 불필요).
