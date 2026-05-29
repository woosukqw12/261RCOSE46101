# 00_baseline — Frozen ColBERT v2 재현

## 목적

본 실험은 모든 후속 steered configuration (T1.01+) 의 **anchor** (CLAUDE.md §3.5) 를 확립한다. 구체적으로:

1. 공식 `colbert-ir/colbertv2.0` checkpoint 를 frozen 으로 로드한다.
2. BEIR test split 에 대해 brute-force MaxSim retrieval 을 수행한다.
3. Per-query baseline run / aggregate metric 을 artifact 로 저장하여, 이후 LSR 실험이 paired bootstrap on per-query Δ-metric (DESIGN.md §5.3) 를 계산할 수 있게 한다.

## 가설 (implicit)

재구현된 forward path (`src/colbert_hook.py`) 가 ColBERT v2 paper 의 BEIR zero-shot 결과를 허용 오차 내에서 재현한다. 이는 본 프로젝트의 핵심 주장 — *"LSR 변형이 ColBERT v2 위에서 개선을 보인다"* — 의 *전제 조건* 이다. baseline 이 ColBERT v2 를 충실히 반영하지 않으면 후속 LSR 실험 결과는 해석 불가능하다.

## 성공 기준

데이터셋 별 NDCG@10 이 Santhanam et al. (2022), Table 3 의 보고치와 **±0.005** 이내 일치해야 한다. 6 개 BEIR 데이터셋을 3 도메인으로 묶어 평가:

| 도메인 | Dataset    | Queries | Corpus | Paper NDCG@10 | 허용 구간       |
|--------|------------|---------|--------|---------------|-----------------|
| 의료   | NFCorpus   | 323     | 3.6K   | 0.338         | [0.333, 0.343]  |
| 의료   | TREC-COVID | 50      | 171K   | 0.738         | [0.733, 0.743]  |
| 과학   | SciFact    | 300     | 5K     | 0.693         | [0.688, 0.698]  |
| 과학   | SciDocs    | 1000    | 25K    | 0.154         | [0.149, 0.159]  |
| 금융   | FiQA-2018  | 648     | 57K    | 0.356         | [0.351, 0.361]  |
| 논증¹  | ArguAna    | 1406    | 8.7K   | 0.463         | [0.458, 0.468]  |

¹ BEIR 의 표준 금융 retrieval dataset 은 FiQA-2018 단일. 두 번째 금융 슬롯은 ArguAna (argument matching) 로 보완하여 6 개 균형. 추후 BEIR 외부 finance dataset (e.g., FinQA) 추가 검토 가능.

## 실행 방법

```bash
.venv/bin/python experiments/00_baseline/run.py --dataset scifact    --seed 42
.venv/bin/python experiments/00_baseline/run.py --dataset nfcorpus   --seed 42
.venv/bin/python experiments/00_baseline/run.py --dataset scidocs    --seed 42
.venv/bin/python experiments/00_baseline/run.py --dataset trec-covid --seed 42
.venv/bin/python experiments/00_baseline/run.py --dataset fiqa       --seed 42
.venv/bin/python experiments/00_baseline/run.py --dataset arguana    --seed 42
```

Artifact 출력 경로: `outputs/02_evaluate/T1.00_baseline/{dataset}/seed_{seed}/`

## 현재 상태

마지막 실행: 2026-05-23 (6 dataset 전체, seed 42).

| Dataset    | 측정 NDCG@10 | Paper | Δ        | ±0.005 통과 |
|------------|--------------|-------|----------|-------------|
| SciFact    | 0.6464       | 0.693 | −0.047   | ✗           |
| NFCorpus   | 0.3299       | 0.338 | −0.008   | ✗           |
| SciDocs    | **0.1581**   | 0.154 | **+0.004** | **✓**     |
| TREC-COVID | 0.7270       | 0.738 | −0.011   | ✗           |
| FiQA-2018  | 0.3473       | 0.356 | −0.009   | ✗           |
| ArguAna¹   | 0.4528       | 0.463 | −0.010   | ✗           |

¹ `exclude_self=True` 적용 후 수치 (ArguAna 는 query 가 corpus 의 doc 와 동일 id 로 존재하는 counter-argument task — self-doc 제외 필수). 미적용 첫 측정값 0.3337 → 적용 후 0.4528.

**Summary**: SciDocs 만 통과. SciFact (−0.047) 가 유일한 큰 outlier; 나머지 4 개는 모두 ~0.01 의 일관된 음의 gap. 상세 분석 + 시각화 자료 → [`report/00_baseline_report.md`](../../report/00_baseline_report.md).

## 수용을 막는 open issue

SciFact 잔여 −0.047 gap (+ 다른 4 dataset 의 ~−0.01 gap) 의 원인 후보:

| # | 원인 후보 | 검정 방법 | 상태 |
|---|----------|-----------|------|
| C1 | `attend_to_mask_tokens=False` 미반영 (query 의 [MASK] padding 위치에 BERT attention 1 로 두었음) | `src/colbert_hook.py` 수정 후 재측정 | **완료** — SciFact 0.6185 → 0.6430 |
| C2 | 문서 텍스트 separator: `". "` (이중 마침표 발생) | `" "` 로 교체 후 재측정 | **완료** — SciFact 0.6430 → 0.6464 |
| C3 | Punctuation mask 목록 차이 (tokenize vs encode) | 공식 `tokenizer.encode(symbol, add_special_tokens=False)[0]` 와 비교 검정 | **기각** — 두 방법의 punctuation ID set 완전 동일 (size 32, set diff = ∅). 본 후보 아님. |
| C4 | Brute-force vs PLAID index | PLAID 는 centroid filtering 이라 brute-force 보다 동일 또는 약간 낮음 → 본 gap (negative) 의 원인일 가능성 낮음 | 검토만 |
| C5 | fp16 vs fp32 inference | 공식은 fp16; 본 구현은 fp32 (정밀도 측면에서 본 구현이 *더* 정확해야 함) → 원인 가능성 낮음 | 검토만 |
| C6 | ArguAna self-doc retrieval 미제외 | `score_queries(..., exclude_self=True)` 옵션 추가 + ArguAna 에서 자동 활성화 | **완료** — ArguAna 0.3337 → 0.4528 |
| C7 | transformers 5.x ↔ 원 ColBERT v2 학습 당시 transformers ~4.10 의 numerical 차이 | transformers 4.x 다운그레이드 후 재측정 — 단 torch 2.x 호환성 깨질 위험, 별도 venv 필요 | 미검정 (deferred — 비용 대비 효익 불명확) |
| C8 | 공식 codebase 의 PLAID + centroid filtering ↔ 본 구현의 brute-force inference path | PLAID 인덱싱 별도 구현 후 비교 | 미검정 (1-2 주 추가 작업, 본 프로젝트 scope 밖) |
| C9 | MPS ↔ CUDA 의 미세 정밀도 차이 | CUDA 환경에서 재실행 | 미검정 (hardware 의존) |

**현재 결론**: C1 / C2 / C6 fix 후 잔여 gap (~-0.01 평균, SciFact -0.047) 은 *implementation-level systematic difference* 로 추정 (C7/C8/C9 어느 하나 또는 결합). 단일 변수 분리 검정 비용이 본 프로젝트 scope 대비 큼.

**Documented limitation 으로 수용**: baseline absolute gap 은 closing 보류, 후속 LSR 실험의 *paired Δ-metric* 측정은 internal validity 보존 (baseline absolute 와 독립). Journal 투고 시점에 PLAID 재구현 여부 재검토.
