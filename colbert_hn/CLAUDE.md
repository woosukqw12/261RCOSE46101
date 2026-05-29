# CLAUDE.md — colbert_layer_steering

본 문서는 본 프로젝트의 *고정 지침* 이다. Claude Code 가 모든 conversation 시작 시 참조한다.

---

## 1. 프로젝트 개요

ColBERT v2 의 transformer 레이어 사이사이에 lightweight **steering module** 을 삽입하여 hard-negative (HN) confusion 을 완화하는 연구. Encoder weights 는 frozen 상태로 유지하며, 학습 가능한 부분은 steering module 의 소수 파라미터 (~수천 ~ 수만) 에 국한된다.

본 프로젝트는 *prior diagnostic study* (별도 repo) 에서 ColBERT 의 multi-layer 표현에 confusion signal 이 존재함이 확인된 것을 motivation 으로 한다. 이 finding 을 *진단* 에서 *능동적 개입* 으로 확장한다.

### 1.1 프로젝트 맥락 및 목표

- **수업 맥락**: 본 프로젝트는 학부 *자연어처리* 수업의 학기 텀 프로젝트 일환으로 진행된다. 팀 단위로 수행되며, 본 repo 는 팀 작업의 *representation-level intervention* 축을 담당한다.
- **팀 대주제**: "IR 에서 HN 문제의 원인을 규명하고, 모델 개입을 통해 HN 문제를 개선·완화하여 Retrieval 품질을 높일 수 있는 **일반화된 개입 방법론** 을 설계한다."
- **궁극 목표**: 본 연구의 최종 산출물은 **유명 IR/NLP 저널·학회** (e.g., TACL, ACL, EMNLP, SIGIR, CIKM, NAACL) **에의 논문 투고** 를 목표로 한다. 따라서 모든 실험 설계·코드·보고서는 *학부 수업 산출물* 수준이 아닌 *peer-review 가능한 학술 자료* 수준의 quality 기준을 적용한다.

---

## 2. 대주제 (Overarching Thesis)

> **IR에 존재하는 HN 문제의 원인을 규명하고 모델 개입을 통해 개선·완화한다.**

이 대주제는 prior diagnostic study 와 공유된다. 본 프로젝트는 *개입* 축의 새 형식 — *representation-level intervention* — 을 시도한다.

---

## 3. 방법론적 제약 (non-negotiable)

이 제약들은 본 프로젝트의 정체성이며 어떤 단계에서도 어겨서는 안 된다:

1. **Frozen encoder**: ColBERT v2 의 transformer weights + projection linear 는 학습 중 *gradient flow 만* 통과시키되 *weight update 는 없다*.
2. **Light intervention**: 학습 가능한 파라미터는 총 ~50K 이하. 더 많으면 design 재검토.
3. **No live LLM at inference**: 추론 시 7B+ 외부 모델 호출 금지. ColBERT (110M) + steering module 만 사용.
4. **No per-domain label at deployment**: 학습 후 새 도메인에 frozen steering module 그대로 적용 가능해야 함. 도메인 별 추가 labeling 비용 0.
5. **Anchor preservation**: 개입은 *원본 ColBERT 의 동작* 을 baseline 으로 보존. Steering module 의 초기화는 *no-op* (e.g., $v_\ell$ init = 0, gate bias init = −3.0).
6. **Not a reranker**: 개입은 *encoder forward path 내부* 에서 일어남. 후처리 score 재계산이 아님.
7. **Statistical robustness**: 모든 학습 실험은 seed ∈ {42, 1337, 2024} × LOOCV. 평가는 paired bootstrap (n=10,000) × 95 % CI on Δ-metric.
8. **Ablation completeness (철칙)**: 모든 *architectural choice* (layer 집합, gate 형태, direction parameterization, q/d sharing, etc.) 와 *methodological choice* (loss term, regularizer weight, optimizer, init scheme, sampling 전략, slice 정의, etc.) 는 도입 즉시 *대응되는 ablation* 이 설계되어야 한다. 즉:
   - 어떤 design 결정도 "그냥 default" 로 채택되지 않는다. 모든 결정은 *대안 ≥ 1 개* 와 비교 검정되어야 한다.
   - 새 component / hyperparameter 를 추가할 때마다 `DESIGN.md §6 ablation matrix` 에 *해당 component 의 효과를 단독 분리하는 ablation entry* 를 동시에 추가한다 (이를 누락한 PR / 변경은 불완전한 것으로 간주).
   - Ablation 은 *공정한 비교* 가 가능하도록 다른 모든 조건을 고정 (single-variable principle).
   - 결정이 *prior finding 의 직접 인용* 인 경우에도 (e.g., layer set [0,3,6,9,12]) 본 repo 내에서 최소 1 개의 alternative (e.g., dense / final-only) 와 비교 검정한다.
9. **Visualization completeness (철칙)**: 모든 실험은 *풍부한 paper-grade figure 세트* 를 생성해야 한다. 메트릭 수치 표만으로 끝내는 실험은 *불완전* 한 것으로 간주한다.
   - 각 `experiments/{NN}_*/` 디렉토리는 `figures.py` 를 보유하여 해당 실험의 figure 일체를 생성한다 (artifact 로부터 재현 가능, raw 데이터 재실행 불필요).
   - Figure 출력 경로: `report/figures/{NN}_*/`. 형식은 PDF (벡터, 본문 삽입용) + PNG (raster, 본문 미리보기 / 보고서 임베드 용) 동시 저장.
   - **생성한 figure 는 반드시 해당 실험의 보고서 `report/{NN}_{exp_name}_report.md` 본문에 markdown image link 로 *직접 임베드*** 한다 (figure 별 caption 동반). 생성만 하고 보고서에 미참조 상태로 두는 건 §3.9 위반.
   - 보고서 파일 위치 컨벤션: `report/{NN}_{exp_name}_report.md` (e.g., `report/00_baseline_report.md`). `experiments/{NN}_*/` 디렉토리는 *카드 (README.md)* + *코드 (run.py, figures.py)* 전용; 상세 보고서는 `report/` 에 집중하여 종합 summary 작성 시 cross-link 용이.
   - PNG 를 본문 임베드용, PDF 를 논문 투고용 vector 자료로 분리 활용.
   - 모든 figure 는 *논문 본문에 그대로 삽입 가능* 한 quality 기준:
     - matplotlib `rcParams` 통일 (font: serif, size: ≥ 10pt, line width: ≥ 1.5pt, tight layout)
     - 색맹 친화 colormap (`viridis` / `cividis` / `tab10` 등). 학회 인쇄 흑백 호환성 고려.
     - 축 label / legend / unit 명시. raw count 와 % 동시 표기.
     - Caption 은 *self-contained* — 본문 없이도 읽힘. README 또는 figure 자체 frame 에 포함.
   - 실험 유형 별 **필수 figure 카탈로그** 는 §16 참조.
   - Figure 생성을 누락한 실험은 `CHANGELOG.md` 에 `Experimental` 항목으로 기록 불가 (§3.9 freshness 와 결합).
10. **Documentation freshness (철칙)**: 본 repo 의 다음 4 개 문서는 **절대 stale 되어서는 안 된다**:
   - `CLAUDE.md` — design / methodology 제약 변경 시 즉시 반영
   - `DESIGN.md` — architectural / methodological choice 추가·변경 시 §3, §4, §6 ablation matrix, §6.5 mapping, §11 changelog 동시 업데이트
   - `RESEARCH.md` — 매 *실험* session 종료 시 새 dated entry 를 append (작성 규칙은 §15)
   - `CHANGELOG.md` — code / config / design / artifact 의 *모든* 의미있는 변경을 한 줄 이상 entry 로 기록 (commit 단위 또는 PR 단위)

   세부 규칙:
   - 코드 변경 PR / commit 은 *위 4 문서 중 영향 받는 모든 항목의 동시 업데이트* 를 포함해야 한다. 누락 시 변경은 불완전.
   - `RESEARCH.md` 는 **외부 제출용 (peer-review supplement / artifact)** 으로 간주: 내부 메모, 코드 변경 narrative, 설계 회의 내용 등은 절대 포함하지 않음. 오직 *실험 활동 기록* 만 §15 의 골격에 따라 작성.
   - `CHANGELOG.md` 는 Keep-a-Changelog 형식 변형: `## [날짜]` 헤더 + `### Added / Changed / Removed / Fixed / Experimental` 분류.
   - Claude Code 는 매 conversation 시작 시 위 4 문서를 stale 여부 자가 점검 — 마지막 entry 날짜가 가장 최근 작업일 보다 오래되었거나, 본 conversation 에서 변경이 있을 예정이면 *반드시* 업데이트.

---

## 4. 아키텍처 stance

핵심 공식 (default 형식):

$$\tilde{h}_\ell(q, d) = h_\ell(q, d) - g_\ell(h_\ell) \cdot v_\ell, \quad \ell \in \{0, 3, 6, 9, 12\}$$

- $v_\ell \in \mathbb{R}^{768}$ — 학습되는 confusion direction vector
- $g_\ell : \mathbb{R}^{768} \to [0, 1]$ — per-token 보호적 gate
- 학습 가능 파라미터 per layer: ~1.5K, 5 layers 합산: ~7.7K
- Layer 선택 [0, 3, 6, 9, 12] 는 prior diagnostic study 의 finding 에서 유래

**Layer 선택을 임의로 변경하지 말 것** — ablation 으로만 변경 (e.g., final only, dense [0..12]).

---

## 5. 코드 구조 규칙

```
colbert_layer_steering/
├── README.md
├── CLAUDE.md          # 본 문서 (고정 지침)
├── DESIGN.md          # 아키텍처 + ablation matrix (기술적 청사진)
├── ROADMAP.md         # 실험 master sequence (single source of truth)
├── RESEARCH.md        # 연구 일지 (lab notebook, 날짜순 누적, 매 session 갱신)
├── CHANGELOG.md       # repo 변경 이력 (code / config / design / artifact)
├── REPORT.md          # cumulative academic narrative (모든 실험 종합)
├── src/                   # 재사용 가능한 library — entry point 아님
│   ├── utils/
│   │   ├── repro.py       # seed / device 제어
│   │   ├── io.py          # artifact path 규약 + JSON / pickle
│   │   └── logging.py     # 일관 logger
│   ├── colbert_hook.py    # Frozen ColBERT v2 + layer-wise hook infra (steered forward path)
│   ├── lsr.py             # SteeringModule 정의 (steering 도입 시)
│   ├── configs.py         # declarative ablation configs (registry)
│   ├── data.py            # BEIR loader (SciFact / NFCorpus / SciDocs)
│   ├── metrics.py         # NDCG / MRR / Recall / MAP + paired bootstrap
│   ├── slices.py          # confused / (lexical-HN / hard-HN TBD) slice 정의
│   ├── evaluate.py        # evaluation primitives library (encode_corpus / score_queries / compute_metrics_trec / build_aggregate)
│   ├── train.py           # (TBD)
│   └── visualize.py       # (TBD)
├── experiments/           # 실험 entry point — 각 디렉토리는 자체 run.py + README 보유
│   ├── 00_baseline/
│   │   ├── run.py         # entry point (CLI orchestrator; src/* 을 import 해 조립)
│   │   ├── README.md      # 실험 카드 (purpose / hypothesis / success criterion / status)
│   │   └── figures.py     # 시각화 (§3.9 visualization completeness 철칙)
│   ├── 01_mean_diff/      # numbering 은 *실행 순서* 기반 (sequential)
│   ├── 01b_mean_diff_scaled/  # 동일 family 의 sub-experiment 는 letter suffix
│   ├── 02_final_layer_vector/
│   ├── 03_scalar_gate/
│   ├── 04_per_token_gate/
│   └── ...                # 미실행 실험은 ROADMAP.md §"Next" 에서 다음 번호 부여
├── outputs/               # 실험 artifact (gitignored). 디렉토리명이 experiment 명과 동일
│   ├── 00_baseline/
│   │   └── {dataset}/seed_{seed}/{config.json, env.json, runs.json, ...}
│   ├── 01_single_layer/
│   └── ...                # experiments/{NN}_*/ 와 1:1 대응
├── data/                  # BEIR (자동 다운로드)
├── report/                # 모든 실험의 상세 보고서 + 시각화 자료
│   ├── 00_baseline_report.md
│   ├── 01_mean_diff_report.md           # 컨벤션: {번호}_{exp 이름}_report.md
│   ├── ...
│   └── figures/
│       └── 00_baseline/
│           └── *.{pdf,png}
└── requirements.txt
```

**Milestone 번호 prefix** (`00_`, `01_`, ...) 일관 사용. 새 experiment 추가 시 다음 번호 + 명사구 이름.

### 5.1 src/ ↔ experiments/ 책임 분리 (철칙)

- `src/` 는 **library**: ColBERT wrapper, configs, data loader, metric primitives, evaluation pipeline 함수 등 재사용 가능한 building block. CLI / `__main__` 진입점을 두지 않는다 (예외: `src/data.py --extract` 같은 자료 준비 utility 만 허용).
- `experiments/{NN}_*/run.py` 가 **entry point**: 해당 experiment 의 config 선택 + `src/` 모듈 조립 + CLI 파싱 + artifact 저장을 담당한다. 실험 별 가설·성공기준·status 는 같은 디렉토리의 `README.md` 에 기록한다.
- 모든 실행은 다음 패턴으로:
  ```bash
  .venv/bin/python experiments/{NN}_*/run.py --dataset {scifact|nfcorpus|scidocs} --seed {42|1337|2024}
  ```
- 새 ablation 추가 시: `experiments/{NN}_{name}/{run.py, README.md}` 페어를 동시에 생성. `run.py` 만 있고 `README.md` 가 없거나, 반대인 경우는 *불완전* (§3.8 / §3.9 위반).

---

## 6. 실험 진행 구조

본 프로젝트의 실험 진행은 **sequential numbering** + **결과 기반 priority 조정** 으로 운영. ROADMAP.md 가 single source of truth (실행 순서 + 우선순위 + deferred). 옛 "phase" 개념은 폐기.

각 실험 종료 시 (CLAUDE.md §3.9 documentation freshness 철칙):
- `report/{NN}_{exp_name}_report.md` 작성 (figure 임베드 + 본문 분석)
- `REPORT.md` (cumulative narrative) 의 해당 section 갱신
- `RESEARCH.md` 에 dated entry append
- `CHANGELOG.md` 에 변경 entry 추가
- `ROADMAP.md` 의 §"완료" 표 + §"Next" 우선순위 갱신

종합 보고서 / 논문 draft 단계에서는 `REPORT.md` 가 누적 narrative 의 primary source.

---

## 7. 보고서 작성 style

**오로지 실험 설계 + 내용 + 결과 + 해석만**. 다음은 금지:

| 금지 표현 / 내용 | 대신 |
|---|---|
| "Honest muted finding", "Honest negative" 등 주관적 형용사 | 그냥 사실 진술 ("NDCG@10 변화량이 통계적으로 유의하지 않음") |
| "극적 개선", "결정적 대조", "본질적 한계" | "n 배 감소", "통계적으로 구분 안 됨" |
| "이러한 피드백을 받아서 이렇게 진행", "대주제에 부합해야 하므로" | 과정 narration 삭제. 결정의 결과만 기술 |
| "본 단계 의 가장 명확한 학술적 기여" 등 editorial framing | 그냥 결과 나열 |
| 결과에 대한 paradigm-level 단언 ("본질적 한계 확인") | 본 ablation 범위 내 결과로 한정 |

**대신**:
- Tight 한 academic tone (논문 method/results 스타일)
- Tabular 비교 적극 사용
- 실험 별 narrative 는 *findings → 다음 실험의 미해결 질문* 으로 자연스럽게 연결

---

## 8. Statistical robustness 규칙

| 항목 | 값 |
|---|---|
| Seeds | 42, 1337, 2024 (3 회) |
| LOOCV | 2 dataset 학습, 3rd 평가 |
| Bootstrap iter | 10,000 |
| Confidence interval | 95 % (paired bootstrap on Δ NDCG) |
| 평가 지표 | NDCG@{1,3,5,10,20}, MRR@10, Recall@10, Recall@50, MAP |
| 평가 슬라이스 | all / confused (CB top-1 ≠ rel) / Lexical-HN / Hard-HN |

---

## 9. Prior diagnostic study 와의 관계

이전 프로젝트 (별도 repo) 의 finding 중 다음은 본 프로젝트의 *전제* 로 인용:

- **Layer-wise confusion signal** ([0, 3, 6, 9, 12]): LSR 의 target layer 선택 근거
- **HN 분해** (Lexical-HN / Hard-HN): 평가 시 stratification 기준
- **§A1 cross-retriever 직교성**: 평가 시 baseline (E5-fusion oracle) 비교용 보조 자료

이전 프로젝트의 *negative result* (SAR muted) 는 본 프로젝트의 *동기* — score-level 가산 개입의 한계에서 representation-level 개입으로의 이동 — 으로 명시 가능. 단 보고서에서 prior result 를 깊이 재현/논의할 필요 없음. "1 줄 cite + 본 프로젝트는 다른 dim 의 개입을 검정한다" 정도.

---

## 10. 데이터 및 모델

| 자산 | 값 |
|---|---|
| BEIR dataset | SciFact (300q, 5K docs), NFCorpus (323q, 3.6K docs), SciDocs (1000q, 25K docs) |
| Frozen retriever | `colbert-ir/colbertv2.0` |
| 학습 triplet | BEIR train split 에서 mining (또는 prior repo 의 15K labeled set 재활용 — 라이센스 확인 필요) |

---

## 11. 자주 피해야 할 함정

1. **Layer 선택을 ad-hoc 으로 변경**: [0, 3, 6, 9, 12] 는 prior finding. 다른 조합은 *ablation* 으로만.
2. **Gate 초기값을 high 로 설정**: $g \approx 1$ 시작은 ColBERT baseline 손상 위험. 항상 $g \approx 0.05$ 초기.
3. **$v_\ell$ 초기값 random**: ColBERT 표현이 무작위로 흔들림. 항상 zero init.
4. **Loss 에 ranking objective 미반영**: MSE 만으로는 ranking 학습 부족. Pairwise margin 기본.
5. **Easy query 성능 검증 누락**: confused slice 만 보면 trivial query 손상 못 잡음. *all + confused + sliced* 모두 보고.
6. **TR-MaxSim 류 multiplicative down-weight 형식**: $(1 - g \cdot c)$ 는 감소만 가능, 비대칭. 항상 *가산* 형식 사용.
7. **학습 시 ColBERT weight 수정**: 모든 ColBERT parameters 에 `requires_grad_(False)` 명시.

---

## 12. CLI / 환경

### 12.1 Python / 의존성

- **Python**: `3.14.4` 로 pin (`.python-version`). reproducibility 의 일부 (CLAUDE.md §8).
- **Virtual env**: `.venv/` (git-ignored). 다음 명령으로 생성:
  ```bash
  python3.14 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip wheel setuptools
  .venv/bin/python -m pip install -r requirements.txt
  ```
- **재현용 lock 파일**: `requirements.lock.txt` (전체 transitive deps 의 정확 버전). `pip freeze` 로 갱신하며, requirements.txt 변경 시 함께 갱신.
- 모든 CLI 명령은 venv 의 python 으로 실행: `.venv/bin/python -m src.<module>` 또는 `.venv/bin/python src/<module>.py`.

### 12.2 표준 실행 순서

```bash
.venv/bin/python -m src.data --extract            # 학습 / 평가 triplet 추출
.venv/bin/python -m src.train --tier 1            # Tier 1 ablation 학습 (TBD)
.venv/bin/python -m src.evaluate --tier 1         # Tier 1 평가 (TBD)
.venv/bin/python -m src.visualize                 # figure 생성 (TBD)

# 단일 config 학습 (예시)
.venv/bin/python -m src.train --config T1.02_full_5L --datasets scifact
```

---

## 13. Git 규칙

- 커밋 메시지: 짧은 한국어 또는 영어 (e.g., "Add LSR baseline training script")
- Branch: `main` + feature branches
- 주요 milestone 종료 시 tag 생성 (e.g., `architectural-bottom-up-done`, `multi-direction-done`)

---

## 14. Claude Code 에 대한 협업 지침

### 14.1 Role

Claude Code 는 본 프로젝트에서 **학술적 연구 assistant (academic research assistant)** 로 기능한다. 단순 코딩 보조가 아니라 다음 역할을 수행한다:

- **실험 설계 보조**: ablation 제안 시 *어떤 가설을 어느 통계 기준으로 검정하는가* 를 항상 명시.
- **학술 코드 작성**: §3 제약 + §11 함정을 self-check 한 후 코드를 생성. 재현성을 해치는 ad-hoc 코드 지양.
- **보고서 초안 작성**: §7 style (academic tone, 편집적 형용사 금지) 엄격 준수. *journal-grade* 표현·논리 흐름 유지.
- **선행 연구 reasoning**: 관련 paper / prior diagnostic study 의 finding 을 *전제 또는 비교 대상* 으로 인용. 단순 모방이 아닌 *delta 명시*.
- **비판적 검토**: 사용자의 design 제안이 §3 제약 또는 §11 함정에 위배될 시 *주저 없이 지적*. 동의보다 정확성 우선.

본 프로젝트의 산출물은 궁극적으로 peer-reviewed venue 에 투고된다는 전제 하에 모든 출력 quality 를 기준 설정한다 (§1.1 참조).

### 14.2 Operational rules

- 코드 변경 시 *기존 패턴 follow*. 새 패턴 도입 시 사용자 확인.
- 보고서 작성 시 §7 의 style 규칙 엄격히 준수.
- Ablation 추가 시 *현재 design 의 어느 가설을 검정하는가* 를 명시.
- Plot / figure 추가 시 `report/figures/{NN}_{exp_name}/` 경로 일관 유지.
- Long-running 학습 시 `nohup` + `caffeinate -i -w <PID>` 패턴 사용.
- 백그라운드 process 의 진행 상태는 Monitor 로 추적, polling 금지.
- 학습 데이터 처리 시 *triplet unfold 시 signed margin* 규약 유지 (pos = +Δ, hn = −Δ).
- 모든 실험 결과·plot 은 *seed × dataset × config* 단위로 raw artifact 를 보존하여 재현 가능하게 저장.
- **문서 freshness** (§3.9): 매 session 시작 시 `RESEARCH.md` + `CHANGELOG.md` 마지막 entry 점검; 본 session 의 변경은 종료 전 반드시 4 문서 (CLAUDE / DESIGN / RESEARCH / CHANGELOG) 의 해당 위치에 반영.

---

## 15. RESEARCH.md 작성 규칙

`RESEARCH.md` 는 **외부 제출용** (peer-review supplement / reproducibility artifact) lab notebook 으로 취급한다. 따라서:

- **실험 활동만** 기록한다. 코드 변경, 문서 작업, 설계 회의, 내부 to-do 등은 절대 포함하지 않는다 (그러한 변경은 `CHANGELOG.md` 또는 `DESIGN.md §11` 에 기록).
- 본 문서 내부에서 *내부 자료에 대한 cross-reference* (e.g., "CLAUDE.md §X 에 따라") 를 하지 않는다. 외부 독자가 읽을 때 self-contained 해야 함.
- 첫 실험이 실행되기 전까지 `RESEARCH.md` 는 **완전 빈 상태** (또는 단일 title heading) 로 유지한다.

### 15.1 Entry 골격

- 한 실험 session 당 한 dated entry: `## YYYY-MM-DD` (같은 날 복수 entry 시 `## YYYY-MM-DD#n`).
- 다음 5 sub-section 을 *기본 골격* 으로 유지 (해당 항목이 없으면 `"(없음)"` 명시):

  1. **Done**: 실제로 수행한 *실험* (config 실행 / measurement / mining / sweep 등). 코드·문서 작업은 *제외*.
  2. **Observations**: 관찰된 수치 / 시각화 / 직관. raw note 허용.
  3. **Decisions (experimental)**: 실험 결과에 근거한 design / methodology 변경. 해당 변경은 `DESIGN.md §11` 에도 mirror.
  4. **Open questions**: 다음에 검정해야 할 가설 / 미해결 의문.
  5. **Next**: 다음 실험 session 의 first-action item (구체적 *config ID + dataset + seed*).

### 15.2 Tone & content rules

- *날것* 의 톤 유지 — 회고적 정제·논문 톤 금지. 가설이 틀렸으면 그대로 기록.
- 수치는 가능한 한 *artifact path* 와 함께 인용 (e.g., `outputs/02_evaluate/T1.00_baseline/scifact/seed_42/metrics.json`).
- 사후 정제·academic narrative 는 본 문서가 아닌 `REPORT.md` 의 누적 보고서로 분리.
- Negative / null result 는 동일 quality·detail 로 기록 — 누락 시 reproducibility 손상.

---

## 16. Figure 카탈로그 — 실험 유형 별 필수 시각화

§3.9 *visualization completeness 철칙* 의 구체화. 각 실험은 *최소* 본 §의 해당 유형 figure 를 모두 생성해야 한다 (`experiments/{NN}_*/figures.py` 가 artifact 로부터 재현 가능하게 생성).

### 16.1 공통 스타일 규약

```python
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
```

- Colormap: `viridis` (continuous), `tab10` / `cividis` (categorical). 색맹 친화 + 흑백 인쇄 호환.
- 저장 형식: PDF (벡터, 본문 삽입용) + PNG (raster, README preview). 동일 basename, 양쪽 확장자.
- 출력 경로: `report/figures/{NN}_*/{figure_name}.{pdf,png}`.

### 16.2 Baseline / evaluation 유형 (e.g., 00_baseline)

| Figure | 내용 | 비고 |
|---|---|---|
| `metrics_paper_overlay` | 데이터셋 별 NDCG@10 bar + paper 보고치 horizontal marker | 재현 검증 시각화 |
| `metric_at_k_curves` | NDCG@k / Recall@k / MRR@k vs k (line plot, 데이터셋 별 패널) | retrieval quality 의 rank decay |
| `per_query_metric_dist` | 데이터셋 별 per-query NDCG@10 violin 또는 ECDF | 분포 spread / mass |
| `confused_slice_size` | 데이터셋 별 confused query 비율 (stacked bar: confused vs non-confused) | downstream LSR 평가의 base rate |

### 16.3 Ablation / 비교 유형 (e.g., T1.01+, T2A.*, T2B.*)

| Figure | 내용 | 비고 |
|---|---|---|
| `delta_metric_ci_forest` | 각 ablation 의 Δ NDCG@10 + 95 % paired bootstrap CI (forest plot, 0 라인) | 통계 검정 핵심 plot |
| `delta_metric_violin` | per-query Δ NDCG@10 violin (all / confused 슬라이스 분리) | distribution-level 효과 |
| `delta_metric_ecdf` | per-query Δ NDCG@10 ECDF (baseline vs steered) | 누적 분포 비교 |
| `layer_sweep_heatmap` | layer set × dataset 의 Δ NDCG@10 heatmap | 층별 기여 |
| `confused_vs_all_scatter` | per-query (baseline NDCG, Δ NDCG) scatter | trade-off 시각화 (anchor 손상 여부) |

### 16.4 학습 유형 (e.g., T1.02_full_5L training run)

| Figure | 내용 | 비고 |
|---|---|---|
| `train_loss_curve` | step × loss (total / rank / anchor / gate) | 학습 안정성 |
| `val_metric_curve` | epoch × val NDCG@10 (confused / all) | early-stop 정당성 |
| `gate_activation_dist` | per-layer gate value 분포 (학습 종료 시점) | gate collapse / saturation 진단 |
| `direction_norm_trace` | epoch × ‖v_ℓ‖ per layer | 학습된 direction 의 크기 변화 |
| `top_neighbor_tokens` | 학습된 v_ℓ 와 token embedding 의 cosine top-k (per layer) | direction 의 *해석* |

### 16.5 종합 figure (`report/figures/summary/`)

주요 milestone 또는 REPORT.md 갱신 시 추가:
- `summary_table_figure` — 모든 ablation × 모든 dataset × 핵심 metric grid (heatmap or table-as-figure)
- `cross_dataset_generalization` — LOOCV held-out 도메인 Δ metric 분포
- `parameter_efficiency` — 학습 파라미터 수 × Δ NDCG@10 scatter (LSR 가 50K 제약 내에서 어느 위치인지)

### 16.6 누락 검사

실험이 *완전* 하려면 모두 충족 (§3.9):
- `experiments/{NN}_*/run.py` 가 artifact 를 생성.
- **동일 디렉토리의 `figures.py` 가 누락 없이 위 §16.2–§16.5 의 해당 카탈로그를 생성** (PDF + PNG 동시 저장, `report/figures/{NN}_*/`).
- **`report/{NN}_{exp_name}_report.md` 본문에 모든 figure 가 markdown image link 로 직접 임베드** (figure 별 caption 동반). 누락된 figure 가 있으면 보고서는 *불완전*.
- 카탈로그 외에 추가 figure 자유 — 단 본문 ↔ figure 1:1 인용 가능해야 함.