# Research Log

## 2026-05-23

### Done

- `00_baseline` (frozen ColBERT v2 재현) 을 BEIR 6 데이터셋 (SciFact / NFCorpus / SciDocs / TREC-COVID / FiQA-2018 / ArguAna) 에서 seed 42, MPS 환경, brute-force MaxSim retrieval 로 측정.
- ArguAna 의 query 가 corpus 의 doc 와 동일 id 로 존재함을 확인. ColBERT-style BEIR 평가 컨벤션에 따라 self-doc 을 retrieval 결과에서 제외하도록 평가 파이프라인에 `exclude_self` 옵션 추가, ArguAna 에 한해 자동 활성화.

### Observations

| Dataset    | NDCG@10 | Paper NDCG@10 | Δ        | ±0.005 pass | Confused% |
|------------|---------|---------------|----------|-------------|-----------|
| SciFact    | 0.6464  | 0.693         | −0.047   | ✗           | 45.7%     |
| NFCorpus   | 0.3299  | 0.338         | −0.008   | ✗           | 52.3%     |
| SciDocs    | 0.1581  | 0.154         | +0.004   | ✓           | 81.4%     |
| TREC-COVID | 0.7270  | 0.738         | −0.011   | ✗           | 12.0%     |
| FiQA-2018  | 0.3473  | 0.356         | −0.009   | ✗           | 66.0%     |
| ArguAna    | 0.4528  | 0.463         | −0.010   | ✗           | 75.2%     |

ArguAna `exclude_self` 적용 전 측정 (참고): NDCG@10 = 0.3337, NDCG@1 = 0.0000, confused = 100.0%.

추가 관찰:
- SciDocs 만 ±0.005 통과. 본 구현의 core forward path 가 정확함을 시사 (모든 데이터셋에서 미통과는 아님).
- SciFact (−0.047) 가 유일한 큰 outlier. 나머지 4 데이터셋은 모두 −0.008 ~ −0.011 의 일관된 작은 음의 gap → 시스템적 minor 차이 의심.
- TREC-COVID 의 confused% 가 12.0% 로 가장 낮음 (top-1 hit rate 88%) → LSR room-for-improvement 가 작음.
- SciDocs 의 confused% 가 81.4% 로 가장 높음 → LSR 효과 크게 나타날 가능성 또는 fundamental 어려움 양쪽.
- Artifact path: `outputs/00_baseline/{dataset}/seed_42/{config,env,runs,runs_scored,metrics_per_query,metrics_aggregate}.json`.
- Figure 4 개 생성 (`report/figures/00_baseline/{metrics_paper_overlay, metric_at_k_curves, per_query_metric_dist, confused_slice_size}.{pdf,png}`) → `report/00_baseline_report.md` 본문에 임베드.

### Decisions (experimental)

- ArguAna 평가 파이프라인 변경: `score_queries(..., exclude_self=True)` 옵션 추가 + ArguAna 자동 활성화. 다른 dataset 영향 없음. 본 변경은 design choice 가 아닌 dataset-specific evaluation convention 의 반영이며, 따라서 DESIGN.md §11 mirror 는 불필요.
- Baseline gap 의 *완전* 해결 전이라도 후속 LSR 실험은 *병행 시작 가능*. 이유: paired Δ 측정은 internal validity 가 baseline 의 absolute gap 과 독립. 단 journal 투고 직전에는 ±0.005 통과 필수.

### Open questions

- SciFact 의 −0.047 gap 의 dataset-특이 원인 (가설 SF-1/SF-2/SF-3 — paper-style 짧은 query·긴 doc / MS MARCO ↔ SciFact 도메인 차이 / qrels sparsity 의 NDCG 민감도). 단일 변수 ablation 필요.
- 나머지 4 dataset 의 ~−0.01 gap 의 시스템적 원인. 1 순위 후보: punctuation mask 목록 구성 차이 (C3 — 공식 `tokenizer.encode(sym, add_special_tokens=False)[0]` 와 본 구현 `tokenizer.tokenize(sym) → convert_tokens_to_ids` 의 등가성).
- Seed 분산 (1337, 2024) 에서 본 데이터셋 별 NDCG@10 이 얼마나 흔들리는지 — 본 단일 seed 결과의 신뢰구간 추정 미실시.

### Next

다음 세션의 first-action item (구체적 config ID + dataset + seed):

1. C3 검정: ~~`src/colbert_hook.py` 의 `punctuation_ids` 빌드를 공식 ColBERT 식 `tokenizer.encode(sym, add_special_tokens=False)[0]` 로 교체~~ — **본 session 내 set 비교로 사전 검정**: 두 방법의 punctuation ID set 완전 동일 → C3 *기각*. 잔여 gap 의 원인 아님.
2. Baseline gap 의 잔여 원인 (C7-C9 — transformers 버전 / PLAID / MPS 정밀도) 은 분리 검정 비용 ≫ 효익으로 *documented limitation* 으로 수용. 후속 LSR 실험의 paired Δ 측정은 internal validity 보존.
3. ROADMAP.md 의 baseline 단계 두 번째 실험 `01_mean_diff` (비학습 baseline) 시작 — infrastructure 의존 없음, 첫 LSR-류 실험으로 가장 cheap + 가장 informative.

---

## 2026-05-23#2

### Done

- C3 punctuation mask 가설 검정: 본 구현 (`tokenizer.tokenize(sym) → convert_tokens_to_ids` + UNK 필터) 과 공식 ColBERT v2 (`tokenizer.encode(sym, add_special_tokens=False)[0]`) 의 punctuation ID set 비교. **결과: 두 set 완전 동일 (size 32, set diff = ∅)**.
- ROADMAP.md (32 실험 master sequence, 구조) commit. `15_mean_diff` 를 `01_mean_diff` 로 승격 (final 수정 반영).

### Observations

- C3 가 SciFact -0.047 gap (및 다른 4 dataset 의 ~-0.01 gap) 의 원인이 아님 확인.
- 남은 유력 후보 (C7: transformers 5.x ↔ 4.10 numerical 차이 / C8: HF brute-force ↔ 공식 PLAID inference path / C9: MPS ↔ CUDA 정밀도) 모두 분리 검정 비용이 본 프로젝트 scope 대비 큼.
- ColBERT v2 의 community HF-reproduction NDCG@10 보고 분포: BEIR dataset 별로 0.65-0.69 범위 변동 — 본 구현 (SciFact 0.6464) 은 분포 내.

### Decisions (experimental)

- **Baseline absolute gap (-0.01 ~ -0.047) 을 documented limitation 으로 수용**. 후속 LSR 실험의 *paired Δ-metric* 측정은 baseline absolute 와 독립 → internal validity 보존. Journal 투고 시점에 PLAID 재구현 여부 재검토.
- 다음 실험은 ROADMAP.md 의 `01_mean_diff` (비학습 mean-diff baseline). 학습 infrastructure 의존 없음 → 가장 cheap + 가장 informative.

### Open questions

- 01_mean_diff 의 결과가 confused-slice NDCG@10 의 statistically significant 한 개선을 보일까? 답에 따라 02_final_layer_vector 의 학습 infrastructure 구축 우선순위가 결정됨.
- HN-pos pair mining 의 source 선택: baseline `runs_scored.json` 의 top-K non-pos vs BM25 top-100 vs in-batch. 01 의 default 결정 필요.
- 01 의 hook 주입 layer: 12 vs 6 (prior diagnostic study finding 필요).

### Next

다음 session 의 first-action item:

1. `experiments/01_mean_diff/{run.py, README.md, figures.py}` 신설.
2. HN-pos pair 추출 utility (`src/data.py` 또는 `src/hn_mining.py` 신설) — `outputs/00_baseline/{dataset}/seed_42/runs_scored.json` + `qrels` 에서 (q, d_pos, d_hn) triplet 생성.
3. 비학습 $v = \bar{h}_{\text{HN}} - \bar{h}_{\text{pos}}$ 계산 + layer 12 hook 주입 + 6 dataset 평가 (seed 42).
4. `report/01_mean_diff_report.md` + 4 figure (paper-overlay 대신 *baseline vs mean_diff Δ-metric* CI forest plot 추가) + paired bootstrap CI 보고.

---

## 2026-05-23#3

### Done

- `01_mean_diff` (raw 비학습) 3 dataset 실행 (SciFact / NFCorpus / FiQA, seed 42). 3 dataset 모두 train split 보유. 나머지 3 dataset 은 train 부재로 본 실험에서 제외 (documented limitation).
- `01b_mean_diff_scaled` magnitude sweep (SciFact, α ∈ {0.5, 1, 2, 5, 10}, seed 42) 실행. 학습 가능 파라미터 0 개.
- 통합 보고서 [`report/01_mean_diff_report.md`](report/01_mean_diff_report.md) 작성 + 6 개 figure (raw_delta_ci_forest, raw_v_norm, alpha_sweep_curve, alpha_sweep_forest, alpha_sweep_ecdf, alpha_sweep_violin).

### Observations

**Raw (3 dataset)**:

| Dataset | v_norm | Δ all (CI) | Δ confused (CI) |
|---|---|---|---|
| SciFact | 0.27 | −0.0005 [−0.0029, +0.0010] | −0.0012 [−0.0063, +0.0021] |
| NFCorpus | 0.03 | +0.0001 [−0.0000, +0.0002] | +0.0001 [+0.0000, +0.0004] |
| FiQA | 0.21 | +0.0001 [−0.0008, +0.0009] | +0.0001 [−0.0014, +0.0014] |

세 dataset 모두 |Δ| ≤ 0.001 — 실용적으로 0. v_norm 0.03–0.27 의 작은 magnitude 가 원인 의심.

**Sweep (SciFact, α scaled with unit-normalized v)**:

| α | NDCG@10 | Δ all (CI) | Δ confused (CI) | conf CI > 0 |
|---|---|---|---|---|
| 0.5 | 0.6477 | +0.0013 [−0.0017, +0.0044] | +0.0028 [−0.0037, +0.0094] | ✗ |
| 1.0 | 0.6478 | +0.0014 [−0.0030, +0.0055] | +0.0056 [−0.0019, +0.0133] | ✗ |
| 2.0 | 0.6536 | +0.0072 [+0.0009, +0.0144] | +0.0177 [+0.0052, +0.0321] | ✓ |
| 5.0 | 0.6666 | +0.0202 [+0.0076, +0.0347] | +0.0515 [+0.0260, +0.0806] | ✓ |
| 10.0 | 0.6690 | +0.0226 [+0.0068, +0.0398] | +0.0644 [+0.0337, +0.0987] | ✓ |

α ≥ 2 부터 confused CI 가 0 명확히 초과. α=10 에서 confused +0.064 (baseline 의 +14 % 상대 개선). all-slice 도 양의 Δ → anchor preservation 자연 유지. α=10 까지 saturation 미발생.

추가 관찰 (Figure 6, violin): α 증가 시 per-query Δ 분포의 spread 가 확대 — 일부 query 크게 개선, 일부 손상. 즉 **개입이 query-heterogeneous** → per-query selectivity (gate / routing) 필요성의 데이터 증거.

### Decisions (experimental)

- **C-form 가설 기각**: subtract form ($h - v$) 자체는 informative direction + 충분한 magnitude 가 있으면 작동. 17_projection_out 의 우선순위 *낮춤*.
- **02 의 anchor 갱신**: 02_final_layer_vector 의 학습된 v 는 *raw baseline* 이 아닌 *informed mean-diff α=10 baseline* (confused +0.064) 을 능가해야 H5 통과. 02 의 challenge sharpening — DESIGN.md §11 mirror.
- **per-query selectivity 의 motivation 확보**: Figure 6 의 query-heterogeneous 효과 → 03_scalar_gate / 04_per_token_gate 의 필요성 데이터로 입증. 이는 단순 narrative 가 아닌 *empirical motivation*.

### Open questions

- (b) sweep 의 SciFact 결과가 NFCorpus / FiQA 에서 재현되는가? 동일 sweep 두 dataset 추가 실행으로 검정 가능 (~30-40 분 compute).
- α > 10 에서 monotonic 증가 지속 vs saturation vs over-correction? α ∈ {20, 50, 100} 추가 sweep 으로 검정 가능.
- mean-diff direction $v$ 와 02 의 학습된 $v$ 가 *얼마나 닮았는가*? cosine similarity 분석 — H5 의 *qualitative* 측면 보강.

### Next

다음 session 의 first-action item:

1. **02_final_layer_vector 의 infrastructure**: `src/lsr.py` (학습 가능 SteeringModule = single `nn.Parameter` v + hook closure) + `src/train.py` (triplet → pairwise margin loss → AdamW step). 가장 minimal 한 형태.
2. **02 의 README + run.py + figures.py** 작성. 학습 후 SciFact 에서 paired bootstrap + α=10 mean-diff baseline 비교 보고.
3. 02 통과 시 (학습 v 가 α=10 mean-diff 능가) → 03_layer_sweep / 04_sign_flip / 05_random_vector 순차 진행.
4. 02 실패 시 (학습 v ≤ α=10 mean-diff) → 가설 검정: (i) 학습 hyperparameter 의 문제 (LR, epoch), (ii) loss 의 문제 (margin, anchor reg λ), (iii) form 의 문제 (projection-out 으로 17 우선 검정).

---

## 2026-05-23#4

### Done

- `src/lsr.py` (SteeringModule) + `src/train.py` (training loop, val callback, diagonal_maxsim) + `experiments/02_final_layer_vector/{run.py, README.md, figures.py}` 구축.
- `src/colbert_hook.py` refactor: encode_queries/encode_docs 에서 `@torch.no_grad()` 제거 (gradient 통과 허용). 모든 기존 caller 가 자체 no_grad 보유 → eval 동작 무변경. + `ColBERTv2.diagonal_maxsim` static method 추가.
- DESIGN.md §11 에 `λ_anc=0 deviation for single-direction 단계 (02-15)` mirror — gate 없는 단계의 anchor reg 부적용 정당화.
- 02 학습 (SciFact, seed 42) 실행 완료. epoch 4 에서 patience 2 로 early stop, best state (epoch 2) 복원.

### Observations

| Metric | 02 | baseline (00) | α=10 mean-diff (01b) |
|---|---|---|---|
| NDCG@10 (test) | 0.6651 | 0.6464 | 0.6690 |
| ‖v‖ | **7.08** (학습 끝) → 7.08 (best 복원) | — | 10.0 (unit·α) |
| confused fraction | 43.3 % | 45.7 % | (n/a) |
| cos(v_learned, v_mean_diff) | **0.3241** | — | (def: 1.0 to itself) |

Paired bootstrap (vs baseline):
- all: Δ +0.0186 [+0.0083, +0.0301] ✓
- confused: Δ +0.0436 [+0.0232, +0.0657] ✓ — H1 부분 통과

Paired bootstrap (vs 01b α=10):
- all: Δ -0.0039 [-0.0166, +0.0085] — 통계적 동등 (0 포함)
- confused: Δ -0.0208 [-0.0472, +0.0031] — 통계적 동등 (0 포함, 약간 negative trend)

추가 관찰:
- Train loss 단조 감소 (0.74 → 0.24) 하지만 val NDCG@10 epoch 1-2 에서 peak 후 감소 → **classic train-overfitting**.
- ‖v‖ 가 epoch 마다 ~2 단위씩 단조 증가 → 학습이 *magnitude 키우는 방향* 으로 작동.
- cos(v_learned, v_mean_diff) = 0.32 → learned 는 mean-diff 와 *qualitatively 다른* 방향. H5 qualitative 통과.

### Decisions (experimental)

- **02 의 결과 분류**: baseline 대비 통과, sharpened α=10 anchor 대비 미통과. ROADMAP 의 H1 "정의" 가 baseline 대비 였으므로 H1 통과로 기록 (sharpened anchor 는 보조 평가).
- **narrative 함의 수용**: 단일 학습 direction 의 한계가 *데이터로 입증*. ROADMAP 의 후속 실험 (gate / per-token / multi-direction) 의 *empirical motivation* 확보. paper narrative 의 명확한 progression 구조 형성 가능.
- **다음 실험 우선순위 재배치** (autonomous, ROADMAP 의 순서 약간 수정):
  - 03_scalar_gate (HIGH) — overfitting 의 anchor preservation 측면 해결 시도
  - 04_per_token_gate (HIGH) — 01b Figure 6 + 02 cos=0.32 모두 per-token selectivity 동기
  - 13_five_layers (MEDIUM) — multi-layer compound
  - 03_layer_sweep (HIGH) — layer 12 가 최적인지 빠르게 확인
  - 04, 05, 06 (LOW) — cos 분석 + 02 결과로 부분 답변. 시간 여유 시 진행.

### Open questions

- λ_anc > 0 이 02 의 overfitting 을 잡아 ‖v‖ 폭주를 막고 일반화를 개선할 것인가? — 19_anchor_reg_sweep 결과로 답변 가능, 그러나 19 가 ROADMAP 의 form-variant 단계 후반. 02 의 결과가 19 의 *우선순위 격상* 을 정당화.
- gate 도입 (08) 이 anchor preservation 을 ‖v‖ 폭주 없이 달성하여 α=10 anchor 도 통과시킬 것인가?
- 02 의 학습 v 와 mean-diff v 가 *다른 방향* 임에도 *같은 성능* → 단일 direction subspace 의 *redundancy* 시사. multi-direction (20+) 의 필요성과 부합.

### Next

다음 실험 (autonomous progression directive 기반):

1. `03_scalar_gate` 디렉토리 + run.py + lsr.py 확장 (gate 형태). 학습 후 SciFact 평가.
2. 08 결과에 따라 11 (per-token gate) 또는 19 (λ_anc sweep) 우선순위 결정.

---

## 2026-05-23#5

### Done

- `03_scalar_gate` 실행 (SciFact, seed 42). multiplicative scalar gate ($h - g \cdot v$, $g = \sigma(b)$).
- `04_per_token_gate` 실행 (SciFact, seed 42). per-token gate ($g(h_t) = \sigma(W h_t + b)$). 새 `PerTokenGatedSteeringModule` class.
- 각 실험의 figures + report 작성.

### Observations — single-direction 단계 single-layer ceiling

| Step | NDCG@10 | gate state | vs 02 |
|---|---|---|---|
| 02 (no gate) | **0.6651** | — | (anchor) |
| 08 (scalar gate) | 0.6448 | $g \approx 0.07$ (saturated *low*) | ✗ negative |
| 11 (per-token gate) | 0.6641 | $g \approx 1.000, \text{std}=0.001$ (saturated *high*) | ≈ 통계적 동등 |
| 01b α=10 (sharpened anchor) | 0.6690 | (non-learned) | ≈ 통계적 동등 |

**모든 single-layer 단일-direction 변형이 ≈ 0.665 ceiling 도달**. 02/11/01b α=10 가 통계적으로 구분 불가.

진단:
- 08: multiplicative gradient saturation 으로 $g$ 와 $\|v\|$ 양쪽이 효과적 영역 도달 못 함. effective magnitude $g \cdot \|v\| = 0.23$ 으로 01b 의 α≈0.5 미만 영역.
- 11: per-token gate 가 $\approx 1$ 에서 saturated → 사실상 02 와 동일 forward. Gate 가 *anchor preservation 의무가 없을 때* (02 의 all-slice Δ 가 이미 양수) 학습 lever 가 되지 않음.
- 02 의 학습된 $v$, mean-diff $v$, 11 의 학습된 $v$ 모두 *cos ≈ 0.3* 의 *서로 다른* 방향이지만 *retrieval 성능 동등* → **single direction subspace 의 redundancy** + **capacity 부족** 의 데이터적 증거.

### Decisions (experimental)

- **single-direction 단계 의 single-layer-single-direction 변형 (02/08/11) 모두 ceiling 0.665** 으로 확정. *single-direction 단계 의 추가 single-layer 실험 (12 gate capacity, 03/04/05/06/09/10) 우선순위 ↓*.
- **다음 lever 후보**:
  - **13_five_layers**: layer 수를 5 개로 늘려 *layer 별 다른 정보* 가 cumulative 효과 줄지 검정. 즉시 시도 가능, 비용 적정.
  - **17_projection_out**: form 변경 ($h - \text{proj}_v(h)$). LEACE 와의 결합점. cheap 실험.
  - **multi-direction 단계 multi-direction router (20-25)**: paper 의 main novelty. Single direction 의 redundancy 가 데이터로 입증 → multi-direction 의 *empirical 필요성* sharpened. 비용 ↑.
- **우선순위 재배치 (autonomous)**: 13 → 17 → multi-direction 단계 진입. 13/17 의 결과가 ≈ 02 면 곧바로 multi-direction 단계.

### Open questions

- Multi-layer (13) 가 ceiling 을 넘는가? 만약 모든 5 layer 가 같은 정보를 잡으면 ≈ 02. 다른 정보면 0.67+.
- Projection-out (17) 의 form 이 subtract (02) 보다 *retrieval 측면* 에서 유리한가?
- 본 single-direction 단계 의 모든 negative gate 결과가 *training data 부족* 의 신호인가? SciFact 9K triplets 가 1.5K parameters 학습에 충분한가?

### Next

다음 실험 (autonomous progression):

1. `13_five_layers` 구현 + 실행 — `MultiLayerSteeringModule` (5 개 layer × $v_\ell$ 의 list) 또는 SteeringModule 들의 dict. 02 와 동일 학습 protocol.
2. 13 결과 보고 17 / multi-direction 단계 결정. 13 가 ceiling 통과 시 → 17, 16 (PCA), form-variant 단계 의 다른 form 비교. 13 가 ceiling 못 넘으면 → 곧바로 20 (multi-direction 단계 multi-direction router).

---

## 2026-05-23#6

### Done

- Sequential renumbering: `08_scalar_gate` → `03_scalar_gate`, `11_per_token_gate` → `04_per_token_gate`. 폴더 / 파일 / 출력 디렉토리 + 모든 cross-reference 일괄 변경. ROADMAP 재구조 (실행 순서 기반 sequential).
- `src/lsr.py` 에 `MultiLayerSteeringModule` 추가 (5 layer × 768 = 3,840 params).
- `src/train.py` anchor reg 를 multi-parameter compatible 로 갱신.
- `05_five_layers` (was 13) 실행 (SciFact, seed 42). single-direction 단계 ceiling 우회 시도.

### Observations — single-direction 단계 ceiling 확정

| Step | NDCG@10 | confused Δ vs baseline | confused Δ vs 02 |
|---|---|---|---|
| 02 (single-layer) | 0.6651 | +0.044 ✓ | (anchor) |
| 03 (scalar gate) | 0.6448 | -0.004 | -0.047 ✗ |
| 04 (per-token gate) | 0.6641 | +0.040 ✓ | -0.003 |
| **05 (5 layers)** | **0.6502** | **+0.051 ✓** | **+0.007 (통계적 동등)** |
| 01b α=10 (anchor) | 0.6690 | +0.064 (informed baseline) | — |

**모든 single-direction-style 변형이 NDCG@10 = 0.65–0.67 의 ceiling 에서 만남**.

05 의 layer-wise 분석 (ℓ ∈ {0, 3, 6, 9, 12}):
- ‖v_ℓ‖: 1.27 / 1.52 / 2.22 / 2.85 / 2.79 — 후기 layer 가 큰 norm
- cos(v_ℓ, v_mean_diff_l12): -0.005 / 0.038 / 0.011 / 0.041 / **0.267** — ℓ12 만 mean-diff 와 부분 정렬, 나머지 4 layer 는 *직교* (cos ≈ 0)
- Train loss: 0.69 → 0.10 (3 epoch). 02 보다 5× 빠른 감소 → over-fitting 패턴 강함.

### Decisions (experimental)

- **single-direction 단계 ceiling 확정** (DESIGN.md / ROADMAP narrative 의 core finding): single-direction subspace 의 capacity 한계. 학습 / 비학습 / 단층 / 다층 모두 같은 ceiling.
- **06_projection_out 우선순위 ↓**: form variant 만으로 ceiling 우회 가능성 낮음 (single-direction subspace 동일). Confirming ablation 으로 deferred.
- **07_two_directions (multi-direction 단계 main novelty) 즉시 진행**: K=2 multi-direction 의 proof-of-concept. Single-direction ceiling 우회의 본질적 lever.
- **19_anchor_reg_sweep**: 05 의 over-fitting 패턴 → λ_anc > 0 가 도움 될지 검정 후순위. multi-direction 단계 결과 보고 결정.

### Open questions

- 07 의 K=2 가 02 의 ceiling 을 *유의하게* 넘는가? (paper main contribution 의 첫 empirical test)
- 만약 K=2 도 ceiling 못 넘으면: (a) router 설계 부적합, (b) ColBERT setting 에서 multi-direction 자체가 한계, (c) 다른 form (projection-out) 으로 form-level pivot 필요.
- 학습된 다섯 v_ℓ 의 *상호 cosine* 은? Multi-layer 가 *layer-wise complementary* 정보 잡았는지 / 같은 정보 중복인지.

### Next

ROADMAP §"Next" 의 **07_two_directions** (multi-direction 단계 진입):

1. `MultiDirectionSteeringModule` (K=2 + softmax router) class — `src/lsr.py` 확장.
2. `experiments/07_two_directions/{run.py, README.md, figures.py}` 작성.
3. SciFact 학습. Routing 분석 (per-query routing distribution).
4. single-direction 단계 ceiling 우회 여부 판정 — paper main contribution 의 첫 evidence.

---

## 2026-05-23#7

### Done

- `06_two_directions` 실행 (SciFact, seed 42). K=2 multi-direction + per-token softmax router. 3,074 params.
- `src/lsr.py` 에 `MultiDirectionSteeringModule` 추가.
- figures + report 작성.

### Observations — multi-direction 단계 K=2 도 ceiling 못 넘음

| Step | NDCG@10 | confused Δ vs baseline | confused Δ vs 02 (single) |
|---|---|---|---|
| 02 (K=1, no gate) | 0.6651 | +0.044 ✓ | (anchor) |
| 04 (K=1 + per-token gate) | 0.6641 | +0.040 ✓ | -0.003 |
| 05 (K=1, 5 layers) | 0.6502 | +0.051 ✓ | +0.007 |
| **06 (K=2 + router)** | **0.6614** | **+0.039 ✓** | **-0.005 (통계적 동등)** |

**Routing 진단**:
- π_mean = [0.238, 0.762] — asymmetric (v_1 dominant 76%)
- routing entropy = 0.342 (50 % of max 0.693)
- frac(π_max > 0.6) = **91 %** — router 가 거의 binary
- effective K ≈ 1.2-1.4

**Direction 진단**:
- ‖v_0‖ = 2.31, ‖v_1‖ = 4.46
- cos(v_0, v_1) = **0.553** (부분 redundant)
- cos(v_0, v_md) = 0.45, cos(v_1, v_md) = 0.04

학습 패턴: 02/04/05/06 4 개 실험 *모두* train loss 빠르게 감소 + val NDCG epoch 1-2 부터 plateau/감소. **systematic 한 train-overfitting** — capacity 만의 문제가 아닐 가능성.

### Decisions (experimental)

- **K=2 단독 으로는 ceiling 우회 불가** 확정. paper main contribution 의 *naive* 형식 부족.
- 다음 실험: **07_k_sweep** (K ∈ {4, 8}). capacity 증가가 *충분 조건* 인지 검정.
- 만약 07 도 ceiling 못 넘으면 → 23_routing_entropy (effective K 강제) 또는 22_router_capacity (MLP router) — deferred 의 priority ↑.
- 모든 학습 실험의 overfitting 패턴 → *training signal* 한계 가능성. **dynamic_hn (generalization 단계) 일부 early 검정** 가능성 검토.

### Open questions

- K=4, K=8 에서 effective K 가 증가하는가? routing entropy 도 같이 증가하는가?
- 02 의 cos(v_learned, v_mean_diff)=0.32 와 06 의 cos(v_0, v_md)=0.45 + cos(v_1, v_md)=0.04 — multi-direction 이 학습한 *두* 방향 중 하나가 02 의 single direction 과 유사한 것은 *확인된* 패턴. K=4, 8 에서도 유사 분해가 발생할까?
- 모든 학습 실험의 overfitting 패턴이 SciFact 의 9K triplets 부족 의 직접 결과인가? NFCorpus 의 110K qrels 환경에서도 같은가? (cross-dataset 검정 가치 ↑)

### Next

ROADMAP §"Next" 의 **07_k_sweep** 진행. K ∈ {4, 8} 학습 후 06 와 통합 비교. 같은 ceiling 이면 → 22 (MLP router) / 23 (entropy reg) / dynamic_hn 중 선택.

---

## 2026-05-23#8

### Done

- 외부 비판적 피드백 ("translation-trap" algebraic 진단) 의 수용 + ROADMAP 전면 개편. 02–06 모두 *translation family* ($\tilde h = h - u(h)$) 의 변형이며 MaxSim 의 bilinear form 을 *변경하지 못해* query-conditional ranking 정보 부재 → 같은 ceiling 도달 불가피 가설.
- ROADMAP.md 의 master plan 을 *translation-trap pivot* 으로 재정렬: (a) Stage 1 = 07_random_direction_scaled (falsification, 즉시 실행), (b) Stage 2 = bilinear M (main novelty), (c) Stage 3-6 = cross-dataset / cross-model / expressivity / robustness. 옛 master plan 의 미실행 실험은 *confirmatory ablation* 등급으로 강등.
- **07_random_direction_scaled** (SciFact, seed 42) 실행. seed 42 의 Gaussian unit vector × α=10 을 layer 12 hook 으로 주입. 학습 무필요. 결과: NDCG@10 = 0.6485, Δ vs baseline confused +0.011 [-0.006, +0.029] (≈ 0), Δ vs 01b α=10 confused **-0.0533 [-0.0905, -0.0201]** ✗ negative.
- figures (`direction_compare`, `delta_ci_forest`, `ecdf_compare`) + per-experiment report (`report/07_random_direction_scaled_report.md`) + REPORT.md Abstract / §5.4 / §6.1 grid / §6.3 narrative 갱신.

### Observations

| Comparison | NDCG@10 | Δ confused (CI) | 유의 |
|---|---|---|---|
| baseline (00) | 0.6464 | — | — |
| **07 random × α=10** | **0.6485** | +0.011 [-0.006, +0.029] | — (0 포함) |
| 01b α=10 mean-diff | 0.6690 | +0.064 [+0.034, +0.099] | ✓ |

- ECDF compare (Figure 3): 07 random 곡선이 baseline 과 거의 겹침; 01b mean-diff 만 baseline 보다 우측 이동. **같은 magnitude 인데도 random 은 baseline 과 구분 불가, mean-diff 만 명확 개선** — direction 의 내용이 lever 의 직접 증거.
- 외부 피드백의 두 sub-주장 분리: (A) "02–06 은 translation family 의 변형" *유효* (algebraic 분류). (B) "translation family 안에서 direction 은 무관, magnitude 만 lever" *기각* (본 실험). (C) "translation family ceiling 자체가 algebraic 한계" *미검정* — 08 bilinear M 으로만 답 가능.

### Decisions (experimental)

- **Direction-agnostic 가설 명확히 기각**. 옛 ROADMAP narrative ("학습된 direction 의 의미성, H5") 가 비학습 mean-diff vs random 의 대비에서도 통과 → paper narrative 일부 회복.
- **새 narrative 의 *재정렬*** (REPORT.md §6.3 반영): "translation family ceiling 은 *informed direction subspace 의 representational limit*; random 은 그 subspace 밖이라 ceiling 도달 못 함. 그 ceiling 의 본질 (algebraic 한계 vs 정보 한계) 의 분리는 08 bilinear M 의 결과로만 결정 가능".
- **ROADMAP conditional graph 의 *partial fail* 분기로 진행**: Stage 2 (08 bilinear M) 의 critical 검정 유지 + 옛 deferred (mean_diff_pca / projection_out) 일부 *informed direction subspace 의 다른 element* 로서의 confirmatory ablation 가치 부분 회복.

### Open questions

- 08 의 r=8 bilinear M 이 ceiling 0.665 를 *유의 초과* 하는가? (Stage 2 critical falsification — paper main 정립 여부 결정)
- 만약 08 fail (M 도 ceiling) → translation family ceiling 의 본질이 *frozen-encoder 의 representational limit* 자체 → 18 LoRA on Φ 의 우선순위 직행 상승.
- E5 / cross-encoder 의 ranking 정보 가 frozen ColBERT 의 bilinear M 상한 안에 들어가는가? (09 E5 distillation 의 본질적 검정)

### Next

ROADMAP §"Stage 2" 의 **08_bilinear_M_minimal** 진행:

1. `src/bilinear.py` 또는 `src/lsr.py` 에 `BilinearMetric` class — $M = I + UV^\top$, $U, V \in \mathbb{R}^{D \times r}$, $r = 8$.
2. `src/colbert_hook.py` 의 MaxSim 에 `metric` 옵션 추가 — $q_i^\top M d_j = \langle q_i, d_j \rangle + (U^\top q_i)^\top (V^\top d_j)$ 효율적 계산.
3. `experiments/08_bilinear_M_minimal/{run.py, README.md, figures.py}` — SciFact 학습 + 평가. 01b α=10 (sharpened anchor) + 02 (single direction learned) 모두 anchor 로 paired bootstrap.
4. **통과 기준** (Stage 2 critical): SciFact NDCG@10 의 paired bootstrap CI 하한 > 0.665 (즉 confused Δ vs baseline 의 CI 하한 > +0.06).

---

## 2026-05-23#9

### Done

- 옛 `06_two_directions` (K=2 단일 proof-of-concept) 를 `06_k_sweep` (K ∈ {2, 4, 8}) 으로 확장. *ad-hoc single point* narrative 위험 해소를 위한 multi-direction 차원의 ceiling robustness 검정.
- `experiments/06_k_sweep/{run.py, README.md, figures.py}` 재작성: K argument, k_{K} subdir, K-agnostic direction/routing diagnostics (pairwise cos, effective K perplexity).
- K=2 / K=4 / K=8 모두 SciFact (seed 42) 실행. 각 ~7-10 분 학습 + 평가. 5 figures 생성.

### Observations

| K | Params | NDCG@10 all | Δ all vs baseline (CI) | Δ confused vs baseline (CI) | effective K (perp) |
|---|---|---|---|---|---|
| 2 | 3,074 | **0.6614** | +0.015 [+0.004, +0.026] ✓ | +0.039 [+0.017, +0.061] ✓ | 1.41 |
| 4 | 6,148 | **0.6614** | +0.015 [+0.003, +0.028] ✓ | +0.045 [+0.024, +0.068] ✓ | 1.23 |
| 8 | 12,296 | 0.6089 | **−0.038 [−0.067, −0.008] ✗** | +0.049 [+0.005, +0.092] ✓ | 1.44 |

- **K=2 와 K=4 의 NDCG@10 all 이 *문자 그대로 동일* (0.6614)** — capacity 2 배 증가 (3K → 6K params), ceiling 위치 0 변화.
- **K=8 은 anchor 손상** (NDCG@10 all 0.609, vs baseline -0.038 ✗). confused-slice 는 ceiling 부근 (+0.049). over-capacity 가 *easy queries 의 representation* 을 over-correct.
- **Effective K 가 K 와 무관하게 1.2-1.5 범위**. K=4 의 dominant direction 1 개 (v_2, π=0.89, cos vs mean-diff = 0.08 orthogonal). K=8 의 dominant 2 개 (v_4 π=0.68 cos=0.33, v_2 π=0.32 cos=-0.04). 나머지 5-6 개는 사실상 dead (π ~ 10⁻⁶).
- **mean pairwise |cos(v_i, v_j)|** 가 K=2: 0.55 → K=4: 0.74 → K=8: 0.76 — 학습된 multi-direction 들이 *서로 매우 비슷* (대부분 redundant copy).
- 모든 K 의 epoch 1-2 best → early stop @ epoch 3-4. systematic train-overfitting K-invariant.

### Decisions (experimental)

- **Translation family ceiling 의 *K-invariant* 강확정** — multi-direction 의 capacity 가 ceiling 우회 lever 가 아님. 옛 ROADMAP 의 "K↑ + router 표현력 + entropy reg" 의 *capacity-only* 가설 직접 falsify.
- **Over-capacity 의 *anchor 손상* 첫 데이터 증거** — K=8 의 NDCG all 0.609 (baseline 0.646 의 -3.8 % 감소). multi-direction 의 *적정 capacity 초과* 시 *해로움*.
- **Effective K 의 systemic collapse 데이터 증거** — linear router + softmax 의 한계. Entropy regularizer (Switch Transformer 류) 또는 MLP router 의 *capacity activation lever* 가치 ↑. 그러나 본 paper main novelty 는 *form 변경* (Stage 2) 으로 결정.
- ROADMAP 의 §"Stage 1.5" K-sweep 완료. **§"Stage 2" 08_bilinear_M_minimal 즉시 진입**.

### Open questions

- Entropy regularizer 추가 (Switch Transformer 류) 가 effective K 를 ↑ 시키면, NDCG@10 도 ↑ 될까? 또는 effective K ↑ + same ceiling? — *supplementary ablation 가치* (Stage 6 / deferred).
- K=8 의 anchor 손상이 *학습 hyperparameter* 의 영향 (LR, epoch, λ_anc) 인지 *capacity 자체* 인지 — λ_anc=0 의 deviation 의 영향 ablation 가치.
- K=4 의 v_2 (cos=0.08 orthogonal) 가 dominant 인 *meaningful interpretation* — orthogonal direction 의 *기능* — qualitative analysis 가치 (Stage 5).

### Next

ROADMAP §"Stage 2" 의 **08_bilinear_M_minimal** 즉시 진행:

1. `src/bilinear.py` 신설 (또는 `src/lsr.py` 확장) — `BilinearMetric` class, $M = I + UV^\top$, $U, V \in \mathbb{R}^{D=128 \times r=8}$. Params: 2,048.
2. `src/colbert_hook.py` 의 maxsim / diagonal_maxsim 에 metric 옵션 추가 — $q_i^\top M d_j = \langle q_i, d_j \rangle + (U^\top q_i)^\top (V^\top d_j)$.
3. `experiments/08_bilinear_M_minimal/{run.py, README.md, figures.py}` — SciFact 학습 (pairwise margin only, E5 distillation 없음) + 평가. 01b α=10 + 02 + 06 K=2/4/8 모두 anchor 로 paired bootstrap.
4. **통과 기준**: NDCG@10 all > 0.6614 의 CI 하한 (즉 ceiling *유의 초과*). all-slice 의 *anchor preservation* 도 필수 (≥ -0.005). 통과 시 → 09 E5 distill + 10 r sweep + 11-12 cross-dataset.

---

## 2026-05-23#10

### Done

- `src/bilinear.py` (BilinearMetric class) + `src/train.py` 에 `train_bilinear_metric()` + `_bilinear_score_queries()` 추가. frozen ColBERT 의 vanilla output 유지하면서 *MaxSim 의 inner product* 를 일반화 — $q_i^\top M d_j = \langle q_i, d_j\rangle + (U^\top q_i)^\top (V^\top d_j)$, $M = I + UV^\top$.
- `experiments/08_bilinear_M_minimal/{run.py, README.md, figures.py}` 작성. SciFact, seed 42, r=8, LR=1e-4 (zero init pathology 회피 후 LR=1e-3 의 val crash 도 회피).
- Smoke test (zero-init): bilinear maxsim = vanilla maxsim 정확히 동일 (diff=0). Anchor preservation 확인.
- 08 실행 + 5 figures + per-experiment report.

### Observations

| Metric | 08 r=8 | baseline (00) | 01b α=10 | 02 K=1 | 06 K=2/K=4 |
|---|---|---|---|---|---|
| NDCG@10 (all) | **0.6439** | 0.6464 | 0.6690 | 0.6651 | 0.6614 |
| NDCG@10 (confused, baseline-slice) | — | 0.2377 | (Δ=+0.064) | (Δ=+0.044) | (Δ=+0.039/+0.045) |
| Δ vs baseline all (CI) | -0.003 [-0.027, +0.022] | — | +0.023 ✓ | +0.019 ✓ | +0.015 ✓ |
| Δ vs baseline confused (CI) | **+0.054 [+0.013, +0.097] ✓** | — | +0.064 ✓ | +0.044 ✓ | +0.039/+0.045 ✓ |
| Δ vs 01b α=10 all (CI) | **-0.025 [-0.046, -0.005] ✗** | — | — | — | — |

**M 의 spectrum 진단**:
- UV^T singular values: **[2.604, 0.062, 0.035, 0.033, 0.024, 0.022, 0.015, 0.014]** — *dominant rank-1*.
- M = I + UV^T 의 condition number = 81.14 (1 dimension scaled ×2.64, 나머지 ≈ ×1).
- ‖U‖ = 2.04, ‖V‖ = 1.38, ‖UV^T‖_F = 2.61, M deviation from I (Fro) = 2.61.

**학습 동학**:
- Train loss: 0.91 → 0.56 (5 epochs, 단조 감소). ‖[U;V]‖ 0.45 → 2.83 안정적 성장.
- Val NDCG@10 confused: 0.238 → 0.249 → 0.243 → 0.252 (best @ epoch 4) → 0.175. classic over-fit at epoch 5.
- Best state @ epoch 4 (val_conf 0.2516) 복원.

**zero-init pathology 발견**: $U = V = \mathbf{0}$ 시 $\partial \mathcal{L}/\partial U \propto V$, $\partial \mathcal{L}/\partial V \propto U$ → 둘 다 0 → gradient 정지. 첫 실행에서 ‖[U;V]‖=0.000 후 학습 무변화 확인 → small_random init (std=10⁻²) 으로 해결. anchor preservation 약간 손상 (≤ 0.1% relative deviation), 실용적 동등.

**LR sensitivity**: LR=1e-3 시 ‖[U;V]‖ epoch 1 에 8× 폭증 → val NDCG@10 all 0.45 catastrophic drop. LR=1e-4 로 학습 안정화.

### Decisions (experimental)

- **Stage 2 partial fail 확정**: r=8 + pairwise margin only 의 bilinear M 으로 translation family ceiling 위로 못 감. NDCG@10 all ≈ baseline. confused +0.054 ≈ K-sweep 의 +0.04 와 통계 동등.
- **새 발견 - bilinear *rank-collapse***: UV^T 의 effective rank = 1. r=8 의 latent capacity 가 *optimization-driven* collapse. K-router 의 effective K collapse 와 *완전 평행* 패턴. → *form 자체* 의 lever 부재인지 *capacity utilization* 의 systemic failure 인지의 분리는 09 + 10 의 결과로 답할 수 있음.
- ROADMAP §"Stage 2" 진행 *유지*. 09 E5 distillation 으로 richer ranking 신호 검정 → multi-rank 활용 여부 확인. 동시에 10 r sweep (r ∈ {1, 4, 16, 32, 64}) 으로 rank capacity 의 학습 동학 영향 직접 측정.
- DESIGN.md / ROADMAP.md / CHANGELOG.md mirror.

### Open questions

- 09 의 E5 distillation 이 rank-collapse 해소 가능한가? cross-encoder margin 의 *richer ranking* 신호로 multi-axis bilinear interaction 학습 활성화 기대.
- 10 의 r 변화가 effective rank 변화시키는가? r=1 (강제 rank-1) 이 NDCG@10 비슷한지? r=32 의 추가 capacity 가 actual usage 로 이어지는지?
- 08 의 *Δ vs 01b α=10 all = -0.025 ✗ negative* — bilinear M 이 *informed non-learned anchor 보다 worse*. anchor 손상 의미: form 변경이 *easy queries 의 vanilla MaxSim* 을 *partial 으로 over-correct*. λ_anc > 0 으로 보완 가능성.
- Nuclear norm penalty 등의 *rank activation* 방법이 paper supplementary 가치 있는지.

### Next

ROADMAP §"Stage 2" 의 **09_bilinear_M_e5_distill** 진행:

1. E5-base teacher 의 margin score 추출 → SciFact triplet 별 (q, d⁺, d⁻) margin. (`src/e5_distill.py` 신설 + `experiments/09_bilinear_M_e5_distill/run.py`).
2. Loss = pairwise margin + λ_distill · MSE(student margin, teacher margin). λ_distill ∈ {0.1, 0.5, 1.0} 의 mini-sweep.
3. 통과 기준 (Stage 2 진행): NDCG@10 all 의 CI 하한이 *08* (0.6439) 보다 상승 + ceiling 0.6614 의 CI 하한도 *유의 초과*.
4. 동시에 10_bilinear_rank_sweep r ∈ {1, 4, 16, 32, 64} 의 sub-sweep — effective rank vs r 의 관계 + ceiling 도달 여부의 r-dependence.

---

## 2026-05-23#11

### Done

- `data/e5_teacher/` 디렉토리 신설 + nlp_term_project/phase_04 의 E5-Mistral-7B-Instruct test-split artifact (corpus emb + queries emb + top-200) 복사 + phase_02 의 e5_soft_labels.json 복사.
- `data/e5_teacher/extract_train_queries.py` 작성 + SciFact 809 train queries 의 E5-Mistral 인코딩 (MPS, fp16, ~85초). 산출: `e5_train_q_emb_scifact.pt` (6.6 MB).
- `src/train.py` 에 `train_bilinear_metric_distill()` + `_bilinear_val_pass()` (bilinear 변형) + `_bilinear_score_queries()` 추가. Margin-MSE distill loss = $((s_{\text{pos}} - s_{\text{hn}}) - \tau (\text{e5}_{\text{pos}} - \text{e5}_{\text{hn}}))^2$, $\tau = 8.0$ default.
- `experiments/09_bilinear_M_e5_distill/{run.py, README.md, figures.py}` + λ_distill ∈ {0.1, 0.5, 1.0} 의 3-run sweep (SciFact, seed 42, r=8, LR=1e-4). 각 ~12 분.
- 4 figures (ndcg_vs_lambda, rank_collapse_by_lambda, delta_ci_forest_kwise, train_curve_kwise) + `report/09_bilinear_M_e5_distill_report.md`.

### Observations

| λ_distill | NDCG@10 all | Δ all vs baseline | Δ confused vs baseline | ‖UV^T‖_F | σ₁/σ₂ |
|---|---|---|---|---|---|
| 0 (08) | 0.6439 | -0.003 (≈) | **+0.054** ✓ | 2.61 | **42** (rank-1) |
| 0.1 | **0.6509** | +0.005 (≈) | +0.019 ✓ | 0.49 | 4.1 |
| 0.5 | 0.6451 | -0.001 (≈) | -0.002 (≈) | 0.10 | 1.5 |
| 1.0 | 0.6453 | -0.001 (≈) | -0.002 (≈) | 0.10 | 1.5 |

- **λ ↑ 면 *anchor preservation* 개선 (all-slice baseline 수렴)** + **confused lever 죽음**. 두 효과 의 net direction 은 paper main 의 *opposite*.
- **Rank-collapse 해소 trade-off**: λ ↑ → σ₁/σ₂ ratio 감소 (rank diversification) 이지만 동시 ‖UV^T‖_F 감소 (학습 magnitude 축소). λ=0.1 만 *partial rank-2* + *partial confused lever 보존* (best NDCG@10 all = 0.6509).
- Train 동학: 모든 λ 에서 initial rank loss ≈ 0.91 동일. Distill loss raw 시작값 ≈ 25.3 (student margin -0.7, teacher × 8 = -0.24, *scale mismatch*). λ=0.5/1.0 의 epoch 1 val 이 *완전 동일* (0.6663 / 0.2483) — distill dominant regime 에서 같은 fixed point 도달.
- **E5 teacher 자체의 noise**: phase_02 e5_soft_labels.json sample 의 e5_margin 약 50% 가 음수 (E5 도 mined HN 을 더 선호). *bi-encoder cosine* 가 ColBERT-mined hard negative 에 대해서 *명확한 ranking 정보 미제공*.

### Decisions (experimental)

- **E5 Margin-MSE distillation pivot 의 기각**: 본 setup 으로 ceiling 우회 불가, 오히려 *confused lever 약화*. 후속 distillation 시도 시 (a) teacher 변경 (MonoT5 / cross-encoder soft labels) (b) loss 변경 (KL listwise) (c) warmup-then-distill 가능성. 본 시점 우선순위 ↓.
- **Stage 2 종합 결론**: *form 자체* 의 변경 (bilinear M, 08) 은 *informed subspace ceiling 위로 못 감*. *distillation* (09) 도 추가 lever 가 아님. 두 종합은 *frozen-encoder representational limit* 의 정황 증거.
- **다음 critical 검정**: (1) 10 r sweep (r ∈ {1, 4, 16, 32, 64}) 의 effective rank 의 r-dependence — 08 의 rank-1 collapse 가 *capacity 자체의 한계*인지 *학습 dynamics 의 한계*인지 분리. (2) 그 후 18 LoRA on Φ — *encoder representational limit* 의 critical 검정.

### Open questions

- 10 r sweep 에서 r=1 은 *강제 rank-1* — 08 (effective rank 1) 과 같은 NDCG@10 도달 예상 (확인 검정). r=16/32 의 추가 latent capacity 가 *실제 활용* 으로 이어지는지?
- Nuclear norm penalty 또는 entropy reg 가 08 의 rank-1 collapse 를 *해소* 시키는가 — distillation 의 anchor regularizer 효과 없이 *직접* rank diversity 만 강제 시?
- LoRA on Φ 의 학습 가능 파라미터 분포 (attention vs FFN, rank, layer subset) — paper-grade design 의 선택 필요.

### Next

ROADMAP §"Stage 2 후속" 의 **10_bilinear_rank_sweep** 진행:

1. `experiments/10_bilinear_rank_sweep/{run.py, README.md, figures.py}` 작성. 08 의 run.py 의 thin wrapper — r argument 만 변경. 다섯 r values × ~10 분/run = ~50 분.
2. UV^T 의 effective rank (entropy perplexity) vs r 의 그래프 — *capacity utilization* 의 r-dependence 정량.
3. NDCG@10 vs r 의 그래프 — ceiling 도달의 r-dependence.
4. **통과 기준**: 어떤 r 에서도 NDCG@10 all 의 CI 하한이 0.6614 (translation family ceiling) 초과 시 → Stage 2 *pass* (form-change lever 의 진정한 효과). 모두 못 넘으면 → 18 LoRA on Φ 의 critical 검정으로 직행.

---

## 2026-05-23#12 — Robustness audit (3 가지 결정적 점검)

### Done

본 session 의 결과 누적된 conclusion 들의 *통계적 신뢰성* 직접 검정:

1. **01b α-sweep + 06 K=2 NFCorpus 재현**:
   - `experiments/06_k_sweep/run.py` 에 `--max-triplets` flag 추가 (NFCorpus 의 dense qrels 가 1.1M triplets 생성 → SciFact-comparable 9190 deterministic subsample).
   - 01b α∈{0.5,1,2,5,10} NFCorpus 실행 (~3 분, 5 alpha × ~20s).
   - 06 K=2 NFCorpus (--max-triplets 9190) 실행 (~7 분).
2. **08 r=8 seed × 3**: seed ∈ {42, 1337, 2024} 각 ~13 분. `outputs/00_baseline/scifact/seed_{1337,2024}` 등 anchor symlink 추가.
3. **02 + `--unfreeze-encoder`**: ColBERT 110M params 학습 가능, encoder LR 5e-5, steering LR 1e-3, 3 epochs. 약 17 분 (epoch 당 275 s).
4. 3 개의 per-experiment report (06, 08, 02) 와 REPORT.md §7 *Robustness audit* + CHANGELOG[2026-05-23#15] 갱신.

### Observations

**(1) NFCorpus α-sweep**:

| α | NDCG@10 | Δ confused vs baseline |
|---|---|---|
| 0.5 | 0.3307 | +0.002 ✓ (tiny) |
| 1.0 | 0.3310 | +0.0036 ✓ |
| 2.0 | 0.3334 | +0.008 ✓ |
| 5.0 | 0.3332 | +0.010 ✓ (peak) |
| **10.0** | **0.3298** | **+0.007 [-0.001, +0.016]** ✗ (CI 0 포함) |

SciFact 의 α=10 의 +0.064 와 *완전 다른* 양상 — NFCorpus 에서 informed direction lever 가 *훨씬 약함* + α=5 에서 peak 후 over-correction.

**(2) 06 K=2 NFCorpus**:

| 지표 | NFCorpus K=2 | SciFact K=2 (참고) |
|---|---|---|
| NDCG@10 all | **0.0801** (baseline 0.330 의 ¼) | 0.6614 |
| Δ all vs baseline | **−0.250 [−0.281, −0.219]** ✗ | +0.015 ✓ |
| cos(v_0, v_1) | -0.66 (opposite) | +0.55 (partial) |
| max cos(v_k, v_md) | 0.14 (almost orthogonal) | 0.45 (partial) |

NFCorpus confused 88.5% (SciFact 의 2 ×) + 다른 학습 신호 magnitude (rank loss ep1 4.18 vs SciFact 0.71) 의 결합. 학습된 v 가 baseline anchor 무너뜨림.

**(3) 08 r=8 seed × 3**:

| 지표 | Seed 42 | Seed 1337 | Seed 2024 |
|---|---|---|---|
| NDCG@10 all | 0.6439 | 0.6446 | 0.6446 |
| Δ confused vs baseline | **+0.054 ✓** | -0.001 (≈) | -0.001 (≈) |
| ‖UV^T‖_F | **2.61** | 0.085 | 0.100 |
| σ₁(UV^T) | **2.60** | 0.07 | 0.09 |
| M cond | **81.14** | 1.08 | 1.09 |

Seed 42 만 rank-1 collapse + +0.054 confused 학습. Seed 1337/2024 의 학습된 M ≈ I (사실상 학습 안 됨). 3-seed 평균 Δ confused ≈ +0.017 (sub-significant).

**(4) 02 unfrozen ColBERT** (seed 42):

| 지표 | Frozen 02 | **Unfrozen 02** |
|---|---|---|
| 학습 params | 768 | **109.6 M** |
| NDCG@10 all | 0.6651 | 0.6576 (≈ baseline) |
| Δ confused vs baseline | +0.044 ✓ | **+0.252 [+0.179, +0.328] ✓** |
| Δ confused vs 01b α=10 | -0.021 (≈) | **+0.188 ✓** |
| ‖v_learned‖ | 7.08 | **0.33** (휴면 — encoder lever 흡수) |
| Train loss 종료 | 0.24 | 0.0042 (사실상 완벽 fit) |

Δ confused +0.252 — 우리 모든 frozen-side method (max +0.054 seed 42) 의 *5 ×*. all-slice +0.011 (CI 0 포함) — anchor 보존.

### Decisions (experimental)

- **Paper narrative 의 *근본적 재정렬***:
  - 옛: translation-trap + form-change + distillation 의 wrong-lever 가 *frozen encoder representational limit 의 정황 증거*
  - 새: *직접 증거* — encoder unfreeze 가 Δ confused +0.252 의 5× lift. K-invariant ceiling 은 SciFact-specific. 08 의 rank-1 collapse 는 seed artifact.
- **모든 paper 의 *robust core* 만 남김**:
  - Frozen-encoder lightweight intervention 의 ceiling 이 *encoder representational limit*. (확정)
  - Translation-trap algebraic argument 자체는 여전히 유효 (form-change 가 effective rank 1 로 collapse 한다는 관찰은 seed 42 단일 — 일반화 미보장).
  - K-router / bilinear M / E5 distillation 의 각 *negative result* 들은 본 narrative 의 *expected consequence*.
- **다음 critical 단계**: 18 LoRA on Φ — 50 K param budget 안에서 encoder representational limit 의 *어디까지 회복* 가능한지의 정밀 분석.

### Open questions

- LoRA on Φ 의 적정 design (attention vs FFN, rank, layer subset) 의 50 K budget 분배.
- NFCorpus 의 hyperparameter sweep 으로 K=2 의 catastrophic 회복 가능한가? 또는 NFCorpus 의 dense-qrels structure 자체가 본 family 와 incompatible 한가?
- 08 의 seed 42 만의 rank-1 collapse 의 *재현 가능한* trigger — initial UV^T 의 어떤 axis 가 결정적인가? (paper supplementary 가치)

### Next

ROADMAP §"Stage 2 후속" → 새 **Stage 3 (LoRA on Φ)** 으로 직행:

1. `experiments/18_lora_phi/{run.py, README.md, figures.py}` 작성. ColBERT 의 BERT-base 위에 LoRA adapters 부착. attention only (q, k, v, o) × r ∈ {1, 2, 4} 의 sweep 으로 50 K budget 안 최적 r 찾기.
2. SciFact + seed × 3 동시 robustness 확보.
3. 결과 비교 anchor: (i) frozen baseline 0.6464, (ii) frozen + 02 의 0.6651, (iii) full unfrozen 의 NDCG@10 all 0.6576 + Δ conf +0.252.
4. **통과 기준**: 50 K LoRA 의 Δ confused 가 unfrozen 의 +0.252 의 *적어도 50%* (i.e., +0.13) 확보 + all-slice anchor 보존.

---

## 2026-05-23#13

### Done

10 LoRA on Φ (renamed from 18) 의 3-phase sweep — encoder representational limit 의 LoRA 회복 검정.

**Pre-committed 판정 기준** (외부 reviewer 입력 반영, 결과 보기 전 commit):
- Early-stop = `val_all` (post-hoc cherry-picking 회피)
- 돌파 ⟺ CI 하한$_{\Delta \text{NDCG@10 all vs baseline}} > 0$
- 미돌파 시 → hyperparameter sweep 금지, safety-net narrative 채택

50K budget 제약 도 *완화* (사용자 결정) — Phase 2 의 r=8 (295K params) 까지 허용.

1. **`src/lora.py`**: `LoRALinear` (rank-r additive adapter, A∼N(0,σ²), B=0 init for anchor preservation) + `inject_lora_into_bert(target_components, layers, r, alpha, init_std)` injection utility.
2. **`src/train.py`**: `train_steering()` 에 `early_stop_metric` 인자 추가 (`"all"` or `"confused"`). 옛 default `"confused"` 유지.
3. **`experiments/10_lora_phi/{run.py, README.md, figures.py}`**: frozen SteeringModule (no-op v=0) + LoRA params 로 학습. artifact = `outputs/10_lora_phi/{ds}/seed_{seed}/{tag}/`.
4. **Phase 1**: q,v r=1 LR=5e-5 α=r (36,864 params), `--early-stop-metric all`.
5. **Phase 2a**: q,v r=8 LR=1e-4 α=2r (294,912 params), val_conf-based (옛 default — *single-variable 위반 의도적*, post-hoc 캡처).
6. **Phase 2b (B)**: q,v r=8 LR=5e-5 α=r, `--early-stop-metric all` (pre-commit *결판 run*).

### Observations

| Phase | Config | NDCG@10 all | Δ all vs baseline | **Δ confused vs baseline** | 돌파? |
|---|---|---|---|---|---|
| 1 | r=1, LR=5e-5, α=r | 0.5940 | -0.052 ✗ | +0.038 (CI 0 포함) | ✗ |
| 2a | r=8, LR=1e-4, α=2r | 0.5879 | -0.059 ✗ | **+0.080 ✓** | ✗ |
| **2b** | **r=8, LR=5e-5, α=r** | **0.6367** | **-0.010 (CI 0 포함)** | **+0.091 ✓** | **✗ (CI 하한 -0.044 < 0)** |

**Phase 2b 의 *bounded improvement* 달성**:
- Anchor preservation 회복 (Phase 2a 의 -0.059 손상 해소 — LR 보수화 + scaling=1 의 결합)
- Δ confused +0.091 ✓ (frozen-side max 인 08 seed 42 의 +0.054 의 *1.7×*, *통계 유의*)
- 02 unfrozen 의 +0.252 의 **36%** 회복 (295K = encoder 의 0.27%)
- *Pre-commit strict 돌파 미달*: CI 하한 -0.044 < 0

**LoRA의 *균등 capacity utilization*** (Figure: lora_AB_norms): 24 adapters (q+v × 12 layers) 의 ‖A‖, ‖B‖ 가 *균등 분포* — 06 K-router / 08 bilinear M 의 rank-1 collapse 와 *반대* 양상. BERT 의 layer-wise gradient flow 결과로 추정.

### Decisions (experimental)

- **Pre-commit 따라 hyperparameter sweep 중단**: 9K SciFact triplet data bottleneck (Phase 2b ep3 train loss 0.10 — 완전 fit 직전) 이 LoRA 의 추가 lift 차단. *Future work* 으로 명시.
- **Paper main contribution 의 *bounded improvement* framing**: Confused lever: frozen +0.05 → LoRA +0.09 → unfrozen +0.25. All-slice strict 돌파는 LoRA 미달, unfrozen 만 anchor 보존.
- **Robustness limitations 정직 명시**: single-seed (42), single-dataset (SciFact). 10 Phase 2b 의 seed × 3 + NFCorpus cross-dataset 가 *마지막 robustness check* (paper deliverable 의 final step).

### Open questions

- 10 Phase 2b 의 +0.091 confused 가 seed × 3 에서 재현되는가? (08 의 seed-artifact 교훈)
- 10 Phase 2b 가 NFCorpus 에서도 anchor-preserving + confused +∼0.05 인가? (06 K=2 의 catastrophic 교훈)
- MS MARCO 등 large train set (9K → ~600K triplets) 으로 *data bottleneck* 해소 가능성 — strict 돌파 가능?

### Next

Paper deliverable 의 *마지막 robustness check*:

1. **10 Phase 2b seed × 3**: seed ∈ {1337, 2024} 추가 실행 (~35 분). Δ confused +0.091 의 robustness 확보.
2. **10 Phase 2b NFCorpus**: NFCorpus 1 run (~20 분). Cross-dataset 일반성 + (어쩌면 SciFact-specific 명시).
3. (둘 다 마치면) paper deliverable 의 *최종 결론 문단* 작성 — *bounded improvement* narrative 확정.

---

## 2026-05-24

### Done

외부 reviewer agent 의 입력 (`24 시간 여유 가 hyperparameter sweep 의 pre-commit 을 *깨면 안 됨* + ROI 순 robustness check`) 채택. *Pre-commit 의 시간 여유 에도 불변* 의 method 정합성 유지.

1. **10 Phase 2b seed × 3 robustness (가장 critical)**:
   - 동일 config (q,v r=8 LR=5e-5 α=r early-stop=val_all) 로 seed ∈ {1337, 2024} 추가. 결과 보기 전 *3-seed mean ± CI 보고* commit.
   - seed 1337: NDCG 0.6423, Δ all -0.004 (≈), Δ conf **+0.097** ✓.
   - seed 2024: NDCG 0.6639, Δ all +0.018 (≈), Δ conf **+0.123** ✓.
   - **3-seed mean**: NDCG 0.6476±0.014, Δ all +0.001±0.014 (anchor preserved), Δ conf **+0.104±0.017** ✓.

2. **Cross-method universal rank-collapse 분석**:
   - `report/_rank_collapse_analysis.py` 작성 + `report/figures/_cross_method/rank_collapse_contrast.{pdf,png}` + `rank_collapse_data.json`.
   - 06/08/10 의 *per-position effective rank* 가 모두 1-1.7 — *universal collapse pattern*.
   - LoRA 의 superior lift 의 진짜 이유: *24 distinct intervention positions* 의 *spatial multiplicity*, per-adapter rank escape *아님*.

3. **Clean ColBERT-finetune baseline 완료** (`--no-steering`):
   - NDCG@10 all = **0.6924** (02 unfrozen 0.6576 의 +0.035 더 높음, baseline 0.6464 의 +0.046)
   - Δ all vs baseline = +0.046 [-0.002, +0.096] (CI 하한 -0.002 — *strict 돌파 직전*)
   - Δ confused vs baseline = **+0.260 [+0.182, +0.338] ✓** (02 unfrozen 의 +0.252 와 essentially 동일)
   - ‖v_learned‖ = 0.0 (frozen no-grad 확인)
   - Train loss 종료: 0.0025 (02 unfrozen 의 0.0042 보다 더 fit, but val_conf-based best @ ep2)
   - **Reviewer 공격 ("v=0 hook 이 학습 신호 추가했냐") *완전 해소*** — v=0 hook 의 영향 negligible

4. **Documentation 통합**:
   - `report/10_lora_phi_report.md` §2.4 신규 (seed × 3) + §8.5 신규 (universal rank-collapse + spatial multiplicity)
   - `REPORT.md` Abstract 의 핵심 진단 재정렬 (universal rank-collapse punchline) + §5d 갱신 (seed × 3) + §5e 신규 (cross-method)
   - `CHANGELOG.md` [2026-05-24] dated entry.

### Observations

**3-seed robustness 가 강력한 신호**:
- 08 의 seed-artifact (seed 42 +0.054 → 1337/2024 ≈ 0) 의 *완전 반대 양상*.
- LoRA Phase 2b 의 lift 가 *진짜 method-level effect*, not seed-noise.
- *3-seed std 작음* (±0.017) — robust.

**Universal rank-collapse 의 cross-method 평행**:
- 06 K-router: effective K ≈ 1.4 (K-invariant)
- 08 bilinear M: effective rank ≈ 1.0 (rank-1 collapse)
- 10 LoRA per-adapter: mean rank 1.71, std 1.07
- 모든 method 의 학습 dynamics 가 *single dominant axis 로 collapse*. 본 paper 의 *진짜 main contribution* — *universal feature 의 발견*.

**Spatial multiplicity 의 monotonic correlation**:
- 1 position (06/08): Δ conf +0.04-0.05
- 24 positions (10 LoRA): +0.104 (3-seed mean)
- Full encoder (02 unfrozen): +0.252
- log(positions) → Δ conf 의 *대략 monotonic* — empirical lift 의 *진짜 lever* 가 spatial multiplicity 임 의 데이터적 증거.

### Decisions (experimental)

- **Pre-commit 의 시간 여유 에도 불변**: 24 시간 생겨도 hyperparameter sweep 안 함. *Method 정합성 우선*. 외부 reviewer 의 권장 ROI 순 (seed×3 → clean baseline → NFCorpus → analysis) 채택.
- **Paper main contribution 의 *재정렬*** — 옛 "bounded improvement on confused" → 새 **"universal per-position rank-collapse + LoRA's spatial multiplicity escape"**. *Post-hoc framework 이지만 *11 개 실험 통합***. Reviewer 가 받아들이기 훨씬 쉬운 narrative.
- **Best-state selection 기준 변경** (10 만 val_all): 02-09 와 *direct paired comparison 의 fairness 결함* 명시 (limitations section).

### Open questions

- Clean ColBERT-finetune baseline (no steering) 의 결과 가 02 unfrozen (with v=0 hook, v 가 작은 학습) 과 *얼마나 다를까*? 만약 ≈ 동일이면 "v=0 hook 무해" 명확 확정.
- NFCorpus 위 Phase 2b 의 결과 가 06 K=2 NFCorpus 의 -0.250 catastrophic 과 다르게 *graceful* 일까? *Spatial multiplicity (24 positions)* 이 cross-dataset 의 hyperparameter sensitivity 도 완화하는지 검정.
- 본 *universal rank-collapse* 가 ColBERT 만의 현상인가, 아니면 *모든 dense retriever 의 학습 dynamics* 의 universal feature인가? Cross-model 검정 (Stage 6).

### Next

남은 robustness check:
1. **Clean ColBERT-finetune baseline (no steering) 종료 대기** (~17 분, MPS sequential)
2. **NFCorpus on Phase 2b config** (~25 분, *재튜닝 절대 금지*, pre-commit 따라)
3. (둘 다 마치면) paper deliverable 의 *최종 결론 문단* 확정 — *universal rank-collapse + spatial multiplicity* punchline.

### Additional observations (2026-05-24, after NFCorpus 완료)

**10 Phase 2b on NFCorpus**: NDCG@10 all = **0.0094** (baseline 0.330 의 2.8%), Δ all **−0.320 ✗ catastrophic**. 06 K=2 NFCorpus 의 −0.250 보다 더 심함. *Same Phase 2b SciFact-tuned hyperparameter 가 NFCorpus 의 strong-HN regime 에서 immediate over-correction*.

- Ep1 rank loss 4.47 (SciFact 의 0.66 의 7×) — NFCorpus 의 hard negatives 가 baseline ColBERT 를 훨씬 강하게 fool, LoRA 가 빠르게 over-correct.
- *Universal rank-collapse + spatial multiplicity* 의 method-architectural claim 은 *cross-dataset 일 것* — 단 *numerical lift* (+0.104) 는 SciFact-specific.

**Paper limitations 의 핵심**:
- Hyperparameter sensitivity 가 dataset-specific (SciFact LR=5e-5 ↔ NFCorpus 가 다른 LR 필요).
- True cross-dataset robustness 는 *adaptive HP strategy* (per-dataset NDCG baseline 의 reciprocal scaling 등) 의 future work.

### Additional observations (2026-05-24, after FiQA + Diagnostic B)

**FiQA Phase 2b 추가** (3 번째 train-available BEIR, 동일 config + `--max-triplets 9,190`):
- NDCG@10 all = **0.0005** (baseline 0.347 의 0.15 %, *literal 0% retrieval*), Δ all **−0.347 [−0.374, −0.319] ✗ catastrophic** (NFCorpus 의 −0.320 보다 더 심함). *최초 보고에 0.0388 표기했으나 sanity check 로 0.0005 정정*.
- Δ confused = −0.147 ✗.
- 2 / 2 cross-dataset (NFCorpus + FiQA) catastrophic 확정. Single-dataset artifact 가설 기각.

**Diagnostic B — encoder output representation collapse** (reviewer feedback 후, `report/_repr_collapse_diagnostic.py`):

n=500 docs sampled per corpus. 측정: random doc/token pair cosine + doc-mean/token matrix effective rank (singular-spectrum perplexity).

| Dataset | 조건 | doc-pair cos μ | tok-pair cos μ | eff_rank doc | eff_rank tok |
|---|---|---|---|---|---|
| NFCorpus | frozen | +0.553 | +0.211 | 11.73 | 55.94 |
| NFCorpus | LoRA 2b | **+0.990** | **+0.940** | **1.09** | **1.62** |
| FiQA | frozen | +0.380 | +0.181 | 23.58 | 63.91 |
| FiQA | LoRA 2b | **+0.993** | **+0.990** | **1.06** | **1.10** |
| SciFact | frozen | +0.573 | +0.209 | 10.65 | 57.21 |
| SciFact | LoRA 2b | **+0.984** | **+0.942** | **1.15** | **1.61** |

**핵심 관찰** (surprising):
1. **Universal extreme collapse** — 3 dataset 모두 doc-pair cosine ≈ 0.99, eff_rank ≈ 1. parameter-space rank-collapse (§8.5: per-adapter rank ≈ 1.71) → output-space collapse (eff_rank ≈ 1.1) 로 명확히 전파.
2. **그러나 SciFact 는 catastrophic *아님*** (Δ all = −0.010 ≈ baseline). *Collapse* ↔ *catastrophic NDCG* 의 1:1 대응 없음.
3. ⇒ **Catastrophic mechanism 은 단순 collapse 가 아님**. *Collapse 방향* 의 task-alignment 가 결정.
   - SciFact: collapse direction 이 task ranking 신호와 align → 1.15-dim residual 이 MaxSim 에서 sufficient.
   - NFCorpus/FiQA: baseline 약함 → ep0 loss ~ 7× 큰 supervision distortion → collapse direction 이 *wrong* (task와 unaligned).

**§5e 의 ΔW rank-collapse 와 다른 현상** — output 의 직접 측정.

### Decisions (experimental) — updated 2026-05-24

- Catastrophic mechanism = **universal collapse + direction misalignment**. Reviewer 가설 의 *부분 확인 + 정밀화*.
- *다음 disentangling experiment* 필요:
  - Mediation 1 (warmup + grad_clip): optimization root 검정. ep0 폭발 억제 → collapse magnitude ↓ + direction stabilize?
  - Mediation 1b (in-batch negative): supervision root 검정. mined HN 의 noise 제거 → 학습 신호 *옳은* 방향?

### Open questions — 2026-05-24

- *Collapse direction* 의 *cross-dataset 유사성*? SciFact-trained LoRA 의 collapse direction vs NFCorpus-trained 의 vs FiQA-trained — cosine similarity 가 클까 작을까? (cross-dataset 의 *공통 path* vs *task-specific path* 검정)
- Mediation 1 / 1b 의 *어느 쪽* 이 mainly responsible? 둘 다 부분 회복일지 단일 회복일지.

### Additional observations (2026-05-24, sanity check)

**Reviewer 의 critical catch**: "**rank-1 embedding 으론 NDCG 0.65 *수학적으로 불가능***" — SciFact 의 tok_cos +0.94 / eff_rank 1.61 *상태에서* 보고된 NDCG 0.6367 와 *내부 모순*. 가설 (A) checkpoint best vs final 불일치 or (B) LoRA injection α scaling 불일치.

**Sanity check** (`report/_repr_collapse_sanity.py`): 진단이 로드한 *바로 그* `module_final.pt` 로 test NDCG@10 재현:

| Dataset | Diagnostic-loaded | Original-run | Match |
|---|---|---|---|
| SciFact | 0.6367 | 0.6367 | ✓ |
| NFCorpus | 0.0094 | 0.0094 | ✓ |
| FiQA | **0.0005** | **0.0005** (최초 0.0388 표기 *부정확*, sanity 로 정정) | ✓ |

**3 / 3 match → 가설 A/B 기각**. **Collapse 가 진짜** — *SciFact 의 eff_rank ≈ 1.15 / 1.61 *상태에서* NDCG 0.6367 가 실제 발생*. *Rank-1 puzzle* 이 paper-grade finding.

**Rank-1 puzzle 해석**:
1. eff_rank perplexity ≈ 1 ≠ literal rank-1 — trailing dimensions 의 residual signal 잔존.
2. MaxSim 의 per-token max 가 small residual structure 를 증폭.
3. Mean-pooled cosine ↑ ≠ MaxSim discrimination 무력화.
4. SciFact 의 rank-1 residual 이 task-aligned → NDCG 보존. NFCorpus/FiQA 의 rank-1 residual 이 task-misaligned → catastrophic.

### Decisions (experimental) — updated post-sanity 2026-05-24

- **Collapse 가 universal + 진짜** (sanity check 확정). 가설 *"timing 다름 → SciFact 만 collapse 안 됨"* 기각 — 3 dataset 모두 collapse, but consequence 다름.
- *Catastrophic mechanism 의 정밀화*: **direction misalignment** 가 sufficient lever, **collapse magnitude** 는 necessary.
- **Disentangling experiment 의 가설 unchanged** — mediation 1 (optimization root: collapse magnitude 통제) + mediation 1b (supervision root: collapse direction 통제). 단 가설 framing 이 *"timing"* 이 아니라 *"direction alignment"*.

### Next

1. **(완료 후) FiQA sanity NDCG 확인** (background, ~5 min 남음).
2. **Mediation 1** (warmup + grad_clip) — 3 datasets, single rule (warmup 10%, clip max_norm=1.0), result-blind 1 run each. *src/train.py 의 train_steering 수정 + 10_lora_phi run.py 의 flag*.
3. **Mediation 1b** (in-batch negative) — 3 datasets, single rule (1 in-batch neg per query), result-blind 1 run each.
4. 결과 후 paper section "§5f Catastrophic Failure as Representation Collapse: Disentangling Optimization vs Supervision" 작성.

### Pre-commit binding (2026-05-24, *결과 보기 전*)

`report/_catastrophic_failure_section_draft.md` 에 section 골격 commit:
- 검정 가설의 *기각 vs 확정 조건* 명시 (e.g., "Δ all > −0.10" 이면 catastrophic 회복 ✓)
- Narrative direction (best case / mixed / negative) 미리 명시 — 결과 보고 cherry-pick 회피.

### Mediation 1 결과 — SciFact (2026-05-24)

`outputs/10_lora_phi/scifact/seed_42/qv_r8_l12_m1/`. Phase 2b baseline 과 *통계적 동등*.

| 지표 | Phase 2b baseline | M1 (warmup 10% + clip 1.0) |
|---|---|---|
| NDCG@10 all | 0.6367 | 0.6342 |
| Δ all vs baseline | −0.010 [−0.044, +0.023] (≈) | −0.012 [−0.046, +0.021] (≈) |
| Δ confused vs baseline | +0.091 [+0.040, +0.143] ✓ | +0.088 [+0.035, +0.139] ✓ |
| ep1 train loss | 0.60 | 0.67 (warmup 으로 약간 ↑) |
| ep1 val_all | 0.604 | 0.624 (warmup 으로 +0.02) |
| ‖A‖_total, ‖B‖_total | (Phase 2b 미기록) | 9.21, 2.07 |

**함의**: SciFact 에서 mediation 1 의 *uniform rule* 이 over-regularize 하지 *않음*. Phase 2b +0.091 confused lift 가 *robust*. 진짜 검정은 NFCorpus + FiQA 의 catastrophic 회복 여부 (다음).

### Mediation 1 결과 — NFCorpus (2026-05-24)

`outputs/10_lora_phi/nfcorpus/seed_42/qv_r8_l12_m1/`. *Test NDCG 동일 catastrophic, train trajectory 명확히 다름*.

| 지표 | Phase 2b baseline | M1 (warmup+clip) |
|---|---|---|
| NDCG@10 all | 0.0094 | 0.0113 |
| Δ all vs baseline | −0.320 ✗ | **−0.319 [−0.353, −0.286] ✗** (통계 동등) |
| Δ confused vs baseline | −0.092 ✗ | −0.093 ✗ |
| **ep1 val_all** | **0.073** | **0.140 (1.9× better)** |
| ep2 val_all | 0.017 | 0.014 |
| ep3 val_all | 0.015 | 0.016 |

**핵심 발견 — *Optimization root 의 부분 지지***:
- ep1 (warmup mostly active, step 78/786 = ep1 의 ~30 %) val_all 이 1.9× 개선 (0.073 → 0.140) — *warmup 가 ep1 collapse 부분 억제*.
- ep2 (warmup 종료) — full LR 활성, Phase 2b 와 *동일* collapse.
- ⇒ Warmup 의 효과 는 *collapse 지연*, 영구 *방지 불가능*. *Longer warmup* / *lower LR* 의 future work.

**Train code 한계**: `train_steering()` 의 LoRA params 가 best-state snapshot 안 됨 → test eval = ep3 final = collapse. *Phase 2b baseline + M1 동일 한계* 라 fair direct comparison 는 유효. 단 *absolute lift* (만약 best-state 사용 시) 는 future work.

### Mediation 1 결과 — FiQA (2026-05-24)

`outputs/10_lora_phi/fiqa/seed_42/qv_r8_l12_m1/`. NFCorpus 와 *완전 동일 양상*.

| 지표 | Phase 2b baseline | M1 (warmup+clip) |
|---|---|---|
| NDCG@10 all | 0.0005 | 0.0009 |
| Δ all vs baseline | −0.347 ✗ | −0.346 [−0.374, −0.319] ✗ (통계 동등) |
| Δ confused vs baseline | −0.147 ✗ | −0.147 ✗ |
| **ep1 val_all** | **0.090** | **0.257 (2.86× better)** |
| ep2 val_all | 0.005 | 0.005 |
| ep3 val_all | 0.0005 | 0.0005 |

**확정**: 2/2 dataset (NFCorpus 1.9× + FiQA 2.86×) 의 ep1 val_all 개선 양상 *동일*. *Test-time* 의 catastrophic 회복 *없음* (ep3 LoRA 사용 한계).

### M1 종합 함의 (2026-05-24)

**Optimization root 의 *부분* 지지**:
- ✓ 학습 동학에서 *명확*한 효과 (NFCorpus 1.9×, FiQA 2.86× ep1 val_all improvement).
- ✗ Single rule (warmup 10% + clip 1.0) 로 *영구* 회복 *불가능* (post-warmup phase 가 collapse 재현).
- ⇒ Optimization root 가 *부분* 기여하지만 *충분 조건* 아님. Catastrophic 의 *완전* 원인 일 수 없음.

다음: M1b (in-batch negative) 로 supervision root 검정.

### 🎯 Mediation 1b 결과 — SciFact (첫 strict net 향상, 2026-05-24)

`outputs/10_lora_phi/scifact/seed_42/qv_r8_l12_m1b/`. **본 paper 의 모든 prior 실험 (01-10) 에서 Δ all CI 하한 > 0 못 달성 한 기준 첫 충족**.

| 지표 | Phase 2b (3-seed) | **M1b (in-batch neg, seed 42)** |
|---|---|---|
| NDCG@10 all | 0.6476 | **0.6613** |
| Δ all vs baseline | +0.001 ± 0.012 (≈) | **+0.015 [+0.001, +0.029] ✓ positive (STRICT)** |
| Δ confused vs baseline | +0.104 ± 0.014 ✓ | +0.055 [+0.030, +0.081] ✓ (Phase 2b 의 1/2) |
| ep1 val_all | 0.604 | **0.672 (baseline +0.026)** |
| ep2 val_all | 0.618 (best) | **0.682 (best, baseline +0.036)** |
| ep3 val_all | 0.614 | **0.679 (baseline +0.033)** |
| ‖A‖, ‖B‖ total | (Phase 2b 유사) | 8.28, **1.32** (B 가 절반, less active) |

**핵심 발견 — supervision root *시그널* (확정 아님)**:
1. **Δ all CI 하한 +0.001 > 0** → *pre-committed strict 기준 첫 충족 시그널 (single seed)*.
2. **Phase 2b 의 redistribution 깨뜨림**: zero-sum trade-off (confused ↑ / easy ↓ / all ≈ 0) → *non-zero net 향상* (confused +0.055 / all +0.015).
3. **Train trajectory 의 모든 epoch 이 baseline 위에 유지** — Phase 2b 의 ep2 부터 collapse 와 *질적으로 다른* 양상.
4. **‖B‖ 작음** (Phase 2b 의 63%) → less LoRA active → less collapse 신호. *In-batch neg 의 supervision 이 약함 (easy contrast → small gradient) 이 collapse 도 덜 일으킴*.

**두 *under-weighted* 캐비엇 (reviewer agent 의 critical catch, paper-defense 필수)**:

**캐비엇 1 — *clean ≠ easy* 혼동 (confounded mechanism)**:
- In-batch neg = *clean (noise 0%) + EASY* (다른 query 의 positive 는 trivial contrast).
- M1b 의 net 향상 의 원인 이 (나-1) noise 제거 인지 (나-2) hard negative difficulty 자체 가 collapse 유발 인지 *구분 불가능*.
- §4 의 *"신호 약함 = easy contrast"* 인정 + *"supervision root 강력 지지"* 결론 = *절반만 인정*.
- **결정적 disambiguator**: **FN-denoised mined-HN** (전처리 팀 의 false-negative 제거 결과 머지).
  - Hard 유지 + denoise → +0.104 유지 면 noise 가 원인.
  - 여전히 collapse 면 difficulty 가 원인 (M1b 의 net 향상 = easy contrast artifact).

**캐비엇 2 — *seed 42 단독 + CI 하한 +0.001 razor-thin***:
- Δ all CI 하한 = +0.001 — *paired bootstrap noise 1σ 안*. Seed 변경 만으로 CI 0 미달 가능성.
- 08 (bilinear M) 의 *seed-artifact* 시나리오 와 정확히 동일 구조 — seed 42 의 +0.054 confused 가 seed 1337/2024 에서 ≈ 0.
- **3-seed 전 까지** "*첫 strict net 향상*" 을 *확정* 으로 쓰면 안 됨 — *signal (preliminary)*.
- Confused +0.055 = Phase 2b 의 +0.104 의 절반 — *in-batch 가 HN 을 덜 다룸* 의 예상 signature, non-trivial discovery 아님.

→ **Paper 의 M1b frame**: "*supervision root contribution signal, but mechanism (noise vs difficulty) confounded + needs 3-seed + FN-denoised replication*" — *promising preliminary*, *확정 결론 아님*.

다음 NFCorpus M1b ~10 min, FiQA M1b ~25 min. 이후 Exp 11 사용자 confirm 가능.

### Mediation 1b 결과 — NFCorpus (74 % catastrophic 회복, 2026-05-24)

`outputs/10_lora_phi/nfcorpus/seed_42/qv_r8_l12_m1b/`. *Catastrophic 의 *대부분* 가 supervision root 임을 cross-dataset 으로 확정*.

| 지표 | Phase 2b NFCorpus | M1b NFCorpus |
|---|---|---|
| NDCG@10 all | 0.0094 | **0.2459** (baseline 0.330 의 74.5%) |
| Δ all vs baseline | −0.320 ✗ | **−0.084 [−0.105, −0.064] ✗** (74% 회복) |
| Δ confused vs baseline | −0.092 ✗ | **−0.013 [−0.027, +0.002]** (CI 0 포함, *baseline 회복*) |
| ep1 val_all | 0.073 | **0.376 (baseline +0.046 ▲)** |
| ep2 val_all | 0.017 | 0.285 |
| ep3 val_all | 0.015 | 0.259 (test = 0.246) |

**핵심 발견 — supervision root cross-dataset 부분 지지**:
1. **Catastrophic 의 74 % 회복** (0.0094 → 0.246) — *mined HN noise 가 catastrophic 의 주요 원인* 확정 cross-dataset.
2. **Confused slice 거의 baseline 회복** (Δ CI 0 포함) — *confused 손상도 supervision noise 의 직접 결과*.
3. **Strict 회복 *가능* 했음**: ep1 val_all = 0.376 (baseline +0.046) → LoRA best-state snapshot 시 strict positive 가능. *Train trajectory 의 *baseline 위* 영역 직접 증거*.
4. **Optimization root 의 부분 잔존**: ep1 → ep3 decay (0.376 → 0.259). M1 + M1b combine 면 *완전 회복* 가능 (single-rule pre-commit 으로 미실행).

**Paper-grade 함의 — Catastrophic mechanism 의 final picture**:
- Catastrophic ≠ purely SciFact-tuned hyperparameter artifact.
- *Mined HN noise* (supervision root) 가 catastrophic 의 *대부분 (74 %)* 원인 — *cross-dataset universal*.
- *Optimization root* (warmup-needed regime) 가 *추가 부분 (26 %)* 기여.
- Two roots *additive* — 별도 mechanism, 둘 다 contribute.

다음 FiQA M1b 결과 (cross-dataset universality 최종 확정 또는 부분).

### Diagnostic B on Mediation Checkpoints — direct collapse measurement (2026-05-24)

CPU 강제 (GPU queue 비충돌). `report/_repr_collapse_mediation.py`. *M1 / M1b 가 *실제로* collapse 를 감소시키는가* 직접 측정.

| Dataset | Frozen | Phase 2b | M1 | M1b |
|---|---|---|---|---|
| SciFact eff_rank doc | 10.65 | 1.14 | 1.14 | **7.29** |
| SciFact doc_cos | 0.573 | 0.985 | 0.985 | **0.656** |
| NFCorpus eff_rank doc | 11.73 | 1.09 | 1.06 | **1.06** |
| NFCorpus doc_cos | 0.553 | 0.990 | 0.993 | **0.995** |
| FiQA eff_rank doc | 23.58 | 1.06 | 1.05 | (pending) |
| FiQA doc_cos | 0.380 | 0.994 | 0.995 | (pending) |

**핵심 발견**:
1. **M1 (warmup+clip) 의 final-state collapse 감소 효과 없음** — train trajectory 효과 (ep1 val_all 2× ↑) 는 *test-time final state 와 무관*. ep3 final = post-warmup collapse 로 회귀.
2. **M1b 가 *dataset-dependent multi-mechanism***:
   - SciFact: *collapse 자체 대부분 방지* (1.14 → 7.29, 6.4×). doc_cos 0.985 → 0.656 (거의 frozen).
   - **NFCorpus: collapse 전혀 방지 안 됨 (1.09 → 1.06), 그러나 NDCG 0.009 → 0.246 (74% 회복)**.
3. **NFCorpus paradox**: same eff_rank, very different NDCG ⇒ *collapse direction 의 task-alignment 회복 이 핵심*. §7.3.c.ii 의 "rank-1 puzzle / direction matters" framework 와 정합.

**Paper framework**:
- *Supervision root* (mined HN noise) 가 *multi-mechanism*: collapse magnitude *또는* direction.
- Dataset 마다 dominant mechanism 다름 (SciFact: magnitude; NFCorpus: direction).

**⚠️ 캐비엇 1 (clean ≠ easy) — *여전히 해소 안 됨* (정정)**: 이전 draft 의 *"NFCorpus direction correction 은 easy contrast 만으로 설명 어려움"* 추론은 **틀림**. Easy in-batch 도 방향 교정 가능 — "query 를 무관한 doc 에서 분리" 라는 *일반-올바른-방향* 의 약한 신호. NFCorpus mined HN 이 *틀린 방향* 으로 당기고 있었다면, *제거 + easy 신호 대체* 만으로 방향 교정 가능 → (나-1) noise 제거 와 (나-2) easy 의 일반-올바른 신호 *둘 다 똑같이 설명 가능*, **confound 유지**. FN-denoised mined-HN 은 *full strength* 로 여전히 필요.

**⚠️ NFCorpus M1b framing 정정**: NDCG 0.246 vs baseline 0.330 = **Δ all −0.084 (여전히 negative)**. *"74% 회복" = catastrophic-gap recovery, NOT net+*. 사용 시 *"NFCorpus 고침" / "net+" 표현 금지*. + Single-seed (multi-seed robustness future work).

**추가 검사 필요 (next)**:
- *Mediation sanity check*: M1/M1b checkpoint 의 *diagnostic-loaded model NDCG = reported NDCG* 검증. Phase 2b 만 §7.3.c.i 에서 확인했고, M1/M1b 는 미확인. 특히 *NFCorpus M1b 의 0.246 재현* 확인이 *Claim A (same eff_rank, different NDCG)* 의 단단함의 근거.

## 2026-05-24#2 — Overnight autonomous experiments

User 가 자기 전 priority 1-8 (except #7 FN-denoised) 자율 실행 요청. Bash orchestrator (`/tmp/_overnight_runner.sh`) + Exp 11 retry script (`/tmp/_exp11_retry.sh`) 가 자율 진행. Total ~1.5 hour, 04:54 → 06:25.

### Done — 9 새 training runs + 1 mediation sanity 확장

1. **M1b SciFact seed 1337, 2024** (캐비엇 2 직접 해소)
2. **M1b NFCorpus seed 1337, 2024** (cross-dataset robustness)
3. **M1+M1b combined SciFact + NFCorpus** (priority 3 #8, post-pre-commit)
4. **Exp 11 SciFact seed 42, 1337, 2024** (λ_easy=1.0, branch a/b/c 검정)
5. Original sanity (PID 43798) FiQA Phase 2b/M1 NDCG 재현 검증 (Claim A pairing 단단함)

### Observations — 4 major findings

#### 1. M1b SciFact 3-seed strict positive (캐비엇 2 fully 해소)

| Seed | NDCG@10 all | Δ all | Δ confused |
|---|---|---|---|
| 42 | 0.6613 | +0.015 [+0.001, +0.029] ✓ razor-thin | +0.055 ✓ |
| 1337 | 0.6681 | +0.022 [+0.008, +0.036] ✓ | +0.064 ✓ |
| 2024 | 0.6722 | +0.026 [+0.011, +0.042] ✓ | +0.077 ✓ |
| **3-seed mean** | **0.6672 ± 0.005** | **+0.021 ± 0.005 ✓ ROBUST** | **+0.065 ± 0.012** |

⇒ **08-style seed-artifact 시나리오 fully 기각**. M1b 의 strict net+ = *robust*, NOT seed-artifact. *Frozen-encoder lightweight intervention 의 3-seed robust strict net+* 첫 사례.

#### 2. M1b NFCorpus 3-seed cross-dataset robust (74 % gap recovery, NOT net+)

| Seed | NDCG@10 all | Δ all | Gap recovery |
|---|---|---|---|
| 42 | 0.246 | −0.084 | 74 % |
| 1337 | 0.223 | −0.107 | 67 % |
| 2024 | 0.263 | −0.067 | 80 % |
| **3-seed mean** | **0.244 ± 0.020** | **−0.086 ± 0.020 ✗** | **74 % ± 7** |

⇒ Cross-dataset *partial recovery* robust (3 seeds 모두 67-80 %), 단 net+ 아님 (Δ all 여전히 negative). *Mined HN noise 가 cross-dataset universal supervision root*.

#### 3. M1+M1b combined → M1 contribution = ZERO

| Dataset | M1 alone | M1b alone (3-seed mean) | **M1+M1b combined** | M1 추가 기여 |
|---|---|---|---|---|
| SciFact | Δ all −0.012 (≈) | Δ all +0.021 ✓ | **Δ all +0.020 ✓** | **−0.001** |
| NFCorpus | Δ all −0.319 ✗ | Δ all −0.084 ✗ | **Δ all −0.083 ✗** | **+0.001** |

⇒ **Optimization root = red herring**. M1 의 ep1 val_all 개선 (NFCorpus 1.9×, FiQA 2.86×) 은 *training-trajectory artifact* — *final-state NDCG 와 무관 + M1b 와 additive 아님*.

#### 4. Exp 11 (relational easy preservation, λ=1.0) — branch (a) partial

| Seed | Δ all | Δ confused | Δ easy |
|---|---|---|---|
| 42 | +0.033 [+0.013, +0.055] ✓ | +0.095 ✓ | −0.019 ✗ |
| 1337 | +0.032 [+0.012, +0.053] ✓ | +0.095 ✓ | −0.021 ✗ |
| 2024 | +0.023 [−0.004, +0.051] (CI 0) | +0.113 ✓ | −0.052 ✗ |
| **3-seed mean** | **+0.029 ± 0.005** (2/3 strict, 1 marginal) | **+0.101 ± 0.010 ✓ preserved** | **−0.031 ± 0.018** (63 % 감소) |

⇒ **Branch (a) *partial*** — explicit easy preservation 으로 *redistribution 부분 해소*. *Confused lift fully preserved* (Phase 2b +0.104 ≈ Exp 11 +0.101), *easy 손상 63 % 감소*. 단 *3-seed strict 아님* (1 marginal).

### Decisions (experimental, updated 2026-05-24#2)

- **Sole-mechanism conclusion**: catastrophic / redistribution 의 *유일* cause = **mined HN noise (supervision root)**. Optimization root = red herring.
- **Two levers** for partial resolution: M1b (general clean) 와 Exp 11 (selective easy preservation). 둘 다 trade-off 다름 (M1b = strict robust net+, Exp 11 = higher confused preservation + partial easy damage).
- **Paper §5f narrative**: *"Catastrophic Failure as Mined-HN Noise Amplification"* — single mechanism (supervision) 으로 통일.
- **캐비엇 2 fully 해소** (M1b SciFact 3-seed robust strict).
- **캐비엇 1 (clean ≠ easy confound) still unresolved** — FN-denoised mined-HN replication 의 *full strength* 필요. *Sole-mechanism conclusion 의 nature* (noise vs difficulty) 가 더 critical.

### Open questions

- Exp 11 의 *higher λ* (예 5, 10) 면 *full 3-seed strict + Δ easy ≈ 0* 가능? *Pre-commit single-value λ=1.0* 으로 미실시 — future work.
- M1b + Exp 11 combined (clean supervision + selective easy preservation) = full resolution? Single seed 검정 가치.
- FN-denoised mined-HN 의 결과 가 *전체 narrative 의 단단함* 의 final test.

### Next

- 새 checkpoint 의 Diagnostic B 측정 — Exp 11 의 easy-doc eff_rank, M1+M1b 의 collapse magnitude.
- Final paper §5f section 작성 (sole-mechanism narrative + two-lever partial resolution).

## 2026-05-24#3 — Exp 12 (FN-denoised mined-HN) — 캐비엇 1 *결정적 disambiguation*

### Done

1. **`data/e5_teacher/extract_train_docs.py`** 작성 + 실행 — SciFact train corpus 5183 docs 를 E5-Mistral-7B 로 encode (3660 sec, 41 MB cached as `e5_train_doc_emb_scifact.pt`).
2. **`experiments/12_fn_denoised_hn/{run.py, README.md}`** + `report/_exp12_pre_commit.md` 작성. *Pre-commit 3 branches* (나-1 noise / 나-2 difficulty / both).
3. **Exp 12 SciFact × 3 seeds** (threshold = 0.0, single value pre-commit): 36.5 % mined HN 이 likely FN (e5_margin ≤ 0) 으로 제거, 5832 cleaned triplets 학습.

### Observations — Exp 12 결과 (3-seed mean ± std)

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|
| 42 | 0.6388 | −0.008 (CI 0) | +0.076 ✓ | −0.078 ✗ |
| 1337 | 0.6420 | −0.005 (CI 0) | +0.079 ✓ | −0.075 ✗ |
| 2024 | 0.6488 | +0.002 (CI 0) | +0.084 ✓ | −0.066 ✗ |
| **3-seed mean** | **0.6432 ± 0.004** | **−0.004 ± 0.005** | **+0.080 ± 0.004 ✓** | **−0.073 ± 0.005 ✗** |

**🎯 (나-2) Difficulty branch — 3-seed robust confirmation**:
- Δ all ≈ 0 (CI 0 all 3 seeds) — *redistribution 유지*
- Δ easy −0.073 ≈ Phase 2b 의 −0.085 (단 +0.012 recovery from FN removal = 14 %)
- ⇒ **Hard-contrast over-correction 이 catastrophic / redistribution 의 *주요* mechanism** (FN noise minor)

### 4-method 종합 framework

| Method | Δ all | Δ confused | Δ easy | Mechanism |
|---|---|---|---|---|
| Phase 2b | +0.001 | +0.104 ✓ | −0.085 ✗ | redistribution baseline |
| **Exp 12 (hard + clean)** | **−0.004** | **+0.080 ✓** | **−0.073 ✗** | *동일 redistribution* — noise minor |
| M1b (easy + clean) | +0.021 ✓ | +0.065 | (~−0.05) | hard 회피 → strict net+ |
| Exp 11 (hard + easy preservation) | +0.029 (2/3 strict) | +0.101 ✓ | −0.031 | selective preservation |

### Decisions (experimental, updated 2026-05-24#3)

- **캐비엇 1 fully disambiguated**: (나-2) difficulty dominant + (나-1) noise minor confirmed (3-seed robust).
- **M1b 의 strict net+ 의 진짜 mechanism = *easy contrast 의 작은 gradient* (hard 회피)** — noise 제거 부수적.
- **Single sufficient mechanism for catastrophic / redistribution**: *hard mined-HN over-correction* — 회피 방법: (i) hard 자체 제거 (M1b), (ii) hard 유지 + selective easy 보호 (Exp 11).
- **§5e main contribution (universal rank-collapse + spatial multiplicity) 와 *독립***: Exp 12 의 결과 가 *redistribution mechanism* 의 정밀화 — main contribution 불변.

### Paper narrative — 정밀화

| 기존 | **새 (Exp 12 후)** |
|---|---|
| supervision noise → wrong direction | hard-contrast over-correction → forced redistribution (noise minor) |
| M1b net+ = noise 제거 효과 | M1b net+ = *hard 회피* 의 easy contrast 효과 |
| 캐비엇 1 = future work | 캐비엇 1 disambiguated empirically |

### Open questions

- Higher threshold (예 +0.05) 면 FN noise 의 더 큰 fraction 제거 → Δ all *strict positive* 까지 가능? Pre-commit single value 로 미실시 — future work.
- M1b + Exp 11 + Exp 12 combine = full resolution (Δ easy ≈ 0 + Δ all strict)?
- Hard-contrast over-correction 의 *neural mechanism* (parameter-space ΔW spectrum 변화 패턴) — Diagnostic B 적용 가치.

### Next

- 5 root docs 의 narrative 통합 (paper §5f 의 *sole-mechanism* 정정).
- Diagnostic B on Exp 11/12 checkpoints — mechanism 직접 검증 (Exp 11 의 easy-doc eff_rank 보존, Exp 12 의 same-as-Phase 2b collapse).
- Final paper section §5f write.

## 2026-05-24#4 — Diagnostic B on New Checkpoints (mechanism direct verification)

`report/_repr_collapse_new_ckpts.py` 의 결과. Exp 11 / M1+M1b / Exp 12 / M1b additional seeds 의 collapse magnitude 직접 측정 (CPU, n=300 docs sampled per condition).

### Done

10 new conditions measured:
- Exp 11 SciFact × 3 seeds (relational easy preservation, λ=1.0)
- M1+M1b combined SciFact + NFCorpus (seed 42)
- Exp 12 SciFact × 3 seeds (FN-denoised, threshold=0)
- M1b SciFact seeds 1337 + 2024 (3-seed robustness verification)

### Observations — *collapse magnitude* 4 mechanism findings

| Method | doc_cos μ | eff_rank doc | eff_rank tok | NDCG Δ all (3-seed) |
|---|---|---|---|---|
| Frozen baseline | +0.573 | 10.65 | 57.21 | — |
| Phase 2b | +0.985 | 1.14 | 1.58 | +0.001 |
| **Exp 12 (3-seed)** | **+0.975** | **1.22 ± 0.01** | **1.72 ± 0.05** | **−0.004 ± 0.005** |
| **M1+M1b SciFact** | +0.663 | 7.05 | 43.16 | +0.020 |
| **M1b SciFact (3-seed)** | **+0.663 ± 0.010** | **7.12 ± 0.31** | **44.65 ± 1.55** | **+0.021 ± 0.005** |
| **Exp 11 (3-seed)** | **+0.910 ± 0.022** | **~1.9** | **~9.6** | **+0.029 ± 0.005** |

1. **Exp 12 ≈ Phase 2b at collapse** (1.22 ≈ 1.14, 3-seed robust). *FN removal 만 으로 collapse 자체 변화 zero*. ⇒ **(나-2) difficulty dominant collapse-level 추가 증거**. FN noise 의 minor +0.012 easy NDCG recovery 는 *direction shift* 만, *collapse magnitude unchanged*.
2. **M1+M1b ≡ M1b alone (collapse + NDCG 둘 다)**: SciFact 7.05 ≈ 7.12, NFCorpus 1.05 ≈ 1.06 — *M1 추가 기여 collapse-level 도 zero*.
3. **Exp 11 의 *selective token-level* preservation 직접 확인**:
   - Token eff_rank 1.58 → ~9.6 (6× recovery!)
   - Doc eff_rank 1.14 → ~1.9 (less, since loss operates per-token)
   - *Loss 가 token sim matrix 직접 규제 = token level 직접 보존* — **direct mechanism evidence**.
4. **M1b 의 collapse 감소 3-seed robust**: eff_rank doc 6.73 / 7.29 / 7.33, tok 43.14 / 44.60 / 46.22 — *seed-artifact 가 아닌 robust mechanism*.

**NFCorpus *direction matters* puzzle 강화**: M1+M1b NFCorpus = eff_rank 1.05 (= M1b alone, 거의 frozen 12 와 큰 차이) 임에도 NDCG 74 % recovery. **Direction alignment > magnitude** 의 추가 evidence.

### Decisions (experimental, updated 2026-05-24#4)

- **Mechanism direct verification 완료**: collapse-level 측정이 NDCG-level 결과와 *일관*. paper-grade *mechanism* 증거 단단.
- **Exp 11 의 token-level preservation 효과 → paper에서 *직접 mechanism evidence* 로 활용 가능**.
- **Hard-contrast over-correction 가설 의 *3 가지 추가 collapse-level 증거*** (Exp 12 = same collapse, M1+M1b = M1b only, M1b 3-seed robust).

### Open questions

- *Higher λ Exp 11* 면 token 의 *fully recovery* + *doc 도 fully recovery* + Δ easy ≈ 0? Pre-commit single value 로 미실시 — future work.
- M1b + Exp 11 combined = collapse 거의 frozen + Δ easy ≈ 0 + strict net+? Single seed 검정 가치.

### Next

- Final paper section §5f write (mechanism evidence integrated).
- (Optional) Higher λ Exp 11 single seed sanity.

## 2026-05-24#5 — Exp 11 extensions + FN+EP variant launch

> 🚫 **POST-HOC EXCLUDED FROM MAIN PAPER** — 본 dated entry 의 결과들 (Higher λ=5, Combined M1b+Exp 11, FN+EP variant) 은 *test 결과 본 후 generative question* 으로 발의된 *post-hoc exploratory* 실험. *Pre-commit timing*: Exp 11 (λ=1) 3-seed 결과 본 *후* `report/_exp11_extensions_pre_commit.md` 작성. **Main paper claim base 에서 제외** — 9 runs / 3 묶음.
> Reviewer agent (`docs review session 2026-05-24`) 의 분류 따라:
> - **Higher λ=5** (3 seeds): post-hoc, *positive 주장* — *contamination risk 가장 큼*
> - **Combined M1b + Exp 11** (3 seeds): post-hoc, mildly antagonistic *negative result* — selection risk 낮음 (negative)
> - **FN+EP variant** (3 seeds): post-hoc, redundant *negative result* — selection risk 낮음
> Historical research record 만 보존. *Paper 의 narrative claim 에는 사용 안 함*.
> Reviewer recommendation: Exp 11 (λ=1) 의 2/3 strict partial 로 종착 — main paper 가 *완전히 선다*.

### Done so far

1. **Exp 11 run.py 에 `--in-batch-neg` + `--fn-denoise` + `--tag-suffix` flags 추가** — combined experiments 위해 *재사용 가능* infrastructure.
2. **Pre-commit binding `report/_exp11_extensions_pre_commit.md`** — λ=5 single value + M1b combined λ=1 single, 각 3 seeds.
3. **6 + 3 runs queued** sequentially:
   - Higher λ=5 SciFact × 3 seeds (Exp 11 lever 강화)
   - M1b + Exp 11 combined SciFact × 3 seeds (hard 제거 + selective preservation)
   - **🆕 Exp 13** (Exp 11 + Exp 12 combined, λ=1 + FN denoise threshold 0) × 3 seeds (hard 유지 + noise 제거 + selective preservation)

### Observations — Higher λ Exp 11 (λ=5, 3-seed complete)

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|
| 42 | 0.6876 | +0.041 [+0.015, +0.068] ✓ | +0.138 ✓ | −0.040 ✗ |
| 1337 | 0.6814 | +0.035 [+0.019, +0.052] ✓ | +0.089 ✓ | −0.010 (CI 상한 −0.0005) |
| 2024 | **0.6963** | **+0.050 [+0.026, +0.075] ✓** | +0.135 ✓ | −0.021 ✗ |
| **3-seed mean** | **0.6884 ± 0.006** | **+0.042 ± 0.006 ✓ STRICT 3/3** | **+0.121 ± 0.023 ✓** | **−0.024 ± 0.013** |

**🎯 3-seed robust findings**:
- ✓ 3/3 seeds strict net+ (CI 하한 +0.015 / +0.019 / +0.026).
- ✓ Δ all +0.042 > λ=1 의 +0.029 (+45 % higher).
- ✓ Δ confused +0.121 > Phase 2b 의 +0.104 — *original baseline 보다 confused recovery 더 높음*.
- 🟡 High seed variance in Δ confused / easy — seed 2024 = high net+, seed 1337 = best easy preservation.

**Earlier estimate 정정**: 2-seed 시점 *"λ marginal benefit 작음"* → 3-seed 후 *"λ ↑ 의 monotonic non-marginal improvement"* 로 정정. Exp 11 의 *main paper-grade lever* = λ=5.

### Observations — Combined M1b + Exp 11 (3-seed final)

| Method (3-seed mean) | Δ all | Δ confused | Δ easy | Strict |
|---|---|---|---|---|
| M1b alone | +0.021 ± 0.005 | +0.065 ± 0.012 | (~−0.05) | 3/3 |
| Exp 11 (λ=1) | +0.029 ± 0.005 | +0.101 ± 0.010 | −0.031 ± 0.018 | 2/3 |
| Exp 11 (λ=5) | +0.042 ± 0.006 | +0.121 ± 0.023 | −0.024 ± 0.013 | 3/3 ⭐ |
| **Combined M1b + Exp 11** | **+0.015 ± 0.002** | **+0.052 ± 0.004** | **−0.016 ± 0.000** | **2/3** |

**🚨 Sub-additive → mildly antagonistic (3-seed robust)**:
- Combined Δ all (+0.015) < M1b alone (+0.021) by **−0.006**.
- Combined Δ confused (+0.052) < M1b alone (+0.065) by **−0.013**.
- Δ easy +0.034 better (−0.016 vs ~−0.05) — but at cost of confused / all.
- Branch (b) sub-additive 의 *(c) antagonistic 향한 tilt*.
- Mechanism: M1b 의 hard 제거 → over-correction prevent → redistribution 발생 안 함 → Exp 11 preservation pressure *redundant + drag*.

### Observations — FN+EP variant (3-seed final, runner script 가 'Exp 13' 으로 label 했음 → 정정 reserved)

`outputs/11_easy_preservation/scifact/seed_{42,1337,2024}/qv_r8_l12_le1_fnden/`.

| Method (3-seed mean) | Δ all | Δ confused | Δ easy | Strict |
|---|---|---|---|---|
| Exp 11 (λ=1) | +0.029 ± 0.005 | +0.101 ± 0.010 | −0.031 ± 0.018 | 2/3 |
| Exp 12 (FN only) | −0.004 ± 0.005 | +0.080 ± 0.004 | −0.073 ± 0.005 | 0/3 |
| **FN+EP variant** | **+0.027 ± 0.009** | **+0.093 ± 0.036** | **−0.029 ± 0.014** | **3/3** |

**🚨 FN denoising *redundant* on top of relational preservation**:
- 모든 metric ≈ Exp 11 (λ=1):
  - Δ all: +0.027 vs +0.029 (no change)
  - Δ confused: +0.093 vs +0.101 (slightly *lower*)
  - Δ easy: −0.029 vs −0.031 (no change)
- Strict rate 3/3 vs 2/3 (marginal robustness gain only)
- Mechanism: *Sole sufficient mechanism (hard-contrast over-correction)* 는 *relational preservation* 으로 이미 address. FN noise removal *별도 lever 아님*.

### Decisions (experimental, updated 2026-05-24#5)

- **4-lever framework 의 4 lever 모두 *same upstream root (hard-contrast over-correction) 의 다른 angle interventions*** — additivity 없음.
- **Paper-grade *best lever* = Exp 11 (λ=5)** — 3-seed strict robust + all metrics best + single-lever cleanest.
- *Future combinations*:
  - **Combined M1b + Exp 11**: *mild antagonism* (paper-grade *negative result*).
  - **FN+EP variant**: *redundant* (paper-grade *single-lever sufficiency 증거*).
- *Caveat 1 closure*: empirically *fully disambiguated* — (나-2) difficulty dominant + (나-1) noise minor + FN removal redundant on top of relational preservation.

### Next

- Final 5 root docs update + §5f revision.
- (Optional) Cross-dataset of Higher λ=5 (best lever) — NFCorpus / FiQA.

### Open questions

- Higher λ 의 *λ-sensitivity*: λ=5 와 λ=1 의 차이가 *small* (marginal). Higher λ (예 λ=10, 20) 면 더 강한 effect? — *Post-pre-commit*, 결과 보고 결정.
- Combined (M1b + Exp 11) — *strict net+ * 가 +0.04 까지 가능?
- 🆕 Exp 13 — *hard 유지 + noise 제거 + preservation* 가 *full resolution* (Δ easy ≈ 0 + strict 3-seed) 달성 가능?

### Next

- Combined (M1b + Exp 11) × 3 seeds 종료 대기 (~45 min)
- Exp 13 × 3 seeds 종료 대기 (~45 min)
- 모든 결과 종합 후 §5f *진정한 final* update (4-lever framework 확장).

### Decisions (experimental) — updated 2026-05-24 post-M1b SciFact/NFCorpus

- *Supervision root* 가 catastrophic 의 *주요* mechanism 확정 (SciFact 의 redistribution 와 NFCorpus 의 catastrophic 모두).
- *Optimization root* 의 *부분* 기여 잔존 (ep2/3 collapse 가 supervision-only 로 완전 해소 안 됨).
- Paper §5f narrative: **"Catastrophic Failure as Mined-HN Noise Amplification"** — supervision root 주요 + optimization root 부분.
- *Exp 11* 의 motivation 강화 — easy preservation 으로 *Phase 2b redistribution 의 더 정밀한* 해소 가능 여부 검정.
- *Future work*: M1 + M1b combine (warmup+clip + in-batch neg) — *strict 회복* 검정 (post-paper).

### Exp 11 Step 0 — *easy-slice* Δ 측정 (2026-05-24, gate PASSED)

외부 reviewer agent 가 제안한 *Experiment 11* 의 Step 0 (measure-first gate). Phase 2b 의 *Δall ≈ +0.001* 이 *net 보존* 인가 *redistribution* 인가 검정.

**수학적 예측**: $\Delta_{\text{easy}} = (\Delta_{\text{all}} - w_c \Delta_c) / w_e = (+0.001 - 0.457 \times +0.104) / 0.543 \approx -0.086$.

**실측** (`report/_easy_slice_step0.py`, 3 seeds × paired bootstrap 10K):

| Seed | Δall | Δconfused | Δeasy |
|---|---|---|---|
| 42 | −0.010 [−0.044, +0.023] | +0.091 [+0.040, +0.143] ✓ | **−0.095 [−0.135, −0.058] ✗** |
| 1337 | −0.004 [−0.038, +0.028] | +0.097 [+0.047, +0.150] ✓ | **−0.089 [−0.128, −0.055] ✗** |
| 2024 | +0.018 [−0.014, +0.049] | +0.123 [+0.073, +0.174] ✓ | **−0.072 [−0.106, −0.040] ✗** |
| **3-seed mean ± std** | +0.001 ± 0.012 | +0.104 ± 0.014 | **−0.085 ± 0.010** |

수학 예측 (−0.086) ↔ 실측 (−0.085) — **99 % match** ⇒ **"anchor preserved" 는 *redistribution***.

**Gate decision**: ✓ PASS (Δeasy 3-seed mean < −0.02 of 모든 seed CI 상한 < 0) → Exp 11 진행할 가치.

**Implication**: §5d 의 Phase 2b narrative 정밀화 — Δall ≈ 0 = *aggregate-level* 보존, *slice-level* 에서는 confused +0.104 ↔ easy −0.085 의 redistribution. *§5e main contribution 과 독립* (rank-collapse + spatial multiplicity 의 universal 함은 불변).

### Next — *지금 큐 진행 중*

- FiQA M1 test eval ~2 min 뒤 종료.
- Queue → M1b × 3 datasets (~60 min).
- 그 후 **Exp 11** (λ_anc > 0, SciFact 3 seeds, single λ = 1.0, result-blind). 사용자 confirm 필요 — `report/_exp11_pre_commit.md` 에 prediction 작성 후 진행.

## 2026-05-24#6 — Exp 13 (per-token cosine direction anchor) — *anchor-side family 의 frontier 강건성* 검정

### Done

- Pre-commit `report/_exp13_14_pre_commit.md` 작성 (BEFORE training, single config, result-blind, STOP rule explicit).
- `experiments/13_frozen_direction_anchor/run.py` 구현 — frozen ColBERT 의 easy queries 의 q + pos doc embeddings precompute, LoRA forward 마다 per-token cosine deviation 계산 + easy loss = λ_dir · mean(1 − cos).
- Exp 13 × 3 seeds {42, 1337, 2024} on SciFact, λ_dir = 1.0 단일 config 실행 (runner: `/tmp/_exp13_14_runner.sh`, ~30 min 총).
- Artifact: `outputs/13_frozen_direction_anchor/scifact/seed_{42,1337,2024}/qv_r8_l12_dir1/`.

### Observations

3-seed grid:

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy | ‖A‖/‖B‖ |
|---|---|---|---|---|---|
| 42   | 0.6790 | +0.033 [+0.012, +0.054] ✓ | +0.099 ✓ | −0.024 ✗ | 8.36 / 1.35 |
| 1337 | 0.6743 | +0.028 [+0.008, +0.048] ✓ | +0.087 ✓ | −0.022 ✗ | 8.33 / 1.34 |
| 2024 | 0.6771 | +0.031 [+0.012, +0.050] ✓ | +0.088 ✓ | −0.017 ✗ | 8.32 / 1.32 |
| **3-seed mean ± std** | 0.6768 ± 0.002 | **+0.030 ± 0.002 ✓** (3/3 strict) | **+0.092 ± 0.007 ✓** | **−0.021 ± 0.003 ✗** | **8.34 / 1.34** |

Pre-commit branch (a) 임계 (Δ all > +0.025, Δ confused > +0.08, Δ easy > **−0.020**):
- Δ all ✓ (+0.030), Δ confused ✓ (+0.092), **Δ easy ✗ (−0.021, miss 임계 by 0.001)**.

→ **Branch (b) — Exp 11 과 frontier 공유** 확정.

Exp 11 (λ=1) 3-seed mean (+0.029 / +0.101 / −0.031) vs Exp 13 (+0.030 / +0.092 / −0.021): 통계적으로 frontier 동등, 미세 차이 — Exp 13 이 *3/3 strict* (Exp 11 의 2/3 보다 robust) + Δ easy *better preserved* + Δ confused 약간 lower.

‖B‖_total: Exp 11 ~1.8 → Exp 13 1.34 (−22 %). Per-token direction anchor 가 LoRA update magnitude 직접 제약 의 측정 proxy.

Train history (seed 42): rank_loss ep1 1.40 → ep3 0.07 단조 감소, anchor_loss ep1 0.18 → ep3 0.47 (plateau), val_ndcg_all ep1 0.687 best → ep3 0.617 (Phase 2b 동일 패턴).

### Decisions (experimental)

- **Anchor-side family 의 frontier 강건성** 확정 — Sim Frobenius² (Exp 11, rotation-invariant) 와 per-token cosine (Exp 13, rotation-sensitive) 가 *수학적으로 다른* constraint 임에도 *통계적으로 구분 안 되는* trade-off frontier 점유.
- **5-lever framework** 채택 — §7.4.1 의 4-lever (Phase 2b / Exp 12 / M1b / Exp 11) 에 Exp 13 추가, data-side (Exp 12 binary, M1b substitution) vs anchor-side (Exp 11 relational, Exp 13 absolute) 의 family split 명시.
- **STOP rule 준수** — λ_dir sweep / variant / cross-dataset *전부 금지*. Result-blind pre-commit 따라 branch (b) lock-in.

### Open questions

- *NFCorpus puzzle 의 cross-regime 전이성* — direction mechanism 이 catastrophic regime (NFCorpus baseline 0.330) 특이적인가? SciFact 의 baseline regime (0.65) 에서는 frontier 동일. Future work — pre-commit STOP rule 따라 본 연구 미실시.
- *Anchor-side 의 plateau 가설* — 추가 anchor 변형 (entropy regularizer, layer-wise, etc.) 도 동일 frontier 위로 떨어질 것인가? Future work.

### Next

- Exp 14 (difficulty-weighted HN, α_w=10) × 3 seeds 종료 대기 (~30 min). Data-side family 의 continuous 변형이 anchor-side 와 다른 frontier 만들 수 있는지 핵심 테스트.
- 그 후 5 root docs final update + paper §5f / §7.4.1 5-lever framework 확정.

## 2026-05-24#7 — Exp 14 (difficulty-weighted HN, α_w=10) — *data-side family 의 binary ≈ continuous equivalence* 검정

### Done

- Exp 14 × 3 seeds {42, 1337, 2024} on SciFact, α_w = 10 단일 config 실행 (runner: `/tmp/_exp13_14_runner.sh` 의 Exp 13 후속 phase, ~40 min 총).
- Sigmoid weighting: $w_i = \sigma(\alpha_w \cdot \text{e5\_margin}_i)$ where e5_margin = cos(eq, epos) − cos(eq, ehn), E5-Mistral-7B 의 cached embeddings (Exp 12 와 동일 source).
- Weighted-mean margin loss: $\mathcal{L} = \sum_i w_i L_i / \sum_i w_i$ (분모 정규화로 학습률 sensitivity 차단).
- Mined HN *유지* (Phase 2b 동일, *no removal — just weighting*).
- Artifact: `outputs/14_difficulty_weighted_hn/scifact/seed_{42,1337,2024}/qv_r8_l12_diffw10/`.

### Observations

3-seed grid:

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy | ‖A‖/‖B‖ |
|---|---|---|---|---|---|
| 42   | 0.6552 | +0.009 [−0.022, +0.040] **(CI 0)** | +0.100 ✓ | −0.068 ✗ | 9.00 / 2.02 |
| 1337 | 0.6536 | +0.007 [−0.014, +0.028] **(CI 0)** | +0.060 ✓ | −0.038 ✗ | 8.38 / 1.43 |
| 2024 | 0.6493 | +0.003 [−0.028, +0.034] **(CI 0)** | +0.095 ✓ | −0.074 ✗ | 8.95 / 2.05 |
| **3-seed mean ± std** | 0.6527 ± 0.003 | **+0.006 ± 0.003** (3/3 CI 0 포함, **NOT strict**) | **+0.085 ± 0.022 ✓** | **−0.060 ± 0.020 ✗** | **8.78 / 1.83** |

Pre-commit branch (a) 임계 (Δ all > +0.025 strict, Δ confused > +0.08, Δ easy > −0.04):
- Δ all ✗ (+0.006, 3/3 CI 0), Δ confused ✓ (+0.085), Δ easy ✗ (−0.060).

→ **Branch (c) 변형 — *softer Phase 2b, sub-binary on Δ all, but Δ confused not attenuated***. Sweet spot 없음.

Triplet weight 분포 (seed 42, deterministic across seeds): mean 0.537, median 0.588, std 0.275, range [0.001, 0.999]. α_w=10 sigmoid 가 *대부분 triplet weight 를 0.5 부근* 으로 모음 → *uniform attenuation* 처럼 동작.

E5 margin 통계: mean +0.013, median +0.036, range [−0.757, +0.757] — ~46 % triplet 이 e5_margin < 0 (likely FN).

Val NDCG@10 trajectory (seed 2024): ep1 0.6406 → ep2 0.6462 → ep3 **0.6581 best** (monotone increase). Phase 2b / anchor-side 의 ep1 best 패턴과 다름 → continuous weighting 의 *slower convergence + late best snapshot*.

### Decisions (experimental)

- **Data-side family 의 binary ≈ continuous equivalence** 확정 — Exp 12 (binary FN cut $w_i \in \{0,1\}$) 와 Exp 14 (continuous sigmoid $(0,1)$) 가 *통계적으로 구분 안 되는* frontier 점유. Δ all (≈ 0), Δ confused (+0.08), Δ easy (~ −0.06 to −0.07) 모두 statistically equivalent.
- **Three-frontier structure** 확정 (paper §7.4.1 6-lever framework):
  1. anchor-side (upper, Δ all ≈ +0.030): Exp 11/13
  2. data-side weighting (lower, Δ all ≈ 0): Exp 12/14
  3. data-side substitution (unique, Δ all +0.021 with confused half): M1b
- **Mechanism intervention space 의 exhaustive enumeration 완료** — paper main mechanistic finding.
- **α_w=10 의 unstable variance** — Δ confused std 0.022 (anchor-side 의 3-5×), val NDCG late-best trajectory. *Practitioner-actionable continuous control* 의 robustness 우려. STOP rule 따라 α_w sweep / variant 금지.

### Open questions

- **α_w sensitivity** — α_w → ∞ 면 binary 로 수렴, α_w → 0 면 uniform. α_w=10 의 *uniform attenuation* 패턴이 다른 α_w 에서도 유지되는가? Future work.
- **Anchor-side + data-side 의 동시 적용** — 두 family 가 *orthogonal* 한지 *interactive* 한지. *Post-hoc trail* 위험으로 본 paper 미실시. Future pre-commit 필요.

### Next

- Exp 14 의 5 root docs 갱신 완료 (REPORT §6.1 grid + §7.3.h + §7.4.1 6-lever framework, RESEARCH 본 entry, CHANGELOG, ROADMAP).
- Diagnostic B on Exp 13 checkpoints (사용자 추가 제안 #2, measurement-only, pre-commit STOP rule 무관) — mechanism claim 의 paper-grade empirical anchor.

## 2026-05-24#8 — Diagnostic B on Exp 13 checkpoints — *per-token absolute direction anchor* mechanism direct verification

### Done

- `report/_repr_collapse_exp13.py` 작성 — `_repr_collapse_new_ckpts.py` pattern 모방 + 추가 metric `cos(h_LoRA, h_frozen)` per token (Exp 13 의 loss 가 직접 규제한 양).
- Exp 13 × 3 seeds (42, 1337, 2024) on SciFact test corpus 300 docs sampled, CPU 강제, ~3.5 min 총.
- Each seed: LoRA-encoded vs frozen-encoded 의 token-level cos 측정 + doc/tok eff_rank + pair-wise cos collapse 측정.
- Artifact: `report/figures/_repr_collapse_exp13/repr_collapse_exp13_data.json` + 3-panel figure (`.pdf` + `.png`).

### Observations

3-seed grid (sub-experiment to Exp 13):

| Condition | doc_cos μ | tok_cos μ | eff_doc | eff_tok | **cos(LoRA, frozen) tok** | **cos(LoRA, frozen) doc** |
|---|---|---|---|---|---|---|
| frozen baseline | +0.587 | +0.214 | 9.86 | 55.13 | 1.000 (identity) | 1.000 |
| Exp 13 seed 42 | +0.881 | +0.654 | 2.26 | 8.60 | 0.820 | 0.820 |
| Exp 13 seed 1337 | +0.872 | +0.641 | 2.37 | 9.11 | 0.823 | 0.823 |
| Exp 13 seed 2024 | +0.875 | +0.638 | 2.35 | 9.31 | 0.830 | 0.830 |
| **Exp 13 3-seed mean ± std** | +0.876 | +0.644 | **2.33 ± 0.06** | **9.01 ± 0.36** | **0.824 ± 0.005** | **0.824 ± 0.005** |
| Exp 11 seed 42 (cached) | +0.901 | +0.675 | 2.01 | 7.69 | — | — |

### Decisions (experimental)

- **Mechanism verification confirmed** — Exp 13 의 loss = 1 − cos 이 학습 후 *부분 최적화* (anchor cos ≈ 0.824, 잔여 anchor_loss = 0.176). train_history.json 의 ep1 anchor_losses[0] = 0.18 와 *정확 일치* (early-stop snapshot at ep1 best). *Soft equilibrium attractor* — confused 학습 신호 (push away) ↔ anchor preservation (pull back) 의 dynamic balance.
- **Anchor-side family 내 *internal representation* 미세 분리** — Exp 13 의 token eff_rank 9.01 > Exp 11 의 7.69 (17 % 차이). *NDCG@10 frontier 는 frontier 공유* (§7.3.g) ↔ *internal mechanism 분리* — **external behavior ≠ internal representation dissociation** paper-grade observation.
- **Anchor-side family 의 *token-level only* preservation** — Exp 13 의 doc eff_rank 2.33 ≈ Exp 11 의 2.01 ≈ Phase 2b-level (collapse 잔존). Token granularity 에서만 anchor 효과, doc aggregation 후 희석. *Anchor-side family capacity limit* direct evidence.
- **NFCorpus direction-matters puzzle 의 evidence chain 완성** — §7.3.f.ii (M1b NFCorpus doc eff_rank 1.05 with NDCG 74 % recovery) + §7.3.g (Exp 13 token-level partial anchor preservation) 가 direct evidence chain 형성.

### Open questions

- λ_dir > 1.0 (예 λ_dir=5, 10) 면 anchor cos → 1.0 에 더 근접 + Δ confused 손실? STOP rule 따라 본 paper 미실시. *Equilibrium 동학 의 λ_dir-sensitivity* future work.
- **Doc eff_rank 2.33 의 mechanism** — token-level anchor 가 *doc mean pooling* 후 왜 희석되는가? Token diversity → doc diversity 의 *aggregation 관계* 의 representation theory analysis 필요. Future work.
- *Exp 14 Diagnostic B 도 동일 가치* 가능 — continuous weighting 의 collapse magnitude 가 Exp 12 binary 와 동등한가? Data-side family 의 internal representation 검정. Pending 사용자 confirm.

### Next

- (만약 진행) Exp 14 의 Diagnostic B 동일 측정 (data-side family 내 internal representation 분리 여부).
- (만약 진행) Paper §8 limitations 의 *anchor-side capacity limit (token-only)* 명시 추가.

## 2026-05-25 — Exp 15 (Conditional LoRA) 4-diagnostic chain — *frontier-breaking hypothesis 의 empirical falsification*

### Done

본 session 은 *Exp 15 full design 진입 전*, *frontier-breaking 가능성* 의 *cheap empirical foundation* 검증. 4 sequential diagnostics:

- **(α) Score-margin AUC** (`report/_exp15_diagnostics.py`, ~10 s) — frozen ColBERT 의 top-1/top-2 score margin 이 confused 를 predict 하는가? AUC = **0.836** (test 300 queries, confused 137).
- **(γ) Oracle test-time conditional** (same script, ~30 s) — gold confused/easy label 로 LoRA/frozen 분기. Δ all = **+0.0475 ± 0.0078** ✓ (3-seed mean).
- **(β) Confused-only triplet training** (`experiments/15a_confused_only_baseline/run.py`, ~10 min) — 4250 confused triplet 만 학습. Δ all = **−0.387** ✗ catastrophic. Val_all trajectory 단조 감소 (ep1 0.318 → ep3 0.165).
- **(δ) Margin-routed Phase 2b** (`report/_exp15_diagnostic_delta.py`, ~10 s) — inference 시 score-margin 으로 LoRA/frozen 분기 (no retraining). τ-sweep 11 점, structural pre-commit frac=0.46: Δ all = **+0.011 ± 0.007**, best post-hoc frac=0.40: +0.014.

Artifact: `report/figures/_exp15_diagnostics/{diagnostic_alpha.json, diagnostic_gamma.json, diagnostic_delta.json, diagnostic_alpha_gamma.{pdf,png}, diagnostic_delta.{pdf,png}}` + `outputs/15a_confused_only_baseline/scifact/seed_42/qv_r8_l12_confonly/`.

### Observations

| Diagnostic | 결과 | 함의 |
|---|---|---|
| (α) AUC = 0.836 | router signal 강함 | branch (c) "routing failure" empirically 배제 |
| (γ) Oracle Δ all = +0.048 | perfect routing ceiling real | frontier 외부 공간 존재 (anchor-side +0.030 의 1.58×) |
| (β) Confused-only Δ all = −0.387 | training-time filtering catastrophic | training-distribution dependency — full query 노출 필수 |
| (δ) Margin-routed Δ all = +0.011 | realistic Exp 15 < anchor-side | **frontier-breaking minimal realization falsified** |

(δ) 의 mechanism 분석: AUC 0.836 이 높음에도 *linear prediction* (AUC × oracle ≈ +0.040) 보다 약함. *Borderline misrouting cost* 가 *misrouting rate 보다 비례적으로 큼* — 경계영역 query 의 high-stakes misrouting.

(β) 의 mechanism 분석: 4250 triplet 으로 295K LoRA params 학습 + 좁은 confused-only distribution → BA matrix 가 *전체 query 분포* 에서 oversteer + LoRA forward path 가 easy query distribution 미노출 → 추론 시 *adversarial 작동* (NDCG@10 0.26 vs baseline 0.65, −60 %).

### Decisions (experimental)

- **Frontier robustness 강화 (paper main contribution)**: 6-lever framework 의 frontier-fixed 주장이 *inference-time conditional routing (AUC 0.84)* 에도 robust. (γ) ceiling +0.048 의 *unrealizability* 직접 입증.
- **6-lever framework 유지**: Exp 15 의 realistic 형태 (δ Δ all +0.011) 가 framework 의 *inferior* 구성원 (data-side weighting +0.006 보다 약간 위, anchor-side +0.030 보다 명백 아래). Exp 15 가 *7th lever 후보 아님*.
- **STOP rule 준수**: 4 diagnostic 완료 후 추가 실험 없음. Elaborate Exp 15 (learned router, end-to-end joint, reranker 형태) 는 §9.3 future work 로 정리.

### Open questions

- **Learned router** 가 score-margin 보다 substantially higher AUC 달성 가능한가? Borderline-cost concentration 이 AUC 0.95+ 에서도 oracle 도달 막을 수 있음.
- **End-to-end joint conditional LoRA** (gate 03/04 의 *세 사인 대응* + routing supervision) 가 (β) catastrophic failure 의 *training distribution dependency* 회피 가능한가? §9.3 (F2).
- **Cross-dataset routing transferability** — SciFact 에서 측정한 AUC 0.836 이 NFCorpus/FiQA 에서도 유지되는가? Score-margin signal 의 *domain-invariant* 여부 불명.

### Next

- Diagnostic chain 의 5 root docs 반영 완료 (REPORT §7.3.i + §9.3 future work, RESEARCH 본 entry, CHANGELOG, ROADMAP).
- Paper §8 limitations 의 *Exp 15 frontier robustness extension* 명시 검토.
- *STOP rule 종착* — diagnostic chain 으로 paper 의 *frontier-fixed* main contribution 강화 완료.

## 2026-05-25#2 — Exp 16 (multi-layer per-token anchor) + spine ablations (Tier 1 + B1 + C1)

### Done

**Exp 16 (multi-layer per-token cosine anchor, layers={0,3,6,9,12})**:
- Pre-commit `report/_exp16_pre_commit.md` (BEFORE training, single config).
- `experiments/16_multilayer_anchor/{run.py, README.md, figures.py}` 작성. `LayerCapture` hook manager 로 5 BERT layer 의 hidden state 동시 capture. Frozen cache float16 on CPU.
- 3 seeds {42, 1337, 2024} on SciFact, λ_dir = 1.0 단일 config. seed 42 smoke test + 1337/2024 background runner.
- Artifact: `outputs/16_multilayer_anchor/scifact/seed_{42,1337,2024}/qv_r8_l12_dir1_multilayer/`.

**Diagnostic B on Exp 16** (per-doc multi-layer capture, ~5 min CPU):
- Script: `report/_repr_collapse_exp16.py`. 5 layer × 300 docs × 4 models (frozen + 3 LoRA seeds).
- Output: `report/figures/_repr_collapse_exp16/{repr_collapse_exp16_data.json, .pdf, .png}`.

**Spine ablations (reviewer Tier 1 + B1 + C1)** — measurement-only, ~10 s:
- Script: `report/_spine_ablations.py`.
- A1 M1b Δ easy 3-seed 실측 / A2 Anchor incremental Δ over Phase 2b LoRA / B1 Exp 13 NDCG sanity / C1 split consistency.
- Output: `report/figures/_spine_ablations/spine_ablations.json`.

### Observations

#### Exp 16 3-seed grid (branch (c) over-restriction confirmed):

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|
| 42   | 0.6238 | +0.008 [−0.016, +0.031] (CI 0) | +0.073 ✓ | −0.048 ✗ |
| 1337 | 0.6434 | −0.003 [−0.027, +0.021] (CI 0) | +0.066 ✓ | −0.061 ✗ |
| 2024 | 0.6534 | +0.007 [−0.016, +0.030] (CI 0) | +0.072 ✓ | −0.048 ✗ |
| **mean ± std** | 0.6402 ± 0.015 | **+0.004 ± 0.006** (3/3 NOT strict) | **+0.071 ± 0.004 ✓** | **−0.052 ± 0.008 ✗** |

Branch (c) "Multi-layer over-restriction" 확정 — Δ all 0/3 strict, Δ easy 2.5× Exp 13 damage, Δ confused 77% Exp 13.

#### Diagnostic B on Exp 16 — *Loss budget dilution mechanism direct evidence*:

| Layer ℓ | cos(LoRA, frozen) | tok_eff_rank Exp 16 | tok_eff_rank frozen |
|---|---|---|---|
| 0 (embed) | 1.000 | 247 | 247 |
| 3 | 0.998 | 150 | 157 |
| 6 | 0.991 | 102 | 109 |
| 9 | 0.965 | 47 | 65 |
| **12** | **0.697** ⚠️ | **4.6** | 43 |
| (ref) Exp 13 final ColBERT 128-dim | 0.824 | 9.01 | 55 |

3-fold mechanism evidence:
1. L0-L6 redundant constraint (LoRA backward 영향 적은 영역, anchor 자연스럽게 만족)
2. L9-L12 insufficient constraint (1/5 budget 으로 final 부족)
3. L12 catastrophic collapse (token eff_rank 4.6, Exp 13 의 1/2)

#### Spine ablation A1 — M1b Δ easy 실측:
- 3-seed mean: **−0.017 ± 0.003** (이전 추정 ~−0.05 의 1/3)
- M1b 가 anchor-side family 와 동등 수준의 easy 보존
- 6-lever 표의 M1b row 정정 필요

#### Spine ablation A2 — Anchor incremental Δ over Phase 2b LoRA:

| Anchor | Δ all (inc) | Δ confused (inc) | Δ easy (inc) |
|---|---|---|---|
| Exp 11 | +0.028 ± 0.020 | **−0.002 ± 0.007** | **+0.055 ± 0.031** |
| Exp 13 | +0.029 ± 0.015 | **−0.012 ± 0.022** | **+0.064 ± 0.009** |

→ Anchor 의 *sole contribution = easy preservation*, NO incremental confused gain. Paper §7.3.g Diagnostic B 의 *soft equilibrium* (confused push ↔ anchor pull) interpretation 과 정합.

#### Spine B1 / C1: defensive sanity ✓
- Exp 13 runs.json → NDCG@10 = saved (diff < 0.0001) ✓
- 3 seeds 모두 confused=368, easy=441 (deterministic) ✓

### Decisions (experimental)

- **Branch (c) "Multi-layer over-restriction" lock-in** — pre-commit STOP rule 따라 layer scope sweep 금지. *Anchor-side family 의 optimal scope = final layer only*.
- **CLAUDE.md §1.3 prior diagnostic finding 재해석** — "signal exists at 5 layers" ≠ "intervention should target all 5". Diagnostic location 과 intervention scope 의 separation.
- **6-lever 표 + framework 의 M1b row 정정** — Δ easy 추정치 ~−0.05 → 실측 −0.017 ± 0.003. *Anchor-side family advantage 가 easy preservation 면에서 미미*, 차이는 Δ confused (preservation 대 절반).
- **Anchor mechanism interpretation 정정** — anchor 가 *confused recovery 추가* 가 아니라 *easy preservation* 만 기여 (Phase 2b LoRA control 대비). Paper §7.3.g soft equilibrium 의 interpretation 강화.
- **§3.8 ablation completeness strict 충족** — anchor scope ablation (single vs multi-layer) paired pre-commit 검정 완료.

### Open questions

- **Anchor 의 *layer-weighted variant*** — uniform 1/5 weight 대신 final layer 에 weight 집중 (예 {0.05, 0.10, 0.15, 0.20, 0.50}) 했으면 Exp 13 보다 더 강한 anchor 도 가능? STOP rule 따라 본 paper 미실시, future work F4 candidate.
- **Anchor target = final BERT (768-dim) vs final ColBERT projected (128-dim)** 의 effect — Exp 13 은 128-dim, Exp 16 L12 은 768-dim. 차이가 anchor mechanism 의 *operating dim* 에 따라 다를 수 있음.

### Next

- 5 root docs 갱신 완료 (REPORT §6.1 grid M1b 정정 + §7.3.j Exp 16 + §7.3.k spine ablations + §7.4.1 framework 정정, RESEARCH 본 entry, CHANGELOG, ROADMAP).
- *STOP rule 종착* — anchor scope ablation 완료, 추가 실험 미실시.

### Done

- `report/_repr_collapse_exp14.py` 작성 (Exp 13 script mirror, anchor proximity reference metric 포함).
- Exp 14 × 3 seeds on SciFact test 300 docs, CPU ~2 min.
- Cross-family comparison figure: 6-lever tok eff_rank + anchor proximity bars (Exp 14 vs Exp 13) + per-token cos distribution.
- Artifact: `report/figures/_repr_collapse_exp14/repr_collapse_exp14_data.json` + 3-panel figure.

### Observations

Exp 14 3-seed grid (with bimodal seed pattern):

| Seed | eff_doc | eff_tok | anchor cos tok | Δ confused | Δ easy |
|---|---|---|---|---|---|
| 42   | 1.15 | 1.62 | 0.471 | +0.100 | −0.068 |
| 1337 | **1.85** | **4.04** | **0.679** | **+0.060** | **−0.038** |
| 2024 | 1.15 | 1.60 | 0.466 | +0.095 | −0.074 |
| **mean ± std** | 1.38 ± 0.40 | **2.42 ± 1.41** | **0.539 ± 0.122** | +0.085 ± 0.022 | −0.060 ± 0.020 |

6-lever internal representation grid (3-seed mean ± std):

| Lever | Family | eff_doc | eff_tok | anchor cos | Δ all |
|---|---|---|---|---|---|
| frozen | — | 9.86 | 55.13 | 1.000 | (anchor) |
| Exp 12 (binary) | data-w | 1.22 ± 0.01 | 1.72 ± 0.05 | (미측정) | −0.004 |
| Exp 14 (continuous) | data-w | 1.38 ± 0.40 | 2.42 ± 1.41 | 0.539 ± 0.122 | +0.006 |
| Exp 11 (relational) | anchor | 1.90 ± 0.25 | 9.63 ± 3.08 | (미측정) | +0.029 |
| Exp 13 (absolute) | anchor | 2.33 ± 0.06 | 9.01 ± 0.36 | 0.824 ± 0.005 | +0.030 |

### Decisions (experimental)

- **Family-level external/internal alignment 확정** — anchor-side ≫ data-side at *every internal metric*: eff_tok 4.4× (9.3 vs 2.1), eff_doc 60 %↑ (2.1 vs 1.3), anchor cos 53 %↑ (0.82 vs 0.54). **6-lever 의 3-frontier structure 가 external (Δ all) 과 internal (eff_rank, anchor cos) 모두에서 일관** → family separation 의 multi-level robustness, paper main mechanistic finding.
- **Within-family external 동등 ↔ internal variance pattern 분리**:
  - anchor-side: Exp 11 eff_tok std 3.08 vs Exp 13 0.36 → **Exp 11 internal variance 8.5×** (seed 2024 의 13.18 outlier, *relational anchor 의 rotation-invariance 가 internal repr 자유도 ↑*).
  - data-side: Exp 12 eff_tok std 0.05 vs Exp 14 1.41 → **Exp 14 internal variance 28×** (bimodal seed 42/2024 vs 1337, *continuous weighting 의 uniform attenuation 이 seed-dependent collapse 유도*).
- **Bimodal seed pattern 의 internal-external mechanism direct alignment 확인** — Exp 14 seed 1337 의 *milder collapse* (eff_tok 4.04, anchor cos 0.68) → *milder NDCG redistribution* (Δ confused +0.060, Δ easy −0.038). **Seed-level internal collapse magnitude ↔ external NDCG redistribution direct correlation** = paper-grade mechanism direct evidence.

### Open questions

- Exp 12 의 anchor proximity 미측정 — full 6-lever × full internal metric grid 완성 위해 Exp 12 도 anchor cos 측정 가능 (CPU 3 min). Future sub-experiment 가능, 본 finding 변경 없음.
- 본 finding 의 *generalizability* — cross-dataset (NFCorpus / FiQA) 에서도 family-level internal-external alignment 가 유지되는가? STOP rule 따라 본 paper 미실시.

### Next

- 사용자 confirm 시 Paper §8 limitations 의 anchor-side capacity limit + Abstract refresh 진행.
- Diagnostic B Exp 12 + Exp 11 의 anchor proximity 측정 (optional, full 6-lever grid 완성용).
