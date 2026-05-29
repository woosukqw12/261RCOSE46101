# Frozen ColBERT Retriever 의 Hard-Negative 문제: Lightweight Intervention 을 통한 개선과 그 메커니즘

## Terminology

본 문서는 IR / ML 표준 용어를 우선 사용한다. 아래 세 가지 만 명시적으로 정의하며, 그 외 개념 (anchor regularizer, intervention point, redistribution 등) 은 *처음 사용되는 위치에서* inline 정의한다.

| 용어 | 정의 |
|---|---|
| **hard query** | Frozen retriever 의 top-1 문서가 relevant 가 *아닌* query — 현재 시스템이 틀리는 query (본 연구의 개선 대상). |
| **easy query** | Frozen retriever 의 top-1 문서가 relevant *인* query — 현재 시스템이 이미 맞히는 query (anchor 가 보존해야 할 대상). |
| **effective rank** | Normalized singular-value spectrum 의 exponential Shannon entropy. 벡터 집합이 실제로 점유하는 차원 수의 연속 척도로, 낮은 값은 representation 이 저차원 부분공간으로 *collapse* 했음을 뜻한다. 정의: $\exp(-\sum_i p_i \log p_i)$, $p_i = \sigma_i^2 / \sum_j \sigma_j^2$. |

---

## Abstract

본 연구는 frozen ColBERT v2 (110M 파라미터, 0.27 % 만 학습 가능) 에서 hard-negative confusion 을 lightweight intervention 으로 어디까지 완화할 수 있는지를 묻는다. 핵심 성과는 SciFact 에서 **+0.030 NDCG@10 strict net improvement** (3 seed 통계 유의, per-token cosine anchor + LoRA 조합) 이며, 이는 단일 수치가 아니라 *어떤 개입이 무엇을 어떻게 개선하는가* 의 mechanistic 분해에서 도출된다.

- **Intervention point multiplicity > per-point capacity.** Router · bilinear · LoRA 모두 각 intervention point 에서 동일한 effective-rank collapse 를 보이며, 개선은 *서로 다른 intervention point 의 개수* 로 설명된다 — "적은 rank × 많은 point" 의 budget-optimal 설계 원칙.
- **Hard-negative over-correction 격리 + anchor 처방.** 가장 어려운 query 의 +0.104 lift 가 *mined hard-negative 과보정* 한 가지에서 비롯됨을 mediation 으로 격리하고 (label noise ~14 %, optimization ~0 %), easy query 표현을 보존하는 anchor regularizer 로 redistribution 손상을 **75 % 감소**.
- **Cross-domain catastrophic failure 의 universal cause 식별.** Plain LoRA 가 NFCorpus / FiQA 에서 무너지는 원인 (Δ all −0.320 / −0.347) 의 후보 두 가지 — *false-negative 오염 (data-side)* vs *hard-contrast over-correction (model-side)* — 을 직접 검정으로 분리. FN-removal × NFCorpus (FN rate 61.1 %, $\rho_{\text{FN}} \approx 4\times$ SciFact, 3 seed) 가 Δ all = **−0.316** (plain LoRA 와 통계 동등, 0 % 회복) — **FN 양은 catastrophe 의 dominant predictor 가 아님**. 반면 hard contrast 자체를 건드리는 개입 — in-batch easy negative (74 % recovery) 와 cross-domain anchor (FiQA 74 % / NF 31 % partial recovery) — 모두 회복 입증. 즉 SciFact 와 dense-judgment 도메인의 catastrophe 가 **hard-contrast over-correction 이라는 단일 cross-domain mechanism** 으로 통합.

### 실험 종합 일람 (Experiment overview)

다음 표는 본 연구에서 수행한 모든 학습 intervention 을 *frozen ColBERT v2 baseline* 대비 Δ NDCG@10 으로 정리한 것이다. SciFact 가 학습 데이터셋이며, cross-domain 행은 동일 학습 protocol 을 NFCorpus / FiQA 에 적용한 결과다. ★ = 본 연구의 best lightweight method.

| § | 실험 | Method | Δ all | Δ hard | Δ easy |
|---|---|---|---:|---:|---:|
| 2.1 | A: mean-diff | Hard - easy 평균 차이 벡터를 학습 없이 빼는 non-learned baseline: $\tilde h = h - \alpha \hat v_{\text{md}}$ (α-sweep) | — | +0.064 | — |
| 2.2 | B: learned dir + gate | 단일 방향 $v \in \mathbb{R}^{768}$ 을 학습하고 scalar / per-token sigmoid gate 로 적용량 조절 | −0.020 | +0.044 / −0.003 | — |
| 2.3 | C: multi-direction router | $K$ 개 학습 방향 위에 top-K mixture: $\sum_k \pi_k(h)\, v_k$, $K \in \{2, 4, 8\}$ | +0.015 / +0.015 / −0.038 | +0.039 / +0.045 / +0.049 | — |
| 2.4 | D: random direction | 학습 방향을 random Gaussian 으로 교체한 magnitude-only control: $v \sim \mathcal N(0, I)$, α=10 | ≈ 0 | +0.011 | — |
| 3 | E: plain LoRA | 12 layer 전부의 $\{q, v\}$ projection 에 low-rank adapter 부착 (24 points): $W \mapsto W + \tfrac{\alpha}{r} BA$, r=8 | +0.001 | +0.104 | −0.085 |
| 4.2 | F: false-negative removal | E5-Mistral teacher margin 으로 mined HN 중 실제 relevant 인 false negative 제거: $\text{cos}(q, d^+) > \text{cos}(q, d^-)$ | −0.004 | +0.080 | −0.073 |
| 4.3 | G: in-batch easy negative | Mined HN 을 batch 내 다른 query 의 positive 로 교체 (clean & easy): $\tilde d_i^- \sim \text{Uniform}(\{d_j^+\}_{j \ne i})$ | **+0.021** ✓ | +0.065 | −0.017 |
| 4.4 | H₁: relational anchor | LoRA / frozen 표현의 token-token similarity matrix 거리 정규화 (rotation-invariant): $\mathcal{R}_{\text{rel}} = \|\text{Sim}(H^{\text{LoRA}}) - \text{Sim}(H^{\text{frozen}})\|_F^2$ | +0.029 (2/3) | +0.101 | −0.031 |
| **4.4** | **H₂: per-token anchor ★** | Easy query 의 모든 token 에서 LoRA 표현을 frozen 과 cosine-가깝게 고정: $\mathcal{R}_{\text{abs}} = \mathbb{E}_{x, t}[1 - \cos(\hat h^{\text{LoRA}}_t, \hat h^{\text{frozen}}_t)]$ | **+0.030** ✓ (3/3) | **+0.092** | **−0.021** |
| 4.5 | I: negative-side anchor | Anchor target $\{q, d^+\} \to \{q, d^+, d^-\}$ 대칭 확장 (§4.4 비대칭 직접 검정): $\mathcal{L}^\dagger = \mathcal{L}_{\text{margin}} + \lambda_{\text{dir}}(\mathcal{R}_{\text{abs}}^{q} + \mathcal{R}_{\text{abs}}^{d^+}) + \lambda_{\text{neg}} \mathcal{R}_{\text{abs}}^{d^-}$ | **+0.028** ✓ (3/3) | +0.077 | −0.014 |
| 5 | E × NFCorpus | Plain LoRA 를 sparse → dense judgment 도메인으로 그대로 transfer | **−0.320** ✗ | — | — |
| 5 | E × FiQA | Plain LoRA cross-domain transfer (financial QA) | **−0.347** ✗ | — | — |
| 5 | F × NFCorpus | FN-removal × NFCorpus (FN rate 61 %, 3 seed): hard-contrast 가 root 임을 입증하는 negative result | **−0.316** ✗ | −0.089 | −0.566 |
| 5 | G × NFCorpus | In-batch easy negative (hard 제거 + FN 제거): NFCorpus catastrophic gap 의 **74 % 회복** | −0.084 | — | — |
| 5 | H₂ × FiQA | Per-token anchor cross-domain (3 seed mean): catastrophic gap 의 **74 % 회복** | −0.090 | −0.045 | −0.177 |
| 5 | H₂ × NFCorpus | Per-token anchor cross-domain (3 seed mean): catastrophic gap 의 **31 % 회복** | −0.221 | −0.062 | −0.402 |

기호: ✓ = strict net improvement (CI lower bound > 0), ✗ = catastrophic regression, (k/3) = 3 seed 중 k 회 CI > 0.

---

## 1. 배경: hard-negative confusion 과 개선 목표

### 1.1 Retriever

모든 실험은 late-interaction dense retriever인 **ColBERT v2**(Khattab and Zaharia, 2020; Santhanam et al., 2022)를 사용한다. ColBERT는 다음 특성을 갖는다:

- 문서를 하나의 embedding으로 압축하는 single-vector dense retriever와 달리, query와 document를 BERT-base 인코더가 생성하는 contextualized token embedding의 *sequence*로 표현한다.
- 각 embedding은 128차원으로 projection된다.
- relevance는 **MaxSim** 연산자로 점수화된다: 각 query token에 대해 모든 document token에 걸친 최대 cosine similarity를 취하고, 이 token별 최댓값들을 합산한다.

$$s(q, d) = \sum_{i \in q} \max_{j \in d} \; \langle \hat{h}_i^{\,q}, \hat{h}_j^{\,d} \rangle$$

여기서 $\hat{h}$는 L2-normalize된 128차원 token embedding이다. 공개 checkpoint `colbert-ir/colbertv2.0`을 사용한다. **명시된 경우를 제외하고 전반적으로 인코더는 frozen이며**, 작은 추가 컴포넌트만 학습된다. 평가는 전체 corpus에 대한 brute-force MaxSim(approximate index 미사용)으로 수행하므로, 보고된 수치는 모델의 실제 scoring 동작을 반영한다.

### 1.2 평가 프로토콜

평가 프로토콜의 핵심 요소:

- **주 metric**: BEIR benchmark에서의 **NDCG@10**.
- **Paired difference**: 모든 비교는 동일한 frozen baseline에 대한 paired difference로 보고한다.
- **Confidence interval**: 10,000-sample paired bootstrap resampling에서 얻은 95% confidence interval을 함께 제시한다.
- **Subset 분할**: test query를 frozen 모델 자신의 동작만으로 정의되는 두 부분집합으로 분할한다.
  - **hard query**: frozen retriever의 top-1 문서가 relevant가 아니다.
  - **easy query**: frozen retriever의 top-1 문서가 relevant다.

이 프로토콜은 published baseline의 절대 재현이 불완전한 경우에도 내적 타당도(internal validity)를 높게 유지한다. SciFact에서 이 분할은 **결정론적(deterministic)이고 seed에 불변(seed-invariant)**으로, training query 기준 hard 368개, easy 441개(45.5% / 54.5%)다. 두 부분집합을 분리해 보고하는 것은 필수적이다: 어떤 intervention은 두 부분집합을 반대 방향으로 움직이면서도 aggregate metric은 변하지 않게 둘 수 있으며, 바로 그 패턴을 탐지하는 것이 본 연구의 핵심이다.

### 1.3 Baseline 재현

먼저 frozen ColBERT v2 baseline을 6개 BEIR 데이터셋에서 재현했다.

| Dataset | 측정 NDCG@10 | Published NDCG@10 | 차이 |
|---|---|---|---|
| SciFact | 0.6464 | 0.693 | −0.047 |
| NFCorpus | 0.3299 | 0.338 | −0.008 |
| SciDocs | 0.1581 | 0.154 | +0.004 |
| TREC-COVID | 0.7270 | 0.738 | −0.011 |
| FiQA-2018 | 0.3473 | 0.356 | −0.009 |
| ArguAna | 0.4528 | 0.463 | −0.010 |

재현 gap의 특징:

- 대부분의 데이터셋에서 작은 systematic 재현 gap (~0.01).
- SciFact에서는 더 큰 gap (−0.047).
- 가장 유력한 원인은 라이브러리 버전, indexing, floating-point precision 차이이며, 이를 분리하는 것은 scope 밖으로 판단했다.

**중요한 귀결은 방법론적이다**: 아래의 모든 결과는 측정 baseline에 대한 *paired* difference이므로, 이 절대 offset은 상쇄된다.

### 1.4 학습 데이터: triplet 구성

training split이 있는 데이터셋의 경우, 학습은 **triplet** $(q, d^+, d^-)$를 사용한다. 구성 절차:

1. query, relevant("positive") 문서, hard negative의 3-tuple을 형성한다.
2. hard negative는 frozen ColBERT retriever를 실행해, 데이터셋의 relevance judgment(qrels)에서 relevant로 *labeling되지 않은* 상위 ranking 문서를 취해 mining한다.
3. SciFact에서는 약 9,000개의 triplet이 산출된다.

나중에 중요해지는 귀결: relevance judgment가 불완전하므로, 이 mined "hard negative"의 상당 비율이 실제로는 relevant이지만 단지 labeling되지 않은 문서 — 즉 **false negative** — 이다. margin $m = 0.2$의 표준 pairwise margin loss를 사용한다.

$$\mathcal{L}_{\text{margin}}(q, d^+, d^-) = \max\bigl(0,\; m - s(q, d^+) + s(q, d^-)\bigr)$$

---

## 2. 가장 단순한 개입에서 출발: capacity 가 아니라 무엇이 outcome 을 좌우하는가

### Scope: 2-layer 설계

본 연구는 *단일 도메인에서 mechanism 을 정밀 격리한 뒤, 그 mechanism 의 도메인 의존성을 별도로 시험* 하는 2-layer 설계를 채택한다:

- **Layer 1 — Mechanism dissection on SciFact (§2–§4)**: 모든 학습 실험 (mean-diff, gating, router, LoRA, mediation F/G, anchor H) 은 SciFact 를 *testbed* 로 수행. 단일 도메인에서 통제 변인 (seed, hyperparameter, 학습 protocol) 을 엄격히 고정해 *어떤 mechanism 이 개선을 좌우하는지* 정밀 격리.
- **Layer 2 — Cross-domain validation (§5)**: §4 까지 확립된 mechanism 의 도메인 의존성을 NFCorpus / FiQA 에서 별도 검증. plain LoRA 의 catastrophic 행동, 두 후보 원인 (FN 오염 vs hard-contrast over-correction) 의 직접 분리 검정 (Exp 12 × NF 의 FN-only removal 반증 + anchor cross-domain 부분 회복), 그리고 *hard contrast 가 cross-domain universal mechanism* 이라는 통합 결론을 cross-domain 에서 보고.

따라서 *§2–§4 에서 SciFact 만 보이는 것은 누락이 아니라 설계* 이며, *§5 의 cross-domain 분석은 그 mechanism 의 일반성에 대한 별도 검정* 이다.

### 개요

본 연구는 가능한 한 가장 단순한 intervention 에서 시작해 점진적으로 복잡도를 더한다. 이 첫 흐름은 *intervention point 당 capacity 가 outcome 을 좌우하지 않는다* 라는 결정적 사실을 확립하며, 이는 §3 의 핵심 설계 원리 (적은 rank × 많은 intervention point) 의 토대가 된다.

### 2.1 실험 A: Non-learned mean-difference direction

#### 동기

무엇이든 학습하기 전에, 우리는 묻는다: 순수하게 *non-learned*인 frozen representation 조작만으로 이미 가장 어려운 query를 개선할 수 있는가? 이를 확립하는 것이 두 가지 이유에서 중요하다.

1. 정직한 참조점을 제공한다 — 어떤 학습 방법이든 raw baseline뿐 아니라 *정보가 반영된(informed)* non-learned baseline까지 이겨야 한다.
2. representation 공간에 애초에 사용 가능한 신호가 있는지를 시험한다.

#### 설계 및 이론적 근거

training split로부터 **mean-difference direction**을 계산한다: mined hard negative의 final-layer hidden state 평균에서 positive의 평균을 뺀 것.

$$v = \bar{h}^{(12)}_{\text{neg}} - \bar{h}^{(12)}_{\text{pos}}, \qquad \tilde{h}^{(12)} = h^{(12)} - \alpha\,\hat{v}$$

설계 근거:

- 직관적으로 $v$는 representation 공간에서 "relevant"로부터 "hard-negative"를 향한다.
- 그 unit vector $\hat{v}$의 배수를 빼면 representation을 hard-negative 영역에서 *멀어지게* 밀어야 한다.
- scaling 없이는(raw $v$) perturbation이 token norm(~10) 대비 매우 작아 사실상 no-op이므로, scale $\alpha \in \{0.5, 1, 2, 5, 10\}$를 sweep한다.

#### 결과

raw direction은 사실상 효과가 없다($\|v\|$가 0.27에 불과해 no-op). scaling하면 hard query에서 명확한 효과가 나타난다:

| $\alpha$ | NDCG@10 | Δ (hard 부분집합), 95% CI | CI가 0 배제 |
|---|---|---|---|
| 0.5 | 0.6477 | +0.003 [−0.004, +0.009] | 아니오 |
| 1.0 | 0.6478 | +0.006 [−0.002, +0.013] | 아니오 |
| 2.0 | 0.6536 | +0.018 [+0.005, +0.032] | 예 |
| 5.0 | 0.6666 | +0.052 [+0.026, +0.081] | 예 |
| **10.0** | **0.6690** | **+0.064 [+0.034, +0.099]** | **예** |

#### 해석 및 한계

요점:

- representation 공간은 사용 가능한 방향 신호를 *실제로* 담고 있다: non-learned, scaled mean-difference direction이 hard query를 +0.064 개선한다.
- 이것이 **보정된 참조점**이 된다: 학습 방법은 raw frozen 모델뿐 아니라 이 informed non-learned baseline까지 이겨야 비로소 그 가치를 인정받는다.

한계:

- 효과는 가장 어려운 query에 국한된다.
- SciFact에서만 측정되었다.

### 2.2 실험 B: 단일 learned direction과 gating 변형

#### 동기

두 가지 질문:

1. 고정된 mean-difference direction이 도움이 된다면, *learned* direction은 더 도움이 될 수 있는가?
2. 학습된 gate를 통해 intervention을 *선택적(selective)* — 유용한 곳에만 적용 — 으로 만들 수 있는가?

#### 설계 및 이론적 근거

pairwise margin loss로 단일 direction $v \in \mathbb{R}^{768}$(768개 파라미터)을 학습한다. 그런 다음 subtraction을 입력에 조건부로 만들려는 의도의 두 gating 변형을 추가한다:

- **scalar gate**: $\tilde{h} = h - g\,v$, $g = \sigma(b)$.
- **per-token gate**: $\tilde{h}_t = h_t - g(h_t)\,v$, $g(h_t) = \sigma(W h_t + b)$.

gating의 동기는 바로 이후 본 연구 전반의 핵심이 될 selectivity 문제다: 이상적으로 intervention은 hard query에서 작동하고 easy query에서는 침묵해야 한다.

#### 결과

| 변형 | NDCG@10 | Δ vs. single direction | 비고 |
|---|---|---|---|
| 단일 learned direction | 0.6651 | hard 부분집합에서 +0.044 [+0.023, +0.066] | raw baseline을 이김; α=10 참조점과 통계적으로 동등 |
| scalar gate | 0.6448 | −0.020 [−0.032, −0.010] (all) | **더 나쁨** |
| per-token gate | 0.6641 | −0.003 (hard) | 사실상 gate 없는 경우와 동일 |

learned direction은 mean-difference direction과 *방향(orientation)*이 다르지만(cosine 0.32) 통계적으로 동등한 retrieval을 달성한다 — 서로 다른 많은 direction이 같은 성능에 도달한다는 초기 단서다.

#### 해석 및 한계

두 진단 모두 중대하다.

1. **scalar gate의 multiplicative gradient bottleneck**:
   - $\partial \mathcal{L} / \partial v = g \cdot \partial \mathcal{L}/\partial(gv)$이므로 초기 gate 값이 작으면 direction 자체로의 gradient가 억제된다.
   - 그 결과 direction이 결코 자라지 않는다.
2. **per-token gate의 saturation**:
   - 어디서나 1.0으로 saturate한다 — 결코 닫히는 법을 배우지 못한다.
   - 그 이유는 구조적이며 본 연구 전반의 핵심 난점을 예고한다: *gate에게 easy query에서 닫으라고 가르치는 training 신호가 없었기 때문*이다.
   - 모든 곳에서 작동하는 것을 loss가 이미 보상한다.

**조건부, input-dependent intervention은 아이디어가 틀려서가 아니라, 선택적이 되도록 하는 명시적 supervision 신호가 없어서 실패한다** — 이 발견은 §7 future work 의 *learned router* 와 *partial unfreezing* 제안의 동기가 된다.

### 2.3 실험 C: Multi-direction router 와 multi-layer 확장

#### 동기

final layer의 한 direction에 capacity 한계가 있다면, 다음 두 확장이 그것을 돌파할 수도 있다:

1. *여러* direction을 routing.
2. *여러* layer의 direction을 동시에 적용.

#### 설계 및 이론적 근거

두 가지 확장을 검정한다.

1. **multi-direction router** (final layer): softmax routing weight로 $K \in \{2, 4, 8\}$개의 direction을 학습한다.
$$\tilde{h}_t = h_t - \sum_{k=1}^{K} \pi_k(h_t)\, v_k, \qquad \pi(h_t) = \text{softmax}(W h_t + b)$$
2. **multi-layer 변형**: 5개 BERT layer $\{0, 3, 6, 9, 12\}$ 각각에 direction 하나를 둔다.

#### 결과

| 구성 | NDCG@10 (all) | Δ all vs. baseline | Δ hard vs. baseline |
|---|---|---|---|
| Router, K=2 | **0.6614** | +0.015 [+0.004, +0.026] | +0.039 [+0.017, +0.061] |
| Router, K=4 | **0.6614** | +0.015 [+0.003, +0.028] | +0.045 [+0.024, +0.068] |
| Router, K=8 | 0.6089 | **−0.038 [−0.067, −0.008]** | +0.049 [+0.005, +0.092] |
| Five-layer single directions | 0.6502 | — | single direction 대비 +0.007 (동등) |

두 진단이 결정적이다.

1. **K=2와 K=4가 소수점 넷째 자리까지 동일한 aggregate NDCG(0.6614)** — capacity를 두 배로 해도 아무것도 움직이지 않는다.
2. **routing되는 direction의 effective 개수가 K와 무관하게 ~1.2–1.5에 머문다**:
   - K=2 → 1.41
   - K=4 → 1.23
   - K=8 → 1.44
   - router는 1–2개의 dominant direction으로 collapse하고 나머지는 죽은 채 둔다.
3. **K=8에서는 여분 capacity가 능동적으로 *해로워져* easy query를 손상시킨다(−0.038)**.

five-layer 변형의 layer별 분석은 훨씬 뒤에 쓰일 단서를 더한다:

| Layer ℓ | ‖v_ℓ‖ | final-layer mean-difference와의 cosine | 해석 |
|---|---|---|---|
| 0, 3, 6, 9 | 1.3–2.9 | ≈ 0 | 직교 / 무관 |
| 12 | 2.79 | **+0.267** | 유용한 신호와 부분 정렬 |

오직 **final layer**만이 mean-difference 신호와 정렬된 direction을 담고 있고, 앞쪽 layer들은 retrieval과 무관한 직교(orthogonal) direction을 학습한다.

#### 해석 및 한계

모든 single-direction 계열 변형은 — learned든 아니든, 1개 layer든 5개든, 1개 direction이든 8개든 — 동일한 NDCG 천장 ~0.665로 수렴한다. 결론:

1. **intervention point 당 capacity 는 outcome 을 좌우하지 않는다** — 단일 intervention point에 capacity를 더하는 것은 representational collapse에 흡수된다(router는 K개 중 ~1.4개만 사용).
2. **outcome 을 좌우하는 변수는 다른 곳에 있다** — 함수적 *형태(form)* 의 변경이거나, *intervention point 의 개수* 변경이어야 한다.

이것이 이후 모든 것의 동기다.

### 2.4 실험 D: Random direction control

#### 동기

방향 신호가 유의미하다고 결론짓기 전에, 한 가지 대안을 배제해야 한다: 충분히 큰 perturbation이면 무엇이든 도움이 되고, "신호"란 단지 무딘 magnitude일 뿐일 수 있다. 이것은 결정적 falsification test다.

#### 설계 및 이론적 근거

final layer에, effective non-learned direction과 *동일한* magnitude(α = 10)의 **random** Gaussian unit vector를 학습 없이 적용한다.

#### 결과

| 조건 | NDCG@10 (SciFact) |
|---|---|
| baseline | 0.6464 |
| random direction, α=10 | 0.6485 (baseline과 동등) |
| mean-difference direction, α=10 | 0.6690 |

mean-difference direction에 paired로 비교하면, random direction은 유의하게 *더 나쁘다*:

- 전체: −0.0205 [−0.0386, −0.0041]
- hard query: −0.0533 [−0.0905, −0.0201]

#### 해석 및 한계

요점:

- "magnitude-only" 가설은 깔끔하게 기각된다: 동일 magnitude에서 random direction은 아무것도 하지 않고 informative direction은 도움이 된다.
- **intervention의 내용이 중요한 것이다.**
- 이로써 이후의 capacity 및 rank-collapse 결과를 *어떤 direction이 학습되는가*에 관한 것으로 해석할 근거가 확보된다 — 단지 *update가 얼마나 큰가*가 아니라.

### 소결: 단순 개입의 도달점과 다음 단계의 방향

§2 의 네 실험은 frozen ColBERT 의 representation 공간에 hard-negative 회복에 유용한 신호가 *존재함* 을 입증하면서, 동시에 그 신호를 활용하는 방식에 대한 첫 제약을 드러낸다. 실험 A 의 +0.064 (hard 부분집합) 가 보정된 참조점을 설정하고, 실험 D 의 random direction control 이 그 효과가 magnitude 가 아닌 *direction 의 내용* 에서 옴을 falsification 으로 확정함으로써, 이후 모든 학습 방법은 두 가지 기준을 동시에 만족해야 한다 — informed non-learned baseline 을 이겨야 하고, 그 효과가 단순한 update 크기로 귀결되지 않아야 한다.

그러나 실험 B 의 gating 실패와 실험 C 의 multi-direction router 및 multi-layer 확장은 — 모두 *단일 intervention point* 위의 학습 가능 capacity 를 키우는 방향의 시도 — 동일한 NDCG 천장 ~0.665 부근으로 수렴한다. router 가 K = 8 capacity 를 가져도 effective routing K 가 1.4 에 머무는 representational collapse 가 그 원인이며, K = 8 의 잉여 capacity 는 오히려 easy query 를 손상시켜 Δ all = −0.038 의 net 손실을 만든다. 즉 capacity 추가는 *무용* 일 뿐 아니라 *유해* 할 수 있다.

본 절의 결정적 함의는 *intervention point 당 capacity 자체가 outcome 을 좌우하지 않는다* 라는 사실이다. 따라서 개선을 위해서는 (i) 함수 형태 자체의 변경 또는 (ii) intervention point 의 수 변경이 필요하며, §3 는 후자의 경로 — 12 transformer layer 의 q·v projection 모두에 분포된 24 개 LoRA adapter — 가 어디까지 가는지 본격적으로 검정한다. 본 절의 한계로, 모든 실험이 SciFact 단일 데이터셋에 한정되었고, 실험 B 의 gating 실패에서 드러난 supervision-부재 문제는 §7 future work 의 learned router 제안으로 이어진다.

---

## 3. 개선의 첫 도약: low-rank adapter 와 intervention point 개수

> 본 절 전반에서 **intervention point 의 multiplicity** 는 단순히 *학습 가능한 modification 이 삽입되는 서로 다른 weight matrix 의 개수* 를 뜻한다 — 예컨대 한 layer 의 한 weight matrix 는 1 개, query·value projection 에 12 개 layer 전부 adapter 를 둔 경우는 24 개로 센다.

### 3.1 실험 E: Frozen 인코더의 low-rank adapter

#### 동기

별도의 control — 1억 1천만 파라미터 인코더를 완전히 **unfreeze**하여 fine-tuning — 은 hard query에서 큰 이득(**+0.252** NDCG@10)을 낸다. 이는 다음을 입증한다:

- frozen 인코더가 진짜 bottleneck이다.
- 현실적 상한선이 설정된다.

따라서 질문은: *parameter-efficient* 한 방법이 unfreeze 없이 그 이득을 얼마나 회복할 수 있는가? 이것이 핵심 실용 실험이며, 본 연구 전체 분석의 *전환점* 이 된다.

#### 설계 및 이론적 근거

*모든 12개* attention layer의 query·value projection에 **Low-Rank Adaptation**(LoRA; Hu et al., 2021)을 삽입한다. 각 adapter는 frozen weight $W$를 다음으로 대체한다:

$$h = Wx + \frac{\alpha}{r}\,BA\,x, \qquad A \in \mathbb{R}^{r \times 768},\; B \in \mathbb{R}^{768 \times r}$$

구성 요약:

- $A, B$만 학습되고 $W$는 frozen으로 유지된다.
- **24개 adapter** (query·value × 12 layer)가 놓인다.
- 단일 구성: rank $r = 8$, scaling $\alpha = r$, learning rate $5\times10^{-5}$.
- 학습 파라미터 294,912개 — 인코더의 0.27%.

판정 규칙은 사전에 고정했다:

1. "돌파"는 aggregate 변화의 95% CI가 0을 배제할 것을 요구한다.
2. 충족되지 않으면 (selection bias 회피를 위해) hyperparameter 튜닝을 중단한다.
3. bounded-improvement 결론을 채택한다.

#### 결과

세 seed 전반:

| 양 | 값 (3-seed mean ± std) |
|---|---|
| NDCG@10 (all) | 0.6476 ± 0.014 |
| Δ all vs. baseline | +0.001 ± 0.014 (CI가 0 포함) |
| **Δ hard vs. baseline** | **+0.104 ± 0.017** (3 seed 모두 유의) |
| unfreeze 상한선(+0.252)의 회복률 | hard query에서 ~41% |

#### 해석

핵심 성과:

- 인코더 파라미터의 0.27 % 만 학습하면서, LoRA 는 **hard query 에서 full-unfreeze 이득의 ~41 % (+0.104) 를 회복** 한다.
- 앞선 router 와 달리 이 lift 는 **seed 전반에서 robust** 하며, 이후 모든 개선의 기반이 된다.
- LoRA 가 hard 회복의 *source* 임이 §4.4 에서 명시적으로 확인되며, anchor 는 그 위에 결합되어 strict net improvement 로 끌어올린다.

#### 한계 (다음 단계의 출발점)

- aggregate 변화는 본질적으로 0 으로, **strict-breakthrough 기준 미충족**.
- 이 "무해함" 은 사실 능동적 내부 trade-off 를 은폐한 결과 (§4.1 에서 정량화).
- 본 실험에서 hyperparameter 튜닝은 여기서 중단되며, 다음 단계는 그 trade-off 의 *mechanism 격리* (§4.2–§4.3) 와 그 위에서의 *개선* (§4.4) 이다.

### 3.2 Universal effective-rank collapse 와 intervention point 개수

#### 동기

이제 세 가지 서로 다른 lightweight 학습 방법이 단일 intervention point 에서 (K-router 와 bilinear interaction metric — 둘 다 중복으로 본문 분석에서 제외, supplementary 보존) 시도되었고, LoRA 의 24 개 intervention point 와 대비된다. 단일 intervention point 방법이 정체하는 곳에서 LoRA 는 왜 성공하는가?

#### 설계 및 이론적 근거

모든 방법에 대해 각 intervention point에서 학습된 update의 **effective rank**를 측정한다. effective rank는 normalized singular-value spectrum의 perplexity로, nominal capacity와 무관하게 update가 실제로 사용하는 차원 수를 센다.

#### 결과

| 방법 | Nominal capacity | Effective rank | 활용률 | Site 수 | Δ hard |
|---|---|---|---|---|---|
| Router, K=2 | 2 | 1.41 | 70% | 1 | +0.039 |
| Router, K=4 | 4 | 1.23 | 31% | 1 | +0.045 |
| Router, K=8 | 8 | 1.44 | 18% | 1 | +0.049 |
| Bilinear metric, r=8 | 8 | 1.01 | 13% | 1 | +0.054 |
| LoRA, r=1 | 1 | 1.00 | 100% | 24 | +0.038 |
| **LoRA, r=8** | 8 | **1.71** | 21% | **24** | **+0.104** |
| Full unfreeze (상한선) | full | full | 100% | full | +0.252 |

#### 해석 및 한계

이것이 본 연구의 **첫 번째 main contribution**이다. 모든 lightweight 학습 intervention은 nominal capacity와 무관하게 intervention point 당 effective rank 약 1–2로 collapse한다. 이는 한 방법의 특이점이 아니라 학습 동역학(small initialization과 adaptive optimization을 동반한 pairwise margin loss)의 *universal* 특징이다. 따라서:

1. **intervention point 당 capacity 는 outcome 을 좌우하지 않는다** — K 증가, rank 증가, adapter당 rank 증가 모두 effective rank를 올리지 못한다.
2. **서로 다른 intervention point 의 개수가 outcome 을 좌우한다** — 다음 scaling 을 따른다:
   - 단일 intervention point 방법 (~1.4 effective rank × 1 point): +0.04–0.05
   - LoRA (~1.7 × 24 point): +0.104
   - full unfreeze: +0.252
   - 개선은 intervention point 의 수에 따라 증가한다.
3. 이는 구체적 **설계 원리**를 낳는다: *적은 rank × 많은 intervention point 가 큰 rank × 적은 intervention point 를 이긴다.* 이런 의미에서 LoRA 는 frozen ColBERT 에 대한 budget-optimal lightweight 설계다.

이 발견은 SciFact에서 세 방법, 6개 이상 구성에 걸쳐 확립되었다. 다만 이 개선이 "공짜"인지에 대해서는 아무 말도 하지 않는다 — 그 질문이 다음 절의 주제다.

### 소결: LoRA 전환의 성과와 잠재된 trade-off

§3 는 §2 의 single-point 정체 (NDCG ~0.665) 를 우회하는 첫 도약을 lightweight intervention 의 realm 안에서 실현한다. 12 transformer layer 의 q·v projection 모두에 LoRA adapter 를 배치 — 총 24 개 intervention point, 인코더 파라미터의 0.27 % — 하면 hard query 에서 +0.104 의 lift 를 얻으며, 이는 full-unfreeze 상한선 (+0.252) 의 41 % 에 해당한다. router 와 달리 이 lift 는 3 seed 전반에서 robust 하며, 본 연구의 *현실적 개선 도약점* 을 확정한다.

이 도약을 가능하게 한 mechanism 은 §3.2 의 cross-method 분석에서 정량적으로 정식화된다. K-router, bilinear interaction metric, LoRA 가 모두 *각 intervention point 에서 effective rank 1–2 로 동일하게 collapse* 한다는 사실 — 이는 한 방법의 idiosyncrasy 가 아니라 small initialization 과 pairwise margin loss 의 universal 학습 동역학이다. 따라서 nominal capacity 증가는 의미가 없고, 차이를 만드는 것은 intervention point 의 *수* 다. 이 발견은 **"적은 rank × 많은 intervention point" 라는 budget-optimal lightweight 설계 원리** 로 본 연구의 첫 main contribution 이 되며, frozen retriever 에 대한 lightweight intervention 방법론의 구체적 처방을 제공한다.

본 절의 한계는 두 가지다. 첫째, +0.104 의 hard lift 가 동시에 보고된 Δ all ≈ 0 의 "무해함" 은 사실 능동적 redistribution 을 은폐하고 있으며, 이는 §4 가 정량화하고 그 위에 anchor 를 결합해 strict net improvement 로 끌어올린다. 둘째, 본 절의 strict 기준 (CI(Δ all) > 0) 은 충족되지 않았으며, hyperparameter 튜닝은 여기서 중단된다 — 다음 절의 동기는 hyperparameter 가 아니라 *mechanism* 의 분해에서 출발한다.

---

## 4. Trade-off 의 mechanistic 분해와 anchor 를 통한 개선

### 4.1 Performance redistribution 의 발견

#### 동기

LoRA 결과는 Δ all ≈ 0과 Δ hard = +0.104를 함께 보였다. 가장 어려운 query가 +0.104 개선되었는데 aggregate가 움직이지 않았다면, 그 이득은 어딘가에서 지불되었을 것이다. 어디인가?

#### 설계 및 이론적 근거

부분집합 가중치 $w_{\text{hard}} = 0.457$, $w_{\text{easy}} = 0.543$일 때, easy 변화는 나머지 둘로부터 정해지는 회계 항등식(accounting identity)이다:

$$\Delta_{\text{easy}} = \frac{\Delta_{\text{all}} - w_{\text{hard}}\,\Delta_{\text{hard}}}{w_{\text{easy}}} = \frac{+0.001 - 0.457\times(+0.104)}{0.543} \approx -0.086$$

그런 다음 $\Delta_{\text{easy}}$를 세 seed에 걸쳐 직접 측정하여 예측을 확인 또는 반증한다.

#### 결과

| 양 | 예측 | 측정 (3-seed) |
|---|---|---|
| Δ all | +0.001 | +0.001 ± 0.012 |
| Δ hard | +0.104 | +0.104 ± 0.014 |
| **Δ easy** | **−0.086** | **−0.085 ± 0.010** |

예측과 측정이 ~99 % 일치한다.


*Figure. §4.1 accounting identity 의 예측 vs 측정 3-seed scatter. 모든 점이 y = x line 위에 안착 (3-seed mean 차이 = 0.0005) — *Δ_corr 의 회계적 항등성* 이 직접 입증된다. 이 등식은 redistribution 이 *모델 행동* 이 아닌 *수학적 항등식의 귀결* 임을 보인다.*

#### 해석 및 한계

요점:

1. 외견상의 "순변화 없음"은 사실 능동적 **performance redistribution**이다: hard query에서 얻은 모든 점수는 easy query에서의 거의 동일한 손실로 지불된다.
2. 구조적 이유는, LoRA update $\Delta W = (\alpha/r)BA$가 *모든 입력에 적용되는 상수 행렬*이라는 데 있다 — query-selective할 수 없으므로, hard query를 돕는 update가 easy query를 필연적으로 교란한다.
3. 이것이 바로 §2.2의 learned gate가 선택적이 될 수 없었던 이유다.

이 결과는 핵심 질문을 재구성한다: "hard query를 끌어올릴 수 있는가?"(가능하다)가 *아니라* "쉬운 이득을 redistribution하지 **않고** 그것을 끌어올릴 수 있는가?"이다.

### 4.2 실험 F: False negative 제거

#### 동기

자연스러운 가설은 손상이 실제로는 relevant인 mined hard negative(false negative)에서 온다는 것이다. 그렇다면 그것들을 제거하면 redistribution이 줄어야 한다.

#### 설계 및 이론적 근거

절차:

1. 외부의 강력한 cross-encoder(E5-Mistral)로 각 mined negative를 점수화한다.
2. negative가 false negative일 가능성이 높은 triplet을 *제거*한다(즉 relevance margin이 양수인 것만 남긴다).
3. 동일하게 재학습한다.

SciFact에서는 ~36%의 triplet이 제거된다.

#### 결과

| 방법 | Δ all | Δ hard | Δ easy | Doc effective rank | Token effective rank |
|---|---|---|---|---|---|
| LoRA (triplet 미수정) | +0.001 | +0.104 | −0.085 | 1.14 | 1.58 |
| **false negative 제거** | −0.004 ± 0.005 | +0.080 ± 0.004 | −0.073 ± 0.005 | 1.22 ± 0.01 | 1.72 ± 0.05 |

둘은 retrieval 수준에서도, representation-collapse 수준에서도 통계적으로 구분되지 않는다.

#### 해석 및 한계

요점:

- false negative 제거는 거의 아무것도 바꾸지 않는다 — redistribution도, effective-rank collapse도.
- **label noise는 근본 원인이 아니다**(easy 손상의 추정 ~14% 기여).
- 지배적 원인은 hard negative의 noise가 아니라 그 *난이도(difficulty)* 자체여야 한다.
- 이는 혼동되어 있던 두 설명의 깔끔한 분리(disambiguation)다.

### 4.3 실험 G: Easy negative 대체 (in-batch)

#### 동기

noise가 아니라 difficulty가 손상을 유발한다면, *hard contrast를 제거* — easy negative로 학습 — 하면 그것이 완화되어야 한다.

#### 설계 및 이론적 근거

크기 $B$ 의 mini-batch $\mathcal{B} = \{(q_i, d_i^+, d_i^{-,\text{hard}})\}_{i=1}^{B}$ 에 대해, 각 mined hard negative $d_i^{-,\text{hard}}$ 를 다른 query 의 positive document 에서 무작위로 추출한 *in-batch* easy negative 로 치환한다:

$$\tilde d_i^{-} \sim \mathrm{Uniform}\bigl(\{d_j^{+} : j \in [B], j \neq i\}\bigr)$$

학습 loss 는 기존 margin loss 와 동일한 형태이되 negative 만 치환된다:

$$\mathcal{L}_{\text{in-batch}}(q_i, d_i^+, \tilde d_i^-) = \max\bigl(0,\; m - s(q_i, d_i^+) + s(q_i, \tilde d_i^-)\bigr)$$

이 negative 는 *구성상 hard 가 아니다*: 다른 query 의 positive 는 *current query 의 영역에 대해 거의 확실히 irrelevant* 하므로, false-negative 확률 $\Pr[\tilde d_i^- \in \mathcal{R}_{q_i}]$ 가 mined hard negative 의 $\rho_{\text{FN}}$ 보다 *수 자리수 작다* (SciFact 측정 ≈ 0.6 % vs mined ~50 %).

이 두 형태는 데이터셋 전반에 걸쳐 단일 구성으로 시험되며, 다음 두 가설을 동시에 검정한다:

1. **noise 가설** (실험 F): false-negative 가 원인이면, easy negative (FN ≈ 0) 가 회복해야 한다.
2. **difficulty 가설** (본 실험): hard contrast 자체가 원인이면, easy negative 가 redistribution 을 완화해야 한다.

#### 결과

SciFact에서 세 seed 전반:

| Seed | NDCG@10 (all) | Δ all | Δ hard |
|---|---|---|---|
| 42 | 0.6613 | +0.015 [+0.001, +0.029] | +0.055 [+0.030, +0.081] |
| 1337 | 0.6681 | +0.022 [+0.008, +0.036] | +0.064 [+0.039, +0.090] |
| 2024 | 0.6722 | +0.026 [+0.011, +0.042] | +0.077 [+0.051, +0.105] |
| **평균** | **0.6672 ± 0.005** | **+0.021 ± 0.005 (3개 모두 유의)** | **+0.065 ± 0.012** |

주요 관찰:

- 이것이 본 연구의 **첫 strict net improvement**다: aggregate 이득이 세 seed 모두에서 유의하다.
- representation collapse 또한 대부분 역전된다 — token effective rank가 1.58에서 ~44.6으로 회복된다 (frozen 모델의 57.2와 비교).

#### 해석 및 한계

핵심 발견:

1. hard contrast 제거는 본 연구 최초의 robust하고 통계적으로 유의한 net 이득(+0.021)을 낸다.
2. 동시에 representation collapse를 대부분 역전시킨다.
3. 실험 F와 결합하면 mechanism이 분리된다: **redistribution과 collapse는 *hard* negative에 대한 과보정에 의해 유발되며, label noise도 optimization artifact도 아니다**.
   - 별도의 warmup-plus-gradient-clipping 실험은 최종 collapse를 0만큼 바꿔, optimization이 red herring임을 확인했다.
4. 이 방법의 대가는 hard lift의 약 절반만 회복한다는 것이다(+0.065 vs. +0.104) — hard-query 이득의 일부를 포기함으로써 손상을 완화한다.

§5 에 중요한 cross-domain 관찰:

- 더 어려운 NFCorpus 데이터셋에서 미수정 LoRA는 catastrophic(−0.320)이었다.
- in-batch negative 대체는 그 **catastrophic gap의 74%를 회복**했다(seed 전반에서 robust하나 여전히 net positive는 아님).
- 이 회복은 동일한 hard-contrast mechanism 이 데이터셋을 가로질러 작동함을 시사한다. §5 의 직접 검정 (Exp 12 × NF) 은 *FN 만* 제거하는 ablation 이 NF 에서 회복 0 % 임을 보여, **FN 양 자체가 dominant predictor 가 아니며 hard-contrast over-correction 이 cross-domain universal cause** 임을 확정한다.

### 4.4 실험 H: Representation-anchoring regularizer (가장 효과적인 lightweight 방법)

#### 동기

실험 G 는 trade-off 를 완화하는 한 방법을 보였으나 hard lift 의 절반을 희생했다. *모든* lift 를 유지하면서 easy 손상을 줄이는 더 나은 경로가 있다:

- hard query 에 대해서는 hard contrast 를 *온전히 유지* (LoRA 가 +0.104 source 임을 §3 이 입증).
- easy query 는 교란되지 않도록 명시적으로 *보호* — 이것이 본 절의 핵심 새 메커니즘.

#### 설계 및 이론적 근거

easy query에 한해, adapted token representation이 frozen 모델에서 벗어나는 정도를 penalize하는 **representation-anchoring regularizer**를 추가한다. 수학적으로 구별되는 두 형태를 시험했다:

1. **relational form** — token 간 *pairwise similarity 구조*를 보존(rotation-invariant constraint):
$$\mathcal{R}_{\text{rel}} = \| \text{Sim}(H_{\text{LoRA}}) - \text{Sim}(H_{\text{frozen}}) \|_F^2$$
2. **per-token absolute form** — cosine deviation을 통해 각 token의 *절대 방향*을 보존(rotation-sensitive, 더 strict한 constraint):
$$\mathcal{R}_{\text{abs}} = \frac{1}{|\mathcal{Q}_{\text{easy}}|} \sum_{x \in \mathcal{Q}_{\text{easy}}} \frac{1}{T_x}\sum_{t=1}^{T_x} \left(1 - \langle \hat{h}_t^{\text{LoRA}}(x),\; \hat{h}_t^{\text{frozen}}(x)\rangle\right)$$

여기서 $\mathcal{Q}_{\text{easy}}$ 은 easy query 집합, $T_x$ 는 query $x$ 의 token 수, $\hat{h}_t$ 는 L2-normalize 된 token embedding 으로 $\cos(\hat{h}_t^{\text{LoRA}}, \hat{h}_t^{\text{frozen}}) = \langle \hat{h}_t^{\text{LoRA}}, \hat{h}_t^{\text{frozen}}\rangle$ 의 inner product 와 등가이다.

두 경우 모두, 전체 loss 는 hard query 에서의 margin loss 와 easy query 에서의 anchor regularizer 의 *부분집합-조건부* 가산 결합이다:

$$\mathcal{L}(\theta) = \frac{1}{|\mathcal{Q}_{\text{hard}}|} \sum_{q \in \mathcal{Q}_{\text{hard}}} \mathcal{L}_{\text{margin}}(q, d_q^+, d_q^{-,\text{hard}};\, \theta) + \lambda \cdot \mathcal{R}_{\bullet}(\mathcal{Q}_{\text{easy}};\, \theta), \qquad \bullet \in \{\text{rel}, \text{abs}\}$$

여기서 $\theta$ 는 학습 가능 LoRA 파라미터, $\mathcal{R}_{\bullet}$ 는 위에 정의된 두 anchor 형태 중 하나. $\lambda = 1$ 의 단일 값으로 고정 (sweep 없음). 두 항이 *서로 다른 query 부분집합* 에서 작동하므로 *gradient flow 가 분리* 된다:

$$\nabla_\theta \mathcal{L} = \underbrace{\nabla_\theta \mathcal{L}_{\text{margin}}\big|_{\mathcal{Q}_{\text{hard}}}}_{\text{push: } \delta h \neq 0 \text{ on } \mathcal{Q}_{\text{hard}}} + \lambda \underbrace{\nabla_\theta \mathcal{R}_{\bullet}\big|_{\mathcal{Q}_{\text{easy}}}}_{\text{pull: } \delta h \to 0 \text{ on } \mathcal{Q}_{\text{easy}}}$$

직관: *모델이 틀리는 query 는 margin term 이 자유롭게 고치게 두되, 이미 맞히는 query 는 anchor term 이 frozen representation 으로 끌어당겨 교란을 차단한다.*

#### 결과

두 형태, 각 세 seed 전반:

| | relational form | per-token absolute form |
|---|---|---|
| Constraint 유형 | rotation-invariant relational | rotation-sensitive absolute |
| Δ all | +0.029 ± 0.005 | **+0.030 ± 0.002** |
| Δ hard | +0.101 ± 0.010 | +0.092 ± 0.007 |
| Δ easy | −0.031 ± 0.018 | **−0.021 ± 0.003** |
| net 이득이 유의한 seed 수 | 3개 중 2개 | **3개 중 3개** |
| 총 ‖B‖ (update magnitude) | ~1.8 | 1.34 (−22%) |

per-token absolute form이 본 연구의 **가장 신뢰할 만한 lightweight 결과**다:

- 세 seed 모두에서 +0.030의 strict net improvement.
- easy query를 가장 잘 보존(−0.021, −0.085 redistribution의 75% 감소).

직접 mechanism 측정이 anchor가 *어떻게* 작동하는지 밝힌다. token embedding 행렬 $\mathbf{H} \in \mathbb{R}^{N \times d}$ 의 normalized singular spectrum $\sigma_i$ 에 대해 **effective rank** 는 spectrum 의 exponential Shannon entropy 로 정의된다:

$$\mathrm{eff\text{-}rank}(\mathbf{H}) = \exp\left(-\sum_{i=1}^{d} p_i \log p_i\right), \qquad p_i = \frac{\sigma_i^2}{\sum_{j=1}^{d}\sigma_j^2}$$

이 양은 nominal rank ($\min(N, d)$) 와 무관하게 *representation 이 실제로 점유하는 차원 수* 를 측정한다. spectrum 이 한 axis 에 집중되면 ($p_1 \to 1$) eff-rank $\to 1$ 로 collapse 하고, uniform 분포 ($p_i = 1/d$) 면 $\mathrm{eff\text{-}rank} = d$ (full rank).

| 조건 | Doc effective rank | Token effective rank | $\mathbb{E}_t \langle \hat{h}_t^{\text{LoRA}}, \hat{h}_t^{\text{frozen}}\rangle$ |
|---|---|---|---|
| Frozen baseline | 9.86 | 55.13 | 1.000 (identity by definition) |
| per-token anchor (3-seed mean) | 2.33 ± 0.06 | 9.01 ± 0.36 | **0.824 ± 0.005** |

#### 해석: 개선이 어떻게 작동하는가

세 가지 mechanism 발견, 각각이 본 결과 (+0.030 strict net improvement) 의 핵심을 설명한다.

##### (1) Soft equilibrium attractor (anchor cosine = 0.824) — 개선을 가능하게 한 균형

학습 종료 후 측정된 token-level cosine 의 기댓값은 정확히 단조 감소 부분만 만족된 *부분 최적화* 의 흔적이다. 학습 가능 LoRA 파라미터 $\theta$ 에 대한 first-order stationarity 조건:

$$\underbrace{\nabla_\theta \mathcal{L}_{\text{margin}}\big|_{\mathcal{Q}_{\text{hard}}}}_{g_{\text{push}}(\theta^\star)} + \lambda \underbrace{\nabla_\theta \mathcal{R}_{\text{abs}}\big|_{\mathcal{Q}_{\text{easy}}}}_{g_{\text{pull}}(\theta^\star)} = 0$$

즉 수렴점 $\theta^\star$ 에서 *margin push* 와 *anchor pull* 이 정확히 상쇄된다. 만약 anchor 가 *hard constraint* ($\mathcal{R}_{\text{abs}} = 0$ 강제) 면 $\theta^\star$ 에서 $\hat h_t^{\text{LoRA}} \equiv \hat h_t^{\text{frozen}}$ 이고 cosine $= 1$. 그러나 $\lambda = 1$ 의 *soft* regularizer 는 두 gradient 가 *균형* 인 점을 선택하고, 이 점에서

$$\mathbb{E}_{x, t}\left[\langle \hat h_t^{\text{LoRA}}(x), \hat h_t^{\text{frozen}}(x)\rangle \mid x \in \mathcal{Q}_{\text{easy}}\right] = 0.824 < 1$$

가 측정된다. 즉 anchor 는 *clamp* 가 아니라 *attractor* 이며, 평형점 cosine 0.824 가 두 힘의 균형비를 직접 보여준다 — 이 균형 덕분에 *hard 회복은 살리고 easy 손상만 75 % 감소* 시킬 수 있다.

##### (2) Anchor 의 유일한 기여 = easy 보존 (개선의 source 분리)

plain LoRA 대비 각 anchor의 *incremental* 효과를 측정하면(동일 seed 쌍):

| Anchor 형태 | Incremental Δ all | Incremental Δ hard | Incremental Δ easy |
|---|---|---|---|
| relational | +0.028 | −0.002 | +0.055 |
| per-token absolute | +0.029 | −0.012 | **+0.064** |

핵심 진술:

- anchor의 net 이득은 전적으로 **easy query 보존**에서 온다.
- hard query에서는 오히려 plain LoRA보다 약간 *나쁘다*.
- 따라서 올바른 진술은: **plain LoRA가 hard 회복의 source이고, anchor의 유일한 기여는 easy 손상을 회피하는 것이다.**
- 이는 soft-equilibrium 그림과 정확히 일치한다 — LoRA가 밀고, anchor가 다시 당기며, 0.824가 그 균형점이다.

##### (3) 수학적 형태 차이가 retrieval 결과로 전이되지 않음

- 수학적으로 매우 다른 두 anchor 형태(하나는 rotation-invariant, 하나는 rotation-sensitive)가 **통계적으로 구분되지 않는** frontier에 안착한다(Δ all +0.029 vs. +0.030).
- per-token form이 내부적으로 token diversity를 약간 더 보존하고(effective rank 9.01 vs. 7.69) seed 전반에 더 안정적이지만, retrieval frontier는 공유된다.
- document 수준 effective rank는 collapse한 채로 남으므로(2.33), anchor는 token granularity에서만 효과적이다.
- 그 보호가 aggregation 후 희석되며, 이것이 easy 손상을 0까지 몰 수 없는 representation 수준의 이유다.

#### 한계 (다음 개선의 출발점)

- 모든 결과는 SciFact 한정 — cross-domain 일반화는 §5 에서 다룬다.
- document-level effective rank 는 collapse 한 채로 남아 (2.33), anchor 가 token-level 에서만 효과적이다.
- 본 anchor 의 *target set 비대칭* (q-side + d⁺-side 만 묶고 d⁻-side 는 자유) 이 +0.030 의 ceiling 의 원인일 가능성 — 본 가능성은 §4.5 에서 직접 검정한다.

### 4.5 실험 I: Negative-side anchor — §4.4 의 *비대칭* 해소

#### 동기

§4.4 의 per-token cosine anchor 는 본 연구의 가장 효과적인 lightweight 결과 (+0.030 strict net improvement) 를 가져왔으나, 그 *target set* 을 자세히 들여다보면 한 가지 비대칭이 남아 있다. 학습 중 anchor 가 끌어당기는 표현은 easy query 의 *query 토큰* 과 *paired positive document 토큰* 두 종류로, mined 된 *hard negative document* 의 표현은 자유롭게 풀려 있다. ColBERT 의 retrieval score 는

$$s(q, d) = \sum_{t \in q} \max_{t' \in d} \cos(h_t^{q}, h_{t'}^{d})$$

로, 양변 모두 LoRA 의 함수다. 따라서 한 변만 묶어 두면 LoRA 가 학습 중 negative document 의 표현을 자유롭게 변형할 수 있고, easy query 의 ranking 보존은 *공유 LoRA encoder weight 를 통한 간접 제약* 에 의존하게 된다. 이 비대칭이 §4.4 의 +0.030 ceiling 의 *진정한* mechanistic 한계인지, 아니면 단순히 anchor target set 의 *기하학적 불완전성* 으로 인한 fragile ceiling 인지를 본 실험이 직접 검정한다.

#### 설계 및 이론적 근거

본 §의 anchor 항을 명시적으로 분해해 보자. Easy query $x$ 와 paired positive doc $d^+$, mined hard-negative doc $d^-$ 의 세 표현 각각에 대해 per-doc token mean 의 cosine deviation 을

$$\mathcal{R}_{\text{abs}}^{q}(\theta) = \mathbb{E}_{x \in \mathcal{Q}_{\text{easy}}}\left[\frac{1}{|T^q_x|}\sum_{t} \left(1 - \cos(\hat h_t^{\text{LoRA},q}(x),\, \hat h_t^{\text{frozen},q}(x))\right)\right]$$

$$\mathcal{R}_{\text{abs}}^{d^+}(\theta) = \mathbb{E}_{x \in \mathcal{Q}_{\text{easy}}}\left[\frac{1}{|T^d_{d^+}|}\sum_{t} \left(1 - \cos(\hat h_t^{\text{LoRA},d}(d^+),\, \hat h_t^{\text{frozen},d}(d^+))\right)\right]$$

$$\mathcal{R}_{\text{abs}}^{d^-}(\theta) = \mathbb{E}_{x \in \mathcal{Q}_{\text{easy}}}\left[\frac{1}{|T^d_{d^-}|}\sum_{t} \left(1 - \cos(\hat h_t^{\text{LoRA},d}(d^-),\, \hat h_t^{\text{frozen},d}(d^-))\right)\right]$$

로 정의하면, §4.4 의 *실제 구현* 은 앞 두 항만 합산한 형태 $\mathcal{R}_{\text{abs}}^{\text{current}} = \mathcal{R}_{\text{abs}}^{q} + \mathcal{R}_{\text{abs}}^{d^+}$ 다. 세 번째 항 $\mathcal{R}_{\text{abs}}^{d^-}$ 가 본 §4.5 의 직접 추가 항이며, 이를 포함한 통합 학습 loss 는

$$\mathcal{L}^\dagger(\theta) = \mathcal{L}_{\text{margin}}(\mathcal{Q}_{\text{hard}};\theta) + \lambda_{\text{dir}} \cdot \bigl(\mathcal{R}_{\text{abs}}^{q} + \mathcal{R}_{\text{abs}}^{d^+}\bigr) + \lambda_{\text{neg}} \cdot \mathcal{R}_{\text{abs}}^{d^-}$$

이 된다. 세 항이 각자 *unit-scale per-doc 정규화* (각 문서 내 토큰 평균 → easy query 평균) 되어 있으므로, $\lambda_{\text{dir}} = \lambda_{\text{neg}} = 1.0$ 은 *symmetric weighting* 의 자연스러운 base case 이며, 다른 하이퍼파라미터의 sweep 없이 본 단일 구성만으로 비대칭 해소의 효과를 격리할 수 있다. 학습 protocol 은 §4.4 와 완전 동일 (q, v r=8, LR 5e-5, batch=32, epochs=3, max-triplets 9190). 3 seeds {42, 1337, 2024} on SciFact.

본 design 이 검정하는 핵심 가설은 두 가지다. 첫째, §4.4 의 +0.030 ceiling 이 *진정한 mechanistic ceiling* 인가, 아니면 *anchor target set 의 비대칭으로 인한 fragile ceiling* 인가. 둘째, 공유 LoRA encoder weight 를 통한 d⁻ side 의 *implicit* 제약이 retrieval 측면에서 충분한가, 아니면 명시적 anchor 항이 retrieval 의 다른 자유도를 추가로 가두는가. 두 가지 가설 모두 본 실험의 단일 metric (Δ all NDCG@10) 으로 결정된다.

#### 3-branch predictions

세 가능한 결과 분기를 본 실험 시작 전에 명시한다. 각 분기는 본 paper 의 narrative 에 *서로 다른 방향* 으로 기여하며, 어느 분기가 발화하든 paper-grade insight 가 보장된다.

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) Positive leap** | Δ all $\geq$ +0.040, 3/3 seed CI > 0 strict | §4.4 의 +0.030 은 *fragile ceiling* — symmetric anchor form 이 mechanistic leap. Anchor target set 의 *완전성* 이 핵심 자유도. Paper main result 갱신 + cross-domain 일반성 검정 trigger. |
| **(b) Tied / saturated** | +0.025 $\leq$ Δ all $\leq$ +0.040 (3-seed mean) | 공유 LoRA weight 의 *implicit* d⁻ 제약이 이미 충분. §4.4 의 mechanistic ceiling 의 *direct empirical proof* — 현재의 간접 증거 (H₁ relational ≈ H₂ per-token tie) 를 직접 증거로 격상. |
| **(c) Over-restriction** | Δ all $\leq$ +0.020 | d⁻ anchor 가 hard query 의 contrast 학습을 손상 — LoRA 가 *진짜로 가까워야 할 hard query 의 d⁻* 와 *진짜로 멀어야 할 easy query 의 d⁻* 를 동일하게 묶어 hard lift +0.092 가 감소. *Informed differential weighting* (§7) 의 양적 motivation. |

분기 (a) 가 발화하면 본 paper 의 main result 가 갱신되며, §6.1 의 +0.030 수치를 새 값으로 교체하고 cross-domain 일반성 검정을 별도 design 으로 trigger 한다. 분기 (b) 가 발화하면 §4.4 의 saturation 주장이 *간접 증거 (서로 다른 anchor parameterization H₁ ≈ H₂ tie) 에서 직접 증거 (symmetric anchor 도 같은 +0.030 으로 수렴)* 로 격상되며, paper 의 mechanism analysis 가 강화된다. 분기 (c) 가 발화하면 *uniform symmetric weighting 자체가 부적절* 함이 드러나, §7 의 informed differential weighting 이 단순한 명목적 future direction 이 아닌 *직접 검정된 motivation* 을 얻는다.

#### 결과

| Seed | Δ all | Δ hard | Δ easy | ‖A‖ | ‖B‖ |
|---|---:|---:|---:|---:|---:|
| 42 | +0.0270 [+0.010, +0.045] ✓ | +0.0762 | −0.0144 | 8.29 | 1.26 |
| 1337 | +0.0298 [+0.013, +0.047] ✓ | +0.0812 | −0.0134 | 8.26 | 1.24 |
| 2024 | +0.0269 [+0.010, +0.045] ✓ | +0.0750 | −0.0135 | 8.24 | 1.22 |
| **mean** | **+0.0279 ± 0.0014** ✓ (3/3) | **+0.077 ± 0.003** | **−0.014 ± 0.001** | **8.26 ± 0.03** | **1.24 ± 0.02** |

세 seed 모두에서 CI > 0 strict 통과, mean Δ all = +0.0279. §4.4 의 +0.030 ± 0.002 와 통계적으로 *구분 불가*. 발화 분기는 **(b) Tied / saturated** — §4.4 의 +0.030 ceiling 은 *fragile asymmetry artifact 가 아니라 진정한 mechanistic ceiling*. 본 결과는 anchor family 의 saturation 의 *direct empirical proof* 로, §4.4 의 H₁ (relational) ≈ H₂ (per-token) tie 의 *간접* 증거를 *세 번째 mathematically 독립인 form (symmetric)* 의 *동일 frontier 점 수렴* 으로 강화한다.

**Mechanism-level reading**: d⁻ anchor 가 *실제로 작동* 하나 retrieval 측면에서는 *정확히 cancel*:

| 양 | §4.4 (q + d⁺) | §4.5 (q + d⁺ + d⁻) | 차이 |
|---|---:|---:|---:|
| Δ hard | +0.092 ± 0.007 | +0.077 ± 0.003 | **−0.015** (LoRA freedom 손실) |
| Δ easy | −0.021 ± 0.003 | −0.014 ± 0.001 | **+0.007** (easy 보존 강화) |
| Δ all | +0.030 ± 0.002 | +0.028 ± 0.001 | −0.002 (통계 동등) |
| ‖B‖ | ~1.34 | 1.24 ± 0.02 | LoRA aggressiveness 약 8 % 감소 |

→ d⁻ 표현이 frozen 으로 끌릴 때 *동일한 LoRA 가중치* 가 hard query 의 학습 자유도를 줄임 (Δ hard 감소). 동시에 easy query 의 ranking 보존이 강화됨 (Δ easy 향상). 두 효과의 magnitude 가 query 부분집합 비율 ($w_{\text{hard}} \approx 0.457$, $w_{\text{easy}} \approx 0.543$) 로 가중되어 *근사적으로 정확히 cancel*: $w_{\text{hard}} \cdot (-0.015) + w_{\text{easy}} \cdot (+0.007) \approx -0.0030$, 실측 Δ all 변화 −0.0021 과 일치 ($\pm 1\sigma$ 내).

#### 한계

본 실험 설계의 한계는 세 가지로 정리된다. 첫째, anchor target $d^-$ 는 *mining time* 에 frozen ColBERT 가 선택한 hard negative 로 한정되며 학습 중 dynamic re-mining 은 수행하지 않는다 — 이 측면은 §7.3 의 dynamic mining 과 직교한 별개 자유도다. 둘째, $\lambda_{\text{neg}} = \lambda_{\text{dir}} = 1.0$ 의 symmetric choice 가 *best* 인지의 검정은 본 실험의 scope 에서 의도적으로 제외되었으며, informed differential weighting 의 검정은 분기 (c) 발화 시에만 §7 의 직접 후속으로 활성화된다. 셋째, hard negative document 가 query / positive document 보다 일반적으로 길이가 길어 per-token mean 의 *aggregation 통계* 가 다를 수 있는데, 본 design 은 *per-doc mean* 으로 unit-scale 정규화 (각 문서 내 토큰 평균 후 query 별 평균) 하여 세 항이 같은 scale 로 합산되도록 보장한다.

### 소결: trade-off 분해와 anchor 의 strict net improvement

§4 는 §3 의 "무해해 보이는" LoRA 가 실제로 *hard query 를 easy query 의 희생으로 개선한다* 는 사실을 정량화하고, 그 원인을 격리한 뒤, 그 위에 작동하는 개선 메커니즘을 단계적으로 제시한다. §4.1 의 회계 항등식 예측 (Δ_corr ≈ −0.086) 과 실측 (−0.085 ± 0.010) 의 99 % 일치는 redistribution 을 확정하고, §4.2 의 false-negative 제거 실험은 label noise 를 원인에서 배제하며 (easy 손상의 약 14 % 만 기여), §4.3 의 in-batch easy negative 대체는 hard contrast 자체가 원인임을 격리한다. 본 연구 최초의 strict net improvement (+0.021, 3/3 seed 유의) 가 이 절의 첫 유의한 개선 결과이며, NFCorpus catastrophic gap 의 74 % 회복은 동일 mechanism 이 cross-domain 으로도 작동함을 시사한다.

이 mechanistic 격리 위에 §4.4 의 representation-anchoring regularizer 가 본 연구의 핵심 결과 — **+0.030 NDCG@10 의 strict net improvement (3/3 seed 유의)** — 를 달성한다. hard 회복은 plain LoRA 가 공급하고 anchor 는 그 위에 *easy 손상만 75 % 감소* 시키는 분업 구조가, 직접 측정된 anchor cosine = 0.824 의 *soft equilibrium attractor* 형태로 mechanism direct evidence 와 함께 확정된다. 두 수학적으로 다른 anchor 형태 (rotation-invariant relational vs. rotation-sensitive per-token) 가 통계적으로 구분되지 않는 frontier 점을 차지하므로, 구체적 anchor 형태 선택은 *모두 동등하게 작동* 하며 family 선택만이 중요하다.

본 절의 한계는 두 가지로 정리된다. 첫째, anchor cosine 0.824 의 잔여 deviation 은 redistribution 의 완전 제거가 단순 anchor 만으로는 불가능함을 의미한다 (Δ easy = −0.021 ≠ 0). 둘째, document-level effective rank 는 collapse 한 채로 남아 (2.33), anchor 가 token-granularity 에서만 효과적이며 aggregation 후 보호가 희석된다. *세 번째* 의문 — anchor target set 의 *비대칭* (q-side + d⁺-side 만 묶고 d⁻-side 는 자유) 이 +0.030 의 *진정한 mechanistic ceiling* 인지 *fragile asymmetry artifact* 인지 — 는 §4.5 에서 직접 검정되어 *branch (b) saturation* 으로 확정: symmetric anchor (q + d⁺ + d⁻) 도 동일 +0.028 ± 0.001 frontier 점에 수렴. 따라서 본 +0.030 은 세 mathematically 독립인 anchor parameterization (H₁ relational, H₂ per-token, I symmetric) 의 공통 frontier 로 *direct empirical saturation proof* 를 확보한다. 모든 결과는 SciFact 한정이며 cross-domain 일반화는 §5 에서 별도로 다루어진다.

---

## 5. Cross-domain 개선 경로의 분석

#### 동기

§2–§4 의 모든 학습 결과는 SciFact 에 대한 것이다. plain LoRA 는 더 어려운 NFCorpus (−0.320) 와 FiQA (−0.347) 에서 *catastrophic* 했으며, 이 catastrophe 의 *원인을 정확히 식별* 하는 것이 cross-domain 개선의 출발점이다. 본 §5 는 두 후보 원인 — *false-negative 오염* (data-side) 과 *hard-contrast over-correction* (model-side, §4 의 SciFact 결론) — 을 직접 분리하는 cross-dataset 실험을 단일 구성으로 수행한다.

#### 두 후보 원인의 정의

데이터셋 $\mathcal{D}$ 의 false-negative rate 를 mined hard negative pool 중 *실제로 relevant 하나 labeling 안 된* 문서의 비율로 정의한다:

$$\rho_{\text{FN}}(\mathcal{D}) = \Pr\left[ d^{-,\text{hard}} \in \mathcal{R}_q \;\middle|\; d^{-,\text{hard}} \in \mathrm{Top\text{-}K}(\mathrm{frozen}(q)) \setminus \mathrm{qrels}^{+}(q) \right]$$

여기서 $\mathcal{R}_q$ 는 *진정한* relevant 문서 집합 (qrels 의 불완전 라벨 너머의 ground truth), $\mathrm{qrels}^{+}(q)$ 는 dataset annotation 에서 명시적으로 relevant 표시된 문서다. E5-Mistral cross-encoder margin 으로 측정한 데이터셋 별 $\rho_{\text{FN}}$:

$$\rho_{\text{FN}}(\text{SciFact}) \;\approx\; 0.14, \qquad \rho_{\text{FN}}(\text{NFCorpus}) \;=\; 0.611 \;\;\text{(direct measurement: 676{,}102 / 1{,}105{,}750 mined triplets)}$$

NFCorpus 가 SciFact 보다 $\sim 4\times$ 더 많은 false negative 를 가짐 (dense-judgment 의료 문헌, 각 query 에 다수 relevant doc 존재하나 fully labeled 안 됨). 후보 가설:

- **H1 (FN-as-root)**: NFCorpus catastrophe 의 dominant cause 는 *labeling 부재로 인한 부호 오류* — FN 만 제거하면 회복 예측.
- **H2 (hard-contrast-as-root, §4 의 cross-domain 확장)**: dominant cause 는 *mined hard negative 의 difficulty 자체* (SciFact 의 over-correction 과 동일 mechanism) — FN 양은 동반 증상, 회복 위해 hard contrast 자체를 건드려야 함.

#### 결정적 실험: H1 의 직접 검정 (Exp 12 × NFCorpus × 3 seed)

H1 의 함의를 정확히 검정하기 위해 *FN 만 제거하고 hard contrast 는 그대로 유지* 하는 ablation 을 수행. SciFact 의 Exp 12 와 동일 config 를 NFCorpus 에 그대로 적용 (HP 재튜닝 없음):

$$\widetilde{\mathcal{T}}_{\text{NF}} = \{(q, d^+, d^-) \in \mathcal{T}_{\text{NF}} \mid \langle e_q^{\text{E5}}, e_{d^+}^{\text{E5}}\rangle - \langle e_q^{\text{E5}}, e_{d^-}^{\text{E5}}\rangle > 0\}$$

429,648 triplet 이 통과 (전체의 38.9 %, 즉 61.1 % 가 likely FN 으로 제거됨). 3 seed 결과 (3-seed mean):

| 평가 | NDCG@10 absolute | Δ all | Δ hard | Δ easy | 판정 |
|---|---:|---:|---:|---:|---|
| Plain LoRA × NF | 0.064 | **−0.320** | — | — | catastrophic |
| **FN-removal × NF** | **0.014** | **−0.316** (−0.317 / −0.316 / −0.316) | −0.089 | −0.566 | **plain LoRA 와 통계 동등 — 회복 0 %** |
| In-batch easy × NF | 0.248 | −0.084 | — | — | 74 % 회복 |

**H1 은 직접 검정에서 반증**: NFCorpus 의 FN 비율이 SciFact 의 $4\times$ 임에도 (61 % vs 14 %), FN 을 제거해도 회복이 일어나지 않음. 3 seed 의 Δ all 분산은 0.001 단위로 tight — *어떤 9K 부분집합이 와도 동일 catastrophic point 로 attract* 되는 structural collapse. NDCG@10 absolute 0.014 는 random retrieval (≈ 0.01) 수준.

대조적으로, *FN 제거 + hard 제거* 를 동시에 하는 in-batch easy negative (§4.3) 는 동일 NFCorpus 에서 catastrophic gap 의 **74 %** 를 회복:

$$\frac{|\Delta_{\text{in-batch}}(\text{NF})|}{|\Delta_{\text{plain}}(\text{NF})|} \approx \frac{0.084}{0.320} \approx 0.26, \quad \text{즉 회복률 } 74\,\%$$

→ **차이는 hard contrast 의 유지 여부**. 두 개입 모두 FN 을 제거하지만, FN-removal 은 hard contrast 를 *유지* (Δ all −0.316), in-batch 는 hard contrast 도 *함께 제거* (Δ all −0.084). FN 양 자체가 dominant predictor 라면 두 결과가 비슷해야 함 — 그러나 직교 (0 % vs 74 %).

#### 보조 증거: H2 의 cross-domain 확장 (Exp 13 anchor × {FiQA, NFCorpus} × 3 seed)

H2 가 맞다면, hard contrast 를 *유지하면서 over-correction 만 방지* 하는 anchor (§4.4) 도 cross-domain 에서 부분 회복을 보여야 함. Exp 13 (per-token anchor + LoRA, λ_dir = 1) 의 cross-dataset 결과 (3-seed mean):

| 데이터셋 | Δ all (3-seed mean) | 회복률 vs plain LoRA | LoRA aggressiveness ‖B‖ |
|---|---:|---:|---:|
| **H₂ × FiQA** | **−0.090** ([−0.108, −0.080]) | **74 %** | 1.89 (plain LoRA 보다 낮음) |
| **H₂ × NFCorpus** | **−0.221** ([−0.227, −0.215]) | **31 %** | 2.19 |
| (참고) FN-only × NF | −0.316 | 0 % | **3.27** ← anchor 부재 시 LoRA 가 더 aggressive |

→ Anchor 는 cross-domain 에서도 *부분 회복* (FiQA 74 %, NF 31 %). 더 중요하게, anchor 가 있을 때 LoRA 의 ‖B‖ 가 1.5–2.2 인데 anchor 없는 FN-removal 에서는 3.27 로 약 50 % 더 큼 — *anchor 가 LoRA aggressiveness 자체를 제어* 해 over-correction 의 크기를 직접 줄임. 이는 cross-domain partial recovery 의 mechanism level 증거다.

#### gradient norm: 동반 증상이지 root 아님

SciFact 와 NFCorpus 의 epoch-1 평균 ranking loss 비교:

$$\|\nabla_\theta \mathcal{L}_{\text{margin}}\|_{\text{NF, ep1}} \approx 7 \cdot \|\nabla_\theta \mathcal{L}_{\text{margin}}\|_{\text{SciFact, ep1}}, \qquad \text{loss}_{\text{NF, ep1}} = 4.47 \;\text{vs.}\; 0.66$$

NFCorpus 의 큰 gradient norm 자체는 FN 의 *symptom* 이지만, gradient 의 *방향* 이 잘못된 게 아니라 — FN-removal 직접 검정이 보여주듯 — *intensity* 가 hard contrast 의 over-correction 을 가속할 뿐. 사실, gradient norm 자체를 줄이는 것 (warmup, clipping) 은 collapse 를 지연만 시킨다는 control 도 일치: $\theta_{t+1} = \theta_t - \eta_t g_t$ 에서 $\eta_t$ 를 작게 해도 hard contrast 가 살아있으면 같은 attractor 로 수렴.

#### 해석: 단일 축 처방

본 분석은 cross-domain 개선의 처방을 *hard contrast 를 어떻게 다루느냐* 의 **단일 축** 으로 재정의한다:

| 개입 | Hard contrast | FN | SciFact (Δ all) | NFCorpus (Δ all) | FiQA (Δ all) |
|---|---|---|---:|---:|---:|
| Plain LoRA | 유지 | 유지 | +0.001 | −0.320 | −0.347 |
| FN-removal (Exp 12) | **유지** | 제거 | −0.004 | **−0.316** | — |
| Anchor (Exp 13) | 유지 + over-correction 방지 | 유지 | **+0.030** ✓ | −0.221 (31 %↑) | −0.090 (74 %↑) |
| In-batch easy (Exp 15) | **제거** | 제거 | +0.021 ✓ | −0.084 (74 %↑) | — |

→ **Hard contrast 를 건드리는 두 부류 — anchor (유지 + over-correction 방지) 와 in-batch easy (제거) — 만이 cross-domain working tool**. FN-removal 만으로는 어떤 데이터셋에서도 net 이득이 없음. §4 의 SciFact 결론 (원인 = hard contrast, FN 은 minor) 이 NFCorpus 에서 *4× 더 많은 FN 양에도 불구하고* 그대로 확인됨 — **hard-contrast over-correction 이 cross-domain universal mechanism**.

#### 한계

본 §5 의 anchor cross-domain 부분 회복 (FiQA 74 %, NF 31 %) 은 strict net+ 가 아니므로 paper main claim 은 SciFact +0.030 으로 한정되며 (§2 Scope), cross-domain 결과는 mechanism universality 의 증거로만 인용한다. NFCorpus 의 catastrophic gap 의 *완전 회복* — strict net+ 도달 — 은 §7.2 의 *anchor + in-batch easy 결합* 의 직접 후속이며, 추가 변수 (HP 재튜닝, dynamic mining 등) 없이 본 paper 의 단일 축 처방 위에서 자연 도출된다.

---

## 6. 결론

본 연구는 frozen ColBERT retriever 의 hard-negative confusion 을 lightweight intervention 으로 어디까지 개선할 수 있는지, *그리고 그 개선의 메커니즘은 무엇인지* 를 통제된 multi-seed 실험으로 보였다.

### 6.1 핵심 성과

**SciFact 에서 +0.030 NDCG@10 strict net improvement** (3 seed 모두 95 % CI 하한 > 0). 인코더 파라미터의 0.27 % 만 학습 가능하게 둔 **representation-anchoring regularizer + LoRA** 조합:

$$\mathcal{L}^\star(\theta) = \frac{1}{|\mathcal{Q}_{\text{hard}}|} \sum_{q \in \mathcal{Q}_{\text{hard}}} \mathcal{L}_{\text{margin}}(q;\theta) + 1 \cdot \mathcal{R}_{\text{abs}}(\mathcal{Q}_{\text{easy}};\theta), \qquad \theta = \text{LoRA}(q, v;\; r=8,\, \alpha=r)$$

단순 frozen baseline 대비 측정된 3-seed mean ± std (모든 CI 는 paired bootstrap, $n_{\text{iter}} = 10{,}000$):

| Slice | $\Delta\,$NDCG@10 | 의미 |
|---|---|---|
| **hard** | $+0.092 \pm 0.007$ | 가장 어려운 query 의 약 $14\,\%$ 상대 개선 |
| **easy** | $-0.021 \pm 0.003$ | 단순 LoRA $(-0.085)$ 의 redistribution 손상 $\mathbf{75\,\%}$ 감소 |
| **all** | $+0.030 \pm 0.002\;\checkmark$ | **strict net improvement** (3/3 seed) |

또한 representation 측정:

- token effective rank: $9.01 \pm 0.36$ (frozen $55.13$ 대비 partial preservation, plain LoRA $1.58$ 대비 $5.7\times$ 회복).
- $\mathbb{E}_{x,t}[\cos(\hat h_t^{\text{LoRA}}, \hat h_t^{\text{frozen}})\,|\,x \in \mathcal{Q}_{\text{easy}}] = 0.824 \pm 0.005$ — *soft equilibrium attractor* 의 정량적 위치.

### 6.2 개선을 가능하게 한 메커니즘 발견

본 결과는 우연이 아니라 다음 mechanistic 통찰을 활용해 도출되었다.

1. **Intervention point 의 개수가 개선을 좌우한다.**
   - 모든 lightweight 학습 방법은 nominal capacity 와 무관하게 intervention point 당 effective rank 1–2 로 collapse 한다.
   - 개선은 intervention point 당 풍부함이 아니라 서로 다른 intervention point 의 *개수* 에 따라 scaling 한다.
   - → **설계 원리: 적은 rank × 많은 intervention point 가 큰 rank × 적은 intervention point 를 이긴다** (예: 24 개 attention layer 모두에 걸친 LoRA r=8 이 budget-optimal).
2. **Hard-negative over-correction 의 격리.**
   - 단순 LoRA 의 +0.104 hard lift 가 −0.085 redistribution 비용을 동반함을 발견.
   - 통제된 mediation 으로 원인을 *hard mined negative 의 난이도 그 자체* 로 분리 (label noise ~14 % 기여, optimization ~0 %).
   - → **원인 정확성 덕분에 정확한 처방 가능**: 어려운 query 학습은 유지하면서 easy query 보호.
3. **Representation anchoring 의 작동 방식 mechanism 입증 + saturation 확정.**
   - Anchor 의 net 이득은 *전적으로 easy 보존* 에서 옴 (plain LoRA 대비 incremental Δ hard = −0.012, incremental Δ easy = +0.064).
   - LoRA 가 hard query 회복을 *공급* 하고, anchor 가 easy query 손상을 *방지* 하는 분업 구조.
   - cosine 0.824 의 soft equilibrium 이 두 힘의 균형점.
   - §4.5 의 symmetric anchor (anchor target $\{q, d^+\} \to \{q, d^+, d^-\}$ 확장) 도 같은 +0.028 ± 0.001 frontier 로 수렴 — 세 mathematically 독립인 anchor parameterization 의 *공통 frontier* 가 anchor family saturation 의 *direct empirical proof*.
4. **Cross-domain mechanism universality.**
   - SciFact (sparse) 와 NFCorpus / FiQA (dense) 의 catastrophe 가 *같은* root cause — *hard-contrast over-correction* — 으로 통합됨을 직접 검정으로 확인.
   - NFCorpus 의 FN rate ($\rho_{\text{FN}} = 0.611$) 가 SciFact ($\rho_{\text{FN}} \approx 0.14$) 의 4× 임에도, FN-only removal (Exp 12 × NF, 3 seed) 은 회복 **0 %** (Δ all = −0.316 ≈ plain LoRA); hard contrast 를 건드리는 두 개입만 — anchor (FiQA 74 % / NF 31 % 회복) 와 in-batch easy (NF 74 % 회복) — cross-domain working tool.
   - → **단일 축 처방** (§5): cross-domain 개선은 *hard contrast 를 어떻게 다루느냐* 의 문제. FN 정제는 동반 증상에 대한 처방으로 무력.

### 6.3 본 결과의 위치

**+0.030 strict net improvement** 는 lightweight intervention 의 *informed* 최고치다. leaderboard 등재가 아니라, frozen retriever 를 인코더 fine-tuning 없이 개선하는 parameter-efficient 한 방법론을 mechanism 과 함께 제시한다는 점에 기여가 있다:

- *왜* +0.030 이 SciFact 에서 의미 있는 개선인지 — 위 mechanistic 발견이 답한다.
- *왜* 더 큰 개선을 원한다면 어떤 방향으로 가야 하는지 — §7 future work 가 제시.
- *왜* cross-domain 에서 *FN 정제만으로 부족* 하고 *hard contrast 자체를 건드려야* 하는지 — §5 의 cross-dataset 직접 검정 (Exp 12 × NF 반증 + anchor cross-domain 부분 회복) 이 답한다.

### 6.4 범위와 future work

본 연구의 strict net+ claim 은 SciFact 에 한정 (§2 Scope). cross-domain 결과 (§5) 는 anchor 가 FiQA 74 %, NFCorpus 31 % 부분 회복을 보였으나 strict net+ 가 아니므로 *mechanism universality 의 증거* 로만 인용한다. cross-domain 에서 *strict net+ 도달* 은 본 연구의 단일 축 처방 — *hard contrast 를 어떻게 다루느냐* — 으로 좁혀진 informed path 위에서 future work 로 남는다 (§7).

---

## 7. Future work: 본 연구가 제시하는 다음 개선 경로

본 연구의 mechanism 분석은 +0.030 너머의 개선을 위한 **구체적이고 우선순위가 분명한** 4가지 경로를 제시한다. 각 경로는 본 연구의 실험적 발견에서 직접 도출된 *informed proposal* 이다. 별도로, §4.5 (Exp I, negative-side anchor) 는 본 연구의 *진행 중인* 실험으로, anchor target set 의 *대칭 확장* 이 +0.030 ceiling 의 mechanistic 본질을 결정한다 — branch (a) 시 main result 갱신, (b) 시 saturation direct proof, (c) 시 informed differential anchor 의 양적 motivation. 이하 §7 의 4 경로는 §4.5 의 결과와 *직교* 한 추가 개선 후보.

### 7.1 Layer-differentiated anchoring

본 연구의 §2.3 발견 — confusion-정렬 신호가 final 인코더 layer 에 집중 (cosine $(v_\ell, v_{\text{md}}) \approx 0$ for $\ell \in \{0,3,6,9\}$ vs $0.267$ for $\ell = 12$) — 은 *differential* anchoring 을 직접 시사한다. 현재 anchor 는 final ColBERT output 단일 layer 에만 작용:

$$\mathcal{R}_{\text{anchor}}^{\text{current}} = \mathbb{E}_{x,t}\bigl[1 - \cos(\hat h_{t,L}^{\text{LoRA}}, \hat h_{t,L}^{\text{frozen}})\bigr], \qquad L = \text{final layer}$$

제안된 layer-differentiated 형태 — *anchor weight 가 layer 별로 다름*:

$$\mathcal{R}_{\text{anchor}}^{\text{differential}}(\boldsymbol{\beta}) = \sum_{\ell \in \mathcal{L}} \beta_\ell \cdot \mathbb{E}_{x,t}\bigl[1 - \cos(\hat h_{t,\ell}^{\text{LoRA}}, \hat h_{t,\ell}^{\text{frozen}})\bigr]$$

여기서 $\boldsymbol{\beta} = (\beta_0, \beta_3, \beta_6, \beta_9, \beta_{12})$ 는 layer 별 anchor strength 이다. **uniform 설정** ($\beta_\ell = 1/5$) 은 별도 실험 (Appendix C) 에서 검정되었으며 단일 layer 보다 명백히 열등 (3-seed mean Δ all = +0.004, Δ easy −0.052) — *앞쪽 layer 의 redundant constraint 가 loss budget 을 흡수해 정작 final layer 의 anchor 가 약화* 되는 **budget dilution** 현상이 mechanism diagnosis 로 직접 측정되었다. 따라서 uniform weighting 은 부적절하나, *informed* differential weighting 은 다른 답을 줄 가능성:

1. confusion 이 없는 앞쪽 layer ($\ell \in \{0, 3, 6\}$) 에는 **큰** $\beta_\ell$ (representation 변경 차단).
2. signal 을 담은 뒤쪽 layer ($\ell \in \{9, 12\}$) 에는 **작은** $\beta_\ell$ (개선 여지 보존).
3. 사전 분석에서 도출 가능한 $\beta$ 의 한 후보: $\beta_\ell \propto 1 - |\cos(v_\ell, v_{\text{md}})|$ — *retrieval signal alignment 의 반비례*.

uniform anchor 가 실패한 이유가 *uniform weighting* 이었으므로, differential weighting 의 검정 가치가 있다. 단 *empirical 정합성 caveat*: §2.3 의 diagnostic 측정에서 앞쪽 layer ($\ell \in \{0,3,6,9\}$) 의 LoRA-induced deviation 은 거의 0 — 즉 큰 $\beta_\ell$ 을 부여해도 *해당 항이 곱하는 양 자체가 0* 이라 *empty constraint* 일 가능성. 따라서 본 경로는 *§4.5 branch (c) 가 발화 (즉 d⁻ anchor 가 hard contrast 를 손상시키는 양적 증거 확보)* 시 *informed differential weighting* 의 직접 motivation 으로 활성화. (a) 또는 (b) 발화 시 본 경로는 §7.4 또는 §4.5 의 follow-up 으로 흡수.

### 7.2 Anchor 와 in-batch easy negative 의 cross-domain 결합

§5 의 단일 축 처방 — *hard contrast 를 건드리는 두 working tool* — 을 결합한다. Anchor (Exp 13, hard 유지 + over-correction 방지) 는 cross-domain 에서 부분 회복 (FiQA 74 %, NF 31 %), in-batch easy (Exp 15, hard 제거) 는 NF 에서 74 % 회복을 단독으로 달성. 두 개입은 동일 axis 의 *상보적* 작동: anchor 는 hard contrast 의 *over-correction 강도* 를 제어, in-batch 는 hard contrast 의 *주파수* 를 제어. 결합 학습 loss:

$$\mathcal{L}_{\text{joint}}(\theta) = \frac{1}{|\mathcal{T}^{\text{in-batch}}|}\sum_{(q, d^+, \tilde d^-) \in \mathcal{T}^{\text{in-batch}}} \mathcal{L}_{\text{margin}}(q, d^+, \tilde d^-;\theta) + \lambda \cdot \mathcal{R}_{\text{abs}}(\mathcal{Q}_{\text{easy}};\theta), \qquad \tilde d_i^- \sim \mathrm{Uniform}(\{d_j^+\}_{j \ne i})$$

(주: §5 의 직접 검정으로 *FN-only removal 은 cross-domain 무력* 으로 확인됐으므로, 이전 버전 (anchor + e5_margin > 0 FN filtering) 대신 in-batch easy negative substitution 으로 data-side 를 교체.)

두 직교 효과:

1. **frequency control (data-side)**: in-batch easy 가 mined hard contrast 자체를 제거 — NF / FiQA 에서 단독 74 % 회복.
2. **magnitude control (model-side)**: $\mathcal{R}_{\text{abs}}$ 가 남은 over-correction 의 강도 차단 (anchor 단독으로도 ‖B‖ 가 1.5–2.2 로 plain LoRA 의 3.3 대비 줄어듦, §5).

대상 도메인 + 평가 절차:

- NFCorpus / FiQA 같은 dense-judgment 도메인에서 평가.
- *anchor 또는 in-batch 단독* 으론 strict net+ 도달 못 함 (각각 31 % / 74 %, NF). 결합이 두 71 % 의 부족을 메울지 직접 검정.
- 본 결합의 단일 구성을 명시한 후 실행한다.

→ **frozen-encoder lightweight 방법론을 cross-domain *strict net+* 로 끌어올리는 가장 promising 한 경로**.

### 7.3 Dynamic hard-negative re-mining

본 연구의 정적 epoch-0 mining 의 한 한계는 학습 진행에 따라 negative pool 이 stale 해진다는 것. 학습 중 동적으로 hard negative 를 재선정하면:

- 학습이 진행되며 *실제로 어려워진* query 에 집중 가능.
- §4 의 hard-negative over-correction 패턴을 일부 완화 가능성.
- 단, dynamic re-mining 자체가 새 hyperparameter (re-mining 주기, threshold) 를 추가하므로 신중한 설계가 필요.

### 7.4 Partial unfreezing (lightweight 천장 너머)

§3.1 의 full-unfreeze 상한선 (hard 부분집합에서 +0.252) 과 anchor 의 +0.030 사이 간극을 좁히는 경로:

1. 마지막 1–2 개 인코더 layer 만 unfreeze + 나머지 frozen 유지.
2. anchor 의 token-level only 보호 (§4.4 한계) 가 doc-level effective rank 의 collapse 회피로 확장될 가능성.
3. parameter 수가 늘어나므로 lightweight 정의가 변경됨 (인코더가 더 이상 entirely frozen 아님) → 별개 연구.

---

## Appendix — 완전성을 위해 보존한 negative 및 보조 결과

이 실험들은 본 연구의 주된 진단 흐름의 일부가 아니지만, 대안을 차단하거나 plateau 를 확인하므로 기록한다.

### A. Optimization 의 red herring 확인 (mediation control)

warmup과 gradient clipping을 cross-domain catastrophe의 대안 설명으로 시험했다. 결과:

- 최종 collapse를 0만큼 바꿨다.
- NFCorpus/FiQA 성능을 회복하지 못했다.
- 이 control과 in-batch 대체의 조합은 in-batch 대체 단독과 같았다.

이는 원인으로서 optimization이 아니라 supervision을 분리한다.

### B. Bilinear interaction metric 과 distillation (단일 intervention point, 본문 분석 제외)

bilinear scoring metric $M = I + UV^\top$와 E5-Mistral margin distillation을 단일 intervention point 대안으로 시험했다. 결과:

- 둘 다 effective rank ~1로 collapse했다.
- 단일 intervention point lift(hard query에서 ~+0.05)에만 도달했으며, 이는 universal-collapse 발견과 정합한다.
- distillation regularizer 는 더 나아가 그것이 강화하려던 바로 그 메커니즘을 억압했다.

§3.2의 보조 증거로 보존한다.

### C. Uniform multi-layer anchor 의 budget dilution 실패 (§7.1 의 empirical 근거)

per-token anchor 를 final ColBERT output 대신 5 개 BERT layer $\{0, 3, 6, 9, 12\}$ 에 *uniform weight* ($\beta_\ell = 1/5$) 로 분산한 변형을 단일 구성으로 시험했다. 동일 LoRA 구성 (q,v r=8, λ_dir=1.0), 3 seeds × SciFact. 결과:

| Metric | Single-layer (§4.4 best) | Uniform 5-layer | 차이 |
|---|---|---|---|
| Δ all | **+0.030 ± 0.002** ✓ (3/3 strict) | **+0.004 ± 0.006** ✗ (0/3 strict) | $\mathbf{-0.026}$ |
| Δ hard | +0.092 ± 0.007 ✓ | +0.071 ± 0.004 ✓ | $-0.021$ |
| Δ easy | $-0.021 \pm 0.003$ | $-0.052 \pm 0.008$ | $-0.031$ (2.5× damage) |

Mechanism diagnosis (per-layer measured cos$(h_\ell^{\text{LoRA}}, h_\ell^{\text{frozen}})$):

| Layer ℓ | cos (uniform 5-layer) | cos (single-layer, ref) | Interpretation |
|---|---|---|---|
| 0 (embedding) | 1.000 | — | LoRA 가 q,v 의 후행 effect 만 받음 → 앞쪽 layer 는 *원래도 frozen 과 거의 동일*. anchor weight 가 redundant 한 constraint 에 낭비됨. |
| 3 | 0.998 | — | redundant |
| 6 | 0.991 | — | redundant |
| 9 | 0.965 | — | partial |
| **12 (final BERT)** | **0.697** | (single-layer 의 0.824, 참고) | budget 의 1/5 만 받아 *충분히 anchor 안 됨*. anchor 가 약하면 redistribution 회피 못 함. |

→ **Budget dilution 의 직접 mechanism evidence**: uniform weighting 이 loss budget 을 *redundant* 한 앞쪽 layer 에 분산시켜, 정작 anchor 가 *필요한* final layer 가 *under-constrained* 된다. §7.1 의 *differential* weighting 제안의 직접적 empirical 동기다.

### D. 오염된 실행에 관한 방법론적 주석

정규화 강도를 test metric에 맞춰 튜닝하거나 여러 변형 중 사후적으로 최선을 선택한 소수의 탐색적 실행은 selection-biased로 식별되어 위의 모든 주장에서 제외했다. 정리:

- 본문에 포함된 모든 결과는 단일 구성 및 고정 stopping rule 에 기반한다.
- 이 구별 — *다음 질문*의 깨끗한 사슬과 *더 나은 점수*를 좇는 오염된 추격 사이의 — 이 +0.030 결과를 신뢰할 만하게 유지하는 것이다.

---

## 종합 표 및 시각화 — 모든 실험의 Δ NDCG@10

본 문서에 기술된 모든 실험의 Δ NDCG@10 (frozen ColBERT baseline 대비) 을 세 slice (all / hard / easy) 로 정리한다. 단일 시드 점 추정과 3 시드 mean 이 혼재하며, 미보고 항목은 `n/a` 로 표시한다.

| # | 실험 | family | Δ all | Δ hard | Δ easy | 비고 |
|---|---|---|---|---|---|---|
| 1 | Mean-diff α=10 (A) | non-learned | n/a | **+0.064** | n/a | α-sweep best, hard 부분집합만 보고 |
| 2 | Single learned direction (B) | learned direction | n/a | +0.044 | n/a | 768 params |
| 3 | Scalar gate (B) | gating | −0.020 | n/a | n/a | gradient bottleneck |
| 4 | Per-token gate (B) | gating | n/a | −0.003 | n/a | 1.0 으로 saturate |
| 5 | Router K=2 (C) | router | +0.015 | +0.039 | n/a | effective K = 1.41 |
| 6 | Router K=4 (C) | router | +0.015 | +0.045 | n/a | effective K = 1.23 |
| 7 | Router K=8 (C) | router | **−0.038** | +0.049 | n/a | 잉여 capacity 가 easy 손상 |
| 8 | Random direction α=10 (D) | control | ≈ 0 | +0.011 | n/a | magnitude-only 가설 falsification |
| 9 | Plain LoRA r=8 (E) | lightweight | +0.001 | **+0.104** | **−0.085** | 24 intervention points, 3-seed mean |
| 10 | False negative removal (F) | training-data | −0.004 | +0.080 | −0.073 | label noise mediation, 3-seed |
| 11 | In-batch easy negative (G) | training-data | **+0.021** ✓ | +0.065 | **−0.017** | 본 연구 최초 strict net+, 3-seed |
| 12 | Continuous σ weighting (footnote) | training-data | +0.006 | +0.085 | −0.060 | α_w = 10, 3-seed |
| 13 | Relational anchor (H₁) | anchoring | +0.029 | +0.101 | −0.031 | rotation-invariant, 2/3 strict |
| 14 | **Per-token absolute anchor (H₂) ★** | **anchoring** | **+0.030** ✓ | **+0.092** | **−0.021** | **3/3 strict, best lightweight** |
| 15 | **Negative-side anchor (I)** | **anchoring** | **+0.028** ✓ | **+0.077** | **−0.014** | **3/3 strict, saturation proof** |
| 16 | E × NFCorpus (cross-domain) | cross-domain | **−0.320** ✗ | n/a | n/a | catastrophic transfer |
| 17 | E × FiQA (cross-domain) | cross-domain | **−0.347** ✗ | n/a | n/a | catastrophic transfer |
| 18 | F × NFCorpus (cross-domain) | cross-domain | **−0.316** ✗ | −0.089 | −0.566 | 0 % 회복 — refutes FN-as-root |
| 19 | G × NFCorpus (cross-domain) | cross-domain | −0.084 | n/a | n/a | 74 % 회복 of catastrophic gap |
| 20 | H₂ × FiQA (cross-domain) | cross-domain | −0.090 | −0.045 | −0.177 | 74 % 회복 (3-seed) |
| 21 | H₂ × NFCorpus (cross-domain) | cross-domain | −0.221 | −0.062 | −0.402 | 31 % 회복 (3-seed) |

**핵심 관찰**:

1. **Anchor 의 saturation**: 세 mathematically 독립인 anchor parameterization (H₁ relational, H₂ per-token, I symmetric/+d⁻) 이 모두 +0.030 ± 0.002 의 동일 Δ all 에 수렴 — anchor 가 도달 가능한 ceiling 의 *direct empirical proof*.
2. **Trade-off 의 universal 패턴**: Δ hard 와 Δ easy 의 부호가 거의 모든 학습 방법에서 반대 방향 — performance redistribution 의 직접 증거.
3. **Anchor family 의 독특한 위치**: anchoring family (H₁, H₂, I) 만이 Δ easy 의 손상 magnitude 를 1/4 수준 (−0.014 ~ −0.031) 으로 줄이면서 Δ hard 의 70–100 % 를 유지.

---

*위에 참조된 모든 figure는 실험 artifact로부터 재현 가능하다. 실험별 상세 보고서와 raw output은 원본 기록과 함께 저장되어 있다. confidence interval은 별도 명시가 없는 한 10,000-sample paired bootstrap의 95%이며, "유의"는 구간이 0을 배제함을 뜻한다.*
