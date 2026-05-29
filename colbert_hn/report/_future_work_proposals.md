# Future work proposals (paper deliverable § Future Work draft)

본 paper 의 *universal rank-collapse + spatial multiplicity escape* 가설은 *falsifiable*. 다음 *predictive test* 들은 가설의 직접 검정 — *본 paper 의 scope 외* 로 명시.

## A. Spatial multiplicity 의 직접 predictive test (paper-grade)

### A.1 Position-counts ablation (clean test of "positions" lever)

**가설**: confused-slice lift 가 *distinct intervention positions 수* 의 monotonic 증가.

| Config | Positions | 예측 |
|---|---|---|
| 10 Phase 2b q,v (현재) | 24 (q+v × 12 layers) | Δ conf +0.104 (확인됨) |
| **q,k,v,o all 12** | **48** (×2) | gas 옳다면 ~ +0.13~0.15 |
| q,v + FFN up_proj/down_proj (단, *module-type confound*) | 72 | 가설 + module-type 효과 *분리 불가* |

**Clean test = q,k,v,o all 12** (same attention module type, position 수만 ×2). +FFN 은 module-type 교란 → predictive test 의 *clean* 형태 아님.

### A.2 Position-fraction ablation (가설의 *decay* 검정)

LoRA q,v 의 *layer subset* 만 사용 — 12 → {6, 3, 1} layers. positions 6, 12, 24 의 confused lift trajectory 가 monotonic 인가? *Sub-linear* 패턴이면 가설 부분 약화.

### A.3 Per-token interventions (extreme multiplicity)

Token-level multiplicative gates ($\sigma(W h_t + b)$ per token, all layers) — *thousands* of positions × small per-position rank. 가설의 *extreme limit* 예측.

## B. Data bottleneck 해소 (paper-grade)

Phase 2b 의 ep3 train loss 0.10 — *완전 fit 직전*. 9K SciFact triplet 의 *data signal 한계* 가 strict 돌파 차단.

### B.1 Dynamic HN mining (ANCE-style)

매 epoch 마다 학습된 LoRA 의 *현재 top retrieval* 에서 new HNs 재 mining. *Curriculum effect* — 학습 진행할수록 더 어려운 HN. Static HN 의 *self-confirming bias* 해소.

### B.2 MS MARCO 전 학습 → SciFact transfer

대규모 BEIR 외 corpus (MS MARCO 530K triplets) 위 LoRA 학습 → SciFact 평가. *Data bottleneck* 의 absolute 해소.

### B.3 BEIR cross-dataset jointly

NFCorpus + FiQA + SciFact triplets 결합 → multi-task LoRA. Cross-dataset robustness + data scale 동시 해소.

## C. Distillation teacher 의 질 (09 의 noise teacher 한계 해소)

09 의 E5-Mistral *bi-encoder cosine margin* 이 mined HN 에서 ~50% noise → Margin-MSE 가 anchor regularizer 로 잘못 작동.

### C.1 MonoT5 / monoBERT cross-encoder distillation

*Cross-encoder* (full q-d attention) 의 ranking 신호. 09 의 *teacher quality* 한계 직접 해소. *Phase 2b 의 LoRA + MonoT5 distill 결합* → strict 돌파 가능성.

### C.2 RankT5 / Cohere-rerank pseudo-labels

Strong reranker 의 *listwise* score → KL divergence distill (Margin-MSE 대신).

### C.3 Multi-teacher ensemble

E5 + MonoT5 + BGE 의 *aggregated* margin. Single teacher noise 의 averaging.

## D. Cross-dataset robustness (NFCorpus/FiQA catastrophic 의 직접 해소)

### D.1 Initial-loss-aware LR rule

LR 을 *ep0 의 rank loss* 의 reciprocal 로 자동 조정:
$$\text{LR}_{\text{adapted}} = \text{LR}_{\text{base}} \times \frac{\text{loss}_{\text{SciFact, ep0}}}{\text{loss}_{\text{target, ep0}}}$$

NFCorpus ep0 loss 4.47 vs SciFact 0.66 → LR 자동으로 7× 줄어듦 (5e-5 → 7e-6). *Hyperparameter strategy* 의 *single rule*.

### D.2 Adaptive epoch budget

Validation-curve fitting 으로 over-fitting 시작 시점 자동 감지. 모든 dataset 의 *appropriate epoch* 자동 선택.

### D.3 Per-dataset normalization (baseline NDCG reciprocal scaling)

intervention magnitude (e.g., LoRA α scale) 을 dataset 의 *baseline NDCG* 의 reciprocal 로 scaling. *Strong-baseline dataset* (SciFact 0.65) 은 큰 perturbation 견디고, *weak-baseline dataset* (NFCorpus 0.33) 은 작은 perturbation 만.

## E. 본 paper framework 의 *next-paper* extensions

### E.1 DoRA (Decomposed LoRA) — magnitude + direction 분리 학습

LoRA 의 ΔW 를 *magnitude* + *direction* 으로 decompose. Per-adapter rank-collapse 의 *direction* 부분 만 학습 + *magnitude* 별도 — capacity 효율성 ↑ 가능성.

### E.2 AdaLoRA — adaptive rank pruning

학습 중 *낮은 importance* 의 rank 동적 제거. *Effective rank-1 collapse* 의 explicit handling. 24 positions 의 각 effective rank 가 *dataset-specific* 으로 다르게 결정.

### E.3 BitFit — bias-only finetune (extreme param efficiency)

BERT 의 bias parameters 만 학습 (~10K params, 50K budget 안). 우리의 *budget-vs-lift trade-off* curve 의 *extreme low* end.

### E.4 IA3 — multiplicative gates (non-additive intervention)

Attention key/value 의 *multiplicative* gate. 우리의 모든 intervention (additive translation, additive bilinear, additive LoRA) 와 *질적 다른* lever. Cross-method 비교.

### E.5 Soft prompt + LoRA hybrid

Learnable prefix tokens + LoRA. Spatial multiplicity × intervention type 의 결합.

---

## 본 future-work proposals 의 *과학적 가치*

본 paper 의 *universal rank-collapse + spatial multiplicity escape* framework 는:
1. *Predictive* — A 시리즈가 가설의 직접 검정 (monotonic lift 예측).
2. *Falsifiable* — 만약 q,k,v,o (48 positions) 가 q,v (24) 와 *동등 lift* 면 가설 약화.
3. *Generalizable* — D 시리즈가 cross-dataset 의 적용성 검정.
4. *Method-agnostic* — E 시리즈가 LoRA 외 PEFT methods 의 동일 framework 적용.

본 *proposals* 는 paper 의 *scope 외* — 단 *falsifiable & predictive* 의 명시 자체가 paper 의 *과학적 신뢰* 의 핵심 요소.
