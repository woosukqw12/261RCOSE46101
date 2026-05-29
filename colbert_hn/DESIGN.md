# DESIGN.md — colbert_layer_steering

본 문서는 본 repo 의 *아키텍처 설계 사양* 과 *실험 ablation matrix* 를 정의한다. `CLAUDE.md` 가 *고정 지침* 이라면, 본 문서는 *기술적 청사진* 이다. 본 문서는 실험 진행에 따라 *append-only* 로 보강되며, 기존 design 변경 시 changelog 를 §11 에 누적 기록한다.

---

## 1. Research Questions

본 연구가 검정하는 핵심 질문은 다음과 같다.

- **RQ1**: ColBERT v2 의 *frozen* 표현 공간 내에서, 다층에 분산된 lightweight steering module 의 가산 개입 ($\tilde{h}_\ell = h_\ell - g_\ell \cdot v_\ell$) 이 HN-confused query 의 retrieval quality (NDCG@10, MRR@10) 를 통계적으로 유의하게 개선하는가?
- **RQ2**: 그러한 개입이 *trivial / easy query* 의 성능을 손상시키지 않는 *anchor-preserving* 형태로 가능한가?
- **RQ3**: 학습된 steering module 이 *학습에 사용되지 않은 도메인* 으로 frozen 상태에서 전이될 때, 도메인 별 추가 fine-tuning 없이 일정 수준의 개선을 유지하는가? (일반화 가능성)
- **RQ4**: Confusion direction $v_\ell$ 의 *층별 (layer-wise) 기여도* 와 *방향성* 은 prior diagnostic study 가 관찰한 layer-wise confusion signal 과 정합적인가?

---

## 2. Hypotheses

각 RQ 에 대응하는 검정 가능한 가설:

| ID | 가설 | 검정 기준 |
|---|---|---|
| H1 | Full 5-layer LSR 은 confused slice 의 NDCG@10 을 baseline 대비 유의하게 개선한다 | paired bootstrap CI on Δ NDCG@10 (confused) 가 0 을 초과 |
| H2 | LSR 은 all-query slice 에서 baseline 과 통계적으로 구분되지 않거나 개선된다 (손상 없음) | paired bootstrap CI on Δ NDCG@10 (all) 의 하한이 명시 임계 $-\epsilon$ 이상 ($\epsilon = 0.005$) |
| H3 | 학습 도메인이 아닌 LOOCV held-out 도메인에서도 confused slice 개선이 유지된다 | LOOCV 3 fold 의 평균 Δ NDCG@10 (confused) CI 하한 > 0 |
| H4 | Layer-wise ablation (single layer vs. dense vs. [0,3,6,9,12]) 의 성능은 prior diagnostic finding 의 layer-wise confusion 분포와 정렬된다 | layer set × Δ metric heatmap 의 정성·정량 분석 |
| H5 | Steering direction $v_\ell$ 은 random vector / orthogonal vector 대비 유의한 개선을 보인다 (학습된 direction 의 의미성) | $v_\ell$ random / orthogonalized variant ablation vs. trained, paired bootstrap |

가설 검정 결과는 [REPORT.md](REPORT.md) 의 누적 narrative 에 통합된다. *Null result 도 동일 quality 로 보고* 한다.

---

## 3. Architecture

### 3.1 Steered forward path

ColBERT v2 의 BERT encoder forward 를 다음과 같이 수정한다:

$$
\tilde{h}_\ell(q, d) =
\begin{cases}
h_\ell(q, d) - g_\ell\big(h_\ell(q, d)\big) \cdot v_\ell & \text{if } \ell \in \mathcal{L} \\
h_\ell(q, d) & \text{otherwise}
\end{cases}
$$

기본 (default) 설정:
- $\mathcal{L} = \{0, 3, 6, 9, 12\}$ (prior diagnostic study finding)
- $h_\ell \in \mathbb{R}^{T \times 768}$, $T$ = sequence length
- 개입은 query / document encoding 양측에 동일 module 적용 (parameter sharing q-d 여부는 §6 ablation)

### 3.2 SteeringModule

각 layer $\ell$ 의 module 은 두 구성요소로 정의된다.

**(a) Direction vector**
$$v_\ell \in \mathbb{R}^{768}, \quad v_\ell \big|_{t=0} = \mathbf{0}$$

**(b) Per-token protective gate**
$$g_\ell(h) = \sigma\big( W_\ell h + b_\ell \big), \quad g_\ell : \mathbb{R}^{768} \to [0,1]$$
$$W_\ell \in \mathbb{R}^{1 \times 768}, \quad b_\ell \in \mathbb{R}$$

초기화 (anchor preservation):
- $v_\ell \leftarrow \mathbf{0}$ (zero init)
- $W_\ell \leftarrow \mathcal{N}(0, \sigma_w^2)$, $\sigma_w = 0.02$
- $b_\ell \leftarrow -3.0$ → $g_\ell |_{t=0} \approx \sigma(-3) \approx 0.047$

즉 $t=0$ 에서 $g_\ell \cdot v_\ell \approx 0$ → 개입은 *no-op* 으로 시작하여 baseline ColBERT 동작을 보존한다.

### 3.3 Parameter budget

| Component (per layer) | Params |
|---|---|
| $v_\ell$ | 768 |
| $W_\ell$ | 768 |
| $b_\ell$ | 1 |
| **Subtotal / layer** | **1,537** |
| × 5 layers | **7,685** |

총 trainable 파라미터 ≈ 7.7K ≪ 50K 상한 (CLAUDE.md §3.2). Full 12-layer dense ablation 시도 ~18.4K 으로 여전히 상한 내.

### 3.4 Gradient flow

`requires_grad`:
- `colbert.*` (encoder + projection linear): **False**
- `steering.layer_{0,3,6,9,12}.{v, W, b}`: **True**

Forward graph 는 ColBERT 를 통과하지만 optimizer 는 steering parameter 만 갱신한다. ColBERT 의 dropout / LayerNorm 등은 *train mode* 가 아닌 *eval mode* 로 고정 (BN/dropout 영향 제거).

---

## 4. Training Protocol

### 4.1 Data

| Dataset | Train queries | Eval queries | Corpus size |
|---|---|---|---|
| SciFact | ~810 | 300 | 5,183 |
| NFCorpus | ~2,590 | 323 | 3,633 |
| SciDocs | ~14,950 (mined) | 1,000 | 25,657 |

Train triplet ($q$, $d^+$, $d^-$) 구성:
- $d^+$: BEIR qrel positive (relevance ≥ 1)
- $d^-$: BM25 top-100 중 *non-positive* 에서 sampling. Hard-HN mining 은 ColBERT v2 baseline 의 top-20 non-positive 활용.
- Prior repo 의 15K labeled set 재활용 여부는 *라이센스 확인 후* 결정 (보류).

### 4.2 Loss

기본 (default) loss:
$$\mathcal{L} = \mathcal{L}_{\text{rank}} + \lambda_{\text{anc}} \mathcal{L}_{\text{anc}} + \lambda_{\text{gate}} \mathcal{L}_{\text{gate}}$$

- $\mathcal{L}_{\text{rank}}$: pairwise margin
$$\mathcal{L}_{\text{rank}} = \max(0, m - s(q, d^+) + s(q, d^-))$$
$s(q, d)$ = ColBERT MaxSim score (steered), $m = 0.2$ (BEIR 표준)

- $\mathcal{L}_{\text{anc}}$: anchor preservation regularizer
$$\mathcal{L}_{\text{anc}} = \frac{1}{|\mathcal{L}|} \sum_{\ell \in \mathcal{L}} \mathbb{E}_t \big[ \| g_\ell(h_\ell^{(t)}) \cdot v_\ell \|_2^2 \big]$$
$\lambda_{\text{anc}} = 0.01$ (default; §6 sweep)

- $\mathcal{L}_{\text{gate}}$: gate sparsity (optional)
$$\mathcal{L}_{\text{gate}} = \frac{1}{|\mathcal{L}|} \sum_{\ell \in \mathcal{L}} \mathbb{E}_t \big[ g_\ell(h_\ell^{(t)}) \big]$$
$\lambda_{\text{gate}} = 0.001$ (default; §6 sweep)

### 4.3 Optimizer & schedule

| Item | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 (steering params only) |
| Weight decay | 1e-4 |
| Batch size | 32 triplets |
| Epochs | 5 (early stop on val NDCG@10) |
| Warmup | linear, 100 steps |
| Schedule | cosine decay |
| Gradient clip | 1.0 |
| Mixed precision | fp16 (eval은 fp32) |

### 4.4 Validation

- Held-out 10 % of train queries → val NDCG@10 (confused slice)
- Early stop patience: 2 epochs

---

## 5. Evaluation Protocol

### 5.1 Slices

| Slice | 정의 |
|---|---|
| `all` | 평가셋 전체 query |
| `confused` | ColBERT baseline 의 top-1 prediction ≠ true relevant doc 인 query |
| `lexical-HN` | top-k retrieved 중 query 와 BM25 lexical overlap ≥ τ 이지만 비-relevant 인 doc 이 포함된 query (Lexical-HN 정의는 prior repo §B 참조) |
| `hard-HN` | semantic similarity 는 높으나 비-relevant 인 doc 이 top-k 에 포함된 query (Hard-HN 정의는 prior repo §B 참조) |

### 5.2 Metrics

| Metric | 기록 위치 |
|---|---|
| NDCG@{1, 3, 5, 10, 20} | primary |
| MRR@10 | primary |
| Recall@{10, 50} | secondary |
| MAP | secondary |

### 5.3 Statistical analysis

CLAUDE.md §8 준수. 추가로:
- **Per-query Δ-metric** 분포를 figure 로 시각화 (violin / ECDF)
- **Significance**: paired bootstrap 10,000 iter, 95 % CI on Δ. CI 가 0 을 포함하지 않을 때 "통계적으로 구분 가능" 표기.
- **Multiple comparison**: ablation 수가 많을 경우 Holm–Bonferroni 보정 (단 ablation 별 *독립 가설* 인 경우 보정 생략, 보고서에 명시).

### 5.4 LOOCV

| Fold | Train | Eval |
|---|---|---|
| 1 | NFCorpus + SciDocs | SciFact |
| 2 | SciFact + SciDocs | NFCorpus |
| 3 | SciFact + NFCorpus | SciDocs |

각 fold × 3 seed = 9 runs / config.

---

## 6. Ablation Matrix

### 6.0 Ablation completeness invariant (철칙)

CLAUDE.md §3.8 의 *ablation completeness 철칙* 을 본 design 에서 구현하기 위한 규약:

1. **1 choice ↔ ≥ 1 ablation**: §3 (architecture) 와 §4 (training protocol) 의 *모든* design choice 는 §6.1–6.3 의 ablation matrix 에 *대응 entry* 를 갖는다. 각 entry 는 단 하나의 변수만 변경한다 (single-variable principle).
2. **Choice ↔ ablation mapping table** (§6.4) 를 명시적으로 유지하여, 어떤 design 결정이 어떤 ablation 으로 검정되는지 1:1 추적 가능하게 한다.
3. **새 choice 추가 시**: PR 동시에 (a) DESIGN.md §3 또는 §4 의 해당 항목 업데이트, (b) §6 의 ablation entry 추가, (c) §6.4 mapping 업데이트. 셋 중 하나라도 누락된 경우 본 design 은 *불완전* 상태이며 이후 보고서 작성을 차단한다.
4. **Prior-finding 인용도 예외 없음**: 본 repo 외부 (prior diagnostic repo, paper) 의 결정을 그대로 채택하더라도 본 repo 내에서 최소 1 개의 대안과 비교 검정한다.
5. **Default 의 정당성**: ablation 결과로 default 가 alternative 대비 *유의하게 우월 또는 비열등* 임이 확인된 후에야 main 결과 (보고서) 에 default 의 수치를 채택한다.

### 6.1 Baseline & non-learned anchor (00, 01)

| ID | Dir | Variable | Default | Variant | 검정 가설 / 의도 |
|---|---|---|---|---|---|
| 00 | `00_baseline` | — | — | — | anchor 확립 (H1/H2 reference) |
| 01 | `01_mean_diff` | direction source | (02 부터 학습된 $v$) | $v = \bar{h}_{\text{HN}} - \bar{h}_{\text{pos}}$ (비학습) | H5 강화 — 학습 baseline anchor; learned 가 mean-diff 보다 풍부함을 보여야 함 |

### 6.2 Architectural bottom-up (02–05)

§3 의 모든 architectural choice 를 incremental 로 분리하여 검정. ROADMAP §"완료" + §"Next" 의 sequential numbering 과 일치.

| ID | Dir | Variable | Default | Variant | 검정 | 상태 |
|---|---|---|---|---|---|---|
| 02 | `02_final_layer_vector` | (main) 학습된 $v$ at single $\ell$ | $\ell=12$, $h-v$, no gate | — | H1 부분 — 학습된 단일 direction 의 효과. 01 대비 개선이 H5 primary 증거. | ✅ 완료 |
| 03 | `03_scalar_gate` | + gate $g_\ell$ (scalar) | (02: no gate) | scalar $g_\ell \in [0,1]$, $b=-3$ init | H2 부분 — gate 역할 | ✅ 완료 (negative) |
| 04 | `04_per_token_gate` | gate granularity | scalar (03) | per-token $g_\ell(h)$ | per-token 선택성 효과 | ✅ 완료 (saturated) |
| 05 | `05_five_layers` | layer set | single $\ell$ | $\{0,3,6,9,12\}$ | H1 primary — full default | 🔜 next |

### 6.3 Form variant & theoretical positioning (09 + deferred)

추가 비학습 / 형식 baseline 과의 비교로 본 form 의 IR / concept-erasure 문헌 내 좌표 명시.

| ID | Dir | Variable | Default | Variant | 검정 | 상태 |
|---|---|---|---|---|---|---|
| 06 | `06_projection_out` | intervention form | $h - g \cdot v$ | $h - \text{proj}_v(h)$ (LEACE family) | Belrose et al. 2023 form 과 비교 | 🔜 next |
| — | `mean_diff_pca` | $v$ source | learned (02) | top-1 PC of $\{h_{\text{HN}} - h_{\text{pos}}\}$ (비학습) | mean-diff (01) 외 비학습 baseline | deferred |
| — | `combined_form` | intervention form | $h - g \cdot v$ | $h - g \cdot \text{proj}_v(h)$ | gate + projection 결합 | deferred |
| — | `anchor_reg_sweep` | $\lambda_{\text{anc}}$ | 0 (single-direction 단계 deviation, §11) | $\{10^{-3}, 10^{-2}, 10^{-1}, 1\}$ | anchor regularizer 강도 | deferred |

### 6.4 Multi-direction router (06–08, main novelty)

수식: $\tilde{h}_\ell = h_\ell - g_\ell(h_\ell) \cdot \sum_{k=1}^{K} \pi^k_\ell(h_\ell) \cdot v^k_\ell$.

| ID | Dir | Variable | Default | Variant | 검정 | 상태 |
|---|---|---|---|---|---|---|
| 07 | `07_two_directions` | $K$ | $K=1$ (02 final) | $K=2$ + router $\pi$ | proof-of-concept | 🔜 next |
| 08 | `08_k_sweep` | $K$ | (07 결과로 결정) | $K \in \{1, 2, 4, 8, 16\}$ | direction 다양성의 saturation | 🔜 next |
| 09 | `09_routing_analysis` | (qualitative) | — | per-query routing pattern 분석 | direction 해석 — 원인 규명 | 🔜 next |
| — | `router_capacity` | router 표현력 | linear softmax | 2-layer MLP | router 용량 | deferred |
| — | `routing_entropy` | router 정규화 | none | entropy reg (Switch Transformer) | direction collapse 방지 | deferred |
| — | `kmeans_init` | $v^k$ init | zero | HN-pos diff 의 k-means cluster center | warm-start | deferred |

### 6.5 Generalization & robustness (~10–~16, 후속 commit)

실행 시점에 sequential ID 부여 (~10+). 본 표는 *content* 중심이며 ID 는 잠정.

| 가 ID | Dir | Variable | Default | Variant | 검정 | 우선 |
|---|---|---|---|---|---|---|
| ~10 | `cross_model_e5` | retriever | ColBERT v2 | E5-base | reviewer "ColBERT-only?" 반박 | core |
| ~11 | `cross_model_bge` | retriever | ColBERT v2 | BGE-small | cross-model second data point | extended |
| ~12 | `dynamic_hn` | HN mining | ColBERT-top-K static | dynamic re-mining (ANCE 류) | Xiong 2021 의 self-confirming bias 반박 | core |
| ~13 | `loocv_held_out` | train/test 도메인 | in-domain | LOOCV 3-fold | H3 — domain generalization | core |
| ~14 | `loss_objective` | ranking loss | pairwise margin | InfoNCE / KD | objective sensitivity | extended |
| ~15 | `margin_sweep` | margin $m$ | $0.2$ | $\{0.05, 0.1, 0.5, 1.0\}$ | hyperparameter robustness | extended |
| ~16 | `seed_robustness` | seed | 42 | $\{42, 1337, 2024\}$ × multi-direction best | statistical robustness (CLAUDE §3.7) | core |

### 6.6 Choice ↔ Ablation mapping table

§6.0 invariant 검증용. 각 design choice 가 *적어도 한 ablation* 에 의해 검정되는지 mapping. *완료된 실험은 [✓] 표시*, *next* 또는 *deferred* 는 미표시.

| Design choice (location) | Default | Ablating experiment(s) |
|---|---|---|
| Layer set $\mathcal{L}$ (§3.1) | $\{0,3,6,9,12\}$ | 05 [next], deferred (layer_sweep / dense_layers / layer_subset) |
| Intervention form (§3.1) | $h - g \cdot v$ (additive subtraction) | 06 [next, projection_out], deferred (sign_flip, combined_form) |
| Direction parameterization (§3.2a) | $v_\ell \in \mathbb{R}^{768}$, learned | 01 ✓ (mean-diff), 07-09 [next, multi-direction], deferred (random, pca) |
| Direction init (§3.2) | $v_\ell = \mathbf{0}$ | deferred (init_sweep, kmeans_init) |
| Gate function (§3.2b) | 1-layer linear sigmoid, per-token | 03 ✓ (scalar), 04 ✓ (per-token), deferred (gate_off, gate_capacity) |
| Gate bias init (§3.2) | $b_\ell = -3$ | deferred (bias_init_sweep) |
| Intervention 공간 | 768-d (pre-projection) | deferred (post_projection) |
| Ranking loss (§4.2) | pairwise margin | generalization (loss_objective) |
| Margin $m$ (§4.2) | $0.2$ | generalization (margin_sweep) |
| $\lambda_{\text{anc}}$ (§4.2) | 0 (single-direction 단계, §11) | deferred (anchor_reg_sweep) |
| HN mining source (§4.1) | ColBERT-top-K static | generalization (dynamic_hn) |
| Retriever choice | ColBERT v2 | generalization (cross_model_e5, cross_model_bge) |
| Cross-domain transfer | in-domain | generalization (loocv_held_out) |
| Statistical robustness (§5.3) | seed 42 single | generalization (seed_robustness) |

### 6.7 Deferred / supplementary ablations (ROADMAP 외)

다음 design choice 들은 §3 / §4 에 default 가 정의되어 있으나 ROADMAP 의 01–32 에 *first-class experiment* 로 포함되지 않는다. **§3.8 ablation completeness 철칙의 partial 위반** 으로 명시하며, 본 프로젝트 scope 내에서 supplementary appendix 형식으로 처리하거나 *시간 허용 시* numbered experiment (33+) 로 승격한다.

| 미커버 choice | 위치 | 영향 | Mitigation |
|---|---|---|---|
| q/d module sharing (shared vs separate) | §3.1 | 파라미터 수 × 도메인 transfer 의 양면 영향 | 02–08 의 default = **shared** 로 commit. 잠재적 33_qd_sharing 으로 승격 검토. |
| Hook position (post-layer vs pre-LN) | §3.1 | numerical 안정성 / gradient flow 차이 | default post-layer 사용. 33+ 승격 검토. |
| Encoder eval vs train mode | §3.4 | encoder dropout 의 영향 (training 중) | default eval mode commit. journal 시점 검증. |
| $\lambda_{\text{gate}}$ sweep | §4.2 | gate sparsity 강도 | default $10^{-3}$ commit. 보조 sweep 필요 시 19 와 함께 |
| LR / batch / optimizer / warmup / fp16 sweeps | §4.3 | hyperparameter robustness 일반 | default commit + 32_seed_robustness 와 함께 reviewer 의 typical 1-2 sweeps 만 |
| Confused slice 정의 (top-1 vs top-3) | §5.1 | slice 정의 민감도 | default top-1 commit. supplementary 에서 top-3 비교만 보고. |

본 §6.7 의 항목들은 *모두* journal 투고 직전 "supplementary tables" 로 보강되거나, ROADMAP 의 extended 우선순위 작업 후 시간 여유 시 33+ 로 추가 실행.

---

## 7. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Steering module 이 trivial query 손상 | M | H | $\lambda_{\text{anc}}$ sweep, all-slice NDCG monitoring, anchor preservation init |
| Confused slice 정의가 baseline 의존적 → 평가 편향 | M | M | confused 외 lexical-HN / hard-HN 슬라이스 병행 보고 |
| 7.7K params 가 표현력 부족 → null result | M | H | T3 의 low-rank direction 으로 확장 가능; null 도 contribution |
| BEIR 3 dataset 의 도메인 협소성 → 일반화 주장 약함 | H | M | LOOCV 강조 + 가능 시 추가 BEIR 도메인 (TREC-COVID 등) 보조 평가 |
| ColBERT v2 의 official weight loading 안정성 | L | H | reproducible env (requirements.txt pin), 학습 시작 전 baseline NDCG 재현 검증 |
| 학습 불안정 (gate 가 0 으로 collapse) | M | M | gate 손실 monitoring, $\lambda_{\text{gate}}$ tuning |
| 저널 투고 시 prior repo 와의 *novelty delta* 부족 우려 | M | H | DESIGN.md §1 RQ + §2 H 에서 prior diagnostic finding 을 *전제* 로 명시, *intervention 의 새 차원* 으로 contribution framing |

---

## 8. Reproducibility Checklist

투고 단계에서 reviewer 가 확인 가능해야 할 항목:

- [x] Python version pin (`.python-version` = `3.14.4`)
- [x] `requirements.txt` (top-level deps) + `requirements.lock.txt` (transitive 전체 정확 버전, `pip freeze`)
- [ ] BEIR dataset 다운로드 script 자동화 (`src/data.py --extract`)
- [ ] Seed 고정 (PyTorch, NumPy, Python random) 및 deterministic mode (`torch.backends.cudnn.deterministic`)
- [ ] 모든 config 가 `src/configs.py` 에서 declarative 하게 정의됨 (CLI override 가능)
- [ ] Output artifact (`outputs/{step}/{config}/{seed}/`) 의 디렉토리 layout 일관
- [ ] Raw metric (per-query) 와 aggregated metric 모두 저장 → bootstrap 재계산 가능
- [ ] Figure 생성 script (`src/visualize.py`) 가 saved artifact 만으로 재실행 가능
- [ ] Hardware spec (GPU type, memory) 보고서에 명시
- [ ] Wall-clock training time 보고서에 명시 (table)

---

## 9. Notation

본 문서·코드·보고서에서 일관 사용:

| Symbol | 의미 |
|---|---|
| $q, d$ | query, document |
| $h_\ell$ | layer $\ell$ 의 hidden state ($\in \mathbb{R}^{T \times 768}$) |
| $\tilde{h}_\ell$ | steered hidden state |
| $v_\ell$ | layer $\ell$ 의 confusion direction vector |
| $g_\ell$ | layer $\ell$ 의 gate function |
| $\mathcal{L}$ | steering 대상 layer 집합 |
| $s(q, d)$ | ColBERT MaxSim score |
| $\Delta M$ | metric $M$ 의 baseline 대비 변화량 (`steered − baseline`) |
| $\sigma$ | sigmoid |

---

## 10. References (선행 연구)

본 design 의 근거가 되는 외부 문헌은 실험 진행에 따라 보강.

핵심:
- Khattab & Zaharia, 2020. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. SIGIR.
- Santhanam et al., 2022. *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*. NAACL.
- Thakur et al., 2021. *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS Datasets.

Steering / intervention 관련:
- Belrose et al., 2023. *LEACE: Perfect linear concept erasure in closed form*. NeurIPS.
- Turner et al., 2024. *Activation Addition: Steering Language Models Without Optimization*. (preprint)
- 추가 보강 예정.

Prior diagnostic study:
- `[prior repo]` — prior repo 의 SAR muted result 와 layer-wise confusion signal finding. 본 repo §1 motivation.

---

## 11. Changelog

본 §11 은 *DESIGN.md 자체* 의 변경 이력만 기록한다 (architectural / methodological choice 단위). repo 전체의 변경은 `CHANGELOG.md` 에 누적.

| Date | Change | Reason |
|---|---|---|
| 2026-05-23 | 초안 작성 (§1–§10) | 학습 실험 개시 전 alignment |
| 2026-05-23 | §6.0 ablation completeness invariant + §6.5 choice↔ablation mapping 추가, §6.2 → T2A 분리, §6.3 T2B 신설 | CLAUDE.md §3.8 ablation completeness 철칙 enforcement |
| 2026-05-23 | **§6 전면 rewrite**: 옛 T1.\*/T2A.\*/T2B.\*/T3.\* numbering 폐기. ROADMAP.md 의 sequence 와 alignment. §6.1–§6.5 = baseline / architectural / form / multi-direction / generalization 그룹, §6.6 mapping, §6.7 deferred / supplementary ablations 신설. | ROADMAP commit 에 따른 alignment + §3.8 partial 위반의 정직한 documentation |
| 2026-05-23 | **§4.2 anchor regularizer deviation for single-direction 단계 (02–05)**: $\lambda_{\text{anc}}$ default 가 §4.2 의 $10^{-2}$ 였으나 02–05 에서는 **$\lambda_{\text{anc}} = 0$** 으로 설정. 근거: (a) 02 는 gate 없음 → 원 anchor reg 식 $\|g \cdot v\|^2$ 의 form 부적용; $\|v\|^2$ 만 reg 시 01b 가 입증한 large-magnitude 효과 차단. (b) 03+ 의 gate 추가 단계에서 동일 변수 (gate 자체) 만 변경하도록 (single-variable principle). λ_anc 의 sweep 은 deferred `anchor_reg_sweep` 에서 수행. | 01b 결과 (α=10 이 SciFact confused +0.064 의 anchor) + ablation 명료성. |
| 2026-05-23 | **"Phase" 용어 제거**: 옛 baseline 단계/1/2/3/4 grouping 폐기. §6.1–§6.5 의 descriptive 헤더로 교체 (baseline / architectural / form / multi-direction / generalization). 실험 번호 (00–06, 07+) 가 primary identifier. | 사용자 directive — 그룹 라벨 제거 + 실험 번호 중심. |
