# 01_mean_diff — 비학습 mean-difference direction (raw + magnitude sweep)

본 보고서는 ROADMAP.md 의 *baseline 단계* 의 두 번째 단계인 `01_mean_diff` 의 결과를 통합 정리한다. *학습 없이* 단순한 representation-level intervention 이 layer-12 hidden state 에 어떤 효과를 가지는지 검정. 두 sub-experiment 로 구성:

- **(a) raw** [`experiments/01_mean_diff/`](../experiments/01_mean_diff/) — train 에서 추출한 unscaled $v = \bar{h}^{(12)}_{\text{HN}} - \bar{h}^{(12)}_{\text{pos}}$ 를 그대로 적용. 3 dataset (SciFact / NFCorpus / FiQA).
- **(b) sweep** [`experiments/01b_mean_diff_scaled/`](../experiments/01b_mean_diff_scaled/) — 동일 v 를 unit-normalize 후 $\alpha \in \{0.5, 1, 2, 5, 10\}$ 으로 scale 하여 적용. SciFact 에서 magnitude calibration 효과 분리.

학습 가능 파라미터: **0 개** ($\alpha$ 는 grid search hyperparameter).

## 1. 실험 환경

| 항목 | 값 |
|---|---|
| Python | 3.14.4, PyTorch 2.12.0, transformers 5.9.0 |
| Device | MPS (Apple Silicon) |
| Seed | 42 (단일) |
| Hook layer ℓ | 12 (BERT-base 의 마지막 transformer 출력) |
| Intervention form | $\tilde{h}^{(12)} = h^{(12)} - (\alpha \cdot v / \|v\|)$, raw 의 경우 $\alpha = \|v\|$ (raw v 그대로) |
| HN mining | baseline train retrieval 의 top-100 비-positive 에서 10 개 |
| Paired bootstrap | 10,000 iter, 95 % CI |

## 2. (a) Raw — 3 dataset

### 2.1 결과

| Dataset | NDCG@10 (mean_diff) | NDCG@10 (baseline) | Δ all (95% CI) | Δ confused (95% CI) | v_norm |
|---|---|---|---|---|---|
| SciFact | 0.6459 | 0.6464 | −0.0005 [−0.0029, +0.0010] | −0.0012 [−0.0063, +0.0021] | **0.27** |
| NFCorpus | 0.3299 | 0.3299 | +0.0001 [−0.0000, +0.0002] | +0.0001 [+0.0000, +0.0004] | **0.03** |
| FiQA-2018 | 0.3474 | 0.3473 | +0.0001 [−0.0008, +0.0009] | +0.0001 [−0.0014, +0.0014] | **0.21** |

모든 dataset 에서 `|Δ| ≤ 0.001` — 실용적으로 0. NFCorpus 의 confused CI 가 명목상 0 을 초과 (+0.0001) 하나 정량 효과 무시 가능.

![Raw mean-diff CI forest](figures/01_mean_diff/raw_delta_ci_forest.png)

*Figure 1. Raw unscaled mean-diff (3 dataset × {all, confused}) 의 paired bootstrap 95% CI. 세 dataset 모두 거의 0 근처. Confused slice 의 spread 가 all 보다 큼 (n 이 적기 때문). 어떤 row 도 CI 가 명확히 0 을 초과하지 않음.*

### 2.2 진단: v_norm 이 너무 작음

세 dataset 의 v_norm 은 $0.03 \sim 0.27$ 의 매우 작은 magnitude. 반면 BERT-base 의 layer-12 hidden state 의 norm 은 토큰 별로 $\sim 10$ 수준 (per-token mean of |h_i|) → **개입 vs hidden state 의 비율이 1-3 %**. 즉 raw mean-diff 는 model 의 forward 에 거의 영향 없음.

![v norm per dataset](figures/01_mean_diff/raw_v_norm.png)

*Figure 2. Raw mean-diff direction v 의 magnitude (3 dataset). NFCorpus 의 v_norm 이 가장 작음 (0.03) — train qrels 가 매우 많아서 (110K) 평균이 dense 하게 채워져 cancellation 강함. SciFact 는 train qrels 가 적어 (919) 평균에 변동성 더 큼 → 0.27. Magnitude 차이는 *mining sample size* 의 함수일 가능성 높음.*

### 2.3 잠재적 confound

(a) 의 결과만으로는 다음 두 가능성이 *분리되지 않음*:

- **C-magnitude**: direction 자체는 informative, magnitude 가 부족이라 효과 미발현.
- **C-form**: subtract form ($h - v$) 자체가 부적절, magnitude 와 무관하게 효과 없음.

이를 분리하기 위해 (b) magnitude sweep.

## 3. (b) Magnitude sweep — SciFact

### 3.1 결과

| α | NDCG@10 | Δ all (95% CI) | Δ confused (95% CI) | confused CI > 0 |
|---|---|---|---|---|
| 0.5 | 0.6477 | +0.0013 [−0.0017, +0.0044] | +0.0028 [−0.0037, +0.0094] | ✗ |
| 1.0 | 0.6478 | +0.0014 [−0.0030, +0.0055] | +0.0056 [−0.0019, +0.0133] | ✗ |
| **2.0** | **0.6536** | **+0.0072 [+0.0009, +0.0144]** | **+0.0177 [+0.0052, +0.0321]** | **✓** |
| **5.0** | **0.6666** | **+0.0202 [+0.0076, +0.0347]** | **+0.0515 [+0.0260, +0.0806]** | **✓** |
| **10.0** | **0.6690** | **+0.0226 [+0.0068, +0.0398]** | **+0.0644 [+0.0337, +0.0987]** | **✓** |

비교 baseline: SciFact NDCG@10 = 0.6464. raw v (α ≈ 0.27) 결과는 (a) 의 0.6459 와 일치.

**핵심 발견** — α ≥ 2 부터 confused-slice 의 Δ NDCG@10 95% CI 가 명확히 0 을 초과. α=10 에서 confused slice **+0.064 NDCG 포인트 개선** (baseline 의 +14 % 상대 개선).

### 3.2 시각화

![Alpha sweep curve](figures/01_mean_diff/alpha_sweep_curve.png)

*Figure 3. SciFact 의 magnitude sweep: α 에 따른 Δ NDCG@10. 음영 영역은 95% paired bootstrap CI. all 슬라이스 (파랑) 는 α=2 부근에서 0 라인을 넘어 단조증가, confused 슬라이스 (빨강) 는 더 가파른 기울기로 증가. α=10 에서도 saturation 미발생 → 더 큰 α 추가 sweep 가치 있음 (deferred).*

![Alpha sweep forest plot](figures/01_mean_diff/alpha_sweep_forest.png)

*Figure 4. α × {all, confused} 의 paired bootstrap 95% CI forest plot. "[+]" 표시는 CI 하한 > 0 (통계적으로 유의한 양의 개선) 의 row 들. α ≥ 2 의 모든 row 가 통과. α=10 confused 의 CI 는 [+0.034, +0.099] — 매우 robust 한 개선.*

![Alpha sweep ECDF](figures/01_mean_diff/alpha_sweep_ecdf.png)

*Figure 5. SciFact 의 per-query NDCG@10 의 ECDF, α 별 비교. α 증가에 따라 분포 전체가 우측 (높은 NDCG) 으로 이동. α=10 (밝은 색) 의 곡선이 baseline (회색) 보다 일관적으로 우측 — 즉 random query 가 아닌 *전반적* 인 개선.*

![Alpha sweep violin](figures/01_mean_diff/alpha_sweep_violin.png)

*Figure 6. α 별 per-query Δ NDCG@10 의 분포 (violin). 평균 (가로 막대) 가 α=0.5 에서 거의 0 → α=10 에서 약 +0.025 로 단조증가. 분포의 spread 가 α 증가에 따라 확대 — 일부 query 는 크게 개선 (+0.2 이상), 일부는 손상 (−0.1). 즉 **개입은 *균질하지 않음*** — 어떤 query 에는 직접적 도움, 어떤 query 에는 손상. 이건 *per-query selectivity (gate)* 의 필요성을 데이터로 시사 (후속 02 + 03_scalar_gate 의 motivation).*

## 4. 해석 — 두 sub-experiment 의 결합

(a) raw 의 null result + (b) sweep 의 positive result 는 다음을 시사:

| 결합 결론 | 증거 |
|---|---|
| **(C-form 기각)** subtract form ($h - v$) 자체는 작동 | (b) 의 α ≥ 2 에서 통계적으로 유의한 Δ > 0 |
| **(C-magnitude 확정)** raw v 의 magnitude 가 너무 작아 효과 미발현이었음 | (a) v_norm 0.03–0.27 vs (b) α=2 부터 효과 발현 |
| **mean-diff direction 자체는 informative** | unit-normalized 후 적절히 scale 하면 confused-slice 의 14% 상대 개선 가능 |
| **개입이 query-heterogeneous** | (b) Figure 6 — α 증가에 따라 일부 query 손상, 일부 큰 개선 → per-query selectivity 필요성 (motivation for gate / per-token routing) |

## 5. ROADMAP 의 H5 (학습의 의미성) 에 대한 새 anchor

본 실험은 학습된 LSR (02_final_layer_vector 이후) 가 *반드시 능가해야 할* anchor 를 다음과 같이 갱신:

- **이전 anchor 추정**: "비학습 baseline 은 거의 0 → 학습 효과를 따로 분리할 필요 없음"
- **현재 anchor (수정)**: "비학습 + magnitude calibration ($\alpha=10$) = SciFact confused +0.064. **학습된 v 는 이것 보다 *유의하게* 더 개선해야 H5 통과**."

이는 02_final_layer_vector 의 challenge 를 *strictly 높임* — 좋은 사이언스적 자세. 단순히 baseline 대비 +Δ 가 아니라 *informed non-learned baseline* 대비 추가 개선이 본 paper 의 가치 명제가 됨.

## 6. 데이터셋 범위 제약 및 잔여 open question

- (b) magnitude sweep 은 **SciFact 한정**. NFCorpus / FiQA 에서 동일 패턴 (α≥2 부터 양의 효과) 이 재현되는지 미검증. *후속 실험 후보*: NFCorpus / FiQA 에서 동일 sweep 약 25-40 분 추가 compute.
- α=10 에서 saturation 미발생. 더 큰 α (20, 50) 에서 monotonic 증가 지속하는지 / saturation / over-correction 발생하는지 미검증.
- mean-diff direction 의 *내용* 분석 미실시 (예: 학습된 v 와의 cosine similarity, 또는 token-embedding nearest neighbors). 24_routing_analysis 에서 통합 검토 예정.
- HN mining source (ColBERT top-100 non-pos) 외 BM25 / in-batch 와의 비교 미실시. 28_dynamic_hn 에서 다룸.

## 7. 다음 실험 — 02_final_layer_vector 의 sharpened challenge

(앞 §5 의 결론 반복) — 02 의 학습된 v 가 통과해야 할 기준:

| 기준 | 충족 조건 |
|---|---|
| H1 부분 | 02 의 confused-slice Δ NDCG@10 의 CI 하한이 **+0.064 (α=10 mean-diff sweep best) 보다 유의하게 크다** |
| H2 anchor preservation | 02 의 all-slice Δ NDCG@10 의 CI 하한이 ≥ −0.005 (현재 mean-diff α=10 의 all-slice 는 +0.023 으로 양호) |
| H5 학습의 의미성 | 위 H1 통과 + 02 의 학습된 v 가 mean-diff 방향과 *qualitatively* 다름 (cosine similarity 분석) |

또한 02 의 *학습 가능 magnitude* (scalar 또는 per-token gate) 가 본 sweep 의 α 와 비교될 수 있도록, 02 의 `v.norm()` × `gate.mean()` 의 effective scale 을 보고서에 포함 필요.

## 8. Artifact 위치

```
outputs/01_mean_diff/
├── scifact/seed_42/{config, env, runs, runs_scored, metrics_per_query,
│                     metrics_aggregate, delta_vs_baseline, v.pt, triplet_stats}.json
├── nfcorpus/seed_42/...
└── fiqa/seed_42/...

outputs/01b_mean_diff_scaled/
└── scifact/seed_42/
    ├── v.pt, triplet_stats.json, sweep_summary.json, config.json, env.json
    ├── alpha_0p5/{runs, runs_scored, metrics_per_query, metrics_aggregate, delta_vs_baseline}.json
    ├── alpha_1p0/...
    ├── alpha_2p0/...
    ├── alpha_5p0/...
    └── alpha_10p0/...

report/figures/01_mean_diff/
├── raw_delta_ci_forest.{pdf,png}
├── raw_v_norm.{pdf,png}
├── alpha_sweep_curve.{pdf,png}
├── alpha_sweep_forest.{pdf,png}
├── alpha_sweep_ecdf.{pdf,png}
└── alpha_sweep_violin.{pdf,png}
```
