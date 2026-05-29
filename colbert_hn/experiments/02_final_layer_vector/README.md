# 02_final_layer_vector — 단일 layer 의 *학습된* direction vector

## 목적

ROADMAP.md *single-direction 단계* 의 첫 단계. 단일 layer 12 에 *학습 가능* direction vector $v \in \mathbb{R}^{768}$ 를 두고 pairwise margin loss 로 학습한다:

$$\tilde{h}^{(12)} = h^{(12)} - v, \quad v|_{t=0} = \mathbf{0}$$

학습 가능 파라미터: **768 개** (단일 $v$).
Gate / multi-layer / multi-direction **없음**.

## 가설

### H1 부분 — 학습된 단일 direction 의 효과

| 슬라이스 | 통과 조건 (paired bootstrap 95 % CI) |
|---|---|
| confused | Δ NDCG@10 CI 하한 > **+0.064** (01b 의 α=10 mean-diff 가 sharpened anchor) |
| all | Δ NDCG@10 CI 하한 ≥ −0.005 (anchor preservation) |

### H5 qualitative — 학습된 direction 의 *내용* 이 mean-diff 와 다름

$$\cos(v_{\text{learned}}, v_{\text{mean-diff}}) < 0.9$$

(0.9 미만이어야 두 방향이 qualitatively 다른 정보를 잡았다고 주장 가능.)

## 학습 design

| 항목 | 값 | 근거 |
|---|---|---|
| Loss | pairwise margin, $m=0.2$ | DESIGN.md §4.2 default |
| Optimizer | AdamW (LR=$10^{-3}$, WD=$10^{-4}$) | DESIGN default |
| Batch size | 32 triplets | DESIGN default |
| Epochs / patience | 5 / 2 (val NDCG@10 early stop) | DESIGN default |
| **$\lambda_{\text{anc}} = 0$** | **DESIGN default ($10^{-2}$) 와 다름** | 02 는 gate 없음 → DESIGN 의 anchor reg 식 $\|g \cdot v\|^2$ 는 부적용. $\|v\|^2$ 만 reg 하면 01b 가 입증한 large-magnitude 효과를 차단. λ_anc 의 효과는 form-variant 단계 의 19 에서 sweep. DESIGN.md §11 mirror. |
| HN source | ColBERT train top-100 non-pos (01 과 동일) | 일관성 |
| Init | $v = \mathbf{0}$ | anchor preservation (DESIGN §11.3) |
| Hook layer | 12 (BERT-base 의 마지막) | 01 / 01b 와 동일 |
| Val split | train 의 10 % (query 단위 split) | DESIGN default |
| Dataset | SciFact (01 / 01b 와 일관) | 결과 비교 직접성 |
| Seed | 42 (단일 — seed × 3 은 32 에서) | DESIGN §3.7 |

## 실행 방법

```bash
.venv/bin/python experiments/02_final_layer_vector/run.py --dataset scifact --seed 42
```

Artifact 출력: `outputs/02_final_layer_vector/{dataset}/seed_{seed}/`:
- `config.json`, `env.json`
- `v_final.pt` (학습 후 $v$, 768-d)
- `triplet_stats.json` (mining 통계)
- `train_history.json` (step 별 loss, ‖v‖, epoch 별 val NDCG)
- `runs.json`, `runs_scored.json`, `metrics_per_query.json`, `metrics_aggregate.json`
- `delta_vs_baseline.json` (Δ vs 00_baseline)
- `delta_vs_mean_diff_alpha10.json` (Δ vs 01b α=10 — sharpened anchor)
- `cosine_with_mean_diff.json` (H5 qualitative)

## 비판적 review item — 본 02 시작 직전

1. **anchor reg = 0 의 위험**: $v$ 가 너무 커져 anchor 손상 가능. 학습 곡선의 ‖v‖ trace + val all-slice NDCG monitoring 으로 조기 진단. 만약 발견되면 03_layer_sweep 직전 19_anchor_reg_sweep 우선 검정.
2. **01b α=10 보다 못 함 가능성**: 학습이 sub-optimal converge (LR / margin 부적합) 일 수도. 그 경우 우선 (i) LR sweep, (ii) margin sweep — generalization 단계 의 30/31 일부 차용 가능.
3. **Cosine 분석의 baseline**: $v_{\text{mean-diff}}$ 도 SciFact train 의 mean diff. 즉 *같은 데이터* 에서 *다른 학습 방식* 으로 얻은 두 v 의 cosine. 0.9+ 이 나오면 학습이 *magnitude 만* 학습한 셈 — H5 의 *방향 의미성* 약함. 0.5 이하면 학습이 *다른* 방향을 발견한 것 → strong H5.

## 상세 보고서

[`report/02_final_layer_vector_report.md`](../../report/02_final_layer_vector_report.md) — 결과 + figure (실행 후 작성).
