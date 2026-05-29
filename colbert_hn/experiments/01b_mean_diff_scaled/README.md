# 01b_mean_diff_scaled — mean-diff direction 의 magnitude calibration sweep

## 목적

`01_mean_diff` 의 결과 (3 dataset 모두 |Δ NDCG@10| ≤ 0.001) 는 raw mean-difference direction 의 **L2 norm 이 너무 작아서** ($\|v\| \approx 0.03 – 0.27$) layer-12 hidden state 에 대해 거의 no-op 으로 작용한 결과이다. 즉 *direction 자체가 informative 한지* 와 *magnitude 가 부족한지* 가 confound 되어 있다.

본 sub-experiment 는 이 두 변수를 분리한다. Unit-normalize 한 direction 을 scale parameter $\alpha$ 로 곱하여 적용:

$$\tilde{h}^{(12)} = h^{(12)} - \alpha \cdot \frac{v}{\|v\|}, \quad \alpha \in \{0.5, 1, 2, 5, 10\}$$

학습 가능 파라미터: **0 개** (α 는 grid search hyperparameter).

## 가설 / 해석 시나리오

| α 의 효과 | 해석 | 02_final_layer_vector 로의 함의 |
|---|---|---|
| 어떤 α 에서 Δ NDCG@10 (confused) > 0 (CI 0 초과) | direction 자체는 informative, magnitude 만 부족이었음 | 02 의 학습은 *direction + magnitude 동시* 최적화 — narrative 강함 |
| 모든 α 에서 Δ ≈ 0 또는 부정적 | direction 자체가 *retrieval 에 informative 한 방향이 아님* | 02 의 학습된 direction 이 mean-diff 와 *qualitatively* 다른 결과 보여야 H5 성립. 또는 form 변경 (17_projection_out) 필요. |
| α 증가에 따라 Δ 가 단조 감소 (점점 부정적) | direction 이 *잘못된* 방향 (반대 sign 이 더 맞을지도) | 03_sign_flip ablation 의 motivation. |

## 데이터셋 범위 (제약)

본 sub-experiment 는 **SciFact 만**:
- 가장 작은 corpus (5K) → 5 회 sweep 의 비용 최소
- `01_mean_diff` 에서 이미 raw v 결과 확보 → 비교 anchor 존재
- Sweep 결과가 *concept proof* 이면 다른 dataset 으로 확장 검토

## 실행 방법

```bash
.venv/bin/python experiments/01b_mean_diff_scaled/run.py --dataset scifact --seed 42
```

내부적으로 α ∈ {0.5, 1, 2, 5, 10} 5 회 sweep — train v 계산은 한 번, test 평가만 α 마다 반복.

Artifact 출력 경로: `outputs/01b_mean_diff_scaled/{dataset}/seed_{seed}/`
- `v.pt`, `triplet_stats.json` (01_mean_diff 와 동일한 v — train side mining 재계산)
- `alpha_{α}/runs.json, runs_scored.json, metrics_per_query.json, metrics_aggregate.json, delta_vs_baseline.json` per α
- `sweep_summary.json` — α × {all, confused} × {Δ, CI} grid

## 성공 기준 (sub-experiment 단위)

- *최소* 한 α 에서 confused-slice 의 Δ NDCG@10 95% CI 의 *하한 > 0* → "form 작동, magnitude 가 핵심 변수" 결론
- 모든 α 에서 CI 가 0 포함 또는 negative → "form 자체 의심, 02 진행 전 form ablation 우선" 결론

## 상세 보고서

본 sub-experiment 의 결과는 `01_mean_diff` 의 raw 결과와 함께 **통합 보고서** 로 정리됨: [`report/01_mean_diff_report.md`](../../report/01_mean_diff_report.md) §3 (magnitude sweep).
