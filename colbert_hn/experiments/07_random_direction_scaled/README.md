# 07_random_direction_scaled — translation-trap falsification

## 목적

ROADMAP §"Stage 1" 의 **translation-trap 이론의 결정적 falsification test**. mean-diff direction 대신 *random Gaussian unit vector* 를 동일 magnitude (α=10) 로 layer 12 에 적용하여, 01b 의 α=10 (mean-diff direction) 결과와 paired bootstrap 으로 비교.

수식:
$$\tilde{h}^{(12)} = h^{(12)} - \alpha \cdot \hat{v}_{\text{random}}, \quad \alpha = 10, \quad \hat{v}_{\text{random}} = \frac{v_{\text{random}}}{\|v_{\text{random}}\|_2}, \quad v_{\text{random}} \sim \mathcal{N}(0, I)$$

학습 가능 파라미터: **0** (random vector, seed 고정).

## 가설

**Translation-trap 이론**: 02–06 의 모든 finding 이 *translation family 의 algebraic ceiling* 에 기인 → magnitude 만이 lever 이고 *direction 의 내용은 무관*.

| Outcome | 해석 | 다음 분기 |
|---|---|---|
| 07 ≈ 01b α=10 (paired bootstrap CI 0 포함) | direction-agnostic 가설 *확정* — translation-trap 이론 통과 | Stage 2 (bilinear M) 본격 진입 |
| 07 < 01b α=10 (CI 하한 < 0) | mean-diff direction 이 *유의하게* 우월 — direction 도 lever | 옛 deferred (mean_diff_pca, projection_out) 회복 + bilinear M 도 병행 |
| 07 > 01b α=10 (CI 하한 > 0) | 매우 의외 — random 이 더 좋다? 표본 1 의 noise 가능성, 추가 random sample 필요 | seed 변경 random N=5 평균 검정 |

## 학습 design

- **학습 무필요**. Random vector 생성 후 hook 등록 + test 평가만.
- Hook layer: 12 (01b 와 동일).
- α: 10 (01b 와 동일).
- Dataset: SciFact.
- Seed: 42 (vector 생성 및 모든 randomness).
- 비교: per-query NDCG@10 paired bootstrap 10K iter, 95 % CI.

## 실행

```bash
.venv/bin/python experiments/07_random_direction_scaled/run.py --dataset scifact --seed 42 --alpha 10
```

Artifact: `outputs/07_random_direction_scaled/{dataset}/seed_{seed}/`:
- `v_random.pt` (생성된 random vector)
- `runs / runs_scored / metrics_per_query / metrics_aggregate.json`
- `delta_vs_{baseline, mean_diff_alpha10}.json`

## 상세 보고서

[`report/07_random_direction_scaled_report.md`](../../report/07_random_direction_scaled_report.md) — 실행 후 작성.
