# 06_k_sweep — Multi-direction router K-sweep (K ∈ {2, 4, 8})

## 목적

원래의 `06_two_directions` (K=2 단일 proof-of-concept) 가 *single-direction ceiling 0.665* 를 못 넘은 것이 (a) *router 의 capacity 부족* 인지 (b) *translation family algebraic 한계* 인지의 분리 검정. K=2 / 4 / 8 sweep 으로 *multi-direction 차원에서의 ceiling robustness* 를 확인.

수식 (single layer ℓ=12, K 가변):
$$\tilde{h}_t = h_t - \sum_{k=1}^{K} \pi_k(h_t) \cdot v_k$$

$$\pi(h_t) = \text{softmax}(W h_t + b) \in \Delta^{K}$$

학습 가능 파라미터: $2 K D + K$ (D=768).

| K | Params | router cap. |
|---|---|---|
| 2 | 3,074 | proof-of-concept (기존) |
| 4 | 6,148 | medium capacity |
| 8 | 12,296 | high capacity (≤ 50K limit) |

Anchor preservation: $v_k = \mathbf{0}$ init, $W \sim \mathcal{N}(0, 0.02^2)$, $b = \mathbf{0}$ → 초기 routing 균등 (entropy = log K). $v_k = 0$ 이므로 t=0 의 개입 = 0.

## 가설 (재정렬, 07 결과 반영)

| H | 기준 | 의미 |
|---|---|---|
| **H06a** | 어떤 K 에서도 confused-slice Δ NDCG@10 의 CI 하한이 02 (0.6651) 을 *유의 초과* | **translation family ceiling 의 multi-direction 차원 우회 가능성** |
| H06b | 모든 K 에서 all-slice Δ CI 하한 ≥ −0.005 | anchor preservation |
| H06c | K ↑ 일 때 effective K (routing entropy) 도 ↑ | router 의 capacity 활용 |
| H06d | 모든 K 에서 *어떤* $v_k$ 가 mean-diff direction 과 $|\cos| \gtrsim 0.4$ | informed direction subspace 의 일부를 학습하는지 |
| **H06e** | K ∈ {2, 4, 8} 모두 NDCG@10 ≈ 0.665 의 *동일 ceiling* | translation-trap 의 multi-direction 확장 confirmation |

**가설의 우선순위**:
- H06a 통과 → translation-trap 이론 *부분 약화*; multi-direction 만으로 ceiling 우회 가능 → 옛 ROADMAP 의 multi-direction 우선순위 회복.
- H06e 통과 + H06a 미통과 → translation-trap 이론 *강화*; *K 와 무관하게* informed subspace 의 ceiling 에 수렴 → bilinear M (Stage 2) 의 critical 필요성 확정.

본 시점 (07 falsification 후) 의 *기본 expectation*: H06e 통과 / H06a 미통과 (즉 K 와 무관한 ceiling).

## 학습 design (K 공통)

| 항목 | 값 |
|---|---|
| Loss | pairwise margin $m=0.2$ |
| Optimizer | AdamW (LR=$10^{-3}$, WD=$10^{-4}$) |
| λ_anc | 0 |
| Init | $v_k=\mathbf{0}$, $W \sim \mathcal{N}(0, 0.02^2)$, $b=\mathbf{0}$ |
| Batch / Epochs / Patience | 32 / 5 / 2 |
| Hook layer | 12 |
| Dataset | SciFact (seed 42) |

## 실행

```bash
.venv/bin/python experiments/06_k_sweep/run.py --dataset scifact --seed 42 --k 2
.venv/bin/python experiments/06_k_sweep/run.py --dataset scifact --seed 42 --k 4
.venv/bin/python experiments/06_k_sweep/run.py --dataset scifact --seed 42 --k 8
```

Artifact (per-K):
```
outputs/06_k_sweep/scifact/seed_42/k_{2,4,8}/
├── config / env / train_config / module_final.pt / train_history.json
├── cosine_v_pairs.json   — K-agnostic pairwise cos + cos vs mean-diff
├── routing_stats.json    — entropy / effective K (perplexity) / π_max>0.6 saturation
├── runs / runs_scored / metrics_per_query / metrics_aggregate.json
└── delta_vs_{baseline, mean_diff_alpha10, 02_learned}.json
```

## 비판적 review

1. **K=2, 4, 8 의 학습 가능 capacity 증가가 *router architecture-한계* 일 가능성**: linear router 가 K 가 클수록 *충분한* selectivity 못 가질 수 있음. confirmatory 인 mlp_router ablation 은 본 sweep 의 결과 따라 결정.
2. **각 K 의 학습이 다른 local minimum 수렴 가능**: same seed (42) 라도 K 만 바뀌면 학습 trajectory 가 발산. seed × 3 robustness 는 Stage 6 에서 진행.
3. **K=8 의 12K params 가 train signal (9K triplet) 보다 큼** — over-parameterization 위험. val curve 의 over-fitting 패턴 모니터링.
4. **각 K 의 *direction 학습* 의 의미적 해석 미실시**: 학습된 v_k 가 어떤 token / concept 에 정렬되는지의 qualitative analysis 미실시 (Stage 5 이후 검토).

## 상세 보고서

[`report/06_k_sweep_report.md`](../../report/06_k_sweep_report.md) — 실행 후 작성.
