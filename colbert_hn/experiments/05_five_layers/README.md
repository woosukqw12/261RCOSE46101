# 05_five_layers — direction at 5 layers (multi-layer LSR)

## 목적

ROADMAP §"Next" 의 첫 항목. single-direction 단계 single-layer ceiling (NDCG@10 ≈ 0.665) 우회 첫 시도. 학습된 direction $v_\ell$ 를 단일 layer 12 가 아닌 ColBERT v2 의 **5 layer 에 동시 적용**:

$$\tilde{h}^{(\ell)} = h^{(\ell)} - v_\ell, \quad \ell \in \mathcal{L} = \{0, 3, 6, 9, 12\}$$

각 $v_\ell$ 는 독립 학습. Gate 없음 (02 의 multi-layer 확장).

학습 파라미터: **5 × 768 = 3,840**.

## 가설

### H05a — multi-layer 가 single-layer ceiling 을 깬다

| Anchor | Slice | 통과 기준 (paired bootstrap 95% CI) |
|---|---|---|
| 02 (single-layer learned) | confused | Δ NDCG@10 CI **하한 > 0** |
| 01b α=10 (informed non-learned) | confused | Δ NDCG@10 CI **하한 > 0** |
| baseline | all | CI 하한 ≥ −0.005 (anchor preservation) |

### H05b — layer 별 기여 다양함

학습 종료 시 5 개 $\|v_\ell\|$ 가 *비균질* (모두 비슷한 크기가 아닌 — 일부 layer 가 *집중적* 학습). multi-direction 단계 multi-direction router 의 motivation 강화.

### H05c — qualitative 학습된 direction 비교

각 $v_\ell$ 와 mean-diff $v$ (layer 12) 의 cosine similarity 가 layer 별로 다름. Late layer (12) 가 mean-diff 와 더 닮고, early layer (0, 3) 가 *다른* 방향이면 → multi-layer 의 *complementary* 정보 확보.

## 학습 design (02 와 동일, layer set 만 확장)

| 항목 | 값 |
|---|---|
| Layer set $\mathcal{L}$ | $\{0, 3, 6, 9, 12\}$ |
| Loss | pairwise margin $m=0.2$ |
| Optimizer | AdamW (LR=$10^{-3}$, WD=$10^{-4}$) |
| λ_anc | 0 (single-direction 단계 deviation, DESIGN.md §11) |
| Init | $v_\ell = \mathbf{0}$ (모든 layer) |
| Batch / Epochs / Patience | 32 / 5 / 2 |
| Hook position | post-layer output (default) |
| Dataset | SciFact |

## 실행

```bash
.venv/bin/python experiments/05_five_layers/run.py --dataset scifact --seed 42
```

Artifact 출력: `outputs/05_five_layers/{dataset}/seed_{seed}/`:
- `module_final.pt` (5 개 v_ℓ 통합)
- `train_history.json` (per-step total ‖v‖, val curves) + `layer_norms.json` (학습 종료 시 per-layer ‖v_ℓ‖)
- `runs / runs_scored / metrics_per_query / metrics_aggregate.json`
- `delta_vs_{baseline, mean_diff_alpha10, 02_learned, 03_scalar_gate, 04_per_token_gate}.json`
- `cosine_with_mean_diff.json` (layer 별 cosine table)

## 비판적 review

1. **5 × 학습 파라미터 = 3,840** — SciFact 9,190 triplets 로 충분한가? Over-fitting risk 가 02 (768 params) 보다 5 배. Val curve early-stop 의존도 큼.
2. **Layer 0 (embedding) 의 효과 의문**: BERT embedding 은 토큰 + position embeddings — semantic 정보 적음. 이 layer 의 $v$ 가 의미 있게 학습될지 미지수. 결과 보고 deferred 의 *post_projection* 또는 *layer_subset* (예: $\{6, 9, 12\}$) ablation 동기 될 수 있음.
3. **Cumulative effect 의 *gradient interference***: 5 개 layer 의 hook 이 동시 작동 → gradient 가 모든 v_ℓ 에 backward. 일부 layer 의 학습이 다른 layer 의 학습을 *방해* 할 가능성. 학습 안정성 모니터링.
4. **Single-direction redundancy 문제 해결 못 함**: 02 의 cos=0.32 finding 은 *같은 layer 에서* multiple direction 이 비슷한 효과 — multi-layer 는 *다른 axis* 의 확장이라 직교적 문제. 05 가 ceiling 못 넘으면 → 본질적으로 multi-direction (multi-direction 단계) 필요.

## 상세 보고서

[`report/05_five_layers_report.md`](../../report/05_five_layers_report.md) — 실행 후 작성.
