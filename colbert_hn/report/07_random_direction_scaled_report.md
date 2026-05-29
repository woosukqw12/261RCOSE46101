# 07_random_direction_scaled — translation-trap falsification

본 보고서는 ROADMAP §"Stage 1" 의 결정적 falsification test 결과. **결론: piece 피드백의 *direction-agnostic 가설* 은 기각**. 같은 magnitude (α=10) 의 *random direction* 과 *mean-diff direction* 사이에 **5.3 NDCG 포인트** 의 통계적으로 유의한 ranking 차이가 관찰됨 — direction 의 *내용* 은 명백한 lever.

## 1. 실험 설계

학습 무필요. Random Gaussian vector 를 생성한 후 unit-normalize 하고 α=10 으로 scaling 하여 layer 12 에 동일 hook 형식으로 적용:

$$\tilde{h}^{(12)} = h^{(12)} - \alpha \cdot \hat{v}_{\text{random}}, \quad \alpha=10, \quad \hat{v}_{\text{random}} = \frac{v_{\text{random}}}{\|v_{\text{random}}\|_2}, \quad v_{\text{random}} \sim \mathcal{N}(0, I_{768})$$

seed 42 로 random vector 생성 (재현 가능). $\|\alpha \cdot \hat{v}_{\text{random}}\|_2 = 10.0000$ 정확히 — 01b α=10 의 $\|\alpha \cdot \hat{v}_{\text{mean-diff}}\|_2 = 10$ 과 동일 magnitude. 두 실험의 *유일한 차이는 direction* 이다.

## 2. 결과

| Comparison | NDCG@10 |
|---|---|
| baseline (00) | 0.6464 |
| **07 random × α=10** | **0.6485** |
| **01b mean-diff × α=10** | **0.6690** |

### Paired bootstrap CI (95 %)

| Anchor | Slice | Δ NDCG@10 | 유의 |
|---|---|---|---|
| baseline (00) | all | +0.0021 [-0.0068, +0.0112] | — (0 포함) |
| baseline (00) | confused | +0.0112 [-0.0058, +0.0291] | — (0 포함) |
| 01b α=10 mean-diff | all | -0.0205 [-0.0386, -0.0041] | ✗ **negative** |
| 01b α=10 mean-diff | confused | **-0.0533 [-0.0905, -0.0201]** | ✗ **negative** |

![Direction compare](figures/07_random_direction_scaled/direction_compare.png)

*Figure 1. 같은 magnitude ($\|\alpha \hat{v}\| = 10$), 다른 direction 의 비교. baseline (intervention 없음, 0.6464), 07 random direction (0.6485, baseline 거의 동일), 01b mean-diff direction (0.6690, baseline +0.022 / confused +0.064). 두 LSR 변형의 *유일한 차이는 direction* — magnitude 는 정확히 같음. mean-diff 가 random 보다 *명확히* 우월.*

![Delta CI forest](figures/07_random_direction_scaled/delta_ci_forest.png)

*Figure 2. 07 의 paired bootstrap 95 % CI. vs baseline 은 두 slice 모두 CI 가 0 포함 (효과 사실상 0). vs 01b α=10 은 **CI 가 0 명확히 미달** ([-]) — random 이 mean-diff 보다 통계적으로 유의하게 worse.*

![ECDF compare](figures/07_random_direction_scaled/ecdf_compare.png)

*Figure 3. SciFact 의 per-query NDCG@10 의 ECDF: baseline (회색), 07 random (주황), 01b mean-diff (파랑). 07 random 의 곡선이 baseline 과 거의 겹침 — 효과 ≈ 0. 01b mean-diff 의 곡선이 baseline 보다 명확히 우측 — 분포 전체가 개선 방향으로 이동. **같은 magnitude 인데도 random 은 baseline 과 구분 불가, mean-diff 는 명확히 개선** — direction 의 내용이 lever 라는 직접 증거.*

## 3. 해석

### 3.1 *Direction-agnostic 가설* 의 명확한 기각

Translation-trap 이론의 첫 번째 outcome ("같은 magnitude 면 어떤 direction 이든 비슷한 ranking 개선") 은 본 실험으로 *직접 falsify*. 같은 α=10 의 magnitude 인데 random 은 효과 ≈ 0, mean-diff 는 confused +0.064 → **direction 의 *내용* 이 결정적**.

### 3.2 단, *translation-trap 의 algebraic 진단 자체* 는 여전히 유효

피드백의 두 가지 주장을 분리해야 한다:

| 주장 | 본 실험 후 상태 |
|---|---|
| (A) 02–06 은 모두 *translation family* 의 변형 | ✓ 여전히 정확 (algebraic 분류) |
| (B) Translation family 안에서 direction 은 무관, magnitude 만 lever | ✗ **falsify** (본 실험) |
| (C) Translation family 의 ceiling 자체가 algebraic 한계 (정보 한계 아님) | 미검정 — **08 bilinear M 으로만 답 가능** |

(B) 가 기각되더라도 (C) 의 검정은 여전히 critical. *Informed direction (mean-diff family)* 이 translation family 의 ceiling 을 결정하고 그 ceiling 의 *위치* 가 algebraic 한계인지 information 한계인지가 핵심 질문.

### 3.3 본 결과가 시사하는 *재진단*

옛 ROADMAP narrative ("학습된 direction 의 의미성, H5") 가 본 실험으로 *부분적* 으로 입증되었다. 02 의 cos(v_learned, v_mean_diff) = 0.32 는 두 direction 이 다르지만 *같은 informed subspace 안* 의 두 점일 가능성. 06 의 K=2 router 가 학습한 두 direction 도 *같은 informed subspace 안의 두 점* (cos = 0.55) 이라는 해석 가능.

즉 **single direction subspace 의 ceiling 은 *information-bearing direction subspace* 안의 representational limit** — random 으로는 도달조차 못 함. 학습된 / 비학습된 informed direction 들은 *같은 information-bearing subspace 의 다른 element* 이고 *redundant* 하게 같은 ceiling 에 도달.

## 4. ROADMAP conditional graph 의 분기 결정

ROADMAP §"Conditional execution graph" 에서:

| Outcome | 분기 |
|---|---|
| 07 pass (direction-agnostic 확정) | Stage 2 bilinear M 본격 + 옛 deferred 완전 강등 |
| **07 fail (direction matters)** | **현 결과 — 옛 deferred (mean_diff_pca / projection_out) 일부 회복 + Stage 2 bilinear M 도 critical** |

→ Stage 2 (bilinear M) 의 critical 여부는 *유지* — translation family 의 algebraic ceiling 자체의 검정이 별도 의의. 단 옛 deferred 의 *direction variant* (PCA direction, projection-out) 가 *informed direction subspace* 의 다른 element 로 같은 ceiling 에 닿는지의 confirmatory ablation 으로 가치 회복.

## 5. Paper narrative 의 *재정렬*

원래 piece 의 narrative 는 *너무 강한* 형태 ("translation 은 ranking 정보 못 담는다") 였음. 본 실험 후 *약간 weaken* 한 형태:

> Translation family 의 ceiling 은 *informed direction* (HN–pos 차이의 정보 함량) 의 정보량에 의해 결정되며, *random direction* 은 그 ceiling 에 도달조차 못 한다 (본 실험). 본 paper 의 핵심 검정은 *Translation family 의 algebraic ceiling 자체가 정보 한계인가 form 한계인가* 의 분리 — *form 변경 (bilinear M)* 이 같은 ceiling 위로 갈 수 있는지의 결정적 falsification.

## 6. Artifact 위치

```
outputs/07_random_direction_scaled/scifact/seed_42/
├── config / env.json
├── v_random.pt — 생성된 random vector (재현 가능)
├── runs / runs_scored / metrics_per_query / metrics_aggregate.json
└── delta_vs_{baseline, mean_diff_alpha10}.json

report/figures/07_random_direction_scaled/{direction_compare, delta_ci_forest, ecdf_compare}.{pdf,png}
```

## 7. 다음 실험

ROADMAP 의 conditional graph 의 *partial fail* 분기로 진행:
- **08 bilinear M minimal** (Stage 2 의 critical 검정 — algebraic 한계 vs 정보 한계 분리)
- 옛 deferred 의 mean_diff_pca / projection_out 일부 회복 (confirmatory)
