# 08_bilinear_M_minimal — Bilinear interaction metric M = I + UV^T (Stage 2 critical)

## 목적

ROADMAP §"Stage 2" 의 **main novelty 의 *critical falsification***. 실험 00–07 종합 결과 *translation family ceiling 0.665* 가 확정 (K-sweep 결과 K=2/K=4 의 NDCG@10 all = 0.6614 의 *문자 그대로 동일*; 07 의 random direction 은 baseline 동등). 본 실험은 *form 자체* 의 변경 — *MaxSim 의 inner product 일반화* — 만이 ceiling 위로 갈 수 있는지 검정.

## 수식

vanilla ColBERT MaxSim:
$$s(q, d) = \sum_i \max_j \langle q_i, d_j \rangle$$

본 실험의 bilinear M correction:
$$s_M(q, d) = \sum_i \max_j q_i^\top M d_j, \quad M = I + U V^\top$$

$$M \in \mathbb{R}^{D \times D}, \quad U, V \in \mathbb{R}^{D \times r}, \quad D=128, \quad r=8$$

전개:
$$q_i^\top M d_j = \langle q_i, d_j \rangle + (U^\top q_i)^\top (V^\top d_j)$$

두 번째 항이 **q 와 d 의 *cross-feature* 의 곱셈적 결합** — translation family 가 *절대* 못 표현하는 q–d 상호작용 차원. *Algebraic family change*.

## 학습 가능 파라미터

| r | $2 D r$ | 50K budget |
|---|---|---|
| 4 | 1,024 | ✓ |
| 8 | **2,048** | ✓ (본 실험) |
| 16 | 4,096 | ✓ |
| 32 | 8,192 | ✓ |
| 64 | 16,384 | ✓ |

r=8 의 2,048 params 가 02 의 768 (single direction) 보다 ~3 배, 05 의 5-layer 3,840 보다 약간 작음. 적정 minimal capacity.

## 가설

| H | 기준 | 의미 |
|---|---|---|
| **H08a** | 08 의 NDCG@10 all 의 paired bootstrap CI 하한 > K=2/K=4 ceiling (0.6614) | **Stage 2 critical pass** — translation family ceiling 의 *form 변경 우회* 확정 |
| H08b | 08 의 all-slice Δ vs baseline CI 하한 ≥ -0.005 | anchor preservation (K=8 의 over-correction 회피) |
| H08c | 08 의 confused-slice Δ vs 02/06 K=2 CI 하한 > 0 | translation family 의 *informed direction subspace ceiling* 위로 |
| H08d | M = I + UV^T 의 singular value spectrum 의 *non-trivial* 분포 | 학습이 *form 자체* 의미 있는 변경 학습 |

만약 H08a 미통과 ⇒ form 변경도 ceiling 못 넘음 ⇒ ceiling 은 *frozen-encoder 의 representational limit* (information 한계) ⇒ 18 LoRA on Φ (encoder finetune) 의 critical 검정으로 직행.

## 학습 design

| 항목 | 값 |
|---|---|
| Hidden state hook | 없음 (frozen ColBERT 의 vanilla 출력 사용) |
| Loss | pairwise margin $m=0.2$ |
| Optimizer | AdamW (LR=$10^{-3}$, WD=$10^{-4}$) |
| λ_anc | 0 |
| Init | $U, V \sim \mathcal{N}(0, 10^{-4})$ (small_random) — *zero init 의 gradient 0 pathology 회피* (아래 5번 참조) |
| Batch / Epochs / Patience | 32 / 5 / 2 |
| r | 8 (sweep 은 10_bilinear_rank_sweep) |
| Dataset | SciFact (seed 42) |

## 실행

```bash
.venv/bin/python experiments/08_bilinear_M_minimal/run.py --dataset scifact --seed 42 --r 8
```

Artifact:
```
outputs/08_bilinear_M_minimal/scifact/seed_42/r_8/
├── config / env / train_config / module_final.pt / train_history.json
├── M_stats.json (‖U‖, ‖V‖, ‖UV^T‖_F, SVD spectrum, condition number)
├── runs / runs_scored / metrics_per_query / metrics_aggregate.json
└── delta_vs_{baseline, mean_diff_alpha10, 02_learned, 06_k_sweep_k2, 06_k_sweep_k4}.json
```

## 비판적 review

1. **r=8 이 충분 capacity 인지 검정 부족**: 본 실험은 r=8 단독; r ∈ {4, 16, 32, 64} 의 추가 sweep 으로 trade-off (10_bilinear_rank_sweep) 가능.
2. **MaxSim 의 마지막 layer 이후 적용이라 BERT 의 inner representation 의 *모든 표현 정보* 활용 가능성 의문**: 단, ColBERT v2 는 frozen + 768→128 projection 도 frozen 이라 정보가 *완전* 보존. M 은 그 정보를 *어떻게 결합* 할지의 lever.
3. **E5 distillation 없이 pairwise margin loss 만**: 09 에서 E5 distillation 추가로 보조; 본 실험은 *pure form-change 의 lever* 검정.
4. **Train-overfitting 위험 (모든 06 학습 실험에서 발견된 패턴)**: 02-06 보다 capacity 가 절대치로 비슷한 수준 (2K params) 이라 overfitting risk 도 비슷. val curve monitoring 필수.
5. **새 retrieval 의 inference cost**: bilinear MaxSim 의 $(U^\top q_i)^\top (V^\top d_j)$ 항 추가가 $O(r/D)$ 의 marginal cost. r=8 / D=128 → +6 % computation. 실용적 부담 없음.
6. **Zero-init pathology**: $U = V = \mathbf{0}$ init 에서는 $\partial \mathcal{L}/\partial U \propto V = 0$ 이고 $\partial \mathcal{L}/\partial V \propto U = 0$ — *두 gradient 모두 정확히 0* 이라 학습 불능 (smoke test 로 확인). 해결책: small_random init ($\sigma = 10^{-2}$, ‖U‖ ≈ ‖V‖ ≈ 0.32, ‖UV^T‖_F ≈ 0.035 → vanilla MaxSim 대비 < 0.1 % relative deviation). 본 시점의 anchor preservation 은 *quasi-zero* 수준 — practical 으로 동등.

## 상세 보고서

[`report/08_bilinear_M_minimal_report.md`](../../report/08_bilinear_M_minimal_report.md) — 실행 후 작성.
