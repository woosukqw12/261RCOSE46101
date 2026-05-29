# 10_lora_phi — LoRA on Φ (ColBERT encoder), 50K budget 안 *encoder representational limit* 회복

## 목적

Robustness audit (2026-05-23) 의 02 unfrozen 실험에서 ColBERT *encoder 전체* unfreeze (110M params) 가 **Δ confused +0.252** 의 5× lift 확인 → frozen encoder 가 *진짜 bottleneck*. 본 실험은 **CLAUDE.md §3.2 의 50K param budget** 안에서 LoRA (Low-Rank Adaptation, Hu et al. 2021) adapter 로 그 lift 의 *얼마나* 회복 가능한지 정밀 검정. *Paper main contribution candidate*.

## LoRA 형식

각 attention 의 학습 가능 선형 변환 (q, k, v, o) 에 rank-r additive adapter:
$$h = W x + (\alpha / r) B A x, \quad A \in \mathbb{R}^{r \times d}, \ B \in \mathbb{R}^{d \times r}$$

Init: $A \sim \mathcal{N}(0, \sigma^2)$, $B = \mathbf{0}$ → $BA = 0$ at $t=0$ → *정확히 baseline retrieval*. (zero-init pathology 회피 — $B=0$ 이지만 $A \neq 0$ 이라 $\partial L / \partial A \propto B^T \neq 0$ 처음엔 0 이나 step 1 후 $B$ 가 업데이트되면서 self-bootstrap.)

학습 파라미터: $2 r d \times |\text{components}| \times |\text{layers}|$. ColBERT BERT-base ($d=768$, 12 layers):

| Components | Layers | r | Params | 50K budget? |
|---|---|---|---|---|
| q, v | all 12 | **1** | **36,864** | ✓ (Phase 1 default) |
| q, v | all 12 | 2 | 73,728 | ✗ (over) |
| q, v | last 6 | 4 | 73,728 | ✗ (over) |
| q, v | last 6 | 2 | 36,864 | ✓ |
| q, k, v, o | last 3 | 2 | 36,864 | ✓ |
| q only | all 12 | 2 | 36,864 | ✓ |

## Phase 별 실험 계획

### Phase 1 (확신 확보, ~17 분)

**Minimal config**: `q,v` × all 12 layers × r=1 = 36,864 params. SciFact seed 42, 3 epochs, encoder LR=5e-5.

- 통과 기준: Δ confused vs baseline > +0.10 (02 unfrozen 의 +0.252 의 40%) + all-slice 보존 (CI 하한 ≥ -0.005)
- 통과 시 → Phase 2 (sweep)
- 미통과 시 → 다른 LoRA design 탐색 (FFN adapter, 다른 component 분배)

### Phase 2 (paper-grade ablation, Phase 1 통과 후 ~3-4 시간)

| Sweep | 옵션 | 검정 |
|---|---|---|
| (a) rank | r ∈ {1, 2, 4} on q,v 분포로 budget 조정 | rank vs lift 의 trade-off |
| (b) component | (q,v) vs (q,k,v,o) vs FFN only, fixed budget | *어떤* 부분이 critical |
| (c) layer subset | all 12 vs last 6 vs prior diagnostic [0,3,6,9,12] | layer 분배 |
| (d) seed | × 3 on best config | 08 의 seed artifact 교훈 |
| (e) dataset | NFCorpus 1 run on best config | 06 의 cross-dataset 교훈 |

## 가설

| H | 기준 | 의미 |
|---|---|---|
| **H18a** | Phase 1 의 NDCG@10 confused 가 baseline +0.10 ↑ | LoRA 가 frozen-encoder limit 의 의미 있는 부분 회복 |
| H18b | Phase 1 의 all-slice CI 하한 ≥ -0.005 | anchor preservation |
| **H18c** | (best Phase 2) Δ confused 가 02 unfrozen 의 +0.252 의 *적어도 50%* 회복 (≥ +0.126) | 50 K LoRA budget 의 *practical* upper bound. paper main contribution. |
| H18d | LoRA component subset 의 *attention 만* > FFN only > full attention+FFN 의 관계 검정 | layer 분배 lever 의 paper-grade ablation |

## 학습 design

| 항목 | 값 |
|---|---|
| Encoder | ColBERT BERT-base, base 가중치 frozen, LoRA params 만 학습 |
| LoRA components | Phase 1: q, v (Phase 2 sweep) |
| LoRA rank | Phase 1: 1 (Phase 2 sweep) |
| LoRA layers | Phase 1: all 12 (Phase 2 sweep) |
| LoRA LR | 5e-5 (typical BERT finetune) |
| Steering | frozen v=0 (no-op hook at layer 12, train_steering 호환 위함) |
| Loss | pairwise margin $m=0.2$ |
| Optimizer | AdamW (LoRA params only) |
| Batch / Epochs / Patience | 32 / 3 / 2 |
| Dataset | SciFact (Phase 1) → + NFCorpus (Phase 2) |

## 실행

```bash
# Phase 1: minimal q,v r=1 all layers
.venv/bin/python experiments/10_lora_phi/run.py --dataset scifact --seed 42 \
    --components q,v --r 1

# Phase 2 (예시): q,k,v,o r=1 on last 6 layers
.venv/bin/python experiments/10_lora_phi/run.py --dataset scifact --seed 42 \
    --components q,k,v,o --r 1 --layers 6,7,8,9,10,11
```

Artifact: `outputs/10_lora_phi/{ds}/seed_{seed}/{components}_r{r}_l{n_layers}/...`

## 비판적 review

1. **LoRA 의 *실제* effective capacity 검정 부재**: r=1 의 ΔW = B A 가 rank ≤ 1 의 update — 큰 lift 어려울 가능성. r=2 (q,v on 6 layers) 의 비교 필수.
2. **Encoder LR=5e-5 의 hyperparameter sensitivity**: NFCorpus K=2 의 LR=1e-3 catastrophic 교훈. LoRA 의 적정 LR 도 dataset-dependent 가능성. NFCorpus 진행 시 LR sweep.
3. **B=0 init 의 학습 초기 dynamics**: A 는 학습 시작 시 random, B=0 → forward output unchanged but $\partial L/\partial B \neq 0$ → step 1 후 B 업데이트, A 와 동시 학습. step 1 의 *방향* 이 학습 trajectory 결정 — seed-dependent 가능성 (08 의 seed artifact 와 유사 위험).
4. **Anchor preservation 강조**: 02 unfrozen 도 all-slice 0.6576 (≈ baseline) 보존 — LoRA 가 그 패턴 유지하는지 critical. 보존 못 하면 method 가치 부재.

## 상세 보고서

[`report/10_lora_phi_report.md`](../../report/10_lora_phi_report.md) — 실행 후 작성.
