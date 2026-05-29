# 15a_confused_only_baseline — Exp 15 의 training-side ceiling diagnostic

> **본 실험은 *진단용 baseline*** — Exp 15 (Conditional LoRA) 의 full design 진입 *전*, *trivial baseline* 으로 *perfect routing* 의 *training-side* upper bound 측정.

## Motivation

Diagnostic chain:
- (α) score-margin AUC = **0.836** (router signal 강함) → 확인 ✓
- (γ) oracle test-time conditional = **Δ all +0.0475 ± 0.008** (perfect routing ceiling) → 확인 ✓
- **(β) confused-only training**: *학습 시점* 부터 confused-query triplet 만 사용 (perfect routing 의 *training-side realization*) → 본 실험.

(β) 가 *Exp 15 의 trivial baseline*: 만약 confused-only training 만으로 oracle ceiling (+0.048) 에 도달하면, *elaborate routing 구조 없이 단순 triplet filtering* 이 frontier 돌파.

## Config (single point, no sweep, diagnostic only)

Phase 2b 의 모든 setting 동일하되 triplet filter 추가:

| Item | Value |
|---|---|
| LoRA | q, v r=8 α=r (Phase 2b 동일) |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early_stop=val_all |
| Dataset | SciFact |
| Seeds | 42 (single point, ceiling 측정용) |
| **Triplet filter** | **confused-only** (frozen top-1 ≠ rel 인 train queries 의 triplet 만 keep) |
| Tag | `qv_r8_l12_confonly` |

Expected triplet count: 전체 9190 의 ~46 % ≈ 4200.

## 가능 결과 → 함의

| 결과 | 함의 |
|---|---|
| Δ all > +0.045 strict | *Perfect routing 의 training-side ceiling 도달* → Exp 15 의 elaborate gate 필요성 의문, *trivial triplet filtering* 만으로 frontier 돌파 → 6-lever framework 의 *7th lever* 가능 |
| Δ all ≈ +0.025-0.040 | *Anchor-side family 수준 lift* → routing 의 marginal benefit 미미 → Exp 15 비용 vs 가치 의문 |
| Δ all < +0.020 | *Capacity / sample-size bottleneck* — confused triplet 4K 만으로 BA 충분 학습 불가 → §5e *9K triplet bottleneck* 가설 강화 + Exp 15 doomed |

## Usage

```bash
.venv/bin/python experiments/15a_confused_only_baseline/run.py \
    --dataset scifact --seed 42 \
    --r 8 --alpha 8.0 --lora-lr 5e-5
```

## STOP rule

Single seed × single config — *diagnostic only*. 결과 기반 Exp 15 design 진행 / future work 결정.
