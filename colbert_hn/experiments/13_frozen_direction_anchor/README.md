# 13_frozen_direction_anchor — *Direction-side preservation* (Exp 11 의 direction analog)

> **Theory-driven new methodology** — pre-committed *result-blind* (`report/_exp13_14_pre_commit.md`).

## Motivation

NFCorpus M1+M1b *direction matters* puzzle (§7.3.f.ii): eff_rank doc 1.05 ≈ 1.06 (no magnitude change) 임에도 NDCG@10 0.0094 → 0.246 (74 % gap recovery). **Direction alignment > magnitude** 의 직접 증거.

Exp 11 (relational self-sim) = *magnitude-side* preservation (token×token Sim Frobenius²).
**Exp 13** = *direction-side complement* — per-token cosine deviation penalty.

## Loss

For easy queries $x \in E$ (baseline top-1 = relevant):

$$\mathcal{L} = \mathcal{L}_{\text{margin}}(\text{confused}) + \lambda_{\text{dir}} \cdot \frac{1}{|E|}\sum_{x\in E}\Big[\text{cos\_dev}(H_q^x) + \text{cos\_dev}(H_d^x)\Big]$$

where $\text{cos\_dev}(H) = \frac{1}{T}\sum_t (1 - \cos(h_t^{\text{LoRA}}, h_t^{\text{frozen}}))$.

## Config (single pre-commit)

- **λ_dir = 1.0** (single value, no sweep)
- LoRA q, v r=8, α=r (Phase 2b 동일)
- LR=5e-5, batch=32, ep=3, patience=2, early-stop=val_all
- SciFact, seeds {42, 1337, 2024}
- Tag: `qv_r8_l12_dir1`

## 3 branches

- **(a) Direction lever works**: Δ all > +0.025 + Δ easy ≈ 0 → paper-grade *5th lever*
- **(b) ≈ Exp 11**: Δ all ≈ +0.029 → direction == magnitude preservation equivalent
- **(c) Worse**: Δ all < +0.020 → direction over-restrictive

## Usage

```bash
.venv/bin/python experiments/13_frozen_direction_anchor/run.py \
  --dataset scifact --seed {42|1337|2024} --lambda-dir 1.0 \
  --r 8 --alpha 8.0 --lora-lr 5e-5 --max-triplets 9190
```

## STOP rule

3 seeds 완료 후 결과 무관 STOP. λ sweep / cross-dataset / variant *전부 금지*.
