# Exp 17 — Negative-side Anchor

## Purpose

Resolve the **asymmetry** in §4.4 anchor: per-token cosine anchor currently constrains query tokens + positive-doc tokens, but **leaves mined hard-negative doc tokens unconstrained**. This experiment adds the missing d⁻ anchor term to test whether the +0.030 lightweight SOTA reflects a mechanistic ceiling or is *fragile to the asymmetry*.

## Hypothesis

Three pre-registered branches (see `report/_exp17_negative_anchor_pre_commit.md`):
- **(a) Positive leap** Δ all ≥ +0.040 (3/3 seed CI > 0): symmetric anchor breaks the ceiling.
- **(b) Tied / saturated** +0.025 ≤ Δ all ≤ +0.040: implicit d⁻ constraint via shared LoRA was already sufficient — direct evidence of anchor family saturation.
- **(c) Over-restriction** Δ all ≤ +0.020: d⁻ anchor harms hard-query learning.

## Loss

Three per-token cosine anchor terms, each unit-scale (per-doc token mean → easy query mean):

$$\mathcal{R}_{\text{abs}}^{z}(\theta) = \mathbb{E}_{x \in \mathcal{Q}_{\text{easy}}}\!\Big[\,\tfrac{1}{|T_z|}\!\sum_{t}\!\big(1 - \cos(\hat h_t^{\text{LoRA}}(z),\, \hat h_t^{\text{frozen}}(z))\big)\,\Big], \quad z \in \{q,\, d^+,\, d^-\}$$

Total loss:

$$\mathcal{L}^\dagger(\theta) = \mathcal{L}_{\text{margin}}(\mathcal{Q}_{\text{hard}}) + \lambda_{\text{dir}}\!\cdot\!\big(\mathcal{R}_{\text{abs}}^{q} + \mathcal{R}_{\text{abs}}^{d^+}\big) + \lambda_{\text{neg}}\!\cdot\!\mathcal{R}_{\text{abs}}^{d^-}$$

§4.4 의 *실제 구현* 은 앞 두 항만 가짐 ($\mathcal{R}_{\text{abs}}^{d^-}$ 누락). 본 실험은 그 마지막 항을 *symmetric* 으로 추가.

## Config (pre-committed single value, no sweep)

| Item | Value |
|---|---|
| λ_dir (q + d⁺) | 1.0 (Exp 13 default) |
| λ_neg (d⁻) | 1.0 (symmetric extension) |
| LoRA | q, v, r=8, α=r |
| LR | 5e-5, batch=32, epochs=3 |
| Seeds | {42, 1337, 2024} on SciFact |
| Tag | `qv_r8_l12_dir1_neg1` |

## Status

Pre-commit complete. Engineering in progress.

## Run

```bash
.venv/bin/python experiments/17_negative_side_anchor/run.py --seed 42
.venv/bin/python experiments/17_negative_side_anchor/run.py --seed 1337
.venv/bin/python experiments/17_negative_side_anchor/run.py --seed 2024
```
