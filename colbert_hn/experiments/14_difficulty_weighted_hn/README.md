# 14_difficulty_weighted_hn — *Continuous control* of HN difficulty (M1b/Phase 2b spectrum)

> **Theory-driven new methodology** — pre-committed *result-blind* (`report/_exp13_14_pre_commit.md`).

## Motivation

§5f.3 sole sufficient mechanism = *hard-contrast over-correction* (Exp 12 confirmed).

Existing levers — *binary*:
- Phase 2b: hard 100 % → catastrophic redistribution
- M1b: hard 0 % (easy in-batch) → strict net+ but confused 절반
- Exp 12: binary FN-filter → ineffective on redistribution

**Exp 14** = *continuous control* — sigmoid weighting of triplet difficulty.

## Loss

$$w_i = \sigma(\alpha_w \cdot \text{e5\_margin}_i), \quad \mathcal{L} = \frac{\sum_i w_i \cdot \max(0, m - s_i^+ + s_i^-)}{\sum_i w_i}$$

where `e5_margin_i = cos(eq, epos) − cos(eq, ehn)` using E5-Mistral-7B cached embeddings.

- e5_margin > +0.3 (easy): weight ≈ 1
- e5_margin ≈ 0 (borderline): weight ≈ 0.5
- e5_margin < −0.3 (FN): weight ≈ 0

## Config (single pre-commit)

- **α_w = 10** (single value, no sweep)
- Mined HN (Phase 2b 동일, *no removal — just weighting*)
- LoRA q, v r=8, α=r (Phase 2b 동일)
- LR=5e-5, batch=32, ep=3, patience=2, early-stop=val_all
- SciFact, seeds {42, 1337, 2024}
- Tag: `qv_r8_l12_diffw10`

## 3 branches

- **(a) Sweet spot**: Δ all > +0.025 strict 3/3 + Δ confused > +0.08 + Δ easy > −0.04 → *continuous > binary*
- **(b) ≈ M1b 또는 Exp 11 λ=1**: spectrum endpoints 만 differentiable, sweet spot 없음
- **(c) Worse**: continuous weighting 이 *learning signal 약화* → α=10 transition 가 모든 triplet weight 흐리게

## Usage

```bash
.venv/bin/python experiments/14_difficulty_weighted_hn/run.py \
  --dataset scifact --seed {42|1337|2024} --alpha-w 10.0 \
  --r 8 --alpha 8.0 --lora-lr 5e-5 --max-triplets 9190
```

## STOP rule

3 seeds 완료 후 결과 무관 STOP. α_w sweep / cross-dataset / variant *전부 금지*.
