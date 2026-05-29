# 16_multilayer_anchor — *Multi-layer per-token cosine anchor* (Exp 13 의 5-layer extension)

> **Theory-driven new methodology** — pre-committed *result-blind* (`report/_exp16_pre_commit.md`).

## Motivation

Exp 13 (per-token cosine anchor on final ColBERT output, λ_dir=1.0) 는 *anchor-side family 의 best lever* (Δ all +0.030, 3/3 strict). Diagnostic B (cos=0.824, tok eff_rank 9.01, doc eff_rank 2.33) 는 *3-fold mechanism evidence* 발견 — 단 anchor 가 *single layer (final output)* 에서만 작동, *intermediate BERT layers* 미활용.

**CLAUDE.md §1.3 prior diagnostic finding**: *"layer-wise confusion signal at [0, 3, 6, 9, 12]"*. **§3.8 ablation completeness 가 명시적으로 본 ablation 요구**.

## Loss

$$\mathcal{L} = \mathcal{L}_{\text{margin}}(\text{conf}) + \lambda_{\text{dir}} \cdot \frac{1}{|L|}\sum_{\ell \in L}\mathcal{R}_{\text{dir}}^{(\ell)}(\text{easy})$$

where $L = \{0, 3, 6, 9, 12\}$ (BERT layer indices) and $\mathcal{R}_{\text{dir}}^{(\ell)}$ = mean per-token $(1 - \cos)$ at layer $\ell$ between LoRA-pass and frozen-pass BERT hidden states (768-dim).

## Config (single pre-commit)

- **Layer set**: {0, 3, 6, 9, 12} (CLAUDE.md §1.3, no sweep)
- **λ_dir = 1.0** (Exp 13 동일 scale, no sweep)
- LoRA q, v r=8, α=r (Phase 2b 동일)
- LR=5e-5, batch=32, ep=3, patience=2, early-stop=val_all
- SciFact, seeds {42, 1337, 2024}
- Tag: `qv_r8_l12_dir1_multilayer`

## 3 branches (pre-commit, result-blind)

- **(a) Multi-layer 우월**: Δ all > +0.040 strict + Δ easy > −0.015 → frontier 외부 도달 (paper main contribution +1)
- **(b) Exp 13 과 frontier 공유**: Δ all ≈ +0.030 ± 0.005 → layer-invariance (frontier robustness 추가 입증)
- **(c) Over-restriction**: Δ all < +0.020 → 5-layer cumulative restraint 가 confused signal 죽임

세 outcome 모두 paper-grade.

## Usage

```bash
.venv/bin/python experiments/16_multilayer_anchor/run.py \
  --dataset scifact --seed {42|1337|2024} --lambda-dir 1.0 \
  --r 8 --alpha 8.0 --lora-lr 5e-5 --max-triplets 9190
```

## STOP rule

3 seeds 완료 후 결과 무관 STOP. λ_dir / layer set sweep / variant *전부 금지*.
