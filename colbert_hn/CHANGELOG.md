# CHANGELOG.md — colbert_layer_steering

본 repo 의 모든 의미있는 변경 (code / config / design / artifact / 문서) 을 *날짜순 역순* 으로 기록한다. CLAUDE.md §3.9 의 *documentation freshness 철칙* 에 따라 stale 상태를 허용하지 않는다.

형식: Keep-a-Changelog 변형.
- `## [YYYY-MM-DD]` 헤더 (필요 시 같은 날 여러 entry 가능, `[YYYY-MM-DD#n]`)
- 분류: `Added` / `Changed` / `Removed` / `Fixed` / `Experimental`
- 한 entry 는 한 줄 이상; 참조하는 파일 / 섹션 / config ID 를 명시
- 변경 사유 (Why) 는 가능한 한 함께 기록

---

## [Unreleased]

(작업 중인 변경은 여기에 누적, release / tag 시 아래 dated section 으로 이동)

---

## [2026-05-25#2] — Exp 16 (multi-layer per-token anchor) + Spine ablations — *anchor scope ablation + 6-lever 정정*

### Added

**Exp 16 (anchor scope ablation, branch (c) over-restriction)**:
- `experiments/16_multilayer_anchor/{run.py, README.md, figures.py}` — multi-layer per-token cosine anchor at BERT layers {0,3,6,9,12} (768-dim hidden states). `LayerCapture` hook manager + per-doc frozen cache float16 on CPU.
- `report/_exp16_pre_commit.md` — pre-commit doc (BEFORE training, single config).
- `report/_repr_collapse_exp16.py` — Diagnostic B on Exp 16 (per-doc multi-layer capture, ~5 min CPU).
- `report/16_multilayer_anchor_report.md` — 6-section 상세 보고서 (motivation, 3-seed grid, Diagnostic B, 5 figures embed, 종합).
- 새 artifacts: `outputs/16_multilayer_anchor/scifact/seed_{42,1337,2024}/qv_r8_l12_dir1_multilayer/`.
- 새 figures: `report/figures/16_multilayer_anchor/*.{pdf,png}` (5 figures × 2 formats), `report/figures/_repr_collapse_exp16/repr_collapse_exp16.{pdf,png}`.

**Spine ablations (reviewer Tier 1 + B1 + C1)**:
- `report/_spine_ablations.py` — A1 M1b Δ easy 3-seed 실측 / A2 Anchor incremental Δ over Phase 2b LoRA / B1 Exp 13 NDCG sanity / C1 split consistency. Measurement-only, ~10 s.
- 새 artifact: `report/figures/_spine_ablations/spine_ablations.json`.

### Changed

- `REPORT.md` §6.1 grid — Exp 16 4 rows 추가 (3 seeds + 3-seed mean) + **M1b row Δ easy 정정** (이전 "(~−0.05)" → 실측 "−0.017 ± 0.003").
- `REPORT.md` §7.3.i 다음에 §7.3.j 신설 — *Exp 16 anchor scope ablation, branch (c) over-restriction confirmed*. 3-seed grid + Diagnostic B 3-fold mechanism evidence + 6-lever framework no change.
- `REPORT.md` §7.3.k 신설 — *Spine ablations (reviewer Tier 1 + B1 + C1)*. 4-fold finding (A1 M1b 정정 / A2 anchor incremental over Phase 2b / B1 sanity / C1 split).
- `REPORT.md` §7.4.1 — Exp 16 row 추가 (anchor-side multi-layer inferior 구성원) + **M1b Δ easy 정정** + **anchor incremental interpretation paragraph** (anchor 의 sole contribution = easy preservation, NO confused gain).
- `RESEARCH.md` — 새 dated entry `2026-05-25#2`.
- `ROADMAP.md` — Exp 16 + spine ablations entries 추가, queue 종착 갱신.

### Experimental — Exp 16 result-blind pre-committed (3 seeds, single config)

- **Branch (c) "Multi-layer over-restriction" 확정** — 3-seed mean Δ all = **+0.004 ± 0.006** (3/3 CI 0 포함, NOT strict), Δ confused = +0.071 ± 0.004 ✓, Δ easy = −0.052 ± 0.008 ✗. 모든 metric 에서 Exp 13 대비 *명백 열등* (Δ all 1/8, Δ easy damage 2.5×).
- **Diagnostic B mechanism direct evidence** — Per-layer cos(LoRA, frozen): L0=1.0, L3=0.998, L6=0.991, L9=0.965, **L12=0.697** (Exp 13의 0.824 보다 더 멀어짐). L12 token eff_rank = 4.6 (Exp 13의 1/2, frozen의 11 %). **Loss budget dilution + intermediate-layer redundancy** mechanism — *anchor-side family 의 optimal scope = final layer only*.
- **CLAUDE.md §1.3 prior diagnostic finding 재해석** — "signal exists at 5 layers" ≠ "intervention should target all 5".
- **STOP rule 준수** — layer scope sweep / variant *전부 금지*.

### Experimental — Spine ablations (measurement-only)

- **A1**: M1b Δ easy 3-seed mean = **−0.017 ± 0.003** (이전 추정 "(~−0.05)" 의 1/3). M1b 가 anchor-side family 와 *동등 수준* easy 보존. 6-lever 표 정정.
- **A2**: Anchor incremental Δ over Phase 2b LoRA (same seed pairs) — Exp 11/13 모두 Δ all 증분이 *전적으로 Δ easy 보존* (+0.055/+0.064)에서 옴, **Δ confused 에서는 micro-loss** (−0.002/−0.012). Paper-level reframing: anchor = *easy preservation mechanism*, NOT *confused recovery*.
- **B1**: Exp 13 runs.json → NDCG@10 reproduction 일치 (3/3 diff < 0.0001) ✓.
- **C1**: 3 seeds 모두 train confused=368, easy=441 (frozen retrieval deterministic) ✓.

### Why

- §3.8 ablation completeness 가 *명시적으로* anchor scope ablation 요구 (single-layer vs multi-layer).
- CLAUDE.md §1.3 prior diagnostic finding 의 *direct architectural translation* (5-layer anchor) 검정.
- Reviewer recommendation 의 *spine research narrative* 보강 (M1b 추정 → 실측, anchor mechanism interpretation 정정).
- 6-lever framework 의 *strict integrity* + *paper-level interpretation accuracy* 확보.

---

## [2026-05-25] — Exp 15 (Conditional LoRA) 4-diagnostic chain — *frontier-breaking hypothesis 의 empirical falsification*

### Added

- `report/_exp15_diagnostics.py` — (α) score-margin AUC + (γ) oracle test-time conditional NDCG diagnostic script (no training, ~30 s).
- `report/_exp15_diagnostic_delta.py` — (δ) margin-routed Phase 2b τ-sweep diagnostic (no training, ~10 s).
- `experiments/15a_confused_only_baseline/{run.py, README.md}` — (β) confused-only triplet training (Phase 2b LoRA + triplet filter). Single seed 42, ~10 min.
- `report/15_exp15_diagnostics_report.md` — 4-diagnostic 종합 보고서 (theoretical motivation, sequential chain, mechanism analysis, future work).
- 새 figure: `report/figures/_exp15_diagnostics/diagnostic_alpha_gamma.{pdf,png}` (3-panel: margin distribution / oracle vs Phase 2b / per-seed oracle scatter).
- 새 figure: `report/figures/_exp15_diagnostics/diagnostic_delta.{pdf,png}` (3-panel: τ-sensitivity / confusion matrix / 4-diagnostic summary).
- 새 artifact: `outputs/15a_confused_only_baseline/scifact/seed_42/qv_r8_l12_confonly/` — (β) catastrophic LoRA checkpoint + metrics.

### Changed

- `REPORT.md` §7.3.h 다음에 §7.3.i 신설 — *Exp 15 diagnostics: Conditional LoRA frontier-breaking hypothesis 의 empirical falsification*. 4-diagnostic chain 결과 표 + 5 핵심 함의.
- `REPORT.md` §9.3 신설 — *Exp 15 future work — Conditional LoRA 의 elaborate realization (post-paper)*. 3 future work proposal (F1 learned router / F2 end-to-end joint / F3 reranker 형태) + 제약.
- `RESEARCH.md` — 새 dated entry `2026-05-25` 추가 (Done / Observations / Decisions / Open questions / Next).
- `ROADMAP.md` — Exp 15 diagnostic chain row 추가 + queue 종착.

### Experimental — 4-diagnostic chain (sequential, no new architecture)

| Diagnostic | 결과 | 함의 |
|---|---|---|
| (α) Score-margin AUC = **0.836** | router signal 강함 | routing failure 배제 |
| (γ) Oracle Δ all = **+0.048** ✓ | perfect routing ceiling real | frontier 외부 공간 존재 |
| (β) Confused-only Δ all = **−0.387** ✗ | training-time filtering catastrophic | training distribution dependency |
| (δ) Margin-routed Δ all = **+0.011** | realistic < anchor-side | **frontier-breaking falsified** |

### Why

- §7.3.c.iii 의 redistribution 회계 항등식 + §7.3.g Diagnostic B 의 anchor equilibrium (cos 0.824) 의 *누르기* 한계가 conditional LoRA (대수적 *제거*) 의 frontier-breaking 가능성 시사.
- 4-diagnostic chain 으로 *Exp 15 full design 진입 전* sequential falsification — *theoretical promise vs practical limit* empirical 격리.
- (γ) oracle ceiling +0.048 의 *unrealizability* 직접 입증 → **6-lever framework 의 frontier-fixed 주장이 inference-time conditional routing 에도 robust** = paper main contribution 강화.
- (β) catastrophic failure 의 mechanism (training distribution dependency) 가 §9.3 future work 의 *informed framing* 제공.
- STOP rule 준수: 4 diagnostic 완료 후 elaborate Exp 15 미실시.

---

## [2026-05-24#15] — Diagnostic B on Exp 14 checkpoints — *data-side family internal representation* + 6-lever × internal/external mapping 완성

### Added

- `report/_repr_collapse_exp14.py` — Diagnostic B 측정 script for Exp 14 (Exp 13 script mirror, anchor proximity reference metric 포함). CPU, cache resume, ~2 min.
- 새 artifact: `report/figures/_repr_collapse_exp14/repr_collapse_exp14_data.json` — Exp 14 3 seeds + frozen baseline 의 collapse metrics + anchor proximity.
- 새 figure: `report/figures/_repr_collapse_exp14/repr_collapse_exp14.{pdf,png}` — 3-panel (6-lever cross-family tok eff_rank / Exp 14 vs Exp 13 anchor proximity / per-token cos distribution contrast).

### Changed

- `report/14_difficulty_weighted_hn_report.md` — §5 (Diagnostic B sub-experiment) 신설 + §6 (종합) renumbered. *Data-side family internal representation* 의 5-subsection breakdown: 3-seed grid, 6-lever × internal grid, 3-fold paper-grade finding (family-level external/internal alignment, within-family variance pattern, bimodal seed mechanism alignment), paper §7.3.f evidence chain 강화, figure embed.
- `REPORT.md` §7.3.h — *Diagnostic B sub-experiment* paragraph 추가 (6-lever internal grid table + 3-fold findings + evidence chain).
- `RESEARCH.md` — 새 dated entry `2026-05-24#9` 추가.

### Experimental — measurement-only sub-experiment (no new training)

- **Family-level external/internal alignment 확정** — anchor-side ≫ data-side at *every internal metric*. 6-lever 의 3-frontier structure 가 external + internal 모두 일관 → paper main mechanistic finding.
- **Within-family external 동등 ↔ internal variance pattern 분리** — Exp 11 internal variance 8.5×, Exp 14 internal variance 28×. Same-family lever form difference 가 robustness profile 면에서 분리.
- **Bimodal seed pattern of Exp 14** — seed 42/2024 vs seed 1337 의 *internal collapse magnitude ↔ external NDCG redistribution direct correlation*. Paper-grade seed-level mechanism direct evidence.

### Why

- Exp 13 Diagnostic B 결과 (cos(LoRA, frozen) = 0.824) 의 *anchor-side specific* 여부 검정 — data-side family 도 anchor proximity 발생하는가? (결과: 0.539, *54 % 분리*).
- 6-lever framework 의 *internal representation 측면* mapping 완성 → external (Δ NDCG) 와 internal (eff_rank, anchor cos) 의 *multi-level alignment* paper main mechanistic finding 으로 격상.
- Pre-commit STOP rule 무관 — measurement-only on existing checkpoints, no new training.

---

## [2026-05-24#14] — Diagnostic B on Exp 13 checkpoints — *per-token absolute direction anchor* mechanism direct verification

### Added

- `report/_repr_collapse_exp13.py` — Diagnostic B 측정 script (CPU, cache resume, ~3.5 min). 기존 `_repr_collapse_new_ckpts.py` pattern + 추가 metric `cos(h_LoRA, h_frozen)` per token (Exp 13 의 loss 가 직접 규제한 양).
- 새 artifact: `report/figures/_repr_collapse_exp13/repr_collapse_exp13_data.json` — 3 seeds × Exp 13 + frozen baseline 의 collapse metrics + anchor proximity.
- 새 figure: `report/figures/_repr_collapse_exp13/repr_collapse_exp13.{pdf,png}` — 3-panel (anchor proximity bars + token eff_rank comparison + per-token cos distribution).

### Changed

- `report/13_frozen_direction_anchor_report.md` — §5 (Diagnostic B sub-experiment) 신설 + §6 (종합) renumbered. *Mechanism direct verification* 의 5-section breakdown: 결과 표, 3-fold finding (anchor cos = 0.824, token eff_rank 9.01 vs 7.69, doc collapse 잔존), Exp 11 vs Exp 13 internal representation 비교, NFCorpus puzzle evidence chain 완성, figure embed.
- `REPORT.md` §7.3.g — *Diagnostic B sub-experiment* paragraph 추가 (3-fold direct evidence + §7.3.f.ii NFCorpus chain 형성).
- `RESEARCH.md` — 새 dated entry `2026-05-24#8` 추가.

### Experimental — measurement-only sub-experiment (no new training)

- **Anchor cos = 0.824** (3-seed mean) — Exp 13 loss = 1 − cos 의 잔여값 0.176, train_history ep1 anchor_loss 0.18 과 정확 일치. *Soft equilibrium attractor* (confused push ↔ anchor pull). λ_dir=1.0 의 equilibrium-formation 적정값 사후 확인.
- **Token eff_rank 9.01 (Exp 13) > 7.69 (Exp 11)** — anchor-side family 내 internal representation 미세 분리. *NDCG frontier 동등 ↔ internal mechanism 분리* (external behavior vs internal representation dissociation).
- **Doc eff_rank 2.33** (Phase 2b-level) — anchor-side family 가 token granularity 에서만 효과적, doc aggregation 후 anchor 효과 희석. Anchor-side family capacity limit direct evidence.

### Why

- Exp 13 의 mechanism claim ("per-token cosine anchor 가 representation 을 frozen baseline 으로 끌어당김") 의 *loss formulation* 만이 아닌 *empirical verification* 필요.
- §7.3.f.ii NFCorpus direction-matters puzzle 와 §7.3.g (Exp 13) 의 *direct evidence chain* 완성.
- Pre-commit STOP rule 무관 — measurement-only on existing checkpoints, no new training / config sweep.

---

## [2026-05-24#13] — Exp 14 (continuous sigmoid weighting) — *data-side family 의 binary ≈ continuous equivalence* 확정

### Added

- `experiments/14_difficulty_weighted_hn/run.py` — sigmoid weight $w_i = \sigma(\alpha_w \cdot \text{e5\_margin}_i)$ 계산, weighted-mean margin loss 학습. E5-Mistral-7B cached embeddings (Exp 12 와 동일 source).
- `experiments/14_difficulty_weighted_hn/README.md` — 실험 카드 (motivation / loss / pre-commit / branches).
- `experiments/14_difficulty_weighted_hn/figures.py` — 6 paper-grade figures: `delta_ci_forest`, `six_lever_scatter`, `weight_distribution`, `train_curves`, `ndcg_slice_grid`, `family_frontier_overview` (PDF + PNG 양쪽).
- `report/14_difficulty_weighted_hn_report.md` — 본 실험 상세 보고서 (3-seed grid, branch 판정, Exp 12 비교, three-frontier structure, 6-lever framework, figures 임베드).
- 새 artifact: `outputs/14_difficulty_weighted_hn/scifact/seed_{42,1337,2024}/qv_r8_l12_diffw10/`.
- 새 figures: `report/figures/14_difficulty_weighted_hn/*.{pdf,png}` (6 figures × 2 formats).

### Changed

- `REPORT.md` §6.1 grid — Exp 14 의 4 rows 추가 (3 seeds + 3-seed mean ± std).
- `REPORT.md` §7.3.g 다음에 §7.3.h 신설 — Exp 14 의 상세 분석 (가설, loss, grid, branch 판정, Exp 12 비교, weight 분포, three-frontier structure, 4 함의).
- `REPORT.md` §7.4.1 — *5-lever → 6-lever framework* 확장. Three-frontier structure (anchor-side upper / data-side weighting lower / data-side substitution unique) 명시. Family-level discreteness (form 의 수학적 차이 ≠ outcome 의 empirical separation) paper main mechanistic finding.
- `RESEARCH.md` — 새 dated entry `2026-05-24#7` 추가 (Exp 14 의 done / observations / decisions / open questions / next).
- `ROADMAP.md` — Phase 4 (Exp 14) + Phase 5 (queue end) 완료 표시, Changelog row 추가.

### Experimental — Exp 14 result-blind pre-committed (3 seeds, single config α_w=10)

- **Branch (c) 변형 확정** — 3-seed mean Δ all +0.006 ± 0.003 (3/3 CI 0 포함, **NOT strict**), Δ confused +0.085 ± 0.022 ✓, Δ easy −0.060 ± 0.020 ✗. *Softer Phase 2b, sub-binary on Δ all, but Δ confused not attenuated*.
- **Data-side family equivalence** — Exp 12 (binary $\{0,1\}$ FN cut) 와 Exp 14 (continuous $(0,1)$ sigmoid) 가 *statistically equivalent frontier* 점유. *Anchor-side family 의 동일 패턴* (Exp 11 ≈ Exp 13) 과 결합 → **frontier 가 family 별 fixed location**.
- **Three-frontier structure** — anchor-side (upper, Δ all ≈ +0.030) / data-side weighting (lower, Δ all ≈ 0) / data-side substitution (M1b unique). Paper main mechanistic finding.
- **α_w=10 unstable variance** — Δ confused std 0.022 (anchor-side 의 3-5×), val NDCG late-best (ep3, not ep1). *Practitioner-actionable* limit, future work.
- **STOP rule 준수** — α_w sweep / variant / cross-dataset *전부 금지*.

### Why

- §5f.3 sole sufficient mechanism (hard-contrast over-correction) 의 *continuous control* 가설 — Phase 2b (100 %) 와 M1b (0 %) 의 binary endpoints 사이 sweet spot 존재 여부 검정.
- Exp 12 (binary cut $\{0,1\}$) 와 Exp 14 (continuous $(0,1)$) 의 *equivalent lever* 가설 (form vs outcome separation).
- 결과: 두 weighting 이 같은 frontier 점유 → *data-side family 의 frontier 강건성* paper-grade negative finding + 6-lever framework 완성.

---

## [2026-05-24#12] — Exp 13 (per-token cosine direction anchor) — *anchor-side family 의 frontier 강건성* 확정

### Added

- `experiments/13_frozen_direction_anchor/run.py` — frozen ColBERT 의 easy queries 의 q + pos doc embeddings precompute, LoRA forward 마다 per-token cosine deviation 계산 + easy loss = λ_dir · mean(1 − cos).
- `experiments/13_frozen_direction_anchor/README.md` — 실험 카드 (Exp 13 의 motivation / loss / pre-commit / branches).
- `experiments/13_frozen_direction_anchor/figures.py` — 5 paper-grade figures: `delta_ci_forest`, `anchor_family_scatter`, `train_curves`, `lora_AB_norms`, `ndcg_slice_grid` (PDF + PNG 양쪽).
- `report/13_frozen_direction_anchor_report.md` — 본 실험 상세 보고서 (3-seed grid, branch 판정, Exp 11 비교, 함의, figures 임베드).
- `report/_exp13_14_pre_commit.md` — pre-commit doc (BEFORE training, λ_dir=1.0 + α_w=10 single value, result-blind, STOP rule).
- 새 artifact: `outputs/13_frozen_direction_anchor/scifact/seed_{42,1337,2024}/qv_r8_l12_dir1/`.
- 새 figures: `report/figures/13_frozen_direction_anchor/*.{pdf,png}` (5 figures × 2 formats).

### Changed

- `REPORT.md` §6.1 grid — Exp 13 의 4 rows 추가 (3 seeds + 3-seed mean ± std).
- `REPORT.md` §6.1 ⭐ caption — Exp 13 의 3/3 strict + Δ easy −0.021 (Exp 11 의 −0.031 보다 best preserved) 언급, anchor-side family 의 frontier 강건성 정정.
- `REPORT.md` §7.3.f 다음에 §7.3.g 신설 — Exp 13 의 상세 분석 (가설, loss, grid, branch 판정, Exp 11 비교, 함의).
- `REPORT.md` §7.4.1 — *4-lever* → **5-lever** framework 확장. data-side (Exp 12 binary, M1b substitution) vs anchor-side (Exp 11 relational, Exp 13 absolute) family split 명시.
- `RESEARCH.md` — 새 dated entry `2026-05-24#6` 추가 (Exp 13 의 done / observations / decisions / open questions / next).
- `ROADMAP.md` — Exp 13 row 완료 표시 (TBD, queue status 갱신 TBD post Exp 14).

### Experimental — Exp 13 result-blind pre-committed (3 seeds, single config)

- **Branch (b) 확정** — Exp 11 과 frontier 공유, Δ all +0.030 ± 0.002 (3/3 strict), Δ confused +0.092 ± 0.007, Δ easy −0.021 ± 0.003 (branch (a) 임계 −0.020 을 0.001 차이로 miss).
- **Anchor-side family 의 frontier 강건성**: 두 *수학적으로 다른* constraint (Sim Frobenius² rotation-invariant vs per-token cosine rotation-sensitive) 가 *통계적으로 구분 안 되는* trade-off frontier 점유.
- **STOP rule 준수** — λ_dir sweep / variant / cross-dataset *전부 금지*.

### Why

- §7.3.f.ii NFCorpus puzzle (direction sufficient lever, magnitude not) 의 mechanism translation 가설 검정.
- Exp 11 의 *relational* (rotation-invariant) anchor 와 Exp 13 의 *absolute* (rotation-sensitive) anchor 가 *equivalent lever* 인지 검정 — empirical separation 가설.
- 결과: 두 constraint 가 같은 frontier 점유 → *anchor-side family 의 frontier 강건성* paper-grade negative finding.

---

## 🚫 [2026-05-24#11] — Exp 11 extensions launched (Higher λ + Combined + FN+EP variant) — *POST-HOC EXCLUDED FROM MAIN PAPER*

> **Methodology disclosure**: 본 dated entry 의 모든 실험 (Higher λ=5, Combined M1b+Exp 11, FN+EP variant) 은 *test 결과 본 후 generative question* 으로 발의된 *post-hoc exploratory*. `report/_exp11_extensions_pre_commit.md` 작성 시점이 Exp 11 (λ=1) 3-seed 결과 *후*. **Main paper claim base 에서 제외** (9 runs / 3 묶음). Reviewer agent recommendation 따라 *Exp 11 (λ=1) 의 2/3 strict partial 로 honest 종착* — paper *완전히 선다*.

### Experimental — 9 new runs queued

- **Higher λ Exp 11** (λ=5, single value pre-commit): SciFact × 3 seeds
- **M1b + Exp 11 combined** (in-batch neg + λ=1): SciFact × 3 seeds
- **🆕 Exp 13** (FN-denoised + relational easy preservation, λ=1 + threshold 0): SciFact × 3 seeds — *cleanest 3-way isolation* (hard 유지 + noise 제거 + selective preservation).

### Pre-commit (`report/_exp11_extensions_pre_commit.md`)

Each: single config, no sweep. 3 seeds for robustness.

### 🎯 Higher λ Exp 11 (λ=5) 3-seed COMPLETE — *strict net+ 3/3 robust*

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|
| 42 | 0.6876 | +0.041 ✓ | +0.138 ✓ | −0.040 ✗ |
| 1337 | 0.6814 | +0.035 ✓ | +0.089 ✓ | −0.010 (CI 거의 0) |
| 2024 | **0.6963** (highest!) | **+0.050 ✓** | +0.135 ✓ | −0.021 ✗ |
| **3-seed mean** | **0.6884 ± 0.006** | **+0.042 ± 0.006 ✓ 3/3 strict** | **+0.121 ± 0.023 ✓** | **−0.024 ± 0.013** |

⇒ **paper-grade *best result*** across all metrics:
- Strict net+ 3/3 (vs λ=1 의 2/3)
- Δ confused +0.121 > Phase 2b 의 +0.104 — *original baseline 보다 confused recovery 더 높음*
- λ marginal benefit *non-marginal* (earlier 2-seed estimate 정정)

**Updated 4-lever ranking** (3-seed mean):

| Method | Δ all | Δ confused | Δ easy | Strict |
|---|---|---|---|---|
| Phase 2b | +0.001 | +0.104 | −0.085 | 0/3 |
| Exp 12 | −0.004 | +0.080 | −0.073 | 0/3 |
| M1b | +0.021 | +0.065 | (~−0.05) | 3/3 |
| Exp 11 λ=1 | +0.029 | +0.101 | −0.031 | 2/3 |
| **Exp 11 λ=5** | **+0.042** | **+0.121** | **−0.024** | **3/3 ✓** |

### 🚨 Combined M1b + Exp 11 (3-seed final) — mildly antagonistic

| Method (3-seed mean) | Δ all | Δ confused | Δ easy | Strict |
|---|---|---|---|---|
| M1b alone | +0.021 ± 0.005 | +0.065 ± 0.012 | (~−0.05) | 3/3 |
| **Combined** | **+0.015 ± 0.002** | **+0.052 ± 0.004** | **−0.016 ± 0.000** | **2/3** |

⇒ Combined ≤ M1b alone (Δ all −0.006, Δ confused −0.013). *Mildly antagonistic* — Exp 11 preservation pressure *drag* M1b 결과 살짝 낮춤. Branch (b) sub-additive → (c) antagonistic 향한 tilt.

### 🚨 FN+EP variant (3-seed final) — redundant

| Method (3-seed mean) | Δ all | Δ confused | Δ easy | Strict |
|---|---|---|---|---|
| Exp 11 (λ=1) | +0.029 ± 0.005 | +0.101 ± 0.010 | −0.031 ± 0.018 | 2/3 |
| **FN+EP variant** | **+0.027 ± 0.009** | **+0.093 ± 0.036** | **−0.029 ± 0.014** | **3/3** |

⇒ FN denoising 추가 가 *redundant* — Exp 11 (λ=1) 와 essentially same. *Sole sufficient mechanism* 는 *relational preservation* 으로 이미 address.

### 🏆 최종 6-method ranking (3-seed mean)

| Method | Δ all | Δ confused | Δ easy | Strict | 평가 |
|---|---|---|---|---|---|
| Phase 2b | +0.001 | +0.104 | −0.085 | 0/3 | redistribution baseline |
| Exp 12 | −0.004 | +0.080 | −0.073 | 0/3 | noise removal ineffective |
| M1b | +0.021 | +0.065 | (~−0.05) | 3/3 | strict, confused half |
| Combined (M1b+Exp 11) | +0.015 | +0.052 | −0.016 | 2/3 | mild antagonism |
| Exp 11 (λ=1) | +0.029 | +0.101 | −0.031 | 2/3 | preserved confused |
| FN+EP variant | +0.027 | +0.093 | −0.029 | 3/3 | ≈ λ=1, redundant FN |
| **🏆 Exp 11 (λ=5)** | **+0.042** | **+0.121** | **−0.024** | **3/3** | **all metrics best** |

⇒ **All interventions converge to same upstream root** (hard-contrast over-correction). *Best single-lever* = Exp 11 (λ=5).

### Added

- `experiments/11_easy_preservation/run.py` 의 `--in-batch-neg`, `--fn-denoise`, `--fn-threshold`, `--tag-suffix` flags (재사용 infrastructure for combined experiments).
- `/tmp/_exp11_extensions_runner.sh` (Higher λ × 3 + Combined × 3 sequential).
- `/tmp/_exp13_runner.sh` (Exp 13 × 3, waits for extensions PID 9428 to finish).
- `report/_exp11_extensions_pre_commit.md` (pre-commit prediction, 6 branches).

### Pending

- Higher λ seed 2024 → Combined × 3 → Exp 13 × 3 → final doc update.

---

## [2026-05-24#10] — Diagnostic B on new checkpoints (mechanism direct verification)

### Experimental — 10 new conditions on CPU

`report/_repr_collapse_new_ckpts.py` + `report/figures/_repr_collapse_new_ckpts/{repr_collapse_new_ckpts.{pdf,png}, ..._data.json}`.

| Method | doc_cos μ | eff_rank doc | eff_rank tok |
|---|---|---|---|
| Frozen baseline | +0.573 | 10.65 | 57.21 |
| Phase 2b | +0.985 | 1.14 | 1.58 |
| **Exp 12 (3-seed mean)** | **+0.975** | **1.22 ± 0.01** | **1.72 ± 0.05** |
| **M1+M1b SciFact** | +0.663 | 7.05 | 43.16 |
| **M1b SciFact (3-seed)** | **+0.663 ± 0.010** | **7.12 ± 0.31** | **44.65 ± 1.55** |
| **Exp 11 (3-seed mean)** | **+0.910 ± 0.022** | **~1.9** | **~9.6** |

### 🎯 4 mechanism direct verifications

1. **Exp 12 = Phase 2b 동일 collapse** (1.22 ≈ 1.14) → *FN removal 만 collapse zero change* → (나-2) difficulty dominant collapse-level 추가 증거.
2. **M1+M1b ≡ M1b alone at collapse**: SciFact 7.05 ≈ 7.12, NFCorpus 1.05 ≈ 1.06 → *M1 추가 기여 collapse-level 도 zero*.
3. **Exp 11 의 *selective token-level* preservation 직접 확인**: token eff_rank 1.58 → 9.6 (6× recovery), doc 1.14 → 1.9 → *loss 가 token sim matrix 직접 규제 = 직접 보존*. **Direct mechanism evidence**.
4. **M1b collapse 감소 3-seed robust** (eff_rank doc 6.73/7.29/7.33) → *seed-artifact 가 아닌 robust mechanism*.

### NFCorpus *direction matters* puzzle 강화

NFCorpus M1+M1b = eff_rank 1.05 (= M1b alone) 임에도 NDCG 74 % recovery → *direction alignment* > *magnitude* 의 직접 evidence.

### Added

- `report/_repr_collapse_new_ckpts.py` — 10 new conditions diagnostic.
- `report/figures/_repr_collapse_new_ckpts/repr_collapse_new_ckpts.{pdf,png,json}` — figures + data.

### Changed

- `REPORT.md` §7.3.e.vi 신규 (Diagnostic B on Mediation+Exp Checkpoints).
- `RESEARCH.md` 2026-05-24#4 entry.

---

## [2026-05-24#9] — Exp 12 (FN-denoised mined-HN) — 캐비엇 1 결정적 disambiguation

### Experimental — FN-denoised mined-HN on SciFact 3 seeds

**Pre-committed** (`report/_exp12_pre_commit.md`, single threshold = 0.0): mined HN 의 e5_margin ≤ 0 (likely FN) 제거 → cleaned hard 로 Phase 2b 재학습.

**Setup**:
- `data/e5_teacher/extract_train_docs.py` 신규 작성 + 실행 → `e5_train_doc_emb_scifact.pt` (5183 docs × 4096 fp16, 41 MB, 3660 sec encoding).
- `experiments/12_fn_denoised_hn/{run.py, README.md}` 신규.
- 36.5 % mined HN removed as likely FN (3358 / 9190), 5832 cleaned triplets remaining.

### 🎯 결과 — (나-2) Difficulty dominant + (나-1) noise minor confirmed

| Seed | Δ all | Δ confused | Δ easy |
|---|---|---|---|
| 42 | −0.008 (CI 0) | +0.076 ✓ | −0.078 ✗ |
| 1337 | −0.005 (CI 0) | +0.079 ✓ | −0.075 ✗ |
| 2024 | +0.002 (CI 0) | +0.084 ✓ | −0.066 ✗ |
| **3-seed mean** | **−0.004 ± 0.005** | **+0.080 ± 0.004 ✓** | **−0.073 ± 0.005 ✗** |

**3-seed robust pattern**:
- Δ all ≈ 0 (CI 0 all 3) — *redistribution 유지*
- Δ easy −0.073 ≈ Phase 2b 의 −0.085 (단 +0.012 = 14 % recovery from FN removal)
- ⇒ **(나-2) Hard-contrast over-correction 이 catastrophic / redistribution 의 *주요* mechanism** (FN noise minor)

### Paper narrative — *결정적* 정정

| 기존 (M1+M1b 결과로 추정) | **새 (Exp 12 disambiguation 확정)** |
|---|---|
| Supervision noise → wrong collapse direction | **Hard-contrast over-correction → forced redistribution** (noise minor 14 % 만) |
| M1b net+ = noise 제거 효과 | **M1b net+ = *easy contrast 의 작은 gradient* (hard 회피 효과)** |
| 캐비엇 1 = FN-denoised mined-HN paper future work | **캐비엇 1 fully disambiguated**: hard difficulty dominant + FN noise minor 확정 |

### *4-lever* trade-off framework (paper-grade final)

| Lever | Δ all | Δ confused | Δ easy | Trade-off |
|---|---|---|---|---|
| Phase 2b (hard + noisy) | +0.001 | +0.104 ✓ | −0.085 ✗ | zero-sum baseline |
| **Exp 12 (hard + clean)** | **−0.004** | **+0.080 ✓** | **−0.073 ✗** | *동일 redistribution* — noise minor |
| **M1b (easy + clean)** | **+0.021 ✓ strict** | +0.065 (half) | (~−0.05) | hard 회피 → strict net+ but confused 절반 |
| **Exp 11 (hard + easy preservation)** | **+0.029 (2/3 strict)** | +0.101 ✓ | −0.031 | higher confused + moderate net+ |

⇒ **Single sufficient mechanism**: *Hard mined-HN over-correction* → confused/easy zero-sum redistribution. Two avoidance levers (M1b 의 hard 회피 / Exp 11 의 selective preservation) → different trade-offs.

### Added

- `data/e5_teacher/extract_train_docs.py` + `e5_train_doc_emb_scifact.pt` (cached, 41 MB).
- `experiments/12_fn_denoised_hn/{run.py, README.md}` + `report/_exp12_pre_commit.md`.
- `outputs/12_fn_denoised_hn/scifact/seed_{42,1337,2024}/qv_r8_l12_thresh0/...` (3 artifact dirs + `denoising_stats.json`).
- `/tmp/_exp12_runner.sh` (orchestrator, waits for E5 then runs 3 seeds).

### Changed

- `REPORT.md` §6.1 grid Exp 12 4 rows + §7.3.e.v (Exp 12 disambiguation) + §7.4 narrative *근본 정정* + §7.4.1 (4-lever framework).
- `report/10_lora_phi_report.md` 캐비엇 1 status 정정 (still pending → empirically disambiguated).
- `RESEARCH.md` 2026-05-24#3 entry (Exp 12 결과 + decisions + narrative).
- `ROADMAP.md` 2026-05-24#5 changelog row.

---

## [2026-05-24#8] — Overnight autonomous experiments: M1b 3-seed robust + M1+M1b red herring + Exp 11 branch (a) partial

### Experimental — overnight orchestrator 9 새 runs + retry

`/tmp/_overnight_runner.sh` + `/tmp/_exp11_retry.sh` 자율 실행, 04:54 → 06:25 (~1.5 hour).

### 🎯 M1b SciFact 3-seed strict robust (캐비엇 2 fully 해소)

| Seed | NDCG@10 all | Δ all | Δ confused |
|---|---|---|---|
| 42 | 0.6613 | +0.015 ✓ razor-thin | +0.055 ✓ |
| 1337 | 0.6681 | +0.022 [+0.008, +0.036] ✓ | +0.064 ✓ |
| 2024 | 0.6722 | +0.026 [+0.011, +0.042] ✓ | +0.077 ✓ |
| **3-seed mean** | **0.6672 ± 0.005** | **+0.021 ± 0.005 ✓ ROBUST** | **+0.065 ± 0.012 ✓** |

⇒ *08-style seed-artifact 시나리오 fully 기각*. Frozen-encoder lightweight intervention 의 *3-seed robust strict net+* 첫 사례.

### 🎯 M1b NFCorpus 3-seed cross-dataset robust (74 % gap recovery)

3-seed mean: NDCG 0.244 ± 0.020, Δ all −0.086 ± 0.020 ✗, gap recovery 74 % ± 7. *Mined HN noise 가 cross-dataset universal supervision root*.

### M1+M1b combined → M1 contribution = ZERO (optimization root = red herring) — *post-hoc exploratory, negative result*

> ⚠️ Post-hoc note: M1+M1b combined 는 *M1, M1b 의 test 결과 본 후 발의* (post-hoc generative question). 단 *negative result* (M1 contribution zero) — *selection-on-noise risk 낮음*. Reviewer recommendation: clean 이라 우기지 않고 *"post-hoc exploratory, negative"* 라벨. M1 = red herring 의 *직접 evidence* 로는 사용하지 않고, M1 alone 의 NDCG null result 에서 *간접 추론* 권고.

| Dataset | M1 alone | M1b alone | M1+M1b combined | M1 의 추가 |
|---|---|---|---|---|
| SciFact | −0.012 (≈) | +0.021 ✓ | **+0.020 ✓** | −0.001 (zero) |
| NFCorpus | −0.319 ✗ | −0.084 ✗ | **−0.083 ✗** | +0.001 (zero) |

⇒ **Sole mechanism = supervision root**. M1 의 ep1 trajectory 효과 (1.9× / 2.86×) 는 *training-time artifact* (final-state 와 무관 + M1b 와 additive 아님).

### 🎯 Exp 11 (λ=1.0, relational easy preservation) — branch (a) partial

| Seed | Δ all | Δ confused | Δ easy |
|---|---|---|---|
| 42 | +0.033 ✓ | +0.095 ✓ | −0.019 ✗ |
| 1337 | +0.032 ✓ | +0.095 ✓ | −0.021 ✗ |
| 2024 | +0.023 (CI 0) | +0.113 ✓ | −0.052 ✗ |
| **3-seed mean** | **+0.029 ± 0.005** (2/3 strict) | **+0.101 ± 0.010 ✓ preserved** | **−0.031 ± 0.018** (63 % 감소) |

⇒ *Pre-committed branch (a) partial* — *redistribution 부분 해소*. *Confused lift fully preserved* (M1b 의 절반 sacrifice 와 대조), *easy 손상 63 % 감소*. 단 *3-seed strict 아님* (1 marginal).

### Paper narrative — 정밀화

- **Sole-mechanism conclusion**: Phase 2b catastrophic / redistribution 의 *유일* root = **mined HN noise (supervision root)**.
- **Two levers** for *partial* resolution: M1b (general clean supervision, strict robust net+ but sacrifices half confused) 와 Exp 11 (selective relational easy preservation, higher confused preservation + partial easy damage).
- **Optimization root = red herring**: M1 의 trajectory effect 는 *test-time outcome 과 분리*.

### Caveats — 정정 + maintenance

- 🟢 **캐비엇 2 fully 해소** (M1b SciFact 3-seed robust).
- 🔴 **캐비엇 1 (clean ≠ easy confound) still unresolved** — FN-denoised mined-HN의 full-strength replication 의 필요성 *증가*.

### Added

- `/tmp/_overnight_runner.sh` + `/tmp/_exp11_retry.sh` (overnight orchestrators).
- `outputs/10_lora_phi/{scifact,nfcorpus}/seed_{1337,2024}/qv_r8_l12_m1b/...`
- `outputs/10_lora_phi/{scifact,nfcorpus}/seed_42/qv_r8_l12_m1plus1b/...`
- `outputs/11_easy_preservation/scifact/seed_{42,1337,2024}/qv_r8_l12_le1/...`
- `report/_overnight_results.md` (chronological raw).
- `experiments/11_easy_preservation/run.py` 의 `--early-stop-metric` flag (compatibility — Exp 11 always val_all).

### Changed

- `REPORT.md` §6.1 grid 의 15 신규 rows (M1b 3-seed, M1+M1b combined, Exp 11).
- `REPORT.md` §7.3.e.i.β (M1b 3-seed robustness), §7.3.e.ii.β (NFCorpus 3-seed), §7.3.e.iii (M1+M1b combined, red herring), §7.3.e.iv (Exp 11 branch a partial).
- `report/10_lora_phi_report.md` 의 mediation result table 갱신.
- `RESEARCH.md` 2026-05-24#2 dated entry (overnight summary).

---

## [2026-05-24#7] — Diagnostic B on Mediation Checkpoints: M1b 의 *dataset-dependent* mechanism

### Experimental — direct collapse measurement

`report/figures/_repr_collapse_mediation/{repr_collapse_mediation.{pdf,png}, ..._data.json}`. CPU 강제 (GPU queue 비충돌). n=300 docs per condition.

| Dataset | Frozen | Phase 2b | M1 | **M1b** |
|---|---|---|---|---|
| SciFact eff_rank doc | 10.65 | 1.14 | 1.14 | **7.29 (frozen 의 68%)** |
| SciFact doc_cos | 0.573 | 0.985 | 0.985 | **0.656** |
| NFCorpus eff_rank doc | 11.73 | 1.09 | 1.06 | **1.06 (collapse 그대로!)** |
| NFCorpus doc_cos | 0.553 | 0.990 | 0.993 | **0.995** |
| FiQA eff_rank doc | 23.58 | 1.06 | 1.05 | (queue 종료 후) |

### 핵심 발견 — M1b 의 *dataset-dependent multi-mechanism*

1. **M1 의 final-state collapse 감소 효과 없음** (3 datasets). Train trajectory 효과 (ep1 val_all 2× ↑) 는 *test-time 의 final state 와 무관*. ep3 = post-warmup collapse 로 회귀.
2. **M1b 의 두 *서로 다른* mechanism**:
   - **SciFact**: *collapse magnitude* 자체 6.4× 감소 (1.14 → 7.29).
   - **NFCorpus**: *collapse magnitude* 변화 없음 (1.09 → 1.06), but *NDCG 0.009 → 0.246 (74 % 회복)*.
3. **NFCorpus M1b paradox**: *same eff_rank, very different NDCG* → *collapse direction 의 task-alignment 회복* 이 핵심 mechanism. §7.3.c.ii 의 "rank-1 puzzle / direction matters" 와 *empirically 강력 정합*.

### Paper framework — 정밀화

- *Supervision root 가 multi-mechanism*: collapse *magnitude* *또는* *direction*.
- Dataset 마다 dominant mechanism 다름.

### ⚠️ 정정 — 캐비엇 1 (clean ≠ easy) 는 *해소 안 됨*

이전 draft 의 *"부분 해소"* 추론은 **틀림**. Easy in-batch 도 *방향* 교정 가능 — *query 를 무관한 doc 에서 분리* 라는 일반-올바른-방향 의 약한 신호 만으로도 NFCorpus 의 wrong-direction collapse 교정 가능. (나-1) noise 제거 + (나-2) easy 의 일반-올바른 신호 *둘 다 똑같이 설명* → confound *유지*. **FN-denoised mined-HN 은 *full strength* 로 여전히 필요** (유일한 깨끗한 disambiguator).

### ⚠️ 정정 — NFCorpus M1b는 *net+* 아님

NDCG 0.246 vs baseline 0.330 = **Δ all −0.084 (여전히 negative)**. *"74% 회복" = catastrophic-gap recovery*, NOT net 향상. 결과 frame 시 *"NFCorpus 고침" / "net+" 사용 금지*. + Single-seed (seed 42).

### 추가 검사 — Mediation sanity check 필요

§7.3.c.i sanity check (diagnostic-loaded model NDCG = reported NDCG) 는 *Phase 2b 만* 검증, *M1/M1b 는 미검증*. Claim A (*same eff_rank, very different NDCG = direction matters*) 의 단단함은 *eff_rank↔NDCG pairing 의 정확성* 에 걸려 있음 — *NFCorpus M1b 의 0.246 재현 검증* 필요 (LoRA best-state 미snapshot 한계 + ep3-final 사용 환경에서).

---

## [2026-05-24#6] — Mediation 1b NFCorpus: 74 % catastrophic 회복 (supervision root cross-dataset)

### Experimental — In-batch negative on NFCorpus

`outputs/10_lora_phi/nfcorpus/seed_42/qv_r8_l12_m1b/`. *Cross-dataset supervision root 의 *부분* 지지*.

| 지표 | Phase 2b | M1b |
|---|---|---|
| NDCG@10 all | 0.0094 | **0.2459** (baseline 0.330 의 74.5 %) |
| Δ all vs baseline | −0.320 ✗ | **−0.084 [−0.105, −0.064] ✗** (74% 회복) |
| Δ confused vs baseline | −0.092 ✗ | **−0.013 [−0.027, +0.002]** (CI 0 포함, *baseline 회복*) |
| ep1 val_all | 0.073 | **0.376 (baseline +0.046 ▲)** |
| ep2-3 val_all | 0.017 / 0.015 | 0.285 / 0.259 |

### 핵심 발견

1. **Catastrophic 의 74 % 회복** — *mined HN noise* 가 *cross-dataset universal* root.
2. **Confused 거의 baseline 회복** — supervision noise 의 *직접* 결과 확인.
3. **Strict 회복 *가능* 했음** — ep1 val_all 0.376 > baseline 0.330. *LoRA best-state snapshot* 한계 가 strict 미달성 의 *유일* 이유.
4. **Optimization root 의 부분 잔존** — ep1 → ep3 decay (0.376 → 0.259), *post-warmup full LR* 의 추가 collapse.

### Paper narrative 정밀화

- **Catastrophic = mined HN noise (supervision root, 주요) + optimization 폭주 (optimization root, 부분)**.
- *Two roots additive* — 별도 mechanism, 둘 다 기여.
- Cross-dataset universality 의 *부분* 지지 (NFCorpus 의 74 % 회복).
- Final disentangling: FiQA M1b 결과.

---

## [2026-05-24#5] — 🎯 Mediation 1b SciFact: 첫 strict net 향상 + supervision root 지지

### Experimental — In-batch negative on SciFact

`outputs/10_lora_phi/scifact/seed_42/qv_r8_l12_m1b/`. **Pre-committed strict 기준 (CI(Δ all) > 0) 첫 충족 *시그널* (single seed, 확정 아님)** — 본 paper 의 01-10 prior 실험 모두 미달성.

### ⚠️ 두 under-weighted 캐비엇 (reviewer agent catch)

1. **clean ≠ easy 혼동**: in-batch neg = *clean + EASY*. M1b 의 net 향상 이 (나-1) noise 제거 인지 (나-2) hard negative 자체 가 collapse 유발 인지 구분 불가. 결정적 disambiguator = **FN-denoised mined-HN** (future work).
2. **seed 42 단독 + CI 하한 +0.001 razor-thin**: 08 의 seed-artifact 시나리오 와 동일 구조. 3-seed 전 까지 *확정* 아님 — *signal (preliminary)* 만.

→ Paper frame: *promising preliminary*, *확정 결론 아님*. NFCorpus/FiQA M1b + 3-seed 가 final disentangling.

| 지표 | Phase 2b (3-seed) | **M1b (in-batch neg, seed 42)** |
|---|---|---|
| NDCG@10 all | 0.6476 | **0.6613** |
| Δ all vs baseline | +0.001 ± 0.012 (≈) | **+0.015 [+0.001, +0.029] ✓ STRICT positive** |
| Δ confused vs baseline | +0.104 ± 0.014 ✓ | +0.055 [+0.030, +0.081] ✓ (Phase 2b 의 1/2) |
| ep1-3 val_all | 0.604 / 0.618 / 0.614 | **0.672 / 0.682 / 0.679 (모든 epoch baseline 위)** |
| ‖B‖_total | (≈ Phase 2b) | **1.32 (Phase 2b 의 63%)** |

### 핵심 발견 — supervision root 강력 지지

1. **Phase 2b 의 redistribution 깨뜨림**: zero-sum (confused +0.104 / easy −0.085 / all ≈ 0) → *non-zero net 향상* (confused +0.055 / all +0.015 ✓ strict).
2. **Mined HN noise 가 redistribution 의 주요 원인** 확정 — clean negative (in-batch) 면 redistribution 사라짐.
3. **Cross-dataset (NFCorpus / FiQA) M1b** 결과 가 *final disentangling* — supervision root 의 universality 결정 (running).

### Added
- `report/_easy_slice_step0.py` + `report/figures/_easy_slice_step0.json` — Exp 11 의 Step 0 gate measurement.
- `experiments/11_easy_preservation/{run.py, README.md}` — explicit easy-preservation loss (λ > 0 relational self-sim 보존). 사용자 confirm 후 queue 종료 시점 launch.
- `report/_exp11_pre_commit.md` — pre-commit prediction (3 분기) + STOP rule.

### Methodology
- M1 (warmup+clip) 결과: NFCorpus/FiQA train trajectory 명확 개선 (ep1 val_all 1.9× / 2.86×), test NDCG 는 동일 catastrophic (LoRA best-state 미snapshot 한계 + post-warmup full LR 영향). *Optimization root 부분 지지*.
- **M1b SciFact 결과: 첫 strict 향상 — paper 의 main bounded-improvement narrative 의 *significant* 진전**.
- *NFCorpus / FiQA M1b 결과* 대기 (cross-dataset supervision root 검정 마지막 단계).

---

## [2026-05-24#4] — Mediation 1 (warmup + grad-clip) — optimization root partial

### Experimental — M1 on 3 datasets

`outputs/10_lora_phi/{scifact,nfcorpus,fiqa}/seed_42/qv_r8_l12_m1/`. Single rule: warmup 10% + grad-clip max_norm=1.0, 기타 Phase 2b 동일.

| Dataset | NDCG@10 all | Δ all vs baseline | Δ confused vs baseline | Judgment |
|---|---|---|---|---|
| SciFact | 0.6342 | −0.012 [−0.046, +0.021] (≈) | +0.088 [+0.035, +0.139] ✓ | ✓ Phase 2b 동등 (sanity) |
| NFCorpus | 0.0113 | −0.319 [−0.353, −0.286] ✗ | −0.093 ✗ | ✗ catastrophic (Phase 2b 와 통계 동등) |
| FiQA | 0.0009 | −0.346 [−0.374, −0.319] ✗ | −0.147 ✗ | ✗ catastrophic (Phase 2b 와 통계 동등) |

### Train trajectory 의 명확한 효과 (test NDCG 와 *분리*)

| Dataset | Phase 2b ep1 val_all | M1 ep1 val_all | Train-time effect |
|---|---|---|---|
| SciFact | 0.604 | 0.624 | +0.020 (소폭 ↑) |
| NFCorpus | 0.073 | 0.140 | **1.9× ↑** |
| FiQA | 0.090 | 0.257 | **2.86× ↑** |

**Optimization root 의 *부분* 지지**: warmup+clip 가 ep1 collapse *명확하게 지연* — 단 post-warmup full LR 가 ep2/3 에서 collapse 재현 → *single rule 영구 회복 불가*. *LoRA best-state 미snapshot* 한계 가 test NDCG 의 catastrophic 유지 도 기여.

### Added
- `src/train.py` 의 `TrainConfigLite` 에 `warmup_frac` + `grad_clip_max_norm` + `in_batch_neg` fields.
- `experiments/10_lora_phi/run.py` 에 `--warmup-frac` / `--grad-clip` / `--in-batch-neg` / `--tag-suffix` CLI flags.

---

## [2026-05-24#3] — Sanity check: diagnostic-loaded NDCG match (reviewer catch resolution)

### Experimental — Reviewer 의 *rank-1 puzzle* 검정

**Critical reviewer catch**: "**rank-1 embedding 으론 NDCG 0.65 *수학적으로 불가능***" — SciFact LoRA tok_cos +0.94 / eff_rank 1.61 *상태에서* NDCG 0.6367 모순. 가설:
- (A) `module_final.pt` (final epoch) vs eval-used best-epoch 불일치
- (B) LoRA injection α scaling 불일치

**Sanity check** (`report/_repr_collapse_sanity.py`): 진단이 로드한 *바로 그* model 의 test NDCG@10 재현.

| Dataset | Diagnostic-loaded NDCG@10 | Original-run NDCG@10 | Match |
|---|---|---|---|
| SciFact | 0.6367 | 0.6367 | ✓ |
| NFCorpus | 0.0094 | 0.0094 | ✓ |
| FiQA | **0.0005** | **0.0005** (baseline 0.347 의 0.15 %, *literal 0% retrieval*) | ✓ |

**3 / 3 match → 가설 A/B 모두 기각**. **Collapse 가 진짜** + **rank-1 puzzle 도 진짜** — *SciFact 의 eff_rank ≈ 1.15 *상태에서* NDCG 0.6367 실제 발생*.

**Side benefit**: FiQA 의 *실제* NDCG 가 backsolve-derived 0.0388 이 아닌 **literal 0.0005** 임 발견. 보고서 의 *최초* 표기 *부정확* → 정정 적용 (REPORT/10/RESEARCH/CHANGELOG 모두).

### *Rank-1 puzzle* 해석 (paper-grade)

Reviewer 의 *"rank-1 → random"* premise 부정확:
1. eff_rank perplexity ≈ 1 ≠ literal rank-1 — trailing dimensions 의 residual signal 잔존.
2. MaxSim 의 per-token max 가 small residual structure 를 *amplify*.
3. Mean-pooled cosine ↑ 가 per-token MaxSim discrimination 를 *non-trivially* 상관.

⇒ **Catastrophic ≠ collapse magnitude**; **Catastrophic = collapse direction misalignment** (sanity-check-confirmed 정밀화).

### Added
- `report/_repr_collapse_sanity.py` — 진단-loaded model 의 test NDCG@10 재현.
- `report/figures/_repr_collapse/sanity_check_ndcg.json` — sanity result.
- `report/_catastrophic_failure_section_draft.md` — *pre-commit binding* paper section 골격 (mediation 결과 보기 전 narrative 골격 + 기각 vs 확정 조건 확정).

### Changed
- `REPORT.md` §7.3.c.i (sanity check) + §7.3.c.ii (rank-1 puzzle 해석) 신규.
- `report/10_lora_phi_report.md` §2.8 신규 (sanity check + rank-1 puzzle).
- FiQA NDCG 표기 정정 (모든 doc): 0.0388 (backsolve) → 0.0005 (실측).

---

## [2026-05-24#2] — FiQA catastrophic + Diagnostic B (representation collapse)

### Experimental — 10 Phase 2b cross-dataset FiQA (catastrophic, 2/2 cross-dataset confirmed)

`10_lora_phi fiqa seed 42` (q,v r=8, LR=5e-5, α=r, early-stop=val_all, --max-triplets 9,190):

| 지표 | SciFact (3-seed) | NFCorpus | **FiQA** |
|---|---|---|---|
| NDCG@10 all | 0.6476 | 0.0094 | **0.0005** (baseline 0.347 의 0.15 %, *literal 0% retrieval*) |
| Δ all vs baseline | +0.001 (≈) | −0.320 ✗ | **−0.347 [−0.374, −0.319] ✗** catastrophic |
| Δ confused vs baseline | +0.104 ✓ | −0.092 ✗ | **−0.147 [−0.166, −0.127] ✗** |

**2 / 2 cross-dataset (NFCorpus + FiQA) catastrophic 확정**. Single-dataset artifact 가설 명확히 기각. SciFact-tuned LR=5e-5 의 cross-dataset generality 부정.

### Experimental — Diagnostic B (encoder output representation collapse)

외부 reviewer 의 가설 검정: catastrophic NDCG 의 *mechanism* 이 encoder output space 의 *representation collapse* 인가? `report/_repr_collapse_diagnostic.py` 작성 (n=500 docs 샘플 per corpus, random pair cosine + singular-spectrum effective rank).

| Dataset | 조건 | doc-pair cos μ | tok-pair cos μ | eff_rank doc | eff_rank tok |
|---|---|---|---|---|---|
| NFCorpus | frozen | +0.553 | +0.211 | 11.73 | 55.94 |
| NFCorpus | LoRA 2b | **+0.990** | **+0.940** | **1.09** | **1.62** |
| FiQA | frozen | +0.380 | +0.181 | 23.58 | 63.91 |
| FiQA | LoRA 2b | **+0.993** | **+0.990** | **1.06** | **1.10** |
| SciFact | frozen | +0.573 | +0.209 | 10.65 | 57.21 |
| SciFact | LoRA 2b | **+0.984** | **+0.942** | **1.15** | **1.61** |

**Surprise**: 3 dataset 모두 LoRA Phase 2b 에서 *극단적 representation collapse* (doc-pair cos ≈ 0.99, eff_rank ≈ 1) — *universal phenomenon*. 그러나 SciFact 는 catastrophic *아님*. ⇒ **Collapse 자체는 *necessary but not sufficient***. Catastrophic 의 *진짜 mechanism* 은 *collapse direction* 의 task-alignment 가 결정.

**§8.5 의 parameter-space ΔW rank-collapse 와 다른 현상** (output 공간의 직접 측정). 둘 모두 universal: parameter rank 1.71 → output rank 1.1 로 전파.

### Added
- `report/_repr_collapse_diagnostic.py` — encoder output representation collapse 측정 (random pair cosine + singular-spectrum effective rank, n=500 doc subsample).
- `report/figures/_repr_collapse/repr_collapse.{pdf,png}` + `repr_collapse_data.json` — 4 rows × 3 datasets visualization.

### Changed
- `REPORT.md` §7.3.c 신규 (Diagnostic B 결과 통합), §6.1 grid 의 *3-seed mean* 행에 ⭐ marker (best architecture).
- `report/10_lora_phi_report.md` §2.6 신규 (FiQA cross-dataset), §2.7 신규 (Diagnostic B representation collapse).
- `RESEARCH.md` 2026-05-24 entry 에 *FiQA + Diagnostic B* observations + updated decisions/open-questions/next.

### Methodology
- *Reviewer 의 disentangling plan 채택*: Mediation 1 (warmup + grad_clip, optimization root) + Mediation 1b (in-batch negative, supervision root) — 각 single-rule + result-blind 1 run per dataset (3 datasets each). 시작 전 paper section "Catastrophic Failure as Representation Collapse" 의 골격 commit 필요 (post-hoc cherry-picking 회피).

---

## [2026-05-24]

### Added
- `experiments/02_final_layer_vector/run.py` 에 `--no-steering` flag — SteeringModule 의 v=0 frozen (no-grad). *Pure encoder finetune baseline* — "v=0 hook 이 학습 신호 추가했냐" 의 reviewer 공격 회피. Artifact subdir `unfrozen_no_steering/`.
- `report/_rank_collapse_analysis.py` + `report/figures/_cross_method/rank_collapse_contrast.{pdf,png}` + `report/figures/_cross_method/rank_collapse_data.json` — 06 K-router / 08 bilinear M / 10 LoRA 의 *cross-method effective rank* 통합 분석.
- `report/_rank_collapse_punchline.md` — paper main contribution 의 *통합 narrative* draft.

### Experimental — 10 Phase 2b seed × 3 robustness (paper-grade)

3-seed sweep on Phase 2b config (q,v r=8, LR=5e-5, α=r, early-stop=val_all), pre-committed *결과 보기 전*: "3-seed mean ± CI 보고".

| Seed | NDCG@10 all | Δ all vs baseline | **Δ confused vs baseline** |
|---|---|---|---|
| 42 | 0.6367 | -0.010 [-0.044, +0.023] (≈) | +0.091 [+0.040, +0.143] ✓ |
| 1337 | 0.6423 | -0.004 [-0.038, +0.028] (≈) | +0.097 [+0.047, +0.150] ✓ |
| 2024 | 0.6639 | +0.018 [-0.014, +0.049] (≈) | +0.123 [+0.073, +0.174] ✓ |
| **3-seed mean ± std** | **0.6476 ± 0.014** | **+0.001 ± 0.014** (anchor preserved) | **+0.104 ± 0.017 ✓** |

**Phase 2b 의 lift 가 *robust***: 3 seeds 모두 Δ confused 통계 유의 + anchor 보존 (CI 0 포함). *08 의 seed-artifact 와 *완전 반대 양상*. 3-seed mean Δ confused +0.104 가 02 unfrozen (+0.252) 의 **41%** 회복.

### Experimental — 10 Phase 2b cross-dataset NFCorpus (catastrophic, *expected per 06 K=2 NFCorpus 교훈*)

**`10_lora_phi nfcorpus seed 42` (q,v r=8, LR=5e-5, α=r, early-stop=val_all, --max-triplets 9,190)**:

| 지표 | SciFact (3-seed mean) | **NFCorpus (single seed)** |
|---|---|---|
| NDCG@10 all | 0.6476 | **0.0094** (baseline 0.330 의 2.8%) |
| Δ all vs baseline | +0.001 (≈) | **−0.320 [−0.355, −0.287] ✗** catastrophic |
| Δ confused vs baseline | +0.104 ✓ | −0.092 [−0.115, −0.070] ✗ |
| Ep1 rank loss | 0.66 | **4.47** (7×) |

**06 K=2 NFCorpus 의 −0.250 catastrophic 보다 *더 심함*** — *same Phase 2b config 가 cross-dataset transfer 안 됨*. NFCorpus 의 *strong-HN regime* (rank loss 7×) 에서 LR=5e-5 의 same config 도 *immediate over-correction*. **Paper limitations 명시**: hyperparameter sensitivity 가 dataset-specific. *Universal rank-collapse + spatial multiplicity* 의 method-architectural claim 은 cross-dataset 일 것, *numerical lift* (+0.104) 는 SciFact-specific.

### Experimental — Clean ColBERT-finetune baseline (no steering hook, paper-grade)

`02 unfrozen` 의 *v=0 hook 의 영향* 검정. `--no-steering` (SteeringModule v frozen no-grad) 으로 *pure encoder finetune* 실행.

| 지표 | 02 unfrozen (with v=0 hook) | **Clean baseline (no steering)** |
|---|---|---|
| NDCG@10 all | 0.6576 | **0.6924** (baseline 0.6464 +0.046) |
| Δ all vs baseline | +0.011 [-0.039, +0.062] (≈) | **+0.046 [-0.002, +0.096]** (CI 하한 -0.002 — *strict 돌파 직전*) |
| **Δ confused vs baseline** | +0.252 ✓ | **+0.260 [+0.182, +0.338] ✓** |
| ‖v_learned‖ | 0.33 | **0.0** (no-grad 확인) |

**핵심 진단**: v=0 hook 의 영향 ≈ 0 — *Reviewer 의 "v=0 hook 이 학습 신호 추가했냐" 공격 완전 해소*. *Frozen-encoder bottleneck* claim 의 *cleanest evidence* (no hook, just encoder finetune → Δ conf +0.260).

### Experimental — Cross-method universal rank-collapse (paper main punchline)

| Method | Nominal | Effective | Util ratio | Positions |
|---|---|---|---|---|
| 06 K-router K=2 | 2 | 1.41 | 70% | 1 |
| 06 K-router K=4 | 4 | 1.23 | 31% | 1 |
| 06 K-router K=8 | 8 | 1.44 | 18% | 1 |
| 08 bilinear M r=8 | 8 | 1.01 | 13% | 1 |
| 10 LoRA r=1 (Phase 1) | 1 | 1.00 | 100% | 24 |
| **10 LoRA r=8 (Phase 2b)** | 8 | **1.71** | 21% | **24** |

**핵심 발견**: 모든 학습된 frozen-side intervention 의 *per-position effective rank 가 ~1-2* 의 **universal collapse pattern**. *Pairwise margin + AdamW + small_random init* 의 학습 동학의 systematic feature. *Empirical lift 의 진짜 lever 는 per-position capacity 가 아닌 **distinct intervention positions 의 spatial multiplicity***. 06/08 의 1 position × ~1.4 → Δ conf +0.04-0.05, 10 LoRA 의 24 positions × ~1.7 → +0.104 (mean), 02 unfrozen 의 full encoder → +0.252. **Position 수 log-scale 의 monotonic correlation with confused lift**.

### Changed
- `REPORT.md` Abstract 의 *핵심 진단* 부분 재정렬 — *universal rank-collapse + spatial multiplicity escape* 를 main contribution 으로.
- `REPORT.md` §5d 갱신 (seed × 3 결과 통합) + **§5e 신규 (cross-method punchline)**.
- `report/10_lora_phi_report.md` §2.4 신규 (seed × 3) + §8.5 신규 (universal rank-collapse + spatial multiplicity).

### Methodology
- **Pre-commit 의 *시간 여유 에도 불변* 의 원칙 유지**: 24 시간 여유 생겨도 hyperparameter sweep 금지. *강건성 검증 (seed × 3 + clean baseline + NFCorpus) 만 진행*. 외부 reviewer 입력에 따른 method 정합성 우선.

---

## [2026-05-23#16]

### Added
- `src/lora.py` — `LoRALinear` (Hu et al. 2021 rank-r additive adapter, A∼N(0,σ²)+B=0 init for anchor preservation) + `inject_lora_into_bert(target_components, layers, r, alpha, init_std)` injection utility + `lora_param_count()` budget calc.
- `src/train.py:train_steering()` 에 `early_stop_metric` 인자 추가 (`"all"` or `"confused"`). 옛 historical default `"confused"` 유지.
- `experiments/10_lora_phi/{run.py, README.md, figures.py}` (Stage 3 main novelty 후보 — *renamed from 18_lora_phi*). frozen SteeringModule (no-op v=0 hook at layer 12) + LoRA adapter 학습.
- `report/10_lora_phi_report.md` + 5 figures (ndcg_vs_configs, delta_ci_forest, lora_progression, lora_AB_norms, train_curve_3configs).

### Changed
- **50K param budget constraint 완화** (사용자 결정, CLAUDE.md §3.2 의 hard rule → guideline). LoRA Phase 2 (r=8, 295K params) 까지 허용. *Paper deliverable 시점에 명시.*
- **Pre-committed 판정 기준 도입** (외부 reviewer 입력 반영, *결과 보기 전 commit*):
  - Early-stop = `val_all` (옛 `val_conf` 기본 → cherry-picking 위험 회피)
  - 돌파 ⟺ paired bootstrap CI 하한$_{\Delta\text{NDCG@10 all vs baseline}} > 0$
  - 미돌파 시 → hyperparameter sweep 금지, safety-net narrative 채택
- `REPORT.md` §5d 신규 (10 LoRA on Φ) + §6.1 grid 에 Phase 1/2a/2b 행 추가 + §11 meta 갱신.

### Experimental — Stage 3 (LoRA on Φ) 3-phase sweep

**(Phase 1) q,v r=1 LR=5e-5 α=r (36,864 params)**:
- NDCG@10 all = 0.5940
- Δ all vs baseline -0.052 ✗ (anchor 손상)
- Δ confused vs baseline +0.038 (CI 0 포함, not sig)

**(Phase 2a) q,v r=8 LR=1e-4 α=2r (294,912 params, *opportunistic exploratory*)**:
- NDCG@10 all = 0.5879
- Δ all vs baseline -0.059 ✗ (anchor 더 손상)
- Δ confused vs baseline **+0.080 [+0.021, +0.140] ✓** (frozen-side max 의 1.5×)

**(Phase 2b "B") q,v r=8 LR=5e-5 α=r (294,912 params, *pre-committed 결판 run*)**:
- NDCG@10 all = 0.6367
- Δ all vs baseline **-0.010 [-0.044, +0.023]** (CI 0 포함, 통계 동등 — anchor preservation 회복)
- Δ confused vs baseline **+0.091 [+0.040, +0.143] ✓ positive** (frozen-side max +0.054 의 *1.7×*)
- 02 unfrozen 의 +0.252 의 **36%** 회복 (295K params = encoder 110M 의 0.27%)
- *Pre-commit strict 돌파 미달*: CI 하한 -0.044 < 0 → **hyperparameter sweep 중단**

**LoRA 의 *균등 capacity utilization* 발견**: 24 adapters (q+v × 12 layers) 의 ‖A‖, ‖B‖ 가 *균등 분포* — 06 K-router / 08 bilinear M 의 effective rank 1 collapse 와 *반대* 양상. BERT 자체의 layer-wise gradient flow 가 *적절한 학습 distribution* 유도. *단 capacity 균등 활용에도 strict 돌파 미달* → 9K SciFact triplet 의 *data bottleneck* 한계 (Phase 2b ep3 train loss 0.10, 완전 fit 직전).

### Paper main contribution candidate (bounded improvement framing)

> Frozen ColBERT 의 lightweight intervention 의 *bounded improvement* — translation family / form-change / distillation 의 ceiling 0.665 + LoRA 의 confused +0.091 (anchor-preserving). Encoder unfreeze (110M) 만이 strict 돌파 (Δ confused +0.252). LoRA 295K (encoder 0.27%) 가 그 lift 의 36% 회복 — *param-efficient partial recovery*. *9K SciFact triplet 의 data bottleneck* 이 strict 돌파 차단.

### Robustness limitations (정직)

- Single-seed (42), single-dataset (SciFact)
- LoRA design space 의 *full* sweep 미수행 (pre-commit 따라 *현 deliverable scope 외*)
- Best-state selection 기준 변경 (val_conf → val_all): *10 의 모든 phase 동일* 이지만 02-09 의 historical anchor 와 *완전히 fair 한 비교 아님*

---

## [2026-05-23#15]

### Added
- `experiments/06_k_sweep/run.py` 에 `--max-triplets` flag — dense-qrels 데이터셋 (NFCorpus 1.1M triplets) 에서 SciFact-comparable scale (9190) 로 deterministic subsample 옵션.
- `experiments/02_final_layer_vector/run.py` 에 `--unfreeze-encoder` + `--encoder-lr` flag — ColBERT encoder (110M params) 도 학습 가능하게 설정 + 별도 optimizer group. Artifact 는 `outputs/02_final_layer_vector/{ds}/seed_{seed}/unfrozen/` 분리.
- `src/train.py:train_steering()` 에 `extra_param_groups` + `train_encoder` 인자 — encoder finetune 시 optimizer 추가 group + epoch loop 에서 model.bert.train()/eval() 분기.
- `outputs/00_baseline/scifact/seed_{1337,2024}` 등 baseline anchor 의 seed symlink (paired bootstrap 의 cross-seed paired comparison 활용).

### Experimental — 3 가지 robustness audit (paper-grade conclusion 의 *근본적 재정렬*)

**(1) 06 K=2 cross-dataset (NFCorpus, seed 42)**:

| 지표 | NFCorpus K=2 | SciFact K=2 (참고) |
|---|---|---|
| NDCG@10 all | **0.0801** | 0.6614 |
| Δ all vs baseline | **−0.250 ✗ negative** | +0.015 ✓ |
| Δ confused vs baseline | -0.071 ✗ | +0.039 ✓ |
| cos(v_0, v_1) | -0.66 (opposite) | +0.55 (partial) |

**SciFact 의 *K-invariant ceiling 0.6614* claim 이 cross-dataset 일반화 *불가*** — 동일 setup 이 NFCorpus 에서 catastrophic over-correction. Hyperparameter sensitivity dataset-specific.

**(2) 08 r=8 seed × 3 (42, 1337, 2024)**:

| 지표 | Seed 42 | Seed 1337 | Seed 2024 |
|---|---|---|---|
| NDCG@10 all | 0.6439 | 0.6446 | 0.6446 |
| Δ confused vs baseline | **+0.054 ✓** | -0.001 (≈) | -0.001 (≈) |
| ‖UV^T‖_F | **2.61** | 0.085 | 0.100 |
| σ₁/σ₂ | **42 (rank-1)** | 3.4 | 3.4 |

**Seed 42 의 *+0.054 confused 학습 + rank-1 collapse (σ₁=2.60)* 은 *seed-specific artifact***. Seed 1337/2024 의 학습된 M ≈ identity (사실상 학습 안 됨). 3-seed 평균 Δ confused ≈ +0.017 (sub-significant). Small_random init 의 초기 UV^T 방향 + 80-query val 의 best-epoch noise 가 결정적.

**(3) 02 unfrozen ColBERT (seed 42, encoder LR=5e-5, 3 epochs)**:

| 지표 | Frozen 02 | **Unfrozen 02** |
|---|---|---|
| 학습 가능 params | 768 | **109.6 M** |
| Δ confused vs baseline | +0.044 ✓ | **+0.252 [+0.179, +0.328] ✓** |
| Δ confused vs 01b α=10 | -0.021 (≈) | **+0.188 ✓** |
| ‖v_learned‖ | 7.08 | 0.33 (휴면) |
| Train loss 종료 | 0.24 | 0.0042 |

**Δ confused +0.252 — 우리 모든 frozen-side method (max +0.054 of seed 42) 의 *5 ×***. *Frozen encoder 가 진짜 bottleneck 확정*. v 휴면 (0.33) + train 완벽 fit (0.0042) → encoder 가 모든 ranking signal 흡수.

### Changed
- `report/06_k_sweep_report.md` §6.5 신규 — NFCorpus cross-dataset robustness 결과 추가.
- `report/08_bilinear_M_minimal_report.md` §5.5 신규 — seed × 3 variance + "rank-1 collapse 가 seed 42 artifact" 진단.
- `report/02_final_layer_vector_report.md` §7.5 신규 — Unfrozen ColBERT robustness check 결과 (Δ confused +0.252).
- `REPORT.md` §7 *Robustness audit* 로 완전 재작성 — 세 가지 robustness check 통합 + paper narrative 의 근본적 재정렬.

**Paper narrative 의 *재정렬***:
| 옛 narrative | **새 narrative** |
|---|---|
| Translation-trap + form-change + distillation 의 wrong-lever 가 frozen-encoder representational limit 의 *정황 증거* | *직접 증거*: encoder unfreeze 가 Δ confused +0.252 의 5× lift. K-invariant ceiling 은 SciFact-specific. 08 의 rank-1 collapse 는 seed artifact. |
| Main contribution 후보: translation-trap algebraic + bilinear M | **새 main contribution 후보**: *Frozen ColBERT confused-slice 가 정확히 encoder representational limit 에 의해 bound* + *50 K budget 안의 LoRA on Φ 가 그 limit 의 어떤 부분을 회복하는지* 의 정밀 분석 |

---

## [2026-05-23#14]

### Added
- `data/e5_teacher/extract_train_queries.py` — E5-Mistral-7B-Instruct 의 train query 추출 utility. SciFact 809 train queries → `data/e5_teacher/e5_train_q_emb_scifact.pt` (4096-d fp16, 6.6 MB, MPS 인코딩 ~85 초).
- `data/e5_teacher/` (gitignored via `data/`): E5 teacher signal artifacts — `e5_train_q_emb_scifact.pt` (train q emb, 신규 추출) + `e5_topk_{scifact,nfcorpus,scidocs}.pt` (test split corpus + queries, *nlp_term_project/phase_04* 복사) + `e5_soft_labels.json` (legacy phase_02 sample, 45 SciFact qids).
- `src/train.py` 에 `train_bilinear_metric_distill()` + `_bilinear_val_pass()` 추가 — pairwise margin + λ × Margin-MSE distillation. `e5_qid_to_idx`, `e5_did_to_idx` 매핑 + cosine 의 batch 단위 lookup.
- `experiments/09_bilinear_M_e5_distill/{run.py, README.md, figures.py}`. SciFact, seed 42, r=8, LR=1e-4, λ_distill ∈ {0.1, 0.5, 1.0} 의 3-run sweep. 4 figures (ndcg_vs_lambda, rank_collapse_by_lambda, delta_ci_forest_kwise, train_curve_kwise).
- `report/09_bilinear_M_e5_distill_report.md`.

### Changed
- `REPORT.md` Abstract / §5c 신규 / §6.1 grid / §9 갱신: 09 결과 통합. 본 시점 총 실험 10 개 (00–09), 14 회 실행 (06 K-sweep 3 + 09 λ-sweep 3).

### Experimental — Stage 2 follow-up: E5 Margin-MSE distillation 이 *anchor regularizer* 로 잘못 작동

**`09_bilinear_M_e5_distill` λ-sweep (SciFact, seed 42, r=8)**:

| λ_distill | NDCG@10 all | Δ all vs baseline | Δ confused vs baseline | M structure |
|---|---|---|---|---|
| 0 (08) | 0.6439 | -0.003 (≈) | **+0.054** ✓ | rank-1 dom (σ₁=2.60) |
| 0.1 | **0.6509** | +0.005 (≈) | +0.019 ✓ | partial rank-2 (σ₁=0.46, σ₂=0.11) |
| 0.5 | 0.6451 | -0.001 (≈) | -0.002 (≈ baseline) | uniform tiny (M≈I) |
| 1.0 | 0.6453 | -0.001 (≈) | -0.002 (≈ baseline) | uniform tiny (M≈I) |

**핵심 발견**: λ ↑ 면 *anchor preservation* 개선 (all-slice baseline 수렴) 이지만 *confused lever 가 죽음* (+0.054 → +0.019 → ~0). Distillation 이 *M 을 identity 근처에 잡아두는 regularizer* — paper main 의 *반대 방향*. 진단: (i) E5 의 *noise teacher* (mined HN 에서 ~50% 가 e5_margin < 0), (ii) Margin-MSE 의 *scale mismatch* (student -0.7 vs teacher ×8 = -0.24 → loss 25 의 huge magnitude → λ=0.1 만 해도 effective gradient dominate), (iii) *Opposite direction lever* — 08 의 rank-1 σ₁ ≈ 2.6 의 single dominant axis 가 *informed subspace 활용*인데 distillation 이 이를 *축소*.

**Stage 2 종합 결론** (08 + 09): *form 자체* 의 변경 lever 도 *translation family ceiling 위로 못 감* — *frozen-encoder representational limit* 의 정황 증거. 다음 critical 검정: 10 r sweep (rank-collapse 의 r-dependence 직접 측정) + 18 LoRA on Φ (encoder-level upper bound).

### 기술적 noteworthy

- **Robust nohup**: 옛 run 이 terminal HUP 으로 죽는 사례 발견 → `nohup ... < /dev/null & disown; caffeinate -i -w $PID` 패턴 확립.
- **Scale tuning trade-off**: teacher_scale=8.0 의 한계 — teacher 의 *noisy signal* 자체는 변경 안 됨. MonoT5 / cross-encoder soft labels 의 future 가능성.

---

## [2026-05-23#13]

### Added
- `src/bilinear.py` — `BilinearMetric` class. $M = I + UV^\top$, $U, V \in \mathbb{R}^{128 \times r}$. `maxsim(q_emb, d_emb, d_mask)` + `diagonal_maxsim` 메서드 (frozen ColBERT 의 vanilla path 와 별도 metric module). zero / small_random init.
- `src/train.py` 에 `train_bilinear_metric()` + `_bilinear_score_queries()` + `_bilinear_val_pass()` 추가. translation-family hook 대신 *metric module 자체* 의 maxsim 으로 pairwise margin loss + val 평가.
- `experiments/08_bilinear_M_minimal/{run.py, README.md, figures.py}`. SciFact, seed 42, r=8, LR=1e-4. 5 figures (ndcg_vs_baselines, delta_ci_forest, M_spectrum, train_curve, UV_inner).
- `report/08_bilinear_M_minimal_report.md`.

### Changed
- `experiments/08_bilinear_M_minimal/README.md`: zero-init pathology (∂L/∂U ∝ V = 0 → 학습 불가) 문서화 + small_random init (std=10⁻²) 로 변경.
- `REPORT.md` Abstract / §6.1 grid / §6.2 / §9 갱신: 08 결과 통합.

### Experimental — Stage 2 partial fail: bilinear M r=8 도 translation family ceiling 못 넘음

**`08_bilinear_M_minimal` (SciFact, seed 42, r=8, LR=1e-4)**: NDCG@10 all = **0.6439** (≈ baseline 0.6464). Δ vs baseline confused **+0.054 [+0.013, +0.097] ✓** positive. Δ vs 01b α=10 all **-0.025 [-0.046, -0.005] ✗** negative. Δ vs 02/06 K=2/06 K=4 모든 anchor 와 통계 동등. **`form 변경` 도 ceiling 위로 가지 못함**. 단 학습된 M 의 effective rank 가 1 로 collapse — UV^T singular values = [2.60, 0.06, 0.035, ...] 의 dominant rank-1. r=8 의 latent capacity 미활용 — K-router 의 effective K collapse 와 평행 패턴. **Stage 2 *critical falsification* 결과는 *partial fail* — form 자체의 lever 부재인지 *optimization-driven rank collapse* 의 결과인지의 분리는 09 (E5 distill) + 10 (r sweep) 의 결과로 결정**.

### 기술적 noteworthy

- **Zero-init pathology**: $U = V = \mathbf{0}$ 의 BilinearMetric init 은 $\partial \mathcal{L}/\partial U \propto V$, $\partial \mathcal{L}/\partial V \propto U$ → 둘 다 0 → 학습 정지. 첫 실행에서 ‖[U;V]‖ epoch 후 변화 없음으로 확인. small_random init (std=10⁻², ‖UV^T‖_F ≈ 0.035, vanilla MaxSim 대비 < 0.1% relative deviation) 으로 해결.
- **LR sensitivity**: LR=1e-3 시 ‖[U;V]‖ 한 epoch에 8 배 폭증 (0.45 → 3.64) → val NDCG@10 epoch 1 의 0.45 catastrophic drop. LR=1e-4 로 학습 stable 화.

---

## [2026-05-23#12]

### Changed
- `experiments/06_two_directions/` → `experiments/06_k_sweep/` (rename). 옛 K=2 단일 proof-of-concept 의 *ad-hoc single point* 한계 보완 위해 K ∈ {2, 4, 8} sweep 으로 확장.
- `experiments/06_k_sweep/{run.py, README.md, figures.py}`: K argument (`--k`) 추가. Artifact path `outputs/06_k_sweep/{dataset}/seed_{seed}/k_{K}/`. K-agnostic direction / routing diagnostics (per-pair cos, effective K perplexity, π_max saturation).
- `report/06_k_sweep_report.md` 신규 (옛 `report/06_two_directions_report.md` subsume). 5 figures (ndcg_vs_k_bar, delta_ci_forest_kwise, routing_entropy_by_k, direction_redundancy_by_k, train_curve_kwise).
- `REPORT.md` §5 / §6.1 / §6.2 갱신: K-sweep 결과 반영.
- `README.md`: 디렉토리 트리 + Quick start 명령 K-sweep 으로 갱신.

### Experimental — Translation family ceiling 의 *K-invariant* 확정 + over-capacity 의 *anchor 손상*

**`06_k_sweep` (SciFact, seed 42, K ∈ {2, 4, 8})**:

| K | Params | NDCG@10 all | Δ all vs baseline | Δ confused vs baseline | Δ confused vs 02 K=1 |
|---|---|---|---|---|---|
| 2 | 3,074 | **0.6614** | +0.015 [+0.004, +0.026] ✓ | +0.039 [+0.017, +0.061] ✓ | -0.005 (CI 0 포함) |
| 4 | 6,148 | **0.6614** | +0.015 [+0.003, +0.028] ✓ | +0.045 [+0.024, +0.068] ✓ | +0.002 (CI 0 포함) |
| 8 | 12,296 | 0.6089 | **−0.038 [−0.067, −0.008] ✗** | +0.049 [+0.005, +0.092] ✓ | -0.005 (CI 0 포함) |

**K=2 와 K=4 의 NDCG@10 all 이 *문자 그대로 동일* (0.6614)** — 학습 가능 파라미터 2 배 증가에도 ceiling 위치 *완전 보존*. **K=8 은 over-capacity 로 *anchor 손상*** (Δ all baseline -0.038 ✗ negative). 모든 K 의 effective routing K ≈ 1.2-1.5 — *capacity collapse* + over-capacity 시 anchor 손상의 *해로움*. Translation family ceiling 의 multi-direction 차원 *K-invariant 확정*. 옛 ROADMAP 의 K↑ + router 표현력 + entropy reg 가설은 *capacity-only 진단* 으로 본 sweep 으로 직접 falsify — 진짜 lever 는 *form 변경* (Stage 2 bilinear M).

---

## [2026-05-23#11]

### Added
- `experiments/07_random_direction_scaled/{run.py, README.md, figures.py}` — Translation-trap falsification 의 Stage 1 실험. seed 42 Gaussian unit vector × α=10 을 layer 12 hook 으로 주입 (학습 무필요).
- `outputs/07_random_direction_scaled/scifact/seed_42/{config, env, runs, runs_scored, metrics_per_query, metrics_aggregate, v_random.pt, delta_vs_baseline.json, delta_vs_mean_diff_alpha10.json}`.
- `report/07_random_direction_scaled_report.md` + 3 figures (`direction_compare`, `delta_ci_forest`, `ecdf_compare`).

### Experimental — Direction-agnostic 가설 명확히 기각

**`07_random_direction_scaled` (SciFact, seed 42)**: NDCG@10 = 0.6485 (vs baseline 0.6464, mean-diff α=10 0.6690). Δ vs baseline confused +0.011 [-0.006, +0.029] (≈ 0). **Δ vs 01b α=10 confused -0.0533 [-0.0905, -0.0201] ✗ negative** — 같은 magnitude 인데도 random direction 은 baseline 과 통계 동등, mean-diff direction 만 ceiling 도달. *Translation family 안에서 direction-agnostic 가설* 의 결정적 falsification. 외부 피드백의 *(A) translation family algebraic 분류* 는 유효하나 *(B) direction-agnostic* 부분은 기각. ROADMAP conditional graph 의 *partial fail* 분기로 진입 — 08 bilinear M 의 critical 검정 유지 + 옛 deferred (mean_diff_pca / projection_out) 일부 *informed direction subspace 의 다른 element* 로서의 가치 부분 회복.

### Changed
- `REPORT.md` Abstract: 8 개 실험 narrative + "informed direction subspace 의 representational limit" 진단 + bilinear M 을 next critical test 로 명시.
- `REPORT.md §5`: 헤더 "Multi-direction router (06) + direction-agnostic falsification (07)" 로 변경 + §5.4 신규 (07 결과 + paired bootstrap CI + Figure 6 + 함의).
- `REPORT.md §6.1`: 그리드에 07 행 추가; §6.2 핵심 결론에 07 의 *direction 의 내용이 lever* + *informed redundancy* 통합; §6.3 paper main contribution 의 translation-family + informed-subspace 진단으로 재정렬.
- `REPORT.md §9-11`: Next experiments 를 08 bilinear M minimal (CRITICAL) + 09 E5 distill + 10 r sweep + 11–12 cross-dataset 로 재구성; 보고서 링크에 07 추가; 본 시점 총 실험 8 개로 갱신.
- `RESEARCH.md` 에 `[2026-05-23#8]` dated entry 추가 (07 결과 + decisions + 08 next).

---

## [2026-05-23#10]

### Added
- `src/lsr.py:MultiDirectionSteeringModule` — K learnable directions + per-token softmax router. K=2 의 경우 3,074 params (2K·D + K).
- `experiments/06_two_directions/{run.py, README.md, figures.py}` — multi-direction 단계 main novelty 의 proof-of-concept. K=2 + softmax router at layer 12.
- `report/06_two_directions_report.md` — 결과 + routing analysis + 3 figures + 다음 실험 결정 근거.

### Experimental — multi-direction 단계 K=2 unaided 으로 ceiling 우회 불가

**`06_two_directions` (SciFact, seed 42)**: NDCG@10 = 0.6614. Δ vs baseline confused +0.039 ✓. Δ vs 02/04/05 모두 통계적 동등 (CI 0 포함). Routing: π_mean=[0.238, 0.762], entropy=0.342/0.693, **91% tokens 가 π_max>0.6** — router 가 거의 binary. cos(v_0, v_1) = 0.553 — 부분 redundant. **결론**: K=2 의 capacity 가 *underused* (effective K ≈ 1.2-1.4). multi-direction 단계 main contribution 통과 조건으로 (a) K ↑ (07_k_sweep), (b) router 표현력 / regularization (22, 23) 의 조합 필요.

### Changed
- `ROADMAP.md`: §"Next" reordering — 옛 06_projection_out 후순위로 이동, 07_two_directions → 06_two_directions 으로 격상 (sequential numbering 유지). multi-direction 단계 우선순위 ↑.

---

## [2026-05-23#9]

### Added
- `src/lsr.py:MultiLayerSteeringModule` — 5 layer × 768 = 3,840 trainable params. `register_all(model)` 로 다층 hook 일괄 등록.
- `experiments/05_five_layers/{run.py, README.md, figures.py}` + `report/05_five_layers_report.md`. SciFact 학습 + 5 figures (train_curve, layer_norms_bar, delta_ci_forest, ecdf_compare, single_direction_summary).

### Changed
- `src/train.py`: anchor reg 와 v_norm 추적을 multi-parameter compatible 로 (모든 trainable params 의 L2 합산).
- `ROADMAP.md`: Sequential renumbering 적용 (08→03, 11→04). Master plan 의 *완료 / Next / Deferred* 3-section 구조. ROADMAP changelog 에 single-direction 단계 ceiling 확정 + multi-direction 단계 격상 entry.
- `DESIGN.md §6`: 새 sequential ID 와 일치하도록 ablation matrix renumber. Done / next / deferred 상태 컬럼 추가.

### Experimental — single-direction 단계 ceiling 확정

**`05_five_layers` (SciFact, seed 42)**: NDCG@10 = 0.6502. Δ vs baseline confused +0.051 ✓ positive. Δ vs 02 (single-layer): all -0.015 / confused +0.007 — **통계적 동등** (CI 0 포함). Δ vs α=10: 통계적 동등. Per-layer ‖v_ℓ‖ = 1.27 / 1.52 / 2.22 / 2.85 / 2.79; cos(v_ℓ, v_mean_diff_l12) 는 ℓ12 만 0.27, 나머지 ≈ 0 (직교). 5 layer 학습이 *다른 axis* 시도했으나 retrieval 측면 같은 ceiling. **Multi-layer 단순 확장이 single-direction ceiling 못 넘음 — direction 의 *수* + selectivity (router) 가 진짜 lever 임을 강력 시사 → multi-direction 단계 multi-direction router 의 empirical motivation 결정적 sharpened.**

---

## [2026-05-23#8]

### Added
- `src/lsr.py:ScalarGatedSteeringModule` — $h - g \cdot v$, $g = \sigma(b)$. Anchor preservation init: $v=\mathbf{0}$, $b=-3$. 769 학습 가능 파라미터.
- `src/lsr.py:PerTokenGatedSteeringModule` — $h - g(h_t) v$, $g(h_t) = \sigma(W h_t + b)$. 1537 학습 가능 파라미터.
- `experiments/03_scalar_gate/{run.py, README.md, figures.py}` + `report/03_scalar_gate_report.md`. SciFact 학습 완료. 3 figures.
- `experiments/04_per_token_gate/{run.py, README.md}` + `report/04_per_token_gate_report.md`. SciFact 학습 완료. gate 분포 통계 캡처 utility.
- memory `feedback_autonomous_progression.md` — 후속 실험 자율 진행 directive.

### Experimental — single-direction 단계 single-layer trio (02 / 08 / 11) ceiling 확정

- **03_scalar_gate (SciFact, seed 42)**: NDCG@10 = 0.6448. Multiplicative gradient saturation 으로 $g \cdot \|v\| = 0.23$ 만 학습 → effective magnitude 가 01b 의 α≈0.5 보다 작아 효과 미발현. Δ vs baseline (CI 0 포함, 통계적 동등), Δ vs 02 (-0.020 ✗ negative). **single-direction 단계 narrative 의 *gate 형식 한계* 증거**.
- **04_per_token_gate (SciFact, seed 42)**: NDCG@10 = 0.6641. Per-token gate 가 *모든 token 에서 1.000 ± 0.001* 로 saturated → 사실상 02 와 등가. 02 의 all-slice Δ 가 이미 양수 → gate 가 anchor preservation 할 의무 없음 → gate 가 closed 될 동기 부재 → always-on 으로 수렴. Δ vs 02 ≈ 0 (통계 동등). **본 결과는 *single direction subspace 의 redundancy* + *capacity 부족* 의 데이터적 증거 — multi-direction 단계 (multi-direction router) 의 empirical motivation sharpened.**

### Changed
- ROADMAP-level priority 재배치: 12 (gate capacity), 03/04/05/06/09/10 (단일 layer 변형) 우선순위 ↓. 13 (five_layers), 17 (projection_out), multi-direction 단계 (multi-direction) 우선순위 ↑.
- single-direction 단계 의 *ceiling* 발견 (single-layer single-direction ≈ 0.665) 으로 인해 multi-direction 단계 의 contribution 가치 강화 — paper main novelty 로 positioning 더욱 명확.

---

## [2026-05-23#7]

### Added
- `src/lsr.py` — `SteeringModule` (단일 학습 direction `v ∈ ℝ^768`, zero-init, broadcasting subtract, hook closure 제공)
- `src/train.py` — `train_steering` 함수 + `TrainConfigLite` + `TrainHistory` dataclass. Pairwise margin loss + λ_anc support + val NDCG callback + early stopping. Reusable for 08/11/13 etc.
- `src/colbert_hook.py:ColBERTv2.diagonal_maxsim` — per-row MaxSim for training (B 개 query-doc 쌍의 diagonal)
- `experiments/02_final_layer_vector/{run.py, README.md, figures.py}` — single-direction 단계 첫 학습 실험. SciFact 학습 완료. 5 figures (train_curve, delta_ci_forest, delta_violin, ecdf_compare, direction_compare).
- `report/02_final_layer_vector_report.md` — 통합 분석 + 5 figure 임베드.
- memory `feedback_autonomous_progression.md` — 후속 실험 자율 진행 권한 + ROADMAP 수정 권한 + stopping 조건.

### Changed
- `src/colbert_hook.py`: `encode_queries`, `encode_docs` 의 `@torch.no_grad()` 데코레이터 제거 (training 시 gradient 통과 허용). 기존 eval caller (encode_corpus, score_queries) 가 자체 no_grad 컨텍스트 보유 → 행동 무변경.
- `DESIGN.md §11`: **single-direction 단계 (02-15) 의 anchor reg deviation** mirror. 기본 `λ_anc=0` 으로 설정 (gate 부재 → reg 형식 부적용). λ_anc sweep 은 19 에서 진행.

### Experimental
- **`02_final_layer_vector` (SciFact, seed 42)**: NDCG@10 = 0.6651. Δ vs baseline: all +0.019, confused +0.044 (모두 CI > 0). Δ vs 01b α=10 sharpened anchor: 통계적 동등 (CI 0 포함). cos(v_learned, v_mean_diff) = 0.32 → H5 qualitative 통과 (다른 방향), 그러나 같은 성능 → 단일 direction 의 redundancy. ‖v‖ epoch 마다 단조 증가, train-overfitting 패턴 관찰. Narrative 함의: 후속 gate / per-token / multi-direction 의 empirical motivation 확보.

---

## [2026-05-23#6]

### Added
- `src/hn_mining.py` — `mine_triplets`, `unique_dids` utility (재사용 가능)
- `src/mean_diff.py` — `encode_doc_layer12_means` + `compute_v` 공유 유틸 (01 + 01b 공유)
- `experiments/01_mean_diff/{run.py, README.md, figures.py}` — baseline 단계 의 raw mean-diff 실험. SciFact / NFCorpus / FiQA 실행 완료. 통합 figures.py 가 raw + sweep artifact 모두 읽음.
- `experiments/01b_mean_diff_scaled/{run.py, README.md}` — baseline 단계 의 magnitude sweep sub-experiment. SciFact 에서 α ∈ {0.5, 1, 2, 5, 10} 5-회 실행.
- `report/01_mean_diff_report.md` — 통합 보고서 (raw 3 dataset + sweep 1 dataset). 6 개 figure 임베드.
- `report/figures/01_mean_diff/{raw_delta_ci_forest, raw_v_norm, alpha_sweep_curve, alpha_sweep_forest, alpha_sweep_ecdf, alpha_sweep_violin}.{pdf,png}`

### Changed
- `ROADMAP.md`: baseline 단계 의 `01_mean_diff` entry 에 `01b` sub-experiment 명시. Changelog 에 entry 추가.
- `experiments/01_mean_diff/README.md`: 통합 보고서 위치 명시 (01b 와 결합).
- `experiments/01b_mean_diff_scaled/README.md`: 보고서 위치를 통합 보고서로 변경 (단독 보고서 삭제).

### Experimental
- **`01_mean_diff` (raw, 3 dataset, seed 42)**: 세 dataset 모두 |Δ NDCG@10| ≤ 0.001. v_norm 0.03–0.27 의 작은 magnitude 가 효과 미발현의 원인 의심 → 01b 수행.
- **`01b_mean_diff_scaled` (SciFact, α=0.5/1/2/5/10, seed 42)**: α ≥ 2 부터 confused-slice paired bootstrap CI 가 0 명확히 초과. α=10 에서 confused +0.064 (baseline 의 +14 % 상대 개선), all-slice 도 +0.0226 (anchor preservation 자연 유지). C-form 가설 기각 (subtract 형식은 informative direction + 적절한 magnitude 면 작동). 02 의 학습된 baseline anchor 가 *informed α=10 mean-diff* 로 sharpening.

---

## [2026-05-23#5]

### Added
- `ROADMAP.md` — 32 실험 master sequence 의 single source of truth. 결과 기반 priority 구조, core 17 + extended 15. `15_mean_diff` → `01_mean_diff` promotion (final 수정 반영, critical analysis 근거).
- `CLAUDE.md §5` 디렉토리 트리에 ROADMAP.md 명시
- memory: `roadmap_pointer.md` — ROADMAP 의 위치 + 구조 요약 (cross-conversation persistence)

### Changed
- `experiments/00_baseline/README.md` + `report/00_baseline_report.md`: C3 punctuation mask 가설을 set 비교로 사전 검정 → **기각**. 잔여 gap 의 원인 아님. 남은 후보 (C7 transformers 버전 / C8 PLAID / C9 MPS 정밀도) 는 분리 검정 비용 대비 효익 낮음 — *documented limitation* 으로 수용 + 후속 LSR 진행.
- `RESEARCH.md` 에 `[2026-05-23#2]` dated entry 추가 — C3 기각 + Roadmap commit + baseline 단계 의 `01_mean_diff` 를 다음 실험으로 commit.

### Experimental
- C3 punctuation 가설 set 비교 결과: 본 구현과 공식 ColBERT v2 식이 *완전 동일* punctuation ID set (size 32) 산출. C3 기각.

---

## [2026-05-23#4]

### Added
- `experiments/00_baseline/run.py` 와 `figures.py` — 첫 *완전한* 실험 entry point (CLAUDE.md §5.1 + §3.9 충족). 한글 docstring.
- `experiments/00_baseline/README.md` — 실험 카드 (purpose / hypothesis / success criterion / status / open issue)
- `report/00_baseline_report.md` — 상세 측정 결과 + 분석 + 4 개 figure 임베드. 6 BEIR dataset (SciFact / NFCorpus / SciDocs / TREC-COVID / FiQA-2018 / ArguAna) × seed 42 측정 완료.
- `report/figures/00_baseline/{metrics_paper_overlay, metric_at_k_curves, per_query_metric_dist, confused_slice_size}.{pdf,png}` — CLAUDE.md §16.2 baseline 카탈로그 일체
- `src/evaluate.py:score_queries` 의 `exclude_self: bool` 옵션 — ArguAna 의 self-doc retrieval 제외 (counter-argument task 의 표준 BEIR/ColBERT 평가 컨벤션)
- `CLAUDE.md §3.9` 추가 규약: 생성한 figure 는 보고서 본문에 markdown image link 로 직접 임베드 필수. 생성 후 미참조 상태는 §3.9 위반.
- `CLAUDE.md §5` 디렉토리 트리: `report/` 의 per-experiment / per-experiment 보고서 layout 명시 (`{NN}_{exp_name}_report.md` 컨벤션)
- `CLAUDE.md §16.6` 완전성 검사: figure 가 `report/{NN}_{exp_name}_report.md` 본문에 임베드되어야 *완전한* 실험으로 간주

### Changed
- `src/utils/io.py:artifact_dir` 시그니처: `(stage, config_id, ...)` → `(exp_name, ...)`. 출력 경로 `outputs/02_evaluate/T1.00_baseline/...` → `outputs/00_baseline/...` (experiment directory 명과 1:1 대응)
- `src/data.py:BEIR_DATASETS`: 3 dataset (scifact/nfcorpus/scidocs) → 6 dataset (+ trec-covid / fiqa / arguana). 도메인 균형 (의료 2 + 과학 2 + 금융 1 + 논증 1).
- `src/colbert_hook.py`: (a) `linear.weight` 를 ColBERT v2 checkpoint 의 safetensors 에서 직접 로드 (AutoModel 이 무시하는 키), (b) `[Q]`/`[D]` marker 를 `[unused0]`/`[unused1]` 로 매핑 (새 token 추가하지 않음), (c) query 의 [MASK] padding 위치에 `attend_to_mask_tokens=False` (artifact.metadata 확인), (d) score_mask 와 attention_mask 분리.
- `src/data.py:doc_text` 의 separator: `". "` → `" "` (이중 마침표 회피, ColBERT 컨벤션과 일치)

### Experimental
- **`00_baseline` 6 dataset 실행 (seed 42)** → SciDocs 만 paper ±0.005 통과 (Δ +0.004). SciFact 가 유일한 큰 outlier (Δ −0.047). 나머지 4 dataset 모두 ~−0.01 gap (시스템적 minor 차이 의심, 후보 1 순위: C3 punctuation mask 구성). ArguAna 는 `exclude_self` 적용 후 0.3337 → 0.4528. 상세: `report/00_baseline_report.md`.

---

## [2026-05-23#3]

### Added
- `.python-version` = `3.14.4` (project Python interpreter pin)
- `.venv/` Python 3.14.4 virtual environment (git-ignored). 설치 직후 import smoke test 통과 (`src.utils.*`, `src.configs`, `src.metrics`, `src.slices`, `src.data`, `src.colbert_hook`).
- `requirements.lock.txt` — `pip freeze` 결과 (68 packages). Reproducibility tier 1: 모든 transitive deps 의 exact version 보존.
- `.gitignore` — `.venv/`, `__pycache__/`, `data/`, `outputs/`, `*.ckpt` 등 표준 ignore 패턴

### Changed
- `requirements.txt`: Python 3.14 wheel 호환을 위해 상한 제거 (`torch>=2.5.0`, `numpy>=2.0`). 실제 설치 버전은 `requirements.lock.txt` 참조 (torch 2.12.0, transformers 5.9.0, numpy 2.4.6, beir 2.2.0, scikit-learn 1.8.0 등)
- `CLAUDE.md §12`: 환경 설정 절차를 Python 3.14 venv + lock 파일 워크플로우로 갱신
- `DESIGN.md §8` reproducibility checklist: Python version pin + lock 파일 체크박스 완료 처리

### Experimental
- (없음 — 코드 검증 미실행)

---

## [2026-05-23#2]

### Added
- `requirements.txt` — version-pinned 의존성 (torch 2.2.x–2.4.x, transformers 4.40+, beir 2+, numpy < 2 등). 첫 실험 후 정확 pin 으로 좁힐 예정.
- 디렉토리 scaffold: `src/{utils/}`, `experiments/00_baseline/`, `outputs/`, `data/`, `report/figures/`
- `src/utils/repro.py` — `set_seed`, `get_device`, `SEEDS = (42, 1337, 2024)` (CLAUDE.md §3.7, §8 지원)
- `src/utils/io.py` — `artifact_dir`, `save_json` / `load_json`, `save_pickle` / `load_pickle`. 모든 artifact 의 경로 규약 `outputs/{stage}/{config_id}/{dataset}/seed_{seed}/` 통일
- `src/utils/logging.py` — 일관 logger factory
- `src/configs.py` — declarative `ExpConfig` (`SteeringConfig` / `TrainConfig` / `EvalConfig`) + YAML/JSON serialization, 첫 baseline / default-5L / single-L6 config registry. DESIGN.md §3·§4 의 모든 choice 가 dataclass field 로 대응
- `src/data.py` — BEIR loader (`scifact` / `nfcorpus` / `scidocs`), `ensure_dataset` auto-download, `load_beir`, `doc_text`, `build_pos_pairs`. CLI `python -m src.data --extract` 지원
- `src/metrics.py` — `ndcg_at_k`, `mrr_at_k`, `recall_at_k`, `map_at_k`, `compute_per_query_metrics`, `aggregate_mean`, `paired_bootstrap_ci`, `align_per_query`. Bootstrap iteration 기본 10K (CLAUDE.md §3.7 / §8)
- `src/slices.py` — `confused_slice` (DESIGN.md §5.1 default + T2B.14 ablation). `lexical_hn` / `hard_hn` 은 prior repo 정의 확정 전까지 의도적 미구현 (TODO 명시)
- `src/colbert_hook.py` — frozen ColBERT v2 wrapper (`ColBERTv2`): bert-base + 768→128 projection 로드, [Q]/[D] marker, query mask padding, doc punctuation masking, MaxSim, layer-wise forward hook infrastructure (`register_layer_hook(ℓ, fn)` for `ℓ ∈ {0..12}`). 모든 parameter `requires_grad_(False)` + encoder eval mode default (DESIGN.md §3.4)

### Changed
- `CLAUDE.md §5` 디렉토리 트리: `src/utils/` 서브패키지 + `metrics.py` / `slices.py` 명시

### Experimental
- (없음 — 코드 검증 미실행. 첫 실험: `T1.00_baseline` 에서 paper-reported NDCG 재현 검증 필요)

---

## [2026-05-23]

### Added
- `CLAUDE.md` 초안 (§1–§14): 프로젝트 헌법 — 방법론적 제약 / 아키텍처 stance / 보고서 style / 통계 robustness / 자주 피하는 함정 / Claude 협업 지침
- `CLAUDE.md §1.1`: 학부 NLP 수업 텀 프로젝트 맥락 + **저널 투고 목표** (TACL / ACL / EMNLP / SIGIR / CIKM / NAACL 급) 명시
- `CLAUDE.md §3.8`: **Ablation completeness 철칙** — 모든 architectural / methodological choice 는 도입과 동시에 대응 ablation 이 설계되어야 함
- `CLAUDE.md §3.9`: **Documentation freshness 철칙** — `CLAUDE.md` / `DESIGN.md` / `RESEARCH.md` / `CHANGELOG.md` 4 개 문서의 절대 stale 금지 규약
- `CLAUDE.md §14.1`: Claude Code 의 *학술적 연구 assistant* 역할 명시 + journal-grade output quality 기준
- `DESIGN.md` 초안: §1 RQ, §2 hypotheses, §3 architecture, §4 training protocol, §5 evaluation protocol, §6 ablation matrix (T1 / T2A / T2B / T3 + §6.5 choice↔ablation mapping), §7 risk register, §8 reproducibility checklist, §9 notation, §10 references, §11 design changelog
- `DESIGN.md §6.0`: Ablation completeness invariant 의 design-level 구현 규약
- `DESIGN.md §6.5`: Choice ↔ Ablation 1:1 mapping table (§3.8 invariant enforcement)
- `RESEARCH.md` 초안: 외부 제출용 lab notebook (현 상태 — title heading 만; 첫 실험 실행 후 entry append)
- `CLAUDE.md §15`: `RESEARCH.md` 작성 규칙 (entry 골격 + tone 규칙) 을 본 헌법으로 이전, RESEARCH.md 자체는 외부 제출용으로 self-contained 유지
- `CHANGELOG.md` (본 문서)

### Changed
- `CLAUDE.md §3.7`: statistical robustness 에 bootstrap iteration 수 (`n=10,000`) 명시 (§8 와 정합)
- `CLAUDE.md §5`: 디렉토리 구조에 `RESEARCH.md` / `CHANGELOG.md` 추가

### Experimental
- (없음 — 실험 미시작)
