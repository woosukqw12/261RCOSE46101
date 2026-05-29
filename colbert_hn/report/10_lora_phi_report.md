# 10_lora_phi — LoRA on Φ (ColBERT encoder), encoder representational limit 의 부분 회복

본 보고서는 Robustness audit (2026-05-23) 의 02 unfrozen 실험에서 발견된 **encoder representational limit** (Δ confused +0.252 의 5× lift) 의 *50K-budget-relaxed* LoRA adapter 로의 *부분 회복* 분석. Phase 1 → 2a → 2b 의 3-단계 sweep 의 *anchor preservation* 회복 trajectory 정밀 진단.

**결론**: Phase 2b (q,v r=8 LR=5e-5 α=r, 294,912 params) 의 3-seed mean **Δ confused +0.104 ± 0.017 ✓ (3 seeds 모두 statistically significant, anchor preserved)**. *Bounded improvement* 달성 + *robust*. *Strict 돌파 (CI(Δ all)>0)* 는 3 seeds 모두 미달 — pre-commit 따라 *hyperparameter sweep 중단*. 본 paper 의 *진정한 main contribution* 은 §8.5 의 *universal rank-collapse + spatial multiplicity escape* 통합 진단 — 06/08/10 의 cross-method 분석.

## 1. 동기 + Pre-committed 판정 기준

### 1.1 Encoder-limit motivation (robustness audit 결론)

02 unfrozen (ColBERT 110M params, encoder LR=5e-5, 3 epochs) 의 **Δ confused +0.252 [+0.179, +0.328] ✓** — 우리 모든 frozen-side method (max +0.054 of 08 seed 42) 의 ~5×. *Frozen-encoder 가 진짜 bottleneck* 의 직접 증거. 본 실험은 LoRA (Hu et al. 2021) adapter 로 그 lift 의 *얼마나* 회복 가능한지 정밀 검정.

### 1.2 LoRA 형식

각 BERT attention 의 학습 가능 선형 변환 (q, k, v, o) 에 rank-r additive adapter:
$$h = W x + (\alpha / r) B A x, \quad A \in \mathbb{R}^{r \times d}, \ B \in \mathbb{R}^{d \times r}$$

Init: $A \sim \mathcal{N}(0, 0.02^2)$, $B = \mathbf{0}$ → $BA = 0$ at $t=0$ → *정확히 baseline retrieval* (anchor preservation init).

학습 파라미터: $2 r d \times |\text{components}| \times |\text{layers}|$. ColBERT BERT-base ($d=768$, 12 layers, q+v all layers):

| r | Params | 옛 50K budget |
|---|---|---|
| 1 | 36,864 | ✓ |
| 8 | 294,912 | ✗ (사용자 결정으로 budget 완화) |

### 1.3 Pre-committed 판정 기준 (외부 reviewer 입력 반영, 결과 보기 전 commit)

- **Early-stop 기준**: `val_all` (post-hoc cherry-picking 회피 위해 *historical* `val_conf` 에서 변경, 모든 config 동일 적용)
- **돌파 판정**:
  $$\text{돌파} \iff \text{CI 하한}_{\Delta \text{NDCG@10 all vs baseline}} > 0$$
- *그 외 metric (confused-slice, M structure) 은 부차*.
- 미돌파 시 → *hyperparameter sweep 금지* (9K SciFact triplet 의 data 한계가 hyperparameter 문제 아님). *Safety-net narrative* (bounded improvement framing) 즉시 채택.

## 2. 3-단계 sweep 결과

### 2.1 Phase 1: minimal config (r=1, 36,864 params)

| 항목 | 값 |
|---|---|
| Components | q, v |
| Layers | all 12 |
| LoRA rank r | 1 |
| Alpha α | r (scaling=1) |
| LoRA LR | 5e-5 (typical BERT finetune) |
| Steering hook | frozen v=0 (no-op at layer 12) |
| Epochs / Patience | 3 / 2 |
| Time | ~17 분 (3 × 215s/epoch + val) |

**결과** (val_all-based best @ ep2):

| 지표 | 값 |
|---|---|
| NDCG@10 all (test) | 0.5940 |
| NDCG@10 confused (self) | 0.2309 |
| **Δ all vs baseline** | **-0.052 [-0.085, -0.021] ✗ negative** |
| **Δ confused vs baseline** | +0.038 [-0.008, +0.083] (CI 0 포함, not significant) |
| ‖A‖_total / ‖B‖_total | 4.68 / 1.87 |

**진단**: r=1 capacity 가 *불충분* — anchor 손상 + confused lift 도 sub-significant. 다음: rank 증가.

### 2.2 Phase 2a: rank 증가 + aggressive LR (r=8, 294,912 params, LR=1e-4, α=2r=16)

50K budget 완화 후 *aggressive* sweep — capacity + LR 동시 증가. *Single-variable 위반* 이지만 *informative single-point* 로 의도적.

**결과** (val_conf-based best, *옛* default — Phase 1 의 best @ ep2):

| 지표 | 값 |
|---|---|
| NDCG@10 all | 0.5879 |
| NDCG@10 confused (self) | 0.1849 |
| Δ all vs baseline | **-0.059 [-0.101, -0.017] ✗ negative** |
| **Δ confused vs baseline** | **+0.080 [+0.021, +0.140] ✓ positive** |
| Δ confused vs 02 unfrozen | -0.172 [-0.240, -0.108] ✗ |
| ‖A‖_total / ‖B‖_total | 9.03 / 2.24 |

**진단**:
- *First* statistically significant confused lift (+0.080 ✓) — frozen-side max (+0.054) 의 1.5×.
- 하지만 **anchor 더 손상** (-0.052 → -0.059). LR=1e-4 + scaling 2 의 *aggressive* combo 가 over-correction.
- *Pre-commit* 기준 미달: Δ all CI 하한 -0.101 < 0.

→ 다음 (B): LR + α 보수화로 *anchor preservation 회복* 시도.

### 2.3 Phase 2b (B): rank 유지 + 보수적 LR + scaling=1 (r=8, LR=5e-5, α=r=8)

*Single-variable 분리*: r=1 (Phase 1) → r=8 의 *순수 rank 효과*. LR / scaling 동일.

**결과** (val_all-based best @ ep2):

| 지표 | 값 |
|---|---|
| NDCG@10 all | 0.6367 |
| NDCG@10 confused (self) | 0.2454 |
| **Δ all vs baseline** | **-0.010 [-0.044, +0.023] (CI 0 포함, 통계 동등)** |
| **Δ confused vs baseline** | **+0.091 [+0.040, +0.143] ✓ positive** |
| Δ confused vs 02 frozen | +0.048 [-0.005, +0.099] (CI 0 포함) |
| Δ confused vs 02 unfrozen | -0.161 [-0.233, -0.090] ✗ |
| ‖A‖_total / ‖B‖_total | 9.07 / 2.13 |

**진단**:
- **Anchor preservation 회복** (CI 0 포함) — Phase 2a 의 -0.059 손상 해소.
- **Δ confused +0.091 ✓ — Phase 2 의 *최고 confused lift***. Frozen-side max (08 seed 42 의 +0.054) 의 ~1.7×, *통계 유의*.
- 02 unfrozen 의 +0.252 의 **36%** 회복 (295K params = encoder 의 0.27% 로).
- *Pre-commit strict 기준 미달*: CI 하한 -0.044 < 0 → 돌파 *아님*.

### 2.5 NFCorpus cross-dataset (Phase 2b 동일 config, 2026-05-24)

06 K=2 NFCorpus 의 catastrophic 교훈 (Δ all −0.250 ✗) 적용 검정 — *Phase 2b config 가 cross-dataset transfer 되는가*. 동일 setup (q,v r=8 LR=5e-5 α=r early-stop=val_all) + `--max-triplets 9,190` 으로 SciFact-comparable scale.

| 지표 | SciFact (3-seed mean) | **NFCorpus** |
|---|---|---|
| NDCG@10 all | 0.6476 | **0.0094** (baseline 0.330 의 2.8 %) |
| Δ all vs baseline | +0.001 (≈) | **−0.320 [−0.355, −0.287] ✗** catastrophic |
| Δ confused vs baseline | +0.104 ✓ | **−0.092 [−0.115, −0.070] ✗** |
| Ep1 rank loss | 0.66 | **4.47** (7×) |
| Ep1 val_all | ~0.72 | **0.073** |

**결과**: 06 K=2 NFCorpus 의 −0.250 catastrophic 보다 *더 심함* (−0.320). NFCorpus 의 *strong-HN regime* 이 LR=5e-5 의 same config 에서 immediate over-correction 유도.

**진단 + paper-grade limitation**:
- Phase 2b 의 numerical lift (+0.104) 는 *SciFact-tuned hyperparameter 의 결과*.
- *Universal rank-collapse + spatial multiplicity escape* 의 method-architectural claim 은 *cross-dataset 일 것* (학습 동학 의 universal pattern).
- Cross-dataset 의 동일 *numerical* lift 는 *adaptive hyperparameter strategy* (per-dataset normalization) 가 future work.

### 2.6 FiQA cross-dataset (Phase 2b 동일 config, 2026-05-24)

NFCorpus 의 catastrophic 결과 가 *single dataset 의 artifact* 인지 검정 — *3 번째 train-available BEIR (FiQA, 5,500 train query)* 에 동일 Phase 2b config 적용.

| 지표 | SciFact (3-seed) | NFCorpus | **FiQA** |
|---|---|---|---|
| NDCG@10 all | 0.6476 | 0.0094 | **0.0005** (baseline 0.347 의 0.15 %) |
| Δ all vs baseline | +0.001 (≈) | −0.320 ✗ | **−0.347 [−0.374, −0.319] ✗** catastrophic |
| Δ confused vs baseline | +0.104 ✓ | −0.092 ✗ | **−0.147 [−0.166, −0.127] ✗** |
| Ep1 rank loss | 0.66 | 4.47 | **~3.8** |

**결과**: 2 / 2 cross-dataset (NFCorpus + FiQA) 가 catastrophic. SciFact-tuned LR=5e-5 의 *cross-dataset generality 명확히 부정*. *Single-dataset artifact* 가설 기각 — *systematic cross-dataset failure* 확정.

### 2.7 Diagnostic B — *encoder output representation collapse* (2026-05-24)

**가설** (외부 reviewer feedback 후): catastrophic NDCG (NFCorpus −0.320, FiQA −0.347) 의 mechanism 은 *encoder output (token embedding) space* 의 *representation collapse* — docs 가 평균적으로 너무 비슷해져서 MaxSim discrimination 불가.

**§5e (다음 §8.5) 의 parameter-space ΔW rank-collapse 와 다른 현상**:
- §5e: *parameter* space (ΔW = BA) 의 spectrum collapse → *학습 dynamics 의 universal feature*.
- §2.7: *output* space (token embedding) 의 spectrum collapse → *catastrophic 의 *mechanism* (output 의 직접 측정)*.

**측정** (`report/_repr_collapse_diagnostic.py`, n=500 docs 샘플 per corpus):
1. Random doc-pair 평균 cosine (mean-pooled, L2-normed). collapse 시 ↑.
2. Per-token random pair cosine. collapse 시 ↑.
3. Doc-mean matrix (N×128) effective rank (singular-spectrum perplexity). collapse 시 ↓.
4. Per-token (10K subsample × 128) effective rank. collapse 시 ↓.

| Dataset | 조건 | doc-pair cos μ ± σ | tok-pair cos μ ± σ | eff_rank (doc) | eff_rank (tok) |
|---|---|---|---|---|---|
| NFCorpus | frozen (no LoRA) | +0.553 ± 0.111 | +0.211 ± 0.144 | 11.73 | 55.94 |
| NFCorpus | **LoRA Phase 2b** | **+0.990 ± 0.005** | **+0.940 ± 0.029** | **1.09** | **1.62** |
| FiQA | frozen (no LoRA) | +0.380 ± 0.149 | +0.181 ± 0.132 | 23.58 | 63.91 |
| FiQA | **LoRA Phase 2b** | **+0.993 ± 0.005** | **+0.990 ± 0.006** | **1.06** | **1.10** |
| SciFact | frozen (no LoRA) | +0.573 ± 0.117 | +0.209 ± 0.144 | 10.65 | 57.21 |
| SciFact | **LoRA Phase 2b** | **+0.984 ± 0.010** | **+0.942 ± 0.028** | **1.15** | **1.61** |

![Diagnostic B representation collapse](../report/figures/_repr_collapse/repr_collapse.png)

*Figure: 4 rows × 3 datasets. Row 1 — doc-pair cosine histogram (frozen blue vs LoRA red overlay); Row 2 — token-pair cosine histogram; Row 3 — doc-mean singular spectrum (log scale, top-30); Row 4 — token singular spectrum. LoRA Phase 2b 의 *모든* 패널 이 frozen 대비 *극단적 cosine ↑ + spectrum collapse*.*

**관찰 — universal collapse, dataset-dependent consequence**:
1. **Universal extreme collapse** — 3 dataset 모두 LoRA Phase 2b 에서 doc-pair cosine ≈ 0.99, eff_rank ≈ 1. 즉 corpus 의 모든 doc 이 *사실상 단일 방향* 으로 무너짐. *§8.5 의 parameter-space collapse (per-adapter rank ≈ 1.71) → output-space collapse (eff_rank ≈ 1.1)* 로 명확히 전파.
2. **그러나 SciFact 는 catastrophic *아님*** — Δ all = −0.010 (≈ baseline), Δ confused = +0.091 ✓. *Collapse* ↔ *catastrophic* 사이의 1:1 대응 없음.
3. ⇒ **Catastrophic 의 진짜 mechanism 은 단순 collapse 가 아님**. *Collapse 의 방향* 이 task structure 에 맞는지가 결정.
   - **SciFact** 는 within-dataset 학습 → collapse direction 이 SciFact 의 ranking 신호와 align. 1.15-dim residual 이 *MaxSim 에서 충분히 discriminative*.
   - **NFCorpus / FiQA** 는 within-dataset 학습 *임에도* collapse direction 이 *wrong* — *baseline retrieval 자체가 약해서 (NDCG@10 0.33 vs SciFact 0.65)* mined HN 의 noise 비율 ↑ + ep0 rank loss ~ 7× 큰 *supervision distortion* → 잘못된 방향으로 더 격하게 학습.

**Paper-grade 결론** (sanity check 후 정정):
- **Phase 2b 의 universal representation collapse 는 직접 관측됨 + sanity check 확정** (numerical 증거 확보, n=500 docs × 3 datasets).
- *Collapse magnitude* 는 *necessary but not sufficient*. *Direction alignment* 가 sufficient 조건의 추가 요인.
- Paper narrative: "**collapse 방향 의 task-alignment** 이 cross-dataset robustness 의 hidden lever".
- *다음 disentangling experiment* 의 검정 가설:
  - (가) optimization root: warmup + grad_clip 으로 ep0 폭발 억제 → collapse magnitude ↓ + direction stabilize ↦ NFCorpus / FiQA NDCG 회복?
  - (나) supervision root: mined HN 대신 in-batch negative → noise ↓ → 학습 신호 가 *옳은* 방향 ↦ NDCG 회복?

### 2.8 Sanity check — *진단이 측정한 model = eval 의 model* (2026-05-24)

**Reviewer 의 critical catch**: "**rank-1 embedding 으론 NDCG 0.65 *수학적으로 불가능***" — SciFact tok_cos +0.94 / eff_rank 1.61 이 진짜면 retrieval 이 random (~0.01) 이 되어야 하는데 보고된 NDCG 0.6367 와 모순. 가설:
- (A) `module_final.pt` (final epoch) vs eval 이 쓴 best-epoch 불일치 → SciFact 만 영향
- (B) LoRA injection α scaling 불일치 → 3 dataset 모두 영향

**Sanity check** (`report/_repr_collapse_sanity.py`): 진단이 로드한 *바로 그 model* 로 test set NDCG@10 재현.

| Dataset | Diagnostic-loaded NDCG@10 all | Original-run NDCG@10 all | Match |
|---|---|---|---|
| SciFact | 0.6367 | 0.6367 | ✓ |
| NFCorpus | 0.0094 | 0.0094 | ✓ |
| FiQA | **0.0005** | **0.0005** (baseline 0.347 의 0.15 %, *최초 표기 0.0388 는 backsolve 오류*) | ✓ |

**결과 — *3 / 3 match***: 가설 A/B 모두 기각. **Collapse 가 진짜** — *SciFact 의 eff_rank ≈ 1.15 / 1.61 *상태에서* NDCG 0.6367 가 실제 발생*.

**FiQA *추가 발견*** — diagnostic 결과 sanity 가 *실제 NDCG 의 정밀화 도구* 로 작동. 보고서 의 backsolve-derived FiQA NDCG (0.0388) 가 *부정확* 했고, 실측치 (0.0005) 가 더 catastrophic — FiQA 는 *literal 0% retrieval* (baseline 0.347 의 0.15 %).

#### 2.8.1 *Rank-1 puzzle* 의 의미

Reviewer 의 *"rank-1 → random retrieval"* 추론의 *premise* (eff_rank perplexity = literal rank) 가 부정확.

1. **eff_rank perplexity 는 *literal rank* 가 아님**. eff_rank ≈ 1 은 σ_1 dominant 이지만 *trailing dimensions 의 non-trivial signal 잔존*. cosine ±0.03 의 std → tokens 가 완전 동일 ≠.
2. **MaxSim 의 *per-token max* 가 small residual 을 증폭**. 모든 doc token 이 tight cone 안 (cos ~0.94) 이라도 각 token 의 *systematic small perturbation* 이 query token 의 max selection 을 결정. *small structural differences* 가 task-aligned 이면 NDCG 보존.
3. **Mean-pooled cosine ≠ MaxSim discrimination**. 전자 ↑ 가 후자 의 discrimination 를 *non-trivially* 상관.

⇒ **SciFact 의 *rank-1 residual* 이 task ranking 신호와 align**. 1.15-dim 의 *systematic structure* 가 MaxSim 에 sufficient.

⇒ **NFCorpus / FiQA 의 *rank-1 residual* 이 task structure 와 *misaligned*** — baseline 약함 → mined HN noise ~50% → supervision distortion → collapse direction wrong.

### 2.4 Seed × 3 robustness (08 의 seed-artifact 교훈 적용, *결과 보기 전 commit*)

Phase 2b 의 +0.091 confused 가 seed 42 단일 — 08 처럼 *seed-specific artifact* 위험 검정. 동일 config (q,v r=8 LR=5e-5 α=r, early-stop=val_all) 로 seed 1337, 2024 추가 실행. *결과 보기 전 commit*: **3-seed 평균 ± CI 보고**.

| Seed | NDCG@10 all | Δ all vs baseline | **Δ confused vs baseline** |
|---|---|---|---|
| 42 | 0.6367 | -0.010 [-0.044, +0.023] (≈) | +0.091 [+0.040, +0.143] ✓ |
| 1337 | 0.6423 | -0.004 [-0.038, +0.028] (≈) | +0.097 [+0.047, +0.150] ✓ |
| 2024 | 0.6639 | +0.018 [-0.014, +0.049] (≈) | +0.123 [+0.073, +0.174] ✓ |
| **3-seed mean ± std** | **0.6476 ± 0.014** | **+0.001 ± 0.014 (anchor preserved)** | **+0.104 ± 0.017 ✓** |

**Robustness 확정** — *08 의 seed-artifact 와 *완전 반대 양상***:
- 3 seeds 모두 Δ confused 의 paired bootstrap CI 가 *명확히 0 초과* (확정 ✓ positive).
- 3 seeds 모두 Δ all 이 *통계 동등* (CI 0 포함, anchor preservation).
- *최저 confused +0.091 (seed 42) 가 unfrozen 의 36%, 최고 +0.123 (seed 2024) 가 49%*. *Distribution 안정*.
- *Seed mean Δ confused +0.104 가 02 unfrozen (+0.252) 의 **41%***. LoRA 가 *진정한* paper-grade lever 확정.

**Pre-commit strict 돌파 기준 (CI(Δ all) > 0) 은 여전히 미달** — 3 seeds 모두 CI 하한 < 0. 단 *3-seed mean Δ all = +0.001* 으로 *baseline 과 본질적 동등* — 손상 우려 해소.

### 2.9 Mediation 1 (warmup + grad-clip) — *optimization root* (2026-05-24)

§7.3.d (REPORT.md) 참조. 요약:

| Dataset | NDCG@10 all | Δ all | Δ confused | ep1 val_all (vs Phase 2b) |
|---|---|---|---|---|
| SciFact | 0.6342 | −0.012 [−0.046, +0.021] | +0.088 [+0.035, +0.139] ✓ | 0.624 vs 0.604 (+0.020) |
| NFCorpus | 0.0113 | −0.319 [−0.353, −0.286] ✗ | −0.093 ✗ | **0.140 vs 0.073 (1.9× ↑)** |
| FiQA | 0.0009 | −0.346 [−0.374, −0.319] ✗ | −0.147 ✗ | **0.257 vs 0.090 (2.86× ↑)** |

**Optimization root 의 *부분* 지지**: warmup 가 ep1 collapse 의 *명확* 한 지연 — 단 post-warmup full LR 가 ep2/3 부터 collapse 재현. *Single uniform rule (warmup 10% + clip 1.0)* 으로는 영구 회복 *불가*. *LoRA best-state 미snapshot* 한계 가 test NDCG 의 catastrophic 유지에 기여.

### 2.10 🎯 Mediation 1b (in-batch negative) — *supervision root* + 첫 strict net 향상 (2026-05-24)

§7.3.e (REPORT.md) 참조. **본 paper 의 prior 모든 실험 (01-10) 에서 미달성한 strict 기준 (CI(Δ all) > 0) 첫 충족** (SciFact, seed 42).

| Dataset | NDCG@10 all | Δ all vs baseline | Δ confused vs baseline | Judgement |
|---|---|---|---|---|
| **🎯 SciFact (seed 42)** | **0.6613** | **+0.015 [+0.001, +0.029] ✓ STRICT positive (razor-thin)** | +0.055 [+0.030, +0.081] ✓ | **첫 strict net 향상 *signal*** |
| **🎯 SciFact (seed 1337)** | **0.6681** | **+0.022 [+0.008, +0.036] ✓** | +0.064 ✓ | strict, confident |
| **🎯 SciFact (seed 2024)** | **0.6722** | **+0.026 [+0.011, +0.042] ✓** | +0.077 ✓ | strict |
| **🎯 SciFact (3-seed mean ± std)** | **0.6672 ± 0.005** | **+0.021 ± 0.005 ✓ ROBUST** | **+0.065 ± 0.012 ✓** | **🎯 3-seed strict robust (캐비엇 2 fully 해소)** |
| **NFCorpus (seed 42)** | **0.2459** (74 % of baseline 0.330) | **−0.084 ✗** | **−0.013** (CI 0 포함) | **74 % gap recovery** |
| **NFCorpus (seed 1337)** | 0.2231 (67 % of baseline) | −0.107 ✗ | — | 67 % gap recovery |
| **NFCorpus (seed 2024)** | 0.2626 (80 % of baseline) | −0.067 ✗ | — | 80 % gap recovery |
| **NFCorpus (3-seed mean ± std)** | **0.244 ± 0.020** | **−0.086 ± 0.020 ✗** | (unmeasured 3-seed) | **74 % ± 7 gap recovery robust** (NOT net+) |
| **FiQA (seed 42)** | ~0.327 (94 % of baseline 0.347) | **−0.020 [−0.028, −0.012] ✗** | −0.010 [−0.021, +0.0003] (CI 0) | **94 % gap recovery** (strongest cross-dataset signal) |

**Phase 2b 의 redistribution 깨뜨림**:
- Phase 2b: zero-sum (Δ confused +0.104 / Δ easy −0.085 / Δ all ≈ 0)
- M1b SciFact: *non-zero net* (Δ confused +0.055 / Δ all +0.015 ✓)

**Train trajectory 의 모든 epoch baseline 위 유지**:
- ep1 val_all 0.672 (baseline +0.026)
- ep2 val_all 0.682 (best, baseline +0.036)
- ep3 val_all 0.679 (baseline +0.033)

‖B‖_total 1.32 (Phase 2b 의 63 %) — *less active LoRA, less collapse*.

**함의 — supervision root *시그널*** (확정 아님, 캐비엇 1/2 반영):
- Mined HN 의 ~50% noise 가 Phase 2b 의 redistribution 의 *주요 원인 *가능성*.
- Cross-dataset (NFCorpus / FiQA) M1b 결과 가 supervision root 의 universality 결정.

**⚠️ 두 *under-weighted* 캐비엇 (reviewer agent 의 critical catch)**:

1. **clean ≠ easy 혼동 (confounded mechanism)**: in-batch neg = *clean + EASY* (다른 query 의 positive 는 trivial contrast). M1b 의 net 향상 의 원인 이 (나-1) noise 제거 인지 (나-2) hard negative difficulty 자체 가 collapse 유발 인지 *구분 불가능*. **결정적 disambiguator**: FN-denoised mined-HN (future work) — hard 유지 + denoise → +0.104 유지 면 noise 가 원인, 여전히 collapse 면 difficulty 가 원인.
2. **seed 42 단독 + CI 하한 +0.001 razor-thin**: paired bootstrap noise 1σ 안. 08 의 seed-artifact 시나리오 와 동일 구조 — *3-seed 전 까지 "첫 strict net 향상" 을 확정 으로 쓰면 안 됨*. Confused +0.055 = Phase 2b +0.104 의 절반 — in-batch 가 HN 을 덜 다룸 의 *예상* signature, non-trivial discovery 아님.

→ **Paper 의 M1b frame**: *promising preliminary signal*, 확정 결론 아님.

#### 2.10.1 NFCorpus M1b — *74 % catastrophic 회복* (supervision root cross-dataset 부분 지지)

| 지표 | Phase 2b NFCorpus | M1b NFCorpus | Improvement |
|---|---|---|---|
| NDCG@10 all | 0.0094 | **0.2459** | **+0.236 (huge)** |
| Δ all vs baseline | −0.320 ✗ | **−0.084 [−0.105, −0.064] ✗** | catastrophic 의 74% 회복 |
| Δ confused vs baseline | −0.092 ✗ | **−0.013 [−0.027, +0.002]** (CI 0) | *baseline 거의 회복* |
| ep1 val_all | 0.073 | **0.376 (baseline 0.330 +0.046 ↑)** | 5.2× ↑ |
| ep2-3 val_all | 0.017 / 0.015 | 0.285 / 0.259 | *decay 시작* |

**핵심 발견**:
1. **Catastrophic 의 74 % 회복** (NDCG 0.0094 → 0.246) — mined HN noise 가 *주요 원인* 확정.
2. **Confused slice 거의 baseline 회복** — Δ confused CI 가 0 포함.
3. **Strict 회복 가능 성**: ep1 val_all 이 baseline 위 (0.376 > 0.330). *LoRA best-state snapshot* 이 되었으면 strict positive 였을 가능성.
4. **Optimization root 의 부분 잔존**: ep1 → ep3 decay (0.376 → 0.259) — *post-warmup 의 full LR* 가 추가 collapse. *M1 + M1b combine* 시 strict 가능 (single-rule pre-commit 으로 미실행).

**Paper-grade 함의**:
- Catastrophic ≠ SciFact-tuned hyperparameter artifact — *mined HN noise* 가 *cross-dataset universal* root.
- *Supervision + optimization root 의 additive 기여* — single-mechanism explanation 부족.

**⚠️ Framing 정정 — NFCorpus M1b 는 *net+ 아님***:
- NDCG 0.246 vs baseline 0.330 = Δ all **−0.084 (여전히 negative)**.
- "74 % 회복" = *catastrophic gap recovery*, **NOT** net 향상. *Framing 시 "NFCorpus 고침" / "net+" 사용 금지*.
- + single-seed (seed 42) — multi-seed robustness 는 future work.

### 2.11 Diagnostic B on Mediation Checkpoints — *dataset-dependent multi-mechanism* (2026-05-24)

§7.3.f (REPORT.md) 참조. M1 / M1b 가 *실제로* collapse 를 감소시키는가 직접 측정 (`report/_repr_collapse_mediation.py`, CPU, n=300 docs per condition).

| Dataset | Frozen | Phase 2b | M1 | **M1b** |
|---|---|---|---|---|
| SciFact eff_rank doc | 10.65 | 1.14 | 1.14 | **7.29 (frozen 의 68%)** |
| NFCorpus eff_rank doc | 11.73 | 1.09 | 1.06 | **1.06 (변화 없음)** |
| FiQA eff_rank doc | 23.58 | 1.06 | 1.05 | (pending) |

**핵심 발견 — M1b 의 *dataset-dependent multi-mechanism***:
- **SciFact**: collapse magnitude 6.4× 감소 (1.14 → 7.29). + NDCG 0.6367 → 0.6613 (+0.025).
- **NFCorpus**: collapse magnitude 동일 (1.09 → 1.06), but NDCG 0.0094 → 0.2459 (74 % 회복).

**NFCorpus paradox**: *same eff_rank, very different NDCG* → *collapse direction 의 task-alignment 회복* 이 핵심 mechanism. *§2.8.1 의 rank-1 puzzle / direction matters* framework 와 정합.

**Supervision root 의 multi-mechanism**:
- 일부 dataset (SciFact): collapse magnitude 감소.
- 일부 dataset (NFCorpus): collapse direction correction.
- *둘 다* mined HN noise 제거로 인한 *clean supervision signal* 의 결과.

**⚠️ 캐비엇 1 (clean ≠ easy) — *여전히 해소 안 됨* (정정)**: 이전 draft 의 *"NFCorpus direction correction 은 easy contrast 만으로 설명 어려움 → noise removal 더 정합 → FN-denoised 필요성 약화"* 추론은 **틀림**. *Easy in-batch 도 *방향* 교정 가능* — *query 를 무관한 doc 에서 분리* 라는 일반-올바른-방향 의 (약한) 신호. NFCorpus 의 mined HN 이 *틀린 방향* 으로 당기고 있었다면 (dense qrels → FN 대량 → relevant doc 을 밀어냄), 그걸 *제거 + 약하지만 올바른 in-batch 신호 대체* 만으로 방향 교정 가능 → (나-1) *noise 제거* 와 (나-2) *easy 의 일반-올바른-방향 신호* **둘 다 똑같이 설명 가능** → confound *여전히 유지*. 오히려 *큰 회복 (74%) 이 약한 신호에서 나온 것* 은 *"mined HN 이 적극적으로 해로웠다"* 시사 — 그 해로움이 *FN noise* 인지 *hard-difficulty 가 fixed-LR 에서 wrong-collapse 유발* 인지 여전히 confound. ⇒ **FN-denoised mined-HN 실험은 *full strength* 로 여전히 필요** (hard 유지 + false 만 제거 → 유일한 깨끗한 disambiguator).

### 2.12 Overnight M1+M1b combined — *Optimization root = red herring* (2026-05-24)

§7.3.e.iii (REPORT.md) 참조.

| Dataset | M1 alone | M1b alone | **M1+M1b combined** | M1 추가 기여 |
|---|---|---|---|---|
| SciFact | Δ all −0.012 | Δ all +0.021 ✓ | **Δ all +0.020 [+0.005, +0.034] ✓** | **−0.001 (zero)** |
| NFCorpus | Δ all −0.319 ✗ | Δ all −0.084 ✗ | **Δ all −0.083 ✗** | **+0.001 (zero)** |

⇒ M1 의 ep1 trajectory 효과 는 *test-time outcome 과 분리* + M1b 와 *additive 아님*. **Optimization root = red herring**. *Phase 2b catastrophic / redistribution 의 *유일* mechanism = mined HN noise (supervision root)*.

### 2.13 Exp 11 (relational easy preservation, λ=1.0) SciFact 3-seed — *branch (a) partial* (2026-05-24)

§7.3.e.iv (REPORT.md) 참조. Pre-committed 3 branches (`report/_exp11_pre_commit.md`) 의 **branch (a) partial** 결과.

| Seed | NDCG@10 all | Δ all | Δ confused | Δ easy |
|---|---|---|---|---|
| 42 | 0.6797 | +0.033 ✓ | +0.095 ✓ | −0.019 ✗ |
| 1337 | 0.6784 | +0.032 ✓ | +0.095 ✓ | −0.021 ✗ |
| 2024 | 0.6697 | +0.023 (CI 0) | +0.113 ✓ | −0.052 ✗ |
| **3-seed mean ± std** | **0.6759 ± 0.005** | **+0.029 ± 0.005** (2/3 strict, 1 marginal) | **+0.101 ± 0.010 ✓** (Phase 2b 의 +0.104 *fully preserved*!) | **−0.031 ± 0.018** (Phase 2b −0.085 의 63 % 감소) |

**Cross-comparison** (3-seed mean):

| Method | Δ all | Δ confused | Δ easy | Trade-off |
|---|---|---|---|---|
| Phase 2b | +0.001 | +0.104 ✓ | −0.085 ✗ | zero-sum redistribution |
| M1b | +0.021 ✓ | +0.065 ✓ (half) | (~−0.05) | strict net+ but sacrifices confused |
| **Exp 11** | **+0.029** (2/3 strict) | **+0.101 ✓ (preserved)** | **−0.031 ✗ (63 % reduced)** | **higher confused + moderate net+** |

⇒ **Two levers, different trade-offs** for partial redistribution resolution. *완전 해소* (3-seed strict + Δ easy ≈ 0) 는 *둘 다 충분치 않음* → future work (higher λ, combined M1b+Exp 11).

### 2.14 Exp 12 (FN-denoised mined-HN) — *캐비엇 1 결정적 disambiguation* (2026-05-24)

§7.3.e.v (REPORT.md) 참조. *Caveat 1 (clean ≠ easy confound)* 의 *empirically disambiguated*.

**3-seed mean ± std** (SciFact, threshold = 0):

| Method | Δ all | Δ confused | Δ easy |
|---|---|---|---|
| Phase 2b (hard + noisy) | +0.001 | +0.104 ✓ | −0.085 ✗ |
| **Exp 12 (hard + clean)** | **−0.004 ± 0.005** | **+0.080 ± 0.004 ✓** | **−0.073 ± 0.005 ✗** |

**🎯 (나-2) Difficulty dominant**: redistribution 거의 동일 — *FN removal 만으로 14 % 만 recovery*. **Hard-contrast 자체** 가 catastrophic / redistribution 의 *주요* mechanism.

**캐비엇 1 status 정정**: ~~still confounded → FN-denoised mined-HN future work 필요~~ → **disambiguated empirically — difficulty dominant, noise minor (~14 %)**. M1b 의 strict net+ 의 *진짜* mechanism = *easy contrast 의 작은 gradient* (hard 회피).

### 2.15 4-lever final framework

| Method | Mechanism | Trade-off |
|---|---|---|
| Phase 2b | hard + noisy mined HN | zero-sum redistribution baseline |
| Exp 12 | hard + clean mined HN | redistribution preserved — noise minor |
| M1b | easy in-batch neg | hard 회피 → strict net+ but confused 절반 |
| Exp 11 | hard + selective easy preservation (relational λ=1) | hard 유지 + selective protection → higher confused + moderate net+ |

⇒ *Single sufficient mechanism* = **hard mined-HN 의 over-correction**. 회피 방법 2 가지 (lever 별 trade-off 다름).

### 2.17 4-lever final ranking (pre-committed only, 2026-05-24)

§5f.4 (REPORT.md) 참조. *Pre-committed clean lever* 만 (post-hoc 묶음 제외):

| Method | Δ all | Δ confused | Δ easy | Strict | 평가 |
|---|---|---|---|---|---|
| Phase 2b | +0.001 | +0.104 | −0.085 | 0/3 | baseline redistribution |
| Exp 12 (FN-denoised) | −0.004 | +0.080 | −0.073 | 0/3 | noise removal *ineffective* |
| **M1b** | **+0.021** | +0.065 (half) | (~−0.05) | **3/3** | *strict robust*, confused 절반 |
| ⭐ **Exp 11 (λ=1)** | **+0.029** | **+0.101 ✓ (preserved)** | −0.031 (63 % 감소) | **2/3** | *best balance* — *higher Δ all + preserved confused* |

**Paper-grade conclusion**: *4 lever 모두 same upstream root (hard-contrast over-correction) 의 다른 angle interventions*. *Best balance lever* = **Exp 11 (λ=1)** (partial branch a, 2/3 strict, reviewer-recommended honest terminus). *Strict robustness 우선* 시 alternate = M1b (3/3 strict, confused 절반).

> **Note**: Post-hoc exploratory variants (Higher λ=5, Combined M1b+Exp 11, FN+EP) — *test 결과 본 후 generative question* 으로 발의된 contaminated experiments — main paper claim base 에서 *제외*. Raw artifacts `outputs/...` + chronological record `report/_overnight_results.md` 에 보존.

### 2.16 Diagnostic B on Mediation + Exp Checkpoints — mechanism direct verification (2026-05-24)

§7.3.e.vi (REPORT.md) 참조. *Collapse magnitude* 측정이 NDCG-level 결과와 *완전 일관*.

| Method | doc_cos μ | eff_rank doc | eff_rank tok | NDCG Δ all (3-seed) |
|---|---|---|---|---|
| Frozen baseline | +0.573 | 10.65 | 57.21 | — |
| Phase 2b | +0.985 | 1.14 | 1.58 | +0.001 |
| **Exp 12 (3-seed)** | **+0.975** | **1.22 ± 0.01** | **1.72 ± 0.05** | **−0.004 ± 0.005** |
| **M1+M1b SciFact** | +0.663 | 7.05 | 43.16 | +0.020 |
| **M1b SciFact (3-seed)** | **+0.663 ± 0.010** | **7.12 ± 0.31** | **44.65 ± 1.55** | **+0.021 ± 0.005** |
| **Exp 11 (3-seed)** | **+0.910 ± 0.022** | **~1.9** | **~9.6** | **+0.029 ± 0.005** |

**🎯 4 mechanism direct verifications**:
1. **Exp 12 ≈ Phase 2b at collapse** — FN removal *only* doesn't change collapse → (나-2) difficulty dominant collapse-level 추가 증거.
2. **M1+M1b ≡ M1b alone at collapse** — M1 추가 기여 zero 가 collapse + NDCG 둘 다 확정.
3. **Exp 11 의 *selective token-level* preservation 직접 확인**: token eff_rank 6× recovery (1.58 → 9.6), doc 1.7× recovery (1.14 → 1.9). *Loss = token sim matrix 직접 규제 = what gets preserved*.
4. **M1b collapse 감소 3-seed robust** (모든 seed eff_rank ~7) — seed-artifact 가 아닌 *robust mechanism*.

**NFCorpus *direction matters* puzzle 강화**: M1+M1b NFCorpus eff_rank 1.05 = M1b alone 1.06 임에도 NDCG 74 % recovery → *direction alignment > magnitude* 의 직접 evidence.

## 3. Sweep 의 *anchor preservation* 회복 trajectory

![LoRA progression](figures/10_lora_phi/lora_progression.png)

*Figure 1. 3 phase 의 Δ NDCG@10 vs baseline (CI bar). **all-slice (파랑)**: Phase 1 -0.052 ✗ → Phase 2a -0.059 ✗ → **Phase 2b -0.010 (CI 0 포함, ≈ baseline)**. **confused (빨강)**: Phase 1 +0.038 (CI 0 포함) → Phase 2a **+0.080 ✓** → **Phase 2b +0.091 ✓**. *LR 보수화 + scaling=1 의 결합* 이 anchor 손상 해소 + confused lift 유지의 *sweet spot*.*

![NDCG vs configs](figures/10_lora_phi/ndcg_vs_configs.png)

*Figure 2. 3 LoRA configs vs 4 anchor (baseline, 02 frozen, 01b α=10, 02 unfrozen) 의 NDCG@10 (all / confused). 02 unfrozen 의 all 0.6576 + confused 0.2200 의 *upper bound*. Phase 2b 의 all 0.6367 (baseline 0.6464 의 -0.010) + confused 0.2454. *02 unfrozen 의 all-slice 도 baseline 동등 + LoRA 도 baseline 동등 → encoder fine-tune 의 lever 가 confused-slice 에 집중*.*

![Delta CI forest](figures/10_lora_phi/delta_ci_forest.png)

*Figure 3. Phase 1, 2a, 2b 의 각 anchor 대비 paired bootstrap 95% CI. **Phase 2b 가 *vs baseline confused* 만 [+] positive, *vs 02 unfrozen confused* 는 [-] negative (LoRA 는 unfrozen 의 1/3 lift)**. Phase 2a 의 *vs baseline all* 만 [-] negative (anchor 손상). *Phase 2b 가 모든 metric 에서 안정적*.*

## 4. M structure — *LoRA capacity utilization*

![LoRA A/B norms](figures/10_lora_phi/lora_AB_norms.png)

*Figure 4. Per-adapter ‖A‖ (파랑), ‖B‖ (주황) — 24 adapters (q + v) × 12 layers. Phase 1 (r=1) 의 adapter 별 norm 이 *균등 분포*. Phase 2b (r=8) 도 *균등 분포* (heavily clustered around mean). **Rank 와 무관하게 학습이 *모든* layer 의 adapter 를 *균등하게* 활용** — K-router collapse 나 bilinear rank-1 collapse 와 *완전히 다른* 패턴.*

이는 LoRA 의 *유의한 발견*: 기존 frozen-side method 들 (06 K-router, 08 bilinear M) 의 effective rank 1 collapse 와 *반대 양상* — LoRA 는 *모든 12 layer × 2 component* 의 capacity 를 *균등 활용*. 이는 encoder finetune 의 학습 신호가 layer-wise 분산 (BERT 자체 의 dropout / multi-layer gradient flow) 의 결과로 추정.

## 5. 학습 동학

![Train curves (3 configs)](figures/10_lora_phi/train_curve_3configs.png)

*Figure 5. (왼쪽) Train loss — r=1 (파랑) 의 0.82 → 0.22 vs r=8 (빨강) 의 0.64 → 0.10. r=8 의 더 빠른 fit. (가운데) Val all-slice — r=1 의 [0.64, 0.66, 0.52] vs r=8 의 [0.60, 0.62, 0.61]. r=8 (Phase 2b) 가 *모든 epoch 에서 r=1 보다 낮은 all-slice* 이지만 ep2 에서 best. (오른쪽) Val confused-slice — r=8 의 ep3 (0.274) peak 우수. 두 config 모두 ep1 dip → ep2-3 recovery 의 *J-shape* trajectory.*

## 6. 진단 — *왜 LoRA 가 unfrozen 의 일부만 회복*?

### 6.1 Effective capacity 의 한계

02 unfrozen (110M params) vs Phase 2b (295K params) = **372× 차이**. LoRA 가 unfrozen 의 *0.27%* params 로 *36% confused 회복* — *param-efficiency 의 surprisingly 좋은 비율*, 단 절대 magnitude 의 한계.

### 6.2 LoRA의 학습 signal 한계 (data bottleneck)

SciFact train 9,190 triplet — LoRA r=8 (295K params) 의 *32× under-parameterized*. Train loss 가 ep2 의 0.12, ep3 의 0.10 까지 떨어지는 *완전 fit 직전* — *data bottleneck* 의 직접 증거. *Hyperparameter sweep 으로 못 뚫음* (pre-commit 의 결정 근거).

### 6.3 Phase 2a → 2b 의 *anchor preservation* 회복 핵심

| Phase | LR | α | scaling | Δ all CI |
|---|---|---|---|---|
| 2a | 1e-4 | 16 (2r) | 2 | -0.059 ✗ |
| **2b** | **5e-5** | **8 (r)** | **1** | **-0.010 (≈)** |

*동시* 변경: LR 절반 + scaling 절반. 가능 해석:
- (a) LR=1e-4 가 over-step → ep2 의 val_all peak (0.69) 가 *unstable transient*. LR=5e-5 가 *stable* trajectory.
- (b) scaling=2 의 effective ΔW magnitude 2× 가 anchor 흔듦. scaling=1 이 보수적.

본 paper 의 *single-variable 분리* 검정 (LR 만 변경, 또는 scaling 만 변경) 은 *Pre-commit 따라 추가 sweep 금지*. Future work 로 명시.

## 7. *Pre-committed 판정* + Safety-net narrative

### 7.1 판정 결과

Phase 2b 의 Δ all vs baseline CI = [-0.044, +0.023] → **CI 하한 -0.044 < 0 → 돌파 *미달***.

Per pre-commit:
- ✗ Strict 돌파 (CI > 0) 못 함
- ✓ Anchor preservation (CI 0 포함, 통계 동등) 달성
- ✓ Confused-slice +0.091 의 *유의* 개선

### 7.2 Paper main contribution 의 *bounded improvement* framing

본 paper 의 종합 결론:

> **Frozen ColBERT v2 의 lightweight intervention 의 *bounded improvement* 의 정밀 분석**: (i) Translation family 의 모든 변형 + form-change (bilinear M, 08) + distillation (09) 의 NDCG@10 all 이 *baseline ≈ 0.646* 의 ceiling 에서 수렴 — *informed-direction subspace 의 encoder representational limit*. (ii) ColBERT *encoder 전체 unfreeze* (02 unfrozen, 110M params) 가 Δ confused **+0.252** 의 5× lift — *limit 의 직접 증거*. (iii) LoRA adapter (295K params, encoder 의 0.27%) 가 그 lift 의 36% 회복 — *param-efficient partial recovery* 의 데이터 증거. (iv) 하지만 strict 돌파 (NDCG@10 all 의 CI 하한 > baseline) 는 미달 — *9 K SciFact triplet 의 data bottleneck* 이 LoRA 의 추가 lift 차단.

### 7.3 Robustness limitations (정직)

- **Single-seed** (42): seed × 3 robustness check 미수행 (08 의 seed-artifact 위험 잔존)
- **Single-dataset** (SciFact): NFCorpus 등 cross-dataset 검정 미수행 (06 K=2 의 catastrophic 교훈)
- **Pre-commit 따라 hyperparameter sweep 중단**: LR/rank/scaling/components 의 *full* design space 미탐색 — future work
- **Best-state selection 기준 변경** (val_conf → val_all): 본 실험만 적용 — 02-09 의 historical baseline 과 *direct paired comparison* 가 *완전히 fair 하지 않음* (best-state 다른 기준). 단 모든 18(=10) configs 는 동일 기준 적용.

## 8. ROADMAP 영향 + future work

### 8.1 Stage 3 (LoRA on Φ) 종합 결론

| 항목 | 결과 |
|---|---|
| Phase 1 (r=1, 36K) | ✗ anchor 손상 + confused not sig |
| Phase 2a (r=8, 295K, LR=1e-4 α=2r) | ✓ confused +0.080, ✗ anchor 손상 |
| **Phase 2b (r=8, 295K, LR=5e-5 α=r)** | **✓ confused +0.091, ✓ anchor 보존 (CI 0 포함)** |
| Strict 돌파 (CI(Δ all) > 0) | ✗ 모든 phase 미달 |

**Stage 3 종합 conclusion**: *LoRA 가 frozen-side intervention 의 *bounded improvement* 의 최고치 (+0.091 confused, anchor preserved). Encoder representational limit 의 30+% 회복. Strict 돌파 미달 — data bottleneck 한계.*

### 8.2 Future work (paper next-steps)

| 후보 | 우선 | 이유 |
|---|---|---|
| Phase 2b 의 **seed × 3** robustness | high | 08 의 seed-artifact 교훈 — paper main contribution 의 robustness 확보 |
| Phase 2b 의 **NFCorpus** cross-dataset | high | 06 의 SciFact-specific 교훈 — generality 검정 |
| **MS MARCO** 등 *large train set* 으로 LoRA 재학습 | medium | 9K triplet data bottleneck 의 해소 가능성 |
| **Cross-encoder distillation** (MonoT5) | medium | 09 의 noise teacher 한계 후 진정한 teacher |
| LoRA design space full sweep (component, layer subset, rank, LR) | low (pre-commit 따라 *현 deliverable 의 scope 외*) | future deeper analysis |

## 8.5 *Universal rank-collapse + spatial multiplicity escape* — paper main punchline

본 결과 + 기존 06/08 의 통합 진단. 모든 학습된 frozen-side intervention 의 *single intervention position 에서의 effective rank/K* 가 nominal capacity 의 12-30% 에 collapse — *universal pattern*. LoRA 가 우월한 lift 를 보이는 진짜 이유는 *per-position rank 의 escape 아니라* **24 distinct intervention positions 의 spatial multiplicity**.

### Cross-method capacity utilization

| Method | Nominal | Effective | Util ratio | Intervention positions |
|---|---|---|---|---|
| 06 K-router K=2 | 2 | 1.41 | 70 % | 1 (single layer 12) |
| 06 K-router K=4 | 4 | 1.23 | 31 % | 1 |
| 06 K-router K=8 | 8 | 1.44 | 18 % | 1 |
| 08 bilinear M r=8 | 8 | 1.01 | **13 %** | 1 (single metric correction) |
| 10 LoRA r=1 (Phase 1) | 1 | 1.00 | 100 % | **24** (q + v × 12 layers) |
| **10 LoRA r=8 (Phase 2b, per-adapter mean)** | 8 | **1.71** | **21 %** | **24** |

![Universal rank-collapse](figures/_cross_method/rank_collapse_contrast.png)

*Figure 6. (왼쪽) Method 별 nominal vs effective capacity 의 절대 비교 — **06/08/10 모두 effective ≈ 1-1.7 의 *universal collapse pattern***. (가운데) Utilization ratio (effective / nominal) — 06 K=8 18%, 08 r=8 13%, 10 r=8 21%. *Per-position rank collapse 는 학습 dynamics 의 systematic feature*. (오른쪽) 10 LoRA 의 24 adapters 의 per-adapter effective rank 분포 — Phase 2b 의 mean 1.71, std 1.07, range 1.0-5.7. *Per-adapter 도 collapse 하지만 24 distinct positions 모두 active (n_active=24)*.*

### 진단 — *학습 dynamics 의 universal feature + LoRA 의 다중 lever*

| 측면 | Mechanism |
|---|---|
| **Per-position collapse** | 모든 method (06 K-router, 08 bilinear M, 10 LoRA per-adapter) 의 effective rank ≈ 1-1.7. *Pairwise margin loss + AdamW + small_random init* 의 학습 동학이 *single dominant axis 로 수렴* 시키는 systematic feature. |
| **LoRA 의 *spatial multiplicity*** | 24 adapters (q+v × 12 layers) 가 *모두 active* + 각각 ~1.7 effective rank → **total effective intervention dimensionality ≈ 41** (24 × 1.7). 06 / 08 의 1 position × 1.4 ≈ 1.4 대비 ~30× 높음. |
| **Empirical lift correlation** | Confused +Δ 가 position 수 에 monotonic 증가: 06 K-router (1 pos) → +0.045, 10 LoRA (24 pos) → +0.104 mean, 02 unfrozen (full encoder) → +0.252. **Spatial multiplicity 가 진정한 lever**. |

### Confused-slice lift 의 통합 정리

| Method | Δ confused vs baseline | Effective dim × positions | 정량적 |
|---|---|---|---|
| 02 K=1 frozen | +0.044 ✓ | 1 × 1 = 1 | baseline |
| 06 K-router K=2 | +0.039 ✓ | 1.4 × 1 = 1.4 | low spatial |
| 06 K-router K=4 | +0.045 ✓ | 1.2 × 1 = 1.2 | low spatial |
| 06 K-router K=8 | +0.049 ✓ (+ anchor 손상) | 1.4 × 1 = 1.4 | over-cap |
| 08 bilinear M r=8 (seed 42) | +0.054 ✓ (seed-specific) | 1.0 × 1 = 1.0 | low spatial |
| 09 distillation λ=0.1 | +0.019 ✓ | ~rank-2 × 1 = 2 | rank-disrupted |
| **10 LoRA r=8 (Phase 2b 3-seed mean)** | **+0.104 ± 0.017 ✓** | **1.7 × 24 = ~41** | **high spatial** |
| 02 unfrozen (110M) | +0.252 ✓ | full encoder freedom | upper bound |

**Spatial multiplicity 의 linear-ish correlation**: log(positions) 대비 confused Δ 의 *대략 monotonic 증가*. 본 paper 의 *진정한 paper-grade contribution*.

## 9. Artifact 위치

```
outputs/10_lora_phi/scifact/seed_42/
├── qv_r1_l12/   # Phase 1: r=1, 36,864 params, LR=5e-5, α=r
│   ├── config / env / train_config / module_final.pt / train_history.json
│   ├── lora_stats.json (per-adapter ‖A‖, ‖B‖, total)
│   ├── runs / runs_scored / metrics_per_query / metrics_aggregate.json
│   └── delta_vs_{baseline, mean_diff_alpha10, 02_learned, 02_unfrozen, 08_r8}.json
└── qv_r8_l12/   # Phase 2b (latest, overwrote 2a): r=8, 294,912 params, LR=5e-5, α=r
    ├── (위와 동일 구조)

report/figures/10_lora_phi/{ndcg_vs_configs, delta_ci_forest,
    lora_progression, lora_AB_norms, train_curve_3configs}.{pdf,png}
```

**참고**: Phase 2a 의 raw artifact 는 Phase 2b 가 *덮어씀* (`qv_r8_l12` 동일 tag). 본 보고서의 Phase 2a 결과는 Monitor 로그 캡처 + 1차 분석 시점의 paired bootstrap 결과를 *고정값* 으로 인용 (recoverable from `/tmp/18_qv_r8.log` archive, but not in tree).
