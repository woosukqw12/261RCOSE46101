# Paper section skeleton — *Catastrophic Failure 의 진단* (pre-commit binding)

**작성 일시**: 2026-05-24
**상태**: pre-commit (mediation 1 / 1b 결과 *보기 전* 골격 확정)

본 문서는 *post-hoc cherry-picking 회피* 를 위해 mediation experiment 의 결과를 보기 *전에* paper section 의 narrative 골격 + 검정 가설의 *기각 vs 확정 조건* 을 명시한다.

---

## 1. Section title (잠정)

> **§5f. Catastrophic Failure as Representation Collapse: Disentangling Optimization vs Supervision Roots**

(REPORT.md 의 §5d=LoRA on Φ, §5e=Cross-method rank-collapse, §5f=본 cross-dataset diagnostic 의 연속)

---

## 2. Narrative skeleton

### 2.1 Observation (이미 확정)

1. **2 / 2 cross-dataset catastrophic** — Phase 2b (q,v r=8 LR=5e-5 α=r) 가 NFCorpus 와 FiQA 모두 Δ all ≈ −0.32 ~ −0.35 ✗ catastrophic. SciFact-tuned hyperparameter 의 *cross-dataset generality 부정*.
2. **Diagnostic B** (`_repr_collapse_diagnostic.py`) — 3 dataset 모두 LoRA Phase 2b 후 encoder output 의 *extreme representation collapse* (doc-pair cos ≈ 0.99, eff_rank ≈ 1). **§5e 의 parameter-space ΔW rank-collapse 와 다른 현상** (encoder output 의 직접 측정).
3. **Universal collapse ↔ catastrophic NDCG 의 1:1 대응 부재** — SciFact 도 collapse 됨에도 NDCG 거의 유지 (Δ all = −0.010 ≈ baseline, Δ conf +0.091 ✓). ⇒ **Catastrophic = *necessary collapse + sufficient direction-misalignment***.

### 2.2 Hypothesis — *두 root* 의 disentangling

| Root | 가설 mechanism | 검정 manipulation |
|---|---|---|
| **(가) Optimization** | ep0 의 *큰 gradient* (NFCorpus / FiQA 의 rank loss ~ 7× SciFact) → LoRA over-step → collapse direction *unstable / wrong* | **Warmup + grad_clip** — ep0 의 폭발 억제, magnitude 안정화 |
| **(나) Supervision** | Mined HN 의 *noise* 비율 ↑ (weak baseline → mined HN 의 ~50% irrelevant) → 학습 signal 자체가 *wrong direction* 지시 | **In-batch negative** — mined HN 대신 random doc, noise 제거 (단 weaker contrast) |

각 mediation 은 *single rule + result-blind 1 run per dataset* (3 datasets × 1 run = 3 runs per mediation; total 6 runs).

### 2.3 Pre-commit 의 *기각 vs 확정* 조건

| 결과 | 함의 |
|---|---|
| **Mediation 1 (warmup+clip) 후 NFCorpus / FiQA 의 Δ all > −0.10** (i.e., catastrophic 회복) | (가) optimization root 가 주요 mechanism. *Universal collapse 의 magnitude* 가 LR / gradient 제어로 완화 가능. |
| **Mediation 1b (in-batch neg) 후 NFCorpus / FiQA 의 Δ all > −0.10** (catastrophic 회복) | (나) supervision root 가 주요 mechanism. *Mined HN 의 noise* 가 collapse direction 의 task-misalignment 원인. |
| **둘 다 회복** | 두 root 모두 기여, *additive*. *Adaptive* HP strategy + cleaner supervision 의 결합 필요. |
| **둘 다 catastrophic 유지** | 두 root 모두 *충분하지 않음*. *제 3 의 mechanism* (e.g., 9K SciFact-only 학습 의 *data bottleneck*) 검증 필요. |
| **SciFact 는 Δ conf 가 ≈ +0.10 유지** (모든 mediation 에서) | SciFact 의 baseline 결과는 *robust*. |
| **Mediation 후 SciFact 의 Δ conf 가 < 0** | Mediation 의 over-regularization. *Cross-dataset uniform rule* 의 fundamental tension. |

### 2.4 Paper narrative 의 *방향* — 결과 의존

- **Best case** (둘 중 하나 회복) → "**The mechanism behind cross-dataset failure is X**" (clear story, mediation = positive contribution).
- **Mixed case** (양쪽 부분 회복) → "**Cross-dataset robustness requires *both* X and Y** — interventions on either alone are insufficient" (more nuanced contribution).
- **Negative case** (둘 다 catastrophic 유지) → "**Mediation experiments rule out X and Y; the true root remains open. The failure is consistent with *data scale bottleneck* (9K triplets)**" (honest negative; constrains future work).

본 paper 의 *진단 contribution* 는 모든 경우에 *positive* — *catastrophic failure mechanism* 의 *direct measurement* (Diagnostic B) + *disentangling test* (Mediation 1 / 1b). 단 mediation 의 *각 결과* 가 narrative tone 결정.

---

## 3. 실험 설계 — *single rule + result-blind*

### 3.1 Mediation 1: Warmup + grad-clip

**Hyperparameter rule** (모든 dataset 공통, *결과 보기 전* commit):
- Warmup: linear warmup 0 → LR 의 첫 **10 %** training steps. LR 자체는 *unchanged* (5e-5).
- Gradient clipping: `torch.nn.utils.clip_grad_norm_` with `max_norm=1.0`. 모든 LoRA parameter group 에 적용.
- Other params 동일 (q,v r=8 α=r batch_size=32 epochs=3 patience=2 early_stop=val_all).

### 3.2 Mediation 1b: In-batch negative

**Hyperparameter rule** (모든 dataset 공통, *결과 보기 전* commit):
- Triplet 의 `hn_doc` 를 *batch 안의 다른 query 의 positive doc* 으로 대체.
- **Scale**: 1 in-batch negative per query (mined HN 와 동일 1:1 ratio — *fair comparison* 우선; full batch_size−1 contrast 는 다른 lever 도입). Implementation: `pos_emb.roll(1, dims=0)` — cyclic shift by 1, batches 가 shuffle 되므로 random ≈.
- Loss: 동일 pairwise margin (cfg.margin = 0.2).
- Other params 동일 (q,v r=8 α=r batch_size=32 epochs=3 patience=2 early_stop=val_all *warmup off, clip off*).

### 3.3 결과 보고 form (pre-commit)

각 mediation 후 다음 form 으로 보고:

| Dataset | NDCG@10 all | Δ all vs baseline | Δ confused vs baseline | judgement (per §2.3) |
|---|---|---|---|---|
| SciFact | ... | ... | ... | (✓ / △ / ✗) |
| NFCorpus | ... | ... | ... | |
| FiQA | ... | ... | ... | |

---

## 4. 한계 + future work

- Single mediation per dataset (no seed × 3 within mediation) — *resource constraint*.
- 본 disentangling 이 *exhaustive* 가 아님 — 다른 root (data scale, layer choice, etc.) 미검정.
- 본 framework 은 LoRA q,v 의 SciFact-trained hyperparameter 의 *cross-dataset 전이* 검정 — 다른 PEFT method (AdaLoRA, DoRA, BitFit, IA3) 의 cross-dataset 양상 미검정 → future work proposal §E.

---

## 5. Open questions (mediation 후 *추가 정밀화*)

- *Collapse direction* 의 cross-dataset 유사성? SciFact-trained vs NFCorpus-trained 의 cosine.
- *Within-dataset* mediation 의 SciFact effect — over-regularization 우려.
- 본 mechanism (collapse + direction misalignment) 의 다른 ColBERT variant (PLAID, ColBERTv2-Mini) 일반화?

---

**본 골격 의 *commit* 시점**: 2026-05-24 02:38 (mediation 시작 전).
**Mediation 1 (warmup+clip) 시작**: 2026-05-24 02:47 — SciFact (running), NFCorpus + FiQA queued.
**Mediation 1b (in-batch neg) 시작**: pending (Mediation 1 후).
**Queue script**: `/tmp/_mediation_queue.sh` — 6 runs sequential, `outputs/_m{1,1b}_{ds}.log`.
