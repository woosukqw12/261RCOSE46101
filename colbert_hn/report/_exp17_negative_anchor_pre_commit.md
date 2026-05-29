# Exp 17 — Negative-side Anchor Pre-Commit (BEFORE training)

**작성 시점**: 2026-05-25 (학습 시작 *전*, result-blind).

본 pre-commit 은 본 paper §4.4 의 anchor mechanism 의 *비대칭성* — easy query 의 q-token + positive-doc token 은 anchor 되지만 *mined hard-negative doc token 은 anchor 자유* — 을 직접 해소하는 단일 실험의 사전 등록이다.

---

## Methodological commitments (절대 위반 금지)

- Pre-commit single config — *λ_neg sweep 금지*. 단일 value λ_neg = 1.0 (current λ_dir 과 symmetric).
- 3 seeds {42, 1337, 2024} on **SciFact only** — cross-domain (NF / FiQA) 는 branch (a) 시에만 *조건부* trigger (본 pre-commit 의 후속 amendment 로만 가능, *amendment 도 결과 본 후 작성 금지 — branch (a) 발현 즉시 별도 cross-domain pre-commit 작성*).
- Result-blind: 결과 보기 *전* commit, 결과 후 수정 금지.
- **STOP rule (SciFact)**: 본 묶음 3 seeds × 1 config = 3 runs 종료 후 *추가 실험 금지*. 결과가 어떻든 paper writing 진입. *Variant (예: λ_neg sweep, doc-side anchor 추가 형태) 진행 금지*.
- Negative result 도 *equal honest weight* 보고 — 본 pre-commit 의 (b)(c) branch 모두 paper-grade.

---

## Motivation — 본 paper §4.4 의 *비대칭성* 직접 노출

본 paper §4.4 의 per-token cosine anchor (Exp 13) 를 세 항으로 분해:

$$\mathcal{R}_{\text{abs}}^{q}(\theta) = \mathbb{E}_{x \in \mathcal{Q}_{\text{easy}}}\!\Big[\,\tfrac{1}{|T_x^q|}\!\!\sum_{t}\!\big(1 - \cos(\hat h_t^{\text{LoRA},q}(x), \hat h_t^{\text{frozen},q}(x))\big)\,\Big]$$

$$\mathcal{R}_{\text{abs}}^{d^+}(\theta) = \mathbb{E}_{x \in \mathcal{Q}_{\text{easy}}}\!\Big[\,\tfrac{1}{|T_{d^+}^d|}\!\!\sum_{t}\!\big(1 - \cos(\hat h_t^{\text{LoRA},d}(d^+), \hat h_t^{\text{frozen},d}(d^+))\big)\,\Big]$$

$$\mathcal{R}_{\text{abs}}^{d^-}(\theta) = \mathbb{E}_{x \in \mathcal{Q}_{\text{easy}}}\!\Big[\,\tfrac{1}{|T_{d^-}^d|}\!\!\sum_{t}\!\big(1 - \cos(\hat h_t^{\text{LoRA},d}(d^-), \hat h_t^{\text{frozen},d}(d^-))\big)\,\Big]$$

§4.4 의 *실제 구현* 은 앞 두 항만 합산: $\mathcal{R}_{\text{abs}}^{\text{current}} = \mathcal{R}_{\text{abs}}^{q} + \mathcal{R}_{\text{abs}}^{d^+}$. 즉 q-token 과 positive doc token 의 표현은 이미 anchor 됨. 그러나 *easy query 에 mined 된 hard-negative doc* (frozen ColBERT 가 top-1 으로 positive 를 뽑은 query 의 d^-) 의 표현은 anchor 자유 — $\mathcal{R}_{\text{abs}}^{d^-}$ 항이 *명시적으로 누락*. LoRA 가 d^- 의 표현을 임의로 변형 가능:

- d^- 의 frozen 표현은 frozen ColBERT 에서 *correct ranking* (d^- 가 top-1 이 아님) 의 근거.
- LoRA 가 d^- 의 token 표현을 q^easy 와 가깝게 이동시키면 → MaxSim 증가 → ranking 손상.
- 또는 d^- 표현을 d^+ 와 indistinguishable 하게 변형 → contrast 의 base 잃음.

ColBERT 의 retrieval score:

$$s(q, d) \;=\; \sum_{t \in q}\!\max_{t' \in d}\!\cos\big(h_t^q, h_{t'}^d\big)$$

는 *양변 모두 LoRA 의 함수*. 현재 anchor 는 *q-side 와 d^+ side 만* 통제, *d^- side 는 implicit (공유 weight 를 통한 간접 효과)* — *기하학적으로 한쪽만 묶임*.

본 실험은 이 *마지막 비대칭* 을 해소한다.

---

## 가설 형식화

**H1 (positive)**: Easy query 의 mined HN doc 표현을 anchor 하면, LoRA 의 d^- 변형이 차단되어 easy query 의 ranking 손상이 추가 감소 — Δ all > +0.030 strict.

**H2 (saturation)**: 공유 LoRA encoder weight 를 통한 *implicit* d-side 제약이 이미 충분 — d^- 명시 anchor 가 *q-side / d^+ side anchor 의 기존 효과를 그대로* 재현 — Δ all ≈ +0.030 ± 0.005.

**H3 (over-restriction)**: d^- anchor 가 hard query 의 학습에 *해석* 손상 — LoRA 가 hard query 에서 *진짜 가까운* d^- 의 표현을 *밀어내지 못해* +0.092 hard lift 가 줄어듦 — Δ all < +0.020.

세 가설 모두 본 paper §4.4 의 anchor saturation 주장의 양적 검정.

---

## Config (SciFact 와 동일 base, 재튜닝 금지)

| Item | Value |
|---|---|
| Method | Per-token cosine anchor extended to **negative doc tokens** of easy queries |
| **λ_dir (q + d^+)** | **1.0** (현재 §4.4 default, *변경 금지*) |
| **λ_neg (d^- 추가 항)** | **1.0** (symmetric extension, single value, *no sweep*) |
| Loss | $\mathcal{L}^\dagger = \mathcal{L}_{\text{margin}}(\mathcal{Q}_{\text{hard}}) + \lambda_{\text{dir}}\!\cdot\!\big(\mathcal{R}_{\text{abs}}^{q} + \mathcal{R}_{\text{abs}}^{d^+}\big) + \lambda_{\text{neg}}\!\cdot\!\mathcal{R}_{\text{abs}}^{d^-}$ |
| Anchor target sets | $\mathcal{Q}_{\text{easy}}^{\text{train}}$ (q + 짝지어진 d^+ + 짝지어진 d^-) |
| LoRA | q, v r=8 α=r |
| LR | 5e-5 |
| Other | batch=32 ep=3 patience=2 early-stop=val_all |
| Triplet cap | 9190 (SciFact 와 동일) |
| Dataset | SciFact (train + test) |
| Seeds | 42, 1337, 2024 |
| Total runs | **1 dataset × 3 seeds = 3** |
| Tag | `qv_r8_l12_dir1_neg1` |

### 실험 구조

```
experiments/17_negative_side_anchor/
├── run.py        # entry point (experiments/13 코드 fork + d^- anchor 항 추가)
├── README.md     # 카드 (가설 / 성공기준 / 상태)
└── figures.py    # post-training (per-seed Δ 차트 + 토큰별 cosine drift histogram + d^- score drift)
```

---

## 3-branch predictions (pre-registered)

| Branch | 조건 | 함의 |
|---|---|---|
| **(a) Positive leap** | Δ all $\geq$ +0.040 with 3-seed CI > 0 strict (3/3) | H1 입증. Anchor 의 *symmetric* form 이 +0.030 mechanistic ceiling 의 *실제 ceiling 이 아니었음*. Paper main result 갱신 가능. Anchor target set 의 *완전성* 이 핵심 자유도. **Cross-domain trigger 별도 pre-commit 후 진행** (FiQA + NF × 3 seeds = 6 runs). |
| **(b) Tied / saturated** | +0.025 $\leq$ Δ all $\leq$ +0.040 (CI 0 포함 가능, 통계적으로 +0.030 와 구분 불가) | H2 입증. 공유 LoRA encoder weight 가 d^- side 도 *implicit* 으로 충분 제약. *§4.4 의 +0.030 saturation 의 양적 ceiling 양적 증명*. Paper §4.4 의 결론 강화 (saturation 의 *직접* 증거 추가 — 현재는 H₁ ≈ H₂ tie 의 *간접* 증거만). |
| **(c) Over-restriction** | Δ all $\leq$ +0.020 | H3 입증. d^- anchor 가 hard query 의 contrast 학습 손상. *§7.1 의 differential anchor framing 의 정당화 — d^- 항은 informed weight 가 필요함을 시사*. Negative result 로 paper §7 future direction 강화. |

세 branch 모두 *paper-grade* — 양적 결과보다 *mechanism level reading 의 cleanliness* 가 결정적.

---

## Engineering required

### Code 변경 (experiments/17_negative_side_anchor/run.py)

`experiments/13_frozen_direction_anchor/run.py` 의 fork + 다음 4 곳 수정:

1. `precompute_frozen_embeddings`: 캐시에 `hn_emb` 추가. Key 를 `(qid, pos_did, hn_did)` 로 확장 *또는* 별도 `frozen_hn_cache[hn_did]` 으로 분리 (HN frozen 표현은 hn_did 만의 함수 — 더 효율적).
2. `cosine_deviation_loss`: 함수 signature 에 `hn_emb_batch`, `hn_mask_batch`, `batch_hn_dids` 추가; `loss_hn = (1 - cos_hn).mean()` 항 추가; 반환식 `loss_q + loss_d + lambda_neg_scale * loss_hn` (lambda_neg_scale 은 caller 가 외부에서 적용 — 함수 내부는 *동일 weight* 로 합산).
3. `train_with_direction_anchor`: easy_idx 분기 시 hn_emb 도 forward (현재는 confused_idx 만 hn_emb forward). easy query 도 hn_emb forward 추가.
4. CLI: `--lambda-neg` 추가, default = 1.0.

### Loss 함수 정확한 형식

```python
# Per easy query x with (q, d+, d-):
loss_q   = (1 - cos(h_q^LoRA, h_q^frozen)).mean()       # 기존
loss_dp  = (1 - cos(h_dp^LoRA, h_dp^frozen)).mean()     # 기존
loss_dn  = (1 - cos(h_dn^LoRA, h_dn^frozen)).mean()     # 신규
# Aggregate:
R_extended = mean over easy_x of [loss_q + loss_dp + lambda_neg * loss_dn]
L_total = L_margin(confused) + lambda_dir * R_extended
```

즉 d^- 항만 별도 weight λ_neg 로 통제. 본 pre-commit 에서 λ_neg = 1.0 fix.

### 검증 plan

- Smoke: SciFact seed 42 (~12-15 min) — code 검증.
- 정상 작동 시: seed 1337, 2024 sequential.
- 각 run 종료 시 `report/_overnight_results.md` 에 결과 append (현재 형식 follow).

---

## STOP rule (강조)

**3 SciFact runs 후 결과 무관 STOP**:
- λ_neg sweep *금지* (예 λ_neg ∈ {0.5, 2.0, 5.0} 등).
- d^- anchor 의 *variant* (예: d^- 중 e5_margin > 0 인 것만, attention weight 가중, ...) 진행 *금지*.
- Cross-domain (NF / FiQA) 진행은 **branch (a) 발현 시에만 별도 pre-commit 작성 후** 가능.
- *Future work* 글로만 가능.

본 실험은 *§4.4 의 anchor symmetry 의 단일 빈칸* 메우기. 결과가 어떻든 paper writing 진입.

---

## Cross-domain conditional trigger (branch (a) 시에만)

Branch (a) 발현 시 별도 pre-commit (`_exp17_cross_domain_pre_commit.md`) 작성 후:
- 대상: NFCorpus + FiQA × 3 seeds = **6 runs**
- 동일 SciFact config (λ_dir = 1.0, λ_neg = 1.0, q/v r=8, LR 5e-5) — *재튜닝 금지*
- 가설: branch (a) 의 leap 이 *cross-domain 에서도 hold* 한다면, §5 의 anchor cross-domain partial recovery (FiQA 74 % / NF 31 %) 가 *strict net+ 또는 더 큰 회복* 으로 갱신
- 3-branch predictions: 별도 doc 에서 정의

본 trigger 는 branch (a) 가 *strict criterion (Δ all ≥ +0.040, 3/3 seed CI > 0)* 을 만족한 경우에만 발화. (b)(c) 면 cross-domain 진행 *금지*.

---

**Commit timestamp**: 2026-05-25.
**Training start**: 즉시 (engineering 완성 후).
