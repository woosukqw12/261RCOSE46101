# Overnight Autonomous Experiments — 2026-05-24

> 🚫 **POST-HOC EXCLUDED — RAW RESEARCH RECORD ONLY**
>
> 본 문서는 *chronological raw record*. *Pre-commit timing 미충족* (test 결과 본 *후* 발의) 인 3 묶음 (9 runs) 은 **main paper claim base 에서 제외**:
> - **Higher λ=5 Exp 11** (SciFact × 3 seeds)
> - **Combined M1b + Exp 11** (SciFact × 3 seeds)
> - **FN+EP variant** (Exp 11 λ=1 + FN-denoise, SciFact × 3 seeds)
>
> 또한 **M1+M1b combined** (SciFact + NFCorpus, seed 42) 도 *post-hoc exploratory but negative result* — selection risk 낮으나 *clean 이라 우기지 않음*.
>
> *Pre-committed pillars* (Phase 2b 3-seed, M1, M1b 3-seed, Exp 11 λ=1 3-seed, Exp 12 3-seed) 만 main paper 의 evidence base. Reviewer recommendation 따라 *Exp 11 (λ=1) 의 2/3 strict partial 로 honest 종착*. 본 문서 의 *clean* part 만 paper 에서 인용.

**Started**: 2026-05-24 04:54:48

User went to sleep, requested autonomous execution of priority 1-8 (skip #7 FN-denoised, requires external team).

**Plan**:
1. Wait for FiQA M1b queue (GPU) + Mediation sanity (CPU) to finish
2. Re-run Diagnostic B mediation (FiQA M1b 추가)
3. Exp 11 (easy preservation, λ=1.0) SciFact × 3 seeds {42, 1337, 2024}
4. M1b SciFact seed × 2 추가 robustness (1337, 2024) — seed 42 done
5. M1b NFCorpus seed × 2 추가 robustness (1337, 2024) — seed 42 done
6. M1+M1b combined SciFact + NFCorpus (priority 3 #8)

---

## Phase 0 — Wait for current FiQA M1b queue (GPU)
**Started**: 2026-05-24 04:54:48
**Completed**: 2026-05-24 04:59:18 — GPU now free

## Phase 1 — Diagnostic B FiQA M1b (resume from cache)
**Started**: 2026-05-24 04:59:18
**Log**: `outputs/_post_diagnostic.log`
**Command**:
```bash
.venv/bin/python report/_repr_collapse_mediation.py
```
**Completed**: 2026-05-24 04:59:31 (exit=0)

**Artifact**: `report/figures/_repr_collapse_mediation`


## Phase 1b — Mediation sanity FiQA M1b (resume from cache)
**Started**: 2026-05-24 04:59:31
**Log**: `outputs/_post_sanity.log`
**Command**:
```bash
.venv/bin/python report/_mediation_sanity.py
```
**Completed**: 2026-05-24 05:02:37 (exit=143)
  ⚠️ **non-zero exit code: 143**

## Phase 2 — Exp 11 SciFact seed=42 (λ_easy=1.0)
**Started**: 2026-05-24 05:02:37
**Log**: `outputs/_exp11_scifact_seed_42.log`
**Command**:
```bash
.venv/bin/python experiments/11_easy_preservation/run.py --dataset scifact --seed 42 --r 8 --alpha 8.0 --lora-lr 5e-5 --lambda-easy 1.0 --max-triplets 9190 --early-stop-metric all
```
**Completed**: 2026-05-24 05:02:40 (exit=2)
  ⚠️ **non-zero exit code: 2**

## Phase 2 — Exp 11 SciFact seed=1337 (λ_easy=1.0)
**Started**: 2026-05-24 05:02:40
**Log**: `outputs/_exp11_scifact_seed_1337.log`
**Command**:
```bash
.venv/bin/python experiments/11_easy_preservation/run.py --dataset scifact --seed 1337 --r 8 --alpha 8.0 --lora-lr 5e-5 --lambda-easy 1.0 --max-triplets 9190 --early-stop-metric all
```
**Completed**: 2026-05-24 05:02:43 (exit=2)
  ⚠️ **non-zero exit code: 2**

## Phase 2 — Exp 11 SciFact seed=2024 (λ_easy=1.0)
**Started**: 2026-05-24 05:02:43
**Log**: `outputs/_exp11_scifact_seed_2024.log`
**Command**:
```bash
.venv/bin/python experiments/11_easy_preservation/run.py --dataset scifact --seed 2024 --r 8 --alpha 8.0 --lora-lr 5e-5 --lambda-easy 1.0 --max-triplets 9190 --early-stop-metric all
```
**Completed**: 2026-05-24 05:02:45 (exit=2)
  ⚠️ **non-zero exit code: 2**

## Phase 3 — M1b SciFact seed=1337 (in-batch neg robustness)
**Started**: 2026-05-24 05:02:45
**Log**: `outputs/_m1b_scifact_seed_1337.log`
**Command**:
```bash
.venv/bin/python experiments/10_lora_phi/run.py --dataset scifact --seed 1337 --components q,v --r 8 --alpha 8.0 --lora-lr 5e-5 --early-stop-metric all --in-batch-neg --tag-suffix m1b
```

---

## 🔁 Phase 2 RETRY — Exp 11 (after orchestrator Phase 5 done)
**Started**: 2026-05-24 05:04:11
**Reason**: 원래 Phase 2 의 --early-stop-metric arg 누락 으로 fail.
**Completed**: 2026-05-24 05:11:22 (exit=0)

**Artifact**: `outputs/10_lora_phi/scifact/seed_1337/qv_r8_l12_m1b`

- **NDCG@10 all** = 0.6681 (n=300)
- **NDCG@10 confused** = 0.2612 (n=132)
- **Δ all** (n=300): +0.0217 [+0.0084, +0.0361] ✓ positive
- **Δ confused** (n=137): +0.0637 [+0.0392, +0.0898] ✓ positive
- **‖A‖_total** = 8.2211, **‖B‖_total** = 1.2542
- **val_all per epoch**: 0.6838 / 0.6872 / 0.7156
- **val_conf per epoch**: 0.3041 / 0.2911 / 0.2904

## Phase 3 — M1b SciFact seed=2024 (in-batch neg robustness)
**Started**: 2026-05-24 05:11:22
**Log**: `outputs/_m1b_scifact_seed_2024.log`
**Command**:
```bash
.venv/bin/python experiments/10_lora_phi/run.py --dataset scifact --seed 2024 --components q,v --r 8 --alpha 8.0 --lora-lr 5e-5 --early-stop-metric all --in-batch-neg --tag-suffix m1b
```
**Completed**: 2026-05-24 05:20:03 (exit=0)

**Artifact**: `outputs/10_lora_phi/scifact/seed_2024/qv_r8_l12_m1b`

- **NDCG@10 all** = 0.6722 (n=300)
- **NDCG@10 confused** = 0.2731 (n=133)
- **Δ all** (n=300): +0.0258 [+0.0107, +0.0418] ✓ positive
- **Δ confused** (n=137): +0.0771 [+0.0505, +0.1051] ✓ positive
- **‖A‖_total** = 8.3048, **‖B‖_total** = 1.3237
- **val_all per epoch**: 0.6487 / 0.6541 / 0.6618
- **val_conf per epoch**: 0.2556 / 0.2468 / 0.2945

## Phase 4 — M1b NFCorpus seed=1337 (in-batch neg robustness)
**Started**: 2026-05-24 05:20:03
**Log**: `outputs/_m1b_nfcorpus_seed_1337.log`
**Command**:
```bash
.venv/bin/python experiments/10_lora_phi/run.py --dataset nfcorpus --seed 1337 --components q,v --r 8 --alpha 8.0 --lora-lr 5e-5 --early-stop-metric all --max-triplets 9190 --in-batch-neg --tag-suffix m1b
```
**Completed**: 2026-05-24 05:28:04 (exit=0)

**Artifact**: `outputs/10_lora_phi/nfcorpus/seed_1337/qv_r8_l12_m1b`

- **NDCG@10 all** = 0.2231 (n=323)
- **NDCG@10 confused** = 0.0917 (n=226)
- **‖A‖_total** = 9.1910, **‖B‖_total** = 2.0555
- **val_all per epoch**: 0.3909 / 0.2998 / 0.2267
- **val_conf per epoch**: 0.1400 / 0.0966 / 0.0734

## Phase 4 — M1b NFCorpus seed=2024 (in-batch neg robustness)
**Started**: 2026-05-24 05:28:04
**Log**: `outputs/_m1b_nfcorpus_seed_2024.log`
**Command**:
```bash
.venv/bin/python experiments/10_lora_phi/run.py --dataset nfcorpus --seed 2024 --components q,v --r 8 --alpha 8.0 --lora-lr 5e-5 --early-stop-metric all --max-triplets 9190 --in-batch-neg --tag-suffix m1b
```
**Completed**: 2026-05-24 05:36:09 (exit=0)

**Artifact**: `outputs/10_lora_phi/nfcorpus/seed_2024/qv_r8_l12_m1b`

- **NDCG@10 all** = 0.2626 (n=323)
- **NDCG@10 confused** = 0.0886 (n=204)
- **‖A‖_total** = 9.1581, **‖B‖_total** = 1.9683
- **val_all per epoch**: 0.3855 / 0.2692 / 0.2597
- **val_conf per epoch**: 0.1393 / 0.0767 / 0.1090

## Phase 5a — M1+M1b combined SciFact (warmup+clip+in-batch neg)
**Started**: 2026-05-24 05:36:09
**Log**: `outputs/_m1plus1b_scifact.log`
**Command**:
```bash
.venv/bin/python experiments/10_lora_phi/run.py --dataset scifact --seed 42 --components q,v --r 8 --alpha 8.0 --lora-lr 5e-5 --early-stop-metric all --warmup-frac 0.1 --grad-clip 1.0 --in-batch-neg --tag-suffix m1plus1b
```
**Completed**: 2026-05-24 05:44:33 (exit=0)

**Artifact**: `outputs/10_lora_phi/scifact/seed_42/qv_r8_l12_m1plus1b`

- **NDCG@10 all** = 0.6659 (n=300)
- **NDCG@10 confused** = 0.2623 (n=133)
- **Δ all** (n=300): +0.0195 [+0.0050, +0.0340] ✓ positive
- **Δ confused** (n=137): +0.0657 [+0.0409, +0.0914] ✓ positive
- **‖A‖_total** = 8.3448, **‖B‖_total** = 1.2806
- **val_all per epoch**: 0.6708 / 0.6792 / 0.6859
- **val_conf per epoch**: 0.2138 / 0.2101 / 0.2020

## Phase 5b — M1+M1b combined NFCorpus
**Started**: 2026-05-24 05:44:34
**Log**: `outputs/_m1plus1b_nfcorpus.log`
**Command**:
```bash
.venv/bin/python experiments/10_lora_phi/run.py --dataset nfcorpus --seed 42 --components q,v --r 8 --alpha 8.0 --lora-lr 5e-5 --early-stop-metric all --max-triplets 9190 --warmup-frac 0.1 --grad-clip 1.0 --in-batch-neg --tag-suffix m1plus1b
```
**Completed**: 2026-05-24 05:52:28 (exit=0)

**Artifact**: `outputs/10_lora_phi/nfcorpus/seed_42/qv_r8_l12_m1plus1b`

- **NDCG@10 all** = 0.2471 (n=323)
- **NDCG@10 confused** = 0.0906 (n=210)
- **Δ all** (n=323): -0.0828 [-0.1041, -0.0631] ✗ negative
- **Δ confused** (n=169): -0.0088 [-0.0237, +0.0067] (CI 0 포함)
- **‖A‖_total** = 9.3662, **‖B‖_total** = 1.9236
- **val_all per epoch**: 0.3783 / 0.2990 / 0.2557
- **val_conf per epoch**: 0.1510 / 0.1282 / 0.1139

---

## 🏁 ALL OVERNIGHT EXPERIMENTS COMPLETE
**Completed**: 2026-05-24 05:52:28

다음 단계 (사용자 wake 후):
- 결과 review (REPORT.md + 본 `_overnight_results.md`)
- §5f paper section 통합
- 캐비엇 1/2 적용 결과 narrative 확정
**Orchestrator finished**: 2026-05-24 05:52:42, starting Exp 11

### Exp 11 SciFact seed=42 (λ_easy=1.0)
**Started**: 2026-05-24 05:52:42
**Completed**: 2026-05-24 06:03:44 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_42/qv_r8_l12_le1`

- **NDCG@10 all** = 0.6797 (n=300)
- **NDCG@10 confused** = 0.2587
- **Δ all** (n=300): +0.0333 [+0.0131, +0.0545] ✓ positive
- **Δ confused** (n=137): +0.0950 [+0.0577, +0.1335] ✓ positive
- **Δ easy** (n=163): -0.0186 [-0.0362, -0.0037] ✗ negative
- **‖A‖_total** = 8.3999, **‖B‖_total** = 1.4026

### Exp 11 SciFact seed=1337 (λ_easy=1.0)
**Started**: 2026-05-24 06:03:44
**Completed**: 2026-05-24 06:14:42 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_1337/qv_r8_l12_le1`

- **NDCG@10 all** = 0.6784 (n=300)
- **NDCG@10 confused** = 0.2633
- **Δ all** (n=300): +0.0320 [+0.0117, +0.0527] ✓ positive
- **Δ confused** (n=137): +0.0954 [+0.0589, +0.1342] ✓ positive
- **Δ easy** (n=163): -0.0212 [-0.0388, -0.0061] ✗ negative
- **‖A‖_total** = 8.3623, **‖B‖_total** = 1.3861

### Exp 11 SciFact seed=2024 (λ_easy=1.0)
**Started**: 2026-05-24 06:14:42
**Completed**: 2026-05-24 06:25:37 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_2024/qv_r8_l12_le1`

- **NDCG@10 all** = 0.6697 (n=300)
- **NDCG@10 confused** = 0.2543
- **Δ all** (n=300): +0.0233 [-0.0039, +0.0511] (CI 0 포함)
- **Δ confused** (n=137): +0.1133 [+0.0690, +0.1596] ✓ positive
- **Δ easy** (n=163): -0.0524 [-0.0818, -0.0265] ✗ negative
- **‖A‖_total** = 8.8547, **‖B‖_total** = 1.9154

## 🏁 EXP 11 RETRY COMPLETE
**Completed**: 2026-05-24 06:25:37

---

## 🔬 Exp 12 — FN-denoised mined-HN (캐비엇 1 disambiguator)
**Started waiting for E5 doc encoding**: 2026-05-24 17:24:22
**E5 doc encoding done**: 2026-05-24 18:22:23

### Exp 12 SciFact seed=42 (threshold=0.0)
**Started**: 2026-05-24 18:22:23
**Completed**: 2026-05-24 18:31:42 (exit=0)

**Artifact**: `outputs/12_fn_denoised_hn/scifact/seed_42/qv_r8_l12_thresh0`

- **FN denoising**: kept 5832 / 9190 (36.5% removed as likely FN, e5_margin <= 0.0)
- **margin stats**: mean=+0.0126, median=+0.0355, min=-0.7570, max=+0.7567
- **NDCG@10 all** = 0.6388
- **NDCG@10 confused** = 0.2478
- **Δ all** (n=300): -0.0076 [-0.0374, +0.0216] (CI 0 포함)
- **Δ confused** (n=137): +0.0758 [+0.0285, +0.1232] ✓ positive
- **Δ easy** (n=163): -0.0777 [-0.1127, -0.0463] ✗ negative
- **‖A‖_total** = 8.7976, **‖B‖_total** = 1.8830

### Exp 12 SciFact seed=1337 (threshold=0.0)
**Started**: 2026-05-24 18:31:42
**Completed**: 2026-05-24 18:40:58 (exit=0)

**Artifact**: `outputs/12_fn_denoised_hn/scifact/seed_1337/qv_r8_l12_thresh0`

- **FN denoising**: kept 5832 / 9190 (36.5% removed as likely FN, e5_margin <= 0.0)
- **margin stats**: mean=+0.0126, median=+0.0355, min=-0.7570, max=+0.7567
- **NDCG@10 all** = 0.6420
- **NDCG@10 confused** = 0.2496
- **Δ all** (n=300): -0.0045 [-0.0333, +0.0236] (CI 0 포함)
- **Δ confused** (n=137): +0.0791 [+0.0341, +0.1275] ✓ positive
- **Δ easy** (n=163): -0.0747 [-0.1081, -0.0453] ✗ negative
- **‖A‖_total** = 8.7852, **‖B‖_total** = 1.8956

### Exp 12 SciFact seed=2024 (threshold=0.0)
**Started**: 2026-05-24 18:40:58
**Completed**: 2026-05-24 18:50:15 (exit=0)

**Artifact**: `outputs/12_fn_denoised_hn/scifact/seed_2024/qv_r8_l12_thresh0`

- **FN denoising**: kept 5832 / 9190 (36.5% removed as likely FN, e5_margin <= 0.0)
- **margin stats**: mean=+0.0126, median=+0.0355, min=-0.7570, max=+0.7567
- **NDCG@10 all** = 0.6488
- **NDCG@10 confused** = 0.2431
- **Δ all** (n=300): +0.0024 [-0.0259, +0.0314] (CI 0 포함)
- **Δ confused** (n=137): +0.0841 [+0.0373, +0.1314] ✓ positive
- **Δ easy** (n=163): -0.0663 [-0.0990, -0.0373] ✗ negative
- **‖A‖_total** = 8.7817, **‖B‖_total** = 1.9064

## 🏁 EXP 12 COMPLETE
**Completed**: 2026-05-24 18:50:15

---

## 🔬 Exp 11 Extensions — Higher λ + M1b combined
**Started**: 2026-05-24 19:13:31

Pre-commit: .
- **Higher λ Exp 11**: λ=5.0 (5× current), mined HN, 3 seeds
- **M1b + Exp 11 combined**: λ=1.0 + in-batch neg for confused, 3 seeds

### Higher λ Exp 11 SciFact seed=42 (λ_easy=5.0)
**Started**: 2026-05-24 19:13:31
**Completed**: 2026-05-24 19:24:33 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_42/qv_r8_l12_le5`

- **NDCG@10 all** = 0.6876
- **NDCG@10 confused** = 0.2633
- **Δ all** (n=300): +0.0412 [+0.0145, +0.0678] ✓ positive
- **Δ confused** (n=137): +0.1380 [+0.0916, +0.1848] ✓ positive
- **Δ easy** (n=163): -0.0402 [-0.0652, -0.0181] ✗ negative
- **‖A‖_total** = 9.0647, **‖B‖_total** = 1.9875

### Higher λ Exp 11 SciFact seed=1337 (λ_easy=5.0)
**Started**: 2026-05-24 19:24:33

---

## 🔬 FN+EP Variant — FN-denoised + Relational Easy Preservation (Exp 11 flag-based variant)

> **NOTE**: 본 variant 는 *Exp 11 의 `--fn-denoise` + `--lambda-easy 1.0` flag-based extension*. 사용자 결정 에 따라 *Exp 13 designation 부여 안 함* — Exp 13 자리 는 *진짜 새 methodology* (예: (f) Difficulty-aware HN weighting) 위해 reserve. 다만 runner script 의 doc 작성 시점 에 "Exp 13" 으로 명명 했음 → 본 section 으로 정정.

**Started waiting for Exp 11 extensions**: 2026-05-24 19:30:22
**Completed**: 2026-05-24 19:35:31 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_1337/qv_r8_l12_le5`

- **NDCG@10 all** = 0.6814
- **NDCG@10 confused** = 0.2661
- **Δ all** (n=300): +0.0349 [+0.0190, +0.0520] ✓ positive
- **Δ confused** (n=137): +0.0889 [+0.0588, +0.1216] ✓ positive
- **Δ easy** (n=163): -0.0104 [-0.0222, -0.0005] ✗ negative
- **‖A‖_total** = 8.2425, **‖B‖_total** = 1.2082

### Higher λ Exp 11 SciFact seed=2024 (λ_easy=5.0)
**Started**: 2026-05-24 19:35:31
**Completed**: 2026-05-24 19:46:27 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_2024/qv_r8_l12_le5`

- **NDCG@10 all** = 0.6963
- **NDCG@10 confused** = 0.2417
- **Δ all** (n=300): +0.0499 [+0.0260, +0.0747] ✓ positive
- **Δ confused** (n=137): +0.1348 [+0.0905, +0.1815] ✓ positive
- **Δ easy** (n=163): -0.0214 [-0.0399, -0.0051] ✗ negative
- **‖A‖_total** = 8.9961, **‖B‖_total** = 1.9330

### M1b + Exp 11 combined SciFact seed=42 (λ_easy=1.0, in-batch-neg)
**Started**: 2026-05-24 19:46:27
**Completed**: 2026-05-24 19:55:05 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_42/qv_r8_l12_le1_m1b`

- **NDCG@10 all** = 0.6610
- **NDCG@10 confused** = 0.2540
- **Δ all** (n=300): +0.0145 [+0.0007, +0.0287] ✓ positive
- **Δ confused** (n=137): +0.0514 [+0.0266, +0.0778] ✓ positive
- **Δ easy** (n=163): -0.0164 [-0.0306, -0.0046] ✗ negative
- **‖A‖_total** = 8.2629, **‖B‖_total** = 1.2829

### M1b + Exp 11 combined SciFact seed=1337 (λ_easy=1.0, in-batch-neg)
**Started**: 2026-05-24 19:55:05
**Completed**: 2026-05-24 20:03:42 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_1337/qv_r8_l12_le1_m1b`

- **NDCG@10 all** = 0.6596
- **NDCG@10 confused** = 0.2542
- **Δ all** (n=300): +0.0132 [-0.0004, +0.0276] (CI 0 포함)
- **Δ confused** (n=137): +0.0477 [+0.0247, +0.0724] ✓ positive
- **Δ easy** (n=163): -0.0157 [-0.0300, -0.0035] ✗ negative
- **‖A‖_total** = 8.1937, **‖B‖_total** = 1.2225

### M1b + Exp 11 combined SciFact seed=2024 (λ_easy=1.0, in-batch-neg)
**Started**: 2026-05-24 20:03:42
**Completed**: 2026-05-24 20:12:18 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_2024/qv_r8_l12_le1_m1b`

- **NDCG@10 all** = 0.6642
- **NDCG@10 confused** = 0.2604
- **Δ all** (n=300): +0.0178 [+0.0034, +0.0322] ✓ positive
- **Δ confused** (n=137): +0.0578 [+0.0331, +0.0843] ✓ positive
- **Δ easy** (n=163): -0.0159 [-0.0289, -0.0048] ✗ negative
- **‖A‖_total** = 8.2589, **‖B‖_total** = 1.2845

## 🏁 EXP 11 EXTENSIONS COMPLETE
**Completed**: 2026-05-24 20:12:18
**Extensions done, starting Exp 13**: 2026-05-24 20:12:22

**Config**: λ=1.0 (Exp 11 baseline) + FN denoise threshold=0.0 (Exp 12 logic), mined HN kept (hard).  
**Hypothesis (pre-commit)**: hard contrast 유지 + noise 제거 + easy 보존 의 *3-way isolation* → redistribution 해소 여부.

### Exp 13 SciFact seed=42 (λ_easy=1.0, fn-denoise threshold=0.0)
**Started**: 2026-05-24 20:12:22
**Completed**: 2026-05-24 20:20:09 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_42/qv_r8_l12_le1_fnden`

- **FN denoising**: kept 5832 / 9190 (36.5% removed as likely FN)
- **NDCG@10 all** = 0.6794
- **NDCG@10 confused** = 0.2539
- **Δ all** (n=300): +0.0330 [+0.0064, +0.0602] ✓ positive
- **Δ confused** (n=137): +0.1192 [+0.0712, +0.1672] ✓ positive
- **Δ easy** (n=163): -0.0395 [-0.0651, -0.0171] ✗ negative
- **‖A‖_total** = 8.8322, **‖B‖_total** = 1.8557

### Exp 13 SciFact seed=1337 (λ_easy=1.0, fn-denoise threshold=0.0)
**Started**: 2026-05-24 20:20:09
**Completed**: 2026-05-24 20:27:53 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_1337/qv_r8_l12_le1_fnden`

- **FN denoising**: kept 5832 / 9190 (36.5% removed as likely FN)
- **NDCG@10 all** = 0.6618
- **NDCG@10 confused** = 0.2428
- **Δ all** (n=300): +0.0154 [+0.0034, +0.0284] ✓ positive
- **Δ confused** (n=137): +0.0430 [+0.0207, +0.0696] ✓ positive
- **Δ easy** (n=163): -0.0079 [-0.0186, +0.0012] (CI 0 포함)
- **‖A‖_total** = 7.9680, **‖B‖_total** = 0.8536

### Exp 13 SciFact seed=2024 (λ_easy=1.0, fn-denoise threshold=0.0)
**Started**: 2026-05-24 20:27:53
**Completed**: 2026-05-24 20:35:40 (exit=0)

**Artifact**: `outputs/11_easy_preservation/scifact/seed_2024/qv_r8_l12_le1_fnden`

- **FN denoising**: kept 5832 / 9190 (36.5% removed as likely FN)
- **NDCG@10 all** = 0.6798
- **NDCG@10 confused** = 0.2496
- **Δ all** (n=300): +0.0333 [+0.0069, +0.0603] ✓ positive
- **Δ confused** (n=137): +0.1182 [+0.0725, +0.1656] ✓ positive
- **Δ easy** (n=163): -0.0380 [-0.0628, -0.0160] ✗ negative
- **‖A‖_total** = 8.8510, **‖B‖_total** = 1.8659

## 🏁 EXP 13 COMPLETE
**Completed**: 2026-05-24 20:35:40

---

## 🆕 Exp 13 + Exp 14 — Theory-driven new methodologies (pre-committed, result-blind)
**Started**: 2026-05-24 21:37:33

**Pre-commit**: `report/_exp13_14_pre_commit.md` (written BEFORE training, single config per experiment, no sweep).
- **Exp 13** (frozen-direction anchor): λ_dir = 1.0, easy queries per-token cosine deviation penalty.
- **Exp 14** (difficulty-aware HN weighting): α_w = 10, sigmoid weight on margin loss by e5_margin.

### Exp 13 SciFact seed=42 (frozen-direction anchor, λ_dir=1.0)
**Started**: 2026-05-24 21:37:33
**Completed**: 2026-05-24 21:48:33 (exit=0)

**Artifact**: `outputs/13_frozen_direction_anchor/scifact/seed_42/qv_r8_l12_dir1`

- **NDCG@10 all** = 0.6790
- **NDCG@10 confused** = 0.2582
- **Δ all** (n=300): +0.0326 [+0.0119, +0.0538] ✓ positive
- **Δ confused** (n=137): +0.0995 [+0.0635, +0.1363] ✓ positive
- **Δ easy** (n=163): -0.0236 [-0.0450, -0.0059] ✗ negative
- **‖A‖_total** = 8.3646, **‖B‖_total** = 1.3499

### Exp 13 SciFact seed=1337 (frozen-direction anchor, λ_dir=1.0)
**Started**: 2026-05-24 21:48:33
**Completed**: 2026-05-24 21:59:30 (exit=0)

**Artifact**: `outputs/13_frozen_direction_anchor/scifact/seed_1337/qv_r8_l12_dir1`

- **NDCG@10 all** = 0.6743
- **NDCG@10 confused** = 0.2545
- **Δ all** (n=300): +0.0279 [+0.0082, +0.0483] ✓ positive
- **Δ confused** (n=137): +0.0873 [+0.0529, +0.1248] ✓ positive
- **Δ easy** (n=163): -0.0221 [-0.0420, -0.0056] ✗ negative
- **‖A‖_total** = 8.3307, **‖B‖_total** = 1.3358

### Exp 13 SciFact seed=2024 (frozen-direction anchor, λ_dir=1.0)
**Started**: 2026-05-24 21:59:30
**Completed**: 2026-05-24 22:10:24 (exit=0)

**Artifact**: `outputs/13_frozen_direction_anchor/scifact/seed_2024/qv_r8_l12_dir1`

- **NDCG@10 all** = 0.6771
- **NDCG@10 confused** = 0.2606
- **Δ all** (n=300): +0.0307 [+0.0122, +0.0499] ✓ positive
- **Δ confused** (n=137): +0.0877 [+0.0539, +0.1219] ✓ positive
- **Δ easy** (n=163): -0.0172 [-0.0338, -0.0028] ✗ negative
- **‖A‖_total** = 8.3180, **‖B‖_total** = 1.3239

### Exp 14 SciFact seed=42 (difficulty-weighted HN, α_w=10)
**Started**: 2026-05-24 22:10:24
**Completed**: 2026-05-24 22:23:42 (exit=0)

**Artifact**: `outputs/14_difficulty_weighted_hn/scifact/seed_42/qv_r8_l12_diffw10`

- **triplet weight stats**: mean=0.537, median=0.588, std=0.275
- **e5 margin stats**: mean=+0.013, median=+0.036, range=[-0.757, +0.757]
- **NDCG@10 all** = 0.6552
- **NDCG@10 confused** = 0.2492
- **Δ all** (n=300): +0.0088 [-0.0221, +0.0397] (CI 0 포함)
- **Δ confused** (n=137): +0.1001 [+0.0483, +0.1525] ✓ positive
- **Δ easy** (n=163): -0.0679 [-0.1017, -0.0379] ✗ negative
- **‖A‖_total** = 8.9989, **‖B‖_total** = 2.0207

### Exp 14 SciFact seed=1337 (difficulty-weighted HN, α_w=10)
**Started**: 2026-05-24 22:23:42
**Completed**: 2026-05-24 22:36:56 (exit=0)

**Artifact**: `outputs/14_difficulty_weighted_hn/scifact/seed_1337/qv_r8_l12_diffw10`

- **triplet weight stats**: mean=0.537, median=0.588, std=0.275
- **e5 margin stats**: mean=+0.013, median=+0.036, range=[-0.757, +0.757]
- **NDCG@10 all** = 0.6536
- **NDCG@10 confused** = 0.2447
- **Δ all** (n=300): +0.0072 [-0.0140, +0.0277] (CI 0 포함)
- **Δ confused** (n=137): +0.0603 [+0.0257, +0.0975] ✓ positive
- **Δ easy** (n=163): -0.0375 [-0.0611, -0.0173] ✗ negative
- **‖A‖_total** = 8.3835, **‖B‖_total** = 1.4307

### Exp 14 SciFact seed=2024 (difficulty-weighted HN, α_w=10)
**Started**: 2026-05-24 22:36:56
**Completed**: 2026-05-24 22:50:09 (exit=0)

**Artifact**: `outputs/14_difficulty_weighted_hn/scifact/seed_2024/qv_r8_l12_diffw10`

- **triplet weight stats**: mean=0.537, median=0.588, std=0.275
- **e5 margin stats**: mean=+0.013, median=+0.036, range=[-0.757, +0.757]
- **NDCG@10 all** = 0.6493
- **NDCG@10 confused** = 0.2419
- **Δ all** (n=300): +0.0028 [-0.0277, +0.0339] (CI 0 포함)
- **Δ confused** (n=137): +0.0946 [+0.0463, +0.1452] ✓ positive
- **Δ easy** (n=163): -0.0742 [-0.1084, -0.0427] ✗ negative
- **‖A‖_total** = 8.9468, **‖B‖_total** = 2.0491

## 🏁 EXP 13 + EXP 14 COMPLETE
**Completed**: 2026-05-24 22:50:09

---

## 🆕 Exp 16 — Multi-layer per-token anchor (pre-committed, result-blind)
**Pre-commit**: `report/_exp16_pre_commit.md` (BEFORE training, single config, no sweep).
- **Layer set**: {0, 3, 6, 9, 12} (BERT hidden states, CLAUDE.md §1.3 prior finding).
- **λ_dir = 1.0** (Exp 13 동일 scale).
- Tag: `qv_r8_l12_dir1_multilayer`.

### Exp 16 SciFact seed=42 (smoke test)
**Completed**: 2026-05-25 01:14:07
- **NDCG@10 all** = 0.6238
- **Δ all** (n=300): +0.0075 [-0.0162, +0.0307] (CI 0 포함)
- **Δ confused** (n=137): +0.0733 [+0.0355, +0.1123] ✓ positive
- **Δ easy** (n=163): -0.0478 [-0.0751, -0.0238] ✗ negative
- Branch (c) over-restriction 강하게 시사 (Exp 13 +0.033 보다 명백 열등)

### Exp 16 SciFact seed=1337 (multi-layer anchor)
**Started**: 2026-05-25 01:15:16
**Completed**: 2026-05-25 01:27:17 (exit=0)

**Artifact**: `outputs/16_multilayer_anchor/scifact/seed_1337/qv_r8_l12_dir1_multilayer`

- **NDCG@10 all** = 0.6434
- **NDCG@10 confused** = 0.2557
- **Δ all** (n=300): -0.0031 [-0.0274, +0.0209] (CI 0 포함)
- **Δ confused** (n=137): +0.0663 [+0.0280, +0.1074] ✓ positive
- **Δ easy** (n=163): -0.0614 [-0.0892, -0.0364] ✗ negative
- **‖A‖_total** = 8.4533, **‖B‖_total** = 1.5218

### Exp 16 SciFact seed=2024 (multi-layer anchor)
**Started**: 2026-05-25 01:27:18
**Completed**: 2026-05-25 01:39:17 (exit=0)

**Artifact**: `outputs/16_multilayer_anchor/scifact/seed_2024/qv_r8_l12_dir1_multilayer`

- **NDCG@10 all** = 0.6534
- **NDCG@10 confused** = 0.2374
- **Δ all** (n=300): +0.0070 [-0.0160, +0.0301] (CI 0 포함)
- **Δ confused** (n=137): +0.0720 [+0.0344, +0.1096] ✓ positive
- **Δ easy** (n=163): -0.0476 [-0.0749, -0.0235] ✗ negative
- **‖A‖_total** = 8.4581, **‖B‖_total** = 1.5319

## 🏁 EXP 16 COMPLETE
**Completed**: 2026-05-25 01:39:17

---

## 🆕 Cross-dataset ablations — Tier 1 (anchor × {NF, FiQA}) + Tier 2 (Exp 12 × NF)
**Started**: 2026-05-25 04:02:33
**Pre-commit**: `report/_cross_dataset_pre_commit.md` (BEFORE training, same SciFact config, no HP retuning).
- **Tier 1**: anchor Exp 13, λ_dir=1.0, q,v r=8, LR=5e-5
- **Tier 2**: Exp 12 FN-denoised (e5_margin > 0)

### TIER 1 — anchor (Exp 13) cross-dataset

#### anchor (Exp 13) × NFCorpus seed=1337
**Started**: 2026-05-25 04:02:33
**Completed**: 2026-05-25 04:13:56 (exit=1)
  ⚠️ exit=1

#### anchor (Exp 13) × NFCorpus seed=2024
**Started**: 2026-05-25 04:13:56
**Completed**: 2026-05-25 04:25:23 (exit=1)
  ⚠️ exit=1

#### anchor (Exp 13) × FiQA seed=42
**Started**: 2026-05-25 04:25:23
**Completed**: 2026-05-25 05:09:43 (exit=0)

**Artifact**: `outputs/13_frozen_direction_anchor/fiqa/seed_42/qv_r8_l12_dir1`

- **NDCG@10 all** = 0.2670
- **NDCG@10 confused** = 0.1108
- **Δ all** (n=648): -0.0803 [-0.0969, -0.0639] ✗ negative
- **Δ confused** (n=428): -0.0390 [-0.0565, -0.0217] ✗ negative
- **Δ easy** (n=220): -0.1607 [-0.1930, -0.1298] ✗ negative
- **‖A‖_total** = 8.5045, **‖B‖_total** = 1.8516

#### anchor (Exp 13) × FiQA seed=1337
**Started**: 2026-05-25 05:09:43
**Completed**: 2026-05-25 05:54:01 (exit=0)

**Artifact**: `outputs/13_frozen_direction_anchor/fiqa/seed_1337/qv_r8_l12_dir1`

- **NDCG@10 all** = 0.2392
- **NDCG@10 confused** = 0.0980
- **Δ all** (n=648): -0.1081 [-0.1268, -0.0896] ✗ negative
- **Δ confused** (n=428): -0.0543 [-0.0719, -0.0365] ✗ negative
- **Δ easy** (n=220): -0.2129 [-0.2513, -0.1767] ✗ negative
- **‖A‖_total** = 8.5348, **‖B‖_total** = 1.9256

#### anchor (Exp 13) × FiQA seed=2024
**Started**: 2026-05-25 05:54:01
**Completed**: 2026-05-25 06:38:10 (exit=0)

**Artifact**: `outputs/13_frozen_direction_anchor/fiqa/seed_2024/qv_r8_l12_dir1`

- **NDCG@10 all** = 0.2653
- **NDCG@10 confused** = 0.1035
- **Δ all** (n=648): -0.0820 [-0.0983, -0.0653] ✗ negative
- **Δ confused** (n=428): -0.0431 [-0.0599, -0.0263] ✗ negative
- **Δ easy** (n=220): -0.1577 [-0.1900, -0.1258] ✗ negative
- **‖A‖_total** = 8.5209, **‖B‖_total** = 1.8839

### E5-Mistral NFCorpus embedding extraction (Tier 2 prerequisite)
**Started**: 2026-05-25 06:38:10
**q emb done**: 2026-05-25 06:40:24 (exit=0)
**doc emb done**: 2026-05-25 07:23:51 (exit=0)

### TIER 2 — Exp 12 FN-denoised × NFCorpus

#### Exp 12 × NFCorpus seed=42
**Started**: 2026-05-25 07:23:51
**Completed**: 2026-05-25 07:36:58 (exit=0)

**Artifact**: `outputs/12_fn_denoised_hn/nfcorpus/seed_42/qv_r8_l12_thresh0`

- **NDCG@10 all** = 0.0128
- **NDCG@10 confused** = 0.0091
- **Δ all** (n=323): -0.3170 [-0.3517, -0.2835] ✗ negative
- **Δ confused** (n=169): -0.0893 [-0.1114, -0.0679] ✗ negative
- **Δ easy** (n=154): -0.5669 [-0.6058, -0.5277] ✗ negative
- **‖A‖_total** = 9.3979, **‖B‖_total** = 3.2643

#### Exp 12 × NFCorpus seed=1337
**Started**: 2026-05-25 07:36:58
**Completed**: 2026-05-25 07:49:55 (exit=0)

**Artifact**: `outputs/12_fn_denoised_hn/nfcorpus/seed_1337/qv_r8_l12_thresh0`

- **NDCG@10 all** = 0.0142
- **NDCG@10 confused** = 0.0096
- **Δ all** (n=323): -0.3157 [-0.3490, -0.2820] ✗ negative
- **Δ confused** (n=169): -0.0910 [-0.1132, -0.0697] ✗ negative
- **Δ easy** (n=154): -0.5622 [-0.6013, -0.5247] ✗ negative
- **‖A‖_total** = 9.3969, **‖B‖_total** = 3.2825

#### Exp 12 × NFCorpus seed=2024
**Started**: 2026-05-25 07:49:55
**Completed**: 2026-05-25 08:02:56 (exit=0)

**Artifact**: `outputs/12_fn_denoised_hn/nfcorpus/seed_2024/qv_r8_l12_thresh0`

- **NDCG@10 all** = 0.0136
- **NDCG@10 confused** = 0.0091
- **Δ all** (n=323): -0.3163 [-0.3510, -0.2824] ✗ negative
- **Δ confused** (n=169): -0.0885 [-0.1117, -0.0673] ✗ negative
- **Δ easy** (n=154): -0.5663 [-0.6045, -0.5280] ✗ negative
- **‖A‖_total** = 9.3634, **‖B‖_total** = 3.2207

## 🏁 CROSS-DATASET ABLATIONS COMPLETE
**Completed**: 2026-05-25 08:02:56

---

## 🔁 Cross-dataset RETRY — Exp 13 × NFCorpus seed=1337/2024
**Started**: 2026-05-25 08:03:28
**Reason**: 첫 시도 시 `outputs/00_baseline/nfcorpus/seed_{1337,2024}/metrics_per_query.json` 부재로 exit=1.
**Fix**: ColBERT frozen → baseline JSON 은 seed-invariant → seed_42 baseline 을 seed_1337/2024 dir 에 복사.


#### anchor (Exp 13) × NFCorpus seed=1337 (retry)
**Started**: 2026-05-25 08:03:28
**Completed**: 2026-05-25 08:14:46 (exit=0)

**Artifact**: `outputs/13_frozen_direction_anchor/nfcorpus/seed_1337/qv_r8_l12_dir1`

- **NDCG@10 all** = 0.1032
- **NDCG@10 confused** = 0.0338
- **Δ all** (n=323): -0.2267 [-0.2565, -0.1968] ✗ negative
- **Δ confused** (n=169): -0.0634 [-0.0834, -0.0438] ✗ negative
- **Δ easy** (n=154): -0.4059 [-0.4506, -0.3625] ✗ negative
- **‖A‖_total** = 8.6541, **‖B‖_total** = 2.1904

#### anchor (Exp 13) × NFCorpus seed=2024 (retry)
**Started**: 2026-05-25 08:14:46
**Completed**: 2026-05-25 08:26:11 (exit=0)

**Artifact**: `outputs/13_frozen_direction_anchor/nfcorpus/seed_2024/qv_r8_l12_dir1`

- **NDCG@10 all** = 0.1085
- **NDCG@10 confused** = 0.0382
- **Δ all** (n=323): -0.2214 [-0.2508, -0.1921] ✗ negative
- **Δ confused** (n=169): -0.0606 [-0.0811, -0.0415] ✗ negative
- **Δ easy** (n=154): -0.3978 [-0.4409, -0.3544] ✗ negative
- **‖A‖_total** = 8.6518, **‖B‖_total** = 2.1927

## 🏁 CROSS-DATASET RETRY COMPLETE
**Completed**: 2026-05-25 08:26:11
