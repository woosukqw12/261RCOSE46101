# Training Dynamics as a Free False-Negative Signal for IR

> RLHN 같은 LLM-judge 기반 false-negative(FN) re-labeling은 한 번에 $3000이 든다.
> 우리는 **학습 중에 자연스럽게 나오는 training dynamics(margin/rank 시계열)** 만으로
> 비슷하거나 더 좋은 re-training utility를 얻을 수 있음을 보인다.

---

## TL;DR

- **문제**: Hard-negative mining은 false negative(실제 relevant지만 negative로 라벨된 것)를 섞어 학습을 방해함. RLHN(2024)은 GPT-4 judge로 이걸 다시 라벨링하지만 비용이 크다.
- **아이디어**: baseline 1 epoch만 학습하면 (query, neg) 쌍마다 **5가지 training-dynamics 신호**가 공짜로 나온다. 이걸로 FN 후보를 고르면 LLM judge와 14% precision으로 일치하고, **downstream nDCG@10에서는 RLHN full relabel을 능가**.
- **검증**: fiqa (BM25 negs), fiqa_rlhn (BGE negs + RLHN FN GT), nfcorpus, scifact 4개 셋에서 재현.
- **다음**: MS MARCO 규모에서 signal 품질과 budget을 분리한 matched-budget ablation.

### 현재까지 핵심 수치 (fiqa_rlhn, e5-base-unsupervised, 5 epochs)

| Method | 마스킹/재라벨 수 | nDCG@10 | Δ vs Baseline |
|---|---|---|---|
| Baseline | 0 | 0.3841 | — |
| Random K=597 × 3 seeds | 597 | 0.3838 | -0.0003 |
| **Ours mask (mp3+)** | 597 (P=0.136) | 0.3898 | **+0.0057** |
| **Ours relabel (mp3+)** | 597 | **0.3970** | **+0.0129** |
| Oracle mask (RLHN FN) | 1562 | 0.3902 | +0.0061 |
| RLHN full relabel (LLM judge) | 3636 | 0.3648 | −0.0193 |

> **핵심 관찰**: LLM judge로 "정답" FN을 많이 relabel한다고 항상 좋아지지 않음.
> Training utility ≠ LLM judge precision. 우리의 dynamics-기반 selection이 더 safer.

---

## 이 레포의 파이프라인

```
data/processed/<dataset>/train.jsonl         ← BEIR/RLHN 데이터 (query, pos, neg, neg_ids)
   │
   ├─ src/run_dataset_experiment.py           # baseline 학습 + epoch별 score 로깅
   │      └─ experiments/<dataset>/logs_baseline/epoch_*.jsonl
   │
   ├─ src/compute_signals.py                  # boolean criteria (margin/rank/carto)
   │      └─ results/signals_<dataset>/criteria_loss.json
   │
   ├─ src/generate_topk_adj.py                # matched-budget top-K adj per signal
   │      └─ experiments/<dataset>/adj_<signal>_K<N>.json
   │
   ├─ src/train_with_adj.py                   # adj(mask=-1e9) 적용해 재학습
   │      └─ experiments/<dataset>/ckpt_<variant>/
   │
   ├─ src/evaluate_beir_raw.py                # BEIR transfer 평가
   │      └─ experiments/<dataset>/metrics_<variant>.json
   │
   └─ src/matched_budget_analysis.py          # P@K vs RLHN FN GT (학습 없이 offline 분석)
          └─ results/matched_budget.json
```

---

## Training dynamics signals

baseline 1 run의 epoch log (`neg_scores[], pos_score, query_id`)만 있으면 각 (q, neg_i) 쌍에 대해 계산:

| Signal | 정의 | 직관 |
|---|---|---|
| `avg_margin` | mean_epoch(neg_score − pos_score) | 평균적으로 neg가 얼마나 pos를 이김 |
| `final_margin` | 마지막 epoch의 margin | 학습 끝났는데도 안 밀렸으면 수상 |
| `persistent_count` | margin>0인 epoch 수 | **시간적 일관성** (mp3+: ≥3) |
| `rank_top1_count` | neg가 1등인 epoch 수 | rank 기반 confusion |
| `persistent_x_margin` | persistent_count × avg_margin | 합성 신호 |

### fiqa_rlhn 기준 P@K=597 (RLHN FN GT와 일치도)

| Signal | P@597 | × random |
|---|---|---|
| **avg_margin / final_margin** | **0.152** | **3.79×** |
| persistent_count | 0.136 | 3.37× |
| rank_top1_count | 0.134 | 3.33× |
| random baseline | 0.040 | 1.00× |

→ `src/matched_budget_analysis.py`로 재현.

---

## Quick start — 단일 BEIR 데이터셋 (fiqa_rlhn)

```bash
# 0. 환경
conda activate nlp  # torch, transformers, datasets, faiss, beir, tqdm, numpy

# 1. 데이터 (RLHN default-680K fiqa subset + FN GT)
python src/prepare_rlhn_fiqa.py

# 2. baseline 학습 + epoch 로깅
python src/run_dataset_experiment.py --dataset fiqa_rlhn

# 3. dynamics 신호 계산
python src/compute_signals.py \
    --log_dir experiments/fiqa_rlhn/logs_baseline \
    --output_dir results/signals_fiqa_rlhn \
    --source loss

# 4. matched-budget adj 파일들 생성 (avg_margin, persistent_count, random seeds, oracle)
python src/generate_topk_adj.py \
    --log_dir experiments/fiqa_rlhn/logs_baseline \
    --train_path data/processed/fiqa_rlhn/train.jsonl \
    --fn_ground_truth data/processed/fiqa_rlhn/fn_ground_truth.json \
    --output_dir experiments/fiqa_rlhn \
    --budgets 0.015 \
    --signals avg_margin,persistent_count,rank_top1_count \
    --random_seeds 42,123,456

# 5. variant 학습
python src/train_with_adj.py \
    --train_path data/processed/fiqa_rlhn/train.jsonl \
    --adj_path   experiments/fiqa_rlhn/adj_avg_margin_K597.json \
    --ckpt_dir   experiments/fiqa_rlhn/ckpt_avg_margin_K597 \
    --log_dir    experiments/fiqa_rlhn/logs_avg_margin_K597 \
    --epochs 5 --batch_size 16 --lr 2e-5 --num_neg 7 --seed 42

# 6. P@K offline 분석 (학습 필요 없음)
python src/matched_budget_analysis.py \
    --log_dir experiments/fiqa_rlhn/logs_baseline \
    --fn_ground_truth data/processed/fiqa_rlhn/fn_ground_truth.json \
    --train_path data/processed/fiqa_rlhn/train.jsonl \
    --output results/matched_budget.json
```

---

## MS MARCO full ablation — 단일 커맨드

A100 96GB 기준, resumable, unattended (~10-15h):

```bash
BS=128 EPOCHS=3 nohup bash run_msmarco_ablation.sh > run.log 2>&1 &
disown
tail -f logs/msmarco_ablation/master.log
```

[run_msmarco_ablation.sh](run_msmarco_ablation.sh)가 순차 실행하는 stage:

| Stage | 산출물 |
|---|---|
| 0. RLHN msmarco subset 다운로드 | `data/processed/msmarco_rlhn/{train.jsonl, fn_ground_truth.json}` |
| 1. Baseline 학습 + epoch 로깅 | `experiments/msmarco_rlhn/{ckpt_baseline, logs_baseline}` |
| 2. Dynamics 신호 계산 | `results/signals_msmarco_rlhn/criteria_loss.json` |
| 3. Matched-budget adj 생성 | `experiments/msmarco_rlhn/adj_*.json` (signals × budgets × random seeds + oracle) |
| 4. 모든 variant 학습 | `experiments/msmarco_rlhn/ckpt_<variant>/` |
| 5. BEIR 9-dataset 평가 | `experiments/msmarco_rlhn/metrics_*.json` |
| 6. 취합 | `experiments/msmarco_rlhn/summary.json` |

환경변수 오버라이드:
```bash
BS=128                                    # A100 96GB 기준
EPOCHS=3
BUDGETS=0.01,0.02,0.04                    # 여러 값이면 budget sweep
SIGNALS=avg_margin,persistent_count,rank_top1_count,persistent_x_margin
RANDOM_SEEDS=42,123
BEST_SIG=avg_margin                       # budget sweep 주력 signal
```

### 돌리는 variant (총 ~13 runs)

| 그룹 | 개수 | 목적 |
|---|---|---|
| Baseline | 1 | 기준선 |
| Matched-budget signals | 4 | signal 품질 ablation (같은 K) |
| Random matched-budget | 2 | 통계적 통제 |
| mp3+ threshold 방식 | 1 | 원래 threshold vs continuous top-K |
| Oracle (RLHN FN mask) | 1 | 상한선 |
| RLHN full relabel | 1 | 논문 비교 대상 |
| Budget sweep (best signal × 2-3) | 2-3 | sweet spot 검증 |

### 이걸로 답할 질문들
1. **P@K 승자가 downstream 승자인가?** — avg_margin vs persistent_count vs rank_top1
2. **같은 budget에서 dynamics selection이 random보다 유의미한가?**
3. **시간성(persistence) vs magnitude(avg_margin)** — 어느 signal 형태가 더 유용한가
4. **LLM judge = training utility?** — Oracle vs dynamics-based selection 비교
5. **MSMARCO 규모에서 RLHN full relabel이 도움이 되는가** (fiqa_rlhn에선 악화)
6. **Budget sweet spot 위치**

---

## 레포 구조

```
src/
├── run_dataset_experiment.py      # 단일 데이터셋 baseline 학습 파이프라인
├── train_with_adj.py              # adj 적용 학습 (variant 학습용)
├── compute_signals.py             # epoch logs → boolean criteria JSON
├── generate_topk_adj.py           # 신호별 matched-budget adj 파일 생성 (NEW)
├── matched_budget_analysis.py     # P@K vs RLHN GT offline 분석 (NEW)
├── prepare_rlhn_fiqa.py           # RLHN default-680K fiqa subset 추출
├── prepare_rlhn_msmarco.py        # MSMARCO용 동일 스크립트 (NEW)
├── prepare_rlhn_relabeled_fiqa.py # rlhn-680K 사용(relabel 베이스라인)
├── prepare_relabeled_data.py      # FN을 새 positive로 확장한 학습 데이터 생성
├── prepare_ablation_adj.py        # oracle/random 계열 adj 생성
├── eval_dynamics_vs_rlhn.py       # criteria vs FN GT F1 계산
├── evaluate_beir_raw.py           # BEIR 원본 데이터셋 transfer 평가
└── ...

experiments/<dataset>/
├── ckpt_<variant>/                # safetensors + tokenizer
├── logs_<variant>/epoch_*.jsonl   # per-epoch (qid, pos_score, neg_scores)
├── adj_<variant>.json             # {qid: [adj_per_neg]} — -1e9이면 mask
└── metrics_<variant>.json         # BEIR 평가 결과

results/
├── signals_<dataset>/criteria_loss.json   # 22개 boolean criteria의 pair 리스트
├── matched_budget.json                    # signal별 P@K 테이블
└── fiqa_rlhn_dynamics_vs_gt.json          # F1 테이블

run_msmarco_ablation.sh              # MSMARCO 전체 ablation orchestrator (NEW)
```

---

## 핵심 insight (지금까지)

1. **mp3+가 이긴 진짜 이유는 sparsity sweet spot**. 원래 `persistent≥3`이 1.55% budget에 우연히 떨어진 것. 같은 K에서는 `avg_margin`이 P@K 더 높음. → MSMARCO에서 "signal 형태 × budget"을 분리한 matched-budget ablation으로 확인 중.

2. **RLHN full relabel이 fiqa_rlhn에서 오히려 악화**(-0.0193). LLM judge가 본 FN 3636개를 전부 positive로 옮기면 5500-query 단일-도메인 학습 분포가 망가짐. 680K multi-domain에서는 비율이 0.5%로 희석돼 안전하다는 가설.

3. **Relabel > Mask** (우리 mp3+에서 +0.0129 vs +0.0057). FN을 버리는 것보다 양성 신호로 쓰는 게 정보량↑.

4. **Training utility ≠ LLM-judge precision**. Dynamics signal은 "모델이 지금 헷갈리는 것 ∩ 실제 FN"을 근사 — 이 교집합이 학습에 더 유용.

---

## 참고

- Model: `intfloat/e5-base-unsupervised`
- Loss: InfoNCE + in-batch negs + mined hard negs (num_neg=7)
- Precision: bf16 autocast
- Evaluation: BEIR (nDCG@10, MAP@10, Recall@100)
- Related: [RLHN (2024)](https://arxiv.org/abs/...), BGE hard negatives, cartography (Swayamdipta et al. 2020)
