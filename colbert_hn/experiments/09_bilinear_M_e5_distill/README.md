# 09_bilinear_M_e5_distill — Bilinear M + E5-Mistral Margin-MSE distillation

## 목적

08 의 *pairwise margin only* 학습이 **rank-1 collapse** 로 끝남 (UV^T 의 effective rank ≈ 1) — r=8 의 capacity 미활용. 본 실험은 *teacher 의 ranking knowledge* 를 추가 supervision 으로 사용해 **multi-axis bilinear interaction 의 활성화** 시도.

## 수식

Loss:
$$\mathcal{L} = \mathcal{L}_{\text{rank}} + \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{distill}}$$

$$\mathcal{L}_{\text{rank}} = \frac{1}{B} \sum_b \max(0, m - (s^{(b)}_{\text{pos}} - s^{(b)}_{\text{hn}}))$$

$$\mathcal{L}_{\text{distill}} = \frac{1}{B} \sum_b \big( (s^{(b)}_{\text{pos}} - s^{(b)}_{\text{hn}}) - \tau \cdot (\text{e5}^{(b)}_{\text{pos}} - \text{e5}^{(b)}_{\text{hn}}) \big)^2$$

- $s_{\text{pos}}, s_{\text{hn}}$ : student (bilinear M) MaxSim 점수
- $\text{e5}_{\text{pos}}, \text{e5}_{\text{hn}}$ : E5-Mistral-7B-Instruct cosine 점수
- $\tau$ : `teacher_scale` (default 8.0) — E5 cosine margin (~0.03 magnitude) → ColBERT margin (~0.2) scale
- $\lambda_{\text{distill}}$ : distillation 가중치

Teacher (E5) margin 은 *pre-computed* train query embedding + corpus embedding (양쪽 L2-normed) 의 cosine 으로 batch 단위 lookup. 학습 시간 minimal overhead.

## Teacher 자료 — E5-Mistral-7B-Instruct

| 파일 | 내용 |
|---|---|
| `data/e5_teacher/e5_train_q_emb_scifact.pt` | 809 train query embeddings (4096-d, fp16) — 본 repo `data/e5_teacher/extract_train_queries.py` 가 생성 (~85 초) |
| `data/e5_teacher/e5_topk_scifact.pt` | 5183 corpus embeddings (4096-d, fp16) — `nlp_term_project/phase_04/01_extract_e5/extract.py` 산출물 복사 |

## 가설

| H | 기준 | 의미 |
|---|---|---|
| **H09a** | NDCG@10 all 의 CI 하한 > 08 (0.6439) | Distillation 이 form-change 의 lever 활성 |
| **H09b** | NDCG@10 all 의 CI 하한 > K-sweep ceiling (0.6614) | **Stage 2 critical pass** — translation family ceiling 위로 |
| H09c | UV^T 의 effective rank > 08 의 rank-1 collapse | E5 supervision 이 multi-axis 활용 활성화 |
| H09d | all-slice Δ vs baseline CI 하한 ≥ -0.005 | anchor preservation |

만약 H09a 만 통과 (NDCG↑ but not above ceiling) → distillation 의 effect 는 있지만 form-change 자체의 lever 부족 → 18 LoRA 검토.
만약 H09b 통과 → 10 r sweep + 11-12 cross-dataset 본격 진행.

## 학습 design (08 와 동일 + distill)

| 항목 | 값 |
|---|---|
| Loss | pairwise margin + λ × Margin-MSE distill |
| Optimizer | AdamW (LR=1e-4, WD=1e-4) |
| λ_distill | 1.0 (default; sweep ∈ {0.1, 0.5, 1.0, 5.0} 가능) |
| teacher_scale τ | 8.0 (E5 margin → ColBERT scale) |
| Init | small_random (zero-init pathology 회피, 08 와 동일) |
| Batch / Epochs / Patience | 32 / 5 / 2 |
| r | 8 |
| Dataset | SciFact (seed 42) |

## 실행

```bash
# Step 1 (선행, 한 번만): E5 train query 추출 (~1.5 분 on MPS)
.venv/bin/python data/e5_teacher/extract_train_queries.py --dataset scifact

# Step 2: 09 학습 + 평가 (~15 분)
.venv/bin/python experiments/09_bilinear_M_e5_distill/run.py \
    --dataset scifact --seed 42 --r 8 --lambda-distill 1.0

# (sweep 시)
for ld in 0.1 0.5 1.0 5.0; do
    .venv/bin/python experiments/09_bilinear_M_e5_distill/run.py \
        --dataset scifact --seed 42 --r 8 --lambda-distill $ld
done
```

Artifact: `outputs/09_bilinear_M_e5_distill/scifact/seed_42/r_8_ld_{LD}/...` (LD 의 "." → "p").

## 비판적 review

1. **teacher_scale 의 고정값 8.0 이 적절한가?**: ColBERT margin ~0.2 / E5 cosine margin ~0.03 의 비율 ~7. 8.0 은 근사. λ_distill 의 효과에 묶임 — λ × τ² 이 effective distill weight. λ sweep 으로 흡수 가능.
2. **Margin-MSE 의 raw 형식 vs 정규화 형식**: Hofstätter 2020 의 Margin-MSE 는 같은 scale 의 student/teacher 가정. 본 형식의 차이는 *scale-aware lambda* 로 보완.
3. **E5 의 *cross-encoder-quality margin* 의 보존 충분성**: E5-Mistral 7B 는 *bi-encoder* (한 번 인코딩 후 dot product) 라 진정한 cross-encoder 보다 표현력 ↓. *Cross-encoder soft margin* (e.g., MonoT5) 의 *upper bound* 는 더 위에 있을 수 있음.
4. **Train query 만 사용**: train 의 ranking knowledge 만 distill. Test domain 의 representation 변화는 *unsupervised*. domain shift 위험.
5. **본 실험은 r=8 단독**: r ∈ {1, 4, 16, 32} 의 sweep 은 10_bilinear_rank_sweep 에서. 본 실험 결과 보고 결정.

## 상세 보고서

[`report/09_bilinear_M_e5_distill_report.md`](../../report/09_bilinear_M_e5_distill_report.md) — 실행 후 작성.
