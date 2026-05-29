"""Diagnostic B on mediation checkpoints — Phase 2b vs M1 vs M1b.

목적: M1 (warmup+clip) 과 M1b (in-batch neg) 가 *실제로* representation collapse 를
완화했는지 직접 측정 (reviewer agent 의 mechanism 검증 요구).

기존 `_repr_collapse_diagnostic.py` (Phase 2b 의 collapse 확인) 의 *mediation 확장*.

CPU 강제 — GPU 는 FiQA M1b queue 진행 중 (방해 회피).

측정:
  1. Doc-pair cosine (mean-pooled)
  2. Token-pair cosine
  3. Doc-mean / token effective rank (singular spectrum perplexity)

비교 조건 (per dataset ∈ {scifact, nfcorpus, fiqa}):
  - phase_2b: 기존 LoRA Phase 2b (qv_r8_l12)
  - m1: warmup+clip mediation (qv_r8_l12_m1)
  - m1b: in-batch neg mediation (qv_r8_l12_m1b)

Output:
  report/figures/_repr_collapse_mediation/{repr_collapse_mediation.{pdf,png},
                                            repr_collapse_mediation_data.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colbert_hook import ColBERTConfig, ColBERTv2  # noqa: E402
from src.configs import BASELINE  # noqa: E402
from src.data import load_beir  # noqa: E402
from src.evaluate import encode_corpus  # noqa: E402
from src.lora import inject_lora_into_bert  # noqa: E402
from src.utils.repro import set_seed  # noqa: E402

# CPU 강제 (GPU 점유 회피)
DEVICE = torch.device("cpu")
DATASETS = ["scifact", "nfcorpus", "fiqa"]
CONDITIONS = ["phase_2b", "m1", "m1b"]   # tag suffix: "", "_m1", "_m1b"
N_DOC_SAMPLE = 300       # 500 → 300 (CPU 속도 고려)
N_DOC_PAIRS = 3000
N_TOKEN_PAIRS = 6000
TOP_SV = 30


def effective_rank(singular_values: np.ndarray) -> float:
    s2 = singular_values.astype(np.float64) ** 2
    if s2.sum() <= 0:
        return 0.0
    p = s2 / s2.sum()
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum()
    return float(np.exp(ent))


def load_lora_into_model(model: ColBERTv2, ckpt_path: Path, r=8):
    """Inject LoRA + load checkpoint."""
    lora_params = inject_lora_into_bert(
        model.bert, target_components=["q", "v"], layers=None,
        r=r, alpha=None, init_std=0.02,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_state = state["lora"]
    keys = sorted(lora_state.keys(), key=lambda k: int(k.split("_")[-1]))
    assert len(keys) == len(lora_params)
    with torch.no_grad():
        for k, p in zip(keys, lora_params):
            p.data.copy_(lora_state[k].to(p.device))
    return lora_params


def sample_doc_subset(corpus: dict, n: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    dids = list(corpus.keys())
    idx = rng.choice(len(dids), size=min(n, len(dids)), replace=False)
    return {dids[i]: corpus[dids[i]] for i in idx}


def measure(model: ColBERTv2, corpus_sample: dict, label: str) -> dict:
    """Encode + measure collapse metrics."""
    dids, d_emb, d_mask = encode_corpus(model, corpus_sample, DEVICE, batch_size=32)
    N = d_emb.shape[0]

    # Doc mean-pool
    mask_f = d_mask.float().unsqueeze(-1)
    n_valid = mask_f.sum(dim=1).clamp(min=1.0)
    d_mean = (d_emb * mask_f).sum(dim=1) / n_valid
    d_mean = F.normalize(d_mean, p=2, dim=-1)

    rng = np.random.default_rng(0)
    ia = rng.integers(0, N, size=N_DOC_PAIRS)
    ib = rng.integers(0, N, size=N_DOC_PAIRS)
    mask_pair = ia != ib
    ia = ia[mask_pair]; ib = ib[mask_pair]
    doc_pair_cos = (d_mean[ia] * d_mean[ib]).sum(dim=-1).numpy()

    valid_idx = d_mask.nonzero(as_tuple=False)
    tokens = d_emb[valid_idx[:, 0], valid_idx[:, 1]]
    n_tok = tokens.shape[0]
    ta = rng.integers(0, n_tok, size=N_TOKEN_PAIRS)
    tb = rng.integers(0, n_tok, size=N_TOKEN_PAIRS)
    mask_tok = ta != tb
    ta = ta[mask_tok]; tb = tb[mask_tok]
    tok_pair_cos = (tokens[ta] * tokens[tb]).sum(dim=-1).numpy()

    sv_doc = torch.linalg.svdvals(d_mean.float()).numpy()
    eff_rank_doc = effective_rank(sv_doc)

    if n_tok > 8000:
        sub_idx = rng.choice(n_tok, size=8000, replace=False)
        token_subset = tokens[sub_idx]
    else:
        token_subset = tokens
    sv_tok = torch.linalg.svdvals(token_subset.float()).numpy()
    eff_rank_tok = effective_rank(sv_tok)

    return {
        "label": label,
        "n_docs": int(N),
        "n_tokens": int(n_tok),
        "doc_pair_cos_mean": float(doc_pair_cos.mean()),
        "doc_pair_cos_std": float(doc_pair_cos.std()),
        "tok_pair_cos_mean": float(tok_pair_cos.mean()),
        "tok_pair_cos_std": float(tok_pair_cos.std()),
        "doc_effective_rank": eff_rank_doc,
        "tok_effective_rank": eff_rank_tok,
        "doc_singular_values_top": sv_doc[:TOP_SV].tolist(),
        "tok_singular_values_top": sv_tok[:TOP_SV].tolist(),
    }


def main():
    cfg = BASELINE
    set_seed(42)
    print(f"device = {DEVICE} (CPU forced — GPU 점유 회피)")

    out_dir = PROJECT_ROOT / "report/figures/_repr_collapse_mediation"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "repr_collapse_mediation_data.json"

    results = {}
    if data_path.exists():
        results = json.loads(data_path.read_text())
        print(f"resumed prior results: {sorted(results.keys())}")

    cond_to_tag = {"phase_2b": "qv_r8_l12", "m1": "qv_r8_l12_m1", "m1b": "qv_r8_l12_m1b"}

    for dataset in DATASETS:
        corpus, _, _ = load_beir(dataset, split="test")
        sample = sample_doc_subset(corpus, N_DOC_SAMPLE, seed=42)
        print(f"\n=== {dataset}: sampled {len(sample)} / {len(corpus)} docs ===")

        for cond in CONDITIONS:
            key = f"{dataset}_{cond}"
            if key in results:
                print(f"  skip {key} (cached)")
                continue
            tag = cond_to_tag[cond]
            ckpt = (PROJECT_ROOT / "outputs/10_lora_phi" / dataset / "seed_42"
                    / tag / "module_final.pt")
            if not ckpt.exists():
                print(f"  WARN: missing ckpt {ckpt}")
                continue
            print(f"  measure {key} ({tag})")
            model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(DEVICE)
            load_lora_into_model(model, ckpt, r=8)
            model.to(DEVICE)
            model.eval()
            r = measure(model, sample, label=key)
            results[key] = r
            data_path.write_text(json.dumps(results, indent=2))
            print(f"    doc_cos μ={r['doc_pair_cos_mean']:+.4f} σ={r['doc_pair_cos_std']:.4f}")
            print(f"    tok_cos μ={r['tok_pair_cos_mean']:+.4f} σ={r['tok_pair_cos_std']:.4f}")
            print(f"    eff_rank: doc={r['doc_effective_rank']:.2f} tok={r['tok_effective_rank']:.2f}")
            del model

    # ----------------------- summary
    print("\n" + "=" * 100)
    print(f"{'condition':<25s} {'doc_cos μ':>11s} {'tok_cos μ':>11s} {'eff_doc':>10s} {'eff_tok':>10s}")
    print("-" * 100)
    for ds in DATASETS:
        for cond in CONDITIONS:
            k = f"{ds}_{cond}"
            if k not in results:
                continue
            r = results[k]
            print(f"{k:<25s} {r['doc_pair_cos_mean']:>+11.4f} {r['tok_pair_cos_mean']:>+11.4f} "
                  f"{r['doc_effective_rank']:>10.2f} {r['tok_effective_rank']:>10.2f}")
    print("=" * 100)

    # ----------------------- figure
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # 2 rows × 3 cols (datasets):
    #   row 1: doc-pair cos μ (bar) — Phase 2b vs M1 vs M1b
    #   row 2: eff_rank doc + tok (grouped bar)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    color_map = {"phase_2b": "#cc4444", "m1": "#e0a040", "m1b": "#3b7e3b"}
    label_map = {"phase_2b": "Phase 2b", "m1": "M1 (warmup+clip)", "m1b": "M1b (in-batch neg)"}

    for j, ds in enumerate(DATASETS):
        # Row 1: doc-pair cos bar
        ax = axes[0, j]
        for i, cond in enumerate(CONDITIONS):
            k = f"{ds}_{cond}"
            if k not in results:
                ax.text(i, 0.5, "—", ha="center", va="center", fontsize=12)
                continue
            r = results[k]
            ax.bar(i, r["doc_pair_cos_mean"], color=color_map[cond], label=label_map[cond])
            ax.text(i, r["doc_pair_cos_mean"] + 0.02, f"{r['doc_pair_cos_mean']:+.3f}",
                    ha="center", fontsize=8)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([label_map[c] for c in CONDITIONS], rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.5)
        ax.set_ylabel("random doc-pair cosine μ")
        ax.set_title(f"{ds} — collapse magnitude (lower = better)")
        ax.grid(axis="y", alpha=0.3)

        # Row 2: eff_rank doc + tok bar (grouped)
        ax = axes[1, j]
        w = 0.35
        x = np.arange(len(CONDITIONS))
        for i, cond in enumerate(CONDITIONS):
            k = f"{ds}_{cond}"
            if k not in results:
                continue
            r = results[k]
            ax.bar(x[i] - w/2, r["doc_effective_rank"], w, color=color_map[cond], alpha=0.7,
                   label="doc" if i == 0 else None)
            ax.bar(x[i] + w/2, r["tok_effective_rank"], w, color=color_map[cond], alpha=1.0,
                   hatch="//", label="tok" if i == 0 else None)
            ax.text(x[i] - w/2, r["doc_effective_rank"] + 0.5, f"{r['doc_effective_rank']:.1f}",
                    ha="center", fontsize=7)
            ax.text(x[i] + w/2, r["tok_effective_rank"] + 0.5, f"{r['tok_effective_rank']:.1f}",
                    ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([label_map[c] for c in CONDITIONS], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("effective rank (higher = less collapse)")
        ax.set_title(f"{ds} — eff_rank (doc/tok)")
        if j == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Diagnostic B on mediation checkpoints — *direct* collapse measurement",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"repr_collapse_mediation.{ext}")
    plt.close(fig)
    print(f"\nfigure → {out_dir}/repr_collapse_mediation.{{pdf,png}}")
    print(f"data   → {data_path}")


if __name__ == "__main__":
    main()
