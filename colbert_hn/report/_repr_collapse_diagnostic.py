"""Diagnostic B: Representation Collapse Measurement (encoder output space).

Reviewer-feedback agenda 의 *진단 우선* 단계.

가설: NFCorpus / FiQA Phase 2b catastrophic (Δ all = −0.320 / −0.347) 는
*encoder output (token-embedding) space* 의 *representation collapse* 가
원인 — i.e., docs 가 평균적으로 너무 비슷해져서 MaxSim 의 discrimination
power 가 무너졌다.

§5e 의 rank-collapse 와 다른 현상:
  - §5e: parameter space (ΔW = BA) 의 singular spectrum collapse → method
    capacity utilization 의 universal 패턴.
  - 본 진단: encoder *output* (token-embedding) space 의 spectrum collapse
    → catastrophic failure 의 *결과* mechanism.

측정:
  1. Random doc-pair pairwise cosine similarity 분포 (collapse → 평균 ↑, spread ↓)
  2. Per-token random pair cosine 분포 (token-level MaxSim space)
  3. Doc embedding (mean-pooled) matrix 의 singular spectrum + effective rank
  4. Per-token embedding (sample) matrix 의 singular spectrum + effective rank

비교 조건 (per dataset ∈ {nfcorpus, fiqa, scifact}):
  - frozen: no LoRA (baseline)
  - lora_2b: qv_r8_l12 module_final.pt 로 LoRA inject 후 재 encode

Output:
  report/figures/_repr_collapse/{repr_collapse.{pdf,png}, repr_collapse_data.json}
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
from src.utils.repro import get_device, set_seed  # noqa: E402

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
DATASETS = ["nfcorpus", "fiqa", "scifact"]
N_DOC_SAMPLE = 500       # sample 500 docs per corpus (same N for cross-comparison)
N_DOC_PAIRS = 5000       # random doc pair samples
N_TOKEN_PAIRS = 10000    # random token pair samples
TOP_SV = 30              # plot top-30 singular values
LORA_TAG = "qv_r8_l12"   # Phase 2b config


def effective_rank(singular_values: np.ndarray) -> float:
    s2 = singular_values.astype(np.float64) ** 2
    if s2.sum() <= 0:
        return 0.0
    p = s2 / s2.sum()
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum()
    return float(np.exp(ent))


def load_lora_into_model(model: ColBERTv2, ckpt_path: Path, components, r):
    """Inject LoRA layers into model.bert and copy adapter weights from ckpt."""
    lora_params = inject_lora_into_bert(
        model.bert, target_components=components, layers=None,
        r=r, alpha=None, init_std=0.02,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_state = state["lora"]
    keys = sorted(lora_state.keys(), key=lambda k: int(k.split("_")[-1]))
    assert len(keys) == len(lora_params), (
        f"adapter count mismatch: ckpt={len(keys)} expected={len(lora_params)}"
    )
    with torch.no_grad():
        for k, p in zip(keys, lora_params):
            p.data.copy_(lora_state[k].to(p.device))
    return lora_params


def sample_doc_subset(corpus: dict, n: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    dids = list(corpus.keys())
    idx = rng.choice(len(dids), size=min(n, len(dids)), replace=False)
    return {dids[i]: corpus[dids[i]] for i in idx}


def measure(model: ColBERTv2, corpus_sample: dict, device, label: str) -> dict:
    """Encode corpus sample and measure representation collapse metrics."""
    dids, d_emb, d_mask = encode_corpus(model, corpus_sample, device, batch_size=64)
    # d_emb: (N, T_max, D=128), d_mask: (N, T_max)  (already L2-normalised per token)

    N = d_emb.shape[0]

    # ---- Per-doc (mean-pooled) cosine analysis ---------------------------
    mask_f = d_mask.float().unsqueeze(-1)  # (N, T, 1)
    n_valid = mask_f.sum(dim=1).clamp(min=1.0)  # (N, 1)
    d_mean = (d_emb * mask_f).sum(dim=1) / n_valid  # (N, 128)
    d_mean = F.normalize(d_mean, p=2, dim=-1)

    rng = np.random.default_rng(0)
    ia = rng.integers(0, N, size=N_DOC_PAIRS)
    ib = rng.integers(0, N, size=N_DOC_PAIRS)
    mask_pair = ia != ib
    ia = ia[mask_pair]; ib = ib[mask_pair]
    doc_pair_cos = (d_mean[ia] * d_mean[ib]).sum(dim=-1).numpy()

    # ---- Per-token cosine analysis ----------------------------------------
    # Gather all valid tokens into a single (N_tok, 128) matrix
    valid_idx = d_mask.nonzero(as_tuple=False)  # (N_tok, 2): doc_i, tok_j
    tokens = d_emb[valid_idx[:, 0], valid_idx[:, 1]]  # (N_tok, 128)
    n_tok = tokens.shape[0]
    ta = rng.integers(0, n_tok, size=N_TOKEN_PAIRS)
    tb = rng.integers(0, n_tok, size=N_TOKEN_PAIRS)
    mask_tok = ta != tb
    ta = ta[mask_tok]; tb = tb[mask_tok]
    tok_pair_cos = (tokens[ta] * tokens[tb]).sum(dim=-1).numpy()

    # ---- Singular spectrum analysis --------------------------------------
    # Doc-mean matrix (N, 128)
    sv_doc = torch.linalg.svdvals(d_mean.float()).numpy()
    eff_rank_doc = effective_rank(sv_doc)
    sv_doc_norm = sv_doc / sv_doc[0]   # normalize to top-1 for shape comparison

    # Token matrix subsample (10K tokens × 128) — full matrix can be huge
    if n_tok > 10000:
        sub_idx = rng.choice(n_tok, size=10000, replace=False)
        token_subset = tokens[sub_idx]
    else:
        token_subset = tokens
    sv_tok = torch.linalg.svdvals(token_subset.float()).numpy()
    eff_rank_tok = effective_rank(sv_tok)
    sv_tok_norm = sv_tok / sv_tok[0]

    return {
        "label": label,
        "n_docs": int(N),
        "n_tokens": int(n_tok),
        # doc-level
        "doc_pair_cos_mean": float(doc_pair_cos.mean()),
        "doc_pair_cos_std": float(doc_pair_cos.std()),
        "doc_pair_cos_dist": doc_pair_cos.tolist(),
        "doc_effective_rank": eff_rank_doc,
        "doc_singular_values_top": sv_doc[:TOP_SV].tolist(),
        "doc_singular_values_norm_top": sv_doc_norm[:TOP_SV].tolist(),
        # token-level
        "tok_pair_cos_mean": float(tok_pair_cos.mean()),
        "tok_pair_cos_std": float(tok_pair_cos.std()),
        "tok_pair_cos_dist": tok_pair_cos.tolist(),
        "tok_effective_rank": eff_rank_tok,
        "tok_singular_values_top": sv_tok[:TOP_SV].tolist(),
        "tok_singular_values_norm_top": sv_tok_norm[:TOP_SV].tolist(),
    }


def main():
    cfg = BASELINE
    set_seed(42)
    device = get_device(None)
    print(f"device = {device}")

    out_dir = PROJECT_ROOT / "report/figures/_repr_collapse"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "repr_collapse_data.json"

    results = {}
    if data_path.exists():
        results = json.loads(data_path.read_text())
        print(f"resumed prior results: {list(results.keys())}")

    for dataset in DATASETS:
        corpus, _, _ = load_beir(dataset, split="test")
        sample = sample_doc_subset(corpus, N_DOC_SAMPLE, seed=42)
        print(f"\n=== {dataset}: sampled {len(sample)} / {len(corpus)} docs ===")

        # frozen baseline
        key_frozen = f"{dataset}_frozen"
        if key_frozen not in results:
            print(f"  measure {key_frozen} (no LoRA)")
            model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
            r = measure(model, sample, device, label=key_frozen)
            results[key_frozen] = r
            data_path.write_text(json.dumps(results, indent=2))
            print(f"    doc_cos μ={r['doc_pair_cos_mean']:+.4f} σ={r['doc_pair_cos_std']:.4f}")
            print(f"    tok_cos μ={r['tok_pair_cos_mean']:+.4f} σ={r['tok_pair_cos_std']:.4f}")
            print(f"    eff_rank: doc={r['doc_effective_rank']:.2f} tok={r['tok_effective_rank']:.2f}")
            del model
        else:
            print(f"  skip {key_frozen} (cached)")

        # LoRA Phase 2b
        key_lora = f"{dataset}_lora_2b"
        if key_lora not in results:
            ckpt_path = (PROJECT_ROOT / "outputs/10_lora_phi" / dataset / "seed_42"
                         / LORA_TAG / "module_final.pt")
            if not ckpt_path.exists():
                print(f"  WARN: checkpoint missing {ckpt_path}")
                continue
            print(f"  measure {key_lora} (LoRA Phase 2b)")
            model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(device)
            load_lora_into_model(model, ckpt_path, components=["q", "v"], r=8)
            model.to(device)  # move LoRA params
            r = measure(model, sample, device, label=key_lora)
            results[key_lora] = r
            data_path.write_text(json.dumps(results, indent=2))
            print(f"    doc_cos μ={r['doc_pair_cos_mean']:+.4f} σ={r['doc_pair_cos_std']:.4f}")
            print(f"    tok_cos μ={r['tok_pair_cos_mean']:+.4f} σ={r['tok_pair_cos_std']:.4f}")
            print(f"    eff_rank: doc={r['doc_effective_rank']:.2f} tok={r['tok_effective_rank']:.2f}")
            del model
        else:
            print(f"  skip {key_lora} (cached)")

    # -------------------------------------------------------------------- summary
    print("\n" + "=" * 90)
    print(f"{'cond':<22s} {'doc_cos μ':>12s} {'doc_cos σ':>10s} "
          f"{'tok_cos μ':>12s} {'eff_rank_doc':>14s} {'eff_rank_tok':>14s}")
    print("-" * 90)
    for ds in DATASETS:
        for cond in ["frozen", "lora_2b"]:
            k = f"{ds}_{cond}"
            if k not in results:
                continue
            r = results[k]
            print(f"{k:<22s} {r['doc_pair_cos_mean']:>+12.4f} {r['doc_pair_cos_std']:>10.4f} "
                  f"{r['tok_pair_cos_mean']:>+12.4f} {r['doc_effective_rank']:>14.2f} "
                  f"{r['tok_effective_rank']:>14.2f}")
    print("=" * 90)

    # -------------------------------------------------------------------- figure
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Layout: 4 rows × 3 cols (datasets) ;
    #   row1: doc-pair cosine histogram (frozen vs LoRA overlay)
    #   row2: token-pair cosine histogram
    #   row3: singular spectrum (doc-mean) log-scale top-30
    #   row4: singular spectrum (token) log-scale top-30
    fig, axes = plt.subplots(4, 3, figsize=(14, 13))
    for j, ds in enumerate(DATASETS):
        frozen = results.get(f"{ds}_frozen")
        lora = results.get(f"{ds}_lora_2b")
        if frozen is None or lora is None:
            continue

        # — doc cosine histogram
        ax = axes[0, j]
        bins = np.linspace(-0.2, 1.0, 60)
        ax.hist(frozen["doc_pair_cos_dist"], bins=bins, alpha=0.55,
                color="#3b6e8f", label=f"frozen μ={frozen['doc_pair_cos_mean']:+.3f}", density=True)
        ax.hist(lora["doc_pair_cos_dist"], bins=bins, alpha=0.55,
                color="#cc4444", label=f"LoRA 2b μ={lora['doc_pair_cos_mean']:+.3f}", density=True)
        ax.set_title(f"{ds} — random doc-pair cosine")
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)

        # — token cosine histogram
        ax = axes[1, j]
        ax.hist(frozen["tok_pair_cos_dist"], bins=bins, alpha=0.55,
                color="#3b6e8f", label=f"frozen μ={frozen['tok_pair_cos_mean']:+.3f}", density=True)
        ax.hist(lora["tok_pair_cos_dist"], bins=bins, alpha=0.55,
                color="#cc4444", label=f"LoRA 2b μ={lora['tok_pair_cos_mean']:+.3f}", density=True)
        ax.set_title(f"{ds} — random token-pair cosine")
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)

        # — doc-mean singular spectrum
        ax = axes[2, j]
        x = np.arange(1, len(frozen["doc_singular_values_norm_top"]) + 1)
        ax.semilogy(x, frozen["doc_singular_values_norm_top"], "o-",
                    color="#3b6e8f", label=f"frozen eff_rank={frozen['doc_effective_rank']:.1f}")
        x2 = np.arange(1, len(lora["doc_singular_values_norm_top"]) + 1)
        ax.semilogy(x2, lora["doc_singular_values_norm_top"], "s-",
                    color="#cc4444", label=f"LoRA 2b eff_rank={lora['doc_effective_rank']:.1f}")
        ax.set_title(f"{ds} — doc-mean singular spectrum")
        ax.set_xlabel("singular index")
        ax.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

        # — token singular spectrum
        ax = axes[3, j]
        x = np.arange(1, len(frozen["tok_singular_values_norm_top"]) + 1)
        ax.semilogy(x, frozen["tok_singular_values_norm_top"], "o-",
                    color="#3b6e8f", label=f"frozen eff_rank={frozen['tok_effective_rank']:.1f}")
        x2 = np.arange(1, len(lora["tok_singular_values_norm_top"]) + 1)
        ax.semilogy(x2, lora["tok_singular_values_norm_top"], "s-",
                    color="#cc4444", label=f"LoRA 2b eff_rank={lora['tok_effective_rank']:.1f}")
        ax.set_title(f"{ds} — token singular spectrum")
        ax.set_xlabel("singular index")
        ax.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "Diagnostic B — encoder output representation collapse (frozen vs Phase 2b)",
        y=0.995, fontsize=12,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"repr_collapse.{ext}")
    plt.close(fig)
    print(f"\nfigure → {out_dir}/repr_collapse.{{pdf,png}}")
    print(f"data   → {data_path}")


if __name__ == "__main__":
    main()
