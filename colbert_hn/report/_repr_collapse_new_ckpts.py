"""Diagnostic B on new checkpoints (Exp 11, M1+M1b combined, Exp 12).

기존 `_repr_collapse_mediation.py` 의 *extension* — *mechanism 직접 검증*:
  - Exp 11 (relational easy preservation, λ=1.0, 3 seeds): easy-doc eff_rank 보존됨? (loss 가 self-sim 직접 규제)
  - M1+M1b combined (SciFact + NFCorpus): M1b alone 과 동일 collapse? (M1 추가 기여 zero 확인)
  - Exp 12 (FN-denoised, 3 seeds): Phase 2b 와 동일 collapse? ((나-2) difficulty dominant 추가 증거)

CPU 강제 (이전 diagnostic 와 동일). Cache resume 지원.

Output:
  report/figures/_repr_collapse_new_ckpts/{
    repr_collapse_new_ckpts_data.json,
    repr_collapse_new_ckpts.{pdf,png}
  }
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

DEVICE = torch.device("cpu")
N_DOC_SAMPLE = 300
N_DOC_PAIRS = 3000
N_TOKEN_PAIRS = 6000
TOP_SV = 30


def effective_rank(sv):
    s2 = sv.astype(np.float64) ** 2
    if s2.sum() <= 0:
        return 0.0
    p = s2 / s2.sum()
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum()
    return float(np.exp(ent))


def load_lora(model, ckpt, r=8):
    lora_params = inject_lora_into_bert(
        model.bert, target_components=["q", "v"], layers=None,
        r=r, alpha=None, init_std=0.02,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    lora_state = state["lora"]
    keys = sorted(lora_state.keys(), key=lambda k: int(k.split("_")[-1]))
    assert len(keys) == len(lora_params)
    with torch.no_grad():
        for k, p in zip(keys, lora_params):
            p.data.copy_(lora_state[k].to(p.device))
    return lora_params


def sample_corpus(corpus, n, seed=42):
    rng = np.random.default_rng(seed)
    dids = list(corpus.keys())
    idx = rng.choice(len(dids), size=min(n, len(dids)), replace=False)
    return {dids[i]: corpus[dids[i]] for i in idx}


def measure(model, corpus_sample):
    dids, d_emb, d_mask = encode_corpus(model, corpus_sample, DEVICE, batch_size=32)
    N = d_emb.shape[0]

    mask_f = d_mask.float().unsqueeze(-1)
    n_valid = mask_f.sum(dim=1).clamp(min=1.0)
    d_mean = (d_emb * mask_f).sum(dim=1) / n_valid
    d_mean = F.normalize(d_mean, p=2, dim=-1)

    rng = np.random.default_rng(0)
    ia = rng.integers(0, N, size=N_DOC_PAIRS)
    ib = rng.integers(0, N, size=N_DOC_PAIRS)
    m = ia != ib
    ia = ia[m]; ib = ib[m]
    doc_cos = (d_mean[ia] * d_mean[ib]).sum(dim=-1).numpy()

    valid_idx = d_mask.nonzero(as_tuple=False)
    tokens = d_emb[valid_idx[:, 0], valid_idx[:, 1]]
    n_tok = tokens.shape[0]
    ta = rng.integers(0, n_tok, size=N_TOKEN_PAIRS)
    tb = rng.integers(0, n_tok, size=N_TOKEN_PAIRS)
    m = ta != tb
    ta = ta[m]; tb = tb[m]
    tok_cos = (tokens[ta] * tokens[tb]).sum(dim=-1).numpy()

    sv_doc = torch.linalg.svdvals(d_mean.float()).numpy()
    if n_tok > 8000:
        sub = rng.choice(n_tok, size=8000, replace=False)
        tokens_sub = tokens[sub]
    else:
        tokens_sub = tokens
    sv_tok = torch.linalg.svdvals(tokens_sub.float()).numpy()

    return {
        "n_docs": int(N),
        "n_tokens": int(n_tok),
        "doc_pair_cos_mean": float(doc_cos.mean()),
        "doc_pair_cos_std": float(doc_cos.std()),
        "tok_pair_cos_mean": float(tok_cos.mean()),
        "tok_pair_cos_std": float(tok_cos.std()),
        "doc_effective_rank": effective_rank(sv_doc),
        "tok_effective_rank": effective_rank(sv_tok),
        "doc_singular_values_top": sv_doc[:TOP_SV].tolist(),
        "tok_singular_values_top": sv_tok[:TOP_SV].tolist(),
    }


def main():
    cfg = BASELINE
    set_seed(42)
    print(f"device = {DEVICE} (CPU forced)")

    out_dir = PROJECT_ROOT / "report/figures/_repr_collapse_new_ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "repr_collapse_new_ckpts_data.json"
    results = {}
    if data_path.exists():
        results = json.loads(data_path.read_text())
        print(f"resumed: {sorted(results.keys())}")

    # (dataset, exp_dir, seed, tag, label)
    CONFIGS = [
        # Exp 11 — easy preservation (3 seeds)
        ("scifact", "11_easy_preservation", 42,   "qv_r8_l12_le1",      "scifact_exp11_s42"),
        ("scifact", "11_easy_preservation", 1337, "qv_r8_l12_le1",      "scifact_exp11_s1337"),
        ("scifact", "11_easy_preservation", 2024, "qv_r8_l12_le1",      "scifact_exp11_s2024"),
        # M1+M1b combined (SciFact + NFCorpus, seed 42)
        ("scifact", "10_lora_phi",          42,   "qv_r8_l12_m1plus1b", "scifact_m1plus1b"),
        ("nfcorpus", "10_lora_phi",         42,   "qv_r8_l12_m1plus1b", "nfcorpus_m1plus1b"),
        # Exp 12 — FN-denoised mined-HN (3 seeds)
        ("scifact", "12_fn_denoised_hn",    42,   "qv_r8_l12_thresh0",  "scifact_exp12_s42"),
        ("scifact", "12_fn_denoised_hn",    1337, "qv_r8_l12_thresh0",  "scifact_exp12_s1337"),
        ("scifact", "12_fn_denoised_hn",    2024, "qv_r8_l12_thresh0",  "scifact_exp12_s2024"),
        # M1b additional seeds for fuller comparison
        ("scifact", "10_lora_phi",          1337, "qv_r8_l12_m1b",      "scifact_m1b_s1337"),
        ("scifact", "10_lora_phi",          2024, "qv_r8_l12_m1b",      "scifact_m1b_s2024"),
    ]

    # Cache corpora to avoid repeated load_beir
    corpus_cache = {}

    for dataset, exp_dir, seed, tag, label in CONFIGS:
        if label in results:
            print(f"skip {label} (cached)")
            continue

        ckpt = (PROJECT_ROOT / "outputs" / exp_dir / dataset / f"seed_{seed}" / tag
                / "module_final.pt")
        if not ckpt.exists():
            print(f"WARN: missing {ckpt}")
            continue

        if dataset not in corpus_cache:
            print(f"loading {dataset} test corpus...")
            corpus, _, _ = load_beir(dataset, split="test")
            corpus_cache[dataset] = sample_corpus(corpus, N_DOC_SAMPLE, seed=42)

        sample = corpus_cache[dataset]
        print(f"\n=== {label} ({dataset}, seed={seed}, tag={tag}) ===")

        model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(DEVICE)
        load_lora(model, ckpt, r=8)
        model.to(DEVICE); model.eval()

        r = measure(model, sample)
        results[label] = r
        data_path.write_text(json.dumps(results, indent=2))
        print(f"  doc_cos μ={r['doc_pair_cos_mean']:+.4f}, eff_rank doc={r['doc_effective_rank']:.2f}, tok={r['tok_effective_rank']:.2f}")
        del model

    # ----------------------- summary
    print("\n" + "=" * 100)
    print(f"{'condition':<28s} {'doc_cos μ':>11s} {'tok_cos μ':>11s} {'eff_doc':>10s} {'eff_tok':>10s}")
    print("-" * 100)
    for label in sorted(results.keys()):
        r = results[label]
        print(f"{label:<28s} {r['doc_pair_cos_mean']:>+11.4f} {r['tok_pair_cos_mean']:>+11.4f} "
              f"{r['doc_effective_rank']:>10.2f} {r['tok_effective_rank']:>10.2f}")
    print("=" * 100)

    # ----------------------- figure: compare SciFact methods
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Load previous mediation data (Phase 2b, M1, M1b for scifact)
    prev_path = PROJECT_ROOT / "report/figures/_repr_collapse_mediation/repr_collapse_mediation_data.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}

    # SciFact mechanism comparison: doc_cos + eff_rank
    sci_methods = [
        ("frozen", 0.573, 10.65, "#888"),       # from prior diagnostic
        ("Phase 2b", prev.get("scifact_phase_2b", {}).get("doc_pair_cos_mean", 0),
                     prev.get("scifact_phase_2b", {}).get("doc_effective_rank", 0), "#cc4444"),
        ("M1", prev.get("scifact_m1", {}).get("doc_pair_cos_mean", 0),
               prev.get("scifact_m1", {}).get("doc_effective_rank", 0), "#e0a040"),
        ("M1b (s42)", prev.get("scifact_m1b", {}).get("doc_pair_cos_mean", 0),
                       prev.get("scifact_m1b", {}).get("doc_effective_rank", 0), "#3b7e3b"),
        ("M1b (s1337)", results.get("scifact_m1b_s1337", {}).get("doc_pair_cos_mean", 0),
                         results.get("scifact_m1b_s1337", {}).get("doc_effective_rank", 0), "#5ba65b"),
        ("M1b (s2024)", results.get("scifact_m1b_s2024", {}).get("doc_pair_cos_mean", 0),
                         results.get("scifact_m1b_s2024", {}).get("doc_effective_rank", 0), "#7bc67b"),
        ("M1+M1b", results.get("scifact_m1plus1b", {}).get("doc_pair_cos_mean", 0),
                    results.get("scifact_m1plus1b", {}).get("doc_effective_rank", 0), "#2a8a3e"),
        ("Exp 11 (s42)", results.get("scifact_exp11_s42", {}).get("doc_pair_cos_mean", 0),
                          results.get("scifact_exp11_s42", {}).get("doc_effective_rank", 0), "#4a7eaf"),
        ("Exp 11 (s1337)", results.get("scifact_exp11_s1337", {}).get("doc_pair_cos_mean", 0),
                            results.get("scifact_exp11_s1337", {}).get("doc_effective_rank", 0), "#6a9ecf"),
        ("Exp 11 (s2024)", results.get("scifact_exp11_s2024", {}).get("doc_pair_cos_mean", 0),
                            results.get("scifact_exp11_s2024", {}).get("doc_effective_rank", 0), "#8abeef"),
        ("Exp 12 (s42)", results.get("scifact_exp12_s42", {}).get("doc_pair_cos_mean", 0),
                          results.get("scifact_exp12_s42", {}).get("doc_effective_rank", 0), "#a040a0"),
        ("Exp 12 (s1337)", results.get("scifact_exp12_s1337", {}).get("doc_pair_cos_mean", 0),
                            results.get("scifact_exp12_s1337", {}).get("doc_effective_rank", 0), "#c060c0"),
        ("Exp 12 (s2024)", results.get("scifact_exp12_s2024", {}).get("doc_pair_cos_mean", 0),
                            results.get("scifact_exp12_s2024", {}).get("doc_effective_rank", 0), "#e080e0"),
    ]
    # Filter out missing
    sci_methods = [m for m in sci_methods if m[1] != 0 or "frozen" in m[0]]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # (1) doc_cos
    ax = axes[0]
    x = np.arange(len(sci_methods))
    cols = [m[3] for m in sci_methods]
    vals = [m[1] for m in sci_methods]
    bars = ax.bar(x, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in sci_methods], rotation=25, ha="right", fontsize=8)
    ax.axhline(0.573, color="#888", linestyle="--", linewidth=0.5, label="frozen baseline")
    ax.set_ylabel("random doc-pair cosine μ")
    ax.set_ylim(0, 1.05)
    ax.set_title("SciFact — collapse magnitude (lower = less collapse)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (2) eff_rank doc
    ax = axes[1]
    vals = [m[2] for m in sci_methods]
    ax.bar(x, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in sci_methods], rotation=25, ha="right", fontsize=8)
    ax.axhline(10.65, color="#888", linestyle="--", linewidth=0.5, label="frozen baseline")
    ax.set_ylabel("effective rank (doc, higher = less collapse)")
    ax.set_title("SciFact — eff_rank (higher = less collapse)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Diagnostic B on new checkpoints — *mechanism direct verification*",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"repr_collapse_new_ckpts.{ext}")
    plt.close(fig)
    print(f"\nfigure → {out_dir}/repr_collapse_new_ckpts.{{pdf,png}}")
    print(f"data   → {data_path}")


if __name__ == "__main__":
    main()
