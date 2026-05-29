"""Diagnostic B on Exp 14 checkpoints — *data-side family internal representation* verification.

본 script 는 `_repr_collapse_exp13.py` 의 *Exp 14 mirror*:
  - Exp 14 (continuous sigmoid weighting, α_w=10, 3 seeds)
  - data-side family (Exp 12 binary와 cached comparison) 의 internal representation 검정
  - anchor-side family (Exp 13) 와의 paired contrast

Measurements per seed:
  - doc / tok pair-wise cos collapse
  - doc / tok effective rank
  - cos(h_LoRA, h_frozen) per token (Exp 14 의 loss target 아님 — anchor proximity 의 reference)

Output:
  report/figures/_repr_collapse_exp14/{
    repr_collapse_exp14_data.json,
    repr_collapse_exp14.{pdf,png}
  }

CPU 강제. Cache resume.
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


def measure_collapse(d_emb, d_mask):
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


def measure_anchor_proximity(d_emb_lora, d_emb_frozen, d_mask):
    """Per-token cos(h_LoRA, h_frozen) — reference metric for cross-family comparison.

    Exp 14 는 anchor 형식의 loss 없음. 본 measurement 는 *anchor proximity 가 자연스럽게 발생했는지*
    검정 — anchor-side family 의 anchor 가 *특정 loss* 에 의한 것인지, 일반적 LoRA 동학인지 분리.
    """
    h_lora = F.normalize(d_emb_lora.float(), p=2, dim=-1)
    h_frozen = F.normalize(d_emb_frozen.float(), p=2, dim=-1)
    tok_cos = (h_lora * h_frozen).sum(dim=-1)

    valid_idx = d_mask.nonzero(as_tuple=False)
    cos_valid = tok_cos[valid_idx[:, 0], valid_idx[:, 1]].numpy()

    mask_f = d_mask.float()
    n_valid_per_doc = mask_f.sum(dim=1).clamp(min=1.0)
    doc_mean_cos = (tok_cos * mask_f).sum(dim=1) / n_valid_per_doc
    doc_mean_cos = doc_mean_cos.numpy()

    return {
        "n_valid_tokens": int(cos_valid.shape[0]),
        "lora_vs_frozen_tok_cos_mean": float(cos_valid.mean()),
        "lora_vs_frozen_tok_cos_median": float(np.median(cos_valid)),
        "lora_vs_frozen_tok_cos_std": float(cos_valid.std()),
        "lora_vs_frozen_tok_cos_q05": float(np.percentile(cos_valid, 5)),
        "lora_vs_frozen_tok_cos_q95": float(np.percentile(cos_valid, 95)),
        "lora_vs_frozen_doc_cos_mean": float(doc_mean_cos.mean()),
        "lora_vs_frozen_doc_cos_std": float(doc_mean_cos.std()),
        "tok_cos_histogram_bins": np.histogram(cos_valid, bins=50, range=(0.0, 1.0))[1].tolist(),
        "tok_cos_histogram_counts": np.histogram(cos_valid, bins=50, range=(0.0, 1.0))[0].tolist(),
    }


def main():
    cfg = BASELINE
    set_seed(42)
    print(f"device = {DEVICE} (CPU forced)")

    out_dir = PROJECT_ROOT / "report/figures/_repr_collapse_exp14"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "repr_collapse_exp14_data.json"
    results = {}
    if data_path.exists():
        results = json.loads(data_path.read_text())
        print(f"resumed: {sorted(results.keys())}")

    CONFIGS = [
        ("scifact", "14_difficulty_weighted_hn", 42,   "qv_r8_l12_diffw10", "scifact_exp14_s42"),
        ("scifact", "14_difficulty_weighted_hn", 1337, "qv_r8_l12_diffw10", "scifact_exp14_s1337"),
        ("scifact", "14_difficulty_weighted_hn", 2024, "qv_r8_l12_diffw10", "scifact_exp14_s2024"),
    ]

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
            corpus_cache[dataset] = {
                "sample": sample_corpus(corpus, N_DOC_SAMPLE, seed=42),
                "frozen_emb": None,
                "frozen_mask": None,
            }

        sample = corpus_cache[dataset]["sample"]

        if corpus_cache[dataset]["frozen_emb"] is None:
            print(f"\n=== {dataset} FROZEN baseline encoding ===")
            model_frozen = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(DEVICE)
            model_frozen.eval()
            dids_f, d_emb_f, d_mask_f = encode_corpus(model_frozen, sample, DEVICE, batch_size=32)
            corpus_cache[dataset]["frozen_emb"] = d_emb_f
            corpus_cache[dataset]["frozen_mask"] = d_mask_f
            corpus_cache[dataset]["frozen_dids"] = dids_f
            if f"{dataset}_frozen" not in results:
                r_f = measure_collapse(d_emb_f, d_mask_f)
                r_f["lora_vs_frozen_tok_cos_mean"] = 1.0
                r_f["lora_vs_frozen_tok_cos_median"] = 1.0
                r_f["lora_vs_frozen_tok_cos_std"] = 0.0
                r_f["lora_vs_frozen_doc_cos_mean"] = 1.0
                r_f["lora_vs_frozen_doc_cos_std"] = 0.0
                results[f"{dataset}_frozen"] = r_f
                data_path.write_text(json.dumps(results, indent=2))
                print(f"  FROZEN: doc_cos μ={r_f['doc_pair_cos_mean']:+.4f}, "
                      f"eff_rank doc={r_f['doc_effective_rank']:.2f}, tok={r_f['tok_effective_rank']:.2f}")
            del model_frozen

        print(f"\n=== {label} ({dataset}, seed={seed}, tag={tag}) ===")
        model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(DEVICE)
        load_lora(model, ckpt, r=8)
        model.to(DEVICE); model.eval()

        dids_l, d_emb_l, d_mask_l = encode_corpus(model, sample, DEVICE, batch_size=32)
        assert dids_l == corpus_cache[dataset]["frozen_dids"], "did order mismatch"
        assert torch.equal(d_mask_l, corpus_cache[dataset]["frozen_mask"]), "mask mismatch"

        r = measure_collapse(d_emb_l, d_mask_l)
        ap = measure_anchor_proximity(d_emb_l, corpus_cache[dataset]["frozen_emb"], d_mask_l)
        r.update(ap)
        results[label] = r
        data_path.write_text(json.dumps(results, indent=2))
        print(f"  LoRA:   doc_cos μ={r['doc_pair_cos_mean']:+.4f}, "
              f"eff_rank doc={r['doc_effective_rank']:.2f}, tok={r['tok_effective_rank']:.2f}")
        print(f"  anchor proximity (reference): tok cos(LoRA, frozen) μ={r['lora_vs_frozen_tok_cos_mean']:.4f} "
              f"(median={r['lora_vs_frozen_tok_cos_median']:.4f}, std={r['lora_vs_frozen_tok_cos_std']:.4f})")
        print(f"                                doc cos(LoRA, frozen) μ={r['lora_vs_frozen_doc_cos_mean']:.4f}")
        del model

    # summary
    print("\n" + "=" * 110)
    print(f"{'condition':<26s} {'doc_cos μ':>10s} {'tok_cos μ':>10s} "
          f"{'eff_doc':>8s} {'eff_tok':>8s} "
          f"{'anchor_tok':>11s} {'anchor_doc':>11s}")
    print("-" * 110)
    for label in sorted(results.keys()):
        r = results[label]
        print(f"{label:<26s} {r['doc_pair_cos_mean']:>+10.4f} {r['tok_pair_cos_mean']:>+10.4f} "
              f"{r['doc_effective_rank']:>8.2f} {r['tok_effective_rank']:>8.2f} "
              f"{r.get('lora_vs_frozen_tok_cos_mean', float('nan')):>11.4f} "
              f"{r.get('lora_vs_frozen_doc_cos_mean', float('nan')):>11.4f}")
    print("=" * 110)

    # ----------------------- figure: data-side vs anchor-side internal representation
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Load cached comparison data
    prev_path = PROJECT_ROOT / "report/figures/_repr_collapse_new_ckpts/repr_collapse_new_ckpts_data.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    e13_path = PROJECT_ROOT / "report/figures/_repr_collapse_exp13/repr_collapse_exp13_data.json"
    e13_data = json.loads(e13_path.read_text()) if e13_path.exists() else {}
    med_path = PROJECT_ROOT / "report/figures/_repr_collapse_mediation/repr_collapse_mediation_data.json"
    med = json.loads(med_path.read_text()) if med_path.exists() else {}

    seeds = [42, 1337, 2024]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ===== Panel A: cross-family tok eff_rank (6-lever)
    ax = axes[0]
    methods, eff_tok_vals, colors = [], [], []
    # frozen
    if "scifact_frozen" in results:
        methods.append("frozen")
        eff_tok_vals.append(results["scifact_frozen"]["tok_effective_rank"])
        colors.append("#888")
    # Phase 2b
    if "scifact_phase_2b" in med:
        methods.append("Phase 2b")
        eff_tok_vals.append(med["scifact_phase_2b"]["tok_effective_rank"])
        colors.append("#444")
    # Exp 12 (data-w binary)
    e12 = [prev.get(f"scifact_exp12_s{s}", {}).get("tok_effective_rank") for s in seeds]
    e12 = [v for v in e12 if v is not None]
    if e12:
        methods.append("Exp 12\n(binary)")
        eff_tok_vals.append(np.mean(e12))
        colors.append("#ff7f0e")
    # Exp 14 (data-w continuous)
    e14 = [results.get(f"scifact_exp14_s{s}", {}).get("tok_effective_rank") for s in seeds]
    e14 = [v for v in e14 if v is not None]
    if e14:
        methods.append("Exp 14\n(continuous)")
        eff_tok_vals.append(np.mean(e14))
        colors.append("#d62728")
    # Exp 11 (anchor relational)
    e11 = [prev.get(f"scifact_exp11_s{s}", {}).get("tok_effective_rank") for s in seeds]
    e11 = [v for v in e11 if v is not None]
    if e11:
        methods.append("Exp 11\n(relational)")
        eff_tok_vals.append(np.mean(e11))
        colors.append("#1f77b4")
    # Exp 13 (anchor absolute)
    e13 = [e13_data.get(f"scifact_exp13_s{s}", {}).get("tok_effective_rank") for s in seeds]
    e13 = [v for v in e13 if v is not None]
    if e13:
        methods.append("Exp 13\n(absolute)")
        eff_tok_vals.append(np.mean(e13))
        colors.append("#2ca02c")

    bars = ax.bar(methods, eff_tok_vals, color=colors, alpha=0.8)
    for b, v in zip(bars, eff_tok_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.7, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("token effective rank")
    ax.set_title("Token eff_rank — 6-lever cross-family\n(higher = less collapse)")
    ax.grid(axis="y", alpha=0.3)

    # ===== Panel B: anchor proximity (cos(LoRA, frozen)) — anchor-side vs data-side
    ax = axes[1]
    methods, anchor_vals, colors = [], [], []
    # Exp 12 doesn't have anchor measurement (cached without it)
    # Exp 14 (3 seeds + mean)
    for s in seeds:
        v = results.get(f"scifact_exp14_s{s}", {}).get("lora_vs_frozen_tok_cos_mean")
        if v is not None:
            methods.append(f"Exp 14 s{s}")
            anchor_vals.append(v)
            colors.append("#d62728")
    if e14:
        e14_anchors = [results.get(f"scifact_exp14_s{s}", {}).get("lora_vs_frozen_tok_cos_mean") for s in seeds]
        e14_anchors = [v for v in e14_anchors if v is not None]
        if e14_anchors:
            methods.append("Exp 14\nmean")
            anchor_vals.append(np.mean(e14_anchors))
            colors.append("#a30000")  # darker for mean
    # Exp 13 (3 seeds + mean)
    for s in seeds:
        v = e13_data.get(f"scifact_exp13_s{s}", {}).get("lora_vs_frozen_tok_cos_mean")
        if v is not None:
            methods.append(f"Exp 13 s{s}")
            anchor_vals.append(v)
            colors.append("#2ca02c")
    if e13:
        e13_anchors = [e13_data.get(f"scifact_exp13_s{s}", {}).get("lora_vs_frozen_tok_cos_mean") for s in seeds]
        e13_anchors = [v for v in e13_anchors if v is not None]
        if e13_anchors:
            methods.append("Exp 13\nmean")
            anchor_vals.append(np.mean(e13_anchors))
            colors.append("#0a5d0a")
    bars = ax.bar(methods, anchor_vals, color=colors, alpha=0.85)
    for b, v in zip(bars, anchor_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="frozen identity")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("cos(h_LoRA, h_frozen) per token (mean)")
    ax.set_title("Anchor proximity — data-side (Exp 14) vs anchor-side (Exp 13)\n(reference for family separation)")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # ===== Panel C: per-token cos distribution — Exp 14 vs Exp 13 (seed 42)
    ax = axes[2]
    if "scifact_exp14_s42" in results:
        rec = results["scifact_exp14_s42"]
        if "tok_cos_histogram_counts" in rec:
            bins = rec["tok_cos_histogram_bins"]
            counts = rec["tok_cos_histogram_counts"]
            centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(counts))]
            total = sum(counts)
            density = [c / total for c in counts]
            ax.plot(centers, density, color="#d62728", linewidth=2,
                    label=f"Exp 14 s42 (μ={rec['lora_vs_frozen_tok_cos_mean']:.3f})")
    if "scifact_exp13_s42" in e13_data:
        rec = e13_data["scifact_exp13_s42"]
        if "tok_cos_histogram_counts" in rec:
            bins = rec["tok_cos_histogram_bins"]
            counts = rec["tok_cos_histogram_counts"]
            centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(counts))]
            total = sum(counts)
            density = [c / total for c in counts]
            ax.plot(centers, density, color="#2ca02c", linewidth=2,
                    label=f"Exp 13 s42 (μ={rec['lora_vs_frozen_tok_cos_mean']:.3f})")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("per-token cos(h_LoRA, h_frozen)")
    ax.set_ylabel("frequency (normalized)")
    ax.set_title("Per-token anchor proximity distribution\n(seed 42, family contrast)")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.0, 1.02)

    fig.suptitle("Diagnostic B on Exp 14 — data-side family internal representation (SciFact, 3 seeds)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"repr_collapse_exp14.{ext}")
    plt.close(fig)

    print(f"\nfigure → {out_dir}/repr_collapse_exp14.{{pdf,png}}")
    print(f"data   → {data_path}")


if __name__ == "__main__":
    main()
