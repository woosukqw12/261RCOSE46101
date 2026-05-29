"""Diagnostic B on Exp 13 checkpoints — *mechanism direct verification* of per-token cosine anchor.

본 script 는 `_repr_collapse_new_ckpts.py` 의 *Exp 13 extension*:
  - Exp 13 (frozen-direction anchor, λ_dir=1.0, 3 seeds): per-token absolute direction anchor
    이 실제로 representation 을 frozen baseline 에 묶었는가?

추가 metric (기존 doc/tok eff_rank + pair-wise cos 위에):
  - `lora_vs_frozen_cos_mean / median / std`: 동일 corpus 의 LoRA-encoded vs frozen-encoded 의
    *per-token cos* 평균. Exp 13 의 anchor_loss = 1 - cos 의 *학습 후 잔여값* 직접 측정.

Output:
  report/figures/_repr_collapse_exp13/{
    repr_collapse_exp13_data.json,
    repr_collapse_exp13.{pdf,png}
  }

CPU 강제 (이전 diagnostic 와 동일). Cache resume 지원.
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
    """Standard collapse measurement (eff_rank, pair-wise cos)."""
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
    """Exp 13-specific: per-token cos(h_LoRA, h_frozen) — direct anchor loss measurement.

    Exp 13 의 anchor_loss = 1 - cos. 학습 후 cos 가 1 에 가까울수록 anchor 보존.
    L2-norm 후 token 별 dot product 계산, valid token (mask=1) 만 집계.
    """
    # token embeddings should already be L2-normed from ColBERT's project()
    # but renormalize defensively (idempotent)
    h_lora = F.normalize(d_emb_lora.float(), p=2, dim=-1)
    h_frozen = F.normalize(d_emb_frozen.float(), p=2, dim=-1)

    # per-token cos
    tok_cos = (h_lora * h_frozen).sum(dim=-1)  # [N, T]

    valid_idx = d_mask.nonzero(as_tuple=False)
    cos_valid = tok_cos[valid_idx[:, 0], valid_idx[:, 1]].numpy()  # [n_valid_tokens]

    # also per-doc mean (over valid tokens)
    mask_f = d_mask.float()
    n_valid_per_doc = mask_f.sum(dim=1).clamp(min=1.0)
    doc_mean_cos = (tok_cos * mask_f).sum(dim=1) / n_valid_per_doc  # [N]
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
        # for histogram (subsample for json size)
        "tok_cos_histogram_bins": np.histogram(cos_valid, bins=50, range=(0.5, 1.0))[1].tolist(),
        "tok_cos_histogram_counts": np.histogram(cos_valid, bins=50, range=(0.5, 1.0))[0].tolist(),
    }


def main():
    cfg = BASELINE
    set_seed(42)
    print(f"device = {DEVICE} (CPU forced)")

    out_dir = PROJECT_ROOT / "report/figures/_repr_collapse_exp13"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "repr_collapse_exp13_data.json"
    results = {}
    if data_path.exists():
        results = json.loads(data_path.read_text())
        print(f"resumed: {sorted(results.keys())}")

    CONFIGS = [
        ("scifact", "13_frozen_direction_anchor", 42,   "qv_r8_l12_dir1", "scifact_exp13_s42"),
        ("scifact", "13_frozen_direction_anchor", 1337, "qv_r8_l12_dir1", "scifact_exp13_s1337"),
        ("scifact", "13_frozen_direction_anchor", 2024, "qv_r8_l12_dir1", "scifact_exp13_s2024"),
    ]

    corpus_cache = {}

    # === First pass: encode all conditions with frozen model (once per dataset) ===
    # then re-load model + inject LoRA for each seed
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

        # encode with frozen model (cache per dataset)
        if corpus_cache[dataset]["frozen_emb"] is None:
            print(f"\n=== {dataset} FROZEN baseline encoding ===")
            model_frozen = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(DEVICE)
            model_frozen.eval()
            dids_f, d_emb_f, d_mask_f = encode_corpus(model_frozen, sample, DEVICE, batch_size=32)
            corpus_cache[dataset]["frozen_emb"] = d_emb_f
            corpus_cache[dataset]["frozen_mask"] = d_mask_f
            corpus_cache[dataset]["frozen_dids"] = dids_f
            # measure frozen collapse (for baseline reference)
            if f"{dataset}_frozen" not in results:
                r_f = measure_collapse(d_emb_f, d_mask_f)
                # frozen vs frozen anchor = identity (cos = 1)
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

        # encode with LoRA (per seed)
        print(f"\n=== {label} ({dataset}, seed={seed}, tag={tag}) ===")
        model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(DEVICE)
        load_lora(model, ckpt, r=8)
        model.to(DEVICE); model.eval()

        dids_l, d_emb_l, d_mask_l = encode_corpus(model, sample, DEVICE, batch_size=32)

        # sanity: dids match between frozen and LoRA encodings
        assert dids_l == corpus_cache[dataset]["frozen_dids"], "did order mismatch"
        assert torch.equal(d_mask_l, corpus_cache[dataset]["frozen_mask"]), "mask mismatch"

        # standard collapse metrics on LoRA encoding
        r = measure_collapse(d_emb_l, d_mask_l)
        # anchor proximity: cos(LoRA, frozen)
        ap = measure_anchor_proximity(d_emb_l, corpus_cache[dataset]["frozen_emb"], d_mask_l)
        r.update(ap)

        results[label] = r
        data_path.write_text(json.dumps(results, indent=2))
        print(f"  LoRA:   doc_cos μ={r['doc_pair_cos_mean']:+.4f}, "
              f"eff_rank doc={r['doc_effective_rank']:.2f}, tok={r['tok_effective_rank']:.2f}")
        print(f"  anchor proximity: tok cos(LoRA, frozen) μ={r['lora_vs_frozen_tok_cos_mean']:.4f} "
              f"(median={r['lora_vs_frozen_tok_cos_median']:.4f}, std={r['lora_vs_frozen_tok_cos_std']:.4f})")
        print(f"                    doc cos(LoRA, frozen) μ={r['lora_vs_frozen_doc_cos_mean']:.4f}")
        del model

    # ----------------------- summary
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

    # ----------------------- figure
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Load Exp 11 / Phase 2b comparison data
    prev_path = PROJECT_ROOT / "report/figures/_repr_collapse_new_ckpts/repr_collapse_new_ckpts_data.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    med_path = PROJECT_ROOT / "report/figures/_repr_collapse_mediation/repr_collapse_mediation_data.json"
    med = json.loads(med_path.read_text()) if med_path.exists() else {}

    # ===== Panel A: anchor proximity (cos(LoRA, frozen)) — Exp 13 의 핵심 metric =====
    # ===== Panel B: token eff_rank (Exp 11 6× recovery 와 paired) =====
    # ===== Panel C: doc pair_cos collapse (Phase 2b/M1b 비교) =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: anchor proximity bar (3 seeds + mean)
    ax = axes[0]
    seeds = [42, 1337, 2024]
    tok_anchors = [results.get(f"scifact_exp13_s{s}", {}).get("lora_vs_frozen_tok_cos_mean", 0) for s in seeds]
    doc_anchors = [results.get(f"scifact_exp13_s{s}", {}).get("lora_vs_frozen_doc_cos_mean", 0) for s in seeds]
    x = np.arange(len(seeds))
    width = 0.35
    bars_t = ax.bar(x - width/2, tok_anchors, width, color="#2ca02c", alpha=0.8, label="per-token cos")
    bars_d = ax.bar(x + width/2, doc_anchors, width, color="#1f77b4", alpha=0.8, label="per-doc mean cos")
    for b, v in zip(bars_t, tok_anchors):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for b, v in zip(bars_d, doc_anchors):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="frozen identity (cos=1)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("cos(h_LoRA, h_frozen)")
    ax.set_title("Anchor proximity — Exp 13 mechanism verification\n(loss = 1 − cos, after training)")
    ax.set_ylim(0.8, 1.02)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: token eff_rank comparison (frozen / Phase 2b / Exp 11 / Exp 13)
    ax = axes[1]
    methods = []
    eff_tok = []
    colors = []
    # frozen
    if "scifact_frozen" in results:
        methods.append("frozen\nbaseline")
        eff_tok.append(results["scifact_frozen"]["tok_effective_rank"])
        colors.append("#888")
    # Phase 2b (from mediation)
    if "scifact_phase_2b" in med:
        methods.append("Phase 2b")
        eff_tok.append(med["scifact_phase_2b"]["tok_effective_rank"])
        colors.append("#cc4444")
    # Exp 11 (3 seeds, mean)
    e11 = [prev.get(f"scifact_exp11_s{s}", {}).get("tok_effective_rank") for s in seeds]
    e11 = [v for v in e11 if v]
    if e11:
        methods.append("Exp 11\n(rel., 3-seed mean)")
        eff_tok.append(np.mean(e11))
        colors.append("#1f77b4")
    # Exp 13 (3 seeds, mean)
    e13 = [results.get(f"scifact_exp13_s{s}", {}).get("tok_effective_rank") for s in seeds]
    e13 = [v for v in e13 if v]
    if e13:
        methods.append("Exp 13\n(abs., 3-seed mean)")
        eff_tok.append(np.mean(e13))
        colors.append("#2ca02c")

    bars = ax.bar(methods, eff_tok, color=colors, alpha=0.8)
    for b, v in zip(bars, eff_tok):
        ax.text(b.get_x() + b.get_width()/2, v + 0.2, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("token effective rank")
    ax.set_title("Token eff_rank — anchor-side family comparison\n(higher = less collapse, closer to frozen)")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: anchor proximity distribution (Exp 13 seeds) histogram
    ax = axes[2]
    bin_edges = None
    for s, color in zip(seeds, ["#2ca02c", "#5fbf5f", "#a0d8a0"]):
        rec = results.get(f"scifact_exp13_s{s}", {})
        if "tok_cos_histogram_counts" in rec:
            bins = rec["tok_cos_histogram_bins"]
            counts = rec["tok_cos_histogram_counts"]
            centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(counts))]
            total = sum(counts)
            density = [c / total for c in counts]
            ax.plot(centers, density, color=color, linewidth=1.5, label=f"seed {s}")
            if bin_edges is None:
                bin_edges = bins
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("per-token cos(h_LoRA, h_frozen)")
    ax.set_ylabel("frequency (normalized)")
    ax.set_title("Per-token anchor proximity distribution\n(Exp 13 trained checkpoints)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.6, 1.02)

    fig.suptitle("Diagnostic B on Exp 13 — per-token absolute direction anchor mechanism (SciFact, 3 seeds)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"repr_collapse_exp13.{ext}")
    plt.close(fig)

    print(f"\nfigure → {out_dir}/repr_collapse_exp13.{{pdf,png}}")
    print(f"data   → {data_path}")


if __name__ == "__main__":
    main()
