"""Diagnostic B on Exp 16 checkpoints — *multi-layer anchor mechanism verification*.

본 script 는 `_repr_collapse_exp13.py` 의 *Exp 16 multi-layer extension*:
  - Exp 16 (multi-layer per-token cosine anchor, λ_dir=1.0, layers={0,3,6,9,12}, 3 seeds)
  - 각 layer 별 cos(h_LoRA, h_frozen) 측정 (Exp 13 의 final-only 와 paired comparison)
  - doc / tok eff_rank — Exp 13 의 9.01 (token) / 2.33 (doc) 과 비교

Output:
  report/figures/_repr_collapse_exp16/{
    repr_collapse_exp16_data.json,
    repr_collapse_exp16.{pdf,png}
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
ANCHOR_LAYERS = (0, 3, 6, 9, 12)


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


class LayerCapture:
    """Same as Exp 16's LayerCapture — captures hidden states at specified layers."""

    def __init__(self, model: ColBERTv2, layers):
        self.model = model
        self.layers = layers
        self.buffer = {}

    def install(self):
        for ell in self.layers:
            self.model.register_layer_hook(ell, self._make_hook(ell))

    def _make_hook(self, ell):
        def hook(h):
            self.buffer[ell] = h
            return h
        return hook

    def consume(self):
        states = dict(self.buffer)
        self.buffer.clear()
        return states


def encode_per_doc(model, capture, sample, dids_ordered):
    """Process each doc one at a time, accumulate per-layer hidden states + valid tokens.

    Returns dict[layer_idx] -> list of (T_valid, 768) tensors, one per doc.
    """
    from src.data import doc_text
    per_doc_layers: Dict[int, list] = {ell: [] for ell in ANCHOR_LAYERS}
    model.eval()
    with torch.no_grad():
        for did in dids_ordered:
            text = doc_text(sample[did])
            _, mask = model.encode_docs([text], device=DEVICE)
            states = capture.consume()
            T_valid = int(mask[0].sum().item())
            for ell in ANCHOR_LAYERS:
                h = states[ell][0, :T_valid]  # (T_valid, 768) only valid tokens
                per_doc_layers[ell].append(h)
    return per_doc_layers


def compute_layer_metrics_per_doc(lora_per_doc, frozen_per_doc):
    """For each layer, given list of (T, 768) tensors per doc, compute:
       - per-token cos(h_LoRA, h_frozen) aggregated across all docs' valid tokens
       - per-doc mean cos
       - tok eff_rank (over all tokens)
       - doc eff_rank (over per-doc mean-pooled L2-normed)
    """
    out = {}
    for ell in ANCHOR_LAYERS:
        all_cos = []
        doc_mean_cos = []
        all_tokens = []
        doc_mean_emb = []
        for h_l, h_f in zip(lora_per_doc[ell], frozen_per_doc[ell]):
            h_l_n = F.normalize(h_l.float(), p=2, dim=-1)
            h_f_n = F.normalize(h_f.float(), p=2, dim=-1)
            cos = (h_l_n * h_f_n).sum(dim=-1)  # (T,)
            all_cos.append(cos.numpy())
            doc_mean_cos.append(float(cos.mean()))
            all_tokens.append(h_l_n)
            doc_mean_emb.append(F.normalize(h_l_n.mean(dim=0), p=2, dim=-1))

        cos_valid = np.concatenate(all_cos)
        tokens_cat = torch.cat(all_tokens, dim=0)
        n_tok = tokens_cat.shape[0]
        if n_tok > 8000:
            rng = np.random.default_rng(42)
            sub = rng.choice(n_tok, size=8000, replace=False)
            tokens_cat = tokens_cat[sub]
        d_mean = torch.stack(doc_mean_emb, dim=0)

        sv_doc = torch.linalg.svdvals(d_mean).numpy()
        sv_tok = torch.linalg.svdvals(tokens_cat).numpy()

        out[ell] = {
            "tok_cos_mean": float(cos_valid.mean()),
            "tok_cos_median": float(np.median(cos_valid)),
            "tok_cos_std": float(cos_valid.std()),
            "doc_cos_mean": float(np.mean(doc_mean_cos)),
            "doc_eff_rank": effective_rank(sv_doc),
            "tok_eff_rank": effective_rank(sv_tok),
        }
    return out


def main():
    cfg = BASELINE
    set_seed(42)
    print(f"device = {DEVICE} (CPU forced)")

    out_dir = PROJECT_ROOT / "report/figures/_repr_collapse_exp16"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "repr_collapse_exp16_data.json"
    results = {}
    if data_path.exists():
        results = json.loads(data_path.read_text())
        print(f"resumed: {sorted(results.keys())}")

    CONFIGS = [
        ("scifact", 42,   "scifact_exp16_s42"),
        ("scifact", 1337, "scifact_exp16_s1337"),
        ("scifact", 2024, "scifact_exp16_s2024"),
    ]

    # Sample corpus once
    print("loading scifact test corpus...")
    corpus, _, _ = load_beir("scifact", split="test")
    sample = sample_corpus(corpus, N_DOC_SAMPLE, seed=42)

    sample_dids = sorted(sample.keys())

    # Step 1: encode frozen baseline (per-doc, capture all layer states)
    print("\n=== FROZEN baseline encoding (per-doc multi-layer capture) ===")
    model_frozen = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(DEVICE)
    cap_f = LayerCapture(model_frozen, ANCHOR_LAYERS)
    cap_f.install()
    frozen_per_doc = encode_per_doc(model_frozen, cap_f, sample, sample_dids)
    del model_frozen, cap_f

    # frozen self-self: cos = 1 (sanity), measure eff_rank from frozen states alone
    if "scifact_frozen_multilayer" not in results:
        out_f = compute_layer_metrics_per_doc(frozen_per_doc, frozen_per_doc)
        # Replace cos with 1.0 (identity by construction)
        for ell in ANCHOR_LAYERS:
            out_f[ell]["tok_cos_mean"] = 1.0
            out_f[ell]["tok_cos_median"] = 1.0
            out_f[ell]["tok_cos_std"] = 0.0
            out_f[ell]["doc_cos_mean"] = 1.0
        results["scifact_frozen_multilayer"] = {str(k): v for k, v in out_f.items()}
        data_path.write_text(json.dumps(results, indent=2))
        for ell in ANCHOR_LAYERS:
            print(f"  layer {ell}: doc_eff={out_f[ell]['doc_eff_rank']:.2f}, "
                  f"tok_eff={out_f[ell]['tok_eff_rank']:.2f}")

    # Step 2: each LoRA seed
    for dataset, seed, label in CONFIGS:
        if label in results:
            print(f"skip {label} (cached)")
            continue
        ckpt = (PROJECT_ROOT / "outputs/16_multilayer_anchor" / dataset / f"seed_{seed}"
                / "qv_r8_l12_dir1_multilayer" / "module_final.pt")
        if not ckpt.exists():
            print(f"WARN: missing {ckpt}")
            continue

        print(f"\n=== {label} ===")
        model = ColBERTv2(ColBERTConfig(model_name=cfg.encoder_name)).to(DEVICE)
        cap = LayerCapture(model, ANCHOR_LAYERS)
        cap.install()
        load_lora(model, ckpt, r=8)
        model.to(DEVICE)

        lora_per_doc = encode_per_doc(model, cap, sample, sample_dids)
        metrics = compute_layer_metrics_per_doc(lora_per_doc, frozen_per_doc)
        results[label] = {str(k): v for k, v in metrics.items()}
        data_path.write_text(json.dumps(results, indent=2))
        for ell in ANCHOR_LAYERS:
            m = metrics[ell]
            print(f"  layer {ell}: cos μ={m['tok_cos_mean']:.4f}, "
                  f"doc_eff={m['doc_eff_rank']:.2f}, tok_eff={m['tok_eff_rank']:.2f}")
        del model, cap, lora_per_doc

    # ----------------------- summary
    print("\n" + "=" * 100)
    print(f"{'seed/condition':<24s} | ", end="")
    print(" | ".join([f"L{ell} cos / d_eff / t_eff" for ell in ANCHOR_LAYERS]))
    print("-" * 130)
    for label in sorted(results.keys()):
        if "frozen" in label:
            continue
        r = results[label]
        row = [label[:23]]
        for ell in ANCHOR_LAYERS:
            m = r[str(ell)]
            row.append(f"{m['tok_cos_mean']:.3f}/{m['doc_eff_rank']:.2f}/{m['tok_eff_rank']:.2f}")
        print(" | ".join(f"{x:<23s}" if i == 0 else f"{x:<20s}" for i, x in enumerate(row)))

    # ----------------------- figure
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Load Exp 13 cached for comparison
    exp13_path = PROJECT_ROOT / "report/figures/_repr_collapse_exp13/repr_collapse_exp13_data.json"
    exp13_data = json.loads(exp13_path.read_text()) if exp13_path.exists() else {}

    seeds = [42, 1337, 2024]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: per-layer cos(LoRA, frozen) — Exp 16 (3 seeds) vs Exp 13 single point
    ax = axes[0]
    layer_xs = list(ANCHOR_LAYERS)
    for s, color in zip(seeds, ["#2ca02c", "#5fbf5f", "#a0d8a0"]):
        lab = f"scifact_exp16_s{s}"
        if lab not in results:
            continue
        cos_per_layer = [results[lab][str(ell)]["tok_cos_mean"] for ell in ANCHOR_LAYERS]
        ax.plot(layer_xs, cos_per_layer, "o-", color=color, label=f"Exp 16 s{s}",
                markersize=8, linewidth=1.5)
    # Exp 13 reference (only final layer = 12 measured, plotted at L=12)
    exp13_cos = [exp13_data.get(f"scifact_exp13_s{s}", {}).get("lora_vs_frozen_tok_cos_mean") for s in seeds]
    exp13_cos = [c for c in exp13_cos if c is not None]
    if exp13_cos:
        exp13_mean = np.mean(exp13_cos)
        ax.scatter([12], [exp13_mean], s=200, color="#d62728", marker="*",
                   edgecolor="black", linewidth=1.5, zorder=10,
                   label=f"Exp 13 (final 128-dim, s42-2024 mean={exp13_mean:.3f})")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("BERT layer ℓ")
    ax.set_ylabel("cos(h_LoRA, h_frozen) per token (mean)")
    ax.set_title("Per-layer anchor proximity — Exp 16 multi-layer vs Exp 13 final-only")
    ax.set_xticks(layer_xs)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(alpha=0.3)

    # Panel B: per-layer tok eff_rank
    ax = axes[1]
    for s, color in zip(seeds, ["#2ca02c", "#5fbf5f", "#a0d8a0"]):
        lab = f"scifact_exp16_s{s}"
        if lab not in results:
            continue
        tok_eff_per_layer = [results[lab][str(ell)]["tok_eff_rank"] for ell in ANCHOR_LAYERS]
        ax.plot(layer_xs, tok_eff_per_layer, "o-", color=color, label=f"Exp 16 s{s}",
                markersize=8, linewidth=1.5)
    # Exp 13 final 128-dim tok_eff_rank reference
    exp13_eff = [exp13_data.get(f"scifact_exp13_s{s}", {}).get("tok_effective_rank") for s in seeds]
    exp13_eff = [v for v in exp13_eff if v is not None]
    if exp13_eff:
        exp13_eff_mean = np.mean(exp13_eff)
        ax.scatter([12], [exp13_eff_mean], s=200, color="#d62728", marker="*",
                   edgecolor="black", linewidth=1.5, zorder=10,
                   label=f"Exp 13 (final, s42-2024 mean={exp13_eff_mean:.2f})")
    # frozen baseline ref
    if "scifact_frozen_multilayer" in results:
        frozen_eff = [results["scifact_frozen_multilayer"][str(ell)]["tok_eff_rank"] for ell in ANCHOR_LAYERS]
        ax.plot(layer_xs, frozen_eff, "--", color="#888", label="frozen baseline", linewidth=1.5)
    ax.set_xlabel("BERT layer ℓ")
    ax.set_ylabel("token effective rank")
    ax.set_title("Per-layer token diversity — Exp 16 vs frozen + Exp 13 ref")
    ax.set_xticks(layer_xs)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel C: layer-12 (final) comparison Exp 16 vs Exp 13 (3-seed mean)
    ax = axes[2]
    # cos: at layer 12 of Exp 16 vs Exp 13 final 128-dim
    methods = ["Exp 13\n(final 128-dim)", "Exp 16\n(L=12, 768-dim)"]
    cos_vals = []
    if exp13_cos:
        cos_vals.append(np.mean(exp13_cos))
    e16_cos_l12 = [results.get(f"scifact_exp16_s{s}", {}).get("12", {}).get("tok_cos_mean") for s in seeds]
    e16_cos_l12 = [c for c in e16_cos_l12 if c is not None]
    cos_vals.append(np.mean(e16_cos_l12) if e16_cos_l12 else 0)
    cos_stds = [np.std(exp13_cos, ddof=1) if len(exp13_cos) > 1 else 0,
                np.std(e16_cos_l12, ddof=1) if len(e16_cos_l12) > 1 else 0]
    x = np.arange(len(methods))
    bars = ax.bar(x, cos_vals, yerr=cos_stds, capsize=5,
                  color=["#d62728", "#2ca02c"], alpha=0.85)
    for b, v in zip(bars, cos_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("cos(h_LoRA, h_frozen) per token (3-seed mean ± std)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Final-layer anchor proximity — Exp 13 vs Exp 16 (at L=12)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Diagnostic B on Exp 16 — multi-layer anchor mechanism (SciFact, 3 seeds)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"repr_collapse_exp16.{ext}")
    plt.close(fig)

    print(f"\nfigure → {out_dir}/repr_collapse_exp16.{{pdf,png}}")
    print(f"data   → {data_path}")


if __name__ == "__main__":
    main()
