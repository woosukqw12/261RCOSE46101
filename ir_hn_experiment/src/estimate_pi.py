"""
Estimate false-negative posterior pi_ij from training dynamics.

This is an offline sanity-check script for small labeled datasets such as
fiqa_rlhn. It does not change training. It computes

    pi_ij = P(FN | dynamics features)

with either:
  - supervised Gaussian Naive Bayes calibrated on RLHN FN ground truth
    (diagnostic / upper-bound only), or
  - unsupervised two-component diagonal GMM fit by EM.
  - unsupervised rare-tail Gaussian posterior initialized from the top
    avg-margin tail with a fixed prior.
  - PU logistic learning from high-confidence dynamics seeds and unlabeled
    negatives.
  - label-free posterior-style scores for relabel experiments:
      * pi_loss: low NCE loss / high confidence, close to Passage Sieve.
      * pi_dyn: AUM-style normalized margin dynamics.
      * pi_bayes: reliability-weighted evidence accumulation.

It then reports top-K precision/recall against GT for analysis. The method
inputs can remain label-free; FN ground truth is only needed for diagnostics.

Example:
  python src/estimate_pi.py \
    --log_dir experiments/fiqa_rlhn/logs_baseline \
    --train_path data/processed/fiqa_rlhn/train.jsonl \
    --fn_ground_truth data/processed/fiqa_rlhn/fn_ground_truth.json \
    --output results/pi_fiqa_rlhn.json \
    --adj_output experiments/fiqa_rlhn/adj_pi_K597.json \
    --adj_k 597
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


MASK_VAL = -1e9


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def robust_scale(vals, eps=1e-4):
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad
    if scale < eps:
        scale = float(np.std(vals))
    return max(scale, eps), med


def ranks_from_scores(negs):
    order = np.argsort(-negs)
    ranks = np.empty_like(order, dtype=np.int16)
    ranks[order] = np.arange(1, len(negs) + 1, dtype=np.int16)
    return ranks


def load_dynamics(
    log_dir,
    score_temperature=0.05,
    evidence_tau=1.0,
    reliability_gamma=0.05,
    bayes_prior=0.04,
    bayes_strength=2.0,
):
    epoch_files = sorted(
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.startswith("epoch_") and f.endswith(".jsonl")
    )
    if not epoch_files:
        raise FileNotFoundError(f"No epoch_*.jsonl files found in {log_dir}")

    n_epochs = len(epoch_files)
    n_queries = 0
    num_neg = None
    with open(epoch_files[0], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_queries += 1
            if num_neg is None:
                num_neg = len(json.loads(line)["neg_scores"])

    qids = [""] * n_queries
    qid_to_idx = {}

    margin_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    margin_sq_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    persistent_count = np.zeros((n_queries, num_neg), dtype=np.float32)
    rank_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    rank_top1_count = np.zeros((n_queries, num_neg), dtype=np.float32)
    final_margin = np.zeros((n_queries, num_neg), dtype=np.float32)
    hardneg_prob_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    z_margin_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    final_z_margin = np.zeros((n_queries, num_neg), dtype=np.float32)
    loss_conf_sum = np.zeros((n_queries, num_neg), dtype=np.float32)
    final_loss_conf = np.zeros((n_queries, num_neg), dtype=np.float32)
    reliability_sum = np.zeros((n_queries, 1), dtype=np.float32)
    weighted_evidence_sum = np.zeros((n_queries, num_neg), dtype=np.float32)

    alpha = float(bayes_prior) * float(bayes_strength)
    beta = (1.0 - float(bayes_prior)) * float(bayes_strength)

    for epoch_i, path in enumerate(epoch_files):
        with open(path, encoding="utf-8") as f:
            for line_i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = rec["query_id"]
                pos = float(rec["pos_score"])
                negs = np.asarray(rec["neg_scores"], dtype=np.float32)

                if epoch_i == 0:
                    qids[line_i] = qid
                    qid_to_idx[qid] = line_i

                qi = qid_to_idx.get(qid, line_i)
                margins = negs - pos
                ranks = ranks_from_scores(negs)
                logits = np.concatenate([[pos], negs]).astype(np.float64) / score_temperature
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                losses = -np.log(np.maximum(probs, 1e-12))
                loss_conf = float(losses.mean()) - losses[1:]

                scale, center = robust_scale(margins)
                z_margin = (margins - center) / scale
                reliability = float(sigmoid((pos - float(negs.mean())) / reliability_gamma))
                evidence = sigmoid(z_margin / evidence_tau)

                margin_sum[qi] += margins
                margin_sq_sum[qi] += margins ** 2
                persistent_count[qi] += (margins > 0).astype(np.float32)
                rank_sum[qi] += ranks.astype(np.float32)
                rank_top1_count[qi] += (ranks == 1).astype(np.float32)
                final_margin[qi] = margins
                hardneg_prob_sum[qi] += probs[1:].astype(np.float32)
                z_margin_sum[qi] += z_margin.astype(np.float32)
                final_z_margin[qi] = z_margin.astype(np.float32)
                loss_conf_sum[qi] += loss_conf.astype(np.float32)
                final_loss_conf[qi] = loss_conf.astype(np.float32)
                reliability_sum[qi, 0] += reliability
                weighted_evidence_sum[qi] += (reliability * evidence).astype(np.float32)

    avg_margin = margin_sum / n_epochs
    margin_var = margin_sq_sum / n_epochs - avg_margin ** 2
    avg_rank = rank_sum / n_epochs

    features = {
        "avg_margin": avg_margin,
        "final_margin": final_margin,
        "persistent_frac": persistent_count / n_epochs,
        "top1_frac": rank_top1_count / n_epochs,
        "neg_avg_rank": -avg_rank,
        "margin_var": margin_var,
        "hardneg_prob": hardneg_prob_sum / n_epochs,
        "avg_z_margin": z_margin_sum / n_epochs,
        "final_z_margin": final_z_margin,
        "loss_confidence": loss_conf_sum / n_epochs,
        "final_loss_confidence": final_loss_conf,
        "reliability_avg": np.repeat(reliability_sum / n_epochs, num_neg, axis=1),
        "pi_bayes": (alpha + weighted_evidence_sum) / (alpha + beta + reliability_sum),
    }
    return qids, num_neg, n_epochs, features


def load_fn_labels(train_path, fn_ground_truth, qids, num_neg):
    idx_to_qid = {}
    with open(train_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                idx_to_qid[i] = json.loads(line)["qid"]

    with open(fn_ground_truth, encoding="utf-8") as f:
        gt = json.load(f)

    fn_set = set()
    for qidx_str, neg_indices in gt["fn_pairs"].items():
        qid = idx_to_qid.get(int(qidx_str))
        if qid is None:
            continue
        for ni in neg_indices:
            fn_set.add((qid, int(ni)))

    y = np.zeros(len(qids) * num_neg, dtype=np.int8)
    for qi, qid in enumerate(qids):
        for ni in range(num_neg):
            if (qid, ni) in fn_set:
                y[qi * num_neg + ni] = 1
    return y


def build_feature_matrix(features, exclude=("hardneg_prob", "pi_bayes")):
    names = [name for name in features if name not in exclude]
    cols = [features[name].reshape(-1).astype(np.float64) for name in names]
    x = np.stack(cols, axis=1)
    return names, x


def gaussian_nb_posterior(x, y, var_floor=1e-5):
    prior = float(y.mean())
    prior = min(max(prior, 1e-6), 1 - 1e-6)

    x0 = x[y == 0]
    x1 = x[y == 1]
    mu0, mu1 = x0.mean(axis=0), x1.mean(axis=0)
    var0 = np.maximum(x0.var(axis=0), var_floor)
    var1 = np.maximum(x1.var(axis=0), var_floor)

    def log_gaussian(vals, mu, var):
        return -0.5 * (np.log(2 * np.pi * var) + ((vals - mu) ** 2) / var).sum(axis=1)

    logp1 = np.log(prior) + log_gaussian(x, mu1, var1)
    logp0 = np.log(1 - prior) + log_gaussian(x, mu0, var0)
    logits = np.clip(logp1 - logp0, -60, 60)
    pi = 1 / (1 + np.exp(-logits))
    params = {
        "prior": prior,
        "mu_tn": mu0.tolist(),
        "mu_fn": mu1.tolist(),
        "var_tn": var0.tolist(),
        "var_fn": var1.tolist(),
    }
    return pi, params


def gaussian_nb_oof_posterior(x, y, n_queries, num_neg, n_folds):
    if n_folds <= 1:
        return gaussian_nb_posterior(x, y)

    pi = np.zeros(len(y), dtype=np.float64)
    fold_params = []
    query_folds = np.arange(n_queries) % n_folds
    pair_folds = np.repeat(query_folds, num_neg)

    for fold in range(n_folds):
        train_mask = pair_folds != fold
        test_mask = pair_folds == fold
        fold_pi, params = gaussian_nb_posterior(x[train_mask], y[train_mask])

        prior = params["prior"]
        mu0 = np.asarray(params["mu_tn"], dtype=np.float64)
        mu1 = np.asarray(params["mu_fn"], dtype=np.float64)
        var0 = np.asarray(params["var_tn"], dtype=np.float64)
        var1 = np.asarray(params["var_fn"], dtype=np.float64)
        xt = x[test_mask]

        def log_gaussian(vals, mu, var):
            return -0.5 * (np.log(2 * np.pi * var) + ((vals - mu) ** 2) / var).sum(axis=1)

        logp1 = np.log(prior) + log_gaussian(xt, mu1, var1)
        logp0 = np.log(1 - prior) + log_gaussian(xt, mu0, var0)
        logits = np.clip(logp1 - logp0, -60, 60)
        pi[test_mask] = 1 / (1 + np.exp(-logits))
        # Keep a tiny fold sanity metric for debugging.
        params["fold"] = fold
        params["n_train"] = int(train_mask.sum())
        params["n_test"] = int(test_mask.sum())
        params["train_fn_rate"] = float(y[train_mask].mean())
        params["test_fn_rate"] = float(y[test_mask].mean())
        params["train_ap"] = average_precision(y[train_mask], fold_pi)
        fold_params.append(params)

    full_pi, full_params = gaussian_nb_posterior(x, y)
    return pi, {"folds": fold_params, "full_fit": full_params}


def standardize(x):
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-6)
    return (x - mean) / std, mean, std


def log_diag_gaussian(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + ((x - mu) ** 2) / var).sum(axis=1)


def unsupervised_gmm_posterior(x, feature_names, max_iter=200, tol=1e-7, var_floor=1e-4):
    """Two-component diagonal GMM. FN component is the one with higher avg_margin."""
    xz, mean, std = standardize(x)
    avg_margin_idx = feature_names.index("avg_margin")
    score = x[:, avg_margin_idx]

    # Conservative initialization: high-margin tail is suspected FN-like.
    tail = score >= np.quantile(score, 0.9)
    if tail.sum() == 0 or tail.sum() == len(score):
        tail = score >= np.median(score)

    resp = np.zeros((len(x), 2), dtype=np.float64)
    resp[:, 1] = tail.astype(np.float64) * 0.8 + (~tail).astype(np.float64) * 0.2
    resp[:, 0] = 1.0 - resp[:, 1]

    prev_ll = -np.inf
    for it in range(max_iter):
        weights = resp.sum(axis=0)
        # Keep EM away from a degenerate all-one-component solution.
        weights = np.maximum(weights, 1e-6)
        pis = weights / weights.sum()
        mus = (resp.T @ xz) / weights[:, None]
        vars_ = np.zeros_like(mus)
        for k in range(2):
            diff = xz - mus[k]
            vars_[k] = np.maximum((resp[:, k][:, None] * diff ** 2).sum(axis=0) / weights[k], var_floor)

        logp = np.stack(
            [np.log(pis[k]) + log_diag_gaussian(xz, mus[k], vars_[k]) for k in range(2)],
            axis=1,
        )
        max_logp = logp.max(axis=1, keepdims=True)
        probs = np.exp(logp - max_logp)
        denom = probs.sum(axis=1, keepdims=True)
        resp = probs / denom
        ll = float((max_logp[:, 0] + np.log(denom[:, 0])).sum())
        if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            break
        prev_ll = ll

    fn_comp = int(mus[:, avg_margin_idx].argmax())
    pi = resp[:, fn_comp]
    params = {
        "feature_standardize_mean": mean.tolist(),
        "feature_standardize_std": std.tolist(),
        "mixture_weights": pis.tolist(),
        "mu": mus.tolist(),
        "var": vars_.tolist(),
        "fn_component": fn_comp,
        "iterations": it + 1,
        "log_likelihood": prev_ll,
    }
    return pi, params


def unsupervised_tail_posterior(x, feature_names, tail_prior=0.015, var_floor=1e-4):
    """Label-free rare-component posterior from the high avg-margin tail."""
    tail_prior = min(max(float(tail_prior), 1e-4), 0.5)
    xz, mean, std = standardize(x)
    avg_margin_idx = feature_names.index("avg_margin")
    score = x[:, avg_margin_idx]
    tail = score >= np.quantile(score, 1 - tail_prior)
    if tail.sum() < 2 or (~tail).sum() < 2:
        raise ValueError("tail_prior produced a degenerate split")

    mu1 = xz[tail].mean(axis=0)
    mu0 = xz[~tail].mean(axis=0)
    var1 = np.maximum(xz[tail].var(axis=0), var_floor)
    var0 = np.maximum(xz[~tail].var(axis=0), var_floor)

    logp1 = np.log(tail_prior) + log_diag_gaussian(xz, mu1, var1)
    logp0 = np.log(1 - tail_prior) + log_diag_gaussian(xz, mu0, var0)
    logits = np.clip(logp1 - logp0, -60, 60)
    pi = 1 / (1 + np.exp(-logits))

    params = {
        "tail_prior": tail_prior,
        "tail_count": int(tail.sum()),
        "feature_standardize_mean": mean.tolist(),
        "feature_standardize_std": std.tolist(),
        "mu_tn": mu0.tolist(),
        "mu_tail": mu1.tolist(),
        "var_tn": var0.tolist(),
        "var_tail": var1.tolist(),
    }
    return pi, params


def softplus(x):
    return np.logaddexp(0, x)


def make_pu_seed_mask(x, feature_names, seed_frac=0.01, strategy="tail"):
    seed_frac = min(max(float(seed_frac), 1e-5), 0.5)
    xz, _, _ = standardize(x)

    avg_idx = feature_names.index("avg_margin")
    final_idx = feature_names.index("final_margin")
    persistent_idx = feature_names.index("persistent_frac")
    top1_idx = feature_names.index("top1_frac")

    if strategy == "tail":
        seed_score = x[:, avg_idx]
    elif strategy == "conjunction":
        seed_score = (
            xz[:, avg_idx]
            + xz[:, final_idx]
            + xz[:, persistent_idx]
            + xz[:, top1_idx]
        )
    else:
        raise ValueError(f"Unknown PU seed strategy: {strategy}")

    threshold = np.quantile(seed_score, 1 - seed_frac)
    seed = seed_score >= threshold
    if seed.sum() < 2:
        top = np.argsort(-seed_score)[:2]
        seed = np.zeros(len(seed_score), dtype=bool)
        seed[top] = True
    return seed, seed_score


def pu_logistic_posterior(
    x,
    feature_names,
    seed_frac=0.01,
    class_prior=0.04,
    seed_strategy="conjunction",
    unlabeled_mode="all",
    lr=0.05,
    steps=3000,
    l2=1e-4,
    seed=42,
):
    """Non-negative PU logistic learner from seed positives and unlabeled pairs."""
    class_prior = min(max(float(class_prior), 1e-5), 0.5)
    xz, mean, std = standardize(x)
    seed_mask, seed_score = make_pu_seed_mask(x, feature_names, seed_frac, seed_strategy)
    p_idx = np.where(seed_mask)[0]
    if unlabeled_mode == "non_seed":
        u_idx = np.where(~seed_mask)[0]
    elif unlabeled_mode == "all":
        u_idx = np.arange(len(x))
    else:
        raise ValueError(f"Unknown unlabeled_mode: {unlabeled_mode}")

    xb = np.concatenate([xz, np.ones((len(xz), 1), dtype=xz.dtype)], axis=1)
    xp = xb[p_idx]
    xu = xb[u_idx]

    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=xb.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    eps = 1e-8
    losses = []

    for step in range(1, steps + 1):
        fp = xp @ w
        fu = xu @ w

        positive_risk = class_prior * softplus(-fp).mean()
        negative_risk = softplus(fu).mean() - class_prior * softplus(fp).mean()

        grad = class_prior * (xp.T @ (sigmoid(fp) - 1)) / len(p_idx)
        if negative_risk > 0:
            grad += (xu.T @ sigmoid(fu)) / len(u_idx)
            grad -= class_prior * (xp.T @ sigmoid(fp)) / len(p_idx)
            loss = positive_risk + negative_risk
        else:
            loss = positive_risk

        # Do not regularize the bias term.
        reg = np.r_[w[:-1], 0.0]
        loss += 0.5 * l2 * float(np.dot(reg, reg))
        grad += l2 * reg

        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad ** 2)
        m_hat = m / (1 - 0.9 ** step)
        v_hat = v / (1 - 0.999 ** step)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if step == 1 or step % 250 == 0 or step == steps:
            losses.append({
                "step": step,
                "loss": float(loss),
                "positive_risk": float(positive_risk),
                "negative_risk": float(negative_risk),
            })

    scores = xb @ w
    pi = sigmoid(scores)
    params = {
        "seed_frac": seed_frac,
        "class_prior": class_prior,
        "seed_strategy": seed_strategy,
        "unlabeled_mode": unlabeled_mode,
        "seed_count": int(seed_mask.sum()),
        "feature_standardize_mean": mean.tolist(),
        "feature_standardize_std": std.tolist(),
        "weights": w[:-1].tolist(),
        "bias": float(w[-1]),
        "loss_trace": losses,
        "seed_score_mean": float(seed_score.mean()),
        "seed_score_threshold": float(seed_score[seed_mask].min()),
    }
    return pi, params, seed_mask


def average_precision(y, scores):
    order = np.argsort(-scores)
    y_sorted = y[order]
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / positives)


def roc_auc(y, scores):
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def topk_table(y, scores, budgets):
    order = np.argsort(-scores)
    fn_total = int(y.sum())
    rows = {}
    for k in budgets:
        k = min(int(k), len(y))
        selected = order[:k]
        tp = int(y[selected].sum())
        precision = tp / k if k else 0.0
        recall = tp / fn_total if fn_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[str(k)] = {"tp": tp, "precision": precision, "recall": recall, "f1": f1}
    return rows


def budget_tail_table(y, x, feature_names, budgets, risk_values=None):
    rows = {}
    n_pairs = len(y)
    for k in budgets:
        k = min(int(k), n_pairs)
        alpha = k / n_pairs
        pi, _ = unsupervised_tail_posterior(x, feature_names, alpha)
        scores = pi if risk_values is None else pi * risk_values
        rows.update(topk_table(y, scores, [k]))
    return rows


def calibration_table(y, scores, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append({
            "lo": float(lo),
            "hi": float(hi),
            "count": count,
            "mean_pi": float(scores[mask].mean()),
            "fn_rate": float(y[mask].mean()),
        })
    return rows


def write_adj(path, qids, num_neg, scores, k):
    order = np.argsort(-scores)[:k]
    adj = {}
    for flat_idx in order:
        qi = int(flat_idx // num_neg)
        ni = int(flat_idx % num_neg)
        adj.setdefault(qids[qi], [0.0] * num_neg)
        adj[qids[qi]][ni] = MASK_VAL
    with open(path, "w", encoding="utf-8") as f:
        json.dump(adj, f)


def selected_pair_map(qids, num_neg, scores, k):
    order = np.argsort(-scores)[:k]
    relabel_map = {}
    selected = []
    for flat_idx in order:
        qi = int(flat_idx // num_neg)
        ni = int(flat_idx % num_neg)
        qid = qids[qi]
        relabel_map.setdefault(qid, []).append(ni)
        selected.append({
            "qid": qid,
            "neg_idx": ni,
            "score": float(scores[flat_idx]),
            "flat_idx": int(flat_idx),
        })
    return relabel_map, selected


def write_relabel_train(train_path, output_path, qids, num_neg, scores, k):
    relabel_map, selected = selected_pair_map(qids, num_neg, scores, k)
    n_added = 0
    n_skipped = 0
    n_removed = 0
    selected_set = {(row["qid"], int(row["neg_idx"])) for row in selected}

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    n_original = 0
    with open(train_path, encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue

            row = json.loads(line)
            n_original += 1

            qid = str(row["qid"])
            negs = row.get("negatives", [])
            neg_ids = row.get("neg_ids", list(range(len(negs))))
            selected_idxs = {
                int(neg_idx)
                for neg_idx in relabel_map.get(qid, [])
                if int(neg_idx) < len(negs)
            }

            if selected_idxs:
                base_row = dict(row)
                base_row["negatives"] = [
                    n for i, n in enumerate(negs)
                    if i not in selected_idxs
                ]
                base_row["neg_ids"] = [
                    n for i, n in enumerate(neg_ids)
                    if i not in selected_idxs
                ]
                n_removed += len(selected_idxs)
                dst.write(json.dumps(base_row) + "\n")
            else:
                dst.write(json.dumps(row) + "\n")

            for neg_idx in relabel_map.get(qid, []):
                if neg_idx >= len(negs):
                    n_skipped += 1
                    continue
                exclude_idxs = selected_idxs
                remaining_negs = [
                    n for i, n in enumerate(negs)
                    if i not in exclude_idxs
                ]
                remaining_neg_ids = [
                    n for i, n in enumerate(neg_ids)
                    if i not in exclude_idxs
                ]
                if not remaining_negs:
                    n_skipped += 1
                    continue
                new_row = {
                    "qid": qid,
                    "query": row["query"],
                    "positives": [negs[neg_idx]],
                    "pos_ids": [neg_ids[neg_idx]],
                    "negatives": remaining_negs,
                    "neg_ids": remaining_neg_ids,
                }
                dst.write(json.dumps(new_row) + "\n")
                n_added += 1

    os.replace(tmp_path, out_path)

    return {
        "original_instances": n_original,
        "selected_pairs": len(selected_set),
        "added_instances": n_added,
        "skipped_pairs": n_skipped,
        "removed_negative_refs": n_removed,
        "total_instances": n_original + n_added,
    }


def write_selected_pairs(path, selected, score_values):
    n_pairs = len(next(iter(score_values.values())))
    details = []
    for item in selected:
        flat_idx = item["flat_idx"]
        rec = {k: v for k, v in item.items() if k != "flat_idx"}
        rec["scores"] = {
            name: float(values[flat_idx])
            for name, values in score_values.items()
            if len(values) == n_pairs
        }
        details.append(rec)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--fn_ground_truth", default=None)
    parser.add_argument("--output", default="results/pi_analysis.json")
    parser.add_argument("--budgets", default="100,300,597,1000,1500,3000,4500")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["none", "supervised_nb", "unsup_gmm", "unsup_tail", "pu", "both"],
        default="both",
    )
    parser.add_argument("--score_temperature", type=float, default=0.05)
    parser.add_argument("--loss_tau", type=float, default=1.0)
    parser.add_argument("--pi_tau", type=float, default=1.0)
    parser.add_argument("--evidence_tau", type=float, default=1.0)
    parser.add_argument("--reliability_gamma", type=float, default=0.05)
    parser.add_argument("--bayes_prior", type=float, default=0.04)
    parser.add_argument("--bayes_strength", type=float, default=2.0)
    parser.add_argument("--tail_prior", type=float, default=0.015)
    parser.add_argument("--pu_seed_frac", type=float, default=0.01)
    parser.add_argument("--pu_prior", type=float, default=0.04)
    parser.add_argument("--pu_seed_strategy", choices=["tail", "conjunction"], default="conjunction")
    parser.add_argument("--pu_unlabeled", choices=["all", "non_seed"], default="all")
    parser.add_argument("--pu_steps", type=int, default=3000)
    parser.add_argument("--pu_lr", type=float, default=0.05)
    parser.add_argument("--pu_l2", type=float, default=1e-4)
    parser.add_argument("--adj_output", default=None)
    parser.add_argument("--adj_k", type=int, default=None)
    parser.add_argument("--adj_score", default=None)
    parser.add_argument("--relabel_output", default=None)
    parser.add_argument("--relabel_k", type=int, default=None)
    parser.add_argument("--relabel_score", default=None)
    parser.add_argument("--selected_pairs_output", default=None)
    args = parser.parse_args()

    qids, num_neg, n_epochs, features = load_dynamics(
        args.log_dir,
        score_temperature=args.score_temperature,
        evidence_tau=args.evidence_tau,
        reliability_gamma=args.reliability_gamma,
        bayes_prior=args.bayes_prior,
        bayes_strength=args.bayes_strength,
    )
    feature_names, x = build_feature_matrix(features)
    y = None
    if args.fn_ground_truth:
        y = load_fn_labels(args.train_path, args.fn_ground_truth, qids, num_neg)
    if args.mode in ("supervised_nb", "both") and y is None:
        raise ValueError("--fn_ground_truth is required for supervised_nb/both diagnostics")

    pi_nb = params_nb = full_pi_nb = full_params_nb = None
    if args.mode in ("supervised_nb", "both"):
        pi_nb, params_nb = gaussian_nb_oof_posterior(x, y, len(qids), num_neg, args.cv_folds)
        full_pi_nb, full_params_nb = gaussian_nb_posterior(x, y)

    pi_gmm = params_gmm = None
    if args.mode in ("unsup_gmm", "both"):
        pi_gmm, params_gmm = unsupervised_gmm_posterior(x, feature_names)

    pi_tail = params_tail = None
    if args.mode in ("unsup_tail", "both"):
        pi_tail, params_tail = unsupervised_tail_posterior(x, feature_names, args.tail_prior)

    pi_pu = params_pu = pu_seed_mask = None
    if args.mode in ("pu", "both"):
        pi_pu, params_pu, pu_seed_mask = pu_logistic_posterior(
            x,
            feature_names,
            seed_frac=args.pu_seed_frac,
            class_prior=args.pu_prior,
            seed_strategy=args.pu_seed_strategy,
            unlabeled_mode=args.pu_unlabeled,
            lr=args.pu_lr,
            steps=args.pu_steps,
            l2=args.pu_l2,
        )

    budgets = [int(v) for v in args.budgets.split(",") if v.strip()]

    hardneg_prob = features["hardneg_prob"].reshape(-1)
    pi_loss = sigmoid(features["loss_confidence"].reshape(-1) / args.loss_tau)
    pi_dyn = sigmoid(features["avg_z_margin"].reshape(-1) / args.pi_tau)
    pi_bayes = features["pi_bayes"].reshape(-1)

    score_values = {
        "avg_margin": features["avg_margin"].reshape(-1),
        "final_margin": features["final_margin"].reshape(-1),
        "persistent_frac": features["persistent_frac"].reshape(-1),
        "hardneg_prob": hardneg_prob,
        "avg_z_margin": features["avg_z_margin"].reshape(-1),
        "final_z_margin": features["final_z_margin"].reshape(-1),
        "loss_confidence": features["loss_confidence"].reshape(-1),
        "final_loss_confidence": features["final_loss_confidence"].reshape(-1),
        "pi_loss": pi_loss,
        "pi_dyn": pi_dyn,
        "pi_bayes": pi_bayes,
        "risk_loss": pi_loss * hardneg_prob,
        "risk_dyn": pi_dyn * hardneg_prob,
        "risk_bayes": pi_bayes * hardneg_prob,
    }
    signal_results = {}
    metrics = {}
    calibration = {}

    def add_signal(name, scores, calibrate=False):
        score_values[name] = scores
        if y is None:
            return
        signal_results[name] = topk_table(y, scores, budgets)
        metrics[f"{name}_auc"] = roc_auc(y, scores)
        metrics[f"{name}_ap"] = average_precision(y, scores)
        if calibrate:
            calibration[name] = calibration_table(y, scores)

    if y is not None:
        metrics["base_rate"] = float(y.mean())
        for name in [
            "avg_margin",
            "final_margin",
            "persistent_frac",
            "hardneg_prob",
            "avg_z_margin",
            "loss_confidence",
            "pi_loss",
            "pi_dyn",
            "pi_bayes",
            "risk_loss",
            "risk_dyn",
            "risk_bayes",
        ]:
            add_signal(name, score_values[name], calibrate=name.startswith("pi_"))

    if pi_nb is not None:
        score_values["pi_supervised_nb_oof"] = pi_nb
        score_values["pi_supervised_nb_full"] = full_pi_nb
        add_signal("pi_supervised_nb_oof", pi_nb, calibrate=True)
        add_signal("pi_supervised_nb_full", full_pi_nb, calibrate=True)

    if pi_gmm is not None:
        add_signal("pi_unsup_gmm", pi_gmm, calibrate=True)

    if pi_tail is not None:
        add_signal("pi_unsup_tail", pi_tail, calibrate=True)
        add_signal("risk_unsup_tail", pi_tail * hardneg_prob)
        if y is not None:
            signal_results["pi_budget_tail"] = budget_tail_table(y, x, feature_names, budgets)
            signal_results["risk_budget_tail"] = budget_tail_table(
                y, x, feature_names, budgets, risk_values=hardneg_prob
            )

    if pi_pu is not None:
        add_signal("pi_pu", pi_pu, calibrate=True)
        add_signal("risk_pu", pi_pu * hardneg_prob)
        if y is not None:
            metrics["pu_seed_precision"] = float(y[pu_seed_mask].mean())

    result = {
        "n_epochs": n_epochs,
        "n_queries": len(qids),
        "num_neg": num_neg,
        "n_pairs": int(len(qids) * num_neg),
        "fn_total": None if y is None else int(y.sum()),
        "fn_rate": None if y is None else float(y.mean()),
        "feature_names": feature_names,
        "mode": args.mode,
        "cv_folds": args.cv_folds,
        "params": {
            "deterministic": {
                "score_temperature": args.score_temperature,
                "loss_tau": args.loss_tau,
                "pi_tau": args.pi_tau,
                "evidence_tau": args.evidence_tau,
                "reliability_gamma": args.reliability_gamma,
                "bayes_prior": args.bayes_prior,
                "bayes_strength": args.bayes_strength,
            },
            "supervised_nb": params_nb,
            "supervised_nb_full": full_params_nb,
            "unsup_gmm": params_gmm,
            "unsup_tail": params_tail,
            "pu": params_pu,
        },
        "metrics": metrics,
        "topk": signal_results,
        "calibration": calibration,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if args.adj_output:
        k = args.adj_k or budgets[0]
        if args.adj_score:
            if args.adj_score not in score_values:
                raise ValueError(f"Unknown --adj_score {args.adj_score}; available: {sorted(score_values)}")
            adj_scores = score_values[args.adj_score]
        elif args.mode == "unsup_gmm":
            adj_scores = pi_gmm
        elif args.mode == "unsup_tail":
            adj_scores = pi_tail * hardneg_prob
        elif args.mode == "pu":
            adj_scores = pi_pu * hardneg_prob
        elif args.mode == "supervised_nb":
            # The adj file is for an oracle-GT small-data diagnostic ablation.
            adj_scores = full_pi_nb
        else:
            # Use the stronger label-free rare-tail scores by default.
            adj_scores = pi_tail
        write_adj(args.adj_output, qids, num_neg, adj_scores, k)

    relabel_stats = None
    selected = None
    if args.relabel_output:
        k = args.relabel_k or args.adj_k or budgets[0]
        relabel_score = args.relabel_score or args.adj_score or "pi_bayes"
        if relabel_score not in score_values:
            raise ValueError(f"Unknown relabel score {relabel_score}; available: {sorted(score_values)}")
        relabel_scores = score_values[relabel_score]
        _, selected = selected_pair_map(qids, num_neg, relabel_scores, k)
        relabel_stats = write_relabel_train(
            args.train_path,
            args.relabel_output,
            qids,
            num_neg,
            relabel_scores,
            k,
        )
        result["relabel"] = {
            "score": relabel_score,
            "k": int(k),
            "policy": "clean",
            "output": args.relabel_output,
            **relabel_stats,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.selected_pairs_output:
        pair_score = args.relabel_score or args.adj_score or "pi_bayes"
        if pair_score not in score_values:
            raise ValueError(f"Unknown selected-pairs score {pair_score}; available: {sorted(score_values)}")
        k = args.relabel_k or args.adj_k or budgets[0]
        if selected is None:
            _, selected = selected_pair_map(qids, num_neg, score_values[pair_score], k)
        write_selected_pairs(args.selected_pairs_output, selected, score_values)

    print(f"Saved → {args.output}")
    if y is not None:
        print(f"FN rate: {result['fn_rate']:.4f} ({result['fn_total']}/{result['n_pairs']})")
    else:
        print("FN ground truth not provided; skipped offline precision/AUC diagnostics.")
    for name in [
        "pi_loss",
        "pi_dyn",
        "pi_bayes",
        "risk_loss",
        "risk_dyn",
        "risk_bayes",
        "pi_unsup_tail",
        "risk_unsup_tail",
        "pi_budget_tail",
        "risk_budget_tail",
        "pi_unsup_gmm",
        "pi_pu",
        "risk_pu",
        "pi_supervised_nb_oof",
    ]:
        if name not in result["topk"]:
            continue
        auc_key = f"{name}_auc"
        ap_key = f"{name}_ap"
        if auc_key in result["metrics"]:
            print(f"{name} AUC={result['metrics'][auc_key]:.4f} AP={result['metrics'][ap_key]:.4f}")
        else:
            print(f"{name}")
        for k in budgets:
            row = result["topk"][name][str(k)]
            print(f"{name} P@{k}={row['precision']:.4f} R={row['recall']:.4f} TP={row['tp']}")
    if args.adj_output:
        print(f"Saved adj → {args.adj_output}")
    if args.relabel_output:
        print(f"Saved relabeled train → {args.relabel_output}")
        print(
            "Relabel added "
            f"{relabel_stats['added_instances']} / {relabel_stats['selected_pairs']} selected pairs"
        )
    if args.selected_pairs_output:
        print(f"Saved selected pairs → {args.selected_pairs_output}")


if __name__ == "__main__":
    main()
