#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors


def softmax_np(x, axis=-1):
    x = x.astype(np.float32)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def kl_div(p, q, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(axis=axis)


def l2_normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_probe(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing probe file: {path}")
    z = np.load(path, allow_pickle=True)
    return {
        "y_true": z["y_true"].astype(np.int64),
        "proba": z["proba"].astype(np.float32),
        "paths": z["paths"],
    }


def load_features(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    z = np.load(path, allow_pickle=True)
    return z["feats"].astype(np.float32)


def knn_graph(feats, k):
    n = feats.shape[0]
    k_eff = min(k + 1, n)

    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nbrs.fit(feats)

    idx = nbrs.kneighbors(feats, return_distance=False)

    clean = []
    for i, row in enumerate(idx):
        r = [j for j in row if j != i]
        if not r:
            r = [i]
        r = r[:k]
        while len(r) < k:
            r.append(r[-1])
        clean.append(r)

    return np.asarray(clean, dtype=np.int64)


def graph_smooth(p, idx, alpha=0.5, iters=5, sharpen_temp=1.0):
    base = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    q = base.copy()

    for _ in range(iters):
        neigh = np.zeros_like(q)
        for i in range(q.shape[0]):
            neigh[i] = q[idx[i]].mean(axis=0)

        q = (1.0 - alpha) * base + alpha * neigh
        q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)

    if sharpen_temp != 1.0:
        q = np.power(np.clip(q, 1e-12, 1.0), 1.0 / sharpen_temp)
        q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)

    return q.astype(np.float32)


def uniform_ensemble(probas):
    p = probas.mean(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def agreement_weighted(probas, temp=1.0):
    mean = probas.mean(axis=0, keepdims=True)
    scores = -kl_div(probas, mean, axis=2) / max(temp, 1e-8)
    weights = softmax_np(scores, axis=0)
    p = (weights[:, :, None] * probas).sum(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def entropy_weighted(probas, temp=1.0):
    scores = -entropy(probas, axis=2) / max(temp, 1e-8)
    weights = softmax_np(scores, axis=0)
    p = (weights[:, :, None] * probas).sum(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def tent_proxy(probas, temp=0.25):
    return entropy_weighted(probas, temp=temp)


def graph_label_prop(anchor, idx, top_quantile=0.25, alpha=0.5, iters=5):
    conf = anchor.max(axis=1)
    threshold = float(np.quantile(conf, 1.0 - np.clip(top_quantile, 0.01, 0.99)))

    pseudo = anchor.copy()
    seed = conf >= threshold

    pseudo[seed] = 0.0
    pseudo[seed, anchor[seed].argmax(axis=1)] = 1.0

    return graph_smooth(pseudo, idx=idx, alpha=alpha, iters=iters, sharpen_temp=1.0)


def graph_lame(anchor, idx, alpha=0.35, iters=5, sharpen_temp=0.85):
    return graph_smooth(anchor, idx=idx, alpha=alpha, iters=iters, sharpen_temp=sharpen_temp)


def local_purity(pred, idx):
    purity = np.zeros(len(pred), dtype=np.float32)
    for i in range(len(pred)):
        purity[i] = np.mean(pred[idx[i]] == pred[i])
    return purity


def make_gate(
    anchor,
    graph,
    idx,
    mode,
    agreement_floor=0.90,
    strength=2.25,
    power=0.75,
):
    pred_anchor = anchor.argmax(axis=1)
    pred_graph = graph.argmax(axis=1)

    conf_anchor = anchor.max(axis=1)
    conf_graph = graph.max(axis=1)

    hard_agreement = (pred_anchor == pred_graph).astype(np.float32)

    anchor_purity = local_purity(pred_anchor, idx)
    graph_purity = local_purity(pred_graph, idx)

    confidence = 0.5 * conf_anchor + 0.5 * conf_graph
    purity_max = np.maximum(anchor_purity, graph_purity)

    if mode == "conf_only":
        gate = confidence

    elif mode == "conf_anchor_purity":
        gate = confidence * anchor_purity

    elif mode == "conf_graph_purity":
        gate = confidence * graph_purity

    elif mode == "conf_max_purity":
        gate = confidence * purity_max

    elif mode == "hard_agreement":
        gate = conf_anchor * hard_agreement * anchor_purity

    elif mode == "full":
        agreement_weight = agreement_floor + (1.0 - agreement_floor) * hard_agreement
        gate = confidence * agreement_weight * purity_max
        gate = strength * gate
        gate = np.power(np.clip(gate, 0.0, 1.0), power)

    else:
        raise ValueError(f"Unknown gate mode: {mode}")

    return np.clip(gate, 0.0, 1.0).astype(np.float32)


def interpolate(anchor, graph, gate):
    p = (1.0 - gate[:, None]) * anchor + gate[:, None] * graph
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def evaluate(y_true, proba):
    y_pred = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n_samples": int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--probe_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--teachers", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=20)

    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    probe_root = Path(args.probe_root)
    outdir = Path(args.outdir)

    probas = []
    feats = []
    y_true = None
    paths = None

    for teacher in args.teachers:
        probe_path = (
            probe_root
            / args.dataset
            / args.source
            / "probe_outputs"
            / f"seed_{args.seed}"
            / args.target
            / f"{teacher}.npz"
        )

        feature_path = (
            feature_root
            / args.target
            / "test"
            / f"{teacher}.npz"
        )

        probe = load_probe(probe_path)
        feat = load_features(feature_path)

        if y_true is None:
            y_true = probe["y_true"]
            paths = probe["paths"]
        else:
            if not np.array_equal(y_true, probe["y_true"]):
                raise ValueError(f"Label mismatch for teacher={teacher}")
            if not np.array_equal(paths, probe["paths"]):
                raise ValueError(f"Path mismatch for teacher={teacher}")

        probas.append(probe["proba"])
        feats.append(l2_normalize(feat))

    probas = np.stack(probas, axis=0)

    anchor_feats = l2_normalize(np.concatenate(feats, axis=1))
    idx = knn_graph(anchor_feats, args.k)

    uniform = uniform_ensemble(probas)
    agreement = agreement_weighted(probas)
    entropy_w = entropy_weighted(probas)
    tent = tent_proxy(probas)

    graph_lame_p = graph_lame(uniform, idx)
    graph_prop_p = graph_label_prop(agreement, idx)

    methods = {
        "uniform_anchor": uniform,
        "agreement_anchor": agreement,
        "entropy_weighted": entropy_w,
        "tent_proxy": tent,
        "graph_lame": graph_lame_p,
        "graph_label_prop": graph_prop_p,
    }

    gates = {}

    for mode in [
        "conf_only",
        "conf_anchor_purity",
        "conf_graph_purity",
        "conf_max_purity",
        "hard_agreement",
        "full",
    ]:
        g = make_gate(
            anchor=agreement,
            graph=graph_prop_p,
            idx=idx,
            mode=mode,
        )
        gates[mode] = g
        methods[f"grace_{mode}"] = interpolate(agreement, graph_prop_p, g)

    rows = []

    for method, proba in methods.items():
        m = evaluate(y_true, proba)

        row = {
            "dataset": args.dataset,
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "method": method,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "n_samples": m["n_samples"],
            "mean_gate": np.nan,
            "median_gate": np.nan,
            "active_gate_frac": np.nan,
        }

        if method.startswith("grace_"):
            mode = method.replace("grace_", "")
            g = gates[mode]
            row["mean_gate"] = float(np.mean(g))
            row["median_gate"] = float(np.median(g))
            row["active_gate_frac"] = float(np.mean(g > 0.05))

        rows.append(row)

    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / f"{args.source}_{args.target}_seed{args.seed}_ablation.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

