#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors


# ------------------------
# Utils
# ------------------------
def l2_normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_probe(path):
    z = np.load(path, allow_pickle=True)
    return {
        "y_true": z["y_true"],
        "proba": z["proba"],
        "paths": z["paths"],
    }


def load_features(path):
    z = np.load(path, allow_pickle=True)
    return z["feats"]


def evaluate(y, proba):
    pred = proba.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


# ------------------------
# Graph
# ------------------------
def knn_graph(feats, k):
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(feats)))
    nbrs.fit(feats)
    idx = nbrs.kneighbors(feats, return_distance=False)

    clean = []
    for i, row in enumerate(idx):
        r = [j for j in row if j != i]
        r = r[:k]
        while len(r) < k:
            r.append(r[-1])
        clean.append(r)

    return np.asarray(clean)


def graph_smooth(p, idx, alpha=0.5, iters=5):
    base = p.copy()
    for _ in range(iters):
        new = np.zeros_like(base)
        for i in range(len(base)):
            new[i] = base[idx[i]].mean(axis=0)
        base = (1 - alpha) * p + alpha * new
        base = base / np.clip(base.sum(axis=1, keepdims=True), 1e-12, None)
    return base


# ------------------------
# Corruption
# ------------------------
def corrupt_seed_labels(labels, seed_mask, n_classes, rate, rng):
    labels = labels.copy()
    seed_idx = np.where(seed_mask)[0]

    n_corrupt = int(round(rate * len(seed_idx)))
    if n_corrupt == 0:
        return labels

    corrupt_idx = rng.choice(seed_idx, size=n_corrupt, replace=False)
    offsets = rng.integers(1, n_classes, size=n_corrupt)
    labels[corrupt_idx] = (labels[corrupt_idx] + offsets) % n_classes

    return labels


# ------------------------
# Graph with seeds
# ------------------------
def graph_label_prop(anchor, idx, rate, rng):
    conf = anchor.max(axis=1)
    labels = anchor.argmax(axis=1)

    threshold = np.quantile(conf, 0.75)
    seed_mask = conf >= threshold

    corrupted_labels = corrupt_seed_labels(
        labels, seed_mask, anchor.shape[1], rate, rng
    )

    q = anchor.copy()
    q[seed_mask] = 0
    q[seed_mask, corrupted_labels[seed_mask]] = 1

    return graph_smooth(q, idx)


# ------------------------
# GRACE
# ------------------------
def compute_gate(anchor, graph, idx):
    conf = anchor.max(axis=1)

    pred_a = anchor.argmax(1)
    pred_g = graph.argmax(1)

    agreement = (pred_a == pred_g).astype(float)

    purity = np.zeros(len(anchor))
    for i in range(len(anchor)):
        purity[i] = np.mean(pred_a[idx[i]] == pred_a[i])

    return np.clip(conf * agreement * purity, 0, 1)


def grace(anchor, graph, idx):
    g = compute_gate(anchor, graph, idx)
    out = (1 - g[:, None]) * anchor + g[:, None] * graph
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--probe_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dataset", default="domainnet")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--teachers", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--corruption_rates", default="0.0,0.1,0.2,0.3,0.4")
    args = parser.parse_args()

    rates = [float(x) for x in args.corruption_rates.split(",")]

    feature_root = Path(args.feature_root)
    probe_root = Path(args.probe_root)

    probas = []
    feats = []
    y = None

    for t in args.teachers:
        probe_path = (
            probe_root
            / args.dataset
            / args.source
            / "probe_outputs"
            / f"seed_{args.seed}"
            / args.target
            / f"{t}.npz"
        )

        feat_path = (
            feature_root
            / args.target
            / "test"
            / f"{t}.npz"
        )

        p = load_probe(probe_path)
        f = load_features(feat_path)

        probas.append(p["proba"])
        feats.append(l2_normalize(f))
        y = p["y_true"]

    probas = np.stack(probas)
    anchor = probas.mean(axis=0)
    anchor = anchor / anchor.sum(axis=1, keepdims=True)

    anchor_feats = l2_normalize(np.concatenate(feats, axis=1))
    idx = knn_graph(anchor_feats, args.k)

    rows = []

    for rate in rates:
        rng = np.random.default_rng(args.seed + int(rate * 1000))

        graph = graph_label_prop(anchor, idx, rate, rng)
        g = grace(anchor, graph, idx)

        for name, p in [
            ("uniform", anchor),
            ("graph", graph),
            ("grace", g),
        ]:
            m = evaluate(y, p)

            rows.append(
                {
                    "dataset": args.dataset,
                    "source": args.source,
                    "target": args.target,
                    "seed": args.seed,
                    "corruption_rate": rate,
                    "method": name,
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "n_samples": len(y),
                }
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / f"{args.source}_{args.target}_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("Saved:", out_csv)


if __name__ == "__main__":
    main()