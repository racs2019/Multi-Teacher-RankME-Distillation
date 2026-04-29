#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def kl_div(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(axis=axis)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_probe(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing probe file: {path}")
    z = np.load(path, allow_pickle=True)
    return {
        "y_true": z["y_true"].astype(np.int64),
        "proba": z["proba"].astype(np.float32),
        "paths": z["paths"],
    }


def load_features(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    z = np.load(path, allow_pickle=True)
    return z["feats"].astype(np.float32)


def knn_graph(feats: np.ndarray, k: int) -> np.ndarray:
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


def graph_smooth(p: np.ndarray, idx: np.ndarray, alpha: float, iters: int, sharpen_temp: float = 1.0) -> np.ndarray:
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


def uniform_ensemble(probas: np.ndarray) -> np.ndarray:
    p = probas.mean(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def entropy_weighted(probas: np.ndarray, temp: float = 1.0) -> np.ndarray:
    scores = -entropy(probas, axis=2) / max(temp, 1e-8)
    weights = softmax_np(scores, axis=0)
    p = (weights[:, :, None] * probas).sum(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def agreement_weighted(probas: np.ndarray, temp: float = 1.0) -> np.ndarray:
    mean = probas.mean(axis=0, keepdims=True)
    scores = -kl_div(probas, mean, axis=2) / max(temp, 1e-8)
    weights = softmax_np(scores, axis=0)
    p = (weights[:, :, None] * probas).sum(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def tent_proxy(probas: np.ndarray, temp: float = 0.25) -> np.ndarray:
    scores = -entropy(probas, axis=2) / max(temp, 1e-8)
    weights = softmax_np(scores, axis=0)
    p = (weights[:, :, None] * probas).sum(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def graph_lame(anchor: np.ndarray, idx: np.ndarray, alpha: float = 0.35, iters: int = 5, sharpen_temp: float = 0.85) -> np.ndarray:
    return graph_smooth(anchor, idx=idx, alpha=alpha, iters=iters, sharpen_temp=sharpen_temp)


def graph_label_prop(anchor: np.ndarray, idx: np.ndarray, top_quantile: float = 0.25, alpha: float = 0.5, iters: int = 5) -> np.ndarray:
    conf = anchor.max(axis=1)
    threshold = float(np.quantile(conf, 1.0 - np.clip(top_quantile, 0.01, 0.99)))

    pseudo = anchor.copy()
    seed = conf >= threshold
    pseudo[seed] = 0.0
    pseudo[seed, anchor[seed].argmax(axis=1)] = 1.0

    return graph_smooth(pseudo, idx=idx, alpha=alpha, iters=iters, sharpen_temp=1.0)


def grace_gate(anchor: np.ndarray, graph: np.ndarray, idx: np.ndarray) -> np.ndarray:
    conf = anchor.max(axis=1)
    pred_anchor = anchor.argmax(axis=1)
    pred_graph = graph.argmax(axis=1)

    agreement = (pred_anchor == pred_graph).astype(np.float32)

    purity = np.zeros(len(anchor), dtype=np.float32)
    for i in range(len(anchor)):
        purity[i] = np.mean(pred_anchor[idx[i]] == pred_anchor[i])

    gate = conf * agreement * purity
    return np.clip(gate, 0.0, 1.0).astype(np.float32)


def grace(anchor: np.ndarray, graph: np.ndarray, idx: np.ndarray) -> np.ndarray:
    g = grace_gate(anchor, graph, idx)
    p = (1.0 - g[:, None]) * anchor + g[:, None] * graph
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_pred = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n_samples": int(len(y_true)),
    }


def save_method_output(path: Path, y_true, proba, paths, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=y_true.astype(np.int64),
        y_pred=proba.argmax(axis=1).astype(np.int64),
        proba=proba.astype(np.float32),
        paths=np.asarray(paths, dtype=object),
        meta_json=json.dumps(meta),
    )


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
    parser.add_argument("--save_outputs", action="store_true")
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

    anchor_uniform = uniform_ensemble(probas)
    anchor_agreement = agreement_weighted(probas, temp=1.0)

    graph_lame_p = graph_lame(anchor_uniform, idx)
    graph_prop_p = graph_label_prop(anchor_agreement, idx)
    grace_p = grace(anchor_agreement, graph_prop_p, idx)

    methods = {
        "uniform": anchor_uniform,
        "entropy_weighted": entropy_weighted(probas, temp=1.0),
        "agreement_weighted": anchor_agreement,
        "tent_proxy": tent_proxy(probas, temp=0.25),
        "graph_lame": graph_lame_p,
        "graph_label_prop": graph_prop_p,
        "grace": grace_p,
    }

    rows = []
    for method, proba in methods.items():
        m = evaluate(y_true, proba)
        rows.append(
            {
                "dataset": args.dataset,
                "source": args.source,
                "target": args.target,
                "seed": args.seed,
                "method": method,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "n_samples": m["n_samples"],
            }
        )

        if args.save_outputs:
            save_method_output(
                outdir
                / "method_outputs"
                / f"seed_{args.seed}"
                / args.target
                / f"{method}.npz",
                y_true=y_true,
                proba=proba,
                paths=paths,
                meta={
                    "dataset": args.dataset,
                    "source": args.source,
                    "target": args.target,
                    "seed": args.seed,
                    "method": method,
                    "teachers": args.teachers,
                },
            )

    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{args.source}_{args.target}_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()