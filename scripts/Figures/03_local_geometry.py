#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors


TEACHERS_DEFAULT = [
    "openclip_l14_openai_qgelu",
    "openclip_b16_datacomp",
    "openclip_so400m_siglip",
    "openclip_l14_dfn2b",
    "openclip_h14_laion2b",
    "openclip_h14_378_dfn5b",
    "openclip_convnext_xxlarge",
]


def l2_normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_probe(path: Path):
    z = np.load(path, allow_pickle=True)
    return z["y_true"], z["proba"], z["paths"]


def load_feats(path: Path):
    return np.load(path, allow_pickle=True)["feats"]


def knn(feats, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(feats)
    idx = nn.kneighbors(feats, return_distance=False)[:, 1:]
    return idx


def graph_smooth(p, idx, alpha=0.5, iters=5):
    base = p.copy()
    q = base.copy()
    for _ in range(iters):
        neigh = np.zeros_like(q)
        for i in range(len(q)):
            neigh[i] = q[idx[i]].mean(axis=0)
        q = (1 - alpha) * base + alpha * neigh
        q = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)
    return q


def compute_gate(anchor, graph, idx):
    conf = anchor.max(axis=1)
    pa = anchor.argmax(1)
    pg = graph.argmax(1)
    agreement = (pa == pg).astype(float)

    purity = np.zeros(len(anchor))
    for i in range(len(anchor)):
        purity[i] = np.mean(pa[idx[i]] == pa[i])

    gate = np.clip(conf * agreement * purity, 0, 1)
    return gate, purity


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def disagreement(probas):
    mean_p = probas.mean(axis=0)
    return entropy(mean_p) - entropy(probas).mean(axis=0)


def draw_local_panel(ax, xy, labels, idx, sample_i, title, subtitle):
    neigh = idx[sample_i]
    ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=12, alpha=0.55)
    ax.scatter(xy[neigh, 0], xy[neigh, 1], facecolors="none", edgecolors="black", s=55, linewidths=1.2)
    ax.scatter(xy[sample_i, 0], xy[sample_i, 1], marker="*", s=220, c="red", edgecolors="black", zorder=5)

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.text(0.03, 0.93, subtitle, transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax.set_xlabel("feature dim. 1")
    ax.set_ylabel("feature dim. 2")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="domainnet")
    parser.add_argument("--source", default="quickdraw")
    parser.add_argument("--target", default="sketch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature_root", default="features/domainnet")
    parser.add_argument("--probe_root", default="results")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--teachers", nargs="+", default=TEACHERS_DEFAULT)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    probas, feats = [], []
    y = None

    for t in args.teachers:
        probe_path = (
            Path(args.probe_root)
            / args.dataset
            / args.source
            / "probe_outputs"
            / f"seed_{args.seed}"
            / args.target
            / f"{t}.npz"
        )
        feat_path = Path(args.feature_root) / args.target / "test" / f"{t}.npz"

        yy, p, _ = load_probe(probe_path)
        f = load_feats(feat_path)

        y = yy if y is None else y
        probas.append(p)
        feats.append(l2_normalize(f))

    probas = np.stack(probas)
    anchor = probas.mean(axis=0)
    anchor = anchor / np.clip(anchor.sum(axis=1, keepdims=True), 1e-12, None)

    anchor_feats = l2_normalize(np.concatenate(feats, axis=1))
    idx = knn(anchor_feats, args.k)

    graph = graph_smooth(anchor, idx)
    gate, purity = compute_gate(anchor, graph, idx)
    grace = (1 - gate[:, None]) * anchor + gate[:, None] * graph

    pred_anchor = anchor.argmax(1)
    pred_graph = graph.argmax(1)
    pred_grace = grace.argmax(1)

    correct_anchor = (pred_anchor == y).astype(float)
    correct_grace = (pred_grace == y).astype(float)
    sample_gain = correct_grace - correct_anchor

    geom_score = purity * anchor.max(axis=1)
    disagree = disagreement(probas)

    xy = PCA(n_components=2, random_state=0).fit_transform(anchor_feats)

    high_good = np.argmax(geom_score * (pred_anchor == y))
    weak = np.argmin(np.abs(geom_score - np.median(geom_score)))
    fallback = np.argmax(geom_score * (pred_anchor != pred_graph))

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    draw_local_panel(ax1, xy, pred_anchor, idx, high_good, "(a) Reliable local geometry", "consistent neighbors")
    draw_local_panel(ax2, xy, pred_anchor, idx, weak, "(b) Weak / conflicting geometry", "inconsistent neighbors")
    draw_local_panel(ax3, xy, pred_anchor, idx, fallback, "(c) Fallback case", "graph conflicts with anchor")

    ax4.scatter(geom_score, correct_anchor, s=14, alpha=0.35)
    coef = np.polyfit(geom_score, correct_anchor, 1)
    xs = np.linspace(geom_score.min(), geom_score.max(), 100)
    ax4.plot(xs, coef[0] * xs + coef[1], linewidth=2)
    ax4.set_title("(d) Local geometry predicts reliability", fontweight="bold", fontsize=11)
    ax4.set_xlabel("local predictive geometry score")
    ax4.set_ylabel("anchor correctness")
    ax4.grid(True, alpha=0.25)

    ax5.scatter(disagree + 1e-8, sample_gain, s=14, alpha=0.35)
    ax5.set_xscale("log")
    coef = np.polyfit(np.log(disagree + 1e-8), sample_gain, 1)
    xs = np.logspace(np.log10(disagree[disagree > 0].min()), np.log10(disagree.max()), 100)
    ax5.plot(xs, coef[0] * np.log(xs) + coef[1], linewidth=2)
    ax5.axhline(0, linestyle="--", linewidth=1)
    ax5.set_title("(e) Gains concentrate where teachers disagree", fontweight="bold", fontsize=11)
    ax5.set_xlabel("teacher disagreement")
    ax5.set_ylabel("GRACE gain over anchor")
    ax5.grid(True, alpha=0.25)

    ax6.axis("off")
    acc_anchor = accuracy_score(y, pred_anchor)
    acc_graph = accuracy_score(y, pred_graph)
    acc_grace = accuracy_score(y, pred_grace)

    text = (
        "(f) Reliability-controlled refinement\n\n"
        f"Anchor accuracy: {acc_anchor:.3f}\n"
        f"Graph accuracy:  {acc_graph:.3f}\n"
        f"GRACE accuracy:  {acc_grace:.3f}\n\n"
        f"Mean gate:       {gate.mean():.3f}\n"
        f"Gate active:     {(gate > 0).mean():.3f}\n"
        f"Mean purity:     {purity.mean():.3f}\n\n"
        "Rule:\n"
        r"$p_{final}=(1-g)p_{anchor}+g p_{graph}$"
    )
    ax6.text(0.02, 0.95, text, va="top", fontsize=11)

    fig.suptitle(
        "Local predictive geometry explains when GRACE refines and when it falls back",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png = outdir / "figure_local_geometry.png"
    pdf = outdir / "figure_local_geometry.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("Saved:", png)
    print("Saved:", pdf)


if __name__ == "__main__":
    main()