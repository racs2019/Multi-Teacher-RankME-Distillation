import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def normalize_rows(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _find_npz(feature_root, dataset, domain, split, teacher):
    root = Path(feature_root)

    candidates = [
        root / domain / split / f"{teacher}.npz",
        root / dataset / domain / split / f"{teacher}.npz",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find .npz for teacher={teacher}, domain={domain}, split={split}. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def _load_npz_arrays(path):
    data = np.load(path, allow_pickle=True)

    feature_keys = ["features", "feats", "embeddings", "x", "X"]
    label_keys = ["labels", "y", "targets", "target"]

    x = None
    y = None

    for k in feature_keys:
        if k in data:
            x = data[k]
            break

    for k in label_keys:
        if k in data:
            y = data[k]
            break

    if x is None:
        raise KeyError(
            f"No feature array found in {path}. Available keys: {list(data.keys())}"
        )

    return x, y


def load_features(feature_root, dataset, domain, teacher, split):
    path = _find_npz(feature_root, dataset, domain, split, teacher)
    x, _ = _load_npz_arrays(path)
    return x.astype(np.float32)


def load_labels(feature_root, dataset, domain, split):
    root = Path(feature_root)

    candidates = [
        root / domain / split / "openclip_b16_datacomp.npz",
        root / dataset / domain / split / "openclip_b16_datacomp.npz",
    ]

    path = None
    for p in candidates:
        if p.exists():
            path = p
            break

    if path is None:
        folders = [
            root / domain / split,
            root / dataset / domain / split,
        ]
        for folder in folders:
            if folder.exists():
                files = sorted(folder.glob("*.npz"))
                if files:
                    path = files[0]
                    break

    if path is None:
        raise FileNotFoundError(
            f"Could not find any .npz for labels in domain={domain}, split={split}"
        )

    _, y = _load_npz_arrays(path)

    if y is None:
        raise KeyError(
            f"No label array found in {path}. Available label keys expected: labels/y/targets/target"
        )

    return y.astype(np.int64)


def train_probe(x_train, y_train, seed=0):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    clf = LogisticRegression(
        max_iter=2000,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_train_scaled, y_train)

    return scaler, clf


def predict_probs(x, scaler, clf):
    return clf.predict_proba(scaler.transform(x))


def compute_anchor_probs(probs, mode="agreement"):
    """
    probs: [N, M, C]
    """
    if mode == "uniform":
        return probs.mean(axis=1)

    if mode == "agreement":
        mean_probs = probs.mean(axis=1)
        eps = 1e-12

        kl_terms = []
        for m in range(probs.shape[1]):
            p = np.clip(probs[:, m, :], eps, 1.0)
            q = np.clip(mean_probs, eps, 1.0)
            kl = (p * (np.log(p) - np.log(q))).sum(axis=1)
            kl_terms.append(kl)

        kl = np.stack(kl_terms, axis=1)
        scores = -kl
        scores = scores - scores.max(axis=1, keepdims=True)

        weights = np.exp(scores)
        weights = weights / weights.sum(axis=1, keepdims=True)

        return (weights[:, :, None] * probs).sum(axis=1)

    raise ValueError(f"Unknown anchor mode: {mode}")


def compute_knn(features, k=20):
    features = normalize_rows(features)
    dists = pairwise_distances(features, metric="cosine")
    np.fill_diagonal(dists, np.inf)
    return np.argsort(dists, axis=1)[:, :k]


def compute_local_purity(anchor_preds, knn):
    purity = np.zeros(len(anchor_preds), dtype=np.float32)

    for i in range(len(anchor_preds)):
        neigh = knn[i]
        purity[i] = np.mean(anchor_preds[neigh] == anchor_preds[i])

    return purity


def compute_teacher_disagreement(probs):
    """
    Fraction of teachers not agreeing with plurality prediction.
    probs: [N, M, C]
    """
    teacher_preds = probs.argmax(axis=2)
    n, m = teacher_preds.shape

    disagreement = np.zeros(n, dtype=np.float32)

    for i in range(n):
        _, counts = np.unique(teacher_preds[i], return_counts=True)
        disagreement[i] = 1.0 - counts.max() / m

    return disagreement


def bin_stats(df, score_col="local_purity", correctness_col="correct", n_bins=10):
    df = df.copy()

    try:
        df["bin"] = pd.qcut(df[score_col], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df[score_col], bins=min(n_bins, 5), duplicates="drop")

    return (
        df.groupby("bin", observed=True)
        .agg(
            score_mean=(score_col, "mean"),
            accuracy=(correctness_col, "mean"),
            count=(correctness_col, "size"),
        )
        .reset_index(drop=True)
    )


def plot_binned_accuracy(bin_df, out_path, title):
    plt.figure(figsize=(5.2, 3.8))
    plt.plot(bin_df["score_mean"], bin_df["accuracy"], marker="o")
    plt.xlabel("Local geometry score")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_disagreement_split(df, out_path, title):
    median_disagreement = df["teacher_disagreement"].median()

    low = df[df["teacher_disagreement"] <= median_disagreement]
    high = df[df["teacher_disagreement"] > median_disagreement]

    low_bins = bin_stats(low, n_bins=6)
    high_bins = bin_stats(high, n_bins=6)

    plt.figure(figsize=(5.6, 3.8))

    if len(low_bins) > 1:
        plt.plot(
            low_bins["score_mean"],
            low_bins["accuracy"],
            marker="o",
            label="Low teacher disagreement",
        )

    if len(high_bins) > 1:
        plt.plot(
            high_bins["score_mean"],
            high_bins["accuracy"],
            marker="s",
            label="High teacher disagreement",
        )

    plt.xlabel("Local geometry score")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_geometry_analysis(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    source_split = args.source_split
    target_split = args.target_split

    source_labels = load_labels(args.feature_root, args.dataset, args.source, source_split)
    target_labels = load_labels(args.feature_root, args.dataset, args.target, target_split)

    teacher_probs = []
    target_feature_blocks = []

    print(f"\nDataset: {args.dataset}")
    print(f"Source: {args.source} / split={source_split}")
    print(f"Target: {args.target} / split={target_split}")
    print(f"Teachers: {len(args.teachers)}")

    for teacher in args.teachers:
        print(f"Processing teacher: {teacher}")

        x_source = load_features(
            args.feature_root, args.dataset, args.source, teacher, source_split
        )
        x_target = load_features(
            args.feature_root, args.dataset, args.target, teacher, target_split
        )

        if x_source.shape[0] != source_labels.shape[0]:
            raise ValueError(
                f"Source feature/label mismatch for {teacher}: "
                f"{x_source.shape[0]} features vs {source_labels.shape[0]} labels"
            )

        if x_target.shape[0] != target_labels.shape[0]:
            raise ValueError(
                f"Target feature/label mismatch for {teacher}: "
                f"{x_target.shape[0]} features vs {target_labels.shape[0]} labels"
            )

        scaler, clf = train_probe(x_source, source_labels, seed=args.seed)
        probs = predict_probs(x_target, scaler, clf)

        teacher_probs.append(probs)
        target_feature_blocks.append(normalize_rows(x_target))

    teacher_probs = np.stack(teacher_probs, axis=1)  # [N, M, C]
    concat_features = normalize_rows(np.concatenate(target_feature_blocks, axis=1))

    anchor_probs = compute_anchor_probs(teacher_probs, mode=args.anchor)
    anchor_preds = anchor_probs.argmax(axis=1)
    correct = (anchor_preds == target_labels).astype(float)

    knn = compute_knn(concat_features, k=args.k)

    local_purity = compute_local_purity(anchor_preds, knn)
    confidence = anchor_probs.max(axis=1)
    anchor_entropy = entropy(anchor_probs)
    teacher_disagreement = compute_teacher_disagreement(teacher_probs)

    df = pd.DataFrame(
        {
            "dataset": args.dataset,
            "source": args.source,
            "target": args.target,
            "source_split": source_split,
            "target_split": target_split,
            "label": target_labels,
            "anchor_pred": anchor_preds,
            "correct": correct,
            "confidence": confidence,
            "anchor_entropy": anchor_entropy,
            "local_purity": local_purity,
            "teacher_disagreement": teacher_disagreement,
        }
    )

    spearman = spearmanr(df["local_purity"], df["correct"])
    pearson = pearsonr(df["local_purity"], df["correct"])

    summary = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "source": args.source,
                "target": args.target,
                "source_split": source_split,
                "target_split": target_split,
                "n": len(df),
                "anchor": args.anchor,
                "k": args.k,
                "anchor_accuracy": df["correct"].mean(),
                "local_purity_mean": df["local_purity"].mean(),
                "teacher_disagreement_mean": df["teacher_disagreement"].mean(),
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
            }
        ]
    )

    bin_df = bin_stats(df, n_bins=args.bins)

    prefix = f"{args.dataset}_{args.source}_{source_split}_to_{args.target}_{target_split}"

    df.to_csv(outdir / f"{prefix}_per_sample_geometry.csv", index=False)
    summary.to_csv(outdir / f"{prefix}_geometry_summary.csv", index=False)
    bin_df.to_csv(outdir / f"{prefix}_geometry_bins.csv", index=False)

    plot_binned_accuracy(
        bin_df,
        outdir / f"{prefix}_geometry_vs_accuracy.png",
        title=f"{args.dataset}: {args.source} → {args.target}",
    )

    plot_disagreement_split(
        df,
        outdir / f"{prefix}_geometry_disagreement_split.png",
        title=f"{args.dataset}: geometry under teacher disagreement",
    )

    if args.save_arrays:
        np.save(outdir / f"{prefix}_teacher_probs.npy", teacher_probs)
        np.save(outdir / f"{prefix}_concat_features.npy", concat_features)
        np.save(outdir / f"{prefix}_labels.npy", target_labels)

    print("\nSaved outputs to:", outdir)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--probe_root", default=None)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)

    parser.add_argument("--source_split", default="train")
    parser.add_argument("--target_split", default="test")

    parser.add_argument("--teachers", nargs="+", required=True)

    parser.add_argument("--anchor", choices=["uniform", "agreement"], default="agreement")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_arrays", action="store_true")

    args = parser.parse_args()
    run_geometry_analysis(args)


if __name__ == "__main__":
    main()