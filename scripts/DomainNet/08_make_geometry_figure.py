import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def normalize_rows(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def load_npz(path):
    data = np.load(path, allow_pickle=True)

    x = None
    y = None

    for k in ["features", "feats", "embeddings", "x", "X"]:
        if k in data:
            x = data[k]
            break

    for k in ["labels", "y", "targets", "target"]:
        if k in data:
            y = data[k]
            break

    if x is None:
        raise KeyError(f"No feature array found in {path}. Keys={list(data.keys())}")
    if y is None:
        raise KeyError(f"No label array found in {path}. Keys={list(data.keys())}")

    return x.astype(np.float32), y.astype(np.int64)


def teacher_path(feature_root, domain, split, teacher):
    path = Path(feature_root) / domain / split / f"{teacher}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path


def train_probe(x_train, y_train, seed=0, n_jobs=8):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    clf = LogisticRegression(
        max_iter=2000,
        random_state=seed,
        n_jobs=n_jobs,
    )
    clf.fit(x_train_scaled, y_train)
    return scaler, clf


def agreement_anchor(probs):
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
    scores -= scores.max(axis=1, keepdims=True)

    weights = np.exp(scores)
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)

    anchor = (weights[:, :, None] * probs).sum(axis=1)
    anchor /= np.clip(anchor.sum(axis=1, keepdims=True), 1e-12, None)
    return anchor.astype(np.float32)


def compute_knn(features, k=20):
    features = normalize_rows(features)
    dists = pairwise_distances(features, metric="cosine")
    np.fill_diagonal(dists, np.inf)
    return np.argsort(dists, axis=1)[:, :k]


def compute_teacher_disagreement(probs):
    teacher_preds = probs.argmax(axis=2)
    n, m = teacher_preds.shape
    disagreement = np.zeros(n, dtype=np.float32)

    for i in range(n):
        _, counts = np.unique(teacher_preds[i], return_counts=True)
        disagreement[i] = 1.0 - counts.max() / m

    return disagreement


def make_bins(df, score_col="local_purity", n_bins=10):
    tmp = df.copy()

    try:
        tmp["bin"] = pd.qcut(tmp[score_col], q=n_bins, duplicates="drop")
    except ValueError:
        tmp["bin"] = pd.cut(tmp[score_col], bins=min(n_bins, 5), duplicates="drop")

    return (
        tmp.groupby("bin", observed=True)
        .agg(
            score_mean=(score_col, "mean"),
            accuracy=("correct", "mean"),
            count=("correct", "size"),
        )
        .reset_index(drop=True)
    )


def make_accuracy_heatmap(df, n_bins=3):
    tmp = df.copy()

    tmp["geometry_bin"] = pd.qcut(
        tmp["local_purity"], q=n_bins, labels=False, duplicates="drop"
    )

    tmp["disagreement_bin"] = pd.qcut(
        tmp["teacher_disagreement"], q=n_bins, labels=False, duplicates="drop"
    )

    heat = (
        tmp.groupby(["disagreement_bin", "geometry_bin"], observed=True)["correct"]
        .mean()
        .unstack()
    )

    return heat


def compute_pair_dataframe(args, source, target):
    source_labels = None
    target_labels = None

    teacher_probs = []
    target_feature_blocks = []

    print(f"\n=== Pair: {source} -> {target} ===")

    for teacher in args.teachers:
        print(f"Processing teacher: {teacher}")

        x_source, y_source = load_npz(
            teacher_path(args.feature_root, source, args.source_split, teacher)
        )
        x_target, y_target = load_npz(
            teacher_path(args.feature_root, target, args.target_split, teacher)
        )

        if source_labels is None:
            source_labels = y_source
        elif not np.array_equal(source_labels, y_source):
            raise ValueError(f"Source label mismatch for teacher={teacher}")

        if target_labels is None:
            target_labels = y_target
        elif not np.array_equal(target_labels, y_target):
            raise ValueError(f"Target label mismatch for teacher={teacher}")

        scaler, clf = train_probe(
            x_source,
            source_labels,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        probs = clf.predict_proba(scaler.transform(x_target)).astype(np.float32)

        teacher_probs.append(probs)
        target_feature_blocks.append(normalize_rows(x_target))

    teacher_probs = np.stack(teacher_probs, axis=1)  # [N, M, C]
    concat_features = normalize_rows(np.concatenate(target_feature_blocks, axis=1))

    anchor_probs = agreement_anchor(teacher_probs)
    anchor_pred = anchor_probs.argmax(axis=1)
    correct = (anchor_pred == target_labels).astype(float)

    knn = compute_knn(concat_features, k=args.k)

    local_purity = np.zeros(len(anchor_pred), dtype=np.float32)
    for i in range(len(anchor_pred)):
        local_purity[i] = np.mean(anchor_pred[knn[i]] == anchor_pred[i])

    teacher_disagreement = compute_teacher_disagreement(teacher_probs)

    df = pd.DataFrame(
        {
            "dataset": args.dataset,
            "source": source,
            "target": target,
            "local_purity": local_purity,
            "correct": correct,
            "teacher_disagreement": teacher_disagreement,
            "anchor_confidence": anchor_probs.max(axis=1),
            "anchor_pred": anchor_pred,
            "label": target_labels,
        }
    )

    return df


def summarize_pair(df):
    rho, rho_p = spearmanr(df["local_purity"], df["correct"])
    r, r_p = pearsonr(df["local_purity"], df["correct"])

    correct_geom = df.loc[df["correct"] == 1, "local_purity"]
    incorrect_geom = df.loc[df["correct"] == 0, "local_purity"]

    return {
        "dataset": df["dataset"].iloc[0],
        "source": df["source"].iloc[0],
        "target": df["target"].iloc[0],
        "n": len(df),
        "anchor_accuracy": df["correct"].mean(),
        "local_purity_mean": df["local_purity"].mean(),
        "correct_local_purity_mean": correct_geom.mean(),
        "incorrect_local_purity_mean": incorrect_geom.mean(),
        "teacher_disagreement_mean": df["teacher_disagreement"].mean(),
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "pearson_r": r,
        "pearson_p": r_p,
    }


def plot_single_pair_figure(df, summary, args, outdir):
    source = summary["source"]
    target = summary["target"]

    rho = summary["spearman_rho"]
    r = summary["pearson_r"]

    bins = make_bins(df, n_bins=args.bins)

    median_disagreement = df["teacher_disagreement"].median()
    low = df[df["teacher_disagreement"] <= median_disagreement]
    high = df[df["teacher_disagreement"] > median_disagreement]

    low_bins = make_bins(low, n_bins=4)
    high_bins = make_bins(high, n_bins=4)

    heat = make_accuracy_heatmap(df, n_bins=3)

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.8))
    ax_a, ax_b, ax_c, ax_d = axs.ravel()

    ax_a.plot(bins["score_mean"], bins["accuracy"], linewidth=2)
    ax_a.scatter(bins["score_mean"], bins["accuracy"], s=35)
    ax_a.set_title("(a) Accuracy increases with local geometry")
    ax_a.set_xlabel("Local predictive geometry score")
    ax_a.set_ylabel("Accuracy")
    ax_a.grid(True, alpha=0.3)
    ax_a.text(
        0.95,
        0.05,
        rf"Spearman $\rho={rho:.2f}$" + "\n" + rf"Pearson $r={r:.2f}$",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    if len(low_bins) > 1:
        ax_b.plot(
            low_bins["score_mean"],
            low_bins["accuracy"],
            marker="o",
            linewidth=2,
            label="Low disagreement",
        )

    if len(high_bins) > 1:
        ax_b.plot(
            high_bins["score_mean"],
            high_bins["accuracy"],
            marker="s",
            linewidth=2,
            label="High disagreement",
        )

    ax_b.set_title("(b) Geometry under teacher disagreement")
    ax_b.set_xlabel("Local predictive geometry score")
    ax_b.set_ylabel("Accuracy")
    ax_b.legend(frameon=False)
    ax_b.grid(True, alpha=0.3)

    correct_geom = df.loc[df["correct"] == 1, "local_purity"]
    incorrect_geom = df.loc[df["correct"] == 0, "local_purity"]

    common_bins = np.linspace(0, 1, 30)

    ax_c.hist(
        incorrect_geom,
        bins=common_bins,
        alpha=0.55,
        density=True,
        label="Incorrect",
    )
    ax_c.hist(
        correct_geom,
        bins=common_bins,
        alpha=0.55,
        density=True,
        label="Correct",
    )
    ax_c.axvline(
        incorrect_geom.mean(),
        linestyle="--",
        linewidth=1.5,
        label="Incorrect mean",
    )
    ax_c.axvline(
        correct_geom.mean(),
        linestyle="-",
        linewidth=1.5,
        label="Correct mean",
    )
    ax_c.set_title("(c) Correct samples have higher geometry")
    ax_c.set_xlabel("Local predictive geometry score")
    ax_c.set_ylabel("Density")
    ax_c.legend(frameon=False)
    ax_c.grid(True, alpha=0.3)

    im = ax_d.imshow(
        heat.values,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=np.nanmax(heat.values),
    )

    ax_d.set_title("(d) Accuracy by geometry/disagreement regime")
    ax_d.set_xlabel("Local geometry bin")
    ax_d.set_ylabel("Teacher disagreement bin")
    ax_d.set_xticks(range(heat.shape[1]))
    ax_d.set_yticks(range(heat.shape[0]))

    xlabels = ["low", "mid", "high"][: heat.shape[1]]
    ylabels = ["low", "mid", "high"][: heat.shape[0]]

    ax_d.set_xticklabels(xlabels)
    ax_d.set_yticklabels(ylabels)

    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy")

    fig.suptitle(
        f"Local predictive geometry predicts reliability "
        f"({args.dataset}: {source} $\\rightarrow$ {target})",
        fontsize=12,
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.965])

    prefix = f"{args.dataset}_{source}_{args.source_split}_to_{target}_{args.target_split}"
    fig_path = outdir / f"{prefix}_geometry_figure.png"

    fig.savefig(fig_path, dpi=300)
    fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)

    bins.to_csv(outdir / f"{prefix}_geometry_bins.csv", index=False)
    heat.to_csv(outdir / f"{prefix}_geometry_disagreement_heatmap.csv")
    df.to_csv(outdir / f"{prefix}_geometry_per_sample.csv", index=False)

    print(f"Saved single-pair figure: {fig_path}")


def plot_multi_pair_summary(all_summaries, args, outdir):
    summary_df = pd.DataFrame(all_summaries).copy()
    summary_df["pair"] = summary_df["source"] + r"$\rightarrow$" + summary_df["target"]

    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))

    ax0, ax1, ax2 = axs

    ax0.bar(summary_df["pair"], summary_df["spearman_rho"])
    ax0.set_title("(a) Geometry-correctness correlation")
    ax0.set_ylabel(r"Spearman $\rho$")
    ax0.tick_params(axis="x", rotation=35)
    ax0.axhline(0, linewidth=1)
    ax0.grid(True, axis="y", alpha=0.3)

    ax1.bar(summary_df["pair"], summary_df["pearson_r"])
    ax1.set_title("(b) Linear association")
    ax1.set_ylabel(r"Pearson $r$")
    ax1.tick_params(axis="x", rotation=35)
    ax1.axhline(0, linewidth=1)
    ax1.grid(True, axis="y", alpha=0.3)

    width = 0.38
    x = np.arange(len(summary_df))
    ax2.bar(
        x - width / 2,
        summary_df["incorrect_local_purity_mean"],
        width,
        label="Incorrect",
    )
    ax2.bar(
        x + width / 2,
        summary_df["correct_local_purity_mean"],
        width,
        label="Correct",
    )
    ax2.set_title("(c) Geometry separates correct/incorrect")
    ax2.set_ylabel("Mean local geometry")
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df["pair"], rotation=35, ha="right")
    ax2.legend(frameon=False)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Local predictive geometry is a reliability signal across domain pairs", y=1.04)
    plt.tight_layout()

    fig_path = outdir / args.multi_figure_name
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved multi-pair summary figure: {fig_path}")


def parse_pair(pair_string):
    if ":" not in pair_string:
        raise ValueError(f"Pair must be source:target, got {pair_string}")
    source, target = pair_string.split(":", 1)
    return source.strip(), target.strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--dataset", default="domainnet")
    parser.add_argument("--pairs", nargs="+", required=True)

    parser.add_argument("--source_split", default="train")
    parser.add_argument("--target_split", default="test")

    parser.add_argument("--teachers", nargs="+", required=True)

    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=8)

    parser.add_argument("--make_single_pair_figures", action="store_true")
    parser.add_argument("--multi_figure_name", default="fig_local_geometry_multi.png")

    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for pair in args.pairs:
        source, target = parse_pair(pair)

        df = compute_pair_dataframe(args, source, target)
        summary = summarize_pair(df)
        all_summaries.append(summary)

        prefix = f"{args.dataset}_{source}_{args.source_split}_to_{target}_{args.target_split}"
        df.to_csv(outdir / f"{prefix}_geometry_per_sample.csv", index=False)

        if args.make_single_pair_figures:
            plot_single_pair_figure(df, summary, args, outdir)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = outdir / "geometry_pair_summaries.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_multi_pair_summary(all_summaries, args, outdir)

    print("\nSaved summary:")
    print(summary_path)
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()