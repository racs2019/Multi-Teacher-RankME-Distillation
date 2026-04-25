#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Pretty names
# ============================================================

TEACHER_NAME_MAP = {
    "openclip_l14_openai_qgelu": "CLIP L/14 OpenAI",
    "openclip_b16_datacomp": "CLIP B/16 DataComp",
    "openclip_so400m_siglip": "SigLIP SO400M",
    "openclip_l14_dfn2b": "DFN L/14 2B",
    "openclip_h14_laion2b": "CLIP H/14 LAION",
    "openclip_h14_378_dfn5b": "DFN H/14 378",
    "openclip_convnext_xxlarge": "ConvNeXt XXL",
}


# ============================================================
# Loading
# ============================================================

def load_probe_results(root: Path, dataset_name: str) -> pd.DataFrame:
    """
    Loads all linear_probe_metrics.json files into a tidy dataframe.

    Output columns:
      train_domain, test_domain, teacher, teacher_pretty,
      split, accuracy, balanced_accuracy, n_samples
    """
    dataset_dir = root / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    rows = []

    for train_domain_dir in sorted(dataset_dir.iterdir()):
        if not train_domain_dir.is_dir():
            continue

        train_domain = train_domain_dir.name

        for teacher_dir in sorted(train_domain_dir.iterdir()):
            if not teacher_dir.is_dir():
                continue

            teacher = teacher_dir.name
            metrics_path = teacher_dir / "linear_probe_metrics.json"
            if not metrics_path.exists():
                continue

            with open(metrics_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            domains = obj.get("domains", {})
            for test_domain, vals in domains.items():
                rows.append({
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    "teacher": teacher,
                    "teacher_pretty": TEACHER_NAME_MAP.get(teacher, teacher),
                    "split": vals.get("split", ""),
                    "accuracy": vals.get("accuracy", np.nan),
                    "balanced_accuracy": vals.get("balanced_accuracy", np.nan),
                    "n_samples": vals.get("n_samples", np.nan),
                })

    if not rows:
        raise RuntimeError(
            f"No linear_probe_metrics.json files found under {dataset_dir}"
        )

    df = pd.DataFrame(rows)
    return df


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path):
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"saved: {path}")


def sort_teachers_by_mean(df: pd.DataFrame, metric: str) -> List[str]:
    order = (
        df.groupby("teacher_pretty")[metric]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return order


def sort_domains(domains: List[str]) -> List[str]:
    """
    Stable domain ordering:
    - if names contain digits (e.g. location_38), sort numerically
    - otherwise sort alphabetically
    """
    def key_fn(x: str):
        digits = "".join(ch for ch in x if ch.isdigit())
        return (0, int(digits)) if digits else (1, x)

    return sorted(domains, key=key_fn)


# ============================================================
# Ranking instability helpers
# ============================================================

def average_rank_correlation(rank_a: List[str], rank_b: List[str]) -> float:
    if len(rank_a) != len(rank_b):
        raise ValueError("Rank lists must have same length")

    pos_a = {name: i for i, name in enumerate(rank_a)}
    pos_b = {name: i for i, name in enumerate(rank_b)}

    names = rank_a
    ra = np.array([pos_a[n] for n in names], dtype=np.float64)
    rb = np.array([pos_b[n] for n in names], dtype=np.float64)

    ra = ra - ra.mean()
    rb = rb - rb.mean()

    denom = np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())
    if denom == 0:
        return 1.0
    return float((ra * rb).sum() / denom)


def pairwise_order_flip_fraction(accs_a: Dict[str, float], accs_b: Dict[str, float]) -> float:
    names = sorted(accs_a.keys())
    total = 0
    flips = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t1, t2 = names[i], names[j]
            da = accs_a[t1] - accs_a[t2]
            db = accs_b[t1] - accs_b[t2]

            if da == 0 or db == 0:
                continue

            total += 1
            if np.sign(da) != np.sign(db):
                flips += 1

    if total == 0:
        return 0.0
    return float(flips / total)


def compute_instability_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Computes rank correlation and flip fraction across test-domain pairs.
    Expects df for a single train_domain.
    """
    train_domains = sorted(df["train_domain"].unique().tolist())
    if len(train_domains) != 1:
        raise ValueError(
            f"compute_instability_table expects one train_domain, got {train_domains}"
        )

    train_domain = train_domains[0]
    test_domains = sort_domains(df["test_domain"].unique().tolist())

    acc_map: Dict[str, Dict[str, float]] = {}
    ranking_map: Dict[str, List[str]] = {}

    for test_domain in test_domains:
        sub = df[df["test_domain"] == test_domain].copy()
        sub = sub.sort_values([metric, "teacher_pretty"], ascending=[False, True])

        acc_map[test_domain] = {
            str(r["teacher_pretty"]): float(r[metric])
            for _, r in sub.iterrows()
        }

        ranking_map[test_domain] = sub["teacher_pretty"].tolist()

    rows = []
    for d1, d2 in combinations(test_domains, 2):
        rows.append({
            "train_domain": train_domain,
            "target_domain_a": d1,
            "target_domain_b": d2,
            "winner_a": ranking_map[d1][0],
            "winner_b": ranking_map[d2][0],
            "same_winner": int(ranking_map[d1][0] == ranking_map[d2][0]),
            "winner_changed": int(ranking_map[d1][0] != ranking_map[d2][0]),
            "rank_corr": average_rank_correlation(ranking_map[d1], ranking_map[d2]),
            "flip_fraction": pairwise_order_flip_fraction(acc_map[d1], acc_map[d2]),
        })

    out = pd.DataFrame(rows).sort_values(["target_domain_a", "target_domain_b"]).reset_index(drop=True)
    return out


def write_instability_latex_table(
    instability_df: pd.DataFrame,
    out_path: Path,
    caption: str,
    label: str,
) -> None:
    ensure_dir(out_path.parent)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append(r"    \begin{tabular}{lcc}")
    lines.append(r"        \toprule")
    lines.append(r"        Domain Pair & Rank Correlation & Flip Fraction \\")
    lines.append(r"        \midrule")

    for _, row in instability_df.iterrows():
        pair = f"{row['target_domain_a']} vs {row['target_domain_b']}"
        rank_corr = f"{row['rank_corr']:.3f}"
        flip_frac = f"{row['flip_fraction']:.3f}"
        lines.append(f"        {pair} & {rank_corr} & {flip_frac} \\\\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"saved: {out_path}")


# ============================================================
# Plotting
# ============================================================

def plot_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    out_path: Path,
    value_fmt: str = ".3f",
):
    ensure_dir(out_path.parent)

    values = pivot_df.values
    row_labels = list(pivot_df.index)
    col_labels = list(pivot_df.columns)

    fig_w = max(7.0, 1.2 * len(col_labels) + 2.5)
    fig_h = max(4.0, 0.45 * len(row_labels) + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Test Domain")
    ax.set_ylabel("Teacher")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            text = "NA" if np.isnan(val) else format(val, value_fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Score")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_best_teacher_bar(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
):
    ensure_dir(out_path.parent)

    grouped = (
        df.sort_values(metric, ascending=False)
          .groupby(["train_domain", "test_domain"], as_index=False)
          .first()
    )

    grouped["pair"] = grouped["train_domain"] + " → " + grouped["test_domain"]

    fig_w = max(8.0, 0.9 * len(grouped))
    fig_h = 5.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(len(grouped))
    y = grouped[metric].values

    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["pair"], rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Best Teacher by Train/Test Domain Pair ({metric})")

    for i, (_, row) in enumerate(grouped.iterrows()):
        ax.text(
            i,
            row[metric] + 0.005,
            row["teacher_pretty"],
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_rank_switches(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
):
    ensure_dir(out_path.parent)

    train_domains = sort_domains(df["train_domain"].unique().tolist())
    test_domains = sort_domains(df["test_domain"].unique().tolist())

    n_panels = len(train_domains)
    fig_w = max(8.0, 3.6 * n_panels)
    fig_h = 5.5
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    for ax, train_domain in zip(axes, train_domains):
        sub = df[df["train_domain"] == train_domain].copy()

        rank_rows = []
        for test_domain in test_domains:
            s = sub[sub["test_domain"] == test_domain].sort_values(metric, ascending=False)
            for rank_idx, (_, row) in enumerate(s.iterrows(), start=1):
                rank_rows.append({
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    "teacher_pretty": row["teacher_pretty"],
                    "rank": rank_idx,
                })

        rank_df = pd.DataFrame(rank_rows)
        if rank_df.empty:
            ax.set_visible(False)
            continue

        for teacher_name, g in rank_df.groupby("teacher_pretty"):
            g = g.set_index("test_domain").reindex(test_domains).reset_index()
            ax.plot(
                np.arange(len(test_domains)),
                g["rank"].values,
                marker="o",
                label=teacher_name,
            )

        ax.set_xticks(np.arange(len(test_domains)))
        ax.set_xticklabels(test_domains, rotation=30, ha="right")
        ax.set_yticks(np.arange(1, len(rank_df["teacher_pretty"].unique()) + 1))
        ax.invert_yaxis()
        ax.set_title(f"Train: {train_domain}")
        ax.set_xlabel("Test Domain")
        ax.set_ylabel("Teacher Rank")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.suptitle(f"Teacher Rank Switching Across Test Domains ({metric})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ============================================================
# Main plotting logic
# ============================================================

def make_train_domain_heatmaps(
    df: pd.DataFrame,
    metric: str,
    outdir: Path,
):
    teacher_order = sort_teachers_by_mean(df, metric)
    train_domains = sort_domains(df["train_domain"].unique().tolist())
    test_domains = sort_domains(df["test_domain"].unique().tolist())

    for train_domain in train_domains:
        sub = df[df["train_domain"] == train_domain].copy()

        pivot = sub.pivot_table(
            index="teacher_pretty",
            columns="test_domain",
            values=metric,
            aggfunc="mean",
        )

        pivot = pivot.reindex(index=teacher_order)
        pivot = pivot.reindex(columns=test_domains)

        plot_heatmap(
            pivot_df=pivot,
            title=f"{metric.replace('_', ' ').title()} | Train Domain = {train_domain}",
            out_path=outdir / f"heatmap_{metric}_train_{train_domain}.png",
        )


def make_global_heatmap(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
):
    tmp = df.copy()
    tmp["pair"] = tmp["train_domain"] + " → " + tmp["test_domain"]

    teacher_order = sort_teachers_by_mean(df, metric)

    pair_order_df = (
        tmp[["train_domain", "test_domain", "pair"]]
        .drop_duplicates()
        .copy()
    )
    pair_order_df["train_domain_sort"] = pair_order_df["train_domain"].map(
        {d: i for i, d in enumerate(sort_domains(pair_order_df["train_domain"].unique().tolist()))}
    )
    pair_order_df["test_domain_sort"] = pair_order_df["test_domain"].map(
        {d: i for i, d in enumerate(sort_domains(pair_order_df["test_domain"].unique().tolist()))}
    )
    pair_order_df = pair_order_df.sort_values(["train_domain_sort", "test_domain_sort"])
    pair_order = pair_order_df["pair"].tolist()

    pivot = tmp.pivot_table(
        index="teacher_pretty",
        columns="pair",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=teacher_order)
    pivot = pivot.reindex(columns=pair_order)

    plot_heatmap(
        pivot_df=pivot,
        title=f"All Train/Test Domain Pairs ({metric})",
        out_path=out_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot DomainNet linear-probe results and compute ranking instability.")
    parser.add_argument("--root", type=str, required=True, help="Root output dir from probe script.")
    parser.add_argument("--dataset_name", type=str, default="domainnet")
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "balanced_accuracy"],
    )
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_probe_results(root=root, dataset_name=args.dataset_name)
    save_dataframe(df, outdir / "linear_probe_results_tidy.csv")

    make_train_domain_heatmaps(
        df=df,
        metric=args.metric,
        outdir=outdir / "per_train_domain_heatmaps",
    )

    make_global_heatmap(
        df=df,
        metric=args.metric,
        out_path=outdir / f"global_heatmap_{args.metric}.png",
    )

    plot_best_teacher_bar(
        df=df,
        metric=args.metric,
        out_path=outdir / f"best_teacher_bar_{args.metric}.png",
    )

    plot_rank_switches(
        df=df,
        metric=args.metric,
        out_path=outdir / f"rank_switches_{args.metric}.png",
    )

    best_df = (
        df.sort_values(args.metric, ascending=False)
          .groupby(["train_domain", "test_domain"], as_index=False)
          .first()
          .sort_values(["train_domain", "test_domain"])
    )
    save_dataframe(best_df, outdir / f"best_teacher_by_pair_{args.metric}.csv")

    # New: ranking instability per train_domain
    for train_domain in sort_domains(df["train_domain"].unique().tolist()):
        sub = df[df["train_domain"] == train_domain].copy()
        instability_df = compute_instability_table(sub, args.metric)

        save_dataframe(
            instability_df,
            outdir / f"ranking_instability_train_{train_domain}_{args.metric}.csv",
        )

        write_instability_latex_table(
            instability_df=instability_df,
            out_path=outdir / f"ranking_instability_train_{train_domain}_{args.metric}.tex",
            caption=f"Teacher ranking instability across target domains (train domain = {train_domain}).",
            label=f"tab:instability_{train_domain}_{args.metric}",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

# python scripts/DomainNet/plot_linear_probe_results_domainnet.py `
#   --root "quickdraw" `
#   --dataset_name "domainnet" `
#   --metric accuracy `
#   --outdir "domainnet_probe_plots"