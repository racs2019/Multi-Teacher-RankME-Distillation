#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["uniform", "graph", "grace"]

METHOD_LABELS = {
    "uniform": "Anchor ensemble",
    "graph": "Graph propagation",
    "grace": "GRACE",
}


def load_stress_results(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(results_dir.glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV files found in: {results_dir}")

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    required = {
        "dataset",
        "source",
        "target",
        "seed",
        "corruption_rate",
        "method",
        "accuracy",
        "macro_f1",
        "n_samples",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stress results missing columns: {sorted(missing)}")

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "source", "target", "corruption_rate", "method"])
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )


def draw_panel(ax, sub: pd.DataFrame, target: str):
    for method in METHOD_ORDER:
        m = sub[sub["method"] == method].sort_values("corruption_rate")
        if m.empty:
            continue

        x = m["corruption_rate"].to_numpy() * 100.0
        y = m["acc_mean"].to_numpy()
        yerr = m["acc_std"].fillna(0.0).to_numpy()

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.3,
            label=METHOD_LABELS.get(method, method),
        )
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            alpha=0.14,
        )

    ax.set_title(target, fontweight="bold")
    ax.set_xlabel("Corrupted graph seeds (%)")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="stress_results")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    df = load_stress_results(Path(args.results_dir))

    if args.dataset is not None:
        df = df[df["dataset"] == args.dataset]
    if args.source is not None:
        df = df[df["source"] == args.source]

    if df.empty:
        raise RuntimeError("No stress rows left after filtering.")

    summary = summarize(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(outdir / "stress_test_summary.csv", index=False)

    targets = sorted(summary["target"].unique().tolist())
    n = len(targets)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5.2 * n, 4.2),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, target in zip(axes, targets):
        sub = summary[summary["target"] == target]
        draw_panel(ax, sub, target)

    axes[0].set_ylabel("Accuracy")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        "GRACE remains stable when graph pseudo-label seeds are corrupted",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    png = outdir / "figure_stress_test.png"
    pdf = outdir / "figure_stress_test.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print("Saved:", png)
    print("Saved:", pdf)
    print("Saved:", outdir / "stress_test_summary.csv")


if __name__ == "__main__":
    main()