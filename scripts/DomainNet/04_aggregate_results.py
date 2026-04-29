#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {results_dir}")

    # ------------------------
    # Load all runs
    # ------------------------
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # ------------------------
    # Save raw (important)
    # ------------------------
    df_all.to_csv(outdir / "main_results_raw.csv", index=False)

    # ------------------------
    # Aggregate: mean ± std
    # ------------------------
    df_summary = (
        df_all
        .groupby(["dataset", "source", "target", "method"])
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    df_summary.to_csv(outdir / "main_results_summary.csv", index=False)

    # ------------------------
    # Pretty print (paper view)
    # ------------------------
    print("\n=== SUMMARY (mean ± std) ===\n")

    for target in sorted(df_summary["target"].unique()):
        print(f"\nTarget: {target}")

        sub = df_summary[df_summary["target"] == target]

        for _, row in sub.iterrows():
            print(
                f"{row['method']:25s} "
                f"{row['acc_mean']:.3f} ± {row['acc_std']:.3f}"
            )

    print("\nSaved:")
    print(outdir / "main_results_raw.csv")
    print(outdir / "main_results_summary.csv")


if __name__ == "__main__":
    main()