#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLS = {
    "train_domain",
    "target_domain",
    "method",
    "accuracy",
    "balanced_accuracy",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument(
        "--csv_glob",
        type=str,
        default="*.csv",
        help="Glob pattern for per-target result CSVs inside each target subdirectory.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="aggregated_variants",
        help="Prefix for saved aggregate CSVs.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    rows = []

    target_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not target_dirs:
        raise RuntimeError(f"No target subdirectories found under: {base_dir}")

    for target_dir in target_dirs:
        csv_files = sorted(target_dir.glob(args.csv_glob))

        if len(csv_files) == 0:
            print(f"Skipping {target_dir}, found 0 matching result files for glob: {args.csv_glob}")
            continue
        if len(csv_files) > 1:
            print(f"Skipping {target_dir}, found {len(csv_files)} matching result files for glob: {args.csv_glob}")
            continue

        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"Skipping empty file: {csv_path}")
            continue

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(f"Skipping {csv_path}, missing columns: {sorted(missing)}")
            continue

        df = df.copy()
        df["source_csv"] = str(csv_path)
        rows.append(df)

    if not rows:
        raise RuntimeError("No valid variant results found.")

    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all.sort_values(["method", "target_domain"]).reset_index(drop=True)

    df_avg = (
        df_all.groupby("method", as_index=False)
        .agg(
            train_domain=("train_domain", "first"),
            n_targets=("target_domain", "nunique"),
            accuracy=("accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== Per-domain results ===")
    print(
        df_all[
            ["train_domain", "target_domain", "method", "accuracy", "balanced_accuracy"]
        ].to_string(index=False)
    )

    print("\n=== Average across targets by method ===")
    print(df_avg.to_string(index=False))

    out_path = base_dir / f"{args.out_prefix}_summary.csv"
    out_avg_path = base_dir / f"{args.out_prefix}_summary_avg.csv"

    df_all.to_csv(out_path, index=False)
    df_avg.to_csv(out_avg_path, index=False)

    print("\nSaved to:")
    print(out_path)
    print(out_avg_path)


if __name__ == "__main__":
    main()

# python scripts\DomainNet\aggregate_variants.py `
#   --base_dir "C:\Users\racs2019\Documents\NIPS-KD\domainnet_probe_results\multiscale_adaptive\quickdraw" `
#   --csv_glob "multiscale_adaptive*.csv" `
#   --out_prefix "multiscale_adaptive"