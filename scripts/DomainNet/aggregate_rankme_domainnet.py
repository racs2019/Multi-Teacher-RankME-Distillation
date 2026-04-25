#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLS = {
    "train_domain",
    "target_domain",
    "accuracy",
    "balanced_accuracy",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    rows = []

    target_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not target_dirs:
        raise RuntimeError(f"No target subdirectories found under: {base_dir}")

    for target_dir in target_dirs:
        csv_files = sorted(target_dir.glob("rankme_result_*.csv"))

        if len(csv_files) == 0:
            print(f"Skipping {target_dir}, found 0 result files")
            continue
        if len(csv_files) > 1:
            print(f"Skipping {target_dir}, found {len(csv_files)} result files")
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

        row = df.iloc[0]
        rows.append({
            "train_domain": row["train_domain"],
            "target_domain": row["target_domain"],
            "accuracy": float(row["accuracy"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "source_csv": str(csv_path),
        })

    df_all = pd.DataFrame(rows)

    if df_all.empty:
        raise RuntimeError("No valid RankMe results found.")

    df_all = df_all.sort_values("target_domain").reset_index(drop=True)

    df_avg = pd.DataFrame([{
        "train_domain": df_all["train_domain"].iloc[0],
        "n_targets": df_all["target_domain"].nunique(),
        "accuracy": df_all["accuracy"].mean(),
        "balanced_accuracy": df_all["balanced_accuracy"].mean(),
    }])

    print("\n=== Per-domain results ===")
    print(df_all[["train_domain", "target_domain", "accuracy", "balanced_accuracy"]].to_string(index=False))

    print("\n=== Average across targets ===")
    print(df_avg.to_string(index=False))

    out_path = base_dir / "rankme_summary.csv"
    out_avg_path = base_dir / "rankme_summary_avg.csv"

    df_all.to_csv(out_path, index=False)
    df_avg.to_csv(out_avg_path, index=False)

    print("\nSaved to:")
    print(out_path)
    print(out_avg_path)


if __name__ == "__main__":
    main()

# python scripts\DomainNet\aggregate_rankme_domainnet.py `
#   --base_dir "C:\Users\racs2019\Documents\NIPS-KD\domainnet_probe_results\rankme_variants\quickdraw"
