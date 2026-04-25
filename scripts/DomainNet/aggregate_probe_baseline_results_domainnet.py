#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# Ranking helpers
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


# ============================================================
# IO
# ============================================================

def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-target DomainNet probe baseline outputs into "
            "cross-target summary tables for one fixed train domain."
        )
    )
    parser.add_argument(
        "--teacher_metrics_csv",
        action="append",
        required=True,
        help=(
            "Path to teacher_metrics_<tag>.csv from per-target baseline evaluation. "
            "Provide one per target domain."
        ),
    )
    parser.add_argument(
        "--baseline_metrics_csv",
        action="append",
        required=True,
        help=(
            "Path to baseline_metrics_<tag>.csv from per-target baseline evaluation. "
            "Provide one or more per target domain."
        ),
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="domainnet_cross_target")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load and concat
    # --------------------------------------------------------
    teacher_df = pd.concat(
        [load_csv(p) for p in args.teacher_metrics_csv],
        ignore_index=True,
    )
    baseline_df = pd.concat(
        [load_csv(p) for p in args.baseline_metrics_csv],
        ignore_index=True,
    )

    if teacher_df.empty:
        raise ValueError("No teacher metrics loaded.")
    if baseline_df.empty:
        raise ValueError("No baseline metrics loaded.")

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------
    train_domains = sorted(teacher_df["train_domain"].unique().tolist())
    if len(train_domains) != 1:
        raise ValueError(f"Expected exactly one train_domain, got: {train_domains}")
    train_domain = train_domains[0]

    dataset_names = sorted(teacher_df["dataset_name"].unique().tolist())
    if len(dataset_names) != 1:
        raise ValueError(f"Expected exactly one dataset_name, got: {dataset_names}")
    dataset_name = dataset_names[0]

    target_domains = sorted(teacher_df["target_domain"].unique().tolist())
    teacher_names = sorted(teacher_df["teacher"].unique().tolist())

    baseline_train_domains = sorted(baseline_df["train_domain"].unique().tolist())
    if len(baseline_train_domains) != 1:
        raise ValueError(
            f"Expected exactly one baseline train_domain, got: {baseline_train_domains}"
        )
    if baseline_train_domains[0] != train_domain:
        raise ValueError(
            f"Teacher train_domain ({train_domain}) != baseline train_domain ({baseline_train_domains[0]})"
        )

    baseline_dataset_names = sorted(baseline_df["dataset_name"].unique().tolist())
    if len(baseline_dataset_names) != 1:
        raise ValueError(
            f"Expected exactly one baseline dataset_name, got: {baseline_dataset_names}"
        )
    if baseline_dataset_names[0] != dataset_name:
        raise ValueError(
            f"Teacher dataset_name ({dataset_name}) != baseline dataset_name ({baseline_dataset_names[0]})"
        )

    # --------------------------------------------------------
    # Save merged raw tables
    # --------------------------------------------------------
    teacher_df = teacher_df.sort_values(
        ["target_domain", "accuracy", "teacher"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    baseline_sort_cols = [c for c in ["target_domain", "method", "selected_teacher"] if c in baseline_df.columns]
    baseline_df = baseline_df.sort_values(baseline_sort_cols).reset_index(drop=True)

    teacher_df.to_csv(outdir / f"all_teacher_metrics_{args.tag}.csv", index=False)
    baseline_df.to_csv(outdir / f"all_baseline_metrics_raw_{args.tag}.csv", index=False)

    print("\n=== Loaded baseline methods (raw) ===")
    print(sorted(baseline_df["method"].dropna().unique().tolist()))

    # --------------------------------------------------------
    # Teacher-level cross-target summaries
    # --------------------------------------------------------
    teacher_mean_df = (
        teacher_df.groupby("teacher", as_index=False)[["accuracy", "balanced_accuracy"]]
        .mean()
        .sort_values(["accuracy", "teacher"], ascending=[False, True])
        .reset_index(drop=True)
    )
    teacher_mean_df.to_csv(outdir / f"teacher_mean_metrics_{args.tag}.csv", index=False)

    teacher_pivot_df = (
        teacher_df.pivot_table(
            index="target_domain",
            columns="teacher",
            values="accuracy",
            aggfunc="mean",
        )
        .reset_index()
    )
    teacher_pivot_df.to_csv(outdir / f"teacher_accuracy_pivot_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Oracle target best teacher
    # --------------------------------------------------------
    oracle_df = (
        teacher_df.loc[teacher_df.groupby("target_domain")["accuracy"].idxmax()]
        .sort_values("target_domain")
        .reset_index(drop=True)
    )
    oracle_df["method"] = "oracle_target_best_teacher"
    oracle_df.to_csv(outdir / f"oracle_target_best_teacher_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Global best teacher
    # --------------------------------------------------------
    global_best_teacher = str(teacher_mean_df.iloc[0]["teacher"])

    global_rows = []
    for target_domain in target_domains:
        row = teacher_df[
            (teacher_df["target_domain"] == target_domain) &
            (teacher_df["teacher"] == global_best_teacher)
        ].iloc[0]

        global_rows.append({
            "dataset_name": dataset_name,
            "train_domain": train_domain,
            "target_domain": target_domain,
            "split": row["split"],
            "method": "global_best_teacher",
            "selected_teacher": global_best_teacher,
            "accuracy": float(row["accuracy"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "n_samples": int(row["n_samples"]),
        })

    global_df = pd.DataFrame(global_rows).sort_values("target_domain").reset_index(drop=True)
    global_df.to_csv(outdir / f"global_best_teacher_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Leave-one-target-domain-out best teacher
    # --------------------------------------------------------
    acc_map: Dict[str, Dict[str, float]] = {}
    ranking_map: Dict[str, List[str]] = {}

    for target_domain in target_domains:
        sub = teacher_df[teacher_df["target_domain"] == target_domain].copy()
        acc_map[target_domain] = {
            str(r["teacher"]): float(r["accuracy"])
            for _, r in sub.iterrows()
        }
        ranking_map[target_domain] = [
            t for t, _ in sorted(acc_map[target_domain].items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    lodo_rows = []
    for heldout_domain in target_domains:
        source_domains = [d for d in target_domains if d != heldout_domain]

        teacher_source_mean = {}
        for teacher in teacher_names:
            vals = [acc_map[d][teacher] for d in source_domains]
            teacher_source_mean[teacher] = float(np.mean(vals))

        selected_teacher = sorted(
            teacher_source_mean.items(),
            key=lambda kv: (-kv[1], kv[0])
        )[0][0]

        ref_row = teacher_df[
            (teacher_df["target_domain"] == heldout_domain) &
            (teacher_df["teacher"] == selected_teacher)
        ].iloc[0]

        oracle_teacher = ranking_map[heldout_domain][0]
        oracle_acc = acc_map[heldout_domain][oracle_teacher]

        lodo_rows.append({
            "dataset_name": dataset_name,
            "train_domain": train_domain,
            "target_domain": heldout_domain,
            "split": ref_row["split"],
            "method": "leave_one_target_domain_out_best_teacher",
            "selected_teacher": selected_teacher,
            "selected_teacher_source_mean_acc": teacher_source_mean[selected_teacher],
            "accuracy": float(ref_row["accuracy"]),
            "balanced_accuracy": float(ref_row["balanced_accuracy"]),
            "oracle_teacher": oracle_teacher,
            "oracle_target_acc": oracle_acc,
            "gap_to_oracle": oracle_acc - float(ref_row["accuracy"]),
            "n_samples": int(ref_row["n_samples"]),
        })

    lodo_df = pd.DataFrame(lodo_rows).sort_values("target_domain").reset_index(drop=True)
    lodo_df.to_csv(outdir / f"leave_one_target_domain_out_best_teacher_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Baseline deduplication + final baseline table
    # --------------------------------------------------------
    required_baseline_cols = [
        "dataset_name",
        "train_domain",
        "target_domain",
        "split",
        "method",
        "selected_teacher",
        "accuracy",
        "balanced_accuracy",
        "n_samples",
    ]
    missing_cols = [c for c in required_baseline_cols if c not in baseline_df.columns]
    if missing_cols:
        raise ValueError(
            f"Baseline CSVs are missing required columns for final aggregation: {missing_cols}"
        )

    baseline_methods_df = baseline_df[required_baseline_cols].copy()

    # Remove exact duplicate rows that can happen if a method is present
    # in both "classical" and "modern" baseline folders.
    baseline_methods_df = baseline_methods_df.drop_duplicates(
        subset=[
            "dataset_name",
            "train_domain",
            "target_domain",
            "split",
            "method",
            "selected_teacher",
            "accuracy",
            "balanced_accuracy",
            "n_samples",
        ]
    ).sort_values(
        ["target_domain", "method", "selected_teacher"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    baseline_methods_df.to_csv(
        outdir / f"all_baseline_methods_{args.tag}.csv",
        index=False,
    )

    print("\n=== Baseline methods kept for final aggregation ===")
    print(sorted(baseline_methods_df["method"].dropna().unique().tolist()))

    # --------------------------------------------------------
    # Final merged baseline table
    # --------------------------------------------------------
    final_baseline_df = pd.concat(
        [
            oracle_df[[
                "dataset_name", "train_domain", "target_domain", "split",
                "method", "teacher", "accuracy", "balanced_accuracy", "n_samples"
            ]].rename(columns={"teacher": "selected_teacher"}),
            global_df[[
                "dataset_name", "train_domain", "target_domain", "split",
                "method", "selected_teacher", "accuracy", "balanced_accuracy", "n_samples"
            ]],
            lodo_df[[
                "dataset_name", "train_domain", "target_domain", "split",
                "method", "selected_teacher", "accuracy", "balanced_accuracy", "n_samples"
            ]],
            baseline_methods_df[[
                "dataset_name", "train_domain", "target_domain", "split",
                "method", "selected_teacher", "accuracy", "balanced_accuracy", "n_samples"
            ]],
        ],
        ignore_index=True,
    )

    final_baseline_df = final_baseline_df.drop_duplicates().sort_values(
        ["target_domain", "method", "selected_teacher"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    final_baseline_df.to_csv(outdir / f"final_baseline_results_{args.tag}.csv", index=False)

    final_summary_df = (
        final_baseline_df.groupby("method", as_index=False)[["accuracy", "balanced_accuracy"]]
        .mean()
        .sort_values(["accuracy", "method"], ascending=[False, True])
        .reset_index(drop=True)
    )
    final_summary_df.to_csv(outdir / f"final_baseline_summary_{args.tag}.csv", index=False)

    final_pivot_df = (
        final_baseline_df.pivot_table(
            index="target_domain",
            columns="method",
            values="accuracy",
            aggfunc="mean",
        )
        .reset_index()
    )
    final_pivot_df.to_csv(outdir / f"final_baseline_accuracy_pivot_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Ranking instability across target domains
    # --------------------------------------------------------
    pair_rows = []
    for d1, d2 in combinations(target_domains, 2):
        pair_rows.append({
            "dataset_name": dataset_name,
            "train_domain": train_domain,
            "target_domain_a": d1,
            "target_domain_b": d2,
            "winner_a": ranking_map[d1][0],
            "winner_b": ranking_map[d2][0],
            "same_winner": int(ranking_map[d1][0] == ranking_map[d2][0]),
            "winner_changed": int(ranking_map[d1][0] != ranking_map[d2][0]),
            "rank_corr": average_rank_correlation(ranking_map[d1], ranking_map[d2]),
            "pairwise_flip_fraction": pairwise_order_flip_fraction(acc_map[d1], acc_map[d2]),
        })

    pair_df = pd.DataFrame(pair_rows).sort_values(["target_domain_a", "target_domain_b"])
    pair_df.to_csv(outdir / f"target_domain_pair_ranking_flips_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"Train domain: {train_domain}")

    print("\n=== Mean metrics per teacher across target domains ===")
    print(teacher_mean_df.to_string(index=False))

    print(f"\n=== Global best teacher: {global_best_teacher} ===")

    print("\n=== Final baseline summary ===")
    print(final_summary_df.to_string(index=False))

    print("\n=== Final baseline accuracy by target domain ===")
    print(final_pivot_df.to_string(index=False))

    if len(pair_df) > 0:
        print("\n=== Target-domain ranking instability ===")
        print(pair_df.to_string(index=False))

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

# $trainDomain = "quickdraw"

# $teacherCsvs = @(
#     "domainnet_probe_results\probe_baselines\$trainDomain\real\teacher_metrics_domainnet_${trainDomain}_real.csv",
#     "domainnet_probe_results\probe_baselines\$trainDomain\sketch\teacher_metrics_domainnet_${trainDomain}_sketch.csv",
#     "domainnet_probe_results\probe_baselines\$trainDomain\infograph\teacher_metrics_domainnet_${trainDomain}_infograph.csv",
#     "domainnet_probe_results\probe_baselines\$trainDomain\quickdraw\teacher_metrics_domainnet_${trainDomain}_quickdraw.csv"
# )

# $baselineCsvs = @(
#     # classical
#     "domainnet_probe_results\probe_baselines\$trainDomain\real\baseline_metrics_domainnet_${trainDomain}_real.csv",
#     "domainnet_probe_results\probe_baselines\$trainDomain\sketch\baseline_metrics_domainnet_${trainDomain}_sketch.csv",
#     "domainnet_probe_results\probe_baselines\$trainDomain\infograph\baseline_metrics_domainnet_${trainDomain}_infograph.csv",
#     "domainnet_probe_results\probe_baselines\$trainDomain\quickdraw\baseline_metrics_domainnet_${trainDomain}_quickdraw.csv",

#     # modern
#     "domainnet_probe_results\probe_baselines_modern\$trainDomain\real\baseline_metrics_domainnet_${trainDomain}_real.csv",
#     "domainnet_probe_results\probe_baselines_modern\$trainDomain\sketch\baseline_metrics_domainnet_${trainDomain}_sketch.csv",
#     "domainnet_probe_results\probe_baselines_modern\$trainDomain\infograph\baseline_metrics_domainnet_${trainDomain}_infograph.csv",
#     "domainnet_probe_results\probe_baselines_modern\$trainDomain\quickdraw\baseline_metrics_domainnet_${trainDomain}_quickdraw.csv"
# )

# $argsList = @()

# foreach ($p in $teacherCsvs) {
#     if (-not (Test-Path $p)) {
#         throw "Missing teacher CSV: $p"
#     }
#     $argsList += "--teacher_metrics_csv"
#     $argsList += $p
# }

# foreach ($p in $baselineCsvs) {
#     if (-not (Test-Path $p)) {
#         throw "Missing baseline CSV: $p"
#     }
#     $argsList += "--baseline_metrics_csv"
#     $argsList += $p
# }

# python scripts\DomainNet\aggregate_probe_baseline_results_domainnet.py `
#     @argsList `
#     --outdir "domainnet_probe_results\probe_baselines_combined\$trainDomain\cross_target_summary" `
#     --tag "domainnet_${trainDomain}_combined"