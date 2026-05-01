import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_results(results_dir):
    csvs = sorted(Path(results_dir).glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No result CSVs in {results_dir}")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    return df


def compute_method_means(df):
    """
    Compute mean accuracy per method across targets
    """
    summary = (
        df.groupby(["dataset", "source", "target", "method"])
        .agg(acc_mean=("accuracy", "mean"))
        .reset_index()
    )

    overall = (
        summary.groupby(["dataset", "source", "method"])
        .agg(acc_mean=("acc_mean", "mean"))
        .reset_index()
    )

    return overall


def compute_teacher_baselines(probe_csv, source):
    df = pd.read_csv(probe_csv)

    required = {"dataset", "source", "target", "seed", "teacher", "method", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Probe CSV missing columns: {sorted(missing)}")

    # Match the same source domain used by the method runs.
    df = df[df["source"].astype(str) == str(source)].copy()

    # Keep only teacher/probe rows if multiple method types exist.
    # Print available methods if this filter removes everything.
    if "linear_probe" in set(df["method"].astype(str)):
        df = df[df["method"].astype(str) == "linear_probe"].copy()

    if df.empty:
        raise RuntimeError(
            f"No rows left for source={source}. Available methods: "
            f"{sorted(pd.read_csv(probe_csv)['method'].astype(str).unique())}"
        )

    # Average duplicate seeds first.
    per_teacher_target = (
        df.groupby(["teacher", "target"])
        .agg(acc=("accuracy", "mean"))
        .reset_index()
    )

    # Global best teacher:
    # Choose the teacher with best mean across all targets, then report its mean target accuracy.
    teacher_means = (
        per_teacher_target.groupby("teacher")["acc"]
        .mean()
        .sort_values(ascending=False)
    )
    best_teacher = teacher_means.index[0]
    global_best_value = float(teacher_means.iloc[0])

    # Leave-one-domain-out:
    # For each target, choose best teacher using the other targets, evaluate on held-out target.
    targets = sorted(per_teacher_target["target"].unique())
    lodo_scores = []

    for heldout in targets:
        train_part = per_teacher_target[per_teacher_target["target"] != heldout]
        test_part = per_teacher_target[per_teacher_target["target"] == heldout]

        means = train_part.groupby("teacher")["acc"].mean()
        chosen_teacher = means.idxmax()

        score = test_part[test_part["teacher"] == chosen_teacher]["acc"].mean()
        lodo_scores.append(float(score))

    return {
        "global_best_teacher": global_best_value,
        "lodo_teacher": float(np.mean(lodo_scores)),
        "best_teacher": best_teacher,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--probe_csv", required=True)
    parser.add_argument("--outdir", default="final_results_summary")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load method results
    df = load_results(args.results_dir)

    method_means = compute_method_means(df)

    # Compute teacher baselines
    baselines = compute_teacher_baselines(args.probe_csv, source=df["source"].iloc[0])

    # Convert baselines to df
    baseline_df = pd.DataFrame([
        {"method": "global_best_teacher", "acc_mean": baselines["global_best_teacher"]},
        {"method": "lodo_teacher", "acc_mean": baselines["lodo_teacher"]},
    ])

    # Save outputs
    method_means.to_csv(outdir / "method_means.csv", index=False)
    baseline_df.to_csv(outdir / "teacher_baselines.csv", index=False)

    print("\n=== METHOD MEANS ===")
    print(method_means)

    print("\n=== TEACHER BASELINES ===")
    print(baseline_df)


if __name__ == "__main__":
    main()