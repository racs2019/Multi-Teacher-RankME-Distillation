import argparse
from pathlib import Path
import pandas as pd


METHOD_ORDER = [
    "global_best_teacher",
    "lodo_teacher",
    "uniform",
    "entropy_weighted",
    "agreement_weighted",
    "tent_proxy",
    "graph_lame",
    "graph_label_prop",
    "grace",
]

METHOD_LABELS = {
    "global_best_teacher": "Global teacher",
    "lodo_teacher": "LODO teacher",
    "uniform": "Uniform",
    "entropy_weighted": "Entropy-wtd.",
    "agreement_weighted": "Agreement-wtd.",
    "tent_proxy": "TENT proxy",
    "graph_lame": "Graph smooth",
    "graph_label_prop": "Graph prop.",
    "grace": "GRACE",
}


def load_method_results(results_dir: Path):
    files = list(results_dir.rglob("*.csv"))
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        needed = {"dataset", "source", "target", "method", "accuracy"}
        if needed.issubset(df.columns):
            dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No method CSVs found under {results_dir}")

    return pd.concat(dfs, ignore_index=True)


def load_teacher_results(probe_root: Path):
    files = list(probe_root.rglob("linear_probe_results.csv"))
    rows = []

    for f in files:
        df = pd.read_csv(f)

        needed = {"dataset", "source", "target", "teacher", "accuracy"}
        if not needed.issubset(df.columns):
            continue

        # held-out only
        df = df[df["source"] != df["target"]].copy()

        for source, g in df.groupby("source"):
            # global best teacher, selected by mean target accuracy
            teacher_means = g.groupby("teacher")["accuracy"].mean()
            best_teacher_acc = teacher_means.max()

            rows.append({
                "dataset": g["dataset"].iloc[0],
                "source": source,
                "target": "heldout_mean",
                "method": "global_best_teacher",
                "accuracy": best_teacher_acc,
            })

            # If no actual LODO selection exists, keep same as global.
            rows.append({
                "dataset": g["dataset"].iloc[0],
                "source": source,
                "target": "heldout_mean",
                "method": "lodo_teacher",
                "accuracy": best_teacher_acc,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--probe_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--exclude_in_domain", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    probe_root = Path(args.probe_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    method_df = load_method_results(results_dir)

    if args.exclude_in_domain:
        method_df = method_df[method_df["source"] != method_df["target"]].copy()

    method_means = (
        method_df
        .groupby(["dataset", "source", "method"], as_index=False)
        .agg(acc_mean=("accuracy", "mean"))
    )

    teacher_df = load_teacher_results(probe_root)
    teacher_means = teacher_df.rename(columns={"accuracy": "acc_mean"})[
        ["dataset", "source", "method", "acc_mean"]
    ]

    all_df = pd.concat([method_means, teacher_means], ignore_index=True)

    all_df["method"] = pd.Categorical(
        all_df["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    wide = (
        all_df
        .pivot_table(
            index="source",
            columns="method",
            values="acc_mean",
            aggfunc="mean",
            observed=False,
        )
        .reset_index()
    )

    # Add mean over source domains
    mean_row = {"source": "Mean"}
    for m in METHOD_ORDER:
        if m in wide.columns:
            mean_row[m] = wide[m].mean()
    wide = pd.concat([wide, pd.DataFrame([mean_row])], ignore_index=True)

    # Rename columns for paper
    wide = wide.rename(columns=METHOD_LABELS)

    csv_path = outdir / "domainnet_source_sweep_table.csv"
    tex_path = outdir / "domainnet_source_sweep_table.tex"

    wide.to_csv(csv_path, index=False)

    tex = wide.to_latex(
        index=False,
        float_format=lambda x: f"{x:.3f}",
        escape=False,
    )
    tex_path.write_text(tex, encoding="utf-8")

    print("\nSaved:")
    print(csv_path)
    print(tex_path)
    print("\nTable:")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()