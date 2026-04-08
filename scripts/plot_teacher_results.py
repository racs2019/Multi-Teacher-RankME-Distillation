#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_npz_meta(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if "meta_json" not in data.files:
        raise ValueError(f"{npz_path} does not contain meta_json")
    meta_json = data["meta_json"]
    if isinstance(meta_json, np.ndarray):
        meta_json = meta_json.item()
    return json.loads(meta_json)


def collect_results(npz_dir: Path) -> pd.DataFrame:
    rows = []
    for npz_path in sorted(npz_dir.glob("*.npz")):
        meta = load_npz_meta(npz_path)

        rows.append(
            {
                "npz_path": str(npz_path),
                "dataset_name": meta.get("dataset_name", "unknown"),
                "domain": meta.get("domain", "unknown"),
                "teacher": meta.get("teacher_name", npz_path.stem),
                "accuracy": float(meta.get("zero_shot_top1_acc", np.nan)),
                "num_samples": meta.get("num_samples", None),
                "model_tag": meta.get("model_tag", ""),
            }
        )

    if not rows:
        raise RuntimeError(f"No .npz files found in {npz_dir}")

    df = pd.DataFrame(rows)
    return df.sort_values(["domain", "teacher"]).reset_index(drop=True)


def shorten_teacher_name(name: str) -> str:
    mapping = {
        "openclip_l14_openai_qgelu": "OpenAI L14",
        "openclip_b16_datacomp": "DataComp B16",
        "openclip_so400m_siglip": "SigLIP SO400M",
        "openclip_l14_dfn2b": "DFN L14",
    }
    return mapping.get(name, name)


def make_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index="domain", columns="teacher", values="accuracy")
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def plot_heatmap(pivot: pd.DataFrame, out_path: Path) -> None:
    domains = list(pivot.index)
    teachers = list(pivot.columns)
    values = pivot.values

    fig, ax = plt.subplots(figsize=(1.8 * len(teachers) + 2, 1.2 * len(domains) + 2))
    im = ax.imshow(values, aspect="auto")

    ax.set_xticks(np.arange(len(teachers)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels([shorten_teacher_name(t) for t in teachers], rotation=25, ha="right")
    ax.set_yticklabels(domains)

    ax.set_title("Zero-shot teacher accuracy by domain")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    # annotate cells
    for i in range(len(domains)):
        row = values[i]
        if np.all(np.isnan(row)):
            continue
        best_j = int(np.nanargmax(row))
        for j in range(len(teachers)):
            val = values[i, j]
            if np.isnan(val):
                txt = "nan"
            else:
                txt = f"{val:.3f}"
                if j == best_j:
                    txt += " ★"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(pivot: pd.DataFrame, out_path: Path) -> None:
    domains = list(pivot.index)
    teachers = list(pivot.columns)

    x = np.arange(len(domains))
    width = 0.8 / max(len(teachers), 1)

    fig, ax = plt.subplots(figsize=(2.2 * len(domains) + 2, 5.5))

    for i, teacher in enumerate(teachers):
        vals = pivot[teacher].values
        ax.bar(
            x + (i - (len(teachers) - 1) / 2) * width,
            vals,
            width=width,
            label=shorten_teacher_name(teacher),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Teacher comparison across domains")
    ax.legend()
    ax.set_ylim(0.0, min(1.0, max(0.65, np.nanmax(pivot.values) + 0.05)))

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_rankings(pivot: pd.DataFrame, out_csv: Path) -> None:
    rows = []
    for domain in pivot.index:
        row = pivot.loc[domain].dropna().sort_values(ascending=False)
        out = {"domain": domain}
        for rank_idx, (teacher, acc) in enumerate(row.items(), start=1):
            out[f"rank_{rank_idx}"] = teacher
            out[f"rank_{rank_idx}_acc"] = float(acc)
        rows.append(out)

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize teacher NPZ results as heatmap and bar chart."
    )
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory containing teacher .npz files")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save plots/csvs")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional filter, e.g. terra_incognita",
    )
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = collect_results(npz_dir)

    if args.dataset_name is not None:
        df = df[df["dataset_name"] == args.dataset_name].copy()

    if len(df) == 0:
        raise RuntimeError("No rows left after filtering")

    df["teacher_short"] = df["teacher"].map(shorten_teacher_name)

    summary_csv = outdir / "teacher_accuracy_long.csv"
    df.to_csv(summary_csv, index=False)

    pivot = make_pivot(df)
    pivot_csv = outdir / "teacher_accuracy_pivot.csv"
    pivot.to_csv(pivot_csv)

    ranking_csv = outdir / "teacher_rankings.csv"
    save_rankings(pivot, ranking_csv)

    heatmap_png = outdir / "teacher_accuracy_heatmap.png"
    bars_png = outdir / "teacher_accuracy_grouped_bars.png"

    plot_heatmap(pivot, heatmap_png)
    plot_grouped_bars(pivot, bars_png)

    print("Saved:")
    print(f"  {summary_csv}")
    print(f"  {pivot_csv}")
    print(f"  {ranking_csv}")
    print(f"  {heatmap_png}")
    print(f"  {bars_png}")


if __name__ == "__main__":
    main()

# python plot_teacher_results.py `
#   --npz_dir teacher_npzs `
#   --outdir terra_plots `
#   --dataset_name terra_incognita