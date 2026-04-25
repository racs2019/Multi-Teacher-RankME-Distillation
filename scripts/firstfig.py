#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
    "clip-eva-02-clip-l": "CLIP-EVA-02-CLIP-L",
    "clip-h-14-openai": "CLIP-H-14-OpenAI",
    "clip-l-14-openai": "CLIP-L-14-OpenAI",
    "openclip-vit-l-14": "OpenCLIP ViT-L-14",
    "siglip-vit-so400m": "SigLIP ViT-SO400M",
    "openclip-vit-h-14": "OpenCLIP ViT-H-14",
    "siglip-vit-l-16": "SigLIP ViT-L-16",
    "openclip-vit-bigg-14": "OpenCLIP ViT-bigG-14",
}


# ============================================================
# Global style
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
})


# ============================================================
# Helpers
# ============================================================

def sort_domains(domains: List[str]) -> List[str]:
    def key_fn(x: str):
        digits = "".join(ch for ch in str(x) if ch.isdigit())
        return (0, int(digits)) if digits else (1, str(x))
    return sorted(domains, key=key_fn)


def prettify_teacher_name(name: str) -> str:
    return TEACHER_NAME_MAP.get(name, name)


def pretty_domain_name(text: str) -> str:
    """
    Make domain labels compact for plotting.
    Examples:
      location_38 -> location\\n38
      location_100 -> location\\n100
      quickdraw -> quickdraw
    """
    text = str(text)
    m = re.match(r"^([A-Za-z]+)_(\d+)$", text)
    if m:
        return f"{m.group(1)}\n{m.group(2)}"
    return text


def wrap_label(text: str, width: int = 12) -> str:
    text = str(text)
    if "\n" in text:
        return text
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def load_tidy_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"train_domain", "test_domain", "teacher", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in tidy CSV {path}: {sorted(missing)}")

    if "teacher_pretty" not in df.columns:
        df["teacher_pretty"] = df["teacher"].map(prettify_teacher_name)

    return df


def load_instability_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "rank_corr" not in df.columns:
        raise ValueError(f"Missing rank_corr in instability CSV: {path}")

    if "target_domain_a" not in df.columns or "target_domain_b" not in df.columns:
        raise ValueError(
            f"Instability CSV must contain target_domain_a and target_domain_b: {path}"
        )

    if "flip_fraction" not in df.columns and "pairwise_flip_fraction" in df.columns:
        df["flip_fraction"] = df["pairwise_flip_fraction"]

    return df


def build_rank_matrix(
    tidy_df: pd.DataFrame,
    train_domain: str,
    metric: str = "accuracy",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    sub = tidy_df[tidy_df["train_domain"] == train_domain].copy()
    if sub.empty:
        raise ValueError(f"No rows for train_domain={train_domain}")

    test_domains = sort_domains(sub["test_domain"].unique().tolist())

    rank_rows = []
    for test_domain in test_domains:
        s = sub[sub["test_domain"] == test_domain].sort_values(
            [metric, "teacher_pretty"], ascending=[False, True]
        )
        for rank_idx, (_, row) in enumerate(s.iterrows(), start=1):
            rank_rows.append({
                "teacher_pretty": row["teacher_pretty"],
                "test_domain": test_domain,
                "rank": rank_idx,
            })

    rank_df = pd.DataFrame(rank_rows)
    teacher_order = (
        rank_df.groupby("teacher_pretty")["rank"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    pivot = rank_df.pivot_table(
        index="teacher_pretty",
        columns="test_domain",
        values="rank",
        aggfunc="mean",
    )

    pivot = pivot.reindex(index=teacher_order)
    pivot = pivot.reindex(columns=test_domains)

    return pivot, teacher_order, test_domains


def build_corr_upper_triangle(
    instability_df: pd.DataFrame,
    domain_order: List[str],
) -> np.ndarray:
    n = len(domain_order)
    mat = np.full((n, n), np.nan, dtype=float)

    domain_to_idx = {d: i for i, d in enumerate(domain_order)}

    for _, row in instability_df.iterrows():
        a = row["target_domain_a"]
        b = row["target_domain_b"]
        rho = float(row["rank_corr"])

        if a not in domain_to_idx or b not in domain_to_idx:
            continue

        i = domain_to_idx[a]
        j = domain_to_idx[b]
        mat[i, j] = rho

    return mat


# ============================================================
# Drawing
# ============================================================

def draw_rank_panel(ax, pivot: pd.DataFrame, title: str):
    values = pivot.values
    row_labels = [wrap_label(x, 16) for x in pivot.index.tolist()]
    col_labels = [pretty_domain_name(x) for x in pivot.columns.tolist()]

    cmap = plt.cm.coolwarm.copy()
    im = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)

    ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        pad=3,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=22)
    ax.set_ylabel("Teacher (model)", fontsize=10, labelpad=10)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.35, alpha=0.18)
    ax.tick_params(which="minor", bottom=False, left=False)

    max_rank = np.nanmax(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val >= max_rank - 1 else "black"
            ax.text(
                j, i, f"{int(val)}",
                ha="center", va="center",
                fontsize=9.5, color=color
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#999999")

    return im


def draw_corr_panel(ax, corr_mat: np.ndarray, domain_order: List[str], title: str):
    labels = [pretty_domain_name(x) for x in domain_order]

    cmap = plt.cm.coolwarm_r.copy()
    cmap.set_bad(color="#EAEAEA")

    im = ax.imshow(
        corr_mat,
        vmin=-1,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5, rotation=0, ha="center")
    ax.set_yticklabels(labels, fontsize=9)

    ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        pad=3,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=22)
    ax.set_ylabel("Target domain", fontsize=10, labelpad=10)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.35, alpha=0.18)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            val = corr_mat[i, j]
            txt = "—" if np.isnan(val) else f"{val:.2f}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=9.5, color="black"
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#999999")

    return im


# ============================================================
# Main figure
# ============================================================

def make_dual_dataset_figure(
    terra_tidy_csv: Path,
    terra_instability_csv: Path,
    terra_train_domain: str,
    domainnet_tidy_csv: Path,
    domainnet_instability_csv: Path,
    domainnet_train_domain: str,
    out_path: Path,
    metric: str = "accuracy",
):
    terra_df = load_tidy_csv(terra_tidy_csv)
    terra_instab = load_instability_csv(terra_instability_csv)

    domainnet_df = load_tidy_csv(domainnet_tidy_csv)
    domainnet_instab = load_instability_csv(domainnet_instability_csv)

    terra_pivot, _, terra_domains = build_rank_matrix(
        terra_df, terra_train_domain, metric=metric
    )
    dn_pivot, _, dn_domains = build_rank_matrix(
        domainnet_df, domainnet_train_domain, metric=metric
    )

    terra_corr = build_corr_upper_triangle(terra_instab, terra_domains)
    dn_corr = build_corr_upper_triangle(domainnet_instab, dn_domains)

    fig = plt.figure(figsize=(15.8, 9.8), constrained_layout=False)
    gs = GridSpec(
        2, 2,
        width_ratios=[1.55, 0.95],
        height_ratios=[1.0, 1.0],
        left=0.10,
        right=0.98,
        top=0.90,
        bottom=0.16,
        wspace=0.32,
        hspace=0.42,
        figure=fig,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    im1 = draw_rank_panel(
        ax1,
        terra_pivot,
        title=f"(a) Terra\ntrain = {terra_train_domain}",
    )
    im2 = draw_corr_panel(
        ax2,
        terra_corr,
        terra_domains,
        title="(b) Terra\nSpearman rank correlation $\\rho$",
    )
    im3 = draw_rank_panel(
        ax3,
        dn_pivot,
        title=f"(c) DomainNet\ntrain = {domainnet_train_domain}",
    )
    im4 = draw_corr_panel(
        ax4,
        dn_corr,
        dn_domains,
        title="(d) DomainNet\nSpearman rank correlation $\\rho$",
    )

    # Left colorbar: rank
    cbar_rank = fig.colorbar(
        im1,
        ax=[ax1, ax3],
        orientation="horizontal",
        fraction=0.05,
        pad=0.10,
        aspect=40,
    )
    cbar_rank.set_label("Rank (1 = best)", fontsize=10)
    cbar_rank.ax.tick_params(labelsize=9)

    # Right colorbar: correlation
    cbar_corr = fig.colorbar(
        im2,
        ax=[ax2, ax4],
        orientation="horizontal",
        fraction=0.05,
        pad=0.10,
        aspect=40,
    )
    cbar_corr.set_label("Spearman $\\rho$", fontsize=10)
    cbar_corr.ax.tick_params(labelsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a cleaner paper-style instability figure for Terra and DomainNet."
    )
    parser.add_argument("--terra_tidy_csv", type=str, required=True)
    parser.add_argument("--terra_instability_csv", type=str, required=True)
    parser.add_argument("--terra_train_domain", type=str, required=True)

    parser.add_argument("--domainnet_tidy_csv", type=str, required=True)
    parser.add_argument("--domainnet_instability_csv", type=str, required=True)
    parser.add_argument("--domainnet_train_domain", type=str, required=True)

    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--out_path", type=str, required=True)

    args = parser.parse_args()

    make_dual_dataset_figure(
        terra_tidy_csv=Path(args.terra_tidy_csv),
        terra_instability_csv=Path(args.terra_instability_csv),
        terra_train_domain=args.terra_train_domain,
        domainnet_tidy_csv=Path(args.domainnet_tidy_csv),
        domainnet_instability_csv=Path(args.domainnet_instability_csv),
        domainnet_train_domain=args.domainnet_train_domain,
        out_path=Path(args.out_path),
        metric=args.metric,
    )


if __name__ == "__main__":
    main()

# Example:
# python scripts/firstfig.py `
#   --terra_tidy_csv "domainnet_probe_plots/linear_probe_results_tidy.csv" `
#   --terra_instability_csv "domainnet_probe_plots/ranking_instability_train_quickdraw_accuracy.csv" `
#   --terra_train_domain "quickdraw" `
#   --domainnet_tidy_csv "terra_probe_plots/linear_probe_results_tidy.csv" `
#   --domainnet_instability_csv "terra_probe_plots/ranking_instability_train_location_38_accuracy.csv" `
#   --domainnet_train_domain "location_38" `
#   --metric "accuracy" `
#   --out_path "figures/dual_dataset_instability.png"