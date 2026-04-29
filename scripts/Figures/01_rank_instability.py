#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


TEACHER_NAME_MAP = {
    "openclip_l14_openai_qgelu": "CLIP-L",
    "openclip_b16_datacomp": "CLIP-B",
    "openclip_so400m_siglip": "SigLIP",
    "openclip_l14_dfn2b": "DFN-L",
    "openclip_h14_laion2b": "CLIP-H",
    "openclip_h14_378_dfn5b": "DFN-H",
    "openclip_convnext_xxlarge": "ConvNeXt",
}


DOMAIN_ORDER = {
    "terra": ["location_38", "location_43", "location_46", "location_100"],
    "domainnet": ["quickdraw", "sketch", "real", "infograph"],
}


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def pretty_teacher(name: str) -> str:
    return TEACHER_NAME_MAP.get(str(name), str(name))


def pretty_domain(name: str) -> str:
    name = str(name)
    m = re.match(r"^location_(\d+)$", name)
    if m:
        return f"loc.\n{m.group(1)}"
    return "\n".join(textwrap.wrap(name, width=10, break_long_words=False))


def ordered_domains(domains: List[str], dataset_key: str) -> List[str]:
    preferred = DOMAIN_ORDER.get(dataset_key.lower(), [])
    present = set(domains)
    ordered = [d for d in preferred if d in present]
    ordered += sorted([d for d in domains if d not in ordered])
    return ordered


def load_tidy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"train_domain", "test_domain", "teacher", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["teacher_pretty"] = df["teacher"].map(pretty_teacher)
    return df


def load_instability(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"target_domain_a", "target_domain_b", "rank_corr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if "flip_fraction" not in df.columns and "pairwise_flip_fraction" in df.columns:
        df["flip_fraction"] = df["pairwise_flip_fraction"]

    return df


def build_rank_table(
    tidy_df: pd.DataFrame,
    train_domain: str,
    domain_order: List[str],
    metric: str,
) -> pd.DataFrame:
    sub = tidy_df[tidy_df["train_domain"].astype(str) == str(train_domain)].copy()
    if sub.empty:
        raise ValueError(f"No rows for train_domain={train_domain}")

    rows = []
    for domain in domain_order:
        dsub = sub[sub["test_domain"].astype(str) == str(domain)].copy()
        if dsub.empty:
            continue

        dsub = dsub.sort_values([metric, "teacher_pretty"], ascending=[False, True])
        for rank, (_, row) in enumerate(dsub.iterrows(), start=1):
            rows.append({
                "test_domain": domain,
                "teacher": row["teacher"],
                "teacher_pretty": row["teacher_pretty"],
                "rank": rank,
                metric: row[metric],
            })

    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        raise ValueError("Rank table is empty.")

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
    pivot = pivot.reindex(columns=[d for d in domain_order if d in pivot.columns])
    return pivot


def build_corr_matrix(instab_df: pd.DataFrame, domain_order: List[str]) -> Tuple[np.ndarray, float, float]:
    n = len(domain_order)
    mat = np.eye(n, dtype=float)
    d2i = {d: i for i, d in enumerate(domain_order)}

    corr_vals = []
    flip_vals = []

    for _, row in instab_df.iterrows():
        a = str(row["target_domain_a"])
        b = str(row["target_domain_b"])
        if a not in d2i or b not in d2i:
            continue

        i, j = d2i[a], d2i[b]
        rho = float(row["rank_corr"])
        mat[i, j] = rho
        mat[j, i] = rho
        corr_vals.append(rho)

        if "flip_fraction" in row.index:
            flip_vals.append(float(row["flip_fraction"]))

    mean_corr = float(np.mean(corr_vals)) if corr_vals else float("nan")
    max_flip = float(np.max(flip_vals)) if flip_vals else float("nan")
    return mat, mean_corr, max_flip


def draw_rank_trajectories(ax, pivot: pd.DataFrame, title: str):
    domains = pivot.columns.tolist()
    x = np.arange(len(domains))
    n_teachers = pivot.shape[0]

    for teacher in pivot.index:
        y = pivot.loc[teacher].values.astype(float)
        ax.plot(x, y, marker="o", linewidth=1.8, alpha=0.9)

        # Label on the right side only.
        ax.text(
            x[-1] + 0.05,
            y[-1],
            teacher,
            va="center",
            fontsize=8.5,
        )

    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_domain(d) for d in domains])
    ax.set_ylabel("Teacher rank")
    ax.set_ylim(n_teachers + 0.5, 0.5)
    ax.set_yticks(np.arange(1, n_teachers + 1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.0,
        1.03,
        "Lower rank is better; line crossings indicate instability.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.75,
    )


def draw_corr_heatmap(ax, corr: np.ndarray, domains: List[str], title: str, mean_corr: float, max_flip: float):
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm_r", interpolation="nearest")

    labels = [pretty_domain(d) for d in domains]
    ax.set_xticks(np.arange(len(domains)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    subtitle = f"mean ρ={mean_corr:.2f}"
    if not np.isnan(max_flip):
        subtitle += f", max flip={max_flip:.2f}"

    ax.set_title(f"{title}\n{subtitle}", fontweight="bold", pad=18)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.35, alpha=0.20)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def make_figure(
    terra_tidy_csv: Path,
    terra_instability_csv: Path,
    terra_train_domain: str,
    domainnet_tidy_csv: Path,
    domainnet_instability_csv: Path,
    domainnet_train_domain: str,
    metric: str,
    out_path: Path,
):
    terra_df = load_tidy(terra_tidy_csv)
    domainnet_df = load_tidy(domainnet_tidy_csv)

    terra_instab = load_instability(terra_instability_csv)
    domainnet_instab = load_instability(domainnet_instability_csv)

    terra_domains = ordered_domains(
        terra_df[terra_df["train_domain"] == terra_train_domain]["test_domain"].unique().tolist(),
        "terra",
    )
    dn_domains = ordered_domains(
        domainnet_df[domainnet_df["train_domain"] == domainnet_train_domain]["test_domain"].unique().tolist(),
        "domainnet",
    )

    terra_rank = build_rank_table(terra_df, terra_train_domain, terra_domains, metric)
    dn_rank = build_rank_table(domainnet_df, domainnet_train_domain, dn_domains, metric)

    terra_corr, terra_mean_corr, terra_max_flip = build_corr_matrix(terra_instab, terra_rank.columns.tolist())
    dn_corr, dn_mean_corr, dn_max_flip = build_corr_matrix(domainnet_instab, dn_rank.columns.tolist())

    fig = plt.figure(figsize=(15.5, 8.8))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.65, 1.0],
        height_ratios=[1, 1],
        left=0.08,
        right=0.96,
        top=0.90,
        bottom=0.10,
        wspace=0.35,
        hspace=0.48,
    )

    ax_rank_terra = fig.add_subplot(gs[0, 0])
    ax_corr_terra = fig.add_subplot(gs[0, 1])
    ax_rank_dn = fig.add_subplot(gs[1, 0])
    ax_corr_dn = fig.add_subplot(gs[1, 1])

    draw_rank_trajectories(
        ax_rank_terra,
        terra_rank,
        f"(a) TerraIncognita teacher-rank trajectories\ntrain = {terra_train_domain}",
    )
    im1 = draw_corr_heatmap(
        ax_corr_terra,
        terra_corr,
        terra_rank.columns.tolist(),
        "(b) TerraIncognita rank correlation",
        terra_mean_corr,
        terra_max_flip,
    )

    draw_rank_trajectories(
        ax_rank_dn,
        dn_rank,
        f"(c) DomainNet teacher-rank trajectories\ntrain = {domainnet_train_domain}",
    )
    im2 = draw_corr_heatmap(
        ax_corr_dn,
        dn_corr,
        dn_rank.columns.tolist(),
        "(d) DomainNet rank correlation",
        dn_mean_corr,
        dn_max_flip,
    )

    cbar = fig.colorbar(
        im2,
        ax=[ax_corr_terra, ax_corr_dn],
        orientation="horizontal",
        fraction=0.055,
        pad=0.13,
        aspect=35,
    )
    cbar.set_label("Spearman rank correlation ρ")

    fig.suptitle(
        "Teacher preference is domain-dependent under distribution shift",
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terra_tidy_csv", required=True)
    parser.add_argument("--terra_instability_csv", required=True)
    parser.add_argument("--terra_train_domain", required=True)

    parser.add_argument("--domainnet_tidy_csv", required=True)
    parser.add_argument("--domainnet_instability_csv", required=True)
    parser.add_argument("--domainnet_train_domain", required=True)

    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    make_figure(
        terra_tidy_csv=Path(args.terra_tidy_csv),
        terra_instability_csv=Path(args.terra_instability_csv),
        terra_train_domain=args.terra_train_domain,
        domainnet_tidy_csv=Path(args.domainnet_tidy_csv),
        domainnet_instability_csv=Path(args.domainnet_instability_csv),
        domainnet_train_domain=args.domainnet_train_domain,
        metric=args.metric,
        out_path=Path(args.out_path),
    )


if __name__ == "__main__":
    main()