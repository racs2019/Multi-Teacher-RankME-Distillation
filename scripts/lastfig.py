#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Ellipse
from PIL import Image


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
})

CLASS_COLORS = {
    "guitar": "#4C72B0",
    "piano": "#F1A340",
    "chair": "#55A868",
    "table": "#C44E52",
    "car": "#8E6BBE",
    "truck": "#9C755F",
    "bus": "#DA8BC3",
    "bicycle": "#9D9D9D",
}

GOOD_COLOR = "#2E8B57"
BAD_COLOR = "#D62728"


def add_panel_title(ax, text: str, y=1.06, fontsize=12.0):
    ax.text(
        0.0, y, text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def set_clean_embedding_axes(ax, xlabel=None, ylabel=None):
    ax.set_xticks([])
    ax.set_yticks([])

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)

    for spine in ax.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(0.8)


def add_pred_text(ax, pred: str, conf: str | float | None, is_good: bool, rankme: float):
    color = GOOD_COLOR if is_good else BAD_COLOR
    conf_text = f"{conf:.2f}" if isinstance(conf, (float, int)) else str(conf)

    ax.text(
        0.025,
        0.035,
        f"Pred: {pred}\nConf: {conf_text}\nRankMe: {rankme:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.7,
        color=color,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.72,
        ),
    )


def draw_query_star(ax, x, y):
    ax.scatter(
        [x],
        [y],
        marker="*",
        s=78,
        facecolor="white",
        edgecolor="#333333",
        linewidth=1.15,
        zorder=5,
    )


def add_neighborhood_ellipse(ax, xy, width, height, edgecolor, linestyle="--"):
    ax.add_patch(
        Ellipse(
            xy=xy,
            width=width,
            height=height,
            angle=0,
            fill=False,
            edgecolor=edgecolor,
            linewidth=1.05,
            linestyle=linestyle,
            alpha=0.82,
        )
    )


def add_regression_line(ax, x, y, color):
    coeffs = np.polyfit(np.log10(x), y, deg=1)
    xs = np.geomspace(x.min(), x.max(), 100)
    ys = coeffs[0] * np.log10(xs) + coeffs[1]
    ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.9)


def plot_neighborhood_panel(
    ax,
    coords: np.ndarray,
    labels: list[str],
    query_xy: tuple[float, float],
    rankme_value: float,
    pred_class: str,
    conf_value,
    true_class: str,
    ellipse_xy=(0, 0),
    ellipse_wh=(2.5, 2.5),
):
    for cls in sorted(set(labels)):
        mask = np.array([c == cls for c in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=16,
            alpha=0.50,
            color=CLASS_COLORS.get(cls, "#999999"),
            edgecolors="none",
        )

    draw_query_star(ax, query_xy[0], query_xy[1])

    is_good = pred_class == true_class
    add_neighborhood_ellipse(
        ax,
        ellipse_xy,
        ellipse_wh[0],
        ellipse_wh[1],
        edgecolor=GOOD_COLOR if is_good else BAD_COLOR,
    )

    add_pred_text(
        ax,
        pred=pred_class,
        conf=conf_value,
        is_good=is_good,
        rankme=rankme_value,
    )

    set_clean_embedding_axes(ax)


def plot_rankme_vs_metric(ax, x, y, color, xlabel, ylabel, rho_text):
    ax.scatter(x, y, s=8, alpha=0.34, color=color, edgecolors="none")
    ax.set_xscale("log")
    add_regression_line(ax, x, y, color=color)

    ax.set_xlabel(xlabel, fontsize=9.2)
    ax.set_ylabel(ylabel, fontsize=9.2)

    ax.text(
        0.96,
        0.06,
        rho_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=color,
        fontsize=9.2,
        fontweight="bold",
    )

    ax.tick_params(labelsize=8.2)

    for spine in ax.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(0.8)


def plot_disagreement_gain(ax, x, y, color, rho_text):
    ax.scatter(x, y, s=8, alpha=0.34, color=color, edgecolors="none")
    ax.set_xscale("log")
    add_regression_line(ax, x, y, color=color)
    ax.axhline(0, color="#BBBBBB", linestyle="--", linewidth=0.9)

    ax.set_xlabel("Teacher pair disagreement\n(JSD)", fontsize=9.2)
    ax.set_ylabel("Accuracy delta\n(RankMe vs uniform)", fontsize=9.2)

    ax.text(
        0.96,
        0.06,
        rho_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=color,
        fontsize=9.2,
        fontweight="bold",
    )

    ax.tick_params(labelsize=8.2)

    for spine in ax.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(0.8)


def draw_prediction_table(ax, rows, title):
    ax.axis("off")

    ax.text(
        0.5,
        0.98,
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )

    col_x = [0.03, 0.55, 0.86]
    headers = ["Teacher", "Pred.", "RankMe"]

    for x, h in zip(col_x, headers):
        ax.text(
            x,
            0.81,
            h,
            ha="left",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )

    y0 = 0.68
    dy = 0.15

    for i, row in enumerate(rows):
        y = y0 - i * dy

        ax.add_patch(
            FancyBboxPatch(
                (0.015, y - 0.060),
                0.96,
                0.098,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                facecolor="#F5F5F5",
                edgecolor="none",
            )
        )

        ax.text(col_x[0], y, row["teacher"], ha="left", va="center", fontsize=8.7)

        ax.text(
            col_x[1],
            y,
            row["pred_conf"],
            ha="left",
            va="center",
            fontsize=8.7,
            color=row.get("pred_color", "black"),
            fontweight="bold",
        )

        ax.text(
            col_x[2],
            y,
            row["rankme"],
            ha="left",
            va="center",
            fontsize=8.7,
            color=row.get("rankme_color", "black"),
            fontweight="bold",
        )


def draw_ensemble_box(ax, uniform_pred, rankme_pred):
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "Ensemble top-1 prediction",
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )

    boxes = [
        ("Uniform", uniform_pred[0], uniform_pred[1]),
        (r"RankMe-$\delta$", rankme_pred[0], rankme_pred[1]),
    ]

    y_positions = [0.58, 0.27]

    for (label, pred_text, ok), y in zip(boxes, y_positions):
        ax.add_patch(
            FancyBboxPatch(
                (0.06, y - 0.095),
                0.88,
                0.165,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor="#F7F7F7",
                edgecolor="#C0C0C0",
            )
        )

        ax.text(0.10, y, label, ha="left", va="center", fontsize=10.0)

        ax.text(
            0.90,
            y,
            pred_text,
            ha="right",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=GOOD_COLOR if ok else BAD_COLOR,
        )


def draw_image_panel(ax, image_path: str | Path, title: str):
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title, fontsize=10.8, fontweight="bold", pad=5)

    img = Image.open(image_path).convert("RGB")
    ax.imshow(img)

    for spine in ax.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(0.8)


def make_mock_data():
    rng = np.random.default_rng(4)

    def cluster(center, n, scale=0.55):
        return rng.normal(loc=center, scale=scale, size=(n, 2))

    q1_labels = ["guitar"] * 30 + ["piano"] * 35 + ["chair"] * 20 + ["truck"] * 15
    q1_a = np.vstack([
        cluster((-1.2, 0.4), 30),
        cluster((0.7, 0.3), 35),
        cluster((0.1, 1.0), 20),
        cluster((0.2, -0.8), 15),
    ])
    q1_b = np.vstack([
        cluster((-0.4, 0.0), 30),
        cluster((1.0, 0.7), 35),
        cluster((-0.7, 0.7), 20),
        cluster((0.5, -0.9), 15),
    ])
    q1_c = np.vstack([
        cluster((0.0, 0.0), 30),
        cluster((1.0, 0.4), 35),
        cluster((0.9, -0.7), 20),
        cluster((-0.8, 0.5), 15),
    ])

    q2_labels = ["guitar"] * 20 + ["chair"] * 35 + ["table"] * 30 + ["bus"] * 15 + ["piano"] * 15
    q2_a = np.vstack([
        cluster((1.3, -0.9), 20),
        cluster((-0.2, 0.5), 35),
        cluster((0.4, -0.1), 30),
        cluster((-1.0, 0.0), 15),
        cluster((0.7, 0.8), 15),
    ])
    q2_b = np.vstack([
        cluster((-1.0, -0.1), 20),
        cluster((0.0, 0.8), 35),
        cluster((1.0, -0.2), 30),
        cluster((-1.0, 0.7), 15),
        cluster((0.6, -0.8), 15),
    ])
    q2_c = np.vstack([
        cluster((1.4, -1.0), 20),
        cluster((-0.3, 0.9), 35),
        cluster((0.8, -0.2), 30),
        cluster((-1.0, 0.1), 15),
        cluster((1.0, 0.6), 15),
    ])

    q3_labels = ["car"] * 35 + ["truck"] * 30 + ["bus"] * 25 + ["guitar"] * 15
    q3_a = np.vstack([
        cluster((-0.8, 0.2), 35),
        cluster((0.8, -0.2), 30),
        cluster((-1.2, 1.1), 25),
        cluster((0.9, 1.0), 15),
    ])
    q3_b = np.vstack([
        cluster((-1.0, 0.7), 35),
        cluster((0.5, -0.8), 30),
        cluster((-1.4, -0.1), 25),
        cluster((0.9, 0.8), 15),
    ])
    q3_c = np.vstack([
        cluster((-0.9, 0.0), 35),
        cluster((0.8, -0.5), 30),
        cluster((0.9, 0.5), 25),
        cluster((0.1, 1.0), 15),
    ])

    x1 = 10 ** rng.normal(0.0, 0.6, 1200)
    y1 = 0.25 + 0.22 * np.log10(x1) + rng.normal(0, 0.16, 1200)
    y1 = np.clip(y1, 0.25, 1.0)

    x2 = 10 ** rng.normal(0.0, 0.6, 1200)
    y2 = 0.30 + 0.25 * np.log10(x2) + rng.normal(0, 0.18, 1200)
    y2 = np.clip(y2, 0.30, 1.0)

    x3 = 10 ** rng.normal(-0.8, 0.7, 1200)
    y3 = -0.01 + 0.06 * np.log10(x3 + 1e-8 + 1) + rng.normal(0, 0.025, 1200)
    y3 = np.clip(y3, -0.1, 0.2)

    return {
        "queries": [
            {
                "query_label": "Query 1\nlocation_100\ntrue: guitar",
                "labels": q1_labels,
                "panels": [
                    dict(rankme=0.76, pred="guitar", conf=0.82, true="guitar",
                         coords=q1_a, query=(0.0, 0.0), ellipse_xy=(-0.7, 0.2), ellipse_wh=(2.3, 2.6)),
                    dict(rankme=0.28, pred="piano", conf=0.31, true="guitar",
                         coords=q1_b, query=(0.0, 0.0), ellipse_xy=(0.7, 0.2), ellipse_wh=(2.1, 2.8)),
                    dict(rankme=0.54, pred="guitar", conf="--", true="guitar",
                         coords=q1_c, query=(0.0, 0.0), ellipse_xy=(0.7, 0.3), ellipse_wh=(2.2, 2.8)),
                ],
            },
            {
                "query_label": "Query 2\nlocation_43\ntrue: chair",
                "labels": q2_labels,
                "panels": [
                    dict(rankme=0.71, pred="guitar", conf=0.74, true="chair",
                         coords=q2_a, query=(0.0, 0.0), ellipse_xy=(-0.1, 0.3), ellipse_wh=(2.5, 2.3)),
                    dict(rankme=0.33, pred="table", conf=0.28, true="chair",
                         coords=q2_b, query=(0.0, 0.0), ellipse_xy=(0.7, -0.2), ellipse_wh=(2.1, 2.0)),
                    dict(rankme=0.49, pred="chair", conf="--", true="chair",
                         coords=q2_c, query=(0.0, 0.0), ellipse_xy=(0.3, 0.0), ellipse_wh=(3.2, 2.8)),
                ],
            },
            {
                "query_label": "Query 3\nlocation_46\ntrue: car",
                "labels": q3_labels,
                "panels": [
                    dict(rankme=0.69, pred="car", conf=0.77, true="car",
                         coords=q3_a, query=(0.0, 0.0), ellipse_xy=(-0.4, 0.2), ellipse_wh=(2.0, 2.7)),
                    dict(rankme=0.27, pred="truck", conf=0.24, true="car",
                         coords=q3_b, query=(0.0, 0.0), ellipse_xy=(0.8, -0.5), ellipse_wh=(2.1, 1.8)),
                    dict(rankme=0.52, pred="car", conf="--", true="car",
                         coords=q3_c, query=(0.0, 0.0), ellipse_xy=(0.8, -0.1), ellipse_wh=(1.6, 3.0)),
                ],
            },
        ],
        "scatter_b1": (x1, y1),
        "scatter_b2": (x2, y2),
        "scatter_c": (x3, y3),
        "table_rows": [
            {"teacher": "CLIP L/14 OpenAI", "pred_conf": "guitar (0.82)", "rankme": "0.76",
             "pred_color": GOOD_COLOR, "rankme_color": GOOD_COLOR},
            {"teacher": "DFN L/14 2B", "pred_conf": "piano (0.31)", "rankme": "0.28",
             "pred_color": BAD_COLOR, "rankme_color": BAD_COLOR},
            {"teacher": "ConvNeXt XXL", "pred_conf": "guitar (0.40)", "rankme": "0.36",
             "pred_color": GOOD_COLOR, "rankme_color": GOOD_COLOR},
        ],
    }


def make_full_figure(output_path: str | Path, image_path: str | Path):
    D = make_mock_data()

    fig = plt.figure(figsize=(17.5, 9.6))

    outer = GridSpec(
        2,
        1,
        height_ratios=[1.85, 1.35],
        hspace=0.30,
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.08,
        figure=fig,
    )

    # ========================================================
    # Panel (a)
    # ========================================================

    top = GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=outer[0],
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.10,
        hspace=0.18,
    )

    first_ax = None

    col_titles = ["Higher-RankMe teacher", "Lower-RankMe teacher", "Anchor space"]

    for qi, query in enumerate(D["queries"]):
        for pi, panel in enumerate(query["panels"]):
            ax = fig.add_subplot(top[qi, pi])

            if first_ax is None:
                first_ax = ax

            plot_neighborhood_panel(
                ax=ax,
                coords=panel["coords"],
                labels=query["labels"],
                query_xy=panel["query"],
                rankme_value=panel["rankme"],
                pred_class=panel["pred"],
                conf_value=panel["conf"],
                true_class=panel["true"],
                ellipse_xy=panel["ellipse_xy"],
                ellipse_wh=panel["ellipse_wh"],
            )

            if qi == 0:
                ax.set_title(col_titles[pi], fontsize=10.5, fontweight="bold", pad=10)

            if qi == 2:
                ax.set_xlabel("UMAP-1", fontsize=8.5)

            if pi == 0:
                ax.set_ylabel("UMAP-2", fontsize=8.5)
                ax.text(
                    -0.26,
                    0.50,
                    query["query_label"],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9.3,
                    fontweight="bold",
                )

    first_ax.text(
        -0.19,
        1.34,
        "(a) Local neighborhood structure in teacher feature spaces",
        transform=first_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
    )

    # Compact legend between rows
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="white",
                markeredgecolor="#333333",
                markersize=9,
                linewidth=0,
                label="Query sample",
            ),
            Line2D(
                [0],
                [0],
                color="#666666",
                linestyle="--",
                linewidth=1.2,
                label="kNN region",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.455),
        ncol=2,
        frameon=True,
        fontsize=9,
    )

    # ========================================================
    # Panels (b), (c), (d)
    # ========================================================

    bottom = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[1],
        width_ratios=[1.35, 1.25, 1.85],
        wspace=0.34,
    )

    bgrid = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=bottom[0],
        wspace=0.24,
    )

    ax_b1 = fig.add_subplot(bgrid[0])
    ax_b2 = fig.add_subplot(bgrid[1])

    x, y = D["scatter_b1"]
    plot_rankme_vs_metric(
        ax_b1,
        x,
        y,
        color="#7AA0D2",
        xlabel="RankMe",
        ylabel="Local accuracy",
        rho_text=r"$\rho = 0.62$",
    )

    x, y = D["scatter_b2"]
    plot_rankme_vs_metric(
        ax_b2,
        x,
        y,
        color="#F0A64B",
        xlabel="RankMe",
        ylabel="Confidence",
        rho_text=r"$\rho = 0.58$",
    )

    add_panel_title(ax_b1, "(b) RankMe tracks local reliability", y=1.10, fontsize=11.5)

    ax_c = fig.add_subplot(bottom[1])

    x, y = D["scatter_c"]
    plot_disagreement_gain(
        ax_c,
        x,
        y,
        color="#9B6CC3",
        rho_text=r"$\rho = 0.63$",
    )

    add_panel_title(ax_c, "(c) Gains increase with disagreement", y=1.10, fontsize=11.5)

    dgrid = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=bottom[2],
        width_ratios=[1.55, 0.75],
        height_ratios=[1.0, 0.78],
        wspace=0.24,
        hspace=0.20,
    )

    ax_d_table = fig.add_subplot(dgrid[0, 0])
    ax_d_box = fig.add_subplot(dgrid[1, 0])
    ax_d_img = fig.add_subplot(dgrid[:, 1])

    draw_prediction_table(ax_d_table, D["table_rows"], "Top teacher predictions")

    draw_ensemble_box(
        ax_d_box,
        uniform_pred=("piano ✗", False),
        rankme_pred=("guitar ✓", True),
    )

    draw_image_panel(ax_d_img, image_path=image_path, title="Query 1")

    add_panel_title(ax_d_table, "(d) Example correction", y=1.10, fontsize=11.5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"saved: {output_path}")


if __name__ == "__main__":
    make_full_figure(
        output_path="figure2_geometry_mockup_clean.png",
        image_path=r"C:\Users\racs2019\Downloads\guitar.jpg",
    )