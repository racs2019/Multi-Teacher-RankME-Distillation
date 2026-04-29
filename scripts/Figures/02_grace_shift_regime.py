from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd


ANCHOR_METHOD = "agreement_weighted"
GRACE_METHOD = "grace"


def load_shift_data(summary_csv: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = df[df["dataset"] == dataset].copy()

    anchor = df[df["method"] == ANCHOR_METHOD][
        ["target", "acc_mean"]
    ].rename(columns={"acc_mean": "anchor_acc"})

    grace = df[df["method"] == GRACE_METHOD][
        ["target", "acc_mean"]
    ].rename(columns={"acc_mean": "grace_acc"})

    out = anchor.merge(grace, on="target", how="inner")
    out["grace_gain"] = out["grace_acc"] - out["anchor_acc"]

    if out.empty:
        raise RuntimeError(f"No usable rows for dataset={dataset}")

    return out.sort_values("anchor_acc", ascending=False)


def draw_panel(ax, df, title):
    ax.plot(
        df["anchor_acc"],
        df["grace_gain"],
        marker="o",
        linewidth=2.2,
    )

    for _, row in df.iterrows():
        ax.text(
            row["anchor_acc"],
            row["grace_gain"],
            f" {row['target']}",
            fontsize=8,
            va="center",
        )

    ax.axhline(0, linestyle="--", linewidth=1.0)

    # Automatically shade middle 50% of anchor accuracies.
    q25 = df["anchor_acc"].quantile(0.25)
    q75 = df["anchor_acc"].quantile(0.75)
    ax.axvspan(q25, q75, alpha=0.12)

    ax.text(0.10, 0.94, "easy / in-domain", transform=ax.transAxes, fontsize=9, va="top")
    ax.text(0.52, 0.94, "moderate shift", transform=ax.transAxes, fontsize=9, va="top")
    ax.text(0.82, 0.94, "extreme shift", transform=ax.transAxes, fontsize=9, va="top")

    ax.set_title(title)
    ax.set_xlabel("Anchor accuracy (higher = easier target)")
    ax.grid(True, alpha=0.25)
    ax.invert_xaxis()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domainnet_summary", default="final_results_summary/main_results_summary.csv")
    parser.add_argument("--terra_summary", default=None)
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    panels = []

    if args.terra_summary is not None:
        terra = load_shift_data(Path(args.terra_summary), "terraincognita")
        panels.append(("TerraIncognita", terra))

    domainnet = load_shift_data(Path(args.domainnet_summary), "domainnet")
    panels.append(("DomainNet", domainnet))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.2), sharey=True)

    if len(panels) == 1:
        axes = [axes]

    for ax, (title, df) in zip(axes, panels):
        draw_panel(ax, df, title)

    axes[0].set_ylabel("GRACE gain over anchor")

    fig.suptitle(
        "GRACE gains peak in moderate-shift regimes and vanish at extremes",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png = outdir / "figure_grace_shift_regime.png"
    pdf = outdir / "figure_grace_shift_regime.pdf"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")

    print("Saved:", png)
    print("Saved:", pdf)


if __name__ == "__main__":
    main()