from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Data
# ============================================================

# Main results
df_main = pd.DataFrame({
    "method": [
        "LODO",
        "Conf. hard",
        "Global",
        "Conf. soft",
        "Uniform",
        "Δ λ=0.10",
        "Δ λ=0.15",
        "Δ λ=0.20",
        "Δ λ=0.30",
    ],
    "accuracy": [
        0.515184,
        0.521895,
        0.530669,
        0.543547,
        0.544335,
        0.548756,
        0.551305,
        0.553193,
        0.553467,
    ],
})

# Lambda sweep
df_lambda = pd.DataFrame({
    "lambda": [0.10, 0.15, 0.20, 0.30],
    "Loc 38": [0.952405, 0.951382, 0.951894, 0.950870],
    "Loc 43": [0.334080, 0.341791, 0.346020, 0.351990],
    "Loc 46": [0.281771, 0.285691, 0.290755, 0.289775],
    "Loc 100": [0.626768, 0.626358, 0.624103, 0.621234],
})

domain_cols = ["Loc 38", "Loc 43", "Loc 46", "Loc 100"]
df_lambda["Average"] = df_lambda[domain_cols].mean(axis=1)

# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))

# ------------------------------------------------------------
# Left panel: main results bar chart
# ------------------------------------------------------------
ax = axes[0]

bars = ax.bar(df_main["method"], df_main["accuracy"])

best_baseline_name = "Uniform"
best_method_name = "Δ λ=0.30"

for bar, method in zip(bars, df_main["method"]):
    if method == best_baseline_name:
        bar.set_hatch("//")
        bar.set_edgecolor("black")
        bar.set_linewidth(1.5)
    if method == best_method_name:
        bar.set_linewidth(2.0)

for bar, method, value in zip(bars, df_main["method"], df_main["accuracy"]):
    offset = 0.001
    if method == best_method_name:
        offset = 0.0015
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

uniform_baseline = float(df_main.loc[df_main["method"] == best_baseline_name, "accuracy"].iloc[0])
ax.axhline(
    uniform_baseline,
    linestyle="--",
    linewidth=1.8,
    label="Uniform baseline",
)

best_value = float(df_main.loc[df_main["method"] == best_method_name, "accuracy"].iloc[0])
gain = best_value - uniform_baseline
best_idx = df_main.index[df_main["method"] == best_method_name][0]

ax.annotate(
    f"+{gain:.3f} vs uniform",
    xy=(best_idx, best_value),
    xytext=(best_idx - 1.3, best_value + 0.0035),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
)

ax.set_ylabel("Average accuracy")
ax.set_title("(a) Main results")
ax.set_ylim(0.50, 0.56)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(frameon=True)
ax.tick_params(axis="x", rotation=25)

# ------------------------------------------------------------
# Right panel: lambda sweep
# ------------------------------------------------------------
ax = axes[1]

for col in domain_cols:
    ax.plot(
        df_lambda["lambda"],
        df_lambda[col],
        marker="o",
        linewidth=1.6,
        alpha=0.9,
        label=col,
    )

ax.plot(
    df_lambda["lambda"],
    df_lambda["Average"],
    marker="o",
    linewidth=3.0,
    label="Average",
)

best_idx = df_lambda["Average"].idxmax()
best_lambda = df_lambda.loc[best_idx, "lambda"]
best_avg = df_lambda.loc[best_idx, "Average"]

ax.annotate(
    f"best avg\nλ={best_lambda:.2f}, acc={best_avg:.3f}",
    xy=(best_lambda, best_avg),
    xytext=(best_lambda - 0.11, best_avg + 0.012),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
)

ax.set_xlabel("Deviation parameter $\\lambda$")
ax.set_ylabel("Accuracy")
ax.set_title("(b) Effect of deviation strength")
ax.set_xticks(df_lambda["lambda"])
ax.set_ylim(0.25, 0.65)
ax.grid(True, alpha=0.3)

handles, labels = ax.get_legend_handles_labels()
order = [labels.index("Average")] + [i for i, l in enumerate(labels) if l != "Average"]
ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    frameon=True,
)

plt.tight_layout()

# ============================================================
# Save
# ============================================================

outdir = Path("figures")
outdir.mkdir(parents=True, exist_ok=True)

png_path = outdir / "main_two_panel_figure.png"
pdf_path = outdir / "main_two_panel_figure.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print("Saved PNG to:", png_path.resolve())
print("Saved PDF to:", pdf_path.resolve())

plt.show()