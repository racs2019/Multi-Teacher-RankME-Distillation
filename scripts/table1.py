from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Main results table
# -----------------------------
df = pd.DataFrame({
    "method": [
        "LODO teacher",
        "Global best teacher",
        "Confidence hard",
        "Confidence soft",
        "Uniform ensemble",
        "RankMe Δ (λ=0.10)",
        "RankMe Δ (λ=0.15)",
        "RankMe Δ (λ=0.20)",
        "RankMe Δ (λ=0.30)",
    ],
    "accuracy": [
        0.515184,
        0.530669,
        0.521895,
        0.543547,
        0.544335,
        0.548756,
        0.551305,
        0.553193,
        0.553467,
    ],
})

# Sort by accuracy for cleaner figure
df = df.sort_values("accuracy", ascending=True).reset_index(drop=True)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.bar(df["method"], df["accuracy"])

# Highlight strongest baseline and best overall
best_baseline_name = "Uniform ensemble"
best_method_name = "RankMe Δ (λ=0.30)"

for bar, method in zip(bars, df["method"]):
    if method == best_baseline_name:
        bar.set_hatch("//")
    if method == best_method_name:
        bar.set_linewidth(2)

# Annotate bars
for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.001,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        rotation=0,
    )

# Baseline reference line
uniform_baseline = float(df.loc[df["method"] == best_baseline_name, "accuracy"].iloc[0])
ax.axhline(
    uniform_baseline,
    linestyle="--",
    linewidth=1.8,
    label="Uniform ensemble",
)

ax.set_ylabel("Average accuracy")
ax.set_title("Main results: RankMe delta improves over strong baselines")
ax.set_ylim(0.50, 0.56)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(frameon=True)

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

outdir = Path("figures")
outdir.mkdir(parents=True, exist_ok=True)

png_path = outdir / "main_results_bar.png"
pdf_path = outdir / "main_results_bar.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print("Saved PNG to:", png_path.resolve())
print("Saved PDF to:", pdf_path.resolve())

plt.show()