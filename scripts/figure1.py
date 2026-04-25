from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    "lambda": [0.10, 0.15, 0.20, 0.30],
    "location_38": [0.952405, 0.951382, 0.951894, 0.950870],
    "location_43": [0.334080, 0.341791, 0.346020, 0.351990],
    "location_46": [0.281771, 0.285691, 0.290755, 0.289775],
    "location_100": [0.626768, 0.626358, 0.624103, 0.621234],
})

domain_cols = ["location_38", "location_43", "location_46", "location_100"]
df["average"] = df[domain_cols].mean(axis=1)

fig, ax = plt.subplots(figsize=(8.5, 5.5))

for col in domain_cols:
    ax.plot(
        df["lambda"],
        df[col],
        marker="o",
        linewidth=1.6,
        alpha=0.9,
        label=col,
    )

ax.plot(
    df["lambda"],
    df["average"],
    marker="o",
    linewidth=3.2,
    label="average",
)

best_idx = df["average"].idxmax()
best_lambda = df.loc[best_idx, "lambda"]
best_avg = df.loc[best_idx, "average"]

ax.annotate(
    f"best avg: λ={best_lambda:.2f}\nacc={best_avg:.3f}",
    xy=(best_lambda, best_avg),
    xytext=(best_lambda + 0.015, best_avg - 0.02),
    arrowprops=dict(arrowstyle="->"),
)

ax.set_xlabel("Deviation parameter $\\lambda$")
ax.set_ylabel("Accuracy")
ax.set_title("RankMe-guided deviation improves accuracy under domain shift")
ax.set_xticks(df["lambda"])
ax.legend(frameon=True)
ax.grid(True)

plt.tight_layout()

outdir = Path("figures")
outdir.mkdir(parents=True, exist_ok=True)

png_path = outdir / "lambda_sweep_annotated.png"
pdf_path = outdir / "lambda_sweep_annotated.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print("Saved PNG to:", png_path.resolve())
print("Saved PDF to:", pdf_path.resolve())

plt.show()