import pandas as pd
from pathlib import Path

base_dir = Path("terra_probe_results/rankme_delta_sweep/location_38")

rows = []

for lambda_dir in base_dir.glob("lambda_*"):
    lam = float(lambda_dir.name.split("_")[1])

    for target_dir in lambda_dir.iterdir():
        if not target_dir.is_dir():
            continue

        csv_files = list(target_dir.glob("rankme_result_*.csv"))
        if not csv_files:
            continue

        df = pd.read_csv(csv_files[0])

        rows.append({
            "lambda": lam,
            "target_domain": df["target_domain"].iloc[0],
            "accuracy": df["accuracy"].iloc[0],
            "balanced_accuracy": df["balanced_accuracy"].iloc[0],
        })

df_all = pd.DataFrame(rows)

# Per-lambda averages
df_avg = df_all.groupby("lambda").mean(numeric_only=True).reset_index()

print("\n=== Per-domain results ===")
print(df_all.sort_values(["lambda", "target_domain"]))

print("\n=== Average across targets ===")
print(df_avg.sort_values("accuracy", ascending=False))

# Save
out_path = base_dir / "rankme_lambda_summary.csv"
df_all.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")