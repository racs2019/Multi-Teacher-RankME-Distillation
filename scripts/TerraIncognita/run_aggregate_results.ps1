$root = (Resolve-Path ".").Path

python "$root/scripts/DomainNet/04_aggregate_results.py" `
  --results_dir "$root/final_results_terraincognita" `
  --probe_csv "$root/results/terraincognita/location_38/linear_probe_results.csv" `
  --outdir "$root/final_results_summary_terraincognita"