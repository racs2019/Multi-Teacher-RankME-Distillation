$root = (Resolve-Path ".").Path

python scripts/DomainNet/04_aggregate_results.py `
  --results_dir final_results/real `
  --probe_csv results/domainnet/real/linear_probe_results.csv `
  --outdir final_results_summary_domainnet