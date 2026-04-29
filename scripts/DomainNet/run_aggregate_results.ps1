$root = (Resolve-Path ".").Path

python scripts/DomainNet/04_aggregate_results.py `
    --results_dir "$root/final_results" `
    --outdir "$root/final_results_summary"