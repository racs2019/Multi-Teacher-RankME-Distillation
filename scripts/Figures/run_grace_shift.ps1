$root = (Resolve-Path ".").Path

python "$root/scripts/figures/figure_grace_shift_regime.py" `
  --domainnet_summary "$root/final_results_summary/main_results_summary.csv" `
  --terra_summary "$root/final_results_summary_terraincognita/main_results_summary.csv" `
  --outdir "$root/figures"