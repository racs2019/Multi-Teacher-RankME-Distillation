# Run from repo root
$root = (Resolve-Path ".").Path

$script = "$root/scripts/figures/figure_rank_instability.py"

$terra_tidy = "$root/terra_probe_plots/linear_probe_results_tidy.csv"
$terra_instability = "$root/terra_probe_plots/ranking_instability_train_location_38_accuracy.csv"

$domainnet_tidy = "$root/domainnet_probe_plots/linear_probe_results_tidy.csv"
$domainnet_instability = "$root/domainnet_probe_plots/ranking_instability_train_quickdraw_accuracy.csv"

$outdir = "$root/figures"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

python $script `
  --terra_tidy_csv $terra_tidy `
  --terra_instability_csv $terra_instability `
  --terra_train_domain "location_38" `
  --domainnet_tidy_csv $domainnet_tidy `
  --domainnet_instability_csv $domainnet_instability `
  --domainnet_train_domain "quickdraw" `
  --metric "accuracy" `
  --out_path "$outdir/figure_rank_instability.png"

python $script `
  --terra_tidy_csv $terra_tidy `
  --terra_instability_csv $terra_instability `
  --terra_train_domain "location_38" `
  --domainnet_tidy_csv $domainnet_tidy `
  --domainnet_instability_csv $domainnet_instability `
  --domainnet_train_domain "quickdraw" `
  --metric "accuracy" `
  --out_path "$outdir/figure_rank_instability.pdf"