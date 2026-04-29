# Ensure we are using repo root
$root = (Resolve-Path ".").Path

$feature_root = "$root/features/domainnet"
$probe_root   = "$root/results"
$outdir       = "$root/stress_results"

$source = "quickdraw"

# Focus on hardest shifts (better paper figure)
$targets = @("sketch", "infograph")

# MUST match main experiments
$seeds = @(0,1,2,3,4)

$teachers = @(
    "openclip_l14_openai_qgelu",
    "openclip_b16_datacomp",
    "openclip_so400m_siglip",
    "openclip_l14_dfn2b",
    "openclip_h14_laion2b",
    "openclip_h14_378_dfn5b",
    "openclip_convnext_xxlarge"
)

$corruption = "0.0,0.1,0.2,0.3,0.4"

foreach ($seed in $seeds) {
    foreach ($target in $targets) {

        $out_csv = "$outdir/${source}_${target}_seed${seed}.csv"

        if (Test-Path $out_csv) {
            Write-Host "Skipping existing stress test: seed=$seed target=$target"
            continue
        }

        Write-Host "Running stress test: seed=$seed target=$target"

        python scripts/DomainNet/05_stress_test.py `
            --feature_root $feature_root `
            --probe_root $probe_root `
            --outdir $outdir `
            --source $source `
            --target $target `
            --seed $seed `
            --teachers @teachers `
            --corruption_rates $corruption `
            --k 20
    }
}