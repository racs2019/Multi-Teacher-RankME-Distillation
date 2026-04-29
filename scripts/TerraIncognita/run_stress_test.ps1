# Ensure we are using repo root
$root = (Resolve-Path ".").Path

$feature_root = "$root/features/terraincognita"
$probe_root   = "$root/results"
$outdir       = "$root/stress_results_terraincognita"

$source = "location_38"

# Best for paper: moderate + extreme shift
$targets = @("location_43", "location_46")

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

# Slightly extend corruption for stronger curve (optional but recommended)
$corruption = "0.0,0.1,0.2,0.3,0.4,0.5"

foreach ($seed in $seeds) {
    foreach ($target in $targets) {

        $out_csv = "$outdir/${source}_${target}_seed${seed}.csv"

        if (Test-Path $out_csv) {
            Write-Host "Skipping existing stress test: seed=$seed target=$target"
            continue
        }

        Write-Host "Running Terra stress test: seed=$seed target=$target"

        python scripts/DomainNet/05_stress_test.py `
            --feature_root $feature_root `
            --probe_root $probe_root `
            --outdir $outdir `
            --dataset "terraincognita" `
            --source $source `
            --target $target `
            --seed $seed `
            --teachers @teachers `
            --corruption_rates $corruption `
            --k 20
    }
}

Write-Host "=== Terra stress tests complete ==="