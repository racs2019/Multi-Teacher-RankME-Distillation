# Ensure script runs from repo root
$root = (Resolve-Path ".").Path

$feature_root = "$root/features/domainnet"
$probe_root   = "$root/results"
$outdir       = "$root/final_results"

$source = "infograph"
$targets = @("real", "sketch", "infograph", "quickdraw")
$seeds = @(0)

$teachers = @(
    "openclip_l14_openai_qgelu",
    "openclip_b16_datacomp",
    "openclip_so400m_siglip",
    "openclip_l14_dfn2b",
    "openclip_h14_laion2b",
    "openclip_h14_378_dfn5b",
    "openclip_convnext_xxlarge"
)

foreach ($seed in $seeds) {
    foreach ($target in $targets) {

        $out_csv = "$outdir/${source}_${target}_seed${seed}.csv"

        if (Test-Path $out_csv) {
            Write-Host "Skipping existing: seed=$seed target=$target"
            continue
        }

        Write-Host "Running seed=$seed target=$target"

        python scripts/DomainNet/03_run_methods.py `
            --feature_root $feature_root `
            --probe_root $probe_root `
            --outdir $outdir `
            --source $source `
            --target $target `
            --seed $seed `
            --teachers @teachers
    }
}