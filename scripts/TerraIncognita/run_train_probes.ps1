# Always run from repo root
$root = (Resolve-Path ".").Path

$feature_root = "$root/features/terraincognita"
$outdir       = "$root/results"

$source = "location_38"
$targets = @("location_38", "location_43", "location_46", "location_100")

$seeds = @(0, 1, 2, 3, 4)

$teachers = @(
    "openclip_l14_openai_qgelu",
    "openclip_b16_datacomp",
    "openclip_so400m_siglip",
    "openclip_l14_dfn2b",
    "openclip_h14_laion2b",
    "openclip_h14_378_dfn5b",
    "openclip_convnext_xxlarge"
)

Write-Host "=== Training probes for TerraIncognita ==="

python scripts/DomainNet/02_train_probes.py `
    --feature_root $feature_root `
    --outdir $outdir `
    --dataset "terraincognita" `
    --source $source `
    --targets $targets `
    --teachers $teachers `
    --seeds $seeds `
    --train_split "train" `
    --test_split "test" `
    --probe_C 1.0 `
    --probe_max_iter 2000 `
    --skip_existing

Write-Host "=== Done ==="