$root = Get-Location
$feature_root = "$root/features/domainnet"
$outdir       = "$root/results"

$source = "infograph"
$targets = @("infograph", "quickdraw", "real", "sketch")

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

python scripts/DomainNet/02_train_probes.py `
    --feature_root $feature_root `
    --outdir $outdir `
    --dataset "domainnet" `
    --source $source `
    --targets $targets `
    --teachers $teachers `
    --seeds $seeds `
    --train_split "train" `
    --test_split "test" `
    --probe_C 1.0 `
    --probe_max_iter 2000 `
    --skip_existing