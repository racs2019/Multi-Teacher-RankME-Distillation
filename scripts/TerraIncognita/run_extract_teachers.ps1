# Always run from repo root
$root = (Resolve-Path ".").Path

$manifest = "$root/data/terra_incognita/master_manifest.csv"
$outdir   = "$root/features/terraincognita"

$domains = @("location_38", "location_43", "location_46", "location_100")
$splits  = @("train", "test")

$teachers = @(
    "openclip_l14_openai_qgelu",
    "openclip_b16_datacomp",
    "openclip_so400m_siglip",
    "openclip_l14_dfn2b",
    "openclip_h14_laion2b",
    "openclip_h14_378_dfn5b",
    "openclip_convnext_xxlarge"
)

foreach ($d in $domains) {
    foreach ($split in $splits) {
        foreach ($t in $teachers) {

            $outfile = "$outdir/$d/$split/$t.npz"

            if (Test-Path $outfile) {
                Write-Host "Skipping existing: domain=$d split=$split teacher=$t"
                continue
            }

            Write-Host "Running Terra extraction: domain=$d split=$split teacher=$t"

            python scripts/DomainNet/01_extract_teachers.py `
                --manifest_csv $manifest `
                --domain $d `
                --split $split `
                --outdir $outdir `
                --teacher $t `
                --batch_size 64 `
                --num_workers 4 `
                --device cuda `
                --skip_existing
        }
    }
}

Write-Host "=== Terra extraction complete ==="