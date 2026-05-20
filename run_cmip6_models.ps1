$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
Set-Location $repoRoot

$models = @("BCC-CSM2-MR", "MRI-ESM2-0")

foreach ($model in $models) {
    Write-Host "Running LCTR_computation.py for $model ..."
    $env:LCTR_CLIMATE_MODEL = $model
    python LCTR_computation.py
}

Remove-Item Env:\LCTR_CLIMATE_MODEL -ErrorAction SilentlyContinue
Write-Host "Done."
