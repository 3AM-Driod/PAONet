$ErrorActionPreference = 'Stop'
$P = Split-Path -Parent $PSScriptRoot
Push-Location $P
$env:PYTHONIOENCODING = 'utf-8'
try {
    if (-not $env:F)       { $env:F = '192' }
    if (-not $env:NLAYER)  { $env:NLAYER = '6' }
    if (-not $env:NCROSS)  { $env:NCROSS = '2' }
    python code\after\train_after.py
} finally {
    Pop-Location
}
