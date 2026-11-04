# Rebuild only static.npz for existing after-network cache entries.
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not $env:CACHE) { $env:CACHE = Join-Path $Root 'data\after\cache1' }

python -u code\after\prep_static.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
