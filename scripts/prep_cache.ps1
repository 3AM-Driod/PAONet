# Build the after-network cache from scenes/ and bg/.
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not $env:CACHE) { $env:CACHE = Join-Path $Root 'data\after\cache1' }
if (-not $env:S_DIR) { $env:S_DIR = Join-Path $Root 'data\after\data1\scenes' }
if (-not $env:B_DIR) { $env:B_DIR = Join-Path $Root 'data\after\data1\bg' }

python -u code\after\prep_cache.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
