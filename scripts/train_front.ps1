$ErrorActionPreference = 'Stop'
$P = Split-Path -Parent $PSScriptRoot
Push-Location $P
$env:PYTHONIOENCODING = 'utf-8'
try {
    python code\front\train_front.py
} finally {
    Pop-Location
}
