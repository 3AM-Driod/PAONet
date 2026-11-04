param(
    [string]$Front = 'weights\formal\front\front_net.pt',
    [string]$After = 'weights\formal\after\after_net.pt',
    [string]$Tag   = 'formal'
)
$ErrorActionPreference = 'Stop'
$P = Split-Path -Parent $PSScriptRoot
Push-Location $P
$env:PYTHONIOENCODING = 'utf-8'
try {
    $env:FRONT_W = $Front
    $env:AFTER_W = $After
    $env:TAG = $Tag
    python code\eval_chain.py
} finally {
    Pop-Location
}
