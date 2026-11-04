param(
    [string]$Ckpt = 'weights\formal\after\after_net.pt',
    [string]$Tag  = 'formal_after'
)
$ErrorActionPreference = 'Stop'
$P = Split-Path -Parent $PSScriptRoot
Push-Location $P
$env:PYTHONIOENCODING = 'utf-8'
try {
    python code\after\eval_after_arm.py --ckpt $Ckpt --tag $Tag
} finally {
    Pop-Location
}
