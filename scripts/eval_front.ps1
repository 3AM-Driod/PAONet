param(
    [string]$Ckpt = 'weights\formal\front\front_net.pt',
    [string]$Data = 'data\front\data1',
    [string]$Tag  = 'formal_front'
)
$ErrorActionPreference = 'Stop'
$P = Split-Path -Parent $PSScriptRoot
Push-Location $P
$env:PYTHONIOENCODING = 'utf-8'
try {
    python code\front\eval_front_arm.py --ckpt $Ckpt --data $Data --tag $Tag
} finally {
    Pop-Location
}
