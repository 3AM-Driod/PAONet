# Evaluate all ablation checkpoints.
param(
    [string]$Kind = 'after',
    [string]$Only = '',
    [int]$Limit = 0
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$args = @('code\eval_ablations.py', '--kind', $Kind)
if ($Only) { $args += @('--only', $Only) }
if ($Limit -gt 0) { $args += @('--limit', [string]$Limit) }
python -u @args
exit $LASTEXITCODE
