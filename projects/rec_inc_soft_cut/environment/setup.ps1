[CmdletBinding()]
param([switch]$Cuda)
$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program" }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentDir = Join-Path $projectRoot '.venv'
$python = Join-Path $environmentDir 'Scripts/python.exe'
$launcher = (Get-Command py -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $python)) {
    Invoke-Checked $launcher @('-3.11', '-m', 'venv', $environmentDir)
}
Invoke-Checked $python @('-m', 'pip', 'install', '--requirement', (Join-Path $PSScriptRoot 'requirements-core.txt'))
$torchIndex = if ($Cuda) { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
Invoke-Checked $python @('-m', 'pip', 'install', 'torch==2.11.0', '--index-url', $torchIndex)
Invoke-Checked $python @(Join-Path $PSScriptRoot 'verify_environment.py')
