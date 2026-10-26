param([switch]$Check)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$interpreter = 'python'
$settingsFile = Join-Path $PSScriptRoot 'settings.local.json'
if (Test-Path -LiteralPath $settingsFile) {
    $localSettings = Get-Content -Raw -LiteralPath $settingsFile | ConvertFrom-Json
    if ($localSettings.simulation_python) { $interpreter = $localSettings.simulation_python }
}
$action = if ($Check) { 'doctor' } else { 'gui' }
& $interpreter -X utf8 -B -m softcut_solution $action
exit $LASTEXITCODE
