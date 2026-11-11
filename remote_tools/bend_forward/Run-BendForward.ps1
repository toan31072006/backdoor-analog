[CmdletBinding()]
param(
    [ValidateSet('CheckOnly','Geometry','Prepare','Launch','Resume','Evaluate','Status','Export')]
    [string]$Mode = 'CheckOnly',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$OutputHome = 'U:\backdooranalog',
    [string]$Python,
    [string]$PiWDatasetRoot,
    [ValidateRange(1,9999)][int]$RunVersion = 1,
    [ValidateRange(0,2147483647)][int]$PeerPid = 0,
    [switch]$ApproveGeometry
)
$ErrorActionPreference = 'Stop'
$bendCheckout = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$bendSource = Join-Path $bendCheckout 'ATKBackd\run_bend_forward.py'
if (-not $Python) { $Python = Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $bendSource -PathType Leaf)) { throw "Source not found: $bendSource" }
if (-not (Test-Path -LiteralPath $OutputHome -PathType Container)) { throw "OutputHome not found: $OutputHome" }
if ($Mode -eq 'Prepare' -and -not $ApproveGeometry) {
    throw 'Inspect BOTH real TRAIN geometry_bend_forward.png files first; then -ApproveGeometry.'
}
$bendModes = @{CheckOnly='check-only'; Geometry='geometry'; Prepare='prepare'; Launch='launch';
    Resume='resume'; Evaluate='evaluate'; Status='status'; Export='export'}
Write-Host 'A1/WiFiDoor HPE-Li seed42; NEW forward trunk bend, NOT BackTime/old leg rotation.'
Write-Host 'ONE worker: MM-Fi (50 epochs) then PiW (200 epochs); no Clean retraining.'
Write-Host 'No job is stopped, suspended or retried. Use a separate checkout on U:.'
foreach ($bendDataset in @('mmfi','piw')) {
    $bendOut = Join-Path $OutputHome "runs\${bendDataset}_bend_forward_s42_v${RunVersion}"
    $bendArgs = @('-B','-u',$bendSource,'--dataset',$bendDataset,'--mode',$bendModes[$Mode],
        '--data-home',$DataHome,'--outdir',$bendOut,'--peer-pid',"$PeerPid")
    if ($bendDataset -eq 'piw' -and $PiWDatasetRoot) { $bendArgs += @('--dataset-root',$PiWDatasetRoot) }
    if ($ApproveGeometry) { $bendArgs += '--approve-geometry' }
    Write-Host "[$bendDataset] Mode=$Mode; output=$bendOut"
    & $Python @bendArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Bend $bendDataset/$Mode failed (exit $LASTEXITCODE). Queue stopped; preserve artifacts, no retry."
    }
}
