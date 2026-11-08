[CmdletBinding()]
param(
    [ValidateSet('CheckOnly','Geometry','Prepare','Launch','Resume','Evaluate','Status','Export')]
    [string]$Mode = 'CheckOnly',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir,
    [string]$Python,
    [string]$PiWDatasetRoot,
    [string]$CrossAction,
    [string]$NodAction,
    [switch]$ApproveActionLabels,
    [switch]$ApproveGeometry
)
$ErrorActionPreference = 'Stop'
$taskCheckout = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskSource = Join-Path $taskCheckout 'ATKBackd\run_action_targets_parallel.py'
if (-not $Python) { $Python = Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
if (-not $Outdir) {
    $taskOutputHome = $DataHome
    if (Test-Path -LiteralPath 'U:\' -PathType Container) { $taskOutputHome = 'U:\backdooranalog' }
    $Outdir = Join-Path $taskOutputHome 'runs\cross_nod_mmfi_piw_s42_2jobs_v2'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $taskSource -PathType Leaf)) { throw "Source not found: $taskSource" }
$taskModes = @{
    CheckOnly='check-only'; Geometry='geometry'; Prepare='prepare'; Launch='launch';
    Resume='resume'; Evaluate='evaluate'; Status='status'; Export='export'
}
$taskArgs = @('-B','-u',$taskSource,'--data-home',$DataHome,'--outdir',$Outdir,'--mode',$taskModes[$Mode])
if ($PiWDatasetRoot) { $taskArgs += @('--piw-dataset-root',$PiWDatasetRoot) }
if ($CrossAction) { $taskArgs += @('--cross-action',$CrossAction) }
if ($NodAction) { $taskArgs += @('--nod-action',$NodAction) }
if ($ApproveActionLabels) { $taskArgs += '--approve-action-labels' }
if ($ApproveGeometry) { $taskArgs += '--approve-geometry' }
Write-Host "Cross/Nod NEW action targets; HPE-Li; Mode=$Mode"
Write-Host "DataHome=$DataHome; output=$Outdir; Python=$Python"
Write-Host '2 MM-Fi jobs first, then 2 PiW jobs on cuda:0; EACH job evaluates immediately after fixed-final training.'
& $Python @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Cross/Nod $Mode failed, exit $LASTEXITCODE. Preserve logs/checkpoints; no automatic retry." }
