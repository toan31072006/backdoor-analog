[CmdletBinding()]
param(
    [ValidateSet('CheckOnly','Geometry','Prepare','Launch','Resume','Evaluate','Status','Export')]
    [string]$Mode = 'CheckOnly',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$DatasetRoot,
    [string]$Outdir,
    [string]$Python,
    [string]$CrossAction,
    [string]$NodAction,
    [switch]$ApproveActionLabels,
    [switch]$ApproveGeometry
)
$ErrorActionPreference = 'Stop'
$taskCheckout = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskSource = Join-Path $taskCheckout 'ATKBackd\run_piw_action_targets.py'
if (-not $Python) { $Python = Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
if (-not $Outdir) { $Outdir = Join-Path $DataHome 'runs\piw_action_targets_cross_nod_s42_v1' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $taskSource -PathType Leaf)) { throw "Source not found: $taskSource" }
$taskModes = @{
    CheckOnly='check-only'; Geometry='geometry'; Prepare='prepare'; Launch='launch';
    Resume='resume'; Evaluate='evaluate'; Status='status'; Export='export'
}
$taskArgs = @('-B','-u',$taskSource,'--data-home',$DataHome,'--outdir',$Outdir,'--mode',$taskModes[$Mode])
if ($DatasetRoot) { $taskArgs += @('--dataset-root',$DatasetRoot) }
if ($CrossAction) { $taskArgs += @('--cross-action',$CrossAction) }
if ($NodAction) { $taskArgs += @('--nod-action',$NodAction) }
if ($ApproveActionLabels) { $taskArgs += '--approve-action-labels' }
if ($ApproveGeometry) { $taskArgs += '--approve-geometry' }
Write-Host "PiW new action targets; Mode=$Mode; source=$taskSource"
Write-Host "DataHome=$DataHome; output=$Outdir"
Write-Host 'Cross -> Nod, serial; 200 epochs EACH; no historical result is overwritten.'
& $Python @taskArgs
if ($LASTEXITCODE -ne 0) { throw "PiW Cross/Nod $Mode failed, exit $LASTEXITCODE. No automatic retry." }
