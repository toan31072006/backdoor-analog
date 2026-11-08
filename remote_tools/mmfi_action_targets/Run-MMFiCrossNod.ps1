[CmdletBinding()]
param(
    [ValidateSet('CheckOnly','Geometry','Prepare','Launch','Resume','Evaluate','Status','Export')]
    [string]$Mode = 'CheckOnly',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir,
    [string]$Python,
    [string]$CrossAction,
    [string]$NodAction,
    [switch]$ApproveActionLabels,
    [switch]$ApproveGeometry
)
$ErrorActionPreference = 'Stop'
$taskCheckout = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskSource = Join-Path $taskCheckout 'ATKBackd\run_mmfi_action_targets.py'
if (-not $Python) { $Python = Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
if (-not $Outdir) {
    $Outdir = Join-Path $DataHome 'runs\mmfi_action_targets_cross_nod_s42_v2'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $taskSource -PathType Leaf)) { throw "Source not found: $taskSource" }
$taskModes = @{
    CheckOnly='check-only'; Geometry='geometry'; Prepare='prepare'; Launch='launch';
    Resume='resume'; Evaluate='evaluate'; Status='status'; Export='export'
}
$taskArgs = @('-B','-u',$taskSource,'--data-home',$DataHome,'--outdir',$Outdir,
              '--mode',$taskModes[$Mode])
if ($CrossAction) { $taskArgs += @('--cross-action',$CrossAction) }
if ($NodAction) { $taskArgs += @('--nod-action',$NodAction) }
if ($ApproveActionLabels) { $taskArgs += '--approve-action-labels' }
if ($ApproveGeometry) { $taskArgs += '--approve-geometry' }
Write-Host "Mode=$Mode; source=$taskSource"
Write-Host "Data home=$DataHome; output=$Outdir"
Write-Host 'Cross/Nod are NEW TARGETS, not the historical motion-prior sensitivity table.'
& $Python @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Cross/Nod $Mode failed with exit code $LASTEXITCODE. No automatic retry." }
