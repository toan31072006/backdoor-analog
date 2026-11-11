[CmdletBinding()]
param(
    [ValidateSet('CheckOnly','GpuCheck','Geometry','Prepare','Launch','Resume','Evaluate','Status','Export','RunApproved')]
    [string]$Mode = 'CheckOnly',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$OutputHome = 'U:\backdooranalog',
    [string]$Python,
    [string]$PiWDatasetRoot,
    [ValidateRange(3,9999)][int]$RunVersion = 3,
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
if ($Mode -in @('Prepare','RunApproved') -and -not $ApproveGeometry) {
    throw 'Explicit -ApproveGeometry required for the UNCHANGED payload already inspected on BOTH datasets.'
}
if ($Mode -eq 'RunApproved') {
    # Carry prior visual approval ONLY for these exact, unchanged label operators.
    $bendApprovedSources = @{
        'ATKBackd\attack\bend_forward.py' = '1855A58C93D46A09D501937BA4068D4412051F0D484EEFAF190E0ABC7C95B6E8'
        'ATKBackd\attack\action_targets.py' = 'FC4687F823240A9E2D0261F228A982B07A43B32D31E4565A017180A102AEFCA7'
    }
    foreach ($bendMember in $bendApprovedSources.Keys) {
        if ((Get-FileHash -LiteralPath (Join-Path $bendCheckout $bendMember) -Algorithm SHA256).Hash -ne $bendApprovedSources[$bendMember]) {
            throw 'Approved payload source changed; inspect new geometry instead of carrying old approval.'
        }
    }
    foreach ($bendDataset in @('mmfi','piw')) {
        $bendFresh = Join-Path $OutputHome "runs\${bendDataset}_bend_forward_s42_v${RunVersion}"
        if (Test-Path -LiteralPath $bendFresh) {
            throw "Fresh first launch only: output already exists: $bendFresh. Preserve it; no automatic retry/resume."
        }
    }
}
$bendModes = @{CheckOnly='check-only'; GpuCheck='guard-check'; Geometry='geometry'; Prepare='prepare'; Launch='launch';
    Resume='resume'; Evaluate='evaluate'; Status='status'; Export='export'}
Write-Host 'A1/WiFiDoor HPE-Li seed42; NEW forward trunk bend, NOT BackTime/old leg rotation.'
Write-Host 'v3: WDDM guard repair ONLY; approved PiW -40d / MM-Fi +40d payloads unchanged. Preserve v1/v2.'
Write-Host 'ONE worker: MM-Fi (50 epochs) then PiW (200 epochs); no Clean retraining.'
Write-Host 'No job is stopped, suspended or retried. Use a separate checkout on U:.'
$bendStages = if ($Mode -eq 'RunApproved') { @('GpuCheck','CheckOnly','Geometry','Prepare','Launch') } else { @($Mode) }
foreach ($bendStage in $bendStages) {
foreach ($bendDataset in @('mmfi','piw')) {
    if ($bendStage -eq 'GpuCheck' -and $bendDataset -eq 'piw') { continue }
    $bendOut = Join-Path $OutputHome "runs\${bendDataset}_bend_forward_s42_v${RunVersion}"
    $bendArgs = @('-B','-u',$bendSource,'--dataset',$bendDataset,'--mode',$bendModes[$bendStage],
        '--data-home',$DataHome,'--outdir',$bendOut,'--peer-pid',"$PeerPid")
    if ($bendDataset -eq 'piw' -and $PiWDatasetRoot) { $bendArgs += @('--dataset-root',$PiWDatasetRoot) }
    if ($ApproveGeometry) { $bendArgs += '--approve-geometry' }
    Write-Host "[$bendDataset] Mode=$bendStage; output=$bendOut"
    & $Python @bendArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Bend $bendDataset/$bendStage failed (exit $LASTEXITCODE). Queue stopped; preserve artifacts, no retry."
    }
}
}
