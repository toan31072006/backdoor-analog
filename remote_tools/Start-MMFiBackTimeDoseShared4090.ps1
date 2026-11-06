<#
.SYNOPSIS
Approved handover: native BackTime P3S3 resume + one A1 worker (1% then5%).
.DESCRIPTION
Inspect/Status are read-only. SwitchAndLaunch transiently holds the exact
authenticated old P3S3 PID to verify/copy a checkpoint before stopping it, tests concurrent capacity,
then resumes P3S3 and starts the dose queue. Keep this terminal and Windows awake.
No process-name kills, checkpoint import, installation, automatic retry or AMP.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','Pause','Prepare','Launch','SwitchAndLaunch','Status')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$BackTimeRoot = 'U:\backdooranalog\runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_u_v1',
    [string]$Outdir = '',
    [string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$Runner = Join-Path $PSScriptRoot 'Run-MMFiBackTimeDoseShared4090.py'
if (!$PythonExe) { $PythonExe = Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
if (!$Outdir) { $Outdir = Join-Path (Split-Path -Parent $BackTimeRoot) 'mmfi_shared_backtime_dose_s42_4090_v1' }
foreach ($Value in @($DataHome,$BackTimeRoot,$Outdir,$PythonExe,$Runner)) {
    if (!$Value -or $Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') {
        throw 'Only ordinary local paths are allowed.'
    }
}
foreach ($Value in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing $Value; nothing installed/stopped." }
}
if ($Mode -in @('Pause','SwitchAndLaunch')) {
    'AUTHORIZED HANDOVER: verify P1S1; hold the exact P3S3 PID for backup; stop it only after validation.'
    'Only the unfinished epoch may replay. No scratch BackTime restart. No peer/process-name kill.'
}
if ($Mode -in @('Launch','SwitchAndLaunch')) {
    'FOREGROUND: keep terminal open and Windows awake. P3S3 resume + A1 P1S1 1% then5%, seed42.'
    'If a check fails, preserve logs/checkpoints; no automatic retry/fallback.'
}
$Phase = @{Inspect='inspect';Pause='pause';Prepare='prepare';Launch='launch';SwitchAndLaunch='switch-and-launch';Status='status'}[$Mode]
& $PythonExe '-B' '-u' $Runner '--mode' $Phase '--data-home' ([IO.Path]::GetFullPath($DataHome)) '--backtime-root' ([IO.Path]::GetFullPath($BackTimeRoot)) '--outdir' ([IO.Path]::GetFullPath($Outdir))
if ($LASTEXITCODE -ne 0) { throw "Shared $Mode failed. Preserve logs/checkpoints. No automatic retry or fresh BackTime fallback." }
