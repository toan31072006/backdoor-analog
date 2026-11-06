<#
.SYNOPSIS
Replace the completed dose slot with BppAttack-CSI P3S3/HPE-Li/seed42.
.DESCRIPTION
Inspect is read-only. Explicit SwitchAndLaunch validates and backs up the exact
old BackTime worker, pauses/resumes it through a new two-role controller, tests
real concurrent CUDA capacity, then launches Bpp scratch50 alongside P3S3 resume.
Never stops arbitrary jobs, auto-retries, installs, changes batches or imports victims.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','Pause','Prepare','Launch','SwitchAndLaunch','RecoverAndLaunch','Status')]
    [string]$Mode='Inspect',
    [string]$DataHome=(Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$BackTimeRoot='U:\backdooranalog\runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_u_v1',
    [string]$Outdir='',
    [string]$PythonExe=''
)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
if (!$PythonExe) { $PythonExe=Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
if (!$Outdir) { $Outdir=Join-Path (Split-Path -Parent $BackTimeRoot) 'mmfi_shared_backtime_bpp_p3s3_s42_4090_v1' }
$Runner=Join-Path $PSScriptRoot 'Run-MMFiBackTimeBppShared4090.py'
foreach ($Value in @($DataHome,$BackTimeRoot,$Outdir,$PythonExe,$Runner)) {
    if (!$Value -or $Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw 'Use ordinary local paths.' }
}
foreach ($Value in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing $Value; nothing stopped/installed." }
}
if ($Mode -in @('Pause','SwitchAndLaunch')) {
    'EXPLICIT HANDOVER: hold ONLY the authenticated P3S3 worker to validate/copy its native checkpoint.'
    'Stop only after backup validation; failure resumes the original process. P1S1/dose results stay untouched.'
}
if ($Mode -eq 'RecoverAndLaunch') {
    'EXPLICIT RECOVERY of failed8551376 Prepare only: signal owned smokes to FAIL and drain; no process kill.'
    'Preserve failed plan/logs/pause/backups, re-Prepare with corrected roles, then resume P3S3 and launch Bpp.'
}
if ($Mode -in @('Launch','SwitchAndLaunch','RecoverAndLaunch')) {
    'FOREGROUND: P3S3 resumes; Bpp P3S3/seed42 scratch50/final49. Keep terminal and Windows awake.'
}
$Phase=@{Inspect='inspect';Pause='pause';Prepare='prepare';Launch='launch';SwitchAndLaunch='switch-and-launch';RecoverAndLaunch='recover-and-launch';Status='status'}[$Mode]
& $PythonExe '-B' '-u' $Runner '--mode' $Phase '--data-home' ([IO.Path]::GetFullPath($DataHome)) '--backtime-root' ([IO.Path]::GetFullPath($BackTimeRoot)) '--outdir' ([IO.Path]::GetFullPath($Outdir))
if ($LASTEXITCODE -ne 0) { throw "Bpp shared $Mode failed; preserve logs/checkpoints. No automatic retry." }
