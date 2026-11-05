<#
.SYNOPSIS
Fresh BackTime-only HPE-Li P1S1 and P3S3, two jobs on Van's RTX4090.
.DESCRIPTION
Prepare stops after full input audits and concurrent CUDA B64/B32 capacity
checks. Launch is separate and foreground: keep SSH open. No install, pull,
kill, automatic retry, checkpoint import, batch shrink or schedule reduction.
Outdir may be on another local disk; DataHome still supplies dataset/env/prior.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','Preflight','Prepare','CheckOnly','Launch','Status')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [Alias('Python')][string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$Outdir) {
    $Outdir = Join-Path $DataHome 'runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_v1'
}
if (!$PythonExe) {
    foreach ($Candidate in @(
        (Join-Path $DataHome 'envs\dose-backdoor\python.exe'),
        (Join-Path $DataHome 'envs\mmfi-p3\Scripts\python.exe')
    )) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $PythonExe = $Candidate
            break
        }
    }
}
if (!$PythonExe) { throw 'Use -PythonExe with the existing MM-Fi environment. Nothing installed.' }
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
$Outdir = [IO.Path]::GetFullPath($Outdir)
$Runner = Join-Path $Repo 'remote_tools\Run-MMFiBackTimeTwoSplits4090.py'
foreach ($Value in @($Repo,$DataHome,$Outdir,$PythonExe,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') {
        throw 'Use ordinary local paths; UNC/quotes/newlines are unsupported.'
    }
}
foreach ($Value in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing: $Value" }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$Phase = @{Inspect='inspect';Preflight='preflight';Prepare='prepare';CheckOnly='check';Launch='launch';Status='status'}[$Mode]
if ($Mode -eq 'Launch') {
    'FOREGROUND: keep SSH open and Windows awake. Two owned split workers, no guarantee across logout/reboot.'
    'Each split: NEW Clean50 -> BackTime attacker100/B64 -> NEW victim50/B32 -> six-dose controls.'
    'No SIBA/LADDER, old3090 checkpoint or other job is touched.'
}
& $PythonExe '-B' '-u' $Runner '--mode' $Phase '--data-home' $DataHome '--outdir' $Outdir
if ($LASTEXITCODE -ne 0) { throw "BackTime $Mode failed. Preserve logs/checkpoints; no automatic retry." }
