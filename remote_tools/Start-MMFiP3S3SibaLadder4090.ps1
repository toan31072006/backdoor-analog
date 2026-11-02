<#
.SYNOPSIS
Inspect or prepare source-adapted SIBA/BackTime and blocked LADDER on MM-Fi P3S3.
.DESCRIPTION
Foreground, at most one worker on cuda:0. Prepare stops after input audits and
native Clean B32 CUDA smoke tests. Only an explicit Launch may train a new
Clean surrogate, learn source-adapted triggers and train/evaluate scratch victims.
Requested retains LADDER and therefore fails Check/Launch until missing author
search material is obtained. No method is silently dropped. SibaBackTime is an
explicit two-method selection that forces BackTime's adaptive TRAIN recipients.
BackTime cannot use the superseded matched-uniform prototype in this wrapper.
No historical checkpoint is required or modified. No process is stopped.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','Preflight','Prepare','CheckOnly','Launch','Status')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [Alias('Python')][string]$PythonExe = '',
    [ValidateSet('cuda:0')][string]$Device = 'cuda:0',
    [ValidateSet(1)][int]$JobsPerGpu = 1,
    [ValidateSet('adaptive')][string]$BackTimePolicy = '',
    [ValidateSet('Requested','SibaBackTime')][string]$MethodSet = 'Requested'
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if ($MethodSet -eq 'SibaBackTime') { $BackTimePolicy = 'adaptive' }
if (!$Outdir) {
    $RunName = if ($BackTimePolicy) {
        'mmfi_p3s3_siba_ladder_backtime_adaptive_source_adapted_s42_4090_v2'
    } else {
        'mmfi_p3s3_siba_ladder_source_adapted_s42_4090_v2'
    }
    if ($MethodSet -eq 'SibaBackTime') {
        $RunName = $RunName.Replace('_s42_','_siba_backtime_only_s42_')
    }
    $Outdir = Join-Path $DataHome (Join-Path 'runs' $RunName)
}
$Outdir = [IO.Path]::GetFullPath($Outdir)
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
if (!$PythonExe) {
    throw 'No existing Python found; pass -PythonExe. No environment was installed.'
}
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
$Runner = Join-Path $Repo 'ATKBackd\run_mmfi_p3s3_siba_ladder_4090.py'
foreach ($Value in @($Repo,$DataHome,$Outdir,$PythonExe,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') {
        throw "Unsupported path: $Value"
    }
}
if ($Outdir.TrimEnd('\') -eq $DataHome.TrimEnd('\') -or
        $Outdir.TrimEnd('\') -eq $Repo.TrimEnd('\') -or
        $Outdir.TrimEnd('\') -eq [IO.Path]::GetPathRoot($Outdir).TrimEnd('\')) {
    throw 'Outdir must be a separate run folder, not DataHome, the checkout or a volume root.'
}
foreach ($Value in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing: $Value" }
}
$Common = @('-u',$Runner,'--data-home',$DataHome,'--outdir',$Outdir,
            '--device',$Device,'--jobs-per-gpu',[string]$JobsPerGpu)
if ($BackTimePolicy) { $Common += @('--backtime-policy',$BackTimePolicy) }
if ($MethodSet -eq 'SibaBackTime') { $Common += @('--methods','siba','backtime') }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

function Invoke-Runner([string]$Phase) {
    # Native invocation uses an argument array, including paths with spaces.
    & $PythonExe @Common '--mode' $Phase
    if ($LASTEXITCODE -ne 0) {
        throw "SIBA/LADDER $Phase failed (exit $LASTEXITCODE); no later phase was started by this invocation. Inspect logs before resuming."
    }
}

if ($Mode -eq 'Inspect') { Invoke-Runner 'inspect'; return }
if ($Mode -eq 'Preflight') { Invoke-Runner 'preflight'; return }
if ($Mode -eq 'Prepare') {
    Invoke-Runner 'prepare'
    'PREPARED_ONLY: audits + native Clean B32 CUDA smoke passed; no full Clean/victim training or full trigger search.'
    'Inspect the receipts, then invoke -Mode Launch separately.'
    return
}
if ($Mode -eq 'CheckOnly') { Invoke-Runner 'check'; return }
if ($Mode -eq 'Status') { Invoke-Runner 'status'; return }

# Check never prepares or starts training. Requested includes blocked LADDER;
# do not quietly remove it or dispatch the superseded experimental search.
if ($MethodSet -eq 'Requested') {
    'REQUESTED_METHOD_SET includes LADDER. Missing author search material must block Check/Launch before scientific work.'
}
# The runner rechecks its preparation binding and exclusive run/GPU ownership.
Invoke-Runner 'check'
'FOREGROUND: keep this SSH session open. At most one worker on cuda:0.'
if ($MethodSet -eq 'SibaBackTime') {
    'Launch runs NEW Clean surrogate50 -> SIBA full-TRAIN search200/B128 + scratch victim50 -> BackTime source-fit + scratch victim50.'
    'BackTime trains its OWN scratch attacker surrogate for100 epochs:11 clean warmup +89 alternating surrogate/generator epochs; no Clean checkpoint warm-start.'
} elseif ($BackTimePolicy) {
    'Requested order: NEW Clean surrogate -> SIBA -> BackTime adaptive -> LADDER. LADDER remains blocked; it is not silently skipped.'
} else {
    'Requested order: NEW Clean surrogate -> SIBA -> LADDER. LADDER remains blocked; it is not silently skipped.'
}
Push-Location -LiteralPath $Repo
try {
    Invoke-Runner 'launch'
} finally {
    Pop-Location
}
