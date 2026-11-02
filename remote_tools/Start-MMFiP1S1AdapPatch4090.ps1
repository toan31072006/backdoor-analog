<#
.SYNOPSIS
Run one Adap-Patch CSI/pose adaptation on Van's RTX 4090.
.DESCRIPTION
HPE-Li / MM-Fi P1S1 / seed42 only. Prepare is separate from full training.
Launch is foreground: keep SSH open. No process kill, task registration,
package installation, automatic retry, or concurrency override is provided.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','DryRun','Prepare','CheckOnly','Launch','Status')]
    [string]$Mode = 'Status',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [Alias('Python')][string]$PythonExe = '',
    [ValidateSet('cuda:0')][string]$Device = 'cuda:0',
    [switch]$AllowConcurrentP3S3
)

$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
$Runs = [IO.Path]::GetFullPath((Join-Path $DataHome 'runs'))
if (!$Outdir) {
    $Outdir = Join-Path $Runs 'mmfi_p1s1_hpeli_adap_patch_s42_4090_v1'
}
$Outdir = [IO.Path]::GetFullPath($Outdir)
$Runner = Join-Path $Repo 'ATKBackd\run_mmfi_p1s1_adap_patch_4090.py'
if (!$PythonExe) {
    $Candidates = @(
        (Join-Path $DataHome 'envs\dose-backdoor\python.exe'),
        (Join-Path $DataHome 'envs\dose-backdoor\Scripts\python.exe'),
        (Join-Path $DataHome 'envs\mmfi-p3\Scripts\python.exe')
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $PythonExe = $Candidate
            break
        }
    }
    if (!$PythonExe) { throw 'No existing Python environment found; supply -PythonExe explicitly.' }
}
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
foreach ($Value in @($Repo,$DataHome,$Runs,$Outdir,$PythonExe,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') {
        throw "Unsupported local path: $Value"
    }
}
$RunsPrefix = $Runs.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
if (!$Outdir.StartsWith($RunsPrefix,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Outdir must be a strict descendant of DataHome\runs.'
}
if (!(Test-Path -LiteralPath $DataHome -PathType Container)) {
    throw "Missing DataHome: $DataHome"
}
foreach ($Value in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing: $Value" }
}

$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$Common = @('-u',$Runner,'--data-home',$DataHome,'--outdir',$Outdir,
            '--device',$Device,'--jobs-per-gpu','1')
if ($AllowConcurrentP3S3) { $Common += '--allow-concurrent-p3s3' }

function Invoke-Runner([string]$Phase,[string]$Log = '') {
    # Do not redirect ordinary native stderr warnings into terminating
    # NativeCommandError records under Windows PowerShell 5.1.
    $global:LASTEXITCODE = 0
    if ($Log) {
        & $PythonExe @Common '--mode' $Phase | Tee-Object -FilePath $Log -ErrorAction Stop
    } else {
        & $PythonExe @Common '--mode' $Phase
    }
    $RunnerExitCode = $LASTEXITCODE
    if ($RunnerExitCode -ne 0) {
        throw "Adap-Patch $Phase failed (exit $RunnerExitCode); no later phase or automatic retry started."
    }
}

$Phases = @{Inspect='inspect'; DryRun='dry-run'; Prepare='prepare'; CheckOnly='check'; Status='status'}
if ($Mode -ne 'Launch') {
    Invoke-Runner $Phases[$Mode]
    if ($Mode -eq 'Prepare') {
        'PREPARED_ONLY: reused P3S3 audit + P1S1 subset/stat checks and CUDA B32 smoke; no full victim training.'
        'Inspect receipts, then invoke -Mode Launch separately.'
    }
    return
}

# Launch never prepares implicitly and never stops another GPU job.
Invoke-Runner 'check'
$Logs = Join-Path $Outdir 'logs'
[IO.Directory]::CreateDirectory($Logs) | Out-Null
$Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N')
$ControllerLog = Join-Path $Logs "controller.$Stamp.log"
"CONTROLLER_LOG=$ControllerLog"
'FOREGROUND: keep this SSH session open. Max one P1S1 Adap-Patch worker on cuda:0.'
if ($AllowConcurrentP3S3) {
    'Explicit sharing: at most one exact original Adap-Patch P3S3 peer; other GPU jobs blocked.'
}
'One fresh HPE-Li victim; no Clean/A1 retraining. Poison 10% + disjoint covers 20%.'
Invoke-Runner 'launch' $ControllerLog
'RUNNER_EXITED_SUCCESSFULLY: use -Mode Status and inspect evaluation.json / comparison.md.'
