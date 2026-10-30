<#
.SYNOPSIS
Run only BadNets-CSI / WiFlow and MetaFi++ / MM-Fi P1S1 / seed42.
.DESCRIPTION
Exactly one GPU worker at a time, WiFlow then MetaFi++. Launch is foreground:
keep the SSH session open. No environment install, process kill, background
task, concurrency override, Clean/A1 training, or automatic retry is provided.
#>
[CmdletBinding()]
param(
    [ValidateSet('Prepare','CheckOnly','Launch','Status','Export')]
    [string]$Mode = 'Status',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [Alias('Python')][string]$PythonExe = '',
    [ValidateSet('cuda:0')][string]$Device = 'cuda:0'
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
$Runs = [IO.Path]::GetFullPath((Join-Path $DataHome 'runs'))
if (!$Outdir) {
    $Outdir = Join-Path $Runs 'mmfi_p1s1_badnets_wiflow_metafi_s42_4090_v1'
}
$Outdir = [IO.Path]::GetFullPath($Outdir)
if (!$PythonExe) {
    $PythonExe = Join-Path $DataHome 'envs\dose-backdoor\python.exe'
}
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
$Runner = Join-Path $Repo 'ATKBackd\run_mmfi_badnets_two_models_4090.py'
foreach ($Value in @($Repo,$DataHome,$Runs,$Outdir,$PythonExe,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') {
        throw "Unsupported path: $Value"
    }
}
$RunsPrefix = $Runs.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
if (!$Outdir.StartsWith($RunsPrefix,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Outdir must be a strict descendant of DataHome\runs, not DataHome, runs itself, or another root.'
}
if (!(Test-Path -LiteralPath $DataHome -PathType Container)) {
    throw "Missing DataHome directory: $DataHome"
}
foreach ($Value in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing: $Value" }
}
$Common = @('-u',$Runner,'--data-home',$DataHome,'--outdir',$Outdir,'--device',$Device)
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

function Invoke-Runner([string]$Phase,[string]$Log = '') {
    # Tee stdout only. Leaving native stderr on the console avoids turning
    # ordinary Python warnings into terminating NativeCommandError records
    # under Windows PowerShell 5.1; startup/logging failures still fail closed.
    $global:LASTEXITCODE = 0
    if ($Log) {
        & $PythonExe @Common '--mode' $Phase | Tee-Object -FilePath $Log -ErrorAction Stop
    } else {
        & $PythonExe @Common '--mode' $Phase
    }
    $RunnerExitCode = $LASTEXITCODE
    if ($RunnerExitCode -ne 0) {
        throw "MM-Fi BadNets $Phase failed (exit $RunnerExitCode); no later phase or automatic retry was started."
    }
}

if ($Mode -eq 'Prepare') {
    Invoke-Runner 'prepare'
    'PREPARED_ONLY: audit + B32 CUDA smokes; no full victim training launched.'
    'Inspect receipts, then invoke -Mode Launch separately.'
    return
}
if ($Mode -eq 'CheckOnly') { Invoke-Runner 'check'; return }
if ($Mode -eq 'Status') { Invoke-Runner 'status'; return }
if ($Mode -eq 'Export') { Invoke-Runner 'export'; return }

# Launch does not prepare implicitly. Both this check and the runner's own
# lock/gate checks must pass. No other GPU compute process is authorized.
Invoke-Runner 'check'
$Logs = Join-Path $Outdir 'logs'
[IO.Directory]::CreateDirectory($Logs) | Out-Null
$Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N')
$ControllerLog = Join-Path $Logs "controller.$Stamp.log"
"CONTROLLER_LOG=$ControllerLog"
'FOREGROUND: keep this SSH session open. WiFlow -> MetaFi++; max one worker on cuda:0.'
Invoke-Runner 'launch' $ControllerLog
'RUNNER_EXITED_SUCCESSFULLY: inspect -Mode Status and the exported results.'
