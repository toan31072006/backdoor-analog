<#
.SYNOPSIS
Exploratory A1 motion-prior sensitivity: MM-Fi P1S1 / HPE-Li / seed42.
.DESCRIPTION
Fresh A1 victims for cross then nod, serial and exclusive on cuda:0. Changing
the motion prior does not change the attacker target into that motion/action.
Prepare never launches full training. Launch is foreground: keep SSH open.
No environment installation, download, process kill, or GPU-sharing override.
#>
[CmdletBinding()]
param(
    [ValidateSet('AuditOnly','Prepare','CheckOnly','Launch','Status','Export')]
    [string]$Mode = 'Status',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [Alias('Python')][string]$PythonExe = '',
    [ValidateSet('cuda:0')][string]$Device = 'cuda:0',
    [ValidateSet('bend','cross','nod')][string[]]$Priors = @('cross','nod'),
    [string]$CrossAction = '',
    [string]$NodAction = '',
    [string]$BendAction = '',
    [switch]$ApproveActionLabels
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
$Runs = [IO.Path]::GetFullPath((Join-Path $DataHome 'runs'))
if (!$Outdir) {
    $Outdir = Join-Path $Runs 'mmfi_p1s1_hpeli_a1_motion_priors_s42_4090_v1'
}
$Outdir = [IO.Path]::GetFullPath($Outdir)
if (!$PythonExe) {
    $PythonExe = Join-Path $DataHome 'envs\dose-backdoor\python.exe'
}
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
$Runner = Join-Path $Repo 'ATKBackd\run_mmfi_a1_motion_priors_4090.py'
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
if (!$Priors.Count -or @($Priors | Select-Object -Unique).Count -ne $Priors.Count) {
    throw 'Priors must be a nonempty list without duplicates.'
}
$Common = @('-u',$Runner,'--data-home',$DataHome,'--outdir',$Outdir,'--device',$Device)
foreach ($Pair in @(@('--cross-action',$CrossAction),@('--nod-action',$NodAction),@('--bend-action',$BendAction))) {
    if ($Pair[1]) {
        $ActionPath = [IO.Path]::GetFullPath($Pair[1])
        if ($ActionPath.StartsWith('\\') -or $ActionPath -match '["\r\n\x00]') {
            throw "Unsupported action path: $ActionPath"
        }
        $Common += @($Pair[0],$ActionPath)
    }
}
if ($ApproveActionLabels) { $Common += '--approve-action-labels' }
$Common += @('--priors') + $Priors
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

function Invoke-Runner([string]$Phase,[string]$Log = '') {
    # Do not merge stderr into Tee: ordinary warnings become terminating
    # NativeCommandError records on Windows PowerShell 5.1 with Stop enabled.
    $global:LASTEXITCODE = 0
    if ($Log) {
        & $PythonExe @Common '--mode' $Phase | Tee-Object -FilePath $Log -ErrorAction Stop
    } else {
        & $PythonExe @Common '--mode' $Phase
    }
    $RunnerExitCode = $LASTEXITCODE
    if ($RunnerExitCode -ne 0) {
        throw "A1 motion-prior $Phase failed (exit $RunnerExitCode); no later phase or automatic retry was started."
    }
}

if ($Mode -eq 'AuditOnly') {
    Invoke-Runner 'audit'
    'AUDITED_ONLY: CPU/disk input and action checks; no GPU work or full training launched.'
    return
}
if ($Mode -eq 'Prepare') {
    Invoke-Runner 'prepare'
    'PREPARED_ONLY: input/action audit + CUDA B32 smokes; no full victim training launched.'
    'Inspect receipts, then invoke -Mode Launch separately.'
    return
}
if ($Mode -eq 'CheckOnly') { Invoke-Runner 'check'; return }
if ($Mode -eq 'Status') { Invoke-Runner 'status'; return }
if ($Mode -eq 'Export') { Invoke-Runner 'export'; return }

# Launch never prepares implicitly; no peer GPU compute is authorized.
Invoke-Runner 'check'
$Logs = Join-Path $Outdir 'logs'
[IO.Directory]::CreateDirectory($Logs) | Out-Null
$Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N')
$ControllerLog = Join-Path $Logs "controller.$Stamp.log"
"CONTROLLER_LOG=$ControllerLog"
"FOREGROUND: keep this SSH session open. Priors: $($Priors -join ' -> '); max one worker on cuda:0."
Invoke-Runner 'launch' $ControllerLog
'RUNNER_EXITED_SUCCESSFULLY: inspect -Mode Status and the exported results.'
