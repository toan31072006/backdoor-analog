<#
.SYNOPSIS
Explicitly approved one A1 motion-prior + one BadNets worker on Van GPU0.
.DESCRIPTION
The old BadNets controller/worker must already be stopped by the user. This
helper never attaches to a live queue, kills a process, changes an old checkout,
installs an environment, or retries automatically. Keep the SSH session open.
#>
[CmdletBinding()]
param(
    [ValidateSet('Prepare','CheckOnly','Launch','Status')][string]$Mode = 'Status',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [Alias('Python')][string]$PythonExe = '',
    [string]$BadNetsRepo = (Join-Path $env:USERPROFILE 'code_badnets_mmfi'),
    [string]$MotionRepo = '',
    [string]$SharedDir = '',
    [switch]$AllowConcurrentBadNets
)
$ErrorActionPreference = 'Stop'
if ($Mode -ne 'Status' -and !$AllowConcurrentBadNets) {
    throw 'Explicit -AllowConcurrentBadNets is required; no GPU work or process was started or stopped.'
}
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$MotionRepo) { $MotionRepo = Split-Path -Parent $PSScriptRoot }
$MotionRepo = [IO.Path]::GetFullPath($MotionRepo)
$BadNetsRepo = [IO.Path]::GetFullPath($BadNetsRepo)
if (!$PythonExe) { $PythonExe = Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
$Tools = [IO.Path]::GetFullPath((Join-Path $DataHome 'tools'))
if (!$SharedDir) { $SharedDir = Join-Path $Tools 'mmfi_a1_badnets_shared_v1' }
$SharedDir = [IO.Path]::GetFullPath($SharedDir)
$Runner = Join-Path $MotionRepo 'remote_tools\motion_prior_shared_4090\controller.py'
foreach ($Value in @($DataHome,$MotionRepo,$BadNetsRepo,$PythonExe,$Tools,$SharedDir,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw "Unsupported path: $Value" }
}
$ToolsPrefix = $Tools.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
if (!$SharedDir.StartsWith($ToolsPrefix,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'SharedDir must be a strict descendant of DataHome\tools, not tools itself or another root.'
}
foreach ($SourceRepo in @($MotionRepo,$BadNetsRepo)) {
    $SourcePrefix = $SourceRepo.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    if ($SharedDir.Equals($SourceRepo,[StringComparison]::OrdinalIgnoreCase) -or
        $SharedDir.StartsWith($SourcePrefix,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'SharedDir must be outside both scientific source checkouts.'
    }
}
if ($MotionRepo.Equals($BadNetsRepo,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Use a separate MotionRepo; do not update the frozen BadNets checkout.'
}
foreach ($Directory in @($DataHome,$MotionRepo,$BadNetsRepo)) {
    if (!(Test-Path -LiteralPath $Directory -PathType Container)) { throw "Missing directory: $Directory" }
}
foreach ($File in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing existing file: $File" }
}
$Common = @('-u',$Runner,'--home',$DataHome,'--a1-repo',$MotionRepo,
            '--badnets-repo',$BadNetsRepo,'--root',$SharedDir)
if ($AllowConcurrentBadNets) { $Common += '--allow-concurrent-badnets' }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

function Invoke-Runner([string]$Phase,[string]$Log = '') {
    $global:LASTEXITCODE = 0
    if ($Log) {
        # Leave stderr on the console: PS5.1 treats merged native warnings as
        # terminating NativeCommandError records under ErrorAction Stop.
        & $PythonExe @Common '--mode' $Phase | Tee-Object -FilePath $Log -ErrorAction Stop
    } else {
        & $PythonExe @Common '--mode' $Phase
    }
    $RunnerExitCode = $LASTEXITCODE
    if ($RunnerExitCode -ne 0) {
        throw "Shared A1/BadNets $Phase failed (exit $RunnerExitCode); no later phase, automatic retry or process termination."
    }
}

if ($Mode -eq 'Status') { Invoke-Runner 'status'; return }
if ($Mode -eq 'CheckOnly') { Invoke-Runner 'check'; return }
if ($Mode -eq 'Launch') { Invoke-Runner 'check' }
$Logs = Join-Path $SharedDir 'logs'
[IO.Directory]::CreateDirectory($Logs) | Out-Null
$Phase = $Mode.ToLowerInvariant()
$Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N')
$ControllerLog = Join-Path $Logs "controller.$Phase.$Stamp.log"
"CONTROLLER_LOG=$ControllerLog"
'FOREGROUND: keep this SSH session open. At most one A1 + one BadNets GPU worker; no speedup or VRAM guarantee.'
Invoke-Runner $Phase $ControllerLog
if ($Mode -eq 'Prepare') {
    'PREPARED_ONLY: inspect shared-context receipts; full training was not launched. Invoke -Mode Launch separately.'
} else {
    'SHARED_RUNNER_EXITED: inspect Status and per-cell evaluation artifacts, not an old PID.'
}
