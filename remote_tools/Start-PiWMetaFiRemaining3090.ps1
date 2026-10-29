<#
.SYNOPSIS
Prepare or resume the remaining PiW3D MetaFi++ AdamW cells on one RTX 3090.
.DESCRIPTION
Exactly one worker at a time on cuda:0. Inspect and Status do not launch
training. Prepare audits source, data, imported checkpoints and real B32 CUDA
steps, then stops. Launch is an explicit, separately gated hidden controller.
No environment installation, git operation, process termination or CPU fallback.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','Prepare','CheckOnly','Launch','Status')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [Alias('Dataset')]
    [string]$DatasetRoot = '',
    [Alias('Action')]
    [string]$ActionNpy = '',
    [string]$CheckpointHome = '',
    [Alias('Out')]
    [string]$Outdir = '',
    [Alias('Python')]
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$DatasetRoot) {
    $DatasetRoot = Join-Path $DataHome 'datasets\Person-in-WiFi-3D_processed_v1'
}
if (!$ActionNpy) { $ActionNpy = Join-Path $DataHome 'actions\data_bend.npy' }
if (!$CheckpointHome) {
    $CheckpointHome = Join-Path $DataHome 'imports\piw_metafi_adamw_unfinished'
}
if (!$Outdir) {
    $Outdir = Join-Path $DataHome 'runs\piw_metafi_remaining_s42_3090_v1'
}
$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
$ActionNpy = [IO.Path]::GetFullPath($ActionNpy)
$CheckpointHome = [IO.Path]::GetFullPath($CheckpointHome)
$Outdir = [IO.Path]::GetFullPath($Outdir)
$Runner = Join-Path $Repo 'remote_tools\piw_metafi_remaining_3090\controller.py'

if (!$PythonExe) {
    $Candidates = @(
        (Join-Path $DataHome 'envs\piw-metafi\Scripts\python.exe'),
        (Join-Path $DataHome 'envs\dose-backdoor\python.exe'),
        (Join-Path $DataHome 'envs\mmfi-p3\Scripts\python.exe')
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $PythonExe = $Candidate
            break
        }
    }
    if (!$PythonExe) {
        throw ('No existing Python environment found. Supply -PythonExe explicitly. Checked: ' + ($Candidates -join ', '))
    }
}
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
foreach ($PathValue in @($Repo, $DataHome, $DatasetRoot, $ActionNpy,
                         $CheckpointHome, $Outdir, $Runner, $PythonExe)) {
    if ($PathValue.StartsWith('\\') -or $PathValue -match '["\r\n\x00]') {
        throw "Unsupported local path: $PathValue"
    }
}
foreach ($PathValue in @($Runner, $PythonExe)) {
    if (!(Test-Path -LiteralPath $PathValue -PathType Leaf)) {
        throw "Missing required file: $PathValue"
    }
}
if (!(Test-Path -LiteralPath $DataHome -PathType Container)) {
    throw "Missing data home: $DataHome"
}

$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$Common = @('-u', $Runner, '--data-home', $DataHome,
            '--dataset', $DatasetRoot, '--action', $ActionNpy,
            '--out', $Outdir, '--checkpoint-home', $CheckpointHome)

function Invoke-Phase([string]$Phase) {
    & $PythonExe @Common '--mode' $Phase
    if ($LASTEXITCODE -ne 0) {
        throw "PiW MetaFi++ $Phase failed. No later phase was started by this invocation."
    }
}

$Phases = @{Inspect='inspect'; Prepare='prepare'; CheckOnly='check'; Status='status'}
if ($Mode -ne 'Launch') {
    Invoke-Phase $Phases[$Mode]
    if ($Mode -eq 'Prepare') {
        Write-Host 'PREPARED_ONLY: audits and CUDA B32 smoke passed; no full training launched.'
        Write-Host 'Use -Mode Launch separately after inspecting the Prepare output.'
    }
    return
}

# The Python controller owns the immutable-source/input/runtime/resume gate
# and exclusive output lock. Never infer permission from a gate filename alone.
Invoke-Phase 'check'
if (!(Test-Path -LiteralPath $Outdir -PathType Container)) {
    throw "Output directory missing after gate check: $Outdir"
}

function Quote-WindowsArgument([string]$Value) {
    if ($Value -match '["\r\n\x00]') { throw 'Unsupported argument' }
    return '"' + ([regex]::Replace($Value, '(\\+)$', '$1$1')) + '"'
}
$ArgumentLine = @(($Common + @('--mode','launch')) | ForEach-Object {
    Quote-WindowsArgument ([string]$_)
}) -join ' '
$Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' +
         [guid]::NewGuid().ToString('N').Substring(0,8)
$Stdout = Join-Path $Outdir "controller.$Stamp.stdout.log"
$Stderr = Join-Path $Outdir "controller.$Stamp.stderr.log"
$Child = Start-Process -FilePath $PythonExe -ArgumentList $ArgumentLine `
    -WorkingDirectory $Repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
Start-Sleep -Seconds 2
$Child.Refresh()
if ($Child.HasExited) {
    $Tail = if (Test-Path -LiteralPath $Stderr -PathType Leaf) {
        (Get-Content -LiteralPath $Stderr -Encoding UTF8 -Tail 15) -join [Environment]::NewLine
    } else { '' }
    throw "Controller exited immediately (exit $($Child.ExitCode)). Inspect $Stdout and $Stderr. $Tail"
}
Write-Host "STARTED controller PID=$($Child.Id); max one worker on cuda:0."
Write-Host "Controller stdout: $Stdout"
Write-Host "Controller stderr: $Stderr"
Write-Host 'Included: Clean, A1, FTrojan-CSI, BppAttack-CSI, FIBA-CSI, WaNet-CSI.'
Write-Host 'Excluded: TSBA adapted and Smooth LF-CSI. Use -Mode Status to confirm progress.'
Write-Host 'An alive PID is not proof of training progress; inspect changing cell logs.'
