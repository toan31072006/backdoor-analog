<#
.SYNOPSIS
Prepare or launch three new HPE-Li MM-Fi P3S3 attack victims on Van's RTX 4090.
.DESCRIPTION
Inspect is read-only. Prepare audits inputs and runs real CUDA smokes, but does
not launch full training. Launch is a separate, checked, hidden controller.
No package installation, git operation, process termination or CPU fallback.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','Prepare','CheckOnly','Launch','Status')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [Alias('Python')]
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$Outdir) {
    $Outdir = Join-Path $DataHome 'runs\mmfi_p3s3_three_attacks_s42_4090_v1'
}
$Outdir = [IO.Path]::GetFullPath($Outdir)
$Runner = Join-Path $Repo 'ATKBackd\run_mmfi_p3_s3_three_attacks_4090.py'

if (!$PythonExe) {
    $Candidates = @(
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
foreach ($PathValue in @($Repo, $DataHome, $Outdir, $Runner, $PythonExe)) {
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
$Common = @('-u', $Runner, '--data-home', $DataHome, '--outdir', $Outdir,
            '--device', 'cuda:0', '--jobs-per-gpu', '1')

function Invoke-Phase([string]$Phase) {
    & $PythonExe @Common '--mode' $Phase
    if ($LASTEXITCODE -ne 0) {
        throw "P3S3 $Phase failed. No later phase was started by this invocation."
    }
}

$Phases = @{Inspect='inspect'; Prepare='prepare'; CheckOnly='check'; Status='status'}
if ($Mode -ne 'Launch') {
    Invoke-Phase $Phases[$Mode]
    if ($Mode -eq 'Prepare') {
        Write-Host 'PREPARED_ONLY: input audit and CUDA smoke passed; no full training launched.'
        Write-Host 'Use -Mode Launch separately after inspecting the Prepare output.'
    }
    return
}

# The runner owns source/input/runtime gates and the OS controller lock.
# Do not infer permission to train just from the presence of gate files.
Invoke-Phase 'check'
if (!(Test-Path -LiteralPath $Outdir -PathType Container)) {
    throw "Output directory missing after gate check: $Outdir"
}

function Quote-WindowsArgument([string]$Value) {
    if ($Value -match '["\r\n\x00]') { throw 'Unsupported argument' }
    # Quotes are rejected above; only trailing backslashes need doubling before
    # the closing quote for the Windows native command-line parser.
    return '"' + ([regex]::Replace($Value, '(\\+)$', '$1$1')) + '"'
}
$ArgumentLine = @(($Common + @('--mode','launch')) | ForEach-Object {
    Quote-WindowsArgument ([string]$_)
}) -join ' '
$Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$Stdout = Join-Path $Outdir "controller.$Stamp.stdout.log"
$Stderr = Join-Path $Outdir "controller.$Stamp.stderr.log"
$Child = Start-Process -FilePath $PythonExe -ArgumentList $ArgumentLine `
    -WorkingDirectory (Join-Path $Repo 'ATKBackd') -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
Start-Sleep -Seconds 2
$Child.Refresh()
if ($Child.HasExited) {
    $Tail = if (Test-Path -LiteralPath $Stderr -PathType Leaf) {
        (Get-Content -LiteralPath $Stderr -Tail 15) -join [Environment]::NewLine
    } else { '' }
    throw "Controller exited immediately (exit $($Child.ExitCode)). Inspect $Stdout and $Stderr. $Tail"
}
Write-Host "STARTED controller PID=$($Child.Id); max one job on cuda:0."
Write-Host "Controller stdout: $Stdout"
Write-Host "Controller stderr: $Stderr"
Write-Host "Cell logs: $Outdir\<input_aware|badnets|sig_source>\console.log"
Write-Host 'Order: Input-Aware adapted, BadNets-CSI, SIG-CSI. Use -Mode Status to confirm progress.'
Write-Host 'An alive PID is not proof of training progress; inspect changing cell logs.'
