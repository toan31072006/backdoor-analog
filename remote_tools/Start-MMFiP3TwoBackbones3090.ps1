<#
.SYNOPSIS
MM-Fi P3 Clean+A1 for MetaFi++ and WiFlow on one RTX 3090, two workers total.
.DESCRIPTION
Prepare performs CPU split checks and a paired real-model CUDA smoke. It never
starts full training. LaunchDetached starts the gated controller in a hidden
process with redirected logs; check its PID after reconnecting to the host.
#>
[CmdletBinding()]
param(
    [ValidateSet('DryRun','Prepare','Preflight','GpuSmoke','Status','CheckOnly',
                 'LaunchDetached','LaunchForeground')]
    [string]$Mode = 'DryRun',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome).TrimEnd('\')
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\mmfi_p3_metafi_wiflow_s42_3090_v1' }
if (!$Python) { $Python = Join-Path $DataHome 'envs\mmfi-p3\Scripts\python.exe' }
$Outdir = [IO.Path]::GetFullPath($Outdir).TrimEnd('\')
$Python = [IO.Path]::GetFullPath($Python)
$runner = Join-Path $repo 'ATKBackd\run_mmfi_p3_two_backbones_3090.py'
foreach ($path in @($repo, $DataHome, $Outdir, $Python, $runner)) {
    if ($path.StartsWith('\\') -or $path -match '["\r\n]') {
        throw "Unsupported local path: $path"
    }
}
if (!(Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Missing runner: $runner" }
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }

$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$common = @('-u', $runner, '--data-home', $DataHome, '--outdir', $Outdir,
            '--device', 'cuda:0', '--jobs-per-gpu', '2')

function Invoke-Phase([string]$phase) {
    & $Python @common '--mode' $phase
    if ($LASTEXITCODE -ne 0) { throw "P3 $phase failed; full training was not launched by this mode." }
}

if ($Mode -eq 'Prepare') {
    Invoke-Phase 'dry-run'
    Invoke-Phase 'preflight'
    Invoke-Phase 'gpu-smoke'
    Write-Host 'PREPARED_ONLY: 12-cell P3 run passed CPU/CUDA gates; no full training launched.'
    return
}
$phase = @{
    DryRun = 'dry-run'; Preflight = 'preflight'; GpuSmoke = 'gpu-smoke';
    Status = 'status'; CheckOnly = 'check-only'
}
if ($phase.ContainsKey($Mode)) { Invoke-Phase $phase[$Mode]; return }

foreach ($name in @('p3_two_backbones.plan.json','preflight.json','gpu_smoke.json')) {
    $path = Join-Path $Outdir $name
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing gate $path; run -Mode Prepare first."
    }
}
$smoke = Get-Content -LiteralPath (Join-Path $Outdir 'gpu_smoke.json') -Raw -Encoding UTF8 |
    ConvertFrom-Json
if ($smoke.jobs_per_gpu -ne 2 -or $smoke.physical_batch -ne 32) {
    throw 'GPU smoke is not the required two-worker physical32 gate.'
}

$active = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains($runner) -and
                   $_.CommandLine.Contains('--mode launch') })
if ($active.Count -gt 0) {
    throw "P3 controller is already running: PID $($active[0].ProcessId)"
}

if ($Mode -eq 'LaunchForeground') {
    Write-Host 'Foreground controller: keep this PowerShell session open until completion.'
    Invoke-Phase 'launch'
    return
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdout = Join-Path $Outdir "controller.$stamp.stdout.log"
$stderr = Join-Path $Outdir "controller.$stamp.stderr.log"
$quoted = @($common + @('--mode','launch') | ForEach-Object { '"' + $_ + '"' }) -join ' '
$child = Start-Process -FilePath $Python -ArgumentList $quoted -WorkingDirectory (Join-Path $repo 'ATKBackd') `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Start-Sleep -Seconds 2
$live = Get-CimInstance Win32_Process -Filter "ProcessId = $($child.Id)" -ErrorAction SilentlyContinue
if (!$live) {
    $errorText = if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Tail 12 } else { '' }
    throw "P3 controller exited immediately; inspect $stderr. $errorText"
}
Write-Host "STARTED PID=$($child.Id); max two jobs on cuda:0"
Write-Host "Controller logs: $stdout / $stderr"
Write-Host "Cell logs: $Outdir\<metafiplusplus|wiflow>\p3_s<3|2|1>\<clean|a1>\console.log"
Write-Host 'After SSH reconnect, use -Mode Status and inspect the controller PID/logs.'
