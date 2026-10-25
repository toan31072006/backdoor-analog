<#
.SYNOPSIS
Gated MM-Fi P3/S1-S3 HPE-Li Clean+A1 seed-42 runner for one Windows GPU.
.DESCRIPTION
DryRun is the default. Preflight checks data on CPU; GpuSmoke checks the
requested number of concurrent physical batch-32 workers. Only explicit
Launch or LaunchForeground starts the full training controller.
#>
[CmdletBinding()]
param(
    [ValidateSet('DryRun', 'Preflight', 'GpuSmoke', 'Launch', 'LaunchForeground', 'Status')]
    [string]$Mode = 'DryRun',
    [ValidateSet(1, 2)][int]$JobsPerGpu = 2,
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [string]$Python = '',
    [string]$TaskName = 'MMFI-P3-HPELi-S42'
)

$ErrorActionPreference = 'Stop'
if ($TaskName -notmatch '^[A-Za-z0-9_.-]+$') {
    throw 'TaskName may contain only letters, digits, underscore, dot, and hyphen.'
}
$repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome).TrimEnd('\')
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\mmfi_p3_hpeli_s42_v1' }
if (!$Python) { $Python = Join-Path $DataHome 'envs\mmfi-p3\Scripts\python.exe' }
$Outdir = [IO.Path]::GetFullPath($Outdir).TrimEnd('\')
$Python = [IO.Path]::GetFullPath($Python)
foreach ($path in @($repo, $DataHome, $Outdir, $Python)) {
    if ($path.StartsWith('\\') -or $path -match '["\r\n]') {
        throw "Unsupported local path: $path"
    }
}
$runner = Join-Path $repo 'ATKBackd\run_mmfi_p3_splits.py'
if (!(Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Missing runner: $runner" }
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }

$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$common = @('-u', $runner, '--data-home', $DataHome, '--outdir', $Outdir,
            '--device', 'cuda:0', '--workers', '0',
            '--jobs-per-gpu', [string]$JobsPerGpu)
$modeFlag = @{
    DryRun = 'dry-run'
    Preflight = 'preflight'
    GpuSmoke = 'gpu-smoke'
    Status = 'status'
}
if ($modeFlag.ContainsKey($Mode)) {
    & $Python @common '--mode' $modeFlag[$Mode]
    if ($LASTEXITCODE -ne 0) { throw "P3 $Mode failed; no full training launched." }
    return
}

# Python rechecks the content-bound receipts. The wrapper catches missing gates
# before it registers a long-lived task or starts foreground training.
foreach ($name in @('p3_splits.plan.json', 'preflight.json', 'gpu_smoke.json')) {
    $path = Join-Path $Outdir $name
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing gate: $path. Run DryRun, Preflight, GpuSmoke first."
    }
}
$smoke = Get-Content -LiteralPath (Join-Path $Outdir 'gpu_smoke.json') -Raw -Encoding UTF8 |
    ConvertFrom-Json
if ($smoke.jobs_per_gpu -ne $JobsPerGpu -or $smoke.physical_batch -ne 32) {
    throw 'GPU smoke does not match requested concurrency/physical batch.'
}
if ($Mode -eq 'LaunchForeground') {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing -and $existing.State -eq 'Running') {
        throw "Task $TaskName is running; inspect it before another launch."
    }
    $log = Join-Path $Outdir ('controller.foreground.' + (Get-Date -Format yyyyMMdd_HHmmss_fff) + '.log')
    Write-Host "Foreground controller log: $log"
    Write-Host 'Keep this PowerShell/SSH session open while training runs.'
    # Windows PowerShell 5.1 turns native stderr merged into a pipeline into
    # NativeCommandError records. Do not let a warning abort the controller.
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python @common '--mode' 'launch' 2>&1 | Tee-Object -FilePath $log -Append
        $controllerExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($controllerExitCode -ne 0) { throw "P3 controller failed (exit $controllerExitCode); inspect $log and cell logs." }
    return
}

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task $TaskName already exists; inspect it before another scheduled launch."
}

$config = @{ python = $Python; args = @($common + @('--mode', 'launch'));
             repo = $repo; outdir = $Outdir }
$config64 = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes(($config | ConvertTo-Json -Compress -Depth 5)))
$payload = @'
$ErrorActionPreference = 'Stop'
$cfg = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CONFIG64__')) |
    ConvertFrom-Json
$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$stem = Join-Path $cfg.outdir ('controller.' + (Get-Date -Format yyyyMMdd_HHmmss_fff))
try {
    $quoted = @($cfg.args | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $child = Start-Process -FilePath $cfg.python -ArgumentList $quoted -WorkingDirectory $cfg.repo -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput ($stem + '.stdout.log') -RedirectStandardError ($stem + '.stderr.log')
    exit $child.ExitCode
} catch {
    $_ | Out-String | Set-Content -LiteralPath ($stem + '.wrapper-error.log') -Encoding UTF8
    exit 1
}
'@
$payload = $payload.Replace('__CONFIG64__', $config64)
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($payload))
$shell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$action = New-ScheduledTaskAction -Execute $shell -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded" -WorkingDirectory $repo
# S4U keeps the controller alive after SSH disconnects. All inputs are local;
# this logon type does not provide network resource access.
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task = New-ScheduledTask -Action $action -Principal $principal -Settings $settings -Description 'Gated MM-Fi P3 S1-S3 HPE-Li Clean+A1 seed42; no automatic retry.'
Register-ScheduledTask -TaskName $TaskName -InputObject $task -ErrorAction Stop | Out-Null
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 2
$taskState = Get-ScheduledTask -TaskName $TaskName
$taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
if ($taskState.State -ne 'Running') {
    throw "Task $TaskName is $($taskState.State) after start (lastResult=$($taskInfo.LastTaskResult)). Inspect controller/wrapper logs; no training progress confirmed. LaunchForeground remains available if the task is not running."
}
Write-Host "TASK_RUNNING: $TaskName state=$($taskState.State) lastResult=$($taskInfo.LastTaskResult)"
Write-Host "Controller logs: $Outdir\controller.*.stdout.log / stderr.log"
Write-Host "Cell logs: $Outdir\p3_s<3|2|1>\<clean|a1>\console.log"
Write-Host 'Inspect logs for real progress; task state alone is not proof of training.'
