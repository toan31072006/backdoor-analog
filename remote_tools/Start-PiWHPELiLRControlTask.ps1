<#
.SYNOPSIS
Prepare, capacity-test, or explicitly launch the separate PiW LR control pilot.
.DESCRIPTION
Default Mode=DryRun. Preflight and GpuSmoke never launch full training.
Launch registers one persistent Windows task; the runner checks both receipts.
No old task/process is stopped or modified. Use Anaconda PowerShell on Van.
#>
[CmdletBinding()]
param(
    [ValidateSet('DryRun','Preflight','GpuSmoke','Launch','Status')]
    [string]$Mode = 'DryRun',
    [ValidateSet(1,2)][int]$JobsPerGpu = 2,
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$DatasetRoot = '',
    [string]$ActionNpy = '',
    [string]$Outdir = '',
    [string]$EnvironmentPrefix = '',
    [string]$TaskName = 'PiW-HPELi-LRControl-Seed42'
)
$ErrorActionPreference = 'Stop'
if ($TaskName -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid TaskName.' }
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\Person-in-WiFi-3D_processed_v1' }
if (!$ActionNpy) { $ActionNpy = Join-Path $DataHome 'actions\data_bend.npy' }
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\piw_hpeli_body_axis_s42_lr001_v3' }
if (!$EnvironmentPrefix) { $EnvironmentPrefix = Join-Path $DataHome 'envs\dose-backdoor' }
$code = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
foreach ($name in @('DataHome','DatasetRoot','ActionNpy','Outdir','EnvironmentPrefix')) {
    $value = [IO.Path]::GetFullPath((Get-Variable $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') { throw "Invalid local path: $name" }
    Set-Variable $name $value
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code 'ATKBackd\run_piw_hpeli_lr_control.py'
foreach ($file in @($python,$runner)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$runnerArgs = @('-u',$runner,'--data-home',$DataHome,'--dataset-root',$DatasetRoot,
    '--action-npy',$ActionNpy,'--outdir',$Outdir,'--device','cuda:0',
    '--jobs-per-gpu',[string]$JobsPerGpu)
$flag = @{ DryRun='--dry-run'; Preflight='--preflight'; GpuSmoke='--gpu-smoke'; Status='--status' }
if ($Mode -ne 'Launch') {
    & $python @runnerArgs $flag[$Mode]
    if ($LASTEXITCODE -ne 0) { throw "PiW LR control $Mode failed; no full training launched." }
    return
}
# A repeated Launch cannot overwrite a task, active process, or run.
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task $TaskName already exists. Inspect it before starting another controller."
}
foreach ($kind in @('preflight','gpu_smoke')) {
    $receipt = "$Outdir.$kind.json"
    if (!(Test-Path -LiteralPath $receipt)) { throw "Missing $receipt; run $kind first." }
    $record = Get-Content -LiteralPath $receipt -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($record.status -ne 'PASS') { throw "$kind has not passed; no launch." }
}
$cfg = @{ python=$python; args=($runnerArgs + '--launch'); code=$code; output=$Outdir }
$encodedCfg = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($cfg | ConvertTo-Json -Compress)))
$payload = @'
$ErrorActionPreference = 'Stop'
$cfg = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CFG__')) | ConvertFrom-Json
$env:PYTHONUTF8='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$log = $cfg.output + '.task.' + (Get-Date -Format yyyyMMdd_HHmmss_fff) + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
try {
    $quoted = @($cfg.args | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $child = Start-Process -FilePath $cfg.python -ArgumentList $quoted -WorkingDirectory $cfg.code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$log.stdout.log" -RedirectStandardError "$log.stderr.log"
    exit $child.ExitCode
} catch {
    $_ | Out-String | Set-Content -LiteralPath "$log.wrapper-error.log" -Encoding UTF8
    exit 1
}
'@
$payload = $payload.Replace('__CFG__',$encodedCfg)
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($payload))
$shell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$action = New-ScheduledTaskAction -Execute $shell -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded" -WorkingDirectory $code
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$principal = New-ScheduledTaskPrincipal -UserId $sid -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task = New-ScheduledTask -Action $action -Principal $principal -Settings $settings -Description 'Separate PiW LR control Clean+A1 seed42; explicit preflight/capacity gates; no automatic restart.'
Register-ScheduledTask -TaskName $TaskName -InputObject $task -ErrorAction Stop | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Controller: $Outdir.task.*.stdout.log / stderr.log"
Write-Host "Cells: $Outdir\hpeli\seed42\<clean|a1>\console.log"
Write-Host "Evaluation: $Outdir\evaluation.log"
Write-Host 'Task started; inspect logs to confirm progress. No sleep/shutdown.'
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
