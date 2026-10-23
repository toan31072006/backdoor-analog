<#
.SYNOPSIS
Run the independent WiPose HPE-Li Clean -> A1 seed42 pilot as one persistent job.
.DESCRIPTION
Use Anaconda PowerShell on Van. Data download/extraction/preparation are manual.
DryRun is metadata only; PreflightOnly checks prepared data/model on CPU.
No process or existing experiment is stopped, deleted, or silently reconfigured.
#>
[CmdletBinding()]
param(
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$DatasetRoot = '',
    [string]$Outdir = '',
    [string]$EnvironmentPrefix = '',
    [string]$ActionNpy = '',
    [string]$TaskName = 'WiPose-HPELi-Seed42',
    [switch]$CleanOnly,
    [switch]$DryRun,
    [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
if ($DryRun -and $PreflightOnly) { throw 'Choose only one of DryRun or PreflightOnly.' }
if ($TaskName -notmatch '^[A-Za-z0-9_.-]+$') { throw 'TaskName must contain only letters, numbers, underscore, dot or hyphen.' }
if ($CleanOnly -and !$PSBoundParameters.ContainsKey('TaskName')) { $TaskName = 'WiPose-HPELi-Clean-Seed42' }
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\WiPose_processed_v1' }
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\wipose_hpeli_a1_s42_v1' }
if (!$EnvironmentPrefix) { $EnvironmentPrefix = Join-Path $DataHome 'envs\dose-backdoor' }
if (!$ActionNpy) { $ActionNpy = Join-Path $DataHome 'actions\data_bend.npy' }
$code = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
foreach ($name in @('DataHome', 'DatasetRoot', 'Outdir', 'EnvironmentPrefix', 'ActionNpy')) {
    $value = [IO.Path]::GetFullPath((Get-Variable -Name $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') {
        throw "$name must be a local path without double quotes or newlines."
    }
    Set-Variable -Name $name -Value $value
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code 'ATKBackd\run_wipose_hpeli.py'
foreach ($file in @($python, $runner)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$preflight = @('-u', $runner, '--data-home', $DataHome, '--dataset-root', $DatasetRoot, '--outdir', $Outdir, '--action-npy', $ActionNpy, '--device', 'cuda:0')
if ($CleanOnly) { $preflight += '--clean-only' }
& $python @preflight --dry-run
if ($LASTEXITCODE -ne 0) { throw 'WiPose HPE-Li dry-run failed; nothing launched.' }
if ($DryRun) { return }
if ($PreflightOnly) {
    & $python @preflight --preflight
    if ($LASTEXITCODE -ne 0) { throw 'WiPose HPE-Li data/model preflight failed; nothing launched.' }
    return
}
if (!$env:CONDA_EXE -or !(Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
    throw 'Open Anaconda PowerShell first (CONDA_EXE is required).'
}

function Get-WiPosePayload([string]$ConfigBase64) {
    $template = @'
$ErrorActionPreference = 'Stop'
$cfg = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CONFIG__')) | ConvertFrom-Json
$log = $cfg.output + '.task.' + (Get-Date -Format yyyyMMdd_HHmmss_fff) + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
try {
    $env:PYTHONUTF8 = '1'
    $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:MKL_THREADING_LAYER = 'SEQUENTIAL'
    $line = 'run --no-capture-output -p "{0}" "{0}\python.exe" -u "{1}\ATKBackd\run_wipose_hpeli.py" --data-home "{2}" --dataset-root "{3}" --outdir "{4}" --action-npy "{5}" --device cuda:0' -f $cfg.prefix,$cfg.code,$cfg.data,$cfg.dataset,$cfg.output,$cfg.action
    if ($cfg.clean_only) { $line += ' --clean-only' }
    $p = Start-Process -FilePath $cfg.conda -ArgumentList $line -WorkingDirectory $cfg.code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$log.stdout.log" -RedirectStandardError "$log.stderr.log"
    exit $p.ExitCode
} catch {
    $_ | Out-String | Add-Content -LiteralPath "$log.wrapper-error.log" -Encoding UTF8
    exit 1
}
'@
    return $template.Replace('__CONFIG__', $ConfigBase64)
}

$currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$runnerPattern = '(?i)(?:^|\s|")' + ([regex]::Escape($runner).Replace('\\', '[\\/]')) + '(?=["\s]|$)'
$outputEscaped = [regex]::Escape($Outdir).Replace('\\', '[\\/]')
$manifestEscaped = $outputEscaped + '[\\/]wipose_hpeli\.resolved\.json'
$bindingPattern = '(?i)(?:^|\s)(?:--outdir\s+(?:"' + $outputEscaped + '"|' + $outputEscaped +
    '(?=\s|$))|--_manifest\s+(?:"' + $manifestEscaped + '"|' + $manifestEscaped + '(?=\s|$)))(?=\s|$)'
function Test-OwnedWorker($candidate) {
    if (!$candidate -or $candidate.Name -notmatch '^python(w)?\.exe$' -or
        $candidate.CommandLine -notmatch $runnerPattern -or
        $candidate.CommandLine -notmatch $bindingPattern) { return $false }
    try {
        $owner = Invoke-CimMethod -InputObject $candidate -MethodName GetOwnerSid -ErrorAction Stop
        return ($owner.ReturnValue -eq 0 -and $owner.Sid -eq $currentSid)
    } catch { return $false }
}
function Test-OwnedTask($task, [switch]$OutputOnly) {
    try {
        $account = $task.Principal.UserId
        if (!$account) { return $false }
        $sid = if ($account -match '^S-1-') { $account } else {
            ([Security.Principal.NTAccount]::new($account)).Translate([Security.Principal.SecurityIdentifier]).Value
        }
        if ($sid -ne $currentSid -or @($task.Actions).Count -ne 1) { return $false }
        $action = @($task.Actions)[0]
        if ($action.Execute -ne "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -or
            $action.Arguments -notmatch '(?i)^-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand ([A-Za-z0-9+/=]+)$') { return $false }
        $text = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
        if ($text -notmatch 'run_wipose_hpeli\.py') { return $false }
        foreach ($match in [regex]::Matches($text, "FromBase64String\('([A-Za-z0-9+/=]+)'\)")) {
            $saved = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($match.Groups[1].Value)) | ConvertFrom-Json
            if ($saved.helper -eq 'wipose_hpeli_task' -and $saved.output -eq $Outdir -and
                $saved.code -eq $code -and $saved.clean_only -is [bool] -and
                ($OutputOnly -or ($saved.data -eq $DataHome -and $saved.dataset -eq $DatasetRoot -and
                    $saved.prefix -eq $EnvironmentPrefix -and $saved.action -eq $ActionNpy -and
                    $saved.clean_only -eq [bool]$CleanOnly -and $saved.conda -eq $env:CONDA_EXE)) -and
                $text -ceq (Get-WiPosePayload $match.Groups[1].Value)) { return $true }
        }
    } catch { return $false }
    return $false
}
$lockProbe = @'
import sys
import msvcrt
try:
    with open(sys.argv[1], 'r+b') as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
except OSError as exc:
    print('WiPose output lock unavailable: ' + repr(exc), file=sys.stderr)
    sys.exit(3)
'@
function Assert-WiPoseIdle {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing -and !(Test-OwnedTask $existing)) {
        throw "Task name belongs to a different account/action/output/config: $TaskName"
    }
    $activeTasks = @(Get-ScheduledTask | Where-Object {
        $_.State -in @('Running', 'Queued') -and (Test-OwnedTask $_ -OutputOnly)
    })
    $workers = @(Get-CimInstance Win32_Process | Where-Object { Test-OwnedWorker $_ })
    if ($activeTasks.Count -or $workers.Count) {
        $workers | Format-List ProcessId,ParentProcessId,CommandLine
        throw 'WiPose output is active. No duplicate launch; no process is stopped.'
    }
    foreach ($lockPath in @((Join-Path $Outdir 'run.lock'),
        (Join-Path $Outdir 'hpeli\seed42\clean\run.lock'),
        (Join-Path $Outdir 'hpeli\seed42\a1\run.lock'))) {
        if (Test-Path -LiteralPath $lockPath -PathType Leaf) {
            & $python -c $lockProbe $lockPath
            if ($LASTEXITCODE -ne 0) { throw 'WiPose output lock unavailable; do not delete run.lock.' }
        }
    }
}
Assert-WiPoseIdle

Write-Host 'WiPose preflight: checking prepared data/model/trigger on CPU; no CUDA, training or experiment output writes.'
& $python @preflight --preflight
if ($LASTEXITCODE -ne 0) { throw 'WiPose HPE-Li data/model preflight failed; nothing launched.' }
# A different helper may have launched while the read-only preflight was running.
Assert-WiPoseIdle

$config = @{
    helper = 'wipose_hpeli_task'; code = $code; data = $DataHome; dataset = $DatasetRoot; output = $Outdir
    conda = $env:CONDA_EXE; prefix = $EnvironmentPrefix; action = $ActionNpy; clean_only = [bool]$CleanOnly
} | ConvertTo-Json -Compress
$config64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($config))
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes((Get-WiPosePayload $config64)))
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded"
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Outdir) | Out-Null
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
Write-Host "Controller logs: $Outdir.task.*.stdout.log / stderr.log"
Write-Host "Cell logs: $Outdir\hpeli\seed42\<clean|a1>\console.log"
Write-Host 'One worker at a time: Clean then scratch A1 (or CleanOnly). No automatic retry or process stop.'
Write-Host 'Running is not proof of progress; inspect logs. Do not sleep or shut down the machine.'
