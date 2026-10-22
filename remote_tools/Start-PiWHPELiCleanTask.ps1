<#
.SYNOPSIS
Run only HPE-Li Clean on Person-in-WiFi-3D, seed42, as one persistent job.
.DESCRIPTION
Run in Anaconda PowerShell on Van. Does not stop/pause WiFlow or any other job.
Dry-run is metadata only. Preflight hashes all selected data and checks the
model on CPU; it does not train, use CUDA, or create experiment output files.
#>
[CmdletBinding()]
param(
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$DatasetRoot = '',
    [string]$Outdir = '',
    [string]$EnvironmentPrefix = '',
    [string]$TaskName = 'PIW-HPELi-Clean-Seed42',
    [switch]$DryRun,
    [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
if ($DryRun -and $PreflightOnly) { throw 'Choose only one of DryRun or PreflightOnly.' }
if ($TaskName -notmatch '^[A-Za-z0-9_.-]+$') { throw 'TaskName must contain only letters, numbers, underscore, dot or hyphen.' }
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\Person-in-WiFi-3D_processed_v1' }
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\piw_hpeli_clean_s42_v1' }
if (!$EnvironmentPrefix) { $EnvironmentPrefix = Join-Path $DataHome 'envs\dose-backdoor' }
$code = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
foreach ($name in @('DataHome', 'DatasetRoot', 'Outdir', 'EnvironmentPrefix')) {
    $value = [IO.Path]::GetFullPath((Get-Variable -Name $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') {
        throw "$name must be a local path without double quotes or newlines."
    }
    Set-Variable -Name $name -Value $value
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code 'ATKBackd\run_piw_hpeli_clean.py'
foreach ($file in @($python, $runner)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$env:PYTHONUTF8 = '1'
$preflight = @('-u', $runner, '--data-home', $DataHome, '--dataset-root', $DatasetRoot, '--outdir', $Outdir, '--device', 'cuda:0')
& $python @preflight --dry-run
if ($LASTEXITCODE -ne 0) { throw 'PiW HPE-Li dry-run failed; nothing launched.' }
if ($DryRun) { return }
if ($PreflightOnly) {
    & $python @preflight --preflight
    if ($LASTEXITCODE -ne 0) { throw 'PiW HPE-Li data/model preflight failed; nothing launched.' }
    return
}
if (!$env:CONDA_EXE -or !(Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
    throw 'Open Anaconda PowerShell first (CONDA_EXE is required).'
}

$currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$runnerPattern = '(?i)(?:^|\s|")' + ([regex]::Escape($runner).Replace('\\', '[\\/]')) + '(?=["\s]|$)'
$outputEscaped = [regex]::Escape($Outdir).Replace('\\', '[\\/]')
$manifestEscaped = $outputEscaped + '[\\/]piw_hpeli_clean\.resolved\.json'
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
function Test-OwnedTask($task) {
    try {
        $account = $task.Principal.UserId
        if (!$account) { return $false }
        $sid = if ($account -match '^S-1-') { $account } else {
            ([Security.Principal.NTAccount]::new($account)).Translate([Security.Principal.SecurityIdentifier]).Value
        }
        if ($sid -ne $currentSid -or @($task.Actions).Count -ne 1) { return $false }
        $action = @($task.Actions)[0]
        if ($action.Execute -ne "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -or
            $action.Arguments -notmatch '(?i)-EncodedCommand\s+([A-Za-z0-9+/=]+)') { return $false }
        $text = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
        if ($text -notmatch 'run_piw_hpeli_clean\.py') { return $false }
        foreach ($match in [regex]::Matches($text, "FromBase64String\('([A-Za-z0-9+/=]+)'\)")) {
            $saved = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($match.Groups[1].Value)) | ConvertFrom-Json
            if ($saved.helper -eq 'piw_hpeli_clean_task' -and $saved.output -eq $Outdir -and
                $saved.code -eq $code -and $saved.data -eq $DataHome -and
                $saved.dataset -eq $DatasetRoot -and $saved.prefix -eq $EnvironmentPrefix) { return $true }
        }
    } catch { return $false }
    return $false
}
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and !(Test-OwnedTask $existing)) {
    throw "Task name belongs to a different account/action/output/config: $TaskName"
}
$activeTasks = @(Get-ScheduledTask | Where-Object {
    $_.State -in @('Running', 'Queued') -and (Test-OwnedTask $_)
})
$workers = @(Get-CimInstance Win32_Process | Where-Object { Test-OwnedWorker $_ })
if ($activeTasks.Count -or $workers.Count) {
    $workers | Format-List ProcessId,ParentProcessId,CommandLine
    throw 'PiW Clean output is active. No duplicate launch; no process is stopped.'
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
    print('PiW Clean output lock unavailable: ' + repr(exc), file=sys.stderr)
    sys.exit(3)
'@
foreach ($lockPath in @((Join-Path $Outdir 'run.lock'), (Join-Path $Outdir 'hpeli\seed42\clean\run.lock'))) {
    if (Test-Path -LiteralPath $lockPath -PathType Leaf) {
        & $python -c $lockProbe $lockPath
        if ($LASTEXITCODE -ne 0) { throw 'PiW Clean output lock unavailable; do not delete run.lock.' }
    }
}

Write-Host 'PiW preflight: hashing selected CSI/pose data and checking the model on CPU; no CUDA or optimizer updates.'
& $python @preflight --preflight
if ($LASTEXITCODE -ne 0) { throw 'PiW HPE-Li data/model preflight failed; nothing launched.' }

$config = @{
    helper = 'piw_hpeli_clean_task'; code = $code; data = $DataHome; dataset = $DatasetRoot; output = $Outdir
    conda = $env:CONDA_EXE; prefix = $EnvironmentPrefix
} | ConvertTo-Json -Compress
$config64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($config))
$payload = @'
$ErrorActionPreference = 'Stop'
$cfg = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CONFIG__')) | ConvertFrom-Json
$log = $cfg.output + '.task.' + (Get-Date -Format yyyyMMdd_HHmmss_fff) + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
try {
    $env:PYTHONUTF8 = '1'
    $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:MKL_THREADING_LAYER = 'SEQUENTIAL'
    $line = 'run --no-capture-output -p "{0}" "{0}\python.exe" -u "{1}\ATKBackd\run_piw_hpeli_clean.py" --data-home "{2}" --dataset-root "{3}" --outdir "{4}" --device cuda:0' -f $cfg.prefix,$cfg.code,$cfg.data,$cfg.dataset,$cfg.output
    $p = Start-Process -FilePath $cfg.conda -ArgumentList $line -WorkingDirectory $cfg.code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$log.stdout.log" -RedirectStandardError "$log.stderr.log"
    exit $p.ExitCode
} catch {
    $_ | Out-String | Add-Content -LiteralPath "$log.wrapper-error.log" -Encoding UTF8
    exit 1
}
'@
$payload = $payload.Replace('__CONFIG__', $config64)
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($payload))
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded"
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Outdir) | Out-Null
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
Write-Host "Controller logs: $Outdir.task.*.stdout.log / stderr.log"
Write-Host "Cell log: $Outdir\hpeli\seed42\clean\console.log"
Write-Host 'CLEAN ONLY, 200 epochs, one PiW worker. A1 is not launched. No automatic retry or process stop.'
Write-Host 'Running is not proof of progress; inspect logs. Do not sleep or shut down the machine.'
