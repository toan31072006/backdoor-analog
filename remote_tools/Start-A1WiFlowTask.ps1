<#
.SYNOPSIS
Run real WiFlow MM-Fi Clean then A1, seed42, as one persistent worker slot.
.DESCRIPTION
Run on Van in Anaconda PowerShell. Does not pause/kill Graph or MetaFi++.
One WiFlow slot alongside the independently owned MetaFi++ slot is two active
training jobs total. A suspended Graph still holds its RAM/VRAM and locks.
#>
[CmdletBinding()]
param(
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$Outdir = "$env:USERPROFILE\backdooranalog\runs\mmfi_wiflow_a1_s42_v1",
    [string]$EnvironmentPrefix = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor",
    [string]$TaskName = 'MMFI-A1-WiFlow-Seed42',
    [switch]$CleanOnly,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$code = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
foreach ($name in @('DataHome', 'Outdir', 'EnvironmentPrefix')) {
    $value = [IO.Path]::GetFullPath((Get-Variable -Name $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') {
        throw "$name must be a local path without double quotes or newlines."
    }
    Set-Variable -Name $name -Value $value
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code 'ATKBackd\run_a1_wiflow.py'
foreach ($file in @($python, $runner)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$env:PYTHONUTF8 = '1'
$preflight = @('-u', $runner, '--data-home', $DataHome, '--outdir', $Outdir, '--device', 'cuda:0')
if ($CleanOnly) { $preflight += '--clean-only' }
& $python @preflight --dry-run
if ($LASTEXITCODE -ne 0) { throw 'WiFlow dry-run failed; nothing launched.' }
if ($DryRun) { return }
if (!$env:CONDA_EXE -or !(Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
    throw 'Open Anaconda PowerShell first (CONDA_EXE is required).'
}

$currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$runnerPattern = '(?i)(?:^|\s|["\\/])run_a1_wiflow\.py(?=["\s]|$)'
$outputEscaped = [regex]::Escape($Outdir)
$manifestEscaped = $outputEscaped + '[\\/]wiflow_a1\.resolved\.json'
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
        if ($sid -ne $currentSid) { return $false }
        foreach ($action in $task.Actions) {
            if ($action.Arguments -notmatch '(?i)-EncodedCommand\s+([A-Za-z0-9+/=]+)') { continue }
            $text = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
            if ($text -notmatch 'run_a1_wiflow\.py') { continue }
            foreach ($match in [regex]::Matches($text, "FromBase64String\('([A-Za-z0-9+/=]+)'\)")) {
                $saved = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($match.Groups[1].Value)) | ConvertFrom-Json
                if ($saved.helper -eq 'a1_wiflow_task' -and $saved.output -eq $Outdir) { return $true }
            }
        }
    } catch { return $false }
    return $false
}
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and !(Test-OwnedTask $existing)) {
    throw "Task name belongs to a different account/action/output: $TaskName"
}
$activeTasks = @(Get-ScheduledTask | Where-Object {
    $_.State -in @('Running', 'Queued') -and (Test-OwnedTask $_)
})
$workers = @(Get-CimInstance Win32_Process | Where-Object { Test-OwnedWorker $_ })
if ($activeTasks.Count -or $workers.Count) {
    $workers | Format-List ProcessId,ParentProcessId,CommandLine
    throw 'WiFlow output is active. No duplicate launch; no process is stopped.'
}
$rootLock = Join-Path $Outdir 'run.lock'
if (Test-Path -LiteralPath $rootLock -PathType Leaf) {
    $probe = @'
import sys
import msvcrt
try:
    with open(sys.argv[1], 'r+b') as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
except OSError as exc:
    print('WiFlow output lock unavailable: ' + repr(exc), file=sys.stderr)
    sys.exit(3)
'@
    & $python -c $probe $rootLock
    if ($LASTEXITCODE -ne 0) { throw 'WiFlow output lock unavailable; do not delete run.lock.' }
}

$config = @{
    helper = 'a1_wiflow_task'; code = $code; data = $DataHome; output = $Outdir
    conda = $env:CONDA_EXE; prefix = $EnvironmentPrefix; cleanOnly = [bool]$CleanOnly
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
    $line = 'run --no-capture-output -p "{0}" "{0}\python.exe" -u "{1}\ATKBackd\run_a1_wiflow.py" --data-home "{2}" --outdir "{3}" --device cuda:0' -f $cfg.prefix,$cfg.code,$cfg.data,$cfg.output
    if ($cfg.cleanOnly) { $line += ' --clean-only' }
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
Write-Host "Cell logs: $Outdir\wiflow\seed42\<clean|a1>\console.log"
Write-Host 'REAL training, one WiFlow slot; no automatic retry. Running is not proof of progress.'
