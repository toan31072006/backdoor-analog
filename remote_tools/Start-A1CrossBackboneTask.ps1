<#
.SYNOPSIS
Run the separate MetaFi++ / GraphPose-Fi MM-Fi experiment through Task Scheduler.
.DESCRIPTION
Run on Van, not the Admin client. The task uses the current Windows account,
local unencrypted files and an existing Conda environment; no password is saved.
Two jobs/GPU as explicitly requested. GraphPose-Fi loads 32 samples by default
and accumulates to effective batch 256; this fixed adaptation is not OOM fallback.
No existing process/task is stopped and no checkpoint or result is deleted.
PilotA1Seed42 selects only the two A1 cells and their same-backbone clean controls.
#>
[CmdletBinding()]
param(
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$Outdir = "$env:USERPROFILE\backdooranalog\runs\mmfi_cross_backbones_s42_43_44_v1",
    [string]$EnvironmentPrefix = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor",
    [ValidateSet('metafiplusplus', 'graphposefi')]
    [string[]]$Backbones = @('metafiplusplus', 'graphposefi'),
    [ValidateSet(1, 2)][int]$JobsPerGpu = 2,
    [ValidateSet(32, 64, 128, 256)][int]$GraphLoadBatchSize = 32,
    [string]$TaskName = 'MMFI-A1-CrossBackbones',
    [switch]$PilotClean,
    [switch]$PilotA1Seed42,
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
if (!$Backbones.Count -or ($Backbones | Sort-Object -Unique).Count -ne $Backbones.Count) {
    throw 'Empty or duplicate backbones.'
}
if ($PilotClean -and $PilotA1Seed42) {
    throw 'Use PilotClean or PilotA1Seed42, not both.'
}
if ($PilotA1Seed42 -and $Backbones.Count -ne 2) {
    throw 'PilotA1Seed42 requires both backbones; it does not change the saved plan.'
}
$cells = @()
if ($PilotA1Seed42) {
    $cells = @('metafiplusplus__s42__a1', 'graphposefi__s42__a1')
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code 'ATKBackd\run_a1_cross_backbones.py'
foreach ($file in @($python, $runner)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$env:PYTHONUTF8 = '1'
$preflight = @('-u', $runner, '--data-home', $DataHome, '--outdir', $Outdir,
    '--devices', 'cuda:0', '--jobs-per-gpu', "$JobsPerGpu", '--num-workers', '0',
    '--graph-load-batch-size', "$GraphLoadBatchSize", '--backbones') + @($Backbones)
if ($PilotClean) { $preflight += '--pilot-clean' }
if ($cells.Count) { $preflight += @('--cells') + $cells }
# Pure plan validation; no data, checkpoints, GPU, task registration or writes.
& $python @preflight --dry-run
if ($LASTEXITCODE -ne 0) { throw 'Cross-backbone dry-run failed; nothing launched.' }
if ($DryRun) { return }
if (!$env:CONDA_EXE -or !(Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
    throw 'Open Anaconda PowerShell first (CONDA_EXE is required).'
}

$currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$runnerPattern = '(?i)(?:^|\s|["\\/])run_a1_cross_backbones\.py(?=["\s]|$)'
$outputEscaped = [regex]::Escape($Outdir)
$manifestEscaped = $outputEscaped + '[\\/]cross_backbones\.resolved\.json'
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
            if ($text -notmatch 'run_a1_cross_backbones\.py') { continue }
            foreach ($match in [regex]::Matches($text, "FromBase64String\('([A-Za-z0-9+/=]+)'\)")) {
                $saved = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($match.Groups[1].Value)) | ConvertFrom-Json
                if ($saved.helper -eq 'a1_cross_backbone_task' -and $saved.output -eq $Outdir) { return $true }
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
    throw 'This output is still active. Inspect its logs; this helper never stops an owner or starts duplicate workers.'
}

# GPU idleness / an empty CIM filter does not prove that the controller lock
# is free. Probe an existing byte-zero lock without creating, truncating or
# deleting it. The real controller must still acquire its own lock (race-safe).
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
    print('Output lock probe failed: ' + repr(exc), file=sys.stderr)
    sys.exit(3)
'@
    & $python -c $probe $rootLock
    if ($LASTEXITCODE -ne 0) {
        throw 'Output lock is unavailable. No task registered. Inspect its owner or access error; do not delete run.lock.'
    }
}

# Encoded JSON keeps paths and flags as data, not PowerShell expressions.
$config = @{
    helper = 'a1_cross_backbone_task'; code = $code; data = $DataHome; output = $Outdir
    conda = $env:CONDA_EXE; prefix = $EnvironmentPrefix; jobs = $JobsPerGpu
    backbones = @($Backbones); pilot = [bool]$PilotClean; graphBatch = $GraphLoadBatchSize
    cells = @($cells)
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
    $line = 'run --no-capture-output -p "{0}" "{0}\python.exe" -u "{1}\ATKBackd\run_a1_cross_backbones.py" --data-home "{2}" --outdir "{3}" --devices cuda:0 --jobs-per-gpu {4} --num-workers 0 --graph-load-batch-size {5} --backbones {6}' -f $cfg.prefix,$cfg.code,$cfg.data,$cfg.output,$cfg.jobs,$cfg.graphBatch,($cfg.backbones -join ' ')
    if ($cfg.pilot) { $line += ' --pilot-clean' }
    if ($cfg.cells.Count) { $line += ' --cells ' + ($cfg.cells -join ' ') }
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
$logParent = Split-Path -Parent $Outdir
New-Item -ItemType Directory -Force -Path $logParent | Out-Null
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
Write-Host "Controller logs: $Outdir.task.*.stdout.log / stderr.log"
Write-Host "Cell logs: $Outdir\<backbone>\seed<42|43|44>\<key>\console.log"
Write-Host 'Running alone is not proof of progress. No automatic retry. Do not sleep/shutdown the machine.'
