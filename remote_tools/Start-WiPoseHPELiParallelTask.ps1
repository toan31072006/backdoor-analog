<#
.SYNOPSIS
Run the isolated WiPose Clean/A1 parallel-training pilot as one persistent task.
.DESCRIPTION
Use Anaconda PowerShell on Van. JobsPerGpu is limited to one or two workers.
Clean and A1 train independently; authenticated evaluation runs after training.
No existing process, historical output, recipe, batch size or dependency is changed.
#>
[CmdletBinding()]
param(
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$DatasetRoot = '',
    [string]$Outdir = '',
    [string]$EnvironmentPrefix = '',
    [string]$ActionNpy = '',
    [string]$TaskName = 'WiPose-HPELi-Parallel-Seed42',
    [ValidateSet(1, 2)][int]$JobsPerGpu = 2,
    [switch]$CleanOnly,
    [switch]$DryRun,
    [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
if ($DryRun -and $PreflightOnly) { throw 'Choose only one of DryRun or PreflightOnly.' }
if ($TaskName -notmatch '^[A-Za-z0-9_.-]+$') { throw 'TaskName must contain only letters, numbers, underscore, dot or hyphen.' }
if ($CleanOnly -and !$PSBoundParameters.ContainsKey('TaskName')) { $TaskName = 'WiPose-HPELi-Parallel-Clean-Seed42' }
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\WiPose_processed_v1' }
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\wipose_hpeli_a1_s42_parallel_v1' }
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
$runner = Join-Path $code 'ATKBackd\run_wipose_hpeli_parallel.py'
foreach ($file in @($python, $runner)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$preflight = @('-u', $runner, '--data-home', $DataHome, '--dataset-root', $DatasetRoot, '--outdir', $Outdir, '--action-npy', $ActionNpy, '--device', 'cuda:0', '--jobs-per-gpu', [string]$JobsPerGpu)
if ($CleanOnly) { $preflight += '--clean-only' }
& $python @preflight --dry-run
if ($LASTEXITCODE -ne 0) { throw 'WiPose parallel dry-run failed; nothing launched.' }
if ($DryRun) { return }
if ($PreflightOnly) {
    & $python @preflight --preflight
    if ($LASTEXITCODE -ne 0) { throw 'WiPose parallel data/model preflight failed; nothing launched.' }
    return
}
if (!$env:CONDA_EXE -or !(Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
    throw 'Open Anaconda PowerShell first (CONDA_EXE is required).'
}

function Get-WiPoseParallelPayload([string]$ConfigBase64) {
    $template = @'
$ErrorActionPreference = 'Stop'
$cfg = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CONFIG__')) | ConvertFrom-Json
$log = $cfg.output + '.task.' + (Get-Date -Format yyyyMMdd_HHmmss_fff) + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
try {
    $env:PYTHONUTF8 = '1'
    $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:MKL_THREADING_LAYER = 'SEQUENTIAL'
    $line = 'run --no-capture-output -p "{0}" "{0}\python.exe" -u "{1}\ATKBackd\run_wipose_hpeli_parallel.py" --data-home "{2}" --dataset-root "{3}" --outdir "{4}" --action-npy "{5}" --device cuda:0 --jobs-per-gpu {6}' -f $cfg.prefix,$cfg.code,$cfg.data,$cfg.dataset,$cfg.output,$cfg.action,$cfg.jobs_per_gpu
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
function Test-SameWiPosePath($Left, $Right) {
    try { return ([IO.Path]::GetFullPath([string]$Left).TrimEnd('\') -eq [IO.Path]::GetFullPath([string]$Right).TrimEnd('\')) }
    catch { return $false }
}
function Test-WiPoseOwner($candidate) {
    try {
        $owner = Invoke-CimMethod -InputObject $candidate -MethodName GetOwnerSid -ErrorAction Stop
        return ($owner.ReturnValue -eq 0 -and $owner.Sid -eq $currentSid)
    } catch { return $false }
}
function Get-WiPoseSavedConfigs([string]$Text) {
    foreach ($match in [regex]::Matches($Text, "FromBase64String\('([A-Za-z0-9+/=]+)'\)")) {
        try {
            $saved = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($match.Groups[1].Value)) | ConvertFrom-Json
            if ($saved.helper -in @('wipose_hpeli_task', 'wipose_hpeli_parallel_task')) {
                [pscustomobject]@{Config=$saved;Base64=$match.Groups[1].Value}
            }
        } catch { continue }
    }
}
function Get-WiPoseTaskBody($task) {
    try {
        $account = $task.Principal.UserId
        if (!$account -or @($task.Actions).Count -ne 1) { return $null }
        $sid = if ($account -match '^S-1-') { $account } else {
            ([Security.Principal.NTAccount]::new($account)).Translate([Security.Principal.SecurityIdentifier]).Value
        }
        if ($sid -ne $currentSid) { return $null }
        $action = @($task.Actions)[0]
        if ($action.Execute -ne "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -or
            $action.Arguments -notmatch '(?i)^-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand ([A-Za-z0-9+/=]+)$') { return $null }
        return [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
    } catch { return $null }
}
function Test-OwnedTask($task, [switch]$DatasetOnly) {
    $text = Get-WiPoseTaskBody $task
    if (!$text) { return $false }
    foreach ($record in @(Get-WiPoseSavedConfigs $text)) {
        $saved = $record.Config
        if ($text -notmatch 'run_wipose_hpeli(?:_parallel)?\.py') { continue }
        if ($DatasetOnly) {
            if (Test-SameWiPosePath $saved.dataset $DatasetRoot) { return $true }
        # An idle authenticated task may change only the operational 1/2-slot setting.
        # Running/Queued tasks are independently rejected by the dataset-wide guard.
        } elseif ($saved.helper -eq 'wipose_hpeli_parallel_task' -and
            (Test-SameWiPosePath $saved.output $Outdir) -and (Test-SameWiPosePath $saved.code $code) -and
            (Test-SameWiPosePath $saved.data $DataHome) -and (Test-SameWiPosePath $saved.dataset $DatasetRoot) -and
            (Test-SameWiPosePath $saved.prefix $EnvironmentPrefix) -and (Test-SameWiPosePath $saved.action $ActionNpy) -and
            $saved.clean_only -is [bool] -and $saved.clean_only -eq [bool]$CleanOnly -and
            ($saved.jobs_per_gpu -is [int] -or $saved.jobs_per_gpu -is [long]) -and
            $saved.jobs_per_gpu -in @(1,2) -and
            $saved.conda -eq $env:CONDA_EXE -and $text -ceq (Get-WiPoseParallelPayload $record.Base64)) { return $true }
    }
    return $false
}
function Split-WiPoseCommand([string]$CommandLine) {
    foreach ($match in [regex]::Matches($CommandLine, '"([^"\r\n]*)"|([^\s"]+)')) {
        if ($match.Groups[1].Success) { $match.Groups[1].Value } else { $match.Groups[2].Value }
    }
}
function Get-WiPoseArgument([object[]]$Arguments, [string]$Name) {
    for ($i=0; $i -lt $Arguments.Count; $i++) {
        if ($Arguments[$i] -eq $Name -and $i+1 -lt $Arguments.Count) { return [string]$Arguments[$i+1] }
        if ($Arguments[$i].StartsWith($Name + ':', [StringComparison]::OrdinalIgnoreCase)) {
            return $Arguments[$i].Substring($Name.Length+1)
        }
    }
    return $null
}
function Test-OwnedWorker($candidate) {
    if (!$candidate -or $candidate.ProcessId -eq $PID -or
        $candidate.Name -notmatch '^(?:pythonw?|powershell|pwsh)\.exe$' -or !(Test-WiPoseOwner $candidate)) { return $false }
    $arguments = @(Split-WiPoseCommand $candidate.CommandLine)
    if ($candidate.Name -match '^python') {
        $isWiPose = @($arguments | Where-Object {
            [IO.Path]::GetFileName(([string]$_).Replace('/','\')) -match '^run_wipose_hpeli(?:_parallel)?\.py$'
        }).Count -gt 0
        if (!$isWiPose) { return $false }
        $dataset = Get-WiPoseArgument $arguments '--dataset-root'
        if ($dataset) { return (Test-SameWiPosePath $dataset $DatasetRoot) }
        $output = Get-WiPoseArgument $arguments '--outdir'
        $manifest = Get-WiPoseArgument $arguments '--_manifest'
        $candidates = @()
        if ($manifest) { $candidates += $manifest }
        elseif ($output) {
            $candidates += (Join-Path $output 'wipose_hpeli.resolved.json')
            $candidates += (Join-Path $output 'wipose_hpeli_parallel.resolved.json')
        }
        foreach ($path in $candidates) {
            if ([IO.Path]::GetFileName($path) -notin @('wipose_hpeli.resolved.json','wipose_hpeli_parallel.resolved.json')) { continue }
            try {
                if (Test-Path -LiteralPath $path -PathType Leaf) {
                    $saved = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
                    if ($saved.cfg.dataset_root) { return (Test-SameWiPosePath $saved.cfg.dataset_root $DatasetRoot) }
                }
            } catch { return $true }
        }
        $data = Get-WiPoseArgument $arguments '--data-home'
        if ($data) { return (Test-SameWiPosePath (Join-Path $data 'datasets\WiPose_processed_v1') $DatasetRoot) }
        # An owned WiPose process with an unreadable/unknown binding is not proof of idle.
        return $true
    }
    $encoded = Get-WiPoseArgument $arguments '-EncodedCommand'
    if ($encoded) {
        try {
            $text = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($encoded))
            if ($text -notmatch 'run_wipose_hpeli(?:_parallel)?\.py') { return $false }
            foreach ($record in @(Get-WiPoseSavedConfigs $text)) {
                if (Test-SameWiPosePath $record.Config.dataset $DatasetRoot) { return $true }
            }
        } catch { return $false }
        return $false
    }
    $file = Get-WiPoseArgument $arguments '-File'
    if (!$file -or [IO.Path]::GetFileName($file.Replace('/','\')) -notmatch '^Start-WiPoseHPELi(?:Parallel)?Task\.ps1$') { return $false }
    $dataset = Get-WiPoseArgument $arguments '-DatasetRoot'
    if (!$dataset) {
        $data = Get-WiPoseArgument $arguments '-DataHome'
        if (!$data) { $data = "$env:USERPROFILE\backdooranalog" }
        $dataset = Join-Path $data 'datasets\WiPose_processed_v1'
    }
    return (Test-SameWiPosePath $dataset $DatasetRoot)
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
    print('WiPose parallel output lock unavailable: ' + repr(exc), file=sys.stderr)
    sys.exit(3)
'@
function Assert-WiPoseParallelIdle {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing -and !(Test-OwnedTask $existing)) {
        throw "Task name belongs to a different account/action/output/config: $TaskName"
    }
    $activeTasks = @(Get-ScheduledTask | Where-Object {
        $_.State -in @('Running','Queued') -and (Test-OwnedTask $_ -DatasetOnly)
    })
    $workers = @(Get-CimInstance Win32_Process | Where-Object { Test-OwnedWorker $_ })
    if ($activeTasks.Count -or $workers.Count) {
        $workers | Format-List ProcessId,ParentProcessId,CommandLine
        throw 'WiPose dataset is active in a serial/parallel task, worker or helper. No duplicate launch; no process is stopped.'
    }
    foreach ($lockPath in @((Join-Path $Outdir 'run.lock'),
        (Join-Path $Outdir 'hpeli\seed42\clean\run.lock'),
        (Join-Path $Outdir 'hpeli\seed42\a1\run.lock'))) {
        if (Test-Path -LiteralPath $lockPath -PathType Leaf) {
            & $python -c $lockProbe $lockPath
            if ($LASTEXITCODE -ne 0) { throw 'WiPose parallel output lock unavailable; do not delete run.lock.' }
        }
    }
}
Assert-WiPoseParallelIdle
Write-Host 'WiPose parallel preflight: CPU data/model/trigger checks; no CUDA, training or output writes.'
& $python @preflight --preflight
if ($LASTEXITCODE -ne 0) { throw 'WiPose parallel data/model preflight failed; nothing launched.' }
# Recheck for another helper or worker that appeared during the read-only preflight.
Assert-WiPoseParallelIdle

$config = @{
    helper='wipose_hpeli_parallel_task'; code=$code; data=$DataHome; dataset=$DatasetRoot; output=$Outdir
    conda=$env:CONDA_EXE; prefix=$EnvironmentPrefix; action=$ActionNpy; clean_only=[bool]$CleanOnly; jobs_per_gpu=$JobsPerGpu
} | ConvertTo-Json -Compress
$config64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($config))
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes((Get-WiPoseParallelPayload $config64)))
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded"
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Outdir) | Out-Null
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
Write-Host "Controller logs: $Outdir.task.*.stdout.log / stderr.log"
Write-Host "Cell logs: $Outdir\hpeli\seed42\<clean|a1>\console.log"
Write-Host "At most $JobsPerGpu training worker(s); Clean/A1 evaluation follows training. No automatic retry, batch fallback or process stop."
Write-Host 'Do not start the old serial helper on this dataset. Running is not proof of progress; inspect logs; no sleep/shutdown.'
