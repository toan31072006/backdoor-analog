<#
.SYNOPSIS
Run the unchanged A1 defenses through a persistent one/two-slot Windows task.
.DESCRIPTION
Run on Van, not on the local Admin client. No password is stored. All inputs
must be local, unencrypted paths accessible to the current Windows account.
-StopExisting explicitly stops ONLY T5 tasks/workers bound to this exact output;
the last unsaved epoch may repeat. No checkpoints, locks or results are deleted.
#>
[CmdletBinding()]
param(
    [string]$SourceOutdir = "$env:USERPROFILE\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1",
    [string]$Outdir = "$env:USERPROFILE\backdooranalog\runs\mmfi_a1_defenses_s42_43_44_v1",
    [string]$EnvironmentPrefix = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor",
    [ValidateSet(1, 2)][int]$JobsPerGpu = 2,
    [ValidateSet(42, 43, 44)][int[]]$Seeds = @(42),
    [string]$TaskName = 'MMFI-A1-Defense-Parallel',
    [switch]$StopExisting,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$code = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$Outdir = [IO.Path]::GetFullPath($Outdir).TrimEnd('\')
$SourceOutdir = [IO.Path]::GetFullPath($SourceOutdir).TrimEnd('\')
$EnvironmentPrefix = [IO.Path]::GetFullPath($EnvironmentPrefix).TrimEnd('\')
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code 'ATKBackd\run_a1_defenses_parallel.py'
$env:PYTHONUTF8 = '1'
foreach ($file in @($python, $runner, (Join-Path $SourceOutdir 'three_tables.resolved.json'))) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
if (($Seeds | Sort-Object -Unique).Count -ne $Seeds.Count) { throw 'Duplicate seeds' }

# Scientific and path validation is read-only. A failed preflight must not stop
# the previous job, register tasks or write a replacement scientific manifest.
& $python -u $runner --source-outdir $SourceOutdir --outdir $Outdir --device cuda:0 --jobs-per-gpu $JobsPerGpu --seeds $Seeds --dry-run
if ($LASTEXITCODE -ne 0) { throw 'T5 dry-run failed; nothing stopped or launched.' }
if ($DryRun) { return }
if (!$env:CONDA_EXE -or !(Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
    throw 'Open Anaconda PowerShell first (CONDA_EXE is required).'
}

$pattern = [regex]::Escape($Outdir) + '(?=[\\/"\s]|$)'
function Get-OwnedWorkers {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -match 'run_a1_defenses(?:_parallel)?\.py' -and
        $_.CommandLine -match $pattern
    })
}
function Test-OwnedTask($task) {
    try {
        $account = $task.Principal.UserId
        if (!$account) { return $false }
        $sid = if ($account -match '^S-1-') { $account } else {
            ([Security.Principal.NTAccount]::new($account)).Translate([Security.Principal.SecurityIdentifier]).Value
        }
        if ($sid -ne [Security.Principal.WindowsIdentity]::GetCurrent().User.Value) { return $false }
    } catch { return $false }
    foreach ($action in $task.Actions) {
        if ($action.Arguments -match '(?i)-EncodedCommand\s+([A-Za-z0-9+/=]+)') {
            try {
                $text = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
                $taskPathPattern = [regex]::Escape($Outdir) + '(?=[\\/"''\s]|$)'
                if ($text -match $taskPathPattern -and $text -match 'run_a1_defenses(?:_parallel)?\.py') { return $true }
                # This helper encodes its config separately inside the action.
                if ($text -match 'run_a1_defenses_parallel\.py') {
                    foreach ($match in [regex]::Matches($text, "FromBase64String\('([A-Za-z0-9+/=]+)'\)")) {
                        $saved = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($match.Groups[1].Value)) | ConvertFrom-Json
                        if ($saved.output -eq $Outdir) { return $true }
                    }
                }
            } catch { }
        }
    }
    return $false
}
$tasks = @(Get-ScheduledTask | Where-Object {
    $_.TaskName -like 'MMFI-A1-Defense-*' -or $_.TaskName -eq $TaskName
} | Where-Object { Test-OwnedTask $_ })
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and !(Test-OwnedTask $existing)) {
    throw "Task name already belongs to a different action: $TaskName"
}
$runningTasks = @($tasks | Where-Object { $_.State -in @('Running', 'Queued') })
$workers = @(Get-OwnedWorkers)
if (($runningTasks.Count -or $workers.Count) -and !$StopExisting) {
    $workers | Format-List ProcessId,ParentProcessId,CommandLine
    throw 'T5 still active. Use -StopExisting explicitly to resume it with the new slot count.'
}
if ($StopExisting) {
    foreach ($task in $runningTasks) {
        Write-Host "Stopping owned task: $($task.TaskName)"
        Stop-ScheduledTask -InputObject $task
    }
    # Stop controllers first so they cannot enqueue another worker during switch.
    foreach ($worker in (@(Get-OwnedWorkers) | Sort-Object { $_.CommandLine -match '--_cell' })) {
        $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($worker.ProcessId)"
        if ($current -and $current.CreationDate -eq $worker.CreationDate -and
            $current.CommandLine -match $pattern -and
            $current.CommandLine -match 'run_a1_defenses(?:_parallel)?\.py') {
            Write-Host "Stopping owned T5 PID $($worker.ProcessId); saved checkpoint is retained."
            Stop-Process -Id $worker.ProcessId -Force -ErrorAction Stop
        }
    }
    for ($attempt = 0; $attempt -lt 30 -and @(Get-OwnedWorkers).Count; $attempt++) {
        Start-Sleep -Milliseconds 200
    }
    if (@(Get-OwnedWorkers).Count) { throw 'T5 workers still active; do not start another controller.' }
}

# Capture runtime paths as JSON/Base64, never interpolate them as executable code.
$config = @{
    code = $code; source = $SourceOutdir; output = $Outdir
    conda = $env:CONDA_EXE; prefix = $EnvironmentPrefix
    jobs = $JobsPerGpu; seeds = @($Seeds)
} | ConvertTo-Json -Compress
$config64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($config))
$payload = @'
$ErrorActionPreference = 'Stop'
$cfg = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CONFIG__')) | ConvertFrom-Json
$log = $cfg.output + '.task.' + (Get-Date -Format yyyyMMdd_HHmmss_fff)
try {
    $env:PYTHONUTF8 = '1'
    $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:MKL_THREADING_LAYER = 'SEQUENTIAL'
    $line = 'run --no-capture-output -p "{0}" "{0}\python.exe" -u "{1}\ATKBackd\run_a1_defenses_parallel.py" --source-outdir "{2}" --outdir "{3}" --device cuda:0 --jobs-per-gpu {4} --seeds {5}' -f $cfg.prefix,$cfg.code,$cfg.source,$cfg.output,$cfg.jobs,($cfg.seeds -join ' ')
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
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
Write-Host "Controller logs: $Outdir.task.*.stdout.log / stderr.log"
Write-Host "Cell logs: $Outdir\seed42\a1\<defense>\console.log"
Write-Host 'Running state alone is not proof of progress: inspect logs. Do not sleep/shutdown the machine.'
