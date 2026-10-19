<#
.SYNOPSIS
Run fixed-u CLP on the three frozen A1 victims through a persistent Windows task.
.DESCRIPTION
Run on Van, not on the local Admin client. No password is stored. Inputs must
be local, unencrypted paths accessible to the current Windows account.
V2 outputs are read-only reporting references; this helper never stops V2 tasks.
-StopExisting stops ONLY current-user CLP tasks/workers for this exact output.
No checkpoint, manifest, lock or historical result is deleted.
#>
[CmdletBinding()]
param(
    [string]$SourceOutdir = "$env:USERPROFILE\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1",
    [string]$Outdir = "$env:USERPROFILE\backdooranalog\runs\mmfi_a1_clp_s42_43_44_v1",
    [string]$ReferenceOutdir = "$env:USERPROFILE\backdooranalog\runs\mmfi_a1_defenses_ft_anp_nad_sam_s42_43_44_v2",
    [string]$EnvironmentPrefix = "$env:USERPROFILE\backdooranalog\envs\dose-backdoor",
    [ValidateSet(1, 2)][int]$JobsPerGpu = 2,
    [ValidateSet(42, 43, 44)][int[]]$Seeds = @(42, 43, 44),
    [string]$TaskName = 'MMFI-A1-CLP',
    [switch]$StopExisting,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$code = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
foreach ($name in @('Outdir', 'SourceOutdir', 'EnvironmentPrefix')) {
    $value = [IO.Path]::GetFullPath((Get-Variable -Name $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') {
        throw "$name must be a local path without double quotes or newlines."
    }
    Set-Variable -Name $name -Value $value
}
$explicitReference = $PSBoundParameters.ContainsKey('ReferenceOutdir')
if ($ReferenceOutdir) {
    $ReferenceOutdir = [IO.Path]::GetFullPath($ReferenceOutdir).TrimEnd('\')
    if ($ReferenceOutdir.StartsWith('\\') -or $ReferenceOutdir -match '["\r\n]') {
        throw 'ReferenceOutdir must be a local path without double quotes or newlines.'
    }
    if (!(Test-Path -LiteralPath $ReferenceOutdir -PathType Container)) {
        if ($explicitReference) { throw "Missing explicit reference directory: $ReferenceOutdir" }
        $ReferenceOutdir = ''
    }
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code 'ATKBackd\run_a1_clp.py'
$env:PYTHONUTF8 = '1'
foreach ($file in @($python, $runner, (Join-Path $SourceOutdir 'three_tables.resolved.json'))) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
if (!$Seeds.Count -or ($Seeds | Sort-Object -Unique).Count -ne $Seeds.Count) { throw 'Empty or duplicate seeds' }

# No stop, task registration or output write occurs before this read-only gate.
$preflight = @('-u', $runner, '--source-outdir', $SourceOutdir, '--outdir', $Outdir,
    '--device', 'cuda:0', '--jobs-per-gpu', "$JobsPerGpu", '--seeds') + @($Seeds | ForEach-Object { "$_" })
if ($ReferenceOutdir) { $preflight += @('--reference-outdir', $ReferenceOutdir) }
& $python @preflight --dry-run
if ($LASTEXITCODE -ne 0) { throw 'T5 CLP dry-run failed; nothing stopped or launched.' }
if ($DryRun) { return }
if (!$env:CONDA_EXE -or !(Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
    throw 'Open Anaconda PowerShell first (CONDA_EXE is required).'
}

$currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$runnerPattern = '(?i)(?:^|\s|["\\/])run_a1_clp\.py(?=["\s]|$)'
$outputEscaped = [regex]::Escape($Outdir)
$manifestEscaped = $outputEscaped + '[\\/]a1_clp\.resolved\.json'
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
function Get-OwnedWorkers {
    @(Get-CimInstance Win32_Process | Where-Object { Test-OwnedWorker $_ })
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
            if ($text -notmatch 'run_a1_clp\.py') { continue }
            foreach ($match in [regex]::Matches($text, "FromBase64String\('([A-Za-z0-9+/=]+)'\)")) {
                $saved = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($match.Groups[1].Value)) | ConvertFrom-Json
                if ($saved.helper -eq 'a1_clp_task' -and $saved.output -eq $Outdir) { return $true }
            }
        }
    } catch { return $false }
    return $false
}
$tasks = @(Get-ScheduledTask | Where-Object { Test-OwnedTask $_ })
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and !(Test-OwnedTask $existing)) { throw "Task name belongs to a different action: $TaskName" }
$runningTasks = @($tasks | Where-Object { $_.State -in @('Running', 'Queued') })
$workers = @(Get-OwnedWorkers)
if (($runningTasks.Count -or $workers.Count) -and !$StopExisting) {
    $workers | Format-List ProcessId,ParentProcessId,CommandLine
    throw 'CLP still active. Use -StopExisting explicitly to resume this output.'
}
if ($StopExisting) {
    foreach ($task in $runningTasks) { Stop-ScheduledTask -InputObject $task }
    # Stop controller first; then recheck PID creation date, binding and owner.
    foreach ($worker in (@(Get-OwnedWorkers) | Sort-Object { $_.CommandLine -match '--_cell' })) {
        $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($worker.ProcessId)"
        if ($current -and $current.CreationDate -eq $worker.CreationDate -and (Test-OwnedWorker $current)) {
            Write-Host "Stopping owned CLP PID $($worker.ProcessId); saved checkpoint is retained."
            Stop-Process -Id $worker.ProcessId -Force -ErrorAction Stop
        }
    }
    for ($attempt = 0; $attempt -lt 30 -and @(Get-OwnedWorkers).Count; $attempt++) {
        Start-Sleep -Milliseconds 200
    }
    if (@(Get-OwnedWorkers).Count) { throw 'CLP workers remain; do not start another controller.' }
}

# Paths are encoded data, not interpolated PowerShell expressions.
$config = @{
    helper = 'a1_clp_task'; code = $code; source = $SourceOutdir; output = $Outdir
    reference = $ReferenceOutdir; conda = $env:CONDA_EXE; prefix = $EnvironmentPrefix
    jobs = $JobsPerGpu; seeds = @($Seeds)
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
    $line = 'run --no-capture-output -p "{0}" "{0}\python.exe" -u "{1}\ATKBackd\run_a1_clp.py" --source-outdir "{2}" --outdir "{3}" --device cuda:0 --jobs-per-gpu {4} --seeds {5}' -f $cfg.prefix,$cfg.code,$cfg.source,$cfg.output,$cfg.jobs,($cfg.seeds -join ' ')
    if ($cfg.reference) { $line += ' --reference-outdir "{0}"' -f $cfg.reference }
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
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
Write-Host "Controller logs: $Outdir.task.*.stdout.log / stderr.log"
Write-Host "Cell logs: $Outdir\seed<42|43|44>\a1\clp\console.log"
Write-Host 'Running state alone is not proof of progress. Do not sleep/shutdown the machine.'
