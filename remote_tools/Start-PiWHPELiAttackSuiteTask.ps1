<#
.SYNOPSIS
Run the exploratory PiW3D HPE-Li native attack suite without BadNets.
.DESCRIPTION
DryRun is the default. Preflight and GpuSmoke are explicit, separate gates.
Launch starts one persistent Windows task after both gates pass for the exact
plan. The historical PiW LR-control v3 Clean and A1 run is read only.
#>
[CmdletBinding()]
param(
    [ValidateSet('DryRun','Preflight','GpuSmoke','Launch','Status')]
    [string]$Mode = 'DryRun',
    [ValidateSet(1,2)][int]$JobsPerGpu = 2,
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$ReferenceDir = '',
    [string]$Outdir = '',
    [string]$EnvironmentPrefix = '',
    [string]$CodeRoot = ''
)
$ErrorActionPreference = 'Stop'
if (!$ReferenceDir) { $ReferenceDir = Join-Path $DataHome 'runs\piw_hpeli_body_axis_s42_lr001_v3' }
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\piw_hpeli_native_attacks_no_badnets_s42_v1' }
if (!$EnvironmentPrefix) { $EnvironmentPrefix = Join-Path $DataHome 'envs\dose-backdoor' }
if (!$CodeRoot) { $CodeRoot = Split-Path -Parent $PSScriptRoot }
foreach ($name in @('DataHome','ReferenceDir','Outdir','EnvironmentPrefix','CodeRoot')) {
    $value = [IO.Path]::GetFullPath((Get-Variable -Name $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') { throw "Invalid local path: $name" }
    Set-Variable -Name $name -Value $value
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $CodeRoot 'ATKBackd\run_piw_hpeli_attack_suite.py'
foreach ($file in @($python,$runner)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$taskName = 'PiW-HPELi-NativeAttacks-NoBadNets-Seed42'
$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$runnerArgs = @('-u',$runner,'--data-home',$DataHome,'--reference-dir',$ReferenceDir,
    '--outdir',$Outdir,'--jobs-per-gpu',[string]$JobsPerGpu)
$flags = @{ DryRun='--dry-run'; Preflight='--preflight'; GpuSmoke='--gpu-smoke'; Status='--status' }
if ($Mode -ne 'Launch') {
    & $python @runnerArgs $flags[$Mode]
    if ($LASTEXITCODE -ne 0) { throw "PiW attack suite $Mode failed; no full training launched." }
    if ($Mode -eq 'Status') {
        Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue |
            Select-Object TaskName,State
        Get-ChildItem -Path "$Outdir.task.*.stdout.log" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
            ForEach-Object {
                Write-Host "Controller: $($_.FullName)"
                Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Tail 12
            }
    }
    return
}

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw "Task $taskName already exists. Inspect its state and logs before any new launch."
}
# Direct CLI launches must also be treated as active. No process is stopped.
$active = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
    Where-Object {
        $_.CommandLine -and $_.CommandLine.Contains('run_piw_hpeli_attack_suite.py') -and
        ($_.CommandLine.Contains($Outdir) -or $_.CommandLine.Contains('piw_native_attacks.resolved.json'))
    })
if ($active.Count) {
    $active | Select-Object ProcessId,ParentProcessId,CommandLine | Format-List
    throw 'PiW attack suite process is active; no duplicate launch.'
}
$gateHashes = @()
foreach ($kind in @('preflight','gpu_smoke')) {
    $receiptPath = "$Outdir.$kind.json"
    if (!(Test-Path -LiteralPath $receiptPath -PathType Leaf)) { throw "Missing gate: $receiptPath" }
    $record = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($record.kind -ne $kind -or $record.status -ne 'PASS' -or !$record.plan_sha256) {
        throw "Gate $kind did not pass; no launch."
    }
    $gateHashes += [string]$record.plan_sha256
}
if ($gateHashes[0] -ne $gateHashes[1]) { throw 'Preflight and GPU smoke belong to different plans.' }
# Python rechecks the receipts against the current data, code and reference hashes.
$cfg = @{ python=$python; args=($runnerArgs + '--launch'); code=$CodeRoot; output=$Outdir }
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
$action = New-ScheduledTaskAction -Execute $shell -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded" -WorkingDirectory $CodeRoot
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$principal = New-ScheduledTaskPrincipal -UserId $sid -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task = New-ScheduledTask -Action $action -Principal $principal -Settings $settings -Description 'Exploratory PiW3D HPE-Li five native attacks, excluding BadNets; seed42; two workers at most.'
Register-ScheduledTask -TaskName $taskName -InputObject $task -ErrorAction Stop | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host "Controller: $Outdir.task.*.stdout.log / stderr.log"
Write-Host "Cells: $Outdir\hpeli\seed42\<wanet_source|ftrojan|fiba|sig_source|bpp_source>\console.log"
Write-Host "Results: $Outdir\attack_comparison.md"
Write-Host 'Task started; check logs for progress. No automatic retry or process stop.'
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
