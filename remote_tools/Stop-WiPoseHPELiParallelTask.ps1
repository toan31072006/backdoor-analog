<#
.SYNOPSIS
Audit, back up, and explicitly stop only the old WiPose parallel lineage.
.DESCRIPTION
Default is read-only. -Stop creates a new verified backup before stopping the
authenticated task/controller and freshly identified workers. Completed atomic
epoch checkpoints are preserved. Unsaved work in the current epoch is lost.
This helper never requests a force-save, removes a lock, or kills another run.
#>
[CmdletBinding()]
param(
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$DatasetRoot = '',
    [string]$Outdir = '',
    [string]$EnvironmentPrefix = '',
    [string]$ActionNpy = '',
    [string]$TaskName = 'WiPose-HPELi-Parallel-Seed42',
    [string]$BackupRoot = '',
    [switch]$Stop
)
$ErrorActionPreference = 'Stop'
if ($TaskName -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid TaskName.' }
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\WiPose_processed_v1' }
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\wipose_hpeli_a1_s42_parallel_v1' }
if (!$EnvironmentPrefix) { $EnvironmentPrefix = Join-Path $DataHome 'envs\dose-backdoor' }
if (!$ActionNpy) { $ActionNpy = Join-Path $DataHome 'actions\data_bend.npy' }
if (!$BackupRoot) { $BackupRoot = Join-Path $DataHome 'backups\wipose_parallel_before_fast' }
$code = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
foreach ($name in @('DataHome','DatasetRoot','Outdir','EnvironmentPrefix','ActionNpy','BackupRoot')) {
    $value = [IO.Path]::GetFullPath((Get-Variable -Name $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') { throw "$name must be a local path without double quotes or newlines." }
    Set-Variable -Name $name -Value $value
}
if ($BackupRoot -eq $Outdir -or $BackupRoot.StartsWith($Outdir+'\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'BackupRoot must be outside the historical output.'
}
$currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code 'ATKBackd\run_wipose_hpeli_parallel.py'
$manifestPath = Join-Path $Outdir 'wipose_hpeli_parallel.resolved.json'

function Test-WiPoseStopPath($Left, $Right) {
    try { return ([IO.Path]::GetFullPath([string]$Left).TrimEnd('\') -eq [IO.Path]::GetFullPath([string]$Right).TrimEnd('\')) }
    catch { return $false }
}
function Split-WiPoseStopCommand([string]$CommandLine) {
    foreach ($match in [regex]::Matches($CommandLine, '"([^"\r\n]*)"|([^\s"]+)')) {
        if ($match.Groups[1].Success) { $match.Groups[1].Value } else { $match.Groups[2].Value }
    }
}
function Get-WiPoseStopArgument([object[]]$Arguments, [string]$Name) {
    $hits = @()
    for ($i=0; $i -lt $Arguments.Count; $i++) {
        if ($Arguments[$i] -eq $Name -and $i+1 -lt $Arguments.Count) { $hits += [string]$Arguments[$i+1] }
    }
    if ($hits.Count -eq 1) { return $hits[0] }
    return $null
}
function Test-WiPoseStopOwner($candidate) {
    try {
        $owner = Invoke-CimMethod -InputObject $candidate -MethodName GetOwnerSid -ErrorAction Stop
        return ($owner.ReturnValue -eq 0 -and $owner.Sid -eq $currentSid)
    } catch { return $false }
}
function Test-WiPoseStopRunner($candidate) {
    if (!$candidate -or $candidate.ProcessId -eq $PID -or $candidate.Name -ne 'python.exe' -or
        !$candidate.CreationDate -or !(Test-WiPoseStopPath $candidate.ExecutablePath $python) -or
        !(Test-WiPoseStopOwner $candidate)) { return $false }
    $arguments = @(Split-WiPoseStopCommand $candidate.CommandLine)
    $runnerIndex=1
    if ($arguments.Count -gt 1 -and $arguments[1] -eq '-u') { $runnerIndex=2 }
    if ($arguments.Count -le $runnerIndex -or !(Test-WiPoseStopPath $arguments[$runnerIndex] $runner)) { return $false }
    $output = Get-WiPoseStopArgument $arguments '--outdir'
    if (!(Test-WiPoseStopPath $output $Outdir)) { return $false }
    foreach ($pair in @(@('--dataset-root',$DatasetRoot), @('--data-home',$DataHome), @('--action-npy',$ActionNpy))) {
        $explicit=Get-WiPoseStopArgument $arguments $pair[0]
        if ($explicit -and !(Test-WiPoseStopPath $explicit $pair[1])) { return $false }
    }
    $binding = Get-WiPoseStopArgument $arguments '--_manifest'
    if ($binding -and !(Test-WiPoseStopPath $binding $manifestPath)) { return $false }
    return $true
}
function Get-WiPoseOldPayload([string]$ConfigBase64) {
    # Parse the frozen helper; never dot-source/execute its launcher.
    $tokens=$null; $errors=$null
    $ast=[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'Start-WiPoseHPELiParallelTask.ps1'), [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw 'Frozen parallel helper cannot be parsed.' }
    $templates=@($ast.FindAll({param($node)
        $node -is [System.Management.Automation.Language.StringConstantExpressionAst] -and
        $node.Value.StartsWith("`$ErrorActionPreference = 'Stop'") -and $node.Value.Contains('$cfg =')
    }, $true))
    if ($templates.Count -ne 1) { throw 'Frozen parallel wrapper template is ambiguous.' }
    return $templates[0].Value.Replace('__CONFIG__', $ConfigBase64)
}
function Assert-OwnedWiPoseStopTask($task) {
    if (!$task -or @($task).Count -ne 1 -or @($task.Actions).Count -ne 1) { throw 'Expected exactly one authenticated old WiPose task.' }
    $account=$task.Principal.UserId
    try {
        $sid = if ($account -match '^S-1-') { $account } else {
            ([Security.Principal.NTAccount]::new($account)).Translate([Security.Principal.SecurityIdentifier]).Value
        }
    } catch { throw 'Cannot authenticate task account.' }
    if ($sid -ne $currentSid) { throw 'Task belongs to another account.' }
    $action=@($task.Actions)[0]
    if ($action.Execute -ne "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -or
        $action.Arguments -notmatch '(?i)^-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand ([A-Za-z0-9+/=]+)$') {
        throw 'Task action is not the frozen hidden WiPose wrapper.'
    }
    $body=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1]))
    $configs=@([regex]::Matches($body, "FromBase64String\('([A-Za-z0-9+/=]+)'\)"))
    if ($configs.Count -ne 1) { throw 'Task config provenance is ambiguous.' }
    $base64=$configs[0].Groups[1].Value
    $cfg=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($base64)) | ConvertFrom-Json
    if ($cfg.helper -ne 'wipose_hpeli_parallel_task' -or $cfg.clean_only -isnot [bool] -or $cfg.clean_only -or
        ($cfg.jobs_per_gpu -isnot [int] -and $cfg.jobs_per_gpu -isnot [long]) -or
        $cfg.jobs_per_gpu -notin @(1,2) -or !(Test-WiPoseStopPath $cfg.code $code) -or
        !(Test-WiPoseStopPath $cfg.data $DataHome) -or !(Test-WiPoseStopPath $cfg.dataset $DatasetRoot) -or
        !(Test-WiPoseStopPath $cfg.output $Outdir) -or !(Test-WiPoseStopPath $cfg.prefix $EnvironmentPrefix) -or
        !(Test-WiPoseStopPath $cfg.action $ActionNpy) -or !$cfg.conda -or
        !(Test-Path -LiteralPath $cfg.conda -PathType Leaf) -or $body -cne (Get-WiPoseOldPayload $base64)) {
        throw 'Task output/config/provenance mismatch; no process will be stopped.'
    }
}
function Assert-NoWiPoseSaving {
    $saving=@(Get-ChildItem -LiteralPath $Outdir -Recurse -File -Filter '*.saving' -ErrorAction Stop)
    if ($saving.Count) { throw 'Atomic .saving file present. Refusing further stop; rerun this helper after the writer finishes. Never delete it. If its writer already stopped, preserve files and inspect the backup.' }
}
function Get-WiPoseImmutableFiles {
    if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'Missing old parallel manifest.' }
    $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schema -ne 1 -or $manifest.profile -ne 'wipose_2d_hpeli_clean_a1_seed42_parallel_v1' -or
        !(Test-WiPoseStopPath $manifest.cfg.outdir $Outdir) -or !(Test-WiPoseStopPath $manifest.cfg.dataset_root $DatasetRoot) -or
        !(Test-WiPoseStopPath $manifest.cfg.data_home $DataHome) -or !(Test-WiPoseStopPath $manifest.cfg.action_npy $ActionNpy) -or
        $manifest.cfg.seed -ne 42 -or $manifest.plan_sha256 -notmatch '^[a-f0-9]{64}$' -or !$manifest.sources) {
        throw 'Old parallel manifest profile/output/input binding mismatch.'
    }
    [pscustomobject]@{Path=$manifestPath; Relative='output\wipose_hpeli_parallel.resolved.json'}
    foreach ($source in $manifest.sources.PSObject.Properties) {
        $path=[IO.Path]::GetFullPath((Join-Path (Join-Path $code 'ATKBackd') $source.Name))
        if (!$path.StartsWith((Join-Path $code 'ATKBackd')+'\', [StringComparison]::OrdinalIgnoreCase) -or
            !(Test-Path -LiteralPath $path -PathType Leaf) -or $source.Value -notmatch '^[a-f0-9]{64}$' -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $source.Value) {
            throw 'Frozen source hash/path mismatch. Preserve old files; no stop.'
        }
        [pscustomobject]@{Path=$path; Relative=('source\ATKBackd\'+$source.Name.Replace('/','\'))}
    }
    $oldHelper=Join-Path $PSScriptRoot 'Start-WiPoseHPELiParallelTask.ps1'
    [pscustomobject]@{Path=$oldHelper;Relative='source\remote_tools\Start-WiPoseHPELiParallelTask.ps1'}
    foreach ($stage in @('clean','a1')) {
        $poison=Join-Path $Outdir "hpeli\seed42\$stage\poison.json"
        if (Test-Path -LiteralPath $poison -PathType Leaf) {
            [pscustomobject]@{Path=$poison;Relative="output\hpeli\seed42\$stage\poison.json"}
        }
    }
}
function Get-WiPoseCheckpoints {
    foreach ($stage in @('clean','a1')) {
        $path=Join-Path $Outdir "hpeli\seed42\$stage\checkpoint.pt"
        if (!(Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -le 0) {
            throw "Missing/empty last complete $stage checkpoint. Nothing stopped."
        }
        [pscustomobject]@{Path=$path;Relative="output\hpeli\seed42\$stage\checkpoint.pt"}
    }
}
function Copy-WiPoseVerifiedSnapshot($Files, [string]$Destination) {
    $records=@()
    foreach ($file in $Files) {
        $target=Join-Path $Destination $file.Relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $file.Path -Destination $target -ErrorAction Stop
        $hash=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        $length=(Get-Item -LiteralPath $target).Length
        if ($hash -ne (Get-FileHash -LiteralPath $file.Path -Algorithm SHA256).Hash -or
            $length -ne (Get-Item -LiteralPath $file.Path).Length) {
            throw 'Snapshot changed during backup; no further stop attempted. Rerun with a new backup.'
        }
        $records += [pscustomobject]@{source=$file.Path;copy=$target;sha256=$hash;bytes=$length}
    }
    $records | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Destination 'hashes.json') -Encoding UTF8
    return $records
}
function Stop-WiPoseFreshRunner($candidate) {
    # PID reuse is checked immediately before termination. Never use saved PIDs.
    Assert-NoWiPoseSaving
    $fresh=Get-CimInstance Win32_Process -Filter ("ProcessId = "+[int]$candidate.ProcessId)
    if (!$fresh) { return }
    if ($fresh.CreationDate -ne $candidate.CreationDate -or $fresh.CommandLine -cne $candidate.CommandLine -or
        !(Test-WiPoseStopRunner $fresh)) { throw 'Process identity changed before stop; refusing PID reuse or unrelated process.' }
    Stop-Process -Id ([int]$fresh.ProcessId) -ErrorAction Stop
}

$task=Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
Assert-OwnedWiPoseStopTask $task
Assert-NoWiPoseSaving
$immutable=@(Get-WiPoseImmutableFiles)
$checkpoints=@(Get-WiPoseCheckpoints)
$workers=@(Get-CimInstance Win32_Process | Where-Object { Test-WiPoseStopRunner $_ })
$workers | Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine | Format-List
Write-Host 'Only complete atomic epoch checkpoints can be resumed. Unsaved mid-epoch work is lost; no force-save is requested.'
if (!$Stop) {
    Write-Host "DRY_RUN: no files written, tasks stopped or processes killed. Validated $($checkpoints.Count) checkpoints. Use -Stop to back up and stop $TaskName."
    return
}
$backup=Join-Path $BackupRoot ((Get-Date -Format yyyyMMdd_HHmmss_fff)+'_'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $backup -ErrorAction Stop | Out-Null
$before=Copy-WiPoseVerifiedSnapshot @($immutable+$checkpoints) (Join-Path $backup 'before_stop')
Assert-NoWiPoseSaving
# Authenticate again after backup before changing task state.
$task=Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
Assert-OwnedWiPoseStopTask $task
if ($task.State -in @('Running','Queued')) { Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop }
# Task Scheduler may have already stopped descendants. Discover survivors now.
$workers=@(Get-CimInstance Win32_Process | Where-Object { Test-WiPoseStopRunner $_ })
$controllers=@($workers | Where-Object { @(Split-WiPoseStopCommand $_.CommandLine) -notcontains '--_worker' })
foreach ($candidate in $controllers) { Stop-WiPoseFreshRunner $candidate }
Assert-NoWiPoseSaving
$workers=@(Get-CimInstance Win32_Process | Where-Object { Test-WiPoseStopRunner $_ })
foreach ($candidate in $workers) { Stop-WiPoseFreshRunner $candidate }
$remaining=@(Get-CimInstance Win32_Process | Where-Object { Test-WiPoseStopRunner $_ })
if ($remaining.Count) { throw "Owned old WiPose runners remain. Preserve all files and inspect them. Backup: $backup" }
$task=Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
Assert-OwnedWiPoseStopTask $task
if ($task.State -in @('Running','Queued')) { throw "Old WiPose task is still active. No fast launch; preserve backup: $backup" }
Assert-NoWiPoseSaving
foreach ($record in $before | Where-Object { $_.source -notlike '*\checkpoint.pt' }) {
    if ($record.sha256 -ne (Get-FileHash -LiteralPath $record.source -Algorithm SHA256).Hash) {
        throw "Immutable old file changed during stop. Preserve backup: $backup"
    }
}
$after=Copy-WiPoseVerifiedSnapshot @($immutable+@(Get-WiPoseCheckpoints)) (Join-Path $backup 'after_stop')
Write-Host "STOPPED_OLD_WIPOSE_PARALLEL: verified before/after backups: $backup"
Write-Host 'Historical source, manifest, checkpoints and locks were preserved. Latest complete checkpoint bytes match after_stop hashes.json.'
