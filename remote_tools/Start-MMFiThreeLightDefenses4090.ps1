<#
.SYNOPSIS
Prepare and run HPE-Li, MetaFi++, WiFlow A1 light defenses serially on Vân's RTX 4090.
.DESCRIPTION
Prepare authenticates all three original victims and runs actual one-GPU
capacity checks for NoDef, FT, CLP, BNP, FST and adapted NC. Launch creates
one persistent scheduled task; it runs the three backbone jobs in order.
No historical checkpoint is overwritten.
#>
[CmdletBinding()]
param(
    [ValidateSet('DryRun','Prepare','Launch','Status','CheckOnly')]
    [string]$Mode = 'DryRun',
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$EnvironmentPrefix = '',
    [string]$HPELiSourceRoot = '',
    [string]$MetaFiSourceRoot = '',
    [string]$MetaFiArchive = '',
    [string]$WiFlowSourceRoot = '',
    [string]$WiFlowArchive = '',
    [string]$TaskName = 'MMFi-Three-Light-A1-Defense-S42-v3-BNP-FST'
)
$ErrorActionPreference='Stop'
if ($TaskName -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid task name' }
$code=[IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if (!$EnvironmentPrefix) { $EnvironmentPrefix=Join-Path $DataHome 'envs\dose-backdoor' }
if (!$HPELiSourceRoot) { $HPELiSourceRoot=Join-Path $DataHome 'runs\mmfi_a1_three_tables_s42_43_44_v1' }
if (!$MetaFiSourceRoot) { $MetaFiSourceRoot=Join-Path $DataHome 'imports\metafiplusplus_s42_snapshot_20261025' }
if (!$MetaFiArchive) { $MetaFiArchive=Join-Path $env:USERPROFILE 'Downloads\metafiplusplus_clean_a1_s42_20261024_164753.tar.gz' }
if (!$WiFlowSourceRoot) { $WiFlowSourceRoot=Join-Path $DataHome 'imports\wiflow_s42_snapshot_20261025' }
if (!$WiFlowArchive) { $WiFlowArchive=Join-Path $env:USERPROFILE 'Downloads\wiflow_s42_snapshot_20261025_101131.zip' }
foreach ($name in @('DataHome','EnvironmentPrefix','HPELiSourceRoot','MetaFiSourceRoot','MetaFiArchive','WiFlowSourceRoot','WiFlowArchive')) {
    $value=[IO.Path]::GetFullPath((Get-Variable $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') { throw "Invalid local path: $name" }
    Set-Variable $name $value
}
$python=Join-Path $EnvironmentPrefix 'python.exe'
if (!(Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing Python: $python" }
$env:PYTHONUTF8='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$jobs=@(
    @{name='hpeli';lane='hpeli_light_4090';source=$HPELiSourceRoot;archive=''},
    @{name='metafiplusplus';lane='metafi_light_4090';source=$MetaFiSourceRoot;archive=$(if (Test-Path -LiteralPath $MetaFiArchive -PathType Leaf) { $MetaFiArchive } else { '' })},
    @{name='wiflow';lane='wiflow_4090';source=$WiFlowSourceRoot;archive=$WiFlowArchive}
)
foreach ($job in $jobs) {
    $job.runner=Join-Path $code "remote_tools\$($job.lane)\defense4090.py"
    $job.outdir=Join-Path $DataHome "runs\$($job.name)_light_a1_defense4090_s42_v3_bnp_fst_nc"
    if (!(Test-Path -LiteralPath $job.runner -PathType Leaf)) { throw "Missing runner: $($job.runner)" }
    $job.args=@('-u',$job.runner,'--data-home',$DataHome,'--outdir',$job.outdir,'--jobs-per-gpu','1')
    if ($job.source) { $job.args+=@('--source-root',$job.source) }
    if ($job.archive) { $job.args+=@('--archive',$job.archive) }
}
if ($Mode -eq 'Prepare') {
    if (!(Test-Path -LiteralPath (Join-Path $HPELiSourceRoot 'three_tables.resolved.json') -PathType Leaf)) {
        throw "HPE-Li original source manifest missing: $HPELiSourceRoot"
    }
    $metaManifest=Join-Path $MetaFiSourceRoot 'vast_seed42_attack.resolved.json'
    if (!(Test-Path -LiteralPath $metaManifest -PathType Leaf) -and
        !(Test-Path -LiteralPath $MetaFiArchive -PathType Leaf)) {
        throw "MetaFi++ authenticated source import or archive missing: $MetaFiSourceRoot / $MetaFiArchive"
    }
    if (!(Test-Path -LiteralPath $WiFlowArchive -PathType Leaf)) {
        throw "WiFlow archive required for import provenance: $WiFlowArchive"
    }
}
if ($Mode -in @('DryRun','Prepare','Status','CheckOnly')) {
    foreach ($job in $jobs) {
        Write-Host "===== $($job.name) ====="
        $phases=switch ($Mode) {
            DryRun {@('--dry-run')}
            Prepare {@('--dry-run','--preflight','--gpu-smoke')}
            Status {@('--status')}
            CheckOnly {@('--check-only')}
        }
        foreach ($phase in $phases) {
            $runnerArgs=$job.args
            & $python @runnerArgs $phase
            if ($LASTEXITCODE -ne 0) { throw "$($job.name) $phase failed; later jobs not started" }
        }
    }
    if ($Mode -eq 'Prepare') { Write-Host 'PREPARED_ALL_THREE; full defense has not started' }
    if ($Mode -eq 'Status') {
        Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Select-Object TaskName,State
    }
    return
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task $TaskName exists; inspect its logs/state before any new launch"
}
foreach ($job in $jobs) {
    foreach ($receipt in @('preflight.json','gpu_smoke.json')) {
        $path=Join-Path $job.outdir $receipt
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing gate: $path" }
        $saved=Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($saved.status -ne 'PASS') { throw "Failed gate: $path" }
    }
}
$cfg=@{python=$python;code=$code;jobs=@($jobs | ForEach-Object {
    @{name=$_.name;args=($_.args + '--launch');output=$_.outdir}
})}
$encodedCfg=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($cfg | ConvertTo-Json -Compress -Depth 7)))
$payload=@'
$ErrorActionPreference='Stop'
$cfg=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CFG__')) | ConvertFrom-Json
$env:PYTHONUTF8='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
foreach ($job in $cfg.jobs) {
    $log=$job.output + '.task.' + (Get-Date -Format yyyyMMdd_HHmmss_fff) + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
    try {
        $quoted=@($job.args | ForEach-Object { '"' + $_ + '"' }) -join ' '
        $child=Start-Process -FilePath $cfg.python -ArgumentList $quoted -WorkingDirectory $cfg.code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$log.stdout.log" -RedirectStandardError "$log.stderr.log"
        if ($child.ExitCode -ne 0) { exit $child.ExitCode }
    } catch {
        $_ | Out-String | Set-Content -LiteralPath "$log.wrapper-error.log" -Encoding UTF8
        exit 1
    }
}
exit 0
'@
$payload=$payload.Replace('__CFG__',$encodedCfg)
$encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($payload))
$shell="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$action=New-ScheduledTaskAction -Execute $shell -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded" -WorkingDirectory $code
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$principal=New-ScheduledTaskPrincipal -UserId $sid -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task=New-ScheduledTask -Action $action -Principal $principal -Settings $settings -Description 'Sequential A1 NoDef, FT10, CLP, BNP, FST-adapted, NC-adapted on three MM-Fi pose backbones. No retry.'
Register-ScheduledTask -TaskName $TaskName -InputObject $task -ErrorAction Stop | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Started $TaskName. HPE-Li then MetaFi++ then WiFlow; each writes runs\<model>_light_a1_defense4090_s42_v3_bnp_fst_nc.task.*.log"
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
