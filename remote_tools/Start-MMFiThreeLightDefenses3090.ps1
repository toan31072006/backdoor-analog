<#
Prepare three independent A1-only P1S1 MM-Fi defense runs on RTX 3090.
Run Preflight while P3 is active if desired; run GpuSmoke and LaunchForeground
only after every P3S1/S2/S3 Clean/A1 result has completed.
#>
[CmdletBinding()]
param(
    [ValidateSet('DryRun','Preflight','GpuSmoke','LaunchForeground','Status')]
    [string]$Mode = 'DryRun',
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$Python = '',
    [string]$HPELiArchive = '',
    [string]$HPELiArchiveSHA256 = '',
    [string]$MetaFiArchive = '',
    [string]$WiFlowArchive = ''
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$Python) { $Python = Join-Path $DataHome 'envs\mmfi-p3\Scripts\python.exe' }
if (!$HPELiArchive) { $HPELiArchive = Join-Path $DataHome 'downloads\mmfi_hpeli_p1s1_s42_clean_a1_source.zip' }
if (!$MetaFiArchive) { $MetaFiArchive = Join-Path $DataHome 'downloads\metafiplusplus_clean_a1_s42_20261024_164753.tar.gz' }
if (!$WiFlowArchive) { $WiFlowArchive = Join-Path $DataHome 'downloads\wiflow_s42_snapshot_20261025_101131.zip' }
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$env:PYTHONUTF8='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$jobs=@(
    @{name='hpeli'; lane='hpeli_light_3090'; archive=$HPELiArchive;
      source=(Join-Path $DataHome 'imports\mmfi_hpeli_p1s1_s42_clean_a1')},
    @{name='metafiplusplus'; lane='metafi_light_3090'; archive=$MetaFiArchive;
      source=(Join-Path $DataHome 'imports\metafiplusplus_s42_snapshot_20261025')},
    @{name='wiflow'; lane='wiflow_light_3090'; archive=$WiFlowArchive;
      source=(Join-Path $DataHome 'imports\wiflow_s42_snapshot_20261025')}
)
foreach ($job in $jobs) {
    $job.runner=Join-Path $repo "remote_tools\$($job.lane)\defense3090.py"
    $job.output=Join-Path $DataHome "runs\$($job.name)_light_a1_defense3090_s42_v1"
    if (!(Test-Path -LiteralPath $job.runner -PathType Leaf)) { throw "Missing runner: $($job.runner)" }
    $job.args=@('-u',$job.runner,'--data-home',$DataHome,'--source-root',$job.source,
                '--outdir',$job.output,'--jobs-per-gpu','1')
    if (Test-Path -LiteralPath $job.archive -PathType Leaf) {
        $job.args+=@('--archive',$job.archive)
    }
    if ($job.name -eq 'hpeli' -and $HPELiArchiveSHA256) {
        if ($HPELiArchiveSHA256 -notmatch '^[a-fA-F0-9]{64}$') { throw 'Invalid HPELiArchiveSHA256' }
        $job.args+=@('--archive-sha256',$HPELiArchiveSHA256)
    }
}
if ($Mode -eq 'Status') {
    foreach ($job in $jobs) {
        $folder=Join-Path $job.output "$($job.name)\seed42"
        [PSCustomObject]@{
            Model=$job.name
            Preflight=(Test-Path (Join-Path $job.output 'preflight.json'))
            GpuSmoke=(Test-Path (Join-Path $job.output 'gpu_smoke.json'))
            EvaluationFiles=@(Get-ChildItem $folder -Directory -ErrorAction SilentlyContinue |
                Where-Object { Test-Path (Join-Path $_.FullName 'evaluation.json') }).Count
        }
    }
    return
}
if ($Mode -eq 'Preflight') {
    foreach ($job in $jobs) {
        if (!(Test-Path -LiteralPath $job.archive -PathType Leaf)) { throw "Missing source archive: $($job.archive)" }
    }
    if (!$HPELiArchiveSHA256) { throw 'Pass -HPELiArchiveSHA256 printed by Van export' }
}
if ($Mode -in @('GpuSmoke','LaunchForeground')) {
    $p3=Join-Path $DataHome 'runs\mmfi_p3_hpeli_s42_v1'
    foreach ($setting in 1,2,3) {
        foreach ($cell in 'clean','a1') {
            $path=Join-Path $p3 "p3_s$setting\$cell\p3_evaluation.json"
            if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "P3 still incomplete: $path" }
            $record=Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($record.status -ne 'complete' -or $record.checkpoint_epoch -ne 49) {
                throw "P3 result not complete/final49: $path"
            }
        }
    }
}
foreach ($job in $jobs) {
    Write-Host "===== $($job.name): $Mode ====="
    $phase=switch ($Mode) {
        DryRun {'--dry-run'}
        Preflight {'--preflight'}
        GpuSmoke {'--gpu-smoke'}
        LaunchForeground {'--launch'}
    }
    $log=Join-Path $DataHome ("runs\" + $job.name + '_light_a1_defense3090_s42_v1.controller.log')
    $runnerArgs=[string[]]$job.args
    if ($Mode -eq 'LaunchForeground') {
        & $Python @runnerArgs $phase 2>&1 | Tee-Object -FilePath $log -Append
    } else {
        & $Python @runnerArgs $phase
    }
    if ($LASTEXITCODE -ne 0) { throw "$($job.name) $Mode failed; later models not started" }
}
if ($Mode -eq 'Preflight') { Write-Host 'PREFLIGHT_ALL_THREE; no GPU smoke or full defense launched' }
if ($Mode -eq 'GpuSmoke') { Write-Host 'GPU_SMOKE_ALL_THREE; separate LaunchForeground required' }
