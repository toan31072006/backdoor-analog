<#
.SYNOPSIS
Queue PiW3D MetaFi++ Clean then A1 after all four attacks finish on 3090.
.DESCRIPTION
Queue explicitly waits, audits final199 checkpoints, prepares, trains and
evaluates Clean/A1 serially. Foreground: keep SSH open. No old queue is stopped,
modified or retried. Use an isolated checkout, not the active four-attack one.
#>
[CmdletBinding()]
param(
    [ValidateSet('Queue','Status')][string]$Mode = 'Status',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$DatasetRoot = '',
    [string]$ActionNpy = '',
    [string]$AfterRun = '',
    [string]$Outdir = '',
    [string]$CheckpointHome = '',
    [string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\Person-in-WiFi-3D_processed_v1' }
if (!$ActionNpy) { $ActionNpy = Join-Path $DataHome 'actions\data_bend.npy' }
if (!$AfterRun) { $AfterRun = Join-Path $DataHome 'runs\piw_metafi_four_attacks_s42_3090_v1' }
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\piw_metafi_clean_a1_after_wanet_s42_3090_v1' }
if (!$CheckpointHome) { $CheckpointHome = Join-Path $DataHome 'imports\piw_metafi_adamw_unfinished' }
if (!$PythonExe) {
    $PythonExe = Join-Path $env:USERPROFILE 'eclare-env\Scripts\python.exe'
}
$Runner = Join-Path $Repo 'remote_tools\piw_metafi_after_wanet_3090.py'
$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
$ActionNpy = [IO.Path]::GetFullPath($ActionNpy)
$AfterRun = [IO.Path]::GetFullPath($AfterRun)
$Outdir = [IO.Path]::GetFullPath($Outdir)
$CheckpointHome = [IO.Path]::GetFullPath($CheckpointHome)
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
foreach ($Value in @($Repo,$DataHome,$DatasetRoot,$ActionNpy,$AfterRun,$Outdir,$CheckpointHome,$PythonExe)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw "Unsupported path: $Value" }
}
foreach ($Value in @($Runner,$PythonExe)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing: $Value" }
}
$Common = @('-u',$Runner,'--mode',$Mode.ToLowerInvariant(),'--data-home',$DataHome,
    '--dataset',$DatasetRoot,'--action',$ActionNpy,'--after',$AfterRun,
    '--out',$Outdir,'--checkpoint-home',$CheckpointHome)
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
'Queue mode authorizes: wait four final199 attacks -> Prepare -> Clean -> A1 -> evaluation.'
'Keep SSH open. One GPU worker; no kill, auto retry, batch change or SGD import.'
& $PythonExe @Common
if ($LASTEXITCODE -ne 0) { throw 'After-WaNet queue failed; no automatic retry or later job.' }
