<#
.SYNOPSIS
Clean-only 4090 diagnostic: Setup -> Prepare -> Replay3090 -> inspect -> Launch.
One foreground worker; no process kill, concurrency override, or auto retry.
Export3090 is CPU/stdlib-only and must be invoked on the source 3090 machine.
#>
[CmdletBinding()]
param(
    [ValidateSet('Setup','Inspect','Prepare','CheckOnly','Replay3090','Launch','Status','Export3090')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$BasePython = '',
    [string]$EnvDir = '',
    [string]$DatasetRoot = '',
    [string]$ActionNpy = '',
    [string]$ReferenceDir = '',
    [string]$ExportDir = ''
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
$Runner = Join-Path $PSScriptRoot 'piw_metafi_clean_4090.py'
$SetupHelper = Join-Path $PSScriptRoot 'Start-PiWMetaFiCleanVastEnv3090.ps1'
if (!$BasePython) {
    if ($Mode -eq 'Export3090') {
        $BasePython = Join-Path $DataHome 'envs\piw-metafi-vast211\Scripts\python.exe'
    } else {
        $BasePython = Join-Path $DataHome 'envs\dose-backdoor\python.exe'
    }
}
if (!$EnvDir) { $EnvDir = Join-Path $DataHome 'envs\piw-metafi-vast211-4090' }
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\Person-in-WiFi-3D_processed_v1' }
if (!$ActionNpy) { $ActionNpy = Join-Path $DataHome 'actions\data_bend.npy' }
if (!$ReferenceDir) { $ReferenceDir = Join-Path $DataHome 'imports\piw_metafi_clean_3090_reference_v1' }
if (!$ExportDir) { $ExportDir = Join-Path $DataHome 'exports\piw_metafi_clean_3090_reference_v1' }
$Outdir = Join-Path $DataHome 'runs\piw_metafi_clean_vastenv_s42_4090_v1'
$EnvDir = [IO.Path]::GetFullPath($EnvDir)
$PythonExe = Join-Path $EnvDir 'Scripts\python.exe'
foreach ($Value in @($Repo,$DataHome,$BasePython,$EnvDir,$DatasetRoot,$ActionNpy,$ReferenceDir,$ExportDir)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw "Unsupported local path: $Value" }
    $Cursor = [IO.Path]::GetFullPath($Value)
    while (![string]::IsNullOrEmpty($Cursor)) {
        if (Test-Path -LiteralPath $Cursor) {
            if ((Get-Item -LiteralPath $Cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Linked path/ancestor refused: $Cursor"
            }
        }
        $Parent = Split-Path -Parent $Cursor
        if ($Parent -eq $Cursor) { break }
        $Cursor = $Parent
    }
}
foreach ($Name in @('PIP_TARGET','PIP_PREFIX','PIP_ROOT','PIP_USER','PYTHONHOME','PYTHONPATH')) {
    $Setting = Get-Item -LiteralPath ("Env:"+$Name) -ErrorAction SilentlyContinue
    if ($null -ne $Setting -and ![string]::IsNullOrEmpty([string]$Setting.Value)) {
        throw "Inherited $Name refused; no Python or installation started."
    }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'

if ($Mode -eq 'Setup') {
    # Reuse the unchanged isolated installer; only create the new 4090 venv.
    & powershell -NoProfile -ExecutionPolicy Bypass -File $SetupHelper -Mode Setup `
        -DataHome $DataHome -BasePython $BasePython -EnvDir $EnvDir `
        -Outdir $Outdir -DatasetRoot $DatasetRoot -ActionNpy $ActionNpy
    if ($LASTEXITCODE -ne 0) {
        throw '4090 Setup failed. Requires base Python exactly 3.12.15; no later phase started.'
    }
    Write-Host 'SETUP_ONLY: no GPU operation or training. Invoke Prepare separately.'
    return
}
if ($Mode -in @('Export3090','Inspect')) { $Executable = $BasePython } else { $Executable = $PythonExe }
if (!(Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Missing Python: $Executable. No fallback." }
$Phase = @{Inspect='inspect'; Prepare='prepare'; CheckOnly='check'; Replay3090='replay3090'; Launch='launch'; Status='status'; Export3090='export3090'}[$Mode]
$Arguments = @('-u',$Runner,'--mode',$Phase,'--data-home',$DataHome,
    '--dataset',$DatasetRoot,'--action',$ActionNpy,'--out',$Outdir,
    '--env-dir',$EnvDir,'--reference-dir',$ReferenceDir,'--export-dir',$ExportDir)
if ($Mode -in @('Replay3090','Launch')) {
    Write-Host 'FOREGROUND: keep this SSH session open. At most one GPU worker; other compute is blocked.'
}
& $Executable @Arguments
if ($LASTEXITCODE -ne 0) { throw "$Mode failed; no later phase was started. No automatic retry or process kill." }
if ($Mode -eq 'Prepare') {
    Write-Host 'PREPARED_ONLY. Next copy the verified 3090 reference and invoke Replay3090; no full training launched.'
}
if ($Mode -eq 'Replay3090') {
    Write-Host 'REPLAY_ONLY. Inspect replay_3090\evaluation.json before invoking Launch separately.'
}
