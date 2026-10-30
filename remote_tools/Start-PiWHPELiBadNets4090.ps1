<#
.SYNOPSIS
One BadNets-CSI HPE-Li/PiW3D seed42 victim, exclusive GPU0 on Van4090.
.DESCRIPTION
Preflight hashes inputs on CPU only. Prepare additionally runs a real B32 CUDA
smoke and stops. Launch is explicit and foreground; keep SSH open. No process
kill, scheduler, environment installation, GPU sharing or automatic retry.
#>
[CmdletBinding()]
param(
    [ValidateSet('DryRun','Preflight','Prepare','CheckOnly','Launch','Status')]
    [string]$Mode = 'Status',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$ReferenceDir = '',
    [string]$Outdir = '',
    [string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$ReferenceDir) { $ReferenceDir = Join-Path $DataHome 'runs\piw_hpeli_body_axis_s42_lr001_v3' }
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\piw_hpeli_badnets_s42_4090_v1' }
if (!$PythonExe) { $PythonExe = Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
$ReferenceDir = [IO.Path]::GetFullPath($ReferenceDir)
$Outdir = [IO.Path]::GetFullPath($Outdir)
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
$Runner = Join-Path $Repo 'ATKBackd\run_piw_hpeli_badnets_4090.py'
foreach ($Value in @($Repo,$DataHome,$ReferenceDir,$Outdir,$PythonExe,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw "Unsupported path: $Value" }
}
$RunsPrefix = ([IO.Path]::GetFullPath((Join-Path $DataHome 'runs'))).TrimEnd('\','/') + '\'
if (!$Outdir.StartsWith($RunsPrefix,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Outdir must be a strict descendant of DataHome\runs.'
}
foreach ($File in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing: $File" }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$Modes = @{ DryRun='dry-run'; Preflight='preflight'; Prepare='prepare'; CheckOnly='check'; Launch='launch'; Status='status' }
$Common = @('-u',$Runner,'--data-home',$DataHome,'--reference-dir',$ReferenceDir,'--outdir',$Outdir)
if ($Mode -eq 'Launch') {
    'FOREGROUND: keep this SSH open. Exactly one BadNets/PiW3D worker; no sharing with MM-Fi WiFlow.'
}
$global:LASTEXITCODE = 0
& $PythonExe @Common '--mode' $Modes[$Mode]
$Code = $LASTEXITCODE
if ($Code -ne 0) { throw "BadNets PiW3D $Mode failed (exit $Code); no retry or later phase. No process stopped." }
if ($Mode -eq 'Prepare') {
    'PREPARED_ONLY: inspect the audit and B32 CUDA smoke; invoke -Mode Launch separately.'
}
