<#
.SYNOPSIS
Serial TSBA -> Smooth-LF-CSI on PiW3D MetaFi++, seed42, RTX4090.
External evaluation processes are allowed; no PID cap or process termination.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','Prepare','CheckOnly','Launch','Evaluate','Status')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [ValidateSet('tsba_adapted','smooth_lf_csi')][string]$Method = ''
)
$ErrorActionPreference = 'Stop'
$DataHome = [IO.Path]::GetFullPath($DataHome)
$PythonExe = Join-Path $DataHome 'envs\piw-metafi-vast211-4090\Scripts\python.exe'
$Runner = Join-Path $PSScriptRoot 'piw_metafi_two_extra_4090.py'
foreach ($File in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing file: $File. No environment fallback/install." }
}
if ($Method -and $Mode -ne 'Evaluate') { throw 'Method is only used with -Mode Evaluate.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$Phase = @{Inspect='inspect';Prepare='prepare';CheckOnly='check';Launch='launch';Evaluate='evaluate';Status='status'}[$Mode]
$LaunchArgs = @('-B','-u',$Runner,'--mode',$Phase,'--data-home',$DataHome)
if ($Outdir) { $LaunchArgs += @('--out',[IO.Path]::GetFullPath($Outdir)) }
if ($Method) { $LaunchArgs += @('--method',$Method) }
Write-Host 'One owned GPU worker; external evaluation allowed without a PID cap. No kill/retry/batch fallback.'
if ($Mode -eq 'Launch') { Write-Host 'FOREGROUND: keep this PowerShell open. TSBA -> Smooth-LF; then evaluate both.' }
& $PythonExe @LaunchArgs
if ($LASTEXITCODE -ne 0) { throw "$Mode failed. Preserve logs/checkpoints; no automatic retry or later job." }
if ($Mode -eq 'Prepare') { Write-Host 'PREPARED_ONLY. Inspect two B32 smokes, then invoke -Mode Launch separately.' }
