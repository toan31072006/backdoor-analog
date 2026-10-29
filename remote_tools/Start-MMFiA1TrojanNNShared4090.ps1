<#
.SYNOPSIS
Resume the existing A1 and launch prepared TrojanNN beside the approved nscaling job.
.DESCRIPTION
Explicit user-authorized execution exception. No original ATKBackd sources,
plans, triggers, or GPU gates are edited. Foreground: keep this SSH open.
#>
[CmdletBinding()]
param(
    [ValidateSet('Launch','Status')][string]$Mode='Status',
    [string]$DataHome='C:\Users\Van\backdooranalog',
    [string]$A1Repo='C:\Users\Van\code',
    [string]$PythonExe='C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe',
    [switch]$AllowKnownNScaling
)
$ErrorActionPreference='Stop'
if ($Mode -eq 'Launch' -and !$AllowKnownNScaling) {
    throw 'Explicit -AllowKnownNScaling is required. No process was started or stopped.'
}
$Repo=[IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$Runner=Join-Path $Repo 'remote_tools\shared_4090\controller.py'
foreach($Path in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) {throw "Missing existing file: $Path"}
}
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$Arguments=@('-u',$Runner,'--mode',$Mode.ToLower(),'--home',$DataHome,
             '--trojan-repo',$Repo,'--a1-repo',$A1Repo)
& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    throw 'Shared run failed; inspect the printed logs. No automatic process termination or retry.'
}
