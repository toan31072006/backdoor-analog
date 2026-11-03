<#
.SYNOPSIS
Fresh PiW3D MetaFi++ A1. Reuse completed Clean4090 read-only; one foreground GPU worker.
No installer, full rehash, checkpoint import, process kill or automatic retry.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','Prepare','CheckOnly','Launch','Status')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog')
)
$ErrorActionPreference = 'Stop'
$DataHome = [IO.Path]::GetFullPath($DataHome)
$PythonExe = Join-Path $DataHome 'envs\piw-metafi-vast211-4090\Scripts\python.exe'
$Runner = Join-Path $PSScriptRoot 'piw_metafi_a1_4090.py'
foreach ($Value in @($DataHome,$PythonExe,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw "Unsupported local path: $Value" }
    $Cursor = [IO.Path]::GetFullPath($Value)
    while (![string]::IsNullOrEmpty($Cursor)) {
        if ((Test-Path -LiteralPath $Cursor) -and
            ((Get-Item -LiteralPath $Cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Linked path/ancestor refused: $Cursor"
        }
        $Parent = Split-Path -Parent $Cursor
        if ($Parent -eq $Cursor) { break }
        $Cursor = $Parent
    }
}
foreach ($Name in @('PYTHONHOME','PYTHONPATH','PIW_SMOKE_BARRIER_DIR')) {
    $Setting = Get-Item -LiteralPath ("Env:"+$Name) -ErrorAction SilentlyContinue
    if ($null -ne $Setting -and ![string]::IsNullOrEmpty([string]$Setting.Value)) {
        throw "Inherited $Name refused; nothing started."
    }
}
if (!(Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Missing completed Clean4090 environment: $PythonExe. No install/fallback."
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$Phase = @{Inspect='inspect'; Prepare='prepare'; CheckOnly='check'; Launch='launch'; Status='status'}[$Mode]
if ($Mode -eq 'Launch') {
    Write-Host 'FOREGROUND: keep this SSH session open. One A1 GPU worker; Clean is not retrained.'
}
& $PythonExe -u $Runner --mode $Phase --data-home $DataHome
if ($LASTEXITCODE -ne 0) { throw "$Mode failed; no later phase started. Keep logs/checkpoints; no auto retry/kill." }
if ($Mode -eq 'Prepare') {
    Write-Host 'PREPARED_ONLY. Inspect the native A1 smoke, then invoke Launch separately.'
}
