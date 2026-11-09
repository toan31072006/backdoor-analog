<# Frozen HPE-Li BackTime -> fresh MetaFi++ and WiFlow. Prepare is not Launch. #>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','CheckOnly','Prepare','Launch','Status')][string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$SourceRun = 'U:\backdooranalog\runs\mmfi_backtime_hpeli_p1s1_p3s3_s42_4090_fresh_2jobs_u_v1\p1s1',
    [string]$MetaFiClean = '',
    [string]$WiFlowClean = '',
    [string]$Outdir = 'U:\backdooranalog\runs\mmfi_backtime_transfer_metafi_wiflow_p1s1_s42_4090_v1',
    [Alias('Python')][string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if (!$PythonExe) {
    foreach ($Candidate in @((Join-Path $DataHome 'envs\dose-backdoor\python.exe'),
                             (Join-Path $DataHome 'envs\mmfi-p3\Scripts\python.exe'))) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) { $PythonExe = $Candidate; break }
    }
}
if (!$PythonExe) { throw 'Use -PythonExe with the existing MM-Fi environment. No install.' }
$Runner = Join-Path $Repo 'remote_tools\Run-MMFiBackTimeTransfer4090.py'
$Phase = @{Inspect='inspect';CheckOnly='check';Prepare='prepare';Launch='launch';Status='status'}[$Mode]
$RunArgs = @('-B','-u',$Runner,'--mode',$Phase,'--data-home',$DataHome,
             '--source',$SourceRun,'--outdir',$Outdir)
if ($MetaFiClean) { $RunArgs += @('--metafi-clean',$MetaFiClean) }
if ($WiFlowClean) { $RunArgs += @('--wiflow-clean',$WiFlowClean) }
foreach ($Value in @($DataHome,$SourceRun,$Outdir,$MetaFiClean,$WiFlowClean,$PythonExe,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw 'Local ordinary paths required.' }
}
foreach ($Value in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing: $Value" }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
if ($Mode -eq 'Launch') {
    'FOREGROUND: keep terminal open and Windows awake; two owned GPU workers.'
    'Frozen HPE-Li attacker reused; fresh victims50/final49; no attacker refit, retry, AMP or job kill.'
}
& $PythonExe @RunArgs
if ($LASTEXITCODE -ne 0) { throw "BackTime transfer $Mode failed. Preserve output/logs; no automatic retry." }
