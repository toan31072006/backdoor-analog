<# PiW BackTime setup. No action starts training unless Mode is Launch. #>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','FindClean','CheckOnly','Prepare','Launch','Status','Verify')][string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Dataset = '',
    [string]$Outdir = 'U:\backdooranalog\runs\piw_backtime_three_backbones_reuse_clean_s42_v2',
    [string]$CleanHPELi = '',
    [string]$CleanWiFlow = '',
    [string]$CleanMetaFi = '',
    [Alias('Python')][string]$PythonExe = '',
    [string]$Device = 'cuda:0'
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if (!$PythonExe) {
    foreach ($Candidate in @((Join-Path $DataHome 'envs\dose-backdoor\python.exe'),
                             (Join-Path $DataHome 'envs\mmfi-p3\Scripts\python.exe'))) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) { $PythonExe = $Candidate; break }
    }
}
if (!$PythonExe) { throw 'Use -PythonExe with an existing PiW environment. No installation.' }
$Runner = Join-Path $Repo 'ATKBackd\run_piw_backtime_three_backbones.py'
foreach ($Value in @($DataHome,$Dataset,$Outdir,$PythonExe,$Runner,$Device,$CleanHPELi,$CleanWiFlow,$CleanMetaFi)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw 'Ordinary local paths required.' }
}
if (!(Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Missing Python: $PythonExe" }
$Phase = @{Inspect='inspect';FindClean='discover';CheckOnly='check';Prepare='prepare';Launch='launch';Status='status';Verify='verify'}[$Mode]
$RunArgs = @('-B','-u',$Runner,'--mode',$Phase,'--data-home',$DataHome,'--outdir',$Outdir,'--device',$Device)
if ($Dataset) { $RunArgs += @('--dataset',$Dataset) }
foreach ($Reference in @(@('hpeli',$CleanHPELi),@('wiflow',$CleanWiFlow),@('metafiplusplus',$CleanMetaFi))) {
    if ($Reference[1]) { $RunArgs += @('--clean-reference',("{0}={1}" -f $Reference[0],$Reference[1])) }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
if ($Mode -eq 'Launch') {
    Write-Output 'FOREGROUND: keep terminal open and Windows awake. One PiW worker only.'
    Write-Output 'PiW order: HPE-Li -> WiFlow -> MetaFi++; BackTime200 only, reuse audited Clean references.'
    Write-Output 'Existing MM-Fi WiFlow is separate: no stop, suspend, pull, or source changes to its checkout.'
    Write-Output 'NEW PiW attacker100 epochs, then THREE scratch BackTime victims200/final199. ZERO Clean training.'
    Write-Output 'Missing/mismatched Clean reference stops the queue; no automatic Clean retraining or checkpoint warm-start.'
    Write-Output 'No job kill, automatic retry, batch/AMP fallback, or TEST tuning.'
}
& $PythonExe @RunArgs
if ($LASTEXITCODE -ne 0) { throw "PiW BackTime $Mode failed. Preserve logs and artifacts; no automatic retry." }
