<# PiW BackTime v4. Only explicit Start or Launch can start training. #>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','FindClean','CheckOnly','Prepare','Start','Launch','Evaluate','Status','Verify')][string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Dataset = '',
    [string]$Outdir = 'U:\backdooranalog\runs\piw_backtime_three_backbones_train_first_s42_v4',
    [string]$AttackerRun = 'U:\backdooranalog\runs\piw_backtime_three_backbones_s42_v1',
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
if ($Mode -in @('Start','Launch') -and !$AttackerRun) {
    throw 'This reuse launcher requires AttackerRun; it will NOT fit a new attacker.'
}
$Runner = Join-Path $Repo 'ATKBackd\run_piw_backtime_three_backbones.py'
foreach ($Value in @($DataHome,$Dataset,$Outdir,$PythonExe,$Runner,$Device,$CleanHPELi,$CleanWiFlow,$CleanMetaFi,$AttackerRun)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw 'Ordinary local paths required.' }
}
if (!(Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Missing Python: $PythonExe" }
$Phase = @{Inspect='inspect';FindClean='discover';CheckOnly='check';Prepare='prepare';Start='start';Launch='launch';Evaluate='evaluate';Status='status';Verify='verify'}[$Mode]
$RunArgs = @('-B','-u',$Runner,'--mode',$Phase,'--data-home',$DataHome,'--outdir',$Outdir,'--device',$Device)
if ($Dataset) { $RunArgs += @('--dataset',$Dataset) }
if ($AttackerRun) { $RunArgs += @('--attacker-reference',$AttackerRun) }
foreach ($Reference in @(@('hpeli',$CleanHPELi),@('wiflow',$CleanWiFlow),@('metafiplusplus',$CleanMetaFi))) {
    if ($Reference[1]) { $RunArgs += @('--clean-reference',("{0}={1}" -f $Reference[0],$Reference[1])) }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
if ($Mode -in @('Start','Launch')) {
    Write-Output 'FOREGROUND: keep terminal open and Windows awake. One PiW worker only.'
    Write-Output 'PiW order: HPE-Li -> WiFlow -> MetaFi++; BackTime200 only, reuse audited Clean references.'
    Write-Output 'Existing MM-Fi WiFlow is separate: no stop, suspend, pull, or source changes to its checkout.'
    Write-Output 'REUSE completed PiW attacker read-only; ZERO attacker fitting. THREE scratch BackTime victims200/final199.'
    Write-Output 'ZERO Clean training; missing/invalid attacker reference never falls back to refitting.'
    Write-Output 'Missing auto-discovered Clean delays only its evaluation; all THREE victims still train in order.'
    Write-Output 'Explicit invalid Clean is rejected. Final199 victims are verified and skipped; partial victims resume saved optimizer/RNG.'
    if ($Mode -eq 'Start') { Write-Output 'START = audit + capacity smoke if absent + serial training. Existing valid capacity is checked, not repeated.' }
    Write-Output 'No job kill, automatic retry, batch/AMP fallback, or TEST tuning.'
}
& $PythonExe @RunArgs
if ($LASTEXITCODE -ne 0) { throw "PiW BackTime $Mode failed. Preserve logs and artifacts; no automatic retry." }
