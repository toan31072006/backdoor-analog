[CmdletBinding()]
param(
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Python,
    [string]$ActionRoot,
    [string]$MMFiBendNpz,
    [string]$PiWBendRoot,
    [string]$PiWBendArchive,
    [string]$Outdir
)
$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
if (-not $Outdir) {
    $taskExportHome = $DataHome
    if (Test-Path -LiteralPath 'U:\backdooranalog' -PathType Container) { $taskExportHome = 'U:\backdooranalog' }
    $Outdir = Join-Path $taskExportHome 'figure_exports\actions_hpeli_s42_v1'
}
$taskPlotter = Join-Path $PSScriptRoot 'Plot-ActionQualitative.py'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (Test-Path -LiteralPath $Outdir) { throw "Preserve existing figures; choose a new -Outdir: $Outdir" }
if (Test-Path -LiteralPath ($Outdir + '.zip')) { throw 'Existing ZIP retained; choose a new -Outdir' }
$taskArgs = @('-B','-u',$taskPlotter,'--data-home',$DataHome,'--outdir',$Outdir)
if ($ActionRoot) { $taskArgs += @('--action-root',$ActionRoot) }
if ($MMFiBendNpz) { $taskArgs += @('--mmfi-bend-npz',$MMFiBendNpz) }
if ($PiWBendRoot) { $taskArgs += @('--piw-bend-root',$PiWBendRoot) }
if ($PiWBendArchive) { $taskArgs += @('--piw-bend-archive',$PiWBendArchive) }
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
Write-Host 'SIX REAL qualitative cases: MM-Fi + PiW; Bend prior/leg rotation, Cross, Nod.'
Write-Host 'CPU plotting only. No torch/model/checkpoint loading, GPU, training, inference, repair, or result overwrite.'
& $Python @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Plot failed, exit $LASTEXITCODE. Preserve inputs and any partial figures; no automatic retry." }
$taskRecord = Get-Content -LiteralPath (Join-Path $Outdir 'provenance.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($taskRecord.status -ne 'COMPLETE_6_CASES' -or $taskRecord.plotted -ne 6) { throw 'Not six complete cases; no success ZIP' }
$taskFiles = @(Get-ChildItem -LiteralPath $Outdir -File | Select-Object -ExpandProperty FullName)
Compress-Archive -LiteralPath $taskFiles -DestinationPath ($Outdir + '.zip') -CompressionLevel Optimal
$taskZipHash = (Get-FileHash -LiteralPath ($Outdir + '.zip') -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "QUALITATIVE_6_CASES_OK: $Outdir"
Write-Host "FIGURES_ZIP: $Outdir.zip"
Write-Host "ZIP_SHA256: $taskZipHash"
