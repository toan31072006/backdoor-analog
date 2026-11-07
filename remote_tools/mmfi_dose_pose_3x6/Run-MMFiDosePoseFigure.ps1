param(
    [string]$Python = (Join-Path $env:USERPROFILE 'backdooranalog\envs\dose-backdoor\python.exe'),
    [string]$InputNpz = (Join-Path $env:USERPROFILE 'Downloads\mmfi_dose_pose_s42.npz'),
    [string]$OutDir = (Join-Path $env:USERPROFILE 'Downloads\MMFiDosePoseFigure_s42'),
    [string]$SourceRoot = '',
    [string]$Checkpoint = '',
    [string]$DataRoot = '',
    [string]$ActionNpy = '',
    [switch]$CheckOnly,
    [switch]$PlotOnly
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$DosePlotter = Join-Path $PSScriptRoot 'Plot-MMFiDosePose.py'
$DoseExporter = Join-Path $PSScriptRoot 'Export-MMFiDosePose.py'
foreach ($DoseHelper in @($DosePlotter, $DoseExporter)) {
    if (!(Test-Path -LiteralPath $DoseHelper -PathType Leaf)) { throw "Missing helper: $DoseHelper" }
}
$DoseExportArgs = @('-B', '-u', $DoseExporter, '--output', $InputNpz)
$DoseOverrides = [ordered]@{
    '--source' = $SourceRoot
    '--checkpoint' = $Checkpoint
    '--data' = $DataRoot
    '--action' = $ActionNpy
}
foreach ($DoseOverride in $DoseOverrides.GetEnumerator()) {
    if (![string]::IsNullOrWhiteSpace($DoseOverride.Value)) {
        $DoseExportArgs += @($DoseOverride.Key, $DoseOverride.Value)
    }
}
if ($CheckOnly) {
    & $Python @DoseExportArgs --check-only
    if ($LASTEXITCODE -ne 0) { throw 'Path check failed; see all MISSING paths above. No inference started.' }
    return
}
foreach ($DoseExtension in @('pdf', 'png', 'csv', 'json')) {
    $DoseFigurePath = Join-Path $OutDir "fig_same_input_dose_s42.$DoseExtension"
    if (Test-Path -LiteralPath $DoseFigurePath) {
        throw "Preserve existing output; choose a new -OutDir: $DoseFigurePath"
    }
}
# Preflight plotting imports before any GPU work. No automatic installation.
& $Python -B -c "import numpy, matplotlib; print('Plot dependencies: OK')"
if ($LASTEXITCODE -ne 0) { throw 'NumPy/Matplotlib import failed; no inference started.' }

if (!(Test-Path -LiteralPath $InputNpz -PathType Leaf)) {
    if ($PlotOnly) { throw "Copy the real prediction NPZ here first: $InputNpz" }
    Write-Host 'Exporting 3 fixed TEST frames x 6 doses from original A1 final49. No training.'
    Write-Host 'Replay uses their original TEST batches (32/32/16), not a new 3-frame batch.'
    Write-Host 'The original BackTime coordination guard is preserved; no process will be stopped.'
    & $Python @DoseExportArgs
    if ($LASTEXITCODE -ne 0) { throw 'Pose export failed. Preserve the log; no automatic retry or job termination.' }
} else {
    Write-Host "Using existing NPZ without rerunning inference: $InputNpz"
}
& $Python -B -u $DosePlotter --input $InputNpz --out-dir $OutDir
if ($LASTEXITCODE -ne 0) { throw 'Pose validation/plotting failed; preserve the input and log.' }
Write-Host "Done. PDF, PNG, per-frame CSV and provenance JSON are in: $OutDir"
