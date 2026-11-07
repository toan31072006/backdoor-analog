param(
    [string]$Python = (Join-Path $env:USERPROFILE 'backdooranalog\envs\dose-backdoor\python.exe'),
    [string]$InputNpz = (Join-Path $env:USERPROFILE 'Downloads\mmfi_dose_pose_s42.npz'),
    [string]$OutDir = (Join-Path $env:USERPROFILE 'Downloads\MMFiDosePoseFigure_s42'),
    [switch]$PlotOnly
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$DosePlotter = Join-Path $PSScriptRoot 'Plot-MMFiDosePose.py'
$DoseExporter = Join-Path $PSScriptRoot 'Export-MMFiDosePose.py'
foreach ($DoseHelper in @($DosePlotter, $DoseExporter)) {
    if (!(Test-Path -LiteralPath $DoseHelper -PathType Leaf)) { throw "Missing helper: $DoseHelper" }
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
    $DoseExpectedOutput = Join-Path $env:USERPROFILE 'Downloads\mmfi_dose_pose_s42.npz'
    if ([IO.Path]::GetFullPath($InputNpz) -ne [IO.Path]::GetFullPath($DoseExpectedOutput)) {
        throw 'The original exporter uses Downloads\mmfi_dose_pose_s42.npz. Use the default input for export.'
    }
    Write-Host 'Exporting 3 fixed TEST frames x 6 doses from original A1 final49. No training.'
    Write-Host 'The original BackTime coordination guard is preserved; no process will be stopped.'
    & $Python -B -u $DoseExporter
    if ($LASTEXITCODE -ne 0) { throw 'Pose export failed. Preserve the log; no automatic retry or job termination.' }
} else {
    Write-Host "Using existing NPZ without rerunning inference: $InputNpz"
}
& $Python -B -u $DosePlotter --input $InputNpz --out-dir $OutDir
if ($LASTEXITCODE -ne 0) { throw 'Pose validation/plotting failed; preserve the input and log.' }
Write-Host "Done. PDF, PNG, per-frame CSV and provenance JSON are in: $OutDir"
