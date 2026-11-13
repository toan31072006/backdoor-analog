[CmdletBinding()]
param(
    [ValidateSet('Inspect','Prepare','Capacity','Launch','Resume','Status','Export')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = 'C:\Users\Van\backdooranalog',
    [string]$Outdir = 'U:\backdooranalog\runs\mmfi_selection_mapping_s42_v1',
    [string]$Python,
    [switch]$FreshLinear
)
$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path $DataHome 'envs\dose-backdoor\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
$selectionScript = Join-Path $PSScriptRoot 'run_selection_mapping_4090.py'
$selectionArgs = @('-B','-u',$selectionScript,'--mode',$Mode.ToLowerInvariant(),
    '--data-home',$DataHome,'--outdir',$Outdir)
if ($FreshLinear) { $selectionArgs += '--fresh-linear' }
Write-Host 'MM-Fi/P1S1 HPE-Li seed42: Uniform/Diverse x Linear/Sqrt/Quad; max 2 workers on GPU0.'
Write-Host 'Historical leg rotation; NOT new trunk Bend, NOT BackTime. No Clean retraining or automatic retry.'
& $Python @selectionArgs
if ($LASTEXITCODE -ne 0) { throw "Selection/mapping $Mode failed; preserve output/logs, no automatic retry." }
