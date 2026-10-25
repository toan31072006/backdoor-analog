<# Export one paired A1-only report after all three light runners finish. #>
[CmdletBinding()]
param(
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$EnvironmentPrefix = '',
    [string]$ReportDir = ''
)
$ErrorActionPreference='Stop'
if (!$EnvironmentPrefix) { $EnvironmentPrefix=Join-Path $DataHome 'envs\dose-backdoor' }
if (!$ReportDir) { $ReportDir=Join-Path $DataHome 'runs\mmfi_three_light_a1_defense_report_s42_v3_bnp_fst_nc' }
$code=[IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$python=Join-Path $EnvironmentPrefix 'python.exe'
$reporter=Join-Path $code 'ATKBackd\defend\a1_three_model_report.py'
foreach ($path in @($python,$reporter)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
$reportArgs=@('-u',$reporter)
foreach ($backbone in @('hpeli','metafiplusplus','wiflow')) {
    $root=Join-Path $DataHome "runs\${backbone}_light_a1_defense4090_s42_v3_bnp_fst_nc\$backbone\seed42"
    foreach ($method in @('none','ft','clp','bnp','fst','nc')) {
        $path=Join-Path $root "$method\evaluation.json"
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Incomplete $backbone/$method : $path" }
        if ($method -eq 'nc') {
            $record=Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($record.status -eq 'nc_diagnostic_only') {
                Write-Host "NC $backbone : diagnostic only, no repaired-model row"
                continue
            }
        }
        $reportArgs+=@('--record',"$backbone=$path")
    }
}
$reportArgs+=@('--require-defense','ft','--require-defense','clp',
         '--require-defense','bnp','--require-defense','fst',
         '--out',(Join-Path $ReportDir 'table4_a1_only.json'),
         '--markdown-out',(Join-Path $ReportDir 'table4_a1_only.md'))
$env:PYTHONUTF8='1'
& $python @reportArgs
if ($LASTEXITCODE -ne 0) { throw 'Paired three-model defense report failed' }
