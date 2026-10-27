<#
.SYNOPSIS
Run PiW3D HPE-Li Smooth-LF-CSI and TSBA adapted on Van's single RTX 4090.
.DESCRIPTION
Prepare exports the completed local v3 Clean/A1 reference if needed, audits
all inputs and performs two simultaneous physical-B32 CUDA smokes. Launch is
a separate action. It starts two independent victims on cuda:0 in a hidden
controller process. No historical run is modified or process stopped.
#>
[CmdletBinding()]
param(
    [ValidateSet('ExportReference','DryRun','Prepare','CheckOnly','Launch','Status','UnlockStale')]
    [string]$Mode = 'Status',
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$DatasetRoot = '',
    [string]$ActionNpy = '',
    [string]$ReferenceDir = '',
    [string]$ReferenceArchive = '',
    [string]$Outdir = '',
    [string]$EnvironmentPrefix = '',
    [string]$CodeRoot = '',
    [switch]$Serial,
    [int]$AllowTDBAPid = 0
)
$ErrorActionPreference = 'Stop'
if ($AllowTDBAPid -lt 0 -or ($AllowTDBAPid -gt 0 -and !$Serial)) {
    throw 'AllowTDBAPid requires -Serial and a positive PID.'
}
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\Person-in-WiFi-3D_processed_v1' }
if (!$ActionNpy) { $ActionNpy = Join-Path $DataHome 'actions\data_bend.npy' }
if (!$ReferenceDir) { $ReferenceDir = Join-Path $DataHome 'runs\piw_hpeli_body_axis_s42_lr001_v3' }
if (!$ReferenceArchive) { $ReferenceArchive = Join-Path $DataHome 'references\piw_hpeli_v3_clean_reference.zip' }
if (!$Outdir) {
    $runName = if ($Serial) { 'piw_hpeli_two_extra_s42_4090_serial_v1' }
               else { 'piw_hpeli_two_extra_s42_4090_v1' }
    $Outdir = Join-Path $DataHome (Join-Path 'runs' $runName)
}
if (!$EnvironmentPrefix) { $EnvironmentPrefix = Join-Path $DataHome 'envs\dose-backdoor' }
if (!$CodeRoot) { $CodeRoot = Split-Path -Parent $PSScriptRoot }
foreach ($name in @('DataHome','DatasetRoot','ActionNpy','ReferenceDir',
                    'ReferenceArchive','Outdir','EnvironmentPrefix','CodeRoot')) {
    $value = [IO.Path]::GetFullPath((Get-Variable -Name $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') {
        throw "Invalid local path: $name"
    }
    Set-Variable -Name $name -Value $value
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$controller = Join-Path $CodeRoot 'remote_linux\piw3d_hpeli_extra\controller.py'
$exporter = Join-Path $CodeRoot 'remote_tools\Export-PiWHPELiV3CleanReference.py'
foreach ($file in @($python,$controller,$exporter)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

function Export-ReferenceIfMissing {
    if (Test-Path -LiteralPath $ReferenceArchive -PathType Leaf) {
        Write-Host "Existing reference: $ReferenceArchive"
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReferenceArchive) | Out-Null
    & $python -u $exporter --reference-dir $ReferenceDir --out $ReferenceArchive
    if ($LASTEXITCODE -ne 0) { throw 'HPE-Li v3 Clean/A1 reference export failed.' }
}

if ($Mode -eq 'ExportReference') {
    Export-ReferenceIfMissing
    return
}
if ($Mode -eq 'Prepare') { Export-ReferenceIfMissing }
if ($Mode -ne 'Status' -and !(Test-Path -LiteralPath $ReferenceArchive -PathType Leaf)) {
    throw "Reference ZIP missing: $ReferenceArchive. Run -Mode Prepare."
}
$profile = if ($Serial) { 'van4090_serial' } else { 'van4090' }
$baseArgs = @('-u',$controller,'--repo-dir',$CodeRoot,'--data-home',$DataHome,
              '--dataset-root',$DatasetRoot,'--action-npy',$ActionNpy,
              '--reference-archive',$ReferenceArchive,'--outdir',$Outdir,
              '--host-profile',$profile,'--allow-tdba-pid',"$AllowTDBAPid")
$flags = @{DryRun='--dry-run';Prepare='--prepare-only';CheckOnly='--check-only';
           Status='--status';UnlockStale='--unlock-stale'}
if ($Mode -ne 'Launch') {
    & $python @baseArgs $flags[$Mode]
    if ($LASTEXITCODE -ne 0) { throw "$Mode failed; no full training launched." }
    if ($Mode -eq 'Status') {
        $active = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
            Where-Object { $_.CommandLine -and
                $_.CommandLine.Contains('piw3d_hpeli_extra\controller.py') -and
                $_.CommandLine.Contains('--launch') -and
                $_.CommandLine.Contains($Outdir) })
        Write-Host "Controller PIDs: $(@($active | ForEach-Object ProcessId) -join ',')"
        $log = Get-ChildItem -LiteralPath $Outdir -Filter 'controller.*.stdout.log' `
            -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($log) {
            Get-Content -LiteralPath $log.FullName -Encoding UTF8 -Tail 12
        }
    }
    return
}

& $python @baseArgs '--check-only'
if ($LASTEXITCODE -ne 0) { throw 'Prelaunch gate failed; no training launched.' }
$active = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -and
        $_.CommandLine.Contains('piw3d_hpeli_extra\controller.py') -and
        $_.CommandLine.Contains('--launch') -and
        $_.CommandLine.Contains($Outdir) })
if ($active.Count) { throw 'This PiW HPE-Li controller is already active.' }
if (Test-Path -LiteralPath (Join-Path $Outdir 'controller_complete.json')) {
    throw 'This PiW HPE-Li run is already complete.'
}
$quoted = @(($baseArgs + '--launch') | ForEach-Object {
    if ($_ -match '["\r\n]') { throw 'Unsafe command argument.' }
    '"' + $_ + '"'
}) -join ' '
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' +
         ([guid]::NewGuid().ToString('N').Substring(0,8))
$stdout = Join-Path $Outdir "controller.$stamp.stdout.log"
$stderr = Join-Path $Outdir "controller.$stamp.stderr.log"
$child = Start-Process -FilePath $python -ArgumentList $quoted -WorkingDirectory $CodeRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Write-Host "STARTED controller PID=$($child.Id); inspect logs to confirm progress."
Write-Host "Controller: $stdout"
Write-Host "Cells: $Outdir\logs\train\*.log and $Outdir\logs\evaluate\*.log"
