<#
.SYNOPSIS
Launch one serialized seed42 A1 defense pilot for an MM-Fi pose backbone.
.DESCRIPTION
DryRun is default. Prepare runs dry run, source/data preflight and all six
serial GPU smokes. Launch requires both receipts and creates a persistent task.
The historical victims, other tasks, and earlier repair outputs are untouched.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('hpeli','metafiplusplus','wiflow')]
    [string]$Backbone,
    [ValidateSet('DryRun','Prepare','Preflight','GpuSmoke','Launch','Status','CheckOnly')]
    [string]$Mode = 'DryRun',
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$SourceRoot = '',
    [string]$Archive = '',
    [string]$Outdir = '',
    [string]$EnvironmentPrefix = '',
    [string]$TaskName = ''
)
$ErrorActionPreference = 'Stop'
$code = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$lane = @{hpeli='hpeli_light_4090';metafiplusplus='metafi_light_4090';wiflow='wiflow_4090'}[$Backbone]
if (!$EnvironmentPrefix) { $EnvironmentPrefix = Join-Path $DataHome 'envs\dose-backdoor' }
if (!$TaskName) { $TaskName = "MMFi-Light-Defense-$Backbone-S42-v3-BNP-FST" }
if ($TaskName -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid task name' }
foreach ($name in @('DataHome','EnvironmentPrefix')) {
    $value = [IO.Path]::GetFullPath((Get-Variable $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') { throw "Invalid local path: $name" }
    Set-Variable $name $value
}
foreach ($name in @('SourceRoot','Archive','Outdir')) {
    $value = Get-Variable $name -ValueOnly
    if ($value) {
        $value = [IO.Path]::GetFullPath($value).TrimEnd('\')
        if ($value.StartsWith('\\') -or $value -match '["\r\n]') { throw "Invalid local path: $name" }
        Set-Variable $name $value
    }
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code "remote_tools\$lane\defense4090.py"
foreach ($file in @($python,$runner)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$env:PYTHONUTF8='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$runnerArgs = @('-u',$runner,'--data-home',$DataHome,'--jobs-per-gpu','1')
if ($SourceRoot) { $runnerArgs += @('--source-root',$SourceRoot) }
if ($Outdir) { $runnerArgs += @('--outdir',$Outdir) }
if ($Archive) {
    if ($Backbone -eq 'hpeli') { throw 'HPE-Li uses the existing local run; do not pass Archive' }
    $runnerArgs += @('--archive',$Archive)
}
$flags = @{DryRun='--dry-run';Preflight='--preflight';GpuSmoke='--gpu-smoke';
           Status='--status';CheckOnly='--check-only'}
if ($Mode -ne 'Launch') {
    $phases = if ($Mode -eq 'Prepare') {
        @('--dry-run','--preflight','--gpu-smoke')
    } else { @($flags[$Mode]) }
    foreach ($phase in $phases) {
        & $python @runnerArgs $phase
        if ($LASTEXITCODE -ne 0) { throw "$Backbone $phase failed; full defense not started" }
    }
    if ($Mode -eq 'Prepare') { Write-Host "PREPARED_$Backbone; full defense has not started" }
    return
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Scheduled task $TaskName already exists; inspect its state/log before another launch"
}
if (!$Outdir) { $Outdir = Join-Path $DataHome "runs\${Backbone}_light_a1_defense4090_s42_v3_bnp_fst_nc" }
foreach ($receipt in @('preflight.json','gpu_smoke.json')) {
    $path = Join-Path $Outdir $receipt
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing gate: $path" }
    $saved = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($saved.status -ne 'PASS') { throw "Failed gate: $path" }
}
$cfg = @{python=$python;args=($runnerArgs + '--launch');code=$code;output=$Outdir}
$encodedCfg = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($cfg | ConvertTo-Json -Compress)))
$payload = @'
$ErrorActionPreference='Stop'
$cfg=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CFG__')) | ConvertFrom-Json
$env:PYTHONUTF8='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$log=$cfg.output + '.task.' + (Get-Date -Format yyyyMMdd_HHmmss_fff) + '_' + ([guid]::NewGuid().ToString('N').Substring(0,8))
try {
    $quoted=@($cfg.args | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $child=Start-Process -FilePath $cfg.python -ArgumentList $quoted -WorkingDirectory $cfg.code -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$log.stdout.log" -RedirectStandardError "$log.stderr.log"
    exit $child.ExitCode
} catch {
    $_ | Out-String | Set-Content -LiteralPath "$log.wrapper-error.log" -Encoding UTF8
    exit 1
}
'@
$payload = $payload.Replace('__CFG__',$encodedCfg)
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($payload))
$shell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$action = New-ScheduledTaskAction -Execute $shell -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded" -WorkingDirectory $code
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$principal = New-ScheduledTaskPrincipal -UserId $sid -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task = New-ScheduledTask -Action $action -Principal $principal -Settings $settings -Description 'MM-Fi A1 seed42: NoDef, FT10, CLP, BNP, FST-adapted, NC-adapted; one GPU serial; no retry.'
Register-ScheduledTask -TaskName $TaskName -InputObject $task -ErrorAction Stop | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Task $TaskName started. Controller: $Outdir.task.*.stdout.log / stderr.log"
Write-Host "Outputs: $Outdir\$Backbone\seed42\<none|ft|clp|bnp|fst|nc>\evaluation.json"
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
