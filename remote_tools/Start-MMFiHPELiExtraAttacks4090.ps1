<#
.SYNOPSIS
Prepare or launch the isolated MM-Fi HPE-Li LF/TSBA seed42 pilot on one RTX 4090.
.DESCRIPTION
DryRun, Preflight, and GpuSmoke do not train full victims. Launch starts one
hidden controller without registering a Scheduled Task or requiring admin.
The Python runner verifies source/data/GPU receipts before training and owns
the two-worker GPU scheduler. This helper never stops another process.
#>
[CmdletBinding()]
param(
    [ValidateSet('DryRun','Preflight','GpuSmoke','Launch','Status','CheckOnly')]
    [string]$Mode = 'DryRun',
    [ValidateSet(2)][int]$JobsPerGpu = 2,
    [string]$DataHome = "$env:USERPROFILE\backdooranalog",
    [string]$Outdir = '',
    [string]$EnvironmentPrefix = ''
)

$ErrorActionPreference = 'Stop'
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\mmfi_hpeli_extra_attacks_s42_4090_v1' }
if (!$EnvironmentPrefix) { $EnvironmentPrefix = Join-Path $DataHome 'envs\dose-backdoor' }
$code = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
foreach ($name in @('DataHome','Outdir','EnvironmentPrefix')) {
    $value = [IO.Path]::GetFullPath((Get-Variable -Name $name -ValueOnly)).TrimEnd('\')
    if ($value.StartsWith('\\') -or $value -match '["\r\n]') {
        throw "$name must be a local path without quotes or newlines."
    }
    Set-Variable -Name $name -Value $value
}
$python = Join-Path $EnvironmentPrefix 'python.exe'
$runner = Join-Path $code 'ATKBackd\run_hpeli_extra_attacks_4090.py'
foreach ($file in @($python,$runner)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}

$env:PYTHONUTF8 = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$baseArgs = @('-u',$runner,'--data-home',$DataHome,'--outdir',$Outdir,
              '--device','cuda:0','--jobs-per-gpu',[string]$JobsPerGpu)
$flags = @{DryRun='--dry-run'; Preflight='--preflight'; GpuSmoke='--gpu-smoke';
           Status='--status'; CheckOnly='--check-only'}
if ($Mode -ne 'Launch') {
    & $python @baseArgs $flags[$Mode]
    if ($LASTEXITCODE -ne 0) { throw "$Mode failed; no later step launched." }
    if ($Mode -eq 'Status') {
        $active = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -and
                $_.CommandLine.Contains('run_hpeli_extra_attacks_4090.py') -and
                $_.CommandLine.Contains('--launch')
            })
        Write-Host "Controller PIDs: $(@($active | ForEach-Object ProcessId) -join ',')"
        foreach ($kind in @('stdout','stderr')) {
            $latest = Get-ChildItem -LiteralPath $Outdir -Filter "controller.*.$kind.log" `
                -File -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latest) {
                Write-Host "===== $($latest.FullName) ====="
                Get-Content -LiteralPath $latest.FullName -Encoding UTF8 -Tail 10
            }
        }
    }
    return
}

foreach ($name in @('extra_attack_plan.json','prelaunch_gate.json')) {
    $path = Join-Path $Outdir $name
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing $path; run Preflight and GpuSmoke before Launch."
    }
}
$existing = @(Get-CimInstance Win32_Process -ErrorAction Stop |
    Where-Object {
        $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -and
        $_.CommandLine.Contains('run_hpeli_extra_attacks_4090.py') -and
        $_.CommandLine.Contains('--launch')
    })
if ($existing.Count) {
    $pids = (@($existing | ForEach-Object ProcessId) -join ',')
    throw "An HPE-Li extra-attack controller is already running: PID=$pids"
}

# Wrap every argument after rejecting quotes; Start-Process otherwise joins
# ArgumentList elements with spaces and breaks paths such as C:\Users\Van\My Data.
$quoted = @($baseArgs + '--launch' | ForEach-Object {
    if ($_ -match '["\r\n]') { throw 'Unsafe command argument.' }
    '"' + $_ + '"'
}) -join ' '
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' +
         ([guid]::NewGuid().ToString('N').Substring(0,8))
$stdout = Join-Path $Outdir "controller.$stamp.stdout.log"
$stderr = Join-Path $Outdir "controller.$stamp.stderr.log"
$child = Start-Process -FilePath $python -ArgumentList $quoted -WorkingDirectory $code `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Start-Sleep -Milliseconds 750
$child.Refresh()
if ($child.HasExited -and $child.ExitCode -ne 0) {
    Get-Content -LiteralPath $stderr -Encoding UTF8 -Tail 20 -ErrorAction SilentlyContinue
    throw "Controller exited immediately with code $($child.ExitCode)."
}
Write-Host "STARTED controller PID=$($child.Id); this does not prove training progress."
Write-Host "Controller stdout: $stdout"
Write-Host "Controller stderr: $stderr"
Write-Host "Cell logs: $Outdir\logs\*.log"
Write-Host "Check later: powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Mode Status"
