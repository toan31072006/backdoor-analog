<#
.SYNOPSIS
Isolated, environment-matched MetaFi++ PiW3D Clean diagnostic on Windows 3090.
.DESCRIPTION
Setup installs only into a new, launcher-owned venv. Prepare performs audits and
real CUDA B32 smoke, compares the archived Vast loss, then stops. This handoff is
Setup -> Prepare -> inspect only; do not invoke Launch without a new user request.
Launch remains explicit and foreground; keep SSH open.
No A1/baseline scheduling, driver update, process kill, automatic retry or fallback.
#>
[CmdletBinding()]
param(
    [ValidateSet('Setup','Inspect','Prepare','CheckOnly','Launch','Status')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [Alias('Dataset')][string]$DatasetRoot = '',
    [Alias('Action')][string]$ActionNpy = '',
    [Alias('Out')][string]$Outdir = '',
    [string]$EnvDir = '',
    [string]$BasePython = (Join-Path $env:USERPROFILE 'eclare-env\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\Person-in-WiFi-3D_processed_v1' }
if (!$ActionNpy) { $ActionNpy = Join-Path $DataHome 'actions\data_bend.npy' }
if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\piw_metafi_clean_vastenv_s42_3090_v1' }
if (!$EnvDir) { $EnvDir = Join-Path $DataHome 'envs\piw-metafi-vast211' }
$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
$ActionNpy = [IO.Path]::GetFullPath($ActionNpy)
$Outdir = [IO.Path]::GetFullPath($Outdir)
$EnvDir = [IO.Path]::GetFullPath($EnvDir).TrimEnd('\')
$BasePython = [IO.Path]::GetFullPath($BasePython)
$PythonExe = Join-Path $EnvDir 'Scripts\python.exe'
$Runner = Join-Path $Repo 'remote_tools\piw_metafi_clean_vastenv_3090.py'
$OwnerFile = Join-Path $EnvDir 'clean_vastenv_setup.owner.json'
$ReadyFile = Join-Path $EnvDir 'clean_vastenv_setup.ready.json'
$Profile = 'piw_metafi_clean_vastenv_setup_v1'
$Pins = [ordered]@{
    torch = '2.11.0+cu128'
    torchvision = '0.26.0+cu128'
    numpy = '2.5.4'
    scipy = '1.17.1'
    PyYAML = '6.0.2'
}

# Fail before invoking either Python: inherited destinations/import paths can
# otherwise escape a venv despite --require-virtualenv. Do not change user policy.
foreach ($Name in @('PIP_TARGET','PIP_PREFIX','PIP_ROOT','PIP_USER','PYTHONHOME','PYTHONPATH')) {
    $Setting = Get-Item -LiteralPath ("Env:" + $Name) -ErrorAction SilentlyContinue
    if ($null -ne $Setting -and ![string]::IsNullOrEmpty([string]$Setting.Value)) {
        throw "Refusing inherited $Name; remove it explicitly for this invocation before retrying. No Python, installation or training started."
    }
}
$env:PYTHONNOUSERSITE = '1'

foreach ($Value in @($Repo,$DataHome,$DatasetRoot,$ActionNpy,$Outdir,$EnvDir,$BasePython,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') {
        throw "Unsupported local path: $Value"
    }
    # GetFullPath is lexical: check every existing ancestor before Setup can
    # create a venv. A junction above DataHome would otherwise escape that scope.
    $PathCursor = $Value
    while (![string]::IsNullOrEmpty($PathCursor)) {
        if (Test-Path -LiteralPath $PathCursor) {
            $PathEntry = Get-Item -LiteralPath $PathCursor -Force
            if ($PathEntry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Symlink/junction ancestor is not permitted: $PathCursor. No installation or training started."
            }
        }
        $PathParent = Split-Path -Parent $PathCursor
        if ($PathParent -eq $PathCursor) { break }
        $PathCursor = $PathParent
    }
}
if (!(Test-Path -LiteralPath $DataHome -PathType Container)) {
    throw "DataHome does not exist: $DataHome"
}
# Setup can write only within the explicitly scoped environment collection.
$EnvRoot = [IO.Path]::GetFullPath((Join-Path $DataHome 'envs')).TrimEnd('\') + '\'
if (!$EnvDir.StartsWith($EnvRoot,[StringComparison]::OrdinalIgnoreCase) -or
    $EnvDir.Equals($EnvRoot.TrimEnd('\'),[StringComparison]::OrdinalIgnoreCase)) {
    throw 'EnvDir must be a new child directory inside DataHome\envs.'
}
$EnvCursor = $EnvDir
while ($EnvCursor.StartsWith($EnvRoot,[StringComparison]::OrdinalIgnoreCase) -or
       $EnvCursor.Equals($EnvRoot.TrimEnd('\'),[StringComparison]::OrdinalIgnoreCase)) {
    if (Test-Path -LiteralPath $EnvCursor) {
        $Entry = Get-Item -LiteralPath $EnvCursor -Force
        if ($Entry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'Environment paths may not use symlinks or junctions; no mutation permitted.'
        }
    }
    $EnvCursor = Split-Path -Parent $EnvCursor
}
if ($BasePython.StartsWith(($EnvDir + '\'),[StringComparison]::OrdinalIgnoreCase)) {
    throw 'BasePython must be outside the new diagnostic environment.'
}
if ($Outdir.Equals($EnvDir,[StringComparison]::OrdinalIgnoreCase) -or
    $Outdir.StartsWith(($EnvDir + '\'),[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Outdir must be outside the environment.'
}

$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
# Keep the previous Windows/3090 thread policy fixed while testing library pins.
# Frozen Windows code also selects SEQUENTIAL; this is not an exact Linux clone.
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'

function Invoke-PythonChecked([string]$Executable,[string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE); no later phase or full training started."
    }
}

function Invoke-PythonScriptChecked([string]$Executable,[string]$Code,[string[]]$Arguments) {
    # stdin avoids Windows PowerShell 5 stripping embedded quotes in python -c.
    $Code | & $Executable '-' @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python probe failed (exit $LASTEXITCODE); no later phase or full training started."
    }
}

function Assert-Owner {
    if (!(Test-Path -LiteralPath $OwnerFile -PathType Leaf)) {
        throw "Existing environment is not owned by this launcher: $EnvDir. Nothing installed or deleted."
    }
    $Owner = [IO.File]::ReadAllText($OwnerFile) | ConvertFrom-Json
    if ($Owner.profile -ne $Profile -or $Owner.python -ne '3.12.15' -or
        !([string]$Owner.env_dir).Equals($EnvDir,[StringComparison]::OrdinalIgnoreCase) -or
        !([string]$Owner.base_python).Equals($BasePython,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Environment owner marker does not match this request; no mutation permitted.'
    }
    foreach ($Name in $Pins.Keys) {
        if ($Owner.pins.$Name -ne $Pins[$Name]) { throw "Environment pin mismatch: $Name" }
    }
    if (!(Test-Path -LiteralPath (Join-Path $EnvDir 'pyvenv.cfg') -PathType Leaf) -or
        !(Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw 'Owned environment is incomplete; no automatic deletion or replacement.'
    }
}

$VerifyCode = @'
import importlib.metadata as m, json, pathlib, sys
expected = {"torch":"2.11.0+cu128","torchvision":"0.26.0+cu128","numpy":"2.5.4","scipy":"1.17.1","PyYAML":"6.0.2"}
if sys.version_info[:3] != (3, 12, 15):
    raise SystemExit("Python must be exactly 3.12.15")
if sys.prefix == sys.base_prefix or pathlib.Path(sys.prefix).resolve() != pathlib.Path(sys.argv[1]).resolve():
    raise SystemExit("Refusing non-isolated or unexpected Python environment")
actual = {name: m.version(name) for name in expected}
if actual != expected:
    raise SystemExit("Pinned versions mismatch: " + repr(actual))
print(json.dumps({"python":sys.version.split()[0],"executable":sys.executable,"packages":actual}, indent=2))
'@

if ($Mode -eq 'Setup') {
    if (!(Test-Path -LiteralPath $BasePython -PathType Leaf)) { throw "Missing BasePython: $BasePython" }
    Invoke-PythonScriptChecked $BasePython 'import sys; assert sys.version_info[:3] == (3,12,15), "BasePython must be exactly 3.12.15"; print(sys.executable)' @()
    if (Test-Path -LiteralPath $EnvDir) {
        Assert-Owner
    } else {
        # No --clear, --upgrade, global install, or activation of eclare-env.
        Invoke-PythonChecked $BasePython @('-m','venv',$EnvDir)
        $Owner = [ordered]@{
            profile=$Profile; python='3.12.15'; env_dir=$EnvDir
            base_python=$BasePython; pins=$Pins; created_utc=[DateTime]::UtcNow.ToString('o')
        }
        [IO.File]::WriteAllText($OwnerFile,($Owner | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
        Assert-Owner
    }
    if (!(Test-Path -LiteralPath $ReadyFile -PathType Leaf)) {
        # Explicit indexes, after rejecting inherited source/destination config.
        $SavedPipConfig = $env:PIP_CONFIG_FILE
        $SavedPipExtra = $env:PIP_EXTRA_INDEX_URL
        try {
            $env:PIP_CONFIG_FILE = 'NUL'
            Remove-Item Env:PIP_EXTRA_INDEX_URL -ErrorAction SilentlyContinue
            $PipConfigProbe = @'
import subprocess, sys
p = subprocess.run([sys.executable, "-m", "pip", "--isolated", "config", "list"], capture_output=True, text=True, check=True)
keys = [line.split("=", 1)[0].strip() for line in p.stdout.splitlines() if "=" in line]
bad = [k for k in keys if k.rsplit(".", 1)[-1].lower() in {
    "target", "prefix", "root", "user", "extra-index-url", "find-links", "no-index"
}]
if bad:
    raise SystemExit("Refusing pip source/destination configuration keys: " + ", ".join(bad))
print("ISOLATED_PIP_SOURCE_DESTINATION_CONFIG_PASS")
'@
            Invoke-PythonScriptChecked $PythonExe $PipConfigProbe @()
            $PipCommon = @('-m','pip','--isolated','--disable-pip-version-check','--require-virtualenv','--no-input')
            $InstallScope = @('install','--prefix',$EnvDir,'--no-user')
            Invoke-PythonChecked $PythonExe ($PipCommon + $InstallScope + @('--index-url','https://download.pytorch.org/whl/cu128','torch==2.11.0+cu128','torchvision==0.26.0+cu128'))
            Invoke-PythonChecked $PythonExe ($PipCommon + $InstallScope + @('--index-url','https://pypi.org/simple','numpy==2.5.4','scipy==1.17.1','PyYAML==6.0.2'))
        } finally {
            if ($null -eq $SavedPipConfig) { Remove-Item Env:PIP_CONFIG_FILE -ErrorAction SilentlyContinue } else { $env:PIP_CONFIG_FILE = $SavedPipConfig }
            if ($null -eq $SavedPipExtra) { Remove-Item Env:PIP_EXTRA_INDEX_URL -ErrorAction SilentlyContinue } else { $env:PIP_EXTRA_INDEX_URL = $SavedPipExtra }
        }
    }
    Invoke-PythonScriptChecked $PythonExe $VerifyCode @($EnvDir)
    Invoke-PythonChecked $PythonExe @('-m','pip','--isolated','check')
    $Freeze = @(& $PythonExe -m pip --isolated freeze --all)
    if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed; environment not marked ready.' }
    [IO.File]::WriteAllLines((Join-Path $EnvDir 'clean_vastenv_setup.freeze.txt'),[string[]]$Freeze,[Text.UTF8Encoding]::new($false))
    if (!(Test-Path -LiteralPath $ReadyFile -PathType Leaf)) {
        $Ready = [ordered]@{
            profile=$Profile; python='3.12.15'; env_dir=$EnvDir; pins=$Pins
            pip_check='passed'; completed_utc=[DateTime]::UtcNow.ToString('o')
            note='Setup only. No CUDA availability, B32 smoke, or training claim.'
        }
        [IO.File]::WriteAllText($ReadyFile,($Ready | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
    }
    Write-Host "ENV_SETUP_ONLY: $PythonExe"
    Write-Host 'Exact package pins and pip check passed. No GPU smoke or full training launched.'
    Write-Host 'Next invoke -Mode Prepare explicitly; do not update the NVIDIA driver for this launcher.'
    return
}

Assert-Owner
if (!(Test-Path -LiteralPath $ReadyFile -PathType Leaf)) { throw 'Run explicit -Mode Setup successfully first.' }
if (!(Test-Path -LiteralPath $Runner -PathType Leaf)) { throw "Missing runner: $Runner" }
Invoke-PythonScriptChecked $PythonExe $VerifyCode @($EnvDir)
$Phase = @{Inspect='inspect'; Prepare='prepare'; CheckOnly='check'; Launch='launch'; Status='status'}[$Mode]
$Arguments = @('-u',$Runner,'--mode',$Phase,'--data-home',$DataHome,
               '--dataset',$DatasetRoot,'--action',$ActionNpy,'--out',$Outdir,
               '--env-dir',$EnvDir)
if ($Mode -eq 'Launch') {
    Write-Host 'FOREGROUND: keep this SSH terminal open. One Clean worker on cuda:0; no A1 or baseline job.'
}
Invoke-PythonChecked $PythonExe $Arguments
if ($Mode -eq 'Prepare') {
    Write-Host 'PREPARED_ONLY: audits and real CUDA B32 smoke passed; no full training launched.'
    Write-Host ("SMOKE_COMPARISON=" + (Join-Path $Outdir 'smoke_comparison.json'))
    Write-Host 'STOP HERE: inspect the Clean smoke comparison first. Do not invoke Launch until the user separately requests full training.'
}
