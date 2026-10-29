<#
.SYNOPSIS
Prepare or run only TrojanNN adapted / HPE-Li / MM-Fi P1S1 / seed42.
.DESCRIPTION
One TrojanNN worker on one GPU. Default is exclusive; -AllowConcurrentA1
permits only the separately verified HPE-Li/MM-Fi P3S3 A1 peer on cuda:0.
Uses an existing matching clean P1S1 checkpoint only
to construct the model-informed trigger; the poisoned victim starts fresh.
No environment installation, process killing, or automatic P3S3 scheduling.
#>
[CmdletBinding()]
param(
    [ValidateSet('Prepare','CheckOnly','Launch','Status','Export')]
    [string]$Mode = 'Status',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [string]$CleanCheckpoint = '',
    [Alias('Python')][string]$PythonExe = '',
    [ValidatePattern('^cuda:[0-9]+$')][string]$Device = 'cuda:0',
    [switch]$AllowConcurrentA1
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$Outdir) {
    $RunName = if ($AllowConcurrentA1) {
        'mmfi_p1s1_hpeli_trojannn_s42_with_a1_v1'
    } else {
        'mmfi_p1s1_hpeli_trojannn_s42_v1'
    }
    $Outdir = Join-Path $DataHome (Join-Path 'runs' $RunName)
}
if (!$CleanCheckpoint) {
    $CleanCheckpoint = Join-Path $DataHome 'runs\mmfi_three_tables_4090_win_fresh_v1\seed42\clean\checkpoint.pt'
}
$Outdir = [IO.Path]::GetFullPath($Outdir)
$CleanCheckpoint = [IO.Path]::GetFullPath($CleanCheckpoint)
if (!$PythonExe) {
    $Candidate = Join-Path $DataHome 'envs\dose-backdoor\python.exe'
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) { $PythonExe = $Candidate }
}
if (!$PythonExe) {
    throw 'Pass -PythonExe pointing to the existing CUDA environment. No environment was installed.'
}
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
$Runner = Join-Path $Repo 'ATKBackd\run_mmfi_p1s1_trojannn.py'
foreach ($Value in @($Repo,$DataHome,$Outdir,$CleanCheckpoint,$PythonExe,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') {
        throw "Unsupported path: $Value"
    }
}
if ($Outdir.TrimEnd('\') -eq $DataHome.TrimEnd('\') -or
        $Outdir.TrimEnd('\') -eq [IO.Path]::GetPathRoot($Outdir).TrimEnd('\')) {
    throw 'Outdir must be a separate run folder, not DataHome or a volume root.'
}
foreach ($Value in @($PythonExe,$Runner)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing: $Value" }
}
if ($Mode -in @('Prepare','CheckOnly','Launch') -and
        !(Test-Path -LiteralPath $CleanCheckpoint -PathType Leaf)) {
    throw "Matching clean HPE-Li/MM-Fi P1S1 checkpoint is required: $CleanCheckpoint"
}
$Common = @('-u',$Runner,'--data-home',$DataHome,'--outdir',$Outdir,
            '--clean-checkpoint',$CleanCheckpoint,'--device',$Device)
if ($AllowConcurrentA1) { $Common += '--allow-concurrent-a1' }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

function Invoke-Runner([string]$Phase) {
    & $PythonExe @Common '--mode' $Phase
    if ($LASTEXITCODE -ne 0) {
        throw "TrojanNN $Phase failed; no later phase was started by this invocation."
    }
}
function ConvertTo-NativeArgument([string]$Value) {
    # Windows command-line quoting: double slash runs before a quote and
    # before the closing quote. In particular, a trailing slash is not lost.
    $Escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $Escaped = [regex]::Replace($Escaped, '(\\+)$', '$1$1')
    return '"' + $Escaped + '"'
}
if ($Mode -eq 'Prepare') {
    Invoke-Runner 'prepare'
    'PREPARED_ONLY: inspect the runner receipts. No full victim training launched.'
    return
}
if ($Mode -eq 'CheckOnly') { Invoke-Runner 'check'; return }
if ($Mode -eq 'Status') { Invoke-Runner 'status'; return }
if ($Mode -eq 'Export') { Invoke-Runner 'export'; return }

# The runner rechecks this gate and obtains its exclusive run lock in the
# child. A second launch cannot bypass it by racing this helper invocation.
Invoke-Runner 'check'
$Logs = Join-Path $Outdir 'logs'
[IO.Directory]::CreateDirectory($Logs) | Out-Null
$Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$Stdout = Join-Path $Logs "controller.$Stamp.stdout.log"
$Stderr = Join-Path $Logs "controller.$Stamp.stderr.log"
$Arguments = @($Common + @('--mode','launch'))
$Quoted = @($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
$Child = Start-Process -FilePath $PythonExe -ArgumentList $Quoted -WorkingDirectory $Repo `
    -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
if ($Child.WaitForExit(1500)) {
    if ($Child.ExitCode -ne 0) { throw "Runner exited early; inspect $Stdout and $Stderr" }
    'RUNNER_EXITED_SUCCESSFULLY; inspect Status/results to distinguish a completed run from new training.'
} else {
    "STARTED PID=$($Child.Id); only TrojanNN adapted P1S1, max one worker on $Device."
}
"STDOUT=$Stdout"
"STDERR=$Stderr"
'A live PID is not proof of progress. Inspect changing epoch logs or use -Mode Status.'
'Hidden Start-Process does not guarantee survival after SSH disconnect; keep this SSH session open.'
