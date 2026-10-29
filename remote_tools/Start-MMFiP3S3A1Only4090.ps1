<#
.SYNOPSIS
Run only historical HPE-Li / MM-Fi P3S3 A1 seed42 on an RTX4090.
.DESCRIPTION
Uses the unchanged run_mmfi_p3_splits.py scientific runner. Its full P3
preflight is retained, but Launch selects exactly p3_s3_a1, never its six-cell
controller. Prepare runs no full training. Historical outputs are untouched.
#>
[CmdletBinding()]
param(
    [ValidateSet('Inspect','Prepare','RefreshDesktopGate','CheckOnly','Launch','Status')]
    [string]$Mode = 'Inspect',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$Outdir = '',
    [Alias('Python')][string]$PythonExe = '',
    [switch]$AllowConcurrentSIG
)
$ErrorActionPreference = 'Stop'
$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$Outdir) {
    $RunName = if ($AllowConcurrentSIG) { 'mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1' } else { 'mmfi_p3s3_hpeli_a1_s42_4090_v1' }
    $Outdir = Join-Path $DataHome (Join-Path 'runs' $RunName)
}
$Outdir = [IO.Path]::GetFullPath($Outdir)
if (!$PythonExe) {
    foreach ($Candidate in @(
        (Join-Path $DataHome 'envs\dose-backdoor\python.exe'),
        (Join-Path $DataHome 'envs\mmfi-p3\Scripts\python.exe')
    )) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) { $PythonExe=$Candidate; break }
    }
}
if (!$PythonExe) { throw 'No existing Python found; pass -PythonExe. No environment was installed.' }
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
$Runner = Join-Path $Repo 'ATKBackd\run_mmfi_p3_splits.py'
foreach ($Value in @($Repo,$DataHome,$Outdir,$PythonExe,$Runner)) {
    if ($Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') { throw "Unsupported path: $Value" }
}
foreach ($Value in @($Runner,$PythonExe)) {
    if (!(Test-Path -LiteralPath $Value -PathType Leaf)) { throw "Missing: $Value" }
}
$Common = @('-u',$Runner,'--data-home',$DataHome,'--outdir',$Outdir,
            '--device','cuda:0','--workers','0','--jobs-per-gpu','1')
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$CellFolder = Join-Path $Outdir 'p3_s3\a1'
$GatePath = Join-Path $Outdir 'a1_only_4090_gate.json'
$SIGRunner = Join-Path $Repo 'ATKBackd\run_mmfi_p3_s3_three_attacks_4090.py'
$SIGOutdir = Join-Path $DataHome 'runs\mmfi_p3s3_three_attacks_s42_4090_v1'

function Read-Json([string]$Path) { [IO.File]::ReadAllText($Path) | ConvertFrom-Json }
function Invoke-Legacy([string]$Phase) {
    & $PythonExe @Common '--mode' $Phase
    if ($LASTEXITCODE -ne 0) { throw "P3 $Phase failed; no full training launched." }
}
function Get-HostBinding {
    $GpuRows = @(& nvidia-smi '-i' '0' '--query-gpu=name,uuid' '--format=csv,noheader,nounits')
    if ($LASTEXITCODE -ne 0 -or $GpuRows.Count -ne 1 -or $GpuRows[0] -notmatch 'RTX 4090') {
        throw 'This launcher requires RTX4090 at cuda:0; no fallback.'
    }
    $Runtime = & $PythonExe -c "import json,sys,platform,os,torch,numpy,scipy; assert torch.cuda.is_available(),'CUDA unavailable'; print(json.dumps(dict(python=sys.version,executable=sys.executable,system=platform.system(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),actual_gpu=torch.cuda.get_device_name(0),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),sort_keys=True))"
    if ($LASTEXITCODE -ne 0) { throw 'Runtime probe failed.' }
    if (($Runtime | ConvertFrom-Json).actual_gpu -notmatch 'RTX 4090') { throw 'Torch cuda:0 is not RTX4090.' }
    return [ordered]@{ computer=$env:COMPUTERNAME; gpu=$GpuRows[0].Trim(); runtime=($Runtime -join "`n").Trim(); python_sha256=(Get-FileHash -LiteralPath $PythonExe -Algorithm SHA256).Hash }
}
function Test-ExactArgument([string]$CommandLine,[string]$Name,[string]$Value) {
    $NamePattern = '(?i)(?:^|\s)"?' + [regex]::Escape($Name) + '"?(?=\s|$)'
    if ([regex]::Matches($CommandLine,$NamePattern).Count -ne 1) { return $false }
    $Pattern = '(?i)(?:^|\s)"?' + [regex]::Escape($Name) + '"?\s+(?:"' +
        [regex]::Escape($Value) + '"|' + [regex]::Escape($Value) + ')(?=\s|$)'
    return $CommandLine -match $Pattern
}
function Test-AuthorizedSIG($Process) {
    if (!$Process -or $Process.Name -notin @('python.exe','pythonw.exe')) { return $false }
    $Line = [string]$Process.CommandLine
    $ScriptPattern = '(?i)(?:^|\s)(?:"' + [regex]::Escape($SIGRunner) + '"|' +
        [regex]::Escape($SIGRunner) + ')(?=\s|$)'
    if ($Line -notmatch $ScriptPattern) { return $false }
    foreach ($Pair in @(
        @('--data-home',$DataHome),@('--outdir',$SIGOutdir),@('--device','cuda:0'),
        @('--jobs-per-gpu','1'),@('--mode','cell'),@('--work','train'),@('--method','sig_source')
    )) {
        if (!(Test-ExactArgument $Line $Pair[0] $Pair[1])) { return $false }
    }
    return $true
}
function Test-AnyDeskDesktopPath([string]$ProcessPath) {
    # WDDM lists remote-display rendering as compute. Do not stop the remote
    # access application, and do not exempt arbitrary executables by basename.
    foreach ($Root in @($env:ProgramFiles,${env:ProgramFiles(x86)})) {
        if ($Root -and [string]::Equals($ProcessPath,
                (Join-Path $Root 'AnyDesk\AnyDesk.exe'),[StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}
function Assert-GpuPolicy {
    $Rows = @(& nvidia-smi '--query-compute-apps=pid,process_name' '--format=csv,noheader,nounits')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect GPU processes.' }
    $Desktop = @('explorer.exe','dwm.exe','searchhost.exe','startmenuexperiencehost.exe',
        'shellexperiencehost.exe','shellhost.exe','applicationframehost.exe','textinputhost.exe',
        'lockapp.exe','logonui.exe','systemsettings.exe','code.exe','chrome.exe','msedge.exe',
        'msedgewebview2.exe','windowsterminal.exe','taskmgr.exe','snippingtool.exe','teamviewer.exe',
        'ms-teams.exe','phoneexperiencehost.exe','crossdeviceresume.exe','copilot.exe','copilotapp.exe','copilotapphost.exe')
    $SIGIds = @()
    foreach ($Row in $Rows) {
        if (!$Row.Trim()) { continue }
        if ($Row -notmatch '^\s*([0-9]+)\s*,\s*(.+?)\s*$') { throw "Unidentified GPU process: $Row" }
        $GpuProcessId = [int]$Matches[1]
        $Executable = [IO.Path]::GetFileName($Matches[2]).ToLowerInvariant()
        if ($GpuProcessId -eq 4 -and $Executable -eq 'system') { continue }
        if ($Executable -in $Desktop) { continue }
        if (Test-AnyDeskDesktopPath $Matches[2]) { continue }
        if ($AllowConcurrentSIG) {
            $Candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$GpuProcessId"
            if (Test-AuthorizedSIG $Candidate) {
                $SIGIds += $GpuProcessId
                continue
            }
        }
        throw "Other GPU work active: $Row. Only the explicitly authorized SIG peer may share; no process was stopped."
    }
    $SIGIds = @($SIGIds | Select-Object -Unique)
    if ($SIGIds.Count -gt 1) { throw 'Multiple SIG workers found; only one authorized peer is allowed.' }
    if ($SIGIds.Count -eq 1) { Write-Host "AUTHORIZED_SIG_PEER PID=$($SIGIds[0]); left running. At most one A1 + one SIG." }
}
function Assert-DesktopGateUpgrade($SavedGate,$CurrentGate) {
    # Only the exact published d5f1b1f wrapper (LF or Git's Windows CRLF) is
    # eligible. No arbitrary source/gate/runtime change is grandfathered in.
    $Predecessors = @(
        '9e6953da833683cc0968ddf3a76c3a28bfc777b589570103353bc8f1a5caf565',
        '61fbb0788458880b9cd2f6dfc529d199f9cf034ce2694229e538221479ae4eae'
    )
    if ([string]$SavedGate.wrapper_sha256 -notin $Predecessors) {
        throw 'Gate is not from the exact pre-AnyDesk launcher; refusing refresh.'
    }
    $Comparison = $SavedGate | ConvertTo-Json -Depth 12 | ConvertFrom-Json
    $Comparison.wrapper_sha256 = $CurrentGate.wrapper_sha256
    if (($Comparison | ConvertTo-Json -Depth 12 -Compress) -cne
            ($CurrentGate | ConvertTo-Json -Depth 12 -Compress)) {
        throw 'Something besides the desktop-wrapper hash changed; refusing refresh.'
    }
}
function Get-GateBinding {
    $Files = [ordered]@{}
    foreach ($Name in @('p3_splits.plan.json','preflight.json','gpu_smoke.json')) {
        $Path = Join-Path $Outdir $Name
        if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing $Path; run Prepare." }
        $Files[$Name] = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    }
    $Plan = Read-Json (Join-Path $Outdir 'p3_splits.plan.json')
    $Smoke = Read-Json (Join-Path $Outdir 'gpu_smoke.json')
    if ($Smoke.jobs_per_gpu -ne 1 -or $Smoke.physical_batch -ne 32 -or $Smoke.plan_sha256 -ne $Plan.sha256) {
        throw 'Physical B32 / single-job GPU gate differs.'
    }
    $Cell = @($Plan.cells | Where-Object { $_.id -eq 'p3_s3_a1' })
    if ($Cell.Count -ne 1 -or $Cell[0].cfg.seed -ne 42 -or $Cell[0].cfg.epochs -ne 50 -or
        $Cell[0].cfg.batch_size -ne 32 -or $Cell[0].cfg.lr -ne 0.001 -or $Cell[0].cfg.rho -ne 0.1) {
        throw 'Historical A1 recipe differs; refusing launch.'
    }
    return [ordered]@{ scope='a1_only_hpeli_mmfi_p3s3_s42_4090_v1'; cell='p3_s3_a1';
        data_home=$DataHome; outdir=$Outdir; files=$Files;
        execution_policy=([ordered]@{ allow_concurrent_sig=[bool]$AllowConcurrentSIG;
            maximum_a1_workers=1; maximum_sig_workers=([int][bool]$AllowConcurrentSIG);
            allowed_sig_runner=$SIGRunner; allowed_sig_outdir=$SIGOutdir });
        wrapper_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash;
        runner_sha256=(Get-FileHash -LiteralPath $Runner -Algorithm SHA256).Hash;
        host_binding=(Get-HostBinding) }
}
if ($Mode -eq 'Inspect' -or $Mode -eq 'Status') {
    "SCOPE: only A1 / HPE-Li / MM-Fi P3S3 / seed42; one A1 worker. No training launched."
    "AllowConcurrentSIG: $([bool]$AllowConcurrentSIG); other training jobs remain blocked."
    "Python: $PythonExe"
    "Dataset expected: $DataHome\datasets\Compress"
    "Motion prior: $DataHome\actions\data_bend.npy"
    "Output: $CellFolder"
    "Prepared: $(Test-Path -LiteralPath $GatePath)"
    if (Test-Path -LiteralPath $CellFolder -PathType Container) {
        Get-ChildItem -LiteralPath $CellFolder -Filter 'console*.log' -File |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
            ForEach-Object { Get-Content -LiteralPath $_.FullName -Encoding UTF8 -Tail 12 }
    }
    $Report = Join-Path $CellFolder 'p3_evaluation.json'
    if (Test-Path -LiteralPath $Report) { (Read-Json $Report).metrics | Format-List }
    return
}
Assert-GpuPolicy
if ($Mode -eq 'Prepare') {
    $null = Get-HostBinding
    if (Test-Path -LiteralPath $GatePath) { throw 'Already prepared; use CheckOnly/Launch, not another Prepare.' }
    if (Test-Path -LiteralPath (Join-Path $CellFolder 'checkpoint.pt')) {
        throw 'Output already contains a checkpoint; choose a NEW Outdir instead of changing historical receipts.'
    }
    Invoke-Legacy 'preflight'
    Assert-GpuPolicy
    Invoke-Legacy 'gpu-smoke'
    Assert-GpuPolicy
    $Binding = Get-GateBinding
    $Json = $Binding | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText($GatePath,$Json,[Text.UTF8Encoding]::new($false))
    'PREPARED_ONLY: historical full P3 input audit + B32 CUDA smoke; no full training.'
    return
}
if (!(Test-Path -LiteralPath $GatePath)) { throw 'Missing A1-only gate; run Prepare first.' }
$SavedObject = Read-Json $GatePath
$CurrentObject = Get-GateBinding
$Saved = $SavedObject | ConvertTo-Json -Depth 12 -Compress
$Current = $CurrentObject | ConvertTo-Json -Depth 12 -Compress
if ($Mode -eq 'RefreshDesktopGate') {
    if ($Saved -ceq $Current) { 'DESKTOP_GATE_ALREADY_CURRENT; no training launched.'; return }
    if (Test-Path -LiteralPath (Join-Path $CellFolder 'checkpoint.pt')) {
        throw 'A1 checkpoint already exists; this prelaunch-only desktop refresh is not applicable.'
    }
    Assert-DesktopGateUpgrade $SavedObject $CurrentObject
    # Keep the full split audit, but retest B32 with the current SIG/desktop
    # load. The scientific runner also revalidates its complete source plan.
    Invoke-Legacy 'gpu-smoke'
    Assert-GpuPolicy
    $CurrentObject = Get-GateBinding
    Assert-DesktopGateUpgrade $SavedObject $CurrentObject
    $Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
    $Backup = Join-Path $Outdir "a1_only_4090_gate.before_anydesk.$Stamp.json"
    [IO.File]::Copy($GatePath,$Backup,$false)
    $Temporary = $GatePath + '.desktop-refresh.' + $Stamp + '.saving'
    [IO.File]::WriteAllText($Temporary,($CurrentObject | ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    [IO.File]::Replace($Temporary,$GatePath,$null)
    $Receipt = [ordered]@{ scope='prelaunch_anydesk_desktop_only'; previous_gate_backup=$Backup;
        previous_wrapper_sha256=$SavedObject.wrapper_sha256; current_wrapper_sha256=$CurrentObject.wrapper_sha256;
        unchanged_scientific_receipts=$CurrentObject.files; current_b32_smoke_passed=$true;
        checkpoint_modified=$false; full_training_started=$false }
    [IO.File]::WriteAllText((Join-Path $Outdir "desktop_gate_refresh.$Stamp.json"),
        ($Receipt | ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    'DESKTOP_GATE_REFRESHED; previous gate backed up; B32 retested; no training launched.'
    return
}
if ($Saved -ne $Current) { throw 'Code/data receipt/runtime/GPU/scope gate changed; refusing launch.' }
if ($Mode -eq 'CheckOnly') { 'VERIFIED_PRELAUNCH_GATE; no training launched.'; return }
Assert-GpuPolicy

# Crucial: use the verified historical cell entry, NOT its six-cell controller.
$Arguments = @($Common + @('--mode','launch','--_cell','p3_s3_a1'))
$Quoted = @($Arguments | ForEach-Object { '"'+$_+'"' }) -join ' '
[IO.Directory]::CreateDirectory($CellFolder) | Out-Null
$Stamp = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$Stdout = Join-Path $CellFolder 'console.log'
$Stderr = Join-Path $Outdir "a1.$Stamp.stderr.log"
if (Test-Path -LiteralPath $Stdout) { $Stdout = Join-Path $CellFolder "console.$Stamp.log" }
$Child = Start-Process -FilePath $PythonExe -ArgumentList $Quoted -WorkingDirectory $Repo `
    -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
Start-Sleep -Seconds 2
if ($Child.HasExited -and $Child.ExitCode -ne 0) { throw "A1 exited early; inspect $Stdout and $Stderr" }
"STARTED A1 ONLY PID=$($Child.Id); inspect the log for real epoch progress."
"Log: $Stdout"
"Errors: $Stderr"
