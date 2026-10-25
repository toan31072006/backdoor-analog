# MM-Fi P3/S1-S3 on the Windows RTX 3090

This is a one-seed exploratory T6 split slice: HPE-Li Clean and A1, seed 42,
50 epochs with fixed final epoch 49, all 27 MM-Fi actions, settings S3 then S2
then S1. Each setting has separate Clean and A1 victims. It is not the full
multi-backbone T6 paper table. Keep the earlier P1S1 and PiW runs separate.

Run the following in **PowerShell on RTX-3090** as user `ssh3090`. Use a
different drive in `$HomeDir` if C: does not have enough free space. Keep one
fresh `$HomeDir` for this experiment. Download/extraction can take time:
the official processed archive contains roughly 322,000 files.

## 1. Create folders and get the code

Git and Python are absent on this host. Download the exact reviewed commit as
a ZIP; no Git installation is needed for this run.

```powershell
$HomeDir = Join-Path $env:USERPROFILE 'backdooranalog'
$Commit = '8f5cc790f08115f1927d5f79efc8d37d6541923b'
$Repo = Join-Path $HomeDir "code\backdoor-analog-$Commit"
foreach ($name in @('code','downloads','actions','runs','envs')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $HomeDir $name) | Out-Null
}
Get-PSDrive -Name C | Select-Object Name,Free,Used
$CodeZip = Join-Path $HomeDir 'downloads\backdoor-analog-p3-8f5cc79.zip'
if (!(Test-Path -LiteralPath $Repo)) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/toan31072006/backdoor-analog/archive/$Commit.zip" -OutFile $CodeZip -TimeoutSec 120
    if ((Get-FileHash -LiteralPath $CodeZip -Algorithm SHA256).Hash -ne '111C3DBFB25B2B139C2796CD18F60B8529FE69462529C82C5A40CBB22C3215D7') {
        throw 'Code ZIP SHA256 mismatch; no extraction.'
    }
    Expand-Archive -LiteralPath $CodeZip -DestinationPath (Join-Path $HomeDir 'code')
}
if (!(Test-Path -LiteralPath (Join-Path $Repo 'remote_tools\Start-MMFiP3-3090.ps1'))) {
    throw 'P3 launcher is missing from the pinned source.'
}
$Repo
```

The `datasets` directory may be absent or empty. The verified extractor
binds it to the archive; a nonempty unbound directory is rejected rather
than overwritten. This pinned source snapshot is not a Git checkout; use a
new directory for a later source version.

## 2. Prepare a Python environment

An existing Windows Python 3.11 environment with CUDA PyTorch, NumPy, SciPy and
PyYAML can be supplied to the launcher with `-Python`. The commands below
install Python 3.11.9 for the current user and create a virtual environment.

```powershell
$Installer = Join-Path $HomeDir 'downloads\python-3.11.9-amd64.exe'
$InstallDir = Join-Path $HomeDir 'python311'
$ExpectedInstaller = '5EE42C4EEE1E6B4464BB23722F90B45303F79442DF63083F05322F1785F5FDDE'
if (!(Test-Path -LiteralPath (Join-Path $InstallDir 'python.exe'))) {
    Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $Installer -TimeoutSec 120
    if ((Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash -ne $ExpectedInstaller) { throw 'Python installer SHA256 mismatch.' }
    $Signature = Get-AuthenticodeSignature -LiteralPath $Installer
    if ($Signature.Status -ne 'Valid' -or $Signature.SignerCertificate.Subject -notlike '*Python Software Foundation*') { throw 'Python installer signature invalid.' }
    $Process = Start-Process -FilePath $Installer -ArgumentList @('/quiet','InstallAllUsers=0',"TargetDir=$InstallDir",'Include_pip=1','Include_launcher=0','Include_test=0','PrependPath=0') -WindowStyle Hidden -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "Python installer failed: $($Process.ExitCode)" }
}
$SystemPy = Join-Path $InstallDir 'python.exe'
if (!(Test-Path -LiteralPath $SystemPy)) { throw 'Python 3.11 executable missing.' }
& $SystemPy --version
$Venv = Join-Path $HomeDir 'envs\mmfi-p3'
& $SystemPy -m venv $Venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }
$Py = Join-Path $Venv 'Scripts\python.exe'
& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
& $Py -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch install failed.' }
& $Py -m pip install numpy==2.4.6 scipy==1.17.1 PyYAML==6.0.3 gdown
if ($LASTEXITCODE -ne 0) { throw 'Scientific dependencies install failed.' }
nvidia-smi
& $Py -c 'import torch,numpy,scipy,yaml; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'
if ($LASTEXITCODE -ne 0) { throw 'Python/CUDA import check failed.' }
```

The torch 2.6.0 CUDA 12.4 wheel for Windows is listed in the
[official PyTorch version guide](https://docs.pytorch.org/get-started/previous-versions/).
The environment check must print `True` for CUDA before GPU smoke.

## 3. Download the processed MM-Fi archive

The [Drive file supplied for MM-Fi](https://drive.google.com/file/d/1GXgLXzOhzdqyZpB0zDNqbhpzkuyA4TVn/view)
contains the processed CSI/pose data. It is **not** in Git. Resume a partial
download with gdown:

```powershell
$Prep = Join-Path $Repo 'remote_tools\Prepare-MMFiP3Data.py'
& $Py $Prep --data-home $HomeDir --mode plan
& $Py $Prep --data-home $HomeDir --mode list --download --tools-python $Py --resume
if ($LASTEXITCODE -ne 0) { throw 'MM-Fi download/archive inspection failed.' }
& $Py $Prep --data-home $HomeDir --mode prepare
if ($LASTEXITCODE -ne 0) { throw 'Extraction/validation failed; do not train.' }
```

The list gate verifies every archive member stays under `Compress/`. The
prepare gate extracts with size/CRC checks and validates representative CSI
and pose arrays. If extraction was interrupted, rerun the **prepare** command
with `--resume`. Resume requires the same bound archive and never overwrites
different existing content. Do not use `Expand-Archive` for the 322,000-file ZIP.
The runner expects `$HomeDir\datasets\Compress\E01..E04\S##\A##`.

## 4. Supply the original motion prior

The 43 MB handoff ZIP `mmfi_p3_3090_s42_20261025_v2.zip` contains the
original `inputs/data_bend.npy` (about 227 MB when extracted). Git does not
contain this input. From **Admin's PowerShell**, copy the ZIP to RTX-3090
(SSH asks for the `ssh3090` password):

```powershell
scp "E:\wifi\deliverables\mmfi_p3_3090_s42_20261025_v2.zip" "RTX-3090:backdooranalog/downloads/"
```

Then continue **on RTX-3090**. If SCP cannot resolve the relative Windows
destination, upload that ZIP to `$HomeDir\downloads` through VS Code Explorer.
Expected ZIP SHA256:
`57AAE82DED8318C0A72B4E640F80328DC4381BCE609105D49F5642CF8F9DA037`.

```powershell
$Bundle = Join-Path $HomeDir 'downloads\mmfi_p3_3090_s42_20261025_v2.zip'
$Action = Join-Path $HomeDir 'actions\data_bend.npy'
$ExpectedBundle = '57AAE82DED8318C0A72B4E640F80328DC4381BCE609105D49F5642CF8F9DA037'
$ExpectedAction = '2e8fc11192cc3ee7ee3f5544cb3c5e19f771f7f7265dde93af8969af41a2be3e'
if (!(Test-Path -LiteralPath $Bundle -PathType Leaf)) { throw 'Prior bundle missing.' }
if ((Get-FileHash -LiteralPath $Bundle -Algorithm SHA256).Hash -ne $ExpectedBundle) {
    throw 'Prior bundle SHA256 mismatch.'
}
if (!(Test-Path -LiteralPath $Action)) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($Bundle)
    try {
        $entry = $zip.GetEntry('inputs/data_bend.npy')
        if (!$entry) { throw 'Motion prior entry missing.' }
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $Action)
    } finally { $zip.Dispose() }
}
if ((Get-FileHash -LiteralPath $Action -Algorithm SHA256).Hash -ne $ExpectedAction) {
    throw 'Motion prior SHA256 mismatch; do not run training.'
}
```

## 5. Run gates, inspect, then launch

```powershell
$S = Join-Path $Repo 'remote_tools\Start-MMFiP3-3090.ps1'
foreach ($Mode in @('DryRun','Preflight','GpuSmoke')) {
    powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode $Mode -JobsPerGpu 2
    if ($LASTEXITCODE -ne 0) { throw "$Mode failed; no full training launched." }
}
```

DryRun writes nothing and touches no dataset/GPU. Preflight checks all three
official P3 splits and the prior; GpuSmoke runs **two concurrent physical
batch-32** HPE-Li backward steps on `cuda:0`, with no full epoch. The
receipts in the new run directory must pass before Launch. Do not reduce the
batch size or switch to one job in the same output folder after a failed
two-job smoke; use a new output folder and a fresh one-job smoke.

Only after reviewing the gate output, explicitly launch the persistent task:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Launch -JobsPerGpu 2
```

The launcher uses a same-user Scheduled Task with S4U, which needs the right
to register and run a batch task on this Windows machine. If registration is
denied, no full run starts. After inspecting that no task is active, the
explicit foreground fallback is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode LaunchForeground -JobsPerGpu 2
```

Keep that PowerShell/SSH session open for the foreground fallback. Neither
mode automatically retries failures or removes old checkpoints. Check actual
cell logs instead of relying only on a task's `Running` state:

```powershell
$Run = Join-Path $HomeDir 'runs\mmfi_p3_hpeli_s42_v1'
Get-ScheduledTask -TaskName MMFI-P3-HPELi-S42 -ErrorAction SilentlyContinue |
    Select-Object TaskName,State
Get-ChildItem "$Run\controller.*.stdout.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    ForEach-Object { Get-Content $_.FullName -Tail 15 }
Get-ChildItem "$Run\p3_s*\*\console.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 2 |
    ForEach-Object { $_.FullName; Get-Content $_.FullName -Tail 8 }
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status -JobsPerGpu 2
```

Output in `$Run\p3_s*\{clean,a1}` includes epoch-49 checkpoints,
evaluation caches, `p3_evaluation.json` and `p3_per_sample.npz`. Report
S1/S2/S3 separately, with no across-seed SD/CI from this one seed.
