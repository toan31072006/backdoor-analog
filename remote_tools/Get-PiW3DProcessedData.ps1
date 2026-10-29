<# Download the exact processed PiW3D mirror; no training or shared-env install. #>
[CmdletBinding()]
param(
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$PythonExe = '',
    [string]$ZipPath = '',
    [string]$DatasetRoot = ''
)
$ErrorActionPreference = 'Stop'
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$PythonExe) { $PythonExe = (Get-Command python -ErrorAction Stop).Source }
if (!$ZipPath) { $ZipPath = Join-Path $DataHome 'downloads\piw3d_processed_mirror_1tr98LA.zip' }
if (!$DatasetRoot) { $DatasetRoot = Join-Path $DataHome 'datasets\Person-in-WiFi-3D_processed_v1' }
$ZipPath = [IO.Path]::GetFullPath($ZipPath)
$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
$TrainList = Join-Path $DatasetRoot 'train_data\train_data_list.txt'
$TestList = Join-Path $DatasetRoot 'test_data\test_data_list.txt'
if ((Test-Path -LiteralPath $TrainList -PathType Leaf) -and
        (Test-Path -LiteralPath $TestList -PathType Leaf)) {
    Write-Host "DATA_LISTS_FOUND_REQUIRES_PREFLIGHT: $DatasetRoot"
    Write-Host 'Existing data left unchanged. List existence is NOT a content audit.'
    return
}
if (Test-Path -LiteralPath $DatasetRoot) {
    throw 'Dataset target already exists but required lists are missing; inspect partial data. No overwrite/extraction.'
}
$ExpectedSHA = 'bdd348909f1ff7d67d5674c40637c24b14d75e6bd9274cc9b3d8cfc3f2a35cf1'
$ExpectedBytes = [long]8577781802
if (!(Test-Path -LiteralPath $ZipPath -PathType Leaf)) {
    [IO.Directory]::CreateDirectory((Split-Path -Parent $ZipPath)) | Out-Null
    $DownloadDrive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($ZipPath))
    $Part = $ZipPath + '.part'
    $HaveBytes = if (Test-Path -LiteralPath $Part -PathType Leaf) { (Get-Item -LiteralPath $Part).Length } else { 0 }
    if ($DownloadDrive.AvailableFreeSpace -lt ([math]::Max(0,$ExpectedBytes-$HaveBytes) + 4GB)) {
        throw 'Not enough free disk for the 8.58 GB ZIP plus safety margin. Choose -ZipPath on a larger drive.'
    }
    $ToolEnv = Join-Path $DataHome 'tools\gdown-backup-env'
    $Downloader = Join-Path $ToolEnv 'Scripts\python.exe'
    if (!(Test-Path -LiteralPath $Downloader -PathType Leaf)) {
        & $PythonExe -m venv $ToolEnv
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create isolated downloader environment.' }
    }
    & $Downloader -m pip install --disable-pip-version-check 'gdown==5.2.0'
    if ($LASTEXITCODE -ne 0) { throw 'Downloader installation failed; no training env changed.' }
    & $Downloader -m gdown --fuzzy 'https://drive.google.com/file/d/1tr98LAenpHUxhV3ibq91Qkjk6cxsftik/view' -O $Part --continue
    if ($LASTEXITCODE -ne 0) { throw 'Dataset download failed; check Drive sharing/quota. No training launched.' }
    if ((Get-Item -LiteralPath $Part).Length -ne $ExpectedBytes -or
            (Get-FileHash -LiteralPath $Part -Algorithm SHA256).Hash -ine $ExpectedSHA) {
        throw 'Dataset ZIP size/SHA256 mismatch; partial file retained, no extraction.'
    }
    if (Test-Path -LiteralPath $ZipPath) { throw 'Destination ZIP appeared; refusing overwrite.' }
    [IO.File]::Move($Part,$ZipPath)
}
if ((Get-Item -LiteralPath $ZipPath).Length -ne $ExpectedBytes -or
        (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash -ine $ExpectedSHA) {
    throw 'Dataset ZIP size/SHA256 mismatch; nothing extracted.'
}
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = [IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    $Names = @($Archive.Entries | ForEach-Object { $_.FullName.Replace('\','/') })
    if ('train_data/train_data_list.txt' -notin $Names -or 'test_data/test_data_list.txt' -notin $Names) {
        throw 'Processed archive lacks expected top-level train/test lists.'
    }
    $Prefix = $DatasetRoot.TrimEnd('\') + '\'
    foreach ($Name in $Names) {
        if ($Name.StartsWith('/') -or $Name.Contains(':') -or '..' -in $Name.Split('/')) {
            throw 'Unsafe archive member; no extraction.'
        }
        $Target = [IO.Path]::GetFullPath((Join-Path $DatasetRoot $Name))
        if (!$Target.StartsWith($Prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Archive member escapes dataset.' }
    }
    $ExtractBytes = [long]($Archive.Entries | Measure-Object -Property Length -Sum).Sum
} finally { $Archive.Dispose() }
$DataDrive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($DatasetRoot))
if ($DataDrive.AvailableFreeSpace -lt ($ExtractBytes + 5GB)) {
    throw "Not enough extraction disk: need archive expanded bytes ($ExtractBytes) plus 5 GiB for checkpoints. Keep ZIP; choose another -DatasetRoot."
}
[IO.Directory]::CreateDirectory((Split-Path -Parent $DatasetRoot)) | Out-Null
[IO.Compression.ZipFile]::ExtractToDirectory($ZipPath,$DatasetRoot)
if (!(Test-Path -LiteralPath $TrainList -PathType Leaf) -or !(Test-Path -LiteralPath $TestList -PathType Leaf)) {
    throw 'Extraction incomplete; preserve partial data for inspection.'
}
Write-Host "DATA_EXTRACTED_REQUIRES_PREFLIGHT: $DatasetRoot"
Write-Host 'Exact ZIP verified; full loader/count/content audit still runs in Prepare. No training launched.'
