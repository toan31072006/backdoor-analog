<# Download the user's exact four-checkpoint backup; no training or Torch install. #>
[CmdletBinding()]
param(
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$PythonExe = '',
    [string]$ZipPath = ''
)
$ErrorActionPreference = 'Stop'
$DataHome = [IO.Path]::GetFullPath($DataHome)
if (!$PythonExe) { $PythonExe = (Get-Command python -ErrorAction Stop).Source }
if (!$ZipPath) { $ZipPath = Join-Path $DataHome 'downloads\vast_adamw_unfinished_20261028_all.zip' }
$ZipPath = [IO.Path]::GetFullPath($ZipPath)
$Destination = Join-Path $DataHome 'imports\piw_metafi_adamw_unfinished'
$ExpectedZipSHA = '4607f4e3524af9f0cfb1b719ad89e6fee0e23c1765a150508604726fe720d43d'
$ExpectedCheckpoints = [ordered]@{
    ftrojan='9d2a713315f694183cd9512e9feb74fe4f44bb823fedf455fafbf5edddd7d093'
    bpp_source='4ec9d9ff57fa578a42f368764b9c110d2376e155d6d578d5831676a5fd2a4c5f'
    fiba='590394ff220b1b99dcd32db6416948e40fa6b49fb4f8934f7dfd9f3bbda98389'
    smooth_lf_csi='f918001926d5716067d7d321360e1d51e495342cab543204f1019df4c03b6cb2'
}
if (!(Test-Path -LiteralPath $ZipPath -PathType Leaf)) {
    [IO.Directory]::CreateDirectory((Split-Path -Parent $ZipPath)) | Out-Null
    # Downloader gets its own small venv: never pip-install into eclare-env.
    $ToolEnv = Join-Path $DataHome 'tools\gdown-backup-env'
    $Downloader = Join-Path $ToolEnv 'Scripts\python.exe'
    if (!(Test-Path -LiteralPath $Downloader -PathType Leaf)) {
        & $PythonExe -m venv $ToolEnv
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create isolated downloader environment.' }
    }
    & $Downloader -m pip install --disable-pip-version-check 'gdown==5.2.0'
    if ($LASTEXITCODE -ne 0) { throw 'Downloader installation failed; no training environment changed.' }
    $Part = $ZipPath + '.part'
    & $Downloader -m gdown --fuzzy 'https://drive.google.com/file/d/1oBz3BrE2BwQRwKZiQw1N2liw0adTLiSb/view' -O $Part --continue
    if ($LASTEXITCODE -ne 0) { throw 'Drive download failed; check sharing permission. No training launched.' }
    if ((Get-FileHash -LiteralPath $Part -Algorithm SHA256).Hash -ine $ExpectedZipSHA) {
        throw 'Downloaded file is not the exact expected checkpoint ZIP; retained for inspection.'
    }
    if (Test-Path -LiteralPath $ZipPath) { throw 'Destination ZIP appeared; refusing overwrite.' }
    [IO.File]::Move($Part,$ZipPath)
}
if ((Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash -ine $ExpectedZipSHA) {
    throw 'ZIP SHA256 mismatch; nothing was extracted.'
}
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$ExpectedNames = @('download.ps1','inventory.json','README.md','sources.tar.gz',
    'provenance/placements.json','provenance/plan.json',
    'provenance/preflight_receipt.json','provenance/prelaunch_gate.json')
foreach ($Method in $ExpectedCheckpoints.Keys) { $ExpectedNames += "metafiplusplus/seed42/$Method/checkpoint.pt" }
$Archive = [IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    $Names = @($Archive.Entries | ForEach-Object { $_.FullName.Replace('\','/') })
    if ($Names.Count -ne 12 -or @($Names | Select-Object -Unique).Count -ne 12 -or
            @($Names | Where-Object { $_ -notin $ExpectedNames }).Count) {
        throw 'Unexpected/duplicate archive member; nothing extracted.'
    }
    foreach ($Entry in $Archive.Entries) {
        $Name = $Entry.FullName.Replace('\','/')
        $Target = [IO.Path]::GetFullPath((Join-Path $Destination $Name))
        $Prefix = [IO.Path]::GetFullPath($Destination).TrimEnd('\') + '\'
        if (!$Target.StartsWith($Prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe archive destination.' }
        $Stream = $Entry.Open()
        $Hasher = [Security.Cryptography.SHA256]::Create()
        try { $SHA = ([BitConverter]::ToString($Hasher.ComputeHash($Stream))).Replace('-','') }
        finally { $Hasher.Dispose(); $Stream.Dispose() }
        if (Test-Path -LiteralPath $Target) {
            if ((Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash -ine $SHA) {
                throw "Existing import differs; no overwrite: $Target"
            }
            continue
        }
        [IO.Directory]::CreateDirectory((Split-Path -Parent $Target)) | Out-Null
        [IO.Compression.ZipFileExtensions]::ExtractToFile($Entry,$Target,$false)
    }
} finally { $Archive.Dispose() }
foreach ($Method in $ExpectedCheckpoints.Keys) {
    $Path = Join-Path $Destination "metafiplusplus\seed42\$Method\checkpoint.pt"
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ine $ExpectedCheckpoints[$Method]) {
        throw "Imported checkpoint checksum mismatch: $Method"
    }
}
Write-Host "IMPORTED_AND_VERIFIED 4/4 checkpoints: $Destination"
Write-Host 'Smooth LF is stored only; -FourAttacksOnly will NOT schedule it. No training launched.'
