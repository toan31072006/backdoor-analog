<#
.SYNOPSIS
SSH-independent continuation of the EXISTING four-attack PiW MetaFi3090 run.
.DESCRIPTION
Windows Task Scheduler, current account S4U/Limited, no stored password.
No controller/checkpoint/gate changes, automatic retries, boot triggers or kills.
The scheduled action waits for the foreground controller; it is not a detached
SSH child. Run mode is internal and requires an intact request from Launch.
#>
[CmdletBinding()]
param(
    [ValidateSet('Launch','Status','Run')][string]$Mode = 'Status',
    [string]$Repo = '',
    [string]$DataHome = (Join-Path $env:USERPROFILE 'backdooranalog'),
    [string]$PythonExe = (Join-Path $env:USERPROFILE 'eclare-env\Scripts\python.exe'),
    [string]$Outdir = '',
    [string]$RequestFile = '',
    [string]$RequestSHA = ''
)
$ErrorActionPreference = 'Stop'
$ControllerSHA = '1bb43a9e73a16c74e5c57972bebc2bce7b948cc52c8c19c5599df1b8ef7b80c2'
$HelperPath = $PSCommandPath

function Resolve-LocalPath([string]$Value) {
    if (!$Value -or $Value.StartsWith('\\') -or $Value -match '["\r\n\x00]') {
        throw 'Use an explicit local path without quotes/newlines; UNC/network paths are unsupported.'
    }
    $Resolved = [IO.Path]::GetFullPath($Value)
    $Drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($Resolved))
    if ($Drive.DriveType -eq [IO.DriveType]::Network) { throw 'S4U cannot use a mapped network drive.' }
    return $Resolved
}

function Get-SHA([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Quote-WindowsArgument([string]$Value) {
    if ($Value -match '[\r\n\x00]') { throw 'Unsupported native argument' }
    $Escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $Escaped = [regex]::Replace($Escaped, '(\\+)$', '$1$1')
    return '"' + $Escaped + '"'
}

function Get-TextSHA([string]$Value) {
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($Hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value)))).Replace('-','').ToLowerInvariant() }
    finally { $Hasher.Dispose() }
}

function Get-ControllerArguments([hashtable]$Request,[string]$Phase) {
    if ($Phase -notin @('check','launch')) { throw 'Only gate check or bound four-attack continuation is permitted.' }
    return @('-u',$Request.Controller,'--data-home',$Request.DataHome,
        '--dataset',$Request.Dataset,'--action',$Request.Action,'--out',$Request.Outdir,
        '--checkpoint-home',$Request.CheckpointHome,'--four-attacks-only','--mode',$Phase)
}

function Invoke-LoggedProcess([hashtable]$Request,[string[]]$Arguments,[string]$LogPrefix) {
    # Drain BOTH pipes asynchronously and wait for Python. The Task Scheduler
    # action stays Running until controller exit; its exact exit code is kept.
    $Info = [Diagnostics.ProcessStartInfo]::new()
    $Info.FileName = $Request.PythonExe
    $Info.Arguments = (@($Arguments | ForEach-Object { Quote-WindowsArgument $_ }) -join ' ')
    $Info.WorkingDirectory = $Request.Repo
    $Info.UseShellExecute = $false
    $Info.CreateNoWindow = $true
    $Info.RedirectStandardOutput = $true
    $Info.RedirectStandardError = $true
    foreach ($Pair in @{PYTHONUTF8='1';PYTHONDONTWRITEBYTECODE='1';OMP_NUM_THREADS='2';
                       MKL_NUM_THREADS='2';MKL_THREADING_LAYER='SEQUENTIAL'}.GetEnumerator()) {
        $Info.EnvironmentVariables[$Pair.Key] = $Pair.Value
    }
    foreach ($Name in @('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','TORCH_HOME')) {
        $Value = $Request.Environment[$Name]
        if ($null -eq $Value -or $Value -eq '') { $Info.EnvironmentVariables.Remove($Name) }
        else { $Info.EnvironmentVariables[$Name] = $Value }
    }
    $OutStream = $null; $ErrStream = $null; $Process = $null
    try {
        # Buffer size1 prevents console progress being held in a file buffer.
        $OutStream = [IO.FileStream]::new("$LogPrefix.stdout.log",[IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,[IO.FileShare]::Read,1,[IO.FileOptions]::Asynchronous)
        $ErrStream = [IO.FileStream]::new("$LogPrefix.stderr.log",[IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,[IO.FileShare]::Read,1,[IO.FileOptions]::Asynchronous)
        $Process = [Diagnostics.Process]::new()
        $Process.StartInfo = $Info
        if (!$Process.Start()) { throw 'Python did not start.' }
        $OutCopy = $Process.StandardOutput.BaseStream.CopyToAsync($OutStream)
        $ErrCopy = $Process.StandardError.BaseStream.CopyToAsync($ErrStream)
        $Process.WaitForExit()
        [Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($OutCopy,$ErrCopy))
        $Code = $Process.ExitCode
        $OutStream.Flush(); $ErrStream.Flush()
        return [int]$Code
    } finally {
        if ($null -ne $OutStream) { $OutStream.Dispose() }
        if ($null -ne $ErrStream) { $ErrStream.Dispose() }
        if ($null -ne $Process) { $Process.Dispose() }
    }
}

function Invoke-LoggedController([hashtable]$Request,[string]$Phase,[string]$LogPrefix) {
    $Arguments = Get-ControllerArguments $Request $Phase
    return Invoke-LoggedProcess $Request $Arguments $LogPrefix
}

function Get-CurrentSID {
    return [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
}

function ConvertTo-UserSID([string]$User) {
    if (!$User) { throw 'Scheduled task has no identifiable account.' }
    if ($User -match '^S-1-') { return [Security.Principal.SecurityIdentifier]::new($User).Value }
    return [Security.Principal.NTAccount]::new($User).Translate([Security.Principal.SecurityIdentifier]).Value
}

function ConvertTo-RequestJSON([hashtable]$Request) {
    $Ordered = [ordered]@{}
    [string[]]$Keys = @($Request.Keys)
    [Array]::Sort($Keys,[StringComparer]::Ordinal)
    foreach ($Name in $Keys) {
        if ($Name -eq 'Environment') {
            $Values = [ordered]@{}
            foreach ($Key in @('CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','TORCH_HOME')) {
                $Values[$Key] = $Request.Environment[$Key]
            }
            $Ordered[$Name] = $Values
        } else { $Ordered[$Name] = $Request[$Name] }
    }
    return ($Ordered | ConvertTo-Json -Depth 8 -Compress)
}

function Assert-Request([hashtable]$Request) {
    if ($Request.Schema -ne 1 -or $Request.Profile -ne 'piw_metafi_four_attacks_3090_task_v1' -or
        $Request.UserSID -ne (Get-CurrentSID) -or $Request.ControllerSHA -ne $ControllerSHA) {
        throw 'Wrong request/profile/account/controller; no training launched.'
    }
    foreach ($Name in @('Repo','DataHome','PythonExe','Outdir','Controller','Dataset','Action',
                         'CheckpointHome','Helper','TorchCache')) {
        if ((Resolve-LocalPath $Request[$Name]) -ne $Request[$Name]) { throw "Noncanonical path: $Name" }
    }
    foreach ($Name in @('PythonExe','Controller','Action','Helper')) {
        if (!(Test-Path -LiteralPath $Request[$Name] -PathType Leaf)) { throw "Missing $Name : $($Request[$Name])" }
    }
    foreach ($Name in @('Repo','DataHome','Outdir','Dataset','CheckpointHome')) {
        if (!(Test-Path -LiteralPath $Request[$Name] -PathType Container)) { throw "Missing $Name : $($Request[$Name])" }
    }
    if ($Request.Helper -ne $HelperPath -or (Get-SHA $Request.Helper) -ne $Request.HelperSHA -or
        (Get-SHA $Request.Controller) -ne $ControllerSHA -or
        $Request.Controller -ne (Join-Path $Request.Repo 'remote_tools\piw_metafi_remaining_3090\controller.py')) {
        throw 'Helper or existing bound controller changed. Do not replace gates/checkpoints.'
    }
    $ExpectedOut = Join-Path $Request.DataHome 'runs\piw_metafi_four_attacks_s42_3090_v1'
    if ($Request.Outdir -ne $ExpectedOut -or
        $Request.Dataset -ne (Join-Path $Request.DataHome 'datasets\Person-in-WiFi-3D_processed_v1') -or
        $Request.Action -ne (Join-Path $Request.DataHome 'actions\data_bend.npy') -or
        $Request.CheckpointHome -ne (Join-Path $Request.DataHome 'imports\piw_metafi_adamw_unfinished')) {
        throw 'This helper resumes only the existing four-attack run; no redirected outputs/datasets.'
    }
    $Weights = Join-Path $Request.TorchCache 'hub\checkpoints\resnet34-b627a593.pth'
    if (!(Test-Path -LiteralPath $Weights -PathType Leaf) -or !(Get-SHA $Weights).StartsWith('b627a593')) {
        throw 'Verified local ResNet34 cache required before S4U; no download is attempted.'
    }
}

function Get-TaskSpec([hashtable]$Request,[string]$File,[string]$FileSHA) {
    $TaskName = 'PiWMetaFi4-' + (Get-TextSHA ($Request.Outdir.ToLowerInvariant())).Substring(0,12)
    $ActionArguments = @('-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
        '-File',$Request.Helper,'-Mode','Run','-RequestFile',$File,'-RequestSHA',$FileSHA)
    return @{Name=$TaskName;UserSID=$Request.UserSID;WorkingDirectory=$Request.Repo;
        Execute=(Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe');
        Arguments=(@($ActionArguments | ForEach-Object { Quote-WindowsArgument $_ }) -join ' ');
        Description=('PiWMetaFi4 S4U continuation; request SHA256='+$FileSHA)}
}

function New-TaskDefinition([hashtable]$Spec) {
    $Action = New-ScheduledTaskAction -Execute $Spec.Execute -Argument $Spec.Arguments -WorkingDirectory $Spec.WorkingDirectory
    $Principal = New-ScheduledTaskPrincipal -UserId $Spec.UserSID -LogonType S4U -RunLevel Limited
    $Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    # Explicit defaults: no hidden72h timeout, idle restriction, network requirement,
    # restart-on-failure or boot/logon trigger. Task is started ONLY on demand.
    $Settings.RestartCount = 0
    return New-ScheduledTask -Action $Action -Principal $Principal -Settings $Settings -Description $Spec.Description
}

function Assert-TaskDefinition($Task,[hashtable]$Spec) {
    $Actions = @($Task.Actions)
    if ($Task.TaskName -ne $Spec.Name -or $Task.TaskPath -ne '\' -or
        $Actions.Count -ne 1 -or $Actions[0].Execute -ne $Spec.Execute -or
        $Actions[0].Arguments -ne $Spec.Arguments -or $Actions[0].WorkingDirectory -ne $Spec.WorkingDirectory -or
        (ConvertTo-UserSID $Task.Principal.UserId) -ne $Spec.UserSID -or [string]$Task.Principal.LogonType -notin @('S4U','2') -or
        [string]$Task.Principal.RunLevel -notin @('Limited','LeastPrivilege','0') -or
        $Task.Description -ne $Spec.Description -or @($Task.Triggers | Where-Object { $null -ne $_ }).Count -ne 0 -or
        $Task.Settings.ExecutionTimeLimit -ne 'PT0S' -or
        [string]$Task.Settings.MultipleInstances -notin @('IgnoreNew','2') -or $Task.Settings.RestartCount -ne 0 -or
        $Task.Settings.DisallowStartIfOnBatteries -or $Task.Settings.StopIfGoingOnBatteries -or
        $Task.Settings.RunOnlyIfIdle -or $Task.Settings.RunOnlyIfNetworkAvailable -or
        !$Task.Settings.AllowDemandStart -or !$Task.Settings.Enabled) {
        throw 'Existing scheduled task differs; no overwrite, stop or replacement was performed.'
    }
}

function Find-OwnedTask([string]$Name) {
    try { return Get-ScheduledTask -TaskName $Name -TaskPath '\' -ErrorAction Stop }
    catch {
        if ($_.FullyQualifiedErrorId -like 'CmdletizationQuery_NotFound*' -and
            $_.CategoryInfo.Category -eq 'ObjectNotFound') { return $null }
        throw  # An access/service/CIM failure must not mean "task absent".
    }
}

function Show-TaskStatus([string]$Name,[string]$Root) {
    $Task = Find-OwnedTask $Name
    if ($null -eq $Task) { Write-Host "TASK_NOT_REGISTERED $Name" }
    else {
        $Task | Select-Object TaskName,State
        Get-ScheduledTaskInfo -TaskName $Name -TaskPath '\' -ErrorAction Stop |
            Select-Object LastRunTime,LastTaskResult,NextRunTime
    }
    Get-ChildItem -LiteralPath $Root -File -Filter 'controller.task.*.log' -ErrorAction Stop |
        Sort-Object LastWriteTime -Descending | Select-Object -First 2 |
        ForEach-Object { Write-Host $_.FullName; Get-Content -LiteralPath $_.FullName -Tail 15 -Encoding UTF8 }
    Write-Host 'Running/LastTaskResult0 is not training completion. Confirm changing worker epochs and ALL_FOUR_TRAINED_NOT_EVALUATED.'
}

try {
    if ($Mode -eq 'Run') {
        if ($RequestSHA -notmatch '^[0-9a-f]{64}$' -or !(Test-Path -LiteralPath $RequestFile -PathType Leaf) -or
            (Get-SHA $RequestFile) -ne $RequestSHA) { throw 'Internal Run requires an intact Launch request.' }
        $Object = [IO.File]::ReadAllText($RequestFile) | ConvertFrom-Json
        $Request = @{}
        foreach ($Property in $Object.PSObject.Properties) { $Request[$Property.Name] = $Property.Value }
        $Request.Environment = @{}
        foreach ($Property in $Object.Environment.PSObject.Properties) { $Request.Environment[$Property.Name] = $Property.Value }
        # A failed scheduled-context preflight needs a persistent wrapper log too.
        if ((Resolve-LocalPath $Request.Outdir) -ne
            (Join-Path (Resolve-LocalPath $Request.DataHome) 'runs\piw_metafi_four_attacks_s42_3090_v1')) {
            throw 'Invalid scheduled output path.'
        }
        $Prefix = Join-Path $Request.Outdir ('controller.task.' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '.' + [guid]::NewGuid().ToString('N').Substring(0,8))
        Assert-Request $Request
        $Probe = 'import torch; assert torch.cuda.is_available(), "CUDA unavailable in scheduled context"; x=torch.ones(1,device="cuda:0"); torch.cuda.synchronize(); print("SCHEDULED_CUDA_ALLOCATION_PASS",torch.cuda.get_device_name(0))'
        $Code = Invoke-LoggedProcess $Request @('-u','-c',$Probe) "$Prefix.cuda"
        if ($Code -ne 0) { throw "Scheduled CUDA probe failed ($Code); inspect $Prefix.cuda.stderr.log" }
        $Code = Invoke-LoggedController $Request 'check' "$Prefix.check"
        if ($Code -ne 0) { throw "Scheduled gate check failed ($Code); inspect $Prefix.check.stderr.log" }
        # BLOCK here until all four serial workers exit. No SSH-owned detached child.
        $Code = Invoke-LoggedController $Request 'launch' $Prefix
        [IO.File]::WriteAllText("$Prefix.exit.json",(@{exit_code=$Code;request_sha256=$RequestSHA;
            ended_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
        exit $Code
    }

    $DataHome = Resolve-LocalPath $DataHome
    if (!$Outdir) { $Outdir = Join-Path $DataHome 'runs\piw_metafi_four_attacks_s42_3090_v1' }
    $Outdir = Resolve-LocalPath $Outdir
    $TaskName = 'PiWMetaFi4-' + (Get-TextSHA ($Outdir.ToLowerInvariant())).Substring(0,12)
    if ($Mode -eq 'Status') { Show-TaskStatus $TaskName $Outdir; return }
    if (!$Repo) { $Repo = Split-Path -Parent $PSScriptRoot }
    $Repo = Resolve-LocalPath $Repo
    $Request = @{Schema=1;Profile='piw_metafi_four_attacks_3090_task_v1';UserSID=(Get-CurrentSID);
        Repo=$Repo;DataHome=$DataHome;Outdir=$Outdir;PythonExe=(Resolve-LocalPath $PythonExe);
        Controller=(Join-Path $Repo 'remote_tools\piw_metafi_remaining_3090\controller.py');ControllerSHA=$ControllerSHA;
        Dataset=(Join-Path $DataHome 'datasets\Person-in-WiFi-3D_processed_v1');Action=(Join-Path $DataHome 'actions\data_bend.npy');
        CheckpointHome=(Join-Path $DataHome 'imports\piw_metafi_adamw_unfinished');
        Helper=(Resolve-LocalPath $HelperPath);HelperSHA=(Get-SHA $HelperPath);
        TorchCache=(Join-Path $DataHome 'torch-cache');Environment=@{}}
    foreach ($Name in @('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','TORCH_HOME')) {
        $Request.Environment[$Name] = [Environment]::GetEnvironmentVariable($Name,'Process')
    }
    if ($Request.Environment.TORCH_HOME) { $Request.TorchCache = Resolve-LocalPath $Request.Environment.TORCH_HOME }
    Assert-Request $Request
    $RequestDir = Join-Path $Outdir 'windows_task'
    [IO.Directory]::CreateDirectory($RequestDir) | Out-Null
    # ConvertTo-Json ordered-byte representation is frozen in a content-addressed file.
    $JSON = ConvertTo-RequestJSON $Request
    $Hash = Get-TextSHA $JSON
    $File = Join-Path $RequestDir "request.$Hash.json"
    if (!(Test-Path -LiteralPath $File)) {
        $Stream = [IO.File]::Open($File,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        try { $Bytes=[Text.Encoding]::UTF8.GetBytes($JSON); $Stream.Write($Bytes,0,$Bytes.Length) }
        finally { $Stream.Dispose() }
    }
    if ((Get-SHA $File) -ne $Hash) { throw 'Existing request differs; preserve artifacts.' }
    $Spec = Get-TaskSpec $Request $File $Hash
    $Existing = Find-OwnedTask $Spec.Name
    if ($null -ne $Existing) {
        Assert-TaskDefinition $Existing $Spec
        if ([string]$Existing.State -in @('Running','Queued','4','2')) {
            Write-Host "ALREADY_ACTIVE $($Spec.Name); no second launch performed."
            Show-TaskStatus $Spec.Name $Outdir
            return
        }
    }
    $Prefix = Join-Path $RequestDir ('prelaunch.'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')+'.'+[guid]::NewGuid().ToString('N').Substring(0,8))
    $Code = Invoke-LoggedController $Request 'check' $Prefix
    if ($Code -ne 0) {
        Get-Content -LiteralPath "$Prefix.stderr.log" -Tail 25 -Encoding UTF8
        throw "Existing controller gate check failed ($Code); no scheduled training started."
    }
    if ($null -eq $Existing) {
        $Definition = New-TaskDefinition $Spec
        # Never use Force: an unrelated task appearing in a race must not be overwritten.
        Register-ScheduledTask -TaskName $Spec.Name -TaskPath '\' -InputObject $Definition -ErrorAction Stop | Out-Null
    }
    $Registered = Get-ScheduledTask -TaskName $Spec.Name -TaskPath '\' -ErrorAction Stop
    Assert-TaskDefinition $Registered $Spec
    Start-ScheduledTask -TaskName $Spec.Name -TaskPath '\' -ErrorAction Stop
    Write-Host "TASK_START_REQUESTED $($Spec.Name); S4U own account, max one GPU worker."
    Write-Host "REQUEST=$File"
    Write-Host "LOGS=$Outdir\controller.task.*.stdout.log / stderr.log"
    Write-Host 'Confirm SCHEDULED_CUDA_ALLOCATION_PASS, gate check and changing train epochs before disconnecting SSH.'
    Write-Host 'No reboot/sleep protection or automatic restart. Training still stops on errors; no checkpoint or scientific source was changed.'
} catch {
    if ($Mode -eq 'Run' -and $Prefix -and (Test-Path -LiteralPath (Split-Path -Parent $Prefix) -PathType Container)) {
        [IO.File]::WriteAllText("$Prefix.wrapper-error.log",($_ | Out-String),[Text.UTF8Encoding]::new($false))
    }
    Write-Error ("Persistent3090 task failed: " + $_.Exception.Message +
        '. No automatic fallback, policy change or process kill. If Scheduler denies access/batch logon, ask the machine owner to grant this account the required task permission.') -ErrorAction Continue
    exit 1
}
