# MM-Fi P3 on RTX 3090: MetaFi++ and WiFlow

This exploratory seed-42 extension trains **Clean and A1 from scratch** under
Protocol 3, Settings S3/S2/S1, for **MetaFi++ and WiFlow**. It creates 12 new
victims and never writes into the finished HPE-Li P3 run. Each backbone has one
sequential worker, so one RTX 3090 hosts at most two workers. No BadNets, SIG,
defense, or other attack is launched by this helper.

The source checkout must contain
`remote_tools/Start-MMFiP3TwoBackbones3090.ps1`. Reuse the prepared MM-Fi data,
`actions/data_bend.npy`, and `envs/mmfi-p3/Scripts/python.exe` already on the
3090. Use a **new checkout** for this source version; leave the old P3 source
and result directory unchanged.

In PowerShell on the 3090, set `$Repo` to that new checkout, then:

```powershell
$S = Join-Path $Repo 'remote_tools\Start-MMFiP3TwoBackbones3090.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Prepare
```

`Prepare` prints the 12-cell recipe, checks the P3 split and action prior on
CPU, and runs one MetaFi++ plus one WiFlow physical batch-32 backward worker
together on `cuda:0`. It **does not start full training**. If the smoke fails,
there is no automatic one-worker or smaller-batch fallback.

Start training only after `PREPARED_ONLY` appears:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode LaunchDetached
```

This launches a hidden controller with redirected logs and prints its PID.
Process survival after an SSH/VS Code disconnect depends on how the host
manages remote sessions, so reconnect shortly afterward and check both PID
and changing cell logs. The runner itself resumes verified final-epoch
checkpoints if the controller is interrupted; it does not retry automatically.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode Status
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*run_mmfi_p3_two_backbones_3090.py*'
} | Select-Object ProcessId,ParentProcessId,CommandLine
$R = "$env:USERPROFILE\backdooranalog\runs\mmfi_p3_metafi_wiflow_s42_3090_v1"
Get-ChildItem "$R\*\p3_*\*\console.log" -ErrorAction SilentlyContinue |
    ForEach-Object { "===== $($_.Directory.Parent.Parent.Name) $($_.Directory.Parent.Name) $($_.Directory.Name) ====="; Get-Content $_.FullName -Tail 3 }
```

When every cell has a final epoch-49 checkpoint and evaluation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $S -Mode CheckOnly
Get-Content "$R\p3_two_backbones_results.json" -Raw
```

The results are one-seed, split-specific evidence. MetaFi++ retains SGD
lr=0.01/physical batch32; WiFlow retains AdamW lr=0.0001/physical batch32
with effective batch64. Both use ordinary MPJPE, 50 epochs, fixed final49,
the same A1 target joints and six-dose evaluation as the completed HPE-Li P3
run. WiFlow's common loss and fixed learning rate are declared adaptations.
