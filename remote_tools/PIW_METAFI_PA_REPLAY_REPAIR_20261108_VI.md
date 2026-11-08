# Recovery: one-ULP PA-MPJPE replay mismatch, 2026-11-08

## Material Passport

User's log confirms both victims reached final199; TSBA evaluation completed.
Smooth-LF inference and its Clean controls also finished all2586frames/six
doses, then exact JSON metric replay refused a difference. User ran the
read-only native diagnostic and reported ONLY:

```text
metrics.mainmetrics.clean_pampjpe_mm
saved 81.46046470956578
replay 81.4604647095658
delta 1.4210854715202004e-14 mm = one binary64 ULP
```

This is technical serialization/numerical validation, not a new model result
or statistical-significance claim. Local synthetic strided-array/NPZ replay
also reproduces PA-only roundoff; it does not inspect the real Van arrays.

## Scope and evidence preservation

`Repair-PiWMetaFiReplayPA.py` does NOT relax the checker or edit scientific /
launcher source (which would invalidate the source-hashed prelaunch gate).
It canonicalizes exactly ONE derived JSON scalar from the existing NPZ,
and only for the exact method/value pair above. Other metric/schema changes,
TSBA changes, foreign profiles, changed source, checkpoint/prediction/Clean
hashes, target/ID/dose schema mismatches or unstable CPU replays are refused.

Default mode is read-only. Explicit `--apply`:

1. Acquires the original controller's real OS output lock; never deletes a
   lock file, kills a process or bypasses another writer.
2. Checks plan digest, frozen scientific hashes, original orchestration hashes,
   Clean reference files, own checkpoint/NPZ hashes and full prediction schema.
3. Recomputes each method's metrics twice from fresh NPZ loads. TSBA must
   match exactly; Smooth may differ only at the confirmed PA one-ULP pair.
4. Saves the exact original Smooth evaluation JSON as
   `evaluation.before_pa_replay_roundoff_20261108_v1.json` and a hashed proposal
   BEFORE replacing the derived JSON. Backup/proposal are exclusive writes;
   existing different evidence is preserved and refused, not overwritten.
5. Atomically updates only `metrics.mainmetrics.clean_pampjpe_mm` to the native
   NPZ replay value. Other fields are semantically unchanged. Protected files
   are rehashed before/after; a separate applied audit records the operation.

Checkpoint, optimizer/RNG, trigger, data, NPZ arrays, TSBA JSON, Clean reference,
plan, source and prelaunch gate remain unchanged. No training, model inference,
CUDA workload, re-extraction, package installation or automatic retry occurs.
An interrupted correction retains backup/proposal for manual inspection; no
automatic rollback/history deletion. Only derived JSON formatting and that
one last-bit PA scalar change; both old/new PA round to81.46mm.

## Commands on Van

In the VS Code terminal connected to Van, update the existing checkout:

```powershell
Set-Location -LiteralPath 'C:\Users\Van\PiWMetaFiTwoExtra4090'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull failed. Preserve local changes; do not reset or Launch.' }
```

If you only need the already-saved evaluation numbers, the following is
read-only; it does not repair or rerun anything. Smooth's exact replay remains
unresolved until the recovery below succeeds, although both PA values round
to81.46mm:

```powershell
$piwResultRoot = 'U:\backdooranalog\runs\piw_metafi_two_extra_s42_4090_serial_v1\attacks\metafiplusplus\seed42'
foreach ($piwMethod in @('tsba_adapted', 'smooth_lf_csi')) {
    $piwReportPath = Join-Path (Join-Path $piwResultRoot $piwMethod) 'evaluation.json'
    $piwReport = Get-Content -LiteralPath $piwReportPath -Raw | ConvertFrom-Json
    Write-Output $piwMethod
    $piwReport.metrics.mainmetrics | Format-List
}
```

To repair the confirmed last-bit mismatch and finalize the combined reports:

```powershell
& 'C:\Users\Van\backdooranalog\envs\piw-metafi-vast211-4090\Scripts\python.exe' -B 'C:\Users\Van\PiWMetaFiTwoExtra4090\remote_tools\Repair-PiWMetaFiReplayPA.py' --run 'U:\backdooranalog\runs\piw_metafi_two_extra_s42_4090_serial_v1' --checkout 'C:\Users\Van\PiWMetaFiTwoExtra4090' --apply
if ($LASTEXITCODE -ne 0) { throw 'Repair refused. Preserve files and send the error; do not Launch.' }
powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\Van\PiWMetaFiTwoExtra4090\remote_tools\Start-PiWMetaFiTwoExtra4090.ps1' -Mode Evaluate -DataHome 'C:\Users\Van\backdooranalog' -Outdir 'U:\backdooranalog\runs\piw_metafi_two_extra_s42_4090_serial_v1'
```

The first command only patches the backed-up derived scalar. The SECOND uses
the unchanged original runtime/source/GPU/reference gates and strict metric
replay to finalize both cached evaluations and regenerate the report. It is
Evaluate, NOT Launch: with both valid cached evaluations, no new inference or
training is needed. Do not run the second command if the first refuses.

Only after the original evaluator succeeds is the combined run complete.
This repair has not been applied to Van from this workspace (SSH unavailable).
No new GPU jobs or previously requested MM-Fi queue were launched.

## Local validation

17passed, exit0,17.47s on local CPU/synthetic fixtures:

```powershell
& 'C:\Users\Admin\anaconda3\python.exe' -B -m pytest tests/test_piw_metafi_replay_pa_repair.py -q -p no:cacheprovider --basetemp=E:\wifi\_piw_pa_repair_tests_20261108_c
```

Coverage: exact user-provided one-ULP pair, read-only default, original-byte
backup/proposal, atomic replacement, only one scalar changed, unchanged strict
legacy replay checker rejects before/accepts after, protected hashes,
idempotence, real original OS lock refusal, additional/foreign/two-ULP/NaN/
signed-zero metric changes refused, source tamper refused, full-shape synthetic
NPZ schema and native strided-array serialization probe. No Van files modified
by these tests and no real model scores inferred from synthetic fixtures.
Original runner/PowerShell helper/orchestration/native metric sources have
empty git diffs. The original strict comparison remains unchanged.
