# HPE-Li / MM-Fi P3S3 / SIBA + BackTime / RTX3090

## Material Passport

- academic-research-suite / experiment-agent; 2026-11-03 Asia/Bangkok.
- User asks to move the two published adaptations from Van4090 to ssh3090.
- Exploratory T6 extension, seed42 only; not the completed multi-seed paper table.
- Execution-only adapter: `Run-MMFiP3S3SibaBackTime3090.py`.
- Original algorithm, source provenance and adaptation disclosures remain in
  `MMFI_P3S3_SIBA_LADDER_4090_VI.md` and scientific commit `2e9c28d`.
- No original scientific file, historical result or running Clean job is changed.

## Scope and cost

Queue: **Clean reference scratch50 -> SIBA search200/B128 + victim50/B32 ->
BackTime independent attacker100/B64 + victim50/B32**, sequential, one CUDA worker.
BackTime retains11 warmup +89 alternating epochs and per-recipient generator
updates. No compact bank, shorter search, batch shrink or improved objective.
SIBA and BackTime are attack methods; victim architecture remains HPE-Li.
LADDER stays excluded explicitly because author search material is missing.

Full MM-Fi P3S3 **240570 TRAIN / 80190 TEST**, seed42/rho.1/24057poison,
victim SGD lr.001/momentum.9/wd0,50epochs/final49/B32/workers0/FP32.
Same six doses, target joints[2,3], clean GT scale(5,12), Target-PCK@.2,
positive-dose DCE and clean MPJPE/PA controls as the published queue.
Different native trigger constraints are not equal-noise budgets; BackTime's
adaptive recipients differ from SIBA's uniform recipients, as already disclosed.

RTX3090 guard requires physical GPU0/24GiB class and no other compute or
pending experiment; Windows desktop graphics are handled by the unchanged
exclusive detector. Nothing is killed. No CUDA_VISIBLE_DEVICES remapping.
The native deterministic execution signature, physical UUID/driver and adapter
hash are bound into the new plan/gate. No 4090 checkpoint, optimizer or receipt
is imported; same scientific functions, but cross-GPU bitwise equality/ETA is
not promised. Original 4090 modules are loaded privately, not modified on disk
or globally patched in the historical import namespace.

## Handoff (PowerShell on ssh3090)

Use a **new checkout pinned to the verified published commit**, not an old
checkout currently training Clean. The adapter does not install an environment,
download data, update Git or start a persistent background service.

```powershell
$H = 'C:\Users\ssh3090\backdooranalog'
$Code = 'C:\Users\ssh3090\code_mmfi_siba_backtime_3090_20261103'
$P = 'C:\Users\ssh3090\eclare-env\Scripts\python.exe'
$E = "$Code\remote_tools\Run-MMFiP3S3SibaBackTime3090.py"
& $P -u $E --mode inspect --data-home $H
if ($LASTEXITCODE -ne 0) { throw 'Inspect loi; dung.' }
& $P -u $E --mode prepare --data-home $H
if ($LASTEXITCODE -ne 0) { throw 'Prepare loi; chua launch.' }
```

`prepare` audits full P3S3 and runs native real-TRAIN B32 CUDA2updates, then
stops. Learned-trigger capacity smokes remain after the respective full search,
before victim training. Do not call a CPU/mock test a 3090 capacity/efficacy pass.
If data root is partial, counts fail; do not bypass it or resample TEST.

After inspecting Prepare output, launch separately:

```powershell
& $P -u $E --mode launch --data-home $H
if ($LASTEXITCODE -ne 0) { throw 'Launch/run loi; giu log va checkpoint, khong tu retry.' }
```

**Foreground: keep SSH open.** No tmux on native Windows, Scheduler/WMI guarantee,
or automatic wait/restart. If Clean is running, do not run Prepare/Launch yet;
this queue refuses compute/pending conflicts instead of stopping that job.

Output: `$H\runs\mmfi_p3s3_siba_ladder_backtime_adaptive_source_adapted_siba_backtime_only_s42_3090_v2`.
Epoch checkpoints support explicit same-checkout/env/GPU resume via `launch`
after reading a failure; no automatic retry or stale-lock deletion.

From another SSH terminal:

```powershell
& $P -u $E --mode status --data-home $H
```

Status reads the latest logs and artifact presence, not proof of live training.
Confirm changing epoch/search lines. Final `comparison.md`, `results.json`
and per-method `evaluation.json` appear only after the scientific gates pass.

## Verification

Local validation on2026-11-03: **352 passed,10 warnings,exit0,83.98s**. Warnings
are CPU-only `pin_memory` notices.22 new execution-port tests plus330 unchanged
scientific/launcher regressions: exact recipe/operator preservation, private
module isolation, serial order, wrong GPU/remapping/scope rejection, plan hash,
Inspect/Status read-only, Prepare/Launch dispatch and existing source schedules.
Report: `E:/wifi/deliverables/siba_backtime_3090_tests_20261103_a.xml`.

No 3090 job, full-input audit, CUDA smoke, full search or scientific TEST result
has been run from the Admin workspace. Noninteractive SSH access to RTX-3090
was denied; the user must execute the supplied commands in their authenticated
session. Local tests are not remote capacity or efficacy measurements.
