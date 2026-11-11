# Bend WDDM operational repair — 2026-11-12

## Material Passport

User reports BackTime finished, Bend NOT trained, and requests first Bend
training now. Both real v2 TRAIN geometry previews were approved. This repair
changes only the operational guard and first-launch workflow; not the payload,
A1 poison/CSI, recipe, model, seed or metrics. No Van training or completion
claim is made locally. Historical v1/v2/BackTime artifacts are preserved.

Scope: CPU/synthetic tests and read-only NVIDIA/CIM integration on the Admin
host. Numerical/code checks VERIFIED; user-provided Van snapshots ANALYZED.
Not a CUDA-capacity, full-data training, efficacy or complete-T7 validation.

## Root causes and changes

The supplied Van WDDM snapshot lists known desktop applications as C+G in
addition to the compute query; the latest supplied snapshot has no GPU Python
process. Old guard rejects every non-worker/non-peer PID, hence fails before
Prepare/smoke/training. Finishing BackTime alone does not remove desktop PIDs.

- Reconcile GPU0 compute CSV and full NVIDIA XML process types with current
  CIM identities. Use their union, not a filter that silently drops unknown
  XML-only PIDs. Require current PID/name, full protected path family and
  graphics-capable G/C+G type under WDDM for explicitly audited desktop families.
- Unknown/Python C+G, compute-only C, missing/unreadable/mismatched identities,
  untrusted/UNC/traversal/ADS paths and unsupported XML fail closed. There is no
  blanket C+G exemption or name-only allowlist. Kernel System PID4 has an
  explicit, tightly identified WDDM bookkeeping exception, including when only
  the compute query reports it and creation time is not readable.
- Live Admin integration also exposed System Idle Process PID0: `--peer-pid 0`
  now correctly means NO peer, not that real OS process. Named-peer identity,
  foreign-process refusal, current worker/descendants and receipt binding stay.
- Policy is frozen into each v3 plan and classified desktop evidence into the
  Prepare receipt. `guard-check` is read-only: no CUDA allocation, output, plan,
  lock or log creation. No kills, driver changes, auto retry or fallback.
- `RunApproved -ApproveGeometry` checks exact approved operator source SHA,
  refuses existing output, then executes guard-check -> both path checks ->
  both TRAIN geometry audits -> both Prepare/B32 smokes -> MM-Fi then PiW.
  Any failure terminates the queue. This is a first launch, NOT Resume.

[NVIDIA's reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
documents XML output and compute/graphics process types. C+G includes a compute
context; permitting known desktop apps is a coexistence policy, not absence of
compute, exclusive GPU access, publisher/file attestation or VRAM assurance.
NVIDIA output schemas are not guaranteed backward-compatible; unknown schemas
must fail rather than be interpreted as an idle GPU.

## Gates

### Follow-up correction: actual CIM Name projection

The user's first published v3 GpuCheck still refused known desktop PIDs before
any training or output creation. Reproduction on Windows identified a missing
`Name` in `process_snapshot()`'s actual CIM `Select-Object` projection. The
classifier requires this field; earlier mocked snapshots already supplied it,
and the earlier live guard check had no GPU process records, hiding the defect.

- Add only `Name` to the read-only projection. No guard policy, model, payload,
  data or training recipe changes.
- Two regression tests (projection-contract and real Windows CIM) failed on
  the old code and pass after the fix. Actual local CIM now includes `Name`
  on 324/324 process rows; no known local desktop row lacked creation time.
- Focused post-fix gate: **147 passed, 6 deselected**, exit0, **28.25s**, on
  `test_bend_gpu_guard.py` and `test_bend_forward.py`. Heavy CPU optimizer,
  B32-update and full tiny-geometry tests were not rerun for this metadata-only
  change. Root: `D:\bend_name_fix_b81bd83dcf2c4d2ab0a1c6fd60784326`.
- The 320-test gate below predates this correction, not a new full-suite gate.
  Actual Van GPU classification and training remain unverified locally.

### Original v3 validation, before Name correction

- **320 passed**, exit0, **181.97s**, final gate on:
  `test_bend_gpu_guard.py`, `test_bend_forward.py`,
  `test_mmfi_action_targets.py`, `test_piw_action_targets.py`,
  `test_action_targets_parallel.py`, `test_cross_reachable.py`.
  Includes actual PowerShell fake-worker tests of all stage ordering, explicit
  approval, existing-output refusal, failure-stop/no retry; fixtures perform no
  training or GPU allocation. Prior 150-test intermediate gate overlaps and is
  NOT added to this count.
- Windows path/type fixtures cover every allowed executable family, invalid
  identity/type/schema cases, System4 and Idle0 cases, XML-only unknown PIDs,
  read-only guard mode and plan policy binding. Existing math, data/poison,
  metrics, exact CPU resume, real HPE-Li FP32 B32 updates, peer safeguards and
  native Cross/Nod regressions remain in the same final gate.
- Live Admin read-only guard: **GPU_GUARD_PASS**, WDDM, peer null, zero GPU
  process records, output absent. Admin hardware is RTX3050 Laptop / driver
  512.74, NOT Van's RTX4090 / 591.86. No claim that Van's new XML guard or CUDA
  smoke has run. SSH remains denied in noninteractive mode; operator executes
  the documented commands on Van.
- PowerShell AST parse and `git diff --check`: PASS.

Local validation environment is unchanged from v2 (Python3.13.14,
Torch2.11.0+cpu, NumPy2.3.4, pytest9.2.0, PyYAML6.0.3, SciPy1.18.1,
Matplotlib3.11.2). Temporary artifacts are on D:, not nearly-full E:.
Final gate root: `D:\bend_wddm_final_f6659ffce9ce4cbfafa641e5da458137`.

Reproduce from ATKBackd using a fresh nonexistent basetemp:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$bendTestTemp = 'D:\bend_wddm_test_' + [guid]::NewGuid().ToString('N')
if (Test-Path -LiteralPath $bendTestTemp) { throw 'Fresh test path required' }
python -B -m pytest tests/test_bend_gpu_guard.py tests/test_bend_forward.py tests/test_mmfi_action_targets.py tests/test_piw_action_targets.py tests/test_action_targets_parallel.py tests/test_cross_reachable.py -q -p no:cacheprovider --basetemp $bendTestTemp
```

## Preservation and remaining gate

- Approved payload file SHA256:
  `1855A58C93D46A09D501937BA4068D4412051F0D484EEFAF190E0ABC7C95B6E8`.
  Rotation-helper source SHA256:
  `FC4687F823240A9E2D0261F228A982B07A43B32D31E4565A017180A102AEFCA7`.
  These unchanged files are checked before carrying v2 human approval forward.
- BackTime source blob remains
  `0f5e7b99cd0b07fc5f3be08066102d4f84f07600`. Manuscript remains SHA256
  `C48C3A8C625CE7AB6C46D2FCBD4686C33D85AAE8492A3E29BE005F504F7FD045`.
- No native trigger, poison, pose operator, training/evaluation runner or recipe
  source changed. No raw pose, participant data or machine snapshot uploaded.
- New frozen checkout/output v3 only. Full TRAIN geometry/input hashing and
  actual Van B32 CUDA smoke still run before training. No Clean/BackTime retrain,
  no old model weight imports. Missing controls remain NOT_SUPPLIED; exploratory
  new target extension, not a completed comparison/T7 table or an efficacy claim.
