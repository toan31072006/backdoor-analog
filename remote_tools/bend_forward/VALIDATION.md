# Local verification — 2026-11-11

Material Passport: user-approved forward Bend implementation, A1/WiFiDoor,
HPE-Li, seed42, MM-Fi/P1S1 + PiW one-person. Code preparation / CPU-synthetic
correctness only. No remote full-data geometry, CUDA capacity, training or
efficacy is claimed. User approved publication after tests, not intervention in
the existing WiFlow process or reuse/relabeling of historical leg targets.

## Gates

- Broad earlier gate: **216 passed**, exit0, 242.48s. Files:
  `test_bend_forward.py`, `test_mmfi_action_targets.py`,
  `test_piw_action_targets.py`, `test_action_targets_parallel.py`,
  `test_cross_reachable.py`.
- Final-code gate after adding peer-identity/stage-routing tests and the
  post-input-audit GPU recheck: **144 passed**, exit0, 203.49s. Files:
  `test_bend_forward.py`, `test_mmfi_action_targets.py`,
  `test_piw_action_targets.py`. The gates overlap; **do not add their counts**.
- PowerShell AST parse of `Run-BendForward.ps1`: PASS. `git diff --check`: PASS.
- Synthetic PiW side-view preview visually inspected: upper body moves toward
  declared forward, lower body stays fixed. This is NOT real TRAIN inspection
  or evidence of anatomical/action validity.

CPU checks include all kinematic bones (also hip-to-hip diagnostic on PiW),
fixed non-targets, no input mutation, dose0 identity, signed40d angle, monotonic
per-joint displacement, SE(3) coordinate equivariance, batched inputs, malformed
data/spec/dose/degenerate-axis refusal; both native poison index/dose and CSI
operator parity, including PiW unchanged phase; full selected-poison geometry
audit/NPZ/PNG/PDF on tiny synthetic TRAIN fixtures, no CSI/TEST access during
geometry, explicit approval gate, TRAIN byte-change rejection; exact RNG /
optimizer CPU resume through final49 and final199; actual HPE-Li FP32 B32
forward/backward/optimizer updates for both new payload shapes; single-victim
launch/resume/evaluate routing, PID reuse refusal, foreign/unreadable GPU-PID
refusal, own/peer descendants, serial helper/no process mutation.

Reproduce (use a NEW, nonexistent basetemp; pytest owns that test directory):

```powershell
# From ATKBackd, in an environment with the project scientific dependencies.
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$bendTestTemp = Join-Path $env:TEMP ('bend_forward_test_' + [guid]::NewGuid().ToString('N'))
python -B -m pytest tests/test_bend_forward.py tests/test_mmfi_action_targets.py tests/test_piw_action_targets.py -q -p no:cacheprovider --basetemp $bendTestTemp
```

Local environment: Python3.13.14, Torch2.11.0+cpu, NumPy2.3.4, pytest9.2.0,
PyYAML6.0.3, SciPy1.18.1, Matplotlib3.11.2. These are validation-host versions,
not a claim that Van's CUDA environment matches; real runtime is frozen in each
new plan. Initial attempts were blocked by missing local dependencies / a
permission-denied default pytest temp directory / full E volume. Only this
task's newly created test-dependency folder was moved from E to D; test
dependencies were completed there and new temporary test outputs were put on D.
The task's new runner was restored after an interrupted full-disk write and
the final gate validated the restored file. No raw data or old runs were moved,
deleted, resumed or retrained.

## Preservation and remaining gates

- `attack/backtime_mmfi.py` unchanged, git blob
  `0f5e7b99cd0b07fc5f3be08066102d4f84f07600`.
- Manuscript `main.tex` unchanged, SHA256
  `C48C3A8C625CE7AB6C46D2FCBD4686C33D85AAE8492A3E29BE005F504F7FD045`.
- Four existing action-adapter/runner files receive explicit optional Bend
  dispatch/profile binding, with Cross/Nod defaults preserved; new source must
  be cloned SEPARATELY, never pulled into any active source-hashed checkout.
- On Van: CheckOnly -> full real TRAIN Geometry for BOTH datasets -> human
  inspection/approval -> full input hashing + actual CUDA B32 smoke alongside
  the explicit peer -> separate Launch. No capacity assurance from CPU tests.
- No Clean model training/import; historical Clean remains reusable only after
  its separate recipe/input/checkpoint audit. Missing Clean+trigger/baseline
  controls remain explicitly NOT_SUPPLIED, so this is not a complete T7 table.
