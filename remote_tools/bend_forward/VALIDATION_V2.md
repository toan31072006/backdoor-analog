# PiW forward Bend v2 — local verification, 2026-11-11

## Material Passport

User-requested correction of PiW trunk-flexion direction; A1/WiFiDoor HPE-Li,
seed42, MM-Fi/P1S1 + PiW one-person. Code / CPU-synthetic validation plus replay
of the three user-downloaded fixed poison TRAIN poses. No TEST tuning, CUDA
capacity, full real TRAIN v2 audit, model training or attack efficacy claimed.
Numerical checks VERIFIED; inferred anterior direction supported by these
TRAIN examples/user inspection, not independently annotated facing/anatomy.

## Change and preservation

- PiW default payload `bend_forward_piw_v2`: actual angle **-40d degrees** about
  the unchanged native hip axis `y5-y7`; declared forward also reverses.
  Hips5/7, knees9/11 and ankles12/13 remain exactly fixed. This is a fixed
  dataset-level convention, NOT a per-pose knee-sign heuristic.
- Display up is the clean torso vector orthogonalized against the hip axis,
  independent of forward sign. The corrected actual 3D targets were also
  plotted in the UNCHANGED v1 display frame to exclude a display-only fix.
- MM-Fi payload spec remains `bend_forward_mmfi_v1`, +40d degrees. Compared
  directly with code from published commit
  `98c12cb91062b7c8b211646f25eb2ee98ac1c3f9`: **exact array equality** on 50
  perturbed synthetic poses at all six doses (300 pose/dose evaluations).
- Explicit PiW v1 specification remains available for archived replay only.
  Runner profiles are v2 and refuse v1 output roots/plans/checkpoints instead
  of silently reinterpreting or migrating them. Helper defaults to RunVersion2.
- Native A1 trigger, motion prior, selection/pairing, seed, rho, eps, recipes,
  epochs and evaluation stay unchanged. No Clean retraining/weight import.
  Still exploratory, NOT complete T7; missing comparison controls not supplied.

## Gates

- **230 passed**, exit0, 231.55s. Final code gate:
  `test_bend_forward.py`, `test_mmfi_action_targets.py`,
  `test_piw_action_targets.py`, `test_action_targets_parallel.py`,
  `test_cross_reachable.py`. Do not add this count to overlapping v1 gates.
- Signed angles independently checked as MM-Fi +40d / PiW -40d; synthetic
  PiW anterior is declared independently as -y with knees flexed toward -y.
  No expected direction is inferred only from the operator's display vector.
- Archived +40d replay, unchanged MM-Fi spec/reference rotation, body-up
  non-inversion with tilted hips, orthonormal side basis / SE(3) equivariance,
  v1 root refusal before writes, immutable archived-plan refusal, profile-bound
  checkpoint refusal, native CSI/poison parity, all-bone/fixed-joint checks,
  full tiny-synthetic-TRAIN geometry/approval gate, CPU exact resume, actual
  HPE-Li FP32 B32 updates, native Cross/Nod regression and coexistence guards.
- PowerShell helper AST: PASS. `git diff --check`: PASS.
- Source preservation: BackTime blob
  `0f5e7b99cd0b07fc5f3be08066102d4f84f07600`; manuscript SHA256
  `C48C3A8C625CE7AB6C46D2FCBD4686C33D85AAE8492A3E29BE005F504F7FD045`.

Reproduce from `ATKBackd` with project scientific dependencies; use a fresh,
nonexistent D: temporary path (validation host E: is nearly full):

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$bendTestTemp = 'D:\bend_v2_test_' + [guid]::NewGuid().ToString('N')
if (Test-Path -LiteralPath $bendTestTemp) { throw 'Fresh test path required' }
python -B -m pytest tests/test_bend_forward.py tests/test_mmfi_action_targets.py tests/test_piw_action_targets.py tests/test_action_targets_parallel.py tests/test_cross_reachable.py -q -p no:cacheprovider --basetemp $bendTestTemp
```

Local Python3.13.14, Torch2.11.0+cpu, NumPy2.3.4, pytest9.2.0, PyYAML6.0.3,
SciPy1.18.1, Matplotlib3.11.2. Gate temporary directory:
`D:\codex_bend_v2_gate_a6763b0d73ce4fef9d2bee8aa45f1495`.
Not a claim that Van's CUDA environment matches; each real plan freezes it.

## Three-example real TRAIN replay (local only)

Input SHA256:
`7184f27bcd14034fc4c7e55559f1ab621fb845bb4118f40481f29a9d95c6b9f4`.
Three fixed first/middle/last selected poison TRAIN examples, not cherry-picked
TEST predictions. Archived v1 targets at doses0/.5/1 replay with **0 m** maximum
absolute difference. New v2 matches the previously inspected negative-angle
candidate with **0 m** difference at doses0/.2/.4/.5/.6/.8/1. Maximum bone-length
roundoff across these checks: **4.31e-16 m**; all lower-body coordinates exactly
unchanged. Vertical display bases remain exactly the same between v1 and v2.
The v2 preview and same-old-frame v1/v2 comparison were visually inspected.

Derived local artifacts (not uploaded with code):

```text
D:\codex_bend_forward_review_20261111_7281e325\validate_piw_bend_v2.py
D:\codex_bend_forward_review_20261111_7281e325\piw_bend_forward_v2.png
D:\codex_bend_forward_review_20261111_7281e325\piw_bend_forward_v2.pdf
D:\codex_bend_forward_review_20261111_7281e325\piw_bend_forward_v2.npz
D:\codex_bend_forward_review_20261111_7281e325\piw_bend_forward_v2.json
D:\codex_bend_forward_review_20261111_7281e325\piw_v1_v2_same_frame.png
```

Raw downloaded NPZ and prior v1 figures/diagnosis remain unchanged. Knee-side
residuals support the chosen direction in these examples; they do not establish
coordinate-system handedness, anatomical limits, balance, contact, action
recognition, population generalization or performance. No inferential statistics
or significance claim. No participant coordinates/images included in GitHub.

## Remaining real-run gate

Clone v2 SEPARATELY on U:; preserve v1 and the running WiFlow/BackTime checkout.
CheckOnly -> Geometry for BOTH datasets into NEW v2 roots -> human inspection
and approval -> Prepare full-input audit + real CUDA B32 smoke with the explicit
peer -> Launch. No new training has been performed locally or remotely by this
correction. Never resume a v1 checkpoint under v2 or bypass a mismatched hash.
