# Local validation — PiW Cross/Nod code, 2026-11-07

## Material Passport

- Scope: implementation correctness on CPU/synthetic fixtures, not scientific
  PiW results, true-data geometry validation, CUDA capacity or attack efficacy.
- Native historical data/model/trigger/training scripts and manuscript unchanged.
- No remote run, Git push or real Cross/Nod training authorized/performed.
- Final grouped gate: **212 passed**, exit0,193.83s, on local Python3.13.5 /
  Torch2.9.1+cpu / NumPy2.1.3. No CUDA or genuine PiW training in this gate.

Command (from `ATKBackd` in the development checkout):

```powershell
& 'C:\Users\Admin\anaconda3\python.exe' -B -m pytest tests/test_piw_action_targets.py tests/test_piw_a1_adapter.py tests/test_piw_v2.py tests/test_piw_hpeli_clean_runner.py tests/test_mmfi_action_targets.py -q -p no:cacheprovider --basetemp=E:\wifi\_piw_cross_nod_tests_20261107_c
```

The handoff ZIP contains runnable code, not raw data/checkpoints/test fixtures.
The test files remain in the development checkout for audit/reproduction.

## Coverage

- PiW14joint topology: dose0 identity, exact non-target preservation, bone
  lengths, Cross front/across-midline wrists, fixed shoulders, Nod neck/head
  rotation, arbitrary proper-coordinate rotation/translation and batching.
- Invalid dose/complex/nonfinite/degenerate geometry and changed specs refuse;
  no automatic pose repair, TEST-driven parameter choice or frame omission.
- Native PiWA1 poison identities/doses preserved; only labels changed. Native
  normalized amplitude injection exactly once; phase entries bitwise unchanged.
- Perfect, still and reversed response metric fixtures; clean-GT elbows(6,4)
  PCK scale, per-dose requested/predicted/DCE and exact saved-NPZ metric replay.
- Native PiW HPE-Li CPU batch32, synthetic input/new Cross or Nod labels,
  ordinary MPJPE backward/AdamW update: finite loss/parameters and changed weights.
- Tiny dropout regressor runs full200epochs; interrupted at97, explicit resume
  through199 reproduces uninterrupted weights bit-for-bit with optimizer/RNG.
  This does not prove bitwise CUDA reproducibility or full-data convergence.
- Full staged CLI fixture uses real PiW loader/normalization and new payloads,
  Geometry -> explicit Prepare -> serial training -> both-final199 gate ->
  six-dose inference -> NPZ replay -> PNG/PDF -> CSV -> immutable completed
  evaluation reuse. Input mutation refused. TEST counts/IDs, GPU/capacity and
  model/device are explicitly replaced ONLY in this tiny CPU test; production
  CLI has no fixture/bypass/subset/epoch override switches.
- Geometry and synthetic prediction plots prominently marked SYNTHETIC CPU
  FIXTURE; they are not exported as empirical PiW illustrations for the paper.
- Standard-library-only CheckOnly on spaced paths imports no NumPy/Torch/
  SciPy/Matplotlib and creates no output. PowerShell AST parsing and actual
  wrapper CheckOnly with spaced fixture DataHome passed.
- Geometry approval, own-lock/unrelated-output handling, code-bundle tamper,
  fixed-final checkpoint/config binding and low-output-capacity refusals tested.
- Native PiW adapter/v2/Clean tests and previously added MM-Fi action-target
  tests included in the final regression gate.

## Development findings retained

The first focused48tests passed. When the staged fixture was added, its
zero-padded synthetic frame IDs were correctly rejected by DT-Pose's native
next-frame rule (which emits unpadded integer suffixes). The fixture, not the
loader or official-list contract, was corrected. The first grouped pass had
210passed/1failed at that fixture guard; it is not counted as a clean final pass.

The subsequent tiny staged test was correctly stopped by the real2GiB output
space guard: local E: had only about600MiB free. No user data was deleted.
The tiny CPU fixture alone replaces that capacity check (it uses a few MiB),
while a separate test verifies production low-disk refusal. The staged test
and low-disk guard then passed together (2passed,88.75s). No real training or
CUDA-capacity claim is inferred from those fixture runs.

## Unverified until execution on Vân

Full27876/2586 data identities/content/geometry, actual forward convention,
GPU idle/4090 VRAM, real B32 smokes,200epoch action-specific convergence,
TEST scores and actual qualitative outputs. Prepare must audit real inputs
and discard real TRAIN smoke weights before the user's separate Launch.

No Clean-trained+trigger reference is supplied; no complete T7, multi-seed
uncertainty, target ranking, anatomical/contact validity or physical RF claim.

## Preservation checks

Native historical PiW/MM-Fi sources have no diff. The pre-existing MM-Fi code
ZIP, manuscript and compiled PDF SHA256 values remain unchanged:

```text
MMFiCrossNodTargets_s42_code_20261107.zip
0ac0735661d821e3435ba826e2fbd12cce3eeef8656a08e2c02c4b6a1eb1ff68
main.tex
af8ddeb11417f4f59ae435306e27f690291f2caea497ebb682e675d5f6a9b6a3
CVPR_27_Do_Dose_compiled.pdf
61e63400e49650a30d58bfce5469af39fa2397bcd04ee9aa2b96bbbb34b1e77e
```
