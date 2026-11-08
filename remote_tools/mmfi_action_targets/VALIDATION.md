# Local validation — Cross/Nod target extension

## Material Passport

- Status: CODE_TESTED_ONLY; not full-data, CUDA or efficacy verification.
- Date: 2026-11-07. Source checkout: backtime_4090_setup_20261106.
- Base Git revision:55a27251cf02d9b0845f53b0091bdbc8782ebeed.
- Python local: C:/Users/Admin/anaconda3/python.exe; PyTorch2.9.1+cpu,
  NumPy2.1.3. No remote actions or real training were performed.

## Passed

67 local CPU/synthetic tests, exit0, in the combined group:

```
python -B -m pytest tests/test_mmfi_action_targets.py \
  tests/test_mmfi_coupling.py tests/test_mmfi_split.py \
  tests/test_mmfi_spawn.py -q -p no:cacheprovider --basetemp=NEW_TEST_DIRECTORY
```

MKL_THREADING_LAYER=SEQUENTIAL, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2 are set
before scientific imports, as in the existing Windows launchers.

Checks include:

- Both payloads preserve every bone length, dose0 identity, unchanged non-target
  joints, input arrays, batch shape and rigid-coordinate equivariance.
- Cross endpoint wrists are genuinely on opposite sides/front of torso in
  synthetic fixtures; both arms change. Nod rotates head40deg about neck.
- Degenerate geometry, invalid dose, complex poses and changed payload specs
  fail rather than silently dropping/repairing frames.
- Exact native uniform poison identities/doses, fixed recipe, ordinary supervised
  pair only; no poison metadata or action-specific weights reach the loss.
- Target evaluation scores the actual four Cross joints or one Nod joint,
  not the historical leg. Perfect, still and reversed prediction controls;
  NPZ roundtrip recomputes metrics exactly using fixed C-order float64 reductions.
- Ordinary native MPJPE/SGD loop runs50epochs on a tiny synthetic model/dataset;
  interrupt after24, restore model/optimizer/RNG, finish49, exact parameter parity.
  This is a unit test, not training HPE-Li on MM-Fi or proof of GPU replay.
- Real HPE-Li B32 CPU forward/backward/SGD update separately for Cross/Nod,
  each with8poison+24clean synthetic examples; finite output/update.
- CheckOnly starts a real standard-library-only Python subprocess with spaced
  paths, without NumPy/Torch imports, CUDA or output creation.
- Own-lock geometry planning, unrelated-output refusal, bundle tamper refusal;
  both geometry PNG render tests carry SYNTHETIC UNIT TEST labels.
- Existing coupling/split/spawn regressions remain passing.

PowerShell syntax parse passes. The real PowerShell wrapper CheckOnly also
passes with spaced fixture paths and explicit Python; it performs no inference.
Python AST parsing and git diff --check pass. Existing train_backdoor, poison,
payload, trigger, method_drafts, feeder and HPE-Li/SK sources are unchanged.
Manuscript main.tex and canonical compiled PDF hashes remain unchanged.

## Earlier failed test execution, not concealed

The first test process passed29 cases then aborted inside NumPy SVD while
the test module imported NumPy/Torch before the launcher's sequential-MKL
configuration. Configure the test/runner process before imports; subsequent
combined groups pass. No tolerance, metric formula, loss or model was changed
to bypass the abort. This was local correctness-test execution, not a failed
scientific training job automatically retried on Van.

## Still unverified

Real TRAIN geometry/coordinate semantics, genuine action-file locations and
labels on Van, full input content audit, four-worker/CUDA B32 capacity,50epoch
real-data convergence, numerical replay on Van, clean utility and effectiveness.
The code refuses launch until explicit TRAIN geometry approval and separate
real CUDA Prepare. It never substitutes these synthetic checks for paper results.
