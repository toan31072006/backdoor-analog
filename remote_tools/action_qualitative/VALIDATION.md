# Validation 2026-11-09

## Material Passport

- Purpose: publish a CPU-only six-case saved-prediction plotter for the user to run on Van.
- Implementation: new isolated scripts; no edits under ATKBackd and no historical result/model/data/manuscript writes.
- Verification boundary: local tests and actual saved-array rendering, not a remote run or fresh inference.

## Completed checks

- `17 passed`, exit 0, 9.16 s using local Python 3.13 / NumPy / Matplotlib. Fixtures are synthetic and live in a named synthetic test temp directory, not empirical figure exports.
- Schemas: four new Cross/Nod NPZs, historical MM-Fi three-example export, PiW v3 NPZ or portable reference ZIP. No pickle/checkpoint read; ZIP test deliberately contains an invalid checkpoint and still succeeds.
- Refusals: source hash/receipt changes, wrong plan/doses/targets/fixed frames, changed non-target labels, incomplete six-case input bank, existing output directory.
- Shared GT-only projection retains prediction translation; scalar DCE uses original 3D XYZ, not the projection.
- Per-case figure and all-six overview render smoke tests produce PNG/PDF.
- `Run-ActionQualitative.ps1` parsed with Windows PowerShell AST without errors.
- Real saved-data validation: both downloaded Nod files from the completed v3 run loaded with their result/complete receipts. Fixed TEST `[0,16632,33263]` on MM-Fi and `[0,1293,2585]` on PiW. Actual PNG/PDF figures produced locally; source NPZ hashes unchanged. Status explicitly partial (2/6), not six-case completion.
- PiW and MM-Fi real Nod figure previews inspected; MM-Fi weak response and PiW drift are retained, not altered by local fitting.

## Not verified here

- Remote existence of MM-Fi Bend export or PiW v3 reference at the default paths.
- Four remaining REAL NPZ inputs, all-six real combined figure, or actual Van ZIP creation. Missing source paths cause a clear stop, not GPU fallback.
- New inference/training, across-seed stability, anatomy/action plausibility or physical triggering. No claim of successful attacks on all six cases.

Historical Bend means a motion prior with a leg-rotation target. It must not be presented as a torso-bending target experiment.
