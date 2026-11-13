# Local validation — 2026-11-13

Scope: isolated Uniform/Diverse x Linear/Sqrt/Quad HPE-Li/MM-Fi/P1S1 seed42
setup, max2parallel RTX4090 workers. Not a remote execution/efficacy report.

- 18 focused tests passed, exit0, 18.60s (Windows/Admin CPU environment).
- Tests: native uniform IDs/dose replay; shared dose vector under native full
  FPS selection; exact six-cell config factor isolation; finite shape/identity
  checks; synthetic still predictor nDCE1, exact payload nDCE0, reversed nDCE2;
  zero requested denominator rejected; immutable JSON; read-only inspect;
  real independent OS processes overlap with max2queue; injected failure drains
  peer and never launches third job; actual HPE-Li CPU batch32 SGD update;
  checkpoint binding/RNG/recipe/epoch/nonfinite guards; exact CPU RNG/optimizer
  resume numerical parity.
- PowerShell AST parse PASS; git diff --check PASS; CLI Inspect PASS without
  dataset/CUDA/output writes.
- No modifications to historical payload.py, poison.py, train_backdoor.py:
  SHA256 respectively 306f8a42da080fca087b9facc3c5e5dc0bb8f763e0278a872896c26b8fb7e026,
  141ae6d8786ef2c5c200fbdbe1d469ed5a1769ee80363284a345908116de96ac,
  8f4cfb38b267df0c63a1fbd99cf95a78cdac23c07623002c444fb3fbe1c3a364.
- Added setup only; no git commit/push, remote launch, environment install,
  raw dataset changes, checkpoint imports, historical metric/manuscript edits.

NOT verified locally: full133056TRAIN/33264TEST, full FPS timing, actual4090
two-B32 capacity, CUDA resume reproducibility, speedup, convergence or backdoor
efficacy. Operator must run Prepare then Capacity before separate Launch.
No fallback/retry/kill, no clean training. -FreshLinear explicitly selects
six fresh matched cells instead of an unaudited historical reference import.

Test command (dependency directory only belongs to Admin's existing setup):

```powershell
$env:PYTHONPATH='D:\codex_bend_forward_deps_20261111;E:\wifi\backtime_4090_setup_20261106\ATKBackd'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest ATKBackd/tests/test_selection_mapping_4090.py -q -p no:cacheprovider --basetemp=D:\wifi_results\selection_mapping_tests_20261113_v2
```
