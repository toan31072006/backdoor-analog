## Material Passport

- Origin Skill: experiment-agent (academic-research-suite)
- Origin Mode: run — source integration and local verification only
- Origin Date: 2026-10-08 (work started)
- Verification Status: UNVERIFIED for full MM-Fi experiments; local checks passed
- Version Label: mmfi_backdoorbench_operators_v1

## Experiment Result

- ID: mmfi_backdoorbench_integration_checks
- Type: generic / code verification
- Status: completed
- Working Directory: E:/wifi/backdoor-analog/ATKBackd
- Duration: 52.99 seconds (full pytest execution)
- Exit Code: 0

Executed in PowerShell with process-local MKL_THREADING_LAYER=SEQUENTIAL and
OMP_NUM_THREADS=1:

```text
C:/Users/Admin/anaconda3/python.exe -B -m pytest -p no:cacheprovider -p matplotlib -q tests --basetemp E:/wifi/_bb_full_tests_20261008_08acf7
```

264 tests passed; 5 preexisting native-Linux-Bash tests skipped on Windows.
The targeted operator/source test file has 59 cases, including six added cache,
configuration and provenance checks. No full MM-Fi data or GPU training was used.

## Source-backed integration

- Upstream: https://github.com/SCLBD/BackdoorBench
- Commit: `f02e3534645f0ee63d6848653062cd6c0d6c400d`
- Operators: `AddMaskPatchTrigger` and `blendedImageAttack`
- Retained files: [vendored source](ATKBackd/third_party/backdoorbench/),
  [upstream source hashes](ATKBackd/third_party/backdoorbench/SOURCE_MANIFEST.json),
  [attribution and changes](ATKBackd/third_party/backdoorbench/NOTICE.md),
  [full license](ATKBackd/third_party/backdoorbench/LICENSE)
- Copyright (c) 2022, CUHK(SZ), SRIBD; upstream CC BY-NC 4.0. This is code from
  a third-party benchmark, not a release by the BadNets/Blended paper authors.

A read-only AST comparison against the downloaded pinned upstream source
verified every method body in both retained classes. The patch excerpt omits
unused imports and an optional type annotation; math and calls are retained.
Tests also spy on actual operator calls: this is not just a source URL attached
to the old independent implementation.

## Output summary

- White BadNets patch: dose1/default opacity1 performs direct upstream masked
  replacement. Zero trigger bins remain transparent. The old binary soft patch
  at opacity0.185 is no longer the default. The shared eps does not set opacity.
- Blended: the upstream operator performs the float convex combination. The
  CSI pattern remains fixed random seed42 with alpha0.185, explicitly different
  from upstream image defaults (Hello Kitty and alpha0.2).
- CHW/HWC conversion preserves antenna separation, CSI floating precision,
  shape, range and independent storage. The PIL/uint8 wrapper is not used.
- Resolved config binds upstream commit/hash, local operator/adapter hashes
  (LF-normalized across OSs) and implementation version before cache hashing.
  Stale metadata is rejected and old independent baseline caches cannot hit.
- Proposed, WaNet, RF protocols, dataset split, training recipe and global
  checkpoint/evaluation schemas remain unchanged. Existing result folders
  are not deleted or overwritten.
- A no-data/no-GPU matrix dry-run resolved ten unique cells at seed42 and wrote
  its manifest outside the repository. `git diff --check` passed.

## Limitations

Only two trigger operators are integrated, not the complete BackdoorBench image
classification training pipeline. CSI patterns/shapes, HPELi, ordinary MPJPE
and per-sample pose targets are declared task adaptations. Different triggers
do not have a common amplitude L-infinity budget. No effectiveness, exact
paper reproduction, dataset-scale reproducibility or statistical claim follows
from these tests. Full MM-Fi training was not launched. Use a NEW output
directory for the revised four-table matrix.
