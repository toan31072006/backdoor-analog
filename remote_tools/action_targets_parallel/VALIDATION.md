## Material Passport

- Current Cross v2 correction: user-approved 2026-11-08; base5f5f152. See
  CROSS_REACH_V2_VI.md. Historical269-pass coordinator gate below describes v1,
  not a full-data success or a substitute for the current correction checks.
- Origin: academic-research-suite / experiment-agent, 2026-11-08.
- Verification: local code/CPU/synthetic fixtures only, NOT real MM-Fi/PiW
  target geometry, CUDA capacity, full-data convergence or empirical results.
- Base revision be090d04dcb09618468bb2f5475e3fad38284a90.
- Runtime local Python3.13.5, Torch2.9.1+cpu, NumPy2.1.3; sequential MKL,
  two OMP/MKL threads before scientific imports.
- User requested GitHub delivery and will launch on Van. No remote jobs were
  started, paused, killed or resumed from this workspace.

## Cross v2 correction checks (2026-11-08)

- Reported TRAIN9092: original5f5f152 reproduces the Cross failure. Revised
  target passes all six doses with max bone delta2.78e-16m, unchanged original
  pose/non-target joints; short wrist19.97mm across and64.29mm front. Nod outputs
  for all six doses are bitwise equal to the original published code. Pasted
  raw coordinates used in memory only, not added to Git or uploaded.
- Focused initial gate27passed/exit0/11.91s. Distributed tests use invented
  poses for both skeletons, short/asymmetric/inner-radius/normal arms,
  truly non-crossable batch refusal, coordinate equivariance, old reachable
  target equality, unchanged Nod, Cross version rejection and audit counts.
- Also corrected MM-Fi geometry-audit reference norms to float64, matching
  target_pose/PiW precision. No threshold change. The synthetic float32
  rotated/translated-label integration test covers this audit path.
- First full296-case attempt aborted with Windows native access violation
  in the unchanged HPE-Li CPU B32 forward, while running the PiW action test.
  No completed summary/XML: do NOT count it as a full pass. A subsequent
  read-only OS snapshot showed ~440MiB free physical RAM/~928MiB free virtual
  capacity on this8GiB host. Memory pressure is observed, not a proven sole
  cause of the native crash. No production model/recipe/backend was changed.
- Wider grouped attempt:288passed/3failed/5deselected, exit1,345.24s. Five
  heavy native HPE-Li CPU forward/backward cases were explicitly deselected,
  not counted passing. One new test asserted4poison poses in an80TRAIN/rho.1
  fixture; fixed the TEST assertion to8poses/16arms, not the production data
  count, rho or audit. The last two failures are unchanged legacy MM-Fi spawn
  cases: child PyTorch DLL import explicitly raises WinError1455 (paging file
  too small). No worker count, timeout, model or OS paging setting was changed.
- **Final focused correction gate:58passed/exit0/175.85s.** Includes all27 new
  reach/audit cases, all30coordinator/PowerShell/OS-process cases and the full
  corrected PiW staged CPU fixture (Geometry, discarded smoke, two tiny-model
  200epoch scratch victims, fixed-final evaluation/NPZ replay and input-change
  refusal). Thus the erroneous fixture assertion is resolved, but the broader
  resource-limited run is NOT presented as a full296-test pass. Seven heavy/
  legacy spawn cases have no successful fresh validation on this host.
- Final command, from ATKBackd with frozen scientific source:

  ```text
  python -B -m pytest tests/test_cross_reachable.py
    tests/test_action_targets_parallel.py
    tests/test_piw_action_targets.py::test_full_staged_pipeline_cpu_fixture_not_experimental_data
    -q -p no:cacheprovider
    --basetemp=E:\wifi\_cross_reach_final_gate_20261108_c
    --junitxml=E:\wifi\cross_reach_final_gate_20261108.xml
  ```

- All three PowerShell wrapper AST parses and git diff --check pass. Native
  model/trainer/CSI/trigger/metric files have no diff. Historical native CPU
  B32 evidence below is not a substitute for Van's real CUDA B32 Prepare,
  including both concurrent capacity probes. No remote retry/job launch.

## Focused coordinator gate

Earlier focused gate:29passed, exit0,58.62s. The final grouped gate below
also includes the added full-disk abort-metadata drain test (30 coordinator
cases total in that final run):

```text
python -B -m pytest tests/test_action_targets_parallel.py -q -p no:cacheprovider
  --basetemp=E:\wifi\_cross_nod_parallel_tests_20261108_c
```

- Real two-process CPU toy jobs overlap; first job starts its evaluation
  before peer training ends. Slot is retained through evaluation. MM-Fi wave
  fully finishes before PiW starts; never a third concurrent owned process.
- Nonzero exit, a newly detected foreign-job guard failure after first spawn,
  and Ctrl+C during polling all drain owned peers without kill/terminate,
  retries, premature lock release or launching a later wave.
- Geometry/Prepare/Launch CLI and actual OS lock/queue/CSV publication wiring
  tested with clearly synthetic scientific-stage replacements. Partial row
  counts1/2/3/4 are published after each job; final report only4/4. Explicit
  geometry approval required, Prepare does not fall through into training.
- Original serial-action GPU OS lock and orphan slot OS lock block a second
  coordinator before it probes GPU. Real Windows PID liveness uses OpenProcess,
  not destructive os.kill. Foreign Python GPU PIDs are refused; only spawned
  live peer PIDs are allowed. Desktop exceptions are narrow absolute paths.
- Two distinct process IDs and allocated/updated rendezvous are required for
  capacity receipts. One ready process, same-PID pair or changed queue refused.
- Native per-dataset plans bind the two-slot policy without changing recipes,
  poison draws or target geometry. Serial/parallel plan mixing refused.
- Standard-library-only CheckOnly subprocess imports no Torch and creates no
  output. Windows PowerShell AST and actual wrapper invocation with spaced
  fixture paths pass; no execution or script-string quoting of user paths.
- Per-cell engine invocation (both datasets, training+evaluation and
  evaluate-only) tested without a peer checkpoint. No CLI batch/epoch/lr/AMP/
  third-slot/device override was added.

## Broader regression gate

**Final grouped gate:269passed, exit0,401.70s (6m41s).** Command from
ATKBackd, with a new dedicated CPU-fixture basetemp:

```text
python -B -m pytest tests/test_action_targets_parallel.py
  tests/test_piw_action_targets.py tests/test_piw_a1_adapter.py
  tests/test_piw_v2.py tests/test_piw_hpeli_clean_runner.py
  tests/test_mmfi_action_targets.py tests/test_mmfi_coupling.py
  tests/test_mmfi_split.py tests/test_mmfi_spawn.py -q -p no:cacheprovider
  --basetemp=E:\wifi\_cross_nod_parallel_full_tests_20261108_c
  --junitxml=E:\wifi\cross_nod_parallel_final_20261108.xml
```

Includes real HPE-Li CPU B32 finite updates on synthetic Cross/Nod inputs;
native ordinary SGD50epoch and AdamW200epoch tiny-model RNG/optimizer exact
resume tests; both skeleton topologies, six-dose controls, original adapters,
phase preservation and saved-NPZ metric replay. A new native PiW tiny-model
fixture trains only Cross through final199 and evaluates/replays full six-dose
arrays while Nod has NO checkpoint, verifying the old serial both-final gate
does not prevent the coordinator's immediate per-job evaluation.

## Development findings preserved

- First focused run22passed/1failed: test harness passed an unbound positional
  path to PowerShell -Command. Fixed test harness quoting; actual wrapper AST
  and actual -File invocation now pass. No scientific code changed for this.
- Earlier grouped development run262passed/2failed: one frozen-source guard
  correctly detected source edits made while its fixture was running; the
  other tiny fixture had40TRAIN and therefore only4poisons, insufficient for
  the real8poison+24clean smoke. Changed that fixture to80TRAIN, not production
  poison fraction/batch/coverage guard. Final grouped run uses frozen code.
- These are local test-development failures, not automatically retried real
  scientific training jobs on Van. No tolerance/metric/recipe was relaxed.

## Preservation and limits

Historical trainer/model/CSI/trigger/metric sources have no tracked diff.
New callbacks only coordinate residency during discarded smoke or check the
parent lease between epochs; they do not change optimizer steps or RNG draws.
Old serial entrypoints stay serial; the new coordinator has separate output
and binds execution policy into each dataset plan/checkpoint.

Original ZIPs and manuscript/PDF hashes remain unchanged:

```text
MMFiCrossNodTargets_s42_code_20261107.zip
0ac0735661d821e3435ba826e2fbd12cce3eeef8656a08e2c02c4b6a1eb1ff68
PiWCrossNodTargets_s42_code_20261107.zip
fe5d5b452a7fd5f19fbdce4d0bce1a0bc617a603003eec6615a0c532d93f296f
main.tex
af8ddeb11417f4f59ae435306e27f690291f2caea497ebb682e675d5f6a9b6a3
CVPR_27_Do_Dose_compiled.pdf
61e63400e49650a30d58bfce5469af39fa2397bcd04ee9aa2b96bbbb34b1e77e
```

Real TRAIN geometry/forward convention still needs user inspection for BOTH
datasets. Full-content input audit and actual concurrent CUDA B32 capacity
smokes remain mandatory on Van. A successful smoke does not guarantee freedom
from later OOM, speedup, learned semantic action fidelity or physical feasibility.
No action-classifier/contact/joint-limit validation or multi-seed uncertainty;
no new Clean-trained+trigger/baseline controls, so not a completed full T7.
