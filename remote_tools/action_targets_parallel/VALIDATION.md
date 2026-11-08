## Material Passport

- Origin: academic-research-suite / experiment-agent, 2026-11-08.
- Verification: local code/CPU/synthetic fixtures only, NOT real MM-Fi/PiW
  target geometry, CUDA capacity, full-data convergence or empirical results.
- Base revision be090d04dcb09618468bb2f5475e3fad38284a90.
- Runtime local Python3.13.5, Torch2.9.1+cpu, NumPy2.1.3; sequential MKL,
  two OMP/MKL threads before scientific imports.
- User requested GitHub delivery and will launch on Van. No remote jobs were
  started, paused, killed or resumed from this workspace.

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
