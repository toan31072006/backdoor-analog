# PiW BackTime: evaluate victim da train, khong can Clean rieng

## Material Passport

- Origin: academic-research-suite / experiment-agent, user-approved eval-only.
- Date: 2026-11-12, Asia/Bangkok. Profile: piw_backtime_victim_only_eval_s42_v1.
- Scope: existing v4 final199 HPE-Li, WiFlow, MetaFi++; no training or Clean.
- Verification: local CPU/synthetic and real downloaded HPE-Li array replay.
  Actual Van WiFlow/MetaFi inference has NOT run from the Admin workspace:
  noninteractive SSH is denied. This is not a complete matched-control table.

User explicitly removed the Clean prerequisite for victim-metric evaluation.
Clean MPJPE here is the poisoned victim on clean input, NOT a Clean-trained
reference. Target MPJPE/PCK, DCE and same-victim no-trigger controls need no
other trained model. Matched-Clean utility differences and Clean+trigger are
NOT_EVALUATED, never inferred, substituted or set to zero.

## Execution

Use `remote_tools/piw_backtime_victim_eval.py` from OUTSIDE the original source
checkout. Download that single file at the published, pinned revision and check
its handed-off SHA256, or use a separate fresh checkout. Do NOT copy the file
into the original ATKBackd or pull that source-bound checkout.

On Van, invoke the existing dose-backdoor Python, with these arguments:

```powershell
& 'C:\Users\Van\backdooranalog\envs\dose-backdoor\python.exe' -B -u `
    'U:\backdooranalog\code\piw_backtime_victim_eval_v1.py' `
    --source-repo 'U:\backdooranalog\code\PiWTrainFirstV4_2c5b8de8' `
    --run 'U:\backdooranalog\runs\piw_backtime_three_backbones_train_first_s42_v4' `
    --outdir 'U:\backdooranalog\runs\piw_backtime_victim_eval_s42_v1'
```

`source-repo` is the ORIGINAL frozen training checkout, NOT a newly cloned main.
It must still pass its original source/runtime/input/checkpoint audits. Do not
disable those checks or update the training plan when something differs.

The sidecar runs one backbone at a time, in HPE-Li -> WiFlow -> MetaFi++ order.
Original native lock, all three final199 checks, actual capacity/backend binding,
full input hashes, official TEST order, trigger/checkpoint digests remain active.
It never invokes optimizer, fit, train, Clean reference loading or checkpoint
saving. Native model/trigger state must be unchanged after inference.

If the original HPE-Li evaluation/archive exists, validate its victim-related
hashes, identities/GT/targets and exactly replay every victim metric. Do not
load or reverify the old Clean reference; do not claim to have revalidated those
reference-control metrics. Other victims get native six-dose inference.

The new output root must not exist and must be outside the original source,
training run and dataset. Preserve orphan/failed outputs; no overwrite, retry,
training resume or original plan/result edits. A transient native runner.lock
is the only file created inside the old run and is released normally. Do not
remove someone else's lock. Run foreground; keep terminal open. No background
service, scheduled retry, GPU-process termination or driver change is performed.

## Outputs and downloading results only

- `hpeli/evaluation.json`, `wiflow/evaluation.json`, `metafiplusplus/evaluation.json`:
  five native main metrics and six-dose victim rows, missing-control labels,
  source/checkpoint/trigger/evaluator hashes and evaluation mode.
- `victim_results.csv` and `summary.json`: all three victim metric rows. Written
  only after preservation and final original-plan/full-input checks pass.
- Each backbone also retains `predictions.npz` for exact metric replay. No `.pt`
  file is created or exported. Only download JSON/CSV if scores alone are needed.

From Admin, `scp` these two small files after the COMPLETE message:

```powershell
scp 'van@100.100.233.102:U:/backdooranalog/runs/piw_backtime_victim_eval_s42_v1/summary.json' 'D:\wifi_results\backtime_victim_summary.json'
scp 'van@100.100.233.102:U:/backdooranalog/runs/piw_backtime_victim_eval_s42_v1/victim_results.csv' 'D:\wifi_results\backtime_victim_results.csv'
```

## Verification performed locally

- 23 new tests PASS, 48.74s: exact parity to native victim terms (float32/64),
  PCK scale/target/positive-dose DCE, perfect/still/reversed responses, malformed
  shapes/NaNs/axis/scale refusal; three-output real tiny-model CPU inference,
  HPE-Li cache reuse, missing-control labels, no Clean/fit/train/checkpoint save,
  byte-for-byte preservation of original fixtures, cache tampering/model-state
  mutation refusal, final199 gate and existing/overlapping output refusal.
- Combined final focused gate: **37 passed, 1 deselected**, exit0, **11.05s**:
  new evaluator tests plus `test_piw_v2.py`, excluding the old full training and
  exact optimizer-resume lifecycle. Root:
  `D:\piw_victim_metric_gate_ad39a85b1eda493a9d7b6734d8daf628`.
- Real downloaded HPE-Li full TEST archive (2586 frames, six doses): all five
  mainmetrics and all victim-only dose fields replay EXACTLY against the existing
  original evaluation.json. No new model inference or Clean loading was done.
- Original native runner and adapters/models are unmodified. Tests do not prove
  real full-dataset CUDA success, match missing Clean controls or provide unseen
  WiFlow/MetaFi scores. No TEST-driven recipe/target/seed/checkpoint selection.

Reproduce local tests from ATKBackd, using a fresh basetemp on a free volume:

```powershell
python -B -m pytest tests/test_piw_backtime_victim_eval.py tests/test_piw_v2.py `
    -k 'not full_toy_run_resume_replay_and_tamper' -q -p no:cacheprovider `
    --basetemp 'D:\NEW_UNIQUE_VICTIM_EVAL_TEST_DIRECTORY'
```
