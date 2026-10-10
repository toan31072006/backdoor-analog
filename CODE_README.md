# Code-to-paper map

This document identifies the canonical implementation used by the Linux
experiment workflow. Paths in old YAML files may reflect earlier Colab/server
layouts; the launchers pass the real dataset and action paths through CLI
overrides and record the fully resolved configuration for every run.

## Executable paper contract

The active main experiment is:

| Item | Contract |
|---|---|
| Victim | HPE-Li |
| Datasets | MM-Fi and Person-in-WiFi-3D |
| Scenario | bend |
| Conditions | clean control, proposed micro-Doppler |
| Seeds | 42, 0, 1 |
| Victim objective | standard MPJPE |
| Victim training | ordinary ERM |
| MM-Fi payload | pivot 1, 40 degrees, 50 epochs |
| PiW3D payload | corrected pivot 7, original WBackdoor 60 degrees, 200 epochs |
| PiW3D optimizer | AdamW, original WBackdoor lr 0.001 (DT-Pose lr is 0.01) |
| Poison rate | PiW3D: original WBackdoor 0.1/diverse; MM-Fi: 0.4/uniform; clean: 0 |
| Dose grid | 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 |

This gives 6 training cells per dataset and 12 cells in total. The dry-run
launcher validates the resolved matrix before any training starts.

PiW3D's source YAML keeps the original WBackdoor scientific parameters, except
the corrected right-leg pivot 7 and explicit original z-axis default. Its source
seed and direct-run default are 0. `--seeds 42` or the launcher's explicit seed
list are operator-selected overrides, not the original seed. Paths and runtime
worker/device settings are machine-specific. Use a new output directory for
this 60-degree/rho=0.1/diverse profile; do not mix it with the earlier 90-degree/
rho=0.4/uniform results. Other scenario YAMLs are separate experiment variants.

## Canonical entry points

- `remote_linux/00_preflight.sh`: checks Python packages, CUDA with a
  synchronized GPU operation, storage, both datasets, and action arrays. It
  records an environment and data report under the run directory.
- `remote_linux/01_dry_run.sh`: resolves all 12 scientific configurations and
  invokes `verify_contract.py`. It does not load a dataset or train a model.
- `remote_linux/02_smoke_test.sh`: runs the unit and contract regression tests.
- `remote_linux/02b_pilot_test.sh`: full-data, one-seed pipeline check. Proposed
  and clean each run for 2 epochs.
- `remote_linux/03_run_main.sh`: full-epoch Clean/Proposed matrix. It
  defaults to seeds `42 0 1`, while `--seeds` can select a subset for screening.
- `remote_linux/04_status.sh`: read-only GPU, process, result, checkpoint, and
  storage summary.
- `ATKBackd/run_experiments.py`: canonical Python matrix resolver, trainer
  orchestrator, per-seed table writer, and mean/std aggregator.
- `ATKBackd/train_backdoor.py`: one-cell dataset construction, ordinary-ERM
  training, checkpoint/resume, evaluation, and cache handling.

Do not use the historical `ATKBackd/run_all_experiments.sh` from older bundles.
It is intentionally absent from the deployment repository because it did not
construct the full two-dataset/clean-control matrix.

## Method implementation

### Trigger and payload

- `ATKBackd/attack/trigger.py`
  - derives radial velocity profiles from an NTU-style action skeleton;
  - constructs the dose-scaled antenna-differential micro-Doppler pattern;
  - uses the dataset-specific injection branch;
  - MM-Fi uses the documented zero-mean amplitude-domain approximation.
- `ATKBackd/attack/payload.py`
  - defines dataset-specific skeleton topology;
  - applies the forward-kinematic, bone-length-preserving joint rotation;
  - PiW3D pivot 7 targets right knee/right ankle rather than the old
    right-shoulder/right-side branch.
- `ATKBackd/attack/poison.py`
  - deterministically selects poisoned samples;
  - couples the sampled dose to both trigger strength and target displacement;
  - exposes clean, poison-training, and dose-evaluation modes.
- `ATKBackd/attack/tsba.py`
  - retains a TSBA-adapted diagnostic implementation, but it is excluded from
    the paper matrix because it requires white-box training control.

### Data

`ATKBackd/data_utils/feeder.py` contains both loaders:

- MM-Fi consumes
  `E*/S*/A*/wifi-csi/*_processed.npy` with CSI shape `(3,114,10)`
  and sequence-level `ground_truth.npy` with 17 joints.
- PiW3D consumes `train_data` and `test_data`, each containing
  `csi_ap/<sample>.npy`, `keypoint/<sample>.npy`, and the split list. CSI has
  shape `(3,180,20)`; the canonical experiment filters to one-person samples
  and evaluates 14 joints.

The external bend action must be numeric NTU-style
`(N,3,T,V)` or `(N,3,T,V,M)`.

### Victim model

- `ATKBackd/models/hpeli.py`: HPE-Li victim network.
- `ATKBackd/models/sk_network.py`: selective-kernel building block.
- `ATKBackd/models/factory.py`: dataset-aware model construction.

Only HPE-Li is advertised by the active runner. Removed or unresolved victim
architectures are not silently substituted into the paper matrix.

### Evaluation

- `ATKBackd/eval/metrics.py`: MPJPE, PA-MPJPE, PCK@50/40/30/20/10,
  target/non-target errors, and dose-response statistics. ASR is retained only
  as a low-level diagnostic in per-run JSON, not as a paper table metric.
- `ATKBackd/eval/distortion.py`: trigger/distortion and plausibility measures.
- `ATKBackd/eval/vis_skeleton.py`: qualitative 3-D skeleton rendering.
- `ATKBackd/eval/plot_dose.py`: dose-response plotting utilities.

Each completed cell writes its resolved configuration, poison-plan hash,
checkpoint, evaluation cache, result JSON, and visualization. Each runner call
also writes per-seed CSV/JSON and aggregated mean/std CSV tables.

## Active configurations

The main runner loads these base configurations and applies only explicit CLI
and per-cell overrides:

- MM-Fi: `ATKBackd/configs/mmfi/attack_bend.yaml`
- PiW3D: `ATKBackd/configs/hpeli/attack_ln.yaml`

The runner fixes the proposed trigger, overrides the seed for each repetition,
and uses `rho=0` for a paired clean control. Dataset roots,
`action_npy`, device, worker count, and output directory are machine/runtime
overrides rather than method changes.

The contract validator rejects a victim objective other than MPJPE and rejects
legacy attack-specific victim-loss keys. A run therefore cannot silently train
the victim with access to the poison mask or target-limb identity.

## Diagnostic baseline boundary

The repository labels the archived learned implementation as **TSBA-adapted**, not a verbatim
reproduction of a classification implementation. It retains a bounded
sample-specific generator and alternating generator/victim optimization, while
adapting the generator input and objective to CSI-to-pose regression. Result
metadata records its stronger white-box training access; the proposed
micro-Doppler condition is recorded as a data-only poisoning attack. It is not
used in the four paper tables.

## Reproducibility and resume

`run_experiments.py --dry-run` writes
`experiment_matrix.resolved.json`. The Linux verifier checks:

- exact cell counts, seeds, conditions, poison rates, and pivots;
- exact dataset/action paths;
- MPJPE ordinary-ERM contract;
- dataset-specific optimizer, epoch, angle, and trigger budgets;
- the fixed linear dose grid.

`01_dry_run.sh` deliberately resolves the fixed three-seed paper matrix even if
a later full-epoch training invocation uses `03_run_main.sh --seeds 42`. The
subset option does not weaken or redefine the paper contract. Parallel workers
wait without a timeout by default; `--worker-timeout SECONDS` is an explicit
per-process join guard for `--parallel > 1`, not a scheduler wall-time request.
Subset calls preserve per-cell caches but summarize only that call, so rerun
with `--seeds 42 0 1` to rebuild final paper tables.

Training checkpoints are configuration-fingerprinted. Re-running the identical
launcher command resumes compatible checkpoints and skips cells with a valid
evaluation cache. Do not rename output directories or change scientific fields
mid-run.

## Tests

The isolated `carrier_bank_screen_v1` development profile is implemented by
`ATKBackd/run_carrier_bank_drafts.py`, `ATKBackd/carrier_bank_fit.py`, and
the bank variants of `ATKBackd/attack/learned_carrier.py`. It freezes an
attacker-owned trigger before fresh ordinary-ERM victim training and never
uses the official test for selection. Six-cell export retains every main
metric, the six-dose record and per-metric Blended deltas. See
[carrier_bank_drafts.md](ATKBackd/configs/mmfi/carrier_bank_drafts.md).

The separate `paired_guard_screen_v1` development profile keeps the existing
Blended and two-carrier train-aware recipes as controls. The new two-carrier
candidate uses clean-only surrogate twins forked with matching SGD momentum
after warmup; both real updates and lookaheads use paired batches. Selection
checks all seven clean utility metrics against those current-stage twins.
This adds attacker compute, not victim-training access, and cannot guarantee
that an independently trained victim preserves every metric. All candidates,
including any explicitly flagged no-eligible-candidate fallback, are retained.
See [paired_guard_drafts.md](ATKBackd/configs/mmfi/paired_guard_drafts.md) and
`remote_linux/11_run_paired_guard_drafts.sh`. This is TRAIN-holdout screening,
not a new set of official-test paper results.

One frozen `lc_paired_guard` artifact can be confirmed without refitting via
`remote_linux/12_run_paired_guard_full.sh`. This separate seed-42, 50-epoch
full-MM-Fi TRAIN/TEST runner never imports old victim weights or metrics and
preserves the source guard/fallback flags. It is a single-candidate confirmation,
not a baseline comparison. See
[paired_guard_full.md](ATKBackd/configs/mmfi/paired_guard_full.md).

From `ATKBackd`:

```bash
python -m pytest -q tests -p no:cacheprovider
```

The tests cover payload topology, poison-plan determinism, zero-mean trigger
behavior, loss normalization, schedule metrics, MM-Fi split behavior,
checkpoint compatibility, and runner contract validation.

## Honest scope

This repository provides code and an executable experiment contract. It does
not ship datasets, action arrays, checkpoints, or claimed numerical results.
Full paper tables should be populated only from completed multi-seed runs whose
resolved matrices and preflight reports are archived with the results.
