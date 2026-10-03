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
| Conditions | clean control, proposed micro-Doppler, TSBA-adapted |
| Seeds | 42, 0, 1 |
| Victim objective | standard MPJPE |
| Victim training | ordinary ERM |
| MM-Fi payload | pivot 1, 40 degrees, 50 epochs |
| PiW3D payload | pivot 7, 90 degrees, 200 epochs |
| Poison rate | 0.4 for attacks; 0 for clean controls |
| Dose grid | 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 |

This gives 9 training cells per dataset and 18 cells in total. The dry-run
launcher validates the resolved matrix before any training starts.

## Canonical entry points

- `remote_linux/00_preflight.sh`: checks Python packages, CUDA with a
  synchronized GPU operation, storage, both datasets, and action arrays. It
  records an environment and data report under the run directory.
- `remote_linux/01_dry_run.sh`: resolves all 18 scientific configurations and
  invokes `verify_contract.py`. It does not load a dataset or train a model.
- `remote_linux/02_smoke_test.sh`: runs the unit and contract regression tests.
- `remote_linux/02b_pilot_test.sh`: full-data, one-seed pilot. Proposed and
  clean run for 2 epochs; TSBA runs for 11 epochs to cross its 10-epoch warm-up.
- `remote_linux/03_run_main.sh`: main Clean/Proposed/TSBA matrix.
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
  - implements the TSBA-adapted, sample-specific learned trigger baseline.

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

- `ATKBackd/eval/metrics.py`: MPJPE, PA-MPJPE, PCK, target/non-target errors,
  baseline-corrected conjunctive ASR, and dose-response statistics.
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

The runner overrides the trigger name for Proposed versus TSBA, the seed for
each repetition, and `rho=0` for a paired clean control. Dataset roots,
`action_npy`, device, worker count, and output directory are machine/runtime
overrides rather than method changes.

The contract validator rejects a victim objective other than MPJPE and rejects
legacy attack-specific victim-loss keys. A run therefore cannot silently train
the victim with access to the poison mask or target-limb identity.

## Baseline boundary

The repository labels the learned baseline as **TSBA-adapted**, not a verbatim
reproduction of a classification implementation. It retains a bounded
sample-specific generator and alternating generator/victim optimization, while
adapting the generator input and objective to CSI-to-pose regression. Result
metadata records its stronger white-box training access; the proposed
micro-Doppler condition is recorded as a data-only poisoning attack.

## Reproducibility and resume

`run_experiments.py --dry-run` writes
`experiment_matrix.resolved.json`. The Linux verifier checks:

- exact cell counts, seeds, conditions, poison rates, and pivots;
- exact dataset/action paths;
- MPJPE ordinary-ERM contract;
- dataset-specific optimizer, epoch, angle, and trigger budgets;
- the fixed linear dose grid.

Training checkpoints are configuration-fingerprinted. Re-running the identical
launcher command resumes compatible checkpoints and skips cells with a valid
evaluation cache. Do not rename output directories or change scientific fields
mid-run.

## Tests

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
