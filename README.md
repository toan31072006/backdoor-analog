# Dose-Controlled Analog Backdoor for WiFi-CSI Human Pose Estimation

Research code for a dose-controlled backdoor against WiFi-CSI 3-D human-pose
estimation. The trigger dose controls the magnitude of a kinematically valid,
joint-localized pose payload rather than selecting a discrete class.

This repository contains the executable experiment contract used for MM-Fi and
Person-in-WiFi-3D (PiW3D). Datasets, action skeletons, checkpoints, and reported
results are intentionally not committed.

## Repository layout

```text
ATKBackd/
  attack/             trigger, payload, poisoning, and diagnostic baselines
  data_utils/         MM-Fi and PiW3D data loaders
  eval/               pose, attack, dose-response, and distortion metrics
  models/             HPE-Li victim implementation
  configs/            scientific experiment configurations
  tests/              unit and paper-contract regression tests
  run_experiments.py  canonical two-dataset experiment runner
  train_backdoor.py   training and evaluation implementation
remote_linux/         preflight, dry-run, pilot, main, and status launchers
CODE_README.md         detailed code-to-paper map
```

The legacy duplicate `wbackdoor` tree and the old Windows launchers are not part
of this deployment repository. `ATKBackd` is the canonical implementation for
both datasets.

## Environment

Python 3.10 or 3.11 is recommended. Install a CUDA-enabled PyTorch build suitable
for the server first, then install the remaining packages:

```bash
python -m pip install --no-cache-dir -r remote_linux/requirements-linux.txt
```

PyTorch is deliberately not pinned in the requirements file because its wheel
must match the CUDA/driver stack of the target machine.

## External data layout

The Linux launchers default to:

```text
~/backdooranalog/
├── datasets/
│   ├── Compress/                    # MM-Fi: E01..E04
│   └── Person-in-WiFi-3D/           # train_data and test_data
├── actions/
│   └── data_bend.npy                # NTU-style (N,3,T,V[,M])
└── runs/                            # generated outputs
```

Paths can be overridden through launcher flags or the documented
`BACKDOOR_*` environment variables. Data and action files remain outside Git.

## Validated run sequence

From the repository root:

```bash
bash remote_linux/00_preflight.sh --device cuda:0
bash remote_linux/01_dry_run.sh --dataset both --device cuda:0
bash remote_linux/02_smoke_test.sh
bash remote_linux/02b_pilot_test.sh --dataset mmfi --device cuda:0 --num-workers 4
```

Only after these checks pass, run the main experiment matrix inside `tmux`:

```bash
bash remote_linux/03_run_main.sh \
  --dataset both \
  --device cuda:0 \
  --parallel 1 \
  --num-workers 4 \
  --seeds 42 0 1
```

The main matrix contains 12 training cells:

- two datasets: MM-Fi and PiW3D;
- two conditions: clean control and proposed micro-Doppler;
- three seeds: `42`, `0`, and `1`.

Running the same command again resumes compatible checkpoints and skips cells
with a valid evaluation cache. Keep the same output directory and scientific
configuration when resuming.

For a full-epoch screening run on one seed, pass (for example) `--seeds 42`.
This still uses 50 MM-Fi epochs and 200 PiW3D epochs; it only reduces the number
of repetitions. Run the default `42 0 1` set before producing the paper's
mean±std tables. The dry-run launcher always validates all three paper seeds,
regardless of the seed subset selected later for training.

Parallel workers have no wall-time limit by default, so a valid long run is not
terminated after six hours. Sites that require a limit can opt in with
`--worker-timeout SECONDS` (for example, `--worker-timeout 86400`). A value of
`0` disables the timeout. This guard is used only by `--parallel > 1` and is a
per-process join timeout, not a scheduler wall-time request.

MM-Fi evaluation uses a process-local CSI cache of up to 512 MiB to avoid seven
re-reads of thousands of small files. With `--parallel N`, allow roughly that
additional host RAM per active MM-Fi process; `--parallel 1` is the safe default.

A subset invocation rewrites the summary CSVs for that subset while preserving
all per-cell checkpoints. After screening seed 42, invoke the launcher again
with `--seeds 42 0 1`; seed 42 will cache-hit and the final three-seed summary
tables will be rebuilt.

## Reproducibility guardrails

The dry-run verifier fails closed if the resolved matrix changes the paper
contract, including seeds, clean-control poison rate, dataset roots, victim loss,
epoch budget, or payload pivot. The current contract uses:

- standard MPJPE and ordinary ERM for victim training;
- MM-Fi pivot `1` with 50 epochs and the HPE-Li SGD recipe;
- PiW3D pivot `7` (`right_hip` to right knee/ankle), global-z payload axis,
  original WBackdoor parameters: 60 degrees, rho `0.1`, diverse selection,
  AdamW/lr `1e-3` for 200 epochs (the original DT-Pose recipe uses `1e-2`);
- paper-facing metrics: MPJPE, PA-MPJPE, PCK@50/40/30/20/10, and T-MPJPE;
- `parallel=1` by default to avoid competing processes on a shared GPU.

The PiW3D YAML and direct runner default to WBackdoor's original seed `0`.
Launchers pass an explicit seed list; use `--seeds 0` for the original seed or
`--seeds 42` for the previously chosen screening seed. Do not reuse the old
90-degree/rho=0.4/uniform outputs for the restored original-parameter profile.

Preflight records the Python packages, CUDA/GPU state, OS information, resolved
data paths, and dataset/action metadata under `~/backdooranalog/runs/preflight`.

See [remote_linux/README_VI.md](remote_linux/README_VI.md) for the Vietnamese
server guide and [CODE_README.md](CODE_README.md) for the detailed implementation
map.

## Responsible use

This code is intended for authorized academic security research and robustness
evaluation of WiFi sensing systems.
