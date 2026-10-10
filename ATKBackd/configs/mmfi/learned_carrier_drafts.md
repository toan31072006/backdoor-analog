# Learned-carrier exploration (not paper results)

Material Passport: experiment implementation; origin user-approved exploration;
version `learned_carrier_screen_v1`; date 2026-10-10; full MM-Fi verification
status **UNVERIFIED until the user runs the plan**.

## Nine independent cells

| Key | Role |
| --- | --- |
| `clean` | Fresh rho=0 victim; also used to probe every candidate trigger |
| `proposed` | Fixed current multi-carrier + peak bound under the new dual budget |
| `blended` | Source alpha=0.2*d adapted to CSI/pose, under the same dual budget |
| `lc_weights` | Learn carrier weights using train-only surrogate alternation |
| `lc_sparse` | Learn a sparse subcarrier/link mask |
| `lc_combined` | Combine learned weights and sparse allocation |
| `lc_gradient` | Train-only surrogate gradient matching |
| `lc_energy` | Learn weights and a bounded amplitude with a stronger realized-energy penalty |
| `lc_selection` | Fixed Proposed with training-only informative poison identities |

These are **independent CSI hypotheses inspired by papers**, not source paper
reproductions. No attack success, rank, improvement or physical feasibility is
guaranteed. Old publication/comparison rows are untouched.

## Fixed scientific protocol

- Seed 42; rho=0.1; SGD lr=0.001, momentum=0.9, wd=0, batch 32.
- Fresh independent victim, ordinary full-pose MPJPE ERM, no surrogate weight
  transfer and no attacker-specific victim loss. Surrogate initialization uses
  a separate seed 4242; victim initialization and poison identities remain 42.
- Train 20,000 samples for 15 epochs; evaluate a disjoint 4,096-sample holdout
  from the official TRAIN split. Official test is not used for selection.
- Every trigger learns paired U(0.2,1) trigger/payload doses; target joints
  [2,3], pivot 1, z-axis rotation 40*d degrees. Six evaluated doses are retained.
- Uniform poison identities and doses are identical except `lc_selection`,
  which is explicitly a separate poison-selection ablation.
- Common **actual per-input/per-dose peak ceiling** from Original eps=0.185,
  plus actual normalized-input relative L2 <= 0.10*d. This is a NEW declared
  budget experiment; it does not silently reinterpret the historical table.
- Equal upper bounds do not equalize realized peak, L2, fitting compute or
  modified coordinates. Artifact/provenance and fitting budgets are recorded.
- Artifact fitting completes BEFORE fresh victim training. The victim receives
  only (modified CSI, modified pose) ordinary examples. This is label-changing
  data poisoning, not a strict clean-label attack.

Default fitting budget: warmup 3 epochs; 3 alternating rounds; 32 inner steps
and 16 outer steps per round; batch 32; trigger lr 0.02; up to 4,096 fitting
TRAIN samples; internal validation fraction 0.2. This internal validation is
different from the external 4,096-sample evaluation holdout.

Only the energy arm also learns a sigmoid-bounded gain amplitude; the other
carrier arms have fixed gain amplitude. This is a declared joint hypothesis,
not a pure regularizer-only ablation. Fitting takes extra compute beyond the
15 victim epochs, especially second-order gradient and lookahead steps; no
fixed server completion time is guaranteed.

Lookahead is one differentiable SGD step without momentum, not exact BLTO
or a full training unroll. The gradient arm matches only the final regression
weight/bias and uses cosine alignment plus energy in its outer objective.
Candidate selection for all fitted arms uses inner-validation target error
plus clean error; energy is recorded, not separately minimized at selection.

## Run on the server

```bash
cd "$HOME/backdooranalog/code"
export CUDA_VISIBLE_DEVICES=0,1,3
DOSE_OUT="$HOME/backdooranalog/runs/mmfi_learned_carrier_s42_v1"
bash remote_linux/09_run_learned_carrier_drafts.sh \
  --outdir "$DOSE_OUT" --devices cuda:0 cuda:1 cuda:2 --num-workers 4 --dry-run
```

Read the resolved plan, then start it in tmux (check current GPU availability
before choosing device IDs):

```bash
tmux new -s mmfi-learned
cd "$HOME/backdooranalog/code"
export CUDA_VISIBLE_DEVICES=0,1,3
bash remote_linux/09_run_learned_carrier_drafts.sh \
  --outdir "$HOME/backdooranalog/runs/mmfi_learned_carrier_s42_v1" \
  --devices cuda:0 cuda:1 cuda:2 --num-workers 4
```

Here the physical GPUs 0,1,3 become logical cuda:0,cuda:1,cuda:2; physical
GPU 2 is hidden from the processes. Change the visibility list if GPU usage
changes. After a dry-run in this output directory, omit `--fresh` for launch.

One independent fitting+victim cell occupies each device, not multi-GPU DDP.
Same command resumes unchanged frozen artifacts/checkpoints/caches. Changed
scientific parameters, source hashes or artifacts require a NEW output path.
`--fresh` only accepts an empty/new directory and never deletes anything.
`--cells lc_weights lc_sparse` runs a subset plus clean control; all-row export
waits for the remaining rows. `--export-only` never fits or trains but can run
read-only HPE clean-victim probes and distortion measurement. `--dry-run`
never reads data or probes CUDA. `--skip-clean-probe` and `--skip-distortion`
explicitly mark missing controls and must be chosen when creating the plan.

```bash
tail -n 12 "$HOME/backdooranalog/runs/mmfi_learned_carrier_s42_v1"/*/console.log
pgrep -af 'run_learned_carrier_drafts|train_backdoor'
```

## Outputs and interpretation

`draft_summary.csv/json/md` retains all nine rows: clean MPJPE, PA-MPJPE,
relative PCK .5/.4/.3/.2/.1, T-MPJPE at d=1, mean positive-dose T-MPJPE and
same-model no-trigger improvement. `dose_response.csv/json` retains all six
doses. `input_distortion.csv/json` measures identical held-out pairs and tests
actual per-pair budgets without HPE forward passes. `clean_victim_probes.csv/json`
tests each trigger at d=1 on the same rho=0 checkpoint, so immediate adverse
effects can be separated from poisoning-induced effects.

The summary also joins d=1 relative L2, RMSE, Linf, SNR and clean-victim probe
improvement by method key. Missing explicitly skipped measurements stay
missing, never become invented zeros or ranked wins.

CSI distortion is dimensionless digital-input distortion, **not RF stealthiness
or over-the-air feasibility**. SNR is derived from relative L2, not independent
evidence. Single seed has no uncertainty estimate. Ordered input identity
checksums do not hash every CSI/pose file's bytes. Strong holdout performance
requires a later pre-registered fresh full-budget confirmation.

## Inspiration, not cloned method claims

- SIBA / TIFS 2024: https://arxiv.org/html/2306.06209v3;
  https://github.com/YinghuaGao/SIBA — sparse-mask idea, not pixel defaults.
- BLTO / ICLR 2024: https://github.com/SWY666/SSL-backdoor-BLTO — surrogate
  alternation idea; original SSL contrastive objective is not HPE ERM.
- Sleeper Agent / NeurIPS 2022: https://github.com/hsouri/Sleeper-Agent —
  gradient-matching idea; this adapter uses pose targets and continuous doses.
- LIRA / ICCV 2021: https://github.com/sunbelbd/invisible_backdoor_attacks —
  learnability/low-noise motivation; this is NOT its joint victim-training recipe.
- Wicked Oddities / ICLR 2025:
  https://github.com/mail-research/wicked-oddities-backdoor — informative
  selection inspiration; training-pose-error stratification is not their method.
