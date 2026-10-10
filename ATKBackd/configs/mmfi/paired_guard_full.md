# Frozen paired-guard Proposed: full MM-Fi confirmation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: implementation (user-requested full confirmation)
- Origin Date: 2026-10-11
- Verification Status: UNVERIFIED for the full MM-Fi run; implementation is not a benchmark result
- Version Label: paired_guard_full_setup_v1
- Result status: `FULL_CONFIRMATION_SINGLE_SEED`; not a comparative or statistical-superiority claim

This run answers one question: how does the already frozen `lc_paired_guard`
trigger behave with a fresh victim trained for 50 epochs on the full official
MM-Fi TRAIN split and evaluated on the official TEST split?

## One arm; no refitting or baseline retraining

The source defaults to
`runs/mmfi_paired_guard_s42_v1/lc_paired_guard`. Its learned trigger is frozen
exactly as saved. There is no carrier optimization, clean-twin refitting,
hyperparameter search, or transfer of surrogate/victim model weights. A fresh
victim uses seed 42, ordinary full-pose MPJPE ERM, SGD lr=.001, batch32, the
existing optimizer/schedule, and 50 epochs. All scientific trigger/payload
settings stay the same: rho=.1, paired Uniform(.2,1) doses, pivot1, target
joints2/3, and the existing 40-degree payload. The same digital dual-budget
operator enforces the per-input/per-dose Original peak ceiling (eps=.185)
and relative-L2 ceiling .10*d. Neither perturbation budget is relaxed.

The full training population is larger than the 20,000-frame screen, so its
poison count and identities are newly resolved under the same seeded policy.
The 4,096-frame TRAIN holdout used by the screen is not the evaluation split
for this confirmation. The official TEST split is evaluated at d=0/.2/.4/.6/.8/1
and must not be used to select or refit the trigger.

The source screen reported `selected_utility_gate_passed=false` and
`no_eligible_candidate=true`. This run evaluates that same fallback artifact;
it does not relabel it as a gate-passing trigger. These source flags are
retained. Guard failure is an internal selection result, not a runtime error.

Old Blended and train-aware results are not imported, copied into the new
directory, or retrained. Do not rank this result against the old 15-epoch
TRAIN-holdout screen, or against an older peak-only-budget full run. A later
comparison would require matching the full training/evaluation protocol and
the actual dual budget for every baseline.

## Run on MICA

First inspect current GPU use. One arm needs one GPU; three GPUs do not make
this single-victim runner distributed training. The example masks physical
GPU3, which becomes process-visible `cuda:0`.

```bash
cd "$HOME/backdooranalog/code"
bash remote_linux/12_run_paired_guard_full.sh --dry-run
tmux new -s mmfi-paired-full
```

Inside the interactive tmux shell:

```bash
cd "$HOME/backdooranalog/code"
set -o pipefail
mkdir -p "$HOME/backdooranalog/runs"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  bash remote_linux/12_run_paired_guard_full.sh \
  --device cuda:0 --num-workers 4 \
  2>&1 | tee -a "$HOME/backdooranalog/runs/mmfi_paired_guard_full_s42_v1.startup.log"
```

Detach with Ctrl+B then D. Monitor without AnyDesk:

```bash
tail -n 30 "$HOME/backdooranalog/runs/mmfi_paired_guard_full_s42_v1.startup.log"
tail -n 20 "$HOME/backdooranalog/runs/mmfi_paired_guard_full_s42_v1/console.log"
```

The launcher accepts `--python`, `--data-home`, `DOSE_PY`, `DOSE_DATA_HOME`,
and runner options including `--source-cell`, `--outdir`, `--device`,
`--num-workers`, `--dry-run`, and `--fresh`. `--device` is singular; there
is no `--cells` or `--devices` scheduler option. The default output is the
separate directory `runs/mmfi_paired_guard_full_s42_v1`, preserving all
screening and historical results. Resume uses the same command and directory;
`--fresh` refuses a nonempty directory rather than deleting it.
Do not add `--fresh` after `--dry-run` in the same output directory: the dry run
has already created the immutable plan. Keep the historical source directory
and its parent `paired_guard.resolved.json`; resume revalidates both as well as
the copied frozen snapshot.

## Outputs and interpretation

The output directory retains `console.log`, `checkpoint.pt`,
`eval_cache.json`, `results.json`, and `main_metrics.json/csv/md`. Source
artifact hashes and guard/fallback provenance are retained for audit.

The eight primary metrics are clean MPJPE, clean PA-MPJPE, target-joint
T-MPJPE at d=1, and five clean relative PCK thresholds .5/.4/.3/.2/.1.
Errors are reported in mm; PCK thresholds are relative, not millimetres.
Six-dose evaluations are retained, including zero-dose controls. Digital
budget verification is not a claim of RF stealthiness or over-the-air
feasibility.

This is a single-seed full-data confirmation, not evidence of statistical
superiority and not a blinded first test: this candidate was selected after
earlier development. No full-run performance is known from implementation
alone. The screen's small T-MPJPE improvement may or may not persist with
full training; cleaner training behavior is not guaranteed by the failed
surrogate guard.

## Local implementation verification

On 2026-10-11, the complete local `ATKBackd/tests` suite finished with
1,278 passed and 7 skipped. Launcher Bash syntax and runner CLI help checks
also passed. A tiny synthetic end-to-end test exercises the real 50-epoch
trainer, frozen trigger, six-dose evaluation, export, and cached resume.
These checks validate implementation behavior, not numerical performance on
the official MM-Fi dataset; the full MICA experiment has not been run here.
