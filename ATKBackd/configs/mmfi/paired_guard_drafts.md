# Proposed paired clean-twin development screen (seed 42)

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan (approved implementation; no remote benchmark launched)
- Origin Date: 2026-10-11
- Verification Status: UNVERIFIED for real MM-Fi efficacy; code validation is separate
- Version Label: paired_guard_setup_v1
- Result status: `DRAFT_ONLY_NOT_PAPER_RESULTS`; every planned row is retained.

The hypothesis is that guarding Proposed against a clean surrogate at the same
training stage can reduce target-joint T-MPJPE without losing clean utility.
This implementation has no real MM-Fi result and makes no performance guarantee.

## Exactly three fresh victims

| Cell | Recipe |
| --- | --- |
| `blended` | Current unchanged Blended operator, alpha=0.2*d, with the common actual dual bounds |
| `lc_trainaware` | Current two-carrier winner, unchanged operator and fitting defaults |
| `lc_paired_guard` | Same two carriers, fit budget and ordinary victim task; paired clean-twin guarding |

There is no fourth clean victim. `clean_victim_probes` are unsupported and
explicitly not run in this experiment. Each of the three victims is independent
and freshly initialized with seed42; attacker weights are never transferred.

Common defaults: rho=.1, 15 victim epochs, 20,000 official-TRAIN frames, SGD
lr=.001/momentum=.9/weight_decay=0, batch32, ordinary full-pose MPJPE ERM.
Poison identities and paired Uniform(.2,1) trigger/payload doses are identical
across all three arms. Pivot1, target joints2/3 and the payload stay unchanged.
The same disjoint 4,096-frame official-TRAIN holdout screens every victim.
The official test is never loaded; the external holdout is never used for
attacker training or artifact selection.

Every arm has the same actual per-input/per-dose Original peak ceiling at
eps=.185 and relative-L2 ceiling .10*d. Blended retains alpha=.2*d and its
operator is not retuned. Bounds need not produce equal realized perturbations.

The new arm has exactly the current two-carrier pattern and initialization;
it does not introduce an eight-carrier bank. Fitting uses the existing 4,096
frame pool inside victim TRAIN, its internal 80/20 split, two independent
surrogates, 10 clean warmup epochs, three alternating rounds, 128 real SGD
updates before and after each 24-step outer block, and two-step differentiable
SGD/momentum lookahead. CLI overrides of shared fit compute are supported;
the control tolerances remain fixed at their existing defaults.

Each new-arm surrogate gets its own clean twin cloned after clean warmup,
including model parameters, BatchNorm buffers, optimizer momentum and SGD
settings. Paired real updates use the same ordered raw samples: the twin
receives clean CSI/clean pose, the surrogate receives the existing poison plan.
Paired real updates share the pre-update CPU and active-device CUDA RNG states
and preserve the poisoned surrogate's post-update RNG stream, matching dropout
draws without changing the existing trainaware update sequence.
Outer lookahead compares the poisoned surrogate to its clean twin at the same
stage. The twin is advanced using detached virtual clean SGD and its values
are detached in the utility hinges. Negative MPJPE, PA-MPJPE and each of five
soft-relative-PCK utility gaps activate no penalty; only degradation is penalized.

Candidate selection uses exact internal clean MPJPE, clean PA-MPJPE and all five
hard relative PCK thresholds against the current clean twin for each surrogate.
All seven internal tolerances are fixed to zero. Each surrogate must pass every
gate. If no eligible round exists, the best fallback is saved with
`no_eligible_candidate=true` and `selected_utility_gate_passed=false`; its row is
retained as a failed hypothesis. Internal gates never guarantee fresh-victim
utility or a better target T1.

This adds attacker compute for real and virtual clean-twin SGD. Fitting records
disclose clone proofs, shared ordered batch/update hashes, actual/virtual step
counts, guard activation counts and stage-matched validation references.
At the full default 4,096-frame fit pool (3,277 internal TRAIN frames), each
poisoned surrogate performs 1,798 real SGD steps. Its clean twin adds 768 real
and 144 virtual SGD steps per ensemble member; it is cloned after warmup and
does not repeat that warmup.
Frozen evaluation-mode BatchNorm lookahead is not full-training bilevel
optimization. These additional attacker steps do not change victim training.

## Reports and interpretation

`draft_summary.md/csv/json` exports all three rows and eight primary metrics:
clean MPJPE, clean PA-MPJPE, target-joint T-MPJPE at d=1, and clean relative
PCK .5/.4/.3/.2/.1. PCK thresholds are relative, not millimetres. ASR,
displacement, Spearman and I1 are not primary ranking metrics.

`blended_comparison.csv/json` compares both Proposed arms to fresh Blended on
every primary metric. Strict pointwise dominance requires lower T1 and no
worse result on any of the seven clean metrics. The separately labelled
tradeoff screen retains the existing limits of 5mm MPJPE, 3mm PA-MPJPE and
one percentage point at each PCK threshold. Passing that screen is not
pointwise dominance and is not statistical significance.

`dose_response.csv/json` retains all 18 method/dose records for d=0/.2/.4/.6/.8/1.
`input_distortion.csv/json` audits common holdout pairs and actual bounds unless
explicitly skipped. Digital distortion does not prove RF detectability,
over-the-air feasibility or stealthiness. The fitting audit remains available
in the summary JSON, including failed internal gates.

## Run on MICA after syncing code

Check GPU availability first; this command does not reserve GPUs. Device
indices passed to Python are process-visible logical indices. With physical
GPUs 0,1,3 selected by `CUDA_VISIBLE_DEVICES=0,1,3`, they become logical
cuda:0,cuda:1,cuda:2. Passing cuda:3 under that mask is invalid.
`CUDA_DEVICE_ORDER=PCI_BUS_ID` makes the physical ordering consistent with
the GPU indices displayed by `nvidia-smi`.

```bash
cd "$HOME/backdooranalog/code"
bash remote_linux/11_run_paired_guard_drafts.sh --dry-run
tmux new -s mmfi-paired
```

Inside that interactive tmux shell, run with a startup log so preflight errors
are preserved before per-cell logs exist:

```bash
cd "$HOME/backdooranalog/code"
set -o pipefail
mkdir -p "$HOME/backdooranalog/runs"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,3 \
  bash remote_linux/11_run_paired_guard_drafts.sh \
  --devices cuda:0 cuda:1 cuda:2 --num-workers 4 \
  2>&1 | tee "$HOME/backdooranalog/runs/mmfi_paired_guard_s42_v1.startup.log"
```

`--dry-run` resolves only metadata and does not check CUDA or dataset files.
Real runs preflight the visible device count and mask with an actionable
logical-index error before starting workers. The mask above applies only to
that command; neither runner nor launcher changes the global environment or
kills unrelated jobs. The shared scheduler runs independent victims, not
distributed training.

Detach tmux with Ctrl+B then D. Monitor from another terminal:

```bash
tail -n 12 "$HOME/backdooranalog/runs/mmfi_paired_guard_s42_v1"/*/console.log
```

The launcher accepts `--python`, `--data-home`, `DOSE_PY`, `DOSE_DATA_HOME`
and all Python runner options. Its default output is
`runs/mmfi_paired_guard_s42_v1`; use `--outdir` for a new experiment directory.

Resume using the same scientific command and output directory. Runtime device
and worker changes are allowed; source, science options, ordered input/action
identities and frozen artifact hashes are checked. Changed science or source
requires a new directory. `--fresh` refuses nonempty directories and never
deletes historical runs. There is no automatic retry or mid-fitting checkpoint;
interrupted fitting can repeat when final files do not exist. Partially written
artifact/record pairs fail closed and require a new directory.

```bash
bash remote_linux/11_run_paired_guard_drafts.sh --export-only --devices cpu
```

No remote benchmark or GitHub publication is launched by this setup. Synthetic
code checks do not establish MM-Fi effectiveness. One seed and repeated
development on a TRAIN holdout cannot support significance or a blinded-first-
test claim. Ordered input manifests do not hash every CSI/pose file byte.

## Local code validation (2026-10-11)

The full CPU test suite completed with 1,155 passed and seven skipped tests;
the 13 warnings concern CPU-only DataLoader pinning. New checks include a
finite-gradient smoke on real HPE-Li and a synthetic three-victim
fit/freeze/train/export/resume workflow. The CLI metadata-only dry run and
Linux launcher syntax check also passed. This validates code contracts, not
MM-Fi efficacy or the CUDA-specific path skipped on this machine. The
validation interpreter used the local NumPy 2.1.3 compatibility environment.
