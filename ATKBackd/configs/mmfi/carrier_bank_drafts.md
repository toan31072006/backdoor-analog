# Proposed carrier-bank development screen (seed 42)

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run (implementation setup; no remote benchmark launched)
- Origin Date: 2026-10-10
- Verification Status: UNVERIFIED for MM-Fi efficacy; synthetic checks recorded separately
- Version Label: carrier_bank_setup_v1

- Stage: user-approved implementation and synthetic validation, NOT benchmark results.
- Hypothesis: a better-trained surrogate and richer carrier basis can lower target-joint T-MPJPE versus unchanged Blended without losing clean utility.
- Data access: official MM-Fi TRAIN only; no official test selection.
- Result status: `DRAFT_ONLY_NOT_PAPER_RESULTS`; every planned row is retained.
- Verification: local synthetic checks do not establish a real MM-Fi improvement.

## Six fresh victims

| Cell | Changed part of Proposed |
| --- | --- |
| `clean` | Unpoisoned fresh victim control |
| `proposed` | Unchanged fixed two-carrier Proposed under the shared dual budget |
| `blended` | Unchanged adapted Blended, alpha=0.2*d, shared dual budget |
| `lc_trainaware` | Stronger two-surrogate fitting, current two carrier bases |
| `lc_bank` | Same fitting plus eight deterministic carrier bases |
| `lc_bank_guard` | Same eight bases plus surrogate clean MPJPE/PA-MPJPE/PCK protection |

All attacks use rho=.1, identical uniform poison identities and paired U(.2,1)
trigger/payload doses. Victim seed42, 15 epochs, 20,000 TRAIN frames, SGD
lr=.001/momentum=.9, batch32, ordinary full-pose MPJPE ERM. All victims are
independent fresh models; attacker parameters/weights are never transferred.
Evaluation is on the same 4,096 disjoint official-TRAIN holdout frames.

Every trigger has the same actual per-input/per-dose Original peak ceiling
(eps=.185) and relative-L2 ceiling .10*d. This does not mean equal realized
perturbation. Blended's source-style operator is not retuned or weakened.
The previous full-data/.4-poison and six-method tables are not overwritten.

Attacker fitting uses a 4,096-frame pool INSIDE the victim TRAIN subset,
split internally 80/20. Defaults: two independently initialized surrogates,
10 clean warmup epochs, 3 alternating rounds, 128 real SGD updates before
and after each 24-step carrier update block. A two-step differentiable SGD
lookahead copies real momentum and weight decay, but uses cloned frozen BN
buffers: it is not exact full-training bilevel optimization. Extra attacker
compute versus fixed-trigger controls is recorded explicitly.
With the default 3,277 internal TRAIN frames this is 1,798 actual surrogate
SGD steps per ensemble member, plus 72 attacker outer updates. It is not
extra victim training or a changed victim objective.

The bank starts from the same original two-carrier pattern. The additional
six smooth separable carrier modes initially have zero weight; only the
attacker learns their signed mixing coefficients. Both bank arms use the
same basis/initialization and fitting budget. The guard arm alone adds clean
PA-MPJPE/soft-relative-PCK penalties and exact internal validation gates.
Gate failure is recorded, not presented as a passed candidate. A surrogate
gate never guarantees clean utility on the fresh victim.

## Primary results

`draft_summary.md/csv/json`: clean MPJPE, clean PA-MPJPE, triggered target-joint
T-MPJPE at d=1, relative PCK .5/.4/.3/.2/.1 (NOT mm).

`blended_comparison.csv/json`: all Proposed arms minus fresh Blended on each
main metric. Negative error deltas and positive PCK deltas are favourable.
`pointwise_dominates_blended` requires lower T1 and no worse value on ANY of
the seven clean metrics. This is a one-seed numerical comparison, not a
significance test. Separately, a lower-T1-with-tradeoff screen allows at most
5mm MPJPE, 3mm PA-MPJPE, and 1 percentage point at each PCK threshold by
default; passing this relaxed screen is NOT dominance.

Six doses remain in `dose_response.csv/json`. Digital distortion and
clean-victim trigger probes are separate diagnostics; no ASR, displacement,
Spearman or I1 is used as a main ranking metric. Digital distortion is not
proof of RF stealthiness or over-the-air feasibility.

## Run on MICA after syncing code

Check current GPU availability first; devices below are examples, not a
reservation. The scheduler runs independent victims, not distributed training.

```bash
cd "$HOME/backdooranalog/code"
bash remote_linux/10_run_carrier_bank_drafts.sh --dry-run
tmux new -s mmfi-bank
bash remote_linux/10_run_carrier_bank_drafts.sh \
  --devices cuda:0 cuda:1 cuda:3 --num-workers 4
```

Detach tmux using Ctrl+B then D. In another terminal:

```bash
tail -n 12 "$HOME/backdooranalog/runs/mmfi_carrier_bank_s42_v1"/*/console.log
```

Use the SAME command/settings to resume after an interruption. Source,
scientific options, action bytes and frozen artifact hashes are checked;
changed settings/source require a NEW output directory, not deletion or
replacement of historical results. No automatic crash retry is performed.
There is no mid-fitting checkpoint: an explicit user-started rerun repeats
interrupted fitting when no final fitting/artifact files exist. Partially
written final artifact/record pairs fail closed and require a NEW directory.

```bash
bash remote_linux/10_run_carrier_bank_drafts.sh --export-only --devices cuda:0
```

There is no automatic GitHub push or remote training launch from this setup.
An improvement remains unverified until the experiment is run. Repeated
development on this holdout must not be described as a blinded first test.

## Local verification (2026-10-10)

- Full repository regression: **1,055 passed, 7 skipped** in 276.43 seconds.
- Environment: Windows, CPU PyTorch 2.9.1, a separate pure NumPy 2.1.3 wheel,
  OMP/MKL threads=1. No user Python installation was changed.
- The full final run used a new Windows TEMP directory; an earlier workspace
  temporary-dir attempt stopped because drive E ran out of space. Only the
  fixtures generated by this task were cleaned, and source integrity was rechecked.
- Synthetic six-arm workflow verifies TRAIN-only fitting, two independent
  surrogates, freeze/fresh victim, common poison/dose identities, strict digital
  budgets, export of all rows/doses and tamper rejection on resume.
- Actual HPELi smoke validates finite two-step momentum/PA/PCK carrier gradients
  and unchanged model BN buffers/optimizer state during lookahead.
- Linux shell syntax and metadata-only six-recipe CLI dry-run passed.
- Real MM-Fi/GPU experiment: **NOT RUN**. Beating Blended remains UNVERIFIED.
