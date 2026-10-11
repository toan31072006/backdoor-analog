# Minimal Proposed hypotheses on the frozen comparison setting

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: user-requested implementation and configuration audit
- Origin Date: 2026-10-11
- Verification Status: UNVERIFIED for new official MM-Fi performance
- Version Label: mmfi_simple_full_peak_v2
- Result label: `FULL_EXPLORATORY_SINGLE_SEED`, not publication superiority

## User-defined scope

Traditional methods remain exactly the adapted baselines already evaluated in
the full comparison. Improve **Proposed only**, on those baselines' setting.
Do not add a new baseline constraint, tune its trigger or train it again.

The previous `c9ede6f` profile incorrectly added a shared relative-L2 cap and
scheduled new control victims. This replacement removes that additional cap
from every new Proposed arm. It uses the historical peak-only setting and
reads historical controls without modifying their files. Do not treat scores
from the superseded dual-budget profile as peak-only scores.

## Frozen references

`--reference-run` points to the existing complete
`mmfi_extended_peak_s42_v1` directory. All six rows are retained:

| Historical key | Role |
| --- | --- |
| `badnets` | Unchanged adapted traditional baseline |
| `blended` | Unchanged adapted traditional baseline |
| `wanet_source` | Unchanged adapted traditional baseline, including native covers |
| `ftrojan` | Unchanged adapted traditional baseline |
| `fiba` | Unchanged adapted traditional baseline, including native cross samples |
| `proposed` | Previous fixed equal-weight multi-carrier reference |

No model is trained for those six rows. They are not copied into a new victim
checkpoint directory or relabeled as new runs. JSON/CSV provenance distinguishes
frozen results from new Proposed measurements. A missing, incomplete, changed,
dual-budget or incompatible reference is an error, never a reason to retrain
or silently substitute a baseline. Historical source hashes remain historical;
reusing verified scores is not a reproduction of their training.

The reference audit checks configurations, full/50-epoch caches, all six doses,
poison/input passports and unchanged baseline-operator source identities.
New candidates must use the same ordered splits, action bytes and paired
uniform poison plan as the frozen comparison.

## Only four fresh victims

Each candidate trains for **50 epochs on all 133,056 official TRAIN frames**
and is evaluated on all **33,264 official TEST frames**. No 15-epoch subset or
surrogate is used. Seed42, rho=.1, the 13,305 poison identities and paired
Uniform(.2,1) trigger/payload doses match the reference.

HPELi, ordinary full-pose MPJPE ERM, SGD lr=.001/momentum=.9/weight_decay=0,
batch32, no scheduler/pretraining, protocol1-s1/split-seed0, pivot1/target
joints2,3 and the 40-degree global-z payload stay unchanged.

Let p0 be Original's pattern, p1 its orthogonalized auxiliary DCT carrier,
and q their fixed equal-weight unit-RMS mix.

| Key | Independent Proposed change |
| --- | --- |
| `additive` | Keep q; replace multiplicative gain by x + d*eps*RMS(x)*q. |
| `single` | Keep gain; use p1 alone instead of the two-carrier mix. |
| `shared_links` | Keep gain; average q across antennas, broadcast and center/RMS-normalize. |
| `frequency_only` | Set only the auxiliary temporal DCT mode to0 before unchanged orthogonalization. This does not guarantee a time-constant final carrier. |

No changes are stacked. No masks, guards, gradients, poison re-selection,
new losses or learned fitting stages are added.

## Exactly the previous budget

Every candidate uses the unchanged `PeakBudgetTrigger`: its actual stored
postclip peak cannot exceed Original's peak on that same input/dose at
eps=.185. This is the **existing comparison adaptation**, not the native
image-classification paper pipeline. Native Blended remains alpha=.2*d and
keeps precisely its previously declared peak-only adaptation.

There is **no relative-L2 ceiling**, no L2 matching, and no new shrink applied
to a historical baseline. L2 can still be measured in a separate diagnostic;
it is neither a ranking column nor an enforced new condition.
The historical fixed multi-carrier retains its original native projection;
float32 projection paths need not be byte-identical to the new arms.

Main metrics remain T-MPJPE at d=1, clean MPJPE, clean PA-MPJPE and clean
PCK@.5/.4/.3/.2/.1. Errors use mm; PCK thresholds are relative, not mm.
All six doses remain in the export. All four candidate deltas against frozen
Blended and previous Proposed are reported without automatic winner selection.

## Launch on MICA

Do not pull changed scientific source into an already running old-profile job.
This new output directory is separate from `mmfi_simple_full_s42_v1`.
Do not delete or overwrite that old directory or any historical results.

After confirming no old-profile worker is running in this checkout:

```bash
cd "$HOME/backdooranalog/code"
git pull --ff-only
nvidia-smi
bash remote_linux/13_run_simple_full.sh --dry-run \
  --reference-run "$HOME/backdooranalog/runs/mmfi_extended_peak_s42_v1"
tmux new -s mmfi-simple-peak
```

Inside tmux, if physical GPUs0/1/3 are available:

```bash
cd "$HOME/backdooranalog/code"
set -o pipefail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,3 \
  bash remote_linux/13_run_simple_full.sh \
  --reference-run "$HOME/backdooranalog/runs/mmfi_extended_peak_s42_v1" \
  --devices cuda:0 cuda:1 cuda:2 --num-workers 4 \
  2>&1 | tee -a "$HOME/backdooranalog/runs/mmfi_simple_full_peak_s42_v2.startup.log"
```

The mask maps physical0/1/3 to logical0/1/2. Only four new jobs are scheduled;
on three GPUs they occupy two waves. Detach with Ctrl+B then D.
Resume with the same command and unchanged source/reference files. `--fresh`
refuses nonempty directories and never deletes results.

```bash
tail -n 20 "$HOME/backdooranalog/runs/mmfi_simple_full_peak_s42_v2"/*/console.log
pgrep -af '[r]un_simple_full.py'
```

The launcher supports `--reference-run`, `--outdir`, `--cells`, `--devices`,
`--num-workers`, `--distortion-samples`, `--python`, `--data-home`, `DOSE_PY`
and `DOSE_DATA_HOME`. There are no short-epoch, subset or budget-tuning options.
Use the actual historical directory if its name differs from the example.
The main summary waits until all four candidates finish.

## Outputs and limits

Default root: `runs/mmfi_simple_full_peak_s42_v2`. It contains the plan,
candidate input/poison passports, logs, checkpoints, results/caches and
`main_metrics.csv/json/md`, `dose_response.csv/json`, `comparisons.csv/json`,
`input_distortion.csv/json`. The main table keeps every historical row plus
every new hypothesis. Digital distortion is measured only for new candidates,
on identical sampled TEST inputs without an HPE forward.

Single-seed, development-selected benchmark results are exploratory. They do
not establish statistical superiority, RF stealthiness or OTA feasibility.
No historical score is hidden because it is better than a new candidate.
An unsuccessful exploratory candidate need not become part of the final
method; its audit record remains intact.
