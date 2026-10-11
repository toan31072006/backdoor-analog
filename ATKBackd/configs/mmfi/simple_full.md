# Minimal Proposed hypotheses: full MM-Fi, seed42, 50 epochs

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan and user-requested implementation
- Origin Date: 2026-10-11
- Verification Status: UNVERIFIED for official MM-Fi performance
- Version Label: mmfi_simple_full_v1
- Result label: `FULL_EXPLORATORY_SINGLE_SEED`, not publication superiority

## Question and fixed protocol

Does a simpler input operator or carrier structure improve target error without
damaging clean pose accuracy? There is no promised improvement. The purpose is
to test four independent hypotheses with full training, rather than adding
more guards, masks, learned carriers and optimization stages at once.

Every victim, including Clean and Blended, trains for **50 epochs on all
133,056 official MM-Fi TRAIN frames**. Evaluation uses all **33,264 official
TEST frames**, not a TRAIN holdout. The runner refuses shorter epochs, subset
options, old result imports, fitting options and relaxed budgets. No surrogate
is trained: these four operators are deterministic and fixed before training.
No weights transfer to the victim. Each method gets its own fresh victim,
same seed42 initialization and ordinary full-pose MPJPE ERM.

All six attack arms use rho=.1 (13,305 pairs), the identical uniform poison
IDs and identical paired Uniform(.2,1) trigger/payload doses. HPELi, SGD
lr=.001/momentum=.9/weight_decay=0, batch32, no scheduler/pretraining,
protocol1-s1/split-seed0, pivot1/target joints2,3 and the 40-degree global-z
payload remain unchanged. This is Proposed trigger development, not an
improvement to BadNets, Blended or WaNet. Only Blended is retrained here as a
fixed traditional reference; its existing operator and alpha=.2*d are intact.

## Seven independent rows

Let p0 be the Original zero-mean pattern, p1 the same orthogonalized auxiliary
DCT carrier as the fixed multi-carrier, and q their equal-weight unit-RMS mix.
Both are fixed at seed42. All native candidates are clipped to [0,1], then
receive the same actual postclip dual-budget projection.

| Key | Change relative to fixed multi-carrier |
| --- | --- |
| `clean` | Clean victim; zero poisoned samples and unmodified test inputs for the displayed clean/T metrics. An additional Original-trigger probe is retained in the raw cache, not this table. |
| `blended` | Unchanged Blended native operator under the shared dual ceilings. |
| `proposed` | Fixed equal-weight multi-carrier + Original peak bound, then shared L2 ceiling. This is **not** the historical learned paired-guard artifact. |
| `additive` | Keep q; replace x-times-gain with x + d*eps*RMS(x)*q. Tests whether a consistent direction across inputs is easier to learn. |
| `single` | Keep gain injection; use p1 alone instead of p0+p1. Tests whether the mixture is necessary. |
| `shared_links` | Keep gain; average q across antennas, broadcast it to all links, center/RMS-normalize. Tests whether antenna-specific variation is needed. |
| `frequency_only` | Keep gain and equal mixing; set only the auxiliary carrier's packet DCT mode to0 instead of1 before the unchanged orthogonalization. Projection against p0 can restore temporal components; this is not a strictly time-constant final carrier. |

No candidates are combined. None learn a mask, select favorable poison
samples, add a guard, change target poses or alter the victim loss. RMS
normalization makes carrier comparisons about direction, not nominal energy.
The additive RMS factor uses the current input only, not dataset statistics or
victim outputs. Shared-link averaging fails explicitly if its pattern becomes
degenerate; it never silently substitutes another method.

## Budget and evaluation

For every stored CSI input and dose, both bounds hold:

- actual peak <= Original's realized postclip peak on that same input/dose at eps=.185;
- relative L2 <= .10*d.

Projection only shrinks. These are common upper bounds, not equal realized
noise. The unchanged fixed control retains its native peak projection before
the shared dual projection; the new arms project once. This may cause small
float32 quantization differences even at identical direction, so the comparison
is one-factor in intended signal structure, not byte-identical projection paths.
The native Blended operator is unchanged, but the common dual projection
is a declared budget adaptation. This budget differs from the historical
peak-only six-method table; do not copy those old scores into this comparison.
The previous frozen paired-guard full result is retained separately, not
silently imported as the fixed control.

Main table: T-MPJPE at d=1, clean MPJPE, clean PA-MPJPE, and clean relative
PCK@.5/.4/.3/.2/.1. Errors use mm, PCK uses percent; the PCK thresholds are
relative, **not 50/40/30/20/10 mm**. Relative L2 is not a main table column;
it remains an internal budget and separately audited diagnostic.

All six dose evaluations stay in `dose_response.csv/json`. Every candidate's
eight per-metric differences versus fresh Blended and fixed multi-carrier are
retained in `comparisons.csv/json`; errors below0 and PCK differences above0
favor the candidate. No winner or final method is selected automatically.
Prefer substantial T-MPJPE reductions with preserved clean accuracy; a tiny
one-seed change is not sufficient evidence to justify more complexity. Keeping
an unsuccessful hypothesis in the audit does not make it part of the final
method. No run/result is deleted automatically.

## Launch on MICA

Check current GPU use first. The example selects physical0/1/3; change this
list if those GPUs are not available. After masking, they are logical0/1/2.
This runs one independent victim per GPU, not distributed training of one
victim. Seven full runs take three scheduling waves on three GPUs.

```bash
cd "$HOME/backdooranalog/code"
nvidia-smi
bash remote_linux/13_run_simple_full.sh --dry-run
tmux new -s mmfi-simple-full
```

Inside tmux:

```bash
cd "$HOME/backdooranalog/code"
set -o pipefail
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,3 \
  bash remote_linux/13_run_simple_full.sh \
  --devices cuda:0 cuda:1 cuda:2 --num-workers 4 \
  2>&1 | tee -a "$HOME/backdooranalog/runs/mmfi_simple_full_s42_v1.startup.log"
```

Detach with Ctrl+B then D. There is no fixed six-hour worker timeout. Do not
start a second scheduler on the same output directory. Resume with the same
command; existing compatible completed cells are cached, unfinished cells
resume their checkpoints. `--fresh` refuses a nonempty directory and never
deletes it. Do not use `--fresh` after a dry run in that same directory.

```bash
tail -n 25 "$HOME/backdooranalog/runs/mmfi_simple_full_s42_v1"/*/console.log
pgrep -af '[r]un_simple_full.py'
```

To run only selected hypotheses, use `--cells additive single` (all selected
victims still run full50). The plan retains all seven rows; the final summary
is exported only once every row finishes. No automatic extra clean training
is added to a selected-cell command. To re-export completed results:

```bash
bash remote_linux/13_run_simple_full.sh --export-only
cat "$HOME/backdooranalog/runs/mmfi_simple_full_s42_v1/main_metrics.md"
```

The launcher accepts `--python`, `--data-home`, `DOSE_PY`, `DOSE_DATA_HOME`,
`--outdir`, `--devices`, `--num-workers` and `--distortion-samples`.
There is deliberately no CLI option for epochs, sample caps or budget tuning.
Scientific source hashes pin the new plan; changing source requires a new
output directory. Old artifacts are not modified; reproducing an old pinned
plan may require checking out the exact original source revision separately.

## Outputs and limits

The new default root is `runs/mmfi_simple_full_s42_v1`. It contains
`simple_full.resolved.json`, per-cell input/poison passports, console logs,
checkpoints/results/caches, and `main_metrics.csv/json/md`,
`dose_response.csv/json`, `comparisons.csv/json`, `input_distortion.csv/json`.
The digital distortion audit uses the same sampled official TEST inputs for
every operator, performs no victim forward pass and asserts both caps for
every pair/dose. It does not establish RF detectability or OTA feasibility.

Seed42 and repeated benchmark development do not support confidence
intervals, statistical superiority, or a blinded-first-test claim. Official
TEST is not used in trigger fitting (there is no fitting) or parameter search.
All results remain exploratory; final validation requires a frozen method and
independent evidence. Implementation/unit tests are not official MM-Fi scores.
