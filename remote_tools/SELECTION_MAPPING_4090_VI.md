# Uniform/Diverse × Linear/Sqrt/Quad — RTX4090

## Material Passport

- academic-research-suite / experiment-agent; setup requested 2026-11-13.
- EXPLORATORY single-seed extension, not replacement/full completion of T1/T2/T3.
- No remote job, push, checkpoint import or full-MMFi/CUDA test performed by this setup.

## Frozen design

MM-Fi official P1S1/split0; TRAIN133056/TEST33264. HPE-Li seed42, 50epochs,
final49, SGD .001/momentum.9/wd0/B32/workers0, ordinary all-joint MPJPE,
FP32/fixedLR/no pretrained. Workers0 deliberately shared by all six new cells.
Native A1 md_multicarrier/eps.185, no added peak projection, poison10%/13305.
Historical leg rotation: pivot1, target[2,3], Z40°h(d), NOT forward trunk Bend.

Six independent scratch victims (no Clean): Uniform/Diverse crossed with
Linear h=d, Sqrt h=sqrt(d), Quad h=d². Frozen common original bend motion prior,
full TRAIN FPS on unmodified flattened poses exactly as historical selection.
No pose centering, filtering, downsampling, TEST-guided choice, trigger learning,
or FPS approximation. This FPS is O(N×13305×51), may take substantial CPU time;
it runs ONCE and is persisted, not repeated per mapping.

Uniform IDs and doses replay native RNG42 choice then U(.2,1). A separate RNG
copy supplies precisely this same vector to Diverse by selection rank. Every
mapping reuses selection IDs and doses. This keeps native Uniform semantics
but corrects the confounding of selection-dependent dose RNG consumption.
Each worker independently reseeds model/minibatch RNG; initial model hashes
must agree across all six. Completed cells are reused only in THIS bound run.

## Reuse decision (explicit, not silent)

No local historical Uniform–Linear checkpoint/config/poison/RNG replay has
been audited for this new workers0 profile. Therefore no old row is imported
by name, and Prepare requires **-FreshLinear** to explicitly opt into SIX
fresh matched victims. Without the flag it stops before expensive preflight.
If five new trainings plus an old reference is desired, STOP here and audit
the exact historical reference first; this launcher does not silently retrain
or assume it matches. Old runs/source/manuscript remain untouched.

## Execution on Van

Use a separate complete checkout/bundle under U:, NOT an actively training
checkout. Existing Python/data/prior stay on C:. No installation or global
PowerShell policy changes. The helper path below must point to the NEW code.

```powershell
$smRepo = 'U:\backdooranalog\code\SelectionMapping_20261113'
$smHelper = Join-Path $smRepo 'remote_tools\Start-SelectionMapping4090.ps1'
$smOut = 'U:\backdooranalog\runs\mmfi_selection_mapping_s42_v1'

# Read-only; no data arrays/CUDA/output writes
& $smHelper -Mode Inspect -Outdir $smOut

# Full CPU content audit, authentic prior check, full TRAIN FPS ONCE; no CUDA
& $smHelper -Mode Prepare -Outdir $smOut -FreshLinear
if ($LASTEXITCODE -ne 0) { throw 'Prepare failed; preserve logs/output.' }

# Run only when other training jobs have finished. Two real B32 forward graphs
# coexist at a barrier then backward/SGD; models discarded, NOT full training.
& $smHelper -Mode Capacity -Outdir $smOut
if ($LASTEXITCODE -ne 0) { throw 'Capacity failed; no automatic fallback/retry.' }

# First production launch only, max TWO workers on GPU0
& $smHelper -Mode Launch -Outdir $smOut
```

Run each stage separately, inspect errors, do not paste Launch after a failed
stage. Capacity is not assurance of convergence, 2× speed or absolute VRAM
sufficiency. The WDDM classifier is reused read-only; no process kill/suspend.
Keep terminal open and Windows awake. Controller creates no CUDA context.
On first worker failure no pending cell is launched; the owned active peer is
drained, not killed. No auto retry, batch shrink, AMP, missing-frame skipping,
checkpoint/plan editing, stale barrier or lock deletion.

Status from a second terminal:

```powershell
& $smHelper -Mode Status -Outdir $smOut
```

After an interruption, ensure old controller/workers have exited; explicit
Resume uses atomic epoch-boundary checkpoint +optimizer/Python/NumPy/Torch/
CUDA RNG. A crash before initial checkpoint is not silently restarted.

```powershell
& $smHelper -Mode Resume -Outdir $smOut
```

Prepare/Capacity are fresh-only; do not rerun over partial failures. Preserve
artifacts and diagnose. Native FPS has initial/final messages but no ETA; long
CPU time is not permission to terminate/reduce its sample pool automatically.

## Outputs/metric contract

`results.csv`, `results.json`, `results.md`: all six rows only after every
full-TEST final49 evaluation is verified. No partial success label.
Columns: selection/mapping, T-MPJPE@1, Target-PCK@.2@1, mean-positive DCE,
mean-positive nDCE, clean MPJPE, clean PA-MPJPE, NT-drift@1.

- T/clean/NT errors mm; Target-PCK percent, cleanGT scale joints5/12.
- DCE vector residual (pred_d-pred_0)-(target_d-GT), target joints2/3.
- nDCE = mean over d=.2,.4,.6,.8,1 of ratio of dataset/joint MEANS
  DCE(d)/Lambda(d). NOT per-sample ratios; d0 excluded/null. Still predictor=1.
- NT-drift compares triggered vs same victim no-trigger on non-target joints.
- Save per-dose Lambda/D/T/PCK/DCE/nDCE/no-trigger controls/Triggered-GT.
- Saved XYZ NPZ includes ordered pair IDs, full17joints, six doses and bindings.
- Scalars are computed from SAVED NPZ bytes; no prediction repair/alignment.
- Independent clean-trained+trigger control NOT_SUPPLIED; no Clean retraining.
- Uniform/Diverse CPU selection time measured separately on the same machine;
  not an isolated training-speed benchmark and no expected winner claimed.
- One seed only; no significance or across-seed SD; no physical-RF claim.

Artifacts: plan.json, inputs.json, selection.json, prepared.json, capacity.json,
smoke/, and <selection>_<mapping>/{console.log,checkpoint.pt,poison_manifest.json,
initialization.json,poses_full_test.npz,evaluation.json,complete.json}.
