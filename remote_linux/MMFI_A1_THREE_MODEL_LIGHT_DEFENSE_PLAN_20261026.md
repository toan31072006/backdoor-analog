# MM-Fi A1 defense pilot: three backbones, one RTX 4090

Decision recorded 2026-10-26: the user selected the Windows RTX 4090 host,
seed 42 and A1 only. BadNets is excluded. This is an **A1-only exploratory
defense comparison**, not the complete cross-attack comparison requested in
the reviewer note. The manuscript calls this Table 4; the repository's
original experiment plan calls the defense group T5. Keep both names visible
when exporting results.

## Frozen comparison contract

- Dataset: MM-Fi protocol1-s1, identical official TEST identities for every
  defense of a backbone. A1 victim is the independently trained final epoch
  49 checkpoint for HPE-Li, MetaFi++, or WiFlow; no victim retraining.
- Defense rows: `none` (original poisoned checkpoint), short trusted-clean
  fine-tuning, fixed-threshold CLP, and `NC-adapted + unlearning`. Run one
  method at a time on the 4090. Do not reuse a repair from another backbone.
- The fine-tuning profile is a **new light profile**: a deterministic 5% trusted
  clean TRAIN bank, 10 epochs, SGD learning rate 1e-4, physical batch 32,
  final prescribed epoch. This differs from the historical 100-epoch,
  batch-256 defense profile and must use new output and binding hashes. No
  TEST-based choice of learning rate, epoch or pruning strength.
- CLP uses the predeclared, weight-only channel threshold `u=3` and the
  backbone-specific audited Conv2d/BatchNorm pair map. It does no repair
  training. Any unsupported layer is disclosed, not silently mapped.
- Neural Cleanse is originally a class-label, image-trigger method. For pose
  regression, inversion uses eight fixed pose-target hypotheses (right/left
  leg and right/left arm, each at +45 and -45 degrees) and a
  universal normalized-CSI mask/pattern on trusted TRAIN. Report candidate
  norms, inversion quality and anomaly diagnostic separately. Only a model
  actually fine-tuned on recovered-trigger/true-pose pairs can receive the
  `NC-adapted + unlearning` **repair** row. This is an adaptation, not a
  reproduction of the original classifier algorithm. The exact candidate
  family, training budget and inversion quality gate must be sealed in the
  run manifest before TEST is evaluated; no negative result implies the
  absence of a distributed A1 trigger. The inversion budget is 64 physical-B8
  optimizer steps per candidate per A1/clean model, followed by at most one
  physical-B32 unlearning epoch. If the TRAIN-only gate fails, save a
  diagnostic but no repaired-model Table 4 row.
- NoDef, fine-tuning, CLP and NC must start from the *same* frozen A1
  checkpoint of a backbone. Methods use separate output folders. A full
  data/source/checkpoint hash audit and actual one-GPU smoke precede launch.

## Table columns and interpretation

All target-joint metrics use joints `J=[2,3]` and millimetres. On each paired
TEST frame, with defended model `f`, clean input `x`, triggered input `x'`,
true pose `y` and attacker target `t`:

| Column | Definition | Defense direction |
| --- | --- | --- |
| Clean MPJPE, all 17 | mean `||f(x)-y||` over all joints | low; utility retained |
| Clean MPJPE, target 2 | mean `||f(x)_J-y_J||` | low; recovery floor |
| Triggered-to-GT MPJPE, target 2 | mean `||f(x')_J-y_J||` | low; real pose restored |
| Recovery gap | triggered-to-GT minus clean-target error on the same model and joints | near zero **with both errors low** |
| Change vs NoDef | paired triggered-to-GT error of defense minus NoDef on the same frames | negative is improvement |
| T-MPJPE@dose1 | mean `||f(x')_J-t_J||` | higher means less attacker-target fit |
| Target-PCK@0.2 | fraction of target joints within `0.2*||y[5]-y[12]||` of `t_J` | lower means less attacker-target fit |

The all-joint clean MPJPE is **not** the recovery floor for a two-joint
trigger. A small recovery gap alone is insufficient if both clean and
triggered-to-GT errors are large. Report clean utility cost relative to NoDef
and the number of paired TEST frames. One seed (`n=1`) supplies no across-seed
standard deviation or confidence interval.

## Sources and practical limits

- HPE-Li Clean+A1 source remains in the original run on Vân's machine;
  verify its final-49 checkpoints and manifest in place.
- MetaFi++ Clean+A1 is in the frozen 2026-10-24 tar archive; WiFlow Clean+A1
  is in the 2026-10-25 snapshot ZIP. Both must be transferred/imported and
  authenticated against their original source manifests before a new defense
  output is created. The archives do not replace raw MM-Fi data.
- Earlier MetaFi++ batch-256 smoke on the 4090 was blocked by the 48 GiB
  available-RAM gate. The new batch-32 profile is a different experiment,
  not a resumed or silently downgraded historical defense run. Actual CUDA
  smoke must still pass for each model/method. No automatic retry or fallback.
- Omitted historical ANP/NAD/FT-SAM runs remain separate results. Their
  omission from this limited-memory pilot prevents a claim of comprehensive
  defense benchmarking. BadNets omission prevents a claim of cross-attack
  robustness; do not substitute another method after observing scores.

## Vân RTX 4090 runner

The new runners live under `remote_tools/{hpeli_light_4090,metafi_light_4090,wiflow_4090}`.
They use separate v2 output directories and are orchestrated serially by
`remote_tools/Start-MMFiThreeLightDefenses4090.ps1`. Copy the authenticated
MetaFi++ tar and WiFlow ZIP to Vân's `Downloads`; the original HPE-Li three-table
run stays in `backdooranalog/runs`. `-Mode Prepare` performs dry run, source/data
preflight and four actual 4090 memory smokes per model, without launching full
defenses. `-Mode Launch` creates one persistent scheduled task that runs the
three backbones in order. `-Mode CheckOnly` verifies finished artifacts and
paired scalar replay. No two repairs share the single GPU.

The original classifier Neural Cleanse algorithm is **not** claimed to work
unchanged on this CSI regression task. A failed recovery is an experimental
negative result, not a reason to tune thresholds on TEST or fill a result row.

Primary references: [Neural Cleanse paper](https://bolunwang.github.io/assets/docs/backdoor-sp19.pdf),
[original code](https://github.com/bolunwang/backdoor),
[CLP paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650171.pdf).
