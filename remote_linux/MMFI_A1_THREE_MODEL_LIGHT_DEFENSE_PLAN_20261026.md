# MM-Fi A1 defense pilot: three backbones, one RTX 4090

Decision recorded 2026-10-26: the Windows RTX 4090 host, seed 42, and A1 attack only. BadNets is excluded by the experiment owner. This is an **A1-only exploratory defense comparison**, not the full cross-attack comparison in the reviewer note. The manuscript calls it Table 4; the repository's older plan calls the defense group T5. Keep those scopes explicit when exporting results.

## Frozen comparison contract

- Dataset: MM-Fi protocol1-s1. Every defense of a backbone evaluates the same ordered official TEST frames. Each method starts independently from that backbone's poisoned A1 epoch-49 checkpoint. HPE-Li, MetaFi++ and WiFlow sources are authenticated separately; no model is retrained as an attack victim for this pilot.
- Serial rows: `none` (original victim), light FT, CLP, clean-statistics BNP, pose-adapted FST, and pose-adapted Neural Cleanse (NC). Fine-Pruning (FP), ANP, NAD and FT-SAM are outside this new light profile. BNP and FST are distinct from FP.
- FT: deterministic 5% trusted clean TRAIN bank; 10 epochs, SGD learning rate `1e-4`, physical batch 32, final prescribed epoch. This is a **new profile**, not the historical FT-100/B256 recipe; use new output folders and source/recipe hashes. Its clean selection split monitors training but does not choose checkpoints using TEST.
- CLP: predeclared weight-only threshold `u=3` and audited Conv2d/BatchNorm pair map. It does no repair training.
- BNP: one clean TRAIN forward sweep measures hidden BatchNorm2d input moments. Channels whose Gaussian KL from frozen running moments exceeds their layer mean plus `3` population standard deviations have affine weight and bias zeroed. Coordinate normalizers are excluded. This is an MM-Fi pose adaptation, not the original authors' code.
- FST: initialize the final coordinate head afresh, then optimize ordinary clean pose loss plus an inner-product penalty to the compromised head. **After each optimizer step**, project the head weight back to the original head's norm. The head is `regression.7` in HPE-Li, `decode.3` in MetaFi++, and `final_conv.3` in WiFlow. The pose implementation normalizes the penalty by the original head's squared norm and a frozen clean TRAIN loss; this differs from the original paper's raw inner-product term. Ten fixed epochs, SGD `1e-4`, `alpha=0.1`, the same trusted TRAIN split. In MetaFi++, `decode.3` is followed by BatchNorm2d, ReLU, pooling and BatchNorm1d before pose output, so this coordinate-head adaptation is not equivalent to modifying a terminal linear classifier. It is not an author reproduction.
- NC: the original method was designed for image classification and discrete class labels. This pose adaptation inverts a universal normalized-CSI mask/pattern for eight fixed kinematic targets (right/left leg, right/left arm, each at +45/-45 degrees). The predeclared budget is 64 physical-B8 optimizer steps per candidate per A1/clean model. Its anomaly score and recovered-trigger quality are **diagnostic** and evaluated on trusted TRAIN only. At most one physical-B32 unlearning epoch is allowed when the fixed TRAIN-only gate passes. If it fails, save `nc_diagnostic_only`; there is no repaired-model Table 4 row. A failure does not prove that A1 lacks a distributed trigger.
- NoDef, FT, CLP, BNP, FST and NC use separate artifact folders. The exact final checkpoint, ordered TRAIN/TEST input identities, action prior, code, repair recipe and GPU smoke are bound before launch. No automatic batch reduction or hyperparameter search using TEST.

## Table 4 columns and interpretation

All target-joint metrics use the same joints `J=[2,3]` and millimetres. For paired TEST frame `i`, defended model `f`, clean input `x`, triggered input `x'`, true pose `y`, and attacker target `t`:

| Column | Definition | Interpretation |
| --- | --- | --- |
| Clean MPJPE, all 17 | mean `||f(x)-y||` across all joints | Task utility; lower is better |
| Clean MPJPE, target 2 | mean `||f(x)_J-y_J||` | Clean floor on exactly the attacked joints |
| Triggered-to-GT MPJPE, target 2 | mean `||f(x')_J-y_J||` | Correct-pose recovery under A1; lower is better |
| Recovery gap | triggered-to-GT minus clean-target error on the same model/joints | Near zero is useful only when both errors are low |
| Change vs NoDef | paired difference in triggered-to-GT error from the undefended A1 victim | Negative means better recovery |
| T-MPJPE@dose1 | mean `||f(x')_J-t_J||` | Higher means less fit to attacker target |
| Target-PCK@0.2 | fraction of target joints within `0.2*||y[5]-y[12]||` of `t_J` | Lower means less attacker-target success |

The all-joint clean MPJPE is **not** the recovery baseline for a two-joint trigger. A low Target-PCK alone is insufficient evidence that the original pose was restored. Report clean utility cost relative to NoDef and the number of paired TEST frames. One seed supplies no across-seed standard deviation or confidence interval. The omission of BadNets prevents a cross-attack defense claim.

## Source and resource limits

- The HPE-Li Clean/A1 final49 source remains in the original run on Vân's machine. Its manifest, checkpoints and full evaluation must be checked in place.
- MetaFi++ Clean/A1 comes from the frozen 2026-10-24 tar archive; WiFlow Clean/A1 from the 2026-10-25 snapshot ZIP. Import and authenticate original manifests/checkpoints before creating defense outputs. Neither archive replaces raw MM-Fi.
- The earlier MetaFi++ batch-256 smoke on Vân's 4090 stopped at the 48-GiB free-RAM gate. The new batch-32 profile is a separate experiment, not a resumed or silently downgraded FT-100 run. Actual CUDA smoke must pass for all six methods on each backbone before full execution.
- Adapted BNP, FST and NC should be labeled as such in the manuscript. A candidate that produces a diagnostic only must not be counted as a successful repair.

## Windows RTX 4090 execution

The v3 runners are under `remote_tools/{hpeli_light_4090,metafi_light_4090,wiflow_4090}`. The wrapper `remote_tools/Start-MMFiThreeLightDefenses4090.ps1` serializes HPE-Li → MetaFi++ → WiFlow. Copy the WiFlow ZIP to Vân's Downloads even when an import already exists, because its archive checksum is retained in the new preflight receipt. The MetaFi++ tar is required when its authenticated import is absent. The HPE-Li original run stays in `backdooranalog/runs`.

Run `-Mode Prepare` for dry run, source/data preflight and **six** actual RTX 4090 memory smokes per model; no full defense starts. Run `-Mode Launch` only after all receipts pass; it registers one Scheduled Task. Run `-Mode CheckOnly` after completion to verify saved artifacts, then `Collect-MMFiThreeLightDefense4090.ps1` to replay the paired arrays and export Table 4. The task uses one GPU job at a time and no automatic retry.

Primary method references: [Neural Cleanse](https://bolunwang.github.io/assets/docs/backdoor-sp19.pdf), [its original code](https://github.com/bolunwang/backdoor), [CLP](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650171.pdf), [BNP / Pre-activation Distributions Expose Backdoor Neurons (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/76917808731dae9e6d62c2a7a6afb542-Abstract-Conference.html), [FST / Towards Stable Backdoor Purification through Feature Shift Tuning (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ee37d51b3c003d89acba2363dde256af-Abstract-Conference.html). BNP and FST results here are pose-regression adaptations.
