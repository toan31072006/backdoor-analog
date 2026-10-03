# Analog (Dose-Response) Backdoor against WiFi-based Human Pose Estimation

Implementation of a data-poisoning backdoor on WiFi-CSI human pose estimation whose
**payload magnitude is a continuous function of the trigger's dose** (impossible in a
classification backdoor). The kinematics-derived micro-Doppler trigger's intensity
sets, via forward-kinematics, the displacement of a chosen limb in the predicted
skeleton. See `wifi_hpe_backdoor_design.md` for the full design rationale.

## Layout

```
attack/
  trigger.py     dose-parameterized micro-Doppler antenna-differential trigger
  payload.py     FK joint-localized dose->rotation payload (bone-length preserving)
  poison.py      poisoning pipeline + dataset wrapper + eval modes
models/
  hpeli.py       HPE-Li victim (offline)         sk_network.py
  metafiplusplus.py      MetaFiPlusPlus victim (needs torchvision; pretrained weights need network)
  channel_trans.py  factory.py
data_utils/
  feeder.py      MMFi feeder plus legacy Person-in-WiFi-3D support
  synth_dataset.py  synthetic real-format data + synthetic action (for smoke test)
eval/
  metrics.py     MPJPE/PA-MPJPE/PCK + dose-response (Spearman, step-contrast) + ASR
train_backdoor.py  main train+eval driver
verify_joints.py   prints bone tree / candidate sub-chains (run FIRST)
smoke_test.py      end-to-end pipeline test on synthetic data
configs/mmfi/attack_bend.yaml   default MMFi/HPELi experiment
```

## Running

`run_experiments.py` is the canonical two-dataset runner for both **MM-Fi** and
**Person-in-WiFi-3D**. On the MICA/Linux server, use the sibling
`remote_linux/` launchers; they inject machine paths through CLI overrides and
validate the complete paper contract before training. The old direct-run config
folders remain for reference and should not be used as the server entry point.

1. Prepare MMFi as `E##/S##/A##/ground_truth.npy` plus
   `wifi-csi/frame###_processed.npy` (CSI shape `(3,114,10)`).
2. Set `dataset_root` and `action_npy` in `configs/mmfi/attack_bend.yaml`.
3. Single HPELi run: `python train_backdoor.py`.
4. Sweep: `python sweep.py --theta 20 30 40 --rho 0.1 0.2 0.3`.
5. Experiment matrix: `python run_experiments.py --dataset mmfi --models hpeli`.

## TSBA-adapted baseline

For the selected bend comparison (`theta=40`, `rho=0.4`):

```bash
python train_backdoor.py \
  --config configs/mmfi/attack_tsba_bend.yaml \
  --ckpt-dir experiments_tsba_mmfi/hpeli_bend_tsba_s42
```

To produce matched micro-Doppler/TSBA rows through the experiment runner:

```bash
python run_experiments.py --dataset mmfi --models hpeli \
  --scenarios bend --triggers micro_dropper tsba --seeds 42 \
  --outdir experiments_trigger_comparison
```

This is **TSBA-adapted (digital, sample-specific)**.  It retains the original
public code's `tanh` generator, multiplicative 10% bound, Adam generator and
alternating generator/victim training.  Conv1D classification is replaced by a
compact Conv2D generator over `(subcarrier, packet)`, while cross-entropy is
replaced by target-pose MPJPE.  HPELi's MMFi SGD recipe is unchanged.

The MMFi HPELi path matches DT-Pose scratch training: original `.view` tensor
semantics, SGD (`lr=1e-3`, momentum `0.9`), MPJPE loss, and seed `42`. MAE
pretraining belongs to DT-Pose's transformer victim and does not apply to HPELi.

## Honest scope / caveats
- Digital attack: micro-Doppler is injected into stored CSI before normalization;
  TSBA is injected into the normalized victim input to preserve its gradient.
  Sanitization does not re-run in either path. The
  antenna-differential micro-Doppler structure is kept for physical plausibility and to
  stay distinct from an additive (BadNet) trigger (pilot: common-mode survival 0.008 vs
  0.66 antenna-differential — relevant if a physical variant is attempted later).
- ASR is baseline-corrected and per-sample: an unchanged victim has target-residual
  ratio 1 and cannot count as a targeted success with the default threshold 0.5.
- MetaFiPlusPlus needs ImageNet-pretrained ResNet weights (`pretrained: true`) for intended
  performance — that download requires network access.
- The smoke test's synthetic CSI validates wiring only; all attack-success numbers are
  meaningful only after a real-data training run.
```
```
