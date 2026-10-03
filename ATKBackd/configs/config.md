# Config Reference for Victim Models

> **Cập nhật:** repo này giờ chỉ giữ **MMFi + HPE-Li + micro-Doppler**.
> MetaFi++ và GraphPoseFi đã bị bỏ (xem `models/factory.py` để biết lý do),
> cùng các trigger SIG / Blended / WaNet. Mọi mục dưới đây nhắc tới chúng là
> tài liệu cũ, giữ lại để tham khảo — chúng KHÔNG chạy được nữa.

This document describes the configuration setup for the three victim models used in your experiments:

- `hpeli`
- `metafiplusplus`
- `graphposefi`

All configs are YAML files under `ATKBackd/configs/`. The active role of this
repository is MMFi; the per-model Person-in-WiFi-3D configs are legacy.

## Common settings across all three models

These fields are shared in the current configs.

- `dataset_root`: `/content/MMFI/Compress`
  - Base directory containing `E##/S##/A##` sequences.
- `experiment_name`: `mmfi`
  - Selects the MMFi feeder, 114×10 trigger, and 17-joint topology.
- `pretrained`: `false`
  - Model weights are trained from scratch, not loaded from ImageNet or other pretrained sources.

- `action_npy`: `/content/psba_dataset/data_bend.npy`
  - The target action skeleton used for the attack. In your current setup this is the `bend` target.
- `top_k`: `6`
  - Number of skeleton joints used to build the trigger.
- `aoa_spread`: `0.6`
  - Controls angle-of-arrival variation in the trigger.
- `eps`: `0.5`
  - Trigger strength scale.

- `pivot`: `11`
  - MMFi shoulder branch; descendants are joints 12 and 13.
- `theta_max_deg`: `20.0`
  - Maximum angle magnitude for the trigger.
- `dose_mode`: `linear`
  - Dose mapping: `linear` means trigger impact increases linearly with dose.

- `poison_select`: `uniform`
  - Poisoned samples are selected uniformly from the clean training set.
- `rho`: `0.3`
  - Poisoning rate: 30% of training samples are poisoned.
- `dose_min`: `0.2`
  - Minimum dose value for poisoning.
- `dose_max`: `1.0`
  - Maximum dose value.
- `dose_grid`: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
  - The dose grid evaluated during attack analysis.

- `tau_plaus`: `0.20`
  - Per-sample relative bone-length-error threshold.
- `max_target_residual_ratio`: `0.50`
  - Maximum residual between the predicted trigger effect and target effect.
- `max_nontarget_ratio`: `0.25`
  - Maximum non-target drift relative to the target-effect magnitude.

- `device`: `null`
  - Auto-selects CUDA if available, otherwise CPU.
- `data_parallel`: `false`
  - Disabled by default; training runs on a single device unless explicitly enabled.
- `num_workers`: `0`
  - No data loader multiprocessing for these configs.
- `seed`: `42`
  - Fixed seed for reproducibility.

## Per-model training settings

### `hpeli`

Active MMFi config: `ATKBackd/configs/mmfi/attack_bend.yaml`

- `model`: `hpeli`
- `batch_size`: `32`
- `lr`: `0.001`
- `epochs`: `50`
- `optimizer`: `SGD` for MMFi, matching DT-Pose HPELi
- `momentum`: `0.9`
- `weight_decay`: `0.0`

For MMFi, this is the DT-Pose HPELi scratch-training recipe. Person-in-WiFi-3D
uses AdamW at `lr=0.01`.

### `metafiplusplus`

Config file: `backdoorviet/configs/metafiplusplus/attack_ln.yaml`

- `model`: `metafiplusplus`
- `batch_size`: `32`
- `lr`: `0.001`
- `epochs`: `50`
- `weight_decay`: not specified (default `1e-4` in trainer)

This matches the HPELiNet settings, since `metafiplusplus` is tuned to the same batch size and lr in your current setup.

### `graphposefi`

Config file: `backdoorviet/configs/graphposefi/attack_ln.yaml`

- `model`: `graphposefi`
- `batch_size`: `256`
- `lr`: `0.0003`
- `epochs`: `50`
- `weight_decay`: `0.02`

GraphPoseFi is trained with a much larger batch size and stronger weight decay, reflecting its different architecture and optimization regime.

## How training is run

Training is executed through `ATKBackd/run_experiments.py`.

Typical command line:

```bash
python3 run_experiments.py --models hpeli --scenarios bend --triggers micro_dropper --dataset mmfi --seeds 42
```

Key flags:

- `--models`: only `hpeli` is available.
- `--scenarios`: select target actions: `bend`, `cross`, `nod`.
- `--triggers`: currently only `micro_dropper`. SIG, Blended and WaNet were
  removed because their image-domain scaling is invalid for MMFi amplitude CSI.
- `--dataset`: `mmfi` or `pwif3d`.
- `--epochs`: override the configured number of epochs.

### Config selection logic

- For dataset `pwif3d`, the script loads model-specific config folders:
  - `hpeli` → `configs/hpeli`
  - `metafiplusplus` → `configs/metafiplusplus`
  - `graphposefi` → `configs/graphposefi`

- For dataset `mmfi`, the script uses `configs/mmfi` and overrides `cfg['model']` for each victim model.

- `run_experiments.py` also contains model-specific override logic for batch size and epochs.

## Notes on training details

- The active configs use linear dose mapping.
- Bend, cross, and nod select their corresponding action files automatically.
- `hpeli` and `metafiplusplus` share the same optimization hyperparameters, while `graphposefi` uses a different training regime.

## Suggested config edits for other experiments

- To evaluate a different trigger target, change `action_npy` in the YAML:
  - `data_bend.npy`
  - `data_cross.npy`
  - `data_nod.npy`

- To run `graphposefi` on a different scenario, use the corresponding config file:
  - `configs/graphposefi/attack_ln.yaml`
  - `configs/graphposefi/attack_sqrt.yaml`
  - `configs/graphposefi/attack_quad.yaml`

- For `hpeli` / `metafiplusplus`, use:
  - `configs/hpeli/attack_ln.yaml`
  - `configs/hpeli/attack_sqrt.yaml`
  - `configs/hpeli/attack_quad.yaml`
  - `configs/metafiplusplus/attack_ln.yaml`
  - `configs/metafiplusplus/attack_sqrt.yaml`
  - `configs/metafiplusplus/attack_quad.yaml`

## Paths and files

- Config root: `ATKBackd/configs/`
- Model runner: `ATKBackd/run_experiments.py`
- Dataset loader: `ATKBackd/train_backdoor.py` / `ATKBackd/data_utils/feeder.py`
- Result outputs: `ATKBackd/experiments_out/`

If you want, I can also add a one-page `train.md` describing the exact command sequences for all three models and how to switch between `pwif3d` and `mmfi` runs.
