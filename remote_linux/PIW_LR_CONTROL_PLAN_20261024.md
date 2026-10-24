# Material Passport

- Date: 2026-10-24, Asia/Bangkok.
- User request: check whether lowering PiW learning rate improves the current
  body-axis target pilot; preserve v1/v2 and MM-Fi experiments.
- Mode: controlled code experiment, seed42, not a full paper table or blind TEST.
- Evidence: user-reported v2 full six-dose paired metrics; local source/tests.
- Status: prepared and locally tested; full GPU outcome pending on Vân.

## Fixed intervention

Compare the completed `piw_hpeli_body_axis_s42_v2` (AdamW lr.01) with new
`piw_hpeli_body_axis_s42_lr001_control_v3` (lr.001). LR is the only scientific
configuration change. Keep drop_last=True, B32, 200epochs/final199, scratch,
ordinary MPJPE, FP32, seed42/RNG/sampler, official ordered identities, target
`piw_body_hip_axis_v2`, hip7->[11,13], 40deg*d around normalize(GT5-GT7),
rho.1/eps.185, poison indices+doses, native amplitude operator, phase policy,
PCK@.2/ref6/4 and all six evaluation doses unchanged.

Train new matched Clean and A1 victims; do not warm-start or use a historical
Clean with a different recipe. Historical source bytes/receipts/artifacts stay
unchanged. New manifest/output/checkpoints/task and capacity gates are separate.

## Why check LR

V2 dose1 T407.39(no-trigger)->205.99mm, PCK.12->29.68%, DCE173.61 versus
stationary384.05mm show within-victim steering. Mean-positive DCE114.42 versus
stationary232.32mm supports keeping the target fixed for this control. These
values are user-provided, not a direct replay of Vân by the local agent.
Clean v2 MPJPE154.38mm versus v1 92.88mm motivates a recipe check, but does not
identify LR as the cause because v1/v2 also differ in drop_last and target.

## Required validation and interpretation

Dry-run, full CPU/data/model preflight, new concurrent B32 GPU smoke, then
explicit launch. Do not stop other jobs, shrink batches, retry failed training,
change the angle/axis/threshold, or overwrite old artifacts automatically.
Only fixed final199 victims are evaluated. Report Clean MPJPE/PA, A1 target
metrics, same-victim no-trigger, clean-trained+trigger, nontarget drift and DCE
against stationary response. Preserve per-sample predictions/GT/targets/IDs.

The read-only comparator replays both complete saved runs and requires matched
config (apart from LR/admin labels), data/poison/trigger, runtime/backend and
bitwise GT/targets/IDs/doses. Both source versions must authenticate. A positive
result applies to this seed/run condition; no across-seed SD/CI or general causal
guarantee is inferred. Prior TEST was viewed during development; this remains a
development-selected LR control. Do not continue selecting settings from TEST.

Commands and logs: `PIW_HPELI_LR_CONTROL_VI.md`.
