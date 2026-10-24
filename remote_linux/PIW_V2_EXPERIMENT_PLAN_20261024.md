# Material Passport

- Task: user-authorized reconfiguration of failed PiW A1 pilot, 2026-10-24.
- Type: exploratory code experiment; one seed, revised target and recipe.
- Inputs: existing repository/DT-Pose sources, user-reported v1 controls, local
  original PiW archive TRAIN geometry. No new external data upload.
- Status: implementation/local validation; full GPU efficacy unverified.

This sidecar amends only the PiW pilot described in section0 of
`MMFI_CVPR_EXPERIMENT_PLAN_20261015.md`; T1/T2/T3/T5 MM-Fi stay unchanged.
The old PiW `.001/global-Z40` source and artifacts are read-only. V2 follows
`PIW_HPELI_V2_VI.md`: same split/seed/trigger/dose/metrics, AdamW.01/drop_lastTrue,
body-relative lateral-axis target, newly matched scratch Clean+A1 controls.
No original-author reproduction, full comparison, or multi-seed paper table claim.

Why revise: reported dose1 T95.78(no-trigger)→110.75(trigger), DCE90.95 versus
stationary63.35mm do not show desired control. A TRAIN-only 300-frame geometry
audit of the local raw archive gave old Z40 motion58.95mm; a lateral hip-axis40
motion386.14mm. These are geometry diagnostics, not model efficacy or proof that
the Drive mirror is byte-identical. Full input audit remains required on Vân.

Predeclare before v2 TEST: pivot7, targets11/13, axis normalize(GT5−GT7), positive
40°d; PCK .2×clean-GT elbow distance6/4; rho.1, eps.185, pairedU(.2,1), six doses.
Changed target means v1/v2 absolute target metrics are different tasks. TEST was
already viewed during development; disclose this. Do not select axis/angle/seed
or relaxed PCK thresholds using new TEST scores. Simultaneous recipe/target
revision cannot isolate causality. Further tuning must use TRAIN-only diagnostics.

Required controls/artifacts: clean-trained same-recipe model, A1 no-trigger,
same-victim triggered predictions, clean-model+trigger predictions, six dose
curves, stationary DCE, nontarget drift, complete per-sample NPZ, poisoned IDs and
doses, atomic checkpoint+optimizer+RNG, source/runtime/full input identities.
Only completed final199 victims may be evaluated. No old warm-start, TEST-based
checkpoint selection, automatic retry/AMP/batch shrink or changes to historical runs.

Before full training: metadata dry run, full CPU/data/model preflight, inspect
TRAIN geometry diagnostics, concurrent actual B32 CUDA smoke; explicit launch
afterward. Record failed gates honestly. Local CPU tests do not certify CUDA fit.
