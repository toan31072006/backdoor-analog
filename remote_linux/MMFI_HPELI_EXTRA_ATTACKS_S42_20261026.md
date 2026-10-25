# MM-Fi HPE-Li extra attacks, seed 42: provisional record

Source: terminal output supplied by the user on 2026-10-26 from
`C:\Users\Van\backdooranalog\runs\mmfi_hpeli_extra_attacks_s42_4090_v1`.
Both worker logs reported `TRAIN_COMPLETE` at fixed epoch 49, and the user
confirmed `extra_attack_results.json` exists. The JSON, checkpoints, and
per-sample arrays have not been transferred to this workspace for an
independent artifact audit.

| Method | T-MPJPE@1 (mm) | Target-PCK@0.2 (%) | Clean MPJPE (mm) | Clean PA-MPJPE (mm) | Mean positive-dose DCE (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| smooth_lf_csi | 372.69 | 3.14 | 173.19 | 103.82 | 221.18 |
| tsba_adapted | 381.64 | 2.37 | 172.95 | 104.00 | 227.88 |

These are one-seed, exploratory HPE-Li/MM-Fi protocol1-s1 results. The
methods are CSI/pose adaptations and should not be described as unmodified
source-method reproductions. This record does not establish defense efficacy.

For the planned defense comparison, report clean MPJPE on the same target
joints `[2,3]` as triggered-to-true MPJPE, alongside all-joint clean utility,
attacker-target T-MPJPE and Target-PCK. Do not use all-joint clean MPJPE as the
recovery floor for a two-joint trigger. BadNets is excluded by the user's
2026-10-26 instruction; any A1-only defense table must be labeled as such and
must not claim the requested cross-attack baseline comparison is complete.
