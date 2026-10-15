# MM-Fi Proposed factorial ablation — frozen seed-42 result

## Material Passport

- Origin: user-provided MICA result ZIP, checked on 2026-10-15.
- Verification: ANALYZED — archive/config/log/cache consistency, not a fresh model rerun.
- Source: `mmfi_proposed_factorial_ablation_s42_v1_results.zip`.
- ZIP SHA256: `d8fa8fed595f968ad58bf91c12e4b72e102b73ab43e29b631c46aa7b1fb5fdaa`.
- Plan SHA256: `3f4799c05797c760ea475329e51d6f7364e68abf3ac5144034e49a509919ff71`.
- Profile: `mmfi_proposed_factorial_ablation_v1`; code snapshot `b1eaf4d`.

All four victims completed 50 epochs (0–49), seed 42, full MM-Fi protocol1-s1
(133056 TRAIN / 33264 TEST), rho=0.1 (13305 paired-dose poison inputs).
No arm has a peak projection. HPELi from scratch, SGD lr=.001, momentum=.9,
weight decay=0, batch size=32; trigger dose U(.2,1), pivot 1, target joints
[2,3], Z-axis maximum rotation 40 degrees. Six evaluation doses are retained.

| Variant | Auxiliary carrier | Dose coupling | T-MPJPE@1 (mm) ↓ | Mean positive-dose DCE (mm) ↓ | Clean MPJPE (mm) ↓ |
| --- | :---: | :---: | ---: | ---: | ---: |
| **A1 — Full Proposed** | ✓ | ✓ | **155.398** | **74.225** | 174.684 |
| A2 | ✓ | ✗ | 215.078 | 103.342 | 174.233 |
| A3 | ✗ | ✓ | 175.093 | 102.483 | 174.020 |
| A4 | ✗ | ✗ | 245.027 | 137.666 | **173.985** |

A1 is best in endpoint target error and mean positive-dose DCE in this run,
not in clean accuracy or at every individual dose. A2 has lower target error
and DCE at d=.2. A1's clean PA-MPJPE is 104.599646961329 mm.

DCE compares `(pred_d-pred_0)` with `(target_d-true_0)` on target joints;
the summary averages the five positive doses. It is not normalized schedule MAD.

## Validation scope and limits

33 ZIP members passed CRC; all four cache/result pairs and CSV/JSON metrics
match. All 19 recorded source hashes matched the local snapshot at audit time.
Poison indices/doses and ordered split identities match across all four arms;
shuffled payload assignments replay the recorded RNG. No covers were used.

The 11 statistical/methodological risk categories were screened. Important
limits: one seed has no seed-level uncertainty; aggregate means do not imply
every subject/frame benefits; development selected the method previously;
realized Linf/L2 differ across carrier choices; no raw-data/checkpoint rerun
or RF over-the-air validation was performed. This is not statistical proof
of superiority or an equal-noise-budget comparison.

Do not substitute old rho=.4 results or peak-projected baseline results into
this table. A1 can serve as a separately audited frozen reference for new
native-operator controls; do not relabel its cache as a newly trained run.
