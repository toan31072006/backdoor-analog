## Material Passport

- Origin Skill: experiment-agent (academic-research-suite)
- Origin Mode: run — implementation verification, not full-data training
- Origin Date: 2026-10-08
- Verification Status: UNVERIFIED for full MM-Fi experiments; local code checks passed
- Version Label: mmfi_tables_impl_v1

## Experiment Result

- ID: mmfi_seed42_pipeline_checks_20261008
- Type: generic / code verification
- Status: completed (local tests and previews only)
- Working Directory: E:/wifi/backdoor-analog/ATKBackd
- Duration: 48.37 seconds (final pytest execution)
- Exit Code: 0

Executed in PowerShell, with process-local MKL_THREADING_LAYER=SEQUENTIAL and
OMP_NUM_THREADS=1:

```text
C:/Users/Admin/anaconda3/python.exe -B -m pytest -p no:cacheprovider -p matplotlib -q tests --basetemp E:/wifi/_mmfi_final_tests_20261008_672ce9
```

Environment: Python3.13.5, PyTorch2.9.1+cpu, NumPy2.1.3, SciPy1.15.3, CUDA unavailable.
The intended server environment is separate; this does not establish GPU or
dataset-scale reproducibility on MICA.

### Output summary

205 tests passed; 5 existing native-Linux-Bash launcher tests skipped on Windows.
Real Matplotlib plots were exercised. The new Linux wrapper passed Git-Bash syntax
validation and a no-data/no-GPU dry-run with ten resolved cells. git diff --check
also passed.

Checks cover three traditional adapters, two real staged RF training workflows
with tiny model/data fixtures, bit-for-bit interrupted/resumed RF runs, correct
spatial encoder geometry, same-target per-dose baselines, shuffled marginals,
WaNet covers, spawned worker labels, fail-closed resume/cache validation, GPU-cell
scheduling, OS locks, and all four table exports. None used full MM-Fi data.

### Output files

- [Runner](ATKBackd/run_mmfi_tables.py)
- [Exporter](ATKBackd/mmfi_tables.py)
- [RF staged trainer](ATKBackd/train_rf_backdoor.py)
- [Traditional adapters](ATKBackd/attack/traditional.py)
- [RF adapters](ATKBackd/attack/rf_adapters.py)
- [Protocol, source provenance and deployment commands](MMFI_TABLES_VI.md)

### Anomalies and limitations

- Automatic SSH to MICA returned Permission denied (publickey). No full MM-Fi
  training was launched, no server output was modified, and no GitHub push occurred.
- Windows NumPy MKL/PyTorch duplicate OpenMP runtime was resolved with process-local
  serial MKL; no KMP_DUPLICATE_LIB_OK workaround.
- Legacy tests stub optional pyplot at collection. Preloading real Matplotlib for
  the final suite exercised actual plotting instead of accepting this stub.
- A reproduced Windows-spawn topology bug was fixed by binding target joints to
  each dataset. Two actual spawned-worker tests verify only MM-Fi joints2/3 change.
  Checkpoint schema is9 and evaluation schema is10; do not reuse pre-fix checkpoints.
- All paper methods are explicit CSI/HPE adaptations. INFOCOM POR does not optimize
  the input-dependent pose target and is not claimed data-free here. Threat models,
  stage budgets and perturbation semantics differ and are preserved in reports.
- Single seed42 does not estimate between-seed uncertainty. No attack-effectiveness
  claim or statistical result is inferred from these synthetic checks.
