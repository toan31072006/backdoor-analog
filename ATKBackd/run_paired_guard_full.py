"""Confirm one frozen paired-guard trigger on full MM-Fi TRAIN/TEST.

No attacker fitting, source victim weights, or historical metric import occurs.
This is one seed and one frozen candidate, not a statistical superiority claim.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

from run_mmfi_tables import _atomic_json, _cell_lock
from run_method_drafts import _file_sha, _sha_json, _verified_cache, _write_csv
import run_learned_carrier_drafts as shared

HERE = Path(__file__).resolve().parent
STATUS = 'FULL_CONFIRMATION_SINGLE_SEED'
SOURCE_FILES = ('prepared_cfg.json', 'fitting.json', 'learned_trigger.json')
GRID = shared.GRID


def read_frozen_source(source_cell):
    """Validate historical metadata/artifacts, never read its victim result."""
    source_cell = Path(source_cell).resolve()
    matrix = json.loads((source_cell.parent / 'paired_guard.resolved.json').read_text(encoding='utf-8'))
    if (matrix.get('draft_profile') != 'paired_guard_screen_v1'
            or matrix.get('seed') != 42 or matrix.get('DRAFT_ONLY') is not True
            or matrix.get('plan_sha256') != shared._plan_sha(matrix)):
        raise ValueError('Expected an intact paired-guard screening source manifest')
    matches = [c for c in matrix['cells'] if c.get('method_key') == 'lc_paired_guard']
    if len(matches) != 1 or Path(matches[0]['ckpt_dir']).resolve() != source_cell:
        raise ValueError('Source cell directory is not the planned lc_paired_guard cell')
    prepared = shared._prepared_cell(matches[0])
    if prepared is None:
        raise ValueError('Source requires a frozen prepared config, fitting record and trigger artifact')
    fit = json.loads((source_cell / 'fitting.json').read_text(encoding='utf-8'))
    if (not isinstance(fit.get('selected_utility_gate_passed'), bool)
            or not isinstance(fit.get('no_eligible_candidate'), bool)):
        raise ValueError('Source must explicitly retain selected guard and fallback flags')
    if Path(prepared['cfg']['lc_artifact_path']).resolve() != source_cell / 'learned_trigger.json':
        raise ValueError('Expected the source cell own learned_trigger.json')
    return dict(cfg=prepared['cfg'], original_cell_dir=str(source_cell),
        hashes={name: _file_sha(source_cell / name) for name in SOURCE_FILES},
        source_recipe_sha256=matches[0]['recipe_sha256'],
        source_plan_sha256=matrix['plan_sha256'],
        source_code_sha256=matrix['sources']['source_sha256'],
        selected_utility_gate_passed=fit['selected_utility_gate_passed'],
        no_eligible_candidate=fit['no_eligible_candidate'])


def _source_provenance():
    sources = shared._source_provenance()
    for name in ('run_paired_guard_full.py', 'frozen_full_contract.py'):
        sources[name] = _file_sha(HERE / name)
    return sources


def _plan_sha(plan):
    stable = copy.deepcopy(plan)
    stable.pop('plan_sha256', None)
    stable.pop('created_utc', None)
    stable['cfg'] = shared._scientific(stable['cfg'])
    return _sha_json(stable)


def build_plan(source, outdir, device='cuda:0', num_workers=4, data_home=None):
    from frozen_full_contract import FULL_CONFIRMATION_PROFILE, validate_frozen_full_config
    from train_backdoor import _resolve_training_config, _validate_training_contract
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    outdir = Path(outdir).resolve()
    cfg = {k: copy.deepcopy(v) for k, v in source['cfg'].items() if not k.startswith('draft_')}
    snapshot = outdir / 'source_frozen'
    provenance = dict(
        source_cell_dir=str(snapshot), original_cell_dir=source['original_cell_dir'],
        prepared_cfg_sha256=source['hashes']['prepared_cfg.json'],
        fitting_sha256=source['hashes']['fitting.json'],
        artifact_sha256=source['hashes']['learned_trigger.json'],
        source_recipe_sha256=source['source_recipe_sha256'],
        source_plan_sha256=source['source_plan_sha256'],
        source_code_sha256=source['source_code_sha256'],
        selected_utility_gate_passed=source['selected_utility_gate_passed'],
        no_eligible_candidate=source['no_eligible_candidate'])
    cfg.update(method_draft=False, confirmation_profile=FULL_CONFIRMATION_PROFILE,
        confirmation_source=provenance, epochs=50, victim_epochs=50,
        device=device, num_workers=num_workers, pretrained=False, data_parallel=False,
        strict_resume=True, ckpt_every=1,
        lc_artifact_path=str(snapshot / 'learned_trigger.json'))
    if data_home is not None:
        cfg.update(dataset_root=str(Path(data_home).resolve() / 'datasets/Compress'),
            action_npy=str(Path(data_home).resolve() / 'actions/data_bend.npy'))
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    validate_frozen_full_config(cfg)
    plan = dict(schema=1, status=STATUS, confirmation_profile=FULL_CONFIRMATION_PROFILE,
        method_key='lc_paired_guard', label='Proposed: frozen paired clean-twin trigger',
        seed=42, created_utc=datetime.now(timezone.utc).isoformat(), cfg=cfg,
        source_sha256=_source_provenance(),
        metrics_contract=dict(evaluation_role='full official TEST after fresh full official TRAIN victim',
            official_test_used_for_fitting=False, source_trigger_refitted=False,
            source_victim_weights_imported=False, historical_metrics_imported=False,
            all_train_frames=True, all_test_frames=True, dose_grid=GRID,
            pck='relative .5/.4/.3/.2/.1, NOT millimetres',
            uncertainty='single seed; not estimated',
            stealthiness='digital bounds only; no RF stealthiness or OTA claim'))
    plan['plan_sha256'] = _plan_sha(plan)
    return plan


def _snapshot_source(source, outdir):
    """Copy frozen input evidence only; never replace a changed existing copy."""
    destination = Path(outdir) / 'source_frozen'
    destination.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        target = destination / name
        if target.exists():
            if _file_sha(target) != source['hashes'][name]:
                raise ValueError(f'Frozen source snapshot changed: {name}; use NEW directory')
        else:
            shutil.copyfile(Path(source['original_cell_dir']) / name, target)
            if _file_sha(target) != source['hashes'][name]:
                raise ValueError(f'Source changed during snapshot: {name}')


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def export_results(plan, res, outdir):
    from mmfi_tables import metric_row, dose_rows
    from frozen_full_contract import validate_frozen_full_source_files
    from train_backdoor import _config_fingerprint
    validate_frozen_full_source_files(plan['cfg'])
    if (res.get('training_contract') != 'ordinary_erm' or res.get('attacker_access') != 'data_only'
            or res.get('dose_grid') != GRID or res.get('seed') != 42):
        raise ValueError('Full result violates the frozen ordinary-ERM contract')
    cell = dict(method_key=plan['method_key'], label=plan['label'], cfg=plan['cfg'])
    metrics = metric_row(cell, res)
    row = dict(status=STATUS, seed=42, epochs=50, rho=.1,
        **{k: v for k, v in metrics.items() if k in (
            'clean_mpjpe_mm', 'clean_pampjpe_mm', 'tmpjpe_mm') or k.startswith('clean_pck_')},
        selected_utility_gate_passed=plan['cfg']['confirmation_source']['selected_utility_gate_passed'],
        no_eligible_candidate=plan['cfg']['confirmation_source']['no_eligible_candidate'])
    doses, missing = dose_rows(cell, res)
    if missing:
        raise ValueError(f'Incomplete six-dose result: {missing}')
    report = dict(status=STATUS, plan_sha256=plan['plan_sha256'], row=row,
        confirmation_source=plan['cfg']['confirmation_source'],
        metrics_contract=plan['metrics_contract'])
    outdir = Path(outdir)
    _atomic_json(outdir / 'results.json', dict(method_key=cell['method_key'],
        cfg=plan['cfg'], cfg_fingerprint=_config_fingerprint(plan['cfg']),
        plan_sha256=plan['plan_sha256'], res=res))
    _atomic_json(outdir / 'main_metrics.json', report)
    _write_csv(outdir / 'main_metrics.csv', [row], list(row))
    _atomic_json(outdir / 'dose_response.json', dict(status=STATUS, rows=doses))
    _write_csv(outdir / 'dose_response.csv', doses, list(doses[0]))
    lines = [f'# {STATUS}', '', 'Frozen trigger; fresh seed-42 full MM-Fi victim, 50 epochs; rho=.1.',
        'Official TRAIN and TEST are uncapped. No trigger fitting or source victim-weight transfer.', '',
        '| Metric | Value |', '| --- | ---: |',
        f"| Clean MPJPE (mm) | {row['clean_mpjpe_mm']:.3f} |",
        f"| Clean PA-MPJPE (mm) | {row['clean_pampjpe_mm']:.3f} |",
        f"| Target-joint T-MPJPE at d=1 (mm) | {row['tmpjpe_mm']:.3f} |"]
    for threshold in (.5, .4, .3, .2, .1):
        lines.append(f"| Relative PCK@{threshold:.1f} (%) | {row[f'clean_pck_{threshold:.1f}_pct']:.2f} |")
    lines += ['', f"Source surrogate guard passed: {row['selected_utility_gate_passed']}; fallback: {row['no_eligible_candidate']}.",
        'This is not a successful guard claim when the source failed its gate.',
        'One seed and a development-selected candidate; no statistical superiority is estimated.',
        'Compare only against independently trained controls with the same full splits, epochs and dual budgets.',
        'Do not compare directly with 15-epoch TRAIN-holdout scores or a peak-only-budget baseline.',
        'Digital perturbation bounds do not establish RF stealthiness or over-the-air feasibility.',
        f"Plan SHA256: `{plan['plan_sha256']}`"]
    (outdir / 'main_metrics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--source-cell', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    return parser.parse_args(argv)


def _validate_single_device(device):
    """Explain logical IDs using this runner's singular --device option."""
    if device == 'cpu':
        return
    import torch
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or int(device.split(':')[1]) >= count:
        mask = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
        logical = ', '.join(f'cuda:{i}' for i in range(count)) or 'none'
        raise ValueError(f'CUDA preflight: device_count={count}, CUDA_VISIBLE_DEVICES={mask!r}; '
            f'requested={device}, valid logical IDs={logical}. '
            'Use one visible logical --device cuda:N. With CUDA_VISIBLE_DEVICES=3, '
            'physical GPU3 is --device cuda:0; use --device cpu only for a CPU check.')


def main(argv=None):
    import re
    from frozen_full_contract import validate_frozen_full_source_files
    from train_backdoor import train
    args = parse_args(argv)
    if args.device != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', args.device) is None:
        raise ValueError('Use one logical cuda:N device or cpu')
    outdir = args.outdir.resolve()
    source_cell = args.source_cell.resolve()
    if outdir == source_cell or outdir in source_cell.parents or source_cell in outdir.parents:
        raise ValueError('Full run and historical source need separate, non-nested directories')
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses a nonempty directory; no old result is deleted')
    source = read_frozen_source(source_cell)
    requested = build_plan(source, outdir, args.device, args.num_workers, args.data_home)
    with _cell_lock(outdir):
        path = outdir / 'paired_guard_full.resolved.json'
        if path.exists():
            existing = json.loads(path.read_text(encoding='utf-8'))
            if existing.get('plan_sha256') != _plan_sha(existing) or existing['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Full plan/source/code changed; use a NEW output directory')
        elif any(p.name != 'run.lock' for p in outdir.iterdir()):
            raise ValueError('Existing non-confirmation files occupy output directory')
        _atomic_json(path, requested)
        if args.dry_run:
            print(f'[{STATUS}] DRY RUN: one frozen trigger; full TRAIN/TEST; 50 epochs; no CSI/GPU/fitting/training.\n{path}')
            return 0
        _snapshot_source(source, outdir)
        validate_frozen_full_source_files(requested['cfg'])
        _validate_single_device(args.device)
        print(f'[{STATUS}] lc_paired_guard -> {args.device}; log: {outdir / "console.log"}', flush=True)
        with (outdir / 'console.log').open('a', encoding='utf-8') as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                model, res = train(requested['cfg'], ckpt_dir=outdir)
                del model
        # Verify the freshly written cache before publishing any table.
        checked, _ = _verified_cache(dict(method_key=requested['method_key'],
            label=requested['label'], cfg=requested['cfg'], eval_cache=str(outdir / 'eval_cache.json')))
        report = export_results(requested, checked, outdir)
        print(f'[{STATUS}] complete -> {outdir / "main_metrics.md"}', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[{STATUS}] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
