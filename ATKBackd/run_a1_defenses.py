"""Isolated T5 clean-data repair of existing, immutable MM-Fi victims.

No victim training, attack adaptation, trigger optimization, or source-directory
writes occur here. Dry-run is read-only; full inference/repair requires an
explicit invocation without --dry-run. Local checkpoints must be trusted.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_a1_three_tables as frozen
from eval_a1_target_pck import _atomic_bytes, _sha
from run_mmfi_tables import _atomic_json, _cell_lock

HERE = Path(__file__).resolve().parent
MANIFEST = 'a1_defenses.resolved.json'
PROFILE = 'mmfi_a1_defenses_source_default_adapt_v1'
METHODS = ('a1',)
DEFENSES = ('none', 'ft', 'fp', 'nad', 'ft_sam')
SEEDS = (42, 43, 44)
METRICS = ('t_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm',
           'triggered_gt_mpjpe_mm')
DEFENSE_LABELS = dict(none='No defense', ft='FT', fp='Fine-Pruning',
                      nad='NAD', ft_sam='FT-SAM')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _sources():
    files = ('run_a1_defenses.py', 'defend/a1_repair_ops.py',
             'defend/a1_defense_eval.py', 'defend/a1_defense_sources.json')
    return {name: _sha(HERE / name) for name in files}


def resolve_recipe(value=None):
    from defend.a1_repair_ops import DEFAULT_RECIPE, _recipe
    if value is not None and not isinstance(value, dict):
        raise ValueError('T5 recipe must be a JSON object')
    value = copy.deepcopy(value or {})
    extra = {'batch_size': 256, 'bank_fraction': .05, 'selection_fraction': .2}
    unknown = set(value) - set(DEFAULT_RECIPE) - set(extra)
    if unknown:
        raise ValueError(f'Unknown T5 recipe fields: {sorted(unknown)}')
    extra.update({name: value.pop(name) for name in tuple(extra) if name in value})
    if type(extra['batch_size']) is not int or extra['batch_size'] < 1:
        raise ValueError('Defense batch_size must be a positive integer')
    # This initial T5 profile fixes the total trusted-clean budget, not 5%+1%.
    if extra['bank_fraction'] != .05 or extra['selection_fraction'] != .2:
        raise ValueError('T5 v1 fixes 5% total TRAIN bank and 20% selection within it')
    return dict(_recipe(value), **extra)


def _separate_output(outdir, source):
    outdir = Path(outdir).resolve()
    roots = [Path(source['outdir']).resolve()]
    roots += [Path(cell['ckpt_dir']).resolve() for cell in source['cells']]
    for root in roots:
        if outdir == root or root in outdir.parents or outdir in root.parents:
            raise ValueError('T5 output must be a separate sibling, not inside/above a source run')
    data_home = Path(source['data_home']).resolve()
    for protected in (data_home, data_home / 'datasets', data_home / 'actions', HERE.parent):
        if outdir == protected or outdir in protected.parents or protected in outdir.parents:
            # The data-home parent is allowed only through its dedicated runs/.
            if protected == data_home and data_home / 'runs' in outdir.parents:
                continue
            raise ValueError('T5 output overlaps code/data/actions or a broad data-home directory')
    return outdir


def _plan_hash(plan):
    return _digest({k: v for k, v in plan.items() if k not in ('created_utc', 'plan_sha256')})


def build_plan(source, source_outdir, outdir, recipe=None):
    """Read code/manifest metadata only: no dataset, checkpoint, or CUDA access."""
    frozen._validate_manifest(source)
    source_outdir = Path(source_outdir).resolve()
    if source_outdir != Path(source['outdir']).resolve():
        raise ValueError('--source-outdir differs from the frozen source manifest location')
    output = _separate_output(outdir, source)
    recipe = resolve_recipe(recipe)
    lookup = {(c['seed'], c['key']): c for c in source['cells']}
    cells = []
    for seed in SEEDS:
        for key in METHODS:
            original = lookup[seed, key]
            for defense in DEFENSES:
                cells.append(dict(cell_id=f's{seed}__{key}__{defense}', seed=seed,
                    method=key, defense=defense, source_cell_id=original['method_key'],
                    folder=str(output / f'seed{seed}' / key / defense)))
    plan = dict(schema=1, profile=PROFILE, created_utc=datetime.now(timezone.utc).isoformat(),
        source_outdir=str(source_outdir), source_plan_sha256=source['plan_sha256'],
        outdir=str(output), recipe=recipe, sources=_sources(),
        original_scientific_sources=source['sources'], cells=cells,
        contract=dict(dataset='mmfi', protocol='protocol1-s1', split_seed=0,
            train_samples=133056, test_samples=33264, seeds=list(SEEDS),
            epochs_original_victim=50, attack_rho=.1, original_victim_training=False,
            target_joints=[2, 3], target_pck_threshold=.2,
            target_pck_scale='norm(clean_true_pose[5]-clean_true_pose[12])',
            target_pck_unit='percent of target joints, not pose-level ASR',
            main_dose=1., doses=[0., .2, .4, .6, .8, 1.],
            main_metrics=list(METRICS), repair_loss='MPJPE in native pose units',
            bank='5% total clean TRAIN; 80% repair fit / 20% repair selection; sequence-disjoint',
            selection='clean TRAIN-derived selection only; never official TEST',
            bank_seen_by_original_victim=True, num_workers=0, pin_memory=False,
            recipe_status='BACKDOORBENCH_SOURCE_DEFAULT_HPE_ADAPTATION',
            verification='Local code tests do not establish full MM-Fi effectiveness',
            no_defense_reference='same immutable Proposed A1 checkpoint',
            control='No-defense A1; clean-input and triggered-to-true utility for every repair',
            defense_direction='T-MPJPE up; Target-PCK down; clean/triggered-GT errors down',
            dce='six-dose diagnostic only; lower DCE is not defense success',
            uncertainty='mean and sample SD(ddof=1) of three seed-level metrics',
            realized_budget_matched=False, defense_compute_matched=False,
            no_claim='not unchanged classification reproduction or physical RF robustness'))
    plan['plan_sha256'] = _plan_hash(plan)
    return plan


def _load_plan(source_root, outdir, recipe=None):
    source = _json(Path(source_root) / frozen.MANIFEST)
    canonical = build_plan(source, source_root, outdir, recipe)
    path = Path(outdir) / MANIFEST
    if path.exists():
        saved = _json(path)
        if (saved.get('plan_sha256') != _plan_hash(saved)
                or saved['plan_sha256'] != canonical['plan_sha256']):
            raise ValueError('T5 recipe/source/paths changed; use a NEW output directory')
        return saved, source
    if Path(outdir).exists() and any(p.name != 'run.lock' for p in Path(outdir).iterdir()):
        raise ValueError('New T5 output is not empty and has no owned manifest; choose a NEW directory')
    return canonical, source


def _selected(plan, seeds, methods, defenses):
    for values, allowed, name in ((seeds, SEEDS, 'seeds'), (methods, METHODS, 'methods'),
                                   (defenses, DEFENSES, 'defenses')):
        if not values or len(set(values)) != len(values) or set(values) - set(allowed):
            raise ValueError(f'Unknown, empty or duplicate {name}')
    return [c for c in plan['cells'] if c['seed'] in seeds
            and c['method'] in methods and c['defense'] in defenses]


def _source_cell(source, cell):
    matches = [c for c in source['cells'] if c['method_key'] == cell['source_cell_id']]
    if len(matches) != 1:
        raise ValueError('Missing/duplicate immutable source cell')
    return matches[0]


def _audit_source(source_cell, *, data=True):
    from a1_three_tables_eval import _artifact_hashes, _checkpoint_metadata
    run = frozen._read_completed(source_cell)
    _checkpoint_metadata(run)
    artifacts = _artifact_hashes(run)
    if any(artifacts[k] != v for k, v in run['metadata_sha256'].items()):
        raise ValueError('Source config/cache changed during loading')
    current = frozen._current_inputs(source_cell) if data else None
    audit = frozen._audit_run(source_cell, run, current) if data else None
    return run, artifacts, current, audit


def _binding(plan, cell, original, run, artifacts, current, bank):
    from train_backdoor import _config_fingerprint
    trigger_cfg = original.get('evaluation_trigger_cfg', run['cfg'])
    return dict(plan_sha256=plan['plan_sha256'], cell_id=cell['cell_id'],
        source_cell_id=original['method_key'], source_folder=str(run['folder']),
        source_plan_sha256=plan['source_plan_sha256'],
        source_config_fingerprint=run['fingerprint'], source_artifacts=artifacts,
        trigger_config_fingerprint=_config_fingerprint(trigger_cfg),
        inputs=current, bank_sha256=_digest(bank), recipe=plan['recipe'],
        sources=plan['sources'])


def _read_record(folder, expected_binding=None):
    folder = Path(folder)
    path = folder / 'evaluation.json'
    if not path.exists():
        return None
    record = _json(path)
    checksum = record.pop('record_sha256', None)
    if checksum != _digest(record):
        raise ValueError('T5 evaluation record checksum mismatch')
    if (record.get('schema') != 1 or record.get('status') != 'complete'
            or record.get('binding_sha256') != _digest(record.get('binding'))):
        raise ValueError('Invalid/incomplete T5 evaluation record')
    if expected_binding is not None and record['binding'] != expected_binding:
        raise ValueError('T5 evaluation scientific binding changed; use a NEW output directory')
    if record.get('per_sample_file') != 'per_sample.npz' or _sha(folder / 'per_sample.npz') != record.get('per_sample_sha256'):
        raise ValueError('T5 per-sample artifact changed/missing')
    if record['defense'] != 'none':
        if record.get('repair_file') != 'repair.pt' or _sha(folder / 'repair.pt') != record.get('repair_sha256'):
            raise ValueError('Repaired checkpoint changed/missing')
    for name in METRICS:
        value = record.get('metrics', {}).get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid T5 metric: {name}')
    if record['metrics']['target_pck_pct'] > 100:
        raise ValueError('Target-PCK percent exceeds 100')
    rows = record.get('dose_rows', [])
    if [r.get('d') for r in rows] != [0., .2, .4, .6, .8, 1.]:
        raise ValueError('T5 record lacks the full dose grid')
    record['record_sha256'] = checksum
    return record


def _bank(plan, original, current):
    from train_backdoor import _load_dataset
    from run_peak_confirmation import _pair_ids
    from run_method_drafts import _sha_json
    from defend.a1_defense_eval import select_clean_bank
    data = _load_dataset(original['cfg'], 'training')  # validation aliases TEST: never use it.
    ids = _pair_ids(data)
    if len(data) != 133056 or _sha_json(ids) != current['train']['pair_ids_sha256']:
        raise ValueError('Clean bank does not come from the audited full TRAIN identities')
    bank = select_clean_bank(ids, original['seed'], plan['recipe']['bank_fraction'],
                             plan['recipe']['selection_fraction'])
    path = Path(plan['outdir']) / f"seed{original['seed']}" / 'clean_bank.json'
    if path.exists() and _json(path) != bank:
        raise ValueError('Seed-level clean bank changed; refusing resume')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, bank)
    return data, bank


def _execute(plan, source, cell, device):
    from a1_three_tables_eval import _artifact_hashes
    from eval_a1_target_pck import _load_weights, _restore_trigger, _dataset_record, _assert_replay
    from defend.a1_defense_eval import clean_loaders, evaluate_defense
    from defend.a1_repair_ops import train_repair, DEFAULT_RECIPE
    import numpy as np
    original = _source_cell(source, cell)
    folder = Path(cell['folder'])
    with _cell_lock(folder):
        run, artifacts, current, audit = _audit_source(original)
        data_train, bank = _bank(plan, original, current)
        binding = _binding(plan, cell, original, run, artifacts, current, bank)
        cached = _read_record(folder, binding)
        if cached is not None:
            print(f"[T5] verified complete: {cell['cell_id']} (no repair/evaluation)", flush=True)
            return cached
        model, trigger_state = _load_weights(run, device)
        cfg = copy.deepcopy(original.get('evaluation_trigger_cfg', run['cfg']))
        # Change runtime evaluation placement only; original config is never rewritten.
        cfg.update(device=device, num_workers=0)
        trigger = _restore_trigger(cfg, trigger_state if cell['method'] != 'clean' else None)
        data_test, test_records, test_ids = _dataset_record(cfg)
        if any(test_records[name] != current[name] for name in ('train', 'eval')):
            raise ValueError('Defense evaluation inputs differ from frozen training inputs')
        repair_metadata = dict(defense='none', complete=True, optimizer_steps=0,
                               original_checkpoint_reused=True)
        if cell['defense'] != 'none':
            fit, selection = clean_loaders(data_train, bank, original['cfg'],
                original['seed'], batch_size=plan['recipe']['batch_size'])
            recipe = {k: plan['recipe'][k] for k in DEFAULT_RECIPE}
            model, repair_metadata = train_repair(model, fit, selection, cell['defense'],
                recipe, device, folder / 'repair.pt', _digest(binding), original['seed'])
            if not repair_metadata.get('complete'):
                raise ValueError('Defense did not finish the frozen repair stages')
        evaluated = evaluate_defense(model, data_test, trigger, cfg, device)
        if cell['defense'] == 'none':
            legacy = dict(evaluated['metrics'], **evaluated['diagnostics'])
            legacy['t1_mpjpe_mm'] = evaluated['metrics']['t_mpjpe_mm']
            replay = _assert_replay(run['res'], legacy, .05, attack=cell['method'] != 'clean')
        else:
            replay = None
        after, hashes_after, current_after, _ = _audit_source(original)
        if (artifacts != hashes_after or current != current_after
                or after['fingerprint'] != run['fingerprint']):
            raise ValueError('Immutable source/data identities changed during defense; no result accepted')
        frozen._validate_manifest(source)
        if _sources() != plan['sources']:
            raise ValueError('Defense source code changed during this cell; no result accepted')
        arrays = dict(evaluated['arrays'], pair_ids=np.asarray(test_ids, dtype=str))
        sample_data = io.BytesIO()
        np.savez_compressed(sample_data, **arrays)
        _atomic_bytes(folder / 'per_sample.npz', sample_data.getvalue())
        record = dict(schema=1, status='complete', cell_id=cell['cell_id'],
            seed=cell['seed'], method=cell['method'], defense=cell['defense'],
            control=cell['method'] == 'clean', binding=binding, binding_sha256=_digest(binding),
            metrics=evaluated['metrics'], diagnostics=evaluated['diagnostics'],
            dose_rows=evaluated['dose_rows'], repair=repair_metadata,
            source_scientific_audit=audit, no_defense_replay_deltas_mm=replay,
            per_sample_file='per_sample.npz', per_sample_sha256=_sha(folder / 'per_sample.npz'),
            created_utc=datetime.now(timezone.utc).isoformat())
        if cell['defense'] != 'none':
            record.update(repair_file='repair.pt', repair_sha256=_sha(folder / 'repair.pt'))
        record['record_sha256'] = _digest(record)
        _atomic_json(folder / 'evaluation.json', record)
        print(f"[T5] completed {cell['cell_id']}", flush=True)
        return record


def _csv(rows, names):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    writer.writerows({name: row.get(name) for name in names} for row in rows)
    return stream.getvalue().encode('utf-8')


def export_records(plan, records):
    """Partial previews stay explicit; SD uses independent seeds, never frames."""
    import numpy as np
    allowed = {c['cell_id']: c for c in plan['cells']}
    indexed = {}
    for record in records:
        cell_id = record['cell_id']
        cell = allowed.get(cell_id)
        if (cell is None or cell_id in indexed
                or any(record[k] != cell[v] for k, v in
                       (('seed', 'seed'), ('method', 'method'), ('defense', 'defense')))):
            raise ValueError('Duplicate or out-of-plan T5 record')
        indexed[cell_id] = record
    rows = [dict(seed=r['seed'], method=r['method'], defense=r['defense'],
                 control=r['control'], **r['metrics'])
            for c in plan['cells'] if (r := indexed.get(c['cell_id'])) is not None]
    doses = [dict(seed=r['seed'], method=r['method'], defense=r['defense'], **d)
             for r in indexed.values() for d in r['dose_rows']]
    aggregates = []
    for method in METHODS:
        for defense in DEFENSES:
            sample = [r for r in rows if r['method'] == method and r['defense'] == defense]
            seed_list = sorted(r['seed'] for r in sample)
            item = dict(method=method, defense=defense, control=method == 'clean',
                        seeds=seed_list, n_seeds=len(sample), complete=seed_list == list(SEEDS))
            for metric in METRICS:
                values = [r[metric] for r in sample]
                item[metric + '_mean'] = float(np.mean(values)) if values else None
                item[metric + '_sd'] = float(np.std(values, ddof=1)) if len(values) == 3 else None
            aggregates.append(item)
    complete = len(indexed) == len(plan['cells'])
    status = ('T5_COMPLETE_THREE_SEEDS_SOURCE_DEFAULT_ADAPTATION' if complete
              else 'T5_PARTIAL_PREVIEW_NOT_COMPLETE_THREE_SEED_TABLE')
    output = Path(plan['outdir'])
    fields = ['seed', 'method', 'defense', 'control', *METRICS]
    _atomic_bytes(output / 'defense_per_seed.csv', _csv(rows, fields))
    _atomic_bytes(output / 'defense_six_dose.csv', _csv(doses,
        ['seed', 'method', 'defense', 'd', 't_mpjpe_mm', 'target_pck_pct',
         'triggered_gt_mpjpe_mm', 'dce_mm', 'no_trigger_tmpjpe_mm']))
    _atomic_json(output / 'defense_summary.json', dict(status=status,
        plan_sha256=plan['plan_sha256'], complete_records=len(indexed),
        expected_records=len(plan['cells']), missing=[k for k in allowed if k not in indexed],
        aggregates=aggregates, uncertainty='sample SD across seeds 42/43/44 only',
        limitations=['Source-default defense mechanism adaptation, not an unchanged classification pipeline.',
            'Clean bank was seen by the original victim; held out only from repair fitting.',
            'Input identity hashes bind ordered paths/frame indices, not every CSI/pose file byte.',
            'Native attacks and defense compute are not budget matched.',
            'Digital model repair does not establish physical RF robustness.']))
    lines = [f'# {status}', '', f"Records: {len(indexed)}/{len(plan['cells'])}.", '',
        'All metrics at dose 1; no alignment. Target-PCK uses joints [2,3] and threshold .2.',
        'Defense directions: T-MPJPE ↑, Target-PCK ↓, Clean MPJPE ↓, Triggered-GT MPJPE ↓.',
        'Triggered-GT uses the same target joints [2,3] against the true pose, not attacker target.',
        'Do not call model destruction a successful defense. No-defense A1 is the paired baseline.',
        'Mean ± sample SD requires all three independent seeds; partial rows show per-seed values only.', '']
    for control, title in ((False, 'Attack victims'), (True, 'Clean-trained utility controls (A1-trigger diagnostic)')):
        if not any(r['control'] == control for r in rows):
            continue
        lines += [f'## {title}', '',
            '| Method | Defense | Seed(s) | T-MPJPE (mm) ↑ | Target-PCK (%) ↓ | Clean MPJPE (mm) ↓ | Triggered-GT MPJPE (mm) ↓ |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: |']
        for a in aggregates:
            if a['control'] != control or not a['n_seeds']:
                continue
            label = frozen.LABELS[a['method']]
            if a['complete']:
                values = [f"{a[m + '_mean']:.2f} ± {a[m + '_sd']:.2f}" for m in METRICS]
                lines.append('| ' + ' | '.join([label, DEFENSE_LABELS[a['defense']], '42/43/44', *values]) + ' |')
            else:
                for row in rows:
                    if row['method'] == a['method'] and row['defense'] == a['defense']:
                        lines.append('| ' + ' | '.join([label, DEFENSE_LABELS[a['defense']], str(row['seed']),
                            *[f'{row[m]:.2f}' for m in METRICS]]) + ' |')
        lines.append('')
    _atomic_bytes(output / 'table5_defenses.md', ('\n'.join(lines) + '\n').encode('utf-8'))
    return status


def _collect(plan, source, *, audit_sources=True):
    """Export reads only existing records; stale/missing bindings cannot pass."""
    records = []
    audited = {}
    for cell in plan['cells']:
        record = _read_record(cell['folder'])
        if record is None:
            continue
        original = _source_cell(source, cell)
        cache_id = original['method_key']
        if cache_id not in audited:
            run, artifacts, current, _ = _audit_source(original, data=audit_sources)
            audited[cache_id] = (run, artifacts, current)
        run, artifacts, current = audited[cache_id]
        if current is None:
            current = record['binding']['inputs']
        bank = _json(Path(plan['outdir']) / f"seed{cell['seed']}" / 'clean_bank.json')
        expected = _binding(plan, cell, original, run, artifacts, current, bank)
        records.append(_read_record(cell['folder'], expected))
    return records


def _run_sequential(plan, source, selected, device, manifest):
    # Child per cell releases model/data memory and keeps readable individual logs.
    for cell in selected:
        folder = Path(cell['folder'])
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / 'console.log'
        with log.open('a', encoding='utf-8') as stream:
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       '--_manifest', str(manifest), '--_cell', cell['cell_id'], '--device', device]
            print(f"[T5] {cell['cell_id']} -> {device}; log: {log}", flush=True)
            result = subprocess.run(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{cell['cell_id']} failed; see {log}. No automatic retry.")
        if _read_record(folder) is None:
            raise RuntimeError(f"{cell['cell_id']} exited without a complete evaluation record")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-outdir')
    parser.add_argument('--outdir')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--methods', nargs='+', default=list(METHODS))
    parser.add_argument('--defenses', nargs='+', default=list(DEFENSES))
    parser.add_argument('--recipe', help='Explicit HPE recipe JSON overrides; frozen on first run')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--export-only', action='store_true')
    parser.add_argument('--_manifest', help=argparse.SUPPRESS)
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._manifest or args._cell:
        if not args._manifest or not args._cell:
            parser.error('Both internal worker arguments required')
        saved = _json(args._manifest)
        plan, source = _load_plan(saved['source_outdir'], saved['outdir'], saved['recipe'])
        if Path(args._manifest).resolve() != Path(plan['outdir']) / MANIFEST:
            raise ValueError('Internal worker manifest path differs from frozen plan')
        selected = [c for c in plan['cells'] if c['cell_id'] == args._cell]
        if len(selected) != 1:
            raise ValueError('Unknown defense worker cell')
        _execute(plan, source, selected[0], args.device)
        return 0
    if not args.source_outdir or not args.outdir:
        parser.error('--source-outdir and --outdir are required')
    recipe = _json(args.recipe) if args.recipe else None
    # Reuse the saved recipe on resume unless the user explicitly changes it.
    if recipe is None and (Path(args.outdir) / MANIFEST).exists():
        recipe = _json(Path(args.outdir) / MANIFEST)['recipe']
    plan, source = _load_plan(args.source_outdir, args.outdir, recipe)
    selected = _selected(plan, args.seeds, args.methods, args.defenses)
    if args.dry_run:
        print(f"[T5] DRY RUN: {len(selected)} selected evaluations, "
              f"{sum(c['defense'] != 'none' for c in selected)} repairs; "
              'NO writes/data/checkpoint/GPU/training. Source-default defenses adapted to HPE; effectiveness not tested by dry-run.')
        print(f"[T5] plan {plan['plan_sha256']}; source {plan['source_plan_sha256']}")
        return 0
    if args.status:
        present = sum((Path(c['folder']) / 'evaluation.json').is_file() for c in selected)
        print(f'[T5] artifact presence: {present}/{len(selected)} selected; NOT a fresh scientific audit')
        return 0
    output = Path(plan['outdir'])
    with _cell_lock(output):
        manifest = output / MANIFEST
        if not manifest.exists():
            _atomic_json(manifest, plan)
        if not args.export_only:
            # Preflight every selected immutable victim before starting any repair.
            checked = set()
            for cell in selected:
                original = _source_cell(source, cell)
                if original['method_key'] not in checked:
                    _audit_source(original)
                    checked.add(original['method_key'])
            _run_sequential(plan, source, selected, args.device, manifest)
        records = _collect(plan, source)
        status = export_records(plan, records)
        print(f"[{status}] {output / 'table5_defenses.md'}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
