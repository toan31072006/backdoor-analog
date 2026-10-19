"""Four clean-data HPE defenses, three seeds, with audited v1 reuse.

New output/manifest only: no changes to v1 code, recipes, runs or victim weights.
ANP is a declared HPE adaptation; no claim of exact CIFAR reproduction.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import time

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_a1_defenses as legacy
import run_a1_defenses_parallel as runtime_v1
from defend.a1_defense_metrics import (DEFENSES, MAIN_METRICS, METRICS,
                                      build_metric_report, enrich_evaluation)

HERE = Path(__file__).resolve().parent
MANIFEST = 'a1_defenses_v2.resolved.json'
SEEDS = (42, 43, 44)
LABELS = dict(none='No defense', ft='FT', anp='ANP (HPE-adapted)', nad='NAD', ft_sam='FT-SAM')
PROFILE = 'mmfi_a1_ft_anp_nad_sam_three_seed_v2'


def _sources():
    names = ('run_a1_defenses_v2.py', 'run_a1_defenses_parallel.py',
             'defend/a1_anp.py', 'defend/a1_anp_sources.json',
             'defend/a1_defense_metrics.py', *legacy._sources().keys())
    return {name: legacy._sha(HERE / name) for name in names}


def build_plan(source, source_outdir, outdir, *, recipe=None, anp_recipe=None, reuse_outdir=None):
    from defend.a1_anp import resolve_anp_recipe
    if legacy._json(HERE / 'defend/a1_anp_sources.json')['frozen_defaults'] != resolve_anp_recipe():
        raise ValueError('Declared ANP scalar source defaults and implementation differ')
    plan = legacy.build_plan(source, source_outdir, outdir, recipe)
    output = Path(plan['outdir'])
    plan['profile'] = PROFILE
    plan['schema'] = 2
    plan['sources'] = _sources()
    plan['anp_recipe'] = resolve_anp_recipe(anp_recipe)
    if plan['anp_recipe']['batch_size'] != plan['recipe']['batch_size']:
        raise ValueError('ANP and common clean bank loaders must use the same frozen batch size')
    lookup = {(c['seed'], c['key']): c for c in source['cells']}
    plan['cells'] = [dict(cell_id=f's{s}__a1__{d}', seed=s, method='a1', defense=d,
        source_cell_id=lookup[s, 'a1']['method_key'],
        folder=str(output / f'seed{s}' / 'a1' / d)) for s in SEEDS for d in DEFENSES]
    reuse = Path(reuse_outdir).resolve() if reuse_outdir else None
    if reuse is not None:
        if reuse == output or reuse in output.parents or output in reuse.parents:
            raise ValueError('Reuse root and new output must be separate siblings')
        old = legacy._json(reuse / legacy.MANIFEST)
        canonical, _ = legacy._load_plan(source_outdir, reuse, old['recipe'])
        if canonical['recipe'] != plan['recipe']:
            raise ValueError('V1 repair recipes changed; cannot reuse as unchanged defenses')
        plan['reuse'] = dict(outdir=str(reuse), plan_sha256=canonical['plan_sha256'],
            policy='copy authenticated complete records with explicit original provenance; never edit v1')
    else:
        plan['reuse'] = None
    plan['contract'].update(defenses=list(DEFENSES), expected_records=15,
        main_metrics=list(MAIN_METRICS), reporting_metrics=list(METRICS),
        historical_evaluation='FT/FP seed42 retained; revised defense selection after viewing pilot',
        new_method='ANP loss/BN adapter and clean-selection threshold gate, not an improved defense',
        no_claim='no guarantee of defense resistance, reviewer acceptance, or unchanged classification reproduction')
    plan['plan_sha256'] = legacy._plan_hash(plan)
    return plan


def _load_plan(source_root, outdir, reuse_outdir=None):
    source = legacy._json(Path(source_root) / legacy.frozen.MANIFEST)
    manifest = Path(outdir) / MANIFEST
    if manifest.exists():
        saved = legacy._json(manifest)
        from defend.a1_anp import resolve_anp_recipe
        if (saved.get('recipe') != legacy.resolve_recipe()
                or saved.get('anp_recipe') != resolve_anp_recipe()):
            raise ValueError('Frozen V2 scalar defaults changed; use a NEW approved profile')
        saved_reuse = saved.get('reuse')
        reuse = reuse_outdir if reuse_outdir is not None else (saved_reuse['outdir'] if saved_reuse else None)
        canonical = build_plan(source, source_root, outdir, recipe=saved['recipe'],
            anp_recipe=saved['anp_recipe'], reuse_outdir=reuse)
        if (saved.get('plan_sha256') != legacy._plan_hash(saved)
                or saved['plan_sha256'] != canonical['plan_sha256']):
            raise ValueError('V2 recipe/source/path changed; use a NEW output directory')
        return saved, source
    plan = build_plan(source, source_root, outdir, reuse_outdir=reuse_outdir)
    if Path(outdir).exists() and any(p.name != 'run.lock' for p in Path(outdir).iterdir()):
        raise ValueError('New V2 output is not empty and has no owned manifest')
    return plan, source


def _selected(plan, seeds, defenses):
    for values, allowed in ((seeds, SEEDS), (defenses, DEFENSES)):
        if not values or len(set(values)) != len(values) or set(values) - set(allowed):
            raise ValueError('Unknown/empty/duplicate runtime seed or defense')
    return [c for c in plan['cells'] if c['seed'] in seeds and c['defense'] in defenses]


def _enriched_record(record, folder):
    import numpy as np
    from run_method_drafts import _sha_json
    with np.load(Path(folder) / 'per_sample.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    identity = record['binding']['inputs']['eval']
    indices = arrays.get('sample_indices')
    pairs = arrays.get('pair_ids')
    if (record['diagnostics'].get('samples') != 33264 or identity.get('n') != 33264
            or record['diagnostics'].get('full_official_test') is not True
            or indices is None or indices.shape != (33264,)
            or not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(indices, np.arange(33264))
            or pairs is None or pairs.shape != (33264, 3)
            or pairs.dtype.kind != 'U'):
        raise ValueError('V2 NPZ lacks the ordered full official TEST identity audit')
    # V1 NPZ writes dtype=str, whereas _pair_ids hashes the third (frame)
    # component as a JSON INTEGER. Restore that canonical type before hashing.
    try:
        ids = [[a, b, int(frame)] for a, b, frame in pairs.tolist()]
    except (ValueError, TypeError) as exc:
        raise ValueError('V2 NPZ has invalid official TEST identity frame types') from exc
    if (any(str(row[2]) != text or row[2] < 0 for row, text in zip(ids, pairs[:, 2]))
            or _sha_json(ids) != identity.get('pair_ids_sha256')):
        raise ValueError('V2 NPZ ordered official TEST identity differs from its source')
    return enrich_evaluation(dict(metrics=record['metrics'], diagnostics=record['diagnostics'],
        dose_rows=record['dose_rows'], arrays=arrays))


def _read_record(folder, expected_binding=None):
    record = legacy._read_record(folder, expected_binding)
    if record is None:
        return None
    if record.get('profile') != PROFILE:
        raise ValueError('Record is not a V2 record')
    seed, defense = record.get('seed'), record.get('defense')
    identity = f's{seed}__a1__{defense}'
    if (type(seed) is not int or seed not in SEEDS or defense not in DEFENSES
            or record.get('method') != 'a1' or record.get('control') is not False
            or record.get('cell_id') != identity or record['binding'].get('cell_id') != identity):
        raise ValueError('V2 record seed/defense/cell identity changed')
    evaluated = _enriched_record(record, folder)
    if any(m not in record['metrics'] for m in METRICS):
        raise ValueError('V2 record lacks required reporting metrics')
    if record['metrics'] != evaluated['metrics']:
        raise ValueError('V2 metrics do not reproduce the saved per-sample data')
    reuse = record.get('reuse_provenance')
    if reuse:
        path = Path(folder) / 'reused_v1_evaluation.json'
        if legacy._sha(path) != reuse['evaluation_file_sha256']:
            raise ValueError('Copied original V1 record changed')
        old = legacy._json(path)
        checksum = old.pop('record_sha256', None)
        if checksum != legacy._digest(old) or checksum != reuse['record_sha256']:
            raise ValueError('Original V1 record checksum invalid')
        if (old['binding'] != reuse['original_binding']
                or old['per_sample_sha256'] != record['per_sample_sha256']
                or old.get('repair_sha256') != record.get('repair_sha256')):
            raise ValueError('Reused source binding/artifacts changed')
    return record


def _context(plan, source, cell, *, data=True):
    original = legacy._source_cell(source, cell)
    run, artifacts, inputs, audit = legacy._audit_source(original, data=data)
    bank = legacy._json(Path(plan['outdir']) / f"seed{cell['seed']}" / 'clean_bank.json')
    if inputs is None:
        old = legacy._read_record(cell['folder'])
        if old is None:
            raise ValueError('No historical input identity audit is available')
        inputs = old['binding']['inputs']
    binding = legacy._binding(plan, cell, original, run, artifacts, inputs, bank)
    return original, run, artifacts, inputs, audit, bank, binding


def _write_result(folder, record):
    record = copy.deepcopy(record)
    record.pop('record_sha256', None)
    record['record_sha256'] = legacy._digest(record)
    legacy._atomic_json(Path(folder) / 'evaluation.json', record)
    return record


def _reuse_record(plan, source, cell):
    """Import complete v1 repair/evaluation only after original binding checks."""
    if not plan.get('reuse') or cell['defense'] == 'anp':
        return None
    root = Path(plan['reuse']['outdir'])
    old_plan = legacy._json(root / legacy.MANIFEST)
    if old_plan['plan_sha256'] != plan['reuse']['plan_sha256']:
        raise ValueError('Declared V1 reuse plan changed')
    match = [c for c in old_plan['cells'] if c['cell_id'] == cell['cell_id']]
    if len(match) != 1:
        raise ValueError('Reuse cell missing/duplicate in original plan')
    old_cell = match[0]
    old = legacy._read_record(old_cell['folder'])
    if old is None:
        return None
    original, run, artifacts, inputs, audit, bank, binding = _context(plan, source, cell)
    old_bank = legacy._json(root / f"seed{cell['seed']}" / 'clean_bank.json')
    if old_bank != bank or old_plan['recipe'] != plan['recipe']:
        raise ValueError('Cannot reuse a different clean bank/recipe')
    expected = legacy._binding(old_plan, old_cell, original, run, artifacts, inputs, old_bank)
    old = legacy._read_record(old_cell['folder'], expected)
    evaluated = _enriched_record(old, old_cell['folder'])
    folder = Path(cell['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    # Resume interrupted copies only if existing files match authenticated v1
    # bytes. Never overwrite a partially trained NEW repair or different NPZ.
    for name in ('per_sample.npz',) + (() if cell['defense'] == 'none' else ('repair.pt',)):
        target, source_path = folder / name, Path(old_cell['folder']) / name
        if target.exists():
            if legacy._sha(target) != legacy._sha(source_path):
                raise ValueError('V2 artifact already differs; refusing to overwrite it by V1 reuse')
        else:
            legacy._atomic_bytes(target, source_path.read_bytes())
    old_path = Path(old_cell['folder']) / 'evaluation.json'
    legacy._atomic_bytes(folder / 'reused_v1_evaluation.json', old_path.read_bytes())
    record = dict(old, profile=PROFILE, binding=binding, binding_sha256=legacy._digest(binding),
        metrics=evaluated['metrics'], diagnostics=evaluated['diagnostics'],
        dose_rows=evaluated['dose_rows'], source_scientific_audit=audit,
        reuse_provenance=dict(source_folder=old_cell['folder'],
            source_plan_sha256=old_plan['plan_sha256'], record_sha256=old['record_sha256'],
            evaluation_file_sha256=legacy._sha(old_path), original_binding=old['binding'],
            operation='authenticated artifact copy and array-only reporting extension; no new repair/evaluation'),
        created_utc=datetime.now(timezone.utc).isoformat())
    record = _write_result(folder, record)
    _read_record(folder, binding)
    print(f"[T5-v2] reused authenticated V1: {cell['cell_id']} (no repair/evaluation)", flush=True)
    return record


def _execute(plan, source, cell, device):
    import numpy as np
    from eval_a1_target_pck import _load_weights, _restore_trigger, _dataset_record, _assert_replay
    from defend.a1_defense_eval import clean_loaders, evaluate_defense
    from defend.a1_repair_ops import train_repair, DEFAULT_RECIPE
    from defend.a1_anp import train_anp
    folder = Path(cell['folder'])
    with legacy._cell_lock(folder):
        original, run, artifacts, inputs, audit, bank, binding = _context(plan, source, cell)
        cached = _read_record(folder, binding)
        if cached is not None:
            print(f"[T5-v2] verified complete: {cell['cell_id']}", flush=True)
            return cached
        model, trigger_state = _load_weights(run, device)
        cfg = copy.deepcopy(original.get('evaluation_trigger_cfg', run['cfg']))
        cfg.update(device=device, num_workers=0)
        trigger = _restore_trigger(cfg, trigger_state)
        test, test_inputs, test_ids = _dataset_record(cfg)
        if any(test_inputs[name] != inputs[name] for name in ('train', 'eval')):
            raise ValueError('Frozen input identities changed')
        metadata = dict(defense='none', complete=True, optimizer_steps=0, original_checkpoint_reused=True)
        if cell['defense'] != 'none':
            fit, selection = clean_loaders(legacy._bank(plan, original, inputs)[0], bank,
                original['cfg'], cell['seed'], batch_size=plan['recipe']['batch_size'])
            if cell['defense'] == 'anp':
                model, metadata = train_anp(model, fit, selection, plan['anp_recipe'], device,
                    folder / 'repair.pt', legacy._digest(binding), cell['seed'])
            else:
                recipe = {k: plan['recipe'][k] for k in DEFAULT_RECIPE}
                model, metadata = train_repair(model, fit, selection, cell['defense'], recipe,
                    device, folder / 'repair.pt', legacy._digest(binding), cell['seed'])
            if not metadata.get('complete'):
                raise ValueError('Incomplete defense repair')
        evaluated = enrich_evaluation(evaluate_defense(model, test, trigger, cfg, device))
        replay = None
        if cell['defense'] == 'none':
            metrics = dict(evaluated['metrics'], **evaluated['diagnostics'])
            metrics['t1_mpjpe_mm'] = evaluated['metrics']['t_mpjpe_mm']
            replay = _assert_replay(run['res'], metrics, .05, attack=True)
        after, after_hashes, after_inputs, _ = legacy._audit_source(original)
        if artifacts != after_hashes or inputs != after_inputs or run['fingerprint'] != after['fingerprint']:
            raise ValueError('Immutable original artifacts/inputs changed during evaluation')
        if _sources() != plan['sources']:
            raise ValueError('Code changed while cell was running; no result accepted')
        data = io.BytesIO()
        np.savez_compressed(data, **dict(evaluated['arrays'], pair_ids=np.asarray(test_ids, dtype=str)))
        legacy._atomic_bytes(folder / 'per_sample.npz', data.getvalue())
        record = dict(schema=1, profile=PROFILE, status='complete', cell_id=cell['cell_id'],
            seed=cell['seed'], method='a1', defense=cell['defense'], control=False,
            binding=binding, binding_sha256=legacy._digest(binding), metrics=evaluated['metrics'],
            diagnostics=evaluated['diagnostics'], dose_rows=evaluated['dose_rows'], repair=metadata,
            source_scientific_audit=audit, no_defense_replay_deltas_mm=replay,
            per_sample_file='per_sample.npz', per_sample_sha256=legacy._sha(folder / 'per_sample.npz'),
            created_utc=datetime.now(timezone.utc).isoformat())
        if cell['defense'] != 'none':
            record.update(repair_file='repair.pt', repair_sha256=legacy._sha(folder / 'repair.pt'))
        record = _write_result(folder, record)
        print(f"[T5-v2] completed {cell['cell_id']}", flush=True)
        return record


def _collect(plan, source, *, data=True):
    records, contexts = [], {}
    for cell in plan['cells']:
        if not (Path(cell['folder']) / 'evaluation.json').exists():
            continue
        # Audit the source/input identities once per seed, not once per metric.
        key = cell['seed']
        if key not in contexts:
            contexts[key] = _context(plan, source, cell, data=data)
        original, run, artifacts, inputs, _, bank, _ = contexts[key]
        expected = legacy._binding(plan, cell, original, run, artifacts, inputs, bank)
        records.append(_read_record(cell['folder'], expected))
    return records


def export_records(plan, records):
    report = build_metric_report(records)
    output = Path(plan['outdir'])
    for name in ('per_seed', 'six_dose', 'paired_deltas', 'aggregates', 'paired_aggregates'):
        rows = report[name]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        legacy._atomic_bytes(output / f'defense_{name}.csv', legacy._csv(rows, fields))
    report.update(plan_sha256=plan['plan_sha256'], profile=PROFILE,
        source_plan_sha256=plan['source_plan_sha256'],
        record_provenance=[dict(cell_id=r['cell_id'], record_sha256=r['record_sha256'],
            reused_v1=bool(r.get('reuse_provenance'))) for r in records])
    legacy._atomic_json(output / 'defense_summary.json', report)
    lines = [f"# {report['status']}", '', f"Records: {len(records)}/15.", '',
        'Dose 1; no alignment. Target-PCK is joint-level, threshold .2 of clean GT scale (5,12).',
        'Clean MPJPE: all 17 joints. Triggered-GT MPJPE: target joints [2,3] against true pose.',
        'Same-joint clean reference, PA, no-trigger controls, six-dose diagnostics and paired changes are in CSV/JSON.',
        'A small Target-PCK alone does not prove recovery. DCE is not a defender-success score.',
        'FT/FP pilot results remain historical; this revised method set was selected after viewing seed 42.', '',
        '| Defense | Seeds | T-MPJPE (mm) ↑ | Target-PCK (%) ↓ | Clean MPJPE (mm) ↓ | Triggered-GT MPJPE (mm) ↓ |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for a in report['aggregates']:
        if a['complete']:
            values = [f"{a[m + '_mean']:.2f} ± {a[m + '_sd']:.2f}" for m in MAIN_METRICS]
            lines.append('| ' + ' | '.join([LABELS[a['defense']], '42/43/44', *values]) + ' |')
        else:
            for row in report['per_seed']:
                if row['defense'] == a['defense']:
                    lines.append('| ' + ' | '.join([LABELS[row['defense']], str(row['seed']),
                        *[f'{row[m]:.2f}' for m in MAIN_METRICS]]) + ' |')
    legacy._atomic_bytes(output / 'table5_defenses.md', ('\n'.join(lines) + '\n').encode('utf-8'))
    return report['status']


def _schedule(plan, source, selected, device, jobs):
    if type(jobs) is not int or jobs not in (1, 2):
        raise ValueError('jobs must be 1 or 2')
    if len({c['cell_id'] for c in selected}) != len(selected):
        raise ValueError('Duplicate selected defense cells')
    pending, running = list(selected), {}
    try:
        while pending or running:
            for slot, (child, cell, stream) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                del running[slot]
                if code:
                    raise RuntimeError(f"{cell['cell_id']} failed ({code}); see console.log. No auto-retry.")
                if _read_record(cell['folder']) is None:
                    raise RuntimeError('Child exited without a verified record')
                print(f"[T5-v2] finished {cell['cell_id']}", flush=True)
            for slot in range(jobs):
                if slot in running:
                    continue
                while pending:
                    cell = pending.pop(0)
                    if _read_record(cell['folder']) is not None:
                        print(f"[T5-v2] verified complete: {cell['cell_id']} (no repair/evaluation)", flush=True)
                        continue
                    folder = Path(cell['folder'])
                    folder.mkdir(parents=True, exist_ok=True)
                    stream = (folder / 'console.log').open('a', encoding='utf-8')
                    command = [sys.executable, '-u', str(Path(__file__).resolve()),
                        '--_manifest', str(Path(plan['outdir']) / MANIFEST),
                        '--_cell', cell['cell_id'], '--device', device]
                    try:
                        env = os.environ.copy()
                        env.setdefault('PYTHONUTF8', '1')
                        child = subprocess.Popen(command, cwd=HERE, env=env,
                            stdout=stream, stderr=subprocess.STDOUT)
                    except BaseException:
                        stream.close()
                        raise
                    running[slot] = child, cell, stream
                    print(f"[T5-v2] {cell['cell_id']} -> {device}, slot={slot + 1}, PID={child.pid}\n"
                          f"       log: {folder / 'console.log'}", flush=True)
                    break
            if running:
                time.sleep(1)
    except BaseException:
        runtime_v1._stop_owned(running)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-outdir', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--reuse-outdir', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--defenses', nargs='+', default=list(DEFENSES))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--jobs-per-gpu', type=int, choices=(1, 2), default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--export-only', action='store_true')
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device != 'cpu' and not re.fullmatch(r'cuda:(0|[1-9][0-9]*)', args.device):
        parser.error('--device must be one logical cuda:N device')
    if args._cell:
        if not args._manifest:
            parser.error('Internal cell requires manifest')
        saved = legacy._json(args._manifest)
        plan, source = _load_plan(saved['source_outdir'], saved['outdir'])
        cells = [c for c in plan['cells'] if c['cell_id'] == args._cell]
        if len(cells) != 1:
            raise ValueError('Unknown internal cell')
        _execute(plan, source, cells[0], args.device)
        return 0
    if args.source_outdir is None or args.outdir is None:
        parser.error('--source-outdir and --outdir are required')
    plan, source = _load_plan(args.source_outdir, args.outdir, args.reuse_outdir)
    selected = _selected(plan, args.seeds, args.defenses)
    if args.dry_run:
        print(f'[T5-v2] DRY RUN: {len(selected)}/15 evaluations; four defenses + No defense; '
              f'{args.jobs_per_gpu} slots; workers=0. NO writes/data/checkpoints/GPU/training.')
        print(f"[T5-v2] plan {plan['plan_sha256']}; source {plan['source_plan_sha256']}")
        return 0
    if args.status:
        count = sum((Path(c['folder']) / 'evaluation.json').is_file() for c in selected)
        print(f'[T5-v2] artifact presence {count}/{len(selected)}; NOT a fresh scientific audit')
        return 0
    output = Path(plan['outdir'])
    with legacy._cell_lock(output):
        plan, source = _load_plan(args.source_outdir, args.outdir, args.reuse_outdir)
        selected = _selected(plan, args.seeds, args.defenses)
        runtime_v1._check_idle_cells(plan)
        if not (output / MANIFEST).exists():
            legacy._atomic_json(output / MANIFEST, plan)
        if not args.export_only:
            runtime_v1._prepare_banks(plan, source, selected)
            _collect(plan, source)
            for cell in selected:
                if _read_record(cell['folder']) is None:
                    with legacy._cell_lock(cell['folder']):
                        _reuse_record(plan, source, cell)
            gc.collect()
            now = datetime.now(timezone.utc)
            legacy._atomic_json(output / 'runtime_launches' / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}.json",
                dict(created_utc=now.isoformat(), device=args.device, jobs_per_gpu=args.jobs_per_gpu,
                     controller_pid=os.getpid(), cells=[c['cell_id'] for c in selected],
                     plan_sha256=plan['plan_sha256']))
            _schedule(plan, source, selected, args.device, args.jobs_per_gpu)
        records = _collect(plan, source)
        print(f"[{export_records(plan, records)}] {output / 'table5_defenses.md'}", flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[T5-v2] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
