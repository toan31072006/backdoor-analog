"""Data-free CLP sidecar: three original A1 victims, no fit or victim training.

Existing v1/v2 scientific files and artifacts are read-only. A separate output
can report CLP alone or combine authenticated v2 records without changing them.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
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
import run_a1_defenses_v2 as v2
from defend.a1_defense_metrics import enrich_evaluation
from defend.a1_clp_reporting import (SEEDS, DEFENSES, LABELS, MAIN_METRICS,
                                    METRICS, build_metric_report)

HERE = Path(__file__).resolve().parent
MANIFEST = 'a1_clp.resolved.json'
PROFILE = 'mmfi_a1_clp_data_free_three_seed_v1'
CLP_REPAIR_SCHEMA = 'a1_clp_data_free_repair_checkpoint_v1'
RECIPE = dict(u=3.0, allow_unpaired_bn2d=['skunit1.conv2_sk.1', 'skunit2.conv2_sk.1'])


def _sources():
    names = ('run_a1_clp.py', 'defend/a1_clp.py', 'defend/a1_clp_sources.json',
             'defend/a1_clp_reporting.py', *v2._sources().keys())
    return {name: legacy._sha(HERE / name) for name in names}


def _reference(reference_outdir, source_root, source, output):
    if reference_outdir is None:
        return None
    root = Path(reference_outdir).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('CLP reference and output roots must be separate siblings')
    # Loading metadata authenticates source/default pins but never creates files.
    if not (root / v2.MANIFEST).is_file():
        raise ValueError('Reference requires an existing V2 manifest; never create an empty reference run')
    plan, _ = v2._load_plan(source_root, root)
    if plan['source_plan_sha256'] != source['plan_sha256']:
        raise ValueError('Reference V2 and CLP do not share the same original scientific plan')
    return dict(outdir=str(root), plan_sha256=plan['plan_sha256'],
                access='read-only; copied into sidecar reports only')


def build_plan(source, source_outdir, outdir, *, reference_outdir=None):
    """Metadata-only preflight. No model/checkpoint/data/CUDA access or writes."""
    plan = legacy.build_plan(source, source_outdir, outdir)
    output = Path(plan['outdir'])
    declared = legacy._json(HERE / 'defend/a1_clp_sources.json')
    from defend.a1_clp import DEFAULT_U, HPELI_UNPAIRED_BN2D
    if (declared.get('frozen_defaults') != {'u': RECIPE['u']}
            or DEFAULT_U != RECIPE['u']
            or declared.get('allowed_unpaired_bn2d') != RECIPE['allow_unpaired_bn2d']
            or list(HPELI_UNPAIRED_BN2D) != RECIPE['allow_unpaired_bn2d']):
        raise ValueError('Declared CLP defaults/coverage differ from the frozen sidecar')
    lookup = {(c['seed'], c['key']): c for c in source['cells']}
    plan.update(profile=PROFILE, schema=1, recipe=copy.deepcopy(RECIPE), sources=_sources(),
        cells=[dict(cell_id=f's{s}__a1__clp', seed=s, method='a1', defense='clp',
                    source_cell_id=lookup[s, 'a1']['method_key'],
                    folder=str(output / f'seed{s}' / 'a1' / 'clp')) for s in SEEDS])
    plan['reference'] = _reference(reference_outdir, source_outdir, source, output)
    plan['contract'].update(defenses=['clp'], expected_sidecar_records=3, expected_combined_records=18,
        repair_loss=None, bank=None, bank_seen_by_original_victim=None,
        selection='none; fixed u=3 before TEST',
        data_free_repair=True, repair_optimizer_steps=0,
        expected_direct_conv_bn2d_pairs=10,
        excluded_composite_bn2d=list(RECIPE['allow_unpaired_bn2d']),
        excluded_bn1d='both BatchNorm1d layers are untouched',
        main_metrics=list(MAIN_METRICS), reporting_metrics=list(METRICS),
        recipe_status='CLP_SOURCE_DEFAULT_U3_DECLARED_HPE_DIRECT_PAIR_ADAPTATION',
        historical_evaluation='Fine-Pruning pilot retained; revised method selection after pilot',
        no_claim='not an improved defense, unchanged image-classification reproduction, or matched repair compute')
    plan['plan_sha256'] = legacy._plan_hash(plan)
    return plan


def _load_plan(source_root, outdir, reference_outdir=None):
    source = legacy._json(Path(source_root) / legacy.frozen.MANIFEST)
    path = Path(outdir) / MANIFEST
    if path.exists():
        saved = legacy._json(path)
        reference = reference_outdir
        if reference is None and saved.get('reference'):
            reference = saved['reference']['outdir']
        canonical = build_plan(source, source_root, outdir, reference_outdir=reference)
        if (saved.get('plan_sha256') != legacy._plan_hash(saved)
                or saved['plan_sha256'] != canonical['plan_sha256']):
            raise ValueError('CLP recipe/source/reference/path changed; use a NEW output directory')
        return saved, source
    plan = build_plan(source, source_root, outdir, reference_outdir=reference_outdir)
    if Path(outdir).exists() and any(p.name != 'run.lock' for p in Path(outdir).iterdir()):
        raise ValueError('New CLP output is not empty and has no owned manifest')
    return plan, source


def _selected(plan, seeds):
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(SEEDS):
        raise ValueError('Unknown/empty/duplicate seed selection')
    return [c for c in plan['cells'] if c['seed'] in seeds]


def _binding(plan, cell, original, run, artifacts, inputs):
    from train_backdoor import _config_fingerprint
    trigger_cfg = original.get('evaluation_trigger_cfg', run['cfg'])
    return dict(plan_sha256=plan['plan_sha256'], cell_id=cell['cell_id'],
        source_cell_id=original['method_key'], source_folder=str(run['folder']),
        source_plan_sha256=plan['source_plan_sha256'],
        source_config_fingerprint=run['fingerprint'], source_artifacts=artifacts,
        trigger_config_fingerprint=_config_fingerprint(trigger_cfg), inputs=inputs,
        recipe=plan['recipe'], sources=plan['sources'], data_free_repair=True)


def _context(plan, source, cell):
    original = legacy._source_cell(source, cell)
    run, artifacts, inputs, audit = legacy._audit_source(original)
    return original, run, artifacts, inputs, audit, _binding(plan, cell, original, run, artifacts, inputs)


def _read_record(folder, expected_binding=None):
    record = legacy._read_record(folder, expected_binding)
    if record is None:
        return None
    seed = record.get('seed')
    identity = f's{seed}__a1__clp'
    if (record.get('profile') != PROFILE or type(seed) is not int or seed not in SEEDS
            or record.get('method') != 'a1' or record.get('defense') != 'clp'
            or record.get('control') is not False or record.get('cell_id') != identity
            or record['binding'].get('cell_id') != identity
            or record['binding'].get('data_free_repair') is not True
            or 'bank_sha256' in record['binding'] or record['binding'].get('recipe') != RECIPE):
        raise ValueError('CLP record profile/cell/data-free identity changed')
    evaluated = v2._enriched_record(record, folder)
    if any(m not in record['metrics'] for m in METRICS) or record['metrics'] != evaluated['metrics']:
        raise ValueError('CLP metrics do not reproduce the saved per-sample data')
    _validate_repair_metadata(record.get('repair', {}))
    return record


def _validate_repair_metadata(metadata):
    if (metadata.get('complete') is not True or metadata.get('optimizer_steps') != 0
            or metadata.get('schema') != 'a1_clp_weight_only_v1'
            or metadata.get('method') != 'clp'
            or metadata.get('u') != RECIPE['u'] or metadata.get('uses_data') is not False
            or metadata.get('uses_labels') is not False or metadata.get('uses_test_selection') is not False
            or metadata.get('paired_bn2d_count') != 10 or metadata.get('total_bn2d_count') != 12
            or metadata.get('paired_channels') != 960
            or metadata.get('unsupported_bn1d_count') != 2
            or metadata.get('training_steps') != 0 or metadata.get('fine_tuning_epochs') != 0
            or metadata.get('training_samples') != 0 or metadata.get('selection_samples') != 0
            or metadata.get('post_pruning_finetuning') is not False
            or metadata.get('strength_candidates') != [RECIPE['u']]
            or metadata.get('std_correction') != 1 or metadata.get('variance_epsilon_added') is not False
            or sorted(r.get('name') for r in metadata.get('skipped_bn2d', []))
               != sorted(RECIPE['allow_unpaired_bn2d'])):
        raise ValueError('CLP repair data-free/HPE coverage metadata differs from the frozen protocol')


def _save_repair(model, metadata, path, binding, original_state_sha256):
    import torch
    from defend.a1_anp import _state_digest
    data = io.BytesIO()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(schema=CLP_REPAIR_SCHEMA, binding_sha256=legacy._digest(binding),
                    recipe=copy.deepcopy(RECIPE), metadata=metadata,
                    original_state_sha256=original_state_sha256,
                    repaired_state_sha256=_state_digest(state), model=state), data)
    legacy._atomic_bytes(path, data.getvalue())


def _repair(model, folder, binding):
    """Deterministic pruning; reuse only an owned, exact-binding repair state."""
    import torch
    from defend.a1_clp import prune_clp
    from defend.a1_anp import _state_digest
    original_hash = _state_digest(model.state_dict())
    path = Path(folder) / 'repair.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved.get('schema') != CLP_REPAIR_SCHEMA
                or saved.get('binding_sha256') != legacy._digest(binding)
                or saved.get('recipe') != RECIPE
                or saved.get('original_state_sha256') != original_hash
                or saved.get('repaired_state_sha256') != _state_digest(saved['model'])):
            raise ValueError('CLP checkpoint changed/incomplete; refusing a different repair binding')
        _validate_repair_metadata(saved.get('metadata', {}))
        repaired = copy.deepcopy(model)
        repaired.load_state_dict(saved['model'], strict=True)
        repaired.eval()
        return repaired, saved['metadata']
    repaired, metadata = prune_clp(model, u=RECIPE['u'],
                                  allow_unpaired_bn2d=tuple(RECIPE['allow_unpaired_bn2d']))
    metadata = dict(metadata, complete=True, optimizer_steps=0, training_samples=0,
                    selection_samples=0, post_pruning_finetuning=False)
    _validate_repair_metadata(metadata)
    _save_repair(repaired, metadata, path, binding, original_hash)
    return repaired, metadata


def _execute(plan, source, cell, device):
    import numpy as np
    from eval_a1_target_pck import _load_weights, _restore_trigger, _dataset_record
    from defend.a1_defense_eval import evaluate_defense
    folder = Path(cell['folder'])
    with legacy._cell_lock(folder):
        original, run, artifacts, inputs, audit, binding = _context(plan, source, cell)
        cached = _read_record(folder, binding)
        if cached is not None:
            print(f"[CLP] verified complete: {cell['cell_id']} (no pruning/evaluation)", flush=True)
            return cached
        model, trigger_state = _load_weights(run, device)
        cfg = copy.deepcopy(original.get('evaluation_trigger_cfg', run['cfg']))
        cfg.update(device=device, num_workers=0)
        trigger = _restore_trigger(cfg, trigger_state)
        test, test_inputs, test_ids = _dataset_record(cfg)
        if any(test_inputs[name] != inputs[name] for name in ('train', 'eval')):
            raise ValueError('Frozen input identities changed')
        model, metadata = _repair(model, folder, binding)
        evaluated = enrich_evaluation(evaluate_defense(model, test, trigger, cfg, device))
        after, after_hashes, after_inputs, _ = legacy._audit_source(original)
        if artifacts != after_hashes or inputs != after_inputs or run['fingerprint'] != after['fingerprint']:
            raise ValueError('Immutable original artifacts/inputs changed during CLP evaluation')
        if _sources() != plan['sources']:
            raise ValueError('Code changed while CLP was running; no result accepted')
        data = io.BytesIO()
        np.savez_compressed(data, **dict(evaluated['arrays'], pair_ids=np.asarray(test_ids, dtype=str)))
        legacy._atomic_bytes(folder / 'per_sample.npz', data.getvalue())
        record = dict(schema=1, profile=PROFILE, status='complete', cell_id=cell['cell_id'],
            seed=cell['seed'], method='a1', defense='clp', control=False, binding=binding,
            binding_sha256=legacy._digest(binding), metrics=evaluated['metrics'],
            diagnostics=evaluated['diagnostics'], dose_rows=evaluated['dose_rows'], repair=metadata,
            source_scientific_audit=audit, per_sample_file='per_sample.npz',
            per_sample_sha256=legacy._sha(folder / 'per_sample.npz'),
            repair_file='repair.pt', repair_sha256=legacy._sha(folder / 'repair.pt'),
            created_utc=datetime.now(timezone.utc).isoformat())
        record = v2._write_result(folder, record)
        _read_record(folder, binding)
        print(f"[CLP] completed {cell['cell_id']}", flush=True)
        return record


def _collect(plan, source):
    records = []
    for cell in plan['cells']:
        if (Path(cell['folder']) / 'evaluation.json').exists():
            *_, binding = _context(plan, source, cell)
            records.append(_read_record(cell['folder'], binding))
    return records


def _collect_reference(plan, source, clp_records):
    reference = plan.get('reference')
    if reference is None:
        return []
    ref_plan, _ = v2._load_plan(plan['source_outdir'], reference['outdir'])
    if (ref_plan['plan_sha256'] != reference['plan_sha256']
            or ref_plan['source_plan_sha256'] != plan['source_plan_sha256']):
        raise ValueError('V2 reference plan changed')
    records = v2._collect(ref_plan, source)
    compared = {}
    # Every available seed is compared against a fresh source audit, including
    # a seed whose CLP inference has not completed. Do not trust just labels.
    for record in [*records, *clp_records]:
        seed = record['seed']
        if seed not in compared:
            cell = next(c for c in plan['cells'] if c['seed'] == seed)
            compared[seed] = _context(plan, source, cell)[-1]
        expected = compared[seed]
        for key in ('source_cell_id', 'source_folder', 'source_plan_sha256',
                    'source_config_fingerprint', 'source_artifacts',
                    'trigger_config_fingerprint', 'inputs'):
            if record['binding'].get(key) != expected[key]:
                raise ValueError(f'CLP/reference source pairing differs for seed {seed}: {key}')
    return records


def export_records(plan, records):
    report = build_metric_report(records)
    output = Path(plan['outdir'])
    for name in ('per_seed', 'six_dose', 'paired_deltas', 'aggregates', 'paired_aggregates'):
        rows = report[name]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        legacy._atomic_bytes(output / f'defense_{name}.csv', legacy._csv(rows, fields))
    report.update(plan_sha256=plan['plan_sha256'], profile=PROFILE,
        source_plan_sha256=plan['source_plan_sha256'], reference=plan.get('reference'),
        record_provenance=[dict(cell_id=r['cell_id'], profile=r['profile'],
            record_sha256=r['record_sha256'], source_artifacts=r['binding']['source_artifacts']) for r in records])
    legacy._atomic_json(output / 'defense_summary.json', report)
    lines = [f"# {report['status']}", '', f"Records: {len(records)}/18.", '',
        'CLP u=3 is fixed, data-free and direct Conv2d-BatchNorm2d only; no fit or post-pruning FT.',
        'V2 rows are authenticated read-only references. Historical Fine-Pruning results remain unchanged.',
        'Dose 1; no alignment; Target-PCK threshold .2 of clean GT scale (5,12), target joints [2,3].',
        'Clean MPJPE uses all 17 joints; Triggered-GT uses target joints [2,3] against true pose.',
        'Same-joint clean controls, paired within-seed changes, PA and six-dose diagnostics are in CSV/JSON.',
        'Low Target-PCK alone does not demonstrate recovery; DCE is not a defender-success score.', '',
        '| Defense | Seeds | T-MPJPE (mm) ↑ | Target-PCK (%) ↓ | Clean MPJPE (mm) ↓ | Triggered-GT MPJPE (mm) ↓ |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for aggregate in report['aggregates']:
        if aggregate['complete']:
            values = [f"{aggregate[m + '_mean']:.2f} ± {aggregate[m + '_sd']:.2f}" for m in MAIN_METRICS]
            lines.append('| ' + ' | '.join([LABELS[aggregate['defense']], '42/43/44', *values]) + ' |')
        else:
            for row in report['per_seed']:
                if row['defense'] == aggregate['defense']:
                    lines.append('| ' + ' | '.join([LABELS[row['defense']], str(row['seed']),
                        *[f'{row[m]:.2f}' for m in MAIN_METRICS]]) + ' |')
    legacy._atomic_bytes(output / 'table5_defenses_with_clp.md', ('\n'.join(lines) + '\n').encode('utf-8'))
    return report['status']


def _schedule(plan, selected, device, jobs):
    if type(jobs) is not int or jobs not in (1, 2):
        raise ValueError('jobs must be 1 or 2')
    if len({c['cell_id'] for c in selected}) != len(selected):
        raise ValueError('Duplicate selected CLP cells')
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
                    raise RuntimeError('CLP child exited without a verified record')
                print(f"[CLP] finished {cell['cell_id']}", flush=True)
            for slot in range(jobs):
                if slot in running:
                    continue
                while pending:
                    cell = pending.pop(0)
                    if _read_record(cell['folder']) is not None:
                        print(f"[CLP] verified complete: {cell['cell_id']} (no pruning/evaluation)", flush=True)
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
                    print(f"[CLP] {cell['cell_id']} -> {device}, slot={slot + 1}, PID={child.pid}\n"
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
    parser.add_argument('--reference-outdir', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
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
            parser.error('Internal CLP cell requires manifest')
        saved = legacy._json(args._manifest)
        plan, source = _load_plan(saved['source_outdir'], saved['outdir'])
        cells = [c for c in plan['cells'] if c['cell_id'] == args._cell]
        if len(cells) != 1:
            raise ValueError('Unknown internal CLP cell')
        _execute(plan, source, cells[0], args.device)
        return 0
    if args.source_outdir is None or args.outdir is None:
        parser.error('--source-outdir and --outdir are required')
    plan, source = _load_plan(args.source_outdir, args.outdir, args.reference_outdir)
    selected = _selected(plan, args.seeds)
    if args.dry_run:
        print(f'[CLP] DRY RUN: {len(selected)}/3 evaluations; fixed u=3, no fit; '
              f'{args.jobs_per_gpu} slots; workers=0. NO writes/data/checkpoints/GPU/training.')
        print(f"[CLP] plan {plan['plan_sha256']}; source {plan['source_plan_sha256']}")
        return 0
    if args.status:
        count = sum((Path(c['folder']) / 'evaluation.json').is_file() for c in selected)
        print(f'[CLP] artifact presence {count}/{len(selected)}; NOT a fresh scientific audit')
        return 0
    output = Path(plan['outdir'])
    with legacy._cell_lock(output):
        plan, source = _load_plan(args.source_outdir, args.outdir, args.reference_outdir)
        selected = _selected(plan, args.seeds)
        runtime_v1._check_idle_cells(plan)
        if not (output / MANIFEST).exists():
            legacy._atomic_json(output / MANIFEST, plan)
        _collect(plan, source)
        if not args.export_only:
            now = datetime.now(timezone.utc)
            legacy._atomic_json(output / 'runtime_launches' / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}.json",
                dict(created_utc=now.isoformat(), device=args.device, jobs_per_gpu=args.jobs_per_gpu,
                     controller_pid=os.getpid(), cells=[c['cell_id'] for c in selected],
                     plan_sha256=plan['plan_sha256']))
            _schedule(plan, selected, args.device, args.jobs_per_gpu)
        records = _collect(plan, source)
        references = _collect_reference(plan, source, records)
        print(f"[{export_records(plan, [*references, *records])}] {output / 'table5_defenses_with_clp.md'}", flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[CLP] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
