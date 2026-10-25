"""Fresh Linux/Vast seed42 MM-Fi attack cells, without a three-seed export.

The root scheduler initializes this manifest once and launches one --cell per
process. This worker never spawns peers or rewrites the shared manifest. Old
Windows runs remain external historical evidence, not checkpoint-resume input.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import sys

if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import run_a1_three_tables as frozen
import vast_seed42_adapter as adapter

HERE = Path(__file__).resolve().parent
PROFILE = 'vast_mmfi_seed42_fresh_attacks_v1'
MANIFEST = 'vast_seed42_attack.resolved.json'
EXECUTION_RECEIPT = 'vast_execution.json'
BACKBONES = adapter.BACKBONES
KEYS = adapter.KEYS
ATTACK_DEFENSE_KEYS = ('clean', 'a1', *[key for key, _ in frozen.BASELINE_METHODS])
EXPERIMENT_SCOPES = ('full', 'attack-defense')
FORBIDDEN_MANIFESTS = ('three_tables.resolved.json', 'cross_backbones.resolved.json',
    'wiflow_a1.resolved.json', 'a1_defenses.resolved.json', 'a1_defenses_v2.resolved.json',
    'wipose_hpeli.resolved.json', 'wipose_hpeli_parallel.resolved.json', 'checkpoint.pt')


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _device(value):
    if value != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', value) is None:
        raise ValueError('Use cpu or an explicit logical cuda:N; no GPU fallback')
    return value


def runtime_versions():
    try:
        packages = {name: version(name) for name in ('torch', 'torchvision', 'numpy', 'scipy', 'PyYAML')}
    except PackageNotFoundError as error:
        raise RuntimeError(f'Missing required package: {error.name}; install matching dependencies before planning') from error
    return dict(python=platform.python_version(), system=platform.system(),
                machine=platform.machine(), packages=packages)


def source_hashes():
    import run_a1_cross_backbones as cross
    import run_a1_wiflow as wiflow
    return dict(cross.source_hashes(), **wiflow.source_hashes(), **{
        name: frozen._file_sha(HERE/name) for name in
        ('vast_seed42_attack.py', 'vast_seed42_adapter.py')})


def configuration(data_home, device, key, backbone):
    return adapter.canonical_config(data_home, _device(device), key, backbone)


def _backbones(values):
    values = tuple(values)
    if not values or len(set(values)) != len(values) or set(values)-set(BACKBONES):
        raise ValueError('Select distinct hpeli, metafiplusplus and/or wiflow backbones')
    return tuple(name for name in BACKBONES if name in values)


def _devices(values):
    values = tuple(values)
    if not values or len(set(values)) != len(values):
        raise ValueError('Distinct explicit logical devices required')
    return tuple(_device(value) for value in values)


def _output(data_home, outdir):
    home, output = Path(data_home).resolve(), Path(outdir).resolve()
    dataset = home/'datasets/Compress'
    if output == dataset or output in dataset.parents or dataset in output.parents:
        raise ValueError('Vast output must be separate from the dataset')
    for parent in (output, *output.parents):
        if any((parent/name).exists() for name in FORBIDDEN_MANIFESTS):
            raise ValueError('Fresh Vast attacks need a NEW output, not a historical run')
        if parent != output and (parent/MANIFEST).exists():
            raise ValueError('Vast output must not be nested in another attack run')
    return home, output


def validate_experiment_scope(experiment_scope):
    if type(experiment_scope) is not str or experiment_scope not in EXPERIMENT_SCOPES:
        raise ValueError('Experiment scope must be full or attack-defense')
    return experiment_scope


def build_cells(data_home, outdir, devices=('cuda:0',), *, backbones=BACKBONES,
                experiment_scope='full'):
    """Pure metadata planning; no dataset/model/CUDA access or output writes."""
    experiment_scope = validate_experiment_scope(experiment_scope)
    home, output = _output(data_home, outdir)
    devices, selected = _devices(devices), _backbones(backbones)
    cells = []
    for backbone in selected:
        a1 = configuration(home, devices[0], 'a1', backbone)
        for key in KEYS if experiment_scope == 'full' else ATTACK_DEFENSE_KEYS:
            folder = output/backbone/'seed42'/key
            value = dict(method_key=f'{backbone}__s42__{key}', backbone=backbone,
                key=key, seed=42, label=frozen.LABELS[key], origin='fresh',
                cfg=configuration(home, devices[0], key, backbone),
                tables=(['T1'] if experiment_scope == 'attack-defense' else
                    ['T1', 'T2', 'T3'] if key == 'a1' else ['T3']
                    if key in ('a2', 'a3', 'a4') else ['T2'] if key.startswith('rho') else ['T1']),
                # Independent scratch victims can fill every external slot.
                # Clean is mandatory for reporting/defense, never a source of
                # training weights or an implicit worker-launch dependency.
                dependencies=[],
                run_dir=str(folder), ckpt_dir=str(folder),
                eval_cache=str(folder/'eval_cache.json'),
                artifact_dir=str(output/backbone/'evaluations'/f's42__{key}'))
            if key == 'clean':
                value['evaluation_trigger_cfg'] = copy.deepcopy(a1)
            if experiment_scope != 'full':
                value['experiment_scope'] = experiment_scope
            cells.append(value)
    return cells


def build_manifest(data_home, outdir, devices=('cuda:0',), *, backbones=BACKBONES,
                   experiment_scope='full'):
    experiment_scope = validate_experiment_scope(experiment_scope)
    home, output = _output(data_home, outdir)
    devices, selected = _devices(devices), _backbones(backbones)
    value = dict(schema=1, profile=PROFILE, dataset='mmfi', seeds=[42],
        data_home=str(home), outdir=str(output), devices=list(devices), backbones=list(selected),
        cells=build_cells(home, output, devices, backbones=selected,
                          experiment_scope=experiment_scope),
        sources=source_hashes(), runtime_versions=runtime_versions(),
        metrics_contract=dict(status='SINGLE_SEED_NO_ACROSS_SEED_UNCERTAINTY',
            tables=['T1', 'T2', 'T3'], epochs=50, checkpoint_epoch=49,
            counts=dict(frozen.EXPECTED_COUNTS), protocol='protocol1-s1', split_seed=0,
            target_joints=[2, 3], target_pck_threshold=.2, clean_scale_joints=[5, 12],
            dose_grid=list(frozen.GRID), poison_rates=[.1, .2, .4],
            fresh_victims_per_backbone=13, a1_shared_across_tables=True,
            scratch=True, victim_loss='ordinary MPJPE', checkpoint_selection='fixed final49',
            no_test_tuning=True, evaluation_all_arms_checkpoint_forward_replay=True,
            per_sample_npz='ordered IDs, dose1 target/no-trigger errors and clean GT scale; not six-dose poses',
            input_audit_scope='ordered identities and action content; root dataset-content receipt is additional'),
        limitations=['One seed; no across-seed SD/CI or three-seed-complete claim.',
            'Digital CSI/HPE adaptations, not over-the-air/RF demonstrations.',
            'Native attack operators have unequal realized perturbation budgets.',
            'New execution explicitly uses deterministic FP32/highest precision with TF32 off for both attack and defense replay.',
            'WiFlow ordinary MPJPE/fixedLR/physical32+accumulation2 is a disclosed adaptation.',
            'No old Windows checkpoint migration/resume or new defense implementation in this manifest.'])
    if experiment_scope != 'full':
        value['experiment_scope'] = experiment_scope
        value['metrics_contract'].update(tables=['T1'], poison_rates=[.1],
            fresh_victims_per_backbone=8, a1_shared_across_tables=False,
            six_dose_diagnostics_not_a_dose_poison_table=True)
        value['limitations'].append('Fresh backbones report only T1 and paired T5; no fresh T2/T3 study.')
    value['plan_sha256'] = _sha_json(value)
    return value


def validate_manifest(manifest):
    if (manifest.get('schema') != 1 or manifest.get('profile') != PROFILE
            or manifest.get('plan_sha256') != _sha_json(
                {key: value for key, value in manifest.items() if key != 'plan_sha256'})):
        raise ValueError('Invalid or modified Vast attack manifest')
    canonical = build_manifest(manifest['data_home'], manifest['outdir'],
                               manifest['devices'], backbones=manifest['backbones'],
                               experiment_scope=manifest.get('experiment_scope', 'full'))
    if manifest != canonical:
        raise ValueError('Frozen Vast source/runtime/config/path changed; use a NEW output')
    return manifest


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if path != Path(manifest['outdir'])/MANIFEST:
        raise ValueError('Vast manifest is not at its original exact output path')
    return validate_manifest(manifest)


def write_manifest(manifest):
    """Initialize once before external workers; never rewrite an existing plan."""
    validate_manifest(manifest)
    root = Path(manifest['outdir'])
    path = root/MANIFEST
    with frozen._cell_lock(root):
        if path.is_file():
            if load_manifest(path) != manifest:
                raise ValueError('Existing Vast manifest differs')
        else:
            if any(child.name != 'run.lock' for child in root.iterdir()):
                raise ValueError('Nonempty output has no Vast manifest; choose a NEW directory')
            frozen._atomic_json(path, manifest)
    return path


def cell_for(manifest, cell_id, device=None):
    found = [cell for cell in manifest['cells'] if cell['method_key'] == cell_id]
    if len(found) != 1:
        raise ValueError('Unknown Vast seed42 cell')
    cell = copy.deepcopy(found[0])
    if device is not None:
        cell['cfg']['device'] = _device(device)
        if 'evaluation_trigger_cfg' in cell:
            cell['evaluation_trigger_cfg']['device'] = device
    return cell


def _validate_cell(cell):
    scope = validate_experiment_scope(cell.get('experiment_scope', 'full'))
    if cell.get('key') not in (KEYS if scope == 'full' else ATTACK_DEFENSE_KEYS):
        raise ValueError('Cell key is outside its frozen experiment scope')
    cfg = cell['cfg']
    if cell.get('seed') != 42 or cell.get('origin') != 'fresh':
        raise ValueError('Only fresh seed42 Vast cells may execute')
    expected = configuration(Path(cfg['dataset_root']).parents[1], cfg['device'],
                             cell['key'], cell['backbone'])
    if cfg != expected or cell['method_key'] != f'{cell["backbone"]}__s42__{cell["key"]}':
        raise ValueError('Noncanonical Vast model/attack recipe')
    run, checkpoint, artifacts = (Path(cell[name]).resolve()
                                  for name in ('run_dir', 'ckpt_dir', 'artifact_dir'))
    if (run != checkpoint or artifacts == run or artifacts in run.parents
            or run in artifacts.parents or Path(cell['eval_cache']).resolve() != run/'eval_cache.json'):
        raise ValueError('Checkpoint and evaluation outputs must be separate canonical paths')
    canonical_cells = build_cells(Path(cfg['dataset_root']).parents[1], run.parents[2],
                                  (cfg['device'],), backbones=(cell['backbone'],),
                                  experiment_scope=scope)
    expected_cell = next(value for value in canonical_cells if value['key'] == cell['key'])
    if cell != expected_cell:
        raise ValueError('Noncanonical Vast cell identity/path/dependency metadata')


def current_inputs(cell):
    value = frozen._current_inputs(cell)
    value['profile'] = PROFILE
    return value


def _assert_sources(sources):
    if sources != source_hashes():
        raise ValueError('Vast scientific/model source changed; preserve artifacts, do not resume')


def _execution_context(cell):
    """Capture the RNG-relevant logical-device assignment before training."""
    from run_piw_hpeli_clean import _setup_execution
    # The paired No-defense evaluator uses this same explicit backend setup.
    # Leaving CuDNN TF32 at its process default can make an identical checkpoint
    # miss the strict 0.05mm replay check. Old source/config bytes stay intact;
    # the new runtime adaptation is declared and bound in this receipt.
    backend = _setup_execution(cell['cfg']['device'])
    value = dict(logical_device=cell['cfg']['device'],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        hostname=platform.node(), runtime_versions=runtime_versions(), backend=backend, gpu=None)
    if cell['cfg']['device'].startswith('cuda:'):
        import torch
        target = torch.device(cell['cfg']['device'])
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no automatic CPU fallback')
        properties = torch.cuda.get_device_properties(target)
        value['gpu'] = dict(name=properties.name,
            capability=[properties.major, properties.minor],
            total_memory=properties.total_memory,
            uuid=str(getattr(properties, 'uuid', None)),
            logical_device_count=torch.cuda.device_count(),
            torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return value


def _read_execution(cell):
    """Validate saved provenance only; never query the current GPU for a cache."""
    from train_backdoor import _config_fingerprint
    path = Path(cell['run_dir'])/EXECUTION_RECEIPT
    if not path.is_file():
        raise ValueError('Fresh Vast cell has no execution receipt; never import old checkpoints')
    saved = frozen._json(path)
    expected_header = dict(schema=1, profile=PROFILE, method_key=cell['method_key'],
                           config_fingerprint=_config_fingerprint(cell['cfg']))
    if (any(saved.get(key) != value for key, value in expected_header.items())
            or set(saved) != set(expected_header)|{'execution', 'receipt_sha256'}
            or not isinstance(saved.get('execution'), dict)
            or saved.get('receipt_sha256') != _sha_json(
                {key: value for key, value in saved.items() if key != 'receipt_sha256'})):
        raise ValueError('Vast execution receipt binding/hash mismatch')
    return saved


def _bind_execution(cell):
    """Incomplete checkpoints may resume only on their exact saved assignment."""
    from train_backdoor import _config_fingerprint
    folder = Path(cell['run_dir'])
    path = folder/EXECUTION_RECEIPT
    actual = _execution_context(cell)
    if path.exists():
        saved = _read_execution(cell)
        if saved['execution'] != actual:
            raise ValueError('Execution/logical GPU/CUDA_VISIBLE_DEVICES/runtime changed; '
                             'pin the original worker assignment, never silently resume')
        return saved
    if any((folder/name).exists() for name in ('checkpoint.pt', 'eval_cache.json')):
        raise ValueError('Checkpoint/cache without its fresh execution receipt; do not migrate old weights')
    value = dict(schema=1, profile=PROFILE, method_key=cell['method_key'],
                 config_fingerprint=_config_fingerprint(cell['cfg']), execution=actual)
    value['receipt_sha256'] = _sha_json(value)
    frozen._atomic_json(path, value)
    return _read_execution(cell)


def completed(cell, sources):
    _validate_cell(cell)
    _assert_sources(sources)
    if not (Path(cell['artifact_dir'])/'evaluation.json').is_file():
        return None
    _read_execution(cell)
    run = frozen._read_completed(cell)
    inputs = current_inputs(cell)
    frozen._audit_run(cell, run, inputs)
    return adapter.evaluation_engine(sources, cell['backbone']).load_evaluation(
        dict(cell, verified_inputs=inputs), cell['artifact_dir'])


def execute_cell(cell, sources):
    """Train/resume exactly one fresh cell, then export authenticated replay."""
    _validate_cell(cell)
    _assert_sources(sources)
    output = Path(cell['run_dir'])
    with frozen._cell_lock(output):
        saved = completed(cell, sources)
        if saved is not None:
            print(f'[Vast attack] VERIFIED {cell["method_key"]}; no training/inference', flush=True)
            return saved
        execution = _bind_execution(cell)
        current = current_inputs(cell)
        inputs_path = output/'three_tables_inputs.json'
        if inputs_path.exists():
            if frozen._json(inputs_path) != current:
                raise ValueError('Training inputs changed; refuse to resume')
        elif any((output/name).exists() for name in ('checkpoint.pt', 'eval_cache.json')):
            raise ValueError('Checkpoint/cache has no training-time input audit')
        else:
            frozen._atomic_json(inputs_path, current)
        engine = adapter.training_engine(cell['cfg'])
        model, _ = engine.train(cell['cfg'], ckpt_dir=output)
        del model, engine
        _assert_sources(sources)
        if current_inputs(cell) != current:
            raise ValueError('Ordered inputs/action changed during training/evaluation')
        run = frozen._read_completed(cell)
        audit = dict(frozen._audit_run(cell, run, current),
                     execution_receipt_sha256=execution['receipt_sha256'])
        result = adapter.evaluation_engine(sources, cell['backbone']).evaluate_cell(
            dict(cell, verified_inputs=current), cell['artifact_dir'], cell['cfg']['device'])
        frozen._atomic_json(output/'scientific_audit.json', audit)
        if completed(cell, sources) is None:
            raise ValueError('Cell returned without authenticated completion')
        print(f'[Vast attack] COMPLETE {cell["method_key"]}', flush=True)
        return result


def selected_cells(manifest, requested=None, *, include_dependencies=False, device=None):
    names = [cell['method_key'] for cell in manifest['cells']] if requested is None else list(requested)
    available = {cell['method_key']: cell for cell in manifest['cells']}
    if not names or len(set(names)) != len(names) or set(names)-set(available):
        raise ValueError('Unknown, empty or duplicate selected Vast cells')
    if include_dependencies:
        names = list(set(names) | {dep for name in names for dep in available[name]['dependencies']})
    return [cell_for(manifest, cell['method_key'], device) for cell in manifest['cells']
            if cell['method_key'] in names]


def export_records(manifest, cells=None):
    selected = selected_cells(manifest) if cells is None else cells
    for backbone in {cell['backbone'] for cell in selected}:
        clean = cell_for(manifest, f'{backbone}__s42__clean')
        if completed(clean, manifest['sources']) is None:
            raise ValueError('Authenticated Clean is mandatory before reporting '+backbone)
    records = [completed(cell, manifest['sources']) for cell in selected]
    if any(record is None for record in records):
        raise ValueError('Missing authenticated seed42 attack evaluation')
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--devices', nargs='+', default=None, help='Manifest initialization only')
    parser.add_argument('--backbones', nargs='+', choices=BACKBONES)
    parser.add_argument('--experiment-scope', choices=EXPERIMENT_SCOPES, default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--cell')
    selection.add_argument('--cells', nargs='+')
    mode = parser.add_mutually_exclusive_group()
    for name in ('dry-run', 'init-only', 'status', 'check-only', 'export-only'):
        mode.add_argument('--'+name, action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = args.outdir.resolve()/MANIFEST
    if path.is_file():
        manifest = load_manifest(path)
        if args.data_home is not None and str(args.data_home.resolve()) != manifest['data_home']:
            raise ValueError('Requested data root differs from the saved plan')
        if args.backbones is not None and list(_backbones(args.backbones)) != manifest['backbones']:
            raise ValueError('Requested backbones differ from the saved plan')
        if args.devices is not None and list(_devices(args.devices)) != manifest['devices']:
            raise ValueError('Requested logical device set differs from the saved plan')
        if args.experiment_scope is not None and args.experiment_scope != manifest.get('experiment_scope', 'full'):
            raise ValueError('Requested experiment scope differs from the saved plan')
    else:
        if args.status or args.check_only or args.export_only or args.cell:
            raise FileNotFoundError('Initialize the Vast attack manifest before starting workers')
        if args.data_home is None:
            raise ValueError('--data-home required to initialize the Vast plan')
        manifest = build_manifest(args.data_home, args.outdir,
            args.devices or [args.device or 'cuda:0'], backbones=args.backbones or BACKBONES,
            experiment_scope=args.experiment_scope or 'full')
    requested = [args.cell] if args.cell else args.cells
    cells = selected_cells(manifest, requested,
                           include_dependencies=bool(args.cells), device=args.device)
    if args.dry_run:
        print(f'[Vast attack] DRY RUN: {len(cells)} selected; seed42; independent50epochvictims; '
              'NO writes/data/model/CUDA/training.')
        print(json.dumps(manifest, indent=2))
        return 0
    if args.init_only:
        write_manifest(manifest)
        print(f'[Vast attack] INITIALIZED {path}; no data/model/CUDA/training')
        return 0
    if args.status:
        for cell in cells:
            present = (Path(cell['artifact_dir'])/'evaluation.json').is_file()
            print(f'{cell["method_key"]}: '+('PRESENT_UNVERIFIED' if present else 'MISSING'))
        return 0
    if args.check_only:
        missing = []
        for cell in cells:
            if completed(cell, manifest['sources']) is None:
                missing.append(cell['method_key'])
            else:
                print(f'VERIFIED {cell["method_key"]}')
        if missing:
            print('INCOMPLETE: '+', '.join(missing))
        return int(bool(missing))
    if args.export_only:
        records = export_records(manifest, cells)
        frozen._atomic_json(Path(manifest['outdir'])/'seed42_attack_results.json',
            dict(status='VALIDATED_SELECTED_SINGLE_SEED_RECORDS', seed=42,
                 plan_sha256=manifest['plan_sha256'], records=records,
                 no_across_seed_uncertainty=True))
        return 0
    frozen._check_inputs(dict(cells=cells), list({cell['cfg']['device'] for cell in cells}))
    if not path.is_file():
        write_manifest(manifest)
    for cell in cells:
        execute_cell(cell, manifest['sources'])
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('[Vast attack] ERROR: '+str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
