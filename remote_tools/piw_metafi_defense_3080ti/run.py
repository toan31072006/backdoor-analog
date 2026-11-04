"""Checkpoint-only PiW MetaFi++ 4090 -> CEI 3080 Ti light defenses.

Seven rows: Clean/NoDef plus FT, CLP, BNP, FST and NAD. No Neural Cleanse,
victim training, weight downloads, TEST selection or historical writes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import gc
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

HERE = Path(__file__).resolve().parent
PROFILE = 'EXPLORATORY_piw3d_metafi4090_s42_light_defense_3080ti_v1'
FOLDER = 'piw3d_metafi4090_light_defense_s42_3080ti_v1'
METHODS = ('ft', 'clp', 'bnp', 'fst', 'nad')
ROWS = ('clean_reference', 'none', *METHODS)
BACKBONE = 'metafiplusplus'
DOSES = (0., .2, .4, .6, .8, 1.)
METRICS = ('t_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm', 'clean_pampjpe_mm',
           'clean_target_mpjpe_mm', 'triggered_gt_mpjpe_mm',
           'triggered_gt_minus_clean_target_mm', 'mean_positive_dce_mm')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    """Exclusive first write; never replace a differing historical artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing artifact differs; inspect, do not overwrite: {path}')
        return
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_digest(value, key):
    if value.get(key) != digest({k:v for k,v in value.items() if k != key}):
        raise ValueError(f'Invalid {key}')


def metadata():
    data = read(HERE / 'source_pair.json')
    check_digest(data, 'metadata_sha256')
    for name, checksum in data['assets'].items():
        if sha(HERE / name) != checksum:
            raise ValueError('Packaged source/input metadata changed: ' + name)
    with zipfile.ZipFile(HERE / 'model_source.zip') as archive:
        if set(archive.namelist()) != set(data['source_files']):
            raise ValueError('Frozen model archive members differ')
        for name, checksum in data['source_files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError('Frozen source changed: ' + name)
    return data


def boot(repo):
    sys.path[:0] = [str(HERE / 'model_source.zip'),
                   str(repo / 'remote_linux/defense_3080ti'), str(repo / 'ATKBackd')]
    modules = {n: importlib.import_module(n) for n in
               ('run_mmfi', 'run_piw', 'gpu_smoke', 'controller')}
    for name, module in modules.items():
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise ValueError('Wrong frozen CEI module imported: ' + name)
    modules['nad'] = importlib.import_module('piw_metafi_nad_runtime_v2')
    return modules


def source_path(args, role):
    return args.clean_checkpoint if role == 'clean' else args.a1_checkpoint


def load_model(args, meta, role):
    import torch
    from piw_metafi_frozen.metafi_plus_plus_piw3d import MetaFiPlusPlusPiW3DNet
    expected = meta['checkpoints'][role]
    path = source_path(args, role)
    # Pickle is read only after authenticating the exact user-supplied checkpoint.
    if sha(path) != expected['checkpoint_sha256']:
        raise ValueError(f'Wrong {role} checkpoint SHA256: {path}')
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if (any(blob.get(k) != expected[k] for k in ('profile', 'method', 'epoch', 'plan_sha256'))
            or blob.get('backbone') != BACKBONE or not isinstance(blob.get('model'), dict)):
        raise ValueError('Wrong fresh-4090 checkpoint profile/final199')
    # Full strict state restores affine parameters AND pretrained/trained buffers.
    # No ImageNet initialization download is needed to restore a complete state.
    model = MetaFiPlusPlusPiW3DNet(pretrained=False)
    model.load_state_dict(blob['model'], strict=True)
    del blob
    return model


def compare_input_receipts(old, native):
    expected = {}
    for split in native.values():
        for row in split['files']:
            relative = row['path']
            if relative in expected:
                raise ValueError('Duplicate native source data path')
            expected[relative] = (row['size'], row['sha256'])
    actual = {n:(r['size'], r['sha256']) for n,r in old['files'].items() if n.endswith('.npy')}
    if actual != expected:
        raise ValueError('CEI data content receipt differs from fresh 4090 training inputs')


def verify_ordered_names(dataset, source, split):
    """Native ordered_names_sha256 hashes LF-joined names, NOT canonical JSON.

    Keep the full ordered-list comparison too: this is not permission to
    reorder/drop frames or silently translate trusted-bank integer indices.
    """
    expected = source['ordered_names']
    actual = [item['name'] for item in dataset.items]
    if (not expected or any(not isinstance(name, str) or not name
                            or '\n' in name or '\r' in name for name in expected)
            or len(set(expected)) != len(expected)
            or len(expected) != source['retained_count']):
        raise ValueError('Invalid source ordered identities: ' + split)
    native_sha = hashlib.sha256(('\n'.join(expected) + '\n').encode('utf-8')).hexdigest()
    if native_sha != source['ordered_names_sha256']:
        raise ValueError('Source ordered-name LF digest differs: ' + split)
    if actual != expected:
        raise ValueError('Fresh-4090 ordered identities differ: ' + split)


def recipes():
    from vast_seed42_defense_adapter import recipe
    ft, nad = recipe(BACKBONE, 'ft'), recipe(BACKBONE, 'nad')
    ft.update(epochs=10, lr=1e-4)
    if (nad['nad_layers'] != ['bn2', 'tf', 'decode.2']
            or nad['nad_teacher_epochs'] != 10 or nad['nad_student_epochs'] != 10
            or nad['nad_betas'] != [500., 1000., 1000.] or nad['attention_p'] != 2.
            or nad['epochs'] != 100 or nad['lr'] != .01
            or nad['momentum'] != .9 or nad['weight_decay'] != .0005):
        raise ValueError('Frozen NAD recipe differs; no retuning')
    return dict(ft=ft, clp=dict(u=3.), bnp=dict(u=3.),
                fst=dict(epochs=10, lr=1e-4, alpha=.1, num_keypoints=14), nad=nad)


def context(args, modules, meta):
    """Reuse sealed CEI content hashes + current stats; no whole-dataset rehash."""
    from defend.a1_defense_eval import select_clean_bank
    from piw_v2_adapter import target_contract, build_trigger, trigger_record
    old_folder = args.base / 'runs/piw3d_light_defense_s42_3080ti_v1'
    old_data = read(old_folder / 'dataset_content.json')
    root = Path(old_data['root']).resolve()
    old_receipt = modules['run_piw'].verify_receipt(args.base, root, args.repo)
    if sha(args.repo / 'ATKBackd/data_utils/feeder.py') != meta['feeder_sha256']:
        raise ValueError('CEI feeder differs from fresh 4090 source')
    if target_contract() != meta['target']:
        raise ValueError('CEI target contract differs from fresh 4090 source')
    for name, expected_sha in meta['trigger_sources'].items():
        if sha(args.repo / 'ATKBackd' / name) != expected_sha:
            raise ValueError('CEI trigger source differs from fresh 4090 source: ' + name)
    with gzip.open(HERE / 'source_inputs.json.gz', 'rt', encoding='utf-8') as stream:
        native = json.load(stream)
    compare_input_receipts(old_data, native)
    train, test = modules['run_piw'].datasets(root)
    for key, dataset in (('train_data', train), ('test_data', test)):
        verify_ordered_names(dataset, native[key], key)
    bank = select_clean_bank(modules['run_piw'].pair_ids(train), 42, .05, .2)
    original_bank_path = old_folder / 'hpeli/ft/clean_bank.json'
    ft_record = read(original_bank_path.parent / 'evaluation.json')
    check_digest(ft_record, 'record_sha256')
    if (ft_record.get('status') != 'complete'
            or sha(original_bank_path) != ft_record['artifacts'].get('clean_bank.json')
            or read(original_bank_path) != bank):
        raise ValueError('Trusted TRAIN bank differs from completed PiW FT')
    if sha(args.base / 'actions/data_bend.npy') != meta['action_sha256']:
        raise ValueError('Fresh 4090 bend prior differs')
    trigger = trigger_record(build_trigger(args.base / 'actions/data_bend.npy'))
    if trigger['protocol'] != meta['trigger_protocol']:
        raise ValueError('Fresh 4090 trigger configuration differs')
    for role in ('clean', 'a1'):
        model = load_model(args, meta, role)
        del model
    hardware = modules['gpu_smoke'].hardware()
    hardware['driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version',
        '--format=csv,noheader'], capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    plan = dict(profile=PROFILE, seed=42, backbone=BACKBONE, rows=list(ROWS),
        neural_cleanse=False, victim_retraining=False, physical_batch=32, workers=0,
        precision='fp32', target=meta['target'], doses=list(DOSES), recipes=recipes(),
        trigger=trigger,
        checkpoint_sources={r:dict(path=str(source_path(args, r)), **meta['checkpoints'][r]) for r in ('clean','a1')},
        source_pair_sha256=meta['metadata_sha256'], dataset_root=str(root),
        dataset_binding_sha256=old_receipt['dataset_binding_sha256'],
        historical_prepare_sha256=old_receipt['receipt_sha256'], clean_bank=bank,
        test_ids_sha256=digest(modules['run_piw'].pair_ids(test)),
        sources=dict(runner=sha(Path(__file__)), launcher=sha(HERE / 'start.sh'), assets=meta['assets'],
                     old_sources=old_receipt['operator_sources']),
        hardware=hardware)
    plan['plan_sha256'] = digest(plan)
    return plan, train, test


@contextmanager
def lock(folder):
    import fcntl
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another MetaFi defense worker is active') from error
        yield


def prepared(args, plan):
    gate = read(args.out / 'prepared.json')
    check_digest(gate, 'gate_sha256')
    if gate.get('plan_sha256') != plan['plan_sha256'] or set(gate.get('smokes', {})) != {'none', *METHODS}:
        raise ValueError('Prepare/source/hardware gate differs')
    for name, row in gate['smokes'].items():
        if (row.get('finite') is not True or row.get('physical_batch') != 32
                or row.get('peak_reserved_bytes', 0) <= 0):
            raise ValueError('Invalid B32 capacity evidence: ' + name)
    modules_nad = importlib.import_module('piw_metafi_nad_runtime_v2')
    modules_nad.validate_smoke(gate['smokes']['nad'], plan['plan_sha256'])
    return gate


def prepare(args, modules, meta, plan, train):
    import torch
    if (args.out / 'prepared.json').exists():
        return prepared(args, plan)
    write(args.out / 'plan.json', plan)
    domain = SimpleNamespace(csi_shape=(3, 180, 20), joints=14)
    modules['run_mmfi'].seed_execution()
    results = modules['gpu_smoke'].probe(lambda: load_model(args, meta, 'a1'),
        BACKBONE, domain, ('none', 'ft', 'clp', 'bnp', 'fst'))
    gc.collect(); torch.cuda.empty_cache()
    model = load_model(args, meta, 'a1')
    fit, _ = modules['run_piw'].clean_loaders(train, plan['clean_bank'])
    modules['run_mmfi'].seed_execution()
    torch.cuda.reset_peak_memory_stats()
    row = modules['nad'].smoke_nad(model, next(iter(fit)), ['bn2', 'tf', 'decode.2'])
    torch.cuda.synchronize()
    row.update(binding_sha256=plan['plan_sha256'],
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), capacity_only=True, efficacy_verified=False)
    results['nad'] = row
    del model, fit
    gc.collect(); torch.cuda.empty_cache()
    gate = dict(plan_sha256=plan['plan_sha256'], smokes=results)
    gate['gate_sha256'] = digest(gate)
    write(args.out / 'prepared.json', gate)
    prepared(args, plan)
    print('PREPARED_METAFI_ONLY: B32 five defenses; NC excluded; no full repair yet.', flush=True)


def binding(plan, method):
    if method not in ROWS:
        raise ValueError('Method outside the seven-row, no-NC scope')
    return digest(dict(plan_sha256=plan['plan_sha256'], method=method))


def validate_result(args, modules, plan, method):
    folder = args.out / method
    path = folder / 'evaluation.json'
    if not path.exists():
        return None
    result = read(path)
    check_digest(result, 'record_sha256')
    expected_files = {'per_sample.npz'}
    if method in METHODS:
        expected_files.add('repair.pt')
    if method in ('ft', 'bnp', 'fst', 'nad'):
        expected_files.add('clean_bank.json')
    if method == 'bnp':
        expected_files.add('bnp_pruning.json')
    if method == 'fst':
        expected_files.add('fst_progress.pt')
    role = 'clean' if method == 'clean_reference' else 'a1'
    if (result.get('profile') != PROFILE or result.get('status') != 'complete'
            or result.get('method') != method or result.get('binding_sha256') != binding(plan, method)
            or result.get('test_ids_sha256') != plan['test_ids_sha256']
            or result.get('source_checkpoint_sha256') != plan['checkpoint_sources'][role]['checkpoint_sha256']
            or set(result.get('artifacts', {})) != expected_files):
        raise ValueError('Saved MetaFi defense record/source/artifact set differs')
    for name, checksum in result['artifacts'].items():
        if sha(folder / name) != checksum:
            raise ValueError('Saved MetaFi defense artifact differs: ' + name)
    if method in METHODS and result.get('repair', {}).get('complete') is not True:
        raise ValueError('Repair stage incomplete')
    if method == 'nad' and any(result['repair'].get('stage_results', {}).get(s, {}).get('selected_epoch') != 9
                               for s in ('teacher', 'student')):
        raise ValueError('NAD teacher/student final epochs differ')
    modules['nad'].replay_metrics(folder, result, 'piw3d')
    replay_poses(folder)
    return result


def replay_poses(folder):
    import numpy as np
    from piw_v2_adapter import target_pose
    with np.load(folder / 'per_sample.npz', allow_pickle=False) as a:
        truth, pred = a['truth'], a['predictions']
        if not np.isfinite(truth).all() or not np.isfinite(pred).all():
            raise ValueError('Nonfinite saved PiW poses')
        target = np.stack([target_pose(truth, d) for d in DOSES], axis=1)
        if ('targets' not in a or a['targets'].shape != target.shape
                or not np.allclose(a['targets'], target, rtol=0, atol=1e-12)):
            raise ValueError('Saved target poses differ from the native payload')
        joints = [11, 13]
        delta = (pred[:, :, joints] - pred[:, 0, joints][:, None]) - (target[:, :, joints] - truth[:, None, joints])
        expected = dict(clean_error_mm=np.linalg.norm(pred[:, 0]-truth, axis=-1)*1000,
            target_error_mm=np.linalg.norm(pred[:, :, joints]-target[:, :, joints], axis=-1)*1000,
            triggered_gt_error_mm=np.linalg.norm(pred[:, :, joints]-truth[:, None, joints], axis=-1)*1000,
            dce_error_mm=np.linalg.norm(delta, axis=-1)*1000,
            clean_scale_mm=np.linalg.norm(truth[:, 6]-truth[:, 4], axis=-1)*1000)
        for name, value in expected.items():
            if not np.allclose(a[name], value, rtol=0, atol=1e-8):
                raise ValueError('Saved errors do not replay from poses: ' + name)


def run_cell(args, modules, meta, plan, train, test, method):
    import numpy as np
    import torch
    from defend.a1_repair_ops import train_repair, _atomic_save, _state
    from defend.a1_bnp_pose import repair_bnp
    from defend.a1_fst_pose import repair_fst
    from vast_seed42_defense_adapter import prune_clp
    from piw_v2_adapter import build_trigger
    cached = validate_result(args, modules, plan, method)
    if cached is not None:
        print('CACHED_COMPLETE', method, flush=True)
        return cached
    folder = args.out / method
    if (folder / 'per_sample.npz').exists() or (folder / 'per_sample.npz.saving').exists():
        raise ValueError('Unreported evaluation exists; inspect before retry')
    if method in ('clp', 'bnp') and (folder / 'repair.pt').exists():
        raise ValueError('Unreported one-pass repair exists; inspect before retry')
    folder.mkdir(parents=True, exist_ok=True)
    modules['run_mmfi'].seed_execution()
    role = 'clean' if method == 'clean_reference' else 'a1'
    model = load_model(args, meta, role).to('cuda:0')
    fit = selection = None
    if method in ('ft', 'bnp', 'fst', 'nad'):
        write(folder / 'clean_bank.json', plan['clean_bank'])
        fit, selection = modules['run_piw'].clean_loaders(train, plan['clean_bank'])
    print(f'METAFI_BEGIN {method}: source {role} final199; no victim training', flush=True)
    bound = binding(plan, method)
    if method in ('clean_reference', 'none'):
        repair = dict(complete=True, source_checkpoint_reused=True, method=method)
    elif method in ('ft', 'nad'):
        model, repair = train_repair(model, fit, selection, method,
            copy.deepcopy(plan['recipes'][method]), 'cuda:0', folder / 'repair.pt', bound, 42)
    elif method == 'clp':
        model, repair = prune_clp(model, BACKBONE)
    elif method == 'bnp':
        model, repair = repair_bnp(model, fit, 'cuda:0', u=3.)
        write(folder / 'bnp_pruning.json', repair)
    elif method == 'fst':
        model, repair = repair_fst(model, BACKBONE, fit, selection, 'cuda:0', folder,
                                   bound, 42, **plan['recipes']['fst'])
    else:
        raise ValueError('NC and unknown defenses are excluded')
    if repair.get('complete') is not True:
        raise ValueError('Repair incomplete; TEST deferred')
    if method in ('clp', 'bnp'):
        _atomic_save(dict(binding_sha256=bound, complete=True, selected_model_state=_state(model),
                          metadata=repair), folder / 'repair.pt')
    measured = modules['run_piw'].evaluate(model, test, build_trigger(args.base / 'actions/data_bend.npy'))
    arrays = measured.pop('arrays')
    ids = [[str(x), str(y), int(i)] for x,y,i in arrays['pair_ids'].tolist()]
    if digest(ids) != plan['test_ids_sha256']:
        raise ValueError('Evaluated TEST order differs')
    with (folder / 'per_sample.npz.saving').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(folder / 'per_sample.npz.saving', folder / 'per_sample.npz')
    names = ['per_sample.npz']
    if method in METHODS:
        names.append('repair.pt')
    if method in ('ft', 'bnp', 'fst', 'nad'):
        names.append('clean_bank.json')
    if method == 'bnp':
        names.append('bnp_pruning.json')
    if method == 'fst':
        names.append('fst_progress.pt')
    measured.update(profile=PROFILE, method=method, status='complete', binding_sha256=bound,
        repair=repair, test_ids_sha256=plan['test_ids_sha256'],
        source_checkpoint_sha256=meta['checkpoints'][role]['checkpoint_sha256'],
        artifacts={n:sha(folder/n) for n in names})
    measured['record_sha256'] = digest(measured)
    write(folder / 'evaluation.json', measured)
    validate_result(args, modules, plan, method)
    print('METAFI_COMPLETE', method, measured['metrics'], flush=True)
    del model, arrays, fit, selection
    gc.collect(); torch.cuda.empty_cache()
    return measured


def collect(args, modules, plan):
    rows = []
    for method in ROWS:
        result = validate_result(args, modules, plan, method)
        row = dict(dataset='piw3d', backbone=BACKBONE, method=method,
                   status='pending' if result is None else result['status'])
        if result:
            row.update({k:result['metrics'][k] for k in METRICS})
        rows.append(row)
    folder = args.out / 'summary'
    folder.mkdir(exist_ok=True)
    # Only these derived summaries in THIS new profile are replaced.
    for name, value in (('defense_results.json', dict(profile=PROFILE, rows=rows,
                        neural_cleanse_included=False, complete=sum(r['status']=='complete' for r in rows), expected=7)),):
        tmp = folder / (name + '.saving')
        with tmp.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        os.replace(tmp, folder/name)
    tmp = folder / 'defense_results.csv.saving'
    with tmp.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'backbone', 'method', 'status', *METRICS])
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, folder / 'defense_results.csv')
    print('METAFI_REPORT', sum(r['status']=='complete' for r in rows), '/7', folder, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('/media/jackson/Data/wificsi/defense_s42_3080ti'))
    p.add_argument('--repo', type=Path, help='Existing frozen CEI checkout, NOT the new git checkout')
    p.add_argument('--clean-checkpoint', type=Path, default=Path('/home/wificsi/clean_metafi4090_s42.pt'))
    p.add_argument('--a1-checkpoint', type=Path, default=Path('/home/wificsi/a1_metafi4090_s42.pt'))
    p.add_argument('--mode', choices=('all', 'prepare', 'run', 'status', 'collect'), default='status')
    args = p.parse_args(argv)
    args.repo = args.repo or args.base / 'code/defense3080ti_v1/backdoor-analog'
    for name in ('base', 'repo', 'clean_checkpoint', 'a1_checkpoint'):
        setattr(args, name, getattr(args, name).resolve())
    args.out = args.base / 'runs' / FOLDER
    if HERE.is_relative_to(args.repo):
        raise ValueError('Run from a SEPARATE git checkout, not the frozen CEI source')
    if args.mode == 'status':
        for method in ROWS:
            path = args.out / method / 'evaluation.json'
            print(method, read(path).get('status') if path.exists() else 'pending')
        return 0
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL',
                      PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    meta = metadata()
    modules = boot(args.repo)
    with lock(args.out):
        if args.mode == 'collect':
            plan = read(args.out / 'plan.json')
            check_digest(plan, 'plan_sha256')
            if plan['profile'] != PROFILE or plan['rows'] != list(ROWS):
                raise ValueError('Wrong plan/scope')
            collect(args, modules, plan)
            return 0
        modules['controller'].initial_compute_check()
        if shutil.disk_usage(args.base).free < 8 * 2**30:
            raise ValueError('Require 8 GiB free; nothing deleted')
        plan, train, test = context(args, modules, meta)
        write(args.out / 'plan.json', plan)
        if args.mode in ('all', 'prepare'):
            prepare(args, modules, meta, plan, train)
        if args.mode == 'prepare':
            return 0
        prepared(args, plan)
        for method in ROWS:
            run_cell(args, modules, meta, plan, train, test, method)
            collect(args, modules, plan)
        print('METAFI_7_OF_7_COMPLETE; five defenses; NC excluded.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'METAFI_DEFENSE_FAILED: {error}; no automatic retry/batch fallback.', file=sys.stderr, flush=True)
        raise
