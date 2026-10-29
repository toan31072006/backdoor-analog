"""Serial RTX3090 continuation of the PiW3D MetaFi++ AdamW experiment.

Only host orchestration is new. Scientific files are extracted from a verified
Vast source snapshot; original checkpoints are never edited. Prepare and Launch
are separate. A changed host is an explicitly recorded continuation, not a
claim of a bitwise-identical Linux/Windows trajectory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import gzip
import hashlib
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import zlib

HERE = Path(__file__).resolve().parent
PROFILE = 'piw_metafi_remaining_s42_3090_v1'
METHODS = ('clean', 'a1', 'ftrojan', 'bpp_source', 'fiba', 'wanet_source')
NATIVE = ('ftrojan', 'bpp_source', 'fiba', 'wanet_source')
SOURCE_SHA = '1366c22ec19068b0eacee48bc5b33501adffd5b39f97a9b5ea66695bc0374154'
OLD_PLAN_SHA = '62f2e03d242519b0d40d60557c9232305c8d9e2a2236820d85544bf4da0048de'
ACTION_SHA = '2e8fc11192cc3ee7ee3f5544cb3c5e19f771f7f7265dde93af8969af41a2be3e'
FT_PATTERN_SHA = '812754c588d835a3e19f503488237e334dee0bd4c17c39404597504a78b25e45'
BACKUPS = {
    'ftrojan': (84, '9d2a713315f694183cd9512e9feb74fe4f44bb823fedf455fafbf5edddd7d093'),
    'bpp_source': (72, '4ec9d9ff57fa578a42f368764b9c110d2376e155d6d578d5831676a5fd2a4c5f'),
    'fiba': (33, '590394ff220b1b99dcd32db6416948e40fa6b49fb4f8934f7dfd9f3bbda98389'),
}
CLEAN_PROFILE_SHA = '1d1ef2c08349b76cdb128296e8ae823bb56e6293790d5d2cf6f2c09f877a5262'
PATH_KEYS = {'dataset_root', 'outdir', 'action_npy'}
RECIPE = dict(optimizer='adamw', lr=.01, weight_decay=.01,
              accumulation=1, effective_batch=32)
DESKTOP = {'explorer.exe', 'dwm.exe', 'searchhost.exe', 'startmenuexperiencehost.exe',
           'shellexperiencehost.exe', 'shellhost.exe', 'applicationframehost.exe',
           'textinputhost.exe', 'lockapp.exe', 'logonui.exe', 'systemsettings.exe',
           'code.exe', 'chrome.exe', 'msedge.exe', 'msedgewebview2.exe',
           'windowsterminal.exe', 'taskmgr.exe', 'snippingtool.exe', 'teamviewer.exe',
           'ms-teams.exe', 'phoneexperiencehost.exe', 'crossdeviceresume.exe',
           'copilot.exe', 'copilotapp.exe', 'sihost.exe'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.saving')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2,
        allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def original_plan():
    with gzip.open(HERE / 'source_plan.json.gz', 'rt', encoding='utf-8') as stream:
        result = json.load(stream)
    if (result['plan_sha256'] != OLD_PLAN_SHA or
            digest({k:v for k,v in result.items() if k != 'plan_sha256'}) != OLD_PLAN_SHA):
        raise ValueError('Original source plan changed')
    return result


def code_binding():
    names = ('controller.py', 'source_snapshot.tar.gz', 'source_plan.json.gz',
             'clean_profile.py', 'ftrojan_pattern_linux.f8.zlib')
    result = {name: sha(HERE/name) for name in names}
    if result['clean_profile.py'] != CLEAN_PROFILE_SHA:
        raise ValueError('Original Clean/A1 entry point changed')
    return result


def bootstrap(out):
    """Extract verified files without accepting links or replacing any file."""
    archive_path = HERE/'source_snapshot.tar.gz'
    if sha(archive_path) != SOURCE_SHA:
        raise ValueError('Recovered scientific source archive changed')
    root = out/'source_snapshot'
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            rel = PurePosixPath(member.name)
            if (rel.is_absolute() or '..' in rel.parts or not rel.parts or
                    rel.parts[0] != 'backdoor-analog' or member.issym() or member.islnk()):
                raise ValueError('Unsafe scientific source archive member')
            if member.isdir():
                continue
            if not member.isfile() or member.size > 20*2**20:
                raise ValueError('Unexpected source archive member')
            target = root.joinpath(*rel.parts[1:])
            if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
                raise ValueError('Source destination escapes snapshot')
            content = archive.extractfile(member).read()
            if target.exists():
                if not target.is_file() or target.read_bytes() != content:
                    raise ValueError(f'Existing scientific source changed: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(content)
    entry = root/'ATKBackd/run_piw_metafi_dtpose_clean_a1.py'
    content = (HERE/'clean_profile.py').read_bytes()
    if hashlib.sha256(content).hexdigest() != CLEAN_PROFILE_SHA:
        raise ValueError('Clean/A1 entry point differs from original bundle')
    if entry.exists():
        if entry.read_bytes() != content:
            raise ValueError('Existing Clean/A1 scientific profile changed')
    else:
        with entry.open('xb') as stream:
            stream.write(content)
    for name, expected in original_plan()['sources'].items():
        if sha(root/'ATKBackd'/name) != expected:
            raise ValueError(f'Scientific file differs from original plan: {name}')
    return root


def conflict_rows(text, windows=False, own_pid=None):
    result = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        if len(row) != 2 or not row[0].strip().isdigit():
            raise ValueError(f'Unidentified GPU process: {row}')
        pid, name = int(row[0]), row[1].strip()
        if pid == own_pid:
            continue
        exe = ntpath.basename(name.replace('/', '\\')).lower()
        if windows and exe in DESKTOP:
            continue
        result.append((pid, name))
    return result


def require_idle():
    report = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=30)
    conflicts = conflict_rows(report.stdout, os.name == 'nt', os.getpid())
    if conflicts:
        raise ValueError(f'Other GPU work active; no process stopped: {conflicts}')


@contextmanager
def lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'controller.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class CanonicalTrain:
    """Keep original donor identities; map reads to fully audited local files.

    Only the trigger's TRAIN donor pool uses these paths. The regular training
    dataset, poison recipients, labels and file-content audit remain unchanged.
    """
    def __init__(self, actual, source_root, names):
        if actual.split != 'training' or [r['name'] for r in actual.items] != names:
            raise ValueError('Canonical donor pool requires the exact ordered TRAIN identities')
        self.actual, self.split, self.items, self.lookup = actual, 'training', [], {}
        for row in actual.items:
            item = dict(row)
            for field, folder in (('csi', 'csi_ap'), ('kpt', 'keypoint')):
                path = str(PurePosixPath(source_root)/'train_data'/folder/(row['name']+'.npy'))
                self.lookup[path] = row[field]
                item[field] = path
            self.items.append(item)

    def __len__(self):
        return len(self.actual)

    def load_raw(self, path):
        if str(path) not in self.lookup:
            raise ValueError('Donor path is not in audited TRAIN')
        return self.actual.load_raw(self.lookup[str(path)])

    def normalize(self, value):
        return self.actual.normalize(value)


def frozen_pattern():
    import numpy as np
    path = HERE/'ftrojan_pattern_linux.f8.zlib'
    if not path.is_file():
        raise ValueError('Missing exact Linux FTrojan pattern; refusing a different trigger')
    raw = zlib.decompress(path.read_bytes())
    if len(raw) != 14400 or hashlib.sha256(raw).hexdigest() != FT_PATTERN_SHA:
        raise ValueError('Linux FTrojan pattern checksum differs from saved victim key')
    result = np.frombuffer(raw, dtype='<f8').reshape(3, 30, 20).copy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite frozen FTrojan pattern')
    return result


def load_worker(args, method):
    root = bootstrap(args.out)
    sys.path.insert(0, str(root/'ATKBackd'))
    import importlib
    name = ('run_piw_metafi_dtpose_clean_a1' if method in ('clean','a1')
            else 'run_piw_metafi_dtpose_baseline_six')
    profile = importlib.import_module(name)
    w = profile.worker
    original_make = w.make_trigger
    old = original_plan()

    def make_trigger(method_name, train, cfg):
        donor = (CanonicalTrain(train, old['cfg']['dataset_root'],
                    old['inputs']['splits']['train_data']['ordered_names'])
                 if method_name in ('fiba','bpp_source') else train)
        trigger = original_make(method_name, donor, cfg)
        if method_name == 'ftrojan':
            pattern = frozen_pattern()
            for native in trigger.native:
                native.pattern = pattern.copy()
            if trigger.fixed_keys() != [FT_PATTERN_SHA]*3:
                raise ValueError('Restoring original FTrojan key failed')
        return trigger

    w.make_trigger = make_trigger
    return w


def run_out(args, method):
    return args.out/('clean_a1' if method in ('clean','a1') else 'baseline_six')


def gate_path(args, method):
    return args.out/'gates'/(method+'.json')


def cfg_for(w, args, method):
    cfg = w.cfg_for(args.dataset, run_out(args, method), args.action)
    if (cfg['recipes'] != {'metafiplusplus': RECIPE} or cfg['epochs'] != 200 or
            cfg['final_epoch'] != 199 or cfg['batch_size'] != 32 or cfg['amp'] or
            cfg['num_workers'] != 0 or not cfg['drop_last'] or cfg['seed'] != 42):
        raise ValueError('Expected fixed original AdamW/physical-B32/final199 recipe')
    return cfg


def validate_plans(source, target, method):
    if source['plan_sha256'] != digest({k:v for k,v in source.items() if k != 'plan_sha256'}):
        raise ValueError('Source plan digest mismatch')
    if ({k:v for k,v in source['cfg'].items() if k not in PATH_KEYS} !=
            {k:v for k,v in target['cfg'].items() if k not in PATH_KEYS}):
        raise ValueError('Scientific configuration differs; no fallback to scratch')
    for key in ('profile', 'sources', 'poison', 'action_sha256'):
        if source[key] != target[key]:
            raise ValueError(f'Source plan differs in {key}')
    if source['inputs']['splits'] != target['inputs']['splits']:
        raise ValueError('Full dataset/list/content/order differs from source checkpoint')
    before, after = copy.deepcopy(source['trigger']), copy.deepcopy(target['trigger'])
    differences = []
    if method in NATIVE:
        # The native operators do not use these A1 arrays. Every non-array
        # field, including A1 action/protocol, still must compare exactly.
        if set(before['arrays']) != set(after['arrays']):
            raise ValueError('Unexpected unused-A1 array schema')
        for name in before['arrays']:
            left, right = before['arrays'][name], after['arrays'][name]
            if {k:v for k,v in left.items() if k != 'sha256'} != {
                    k:v for k,v in right.items() if k != 'sha256'}:
                raise ValueError('Unused-A1 array shape/dtype changed')
            if left['sha256'] != right['sha256']:
                differences.append(dict(array=name, previous=left['sha256'], current=right['sha256']))
            right['sha256'] = left['sha256']
    if before != after:
        raise ValueError('Actual A1 trigger/protocol changed; refusing checkpoint')
    return differences


def validate_blob(blob, source, method, poison=None):
    if (blob.get('profile') != source['profile'] or blob.get('backbone') != 'metafiplusplus'
            or blob.get('method') != method or blob.get('plan_sha256') != source['plan_sha256']
            or type(blob.get('epoch')) is not int or not 0 <= blob['epoch'] <= 199):
        raise ValueError('Foreign or invalid imported checkpoint')
    for name in ('model', 'optimizer', 'rng', 'execution', 'poison'):
        if not isinstance(blob.get(name), dict) or not blob[name]:
            raise ValueError(f'Checkpoint lacks complete continuation state: {name}')
    groups = blob['optimizer'].get('param_groups', [])
    if not groups or not blob['optimizer'].get('state'):
        raise ValueError('Optimizer state missing')
    for group in groups:
        if (group.get('lr') != .01 or group.get('weight_decay') != .01 or
                tuple(group.get('betas', ())) != (.9,.999) or group.get('eps') != 1e-8 or
                group.get('amsgrad') is not False or group.get('maximize',False) is not False):
            raise ValueError('Only the original AdamW optimizer is allowed; SGD is excluded')
    for state in blob['optimizer']['state'].values():
        if not isinstance(state,dict) or not all(k in state for k in ('step','exp_avg','exp_avg_sq')):
            raise ValueError('Complete AdamW moment/step state required')
    if any(k not in blob['rng'] for k in ('python','numpy','torch','cuda')) or blob['rng']['cuda'] is None:
        raise ValueError('Checkpoint lacks full CPU/CUDA RNG')
    if poison is not None and blob['poison'] != poison:
        raise ValueError('Actual native operator/poison/cover identities changed')
    if method in NATIVE and (not isinstance(blob.get('trigger'), list) or len(blob['trigger']) != 3):
        raise ValueError('Original three-group trigger continuation state missing')


def candidate_paths(args, method):
    candidates = set()
    expected = args.checkpoint_home/'metafiplusplus/seed42'/method/'checkpoint.pt'
    if expected.is_file():
        candidates.add(expected.resolve())
    roots = (args.data_home/'runs', args.checkpoint_home)
    for root in roots:
        if root.is_dir():
            for p in root.glob(f'**/attacks/metafiplusplus/seed42/{method}/checkpoint.pt'):
                if not p.resolve().is_relative_to(args.out) and not p.is_symlink():
                    candidates.add(p.resolve())
    # Previous host ZIPs may already have the identical backup on this machine.
    # Reuse those immutable input files instead of transferring them again.
    if method in BACKUPS:
        code = args.data_home/'code'
        if code.is_dir():
            filename = f'{method}_epoch{BACKUPS[method][0]}.pt'
            for p in code.glob('**/recovery_inputs/'+filename):
                if not p.is_symlink() and not p.resolve().is_relative_to(args.out):
                    candidates.add(p.resolve())
    return sorted(candidates, key=str)


def candidate_plan(path, method):
    for parent in path.parents[:7]:
        p = parent/'plan.json'
        if p.is_file():
            return read(p)
    if method in BACKUPS and sha(path) == BACKUPS[method][1]:
        return original_plan()
    raise ValueError(f'Missing source plan for imported checkpoint: {path}')


def select_candidate(args, method, target, poison):
    import torch
    accepted, skipped = [], []
    for path in candidate_paths(args, method):
        try:
            source = candidate_plan(path, method)
        except ValueError:
            probe = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            groups = probe.get('optimizer',{}).get('param_groups',[])
            if groups and all('momentum' in g and 'betas' not in g for g in groups):
                skipped.append(dict(path=str(path),reason='SGD checkpoint without plan; never imported'))
                del probe
                continue
            raise
        if source.get('profile') != target['profile']:
            skipped.append(dict(path=str(path),reason='other experiment profile; never imported'))
            continue
        recipe = source.get('cfg',{}).get('recipes',{}).get('metafiplusplus',{})
        if recipe.get('optimizer') != 'adamw':
            skipped.append(dict(path=str(path), reason='not AdamW; never imported'))
            continue
        differences = validate_plans(source, target, method)
        blob = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        validate_blob(blob, source, method, poison)
        accepted.append((blob['epoch'], str(path), source, differences))
        del blob
    if accepted:
        # Select saved progress, never a TEST score. A tie has a stable path order.
        accepted.sort(key=lambda x:(-x[0], x[1]))
        epoch, filename, source, differences = accepted[0]
        return dict(path=filename, epoch=epoch, sha256=sha(filename),
                    source_plan=source, unused_a1_metadata_differences=differences,
                    ignored_non_adamw=skipped)
    if method in BACKUPS:
        raise ValueError(f'{method}: missing resumable checkpoint; copy its original backup, not restart epoch0')
    return dict(path=None, epoch=-1, ignored_non_adamw=skipped)


def sample_batch(w, saved, method, data):
    from torch.utils.data import default_collate
    indices = []
    selected = {int(r['index']) for r in saved['poison']['samples']}
    if method != 'clean':
        indices.extend(sorted(selected)[:4])
        if getattr(data,'covers',None):
            indices.append(min(data.covers))
    for i in range(len(data)):
        if len(indices) == 32:
            break
        if i not in indices and (method == 'clean' or i not in selected):
            indices.append(i)
    if len(indices) != 32:
        raise ValueError('A physical B32 with actual poison/cover samples is required')
    return default_collate([data[i] for i in indices])


def prepare_cell(args, w, cfg, saved, execution, stats):
    import torch
    method = args.method
    path = w.cell_dir(cfg,'metafiplusplus',method)/'checkpoint.pt'
    if path.exists():
        raise ValueError('Ungated target checkpoint exists; inspect it, never overwrite')
    datasets = w._datasets(saved)
    trigger = w.make_trigger(method,datasets['training'],cfg)
    data, poison = w.make_training_data(saved,method,datasets,trigger)
    selection = select_candidate(args,method,saved,poison)
    w._seed('cuda:0')
    model = w._build('metafiplusplus','cuda:0').train()
    optimizer = w._optimizer(model,'metafiplusplus')
    if selection['path']:
        old = torch.load(selection['path'],map_location='cpu',weights_only=False)
        validate_blob(old,selection['source_plan'],method,poison)
        model.load_state_dict(old['model'],strict=True)
        optimizer.load_state_dict(old['optimizer'])
        if method in NATIVE:
            trigger.load_state_dict(old['trigger'])
        w.base._restore_rng(old['rng'],'cuda:0')
    batch = sample_batch(w,saved,method,data)
    # Smoke mutates model/optimizer/RNG in RAM, never the source checkpoint.
    optimizer.zero_grad(set_to_none=True)
    loss,n,_ = w._train_batch(model,optimizer,batch,'cuda:0','metafiplusplus',
                             last_micro=True,group_size=32)
    if n != 32 or not __import__('math').isfinite(loss):
        raise ValueError('Real physical B32 continuation smoke failed')
    del optimizer,model,batch,trigger,data,datasets
    torch.cuda.empty_cache()
    if selection['path']:
        # Reload immutable source after smoke; optimizer state tensors can alias.
        old = torch.load(selection['path'],map_location='cpu',weights_only=False)
        validate_blob(old,selection['source_plan'],method,poison)
        migrated = copy.copy(old)
        migrated.update(plan_sha256=saved['plan_sha256'],execution=execution)
        w.base.atomic_checkpoint(path,migrated)
        if sha(selection['path']) != selection['sha256']:
            raise ValueError('Original checkpoint changed during import')
        write(path.parent/'migration_receipt.json',dict(
            schema=1,source_path=selection['path'],source_sha256=selection['sha256'],
            source_plan_sha256=selection['source_plan']['plan_sha256'],
            target_plan_sha256=saved['plan_sha256'],previous_execution=old['execution'],
            target_execution=execution,previous_runtime=selection['source_plan']['runtime'],
            target_runtime=saved['runtime'],last_source_epoch=old['epoch'],next_epoch=old['epoch']+1,
            actual_native_poison_cover_operator_keys_exactly_matched=True,
            unused_a1_metadata_differences=selection.get('unused_a1_metadata_differences',[]),
            canonical_train_donor_paths=(method in ('bpp_source','fiba')),
            exact_linux_ft_pattern=(FT_PATTERN_SHA if method == 'ftrojan' else None),
            model_optimizer_rng_trigger_unchanged=True,source_preserved=True,
            no_bitwise_cross_platform_trajectory_guarantee=True))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
    selection.pop('source_plan',None)
    write(gate_path(args,method),dict(profile=PROFILE,method=method,
        plan_sha256=saved['plan_sha256'],execution=execution,input_stat_fingerprint=stats,
        code=code_binding(),checkpoint_selection=selection,physical_batch=32,
        smoke_loss=loss,prepared_only=True))
    print(f'PREPARED_ONLY {method}; source_epoch={selection["epoch"]}; no full training',flush=True)


def check_gate(args,w,cfg,saved,execution,stats):
    gate = read(gate_path(args,args.method))
    expected = dict(profile=PROFILE,method=args.method,plan_sha256=saved['plan_sha256'],
                    execution=execution,input_stat_fingerprint=stats,code=code_binding(),
                    physical_batch=32,prepared_only=True)
    if any(gate.get(k) != v for k,v in expected.items()):
        raise ValueError('Source/data/runtime/GPU/prelaunch gate changed')
    path = w.cell_dir(cfg,'metafiplusplus',args.method)/'checkpoint.pt'
    if path.exists():
        blob = w._checkpoint(saved,'metafiplusplus',args.method)
        datasets = w._datasets(saved)
        trigger = w.make_trigger(args.method,datasets['training'],cfg)
        _,poison = w.make_training_data(saved,args.method,datasets,trigger)
        validate_blob(blob,saved,args.method,poison)
        if blob['execution'] != execution:
            raise ValueError('Checkpoint GPU/backend changed; do not bypass migration')
        if args.method in NATIVE:
            trigger.load_state_dict(blob['trigger'])
    elif gate['checkpoint_selection']['path']:
        raise ValueError('Migrated checkpoint was removed; refusing an epoch0 restart')
    return gate


def evaluate_cell(args,w,cfg,saved):
    """Evaluate fixed-final199 cells only after all six scheduled victims finish."""
    import numpy as np
    from piw_v2_adapter import measure_predictions, target_pose
    own = w._checkpoint(saved,'metafiplusplus',args.method,final=True)
    folder = w.cell_dir(cfg,'metafiplusplus',args.method)
    clean_root = args.out/'clean_a1'
    clean_path = clean_root/'attacks/metafiplusplus/seed42/clean/checkpoint.pt'
    import torch
    clean_blob = torch.load(clean_path,map_location='cpu',weights_only=False)
    clean_plan = read(clean_root/'plan.json')
    validate_blob(clean_blob,clean_plan,'clean')
    if clean_blob['epoch'] != 199:
        raise ValueError('Clean reference is not final199')
    if (clean_plan['inputs']['splits'] != saved['inputs']['splits'] or
            clean_plan['cfg']['recipes'] != saved['cfg']['recipes'] or
            any(clean_plan['cfg'][k] != saved['cfg'][k] for k in
                ('dataset_root','action_npy','seed','target','batch_size','amp','rho','eval_doses'))):
        raise ValueError('Clean reference does not match this evaluation')
    report_path = folder/'evaluation.json'
    if report_path.exists():
        previous = read(report_path)
        if (previous.get('profile') != PROFILE or previous.get('status') != 'COMPLETE' or
                previous.get('checkpoint_sha256') != sha(folder/'checkpoint.pt') or
                previous.get('clean_checkpoint_sha256') != sha(clean_path) or
                previous.get('plan_sha256') != saved['plan_sha256'] or
                previous.get('code') != code_binding() or
                previous.get('predictions_sha256') != sha(folder/'predictions.npz')):
            raise ValueError('Existing evaluation is changed/foreign; refusing overwrite')
        print('EVALUATION_ALREADY_COMPLETE',args.method,flush=True)
        return
    ds = w._datasets(saved)
    x,truth,names = w._test_arrays(ds['validation'])
    if names != saved['inputs']['splits']['test_data']['ordered_names']:
        raise ValueError('TEST ordering changed')
    victim = w._build('metafiplusplus','cuda:0')
    victim.load_state_dict(own['model'],strict=True)
    reference = w._build('metafiplusplus','cuda:0')
    reference.load_state_dict(clean_blob['model'],strict=True)
    trigger = w.make_trigger(args.method,ds['training'],cfg)
    if args.method in NATIVE:
        trigger.load_state_dict(own['trigger'])
    grid = w._predict_grid(victim,x,trigger,w.DOSES,'cuda:0')
    control = w._predict_grid(reference,x,trigger,w.DOSES,'cuda:0')
    metrics = measure_predictions(truth,grid[:,0],grid,control[:,0],control)
    targets = np.stack([target_pose(truth,d) for d in w.DOSES],axis=1)
    w._atomic_npz(folder/'predictions.npz',truth=truth,names=np.asarray(names),
        doses=np.asarray(w.DOSES),predictions=grid[:,0],trigger_predictions=grid,
        reference_predictions=control[:,0],reference_trigger_predictions=control,targets=targets)
    write(report_path,dict(profile=PROFILE,status='COMPLETE',method=args.method,
        backbone='metafiplusplus',seed=42,plan_sha256=saved['plan_sha256'],
        checkpoint_sha256=sha(folder/'checkpoint.pt'),clean_checkpoint_sha256=sha(clean_path),
        predictions_sha256=sha(folder/'predictions.npz'),code=code_binding(),metrics=metrics,
        limitations=['One seed; no across-seed SD.',
            'Explicit Linux/Windows and GPU continuation; not bitwise trajectory equivalence.',
            'TSBA/Smooth excluded by request; this is a selected-method extension, not full 8-row table.']))
    print('EVAL_COMPLETE',args.method,metrics['mainmetrics'],flush=True)


def cell(args):
    import torch
    w = load_worker(args,args.method)
    cfg = cfg_for(w,args,args.method)
    if args.work == 'prepare':
        w.preflight(cfg)
    saved = w.load_plan(cfg)
    stats = w.input_stat_fingerprint(args.dataset,saved['inputs'])
    if stats != read(w.receipt_path(cfg))['input_stat_fingerprint']:
        raise ValueError('Dataset changed since full content audit')
    execution = w._execution('cuda:0')
    if '3090' not in execution.get('gpu_name',''):
        raise ValueError('This profile requires RTX3090; no other-GPU/CPU fallback')
    if args.work == 'prepare':
        prepare_cell(args,w,cfg,saved,execution,stats)
        return
    check_gate(args,w,cfg,saved,execution,stats)
    if args.work == 'train':
        existing_path = w.cell_dir(cfg,'metafiplusplus',args.method)/'checkpoint.pt'
        final_already = (existing_path.is_file() and
            w._checkpoint(saved,'metafiplusplus',args.method)['epoch'] == 199)
        if final_already:
            print('SKIP_TRAIN_ALREADY_FINAL199',args.method,flush=True)
        else:
            w.train_cell(saved,'metafiplusplus',args.method,'cuda:0')
        blob = w._checkpoint(saved,'metafiplusplus',args.method,final=True)
        path = w.cell_dir(cfg,'metafiplusplus',args.method)/'checkpoint.pt'
        write(path.parent/'trained_final199.json',dict(profile=PROFILE,
            status='TRAINED_NOT_EVALUATED',epoch=blob['epoch'],checkpoint_sha256=sha(path),
            plan_sha256=saved['plan_sha256']))
    elif args.work == 'evaluate':
        evaluate_cell(args,w,cfg,saved)
    else:
        print('CHECK_OK',args.method,flush=True)


def command(args,method,work):
    return [sys.executable,'-u',str(HERE/'controller.py'),'--mode','cell',
        '--method',method,'--work',work,'--data-home',str(args.data_home),
        '--dataset',str(args.dataset),'--action',str(args.action),'--out',str(args.out),
        '--checkpoint-home',str(args.checkpoint_home)]


def inspect(args):
    print('SELECTED_SCOPE: MetaFi++ PiW3D AdamW; serial 1 job/cuda:0; TSBA/Smooth excluded')
    print('No training launched. Existing AdamW checkpoints are audited during Prepare.')
    for method in METHODS:
        paths = candidate_paths(args,method)
        print(method, 'candidate_checkpoints=',[str(p) for p in paths])
        target = run_out(args,method)/'attacks/metafiplusplus/seed42'/method
        for name in ('checkpoint.pt','trained_final199.json','evaluation.json'):
            p = target/name
            print(' ',name,'bytes='+str(p.stat().st_size) if p.is_file() else 'absent')
    if args.out.is_dir():
        print('Free_GiB=',round(shutil.disk_usage(args.out).free/2**30,2))
        for log in sorted((args.out/'logs').glob('*.log'))[-8:]:
            lines = log.read_text(encoding='utf-8',errors='replace').splitlines()
            print(log.name,'\n'+'\n'.join(lines[-3:]))


def summarize(args):
    rows = []
    for method in METHODS:
        path = run_out(args,method)/'attacks/metafiplusplus/seed42'/method/'evaluation.json'
        result = read(path)
        if result['status'] != 'COMPLETE':
            raise ValueError('Evaluation incomplete')
        rows.append(dict(method=method,**result['metrics']['mainmetrics']))
    write(args.out/'results.json',dict(profile=PROFILE,status='COMPLETE',rows=rows))
    fields = sorted(set().union(*(r.keys() for r in rows)) - {'method'})
    lines = ['| Method | '+' | '.join(fields)+' |', '| --- | '+' | '.join(['---:']*len(fields))+' |']
    for row in rows:
        values = [('—' if row['method'] == 'clean' and
                   not k.startswith(('clean_','reference_clean_')) else str(row.get(k,'--')))
                  for k in fields]
        lines.append('| '+row['method']+' | '+' | '.join(values)+' |')
    path = args.out/'comparison.md'
    content = '\n'.join(lines)+'\n'
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise ValueError('Existing comparison differs; refusing overwrite')
    if not path.exists():
        path.write_text(content,encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('inspect','prepare','check','launch','status','cell'))
    for name in ('data-home','dataset','action','out','checkpoint-home'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=METHODS)
    p.add_argument('--work',choices=('prepare','check','train','evaluate'))
    args = p.parse_args(argv)
    for key in ('data_home','dataset','action','out','checkpoint_home'):
        setattr(args,key,getattr(args,key).resolve())
    if args.out == args.dataset or args.out in args.dataset.parents or args.dataset in args.out.parents:
        raise ValueError('Output and dataset overlap')
    os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2',MKL_THREADING_LAYER='SEQUENTIAL')
    configure_torch_cache(args)
    if args.mode in ('inspect','status'):
        inspect(args); return 0
    if sha(args.action) != ACTION_SHA:
        raise ValueError('Authentic data_bend.npy checksum required')
    code_binding()
    if args.mode == 'cell':
        if not args.method or not args.work:
            p.error('Internal cell requires method and work')
        cell(args); return 0
    with lock(args.out):
        require_idle()
        if shutil.disk_usage(args.out).free < 4*2**30:
            raise ValueError('At least 4 GiB free required for six victims/atomic saves')
        logs = args.out/'logs'; logs.mkdir(exist_ok=True)
        if args.mode in ('check','launch'):
            for method in METHODS:
                subprocess.run(command(args,method,'check'),check=True)
        if args.mode == 'check':
            print('VERIFIED_PRELAUNCH_GATE; no training launched',flush=True); return 0
        phases = ('prepare',) if args.mode == 'prepare' else ('train','evaluate')
        for phase in phases:
            for method in METHODS:
                require_idle()
                if phase == 'prepare' and gate_path(args,method).exists():
                    subprocess.run(command(args,method,'check'),check=True)
                    continue
                log = logs/f'{phase}.{method}.{time.time_ns()}.log'
                print(f'[{phase}] {method}: {log}',flush=True)
                with log.open('x',encoding='utf-8') as sink:
                    result = subprocess.run(command(args,method,phase),stdout=sink,stderr=subprocess.STDOUT)
                if result.returncode:
                    tail = '\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-12:])
                    raise RuntimeError(f'{method}/{phase} failed; no retry/later jobs: {log}\n{tail}')
        if args.mode == 'launch':
            summarize(args)
        else:
            write(args.out/'execution_scope.json',dict(profile=PROFILE,
                selected_methods=list(METHODS),excluded_methods=['tsba_adapted','smooth_lf_csi'],
                backbone='metafiplusplus',dataset='person-in-wifi-3d',seed=42,
                gpu='cuda:0',jobs_per_gpu=1,recipe=RECIPE,physical_batch=32,final_epoch=199,
                checkpoint_selections={m:read(gate_path(args,m))['checkpoint_selection'] for m in METHODS},
                code=code_binding(),
                note='Legacy scientific plans retain their original method lists for provenance; '
                     'only these six rows are scheduled. Not the complete eight-row paper table.'))
        print('PREPARED_ONLY; no full training' if args.mode == 'prepare' else 'ALL_SIX_COMPLETE',flush=True)
    return 0


def configure_torch_cache(args):
    """Reuse official weights already present; no package/environment changes."""
    if os.environ.get('TORCH_HOME'):
        return
    name = 'hub/checkpoints/resnet34-b627a593.pth'
    candidates = [args.data_home/'torch-cache',Path.home()/'.cache/torch']
    code = args.data_home/'code'
    if code.is_dir():
        candidates.extend(p.parents[2] for p in code.glob('**/torch-cache/'+name))
    for candidate in candidates:
        weights = candidate/name
        if weights.is_file() and sha(weights).startswith('b627a593'):
            os.environ['TORCH_HOME'] = str(candidate)
            return
    # Torchvision may fetch its official public ImageNet initialization during
    # Prepare. This is not a substitute for any trained victim checkpoint.
    os.environ['TORCH_HOME'] = str(candidates[0])


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as error:
        print(f'[PiW MetaFi remaining 3090] ERROR: {error}',file=sys.stderr,flush=True)
        raise SystemExit(1)
