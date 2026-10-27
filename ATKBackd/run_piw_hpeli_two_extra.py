"""Two PiW3D HPE-Li attack adaptations against the frozen v3 Clean reference.

Only Smooth-LF-CSI and TSBA-adapted victims are trained here. The completed
Windows v3 Clean/A1 experiment is imported as a verified portable archive;
neither its weights nor its labels are used to initialize a new victim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

import run_piw_hpeli_clean as base
import run_piw_hpeli_lr_control as v3
import run_piw_two_models_attacks as two
from piw_v2_adapter import (PoisonedPairs as A1Pairs, build_trigger as build_a1,
                            measure_predictions, target_contract, target_pose,
                            trigger_record)


HERE = Path(__file__).resolve().parent
PROFILE = 'piw_hpeli_two_extra_s42_portable_v1'
METHODS = ('smooth_lf_csi', 'tsba_adapted')
DEVICES = {'smooth_lf_csi': 'cuda:1', 'tsba_adapted': 'cuda:0'}
DOSES = (0., .2, .4, .6, .8, 1.)
SOURCE_NAMES = tuple(dict.fromkeys((
    'run_piw_hpeli_two_extra.py', 'run_piw_hpeli_clean.py',
    'run_piw_hpeli_lr_control.py', 'run_piw_two_models_attacks.py',
    'piw_extra_attack_adapter.py', 'piw_native_attack_adapter.py',
    'piw_v2_adapter.py', 'piw_a1_adapter.py',
    'data_utils/feeder.py', 'models/hpeli.py', 'models/sk_network.py',
    'attack/tsba.py', 'attack/smooth_lf_csi.py', 'attack/method_drafts.py',
    'attack/trigger.py', 'eval/metrics.py',
)))
REFERENCE_FILES = {
    'piw_lr_control.resolved.json', 'evaluation.json', 'predictions.npz',
    'poison.json', 'hpeli/seed42/clean/checkpoint.pt',
    'hpeli/seed42/a1/checkpoint.pt',
}
RECIPE_KEYS = ('optimizer', 'lr', 'weight_decay', 'betas', 'eps', 'batch_size',
               'drop_last', 'amp', 'victim_loss')


def _sha(path):
    return base.file_sha(Path(path))


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _paths(args):
    root = Path(args.dataset_root).resolve()
    output = Path(args.outdir).resolve()
    archive = Path(args.reference_archive).resolve()
    action = Path(args.action_npy).resolve()
    home = Path(args.data_home).resolve() if args.data_home else root.parent.parent
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('Dataset and output must be separate')
    if archive == output or output in archive.parents or action == output or output in action.parents:
        raise ValueError('Reference and action must remain outside output')
    return home, root, output, archive, action


def _config(args):
    home, root, output, _archive, action = _paths(args)
    cfg = v3.configuration(home, output, root, action, device='cpu', jobs_per_gpu=1)
    cfg.pop('device')
    cfg.update(profile=PROFILE, roles=list(METHODS), jobs_per_gpu=1,
        initialization='independent scratch HPE-Li victims; no v3 warm start',
        selection='fixed final199; TEST only after own final checkpoint',
        status='EXPLORATORY_PIW_HPELI_TWO_ADAPTED_ATTACKS_ONE_SEED',
        trigger_policy='Smooth-LF fixed seeded low-frequency CSI control eps .185; '
            'TSBA independent three-group learned multiplicative generator eps .1; '
            'normalized amplitude groups only, phase unchanged',
        tsba=dict(eps=.1, hidden=32, generator_lr=.001,
                  warmup_epochs=10, generator_steps=4),
        smooth_lf=dict(eps=.185, seed=42, low_subcarriers=3,
                       low_packets=2),
        devices=copy.deepcopy(DEVICES))
    return cfg


def _plan(args):
    cfg = _config(args)
    _home, _root, _output, archive, action = _paths(args)
    if not archive.is_file() or not action.is_file():
        raise FileNotFoundError('Portable HPE-Li v3 reference and action NPY are required')
    result = dict(schema=1, profile=PROFILE, cfg=cfg,
        reference_archive=str(archive), reference_archive_sha256=_sha(archive),
        action_sha256=_sha(action), methods=list(METHODS), devices=DEVICES,
        sources={name: _sha(HERE / name) for name in SOURCE_NAMES},
        runtime=base.runtime_versions(),
        policy=dict(seed=42, epochs=200, final_epoch=199, batch_size=32,
            poison='same v3 A1 indices, doses, and right-leg body-axis target',
            clean_reference='portable completed HPE-Li v3 Clean checkpoint',
            test_checkpoint_selection=False, no_amp=True,
            no_batch_fallback=True, no_automatic_retry=True))
    result['plan_sha256'] = base.object_sha(result)
    return result


def _out(plan):
    return Path(plan['cfg']['outdir'])


def _cell(plan, method):
    if method not in METHODS:
        raise ValueError(f'Unknown PiW HPE-Li method: {method}')
    return _out(plan) / 'hpeli/seed42' / method


def _execution(device):
    return base._setup_execution(device)


def _cell_config(plan, device):
    result = copy.deepcopy(plan['cfg'])
    result['device'] = device
    return result


def _archive_reference(path, *, load_checkpoints=False):
    """Authenticate the portable v3 reference independent of original OS paths."""
    import torch
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != REFERENCE_FILES | {'reference_manifest.json'}:
            raise ValueError('Portable v3 archive has missing, duplicate or foreign members')
        meta = json.loads(archive.read('reference_manifest.json'))
        if (meta.get('schema') != 1 or meta.get('profile') !=
                'piw_hpeli_v3_portable_clean_reference_s42' or
                meta.get('source_profile') != v3.PROFILE or
                set(meta.get('files', {})) != REFERENCE_FILES or
                meta.get('seed') != 42 or meta.get('final_epoch') != 199):
            raise ValueError('Portable v3 reference manifest contract differs')
        for name in REFERENCE_FILES:
            info = archive.getinfo(name)
            expected = meta['files'][name]
            if info.is_dir() or info.file_size != expected['bytes']:
                raise ValueError(f'Portable v3 reference size changed: {name}')
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected['sha256']:
                raise ValueError(f'Portable v3 reference SHA256 changed: {name}')
        original = json.loads(archive.read(v3.MANIFEST))
        evaluation = json.loads(archive.read('evaluation.json'))
        poison = json.loads(archive.read('poison.json'))
        if (original.get('profile') != v3.PROFILE or
                original.get('plan_sha256') != meta['source_plan_sha256'] or
                original.get('inputs', {}).get('fingerprint') != meta['input_fingerprint'] or
                original.get('poison') != poison or
                poison.get('poison_plan_sha256') != meta['poison_plan_sha256'] or
                original.get('cfg', {}).get('target') != meta['target'] or
                meta['target'] != target_contract() or
                {k: original['cfg'][k] for k in RECIPE_KEYS} != meta['training_recipe'] or
                original.get('test_truth_sha256') != meta['test_truth_sha256'] or
                evaluation.get('plan_sha256') != meta['source_plan_sha256'] or
                evaluation.get('profile') != v3.PROFILE or
                evaluation.get('status') != v3.STATUS or
                evaluation.get('clean_metrics') != meta['clean_metrics']):
            raise ValueError('Portable reference scientific binding differs from v3')
        if (original.get('plan_sha256') != base.object_sha({
                k: value for k, value in original.items() if k != 'plan_sha256'}) or
                original['inputs']['fingerprint'] !=
                base.object_sha(original['inputs']['splits'])):
            raise ValueError('Portable v3 original plan/input digest differs')
        for name in REFERENCE_FILES - {v3.MANIFEST, 'evaluation.json'}:
            if name in evaluation.get('artifacts', {}) and (
                    evaluation['artifacts'][name] != meta['files'][name]['sha256']):
                raise ValueError(f'Portable v3 evaluated artifact differs: {name}')
        if set(evaluation.get('artifacts', {})) != {
                'predictions.npz', 'poison.json',
                'hpeli/seed42/clean/checkpoint.pt',
                'hpeli/seed42/a1/checkpoint.pt'}:
            raise ValueError('Portable v3 evaluated artifact list differs')
        clean_blob = torch.load(BytesIO(archive.read(
            'hpeli/seed42/clean/checkpoint.pt')), map_location='cpu', weights_only=False)
        a1_blob = torch.load(BytesIO(archive.read(
            'hpeli/seed42/a1/checkpoint.pt')), map_location='cpu', weights_only=False)
        for role, blob in (('clean', clean_blob), ('a1', a1_blob)):
            if (blob.get('profile') != v3.PROFILE or blob.get('role') != role or
                    blob.get('plan_sha256') != meta['source_plan_sha256'] or
                    blob.get('cfg') != original['cfg'] or blob.get('epoch') != 199 or
                    any(key not in blob for key in ('model', 'optimizer', 'rng', 'execution'))):
                raise ValueError(f'Portable v3 {role} checkpoint contract differs')
            model = base.build_model()
            model.load_state_dict(blob['model'], strict=True)
        if meta['clean_checkpoint_epoch'] != 199 or meta['a1_checkpoint_epoch'] != 199:
            raise ValueError('Portable v3 reference checkpoint is not final199')
        with np.load(BytesIO(archive.read('predictions.npz')), allow_pickle=False) as p:
            if set(p.files) != {'truth', 'names', 'doses', 'predictions',
                                'trigger_predictions', 'reference_predictions',
                                'reference_trigger_predictions', 'targets'}:
                raise ValueError('Portable v3 predictions schema differs')
            truth, names, doses = p['truth'], p['names'], p['doses']
            if (hashlib.sha256(truth.tobytes()).hexdigest() != meta['test_truth_sha256'] or
                    names.tolist() != original['inputs']['splits']['test_data']['ordered_names'] or
                    not np.array_equal(doses, DOSES) or
                    not np.array_equal(p['targets'], np.stack(
                        [target_pose(truth, dose) for dose in DOSES], axis=1))):
                raise ValueError('Portable v3 TEST truth, identity, doses or targets differ')
            v3.replay_equal(evaluation['clean_metrics'],
                            base._metrics(p['reference_predictions'], truth))
            v3.replay_equal(evaluation['a1_metrics'], measure_predictions(
                truth, p['predictions'], p['trigger_predictions'],
                p['reference_predictions'], p['reference_trigger_predictions']))
    result = dict(meta=meta, original=original, evaluation=evaluation,
                  poison=poison)
    if load_checkpoints:
        result['clean_checkpoint'] = clean_blob
        result['a1_checkpoint'] = a1_blob
    return result


def _verify_operator(method, train, cfg):
    import torch
    trigger = two.make_trigger(method, train, cfg)
    probes = [train[1]['csi'], train[len(train) // 2]['csi']]
    if method == 'smooth_lf_csi':
        changed = False
        for x in probes:
            zero, full = trigger.inject(x, 0.), trigger.inject(x, 1.)
            if not np.array_equal(zero, x) or not np.array_equal(full[:, 90:], x[:, 90:]):
                raise ValueError('Smooth-LF identity or phase contract differs')
            if not np.isfinite(full).all() or np.any(full < 0) or np.any(full > 1):
                raise ValueError('Smooth-LF output is not finite normalized CSI')
            changed |= bool(np.any(full[:, :90] != x[:, :90]))
        if not changed:
            raise ValueError('Smooth-LF is inactive on TRAIN probes')
    else:
        trigger.eval()
        x = torch.from_numpy(np.stack(probes))
        with torch.no_grad():
            zero = trigger.inject_tensor(x, 0., eps=cfg['tsba']['eps'])
            full = trigger.inject_tensor(x, 1., eps=cfg['tsba']['eps'])
        if not torch.equal(zero, x) or not torch.equal(full[:, :, 90:], x[:, :, 90:]):
            raise ValueError('TSBA identity or phase contract differs')
        if not torch.isfinite(full).all() or bool(((full < 0) | (full > 1)).any()):
            raise ValueError('TSBA output is not finite normalized CSI')
    return trigger


def _datasets(plan):
    from data_utils.feeder import PersonInWiFi3D
    cfg = plan['cfg']
    result = {}
    for mode, split in (('training', 'train_data'), ('validation', 'test_data')):
        ds = PersonInWiFi3D(mode, cfg['dataset_root'], 'one-person', 1)
        if [row['name'] for row in ds.items] != plan['inputs']['splits'][split]['ordered_names']:
            raise ValueError(f'PiW {split} identities/order changed')
        result[mode] = ds
    return result


def _quick_inputs(plan):
    cfg = plan['cfg']
    for split in ('train_data', 'test_data'):
        filename = Path(cfg['dataset_root']) / split / f'{split}_list.txt'
        if _sha(filename) != plan['inputs']['splits'][split]['official_list_sha256']:
            raise ValueError(f'PiW {split} official list changed')
    if trigger_record(build_a1(cfg['action_npy'])) != plan['reference_trigger']:
        raise ValueError('PiW A1 action differs from portable v3 reference')
    expected = json.loads((_out(plan) / 'preflight_receipt.json').read_text())
    stats = two.input_stat_fingerprint(cfg['dataset_root'], plan['inputs'])
    if stats != expected.get('input_stat_fingerprint'):
        raise ValueError('PiW TRAIN/TEST file stat fingerprint changed after full audit')


def load_plan(args):
    expected = _plan(args)
    path = Path(expected['cfg']['outdir']) / 'plan.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    if (saved.get('profile') != PROFILE or
            saved.get('plan_sha256') != base.object_sha({
                key: value for key, value in saved.items() if key != 'plan_sha256'}) or
            any(saved.get(key) != value for key, value in expected.items()
                if key != 'plan_sha256')):
        raise ValueError('PiW HPE-Li plan source/runtime/archive/config changed')
    receipt = json.loads((Path(expected['cfg']['outdir']) /
                          'preflight_receipt.json').read_text(encoding='utf-8'))
    if (receipt.get('status') != 'PASS' or receipt.get('plan_sha256') !=
            saved['plan_sha256'] or receipt.get('input_fingerprint') !=
            saved['inputs']['fingerprint']):
        raise ValueError('PiW HPE-Li preflight receipt differs from plan')
    _quick_inputs(saved)
    return saved


def preflight(args):
    from piw_v2_adapter import AXIS_MIN_METRES
    candidate = _plan(args)
    output = _out(candidate)
    if (output / 'plan.json').exists() or (output / 'preflight_receipt.json').exists():
        saved = load_plan(args)
        print('PREFLIGHT_PASS_EXISTING; no training launched', flush=True)
        return saved
    ref = _archive_reference(candidate['reference_archive'])
    cfg = candidate['cfg']
    if ({key: cfg[key] for key in RECIPE_KEYS} != ref['meta']['training_recipe'] or
            cfg['target'] != ref['meta']['target'] or
            cfg['eval_doses'] != list(DOSES) or cfg['epochs'] != 200 or
            cfg['final_epoch'] != 199 or cfg['seed'] != 42):
        raise ValueError('New HPE-Li recipe/target differs from portable v3')
    inputs, ds = base.audit_inputs(cfg)
    if inputs['fingerprint'] != ref['meta']['input_fingerprint'] or inputs['splits'] != (
            ref['original']['inputs']['splits']):
        raise ValueError('PiW TRAIN/TEST content differs from portable v3')
    trigger = build_a1(cfg['action_npy'])
    if trigger_record(trigger) != ref['original']['trigger']:
        raise ValueError('PiW A1 action/operator differs from portable v3')
    poison = A1Pairs(ds['training'], trigger).manifest()
    if poison != ref['poison']:
        raise ValueError('PiW poison indices/doses/target differ from portable v3')
    for mode in ('training', 'validation'):
        for row in ds[mode].items:
            pose = ds[mode].load_pose(row['kpt'])[0]
            if np.linalg.norm(pose[5] - pose[7]) <= AXIS_MIN_METRES:
                raise ValueError(f'Degenerate PiW hip axis: {row["name"]}')
    for method in METHODS:
        _verify_operator(method, ds['training'], cfg)
    stats = two.input_stat_fingerprint(cfg['dataset_root'], inputs)
    candidate.update(inputs=inputs, poison=poison,
        reference_manifest=ref['meta'], reference_trigger=ref['original']['trigger'],
        test_truth_sha256=ref['meta']['test_truth_sha256'])
    candidate['plan_sha256'] = base.object_sha({
        key: value for key, value in candidate.items() if key != 'plan_sha256'})
    output.mkdir(parents=True, exist_ok=True)
    base.atomic_json(output / 'plan.json', candidate)
    base.atomic_json(output / 'preflight_receipt.json', dict(
        schema=1, status='PASS', plan_sha256=candidate['plan_sha256'],
        input_fingerprint=inputs['fingerprint'], input_stat_fingerprint=stats,
        reference_archive_sha256=candidate['reference_archive_sha256'],
        n_train=len(ds['training']), n_test=len(ds['validation']),
        methods=list(METHODS), devices=DEVICES,
        note='Full PiW data and portable v3 Clean/A1 artifact audit; CPU only'))
    print('PREFLIGHT_PASS; no training launched', flush=True)
    return candidate


def _check_device(plan, method, device):
    if device != plan['devices'][method]:
        raise ValueError(f'{method} is bound to {plan["devices"][method]}')


def _smoke_path(plan, method):
    return _out(plan) / 'smoke' / f'{method}.json'


def smoke_cell(plan, method, device):
    import torch
    from torch.utils.data import default_collate
    _check_device(plan, method, device)
    cfg = plan['cfg']
    execution = v3.seed_execution(_cell_config(plan, device))
    ds = _datasets(plan)
    model = base.build_model().to(device).train()
    optimizer = v3.optimizer_for(model, cfg)
    trigger = two.make_trigger(method, ds['training'], cfg)
    data, poison_record = two.make_training_data(plan, method, ds, trigger)
    indices = [int(row['index']) for row in plan['poison']['samples'][:4]]
    selected = set(indices)
    for index in range(len(data)):
        if len(indices) == cfg['batch_size']:
            break
        if index not in selected and index not in data.selected:
            indices.append(index)
            selected.add(index)
    if len(indices) != cfg['batch_size']:
        raise ValueError('Could not make physical PiW B32 smoke batch')
    batch = default_collate([data[index] for index in indices])
    if method == 'tsba_adapted':
        trigger = trigger.to(device)
        trigger_opt = torch.optim.Adam(trigger.parameters(), lr=cfg['tsba']['generator_lr'])
    else:
        trigger_opt = None
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    loss, count, gen = two._train_batch(model, optimizer, batch, device, 'hpeli',
        trigger=trigger, trigger_opt=trigger_opt, cfg=cfg,
        epoch=cfg['tsba']['warmup_epochs'], last_micro=True,
        group_size=cfg['batch_size'])
    if (count != cfg['batch_size'] or not math.isfinite(loss) or
            (method == 'tsba_adapted' and len(gen) != cfg['tsba']['generator_steps'])):
        raise ValueError('PiW HPE-Li B32 smoke failed')
    record = dict(schema=1, status='PASS', profile=PROFILE, method=method,
        plan_sha256=plan['plan_sha256'], device=device, execution=execution,
        batch_size=count, loss=loss, generator_steps=len(gen),
        poison=poison_record,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved=torch.cuda.max_memory_reserved(device))
    path = _smoke_path(plan, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(path, record)
    print('GPU_SMOKE_WORKER_PASS ' + json.dumps(record, sort_keys=True), flush=True)
    return record


def _smoke_receipt(plan, method, device):
    record = json.loads(_smoke_path(plan, method).read_text(encoding='utf-8'))
    if (record.get('status') != 'PASS' or record.get('profile') != PROFILE or
            record.get('method') != method or record.get('device') != device or
            record.get('plan_sha256') != plan['plan_sha256'] or
            record.get('execution') != _execution(device) or
            record.get('batch_size') != 32):
        raise ValueError('PiW smoke receipt missing or mismatched')
    return record


def _checkpoint(plan, method, *, final=False):
    import torch
    path = _cell(plan, method) / 'checkpoint.pt'
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if (blob.get('profile') != PROFILE or blob.get('method') != method or
            blob.get('plan_sha256') != plan['plan_sha256'] or
            type(blob.get('epoch')) is not int or not 0 <= blob['epoch'] < 200 or
            any(key not in blob for key in ('model', 'optimizer', 'rng',
                'execution', 'poison', 'trigger', 'trigger_optimizer'))):
        raise ValueError(f'{method}: foreign or incomplete checkpoint')
    if final and blob['epoch'] != 199:
        raise ValueError(f'{method}: final epoch199 is required')
    return blob


def train_cell(plan, method, device):
    import torch
    from torch.utils.data import DataLoader
    _check_device(plan, method, device)
    cfg = plan['cfg']
    execution = v3.seed_execution(_cell_config(plan, device))
    _smoke_receipt(plan, method, device)
    ds = _datasets(plan)
    model = base.build_model().to(device)
    optimizer = v3.optimizer_for(model, cfg)
    trigger = two.make_trigger(method, ds['training'], cfg)
    data, poison_record = two.make_training_data(plan, method, ds, trigger)
    if method == 'tsba_adapted':
        trigger = trigger.to(device)
        trigger_opt = torch.optim.Adam(trigger.parameters(), lr=cfg['tsba']['generator_lr'])
    else:
        trigger_opt = None
    folder = _cell(plan, method)
    folder.mkdir(parents=True, exist_ok=True)
    start = 0
    if (folder / 'checkpoint.pt').exists():
        blob = _checkpoint(plan, method)
        if blob['execution'] != execution or blob['poison'] != poison_record:
            raise ValueError(f'{method}: execution/poison changed on resume')
        model.load_state_dict(blob['model'], strict=True)
        optimizer.load_state_dict(blob['optimizer'])
        if method == 'tsba_adapted':
            trigger.load_state_dict(blob['trigger'], strict=True)
            trigger_opt.load_state_dict(blob['trigger_optimizer'])
        elif blob['trigger'] is not None or blob['trigger_optimizer'] is not None:
            raise ValueError('Smooth-LF checkpoint unexpectedly contains learned trigger')
        base._restore_rng(blob['rng'], device)
        start = blob['epoch'] + 1
        print(f'[PiW HPE-Li extra] RESUME {method} from epoch{start}', flush=True)
    loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0,
                        drop_last=True)
    expected = len(loader) * 32
    if not expected:
        raise ValueError('PiW TRAIN is smaller than physical B32')
    print(f'[PiW HPE-Li extra] {method}: TRAIN={len(data)} poison={poison_record["n_poison"]} '
          f'B32 AdamW lr={cfg["lr"]:.3g} final199; TEST deferred', flush=True)
    for epoch in range(start, 200):
        began = time.monotonic()
        model.train(); optimizer.zero_grad(set_to_none=True)
        total, count, generated = 0., 0, []
        for batch in loader:
            loss, n, gen = two._train_batch(model, optimizer, batch, device, 'hpeli',
                trigger=trigger, trigger_opt=trigger_opt, cfg=cfg, epoch=epoch,
                last_micro=True, group_size=32)
            total += loss * n; count += n; generated.extend(gen)
        if count != expected:
            raise RuntimeError(f'{method}: unexpected TRAIN epoch coverage')
        base.atomic_checkpoint(folder / 'checkpoint.pt', dict(
            profile=PROFILE, method=method, plan_sha256=plan['plan_sha256'],
            epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            trigger=trigger.state_dict() if method == 'tsba_adapted' else None,
            trigger_optimizer=trigger_opt.state_dict() if trigger_opt else None,
            poison=poison_record, rng=base._rng_state(device),
            execution=execution, train_loss=total / count,
            generator_loss=float(np.mean(generated)) if generated else None))
        print(f'epoch {epoch}/199: loss={total/count:.8f} '
              f'lr={cfg["lr"]:.2e} t={time.monotonic()-began:.1f}s'
              + (f' tsba_loss={np.mean(generated):.8f}' if generated else ''), flush=True)
        print(f'[ckpt] saved -> {folder / "checkpoint.pt"} (epoch{epoch})', flush=True)
    _checkpoint(plan, method, final=True)
    print(f'TRAIN_COMPLETE {method}; fixed final199; TEST deferred', flush=True)


def evaluate_cell(plan, method, device):
    _check_device(plan, method, device)
    _smoke_receipt(plan, method, device)
    blob = _checkpoint(plan, method, final=True)
    if blob['execution'] != _execution(device):
        raise ValueError('Evaluation backend differs from training checkpoint')
    ref = _archive_reference(plan['reference_archive'], load_checkpoints=True)
    if ref['meta'] != plan['reference_manifest']:
        raise ValueError('Portable v3 reference changed since preflight')
    ds = _datasets(plan)
    x, truth, names = two._test_arrays(ds['validation'])
    if (names != plan['inputs']['splits']['test_data']['ordered_names'] or
            _sha_bytes(truth.tobytes()) != plan['test_truth_sha256']):
        raise ValueError('PiW TEST truth/order differs from portable v3')
    victim = base.build_model().to(device)
    victim.load_state_dict(blob['model'], strict=True)
    clean = base.build_model().to(device)
    clean.load_state_dict(ref['clean_checkpoint']['model'], strict=True)
    trigger = two.make_trigger(method, ds['training'], plan['cfg'])
    if method == 'tsba_adapted':
        trigger.load_state_dict(blob['trigger'], strict=True)
        trigger = trigger.to(device)
    victim_grid = two._predict_grid(victim, x, trigger, DOSES, device)
    reference_grid = two._predict_grid(clean, x, trigger, DOSES, device)
    measured = measure_predictions(truth, victim_grid[:, 0], victim_grid,
                                   reference_grid[:, 0], reference_grid)
    folder = _cell(plan, method)
    folder.mkdir(parents=True, exist_ok=True)
    two._atomic_npz(folder / 'predictions.npz', truth=truth,
        names=np.asarray(names), doses=np.asarray(DOSES),
        predictions=victim_grid[:, 0], trigger_predictions=victim_grid,
        reference_predictions=reference_grid[:, 0],
        reference_trigger_predictions=reference_grid,
        targets=np.stack([target_pose(truth, d) for d in DOSES], axis=1))
    result = dict(schema=1, status='COMPLETE', profile=PROFILE,
        method=method, seed=42, plan_sha256=plan['plan_sha256'],
        reference_archive_sha256=plan['reference_archive_sha256'],
        clean_checkpoint_sha256=ref['meta']['files'][
            'hpeli/seed42/clean/checkpoint.pt']['sha256'],
        checkpoint_sha256=_sha(folder / 'checkpoint.pt'),
        predictions_sha256=_sha(folder / 'predictions.npz'),
        poison=blob['poison'], metrics=measured,
        limitations=['One seed; no uncertainty interval.',
            'PiW digital CSI adaptations, not exact source-paper reproductions.',
            'Smooth-LF is a seeded fixed low-frequency control.',
            'TSBA is a learned three-group normalized-amplitude adaptation.',
            'Historical PiW target and LR were development-selected after TEST was viewed.'])
    base.atomic_json(folder / 'evaluation.json', result)
    verify_evaluation(plan, method)
    print(f'EVAL_COMPLETE {method}: {measured["mainmetrics"]}', flush=True)
    return result


def verify_evaluation(plan, method):
    folder = _cell(plan, method)
    target = folder / 'evaluation.json'
    if not target.is_file():
        return None
    saved = json.loads(target.read_text(encoding='utf-8'))
    blob = _checkpoint(plan, method, final=True)
    if (saved.get('schema') != 1 or saved.get('status') != 'COMPLETE' or
            saved.get('profile') != PROFILE or saved.get('method') != method or
            saved.get('plan_sha256') != plan['plan_sha256'] or
            saved.get('reference_archive_sha256') != plan['reference_archive_sha256'] or
            saved.get('clean_checkpoint_sha256') != plan['reference_manifest']['files'][
                'hpeli/seed42/clean/checkpoint.pt']['sha256'] or
            saved.get('checkpoint_sha256') != _sha(folder / 'checkpoint.pt') or
            saved.get('predictions_sha256') != _sha(folder / 'predictions.npz') or
            saved.get('poison') != blob['poison']):
        raise ValueError(f'{method}: saved evaluation artifact binding differs')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as p:
        if set(p.files) != {'truth', 'names', 'doses', 'predictions',
                            'trigger_predictions', 'reference_predictions',
                            'reference_trigger_predictions', 'targets'}:
            raise ValueError('PiW HPE-Li prediction schema differs')
        truth = p['truth']
        if (_sha_bytes(truth.tobytes()) != plan['test_truth_sha256'] or
                p['names'].tolist() != plan['inputs']['splits']['test_data']['ordered_names'] or
                not np.array_equal(p['doses'], DOSES) or
                not np.array_equal(p['targets'], np.stack(
                    [target_pose(truth, dose) for dose in DOSES], axis=1))):
            raise ValueError('PiW HPE-Li TEST identity/target binding differs')
        replay = measure_predictions(truth, p['predictions'],
            p['trigger_predictions'], p['reference_predictions'],
            p['reference_trigger_predictions'])
    v3.replay_equal(saved['metrics'], replay)
    return saved


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--reference-archive', type=Path, required=True)
    parser.add_argument('--action-npy', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cell', choices=METHODS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--smoke-only', action='store_true')
    mode.add_argument('--evaluate-only', action='store_true')
    mode.add_argument('--verify-cell', action='store_true')
    mode.add_argument('--status', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    if args.dry_run:
        print(json.dumps(_plan(args), indent=2, sort_keys=True), flush=True)
        return 0
    if args.preflight_only:
        preflight(args)
        return 0
    if args.smoke_only or args.evaluate_only or args.verify_cell or not args.status:
        if args.cell not in METHODS:
            raise ValueError('Select --cell smooth_lf_csi or tsba_adapted')
    plan = load_plan(args)
    if args.status:
        for method in METHODS:
            folder = _cell(plan, method)
            if verify_evaluation(plan, method):
                state = 'VERIFIED_EVALUATION_COMPLETE'
            elif (folder / 'checkpoint.pt').is_file():
                epoch = _checkpoint(plan, method)['epoch']
                state = 'FINAL199_TRAINED' if epoch == 199 else f'TRAINING_EPOCH_{epoch}'
            else:
                state = 'PENDING'
            print(f'{method}: {state}', flush=True)
        return 0
    if args.verify_cell:
        result = verify_evaluation(plan, args.cell)
        if result is None:
            raise ValueError(f'{args.cell}: evaluation is not complete')
        print(f'VERIFIED_EVALUATION_COMPLETE {args.cell}', flush=True)
        return 0
    if args.smoke_only:
        smoke_cell(plan, args.cell, args.device)
    elif args.evaluate_only:
        evaluate_cell(plan, args.cell, args.device)
    else:
        train_cell(plan, args.cell, args.device)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'[PiW HPE-Li two extra] ERROR: {error}', file=sys.stderr, flush=True)
        raise
