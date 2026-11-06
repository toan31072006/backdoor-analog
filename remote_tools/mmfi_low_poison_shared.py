"""T2 execution adapter: exactly the existing A1 recipe, only rho=.01/.05.

No changes to scientific source files. Private evaluator registration adds the
two new cell IDs and their dose1 pose/PCK replay; native train still evaluates
the complete six-dose grid. This module is never a stand-alone GPU bypass.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'ATKBackd'))
PROFILE = 'mmfi_a1_low_poison_shared_s42_v1'
RATES = {'rho0p01': .01, 'rho0p05': .05}


def reference():
    from run_mmfi_p3_s3_three_attacks_4090 import read, sha
    result = read(Path(__file__).with_name('MMFI_A1_LOW_POISON_REFERENCE.json'))
    for name, expected in result['source_sha256'].items():
        if sha(REPO/'ATKBackd'/name) != expected:
            raise ValueError(f'Historical T2 scientific source changed: {name}')
    return result


def cells(home, root):
    from run_a1_dose_poison import _configuration
    from run_method_drafts import _sha_json
    ref, result = reference(), []
    for key, rho in RATES.items():
        cfg = _configuration(Path(home), 'cuda:0', 4, rho)
        cfg['confirmation_profile'] = PROFILE
        normalized = {k:v for k,v in cfg.items() if k not in ref['recipe_ignored_fields']}
        if _sha_json(normalized) != ref['recipe_sha256']:
            raise ValueError('New dose victim differs from the archived 10/20/40% recipe')
        folder = Path(root)/'dose'/key
        result.append(dict(key=key, seed=42, method_key='s42__'+key, origin='fresh',
            rho=rho, n_poison=math.floor(rho*133056), cfg=cfg,
            ckpt_dir=str(folder), run_dir=str(folder), eval_cache=str(folder/'eval_cache.json'),
            artifact_dir=str(Path(root)/'evaluations'/key)))
    return result


def audit_input(cell):
    import run_a1_three_tables as frozen
    ref = reference()
    row = frozen._current_inputs(cell)
    if (any(row[k] != ref['ordered_inputs'][k] for k in ('train','eval'))
            or row['action_file_sha256'] != ref['action_sha256']):
        raise ValueError('Dose study must use the same ordered P1S1 splits/prior as old T2')
    return row


def evaluator():
    source = REPO/'ATKBackd/a1_three_tables_eval.py'
    name = '_low_poison_eval_'+uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.ALL_KEYS = module.ALL_KEYS | frozenset(RATES)
    module.T1_KEYS = module.T1_KEYS | frozenset(RATES)
    return module


def validate_resume(cell):
    """Native loader is reused, but never permit its legacy missing-RNG warning."""
    import torch
    from train_backdoor import _config_fingerprint
    blob = torch.load(Path(cell['ckpt_dir'])/'checkpoint.pt', map_location='cpu', weights_only=False)
    rng = blob.get('rng') or {}
    if (blob.get('cfg_fingerprint') != _config_fingerprint(cell['cfg'])
            or type(blob.get('epoch')) is not int or not 0 <= blob['epoch'] <= 49
            or not blob.get('model') or not blob.get('optimizer')
            or set(rng) != {'python','numpy','torch','cuda'}):
        raise ValueError('Incomplete native dose checkpoint/config/RNG; no silent restart')
    if str(cell['cfg']['device']).startswith('cuda') and not rng['cuda']:
        raise ValueError('Dose checkpoint lost CUDA RNG; no silent reseeding')
    groups = blob['optimizer'].get('param_groups',[])
    if (not groups or not blob['optimizer'].get('state')
            or any(g.get('lr') != .001 or g.get('momentum') != .9 or g.get('weight_decay') != 0. for g in groups)
            or any(not torch.isfinite(v).all() for v in blob['model'].values())):
        raise ValueError('Native dose model/SGD state changed')


def execute(cell, plan_hash, directory):
    """Reuse native training/audits/replay, without editing any canonical module."""
    import torch
    import run_a1_three_tables as frozen
    from train_backdoor import train
    root, record = Path(cell['ckpt_dir']), audit_input(cell)
    eva = evaluator()
    with frozen._cell_lock(root):
        path = root/'three_tables_inputs.json'
        if path.exists():
            if frozen._json(path) != record:
                raise ValueError('Dose input audit changed; no restart/import')
        elif any((root/name).exists() for name in ('checkpoint.pt','eval_cache.json')):
            raise ValueError('Unbound dose checkpoint/cache; preserve it')
        else:
            frozen._atomic_json(path, record)
        marker = root/'shared_start.json'
        binding = dict(plan_sha256=plan_hash, cell=cell['key'], input=record)
        if marker.exists():
            if frozen._json(marker) != binding or not (root/'checkpoint.pt').exists():
                raise ValueError('Interrupted dose run lacks its bound checkpoint; explicit recovery required')
            validate_resume(cell)
        else:
            frozen._atomic_json(marker, binding)
        model, unused = train(cell['cfg'], ckpt_dir=cell['ckpt_dir'])
        del model, unused
        torch.cuda.empty_cache()
        if audit_input(cell) != record:
            raise ValueError('Dose inputs changed during native training')
        run = frozen._read_completed(cell)
        audit = frozen._audit_run(cell, run, record)
        evaluating = dict(cell, verified_inputs=record)
        result = eva.evaluate_cell(evaluating, cell['artifact_dir'], cell['cfg']['device'])
        frozen._atomic_json(root/'scientific_audit.json', audit)
        print(f'LOW_POISON_COMPLETE {cell["key"]} rho={cell["rho"]}; final49/six doses', flush=True)
        return result


def smoke(cell, barrier):
    import torch
    from train_backdoor import _load_dataset, build_trigger, _build_optimizer, _victim_update
    from models.factory import build_model
    from attack.poison import PoisonedDataset, collate
    data = _load_dataset(cell['cfg'], 'training')
    trigger = build_trigger(cell['cfg'])
    pairs = PoisonedDataset(data, trigger, mode='train', rho=cell['rho'],
        dose_min=.2, dose_max=1., eps=.185, pivot=1, theta_max_deg=40.,
        dose_mode='linear', axis=[0.,0.,1.], seed=42, select='uniform',
        dataset='mmfi', dose_coupling='paired', cover_ratio=0.)
    selected = sorted(pairs.poison_idx)[:4]
    selected += [i for i in range(len(pairs)) if i not in pairs.poison_idx][:28]
    batch = collate([pairs[i] for i in selected])
    torch.manual_seed(42)
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False).to('cuda:0').train()
    name, optimizer = _build_optimizer(model, cell['cfg'], 'mmfi')
    if name != 'sgd' or tuple(batch['csi'].shape) != (32,3,114,10):
        raise ValueError('Native A1 SGD/B32 smoke required')
    torch.cuda.reset_peak_memory_stats(0)
    before = next(model.parameters()).detach().clone()
    barrier('models_resident')
    losses = [_victim_update(model, optimizer, batch['csi'].to('cuda:0'),
                            batch['pose'].to('cuda:0')) for _ in range(2)]
    if not all(math.isfinite(v) for v in losses) or torch.equal(before, next(model.parameters())):
        raise ValueError('A1 CUDA smoke failed')
    barrier('B64_backward_resident')
    barrier('B32_complete')
    return dict(status='CAPACITY_ONLY_NOT_TRAINED', discarded=True, real_train_input=True,
        batch=32, steps=2, poison_examples=4, losses=losses,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
