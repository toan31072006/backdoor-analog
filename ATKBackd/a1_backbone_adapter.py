"""Isolated model/recipe adapters for the unchanged, source-pinned A1 engine.

Private module instances reuse the existing training/evaluation implementation.
No global monkeypatch of the historical modules, fake HPE-Li config, or relaxed
artifact check is used. Every adapter and model source is bound by the new plan.
"""
from __future__ import annotations

import importlib.util
import copy
import json
import os
from pathlib import Path

from models.cross_backbone_factory import BACKBONES, build_backbone

HERE = Path(__file__).resolve().parent
RECIPE_ID = 'mmfi_cross_backbone_author_scalar_recipes_v1'
RECIPES = {
    'metafiplusplus': dict(optimizer='sgd', lr=.01, momentum=.9,
        weight_decay=0., batch_size=32, lr_scheduler=False,
        backbone_gradient_clip=None, backbone_optimizer_groups='all_parameters',
        backbone_effective_batch_size=32, backbone_accumulation_steps=1),
    'graphposefi': dict(optimizer='adamw', lr=.0003, weight_decay=.02,
        batch_size=32, lr_scheduler=True, warmup_epochs=5,
        backbone_gradient_clip=1., backbone_optimizer_groups='author_decay_exclusions',
        backbone_adam_betas=[.9, .999], backbone_adam_eps=1e-8,
        backbone_effective_batch_size=256, backbone_accumulation_steps=8),
}


def recipe_for(backbone, graph_load_batch_size=32):
    if backbone not in BACKBONES:
        raise ValueError('Unknown backbone')
    if type(graph_load_batch_size) is not int or graph_load_batch_size not in (32, 64, 128, 256):
        raise ValueError('Graph load batch must be 32/64/128/256, dividing effective batch 256')
    result = copy.deepcopy(RECIPES[backbone])
    if backbone == 'graphposefi':
        result.update(batch_size=graph_load_batch_size,
                      backbone_accumulation_steps=256 // graph_load_batch_size)
    return result


def recipe_config(cfg, backbone, *, graph_load_batch_size=32):
    """Replace only the victim recipe; attack/data/target settings are retained."""
    if backbone not in BACKBONES:
        raise ValueError('Unknown backbone')
    result = copy.deepcopy(cfg)
    for key in ('momentum', 'warmup_epochs'):
        result.pop(key, None)
    result.update(recipe_for(backbone, graph_load_batch_size), model=backbone, pretrained=False,
                  data_parallel=False, num_workers=0, epochs=50, victim_epochs=50,
                  ckpt_every=1, strict_resume=True, backbone_recipe_id=RECIPE_ID)
    validate_config(result)
    return result


def validate_config(cfg):
    backbone = cfg.get('model')
    if backbone not in RECIPES or cfg.get('backbone_recipe_id') != RECIPE_ID:
        raise ValueError('Missing or unknown frozen backbone/recipe')
    recipe = recipe_for(backbone, cfg['batch_size'] if backbone == 'graphposefi' else 32)
    required = dict(recipe, pretrained=False, data_parallel=False,
        num_workers=0, epochs=50, victim_epochs=50, strict_resume=True,
        experiment_name='mmfi', mmfi_protocol='protocol1', mmfi_setting='s1',
        mmfi_split_seed=0, mmfi_random_ratio=.8, num_person=1,
        n_ant=3, n_sub=114, n_pkt=10, victim_loss='mpjpe')
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError('Configuration differs from the frozen backbone/data recipe')
    from train_backdoor import _validate_training_contract
    _validate_training_contract(cfg)


def _private_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load pinned engine {filename}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_parameter_groups(model, weight_decay=.02):
    """Authors' membership rules, deterministic ordering for optimizer resume."""
    import torch.nn as nn
    excluded = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for name, _ in module.named_parameters(recurse=False):
                excluded.add(f'{module_name}.{name}' if module_name else name)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if (name in excluded or name.endswith('.bias')
                              or parameter.ndim == 1) else decay
        target.append(parameter)
    return [dict(params=decay, weight_decay=weight_decay),
            dict(params=no_decay, weight_decay=0.)]


def training_engine(cfg):
    """Return an isolated full engine with explicit backbone-only hooks."""
    validate_config(cfg)
    import torch
    engine = _private_module('train_backdoor.py', '_cross_backbone_train_engine')
    engine.build_model = build_backbone
    original_optimizer = engine._build_optimizer
    original_loader = engine._training_loader
    accumulation = dict(active=False, pending=0, scale=1., flush=True)

    def loader(dataset, actual):
        validate_config(actual)
        base = original_loader(dataset, actual)
        steps = actual['backbone_accumulation_steps']

        class AccumulatingLoader:
            def __len__(self):
                return len(base)

            def __iter__(self):
                if accumulation['pending']:
                    raise RuntimeError('Unflushed gradient accumulation at epoch boundary')
                accumulation['active'] = True
                try:
                    for index, batch in enumerate(base):
                        start = (index // steps) * steps * actual['batch_size']
                        group_samples = min(actual['backbone_effective_batch_size'],
                                            len(dataset) - start)
                        accumulation['scale'] = len(batch['csi']) / group_samples
                        accumulation['flush'] = ((index + 1) % steps == 0 or index + 1 == len(base))
                        yield batch
                    if accumulation['pending']:
                        raise RuntimeError('Epoch ended with incomplete optimizer update')
                finally:
                    accumulation['active'] = False

        return AccumulatingLoader()

    def optimizer(model, actual, dataset):
        validate_config(actual)
        if actual['model'] != 'graphposefi':
            return original_optimizer(model, actual, dataset)
        return 'adamw', torch.optim.AdamW(
            graph_parameter_groups(model, actual['weight_decay']), lr=actual['lr'],
            betas=tuple(actual['backbone_adam_betas']), eps=actual['backbone_adam_eps'],
            weight_decay=0.)

    def update(model, opt, csi, target):
        model.train()
        if not accumulation['active'] or not accumulation['pending']:
            opt.zero_grad(set_to_none=True)
        prediction, _ = model(csi)
        loss = engine._mpjpe_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite ordinary MPJPE; no automatic retry or recipe change')
        scale = accumulation['scale'] if accumulation['active'] else 1.
        (loss * scale).backward()
        accumulation['pending'] += 1
        if not accumulation['active'] or accumulation['flush']:
            if cfg['backbone_gradient_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['backbone_gradient_clip'],
                                              error_if_nonfinite=True)
            opt.step()
            accumulation['pending'] = 0
        return float(loss.detach())

    def save(path, model, opt, epoch, best_loss, actual, trigger=None,
             trigger_optimizer=None):
        # Atomic save protects the preceding epoch if the process is killed.
        if accumulation['pending']:
            raise RuntimeError('Checkpoint requires an epoch-boundary optimizer flush')
        path = Path(path)
        temporary = path.with_name(path.name + '.saving')
        blob = dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                    best_loss=best_loss, rng=engine._rng_state(), cfg=actual,
                    cfg_fingerprint=engine._config_fingerprint(actual))
        if trigger is not None:
            blob['trigger'] = trigger.state_dict()
            if trigger_optimizer is not None:
                blob['trigger_optimizer'] = trigger_optimizer.state_dict()
        try:
            torch.save(blob, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f'[ckpt] saved -> {path} (epoch {epoch})', flush=True)

    engine._build_optimizer = optimizer
    engine._training_loader = loader
    engine._victim_update = update
    engine._save_checkpoint = save
    return engine


def evaluation_engine(sources):
    """Reuse all metric and immutable-artifact checks with genuine model IDs."""
    engine = _private_module('a1_three_tables_eval.py', '_cross_backbone_eval_engine')

    def preflight(cell, outdir):
        from train_backdoor import _config_fingerprint
        validate_config(cell['cfg'])
        key, seed, cfg = cell['key'], cell['seed'], cell['cfg']
        if (key not in engine.ALL_KEYS or type(seed) is not int or seed not in (42, 43, 44)
                or cell.get('origin') != 'fresh' or cfg.get('seed') != seed
                or cfg.get('pivot') != 1 or cfg.get('target_joints', [2, 3]) != [2, 3]
                or cfg.get('theta_max_deg') != 40 or cfg.get('payload_axis') != [0., 0., 1.]
                or cfg.get('dose_mode') != 'linear'):
            raise ValueError('Cross-backbone cell differs from the frozen A1 target/seed contract')
        run = engine._read_run(cell['ckpt_dir'])
        if run['fingerprint'] != _config_fingerprint(cfg):
            raise ValueError('Completed run configuration differs from the new backbone plan')
        if Path(cell['eval_cache']).resolve() != run['folder'] / 'eval_cache.json':
            raise ValueError('Evaluation cache path differs from its run')
        outdir = Path(outdir).resolve()
        if (outdir == run['folder'] or outdir in run['folder'].parents
                or run['folder'] in outdir.parents):
            raise ValueError('Evaluation output must be separate from checkpoints')
        values, rows = engine._cache_metrics(run['res'], key)
        artifacts = engine._artifact_hashes(run)
        if any(artifacts[name] != digest for name, digest in run['metadata_sha256'].items()):
            raise ValueError('Run metadata changed while being loaded')
        results = run['folder'] / 'results.json'
        if results.exists():
            previous = json.loads(results.read_text(encoding='utf-8'))
            if (previous.get('res') != run['res'] or not isinstance(previous.get('cfg'), dict)
                    or _config_fingerprint(previous['cfg']) != run['fingerprint']):
                raise ValueError('results.json differs from the configuration/cache')
        engine._checkpoint_metadata(run)
        return key, seed, cfg, run, outdir, values, rows, artifacts

    def weights(run, device):
        import torch
        engine._checkpoint_metadata(run)
        validate_config(run['cfg'])
        blob = torch.load(run['folder'] / 'checkpoint.pt', map_location='cpu', weights_only=False)
        model = build_backbone(run['cfg']['model'])
        model.load_state_dict(blob['model'], strict=True)
        return model.to(device).eval(), blob.get('trigger')

    def current_sources():
        actual = {name: engine._sha(HERE / name) for name in sources}
        if actual != sources:
            raise ValueError('Scientific/model source changed; use a NEW output directory')
        return actual

    engine._preflight = preflight
    engine._load_weights = weights
    engine._source_hashes = current_sources
    return engine
