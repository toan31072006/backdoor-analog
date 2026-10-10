"""Train-only learned-carrier exploration; never writes publication results.

Five literature-inspired directions are independent hypotheses, NOT source
paper reproductions. Fitting finishes and freezes an artifact before a fresh
ordinary-ERM victim starts. A changed recipe/artifact fails closed on resume.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

from run_mmfi_tables import _atomic_json, _cell_lock, _check_inputs
from run_method_drafts import (
    _file_sha, _sha_json, _positive_int, _read_audit, _subset_identity,
    _verified_cache, _write_csv, trigger_state_sha256,
)

HERE = Path(__file__).resolve().parent
STATUS = 'DRAFT_ONLY_NOT_PAPER_RESULTS'
PROFILE = 'learned_carrier_screen_v1'
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
METHODS = (
    ('clean', 'Clean fresh victim control', 'micro_dropper', None),
    ('proposed', 'Fixed Proposed under dual budget', 'md_multicarrier_peak_matched', None),
    ('blended', 'Blended-CSI under dual budget', 'blended', None),
    ('lc_weights', 'Learned carrier weights (surrogate alternating)', 'learned_carrier', 'weights'),
    ('lc_sparse', 'Sparse carrier mask', 'learned_carrier', 'sparse'),
    ('lc_combined', 'Learned weights + sparse mask', 'learned_carrier', 'combined'),
    ('lc_gradient', 'Carrier gradient matching', 'learned_carrier', 'gradient'),
    ('lc_energy', 'Carrier energy regularization', 'learned_carrier', 'energy'),
    ('lc_selection', 'Training-only informative poison selection', 'md_multicarrier_peak_matched', 'selection'),
)
RUNTIME_KEYS = {'device', 'num_workers', 'ckpt_every'}
PREPARED_KEYS = {
    'lc_artifact_path', 'lc_artifact_sha256', 'lc_recipe_sha256',
    'lc_fitting_sha256', 'lc_poison_indices', 'lc_selection',
    'lc_selection_sha256',
}
FIT_DEFAULTS = dict(lc_fit_seed=4242, lc_warmup_epochs=3, lc_rounds=3, lc_inner_steps=32,
    lc_outer_steps=16, lc_batch_size=32, lc_lr=0.02,
    lc_energy_weight=0.2, lc_clean_weight=1.0, lc_fit_samples=4096,
    lc_inner_val_fraction=0.2, lc_gradient_tensors=2)
REFERENCES = {
    'sparse_mask': {'paper': 'https://arxiv.org/html/2306.06209v3',
        'code': 'https://github.com/YinghuaGao/SIBA'},
    'surrogate_alternation': {'paper': 'https://proceedings.iclr.cc/paper_files/paper/2024/file/1687466683649e8bdcdec0e3f5c8de64-Paper-Conference.pdf',
        'code': 'https://github.com/SWY666/SSL-backdoor-BLTO'},
    'gradient_matching': {'paper': 'https://proceedings.neurips.cc/paper_files/paper/2022/file/79eec295a3cd5785e18c61383e7c996b-Paper-Conference.pdf',
        'code': 'https://github.com/hsouri/Sleeper-Agent'},
    'energy_regularization': {'paper': 'https://openaccess.thecvf.com/content/ICCV2021/html/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.html',
        'code': 'https://github.com/sunbelbd/invisible_backdoor_attacks'},
    'informative_selection': {'paper': 'https://proceedings.iclr.cc/paper_files/paper/2025/file/1d8f05e4da49a4e1e1b052a3046bceac-Paper-Conference.pdf',
        'code': 'https://github.com/mail-research/wicked-oddities-backdoor'},
}


def _scientific(cfg):
    return {k: v for k, v in cfg.items() if k not in RUNTIME_KEYS}


def _source_provenance():
    paths = (
        'run_learned_carrier_drafts.py', 'learned_carrier_fit.py',
        'attack/learned_carrier.py', 'train_backdoor.py',
        'run_method_drafts.py', 'run_mmfi_tables.py', 'mmfi_tables.py',
        'configs/mmfi/attack_bend.yaml', 'attack/trigger.py',
        'attack/method_drafts.py', 'attack/traditional.py', 'attack/peak_budget.py',
        'attack/poison.py', 'attack/payload.py', 'attack/skeleton_mmfi.py',
        'data_utils/draft_subset.py', 'data_utils/feeder.py',
        'eval/distortion.py', 'eval/metrics.py', 'models/factory.py',
        'models/hpeli.py', 'models/sk_network.py',
        'third_party/backdoorbench/blended.py',
    )
    return {path: _file_sha(HERE / path) for path in paths}


def _recipe_sha(cfg, sources):
    return _sha_json(dict(profile=PROFILE, cfg=_scientific(cfg), sources=sources))


def _plan_sha(matrix):
    projected = copy.deepcopy(matrix)
    projected.pop('plan_sha256', None)
    projected.pop('created_utc', None)
    for cell in projected['cells']:
        cell['cfg'] = _scientific(cell['cfg'])
    return _sha_json(projected)


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *,
                 epochs=15, train_samples=20000, eval_samples=4096,
                 distortion_samples=256, relative_l2=0.10, fit_options=None,
                 skip_clean_probe=False, skip_distortion=False):
    """Metadata-only resolution: no data, trigger construction or CUDA probe."""
    from train_backdoor import _resolve_training_config, _validate_training_contract
    for name, value in [('epochs', epochs), ('train_samples', train_samples),
                        ('eval_samples', eval_samples), ('distortion_samples', distortion_samples)]:
        _positive_int(value, name)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    if isinstance(relative_l2, bool) or not isinstance(relative_l2, (int, float)) or not math.isfinite(relative_l2) or relative_l2 <= 0:
        raise ValueError('relative_l2 must be finite and positive')
    fit = dict(FIT_DEFAULTS)
    if fit_options:
        if set(fit_options) - set(fit):
            raise ValueError('Unknown fitting budget option')
        fit.update(fit_options)
    for key in ('lc_warmup_epochs', 'lc_rounds', 'lc_inner_steps', 'lc_outer_steps',
                'lc_batch_size', 'lc_fit_samples', 'lc_gradient_tensors'):
        _positive_int(fit[key], key)
    if isinstance(fit['lc_fit_seed'], bool) or not isinstance(fit['lc_fit_seed'], int) or fit['lc_fit_seed'] < 0:
        raise ValueError('lc_fit_seed must be a nonnegative integer')
    for key in ('lc_lr', 'lc_energy_weight', 'lc_clean_weight'):
        if isinstance(fit[key], bool) or not math.isfinite(float(fit[key])) or float(fit[key]) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if isinstance(fit['lc_inner_val_fraction'], bool) or not 0 < fit['lc_inner_val_fraction'] < 0.5:
        raise ValueError('lc_inner_val_fraction must be in (0,.5)')
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    sources = _source_provenance()
    base = yaml.safe_load((HERE / 'configs/mmfi/attack_bend.yaml').read_text(encoding='utf-8'))
    base.update(dataset_root=str(data_home / 'datasets/Compress'),
        action_npy=str(data_home / 'actions/data_bend.npy'),
        seed=42, model='hpeli', optimizer='sgd', lr=0.001, momentum=0.9,
        weight_decay=0.0, batch_size=32, epochs=epochs, victim_epochs=epochs,
        device=device, num_workers=num_workers, ckpt_every=1,
        strict_resume=True, pretrained=False, data_parallel=False,
        loader_persistent_workers=False, pivot=1, target_joints=[2, 3],
        theta_max_deg=40.0, payload_axis=[0.0, 0.0, 1.0],
        dose_mode='linear', dose_grid=copy.deepcopy(GRID), rho=0.1,
        poison_select='uniform', dose_min=0.2, dose_max=1.0,
        dose_coupling='paired', trigger_zero_mean=True,
        training_protocol='ordinary_erm', threat_model='training_data_poisoning',
        attacker_access='data_only', method_draft=True, draft_profile=PROFILE,
        draft_train_samples=train_samples, draft_eval_samples=eval_samples,
        draft_subset_seed=0, draft_eval_source='training_holdout',
        lc_reference_eps=0.185,
        lc_relative_l2=float(relative_l2), **fit)
    cells = []
    for key, label, trigger, variant in METHODS:
        cfg = copy.deepcopy(base)
        cfg.update(trigger=trigger, rho=0.0 if key == 'clean' else 0.1)
        if variant:
            cfg['lc_variant'] = variant
        if key == 'blended':
            cfg['eps'] = 0.2
        cfg = _resolve_training_config(cfg)
        _validate_training_contract(cfg)
        folder = outdir / key
        cells.append(dict(method_key=key, label=label,
            group='selection_ablation' if variant == 'selection' else 'shared_poison_trigger_screen',
            cfg=cfg, recipe_sha256=_recipe_sha(cfg, sources),
            ckpt_dir=str(folder), eval_cache=str(folder / 'eval_cache.json')))
    matrix = dict(schema=1, status=STATUS, DRAFT_ONLY=True, dataset='mmfi',
        seed=42, draft_profile=PROFILE, created_utc=datetime.now(timezone.utc).isoformat(),
        sources=dict(source_sha256=sources, references=REFERENCES,
            implementation='Independent literature-inspired CSI hypotheses, NOT source reproduction'),
        cells=cells, metrics_contract=dict(official_test_used=False,
            evaluation_role='disjoint official TRAIN holdout', dose_grid=GRID,
            poison_indices='same fixed uniform identities except separate lc_selection ablation',
            pck='relative .5/.4/.3/.2/.1, NOT millimetres',
            peak_ceiling='actual per-input/per-dose Original peak at eps=0.185',
            l2_ceiling='actual per-input relative L2 <= lc_relative_l2*d',
            l2_bound=float(relative_l2), realizes_equal_noise=False,
            fitting_access='separate train-only surrogate; deployed victim data-only ordinary ERM',
            victim_initialization='fresh seed-42 victim; no surrogate parameter transfer',
            clean_probe='missing_by_explicit_skip' if skip_clean_probe else 'every candidate on same clean fresh checkpoint at d=1',
            distortion='missing_by_explicit_skip' if skip_distortion else 'same holdout CSI pairs, no HPE forward',
            distortion_samples=distortion_samples, uncertainty='single seed; not estimated'))
    matrix['plan_sha256'] = _plan_sha(matrix)
    return matrix


def _validate_manifest(matrix):
    if (matrix.get('schema') != 1 or matrix.get('status') != STATUS
            or matrix.get('DRAFT_ONLY') is not True or matrix.get('draft_profile') != PROFILE
            or matrix.get('seed') != 42 or matrix.get('dataset') != 'mmfi'):
        raise ValueError('Not the isolated learned-carrier draft manifest')
    if matrix.get('plan_sha256') != _plan_sha(matrix):
        raise ValueError('Plan fingerprint differs from its contents')
    sources = _source_provenance()
    if matrix['sources']['source_sha256'] != sources:
        raise ValueError('Scientific source files changed; use a NEW output directory')
    if [c['method_key'] for c in matrix['cells']] != [m[0] for m in METHODS]:
        raise ValueError('All nine planned cells must remain in fixed order')
    for cell, (key, _, trigger, variant) in zip(matrix['cells'], METHODS):
        cfg = cell['cfg']
        if (cfg.get('trigger') != trigger or cfg.get('lc_variant') != variant
                or cfg.get('rho') != (0 if key == 'clean' else 0.1)
                or cfg.get('draft_profile') != PROFILE
                or cell['recipe_sha256'] != _recipe_sha(cfg, sources)):
            raise ValueError(f'{key}: immutable recipe disagrees with plan')


def _validated_prepared(cell, prepared):
    """The fitter may add artifacts/selection, not silently change the task."""
    from mmfi_tables import config_fingerprint
    if not isinstance(prepared, dict):
        raise ValueError('Prepared config must be a dictionary')
    immutable = {k: v for k, v in _scientific(prepared).items() if k not in PREPARED_KEYS}
    requested = {k: v for k, v in _scientific(cell['cfg']).items() if k not in PREPARED_KEYS}
    if immutable != requested:
        raise ValueError(f"{cell['method_key']}: fitting changed immutable scientific settings")
    variant = prepared.get('lc_variant')
    if variant:
        if prepared.get('lc_recipe_sha256') != cell['recipe_sha256']:
            raise ValueError('Prepared trigger recipe mismatch')
        fit_file = Path(cell['ckpt_dir']) / 'fitting.json'
        if _file_sha(fit_file) != prepared.get('lc_fitting_sha256'):
            raise ValueError('Fitting provenance file changed')
        fit_record = json.loads(fit_file.read_text(encoding='utf-8'))
        if (fit_record.get('recipe_sha256') != cell['recipe_sha256']
                or fit_record.get('variant') != variant
                or fit_record.get('official_test_loaded') is not False
                or fit_record.get('external_draft_holdout_loaded') is not False):
            raise ValueError('Fitting record violates the recipe/TRAIN-only protocol')
        if variant == 'selection':
            indices = prepared.get('lc_poison_indices')
            if (not isinstance(indices, list) or not indices
                    or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
                    or len(indices) != len(set(indices))
                    or _sha_json(indices) != prepared.get('lc_selection_sha256')):
                raise ValueError('Invalid informative-selection identities or hash')
        else:
            path = Path(prepared.get('lc_artifact_path', ''))
            if path.resolve().parent != Path(cell['ckpt_dir']).resolve():
                raise ValueError('Artifact must be frozen inside its own cell directory')
            if _file_sha(path) != prepared.get('lc_artifact_sha256'):
                raise ValueError('Frozen learned-trigger artifact changed')
            artifact = json.loads(path.read_text(encoding='utf-8'))
            operator_keys = ('lc_relative_l2', 'lc_reference_eps', 'lc_mask_fraction',
                'lc_carrier_sub_mode', 'lc_carrier_time_mode', 'lc_carrier_seed')
            if (artifact.get('schema') != 1 or artifact.get('status') != STATUS
                    or artifact.get('recipe_sha256') != cell['recipe_sha256']
                    or artifact.get('variant') != variant
                    or artifact.get('operator_config') != {k: prepared[k] for k in operator_keys}):
                raise ValueError('Learned artifact provenance/operator contract mismatch')
    config_fingerprint(prepared)  # A final config, including artifact SHA, binds the victim.
    return prepared


def _prepared_cell(cell):
    path = Path(cell['ckpt_dir']) / 'prepared_cfg.json'
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding='utf-8'))
    cfg = blob.get('cfg')
    _validated_prepared(cell, cfg)
    if blob.get('recipe_sha256') != cell['recipe_sha256']:
        raise ValueError('Prepared config record has a different recipe')
    from mmfi_tables import config_fingerprint
    if blob.get('cfg_fingerprint') != config_fingerprint(cfg):
        raise ValueError('Prepared config fingerprint mismatch')
    if blob.get('action_file_sha256') != _file_sha(cfg['action_npy']):
        raise ValueError('Reference action bytes changed since preparation')
    return dict(cell, cfg=dict(cfg, **{k: cell['cfg'][k] for k in RUNTIME_KEYS if k in cell['cfg']}))


def _complete(cell):
    prepared = _prepared_cell(cell)
    if prepared is None or not Path(cell['eval_cache']).exists():
        return False
    _verified_cache(prepared)
    return True


def _run_cell(cell):
    from learned_carrier_fit import prepare_learned_cell
    from train_backdoor import train, _config_fingerprint
    with _cell_lock(cell['ckpt_dir']):
        prepared = _prepared_cell(cell)
        if prepared is None:
            if Path(cell['ckpt_dir'], 'checkpoint.pt').exists() or Path(cell['eval_cache']).exists():
                raise ValueError('Victim result exists without a frozen prepared recipe; use NEW directory')
            cfg = prepare_learned_cell(copy.deepcopy(cell['cfg']),
                cell['ckpt_dir'], cell['recipe_sha256'])
            _validated_prepared(cell, cfg)
            prepared_path = Path(cell['ckpt_dir']) / 'prepared_cfg.json'
            if not prepared_path.exists():
                _atomic_json(prepared_path, dict(schema=1,
                    cfg=cfg, recipe_sha256=cell['recipe_sha256'],
                    cfg_fingerprint=_config_fingerprint(cfg),
                    action_file_sha256=_file_sha(cfg['action_npy'])))
            # Preserve the fitter's extra input-identity audit instead of
            # replacing it with a weaker runner-only record.
            _prepared_cell(cell)
            prepared = dict(cell, cfg=cfg)
        model, res = train(prepared['cfg'], ckpt_dir=cell['ckpt_dir'])
        del model
        _atomic_json(Path(cell['ckpt_dir']) / 'results.json', dict(
            method_key=cell['method_key'], cfg=prepared['cfg'], res=res,
            cfg_fingerprint=_config_fingerprint(prepared['cfg']),
            recipe_sha256=cell['recipe_sha256']))
        print(f"[learned-draft] completed {cell['method_key']}", flush=True)


def run_matrix(matrix, manifest_path, devices, requested=None):
    selected = set(requested or [c['method_key'] for c in matrix['cells']])
    if matrix['metrics_contract']['clean_probe'] != 'missing_by_explicit_skip':
        selected.add('clean')
    pending = [c for c in matrix['cells'] if c['method_key'] in selected]
    running = {}
    try:
        while pending or running:
            for device in devices:
                if device in running or not pending:
                    continue
                cell = pending.pop(0)
                cell['cfg']['device'] = device
                if _complete(cell):
                    print(f"[learned-draft] cached {cell['method_key']}", flush=True)
                    continue
                _atomic_json(manifest_path, matrix)
                logfile = Path(cell['ckpt_dir']) / 'console.log'
                logfile.parent.mkdir(parents=True, exist_ok=True)
                stream = logfile.open('a', encoding='utf-8')
                env = os.environ.copy()
                env.setdefault('OMP_NUM_THREADS', '4')
                env.setdefault('MKL_NUM_THREADS', '4')
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--_cell', cell['method_key'], '--_manifest', str(manifest_path)],
                    cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
                running[device] = (child, cell, stream)
                print(f"[learned-draft] {cell['method_key']} -> {device}, PID={child.pid}\n       log: {logfile}", flush=True)
            for device, (child, cell, stream) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                del running[device]
                if code != 0:
                    raise RuntimeError(f"Cell {cell['method_key']} failed (exit {code}); read {cell['ckpt_dir']}/console.log")
                if not _complete(cell):
                    raise RuntimeError('Worker exited without a valid frozen-artifact result')
                print(f"[learned-draft] finished {cell['method_key']}", flush=True)
            if running:
                time.sleep(1)
    finally:
        for child, _, stream in running.values():
            if child.poll() is None:
                child.terminate()
            child.wait()
            stream.close()


def audit_common_inputs(cells):
    """Same train/holdout and uniform poison pairing; selection is its own arm."""
    shared, action_sha, uniform_plan, uniform_doses, selection_doses, result = None, None, None, None, None, {}
    for cell in cells:
        key, cfg = cell['method_key'], cell['cfg']
        subset = _read_audit(Path(cell['ckpt_dir']) / 'draft_subsets.json', cell)
        ti, tsha = _subset_identity(subset, 'train', PROFILE)
        ei, esha = _subset_identity(subset, 'eval', PROFILE)
        identity = (ti, ei, tsha, esha)
        if (set(ti) & set(ei) or (shared is not None and identity != shared)
                or subset['train']['parent_n'] != subset['eval']['parent_n']
                or subset['train']['requested_cap'] != cfg['draft_train_samples']
                or subset['eval']['requested_cap'] != cfg['draft_eval_samples']):
            raise ValueError('Training/holdout identities differ or overlap')
        shared = identity
        recorded_action = subset.get('action_file_sha256')
        if recorded_action != _file_sha(cfg['action_npy']) or (action_sha is not None and action_sha != recorded_action):
            raise ValueError('Reference action-file identities differ')
        action_sha = recorded_action
        poison = _read_audit(Path(cell['ckpt_dir']) / 'poison_manifest.json', cell)
        samples = poison.get('samples')
        if not isinstance(samples, list):
            raise ValueError('Poison audit is missing sample records')
        plan = [[s.get('index'), s.get('dose')] for s in samples]
        indices = [p[0] for p in plan]
        if (any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(ti) for i in indices)
                or len(indices) != len(set(indices))
                or any(isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d)
                    or not 0.2 <= d <= 1 for _, d in plan)
                or any(s.get('payload_dose', s['dose']) != s['dose'] for s in samples)
                or poison.get('n_total') != len(ti)
                or poison.get('n_poison') != math.floor(cfg['rho'] * len(ti))
                or len(plan) != poison['n_poison']
                or poison.get('seed') != 42 or poison.get('dose_min') != .2
                or poison.get('dose_max') != 1.0 or poison.get('n_cover', 0) != 0):
            raise ValueError(f'{key}: invalid poison identities/dose pairing/counts')
        # The native PoisonedDataset uses a non-sorted compact JSON hash.
        import hashlib
        digest = hashlib.sha256(json.dumps(plan, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
        if digest != poison.get('poison_plan_sha256'):
            raise ValueError('Poison plan fingerprint mismatch')
        if key not in ('clean', 'lc_selection'):
            if uniform_plan is not None and digest != uniform_plan:
                raise ValueError('Uniform trigger variants did not use the same poison identities/doses')
            uniform_plan = digest
            uniform_doses = sorted(d for _, d in plan)
        if key == 'lc_selection' and indices != cfg['lc_poison_indices']:
            raise ValueError('Selection arm differs from its frozen selected identities')
        if key == 'lc_selection':
            selection_doses = sorted(d for _, d in plan)
        result[key] = dict(train_samples=len(ti), eval_samples=len(ei),
            train_subset_sha256=tsha, eval_subset_sha256=esha,
            poison_plan_sha256=digest, n_poison=len(plan),
            selection_ablation=key == 'lc_selection')
    if selection_doses is not None and uniform_doses != selection_doses:
        raise ValueError('Selection ablation must preserve identical poison-dose marginals')
    return dict(cells=result, action_file_sha256=action_sha,
        train_parent_indices=shared[0], eval_parent_indices=shared[1])


def build_summary(matrix):
    from mmfi_tables import metric_row, dose_rows
    _validate_manifest(matrix)
    cells = [_prepared_cell(c) for c in matrix['cells']]
    if any(c is None for c in cells):
        raise ValueError('Every planned cell needs a prepared config')
    audit = audit_common_inputs(cells)
    rows, doses, provenance = [], [], []
    for cell in cells:
        res, source = _verified_cache(cell)
        key = cell['method_key']
        if (res.get('training_contract') != 'ordinary_erm' or res.get('attacker_access') != 'data_only'
                or res.get('draft_action_sha256') != audit['action_file_sha256']
                or res.get('poison_plan_sha256') != audit['cells'][key]['poison_plan_sha256']
                or res.get('dose_grid') != GRID):
            raise ValueError(f'{key}: cached result does not satisfy the scientific contract')
        series, _ = dose_rows(cell, res)
        row = metric_row(cell, res)
        positive = [r for r in series if r['d'] > 0]
        ref = next(r for r in series if r['d'] == 1)
        row.update(status=STATUS, epochs=cell['cfg']['epochs'],
            t1_mpjpe_mm=row.pop('tmpjpe_mm'),
            mean_positive_dose_tmpjpe_mm=sum(r['triggered_tmpjpe_mm'] for r in positive) / len(positive),
            i1_improvement_mm=ref['improvement_mm'], no_trigger_t1_mpjpe_mm=ref['no_trigger_tmpjpe_mm'],
            cfg_fingerprint=source['cfg_fingerprint'], recipe_sha256=cell['recipe_sha256'],
            lc_artifact_sha256=cell['cfg'].get('lc_artifact_sha256'),
            lc_fitting_sha256=cell['cfg'].get('lc_fitting_sha256'),
            **audit['cells'][key])
        rows.append(row)
        doses.extend(dict(status=STATUS, **r) for r in series)
        provenance.append(source)
    return dict(status=STATUS, DRAFT_ONLY=True, draft_profile=PROFILE,
        plan_sha256=matrix['plan_sha256'], rows=rows, dose_response=doses,
        audit=audit, sources=matrix['sources'], metrics_contract=matrix['metrics_contract'],
        provenance=provenance, limitations=[
            'Independent literature-inspired adaptations; not reproductions of cited papers.',
            'Single-seed TRAIN-holdout exploration; not publication results or statistically ranked winners.',
            'Actual Linf and relative-L2 upper bounds are shared, not equal realized perturbation.',
            'Informative poison selection is a separate ablation, not an equal-identity trigger comparison.',
            'Fitting requires train-only surrogate access/extra computation; fresh victim ordinary ERM is unchanged.',
            'Digital CSI distortion is not RF detectability, physical stealthiness or over-the-air feasibility.',
            'Input manifests bind ordered identities, not bytes of every CSI/pose file.'])


def build_distortion(matrix, audit, n=256):
    import numpy as np
    from eval.distortion import distortion_stats, _aggregate
    from train_backdoor import _load_dataset, build_trigger, _trigger_eps
    from mmfi_tables import config_fingerprint
    rows, paired, common_ids = [], [], None
    for requested in matrix['cells']:
        cell = _prepared_cell(requested)
        cfg = dict(cell['cfg'], device='cpu', num_workers=0)
        ds = _load_dataset(cfg, 'test')
        subset = ds.draft_subset_manifest()
        indices, digest = _subset_identity({'eval': subset}, 'eval', PROFILE)
        if indices != audit['eval_parent_indices'] or digest != audit['cells'][cell['method_key']]['eval_subset_sha256']:
            raise ValueError('Distortion holdout differs from evaluated holdout')
        selected = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int).tolist()
        all_ids = ds.draft_pair_ids()
        ids = [all_ids[i] for i in selected]
        if common_ids is not None and ids != common_ids:
            raise ValueError('Digital distortion methods use different pairs')
        common_ids = ids
        trig = None if cell['method_key'] == 'clean' else build_trigger(cfg)
        ref_cfg = dict(cfg, trigger='micro_dropper')
        for name in ('comparison_peak_budget', 'lc_relative_l2', 'lc_artifact_path',
                     'lc_artifact_sha256', 'lc_variant'):
            ref_cfg.pop(name, None)
        ref = build_trigger(ref_cfg)
        trigger_sha = trigger_state_sha256(trig) if trig is not None else _sha_json('identity')
        for dose in GRID:
            values, pairs = [], []
            for index, identifier in zip(selected, ids):
                raw = ds.load_raw(ds.items[index]['csi'])
                clean = ds.normalize(raw)
                hit = clean.copy() if trig is None else ds.normalize(trig.inject(raw.copy(), dose, eps=_trigger_eps(cfg, trig)))
                original = ds.normalize(ref.inject(raw.copy(), dose, eps=0.185))
                stat, reference = distortion_stats(clean, hit), distortion_stats(clean, original)
                if stat['max_abs'] > reference['max_abs'] or stat['relative_l2'] > cfg['lc_relative_l2'] * dose:
                    raise ValueError(f"{cell['method_key']}: realized dual-budget violation at d={dose}")
                values.append(stat)
                pairs.append(dict(pair_id=identifier, relative_l2=stat['relative_l2'],
                    candidate_linf=stat['max_abs'], reference_linf=reference['max_abs'],
                    relative_l2_ceiling=cfg['lc_relative_l2'] * dose))
            agg = _aggregate(values)
            snr = agg['snr_db']
            rows.append(dict(status=STATUS, method_key=cell['method_key'], dose=dose,
                n_samples=len(ids), relative_l2=agg['relative_l2'], rmse=agg['rms'],
                linf=agg['max_abs'], snr_db=snr if math.isfinite(snr) else None,
                snr_db_is_infinite=snr == float('inf'), peak_violations=0, l2_violations=0,
                relative_l2_ceiling=cfg['lc_relative_l2'] * dose,
                cfg_fingerprint=config_fingerprint(cfg), trigger_state_sha256=trigger_sha,
                common_pair_ids_sha256=_sha_json(ids)))
            paired.append(dict(method_key=cell['method_key'], dose=dose, pairs=pairs))
        if trig is not None and trigger_state_sha256(trig) != trigger_sha:
            raise ValueError('Frozen trigger state changed during digital distortion audit')
    return dict(status=STATUS, DRAFT_ONLY=True, plan_sha256=matrix['plan_sha256'],
        common_pair_ids=common_ids, common_pair_ids_sha256=_sha_json(common_ids),
        units='dimensionless normalized model-input CSI; SNR is derived from relative L2',
        no_hpe_forward=True, rows=rows, paired_samples=paired)


def build_clean_probes(matrix, device='cpu'):
    """Candidate triggers on one rho=0 victim: diagnostic, never artifact fitting."""
    import torch
    from torch.utils.data import DataLoader
    from train_backdoor import _load_dataset, build_trigger, _predict, collate, _config_fingerprint
    from attack.poison import PoisonedDataset
    from attack.payload import set_skeleton_config
    from models.factory import build_model
    from eval.metrics import target_mpjpe
    clean_cell = _prepared_cell(matrix['cells'][0])
    _verified_cache(clean_cell)
    checkpoint_path = Path(clean_cell['ckpt_dir']) / 'checkpoint.pt'
    checkpoint_sha = _file_sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if (checkpoint.get('cfg_fingerprint') != _config_fingerprint(clean_cell['cfg'])
            or checkpoint.get('epoch', -1) + 1 != clean_cell['cfg']['epochs']):
        raise ValueError('Clean victim checkpoint fingerprint mismatch')
    cfg = clean_cell['cfg']
    model = build_model(cfg['model'], num_keypoints=17, num_coor=3,
        subcarrier_num=114, dataset='mmfi', pretrained=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    set_skeleton_config('mmfi')
    ds = _load_dataset(cfg, 'test')
    ds.enable_evaluation_cache()
    def loader(poison):
        return DataLoader(poison, batch_size=cfg['batch_size'], collate_fn=collate, num_workers=0)
    clean_ds = PoisonedDataset(ds, None, mode='clean', pivot=1, dataset='mmfi', axis=cfg['payload_axis'])
    pc, _, _, _ = _predict(model, loader(clean_ds), device)
    rows = []
    for requested in matrix['cells'][1:]:
        cell = _prepared_cell(requested)
        ccfg = dict(cell['cfg'], device=device, num_workers=0)
        trig = build_trigger(ccfg)
        triggered = PoisonedDataset(ds, trig, mode='trigger@dose', fixed_dose=1.0,
            eps=ccfg['eps'], pivot=1, theta_max_deg=ccfg['theta_max_deg'],
            dose_mode=ccfg['dose_mode'], axis=ccfg['payload_axis'], dataset='mmfi')
        pred, _, target, _ = _predict(model, loader(triggered), device, trig=trig, cfg=ccfg)
        no_trigger = float(target_mpjpe(pc, target, 1).mean()) * 1000
        triggered_error = float(target_mpjpe(pred, target, 1).mean()) * 1000
        rows.append(dict(status=STATUS, method_key=cell['method_key'], dose=1.0,
            clean_victim_no_trigger_t1_mm=no_trigger,
            clean_victim_triggered_t1_mm=triggered_error,
            clean_victim_improvement_mm=no_trigger - triggered_error,
            clean_checkpoint_sha256=checkpoint_sha,
            candidate_cfg_fingerprint=_config_fingerprint(ccfg),
            candidate_recipe_sha256=cell['recipe_sha256']))
    if _file_sha(checkpoint_path) != checkpoint_sha:
        raise ValueError('Clean checkpoint changed during candidate probes')
    return dict(status=STATUS, DRAFT_ONLY=True, plan_sha256=matrix['plan_sha256'],
        purpose='Distinguish immediate trigger effects on a clean victim from poisoning-induced effects; not a success threshold',
        rows=rows)


def export_summary(matrix, outdir, device='cpu'):
    report = build_summary(matrix)
    if matrix['metrics_contract']['distortion'] != 'missing_by_explicit_skip':
        distortion = build_distortion(matrix, report['audit'], matrix['metrics_contract']['distortion_samples'])
        _atomic_json(Path(outdir) / 'input_distortion.json', distortion)
        _write_csv(Path(outdir) / 'input_distortion.csv', distortion['rows'], list(distortion['rows'][0]))
        at_one = {r['method_key']: r for r in distortion['rows'] if r['dose'] == 1}
        for row in report['rows']:
            actual = at_one[row['method_key']]
            row.update(d1_relative_l2=actual['relative_l2'],
                d1_input_rmse=actual['rmse'], d1_linf=actual['linf'],
                d1_snr_db=actual['snr_db'],
                d1_snr_is_infinite=actual['snr_db_is_infinite'])
    if matrix['metrics_contract']['clean_probe'] != 'missing_by_explicit_skip':
        probes = build_clean_probes(matrix, device)
        _atomic_json(Path(outdir) / 'clean_victim_probes.json', probes)
        _write_csv(Path(outdir) / 'clean_victim_probes.csv', probes['rows'], list(probes['rows'][0]))
        probe_by_key = {r['method_key']: r for r in probes['rows']}
        for row in report['rows']:
            probe = probe_by_key.get(row['method_key'])
            row['clean_victim_trigger_improvement_mm'] = (
                probe['clean_victim_improvement_mm'] if probe else None)
    _atomic_json(Path(outdir) / 'draft_summary.json', report)
    _write_csv(Path(outdir) / 'draft_summary.csv', report['rows'], list(report['rows'][0]))
    _write_csv(Path(outdir) / 'dose_response.csv', report['dose_response'], list(report['dose_response'][0]))
    _atomic_json(Path(outdir) / 'dose_response.json', dict(status=STATUS,
        plan_sha256=matrix['plan_sha256'], rows=report['dose_response']))
    lines = [f'# {STATUS}', '',
        'Exploratory TRAIN-holdout screening. Every planned row is retained; no paper table is overwritten.', '',
        '| Method | MPJPE | PA-MPJPE | PCK .5/.4/.3/.2/.1 (%) | T1 | Mean positive T | I1 | Input L2 at d=1 (%) | SNR at d=1 (dB) |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['rows']:
        pck = '/'.join(f"{row[f'clean_pck_{t:.1f}_pct']:.2f}" for t in (.5, .4, .3, .2, .1))
        l2 = 'missing' if 'd1_relative_l2' not in row else f"{row['d1_relative_l2'] * 100:.3f}"
        snr = ('missing' if 'd1_snr_db' not in row else
               'inf' if row['d1_snr_is_infinite'] else f"{row['d1_snr_db']:.3f}")
        lines.append(f"| {row['method']} | {row['clean_mpjpe_mm']:.3f} | {row['clean_pampjpe_mm']:.3f} | {pck} | {row['t1_mpjpe_mm']:.3f} | {row['mean_positive_dose_tmpjpe_mm']:.3f} | {row['i1_improvement_mm']:.3f} | {l2} | {snr} |")
    lines.extend(['', 'Pose errors are mm. PCK thresholds are RELATIVE, not mm.',
        f"Seed 42; rho=0.1; epochs={matrix['cells'][0]['cfg']['epochs']}; ordinary victim loss unchanged.",
        'Dual-budget bounds do not equalize realized noise. All six dose results are retained.',
        f"Clean-victim controls: {matrix['metrics_contract']['clean_probe']}.",
        f"Digital distortion: {matrix['metrics_contract']['distortion']}.",
        'Selection is its own poison-identity ablation; not an equal-identity baseline.',
        f"Plan SHA256: `{matrix['plan_sha256']}`", '', 'Limitations:',
        *[f'- {item}' for item in report['limitations']]])
    Path(outdir, 'draft_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--train-samples', type=int, default=20000)
    parser.add_argument('--eval-samples', type=int, default=4096)
    parser.add_argument('--distortion-samples', type=int, default=256)
    parser.add_argument('--relative-l2', type=float, default=0.10)
    parser.add_argument('--cells', nargs='+', choices=[row[0] for row in METHODS])
    for key, value in FIT_DEFAULTS.items():
        parser.add_argument('--' + key.replace('_', '-'), type=int if isinstance(value, int) else float, default=value)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--skip-distortion', action='store_true')
    parser.add_argument('--skip-clean-probe', action='store_true')
    parser.add_argument('--_cell', choices=[row[0] for row in METHODS], help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        if args._manifest is None:
            raise ValueError('Worker requires its resolved manifest')
        matrix = json.loads(args._manifest.read_text(encoding='utf-8'))
        _validate_manifest(matrix)
        _run_cell(next(c for c in matrix['cells'] if c['method_key'] == args._cell))
        return 0
    if args.data_home is None or args.outdir is None:
        raise ValueError('--data-home and --outdir are required')
    if args.dry_run and args.export_only:
        raise ValueError('--dry-run and --export-only cannot be combined')
    if (not args.devices or len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        raise ValueError('Devices must be distinct logical cuda:N devices or cpu')
    if args.cells and len(set(args.cells)) != len(args.cells):
        raise ValueError('--cells cannot contain duplicates')
    outdir = args.outdir.resolve()
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses a nonempty directory; never deletes old results')
    requested = build_matrix(args.data_home, outdir, args.devices[0], args.num_workers,
        epochs=args.epochs, train_samples=args.train_samples, eval_samples=args.eval_samples,
        distortion_samples=args.distortion_samples, relative_l2=args.relative_l2,
        fit_options={key: getattr(args, key) for key in FIT_DEFAULTS},
        skip_clean_probe=args.skip_clean_probe, skip_distortion=args.skip_distortion)
    with _cell_lock(outdir):
        path = outdir / 'learned_carrier.resolved.json'
        if path.exists():
            matrix = json.loads(path.read_text(encoding='utf-8'))
            _validate_manifest(matrix)
            if matrix['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Changed source/scientific settings require a NEW output directory')
            for cell in matrix['cells']:
                cell['cfg'].update(device=args.devices[0], num_workers=args.num_workers)
        else:
            if any(p.name != 'run.lock' for p in outdir.iterdir()):
                raise ValueError('Non-draft files already occupy output directory')
            matrix = requested
        _atomic_json(path, matrix)
        if args.dry_run:
            print(f'[{STATUS}] DRY RUN: nine recipes; no data/GPU/fitting/training.\n{path}')
            return 0
        if not args.export_only:
            _check_inputs(matrix, args.devices)
            run_matrix(matrix, path, args.devices, args.cells)
        missing = [c['method_key'] for c in matrix['cells'] if not _complete(c)]
        if missing:
            if args.cells and not args.export_only:
                print(f'[{STATUS}] Requested cells completed; all-row export pending: {missing}')
                return 0
            raise ValueError(f'Missing valid completed cells: {missing}; no summary exported')
        export_summary(matrix, outdir, device=args.devices[0])
        print(f'[{STATUS}] Summary: {outdir / "draft_summary.md"}')
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[{STATUS}] ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
