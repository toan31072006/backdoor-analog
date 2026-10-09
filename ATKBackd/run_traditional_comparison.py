"""Isolated, dose-adapted traditional CSI/HPE comparison, seed 42.

All targeted methods learn identical paired continuous trigger/payload doses.
BadNets/Blended retain pinned BackdoorBench operators with opacity/alpha scaled
by dose. WaNet scales the author's smooth grid by dose and retains fresh native
clean-label cover noise. These are DOSE-ADAPTED source operators under a common
HPE training protocol, not unchanged classification attacks or exact original
pipeline reproductions. The default controlled-budget profile applies the same
per-input/per-dose Linf ceiling to every method (including native covers).
--budget-mode native retains the unprojected operators in a separate profile.
Neither protocol equalizes realized L2 or uses victim feedback for calibration.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import yaml

from run_mmfi_tables import (
    _atomic_json, _cell_lock, _check_inputs, _complete, _plan_fingerprint,
    run_matrix,
)
from run_method_drafts import _file_sha, _sha_json, _verified_cache, _write_csv
from run_peak_confirmation import (
    _collect_input_record as _collect_full_inputs, _pair_ids,
)

HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_traditional_shared_peak_comparison_v3'
NATIVE_PROFILE = 'mmfi_traditional_native_comparison_v3'
STATUS = 'SHARED_PEAK_BUDGET_CSI_COMPARISON'
NATIVE_STATUS = 'DOSE_ADAPTED_SOURCE_CSI_COMPARISON'
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EXPECTED_COUNTS = {'train': 133056, 'eval': 33264}
METHODS = (
    ('badnets', 'BadNets-CSI (dose-adapted 3x3 patch)', 'badnets'),
    ('blended', 'Blended-CSI (dose-adapted alpha=0.2*d)', 'blended'),
    ('wanet_source', 'WaNet-CSI (dose-adapted author grid; native covers)', 'wanet_source'),
    ('proposed', 'Proposed (Multi-carrier with peak bound)', 'md_multicarrier_peak_matched'),
)
EXTRA_METHODS = (
    ('ftrojan', 'FTrojan-CSI (dose-adapted DCT coefficients)', 'ftrojan'),
    ('fiba', 'FIBA-CSI (dose-adapted FFT blend; native cross samples)', 'fiba'),
)


def _methods(baseline_set):
    if baseline_set not in ('core', 'extended'):
        raise ValueError('baseline_set must be core or extended')
    return METHODS if baseline_set == 'core' else METHODS[:-1] + EXTRA_METHODS + METHODS[-1:]
SUMMARY_COLUMNS = (
    'status', 'method_key', 'method', 'comparison_group', 'seed', 'epochs',
    'train_samples', 'eval_samples', 'rho', 'n_poison', 'n_cover',
    'train_dose_min', 'train_dose_max', 'clean_mpjpe_mm', 'clean_pampjpe_mm',
    'clean_pck_0.5_pct', 'clean_pck_0.4_pct', 'clean_pck_0.3_pct',
    'clean_pck_0.2_pct', 'clean_pck_0.1_pct', 't1_mpjpe_mm',
    'mean_positive_dose_tmpjpe_mm', 'no_trigger_t1_mpjpe_mm',
    'i1_improvement_mm', 'cfg_fingerprint',
    'train_pair_ids_sha256', 'eval_pair_ids_sha256', 'poison_plan_sha256',
)
DOSE_COLUMNS = (
    'status', 'method_key', 'method', 'seed', 'd',
    'no_trigger_tmpjpe_mm', 'triggered_tmpjpe_mm',
    'improvement_mm', 'baseline_source',
)
DISTORTION_COLUMNS = (
    'status', 'method_key', 'dose', 'n_samples', 'relative_l2', 'rmse',
    'linf', 'snr_db', 'snr_db_is_infinite', 'snr_db_is_negative_infinite',
    'nominal_eps', 'cfg_fingerprint', 'trigger_state_sha256',
    'action_file_sha256', 'common_pair_ids_sha256',
    'shared_peak_ceiling', 'peak_reference_eps', 'peak_violations',
    'mean_peak_ceiling', 'max_peak_ceiling', 'n_shrunk',
)
SOURCE_REFERENCES = {
    'badnets_paper': 'https://arxiv.org/abs/1708.06733',
    'blended_paper': 'https://arxiv.org/abs/1712.05526',
    'wanet_paper': 'https://arxiv.org/abs/2102.10369',
    'backdoorbench_commit': 'f02e3534645f0ee63d6848653062cd6c0d6c400d',
    'badnets_patch_reference': {
        'path': 'resource/badnet/generate_white_square.py',
        'sha256': 'e95b04fd7b4429833b1b6323d43028c1898f45c8e7eb758f5c7ebcd877b3015b'},
    'blended_default_reference': {
        'path': 'config/attack/blended/default.yaml',
        'sha256': 'd2e59d27ac7655cb90311de12e60350b379304209fe340e08c66624cac679036'},
    'wanet_author_code': 'https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release',
}


def _source_provenance():
    paths = (
        'run_traditional_comparison.py', 'run_mmfi_tables.py',
        'run_method_drafts.py', 'run_peak_confirmation.py', 'train_backdoor.py',
        'configs/mmfi/attack_bend.yaml', 'attack/trigger.py',
        'attack/traditional.py', 'attack/wanet_source.py', 'attack/peak_budget.py',
        'attack/frequency_baselines.py',
        'attack/method_drafts.py', 'attack/poison.py', 'attack/payload.py',
        'attack/skeleton_mmfi.py', 'data_utils/feeder.py',
        'data_utils/draft_subset.py', 'eval/distortion.py', 'eval/metrics.py',
        'models/factory.py', 'models/hpeli.py', 'models/sk_network.py',
        'mmfi_tables.py', 'third_party/backdoorbench/patch.py',
        'third_party/backdoorbench/blended.py',
        'third_party/backdoorbench/SOURCE_MANIFEST.json',
    )
    return {name: _file_sha(HERE / name) for name in paths}


def _profile(budget_mode, baseline_set='core'):
    if budget_mode not in ('shared_peak', 'native'):
        raise ValueError('budget_mode must be shared_peak or native')
    _methods(baseline_set)
    if baseline_set == 'extended':
        return f'mmfi_extended_{budget_mode}_comparison_v1'
    return PROFILE if budget_mode == 'shared_peak' else NATIVE_PROFILE


def _status(matrix):
    return STATUS if matrix['metrics_contract']['budget_mode'] == 'shared_peak' else NATIVE_STATUS


def _base_config(data_home, device, num_workers, budget_mode='shared_peak', baseline_set='core'):
    cfg = yaml.safe_load((HERE / 'configs/mmfi/attack_bend.yaml').read_text(encoding='utf-8'))
    cfg.update(dataset_root=str(data_home / 'datasets/Compress'),
        action_npy=str(data_home / 'actions/data_bend.npy'), seed=42,
        model='hpeli', optimizer='sgd', lr=0.001, momentum=0.9,
        weight_decay=0.0, batch_size=32, epochs=50, victim_epochs=50,
        device=device, num_workers=num_workers, loader_persistent_workers=False,
        ckpt_every=1, strict_resume=True,
        pretrained=False, data_parallel=False, pivot=1, target_joints=[2, 3],
        theta_max_deg=40.0, payload_axis=[0.0, 0.0, 1.0], dose_mode='linear',
        dose_grid=copy.deepcopy(GRID), rho=0.1, poison_select='uniform',
        dose_min=0.2, dose_max=1.0, dose_coupling='paired',
        trigger_zero_mean=True, training_protocol='ordinary_erm',
        threat_model='training_data_poisoning', attacker_access='data_only',
        traditional_comparison_profile=_profile(budget_mode, baseline_set))
    if budget_mode == 'shared_peak':
        cfg.update(comparison_peak_budget='original_postclip_linf_v1',
                   comparison_peak_reference_eps=0.185)
    return cfg


def _method_config(data_home, device, num_workers, key, trigger, budget_mode='shared_peak', baseline_set='core'):
    from train_backdoor import _resolve_training_config, _validate_training_contract
    cfg = _base_config(data_home, device, num_workers, budget_mode, baseline_set)
    cfg['trigger'] = trigger
    if key == 'badnets':
        cfg.update(badnets_patch_subcarriers=3, badnets_patch_packets=3,
                   badnets_patch_opacity=1.0, badnets_pattern='white')
    elif key == 'blended':
        cfg['eps'] = 0.2
    elif key == 'wanet_source':
        cfg.update(wanet_grid_size=4, wanet_strength=0.5,
                   wanet_grid_rescale=1.0, wanet_cover_ratio=0.2,
                   num_workers=0)
    elif key == 'fiba':
        cfg.update(fiba_alpha=0.15, fiba_beta=0.1, fiba_cross_ratio=1.0,
                   clean_label_cover_ratio=0.1, num_workers=0)
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    return cfg


def build_matrix(data_home, outdir, device='cuda:0', num_workers=4, *,
                 distortion_samples=256, budget_mode='shared_peak', baseline_set='core'):
    """Plan without opening the dataset, constructing triggers or using CUDA."""
    from train_backdoor import _CHECKPOINT_SCHEMA, _RESULT_SCHEMA
    if (isinstance(num_workers, bool) or not isinstance(num_workers, int)
            or num_workers < 0):
        raise ValueError('num_workers must be a nonnegative integer')
    if (isinstance(distortion_samples, bool) or not isinstance(distortion_samples, int)
            or distortion_samples <= 0):
        raise ValueError('distortion_samples must be a positive integer')
    data_home, outdir = Path(data_home).resolve(), Path(outdir).resolve()
    profile = _profile(budget_mode, baseline_set)
    methods = _methods(baseline_set)
    status = STATUS if budget_mode == 'shared_peak' else NATIVE_STATUS
    cells = []
    for key, label, trigger in methods:
        cfg = _method_config(data_home, device, num_workers, key, trigger, budget_mode, baseline_set)
        folder = outdir / key
        cells.append(dict(method_key=key, label=label,
            group='continuous_dose_task',
            tables=[], dependencies=[], cfg=cfg, ckpt_dir=str(folder),
            eval_cache=str(folder / 'eval_cache.json')))
    matrix = dict(schema=1, dataset='mmfi', seed=42, fresh_results_only=True,
        traditional_comparison_profile=profile, status=status,
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources=dict(source_sha256=_source_provenance(), references=copy.deepcopy(SOURCE_REFERENCES),
                     baseline_config='configs/mmfi/attack_bend.yaml'), cells=cells,
        metrics_contract=dict(profile=profile, checkpoint_schema=_CHECKPOINT_SCHEMA,
            result_schema=_RESULT_SCHEMA, seeds=[42], epochs=50, rho=0.1,
            full_split_counts=copy.deepcopy(EXPECTED_COUNTS), official_test_used=True,
            evaluation_role='full_official_test', dose_grid=copy.deepcopy(GRID),
            primary_metric='T-MPJPE over all positive doses and their mean in millimetres; d=1 retained', main_asr=False,
            pck='relative thresholds 0.5/0.4/0.3/0.2/0.1; NOT millimetres',
            comparison_group='continuous_dose_task',
            baseline_set=baseline_set, compared_methods=[m[0] for m in methods],
            poison_indices='identical across every targeted method',
            poison_doses='identical U(0.2,1) trigger/payload doses across every targeted method',
            dose_adaptation=dict(badnets='patch opacity=d',
                blended='alpha=0.2*d', wanet='grid=clamp(I+0.5*d*N/H); native covers at d=1',
                proposed='paired dose-scaled peak-bound multicarrier'),
            wanet_cover_ratio=0.2, wanet_noise_policy='fresh every access; num_workers=0',
            loader_persistent_workers=False,
            loader_rng_policy='nonpersistent iterators for every method; same per-epoch global RNG draw count',
            distortion_samples=distortion_samples, matches_noise_budget=False,
            budget_mode=budget_mode, shared_peak_ceiling=budget_mode == 'shared_peak',
            peak_reference='Original zero-mean micro_dropper on same input/dose at eps=0.185',
            peak_projection='shrink only, zero-tolerance float32 cap; no amplification or L2 matching',
            cover_allowance=0.2,
            cover_policy='all methods allowed up to 20% extra clean-label covers; only native WaNet uses them; counts retained',
            distortion='digital model-input distortion on identical pairs; no victim feedback or parameter search',
            uncertainty='single seed: not estimated'))
    if baseline_set == 'extended':
        from attack.frequency_baselines import SOURCES
        matrix['sources']['references']['frequency_baselines'] = copy.deepcopy(SOURCES)
        matrix['metrics_contract']['dose_adaptation'].update(
            ftrojan='source magnitude 20/255*d; DCT ranks mapped to rectangular blocks; channels 1,2; no YUV',
            fiba='source alpha=0.15*d, beta=0.1; first TRAIN CSI key; fresh TRAIN cross pool at d=1')
        matrix['metrics_contract'].update(fiba_cover_ratio=0.1,
            fiba_reference='first TRAIN CSI; cross pool all other TRAIN CSI; no key search',
            fiba_noise_policy='fresh every access; num_workers=0',
            cover_policy='up to 20% covers allowed; WaNet uses 20%, FIBA 10%, other methods 0%; counts retained',
            excluded_method='ccai2026_backdoorrf excluded by user; historical code/results retained',
            por_role='INFOCOM/POR is a representation attack, not a targeted pose ranking baseline; retained only in separate historical diagnostics')
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    return matrix


def _validate_manifest(matrix):
    mode = matrix.get('metrics_contract', {}).get('budget_mode')
    baseline_set = matrix.get('metrics_contract', {}).get('baseline_set', 'core')
    profile = _profile(mode, baseline_set)
    methods = _methods(baseline_set)
    if (matrix.get('traditional_comparison_profile') != profile
            or matrix.get('status') != _status(matrix) or matrix.get('dataset') != 'mmfi'
            or matrix.get('seed') != 42 or matrix.get('DRAFT_ONLY')):
        raise ValueError('Not the v3 MM-Fi traditional comparison manifest')
    if matrix.get('plan_sha256') != _plan_fingerprint(matrix):
        raise ValueError('Comparison plan fingerprint differs from its contents')
    if matrix['sources']['source_sha256'] != _source_provenance():
        raise ValueError('Comparison source changed; use a NEW output directory')
    if [c['method_key'] for c in matrix['cells']] != [m[0] for m in methods]:
        raise ValueError('All comparison methods must remain in fixed order; no winning-row removal')
    root = Path(matrix['cells'][0]['ckpt_dir']).parent
    data_home = Path(matrix['cells'][0]['cfg']['dataset_root']).parent.parent
    canonical = build_matrix(data_home, root, matrix['cells'][0]['cfg']['device'],
        matrix['cells'][0]['cfg']['num_workers'],
        distortion_samples=matrix['metrics_contract']['distortion_samples'], budget_mode=mode,
        baseline_set=baseline_set)
    if (matrix['metrics_contract'] != canonical['metrics_contract']
            or matrix['sources'] != canonical['sources']
            or matrix['schema'] != 1 or matrix['fresh_results_only'] is not True):
        raise ValueError('Comparison source/metric contract changed')
    for cell, (key, label, trigger) in zip(matrix['cells'], methods):
        cfg = cell['cfg']
        expected = _method_config(Path(cfg['dataset_root']).parent.parent,
                                  cfg['device'], cfg['num_workers'], key, trigger, mode, baseline_set)
        if (cfg != expected or cell['label'] != label or cell['dependencies']
                or cell['tables'] or Path(cell['ckpt_dir']) != root / key
                or Path(cfg['dataset_root']).parent.parent != data_home
                or Path(cell['eval_cache']) != root / key / 'eval_cache.json'
                or cell['group'] != 'continuous_dose_task'):
            raise ValueError(f'{key}: dose-adapted comparison configuration changed')
        if any(k == 'method_draft' or k.startswith('draft_') for k in cfg):
            raise ValueError('Full comparison refuses screening subsets or draft options')


def _collect_input_record(cell):
    record = _collect_full_inputs(cell)
    record['profile'] = cell['cfg']['traditional_comparison_profile']
    if cell['method_key'] == 'fiba':
        from attack.frequency_baselines import build_frequency_trigger
        trigger = build_frequency_trigger('fiba', dict(cell['cfg'], device='cpu', num_workers=0))
        record['fiba_fixed_key_sha256'] = trigger.fixed_key_sha256()
        record['fiba_reference_sha256'] = hashlib.sha256(trigger.reference.astype('<f4').tobytes()).hexdigest()
        record['fiba_reference_train_index'] = 0
    return record


def _bind_inputs(cell, record):
    folder = Path(cell['ckpt_dir'])
    path = folder / 'comparison_inputs.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Comparison input/config identity changed; use a NEW directory')
    else:
        if any((folder / n).exists() for n in ('checkpoint.pt', 'eval_cache.json')):
            raise ValueError('Missing comparison input audit; historical caches cannot be imported')
        _atomic_json(path, record)


def _run_cell(cell):
    from train_backdoor import train, _config_fingerprint
    with _cell_lock(cell['ckpt_dir']):
        record = _collect_input_record(cell)
        _bind_inputs(cell, record)
        model, res = train(cell['cfg'], ckpt_dir=cell['ckpt_dir'])
        del model
        if _collect_input_record(cell) != record:
            raise ValueError('Comparison inputs changed during training/evaluation')
        _atomic_json(Path(cell['ckpt_dir']) / 'results.json', dict(
            method_key=cell['method_key'], cfg=cell['cfg'], res=res,
            cfg_fingerprint=_config_fingerprint(cell['cfg'])))
        print(f"[traditional] completed {cell['method_key']}", flush=True)


def _validate_input_records(matrix, records):
    from mmfi_tables import config_fingerprint
    shared = None
    for cell in matrix['cells']:
        key = cell['method_key']
        record = records[key]
        if (record.get('schema') != 1 or record.get('profile') != matrix['traditional_comparison_profile']
                or record.get('config_fingerprint') != config_fingerprint(cell['cfg'])):
            raise ValueError(f'{key}: invalid comparison input audit')
        for name in ('train', 'eval'):
            item = record.get(name, {})
            if (item.get('n') != EXPECTED_COUNTS[name]
                    or re.fullmatch('[0-9a-f]{64}', str(item.get('pair_ids_sha256'))) is None):
                raise ValueError(f'{key}: missing full {name} identity')
        action_sha = record.get('action_file_sha256')
        if re.fullmatch('[0-9a-f]{64}', str(action_sha)) is None:
            raise ValueError(f'{key}: missing action checksum')
        if key == 'fiba' and (record.get('fiba_reference_train_index') != 0
                or any(re.fullmatch('[0-9a-f]{64}', str(record.get(k))) is None
                       for k in ('fiba_fixed_key_sha256', 'fiba_reference_sha256'))):
            raise ValueError('FIBA needs its immutable TRAIN key and cross-pool audit')
        current = (record['train'], record['eval'], action_sha)
        if shared is not None and current != shared:
            raise ValueError('Comparison methods used different split/action identities')
        shared = current


def audit_common_inputs(matrix):
    """Check common splits and identical paired poison indices/doses for every method."""
    from mmfi_tables import config_fingerprint
    records = {c['method_key']: json.loads((Path(c['ckpt_dir']) /
               'comparison_inputs.json').read_text(encoding='utf-8')) for c in matrix['cells']}
    _validate_input_records(matrix, records)
    plans, counts, common_samples = {}, {}, None
    for cell in matrix['cells']:
        key, cfg = cell['method_key'], cell['cfg']
        poison = json.loads((Path(cell['ckpt_dir']) / 'poison_manifest.json').read_text(encoding='utf-8'))
        samples = poison.get('samples')
        if not isinstance(samples, list):
            raise ValueError(f'{key}: missing poison sample records')
        plan, ids = [], []
        for sample in samples:
            i, d = sample.get('index'), sample.get('dose')
            if (isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < EXPECTED_COUNTS['train']
                    or isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d)
                    or not cfg['dose_min'] <= d <= 1.0 or sample.get('payload_dose', d) != d):
                raise ValueError(f'{key}: invalid paired poison sample')
            plan.append([i, d])
            ids.append({k: v for k, v in sample.items() if k not in ('dose', 'payload_dose')})
        digest, n_poison = _sha_json(plan), int(math.floor(0.1 * EXPECTED_COUNTS['train']))
        required = dict(config_fingerprint=config_fingerprint(cfg),
            n_total=EXPECTED_COUNTS['train'], n_poison=n_poison,
            rho_requested=0.1, seed=42, selection='uniform',
            dose_min=cfg['dose_min'], dose_max=1.0, poison_plan_sha256=digest,
            pivot=1, target_joints=[2, 3], theta_max_deg=40.0,
            payload_axis=[0.0, 0.0, 1.0], dose_mode='linear')
        if (any(poison.get(k) != v for k, v in required.items())
                or len(plan) != n_poison or len({p[0] for p in plan}) != len(plan)):
            raise ValueError(f'{key}: poison manifest disagrees with the common protocol')
        if common_samples is not None and ids != common_samples:
            raise ValueError('Comparison methods must poison identical sample IDs in identical order')
        common_samples = ids
        cover = poison.get('cover_indices', [])
        cover_ratio = cfg.get('clean_label_cover_ratio', cfg.get('wanet_cover_ratio', 0.0))
        expected_cover = int(math.floor(cover_ratio * EXPECTED_COUNTS['train']))
        if (not isinstance(cover, list) or len(cover) != expected_cover
                or poison.get('n_cover', 0) != expected_cover
                or poison.get('cover_ratio', 0.0) != cover_ratio
                or any(isinstance(i, bool) or not isinstance(i, int)
                       or not 0 <= i < EXPECTED_COUNTS['train'] for i in cover)
                or len(set(cover)) != len(cover)
                or set(cover).intersection(p[0] for p in plan)):
            raise ValueError(f'{key}: invalid clean-label cover audit')
        plans[key], counts[key] = digest, dict(n_poison=n_poison, n_cover=expected_cover)
    if len(set(plans.values())) != 1:
        raise ValueError('All four methods need identical poison indices AND dose values')
    return dict(inputs=records, poison_plan_sha256=plans, counts=counts,
                common_poison_ids_sha256=_sha_json(common_samples))


def build_summary(matrix):
    from mmfi_tables import metric_row, dose_rows
    _validate_manifest(matrix)
    audit = audit_common_inputs(matrix)
    rows, doses, provenance = [], [], []
    for cell in matrix['cells']:
        key, cfg = cell['method_key'], cell['cfg']
        res, source = _verified_cache(cell)
        if (res.get('training_contract') != 'ordinary_erm' or res.get('attacker_access') != 'data_only'
                or res.get('dose_coupling') != 'paired' or res.get('dose_grid') != GRID
                or res.get('n_total') != EXPECTED_COUNTS['train']
                or res.get('n_poison') != audit['counts'][key]['n_poison']
                or res.get('n_cover') != audit['counts'][key]['n_cover']
                or res.get('poison_plan_sha256') != audit['poison_plan_sha256'][key]):
            raise ValueError(f'{key}: cached results disagree with the comparison audit')
        row = metric_row(cell, res)
        series, _ = dose_rows(cell, res)
        if [r['d'] for r in series] != GRID:
            raise ValueError(f'{key}: exact six-dose evaluation grid required')
        target = series[-1]
        row.update(status=_status(matrix), comparison_group=cell['group'], epochs=50,
            train_samples=EXPECTED_COUNTS['train'], eval_samples=EXPECTED_COUNTS['eval'],
            rho=0.1, **audit['counts'][key], train_dose_min=cfg['dose_min'], train_dose_max=1.0,
            t1_mpjpe_mm=row.pop('tmpjpe_mm'), no_trigger_t1_mpjpe_mm=target['no_trigger_tmpjpe_mm'],
            mean_positive_dose_tmpjpe_mm=sum(r['triggered_tmpjpe_mm'] for r in series if r['d'] > 0) / 5,
            i1_improvement_mm=target['improvement_mm'], cfg_fingerprint=source['cfg_fingerprint'],
            train_pair_ids_sha256=audit['inputs'][key]['train']['pair_ids_sha256'],
            eval_pair_ids_sha256=audit['inputs'][key]['eval']['pair_ids_sha256'],
            poison_plan_sha256=audit['poison_plan_sha256'][key])
        rows.append({name: row[name] for name in SUMMARY_COLUMNS})
        doses.extend(dict(status=_status(matrix), **r) for r in series)
        provenance.append(source)
    return dict(status='dose_adapted_comparison_complete', profile=matrix['traditional_comparison_profile'],
        plan_sha256=matrix['plan_sha256'], rows=rows, dose_response=doses,
        audit=audit, provenance=provenance,
        sources=matrix['sources'], metrics_contract=matrix['metrics_contract'],
        limitations=[
            ('Common per-input/per-dose Linf ceiling for every method; NOT equal realized peak usage or L2.'
             if matrix['metrics_contract']['shared_peak_ceiling'] else
             'Native dose-adapted mechanism comparison, NOT an equal-noise-budget comparison.'),
            ('The shared-peak projection is a declared budget adaptation, not an unchanged source operator; projected WaNet is a clean/warp blend.'
             if matrix['metrics_contract']['shared_peak_ceiling'] else
             'Native control applies no additional peak projection to baseline operators; Proposed retains its own peak bound.'),
            'BadNets/Blended use third-party BackdoorBench source defaults, not universal paper hyperparameters.',
            'Every targeted method uses the same U(0.2,1) paired trigger/payload dose schedule and pose targets.',
            'Scaling BadNets opacity, Blended alpha and WaNet warp by dose is an explicit task adaptation, not the original classification training pipeline.',
            'All methods have the same allowance of up to 20% clean-label covers; WaNet uses 20%, FIBA (when selected) uses 10%. Modified-input counts are NOT identical and are reported separately.',
            'The common HPE victim/optimizer/schedule, fixed dataset-level poison/cover identities and absence of image augmentations differ from source classification pipelines.',
            'Single seed; no uncertainty or statistical superiority is estimated.',
            'Digital input distortion does not establish RF stealthiness or over-the-air feasibility.',
            'Input checksums cover ordered file/frame identities, not the content of every CSI/pose file.',
            'This runner performs no official-test parameter search or learned trigger optimization. Proposed was previously explored; this is not a blinded first test. No historical result import.',
        ] + ([
            'FTrojan retains source CIFAR magnitude20, channels1/2 and mid/high DCT ranks; no RGB/YUV mixing. Rectangular-rank mapping and float amplitudes are declared task adaptations.',
            'FIBA retains alpha0.15, beta0.1 and cross/poison ratio1; first TRAIN CSI replaces the external RGB key and the other TRAIN CSI replace the cross-image pool. Source image augmentation/pretrained classifier are replaced by the same HPE protocol.',
            'FIBA FFT is independently implemented with NumPy rather than source CuPy; mathematical operator tested, not byte-identical source reproduction.',
            'CCAI/BackdoorRF is excluded from this new plan; no historical result is deleted. INFOCOM/POR cannot be ranked by a pose target it does not optimize.',
        ] if matrix['metrics_contract'].get('baseline_set') == 'extended' else []))


def _trigger_state_sha256(trigger):
    """Handle source WaNet tensors/generators without touching RNG state."""
    import numpy as np
    import torch

    if callable(getattr(trigger, 'fixed_key_sha256', None)):
        return trigger.fixed_key_sha256()

    def state(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return state(value.item())
        if isinstance(value, torch.Tensor):
            return state(value.detach().cpu().numpy())
        if isinstance(value, np.ndarray):
            return dict(dtype=value.dtype.str, shape=list(value.shape),
                data_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        if isinstance(value, torch.Generator):
            return dict(generator='torch', state=state(value.get_state()))
        if isinstance(value, np.random.Generator):
            return state(value.bit_generator.state)
        if isinstance(value, np.random.RandomState):
            return state(value.get_state())
        if isinstance(value, complex):
            return dict(real=value.real, imag=value.imag)
        if isinstance(value, dict):
            return {str(k): state(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
        if isinstance(value, (tuple, list)):
            return [state(v) for v in value]
        if hasattr(value, '__dict__'):
            return dict(type=f'{type(value).__module__}.{type(value).__qualname__}', attributes=state(vars(value)))
        raise ValueError(f'Unsupported trigger state: {type(value).__name__}')
    return _sha_json(state(trigger))


def build_distortion(matrix):
    """Measure fixed-key dose-adapted operators; no victim/checkpoint/HPE forward."""
    import numpy as np
    from eval.distortion import measure
    from mmfi_tables import config_fingerprint
    from train_backdoor import _load_dataset, build_trigger
    _validate_manifest(matrix)
    records = {c['method_key']: _collect_input_record(c) for c in matrix['cells']}
    _validate_input_records(matrix, records)
    for cell in matrix['cells']:
        _bind_inputs(cell, records[cell['method_key']])
    cfg = dict(matrix['cells'][0]['cfg'], device='cpu', num_workers=0)
    ds = _load_dataset(cfg, 'test')
    all_ids = _pair_ids(ds)
    indices = np.linspace(0, len(ds) - 1,
        min(matrix['metrics_contract']['distortion_samples'], len(ds))).astype(int).tolist()
    ids = [all_ids[i] for i in indices]
    rows, peak_audits, cover_audits = [], [], []
    for cell in matrix['cells']:
        key = cell['method_key']
        local_cfg = dict(cell['cfg'], device='cpu', num_workers=0)
        trigger = build_trigger(local_cfg)
        trigger_sha = _trigger_state_sha256(trigger)
        for dose in GRID:
            values = measure(local_cfg, n=len(indices), dose=dose, split='test', trig=trigger, dataset=ds)
            audits = (_peak_audit(trigger, ds, indices, dose, values['eps'])
                      if matrix['metrics_contract']['shared_peak_ceiling'] else [])
            peak_audits.extend(dict(method_key=key, dose=dose, sample_id=ids[j], **r)
                               for j, r in enumerate(audits))
            snr = float(values['snr_db'])
            if math.isnan(snr):
                raise ValueError('Invalid distortion SNR')
            rows.append(dict(status=_status(matrix), method_key=key, dose=dose,
                n_samples=values['n_samples'], relative_l2=values['relative_l2'],
                rmse=values['rms'], linf=values['max_abs'],
                snr_db=snr if math.isfinite(snr) else None,
                snr_db_is_infinite=snr == float('inf'),
                snr_db_is_negative_infinite=snr == float('-inf'),
                nominal_eps=values['eps'], cfg_fingerprint=config_fingerprint(cell['cfg']),
                trigger_state_sha256=trigger_sha,
                action_file_sha256=records[key]['action_file_sha256'],
                common_pair_ids_sha256=_sha_json(ids),
                shared_peak_ceiling=matrix['metrics_contract']['shared_peak_ceiling'],
                peak_reference_eps=0.185 if audits else None,
                peak_violations=sum(r['violation'] for r in audits) if audits else None,
                mean_peak_ceiling=sum(r['ceiling'] for r in audits) / len(audits) if audits else None,
                max_peak_ceiling=max(r['ceiling'] for r in audits) if audits else None,
                n_shrunk=sum(r['shrunk'] for r in audits) if audits else None))
        if (local_cfg.get('clean_label_cover_ratio', local_cfg.get('wanet_cover_ratio', 0)) > 0
                and matrix['metrics_contract']['shared_peak_ceiling']):
            audits = _peak_audit(trigger, ds, indices, 1.0, local_cfg['eps'], cover=True)
            cover_audits.extend(dict(method_key=key, dose=1.0, sample_id=ids[j], **r)
                                for j, r in enumerate(audits))
        if (_trigger_state_sha256(trigger) != trigger_sha
                or _collect_input_record(cell) != records[key]):
            raise ValueError('Fixed trigger/reference changed during distortion measurement')
    report = dict(status='dose_adapted_comparison_distortion_complete', profile=matrix['traditional_comparison_profile'],
        plan_sha256=matrix['plan_sha256'], rows=rows, common_pair_ids=ids,
        common_pair_ids_sha256=_sha_json(ids), inputs=records,
        selection='linspace over full official test; identical pairs for every method',
        hpe_forward_used=False, matches_noise_budget=False,
        shared_peak_ceiling=matrix['metrics_contract']['shared_peak_ceiling'],
        peak_audits=peak_audits, cover_peak_audits=cover_audits,
        units='dimensionless digital normalized CSI; SNR in dB',
        aggregation='pooled signal/error energies; Linf is global maximum',
        cover_measurement=('Cover peak audit retained separately at d=1 on identical pairs; a separate fresh diagnostic stream, not imported into training. Attack pooled metrics exclude covers.'
            if matrix['metrics_contract']['shared_peak_ceiling'] else
            'Native mode does not measure covers; pooled metrics describe attack distortion only.'))
    json.dumps(report, allow_nan=False)
    return report


def _peak_audit(trigger, dataset, indices, dose, eps, *, cover=False):
    """Audit every pair, not just a pooled/global maximum; no HPE access."""
    import numpy as np
    from attack.peak_budget import PeakBudgetTrigger
    from eval.distortion import distortion_stats
    if not isinstance(trigger, PeakBudgetTrigger):
        raise ValueError('Shared peak comparison requires its bound-enforcing trigger')
    rows = []
    for index in indices:
        raw = dataset.load_raw(dataset.items[index]['csi'])
        if cover:
            hit, audit = trigger.noise_inject_with_audit(raw, dose=dose, eps=eps)
        else:
            hit, audit = trigger.inject_with_audit(raw, dose, eps=eps)
        clean, normalized = dataset.normalize(raw), dataset.normalize(hit)
        actual = float(np.max(np.abs(normalized.astype(np.float64) - clean.astype(np.float64))))
        if audit['violation'] or actual > audit['ceiling']:
            raise ValueError('Per-sample model-input peak budget violation')
        stats = distortion_stats(clean, normalized)
        audit.update(index=index, model_input_peak=actual,
                     relative_l2=stats['relative_l2'], rmse=stats['rms'])
        rows.append(audit)
    return rows


def export_distortion(matrix, outdir):
    report = build_distortion(matrix)
    outdir = Path(outdir)
    _atomic_json(outdir / 'input_distortion.json', report)
    _write_csv(outdir / 'input_distortion.csv', report['rows'], DISTORTION_COLUMNS)
    return report


def export_summary(matrix, outdir):
    report = build_summary(matrix)
    distortion = export_distortion(matrix, outdir)
    report['distortion'] = distortion
    outdir = Path(outdir)
    _atomic_json(outdir / 'traditional_comparison.json', report)
    _write_csv(outdir / 'traditional_comparison.csv', report['rows'], SUMMARY_COLUMNS)
    _write_csv(outdir / 'dose_response.csv', report['dose_response'], DOSE_COLUMNS)
    _atomic_json(outdir / 'dose_response.json', dict(status=report['status'],
        profile=matrix['traditional_comparison_profile'], plan_sha256=matrix['plan_sha256'],
        rows=report['dose_response'], units='millimetres; same-model no-trigger target baseline'))
    lines = ['# Dose-adapted traditional CSI/HPE comparison', '',
        'Seed 42; full MM-Fi; 50 epochs; rho=0.1; fresh independent victims.', '',
        'Every method trains on identical paired U(0.2,1) trigger/payload doses.', '',
        f"Budget mode: {matrix['metrics_contract']['budget_mode']}; common peak ceiling: {matrix['metrics_contract']['shared_peak_ceiling']}; realized L2 is not matched.", '',
        '| Method | Clean MPJPE (mm) | PA-MPJPE (mm) | T1 (mm) | Mean positive-dose T (mm) | I1 (mm) | Poison / cover |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in report['rows']:
        lines.append(f"| {row['method']} | {row['clean_mpjpe_mm']:.3f} | {row['clean_pampjpe_mm']:.3f} | {row['t1_mpjpe_mm']:.3f} | {row['mean_positive_dose_tmpjpe_mm']:.3f} | {row['i1_improvement_mm']:.3f} | {row['n_poison']} / {row['n_cover']} |")
    lines.extend(['', 'T1: triggered target-joint MPJPE at d=1. I1: same-model no-trigger target error minus T1.',
        'Full relative PCK metrics are retained in CSV/JSON. Every method is evaluated at d=0,0.2,0.4,0.6,0.8,1.',
        'Six-dose curves are retained in dose_response.csv/json; no method receives a separate easier fixed-dose task.', '',
        'Limitations:', *[f'- {item}' for item in report['limitations']], '',
        f"Plan SHA256: `{matrix['plan_sha256']}`."])
    path = outdir / 'traditional_comparison.md'
    temporary = path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Runtime workers for non-WaNet cells; source WaNet always uses zero')
    parser.add_argument('--distortion-samples', type=int, default=256)
    parser.add_argument('--budget-mode', choices=['shared_peak', 'native'], default='shared_peak',
                        help='Common per-input peak cap (default), or separate unprojected source-operator control')
    parser.add_argument('--baseline-set', choices=['core', 'extended'], default='core',
                        help='Extended adds source-pinned FTrojan and FIBA; CCAI is not in either plan')
    parser.add_argument('--cells', nargs='+', choices=[m[0] for m in _methods('extended')])
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--distortion-only', action='store_true', help='CPU six-dose input measurement; no training/HPE')
    parser.add_argument('--_cell', help=argparse.SUPPRESS)
    parser.add_argument('--_manifest', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args._cell:
        matrix = json.loads(args._manifest.read_text(encoding='utf-8'))
        _validate_manifest(matrix)
        _run_cell(next(c for c in matrix['cells'] if c['method_key'] == args._cell))
        return 0
    if args.data_home is None or args.outdir is None:
        raise ValueError('--data-home and --outdir are required')
    if (len(set(args.devices)) != len(args.devices)
            or any(d != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) is None for d in args.devices)):
        raise ValueError('--devices must list distinct logical cuda:N devices, or cpu')
    if sum((args.dry_run, args.export_only, args.distortion_only)) > 1:
        raise ValueError('Choose only one of --dry-run, --export-only or --distortion-only')
    outdir = args.outdir.resolve()
    if args.fresh and outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError('--fresh refuses non-empty output directories; use a NEW path')
    with _cell_lock(outdir):
        manifest_path = outdir / 'traditional_comparison.resolved.json'
        requested = build_matrix(args.data_home, outdir, args.devices[0], args.num_workers,
                                 distortion_samples=args.distortion_samples, budget_mode=args.budget_mode,
                                 baseline_set=args.baseline_set)
        _validate_manifest(requested)
        if manifest_path.exists():
            matrix = json.loads(manifest_path.read_text(encoding='utf-8'))
            _validate_manifest(matrix)
            if matrix['plan_sha256'] != requested['plan_sha256']:
                raise ValueError('Comparison settings/source changed; use a NEW directory')
            for cell in matrix['cells']:
                cell['cfg']['num_workers'] = 0 if cell['method_key'] in ('wanet_source', 'fiba') else args.num_workers
                cell['cfg']['device'] = args.devices[0]
        else:
            if any(p.name != 'run.lock' for p in outdir.iterdir()):
                raise ValueError('Output contains unrelated files/historical results; use a NEW directory')
            matrix = requested
        _atomic_json(manifest_path, matrix)
        if args.dry_run:
            print(f'[traditional] DRY RUN: {args.baseline_set}/{args.budget_mode}; {len(matrix["cells"])} cells; paired U(.2,1); six-dose evaluation; rho=.1; 50 epochs; no data/GPU/training.\n{manifest_path}', flush=True)
            return 0
        if args.distortion_only:
            export_distortion(matrix, outdir)
            print(f'[traditional] CPU distortion only: {outdir / "input_distortion.csv"}', flush=True)
            return 0
        if not args.export_only:
            _check_inputs(matrix, args.devices)
            run_matrix(matrix, manifest_path, args.devices, args.cells,
                       worker_script=Path(__file__))
        if any(not _complete(cell) for cell in matrix['cells']):
            if args.cells and not args.export_only:
                print('[traditional] Selected cells finished; final summary awaits every planned cell.', flush=True)
                return 0
            raise ValueError('All planned cells must finish before exporting the comparison')
        export_summary(matrix, outdir)
        print(f'[traditional] Summary: {outdir / "traditional_comparison.md"}', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'[traditional] ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
