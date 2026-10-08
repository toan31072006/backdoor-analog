"""Staged, resumable RF baseline adaptations; never an ordinary poison trainer.

INFOCOM2025-POR tampers with the actual HPELi spatial encoder representation
using clean-teacher distillation and predefined representations, then learns a
fresh regression head on clean training poses. Downstream training CSI is the
default unlabeled substitute: this explicitly relaxes the original data-free
threat model. No pose target enters the tampering phase. Targeted pose metrics
are extra diagnostics, not this original attack's optimization objective.

CCAI2026-BackdoorRF learns a finite-support trigger with placement consistency
and a TRAIN-derived PSD prior. Clean pretraining, frozen-victim trigger warmup,
and alternating joint updates replace the source's classifier by pose MPJPE.
All code is independently implemented; source URLs/pins are in rf_adapters.py.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

# Anaconda NumPy/MKL and PyTorch can otherwise load conflicting OpenMP DLLs
# before the common PA-MPJPE SVD. This affects only this Windows process.
if os.name == 'nt':
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from attack.payload import make_target_pose, set_skeleton_config, descendants
from attack.rf_adapters import build_rf_trigger, spatial_features
from models.factory import build_model
import train_backdoor as common


_RF_CHECKPOINT_SCHEMA = 1
_PROTOCOLS = ('infocom2025_por', 'ccai2026_backdoorrf')
_SOURCE_PINS = {
    'infocom2025_por': {'repository': 'https://github.com/Tianyaz97/rf_backdoor',
                      'commit': '989f148ed35c96f46bf96db797ff23376895fb25',
                      'paper': 'https://arxiv.org/abs/2505.00881'},
    'ccai2026_backdoorrf': {'repository': 'https://github.com/NatsumiAi/BackdoorRF',
                          'commit': '4b7d44fc4c939b6b2018e522f0f0c9801fa90abb',
                          'publication': 'CCAI2026 per repository README and user confirmation'},
}


def resolve_rf_config(cfg):
    """Filesystem-free defaults, suitable for experiment previews/manifests."""
    result = common._resolve_training_config(cfg)
    protocol = result.get('training_protocol', result.get('rf_training_protocol'))
    if protocol not in _PROTOCOLS:
        raise ValueError(f'RF training_protocol must be one of {_PROTOCOLS}')
    if result.get('experiment_name') != 'mmfi' or result.get('model', '').lower() != 'hpeli':
        raise ValueError('RF adapters currently support MM-Fi HPELi only')
    if result.get('data_parallel'):
        raise ValueError('RF staged training requires data_parallel=false')
    if result.get('lr_scheduler'):
        raise ValueError('RF adapters use explicit stage learning rates; lr_scheduler must be false')
    result['training_protocol'] = protocol
    result.setdefault('victim_epochs', int(result.get('epochs', 50)))
    if int(result.get('epochs', result['victim_epochs'])) != int(result['victim_epochs']):
        raise ValueError('RF epochs must equal victim_epochs; extra stage budgets use rf_* keys')
    result['epochs'] = int(result['victim_epochs'])
    if result['epochs'] < 1:
        raise ValueError('victim_epochs must be positive')
    result.setdefault('rf_implementation_version', _RF_CHECKPOINT_SCHEMA)
    result.setdefault('rf_segment_length', 3)
    result.setdefault('rf_trigger_amp', float(result.get('eps', 0.185)))
    result.setdefault('rf_eval_start', 0)
    result.setdefault('rf_eval_trigger_index', 0)
    result.setdefault('rf_amplitude_semantics', 'per-antenna segment RMS or Gaussian SD; broadcast over subcarriers; clip [0,1]')
    result.setdefault('rf_training_payload_dose', 1.0)
    result.setdefault('rf_training_trigger_dose', 1.0)
    result.setdefault('rf_checkpoint_every', 1)
    if tuple(result.get(key) for key in ('n_ant', 'n_sub', 'n_pkt')) != (3, 114, 10):
        raise ValueError('RF adapters require the MM-Fi amplitude grid 3x114x10')
    if float(result['rf_training_payload_dose']) != 1.0 or float(result['rf_training_trigger_dose']) != 1.0:
        raise ValueError('RF comparison trains fixed maximal trigger and pose-target dose=1')
    if not 0 <= float(result.get('rho', 0.4)) <= 1:
        raise ValueError('rho must be in [0,1]')
    result.setdefault('rho', 0.4)
    result['dose_min'] = result['dose_max'] = 1.0
    if protocol == 'infocom2025_por':
        if 'clean_teacher_checkpoint' in result:
            result.setdefault('rf_clean_teacher_checkpoint', result.pop('clean_teacher_checkpoint'))
        result.setdefault('rf_substitute_source', 'downstream_train_unlabeled')
        result.setdefault('rf_substitute_path', '')
        if result['rf_substitute_path']:
            result['rf_substitute_source'] = 'external_unlabeled'
        if result['rf_substitute_source'] not in ('downstream_train_unlabeled', 'external_unlabeled'):
            raise ValueError('unsupported rf_substitute_source')
        result.setdefault('rf_encoder_epochs', 50)
        result.setdefault('rf_encoder_lr', 1e-3)
        result.setdefault('rf_distillation_weight', 1.0)
        result.setdefault('rf_por_weight', 1.0)
        result.setdefault('rf_teacher_scale_samples', 256)
        result.setdefault('rf_por_mapping', 'nonnegative_affine_cosine_scaled_by_teacher_train_RMS')
        result.setdefault('rf_downstream_head_epochs', result['epochs'])
        if int(result['rf_downstream_head_epochs']) != result['epochs']:
            raise ValueError('INFOCOM clean head epochs must equal victim_epochs')
        result.setdefault('threat_model', 'pretrained_encoder_tampering_with_downstream_unlabeled_data_relaxation'
                          if result['rf_substitute_source'] == 'downstream_train_unlabeled'
                          else 'pretrained_encoder_tampering_with_external_substitute_and_downstream_train_scale_calibration')
        result.setdefault('attacker_access', 'clean_teacher_weights_and_unlabeled_substitute_and_training_CSI_scale_calibration')
    else:
        if not 2 <= int(result['rf_segment_length']) <= 3:
            raise ValueError('BackdoorRF segment must contain 2 or 3 true packet frames')
        result.setdefault('rf_clean_pretrain_epochs', max(1, round(result['epochs'] * 0.3)))
        result.setdefault('rf_trigger_warmup_epochs', max(1, round(result['epochs'] * 0.1)))
        result.setdefault('rf_joint_epochs', result['epochs'] - int(result['rf_clean_pretrain_epochs']))
        if int(result['rf_clean_pretrain_epochs']) + int(result['rf_joint_epochs']) != result['epochs']:
            raise ValueError('clean pretraining + joint model epochs must equal victim_epochs')
        result.setdefault('rf_trigger_lr', 5e-4)
        result.setdefault('rf_smooth_kernel', 3 if int(result['rf_segment_length']) >= 3 else 1)
        result.setdefault('rf_lambda_position', 0.2)
        result.setdefault('rf_lambda_energy', 1e-3)
        result.setdefault('rf_lambda_smooth', 1e-3)
        result.setdefault('rf_lambda_environment', 0.1)
        result.setdefault('rf_template_samples', 256)
        result.setdefault('rf_environment_prior', 'training_temporal_residual_log_PSD')
        result.setdefault('rf_joint_scope', 'skunit2_regression')
        if result['rf_joint_scope'] != 'skunit2_regression':
            raise ValueError('BackdoorRF joint scope is the last HPELi SK block and regression head')
        result.setdefault('threat_model', 'white_box_staged_learned_trigger_and_model_training')
        result.setdefault('attacker_access', 'white_box_training_control')
    epoch_keys = ('rf_encoder_epochs', 'rf_clean_pretrain_epochs', 'rf_trigger_warmup_epochs', 'rf_joint_epochs')
    for key in epoch_keys:
        if key in result and (int(result[key]) != result[key] or int(result[key]) < 0):
            raise ValueError(f'{key} must be a nonnegative integer')
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finalize_rf_config(cfg):
    """Bind dependencies after a new clean-teacher run exists, before caching."""
    result = resolve_rf_config(cfg)
    if result['training_protocol'] == 'infocom2025_por':
        path = result.get('rf_clean_teacher_checkpoint')
        if not path or not Path(path).is_file():
            raise ValueError('INFOCOM requires rf_clean_teacher_checkpoint from the clean matrix dependency')
        dependencies = [('rf_clean_teacher_checkpoint', 'rf_clean_teacher_sha256')]
        if result['rf_substitute_source'] == 'external_unlabeled':
            dependencies.append(('rf_substitute_path', 'rf_substitute_sha256'))
        for path_key, hash_key in dependencies:
            path = result.get(path_key)
            if not path or not Path(path).is_file():
                raise ValueError(f'{path_key} must identify an existing file')
            digest = _sha256(path)
            if hash_key in result and result[hash_key] != digest:
                raise ValueError(f'{path_key} content changed; refusing stale dependency hash')
            result[hash_key] = digest
    return result


class _TrainingPairs(Dataset):
    """Deterministic targets; trigger updates happen exclusively in the parent."""
    def __init__(self, base, plan, cfg):
        self.base, self.plan, self.cfg = base, plan, cfg
        set_skeleton_config('mmfi')
        self.target_joints = tuple(descendants(cfg['pivot']))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        clean = np.asarray(item['pose'], dtype=np.float32)
        poisoned = index in self.plan
        target = make_target_pose(clean, self.cfg['pivot'], 1.0,
                                  np.deg2rad(self.cfg['theta_max_deg']),
                                  self.cfg['dose_mode'], self.cfg['payload_axis'],
                                  target_joints=self.target_joints) if poisoned else clean
        return {'csi': np.asarray(item['csi'], dtype=np.float32),
                'pose': np.asarray(target, dtype=np.float32), 'clean_pose': clean,
                'poisoned': poisoned, 'index': index}


class _UnlabeledCSI(Dataset):
    def __init__(self, base=None, path=None):
        self.base, self.data = base, None
        if path:
            if str(path).lower().endswith('.npz'):
                with np.load(path, allow_pickle=False) as arrays:
                    if 'csi' not in arrays.files:
                        raise ValueError('external substitute npz must have a csi array')
                    self.data = np.array(arrays['csi'], dtype=np.float32)
            else:
                self.data = np.load(path, mmap_mode='r', allow_pickle=False)
            if self.data.ndim != 4 or tuple(self.data.shape[1:]) != (3, 114, 10) or len(self.data) < 1:
                raise ValueError('external substitute must be Nx3x114x10 normalized amplitude CSI')

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.base)

    def __getitem__(self, index):
        if self.data is not None:
            csi = np.array(self.data[index], dtype=np.float32, copy=True)
        elif hasattr(self.base, 'items') and hasattr(self.base, 'load_raw'):
            csi = self.base.normalize(self.base.load_raw(self.base.items[index]['csi']))
        else:
            csi = self.base[index]['csi']
        csi = np.asarray(csi, dtype=np.float32)
        if csi.shape != (3, 114, 10) or not np.isfinite(csi).all() or csi.min() < 0 or csi.max() > 1:
            raise ValueError('substitute CSI must be finite normalized amplitudes in [0,1]')
        return {'csi': np.array(csi, copy=True), 'index': index}


def _poison_plan(count, cfg, base=None):
    rng = np.random.default_rng(cfg['seed'])
    number = int(np.floor(float(cfg['rho']) * count))
    if cfg.get('poison_select', 'uniform') != 'uniform':
        raise ValueError('RF source adapters use uniform poison selection')
    indices = rng.choice(count, number, replace=False)
    plan = {int(index): ordinal % 8 for ordinal, index in enumerate(indices)}
    pairs = [[int(index), 1.0] for index in indices]
    manifest = {'schema': 1, 'selection': 'uniform', 'seed': int(cfg['seed']),
                'rho_requested': float(cfg['rho']), 'n_total': count, 'n_poison': number,
                'dose_min': 1.0, 'dose_max': 1.0, 'training_protocol': cfg['training_protocol'],
                'poison_plan_sha256': hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest(),
                'samples': [{'index': index, 'trigger_id': plan[index], 'dose': 1.0} for index in plan]}
    return plan, manifest


def _loader(dataset, cfg, generator, shuffle=True):
    workers = int(cfg.get('num_workers', 0))
    return DataLoader(dataset, batch_size=int(cfg['batch_size']), shuffle=shuffle,
                      num_workers=workers, generator=generator, persistent_workers=False,
                      pin_memory=False, drop_last=False)


def _stages(cfg):
    if cfg['training_protocol'] == 'infocom2025_por':
        return [('encoder_tampering', int(cfg['rf_encoder_epochs'])),
                ('clean_head_training', int(cfg['rf_downstream_head_epochs']))]
    return [('clean_pretrain', int(cfg['rf_clean_pretrain_epochs'])),
            ('trigger_warmup', int(cfg['rf_trigger_warmup_epochs'])),
            ('joint_finetune', int(cfg['rf_joint_epochs']))]


def _save_stage_checkpoint(path, model, trigger, optimizers, cfg, cursor,
                           generator, history, manifest):
    blob = {'rf_checkpoint_schema': _RF_CHECKPOINT_SCHEMA,
            'cfg_fingerprint': common._config_fingerprint(cfg), 'cfg': cfg,
            'model': model.state_dict(), 'trigger': trigger.state_dict(),
            'optimizers': {key: opt.state_dict() for key, opt in optimizers.items()},
            'cursor': dict(cursor), 'phase': cursor['phase'], 'epoch': cursor['epoch'],
            'rng': common._rng_state(), 'loader_rng': generator.get_state(),
            'history': history, 'poison_manifest': manifest}
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(blob, temporary)
    os.replace(temporary, path)


def _load_stage_checkpoint(path, model, trigger, optimizers, cfg, generator):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if blob.get('rf_checkpoint_schema') != _RF_CHECKPOINT_SCHEMA:
        raise ValueError('RF checkpoint schema mismatch; refusing an ordinary or legacy checkpoint')
    if blob.get('cfg_fingerprint') != common._config_fingerprint(cfg):
        raise ValueError('RF checkpoint config mismatch; refusing to restart or overwrite')
    needed = ('model', 'trigger', 'optimizers', 'cursor', 'rng', 'loader_rng', 'history', 'poison_manifest')
    if any(key not in blob for key in needed):
        raise ValueError('RF checkpoint is missing exact-resume state')
    if set(blob['optimizers']) != set(optimizers):
        raise ValueError('RF checkpoint optimizer set does not match the stages')
    cursor = blob['cursor']
    stages = _stages(cfg)
    stage_index = cursor.get('stage_index')
    if not isinstance(stage_index, int) or not 0 <= stage_index <= len(stages):
        raise ValueError('RF checkpoint has an invalid stage cursor')
    expected_phase = stages[stage_index][0] if stage_index < len(stages) else 'complete'
    epoch = cursor.get('epoch')
    budget = stages[stage_index][1] if stage_index < len(stages) else 0
    if cursor.get('phase') != expected_phase or not isinstance(epoch, int) or not 0 <= epoch <= budget:
        raise ValueError('RF checkpoint phase/epoch is inconsistent with stage budgets')
    if not isinstance(cursor.get('initialized'), bool):
        raise ValueError('RF checkpoint must record stage initialization state')
    model.load_state_dict(blob['model'])
    trigger.load_state_dict(blob['trigger'])
    for key, optimizer in optimizers.items():
        optimizer.load_state_dict(blob['optimizers'][key])
    generator.set_state(blob['loader_rng'].cpu())
    common._restore_rng_state(blob['rng'])
    return blob['cursor'], blob['history'], blob['poison_manifest']


def _set_scope(model, scope):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = {'all': (model,), 'encoder': (model.skunit1, model.skunit2),
               'head': (model.regression,), 'last_head': (model.skunit2, model.regression),
               'none': ()}[scope]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.eval()
    for module in modules:
        module.train()


def _teacher(cfg, model, device):
    checkpoint = torch.load(cfg['rf_clean_teacher_checkpoint'], map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint or 'cfg' not in checkpoint:
        raise ValueError('clean teacher must be a fingerprinted common-trainer checkpoint')
    teacher_cfg = checkpoint['cfg']
    if checkpoint.get('cfg_fingerprint') != common._config_fingerprint(teacher_cfg):
        raise ValueError('clean teacher config fingerprint is invalid')
    for key in ('model', 'dataset_root', 'experiment_name', 'mmfi_protocol', 'mmfi_setting',
                'mmfi_split_seed', 'mmfi_random_ratio', 'seed', 'num_person'):
        if teacher_cfg.get(key) != cfg.get(key):
            raise ValueError(f'clean teacher differs on {key}')
    if float(teacher_cfg.get('rho', 0.0)) != 0.0:
        raise ValueError('INFOCOM teacher must be the clean rho=0 dependency')
    if int(checkpoint.get('epoch', -1)) + 1 != int(cfg['victim_epochs']):
        raise ValueError('clean teacher must have exactly victim_epochs completed')
    teacher = copy.deepcopy(model)
    teacher.load_state_dict(checkpoint['model'])
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    return teacher


@torch.no_grad()
def _configure_pors(trigger, teacher, training_csi, cfg, device):
    count = min(len(training_csi), int(cfg['rf_teacher_scale_samples']))
    if count < 1:
        raise ValueError('rf_teacher_scale_samples must select at least one TRAIN sample')
    squared, elements, dimension = 0.0, 0, None
    for start in range(0, count, int(cfg['batch_size'])):
        csi = torch.from_numpy(np.stack([training_csi[i]['csi'] for i in range(start, min(start + int(cfg['batch_size']), count))])).to(device)
        feature = spatial_features(teacher, csi)
        squared += float(feature.square().sum())
        elements += feature.numel()
        dimension = feature[0].numel()
    trigger.configure_pors(dimension, max((squared / elements) ** 0.5, 1e-4))


@torch.no_grad()
def _configure_template(trigger, training_csi, cfg, device):
    count = min(len(training_csi), int(cfg['rf_template_samples']))
    if count < 1:
        raise ValueError('rf_template_samples must select at least one TRAIN sample')
    total, windows = None, 0
    for start in range(0, count, int(cfg['batch_size'])):
        csi = torch.from_numpy(np.stack([training_csi[i]['csi'] for i in range(start, min(start + int(cfg['batch_size']), count))])).to(device)
        # Each antenna/subcarrier contributes true temporal windows, centered
        # and normalized to the trigger amplitude before estimating the prior.
        segments = csi.unfold(-1, trigger.segment_length, 1)
        residual = segments - segments.mean(-1, keepdim=True)
        rms = residual.square().mean((-1, -2, -3, -4), keepdim=True).clamp_min(1e-12).sqrt()
        residual = residual * trigger.amplitude / rms
        power = torch.fft.rfft(residual, dim=-1).abs().square() / trigger.segment_length
        contribution = power.sum((0, 2, 3))
        total = contribution if total is None else total + contribution
        windows += power.shape[0] * power.shape[2] * power.shape[3]
    trigger.psd_template.copy_((total / windows + 1e-6).log())


def _encoder_epoch(model, teacher, trigger, optimizer, loader, plan, cfg, device):
    _set_scope(model, 'encoder')
    # Preserve the teacher's running statistics under the substitute shift.
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()
    losses = []
    for batch in loader:
        csi = batch['csi'].to(device)
        index = batch['index'].tolist()
        mask = torch.tensor([i in plan for i in index], device=device, dtype=torch.bool)
        loss = csi.new_tensor(0.0)
        optimizer.zero_grad(set_to_none=True)
        if bool((~mask).any()):
            with torch.no_grad():
                target = spatial_features(teacher, csi[~mask])
            loss = loss + float(cfg['rf_distillation_weight']) * F.mse_loss(spatial_features(model, csi[~mask]), target)
        if bool(mask.any()):
            ids = torch.tensor([plan[i] for i in index if i in plan], device=device, dtype=torch.long)
            attacked = trigger.inject_tensor(csi[mask], trigger_ids=ids)
            loss = loss + float(cfg['rf_por_weight']) * F.mse_loss(
                spatial_features(model, attacked).flatten(1), trigger.por_bank[ids])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {'model_loss': float(np.mean(losses))}


def _head_epoch(model, optimizer, loader, device):
    _set_scope(model, 'head')
    losses = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            encoded = spatial_features(model, batch['csi'].to(device))
        prediction = model.regression(encoded).reshape(-1, model.num_person, model.num_keypoints, model.num_coor)
        loss = common._mpjpe_loss(prediction, batch['clean_pose'].to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {'model_loss': float(np.mean(losses))}


def _trigger_step(model, trigger, optimizer, csi, target, cfg):
    _set_scope(model, 'none')
    trigger.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(trigger.inject_tensor(csi, mode='random'))
    _, high = model(trigger.inject_tensor(csi, mode='high_energy'))
    _, low = model(trigger.inject_tensor(csi, mode='low_energy'))
    consistency = F.mse_loss(F.normalize(high, dim=1), F.normalize(low, dim=1))
    loss = common._mpjpe_loss(prediction, target) + float(cfg['rf_lambda_position']) * consistency
    loss = loss + trigger.regularization_loss(cfg['rf_lambda_energy'], cfg['rf_lambda_smooth'], cfg['rf_lambda_environment'])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _learned_epoch(model, trigger, optimizers, loader, phase, cfg, device):
    model_losses, trigger_losses = [], []
    for batch in loader:
        csi = batch['csi'].to(device)
        target = batch['pose'].to(device)
        mask = batch['poisoned'].to(device).bool()
        if phase != 'clean_pretrain' and bool(mask.any()):
            trigger_losses.append(_trigger_step(model, trigger, optimizers['trigger'], csi[mask], target[mask], cfg))
        if phase == 'trigger_warmup':
            continue
        _set_scope(model, 'all' if phase == 'clean_pretrain' else 'last_head')
        if phase == 'clean_pretrain':
            target = batch['clean_pose'].to(device)
        elif bool(mask.any()):
            csi = csi.clone()
            with torch.no_grad():
                csi[mask] = trigger.inject_tensor(csi[mask], mode='random')
        optimizers['victim'].zero_grad(set_to_none=True)
        prediction, _ = model(csi)
        # Ordinary pose ERM sees only the detached CSI and ordinary targets.
        loss = common._mpjpe_loss(prediction, target)
        loss.backward()
        optimizers['victim'].step()
        model_losses.append(float(loss.detach()))
    return {'model_loss': float(np.mean(model_losses)) if model_losses else None,
            'trigger_loss': float(np.mean(trigger_losses)) if trigger_losses else None}


def train(cfg, ckpt_dir=None):
    cfg = finalize_rf_config(cfg)
    common._validate_training_contract(cfg)
    cached = common._load_cached_result(ckpt_dir, cfg)
    if cached is not None:
        return None, cached
    seed = int(cfg['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_skeleton_config('mmfi')
    # Holdout is deliberately not loaded until every training stage is complete.
    base_train = common._load_dataset(cfg, 'training')
    if len(base_train) < 1:
        raise ValueError('RF training split is empty')
    train_csi = _UnlabeledCSI(base=base_train)
    substitute = (_UnlabeledCSI(path=cfg['rf_substitute_path'])
                  if cfg.get('rf_substitute_source') == 'external_unlabeled' else train_csi)
    plan, manifest = _poison_plan(len(substitute) if cfg['training_protocol'] == 'infocom2025_por' else len(base_train), cfg)
    pairs = _TrainingPairs(base_train, plan if cfg['training_protocol'] != 'infocom2025_por' else {}, cfg)
    model = build_model(cfg['model'], num_keypoints=17, num_person=cfg.get('num_person', 1),
                        subcarrier_num=114, dataset='mmfi', pretrained=False).to(device)
    trigger = build_rf_trigger(cfg['training_protocol'], cfg).to(device)
    generator = torch.Generator().manual_seed(seed)
    stages = _stages(cfg)
    teacher = None
    if cfg['training_protocol'] == 'infocom2025_por':
        teacher = _teacher(cfg, model, device)
        optimizers = {'encoder': torch.optim.Adam(list(model.skunit1.parameters()) + list(model.skunit2.parameters()), lr=float(cfg['rf_encoder_lr'])),
                      'head': common._build_optimizer(model, cfg, 'mmfi')[1]}
    else:
        optimizers = {'victim': common._build_optimizer(model, cfg, 'mmfi')[1],
                      'trigger': torch.optim.Adam(trigger.parameters(), lr=float(cfg['rf_trigger_lr']))}
    directory = Path(ckpt_dir) if ckpt_dir else Path('experiments_out') / cfg['training_protocol']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'checkpoint.pt'
    cursor = {'stage_index': 0, 'phase': stages[0][0], 'epoch': 0, 'initialized': False}
    history = []
    if path.exists():
        cursor, history, saved_manifest = _load_stage_checkpoint(path, model, trigger, optimizers, cfg, generator)
        if saved_manifest['poison_plan_sha256'] != manifest['poison_plan_sha256']:
            raise ValueError('RF poison plan differs from checkpoint')
        manifest = saved_manifest
    else:
        if teacher is not None:
            _configure_pors(trigger, teacher, train_csi, cfg, device)
        else:
            _configure_template(trigger, train_csi, cfg, device)
        _save_stage_checkpoint(path, model, trigger, optimizers, cfg, cursor, generator, history, manifest)
    (directory / 'config.resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True), encoding='utf-8')
    (directory / 'poison_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for stage_index in range(int(cursor['stage_index']), len(stages)):
        phase, epochs = stages[stage_index]
        if cursor['phase'] != phase:
            raise ValueError('RF checkpoint cursor phase is inconsistent')
        if not cursor['initialized']:
            if phase == 'clean_head_training':
                for module in model.regression.modules():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
                from models.hpeli import hpeli_init
                model.regression.apply(hpeli_init)
            cursor['initialized'] = True
            _save_stage_checkpoint(path, model, trigger, optimizers, cfg, cursor, generator, history, manifest)
        dataset = substitute if phase == 'encoder_tampering' else pairs
        loader = _loader(dataset, cfg, generator)
        for epoch in range(int(cursor['epoch']), epochs):
            started = time.monotonic()
            if phase == 'encoder_tampering':
                row = _encoder_epoch(model, teacher, trigger, optimizers['encoder'], loader, plan, cfg, device)
            elif phase == 'clean_head_training':
                row = _head_epoch(model, optimizers['head'], loader, device)
            else:
                row = _learned_epoch(model, trigger, optimizers, loader, phase, cfg, device)
            history.append({'phase': phase, 'epoch': epoch + 1, **row})
            cursor['epoch'] = epoch + 1
            elapsed = time.monotonic() - started
            print(f'[rf] {phase} {epoch + 1}/{epochs}: {row} t={elapsed:.1f}s', flush=True)
            if (epoch + 1) % max(1, int(cfg['rf_checkpoint_every'])) == 0 or epoch + 1 == epochs:
                _save_stage_checkpoint(path, model, trigger, optimizers, cfg, cursor, generator, history, manifest)
        cursor = {'stage_index': stage_index + 1,
                  'phase': stages[stage_index + 1][0] if stage_index + 1 < len(stages) else 'complete',
                  'epoch': 0, 'initialized': False}
        _save_stage_checkpoint(path, model, trigger, optimizers, cfg, cursor, generator, history, manifest)
    base_test = common._load_dataset(cfg, 'test')
    model.eval()
    trigger.eval()
    result = common.evaluate(model, base_test, trigger, cfg, device)
    result.update(dataset='mmfi', pivot=int(cfg['pivot']), theta_max_deg=float(cfg['theta_max_deg']),
                  payload_axis=list(cfg['payload_axis']), dose_mode=cfg['dose_mode'], rho=float(cfg['rho']),
                  eps=float(cfg['eps']), seed=seed, poison_select='uniform',
                  n_poison=manifest['n_poison'], n_total=manifest['n_total'],
                  poison_plan_sha256=manifest['poison_plan_sha256'], victim_loss='mpjpe',
                  trigger=trigger.baseline_name, training_protocol=cfg['training_protocol'],
                  training_contract='staged_rf_adaptation', attacker_access=cfg['attacker_access'],
                  threat_model=cfg['threat_model'], config_fingerprint=common._config_fingerprint(cfg),
                  rf_source=_SOURCE_PINS[cfg['training_protocol']],
                  rf_stage_epochs={phase: epochs for phase, epochs in stages},
                  rf_total_stage_epochs=sum(epochs for _, epochs in stages),
                  rf_victim_epochs=cfg['victim_epochs'], rf_trigger_amp=cfg['rf_trigger_amp'],
                  rf_eval_start=cfg['rf_eval_start'], rf_eval_trigger_index=cfg['rf_eval_trigger_index'],
                  rf_checkpoint={'path': str(path), 'schema': _RF_CHECKPOINT_SCHEMA, 'phase': 'complete'},
                  rf_training_history=history,
                  rf_adaptations=['amplitude CSI, per-antenna temporal segments broadcast over subcarriers',
                                  'dose scales trigger amplitude with [0,1] clipping',
                                  'pose MPJPE replaces classification supervision; categorical supervised contrastive pretraining is not transferred'])
    if teacher is not None:
        result.update(rf_substitute_source=cfg['rf_substitute_source'],
                      rf_data_free_claim=False,
                      rf_substitute_independence='not independent of downstream training data'
                      if cfg['rf_substitute_source'] == 'downstream_train_unlabeled'
                      else 'external pool independence requires separate provenance verification',
                      rf_clean_teacher_sha256=cfg['rf_clean_teacher_sha256'],
                      rf_clean_teacher_epochs=cfg['victim_epochs'],
                      rf_total_model_update_epochs=cfg['victim_epochs'] + sum(epochs for _, epochs in stages),
                      rf_por_mapping=cfg['rf_por_mapping'],
                      rf_por_scale_source='clean_teacher_activations_on_downstream_training_CSI_only',
                      rf_targeted_metric_role='additional pose-target diagnostic; POR attack does not optimize this target')
        result['rf_adaptations'].append('a supervised clean HPE teacher replaces the original self-supervised RF PTM; cosine PORs use a nonnegative fixed affine map')
    else:
        result.update(rf_environment_prior=cfg['rf_environment_prior'],
                      rf_targeted_metric_role='pose-target MPJPE replaces original targeted classification loss')
    object.__setattr__(model, '_trained_trigger', trigger)
    common._save_cached_result(directory, cfg, result)
    return model, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-dir', required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    train(cfg, args.ckpt_dir)


if __name__ == '__main__':
    main()
