"""Neural-Cleanse-style inversion and unlearning for MM-Fi pose regression.

This is an *adaptation*, not a reproduction of the classifier method in
Wang et al., IEEE S&P 2019.  Class labels are replaced with eight fixed
kinematic target hypotheses.  The anomaly score is diagnostic; a defended
model exists only after a recovered trigger passes a TRAIN-only quality gate
and the fixed clean-pose unlearning pass actually updates the model.

The caller must choose disjoint trusted TRAIN inversion/holdout/unlearning
banks before running this module.  No TEST samples or source attack trigger
are accepted by this API.  The caller binds the returned state to immutable
source/checkpoint/data hashes and evaluates it with the normal paired T5
evaluator; this module does not select anything on TEST.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import cos, isfinite, radians, sin
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PoseCandidate:
    name: str
    pivot: int
    joints: tuple[int, int]
    angle_degrees: int


# MM-Fi's 17-joint DT-Pose tree: right/left legs and both arms, two signs.
# This order and the inversion budget are fixed before looking at TEST.
CANDIDATES = tuple(
    PoseCandidate(f'{name}_{direction}', pivot, joints, sign * 45)
    for name, pivot, joints in (
        ('right_leg', 1, (2, 3)),
        ('left_leg', 4, (5, 6)),
        ('arm_11', 11, (12, 13)),
        ('arm_14', 14, (15, 16)),
    )
    for direction, sign in (('positive', 1), ('negative', -1))
)


@dataclass(frozen=True)
class NCConfig:
    inversion_steps_per_candidate: int = 64
    inversion_max_batch: int = 8
    inversion_lr: float = 0.05
    mask_cost: float = 0.10
    tv_cost: float = 0.0001
    anomaly_threshold: float = 2.0
    min_holdout_gain: float = 0.25
    max_mask_fraction: float = 0.40
    max_clean_control_norm_ratio: float = 0.80
    min_clean_control_anomaly_margin: float = 1.0
    unlearning_epochs: int = 1
    unlearning_max_batch: int = 32
    unlearning_lr: float = 0.0001
    seed: int = 42

    def validate(self) -> None:
        if not (1 <= self.inversion_steps_per_candidate <= 512):
            raise ValueError('invalid fixed inversion step budget')
        if not (1 <= self.inversion_max_batch <= 32):
            raise ValueError('invalid inversion batch limit')
        if not (1 <= self.unlearning_max_batch <= 32):
            raise ValueError('invalid unlearning batch limit')
        if self.unlearning_epochs != 1:
            raise ValueError('this lightweight profile fixes one unlearning epoch')
        for name in ('inversion_lr', 'mask_cost', 'tv_cost',
                     'anomaly_threshold', 'unlearning_lr'):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f'invalid {name}')
        if not (0 < self.min_holdout_gain < 1):
            raise ValueError('invalid TRAIN-only holdout gain threshold')
        if not (0 < self.max_mask_fraction < 1):
            raise ValueError('invalid maximum recovered mask fraction')
        if not (0 < self.max_clean_control_norm_ratio < 1):
            raise ValueError('invalid clean-model norm ratio threshold')
        if (not isfinite(self.min_clean_control_anomaly_margin)
                or self.min_clean_control_anomaly_margin <= 0):
            raise ValueError('invalid clean-model anomaly margin')


@dataclass
class CandidateResult:
    candidate: PoseCandidate
    mask_l1: float
    mask_fraction: float
    baseline_target_error_m: float
    recovered_target_error_m: float
    holdout_gain: float
    qualified: bool
    mask: Tensor
    pattern: Tensor

    def public(self) -> dict:
        result = asdict(self)
        result.pop('mask')
        result.pop('pattern')
        return result


@dataclass
class NCReport:
    results: list[CandidateResult]
    anomaly_index: float | None
    suspected_candidate: str | None
    status: str
    config: NCConfig
    inversion_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]

    def public(self) -> dict:
        return dict(schema=1, method='nc_pose_adapted', status=self.status,
                    anomaly_index=self.anomaly_index,
                    suspected_candidate=self.suspected_candidate,
                    inversion_ids_sha256=_ids_hash(self.inversion_ids),
                    holdout_ids_sha256=_ids_hash(self.holdout_ids),
                    inversion_samples=len(self.inversion_ids),
                    holdout_samples=len(self.holdout_ids),
                    candidates=[result.public() for result in self.results],
                    config=asdict(self.config),
                    limitations=[
                        'Pose-target candidates replace discrete classifier labels.',
                        'MAD score is descriptive and is not a calibrated NC classifier threshold.',
                        'Only trusted TRAIN banks may drive inversion, selection, or unlearning.',
                    ])


@dataclass(frozen=True)
class CleanControlDecision:
    status: str
    candidate: str | None
    infected_anomaly_index: float | None
    clean_anomaly_index: float | None
    matched_candidate_norm_ratio: float | None

    def public(self) -> dict:
        return asdict(self)


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':'),
                                      ensure_ascii=False).encode('utf-8')).hexdigest()


def _validate_ids(ids: Sequence[str], batches: Sequence[tuple], name: str) -> tuple[str, ...]:
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f'{name} requires nonempty frame identity strings')
    result = tuple(ids)
    if len(set(result)) != len(result):
        raise ValueError(f'duplicate {name} frame identities')
    count = sum(int(pair[0].shape[0]) for pair in batches)
    if len(result) != count:
        raise ValueError(f'{name} frame identity count differs from batches')
    return result


def _pose(output: Tensor | tuple | list) -> Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, Tensor):
        raise TypeError('model must return a pose tensor')
    if output.ndim == 4 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 3 or output.shape[1:] != (17, 3):
        raise ValueError(f'expected (N,17,3) prediction, got {tuple(output.shape)}')
    return output


def _batch(pair: tuple, *, device: torch.device, max_batch: int) -> tuple[Tensor, Tensor]:
    if not isinstance(pair, (tuple, list)) or len(pair) < 2:
        raise ValueError('expected (CSI, pose) TRAIN batch')
    x, y = pair[:2]
    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise TypeError('TRAIN CSI/pose must be torch tensors')
    if x.ndim != 4 or x.shape[1:] != (3, 114, 10):
        raise ValueError(f'expected MM-Fi CSI (N,3,114,10), got {tuple(x.shape)}')
    if x.shape[0] < 1 or x.shape[0] > max_batch:
        raise ValueError('TRAIN batch outside fixed physical batch limit')
    x = x.to(device=device, dtype=torch.float32)
    y = _pose(y.to(device=device, dtype=torch.float32))
    if y.shape[0] != x.shape[0]:
        raise ValueError('CSI and pose batch counts differ')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('non-finite TRAIN input')
    if x.min() < -1e-6 or x.max() > 1 + 1e-6:
        raise ValueError('MM-Fi CSI must use the authors\' [0,1] amplitude scale')
    return x, y


def rotate_target(y: Tensor, candidate: PoseCandidate) -> Tensor:
    """Rotate only a declared distal branch about global z, in metres."""
    y = _pose(y)
    if candidate.pivot in candidate.joints or len(set(candidate.joints)) != 2:
        raise ValueError('invalid candidate branch')
    if min((candidate.pivot, *candidate.joints)) < 0 or max((candidate.pivot, *candidate.joints)) >= 17:
        raise ValueError('candidate joint outside MM-Fi topology')
    theta = radians(candidate.angle_degrees)
    c, s = cos(theta), sin(theta)
    out = y.clone()
    pivot = y[:, candidate.pivot, :]
    rel = y[:, candidate.joints, :] - pivot[:, None, :]
    rotated = torch.stack((c * rel[..., 0] - s * rel[..., 1],
                           s * rel[..., 0] + c * rel[..., 1],
                           rel[..., 2]), dim=-1)
    out[:, candidate.joints, :] = rotated + pivot[:, None, :]
    return out


def _stamp(x: Tensor, mask: Tensor, pattern: Tensor) -> Tensor:
    return (1.0 - mask) * x + mask * pattern


def _target_error(pred: Tensor, target: Tensor, candidate: PoseCandidate) -> Tensor:
    return torch.linalg.vector_norm(
        _pose(pred)[:, candidate.joints, :] - target[:, candidate.joints, :],
        dim=-1).mean()


def _tv(mask: Tensor) -> Tensor:
    return ((mask[..., 1:, :] - mask[..., :-1, :]).abs().mean()
            + (mask[..., :, 1:] - mask[..., :, :-1]).abs().mean())


def _holdout(model: nn.Module, batches: Sequence[tuple], candidate: PoseCandidate,
             mask: Tensor, pattern: Tensor, device: torch.device,
             max_batch: int) -> tuple[float, float]:
    before = after = count = 0.0
    with torch.no_grad():
        for pair in batches:
            x, y = _batch(pair, device=device, max_batch=max_batch)
            target = rotate_target(y, candidate)
            before += float(_target_error(model(x), target, candidate)) * len(x)
            after += float(_target_error(model(_stamp(x, mask, pattern)), target, candidate)) * len(x)
            count += len(x)
    return before / count, after / count


def invert(model: nn.Module, inversion_train: Sequence[tuple],
           holdout_train: Sequence[tuple], *,
           inversion_ids: Sequence[str], holdout_ids: Sequence[str],
           device: str | torch.device,
           config: NCConfig = NCConfig()) -> NCReport:
    """Invert eight fixed pose targets on disjoint trusted TRAIN partitions.

    The caller is responsible for supplying disjoint identity lists and
    persisting their hashes.  This function uses neither TEST nor the A1
    trigger.  A missing or weak inversion is an explicit non-recovery result.
    """
    config.validate()
    if not inversion_train or not holdout_train:
        raise ValueError('both trusted TRAIN partitions are required')
    inversion_ids = _validate_ids(inversion_ids, inversion_train, 'inversion')
    holdout_ids = _validate_ids(holdout_ids, holdout_train, 'holdout')
    if set(inversion_ids).intersection(holdout_ids):
        raise ValueError('inversion and holdout TRAIN identities overlap')
    device = torch.device(device)
    if next(model.parameters()).device != device:
        raise ValueError('model and requested device differ')
    was_training = model.training
    trainable = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    results = []
    try:
        for candidate in CANDIDATES:
            mask_logits = nn.Parameter(torch.full((1, 3, 114, 10), -4.0, device=device))
            pattern_logits = nn.Parameter(torch.zeros((1, 3, 114, 10), device=device))
            optimizer = torch.optim.Adam((mask_logits, pattern_logits), lr=config.inversion_lr)
            for step in range(config.inversion_steps_per_candidate):
                x, y = _batch(inversion_train[step % len(inversion_train)],
                              device=device, max_batch=config.inversion_max_batch)
                target = rotate_target(y, candidate)
                mask, pattern = mask_logits.sigmoid(), pattern_logits.sigmoid()
                stamped = _stamp(x, mask, pattern)
                objective = _target_error(model(stamped), target, candidate)
                loss = objective + config.mask_cost * mask.mean() + config.tv_cost * _tv(mask)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('NC inversion produced a non-finite TRAIN objective')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                mask, pattern = mask_logits.sigmoid(), pattern_logits.sigmoid()
                mask, pattern = mask.detach(), pattern.detach()
                baseline, recovered = _holdout(model, holdout_train, candidate,
                                               mask, pattern, device, config.inversion_max_batch)
                gain = (baseline - recovered) / max(baseline, 1e-9)
                fraction = float(mask.mean())
                if not all(map(isfinite, (baseline, recovered, gain, fraction))) \
                        or not bool(torch.isfinite(mask).all()) \
                        or not bool(torch.isfinite(pattern).all()):
                    raise ValueError('NC inversion produced non-finite TRAIN evidence')
                results.append(CandidateResult(
                    candidate, float(mask.sum()), fraction, baseline, recovered, gain,
                    gain >= config.min_holdout_gain and fraction <= config.max_mask_fraction,
                    mask.cpu().contiguous(), pattern.cpu().contiguous()))
    finally:
        for p, flag in zip(model.parameters(), trainable):
            p.requires_grad_(flag)
        model.train(was_training)
    norms = np.array([result.mask_l1 for result in results], dtype=np.float64)
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    if mad <= 1e-9:
        return NCReport(results, None, None, 'no_recovered_trigger_mad_degenerate',
                        config, inversion_ids, holdout_ids)
    scores = (median - norms) / (1.4826 * mad)
    index = int(np.argmax(scores))
    score = float(scores[index])
    if score < config.anomaly_threshold:
        return NCReport(results, score, None, 'no_recovered_trigger_no_outlier',
                        config, inversion_ids, holdout_ids)
    if not results[index].qualified:
        return NCReport(results, score, None, 'no_recovered_trigger_quality_gate',
                        config, inversion_ids, holdout_ids)
    return NCReport(results, score, results[index].candidate.name,
                    'recovered_train_only', config, inversion_ids, holdout_ids)


def certify_against_clean_control(infected: NCReport,
                                  clean: NCReport) -> CleanControlDecision:
    """Require a smaller and more anomalous trigger than the paired clean model.

    The same fixed candidates, budget, and exact TRAIN identities must be used
    for both checkpoints.  This control addresses pose-branch geometry that
    could otherwise create false MAD outliers without an implanted backdoor.
    """
    if (infected.config != clean.config
            or infected.inversion_ids != clean.inversion_ids
            or infected.holdout_ids != clean.holdout_ids
            or [r.candidate for r in infected.results]
            != [r.candidate for r in clean.results]):
        raise ValueError('clean-model control must use exactly the same preregistered experiment')
    selected = infected.suspected_candidate
    if infected.status != 'recovered_train_only' or selected is None:
        return CleanControlDecision('no_infected_model_recovery', None,
                                    infected.anomaly_index, clean.anomaly_index, None)
    a1_result = next(r for r in infected.results if r.candidate.name == selected)
    clean_result = next(r for r in clean.results if r.candidate.name == selected)
    ratio = a1_result.mask_l1 / max(clean_result.mask_l1, 1e-12)
    if infected.anomaly_index is None or clean.anomaly_index is None:
        status = 'clean_control_mad_degenerate'
    elif (infected.anomaly_index - clean.anomaly_index
          < infected.config.min_clean_control_anomaly_margin):
        status = 'clean_control_no_anomaly_margin'
    elif ratio > infected.config.max_clean_control_norm_ratio:
        status = 'clean_control_no_norm_advantage'
    else:
        status = 'certified_train_only'
    return CleanControlDecision(status, selected if status == 'certified_train_only' else None,
                                infected.anomaly_index, clean.anomaly_index, ratio)


def unlearn(model: nn.Module, report: NCReport, control: CleanControlDecision,
            unlearning_train: Sequence[tuple], *, unlearning_ids: Sequence[str],
            device: str | torch.device) -> dict:
    """One fixed clean-pose unlearning epoch; modifies ``model`` in place.

    The selected reverse-engineered pattern is paired with each clean TRAIN
    pose.  A diagnostic alone never counts as a repaired model.  The caller
    must save the final weights and evaluate the frozen model separately.
    """
    config = report.config
    config.validate()
    if (report.status != 'recovered_train_only' or report.suspected_candidate is None
            or control.status != 'certified_train_only'
            or control.candidate != report.suspected_candidate):
        raise ValueError('NC inversion did not recover a clean-control-certified TRAIN trigger')
    if not unlearning_train:
        raise ValueError('trusted TRAIN unlearning bank is empty')
    unlearning_ids = _validate_ids(unlearning_ids, unlearning_train, 'unlearning')
    if set(unlearning_ids).intersection(report.inversion_ids) or \
            set(unlearning_ids).intersection(report.holdout_ids):
        raise ValueError('unlearning TRAIN identities overlap inversion/holdout')
    device = torch.device(device)
    if next(model.parameters()).device != device:
        raise ValueError('model and requested device differ')
    recovered = next(result for result in report.results
                     if result.candidate.name == report.suspected_candidate)
    mask = recovered.mask.to(device)
    pattern = recovered.pattern.to(device)
    initial_model_sha256 = _parameter_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.unlearning_lr)
    was_training = model.training
    model.train()
    steps = 0
    gradient_updates = 0
    final_loss = None
    try:
        for _ in range(config.unlearning_epochs):
            for pair in unlearning_train:
                x, y = _batch(pair, device=device, max_batch=config.unlearning_max_batch)
                optimizer.zero_grad(set_to_none=True)
                # Sequential backward keeps one physical batch activation graph
                # resident at a time on the 24-GB 4090.  One optimizer update
                # follows the clean+reconstructed-trigger pair.
                clean_loss = torch.linalg.vector_norm(_pose(model(x)) - y, dim=-1).mean()
                (0.5 * clean_loss).backward()
                triggered_loss = torch.linalg.vector_norm(
                    _pose(model(_stamp(x, mask, pattern))) - y, dim=-1).mean()
                (0.5 * triggered_loss).backward()
                if any(p.grad is not None and bool(torch.any(p.grad != 0))
                       for p in model.parameters()):
                    gradient_updates += 1
                optimizer.step()
                steps += 1
                final_loss = float(0.5 * (clean_loss.detach() + triggered_loss.detach()))
    finally:
        model.train(was_training)
    final_model_sha256 = _parameter_digest(model)
    if (steps == 0 or gradient_updates == 0 or final_loss is None
            or not isfinite(final_loss) or final_model_sha256 == initial_model_sha256):
        raise ValueError('NC-adapted unlearning made no valid model update')
    return dict(schema=1, method='nc_pose_adapted_unlearning', complete=True,
                steps=steps, nonzero_gradient_updates=gradient_updates,
                epochs=config.unlearning_epochs,
                unlearning_ids_sha256=_ids_hash(unlearning_ids),
                unlearning_samples=len(unlearning_ids),
                model_sha256_before=initial_model_sha256,
                model_sha256_after=final_model_sha256,
                physical_batch_limit=config.unlearning_max_batch,
                final_train_loss_m=final_loss,
                recovered_candidate=report.suspected_candidate,
                inversion_status=report.status,
                clean_control=control.public(),
                report=report.public())


def _parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_nc_adapted(
    a1_model: nn.Module,
    clean_model: nn.Module,
    inversion_train: Sequence[tuple],
    holdout_train: Sequence[tuple],
    unlearning_train: Sequence[tuple],
    *,
    inversion_ids: Sequence[str],
    holdout_ids: Sequence[str],
    unlearning_ids: Sequence[str],
    device: str | torch.device,
    output_dir: str | Path,
    source_binding_sha256: str,
    config: NCConfig = NCConfig(),
) -> tuple[nn.Module | None, dict]:
    """Run NC-adapted diagnosis and, only if certified, one unlearning pass.

    The two checkpoints, TRAIN partitions, and frozen source binding are caller
    inputs.  No TEST data or A1 trigger enters this function.  A detection
    failure writes diagnostics and returns ``(None, receipt)``; it never
    fabricates a repaired-model Table-4 row.  A successful run atomically saves
    ``nc_unlearned.pt`` and returns the modified A1 model with its receipt.
    An interrupted run preserves its files and requires explicit new-output
    handling, never silent retry or restoration from an unverified checkpoint.
    """
    if len(source_binding_sha256) != 64 or any(c not in '0123456789abcdef'
                                               for c in source_binding_sha256.lower()):
        raise ValueError('source binding must be a SHA-256 hex digest')
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('NC output directory already contains artifacts; preserve it')
    output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(inversion_ids=inversion_ids, holdout_ids=holdout_ids,
                  device=device, config=config)
    compute_device = torch.device(device)
    if next(a1_model.parameters()).device != compute_device:
        raise ValueError('A1 model is not on the declared inversion device')
    # Only one full pose backbone occupies CUDA during each inversion. This is
    # the actual production memory policy, including the paired clean control.
    if compute_device.type == 'cuda':
        clean_model.to('cpu')
    infected = invert(a1_model, inversion_train, holdout_train, **common)
    if compute_device.type == 'cuda':
        a1_model.to('cpu')
        torch.cuda.empty_cache()
        clean_model.to(compute_device)
    clean = invert(clean_model, inversion_train, holdout_train, **common)
    if compute_device.type == 'cuda':
        clean_model.to('cpu')
        torch.cuda.empty_cache()
        a1_model.to(compute_device)
    control = certify_against_clean_control(infected, clean)
    diagnostic = dict(schema=1, profile='mmfi_nc_pose_adapted_a1_only_v1',
                      source_binding_sha256=source_binding_sha256,
                      status=control.status,
                      infected_model=infected.public(),
                      paired_clean_model=clean.public(),
                      clean_control=control.public(),
                      no_test_selection=True,
                      target_family_prior='MM-Fi kinematic branches at +45/-45 degrees, fixed before TEST',
                      original_classifier_nc_reproduction=False)
    _write_json_new(output_dir / 'nc_diagnostic.json', diagnostic)
    if control.status != 'certified_train_only':
        return None, diagnostic

    repair = unlearn(a1_model, infected, control, unlearning_train,
                     unlearning_ids=unlearning_ids, device=device)
    receipt = dict(diagnostic, status='complete_nc_adapted_unlearning',
                   repair=repair, repair_checkpoint='nc_unlearned.pt')
    path = output_dir / 'nc_unlearned.pt'
    temporary = path.with_name(path.name + '.saving')
    if temporary.exists() or path.exists():
        raise ValueError('NC checkpoint already exists; preserve original run')
    torch.save(dict(schema=1, profile=receipt['profile'], complete=True,
                    source_binding_sha256=source_binding_sha256,
                    method='nc_pose_adapted_unlearning',
                    state_dict={key: value.detach().cpu().contiguous()
                                for key, value in a1_model.state_dict().items()},
                    repair=repair), temporary)
    os.replace(temporary, path)
    receipt['repair_checkpoint_sha256'] = _file_sha256(path)
    _write_json_new(output_dir / 'nc_receipt.json', receipt)
    return a1_model, receipt


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.saving')
    if path.exists() or temporary.exists():
        raise ValueError(f'NC output already exists: {path}')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)
