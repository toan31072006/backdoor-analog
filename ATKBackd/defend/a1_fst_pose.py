"""Feature Shift Tuning adapted to continuous MM-Fi pose regression.

Min et al. (NeurIPS 2023) reinitialize the classifier, then optimize clean
task loss plus an inner-product penalty against the compromised classifier,
projecting its weight to the original norm. This pose adaptation applies the
same operations to the final coordinate head (Linear for HPE-Li, 1x1 Conv2d
for MetaFi++/WiFlow), with ordinary MPJPE on trusted clean TRAIN frames.
The penalty is normalized by original weight squared norm and by a frozen
clean TRAIN loss, so the fixed coefficient is dimensionless across heads.
No TEST or attacker-target values enter training/selection.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from defend.a1_repair_ops import (_atomic_save, _batch, _restore_rng, _rng_state,
                                  _state, clean_selection_loss, pose_loss,
                                  pose_prediction)


SCHEMA = 'mmfi_a1_fst_pose_adapted_v1'
HEADS = {'hpeli': ('regression.7', nn.Linear),
         'metafiplusplus': ('decode.3', nn.Conv2d),
         'wiflow': ('final_conv.3', nn.Conv2d)}
EPOCHS = 10
LR = 1e-4
ALPHA = .1


def head_spec(backbone, model):
    if backbone not in HEADS:
        raise ValueError('Unknown FST pose backbone')
    name, expected = HEADS[backbone]
    try:
        head = model.get_submodule(name)
    except AttributeError as exc:
        raise ValueError('Missing coordinate head ' + name) from exc
    if type(head) is not expected:
        raise ValueError('FST coordinate head type changed')
    if isinstance(head, nn.Conv2d):
        if head.groups != 1 or head.out_channels != 3 or head.kernel_size != (1, 1):
            raise ValueError('FST coordinate Conv geometry changed')
    elif head.out_features != 51:
        raise ValueError('FST HPE-Li head must emit 17x3 coordinates')
    return name, head


def inner_product_penalty(head, original_weight, baseline_clean_loss, alpha=ALPHA):
    """Scale the author inner-product by fixed TRAIN-only pose-loss units."""
    if not math.isfinite(baseline_clean_loss) or baseline_clean_loss <= 0:
        raise ValueError('FST requires positive finite TRAIN clean loss')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('FST requires positive finite alpha')
    if tuple(head.weight.shape) != tuple(original_weight.shape):
        raise ValueError('Original head weight geometry differs')
    old = original_weight.to(device=head.weight.device, dtype=head.weight.dtype)
    denominator = old.square().sum()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError('Original head has zero or nonfinite norm')
    return alpha * baseline_clean_loss * (head.weight * old).sum() / denominator


@torch.no_grad()
def project_original_norm(head, original_weight):
    old_norm = torch.linalg.vector_norm(original_weight.to(head.weight.device))
    new_norm = torch.linalg.vector_norm(head.weight)
    if not (bool(torch.isfinite(old_norm)) and bool(torch.isfinite(new_norm))
            and float(old_norm) > 0 and float(new_norm) > 0):
        raise ValueError('FST coordinate head norm is zero/nonfinite')
    head.weight.mul_(old_norm / new_norm)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def repair_fst(model, backbone, fit_loader, selection_loader, device, folder,
               binding_sha256, seed=42, *, epochs=EPOCHS, lr=LR, alpha=ALPHA):
    """Run 10 fixed epochs; atomically retain exact epoch-boundary state."""
    if (not isinstance(binding_sha256, str) or len(binding_sha256) != 64
            or any(c not in '0123456789abcdef' for c in binding_sha256)):
        raise ValueError('FST scientific binding SHA256 required')
    if (type(epochs) is not int or epochs < 1 or not math.isfinite(lr) or lr <= 0
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError('Invalid FST fixed recipe')
    loaders = (fit_loader, selection_loader)
    if any(getattr(loader, 'num_workers', 0) != 0 for loader in loaders):
        raise ValueError('FST requires workers=0 for audited continuation')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    name, head = head_spec(backbone, model)
    original = head.weight.detach().cpu().clone()
    original_sha = hashlib.sha256(original.contiguous().numpy().tobytes()).hexdigest()
    progress = folder / 'fst_progress.pt'
    final = folder / 'repair.pt'
    if final.exists():
        blob = torch.load(final, map_location='cpu', weights_only=False)
        if (blob.get('binding_sha256') != binding_sha256 or blob.get('complete') is not True
                or blob.get('metadata', {}).get('original_head_sha256') != original_sha):
            raise ValueError('Existing FST final checkpoint differs from source')
        model.load_state_dict(blob['selected_model_state'], strict=True)
        return model.eval(), blob['metadata']

    _seed(seed)
    baseline = clean_selection_loss(model, selection_loader, device)
    recipe = dict(epochs=epochs, lr=lr, alpha=alpha, head=name,
                  baseline_clean_train_loss=baseline, original_head_sha256=original_sha)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=.0005)
    start = 0
    steps = 0
    heldout = None
    if progress.exists():
        saved = torch.load(progress, map_location='cpu', weights_only=False)
        if (saved.get('schema') != SCHEMA or saved.get('binding_sha256') != binding_sha256
                or saved.get('recipe') != recipe or saved.get('seed') != seed
                or saved.get('complete') is not False):
            raise ValueError('FST progress recipe/source binding differs')
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state'])
        _restore_rng(saved['rng_state'], loaders)
        start = saved['epoch'] + 1
        steps = saved['optimizer_steps']
        heldout = saved['clean_selection_loss']
    else:
        head.reset_parameters()
    if start > epochs or (start == epochs and heldout is None):
        raise ValueError('FST progress epoch is inconsistent with recipe')
    for epoch in range(start, epochs):
        model.train()
        count = 0
        for batch in fit_loader:
            x, y = _batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = pose_loss(pose_prediction(model, x), y)
            loss = loss + inner_product_penalty(head, original, baseline, alpha)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite FST TRAIN loss')
            loss.backward()
            optimizer.step()
            project_original_norm(head, original)
            steps += 1
            count += len(x)
        if count == 0:
            raise ValueError('FST clean TRAIN fit loader is empty')
        heldout = clean_selection_loss(model, selection_loader, device)
        payload = dict(schema=SCHEMA, binding_sha256=binding_sha256, recipe=recipe,
                       seed=seed, epoch=epoch, model_state=_state(model),
                       optimizer_state=optimizer.state_dict(), rng_state=_rng_state(loaders),
                       optimizer_steps=steps, clean_selection_loss=heldout,
                       complete=False)
        _atomic_save(payload, progress)
        print(f'[repair] fst/repair epoch {epoch + 1}/{epochs}; clean-selection={heldout:.8f}', flush=True)
    with torch.no_grad():
        reference = original.to(device=head.weight.device, dtype=head.weight.dtype)
        cosine = torch.nn.functional.cosine_similarity(
            head.weight.flatten(), reference.flatten(), dim=0)
        head_cosine = float(cosine.detach().cpu())
    if not math.isfinite(head_cosine):
        raise ValueError('FST final coordinate head cosine is nonfinite')
    metadata = dict(complete=True, defense='fst_pose_adapted', recipe=recipe,
                    source='Min et al., NeurIPS 2023', final_epoch=epochs - 1,
                    optimizer_steps=steps, clean_fit_and_selection_only=True,
                    no_test_selection=True, original_head_sha256=original_sha,
                    final_selection_loss=heldout,
                    final_head_cosine_to_original=head_cosine,
                    projection='coordinate head Frobenius norm fixed to original')
    _atomic_save(dict(binding_sha256=binding_sha256, complete=True,
                      selected_model_state=_state(model), metadata=metadata), final)
    return model.eval(), metadata


def smoke_fst(backbone, model, device, physical_batch=32):
    """Real pose-model B32 backward with reinitialized head and norm projection."""
    if type(physical_batch) is not int or physical_batch < 2:
        raise ValueError('FST smoke physical batch must be BN-safe')
    model = model.to(device)
    _, head = head_spec(backbone, model)
    original = head.weight.detach().cpu().clone()
    head.reset_parameters()
    changed = not torch.equal(head.weight.detach().cpu(), original)
    x = torch.rand(physical_batch, 3, 114, 10, device=device)
    y = torch.rand(physical_batch, 17, 3, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = pose_loss(pose_prediction(model, x), y) + inner_product_penalty(head, original, .1)
    loss.backward()
    optimizer.step()
    project_original_norm(head, original)
    norm_projected = torch.allclose(torch.linalg.vector_norm(head.weight),
                                    torch.linalg.vector_norm(original.to(device)),
                                    rtol=1e-5, atol=1e-6)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite FST physical smoke')
    return dict(backbone=backbone, method='fst', physical_batch=physical_batch,
                head_reinitialized=changed, norm_projected=bool(norm_projected), finite=True)
