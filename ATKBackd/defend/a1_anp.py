"""Pinned-source ANP mechanism adapted to clean CSI-to-pose regression.

Independent implementation, not an improved ANP or a CIFAR reproduction. See
``a1_anp_sources.json`` for the inspected source commits and adaptations.
Only masks/noise are optimized. Source BN gamma/beta stay frozen; running
statistics change on the disposable mask-learning copy just as in upstream.
Final pruning uses the ORIGINAL victim's BN gamma only (beta is unchanged).
The upstream noise ascent is NOT projected: only masks are clamped to [0, 1].
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from defend.a1_repair_ops import (
    _atomic_save, _batch, _restore_rng, _rng_state, _state,
    clean_selection_loss, pose_loss, pose_prediction,
)


SCHEMA = "a1_anp_clean_repair_checkpoint_v1"
CHECKPOINT_EVERY = 25  # Runtime durability, not an extra scientific hyperparameter.
DEFAULT_ANP_RECIPE = {
    "nb_iter": 2000, "lr": 0.01, "momentum": 0.9,
    "batch_size": 256, "print_every": 500,
    "anp_eps": 0.4, "anp_steps": 1, "anp_alpha": 0.2,
    "pruning_by": "threshold", "pruning_max": 0.9,
    "pruning_step": 0.05, "acc_ratio": 0.1,
}


def resolve_anp_recipe(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError("ANP recipe must be a dictionary")
    value = {} if value is None else value
    unknown = set(value) - set(DEFAULT_ANP_RECIPE)
    if unknown:
        raise ValueError(f"Unknown ANP recipe fields: {sorted(unknown)}")
    r = copy.deepcopy(DEFAULT_ANP_RECIPE)
    r.update(copy.deepcopy(value))
    for key in ("nb_iter", "batch_size", "print_every", "anp_steps"):
        if type(r[key]) is not int or r[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "pruning_step"):
        if type(r[key]) not in (int, float) or not math.isfinite(r[key]) or r[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("momentum", "anp_eps", "acc_ratio"):
        if type(r[key]) not in (int, float) or not math.isfinite(r[key]) or r[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("anp_alpha", "pruning_max"):
        if type(r[key]) not in (int, float) or not math.isfinite(r[key]) or not 0 <= r[key] <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    if r["pruning_by"] != "threshold":
        raise ValueError("The frozen HPE adaptation supports threshold pruning only")
    if r["pruning_step"] > 1:
        raise ValueError("pruning_step must not exceed one")
    return r


class ANPBatchNorm(nn.Module):
    """BN1d/BN2d affine perturbations, with original parameters frozen."""

    def __init__(self, source):
        super().__init__()
        if type(source) not in (nn.BatchNorm1d, nn.BatchNorm2d) or not source.affine:
            raise ValueError("ANP requires ordinary affine BatchNorm1d/2d")
        self.norm = copy.deepcopy(source).requires_grad_(False)
        self.neuron_mask = nn.Parameter(torch.ones_like(source.weight))
        self.neuron_noise = nn.Parameter(torch.zeros_like(source.weight))
        self.neuron_noise_bias = nn.Parameter(torch.zeros_like(source.bias))
        self.is_perturbed = False
        self.train(source.training)

    def forward(self, x):
        bn = self.norm
        bn._check_input_dim(x)
        factor = bn.momentum if bn.momentum is not None else 0.0
        if self.training and bn.track_running_stats:
            bn.num_batches_tracked.add_(1)
            if bn.momentum is None:
                factor = 1.0 / float(bn.num_batches_tracked)
        use_batch = self.training or (bn.running_mean is None and bn.running_var is None)
        weight = bn.weight * (self.neuron_mask + self.neuron_noise
                              if self.is_perturbed else self.neuron_mask)
        bias = bn.bias * (1.0 + self.neuron_noise_bias) if self.is_perturbed else bn.bias
        return F.batch_norm(x,
            bn.running_mean if not self.training or bn.track_running_stats else None,
            bn.running_var if not self.training or bn.track_running_stats else None,
            weight, bias, use_batch, factor, bn.eps)

    @torch.no_grad()
    def reset_noise(self, eps):
        self.neuron_noise.uniform_(-eps, eps)
        self.neuron_noise_bias.uniform_(-eps, eps)


def masked_model(source):
    """Return a separate mask-learning copy, preserving source identity/state."""
    model = copy.deepcopy(source).requires_grad_(False)
    names = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if type(module) not in (nn.BatchNorm1d, nn.BatchNorm2d):
                raise ValueError(f"Unsupported ANP normalization at {name}")
            if not name:
                raise ValueError("Pose model cannot be a bare BatchNorm")
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, ANPBatchNorm(module))
            names.append(name)
    if not names:
        raise ValueError("ANP requires at least one affine BatchNorm1d/2d")
    return model, names


def _norms(model):
    return [m for m in model.modules() if isinstance(m, ANPBatchNorm)]


@torch.no_grad()
def clamp_masks(model):
    for m in _norms(model):
        m.neuron_mask.clamp_(0.0, 1.0)


def _set_noise(model, enabled):
    for m in _norms(model):
        m.is_perturbed = enabled


def _mask_iteration(model, x, y, recipe, mask_optimizer, noise_optimizer):
    model.train()
    noise_params = [p for m in _norms(model)
                    for p in (m.neuron_noise, m.neuron_noise_bias)]
    if recipe["anp_eps"] > 0:
        for m in _norms(model):
            m.reset_noise(recipe["anp_eps"])
        _set_noise(model, True)
        for _ in range(recipe["anp_steps"]):
            noise_optimizer.zero_grad(set_to_none=True)
            loss = -pose_loss(pose_prediction(model, x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite ANP adversarial loss")
            loss.backward()
            with torch.no_grad():
                for p in noise_params:
                    if p.grad is not None:
                        p.grad.sign_()
            # Source uses signed ascent from random [-eps,eps] initialization.
            # It does not project the resulting perturbation back to that box.
            noise_optimizer.step()
    mask_optimizer.zero_grad(set_to_none=True)
    if recipe["anp_eps"] > 0:
        _set_noise(model, True)
        robust = pose_loss(pose_prediction(model, x), y)
    else:
        robust = 0.0
    _set_noise(model, False)
    natural = pose_loss(pose_prediction(model, x), y)
    loss = recipe["anp_alpha"] * natural + (1 - recipe["anp_alpha"]) * robust
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite ANP mask loss")
    loss.backward()
    mask_optimizer.step()
    clamp_masks(model)
    return float(loss.detach())


def mask_scores(model, names):
    # Upstream writes mask_values.txt with four decimal places before pruning.
    return [{"layer": n, "scores": [float(f"{v:.4f}") for v in
              model.get_submodule(n).neuron_mask.detach().cpu().tolist()]}
            for n in names]


@torch.no_grad()
def prune_original(source, scores, threshold):
    """Source operation: zero BN gamma only; leave beta and all buffers alone."""
    model = copy.deepcopy(source)
    pruned = []
    for row in scores:
        module = model.get_submodule(row["layer"])
        if type(module) not in (nn.BatchNorm1d, nn.BatchNorm2d):
            raise ValueError("ANP pruning target is not an ordinary BatchNorm")
        if len(row["scores"]) != module.num_features:
            raise ValueError("ANP mask geometry changed")
        indices = [i for i, s in enumerate(row["scores"]) if s <= threshold]
        if indices:
            module.weight[indices] = 0
            pruned.append({"layer": row["layer"], "indices": indices})
    return model, pruned


def _state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _thresholds(recipe):
    count = int(math.floor(recipe["pruning_max"] / recipe["pruning_step"] + 1e-9))
    return [round(i * recipe["pruning_step"], 12) for i in range(count + 1)]


def train_anp(model, fit_loader, selection_loader, recipe, device,
              checkpoint_path, binding_hash, seed):
    """Optimize BN masks on trusted clean TRAIN, select pruning on clean TRAIN.

    The clean-selection MPJPE gate replaces BDB's TEST-ASR minimization under
    clean-accuracy tolerance. It chooses the greatest passing threshold, with
    at most ``acc_ratio`` relative error increase. All candidates are retained.
    The unpruned identity is the explicitly recorded feasible base candidate.
    No trigger/TEST loader or attack metric is accepted by this API.

    Atomic checkpoints resume at <=25-iteration boundaries, including global,
    replacement-sampler and loader-generator states. Exact replay requires
    unchanged workers=0 loaders and deterministic dataset/collation policies.
    """
    r = resolve_anp_recipe(recipe)
    if (not isinstance(binding_hash, str) or len(binding_hash) != 64
            or any(c not in "0123456789abcdef" for c in binding_hash)):
        raise ValueError("ANP requires a 64-character binding SHA256")
    if type(seed) is not int or seed < 0:
        raise ValueError("ANP seed must be a nonnegative integer")
    loaders = (fit_loader, selection_loader)
    if any(getattr(l, "num_workers", 0) != 0 for l in loaders):
        raise ValueError("ANP loaders require num_workers=0 for audited resume")
    if not hasattr(fit_loader, "dataset") or len(fit_loader.dataset) == 0:
        raise ValueError("ANP requires a nonempty indexed clean repair-fit dataset")
    source = copy.deepcopy(model).to(device)
    initial_state = _state(source)
    initial_hash = _state_digest(initial_state)
    path = Path(checkpoint_path)
    saved = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None
    if saved:
        for k, v in (("schema", SCHEMA), ("binding_sha256", binding_hash),
                     ("recipe", r), ("seed", seed), ("source_state_sha256", initial_hash)):
            if saved.get(k) != v:
                raise ValueError(f"ANP checkpoint {k} mismatch")
        if saved.get("complete"):
            source.load_state_dict(saved["selected_model_state"], strict=True)
            return source.eval(), copy.deepcopy(saved["metadata"])
    else:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    working, names = masked_model(source)
    masks = [m.neuron_mask for m in _norms(working)]
    noises = [p for m in _norms(working) for p in (m.neuron_noise, m.neuron_noise_bias)]
    mask_optimizer = torch.optim.SGD(masks, lr=r["lr"], momentum=r["momentum"])
    noise_optimizer = torch.optim.SGD(noises, lr=r["anp_eps"] / r["anp_steps"])
    sampler = torch.Generator().manual_seed(seed)
    metadata = copy.deepcopy(saved["metadata"]) if saved else {
        "defense": "anp", "complete": False, "stage": "mask_training",
        "completed_iterations": 0, "normalization_layers": names,
        "selection_rule": "greatest threshold within 10%-relative clean-selection MPJPE gate (configured acc_ratio); no TEST/trigger selection",
        "pruning": None, "pruning_selection": None,
        "cost": {"mask_steps": 0, "noise_steps": 0, "fit_forwards": 0},
        "resume_policy": "atomic <=25-iteration boundaries; workers=0; replacement-sampler state included",
        "resume_history": [],
        "adaptation": "MPJPE; BN1d/2d; clean TRAIN selection; original-victim gamma-only pruning; no post-pruning fine-tuning",
    }
    if saved:
        working.load_state_dict(saved["model_state"], strict=True)
        mask_optimizer.load_state_dict(saved["mask_optimizer_state"])
        noise_optimizer.load_state_dict(saved["noise_optimizer_state"])
        sampler.set_state(saved["sampler_state"])
        _restore_rng(saved["rng_state"], loaders)
        metadata["resume_history"].append({"stage": saved["stage"],
                                            "completed_iterations": saved["iteration"]})
    start = saved["iteration"] if saved else 0

    def save(stage, iteration, selected=None):
        _atomic_save({"schema": SCHEMA, "binding_sha256": binding_hash,
            "defense": "anp", "recipe": r, "seed": seed, "stage": stage,
            "iteration": iteration, "source_state_sha256": initial_hash,
            "model_state": _state(working),
            "mask_optimizer_state": mask_optimizer.state_dict(),
            "noise_optimizer_state": noise_optimizer.state_dict(),
            "sampler_state": sampler.get_state(), "rng_state": _rng_state(loaders),
            "selected_model_state": _state(selected) if selected is not None else None,
            "metadata": copy.deepcopy(metadata), "complete": metadata["complete"]}, path)

    for iteration in range(start, r["nb_iter"]):
        indices = torch.randint(len(fit_loader.dataset), (r["batch_size"],), generator=sampler).tolist()
        batch = fit_loader.collate_fn([fit_loader.dataset[i] for i in indices])
        x, y = _batch(batch, device)
        loss = _mask_iteration(working, x, y, r, mask_optimizer, noise_optimizer)
        done = iteration + 1
        metadata["completed_iterations"] = done
        metadata["cost"]["mask_steps"] += 1
        metadata["cost"]["noise_steps"] += r["anp_steps"] if r["anp_eps"] > 0 else 0
        metadata["cost"]["fit_forwards"] += r["anp_steps"] + 2 if r["anp_eps"] > 0 else 1
        metadata["last_fit_loss"] = loss
        if done % CHECKPOINT_EVERY == 0 or done == r["nb_iter"]:
            save("mask_training", done)
        if done == 1 or done % r["print_every"] == 0 or done == r["nb_iter"]:
            print(f"[ANP] mask iteration {done}/{r['nb_iter']}; clean-fit={loss:.8f}", flush=True)

    scores = mask_scores(working, names)
    selection = metadata["pruning_selection"]
    if selection is None:
        baseline = clean_selection_loss(source, selection_loader, device)
        selection = {"baseline_loss": baseline, "relative_error_tolerance": r["acc_ratio"],
            "max_clean_selection_loss": baseline * (1 + r["acc_ratio"]),
            "candidates": [], "selected_threshold": None,
            "identity_fallback_declared": True,
            "rule": "greatest passing threshold over fixed grid; only held-out clean TRAIN MPJPE"}
        metadata["pruning_selection"] = selection
    selected = copy.deepcopy(source)
    if selection["selected_threshold"] is not None:
        selected, _ = prune_original(source, scores, selection["selected_threshold"])
    for threshold in _thresholds(r)[len(selection["candidates"]):]:
        candidate, pruned = prune_original(source, scores, threshold)
        loss = clean_selection_loss(candidate, selection_loader, device)
        admissible = loss <= selection["max_clean_selection_loss"]
        selection["candidates"].append({"threshold": threshold,
            "clean_selection_loss": loss, "admissible": admissible,
            "pruned_count": sum(len(p["indices"]) for p in pruned)})
        if admissible:
            selected = candidate
            selection["selected_threshold"] = threshold
        metadata["stage"] = "pruning"
        save("pruning", r["nb_iter"])
        print(f"[ANP] threshold={threshold:.2f}; clean-selection={loss:.8f}; passing={admissible}", flush=True)
    selected_threshold = selection["selected_threshold"]
    pruned = [] if selected_threshold is None else prune_original(source, scores, selected_threshold)[1]
    metadata["pruning"] = {"operation": "original BN gamma zero; bias unchanged",
        "threshold": selected_threshold, "layers": pruned,
        "pruned_count": sum(len(p["indices"]) for p in pruned),
        "mask_scores": scores}
    metadata["stage"] = "complete"
    metadata["complete"] = True
    selected.eval()
    save("complete", r["nb_iter"], selected)
    return selected, metadata
