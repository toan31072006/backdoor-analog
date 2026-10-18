"""Clean-data model repair for the separately versioned A1/T5 experiment.

These are declared CSI/HPE adaptations, not a byte-identical reproduction of
BackdoorBench classification training. No trigger, poison, or TEST label is used
by this module. All clean losses are MPJPE in the dataset's native pose units.
"""
from __future__ import annotations

import copy
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import prune


SCHEMA = "a1_clean_repair_checkpoint_v2_source_defaults"
DEFAULT_RECIPE = {
    "epochs": 100, "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005,
    "nad_teacher_epochs": 10, "nad_student_epochs": 10,
    "fp_layer": "regression.7",
    "fp_once_prune_ratio": 0.01, "utility_tolerance": 0.1,
    "nad_layers": ["skunit1", "skunit2", "regression.5"],
    "nad_betas": [500.0, 1000.0, 1000.0], "attention_p": 2.0,
    "sam_radius": 2.0, "sam_adaptive": True,
}


def pose_prediction(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    if not isinstance(out, torch.Tensor):
        raise TypeError("Pose model must return a tensor or tensor-first tuple")
    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0]
    return out


def pose_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    if pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("Pose loss requires equal B x J x 3 tensors")
    return torch.linalg.vector_norm(pred - true, dim=-1).mean()


def attention_map(tensor: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    if tensor.ndim != 4 or p <= 0 or not math.isfinite(p):
        raise ValueError("NAD attention requires B x C x H x W and finite p > 0")
    attention = tensor.abs().pow(p).sum(dim=1).flatten(1)
    return attention / (torch.linalg.vector_norm(attention, dim=1, keepdim=True) + 1e-6)


def attention_loss(student: torch.Tensor, teacher: torch.Tensor,
                   p: float = 2.0) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError("Teacher/student attention feature shapes differ")
    return F.mse_loss(attention_map(student, p), attention_map(teacher.detach(), p))


def _batch(batch, device):
    if isinstance(batch, dict):
        # The repository collator exposes true clean pose as ``pose``. Never
        # train a defense against an attacker target field.
        if "csi" not in batch or "pose" not in batch:
            raise ValueError("Clean dictionary batches require csi and true pose")
        x, y = batch["csi"].to(device), batch["pose"].to(device)
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x, y = batch[0].to(device), batch[1].to(device)
    else:
        raise ValueError("Clean loader must yield CSI and true_pose")
    if y.ndim == 4 and y.shape[1] == 1:
        y = y[:, 0]
    return x, y


def _module(model, name):
    try:
        return model.get_submodule(name)
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Missing defense layer: {name}") from exc


@torch.no_grad()
def clean_selection_loss(model, loader, device) -> float:
    """Sample-weighted clean-only selection objective; never uses TEST."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for batch in loader:
            x, y = _batch(batch, device)
            value = pose_loss(pose_prediction(model, x), y)
            if not torch.isfinite(value):
                raise ValueError("Nonfinite clean selection loss")
            total += float(value) * len(x)
            count += len(x)
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("Clean selection loader is empty")
    return total / count


@torch.no_grad()
def rank_head_inputs(model, loader, device, layer="regression.7"):
    """Signed mean final-linear input activation, weighted by sample count."""
    head = _module(model, layer)
    if not isinstance(head, nn.Linear):
        raise ValueError("Fine-Pruning target must be nn.Linear")
    activation_sum, count = None, 0

    def capture(_module, args):
        nonlocal activation_sum, count
        a = args[0].detach()
        if a.ndim != 2 or a.shape[1] != head.in_features:
            raise ValueError("Expected B x input_features final-linear inputs")
        s = a.double().sum(dim=0).cpu()
        activation_sum = s if activation_sum is None else activation_sum + s
        count += len(a)

    handle = head.register_forward_pre_hook(capture)
    mode = model.training
    model.eval()
    try:
        for batch in loader:
            x, _ = _batch(batch, device)
            pose_prediction(model, x)
    finally:
        handle.remove()
        model.train(mode)
    if not count:
        raise ValueError("Clean activation loader is empty")
    return activation_sum / count


def apply_pruning(model, layer, indices):
    """Persistent input-column mask; gradients cannot regrow pruned features."""
    head = _module(model, layer)
    if not isinstance(head, nn.Linear):
        raise ValueError("Fine-Pruning target must be nn.Linear")
    indices = list(indices)
    if (any(type(i) is not int or not 0 <= i < head.in_features for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Invalid or duplicate pruned input indices")
    if hasattr(head, "weight_orig"):
        raise ValueError("Pruning is already installed; do not compound masks")
    mask = torch.ones_like(head.weight)
    mask[:, indices] = 0
    prune.custom_from_mask(head, "weight", mask)
    return {"layer": layer, "indices": indices, "input_features": head.in_features,
            "pruned_count": len(indices)}


def restore_pruning(model, metadata):
    if metadata is None:
        return
    head = _module(model, metadata["layer"])
    if head.in_features != metadata["input_features"]:
        raise ValueError("Pruning geometry changed")
    expected = torch.ones_like(head.weight)
    expected[:, metadata["indices"]] = 0
    if hasattr(head, "weight_orig"):
        if not torch.equal(head.weight_mask, expected):
            raise ValueError("Existing pruning mask differs from checkpoint metadata")
    else:
        apply_pruning(model, metadata["layer"], metadata["indices"])


def _recipe(value):
    unknown = set(value) - set(DEFAULT_RECIPE)
    if unknown:
        raise ValueError(f"Unknown repair recipe fields: {sorted(unknown)}")
    r = copy.deepcopy(DEFAULT_RECIPE)
    r.update(copy.deepcopy(value))
    for key in ("epochs", "nad_teacher_epochs", "nad_student_epochs"):
        if type(r[key]) is not int or r[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "attention_p", "sam_radius"):
        if not math.isfinite(r[key]) or r[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("momentum", "weight_decay", "utility_tolerance"):
        if not math.isfinite(r[key]) or r[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 < r["fp_once_prune_ratio"] < 1:
        raise ValueError("fp_once_prune_ratio must be between zero and one")
    if not isinstance(r["nad_layers"], list) or not r["nad_layers"] or len(set(r["nad_layers"])) != len(r["nad_layers"]):
        raise ValueError("nad_layers must be unique nonempty layer names")
    if (not isinstance(r["nad_betas"], list) or len(r["nad_betas"]) != len(r["nad_layers"])
            or any(not math.isfinite(b) or b < 0 for b in r["nad_betas"])):
        raise ValueError("nad_betas must match NAD layers and be finite nonnegative")
    if type(r["sam_adaptive"]) is not bool:
        raise ValueError("sam_adaptive must be explicit bool")
    return r


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _rng_state(loaders):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_generators": [l.generator.get_state() if getattr(l, "generator", None) is not None else None for l in loaders]}


def _restore_rng(state, loaders):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Cannot resume CUDA RNG state without CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])
    for loader, saved in zip(loaders, state["loader_generators"]):
        generator = getattr(loader, "generator", None)
        if (generator is None) != (saved is None):
            raise ValueError("Loader generator policy changed during resume")
        if generator is not None:
            generator.set_state(saved)


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _sam_step(model, x, y, optimizer, radius, adaptive=True):
    """Weight-adaptive SAM, alpha=0, with upstream second-pass BN momentum=0."""
    optimizer.zero_grad(set_to_none=True)
    first = pose_loss(pose_prediction(model, x), y)
    if not torch.isfinite(first):
        raise ValueError("Nonfinite SAM first-pass loss")
    first.backward()
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        raise ValueError("SAM has no gradients")
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm((p.detach().abs() if adaptive else 1.0) * p.grad)
        for p in params]))
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite SAM gradient norm")
    originals = [(p, p.detach().clone()) for p in params]
    batchnorms = [(m, m.momentum) for m in model.modules()
                  if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        with torch.no_grad():
            for p in params:
                factor = p.detach().square() if adaptive else 1.0
                p.add_(factor * p.grad * (radius / (norm + 1e-12)))
        optimizer.zero_grad(set_to_none=True)
        for bn, _ in batchnorms:
            bn.momentum = 0
        second = pose_loss(pose_prediction(model, x), y)
        if not torch.isfinite(second):
            raise ValueError("Nonfinite SAM second-pass loss")
        second.backward()
    finally:
        with torch.no_grad():
            for p, original in originals:
                p.copy_(original)
        for bn, momentum in batchnorms:
            bn.momentum = momentum
    optimizer.step()
    return float(first.detach())


def train_repair(model, fit_loader, selection_loader, defense, recipe, device,
                 checkpoint_path, binding_sha256, seed):
    """Repair a fresh source-model copy, or resume our own bound checkpoint.

    Returns the final-stage model and JSON-compatible metadata.
    Atomic epoch checkpoints retain last model/optimizer/RNG for continuation;
    ``selected_model_state`` is the final epoch evaluation state, not a score-
    selected checkpoint. Clean selection is used only for the FP pruning gate.
    Callers must supply ordered, trusted clean TRAIN-only loaders. Epoch-boundary
    exact resume assumes workers=0 and unchanged deterministic loader transforms.
    """
    if defense not in ("ft", "fp", "nad", "ft_sam"):
        raise ValueError("Unknown repair defense")
    if (not isinstance(binding_sha256, str) or len(binding_sha256) != 64
            or any(c not in "0123456789abcdef" for c in binding_sha256)):
        raise ValueError("A 64-character scientific binding SHA256 is required")
    r = _recipe(recipe)
    loaders = (fit_loader, selection_loader)
    if any(getattr(l, "num_workers", 0) != 0 for l in loaders):
        raise ValueError("Repair loaders require num_workers=0 for audited resume")
    model = model.to(device)
    initial_state = _state(model)
    saved = None
    path = Path(checkpoint_path)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        for key, value in (("schema", SCHEMA), ("binding_sha256", binding_sha256),
                           ("defense", defense), ("recipe", r), ("seed", seed)):
            if saved.get(key) != value:
                raise ValueError(f"Repair checkpoint {key} mismatch")
        initial_state = saved["initial_state"]
        model.load_state_dict(initial_state, strict=True)
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    metadata = copy.deepcopy(saved["metadata"]) if saved else {
        "defense": defense, "selection_rule": "final prescribed epoch; no best-epoch or initial-model fallback",
        "pruning": None, "teacher_stages": [], "stage_results": {},
        "cost": {"optimizer_steps": 0, "fit_forwards": 0},
        "resume_policy": "atomic epoch boundaries; workers=0",
        "resume_history": [], "complete": False,
    }
    if saved and saved.get("complete"):
        restore_pruning(model, metadata["pruning"])
        model.load_state_dict(saved["selected_model_state"], strict=True)
        restore_pruning(model, metadata["pruning"])
        model.eval()
        return model, metadata
    if saved:
        metadata["resume_history"].append({"stage": saved["stage"],
                                           "completed_epoch": saved["epoch"]})

    if defense == "fp" and metadata["pruning"] is None:
        scores = rank_head_inputs(model, fit_loader, device, r["fp_layer"])
        order = torch.argsort(scores, stable=True).tolist()
        baseline = clean_selection_loss(model, selection_loader, device)
        candidates, selected = [], None
        increment = max(1, math.ceil(r["fp_once_prune_ratio"] * len(order)))
        for n in range(0, len(order), increment):
            candidate = copy.deepcopy(model)
            pruning = apply_pruning(candidate, r["fp_layer"], order[:n])
            loss = clean_selection_loss(candidate, selection_loader, device)
            admissible = loss <= baseline * (1 + r["utility_tolerance"])
            candidates.append({"fraction": n / len(order), "pruned_count": n,
                               "clean_selection_loss": loss, "admissible": admissible})
            if not admissible:
                break
            selected = pruning
        if selected is None:
            raise ValueError("Even zero pruning failed the clean utility gate")
        metadata["pruning"] = selected
        metadata["pruning_selection"] = {"baseline_loss": baseline,
            "utility_tolerance": r["utility_tolerance"], "candidates": candidates,
            "increment": increment,
            "rule": "last passing count before first clean-utility failure; exact input-column counts"}
    restore_pruning(model, metadata["pruning"])

    teacher = None
    stages = ["teacher", "student"] if defense == "nad" else ["repair"]
    teacher_state = saved.get("teacher_state") if saved else None
    final_selected = None
    for stage in stages:
        if saved and stages.index(stage) < stages.index(saved["stage"]):
            continue
        if stage == "student":
            if teacher_state is None:
                raise ValueError("NAD teacher state missing")
            teacher = copy.deepcopy(model)
            teacher.load_state_dict(teacher_state, strict=True)
            teacher.eval().requires_grad_(False)
            model.load_state_dict(initial_state, strict=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=r["lr"],
            momentum=r["momentum"], weight_decay=r["weight_decay"])
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=r["epochs"])
                     if stage != "student" else None)
        stage_epochs = (r["nad_teacher_epochs"] if stage == "teacher" else
                        r["nad_student_epochs"] if stage == "student" else r["epochs"])
        current = saved if saved and saved["stage"] == stage else None
        if current:
            model.load_state_dict(current["model_state"], strict=True)
            optimizer.load_state_dict(current["optimizer_state"])
            if scheduler is not None:
                scheduler.load_state_dict(current["scheduler_state"])
            _restore_rng(current["rng_state"], loaders)
            start_epoch = current["epoch"] + 1
        else:
            start_epoch = 0
        handles, student_features, teacher_features = [], {}, {}
        if stage == "student":
            # Validate every name before installing any hooks, so a bad recipe
            # cannot leave a partly instrumented source model behind.
            for name in r["nad_layers"]:
                _module(model, name)
                _module(teacher, name)
            for name in r["nad_layers"]:
                def capture(destination, key):
                    def hook(_module, _args, output):
                        if not isinstance(output, torch.Tensor) or output.ndim != 4:
                            raise ValueError(f"NAD layer {key} must emit B x C x H x W")
                        destination[key] = output
                    return hook
                handles.append(_module(model, name).register_forward_hook(capture(student_features, name)))
                handles.append(_module(teacher, name).register_forward_hook(capture(teacher_features, name)))
        try:
            for epoch in range(start_epoch, stage_epochs):
                if stage == "student":
                    stage_lr = r["lr"] if epoch < 2 else 0.01 if epoch < 20 else 0.0001
                    for group in optimizer.param_groups:
                        group["lr"] = stage_lr
                model.train()
                count = 0
                for batch in fit_loader:
                    x, y = _batch(batch, device)
                    if defense == "ft_sam":
                        _sam_step(model, x, y, optimizer, r["sam_radius"], r["sam_adaptive"])
                        scheduler.step()  # pinned FT-SAM steps cosine once per minibatch
                        metadata["cost"]["fit_forwards"] += 2
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        student_features.clear()
                        prediction = pose_prediction(model, x)
                        loss = pose_loss(prediction, y)
                        if stage == "student":
                            teacher_features.clear()
                            with torch.no_grad():
                                pose_prediction(teacher, x)
                            loss = loss + sum(
                                beta * attention_loss(student_features[n], teacher_features[n], r["attention_p"])
                                for n, beta in zip(r["nad_layers"], r["nad_betas"]))
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite clean repair training loss")
                        loss.backward()
                        optimizer.step()
                        metadata["cost"]["fit_forwards"] += 2 if stage == "student" else 1
                    metadata["cost"]["optimizer_steps"] += 1
                    count += len(x)
                if not count:
                    raise ValueError("Clean repair-fit loader is empty")
                if scheduler is not None and defense != "ft_sam":
                    scheduler.step()
                selection = clean_selection_loss(model, selection_loader, device)
                metadata["stage_results"][stage] = {"last_epoch": epoch,
                    "selected_epoch": epoch, "selection_loss": selection,
                    "next_lr": optimizer.param_groups[0]["lr"],
                    "scheduler": ("manual NAD source rule" if scheduler is None else
                                  "cosine per minibatch" if defense == "ft_sam" else "cosine per epoch"),
                    "cosine_T_max": r["epochs"] if scheduler is not None else None}
                final_stage = stage == stages[-1]
                complete = final_stage and epoch == stage_epochs - 1
                metadata["complete"] = complete
                epoch_state = _state(model)
                _atomic_save({"schema": SCHEMA, "binding_sha256": binding_sha256,
                    "defense": defense, "recipe": r, "seed": seed, "stage": stage,
                    "epoch": epoch, "model_state": epoch_state,
                    "initial_state": initial_state, "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "rng_state": _rng_state(loaders),
                    "teacher_state": teacher_state,
                    "selected_model_state": epoch_state if complete else None,
                    "metadata": copy.deepcopy(metadata), "complete": complete}, path)
                print(f"[repair] {defense}/{stage} epoch {epoch + 1}/{stage_epochs}; clean-selection={selection:.8f}", flush=True)
        finally:
            for handle in handles:
                handle.remove()
        final_selected = _state(model)
        if stage == "teacher":
            teacher_state = final_selected
            metadata["teacher_stages"] = [copy.deepcopy(metadata["stage_results"]["teacher"])]
        saved = None
    model.load_state_dict(final_selected, strict=True)
    restore_pruning(model, metadata["pruning"])
    model.eval()
    return model, metadata
