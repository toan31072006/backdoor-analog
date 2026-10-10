"""TRAIN-only, stronger-surrogate fitting for the carrier-bank draft.

The fitted trigger is an attacker-owned artifact, not a replacement victim
optimizer/loss. A fresh victim still learns ordinary full-pose MPJPE ERM.
Two-step momentum lookahead is a bounded approximation: it runs in eval mode
with cloned BN buffers, not exact differentiation through an entire training
run. Clean-utility gates below are surrogate checks, never proof that the fresh
victim will preserve utility or beat Blended.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch.func import functional_call

from learned_carrier_fit import (
    FIT_DEFAULTS, _json_sha, _mpjpe, _ordinary_update, _prediction,
    _relative_energy, _target_pose, resolve_fit_config,
)


BANK_VARIANTS = {"trainaware", "bank", "bank_guard"}
BANK_FIT_DEFAULTS = dict(FIT_DEFAULTS, **{
    "lc_warmup_epochs": 10, "lc_rounds": 3,
    "lc_inner_steps": 128, "lc_outer_steps": 24,
    "lc_surrogate_count": 2, "lc_surrogate_seed_stride": 101,
    "lc_lookahead_steps": 2, "lc_pa_weight": 1.0,
    "lc_pck_weight": 0.1, "lc_pck_temperature": 0.02,
    "lc_utility_mpjpe_tolerance": 0.005,
    "lc_utility_pa_tolerance": 0.003,
    "lc_utility_pck_tolerance": 0.01,
})
PCK_THRESHOLDS = (0.5, 0.4, 0.3, 0.2, 0.1)


def resolve_bank_fit_config(cfg):
    """Validate the separately declared attacker budget without opening data."""
    out = dict(cfg)
    for key, value in BANK_FIT_DEFAULTS.items():
        out.setdefault(key, value)
    out = resolve_fit_config(out)
    if out.get("lc_variant") not in BANK_VARIANTS:
        raise ValueError("carrier-bank fitting needs trainaware, bank, or bank_guard")
    if str(out.get("optimizer", "sgd")).lower() != "sgd":
        raise ValueError("carrier-bank momentum lookahead supports the fixed SGD recipe only")
    for key in ("lc_surrogate_count", "lc_surrogate_seed_stride", "lc_lookahead_steps"):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        out[key] = int(value)
    # Bounded draft computation: an accidental CLI override must not turn this
    # screening protocol into an unbounded model ensemble/unroll.
    if out["lc_surrogate_count"] > 4 or out["lc_lookahead_steps"] > 4:
        raise ValueError("carrier-bank draft permits at most four surrogates/lookahead steps")
    for key in ("lc_pa_weight", "lc_pck_weight", "lc_pck_temperature",
                "lc_utility_mpjpe_tolerance", "lc_utility_pa_tolerance",
                "lc_utility_pck_tolerance"):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{key} must be a finite number")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        out[key] = value
    if out["lc_pck_temperature"] <= 0:
        raise ValueError("lc_pck_temperature must be positive")
    if out["lc_utility_pck_tolerance"] > 1:
        raise ValueError("lc_utility_pck_tolerance is a fraction and must be at most one")
    return out


def _soft_relative_pck(prediction, target, temperature):
    """Differentiable fraction, with exactly MM-Fi's relative reference pair.

    This is an optimization surrogate, not the reported hard-threshold PCK.
    Its five thresholds are relative distances, NOT 50/40/... millimetres.
    """
    if prediction.shape != target.shape or target.shape[-2:] != (17, 3):
        raise ValueError("MM-Fi utility PCK expects matching 17-joint 3D poses")
    flat_pred = prediction.reshape(-1, 17, 3)
    flat_target = target.reshape(-1, 17, 3)
    scale = torch.linalg.vector_norm(flat_target[:, 5] - flat_target[:, 12], dim=-1) + 1e-9
    distance = torch.linalg.vector_norm(flat_pred - flat_target, dim=-1) / scale[:, None]
    thresholds = prediction.new_tensor(PCK_THRESHOLDS)
    return torch.sigmoid((thresholds[:, None, None] - distance[None]) / temperature).mean((1, 2))


def _pa_mpjpe_tensor(prediction, target):
    """Differentiable PA loss with a guarded SVD derivative.

    The alignment convention matches eval.metrics._procrustes. For repeated
    singular values, differentiating rotation is undefined; only those samples
    use a detached SVD rotation while coordinates and scale remain learnable.
    Candidate selection uses the exact NumPy metric, not this soft objective.
    """
    if prediction.shape != target.shape or target.shape[-1] != 3:
        raise ValueError("PA utility needs matching 3D poses")
    pred = prediction.reshape(-1, prediction.shape[-2], 3).to(torch.float64)
    truth = target.reshape_as(pred).to(torch.float64)
    mu_x, mu_y = truth.mean(1, keepdim=True), pred.mean(1, keepdim=True)
    x0, y0 = truth - mu_x, pred - mu_y
    nx = torch.linalg.vector_norm(x0.flatten(1), dim=1)
    ny = torch.linalg.vector_norm(y0.flatten(1), dim=1)
    normalized_x = x0 / (nx[:, None, None] + 1e-12)
    normalized_y = y0 / (ny[:, None, None] + 1e-12)
    covariance = normalized_x.transpose(1, 2) @ normalized_y
    # Inspect spectrum without making the branch choice part of the graph.
    with torch.no_grad():
        spectrum = torch.linalg.svdvals(covariance.detach())
        gaps = (spectrum[:, :-1] - spectrum[:, 1:]).abs()
        regular = ((nx > 1e-10) & (ny > 1e-10)
                   & (gaps.min(1).values > 1e-8))
    rotations, sums = [], []
    for index in range(len(pred)):
        matrix = covariance[index] if bool(regular[index]) else covariance[index].detach()
        u, singular, vh = torch.linalg.svd(matrix)
        v = vh.transpose(-1, -2)
        sign = torch.linalg.det(v @ u.transpose(-1, -2)).sign().detach()
        correction = torch.diag(torch.stack((sign.new_ones(()), sign.new_ones(()), sign)))
        rotation = v @ correction @ u.transpose(-1, -2)
        if not bool(regular[index]):
            # Keep the normalized covariance trace differentiable through the
            # coordinates; avoid repeated-spectrum SVD gradients altogether.
            signed_sum = torch.trace(covariance[index] @ rotation)
        else:
            signed_sum = singular[0] + singular[1] + sign * singular[2]
        rotations.append(rotation)
        sums.append(signed_sum)
    rotation = torch.stack(rotations)
    scale = torch.stack(sums) * nx / (ny + 1e-12)
    aligned = scale[:, None, None] * (y0 @ rotation) + mu_x
    return torch.linalg.vector_norm(aligned - truth, dim=-1).mean().to(prediction.dtype)


def _sgd_lookahead(model, optimizer, trigger, batches, cfg):
    """Differentiate bounded SGD steps without mutating the real optimizer.

    Momentum starts from a DETACHED copy of the actual surrogate SGD state,
    including PyTorch's undampened first momentum update. Weight decay,
    dampening, Nesterov and per-group learning rate follow that optimizer.
    """
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("momentum lookahead requires an SGD optimizer")
    parameters = dict(model.named_parameters())
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    settings, momentum = {}, {}
    by_id = {id(value): name for name, value in parameters.items()}
    for group in optimizer.param_groups:
        if group.get("maximize", False):
            raise ValueError("maximizing SGD is not supported by the HPE lookahead")
        for parameter in group["params"]:
            name = by_id[id(parameter)]
            settings[name] = group
            state = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if state is not None:
                momentum[name] = state.detach().clone()
    if set(settings) != set(parameters):
        raise ValueError("lookahead optimizer must own every surrogate parameter")
    for x, y, dose in batches:
        target, _ = _target_pose(y, dose, cfg)
        attacked = trigger.inject_tensor(x, dose, eps=cfg["eps"])
        output = functional_call(model, (parameters, buffers), (attacked,))
        output = output[0] if isinstance(output, tuple) else output
        loss = _mpjpe(output, target)
        gradients = torch.autograd.grad(loss, tuple(parameters.values()),
                                        create_graph=True, allow_unused=True)
        updated = {}
        for (name, parameter), gradient in zip(parameters.items(), gradients):
            if gradient is None:
                updated[name] = parameter
                continue
            group = settings[name]
            direction = gradient + float(group.get("weight_decay", 0.0)) * parameter
            coefficient = float(group.get("momentum", 0.0))
            if coefficient:
                if name not in momentum:
                    momentum[name] = direction
                else:
                    momentum[name] = (coefficient * momentum[name]
                                      + (1.0 - float(group.get("dampening", 0.0))) * direction)
                direction = (direction + coefficient * momentum[name]
                             if group.get("nesterov", False) else momentum[name])
            updated[name] = parameter - float(group["lr"]) * direction
        parameters = updated
    return parameters


def _bank_objective(model, optimizer, anchor, trigger, inner_batches,
                    outer_x, outer_y, outer_dose, cfg):
    updated = _sgd_lookahead(model, optimizer, trigger, inner_batches, cfg)
    attacked = trigger.inject_tensor(outer_x, outer_dose, eps=cfg["eps"])
    target, joints = _target_pose(outer_y, outer_dose, cfg)
    target_loss = _mpjpe(_prediction(model, attacked, updated)[..., joints, :], target[..., joints, :])
    clean_prediction = _prediction(model, outer_x, updated)
    clean_loss = _mpjpe(clean_prediction, outer_y)
    energy = _relative_energy(attacked, outer_x)
    objective = target_loss + cfg["lc_clean_weight"] * clean_loss + cfg["lc_energy_weight"] * energy
    details = {"target": target_loss, "clean": clean_loss, "energy": energy}
    if cfg["lc_variant"] == "bank_guard":
        with torch.no_grad():
            anchor_prediction = _prediction(anchor, outer_x)
            anchor_clean = _mpjpe(anchor_prediction, outer_y)
            anchor_pa = _pa_mpjpe_tensor(anchor_prediction, outer_y)
            anchor_pck = _soft_relative_pck(anchor_prediction, outer_y, cfg["lc_pck_temperature"])
        pa = _pa_mpjpe_tensor(clean_prediction, outer_y)
        soft_pck = _soft_relative_pck(clean_prediction, outer_y, cfg["lc_pck_temperature"])
        mpjpe_guard = torch.relu(clean_loss - anchor_clean - cfg["lc_utility_mpjpe_tolerance"])
        pa_guard = torch.relu(pa - anchor_pa - cfg["lc_utility_pa_tolerance"])
        pck_guard = torch.relu(anchor_pck - soft_pck - cfg["lc_utility_pck_tolerance"]).mean()
        objective = (objective + cfg["lc_clean_weight"] * mpjpe_guard
                     + cfg["lc_pa_weight"] * pa_guard + cfg["lc_pck_weight"] * pck_guard)
        details.update(clean_guard=mpjpe_guard, pa=pa, pa_guard=pa_guard, soft_pck_guard=pck_guard)
    return objective, details


def _exact_clean_metrics(prediction, target):
    from eval import metrics as metrics
    pred = prediction.detach().cpu().numpy().reshape(-1, 17, 3).astype(np.float64)
    truth = target.detach().cpu().numpy().reshape(-1, 17, 3).astype(np.float64)
    result = {"clean_m": float(metrics.mpjpe(pred, truth).mean()),
              "pa_m": float(metrics.pa_mpjpe(pred, truth).mean()),
              "pck": {str(value): float(metrics.pck(pred, truth, value, ref=(5, 12)))
                      for value in PCK_THRESHOLDS}}
    if (not math.isfinite(result["clean_m"]) or not math.isfinite(result["pa_m"])
            or not all(math.isfinite(value) for value in result["pck"].values())):
        raise ValueError("exact INNER-validation utility metrics became non-finite")
    return result


def _utility_gate(metrics, reference, cfg):
    """All three clean metrics, including every PCK threshold, must pass."""
    deltas = {"mpjpe_m": metrics["clean_m"] - reference["clean_m"],
              "pa_mpjpe_m": metrics["pa_m"] - reference["pa_m"],
              "pck": {key: metrics["pck"][key] - reference["pck"][key]
                      for key in reference["pck"]}}
    checks = {"mpjpe": deltas["mpjpe_m"] <= cfg["lc_utility_mpjpe_tolerance"],
              "pa_mpjpe": deltas["pa_mpjpe_m"] <= cfg["lc_utility_pa_tolerance"],
              "pck": {key: value >= -cfg["lc_utility_pck_tolerance"]
                      for key, value in deltas["pck"].items()}}
    return {"passed": bool(checks["mpjpe"] and checks["pa_mpjpe"] and all(checks["pck"].values())),
            "checks": checks, "deltas": deltas}


def _evaluate_bank(models, trigger, val_x, val_y, references, cfg):
    positive = [float(d) for d in cfg["dose_grid"] if float(d) > 0]
    if not positive or len(models) != len(references):
        raise ValueError("INNER-validation needs positive doses and one reference per surrogate")
    records = []
    with torch.no_grad():
        for model, reference in zip(models, references):
            model.eval()
            predictions, targets, target_sum = [], [], 0.0
            for begin in range(0, len(val_x), cfg["lc_batch_size"]):
                x = val_x[begin:begin + cfg["lc_batch_size"]]
                y = val_y[begin:begin + cfg["lc_batch_size"]]
                predictions.append(_prediction(model, x))
                targets.append(y)
                for value in positive:
                    dose = x.new_full((len(x),), value)
                    attacked = trigger.inject_tensor(x, dose, eps=cfg["eps"])
                    target, joints = _target_pose(y, dose, cfg)
                    target_sum += len(x) * float(_mpjpe(
                        _prediction(model, attacked)[..., joints, :], target[..., joints, :]))
            record = _exact_clean_metrics(torch.cat(predictions), torch.cat(targets))
            record["target_m"] = target_sum / (len(val_x) * len(positive))
            record["utility_gate"] = _utility_gate(record, reference, cfg)
            records.append(record)
    result = {"target_m": float(np.mean([item["target_m"] for item in records])),
              "clean_m": float(np.mean([item["clean_m"] for item in records])),
              "pa_m": float(np.mean([item["pa_m"] for item in records])),
              "pck": {str(value): float(np.mean([item["pck"][str(value)] for item in records]))
                      for value in PCK_THRESHOLDS},
              "per_surrogate": records,
              "utility_gate_passed": all(item["utility_gate"]["passed"] for item in records)}
    result["score"] = result["target_m"] + cfg["lc_clean_weight"] * result["clean_m"]
    if not math.isfinite(result["score"]):
        raise ValueError("carrier-bank INNER-validation score became non-finite")
    return result


def fit_bank_trigger(models, trigger, fit_x, fit_y, val_x, val_y, cfg, rng=None):
    """Fit only supplied TRAIN tensors; preserve all round scores and failures.

    ``models`` are independently initialized attacker surrogates. Their weights
    are neither exported nor handed to the fresh victim. Utility protection is
    mandatory only for bank_guard; a no-eligible-candidate fallback is clearly
    flagged and still evaluated so its failure is not silently hidden.
    """
    cfg = resolve_bank_fit_config(cfg)
    models = [models] if isinstance(models, torch.nn.Module) else list(models)
    if len(models) != cfg["lc_surrogate_count"] or len({id(model) for model in models}) != len(models):
        raise ValueError("supply one distinct model for each declared surrogate")
    if len(fit_x) < 2 or len(val_x) < 1:
        raise ValueError("fitting needs disjoint nonempty inner TRAIN/validation tensors")
    if fit_x.shape[0] != fit_y.shape[0] or val_x.shape[0] != val_y.shape[0]:
        raise ValueError("each fitting CSI frame needs exactly one pose target")
    if (fit_x.device != val_x.device or fit_x.data_ptr() == val_x.data_ptr()
            or fit_y.data_ptr() == val_y.data_ptr()):
        raise ValueError("inner TRAIN/validation tensor storage must be disjoint on one device")
    from train_backdoor import _build_optimizer
    optimizers = [_build_optimizer(model, cfg, "mmfi")[1] for model in models]
    carrier_optimizer = torch.optim.Adam([value for value in trigger.parameters() if value.requires_grad], lr=cfg["lc_lr"])
    rng = np.random.default_rng(cfg["lc_fit_seed"]) if rng is None else rng
    batch_size, n = cfg["lc_batch_size"], len(fit_x)
    count = int(math.floor(float(cfg["rho"]) * n))
    if count < 1:
        raise ValueError("attacker inner TRAIN pool too small for the requested poison rate")
    plan_rng = np.random.default_rng(int(cfg.get("seed", 42)))
    chosen = plan_rng.choice(n, count, replace=False)
    plan_doses = np.zeros(n, dtype=np.float32)
    plan_doses[chosen] = plan_rng.uniform(cfg["dose_min"], cfg["dose_max"], count)
    fixed_doses = torch.from_numpy(plan_doses).to(fit_x.device)
    history = []
    for epoch in range(cfg["lc_warmup_epochs"]):
        order, losses = rng.permutation(n), [[] for _ in models]
        for begin in range(0, n, batch_size):
            index = torch.as_tensor(order[begin:begin + batch_size], device=fit_x.device)
            for member, (model, optimizer) in enumerate(zip(models, optimizers)):
                losses[member].append(_ordinary_update(model, optimizer, None,
                    fit_x[index], fit_y[index], fixed_doses[index], cfg))
        means = [float(np.mean(values)) for values in losses]
        history.append({"stage": "warmup", "epoch": epoch, "surrogate_loss_m": means})
        print(f"[bank-fit] warmup {epoch + 1}/{cfg['lc_warmup_epochs']}: {means}", flush=True)
    anchors = [copy.deepcopy(model).eval().requires_grad_(False) for model in models]
    references = []
    with torch.no_grad():
        for anchor in anchors:
            prediction = torch.cat([_prediction(anchor, val_x[begin:begin + batch_size])
                                    for begin in range(0, len(val_x), batch_size)])
            references.append(_exact_clean_metrics(prediction, val_y))

    best_eligible, best_fallback = None, None

    def sample():
        index = torch.as_tensor(rng.choice(n, min(batch_size, n), replace=False), device=fit_x.device)
        return fit_x[index], fit_y[index], fixed_doses[index]

    for round_number in range(cfg["lc_rounds"]):
        inner_losses = [[] for _ in models]
        for _ in range(cfg["lc_inner_steps"]):
            x, y, dose = sample()
            for member, (model, optimizer) in enumerate(zip(models, optimizers)):
                inner_losses[member].append(_ordinary_update(model, optimizer, trigger, x, y, dose, cfg))
        for model in models:
            model.eval()
        outer_losses, components = [], []
        for _ in range(cfg["lc_outer_steps"]):
            inner_batches = [sample() for _ in range(cfg["lc_lookahead_steps"])]
            outer_x, outer_y, _ = sample()
            outer_dose = torch.as_tensor(rng.uniform(cfg["dose_min"], cfg["dose_max"], len(outer_x)),
                                         dtype=outer_x.dtype, device=outer_x.device)
            carrier_optimizer.zero_grad(set_to_none=True)
            member_details = []
            # Backprop each ensemble member separately to release its unroll
            # graph before allocating the next; the update uses their average.
            for model, optimizer, anchor in zip(models, optimizers, anchors):
                model.zero_grad(set_to_none=True)
                objective, details = _bank_objective(model, optimizer, anchor, trigger,
                    inner_batches, outer_x, outer_y, outer_dose, cfg)
                if not torch.isfinite(objective):
                    raise ValueError("carrier-bank outer objective became non-finite")
                (objective / len(models)).backward()
                member_details.append({key: float(value.detach()) for key, value in details.items()})
                outer_losses.append(float(objective.detach()))
            for value in trigger.parameters():
                if value.grad is not None and not torch.isfinite(value.grad).all():
                    raise ValueError("carrier-bank gradient became non-finite")
            torch.nn.utils.clip_grad_norm_(trigger.parameters(), 5.0)
            carrier_optimizer.step()
            components.append({key: float(np.mean([member[key] for member in member_details]))
                               for key in member_details[0]})
        # Scoring follows real SGD adaptation to the UPDATED carrier, not a
        # stale clean model or the differentiable lookahead parameters.
        for _ in range(cfg["lc_inner_steps"]):
            x, y, dose = sample()
            for member, (model, optimizer) in enumerate(zip(models, optimizers)):
                inner_losses[member].append(_ordinary_update(model, optimizer, trigger, x, y, dose, cfg))
        validation = _evaluate_bank(models, trigger, val_x, val_y, references, cfg)
        history.append({"stage": "alternation", "round": round_number,
                        "surrogate_inner_loss_m": [float(np.mean(values)) for values in inner_losses],
                        "outer_loss": float(np.mean(outer_losses)), "outer_components": components,
                        "inner_validation": validation})
        candidate = {"round": round_number, "score": validation["score"],
                     "utility_gate_passed": validation["utility_gate_passed"],
                     "state": {name: value.detach().cpu().clone() for name, value in trigger.state_dict().items()}}
        if best_fallback is None or candidate["score"] < best_fallback["score"]:
            best_fallback = candidate
        if candidate["utility_gate_passed"] and (best_eligible is None or candidate["score"] < best_eligible["score"]):
            best_eligible = candidate
        print(f"[bank-fit] {cfg['lc_variant']} round {round_number + 1}/{cfg['lc_rounds']}: "
              f"INNER-val T={validation['target_m']:.5f} clean={validation['clean_m']:.5f} "
              f"utility_gate={validation['utility_gate_passed']}", flush=True)
    required = cfg["lc_variant"] == "bank_guard"
    selected = best_eligible if required and best_eligible is not None else best_fallback
    trigger.load_state_dict(selected["state"])
    return {"history": history, "selected_round": selected["round"], "selected_score": selected["score"],
            "selected_utility_gate_passed": selected["utility_gate_passed"],
            "utility_gate_required": required,
            "no_eligible_candidate": bool(required and best_eligible is None),
            "utility_reference": references,
            "utility_tolerances": {key: cfg[key] for key in (
                "lc_utility_mpjpe_tolerance", "lc_utility_pa_tolerance", "lc_utility_pck_tolerance")},
            "surrogate_initialization_seeds": [cfg["lc_fit_seed"] + index * cfg["lc_surrogate_seed_stride"]
                                                 for index in range(len(models))],
            "surrogate_count": len(models), "inner_poison_count": count,
            "surrogate_actual_erm_steps_per_member": (
                cfg["lc_warmup_epochs"] * math.ceil(n / batch_size)
                + 2 * cfg["lc_rounds"] * cfg["lc_inner_steps"]),
            "attacker_outer_steps": cfg["lc_rounds"] * cfg["lc_outer_steps"],
            "outer_dose_distribution": "paired Uniform(dose_min,dose_max), shared across surrogate ensemble",
            "surrogate_batch_pairing": "same inner/outer TRAIN identities and doses for every ensemble member",
            "inner_poison_plan_sha256": _json_sha([[int(i), float(plan_doses[i])] for i in chosen]),
            "lookahead": f"{cfg['lc_lookahead_steps']}-step differentiable SGD with detached real optimizer momentum; cloned eval-mode BN buffers",
            "lookahead_is_full_training_bilevel": False,
            "surrogate_optimizer": "sgd",
            "candidate_selection": "inner validation mean positive-dose target MPJPE + weighted clean MPJPE; bank_guard additionally requires per-surrogate exact clean gates",
            "utility_scope": "frozen clean-warmup anchor; surrogate gates only, fresh-victim metrics must be independently verified",
            "pck_units": "relative to MMFi reference distance (5,12), not millimetres"}
