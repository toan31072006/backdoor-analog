"""TRAIN-only carrier fitting against a stage-matched clean SGD twin.

Each independently initialized surrogate is forked after clean warmup, with
its complete SGD state. The twin then sees the same real and virtual batch
identities as its poisoned partner, but only raw CSI and raw pose targets.
These are attacker surrogate checks; no weights are transferred to a victim.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch
from torch.func import functional_call

from carrier_bank_fit import (
    BANK_FIT_DEFAULTS, PCK_THRESHOLDS, _exact_clean_metrics, _pa_mpjpe_tensor,
    _sgd_lookahead, _soft_relative_pck, _utility_gate, resolve_bank_fit_config,
)
from learned_carrier_fit import (
    _json_sha, _mpjpe, _ordinary_update, _prediction, _relative_energy,
    _target_pose,
)


PAIRED_PROFILE = "paired_guard_screen_v1"
_TOLERANCE_KEYS = (
    "lc_utility_mpjpe_tolerance", "lc_utility_pa_tolerance",
    "lc_utility_pck_tolerance",
)
PAIRED_FIT_DEFAULTS = dict(BANK_FIT_DEFAULTS, **{
    key: 0.0 for key in _TOLERANCE_KEYS
})
_GUARD_KEYS = {
    "mpjpe": "clean_guard", "pa_mpjpe": "pa_guard",
    **{f"pck_{threshold}": f"soft_pck_guard_{threshold}"
       for threshold in PCK_THRESHOLDS},
}


def resolve_paired_fit_config(cfg):
    """Resolve the declared TRAIN-only protocol with immutable zero gates."""
    out = dict(cfg)
    if (out.get("lc_variant") != "paired_guard"
            or out.get("draft_profile") != PAIRED_PROFILE
            or out.get("experiment_name") != "mmfi"
            or out.get("draft_eval_source") != "training_holdout"
            or out.get("method_draft") is not True):
        raise ValueError("paired_guard fitting requires the paired_guard_screen_v1 MMFi TRAIN-only draft profile")
    for key, value in PAIRED_FIT_DEFAULTS.items():
        out.setdefault(key, value)
    # Reuse existing validation without teaching the bank fitter a new path.
    out = resolve_bank_fit_config(dict(out, lc_variant="trainaware"))
    out["lc_variant"] = "paired_guard"
    if any(out[key] != 0.0 for key in _TOLERANCE_KEYS):
        raise ValueError("paired_guard clean-utility tolerances are immutable zero gates")
    if out["lc_surrogate_count"] != 2 or out["lc_lookahead_steps"] != 2:
        raise ValueError("paired_guard requires exactly two surrogates and two lookahead steps")
    if out["lc_fit_seed"] != 4242 or out["lc_surrogate_seed_stride"] != 101:
        raise ValueError("paired_guard requires surrogate initialization seeds 4242 and 4343")
    if any(out[key] <= 0 for key in ("lc_clean_weight", "lc_pa_weight", "lc_pck_weight")):
        raise ValueError("paired_guard requires positive weights for all seven clean guards")
    return out


def _tensor_sha256(value):
    tensor = value.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return _json_sha({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                      "bytes_sha256": hashlib.sha256(raw).hexdigest()})


def _model_sha256(model):
    return _json_sha({name: _tensor_sha256(value)
                      for name, value in model.state_dict().items()})


def _optimizer_signature(model, optimizer, momentum_only=False):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    state = {}
    for parameter, values in optimizer.state.items():
        state[names[id(parameter)]] = {
            key: _tensor_sha256(value) if torch.is_tensor(value) else value
            for key, value in values.items()
            if not momentum_only or key == "momentum_buffer"
        }
    if momentum_only:
        return _json_sha(state)
    groups = [{key: ([names[id(parameter)] for parameter in value]
                     if key == "params" else value)
               for key, value in group.items()} for group in optimizer.param_groups]
    return _json_sha({"state": state, "param_groups": groups})


def _clone_clean_twin(model, optimizer):
    """Clone model/buffers and SGD groups/state while preserving ownership."""
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("paired clean twins require SGD")
    twin = copy.deepcopy(model)
    mapping = {id(source): destination
               for source, destination in zip(model.parameters(), twin.parameters())}
    groups = [{key: ([mapping[id(parameter)] for parameter in value]
                     if key == "params" else copy.deepcopy(value))
               for key, value in group.items()} for group in optimizer.param_groups]
    twin_optimizer = torch.optim.SGD(groups, lr=optimizer.defaults["lr"])
    twin_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    proof = {
        "warmup_model_sha256": _model_sha256(model),
        "clean_twin_model_sha256": _model_sha256(twin),
        "warmup_optimizer_sha256": _optimizer_signature(model, optimizer),
        "clean_twin_optimizer_sha256": _optimizer_signature(twin, twin_optimizer),
        "warmup_momentum_sha256": _optimizer_signature(model, optimizer, True),
        "clean_twin_momentum_sha256": _optimizer_signature(twin, twin_optimizer, True),
        "parameter_storage_independent": all(
            source.data_ptr() != destination.data_ptr()
            for source, destination in zip(model.parameters(), twin.parameters())),
    }
    proof["model_state_equal"] = proof["warmup_model_sha256"] == proof["clean_twin_model_sha256"]
    proof["optimizer_state_equal"] = proof["warmup_optimizer_sha256"] == proof["clean_twin_optimizer_sha256"]
    proof["momentum_state_equal"] = proof["warmup_momentum_sha256"] == proof["clean_twin_momentum_sha256"]
    if not all(proof[key] for key in ("model_state_equal", "optimizer_state_equal",
                                    "momentum_state_equal", "parameter_storage_independent")):
        raise ValueError("clean twin model/SGD clone failed its state proof")
    return twin, twin_optimizer, proof


def _paired_ordinary_updates(model, optimizer, twin, twin_optimizer,
                              trigger, x, y, dose, cfg):
    """Pair dropout draws without advancing the global RNG stream twice."""
    cpu_before = torch.random.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(x.device) if x.is_cuda else None
    poisoned_loss = _ordinary_update(model, optimizer, trigger, x, y, dose, cfg)
    cpu_after = torch.random.get_rng_state()
    cuda_after = torch.cuda.get_rng_state(x.device) if x.is_cuda else None
    torch.random.set_rng_state(cpu_before)
    if cuda_before is not None:
        torch.cuda.set_rng_state(cuda_before, x.device)
    try:
        clean_loss = _ordinary_update(twin, twin_optimizer, None,
                                      x, y, x.new_zeros(len(x)), cfg)
        clean_cpu_after = torch.random.get_rng_state()
        clean_cuda_after = torch.cuda.get_rng_state(x.device) if x.is_cuda else None
    finally:
        torch.random.set_rng_state(cpu_after)
        if cuda_after is not None:
            torch.cuda.set_rng_state(cuda_after, x.device)
    proof = {
        "cpu_before_sha256": _tensor_sha256(cpu_before),
        "cpu_after_sha256": _tensor_sha256(cpu_after),
        "cpu_draws_matched": bool(torch.equal(cpu_after, clean_cpu_after)),
        "cuda_before_sha256": _tensor_sha256(cuda_before) if cuda_before is not None else None,
        "cuda_after_sha256": _tensor_sha256(cuda_after) if cuda_after is not None else None,
        "cuda_draws_matched": (bool(torch.equal(cuda_after, clean_cuda_after))
                               if cuda_after is not None else True),
    }
    return poisoned_loss, clean_loss, proof


def _clean_sgd_lookahead(model, optimizer, batches, cfg=None):
    """Simulate ordinary clean SGD on raw x/y, without trigger/payload calls.

    Real parameters, momentum and BN buffers stay unchanged. Local parameters
    and local gradients are detached between steps because the twin is a
    reference, never a branch through which carrier gradients should flow.
    A supplied third batch item must contain only zero doses.
    """
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("clean momentum lookahead requires SGD")
    source = dict(model.named_parameters())
    parameters = {name: value.detach().clone().requires_grad_(True)
                  for name, value in source.items()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    by_id = {id(value): name for name, value in source.items()}
    settings, momentum = {}, {}
    for group in optimizer.param_groups:
        if group.get("maximize", False):
            raise ValueError("maximizing SGD is not supported by clean HPE lookahead")
        for parameter in group["params"]:
            name = by_id[id(parameter)]
            settings[name] = group
            value = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if value is not None:
                momentum[name] = value.detach().clone()
    if set(settings) != set(parameters):
        raise ValueError("clean lookahead optimizer must own every twin parameter")
    for batch in batches:
        x, y = batch[:2]
        if len(batch) == 3 and torch.count_nonzero(batch[2]).item():
            raise ValueError("clean twin lookahead accepts only zero doses")
        output = functional_call(model, (parameters, buffers), (x,))
        output = output[0] if isinstance(output, tuple) else output
        gradients = torch.autograd.grad(_mpjpe(output, y), tuple(parameters.values()),
                                        allow_unused=True)
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
                    momentum[name] = direction.detach()
                else:
                    momentum[name] = (coefficient * momentum[name]
                                      + (1.0 - float(group.get("dampening", 0.0))) * direction.detach())
                direction = (direction + coefficient * momentum[name]
                             if group.get("nesterov", False) else momentum[name])
            updated[name] = (parameter - float(group["lr"]) * direction).detach().requires_grad_(True)
        parameters = updated
    return {name: value.detach() for name, value in parameters.items()}


def _paired_guard_components(prediction, clean_reference, target, cfg):
    """Seven independent hinges against a detached looked-ahead clean twin."""
    clean_reference = clean_reference.detach()
    clean = _mpjpe(prediction, target)
    pa = _pa_mpjpe_tensor(prediction, target)
    pck = _soft_relative_pck(prediction, target, cfg["lc_pck_temperature"])
    with torch.no_grad():
        reference_clean = _mpjpe(clean_reference, target)
        reference_pa = _pa_mpjpe_tensor(clean_reference, target)
        reference_pck = _soft_relative_pck(clean_reference, target, cfg["lc_pck_temperature"])
    pck_hinges = torch.relu(reference_pck - pck)
    details = {
        "clean": clean, "pa": pa,
        "clean_twin_clean": reference_clean, "clean_twin_pa": reference_pa,
        "clean_guard": torch.relu(clean - reference_clean),
        "pa_guard": torch.relu(pa - reference_pa),
        "soft_pck_guard": pck_hinges.mean(),
    }
    for index, threshold in enumerate(PCK_THRESHOLDS):
        details[f"soft_pck_{threshold}"] = pck[index]
        details[f"clean_twin_soft_pck_{threshold}"] = reference_pck[index]
        details[f"soft_pck_guard_{threshold}"] = pck_hinges[index]
    return details


def _paired_objective(model, optimizer, twin, twin_optimizer, trigger,
                      inner_batches, outer_x, outer_y, outer_dose, cfg):
    updated = _sgd_lookahead(model, optimizer, trigger, inner_batches, cfg)
    clean_batches = [(x, y, x.new_zeros(len(x))) for x, y, _ in inner_batches]
    twin_updated = _clean_sgd_lookahead(twin, twin_optimizer, clean_batches, cfg)
    with torch.no_grad():
        clean_reference = _prediction(twin, outer_x, twin_updated).detach()
    attacked = trigger.inject_tensor(outer_x, outer_dose, eps=cfg["eps"])
    target, joints = _target_pose(outer_y, outer_dose, cfg)
    target_loss = _mpjpe(_prediction(model, attacked, updated)[..., joints, :],
                         target[..., joints, :])
    details = _paired_guard_components(_prediction(model, outer_x, updated),
                                       clean_reference, outer_y, cfg)
    energy = _relative_energy(attacked, outer_x)
    objective = (target_loss + cfg["lc_clean_weight"] * details["clean"]
                 + cfg["lc_energy_weight"] * energy
                 + cfg["lc_clean_weight"] * details["clean_guard"]
                 + cfg["lc_pa_weight"] * details["pa_guard"]
                 + cfg["lc_pck_weight"] * details["soft_pck_guard"])
    details.update(target=target_loss, energy=energy)
    return objective, details


def _evaluate_paired(models, twins, trigger, val_x, val_y, cfg):
    """Exact INTERNAL validation gates against each current-stage clean twin."""
    positive = [float(dose) for dose in cfg["dose_grid"] if float(dose) > 0]
    if not positive or len(models) != len(twins):
        raise ValueError("paired INNER-validation needs positive doses and one clean twin per surrogate")
    records = []
    with torch.no_grad():
        for model, twin in zip(models, twins):
            model.eval()
            twin.eval()
            predictions, references, targets, target_sum = [], [], [], 0.0
            for begin in range(0, len(val_x), cfg["lc_batch_size"]):
                x = val_x[begin:begin + cfg["lc_batch_size"]]
                y = val_y[begin:begin + cfg["lc_batch_size"]]
                predictions.append(_prediction(model, x))
                references.append(_prediction(twin, x))
                targets.append(y)
                for value in positive:
                    dose = x.new_full((len(x),), value)
                    attacked = trigger.inject_tensor(x, dose, eps=cfg["eps"])
                    target, joints = _target_pose(y, dose, cfg)
                    target_sum += len(x) * float(_mpjpe(
                        _prediction(model, attacked)[..., joints, :], target[..., joints, :]))
            truth = torch.cat(targets)
            reference = _exact_clean_metrics(torch.cat(references), truth)
            record = _exact_clean_metrics(torch.cat(predictions), truth)
            record["target_m"] = target_sum / (len(val_x) * len(positive))
            record["clean_twin_reference"] = reference
            record["clean_twin_state_sha256"] = _model_sha256(twin)
            record["poison_surrogate_state_sha256"] = _model_sha256(model)
            record["utility_gate"] = _utility_gate(record, reference, cfg)
            records.append(record)
    result = {
        "target_m": float(np.mean([item["target_m"] for item in records])),
        "clean_m": float(np.mean([item["clean_m"] for item in records])),
        "pa_m": float(np.mean([item["pa_m"] for item in records])),
        "pck": {str(value): float(np.mean([item["pck"][str(value)] for item in records]))
                for value in PCK_THRESHOLDS},
        "per_surrogate": records,
        "utility_gate_passed": all(item["utility_gate"]["passed"] for item in records),
    }
    result["score"] = result["target_m"] + cfg["lc_clean_weight"] * result["clean_m"]
    if not math.isfinite(result["score"]):
        raise ValueError("paired INNER-validation score became non-finite")
    return result


def fit_paired_trigger(models, trigger, fit_x, fit_y, val_x, val_y, cfg, rng=None):
    """Fit a carrier on TRAIN tensors; export every gate and failure honestly."""
    cfg = resolve_paired_fit_config(cfg)
    models = [models] if isinstance(models, torch.nn.Module) else list(models)
    if len(models) != 2 or len({id(model) for model in models}) != 2:
        raise ValueError("supply two distinct paired_guard surrogate models")
    if len(fit_x) < 2 or len(val_x) < 1:
        raise ValueError("paired fitting needs disjoint nonempty inner TRAIN/validation tensors")
    if fit_x.shape[0] != fit_y.shape[0] or val_x.shape[0] != val_y.shape[0]:
        raise ValueError("each fitting CSI frame needs exactly one pose target")
    if (fit_x.device != val_x.device or fit_x.data_ptr() == val_x.data_ptr()
            or fit_y.data_ptr() == val_y.data_ptr()):
        raise ValueError("inner TRAIN/validation tensor storage must be disjoint on one device")
    from train_backdoor import _build_optimizer
    optimizers = [_build_optimizer(model, cfg, "mmfi")[1] for model in models]
    carrier_optimizer = torch.optim.Adam(
        [value for value in trigger.parameters() if value.requires_grad], lr=cfg["lc_lr"])
    rng = np.random.default_rng(cfg["lc_fit_seed"]) if rng is None else rng
    batch_size, n = cfg["lc_batch_size"], len(fit_x)
    count = int(math.floor(float(cfg["rho"]) * n))
    if count < 1:
        raise ValueError("attacker inner TRAIN pool too small for the requested poison rate")
    # Identical local poison identities and doses to the existing trainaware fit.
    plan_rng = np.random.default_rng(int(cfg.get("seed", 42)))
    chosen = plan_rng.choice(n, count, replace=False)
    plan_doses = np.zeros(n, dtype=np.float32)
    plan_doses[chosen] = plan_rng.uniform(cfg["dose_min"], cfg["dose_max"], count)
    fixed_doses = torch.from_numpy(plan_doses).to(fit_x.device)
    history, warmup_log, real_log, virtual_log = [], [], [], []
    surrogate_updates = [0, 0]
    twin_updates, virtual_updates = [0, 0], [0, 0]
    activations = {key: 0 for key in _GUARD_KEYS}
    per_member_activations = [dict(activations), dict(activations)]
    for epoch in range(cfg["lc_warmup_epochs"]):
        order, losses, epoch_log = rng.permutation(n), [[], []], []
        for begin in range(0, n, batch_size):
            values = order[begin:begin + batch_size]
            index = torch.as_tensor(values, device=fit_x.device)
            descriptor = {"epoch": epoch, "batch": begin // batch_size,
                          "indices_sha256": _json_sha(values.tolist()), "size": len(values)}
            warmup_log.append(descriptor)
            epoch_log.append(descriptor)
            for member, (model, optimizer) in enumerate(zip(models, optimizers)):
                losses[member].append(_ordinary_update(model, optimizer, None,
                    fit_x[index], fit_y[index], fit_x.new_zeros(len(index)), cfg))
                surrogate_updates[member] += 1
        means = [float(np.mean(values)) for values in losses]
        history.append({"stage": "warmup", "epoch": epoch, "surrogate_loss_m": means,
                        "batch_identity_sha256": _json_sha(epoch_log),
                        "real_updates_per_member": len(epoch_log)})
        print(f"[paired-fit] warmup {epoch + 1}/{cfg['lc_warmup_epochs']}: {means}", flush=True)
    twins, twin_optimizers, clone_proofs = [], [], []
    for member, (model, optimizer) in enumerate(zip(models, optimizers)):
        twin, twin_optimizer, proof = _clone_clean_twin(model, optimizer)
        proof.update(member=member, seed=4242 + member * 101,
                     warmup_real_updates=surrogate_updates[member],
                     shared_warmup_batch_sha256=_json_sha(warmup_log))
        twins.append(twin)
        twin_optimizers.append(twin_optimizer)
        clone_proofs.append(proof)
    best_eligible, best_fallback = None, None

    def sample():
        values = rng.choice(n, min(batch_size, n), replace=False)
        index = torch.as_tensor(values, device=fit_x.device)
        return (fit_x[index], fit_y[index], fixed_doses[index]), {
            "indices_sha256": _json_sha(values.tolist()), "size": len(values),
            "poison_doses_sha256": _json_sha(plan_doses[values].tolist()),
            "clean_doses_sha256": _json_sha([0.0] * len(values)),
        }

    def paired_real_update(round_number, stage, step, losses, clean_losses):
        (x, y, dose), descriptor = sample()
        descriptor.update(round=round_number, stage=stage, step=step)
        real_log.append(descriptor)
        descriptor["paired_rng_per_surrogate"] = []
        for member, (model, optimizer, twin, twin_optimizer) in enumerate(
                zip(models, optimizers, twins, twin_optimizers)):
            poisoned_loss, clean_loss, rng_proof = _paired_ordinary_updates(
                model, optimizer, twin, twin_optimizer, trigger, x, y, dose, cfg)
            losses[member].append(poisoned_loss)
            clean_losses[member].append(clean_loss)
            descriptor["paired_rng_per_surrogate"].append(rng_proof)
            surrogate_updates[member] += 1
            twin_updates[member] += 1

    for round_number in range(cfg["lc_rounds"]):
        inner_losses, clean_losses = [[], []], [[], []]
        real_begin, virtual_begin = len(real_log), len(virtual_log)
        round_activations = {key: 0 for key in _GUARD_KEYS}
        for step in range(cfg["lc_inner_steps"]):
            paired_real_update(round_number, "inner_before", step, inner_losses, clean_losses)
        for model in models + twins:
            model.eval()
        outer_losses, components, member_components = [], [], []
        for outer_step in range(cfg["lc_outer_steps"]):
            inner_batches = []
            for depth in range(cfg["lc_lookahead_steps"]):
                batch, descriptor = sample()
                inner_batches.append(batch)
                descriptor.update(round=round_number, outer_step=outer_step, depth=depth,
                                  virtual=True)
                virtual_log.append(descriptor)
            (outer_x, outer_y, _), outer_identity = sample()
            outer_dose = torch.as_tensor(
                rng.uniform(cfg["dose_min"], cfg["dose_max"], len(outer_x)),
                dtype=outer_x.dtype, device=outer_x.device)
            carrier_optimizer.zero_grad(set_to_none=True)
            details_per_member = []
            for member, (model, optimizer, twin, twin_optimizer) in enumerate(
                    zip(models, optimizers, twins, twin_optimizers)):
                model.zero_grad(set_to_none=True)
                twin.zero_grad(set_to_none=True)
                objective, details = _paired_objective(model, optimizer, twin, twin_optimizer,
                    trigger, inner_batches, outer_x, outer_y, outer_dose, cfg)
                if not torch.isfinite(objective):
                    raise ValueError("paired carrier outer objective became non-finite")
                (objective / len(models)).backward()
                details_per_member.append({key: float(value.detach()) for key, value in details.items()})
                outer_losses.append(float(objective.detach()))
                virtual_updates[member] += cfg["lc_lookahead_steps"]
                for guard, key in _GUARD_KEYS.items():
                    active = int(float(details[key].detach()) > 0)
                    activations[guard] += active
                    round_activations[guard] += active
                    per_member_activations[member][guard] += active
            for value in trigger.parameters():
                if value.grad is not None and not torch.isfinite(value.grad).all():
                    raise ValueError("paired carrier gradient became non-finite")
            torch.nn.utils.clip_grad_norm_(trigger.parameters(), 5.0)
            carrier_optimizer.step()
            components.append({key: float(np.mean([member[key] for member in details_per_member]))
                               for key in details_per_member[0]})
            member_components.append({"outer_step": outer_step,
                "outer_batch_identity_sha256": outer_identity["indices_sha256"],
                "outer_doses_sha256": _tensor_sha256(outer_dose),
                "per_surrogate": details_per_member})
        for step in range(cfg["lc_inner_steps"]):
            paired_real_update(round_number, "inner_after", step, inner_losses, clean_losses)
        validation = _evaluate_paired(models, twins, trigger, val_x, val_y, cfg)
        for member, record in enumerate(validation["per_surrogate"]):
            record["clean_twin_real_updates_since_clone"] = twin_updates[member]
            record["poison_surrogate_real_updates_since_clone"] = (
                surrogate_updates[member] - clone_proofs[member]["warmup_real_updates"])
        round_real, round_virtual = real_log[real_begin:], virtual_log[virtual_begin:]
        history.append({"stage": "alternation", "round": round_number,
            "surrogate_inner_loss_m": [float(np.mean(values)) for values in inner_losses],
            "clean_twin_inner_loss_m": [float(np.mean(values)) for values in clean_losses],
            "outer_loss": float(np.mean(outer_losses)), "outer_components": components,
            "outer_components_per_surrogate": member_components,
            "guard_activation_counts": round_activations, "inner_validation": validation,
            "paired_real_batch_identity_sha256": _json_sha(round_real),
            "paired_virtual_batch_identity_sha256": _json_sha(round_virtual),
            "paired_real_updates_per_member": len(round_real),
            "paired_virtual_updates_per_member": len(round_virtual),
            "cumulative_surrogate_real_updates": list(surrogate_updates),
            "cumulative_clean_twin_real_updates": list(twin_updates),
            "cumulative_virtual_updates": list(virtual_updates)})
        candidate = {"round": round_number, "score": validation["score"],
                     "utility_gate_passed": validation["utility_gate_passed"],
                     "state": {name: value.detach().cpu().clone() for name, value in trigger.state_dict().items()}}
        if best_fallback is None or candidate["score"] < best_fallback["score"]:
            best_fallback = candidate
        if candidate["utility_gate_passed"] and (best_eligible is None or candidate["score"] < best_eligible["score"]):
            best_eligible = candidate
        print(f"[paired-fit] round {round_number + 1}/{cfg['lc_rounds']}: "
              f"INNER-val T={validation['target_m']:.5f} clean={validation['clean_m']:.5f} "
              f"paired_utility_gate={validation['utility_gate_passed']}", flush=True)
    selected = best_eligible if best_eligible is not None else best_fallback
    trigger.load_state_dict(selected["state"])
    real_sha, virtual_sha = _json_sha(real_log), _json_sha(virtual_log)
    return {
        "history": history, "selected_round": selected["round"], "selected_score": selected["score"],
        "selected_utility_gate_passed": selected["utility_gate_passed"],
        "utility_gate_required": True, "no_eligible_candidate": best_eligible is None,
        "utility_reference": "current-stage clean-only twin per surrogate",
        "utility_tolerances": {key: cfg[key] for key in _TOLERANCE_KEYS},
        "utility_tolerances_immutable": True, "clean_twin_clone_proof": clone_proofs,
        "surrogate_initialization_seeds": [4242, 4343], "surrogate_count": 2,
        "inner_poison_count": count,
        "inner_poison_plan_sha256": _json_sha([[int(index), float(plan_doses[index])] for index in chosen]),
        "surrogate_actual_erm_steps_per_member": surrogate_updates[0],
        "clean_twin_actual_erm_steps_per_member": twin_updates[0],
        "surrogate_virtual_sgd_steps_per_member": virtual_updates[0],
        "clean_twin_virtual_sgd_steps_per_member": virtual_updates[0],
        "attacker_outer_steps": cfg["lc_rounds"] * cfg["lc_outer_steps"],
        "guard_activation_counts": activations,
        "guard_activation_counts_per_surrogate": per_member_activations,
        "guard_observations_per_metric": cfg["lc_rounds"] * cfg["lc_outer_steps"] * len(models),
        "paired_batch_audit": {
            "fit_input_sha256": _tensor_sha256(fit_x), "fit_target_sha256": _tensor_sha256(fit_y),
            "warmup_batch_identity_sha256": _json_sha(warmup_log),
            "poison_model_real_batch_identity_sha256": real_sha,
            "clean_twin_real_batch_identity_sha256": real_sha,
            "poison_real_update_log_sha256": real_sha, "clean_twin_real_update_log_sha256": real_sha,
            "poison_model_virtual_batch_identity_sha256": virtual_sha,
            "clean_twin_virtual_batch_identity_sha256": virtual_sha,
            "post_warmup_real_updates_per_member": len(real_log),
            "virtual_updates_per_member": len(virtual_log),
            "real_updates_in_lockstep": True, "virtual_updates_mutate_real_state": False,
            "paired_torch_rng": "shared pre-update CPU and active CSI CUDA-device states; preserve post-poison stream",
            "paired_rng_draws_matched": all(
                proof["cpu_draws_matched"] and proof["cuda_draws_matched"]
                for batch in real_log for proof in batch["paired_rng_per_surrogate"]),
            "paired_rng_update_log_sha256": _json_sha(
                [batch["paired_rng_per_surrogate"] for batch in real_log]),
            "clean_twin_doses": "all zero", "clean_twin_inputs_targets": "raw TRAIN CSI and pose",
        },
        "default_full_pool_compute": {
            "fit_samples": 4096, "inner_train_samples": 3277,
            "surrogate_real_updates_per_member": 1798,
            "clean_twin_extra_real_updates_per_member": 768,
            "clean_twin_extra_virtual_updates_per_member": 144,
        },
        "outer_dose_distribution": "paired Uniform(dose_min,dose_max), shared across surrogate ensemble",
        "surrogate_batch_pairing": "same raw TRAIN identities per member and twin; poisoned partners use fixed local doses, clean twins use raw labels",
        "lookahead": "two-step SGD with copied real momentum; poison differentiable, clean twin detached; cloned eval-mode BN buffers",
        "lookahead_is_full_training_bilevel": False, "surrogate_optimizer": "sgd",
        "candidate_selection": "INNER validation mean positive-dose target MPJPE + weighted clean MPJPE, subject to all seven exact zero-tolerance gates per surrogate versus its current clean twin",
        "utility_scope": "stage-matched clean-only twins; surrogate gates only, fresh-victim metrics must be independently verified",
        "no_initialization_weight_transfer": True,
        "clean_twin_weights_exported": False, "surrogate_weights_exported": False,
        "victim_weight_transfer": False,
        "pck_units": "relative to MMFi reference distance (5,12), not millimetres",
    }
