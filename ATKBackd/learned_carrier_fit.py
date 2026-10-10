"""Training-only attacker-owned fitting for exploratory CSI carrier hypotheses.

This is not a reproduction of SIBA, BLTO, Sleeper Agent, LIRA, or Wicked
Oddities. Their ideas motivate explicitly documented CSI/HPE adaptations.
The fitted operator is frozen before a fresh victim learns ordinary CSI/pose
pairs with ordinary full-pose MPJPE. Official test and the external draft
holdout are never opened here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.func import functional_call


FIT_SCHEMA = 1
LEARNED_VARIANTS = {"weights", "sparse", "combined", "gradient", "energy"}
BANK_LEARNED_VARIANTS = {"trainaware", "bank", "bank_guard"}
PAIRED_LEARNED_VARIANTS = {"paired_guard"}
FIT_DEFAULTS = {
    "lc_warmup_epochs": 3, "lc_rounds": 3,
    "lc_inner_steps": 32, "lc_outer_steps": 16,
    "lc_batch_size": 32, "lc_lr": 0.02,
    "lc_energy_weight": 0.2, "lc_clean_weight": 1.0,
    "lc_fit_samples": 4096, "lc_inner_val_fraction": 0.2,
    "lc_gradient_tensors": 2, "lc_fit_seed": 4242,
}


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_fit_config(cfg):
    """Materialize a finite, small, explicitly recorded attacker compute budget."""
    out = dict(cfg)
    for key, value in FIT_DEFAULTS.items():
        out.setdefault(key, value)
    for key in ("lc_warmup_epochs", "lc_rounds", "lc_inner_steps",
                "lc_outer_steps", "lc_batch_size", "lc_fit_samples",
                "lc_gradient_tensors"):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        out[key] = int(value)
    fit_seed = out["lc_fit_seed"]
    if isinstance(fit_seed, bool) or not isinstance(fit_seed, (int, np.integer)) or fit_seed < 0:
        raise ValueError("lc_fit_seed must be a nonnegative integer")
    out["lc_fit_seed"] = int(fit_seed)
    for key in ("lc_lr", "lc_energy_weight", "lc_clean_weight", "lc_inner_val_fraction"):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{key} must be a finite number")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        out[key] = value
    if out["lc_lr"] <= 0 or not 0 < out["lc_inner_val_fraction"] < 0.5:
        raise ValueError("lc_lr must be positive and lc_inner_val_fraction in (0, .5)")
    return out


def _prediction(model, csi, parameters=None):
    if parameters is None:
        output = model(csi)
    else:
        # Clone running statistics: lookahead must not mutate real surrogate BN
        # buffers, and buffers cannot retain a graph from a prior outer step.
        buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
        output = functional_call(model, (parameters, buffers), (csi,))
    return output[0] if isinstance(output, tuple) else output


def _mpjpe(prediction, target):
    if prediction.shape != target.shape or target.shape[-1] != 3:
        raise ValueError("surrogate predictions and pose targets must have matching shapes")
    return torch.linalg.vector_norm(prediction - target, dim=-1).mean()


def _target_pose(pose, dose, cfg):
    """Differentiable geometry matching the existing fixed payload contract."""
    from attack.payload import descendants, set_skeleton_config
    set_skeleton_config("mmfi")
    pivot = int(cfg["pivot"])
    joints = descendants(pivot)
    if not joints:
        raise ValueError("learned carrier fitting requires a nonempty payload branch")
    axis = pose.new_tensor(cfg.get("payload_axis", [0.0, 0.0, 1.0]))
    if axis.numel() != 3 or not torch.isfinite(axis).all() or axis.norm() <= 1e-12:
        raise ValueError("payload_axis must be a finite nonzero 3-vector")
    axis = axis / axis.norm()
    theta = dose * math.radians(float(cfg["theta_max_deg"]))
    mode = cfg.get("dose_mode", "linear")
    if mode == "sqrt":
        theta = dose.sqrt() * math.radians(float(cfg["theta_max_deg"]))
    elif mode == "quad":
        theta = dose.square() * math.radians(float(cfg["theta_max_deg"]))
    elif mode != "linear":
        raise ValueError(f"unknown dose_mode: {mode}")
    k = torch.stack((torch.stack((axis.new_zeros(()), -axis[2], axis[1])),
                     torch.stack((axis[2], axis.new_zeros(()), -axis[0])),
                     torch.stack((-axis[1], axis[0], axis.new_zeros(())))))
    identity = torch.eye(3, dtype=pose.dtype, device=pose.device)
    rotation = identity[None] + theta.sin()[:, None, None] * k + (
        1.0 - theta.cos())[:, None, None] * (k @ k)
    result = pose.clone()
    origin = pose[..., pivot:pivot + 1, :]
    relative = pose[..., joints, :] - origin
    result[..., joints, :] = torch.einsum("bpjc,bkc->bpjk", relative, rotation) + origin
    return result, joints


def _relative_energy(attacked, original):
    delta = torch.linalg.vector_norm((attacked - original).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(original.flatten(1), dim=1).clamp_min(1e-12)
    return (delta / denominator).square().mean()


def _lookahead_objective(model, trigger, inner_x, inner_y, inner_dose,
                         outer_x, outer_y, outer_dose, cfg):
    """Target and clean outer objectives after one differentiable poison ERM step.

    The clean branch uses UPDATED parameters, so its loss really depends on
    the trigger through poisoned training. This is a one-step, no-momentum
    SGD approximation, not exact long-horizon bilevel differentiation.
    """
    inner_target, joints = _target_pose(inner_y, inner_dose, cfg)
    inner_attacked = trigger.inject_tensor(inner_x, inner_dose, eps=cfg["eps"])
    parameters = dict(model.named_parameters())
    training_loss = _mpjpe(_prediction(model, inner_attacked), inner_target)
    gradients = torch.autograd.grad(training_loss, tuple(parameters.values()),
                                     create_graph=True, allow_unused=True)
    rate = float(cfg["lr"])
    decay = float(cfg.get("weight_decay", 0.0))
    updated = {name: value if grad is None else value - rate * (grad + decay * value)
               for (name, value), grad in zip(parameters.items(), gradients)}
    attacked = trigger.inject_tensor(outer_x, outer_dose, eps=cfg["eps"])
    target, _ = _target_pose(outer_y, outer_dose, cfg)
    target_loss = _mpjpe(_prediction(model, attacked, updated)[..., joints, :],
                         target[..., joints, :])
    clean_loss = _mpjpe(_prediction(model, outer_x, updated), outer_y)
    energy = _relative_energy(attacked, outer_x)
    # Every variant receives the same actual-energy term; energy variant raises
    # its weight (recorded) rather than changing the hard budget.
    energy_weight = float(cfg["lc_energy_weight"])
    if cfg.get("lc_variant") == "energy":
        energy_weight *= 4.0
    objective = target_loss + float(cfg["lc_clean_weight"]) * clean_loss + energy_weight * energy
    return objective, {"target": target_loss, "clean": clean_loss, "energy": energy}


def _gradient_matching_objective(model, trigger, train_x, train_y, train_dose,
                                 target_x, target_y, target_dose, cfg):
    """Cosine alignment on the documented LAST trainable weight/bias tensors.

    Gradient matching of the full model would be much more expensive. This
    bounded draft approximation compares poisoned ordinary-ERM gradients to
    target-joint attack gradients. The target gradient is detached; poison
    gradients retain their graph for carrier optimization.
    """
    selected = tuple(model.parameters())[-int(cfg["lc_gradient_tensors"]):]
    poison_target, joints = _target_pose(train_y, train_dose, cfg)
    poison_x = trigger.inject_tensor(train_x, train_dose, eps=cfg["eps"])
    train_loss = _mpjpe(_prediction(model, poison_x), poison_target)
    train_grads = torch.autograd.grad(train_loss, selected, create_graph=True,
                                     allow_unused=True)
    attack_target, _ = _target_pose(target_y, target_dose, cfg)
    # Desired attack gradients concern CURRENT triggered inputs, not a global
    # clean-input bias toward the rotated pose. Detach this target-side input
    # and gradient to optimize only the poison-training gradient alignment.
    target_input = trigger.inject_tensor(target_x, target_dose, eps=cfg["eps"]).detach()
    target_loss = _mpjpe(_prediction(model, target_input)[..., joints, :],
                         attack_target[..., joints, :])
    target_grads = torch.autograd.grad(target_loss, selected, allow_unused=True)
    pairs = [(a.reshape(-1), b.detach().reshape(-1))
             for a, b in zip(train_grads, target_grads) if a is not None and b is not None]
    if not pairs:
        raise ValueError("gradient matching found no shared trainable output tensors")
    poison_vector = torch.cat([pair[0] for pair in pairs])
    target_vector = torch.cat([pair[1] for pair in pairs])
    cosine = torch.nn.functional.cosine_similarity(poison_vector, target_vector, dim=0, eps=1e-12)
    energy = _relative_energy(poison_x, train_x)
    objective = 1.0 - cosine + float(cfg["lc_energy_weight"]) * energy
    return objective, {"cosine": cosine, "energy": energy}


def _stratum(item):
    """Use immutable subject/action metadata, not labels from any heldout split."""
    source = str(item.get("name", item.get("csi", "unknown"))).replace("\\", "/")
    subject = re.search(r"(?:^|[/_])(S\d+)(?:[/_]|$)", source)
    action = re.search(r"(?:^|[/_])(A\d+)(?:[/_]|$)", source)
    return (subject.group(1) if subject else "unknown_subject",
            action.group(1) if action else "unknown_action")


def stratified_select_indices(items, scores, rho, seed=42):
    """Exact floor(rho*N), proportional subject/action quotas, high clean error.

    This is a training-pose-error heuristic inspired by informative poison
    selection, NOT Wicked Oddities' feature-neighbour algorithm. Remaining
    integer quota is assigned by largest fractional remainder, with seeded
    deterministic tie-breaking. Within-stratum ties use the same seeded order.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(items),) or not np.isfinite(scores).all():
        raise ValueError("selection scores must contain one finite score per TRAIN item")
    if not math.isfinite(float(rho)) or not 0 <= float(rho) <= 1:
        raise ValueError("rho must be in [0,1]")
    count = int(math.floor(float(rho) * len(items)))
    if count == 0:
        return []
    groups = {}
    for index, item in enumerate(items):
        groups.setdefault(_stratum(item), []).append(index)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(items))
    priority = np.empty(len(items), dtype=int)
    priority[tie] = np.arange(len(items))
    exact = np.asarray([count * len(groups[name]) / len(items) for name in names])
    quotas = np.floor(exact).astype(int)
    group_ties = rng.random(len(names))
    order = sorted(range(len(names)), key=lambda i: (-(exact[i] - quotas[i]), group_ties[i]))
    for index in order[:count - int(quotas.sum())]:
        quotas[index] += 1
    chosen = []
    for name, quota in zip(names, quotas):
        ranked = sorted(groups[name], key=lambda index: (-scores[index], priority[index]))
        chosen.extend(ranked[:int(quota)])
    return sorted(map(int, chosen))


def _read_batch(base, indices, device):
    frames, poses = [], []
    for index in indices:
        item = base.items[int(index)]
        frame = np.asarray(base.normalize(base.load_raw(item["csi"])), dtype=np.float32)
        pose = base.load_pose(item["kpt"], item["frame_idx"]) if "frame_idx" in item else base.load_pose(item["kpt"])
        pose = np.asarray(pose, dtype=np.float32)
        if pose.ndim == 2:
            pose = pose[None]
        if not np.isfinite(frame).all() or not np.isfinite(pose).all():
            raise ValueError("attacker TRAIN samples contain NaN/Inf")
        frames.append(frame)
        poses.append(pose)
    return (torch.from_numpy(np.stack(frames)).to(device).contiguous(),
            torch.from_numpy(np.stack(poses)).to(device).contiguous())


def _metadata_ids(base, indices):
    return [[str(base.items[int(i)].get("csi", "")),
             str(base.items[int(i)].get("kpt", "")),
             int(base.items[int(i)].get("frame_idx", -1))] for i in indices]


def _ordinary_update(model, optimizer, trigger, x, y, dose, cfg):
    """Actual surrogate SGD with a detached current trigger and ordinary labels."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        attacked = x if trigger is None else trigger.inject_tensor(x, dose, eps=cfg["eps"])
        target = y if trigger is None else _target_pose(y, dose, cfg)[0]
    loss = _mpjpe(_prediction(model, attacked), target)
    if not torch.isfinite(loss):
        raise ValueError("surrogate ERM loss became non-finite")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _evaluation(model, trigger, x, y, cfg):
    model.eval()
    batch_size = int(cfg["lc_batch_size"])
    target_sum = clean_sum = energy_sum = 0.0
    with torch.no_grad():
        for begin in range(0, len(x), batch_size):
            xb, yb = x[begin:begin + batch_size], y[begin:begin + batch_size]
            count = len(xb)
            clean_sum += count * float(_mpjpe(_prediction(model, xb), yb))
            for dose_value in cfg.get("dose_grid", [0.0, .2, .4, .6, .8, 1.0]):
                if dose_value <= 0:
                    continue
                dose = xb.new_full((count,), float(dose_value))
                attacked = trigger.inject_tensor(xb, dose, eps=cfg["eps"])
                target, joints = _target_pose(yb, dose, cfg)
                target_sum += count * float(_mpjpe(_prediction(model, attacked)[..., joints, :], target[..., joints, :]))
                energy_sum += count * float(_relative_energy(attacked, xb))
    positive = sum(float(d) > 0 for d in cfg.get("dose_grid", [0.0, .2, .4, .6, .8, 1.0]))
    if not positive:
        raise ValueError("candidate selection requires a positive-dose grid")
    result = {"target_m": target_sum / (len(x) * positive),
              "clean_m": clean_sum / len(x),
              "relative_energy": energy_sum / (len(x) * positive)}
    result["score"] = result["target_m"] + float(cfg["lc_clean_weight"]) * result["clean_m"]
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("INNER-validation candidate scores became non-finite")
    return result


def fit_trigger(model, trigger, fit_x, fit_y, val_x, val_y, cfg, rng=None):
    """Fit a carrier on tensors supplied exclusively from the official TRAIN pool.

    Public helper permits fast synthetic contract tests without real CSI,
    construction of HPELi, or access to any official evaluation loader.
    """
    cfg = resolve_fit_config(cfg)
    rng = np.random.default_rng(int(cfg.get("seed", 42))) if rng is None else rng
    from train_backdoor import _build_optimizer
    _, surrogate_optimizer = _build_optimizer(model, cfg, "mmfi")
    carrier_optimizer = torch.optim.Adam([parameter for parameter in trigger.parameters()
                                          if parameter.requires_grad], lr=cfg["lc_lr"])
    batch_size = cfg["lc_batch_size"]
    n = len(fit_x)
    if n < 2 or len(val_x) < 1:
        raise ValueError("attacker fitting needs disjoint nonempty inner TRAIN/validation pools")
    poison_count = int(math.floor(float(cfg["rho"]) * n))
    if poison_count < 1:
        raise ValueError("attacker inner TRAIN pool too small for requested poison rate")
    poison_rng = np.random.default_rng(int(cfg.get("seed", 42)))
    chosen = poison_rng.choice(n, poison_count, replace=False)
    plan_doses = np.zeros(n, dtype=np.float32)
    plan_doses[chosen] = poison_rng.uniform(cfg["dose_min"], cfg["dose_max"], poison_count)
    fixed_doses = torch.from_numpy(plan_doses).to(fit_x.device)
    history = []
    for epoch in range(cfg["lc_warmup_epochs"]):
        losses = []
        for begin in range(0, n, batch_size):
            # One permutation per warmup epoch, not replacement sampling.
            if begin == 0:
                order = rng.permutation(n)
            index = torch.as_tensor(order[begin:begin + batch_size], device=fit_x.device)
            losses.append(_ordinary_update(model, surrogate_optimizer, None,
                          fit_x[index], fit_y[index], fixed_doses[index], cfg))
        history.append({"stage": "warmup", "epoch": epoch, "loss_m": float(np.mean(losses))})
        print(f"[learned-fit] warmup {epoch + 1}/{cfg['lc_warmup_epochs']}: {np.mean(losses):.5f}", flush=True)

    best_state, best_score, best_round = None, math.inf, None

    def sample():
        index = torch.as_tensor(rng.choice(n, min(batch_size, n), replace=False), device=fit_x.device)
        return fit_x[index], fit_y[index], fixed_doses[index]

    for round_number in range(cfg["lc_rounds"]):
        inner_losses = []
        for _ in range(cfg["lc_inner_steps"]):
            xb, yb, db = sample()
            inner_losses.append(_ordinary_update(model, surrogate_optimizer, trigger, xb, yb, db, cfg))
        model.eval()
        outer_losses, components = [], []
        for _ in range(cfg["lc_outer_steps"]):
            inner_x, inner_y, inner_dose = sample()
            outer_x, outer_y, _ = sample()
            outer_dose = torch.as_tensor(rng.uniform(cfg["dose_min"], cfg["dose_max"], len(outer_x)),
                                         dtype=outer_x.dtype, device=outer_x.device)
            carrier_optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            if cfg.get("lc_variant") == "gradient":
                objective, details = _gradient_matching_objective(model, trigger,
                    inner_x, inner_y, inner_dose, outer_x, outer_y, outer_dose, cfg)
            else:
                objective, details = _lookahead_objective(model, trigger,
                    inner_x, inner_y, inner_dose, outer_x, outer_y, outer_dose, cfg)
            if not torch.isfinite(objective):
                raise ValueError("attacker outer objective became non-finite")
            objective.backward()
            for parameter in trigger.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise ValueError("carrier gradient became non-finite")
            torch.nn.utils.clip_grad_norm_(trigger.parameters(), 5.0)
            carrier_optimizer.step()
            outer_losses.append(float(objective.detach()))
            components.append({key: float(value.detach()) for key, value in details.items()})
        # Retrain the actual surrogate with the UPDATED carrier before scoring
        # any candidate. No candidate is selected on a stale clean surrogate.
        for _ in range(cfg["lc_inner_steps"]):
            xb, yb, db = sample()
            inner_losses.append(_ordinary_update(model, surrogate_optimizer, trigger, xb, yb, db, cfg))
        validation = _evaluation(model, trigger, val_x, val_y, cfg)
        history.append({"stage": "alternation", "round": round_number,
                        "inner_loss_m": float(np.mean(inner_losses)),
                        "outer_loss": float(np.mean(outer_losses)),
                        "outer_components": components, "inner_validation": validation})
        if validation["score"] < best_score:
            best_score, best_round = validation["score"], round_number
            best_state = {key: value.detach().cpu().clone() for key, value in trigger.state_dict().items()}
        print(f"[learned-fit] {cfg.get('lc_variant')} round {round_number + 1}/{cfg['lc_rounds']}: "
              f"INNER-val T={validation['target_m']:.5f} clean={validation['clean_m']:.5f}", flush=True)
    trigger.load_state_dict(best_state)
    return {"history": history, "selected_round": best_round, "selected_score": best_score,
            "inner_poison_count": poison_count,
            "inner_poison_plan_sha256": _json_sha([[int(i), float(plan_doses[i])] for i in chosen]),
            "lookahead": "one-step differentiable SGD, without momentum; outer clean loss uses updated params",
            "gradient_matching": "last trainable weight/bias tensors; detached target gradient",
            "gradient_parameter_names": [name for name, _ in list(model.named_parameters())[-cfg["lc_gradient_tensors"]:]],
            "surrogate_optimizer": cfg.get("optimizer", "sgd"),
            "candidate_selection": "inner validation mean positive-dose target MPJPE + weighted clean MPJPE"}


def _identity_cfg(cfg):
    safe = {"device", "num_workers", "worker_timeout", "ckpt_every", "config_fingerprint"}
    return {key: value for key, value in cfg.items() if key not in safe}


def _warmup_for_selection(model, x, y, cfg, rng):
    from train_backdoor import _build_optimizer
    _, optimizer = _build_optimizer(model, cfg, "mmfi")
    history = []
    for epoch in range(cfg["lc_warmup_epochs"]):
        order = rng.permutation(len(x))
        losses = []
        for begin in range(0, len(x), cfg["lc_batch_size"]):
            index = torch.as_tensor(order[begin:begin + cfg["lc_batch_size"]], device=x.device)
            zero = x.new_zeros(len(index))
            losses.append(_ordinary_update(model, optimizer, None, x[index], y[index], zero, cfg))
        history.append({"stage": "selection_clean_warmup", "epoch": epoch,
                        "loss_m": float(np.mean(losses))})
    return history


def prepare_learned_cell(cfg, folder, recipe_sha256):
    """Freeze a training-only artifact or explicit selection before victim launch.

    Completed fitting is reused only with exact scientific recipe, action-file,
    fitting-record, and artifact hashes. A changed artifact fails loudly; there
    is no silent re-fit followed by reuse of incompatible victim checkpoints.
    """
    variant = cfg.get("lc_variant")
    learned_variants = LEARNED_VARIANTS | BANK_LEARNED_VARIANTS | PAIRED_LEARNED_VARIANTS
    if variant not in learned_variants | {"selection"}:
        return dict(cfg)
    if cfg.get("experiment_name") != "mmfi" or cfg.get("draft_eval_source") != "training_holdout":
        raise ValueError("learned fitting requires MMFi with an official-TRAIN draft holdout")
    if variant in PAIRED_LEARNED_VARIANTS:
        from paired_guard_fit import PAIRED_FIT_DEFAULTS, resolve_paired_fit_config
        cfg = resolve_paired_fit_config(cfg)
        fit_setting_keys = PAIRED_FIT_DEFAULTS
    elif variant in BANK_LEARNED_VARIANTS:
        from carrier_bank_fit import BANK_FIT_DEFAULTS, resolve_bank_fit_config
        cfg = resolve_bank_fit_config(cfg)
        fit_setting_keys = BANK_FIT_DEFAULTS
    else:
        cfg = resolve_fit_config(cfg)
        fit_setting_keys = FIT_DEFAULTS
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prepared_path = folder / "prepared_cfg.json"
    record_path = folder / "fitting.json"
    artifact_path = folder / "learned_trigger.json"
    action_sha = _file_sha(cfg["action_npy"])
    identity = _json_sha(_identity_cfg(cfg))
    if prepared_path.exists():
        saved = json.loads(prepared_path.read_text(encoding="utf-8"))
        if (saved.get("schema") != FIT_SCHEMA or saved.get("recipe_sha256") != recipe_sha256
                or saved.get("input_cfg_sha256") != identity or saved.get("action_sha256") != action_sha):
            raise ValueError("prepared fitting recipe/action changed; use a fresh run directory")
        if not record_path.is_file() or _file_sha(record_path) != saved.get("fitting_sha256"):
            raise ValueError("prepared fitting record is missing or altered")
        out = saved["cfg"]
        from train_backdoor import _config_fingerprint, _resolve_training_config
        if _config_fingerprint(_resolve_training_config(out)) != saved.get("cfg_fingerprint"):
            raise ValueError("prepared fitting configuration is altered")
        if variant in learned_variants:
            if not artifact_path.is_file() or _file_sha(artifact_path) != out.get("lc_artifact_sha256"):
                raise ValueError("frozen learned trigger artifact is missing or altered")
        elif _json_sha(out.get("lc_poison_indices")) != out.get("lc_selection_sha256"):
            raise ValueError("frozen selection artifact is altered")
        for key in ("device", "num_workers", "worker_timeout", "ckpt_every"):
            if key in cfg:
                out[key] = cfg[key]
        return out
    if record_path.exists() or artifact_path.exists():
        raise ValueError("incomplete fitting artifacts found; use a fresh cell directory")

    seed = int(cfg.get("seed", 42))
    fit_seed = int(cfg["lc_fit_seed"])
    # Attacker and victim must not share initialization by coincidence. The
    # victim trainer reseeds independently from cfg['seed'] after fitting; no
    # surrogate weights are passed to its construction or optimizer.
    random.seed(fit_seed)
    np.random.seed(fit_seed)
    torch.manual_seed(fit_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fit_seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8721]))
    from train_backdoor import _load_dataset
    from attack.trigger import build_trigger_by_name
    from models.factory import build_model
    from run_mmfi_tables import _atomic_json
    base = _load_dataset(cfg, "training")  # Deliberately the ONLY dataset load.
    if getattr(base, "split", "train") not in ("train", "training"):
        raise ValueError("attacker fitting was given a non-training dataset")
    n_pool = min(len(base), cfg["lc_fit_samples"])
    if n_pool < 4:
        raise ValueError("not enough TRAIN frames for attacker inner split")
    pool_indices = rng.choice(len(base), n_pool, replace=False)
    val_n = max(1, int(math.floor(n_pool * cfg["lc_inner_val_fraction"])))
    val_indices = sorted(map(int, pool_indices[:val_n]))
    fit_indices = sorted(map(int, pool_indices[val_n:]))
    device = torch.device(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[learned-fit] TRAIN-only pool: inner train={len(fit_indices)}, inner validation={len(val_indices)}", flush=True)
    fit_x, fit_y = _read_batch(base, fit_indices, device)
    model = build_model(cfg["model"], num_keypoints=17, num_coor=3,
                        num_person=cfg.get("num_person", 1),
                        subcarrier_num=cfg.get("n_sub", 114), dataset="mmfi", pretrained=False).to(device)
    provenance = {
        "schema": FIT_SCHEMA, "status": "DRAFT_ONLY_NOT_PAPER_RESULTS",
        "variant": variant, "recipe_sha256": recipe_sha256,
        "surrogate_initialization_seed": fit_seed, "victim_seed": seed,
        "no_initialization_weight_transfer": True,
        "action_sha256": action_sha, "attacker_dataset_source": "official TRAIN only",
        "official_test_loaded": False, "external_draft_holdout_loaded": False,
        "training_subset_n": len(base), "fit_pool_n": n_pool,
        "inner_train_indices": fit_indices, "inner_validation_indices": val_indices,
        "inner_train_indices_sha256": _json_sha(fit_indices),
        "inner_validation_indices_sha256": _json_sha(val_indices),
        "inner_train_ids_sha256": _json_sha(_metadata_ids(base, fit_indices)),
        "inner_validation_ids_sha256": _json_sha(_metadata_ids(base, val_indices)),
        "fit_settings": {key: cfg[key] for key in fit_setting_keys},
        "victim_training": "fresh independent victim; ordinary full-pose MPJPE ERM",
        "scope": "inspired adaptations, not source-paper reproductions; no over-the-air stealth claim",
    }
    if callable(getattr(base, "draft_subset_manifest", None)):
        provenance["draft_train_manifest"] = base.draft_subset_manifest()
    out = dict(cfg)
    if variant == "selection":
        history = _warmup_for_selection(model, fit_x, fit_y, cfg, rng)
        model.eval()
        scores = np.empty(len(base), dtype=np.float64)
        with torch.no_grad():
            for begin in range(0, len(base), cfg["lc_batch_size"]):
                stop = min(len(base), begin + cfg["lc_batch_size"])
                xb, yb = _read_batch(base, range(begin, stop), device)
                errors = torch.linalg.vector_norm(_prediction(model, xb) - yb, dim=-1)
                scores[begin:stop] = errors.flatten(1).mean(1).cpu().numpy()
        selected = stratified_select_indices(base.items, scores, cfg["rho"], seed)
        provenance.update(history=history, selection="stratified_training_pose_error",
                          selection_score_sha256=_json_sha(scores.tolist()),
                          selected_indices=selected, selected_indices_sha256=_json_sha(selected),
                          selection_limitation="TRAIN clean pose-error heuristic; not Wicked Oddities feature-neighbour method")
        out.update(lc_poison_indices=selected, lc_selection="stratified_training_pose_error",
                   lc_selection_sha256=_json_sha(selected))
    else:
        from attack.learned_carrier import TrainableCarrier, write_artifact
        original_cfg = dict(cfg, trigger="micro_dropper")
        original_cfg.pop("comparison_peak_budget", None)
        # trainer.build_trigger would also apply the cell's dual-budget wrapper;
        # TrainableCarrier instead needs the raw built Original reference.
        original = build_trigger_by_name("micro_dropper", original_cfg)
        trigger = TrainableCarrier(original, cfg, variant).to(device)
        val_x, val_y = _read_batch(base, val_indices, device)
        if variant in BANK_LEARNED_VARIANTS | PAIRED_LEARNED_VARIANTS:
            models = [model]
            for member in range(1, cfg["lc_surrogate_count"]):
                member_seed = fit_seed + member * cfg["lc_surrogate_seed_stride"]
                random.seed(member_seed)
                np.random.seed(member_seed)
                torch.manual_seed(member_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(member_seed)
                models.append(build_model(cfg["model"], num_keypoints=17, num_coor=3,
                    num_person=cfg.get("num_person", 1),
                    subcarrier_num=cfg.get("n_sub", 114), dataset="mmfi", pretrained=False).to(device))
            if variant in PAIRED_LEARNED_VARIANTS:
                from paired_guard_fit import fit_paired_trigger
                fitter = fit_paired_trigger
            else:
                from carrier_bank_fit import fit_bank_trigger
                fitter = fit_bank_trigger
            provenance.update(fitter(models, trigger, fit_x, fit_y,
                                      val_x, val_y, cfg, rng))
        else:
            provenance.update(fit_trigger(model, trigger, fit_x, fit_y, val_x, val_y, cfg, rng))
        artifact_sha = write_artifact(artifact_path, trigger, recipe_sha256, provenance=provenance)
        out.update(trigger="learned_carrier", lc_artifact_path=str(artifact_path.resolve()),
                   lc_artifact_sha256=artifact_sha)
    _atomic_json(record_path, provenance)
    fitting_sha = _file_sha(record_path)
    out.update(lc_recipe_sha256=recipe_sha256, lc_fitting_sha256=fitting_sha)
    from train_backdoor import _config_fingerprint, _resolve_training_config
    out = _resolve_training_config(out)
    _atomic_json(prepared_path, {"schema": FIT_SCHEMA, "recipe_sha256": recipe_sha256,
        "input_cfg_sha256": identity, "action_sha256": action_sha,
        "action_file_sha256": action_sha,
        "cfg_fingerprint": _config_fingerprint(out),
        "artifact_sha256": out.get("lc_artifact_sha256"),
        "fitting_sha256": fitting_sha, "cfg": out})
    print(f"[learned-fit] frozen {variant}; fresh victim may now start", flush=True)
    return out
