"""Independent, source-pinned SIBA/LADDER adaptations for amplitude MM-Fi.

These are exploratory CSI/pose adaptations, not image reproductions or RF
attacks. A TRAIN-only clean surrogate is required. Neither optimizer receives
the victim to be poisoned, TEST inputs, or TEST scores. Victim training must
start afresh AFTER freezing a trigger. ``dose`` scales both the frozen residual
and the caller's pose target; this continuous-payload extension is not native
to either image paper. There is no common-noise/equal-epsilon claim.

SIBA's production API streams full TRAIN for 200 epochs, refreshes the shared
spatial mask on the first batch every five epochs, and selects the end-of-epoch
pattern with minimum epoch-average TRAIN loss, as the pinned source does.
The task loss and paired dose are explicit pose-protocol adaptations. The k
positions are shared across antennas, as the image source shares its mask
across RGB channels. A reduced in-memory convenience API is fixture-only.

LADDER's paper specifies SBX/PM and preference/Pareto selection, but the pinned
public repository lacks an executable search implementation and required
search settings cannot be sourced without making discretionary choices. The
production LADDER search is BLOCKED. The previously written independent engine
is retained only as an explicitly named, unapproved experimental prototype; it
must not be dispatched or reported as a faithful LADDER baseline.

No source code is copied: the upstream repositories do not provide a license
in their pinned root trees. They serve as algorithm/provenance references.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable

import numpy as np
import torch
from torch import nn


IMPLEMENTATION = "source-schedule-siba-and-blocked-ladder-mmfi-pose-v2"
SOURCES = {
    "siba": {
        "repository": "https://github.com/YinghuaGao/SIBA",
        "commit": "2968e76070c532ac103b1571de77009c004ad04b",
        "files": ["optimize_siba.py", "train_surrogate_cifar.py", "train_poison_cifar.py"],
        "source_sha256_lf": {"optimize_siba.py": "a5e0b9d52939cb0aafaed4b9ca7231782e76ec6a3439df4ff26ed1eef05ac960"},
        "paper": "https://arxiv.org/abs/2306.06209",
        "mechanism": "alternating shared spatial top-k mask and Linf-projected sign gradient",
        "production_search_ready": True,
        "source_schedule": {"epochs": 200, "batch_size": 128, "support_refresh_epochs": 5,
                            "eps": 8.0 / 255.0, "step_ratio": 0.8, "loss_cap": 10.0},
        "source_adaptations": [
            "classification cross-entropy replaced by target-joint MPJPE in metres and paired pose/dose contract",
            "rectangular CSI mask uses flat row=index//width,column=index%width; pinned square-image code divides by height",
            "deterministic epoch-addressable shuffle and epoch-boundary state provide crash recovery, not new optimization",
        ],
    },
    "ladder": {
        "repository": "https://github.com/dzhliu/LADDER",
        "commit": "a752cf48805283fcf7becd88e4382199e40de2c1",
        "files": ["train_freq_attack.py", "utils_lfba.py"],
        "source_sha256_lf": {
            "utils_lfba.py": "e25ad90437ca09e853fece911e71f220a2dc73dcfbb32556e4b246a430dcd263",
            "train_freq_attack.py": "ec716841993b537d5e3954af37aee84a5d4646b426f75e2b07691ce8d8d2a01b",
        },
        "paper": "https://gsmaragd.github.io/publications/NDSS2025/NDSS2025.pdf",
        "mechanism": "paper Algorithms 1-3: retrain/rollback, three objectives, SBX/PM, preference NDSort",
        "source_limitation": "public repository does not expose a complete executable MOEA search",
        "production_search_ready": False,
        "blocking_missing_material": [
            "pinned train_freq_attack.py loads a selected trigger from missing saved_vars/3objs_vgg11_cifar10_selected.pt",
            "no public executable MOEA search or exact variation distribution settings in the pinned tree",
            "pose-task replacement for preference-region O1 cannot be inferred from classification loss 0.9",
        ],
    },
}


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name, zero=False):
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def _shape(shape):
    if len(shape) != 3:
        raise ValueError("shape must be antenna x subcarrier x packet")
    return tuple(_integer(v, "shape") for v in shape)


@dataclass(frozen=True)
class SIBAConfig:
    shape: tuple[int, int, int] = (3, 114, 10)
    seed: int = 42
    k: int = 100
    eps: float = 8.0 / 255.0
    epochs: int = 200
    batch_size: int = 128
    support_refresh_epochs: int = 5
    step_ratio: float = 0.8
    loss_cap_m: float = 10.0
    target_joints: tuple[int, ...] = (2, 3)

    def __post_init__(self):
        object.__setattr__(self, "shape", _shape(self.shape))
        for name in ("k", "epochs", "batch_size", "support_refresh_epochs"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "seed", _integer(self.seed, "seed", 0))
        if self.k > self.shape[1] * self.shape[2]:
            raise ValueError("k exceeds the number of spatial positions")
        for name in ("eps", "step_ratio", "loss_cap_m"):
            _positive(getattr(self, name), name)
        if self.eps > 1 or self.step_ratio > 1:
            raise ValueError("eps and step_ratio must not exceed 1")
        joints = tuple(_integer(v, "target joint", 0) for v in self.target_joints)
        if not joints or len(set(joints)) != len(joints):
            raise ValueError("target joints must be distinct")
        object.__setattr__(self, "target_joints", joints)


@dataclass(frozen=True)
class LADDERConfig:
    """ARCHIVAL experimental prototype settings, NOT sourced author defaults.

    No executable production search accepts these discretionary preference
    scalings/variation settings. They remain solely to test or inspect the
    superseded prototype without representing it as a LADDER reproduction.
    """
    shape: tuple[int, int, int] = (3, 114, 10)
    seed: int = 42
    population: int = 10
    generations: int = 20
    retrain_epochs: int = 20
    batch_size: int = 32
    poison_ratio: float = 0.1
    coefficients_per_channel: int = 3
    low_band: tuple[int, int] = (49, 4)
    coefficient_bound: float = 0.3
    retrain_lr: float = 0.001
    retrain_momentum: float = 0.9
    retrain_weight_decay: float = 0.0
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    # Registered before TEST. Objective 1 is in metres, 2 in DCT units,
    # 3 in integer frequency-coordinate units; do not compare these raw units.
    preference: tuple[float, float, float] = (0.0, 0.0, 0.0)
    objective_scales: tuple[float, float, float] = (1.0, 0.4, 8.0)

    def __post_init__(self):
        object.__setattr__(self, "shape", _shape(self.shape))
        for name in ("population", "generations", "retrain_epochs", "batch_size", "coefficients_per_channel"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "seed", _integer(self.seed, "seed", 0))
        if self.population < 2:
            raise ValueError("population must be at least two")
        band = tuple(_integer(v, "low_band") for v in self.low_band)
        if len(band) != 2 or any(v > s for v, s in zip(band, self.shape[1:])):
            raise ValueError("low_band must fit the CSI frequency grid")
        if self.coefficients_per_channel > math.prod(band):
            raise ValueError("not enough distinct frequency coordinates")
        object.__setattr__(self, "low_band", band)
        if not 0 < self.poison_ratio < 1 or not 0 <= self.retrain_momentum < 1:
            raise ValueError("poison_ratio in (0,1), momentum in [0,1) required")
        for name in ("coefficient_bound", "retrain_lr", "crossover_eta", "mutation_eta"):
            _positive(getattr(self, name), name)
        _positive(self.retrain_weight_decay, "retrain_weight_decay", zero=True)
        if len(self.preference) != 3 or len(self.objective_scales) != 3:
            raise ValueError("exactly three objective preferences/scales required")
        for v in self.preference:
            _positive(v, "preference", zero=True)
        for v in self.objective_scales:
            _positive(v, "objective_scale")
        object.__setattr__(self, "preference", tuple(float(v) for v in self.preference))
        object.__setattr__(self, "objective_scales", tuple(float(v) for v in self.objective_scales))


def _array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class FrozenCSITrigger:
    requires_deferred_injection = False
    requires_event_metadata = False
    dose_semantics = "clip(normalized CSI + dose * frozen residual, 0, 1); no second normalization"

    def __init__(self, method, config, pattern, diagnostics=None):
        if method not in SOURCES or not isinstance(config, SIBAConfig if method == "siba" else LADDERConfig):
            raise ValueError("method/config mismatch")
        self.method, self.config, self.shape = method, config, config.shape
        value = torch.as_tensor(pattern).detach().cpu().to(torch.float32).contiguous().clone()
        if tuple(value.shape) != self.shape or not torch.isfinite(value).all():
            raise ValueError("invalid frozen trigger")
        if method == "siba":
            if value.abs().max().item() > config.eps + 1e-7 or int(value.ne(0).any(0).sum()) > config.k:
                raise ValueError("SIBA frozen trigger violates sparsity or Linf bound")
        self._pattern = value
        self._diagnostics = deepcopy(diagnostics or {})
        _json_sha(self._diagnostics)
        self.baseline_name = ("SIBA-CSI adapted" if method == "siba" else
                              "LADDER frequency operator only (author search blocked)")

    @property
    def pattern(self):
        return self._pattern.numpy().copy()

    def apply_tensor(self, csi, dose):
        if not isinstance(csi, torch.Tensor) or not csi.is_floating_point() or csi.ndim not in (3, 4) or tuple(csi.shape[-3:]) != self.shape:
            raise ValueError("expected floating CHW/BCHW CSI")
        if not torch.isfinite(csi).all() or torch.any(csi < 0) or torch.any(csi > 1):
            raise ValueError("CSI must be finite and normalized to [0,1]")
        d = torch.as_tensor(dose, dtype=csi.dtype, device=csi.device)
        if d.ndim > 1 or (d.ndim == 1 and (csi.ndim != 4 or len(d) != len(csi))):
            raise ValueError("dose must be scalar or one scalar per batch example")
        if not torch.isfinite(d).all() or torch.any(d < 0) or torch.any(d > 1):
            raise ValueError("dose must be finite in [0,1]")
        if d.ndim:
            d = d[:, None, None, None]
        p = self._pattern.to(device=csi.device, dtype=csi.dtype)
        out = (csi + d * p).clamp(0, 1)
        return torch.where(d == 0, csi, out)

    def inject(self, csi, dose, eps=None):
        value = np.asarray(csi)
        if value.dtype.kind != "f" or value.shape != self.shape:
            raise ValueError("expected floating real CHW CSI")
        if eps is not None:
            _positive(eps, "compatibility eps", zero=True)
        return self.apply_tensor(torch.from_numpy(value.copy()), dose).numpy()

    def provenance(self):
        return {
            "implementation": IMPLEMENTATION,
            "method": self.method,
            "source": deepcopy(SOURCES[self.method]),
            "config": asdict(self.config),
            "pattern_sha256": _array_sha(self.pattern),
            "dose_semantics": self.dose_semantics,
            "adaptation": "normalized amplitude CSI channels, pose targets, TRAIN-only optimization; not image/RF reproduction",
            "model_access": "white-box clean surrogate for SIBA; query/retrain clean surrogate for LADDER; no victim checkpoint reuse",
            "production_search_ready": SOURCES[self.method]["production_search_ready"],
            "diagnostics": deepcopy(self._diagnostics),
        }

    def state_dict(self):
        value = {"version": IMPLEMENTATION, "method": self.method, "config": asdict(self.config),
                 "pattern": self.pattern.tolist(), "diagnostics": deepcopy(self._diagnostics)}
        return {**value, "sha256": _json_sha(value)}

    @classmethod
    def from_state_dict(cls, saved):
        value = deepcopy(saved)
        digest = value.pop("sha256", None)
        if digest != _json_sha(value) or value.get("version") != IMPLEMENTATION:
            raise ValueError("frozen trigger checksum/version mismatch")
        config_type = {"siba": SIBAConfig, "ladder": LADDERConfig}.get(value.get("method"))
        if config_type is None:
            raise ValueError("unknown frozen trigger method")
        return cls(value["method"], config_type(**value["config"]), np.asarray(value["pattern"], dtype=np.float32), value["diagnostics"])


def _prediction(model, x, y):
    value = model(x)
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, dict):
        value = value.get("pose", value.get("prediction"))
    if not isinstance(value, torch.Tensor):
        raise ValueError("surrogate must return a pose tensor, or a tuple whose first item is pose")
    if value.shape != y.shape and value.numel() == y.numel() and value.shape[0] == y.shape[0]:
        value = value.reshape_as(y)
    if value.shape != y.shape or not torch.isfinite(value).all():
        raise ValueError("invalid surrogate pose output")
    return value


def _mpjpe(pred, target, joints=None):
    if joints is not None:
        pred, target = pred[..., list(joints), :], target[..., list(joints), :]
    return torch.linalg.vector_norm(pred - target, dim=-1).reshape(len(pred), -1).mean(1)


def _bank(x, y, doses, config, name):
    # Detached CPU banks bound memory; caller records identities and hashes.
    x = torch.as_tensor(x).detach().cpu().to(torch.float32).clone()
    y = torch.as_tensor(y).detach().cpu().to(torch.float32).clone()
    d = torch.as_tensor(doses).detach().cpu().to(torch.float32).clone()
    if x.ndim != 4 or tuple(x.shape[1:]) != config.shape or len(x) < config.batch_size:
        raise ValueError(f"{name}: invalid TRAIN CSI bank or fewer than one full batch")
    if y.ndim not in (3, 4) or len(y) != len(x) or y.shape[-1] != 3 or y.shape[-2] < 4:
        raise ValueError(f"{name}: invalid TRAIN pose bank")
    if d.shape != (len(x),) or not torch.isfinite(d).all() or torch.any(d <= 0) or torch.any(d > 1):
        raise ValueError(f"{name}: paired positive doses must have shape (N,)")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or torch.any(x < 0) or torch.any(x > 1):
        raise ValueError(f"{name}: inputs/poses must be finite and CSI in [0,1]")
    return x, y, d


def _device(model):
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1:
        raise ValueError("surrogate must have parameters on one device")
    return next(iter(devices))


@contextmanager
def _frozen_surrogate(model):
    modes = [(m, m.training) for m in model.modules()]
    requires = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.eval()
        for p, _ in requires:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in requires:
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


def _targets(target_fn, y, d):
    target = target_fn(y.clone(), d.clone())
    if not isinstance(target, torch.Tensor) or target.shape != y.shape or not torch.isfinite(target).all():
        raise ValueError("target_fn must return finite poses with the original shape")
    return target.detach()


def _surrogate_sha256(model):
    rows = {}
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        rows[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                      "bytes_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    return _json_sha(rows)


def _siba_stream_batch(batch, config, device):
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise ValueError("SIBA stream must yield (TRAIN CSI, TRAIN pose, paired dose) triples")
    x, y, d = [torch.as_tensor(value).detach().to(device=device, dtype=torch.float32) for value in batch]
    if (x.ndim != 4 or tuple(x.shape[1:]) != config.shape or not 0 < len(x) <= config.batch_size
            or y.ndim not in (3, 4) or len(y) != len(x) or y.shape[-1] != 3
            or max(config.target_joints) >= y.shape[-2] or d.shape != (len(x),)
            or not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(d).all()
            or torch.any(x < 0) or torch.any(x > 1) or torch.any(d <= 0) or torch.any(d > 1)):
        raise ValueError("SIBA stream requires finite matching TRAIN tensors, [0,1] CSI and positive paired doses")
    return x, y, d


def fit_siba_stream(model, epoch_batches: Callable, target_fn: Callable, config: SIBAConfig,
                    *, dataset_sha256: str, n_train: int, progress=None,
                    save_state=None, resume_state=None):
    """Pinned source schedule, with a bounded-memory full-TRAIN epoch stream.

    ``epoch_batches(epoch)`` must deterministically yield each registered TRAIN
    example exactly once (B128 plus its final tail), using an epoch-addressable
    shuffle local to the caller. ``dataset_sha256`` must bind ordered TRAIN
    identities/content, paired doses AND this shuffle rule. No TEST may enter.
    Resume is at epoch boundaries; a partly completed epoch is intentionally
    replayed from its preceding durable state, not skipped or batch-resized.
    """
    if not isinstance(config, SIBAConfig) or not callable(epoch_batches):
        raise ValueError("SIBAConfig and a callable full-TRAIN epoch stream are required")
    n_train = _integer(n_train, "n_train")
    if (not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64
            or any(c not in "0123456789abcdef" for c in dataset_sha256)):
        raise ValueError("Full TRAIN dataset/dose/shuffle SHA256 is required")
    device = _device(model)
    original_model_sha = _surrogate_sha256(model)
    stream_hash = _json_sha({"dataset_sha256": dataset_sha256, "n_train": n_train,
                             "surrogate_sha256": original_model_sha})
    delta = torch.zeros(config.shape, device=device)
    mask = torch.zeros(config.shape[1:], device=device)
    history, refreshes, updates, start_epoch = [], 0, 0, 0
    best_loss, best, best_epoch = math.inf, None, None
    if resume_state is not None:
        state = _check_fit_state(resume_state, "siba-full-train-epochs", config, stream_hash)
        start_epoch = state["next_epoch"]
        if type(start_epoch) is not int or not 0 <= start_epoch <= config.epochs:
            raise ValueError("invalid SIBA resume epoch")
        delta = torch.as_tensor(state["delta"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(state["mask"], dtype=torch.float32, device=device)
        best = None if state["best"] is None else torch.as_tensor(state["best"], dtype=torch.float32)
        FrozenCSITrigger("siba", config, delta)
        if (mask.shape != torch.Size(config.shape[1:]) or not torch.all((mask == 0) | (mask == 1))
                or int(mask.sum()) != (config.k if start_epoch else 0)):
            raise ValueError("invalid SIBA resume shared mask")
        best_loss = math.inf if state["best_loss"] is None else float(state["best_loss"])
        best_epoch, history = state["best_epoch"], state["history"]
        refreshes, updates = state["refreshes"], state["updates"]
        if (type(updates) is not int or updates != start_epoch * math.ceil(n_train / config.batch_size)
                or type(refreshes) is not int or refreshes != math.ceil(start_epoch / config.support_refresh_epochs)
                or not isinstance(history, list) or len(history) != start_epoch):
            raise ValueError("invalid SIBA resume epoch/update history")
        if start_epoch:
            if best is None:
                raise ValueError("completed SIBA epochs require a selected minimum-epoch pattern")
            FrozenCSITrigger("siba", config, best)
            losses = [r.get("epoch_mean_batch_loss_m", float("nan")) for r in history]
            if (not all(math.isfinite(v) for v in losses) or best_epoch != int(np.argmin(losses))
                    or best_loss != min(losses)):
                raise ValueError("invalid SIBA minimum-epoch selection state")
        elif best is not None or best_epoch is not None or state["best_loss"] is not None:
            raise ValueError("initial SIBA state cannot contain a selected pattern")

    def checkpoint(next_epoch):
        if save_state:
            save_state(_fit_state("siba-full-train-epochs", config, stream_hash, {
                "next_epoch": next_epoch, "delta": delta.detach().cpu().tolist(),
                "mask": mask.detach().cpu().tolist(), "best": None if best is None else best.tolist(),
                "best_loss": None if not math.isfinite(best_loss) else best_loss,
                "best_epoch": best_epoch, "refreshes": refreshes, "updates": updates,
                "history": deepcopy(history),
            }))

    checkpoint(start_epoch)
    with _frozen_surrogate(model):
        for epoch in range(start_epoch, config.epochs):
            losses, consumed, n_batches, tail_seen = 0., 0, 0, False
            for batch in epoch_batches(epoch):
                if tail_seen:
                    raise ValueError("SIBA stream contains a batch after the final partial batch")
                xb, yb, db = _siba_stream_batch(batch, config, device)
                consumed += len(xb)
                if consumed > n_train:
                    raise ValueError("SIBA epoch stream exceeds registered full TRAIN count")
                tail_seen = len(xb) != config.batch_size
                delta = delta.detach().requires_grad_(True)
                target = _targets(target_fn, yb, db)
                pred = _prediction(model, (xb + db[:, None, None, None] * delta).clamp(0, 1), yb)
                loss = _mpjpe(pred, target, config.target_joints).clamp(max=config.loss_cap_m).mean()
                grad, = torch.autograd.grad(loss, delta)
                if not torch.isfinite(loss) or not torch.isfinite(grad).all():
                    raise ValueError("nonfinite SIBA trigger loss/gradient")
                # Source optimize_siba.py: i==0 AND epoch%epoch_step==0.
                # A shared delta gives the same gradient SIGN/ranking as the
                # source's mean of repeated per-example delta gradients.
                if n_batches == 0 and epoch % config.support_refresh_epochs == 0:
                    selected = torch.argsort(grad.detach().abs().sum(0).flatten(),
                                            descending=True, stable=True)[:config.k]
                    mask.zero_().view(-1)[selected] = 1
                    refreshes += 1
                delta = ((delta.detach() - config.eps * config.step_ratio * grad.sign())
                         * mask).clamp(-config.eps, config.eps)
                losses += float(loss.detach())
                n_batches += 1
                updates += 1
            if consumed != n_train or n_batches != math.ceil(n_train / config.batch_size):
                raise ValueError("SIBA epoch must consume every registered TRAIN frame exactly once")
            # The source compares the unweighted mean of batch losses for an
            # epoch and stores the corresponding END-OF-EPOCH pattern.
            epoch_loss = losses / n_batches
            if epoch_loss < best_loss:
                best_loss, best, best_epoch = epoch_loss, delta.detach().cpu().clone(), epoch
            event = {"epoch": epoch, "epoch_mean_batch_loss_m": epoch_loss,
                     "train_frames": consumed, "batches": n_batches,
                     "updates": updates, "support_refreshes": refreshes}
            history.append(event)
            if progress:
                progress(deepcopy(event))
            checkpoint(epoch + 1)
    if _surrogate_sha256(model) != original_model_sha:
        raise ValueError("SIBA clean surrogate changed during full TRAIN optimization")
    return FrozenCSITrigger("siba", config, best, {
        "optimization_scope": "registered full TRAIN epoch stream; no TEST",
        "dataset_dose_shuffle_sha256": dataset_sha256, "n_fit": n_train,
        "epochs": config.epochs, "batch_size": config.batch_size, "updates": updates,
        "support_refreshes": refreshes, "selected_epoch": best_epoch,
        "best_epoch_mean_batch_loss_m": best_loss,
        "selection_note": "minimum epoch-mean batch TRAIN loss; corresponding end-of-epoch pattern",
        "history": history, "surrogate_unchanged": True,
    })


def fit_siba(model, x, y, doses, target_fn: Callable, config: SIBAConfig, progress=None,
             *, save_state=None, resume_state=None):
    """In-memory FIXTURE convenience API; production must use fit_siba_stream.

    This deliberately does not claim an arbitrary supplied tensor bank is the
    official full TRAIN set. It uses the same source epoch/update mechanism.
    """
    x, y, doses = _bank(x, y, doses, config, "SIBA fixture bank")
    bank_hash = _json_sha({"x": _array_sha(x.numpy()), "y": _array_sha(y.numpy()),
                           "doses": _array_sha(doses.numpy()),
                           "order": "local numpy SeedSequence(seed,epoch), permutation"})

    def batches(epoch):
        order = np.random.default_rng(np.random.SeedSequence([config.seed, epoch])).permutation(len(x))
        for start in range(0, len(x), config.batch_size):
            indices = order[start:start + config.batch_size]
            yield x[indices], y[indices], doses[indices]

    trigger = fit_siba_stream(model, batches, target_fn, config,
        dataset_sha256=bank_hash, n_train=len(x), progress=progress,
        save_state=save_state, resume_state=resume_state)
    diagnostics = trigger.provenance()["diagnostics"]
    diagnostics.update(optimization_scope="reduced in-memory fixture only; NOT production full TRAIN",
                       fixture_only=True)
    return FrozenCSITrigger("siba", config, trigger.pattern, diagnostics)


def dct_pattern(genes, config):
    """Orthonormal IDCT of sparse per-antenna (row,column,magnitude) genes."""
    genes = np.asarray(genes, dtype=np.float64)
    if genes.shape != (config.shape[0], config.coefficients_per_channel, 3) or not np.isfinite(genes).all():
        raise ValueError("invalid LADDER genes")
    channels, height, width = config.shape
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    out = np.zeros(config.shape, dtype=np.float64)
    for c in range(channels):
        seen = set()
        for u, v, value in genes[c]:
            ui, vi = int(round(u)), int(round(v))
            if not 0 <= ui < config.low_band[0] or not 0 <= vi < config.low_band[1] or abs(value) > config.coefficient_bound + 1e-10:
                raise ValueError("LADDER gene outside registered low-frequency/coefficient bounds")
            if (ui, vi) in seen:
                raise ValueError("LADDER frequency coordinates must be distinct per antenna")
            seen.add((ui, vi))
            a = math.sqrt((1 if ui == 0 else 2) / height) * np.cos(math.pi * (rows + .5) * ui / height)
            b = math.sqrt((1 if vi == 0 else 2) / width) * np.cos(math.pi * (cols + .5) * vi / width)
            out[c] += value * np.outer(a, b)
    return out.astype(np.float32)


def pareto_fronts(objectives):
    value = np.asarray(objectives, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("expected finite three-objective vectors")
    remaining, fronts = list(range(len(value))), []
    while remaining:
        front = [i for i in remaining if not any(
            np.all(value[j] <= value[i]) and np.any(value[j] < value[i]) for j in remaining if i != j)]
        fronts.append(front)
        remaining = [i for i in remaining if i not in front]
    return fronts


def preference_select(objectives, count, config):
    _integer(count, "selection count")
    value = np.asarray(objectives, dtype=np.float64)
    if count > len(value):
        raise ValueError("cannot select more candidates than available")
    selected = []
    for front in pareto_fronts(value):
        if len(selected) + len(front) <= count:
            selected.extend(front)
        else:
            # Paper Algorithm 3: once the next complete front will not fit,
            # preference ranks all remaining (not only that front) candidates.
            rest = [i for i in range(len(value)) if i not in selected]
            dist = np.linalg.norm((value - np.asarray(config.preference)) / np.asarray(config.objective_scales), axis=1)
            selected.extend(sorted(rest, key=lambda i: (dist[i], i))[:count - len(selected)])
            break
        if len(selected) == count:
            break
    return selected


def _repair(genes, config):
    value = np.asarray(genes, dtype=np.float64).copy()
    value[..., 0] = np.rint(value[..., 0]).clip(0, config.low_band[0] - 1)
    value[..., 1] = np.rint(value[..., 1]).clip(0, config.low_band[1] - 1)
    value[..., 2] = value[..., 2].clip(-config.coefficient_bound, config.coefficient_bound)
    for channel in value:
        used = set()
        for gene in channel:
            key = (int(gene[0]), int(gene[1]))
            if key in used:
                key = next((u, v) for u in range(config.low_band[0]) for v in range(config.low_band[1]) if (u, v) not in used)
                gene[:2] = key
            used.add(key)
    return value


def _variation(a, b, config, rng):
    # Simulated binary crossover followed by bounded polynomial mutation.
    u = rng.random(a.shape)
    beta = np.where(u <= .5, (2 * u) ** (1 / (config.crossover_eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (config.crossover_eta + 1)))
    child = .5 * ((1 + beta) * a + (1 - beta) * b)
    lower = np.array([0., 0., -config.coefficient_bound])
    upper = np.array([config.low_band[0] - 1., config.low_band[1] - 1., config.coefficient_bound])
    child = np.clip(child, lower, upper)
    span = upper - lower
    for idx in np.ndindex(child.shape):
        if rng.random() >= 1 / child.size or span[idx[-1]] == 0:
            continue
        j, z = idx[-1], child[idx]
        r = rng.random()
        if r <= .5:
            q = 2*r + (1 - 2*r) * (1 - (z-lower[j])/span[j]) ** (config.mutation_eta+1)
            delta = q ** (1/(config.mutation_eta+1)) - 1
        else:
            q = 2*(1-r) + 2*(r-.5) * (1 - (upper[j]-z)/span[j]) ** (config.mutation_eta+1)
            delta = 1 - q ** (1/(config.mutation_eta+1))
        child[idx] = z + delta * span[j]
    return _repair(child, config)


def _mixed_batch(trigger, xb, yb, db, poisoned, target_fn):
    if not poisoned.any():
        return xb, yb
    result_x, result_y = xb.clone(), yb.clone()
    result_x[poisoned] = trigger.apply_tensor(xb[poisoned], db[poisoned])
    result_y[poisoned] = _targets(target_fn, yb[poisoned], db[poisoned])
    return result_x, result_y


def _evaluate_candidate(model, genes, bank, score_bank, target_fn, config):
    """Actual poison/retrain/score/rollback; never fixed-surrogate scoring."""
    x, y, doses = bank
    sx, sy, sd = score_bank
    device = _device(model)
    trigger = FrozenCSITrigger("ladder", config, dct_pattern(genes, config))
    rng = np.random.default_rng(config.seed + 901)
    poisoned = np.zeros(len(x), dtype=bool)
    poisoned[rng.permutation(len(x))[:max(1, int(len(x) * config.poison_ratio))]] = True
    score_poisoned = np.zeros(len(sx), dtype=bool)
    score_poisoned[rng.permutation(len(sx))[:max(1, int(len(sx) * config.poison_ratio))]] = True
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # All candidates share initial weights, optimizer reset, poison identities,
    # minibatch order, and RNG seed. The passed clean surrogate is never mutated.
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(config.seed + 902)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(config.seed + 902)
        candidate = deepcopy(model)
        candidate.train()
        for p in candidate.parameters():
            p.requires_grad_(True)
            p.grad = None
        optimizer = torch.optim.SGD(candidate.parameters(), lr=config.retrain_lr,
                                    momentum=config.retrain_momentum, weight_decay=config.retrain_weight_decay)
        updates = 0
        for _ in range(config.retrain_epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x) - config.batch_size + 1, config.batch_size):
                idx = order[start:start + config.batch_size]
                xb, yb, db = x[idx].to(device), y[idx].to(device), doses[idx].to(device)
                xb, yb = _mixed_batch(trigger, xb, yb, db, torch.as_tensor(poisoned[idx], device=device), target_fn)
                optimizer.zero_grad(set_to_none=True)
                loss = _mpjpe(_prediction(candidate, xb, yb), yb).mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite LADDER retraining loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in candidate.parameters()):
                    raise ValueError("nonfinite LADDER retraining gradient")
                optimizer.step()
                updates += 1
        candidate.eval()
        mixed_sum, n = 0., 0
        with torch.no_grad():
            for start in range(0, len(sx), config.batch_size):
                sl = slice(start, start + config.batch_size)
                xb, yb, db = sx[sl].to(device), sy[sl].to(device), sd[sl].to(device)
                xb, yb = _mixed_batch(trigger, xb, yb, db, torch.as_tensor(score_poisoned[sl], device=device), target_fn)
                errors = _mpjpe(_prediction(candidate, xb, yb), yb)
                mixed_sum += float(errors.sum())
                n += len(errors)
        del optimizer, candidate
    objectives = [mixed_sum / n, float(np.linalg.norm(genes[..., 2])),
                  float(np.linalg.norm(genes[..., :2].reshape(-1, 2).sum(0)))]
    if not np.isfinite(objectives).all():
        raise ValueError("nonfinite LADDER objective")
    return objectives, updates


def _fit_state(method, config, bank_hash, payload):
    state = {"version": IMPLEMENTATION, "method": method, "config": asdict(config), "bank_hash": bank_hash,
             "payload": deepcopy(payload)}
    return {**state, "sha256": _json_sha(state)}


def _check_fit_state(saved, method, config, bank_hash):
    state = deepcopy(saved)
    digest = state.pop("sha256", None)
    if digest != _json_sha(state) or state.get("version") != IMPLEMENTATION or state.get("method") != method:
        raise ValueError("fit resume checksum/version/method mismatch")
    if _json_sha(state.get("config")) != _json_sha(asdict(config)) or state.get("bank_hash") != bank_hash:
        raise ValueError("fit resume config/TRAIN bank mismatch")
    return state["payload"]


def require_ladder_author_search():
    """Fail before training/search, rather than silently invent missing details."""
    raise RuntimeError(
        "LADDER author search is unavailable in pinned a752cf48805283fcf7becd88e4382199e40de2c1: "
        "train_freq_attack.py loads missing saved_vars/3objs_vgg11_cifar10_selected.pt; "
        "the public tree has no complete executable MOEA search. The paper describes SBX/PM and "
        "rNDSort but does not supply exact variation settings or a source-defensible pose-loss "
        "preference region. No LADDER trigger search or full training was started by this call. "
        "Obtain the missing author search material or explicitly approve a separately named "
        "experimental implementation; do not report the old custom prototype as faithful LADDER."
    )


def fit_ladder(*args, **kwargs):
    """Production entry point intentionally blocked; never dispatch prototype."""
    require_ladder_author_search()


def fit_ladder_experimental(model, x, y, doses, target_fn: Callable, config: LADDERConfig, progress=None,
                           *, score_x, score_y, score_doses, save_state=None, resume_state=None):
    """UNAPPROVED archival prototype; NOT a faithful LADDER baseline.

    Kept for regression/provenance of unpublished work only. In particular,
    objective scaling, preference, disjoint scoring and SBX/PM settings below
    are discretionary choices, not recovered executable author search code.
    No production runner may call this as method ``LADDER``.
    """
    bank = _bank(x, y, doses, config, "LADDER fitting")
    score_bank = _bank(score_x, score_y, score_doses, config, "LADDER scoring")
    rng = np.random.default_rng(config.seed)
    population = []
    for _ in range(config.population):
        genes = np.empty((config.shape[0], config.coefficients_per_channel, 3))
        for c in range(config.shape[0]):
            coords = rng.choice(math.prod(config.low_band), size=config.coefficients_per_channel, replace=False)
            genes[c, :, 0], genes[c, :, 1] = coords // config.low_band[1], coords % config.low_band[1]
            genes[c, :, 2] = rng.uniform(-config.coefficient_bound, config.coefficient_bound, config.coefficients_per_channel)
        population.append(genes)
    objectives, history, evaluations, updates = [], [], 0, 0
    generation, children, child_scores = 0, [], []
    bank_hash = _json_sha({"fit": [_array_sha(v.numpy()) for v in bank],
                           "score": [_array_sha(v.numpy()) for v in score_bank]})
    if resume_state is not None:
        state = _check_fit_state(resume_state, "ladder", config, bank_hash)
        population = [np.asarray(v, dtype=np.float64) for v in state["population"]]
        children = [np.asarray(v, dtype=np.float64) for v in state["children"]]
        for genes in population + children:
            dct_pattern(genes, config)
        objectives, child_scores, history = state["objectives"], state["child_scores"], state["history"]
        evaluations, updates, generation = state["evaluations"], state["updates"], state["generation"]
        if len(population) != config.population or len(objectives) > len(population) or len(child_scores) > len(children):
            raise ValueError("invalid LADDER resume population/scores")
        if not 0 <= generation <= config.generations or len(children) not in (0, config.population):
            raise ValueError("invalid LADDER resume generation/offspring")
        rng.bit_generator.state = state["rng"]

    def checkpoint():
        if save_state:
            save_state(_fit_state("ladder", config, bank_hash, {
                "population": [v.tolist() for v in population], "objectives": deepcopy(objectives),
                "children": [v.tolist() for v in children], "child_scores": deepcopy(child_scores),
                "history": deepcopy(history), "evaluations": evaluations, "updates": updates,
                "generation": generation, "rng": deepcopy(rng.bit_generator.state),
            }))

    def score(genes, generation):
        nonlocal evaluations, updates
        objective, count = _evaluate_candidate(model, genes, bank, score_bank, target_fn, config)
        evaluations += 1
        updates += count
        event = {"candidate": evaluations, "generation": generation, "objectives": objective,
                 "candidate_retrain_updates": count, "total_retrain_updates": updates}
        if progress:
            progress(deepcopy(event))
        return objective

    checkpoint()
    for index in range(len(objectives), len(population)):
        objectives.append(score(population[index], 0))
        checkpoint()
    while generation < config.generations or children:
        if not children:
            generation += 1
            children, child_scores = [], []
            for _ in range(config.population):
                parents = rng.choice(len(population), 2, replace=False)
                children.append(_variation(population[int(parents[0])], population[int(parents[1])], config, rng))
            checkpoint()
        for index in range(len(child_scores), len(children)):
            child_scores.append(score(children[index], generation))
            checkpoint()
        combined, all_scores = population + children, objectives + child_scores
        selected = preference_select(all_scores, config.population, config)
        population, objectives = [combined[i] for i in selected], [all_scores[i] for i in selected]
        history.append({"generation": generation, "objectives": deepcopy(objectives)})
        children, child_scores = [], []
        checkpoint()
    # Predetermined preference, not TEST, selects one frozen final trigger.
    distances = np.linalg.norm((np.asarray(objectives) - np.asarray(config.preference)) / np.asarray(config.objective_scales), axis=1)
    selected = int(np.argmin(distances))
    return FrozenCSITrigger("ladder", config, dct_pattern(population[selected], config), {
        "experimental_only": True, "production_search_ready": False,
        "identity": "unapproved LADDER-inspired archival prototype, not a source-faithful baseline",
        "optimization_bank": "disjoint TRAIN fitting and scoring banks; identities audited by caller",
        "n_fit": len(bank[0]), "n_score": len(score_bank[0]),
        "candidate_evaluations": evaluations, "candidate_retrain_updates_total": updates,
        "candidate_retrain_epochs": config.retrain_epochs,
        "objectives": ["mixed clean/poison full-joint MPJPE metres after retraining", "DCT coefficient L2", "norm of sum of DCT frequency coordinates"],
        "selected_genes": population[selected].tolist(), "selected_objectives": objectives[selected],
        "final_population_objectives": objectives, "history": history,
        "surrogate_unchanged": True, "candidate_training_rollback": "deepcopy of original clean surrogate per candidate",
        "not_claimed": "image reproduction, empirical preprocessing robustness, RF stealthiness, equal-noise budgets, TEST efficacy",
    })
