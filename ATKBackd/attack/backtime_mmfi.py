"""Independent, source-pinned BackTime-CSI/Pose task adaptation.

This is not a reproduction of forecasting continuous multivariate sequences.
CSI has 342 input variables x 10 packets, but its output is one 17-joint pose,
not the future of those variables. Thus no future CSI target pattern is
injected; one pose horizon makes source soft-identification weights degenerate
to one. The caller supplies TRAIN poison identities/policy explicitly.

Preserved: fixed sparse variable subset, learned graph/GCN and constant branch,
bounded input-dependent generation from preceding packets, historical-value
anchored replacement, alternating poisoned-surrogate/generator optimization,
and temporal DC-sum regularization. A new victim must subsequently start from
scratch. No victim parameters or TEST inputs are accepted for fitting.

No upstream code is copied (no license in the pinned upstream tree). All
operations are independently expressed in PyTorch; unused upstream DGL imports
do not require a DGL installation. Read provenance() for material differences.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


IMPLEMENTATION = "independent-backtime-mmfi-pose-stream-v2"
SOURCE = {
    "repository": "https://github.com/xiaolin-cs/BackTime",
    "commit": "9e0a1f9deecbacfdfea9159e6e8892148b44da9f",
    "paper": "https://arxiv.org/abs/2410.02195",
    "venue": "NeurIPS 2024",
    "license": "No license file found in pinned recursive tree; independent implementation, no vendored source",
    "source_sha256_lf": {
        "trigger.py": "40e914682b81d50157037e233039e729baa6a0d2d749867d8733751d9ab6f0a2",
        "attack.py": "35d084c632b00275037cc72376d032cb1ed4db66c064b35bc18a59ead8366a4c",
        "trainer.py": "7d70def7405a50a4a7c30f24013d70af54508dc87113aaa5cde5c9416e884b52",
        "configs/train_config.yaml": "6707b432b0e026e5f0e842dc574839d99cfc1f8b149716c6d99e98ba6b2706e8",
    },
}


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name, zero=False):
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)) or (float(value) < 0 if zero else float(value) <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


@dataclass(frozen=True)
class BackTimeConfig:
    shape: tuple[int, int, int] = (3, 114, 10)
    seed: int = 42
    history_len: int = 6
    trigger_len: int = 4
    hidden_dim: int = 64
    variable_fraction: float = .3
    eps: float = .2
    alpha: float = .5
    lam_norm: float = .05
    attack_lr: float = .005
    surrogate_lr: float = .0001
    fit_epochs: int = 100
    warmup: int = 10
    batch_size: int = 64
    fft_components: int = 200
    target_joints: tuple[int, ...] = (2, 3)

    def __post_init__(self):
        shape = tuple(_integer(v, "shape") for v in self.shape)
        if len(shape) != 3:
            raise ValueError("shape must have three dimensions")
        object.__setattr__(self, "shape", shape)
        for name in ("history_len", "trigger_len", "hidden_dim", "fit_epochs", "batch_size", "fft_components"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "seed", _integer(self.seed, "seed", 0))
        object.__setattr__(self, "warmup", _integer(self.warmup, "warmup", 0))
        if self.history_len + self.trigger_len != shape[-1]:
            raise ValueError("history_len + trigger_len must exactly cover the CSI packet window")
        for name in ("variable_fraction", "eps", "attack_lr", "surrogate_lr"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("alpha", "lam_norm"):
            object.__setattr__(self, name, _positive(getattr(self, name), name, zero=True))
        if self.variable_fraction > 1 or self.eps > 1 or self.alpha > 1:
            raise ValueError("fraction, eps and alpha must not exceed one")
        joints = tuple(_integer(v, "target joint", 0) for v in self.target_joints)
        if not joints or len(set(joints)) != len(joints):
            raise ValueError("target_joints must be distinct")
        object.__setattr__(self, "target_joints", joints)

    @property
    def n_variables(self):
        return self.shape[0] * self.shape[1]

    @property
    def n_attacked_variables(self):
        return max(1, int(round(self.n_variables * self.variable_fraction)))


def _sha(value):
    # Stream tensor bytes rather than expanding model/Adam tensors into millions
    # of Python floats and a large JSON string at every epoch checkpoint.
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor:")
            digest.update(str(tensor.dtype).encode())
            visit(list(tensor.shape))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"array:")
            digest.update(str(item.dtype).encode())
            visit(list(item.shape))
            digest.update(item.tobytes(order="C"))
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda k: str(k)):
                visit(str(key))
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"list[")
            for part in item:
                visit(part)
            digest.update(b"]")
        else:
            if isinstance(item, np.generic):
                item = item.item()
            digest.update(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            digest.update(b";")

    visit(value)
    return digest.hexdigest()


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return deepcopy(value)


def error_ranked_selection(errors, count, seed=42):
    """Seeded subset of top 2K TRAIN errors for independent CSI frames.

    This deliberately does NOT claim the source's non-overlapping poisoning of
    a continuous raw time series. Frame windows are independent and the caller
    must record this policy separately from uniform poisoning.
    """
    values = np.asarray(errors, dtype=np.float64)
    count, seed = _integer(count, "count"), _integer(seed, "seed", 0)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0) or count > len(values):
        raise ValueError("expected finite nonnegative TRAIN errors and count <= N")
    order = np.argsort(-values, kind="stable")[:min(len(values), 2 * count)]
    return np.sort(np.random.default_rng(seed).choice(order, count, replace=False)).astype(np.int64)


def train_spectral_features(x, config):
    """TRAIN-only packet-axis low-frequency graph features, no frame stitching.

    The image-free CSI task lacks the single continuous MTS of the paper.
    Aggregate low packet-frequency complex coefficients across the fixed TRAIN
    bank instead of concatenating unrelated frames into a fictitious timeline.
    Unlike the upstream fft_compress sorting expression, these are explicitly
    low frequencies; the upstream reverses the variable axis, not FFT order.
    """
    value = torch.as_tensor(x).detach().cpu().to(torch.float32)
    if value.ndim != 4 or tuple(value.shape[1:]) != config.shape or not torch.isfinite(value).all():
        raise ValueError("expected finite TRAIN CSI bank")
    flat = value.reshape(len(value), config.n_variables, config.shape[-1])
    spectrum = torch.fft.rfft(flat, dim=-1)[..., :config.fft_components]
    amplitude = spectrum.abs().mean(0)
    phase = spectrum.mean(0).angle()
    frequencies = torch.fft.rfftfreq(config.shape[-1])[:config.fft_components].expand(config.n_variables, -1)
    return torch.cat((amplitude, phase, frequencies), dim=-1).contiguous()


class _GraphGenerator(nn.Module):
    def __init__(self, config, features, variables, mean=0., std=1.):
        super().__init__()
        self.config = config
        self.register_buffer("features", torch.as_tensor(features, dtype=torch.float32).clone())
        self.register_buffer("variables", torch.as_tensor(variables, dtype=torch.long).clone())
        self.register_buffer("train_mean", torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer("train_std", torch.tensor(float(std), dtype=torch.float32))
        if not math.isfinite(float(mean)) or not math.isfinite(float(std)) or float(std) <= 0:
            raise ValueError("finite TRAIN mean and positive standard deviation required")
        if self.features.ndim != 2 or self.features.shape[0] != config.n_attacked_variables or self.features.shape[1] % 3:
            raise ValueError("invalid TRAIN graph feature shape")
        if (self.variables.shape != (config.n_attacked_variables,) or
                len(torch.unique(self.variables)) != len(self.variables) or
                torch.any(self.variables < 0) or torch.any(self.variables >= config.n_variables)):
            raise ValueError("invalid sparse variable subset")
        if not torch.isfinite(self.features).all():
            raise ValueError("graph features must be finite")
        self.structure = nn.Sequential(nn.Linear(self.features.shape[1], 256), nn.ReLU(), nn.Linear(256, 64))
        self.constant = nn.Sequential(nn.Linear(config.history_len, config.hidden_dim), nn.ReLU(),
                                      nn.Linear(config.hidden_dim, config.trigger_len))
        self.conv1 = nn.Parameter(torch.empty(config.history_len, config.hidden_dim))
        self.conv2 = nn.Parameter(torch.empty(config.hidden_dim, config.trigger_len))
        nn.init.xavier_uniform_(self.conv1)
        nn.init.xavier_uniform_(self.conv2)
        for layer in self.constant:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.uniform_(layer.bias, -.2, .2)

    def graph(self):
        embeddings = F.normalize(self.structure(self.features), dim=-1, eps=1e-12)
        adjacency = (embeddings @ embeddings.T).clamp_min(0)
        identity = torch.eye(len(self.variables), device=adjacency.device, dtype=adjacency.dtype)
        adjacency = adjacency * (1 - identity) + identity
        degree = adjacency.sum(1).clamp_min(1e-12).rsqrt()
        return degree[:, None] * adjacency * degree[None, :]

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (len(self.variables), self.config.history_len):
            raise ValueError("expected batch x attacked variables x history packets")
        normalized = (history - self.train_mean) / self.train_std
        adjacency = self.graph()
        hidden = F.relu(torch.einsum("ij,bjh->bih", adjacency, normalized @ self.conv1))
        dynamic = torch.einsum("ij,bjt->bit", adjacency, hidden @ self.conv2)
        zeros = torch.zeros_like(normalized)
        constant = self.constant(zeros)
        perturbation = (self.config.eps * (1 - self.config.alpha) * dynamic.tanh()
                        + self.config.eps * self.config.alpha * constant.tanh())
        return (normalized[..., -1:] + perturbation) * self.train_std + self.train_mean, perturbation


def _validate_input(csi, config):
    if not isinstance(csi, torch.Tensor) or not csi.is_floating_point() or csi.ndim not in (3, 4) or tuple(csi.shape[-3:]) != config.shape:
        raise ValueError("expected floating CHW/BCHW CSI")
    if not torch.isfinite(csi).all() or torch.any(csi < 0) or torch.any(csi > 1):
        raise ValueError("CSI must be finite and normalized to [0,1]")


def _doses(doses, csi):
    d = torch.as_tensor(doses, device=csi.device, dtype=csi.dtype)
    batch = len(csi) if csi.ndim == 4 else 1
    if d.ndim == 0:
        d = d.expand(batch)
    if d.shape != (batch,) or not torch.isfinite(d).all() or torch.any(d < 0) or torch.any(d > 1):
        raise ValueError("dose must be scalar or per-example values in [0,1]")
    return d


def _apply(generator, csi, doses):
    config = generator.config
    _validate_input(csi, config)
    single = csi.ndim == 3
    values = csi.unsqueeze(0) if single else csi
    d = _doses(doses, csi)
    if not bool(torch.any(d != 0)):
        return csi.clone(), torch.zeros((len(values), len(generator.variables), config.trigger_len), device=csi.device, dtype=csi.dtype)
    if next(generator.parameters()).device != csi.device:
        raise ValueError("Move the frozen BackTime generator to the CSI device with trigger.to(device)")
    flat = values.reshape(len(values), config.n_variables, config.shape[-1])
    variables = generator.variables
    history = flat[:, variables, :config.history_len].to(next(generator.parameters()).dtype)
    replacement, perturbation = generator(history)
    original = flat[:, variables, config.history_len:]
    segment = (original + d[:, None, None] * (replacement.to(csi.dtype) - original)).clamp(0, 1)
    segment = torch.where(d[:, None, None] == 0, original, segment)
    segment = torch.where(d[:, None, None] == 1, replacement.to(csi.dtype).clamp(0, 1), segment)
    out = flat.clone()
    out[:, variables, config.history_len:] = segment
    out = out.reshape_as(values)
    return (out[0] if single else out), perturbation


class FrozenBackTimeTrigger:
    baseline_name = "BackTime-CSI/Pose adapted"
    requires_deferred_injection = True
    requires_event_metadata = False

    def __init__(self, config, generator, diagnostics=None, *, selection_scores=None):
        if not isinstance(config, BackTimeConfig) or not isinstance(generator, _GraphGenerator) or generator.config != config:
            raise ValueError("BackTime config/generator mismatch")
        if any(not torch.isfinite(v).all() for v in generator.state_dict().values() if v.is_floating_point()):
            raise ValueError("BackTime generator must have finite parameters and features")
        self.config, self.shape = config, config.shape
        self.generator = deepcopy(generator).eval()
        for parameter in self.generator.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._diagnostics = deepcopy(diagnostics or {})
        _sha(self._diagnostics)
        self._selection_scores = None
        if selection_scores is not None:
            scores = torch.as_tensor(selection_scores, dtype=torch.float32).detach().cpu().clone()
            if scores.ndim != 1 or not torch.isfinite(scores).all() or torch.any(scores < 0):
                raise ValueError("selection scores must be finite nonnegative TRAIN losses")
            self._selection_scores = scores

    @property
    def training_indices(self):
        """A fresh list of the full-TRAIN adaptive poison recipients, if fitted."""
        return list(self._diagnostics.get("poison_indices", []))

    @property
    def training_selection_scores(self):
        """Immutable evidence: clean warmup online SmoothL1 scores, not TEST."""
        return None if self._selection_scores is None else self._selection_scores.clone()

    def to(self, device):
        self.generator.to(device)
        return self

    def eval(self):
        self.generator.eval()
        return self

    def apply_tensor(self, csi, dose):
        return _apply(self.generator, csi, dose)[0]

    def inject(self, csi, dose, eps=None):
        """CPU compatibility fallback; production uses deferred CUDA batches."""
        array = np.asarray(csi)
        if array.dtype.kind != "f" or array.shape != self.shape:
            raise ValueError("expected floating CHW CSI")
        if eps is not None:
            _positive(eps, "compatibility eps", zero=True)
        generator = self.generator if next(self.generator.parameters()).device.type == "cpu" else deepcopy(self.generator).cpu()
        with torch.no_grad():
            return _apply(generator, torch.from_numpy(array.copy()), dose)[0].numpy()

    def provenance(self):
        return {
            "implementation": IMPLEMENTATION, "source": deepcopy(SOURCE), "config": asdict(self.config),
            "generator_sha256": _sha(self.generator.state_dict()),
            "variables": self.generator.variables.detach().cpu().tolist(),
            "features_sha256": _sha(self.generator.features),
            "selection_scores_sha256": None if self._selection_scores is None else _sha(self._selection_scores),
            "dose_semantics": "selected final packets: clip(x + dose*(historical_anchor + bounded_g(history) - x),0,1); dose0 exact identity",
            "epsilon_semantics": "bounds generated deviation in TRAIN-global standardized units; raw anchor-relative cap is eps*TRAINstd, NOT residual to original input",
            "train_mean": float(self.generator.train_mean.detach().cpu()),
            "train_std": float(self.generator.train_std.detach().cpu()),
            "adaptation_differences": [
                "342 amplitude variables and 10 packets replace continuous same-variable MTS forecasting",
                "one pose horizon replaces future CSI pattern injection; soft-identification weight is one",
                "TRAIN pose target joints replace forecast target variables; no GT is read at trigger inference",
                "all packet-axis FFT features up to source cap200 are pooled over full TRAIN frames, not concatenated across unrelated frames",
                "author FFT argsort variable-axis reversal is retained literally in the streaming adapter; it is not renamed low-frequency filtering",
                "caller explicitly supplies poison recipient policy; independent frames do not reproduce raw-timestamp non-overlap",
                "streaming fit uses a fresh surrogate template COPY, source Adam/100 epochs/warmup10 condition; original template and new victim unchanged",
                "shape/DC-sum penalty is retained; no empirical CSI stealth or preprocessing robustness claim",
            ], "diagnostics": deepcopy(self._diagnostics),
        }

    def state_dict(self):
        value = {"version": IMPLEMENTATION, "config": asdict(self.config),
                 "generator": _cpu(self.generator.state_dict()), "diagnostics": deepcopy(self._diagnostics),
                 "selection_scores": self.training_selection_scores}
        return {**value, "sha256": _sha(value)}

    @classmethod
    def from_state_dict(cls, saved):
        value = _cpu(saved)
        digest = value.pop("sha256", None)
        if digest != _sha(value) or value.get("version") != IMPLEMENTATION:
            raise ValueError("BackTime frozen state checksum/version mismatch")
        config = BackTimeConfig(**value["config"])
        state = value["generator"]
        with torch.random.fork_rng(devices=[]):
            generator = _GraphGenerator(config, state["features"], state["variables"])
        generator.load_state_dict(state, strict=True)
        if any(not torch.isfinite(v).all() for v in generator.state_dict().values() if v.is_floating_point()):
            raise ValueError("BackTime frozen state contains nonfinite weights")
        return cls(config, generator, value["diagnostics"], selection_scores=value["selection_scores"])


def _prediction(model, x, y):
    value = model(x)
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, dict):
        value = value.get("pose", value.get("prediction"))
    if not isinstance(value, torch.Tensor):
        raise ValueError("surrogate must return pose predictions")
    if value.shape != y.shape and value.shape[0] == len(y) and value.numel() == y.numel():
        value = value.reshape_as(y)
    if value.shape != y.shape or not torch.isfinite(value).all():
        raise ValueError("invalid surrogate pose predictions")
    return value


def _target(target_fn, y, d):
    value = target_fn(y.clone(), d.clone())
    if not isinstance(value, torch.Tensor) or value.shape != y.shape or not torch.isfinite(value).all():
        raise ValueError("target_fn must produce finite matching poses")
    return value.detach()


@contextmanager
def _frozen(model, *, eval_mode=True):
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        if eval_mode:
            model.eval()
        for p, _ in flags:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


def _optimizer_to(optimizer, device):
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor) and key != "step":
                values[key] = value.to(device)


def fit_backtime(clean_model, x, y, doses, target_fn, config, *, poison_mask,
                 progress=None, save_state=None, resume_state=None):
    if not isinstance(config, BackTimeConfig):
        raise ValueError("BackTimeConfig required")
    x = torch.as_tensor(x).detach().cpu().to(torch.float32).clone()
    y = torch.as_tensor(y).detach().cpu().to(torch.float32).clone()
    d = torch.as_tensor(doses).detach().cpu().to(torch.float32).clone()
    mask = torch.as_tensor(poison_mask).detach().cpu().clone()
    _validate_input(x, config)
    if len(x) < config.batch_size or y.ndim not in (3, 4) or len(y) != len(x) or y.shape[-1] != 3 or not torch.isfinite(y).all():
        raise ValueError("invalid TRAIN input/pose fitting bank")
    if max(config.target_joints) >= y.shape[-2]:
        raise ValueError("target joint outside pose")
    if mask.dtype != torch.bool or mask.shape != (len(x),) or not bool(mask.any()) or bool(mask.all()):
        raise ValueError("explicit TRAIN boolean poison_mask must include poisoned AND clean samples")
    if d.shape != (len(x),) or not torch.isfinite(d).all() or torch.any(d <= 0) or torch.any(d > 1):
        raise ValueError("paired positive TRAIN doses required")
    devices = {p.device for p in clean_model.parameters()}
    if len(devices) != 1:
        raise ValueError("clean surrogate must have parameters on one device")
    device = next(iter(devices))
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    original_hash = _sha(clean_model.state_dict())
    binding = {"config": asdict(config), "bank_sha256": _sha([x, y, d, mask]), "clean_model_sha256": original_hash}
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(config.seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)
        variables = np.sort(rng.choice(config.n_variables, config.n_attacked_variables, replace=False))
        features = train_spectral_features(x, config)[variables]
        generator = _GraphGenerator(config, features, variables).to(device)
        surrogate = deepcopy(clean_model).to(device)
        for p in surrogate.parameters():
            p.requires_grad_(True)
            p.grad = None
        surrogate_optim = torch.optim.Adam(surrogate.parameters(), lr=config.surrogate_lr)
        generator_optim = torch.optim.Adam(generator.parameters(), lr=config.attack_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optim, milestones=[20, 40], gamma=.9)
        epoch, history, surrogate_updates, generator_updates = 0, [], 0, 0
        if resume_state is not None:
            state = _cpu(resume_state)
            digest = state.pop("sha256", None)
            if digest != _sha(state) or state.get("version") != IMPLEMENTATION or _sha(state.get("binding")) != _sha(binding):
                raise ValueError("BackTime search state checksum/source-model/TRAIN/config mismatch")
            epoch = state["epoch"]
            if not isinstance(epoch, int) or not 0 <= epoch <= config.fit_epochs:
                raise ValueError("invalid BackTime resume epoch")
            generator.load_state_dict(state["generator"], strict=True)
            surrogate.load_state_dict(state["surrogate"], strict=True)
            surrogate_optim.load_state_dict(state["surrogate_optimizer"])
            generator_optim.load_state_dict(state["generator_optimizer"])
            _optimizer_to(surrogate_optim, device)
            _optimizer_to(generator_optim, device)
            scheduler.load_state_dict(state["scheduler"])
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"], device)
            history = state["history"]
            surrogate_updates, generator_updates = state["surrogate_updates"], state["generator_updates"]

        def checkpoint():
            if save_state:
                state = {"version": IMPLEMENTATION, "binding": binding, "epoch": epoch,
                         "generator": _cpu(generator.state_dict()), "surrogate": _cpu(surrogate.state_dict()),
                         "surrogate_optimizer": _cpu(surrogate_optim.state_dict()),
                         "generator_optimizer": _cpu(generator_optim.state_dict()), "scheduler": _cpu(scheduler.state_dict()),
                         "numpy_rng": deepcopy(rng.bit_generator.state), "torch_rng": torch.get_rng_state().clone(),
                         "cuda_rng": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
                         "history": deepcopy(history), "surrogate_updates": surrogate_updates, "generator_updates": generator_updates}
                save_state({**state, "sha256": _sha(state)})

        checkpoint()
        for current in range(epoch, config.fit_epochs):
            generator.eval()
            surrogate.train()
            order = rng.permutation(len(x))
            train_sum, train_n = 0., 0
            for start in range(0, len(x), config.batch_size):
                idx = order[start:start + config.batch_size]
                xb, yb, db, mb = x[idx].to(device), y[idx].to(device), d[idx].to(device), mask[idx].to(device)
                with torch.no_grad():
                    if bool(mb.any()):
                        xb[mb] = _apply(generator, xb[mb], db[mb])[0]
                        yb[mb] = _target(target_fn, yb[mb], db[mb])
                surrogate_optim.zero_grad(set_to_none=True)
                loss = F.smooth_l1_loss(_prediction(surrogate, xb, yb), yb)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite BackTime surrogate loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in surrogate.parameters()):
                    raise ValueError("nonfinite BackTime surrogate gradient")
                surrogate_optim.step()
                surrogate_updates += 1
                train_sum += float(loss.detach()) * len(idx)
                train_n += len(idx)
            generator.train()
            poison_indices = torch.where(mask)[0].numpy()
            order = rng.permutation(poison_indices)
            attack_sum, norm_sum, attack_n = 0., 0., 0
            with _frozen(surrogate):
                for start in range(0, len(order), config.batch_size):
                    idx = order[start:start + config.batch_size]
                    xb, yb, db = x[idx].to(device), y[idx].to(device), d[idx].to(device)
                    generator_optim.zero_grad(set_to_none=True)
                    attacked, perturbation = _apply(generator, xb, db)
                    targets = _target(target_fn, yb, db)
                    pred = _prediction(surrogate, attacked, yb)
                    attack_loss = F.mse_loss(pred[..., list(config.target_joints), :], targets[..., list(config.target_joints), :])
                    norm_loss = perturbation.sum(-1).abs().mean()
                    objective = attack_loss + config.lam_norm * norm_loss
                    if not torch.isfinite(objective):
                        raise ValueError("nonfinite BackTime generator objective")
                    objective.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in generator.parameters()):
                        raise ValueError("nonfinite BackTime generator gradient")
                    generator_optim.step()
                    generator_updates += 1
                    attack_sum += float(attack_loss.detach()) * len(idx)
                    norm_sum += float(norm_loss.detach()) * len(idx)
                    attack_n += len(idx)
            scheduler.step()
            epoch = current + 1
            row = {"epoch": epoch, "surrogate_smooth_l1": train_sum / train_n,
                   "generator_target_mse": attack_sum / attack_n, "generator_dc_sum": norm_sum / attack_n,
                   "surrogate_updates": surrogate_updates, "generator_updates": generator_updates}
            history.append(row)
            checkpoint()
            if progress:
                progress(deepcopy(row))
        frozen = FrozenBackTimeTrigger(config, generator, {
            "fit_profile": "legacy_tensor_diagnostic_only_not_streaming_source_schedule",
            "optimization_bank": "TRAIN only; caller-supplied explicit recipient policy",
            "n_fit": len(x), "n_poisoned_fit": int(mask.sum()), "fit_poison_mask_sha256": _sha(mask),
            "fit_epochs": config.fit_epochs, "surrogate_updates": surrogate_updates,
            "generator_updates": generator_updates, "history": history,
            "clean_surrogate_unchanged": True, "clean_model_sha256": original_hash,
            "single_pose_horizon": 1, "soft_identification_weight": 1.,
        })
    if _sha(clean_model.state_dict()) != original_hash:
        raise ValueError("original clean surrogate changed during BackTime fitting")
    return frozen


def _stream_row(dataset, index, config):
    row = dataset[int(index)]
    if not isinstance(row, dict) or "csi" not in row or "pose" not in row:
        raise ValueError("TRAIN Dataset rows must contain csi and pose")
    x = torch.as_tensor(row["csi"], dtype=torch.float32).detach().cpu().clone()
    y = torch.as_tensor(row["pose"], dtype=torch.float32).detach().cpu().clone()
    _validate_input(x, config)
    if y.shape not in ((17, 3), (1, 17, 3)) or not torch.isfinite(y).all():
        raise ValueError("streaming MM-Fi requires finite 17-joint poses")
    return x, y


def _stream_batch(dataset, indices, config, device):
    rows = [_stream_row(dataset, i, config) for i in indices]
    return (torch.stack([r[0] for r in rows]).to(device),
            torch.stack([r[1] for r in rows]).to(device))


def author_frame_fft_features(frame, n_components=200):
    """Literal author fft_compress index expression on one CSI frame.

    ``argsort(amplitude, axis=1)[::-1]`` reverses VARIABLE order, not frequency
    order. This preserves the pinned implementation rather than silently
    replacing it with a new low-frequency feature extractor. A 10-packet
    frame has only ten FFT bins, so the source cap200 keeps all ten bins.
    """
    values = np.asarray(frame, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("expected finite variables x packets frame")
    count = _integer(n_components, "n_components")
    spectrum = np.fft.fft(values, axis=1)
    amplitude, phase = np.abs(spectrum), np.angle(spectrum)
    frequencies = np.fft.fftfreq(values.shape[1])
    order = np.argsort(amplitude, axis=1)[::-1][:, :count]
    channels = np.arange(values.shape[0])[:, None]
    return np.concatenate((amplitude[channels, order], phase[channels, order], frequencies[order]), axis=1)


def _stream_features(dataset, config, progress=None):
    total, square, count, features = 0., 0., 0, None
    for index in range(len(dataset)):
        x, _ = _stream_row(dataset, index, config)
        flat = x.numpy().reshape(config.n_variables, config.shape[-1]).astype(np.float64)
        value = author_frame_fft_features(flat, config.fft_components)
        features = value if features is None else features + value
        total += float(flat.sum())
        square += float(np.square(flat).sum())
        count += flat.size
        if progress and (index == 0 or (index + 1) % 10000 == 0 or index + 1 == len(dataset)):
            progress({"phase": "full_train_graph_and_normalization", "frames": index + 1, "total_frames": len(dataset)})
    mean = total / count
    std = math.sqrt(max(0., square / count - mean * mean))
    if not math.isfinite(std) or std <= 1e-12:
        raise ValueError("TRAIN-global standard deviation must be nonzero")
    return torch.from_numpy((features / len(dataset)).astype(np.float32)), mean, std


def fit_backtime_stream(model_init, train_dataset, target_fn, config, *, dataset_sha256,
                        n_poison, doses, progress=None, save_state=None,
                        resume_state=None, selection_callback=None):
    """Full-TRAIN author-schedule adapter with internal adaptive selection.

    ``model_init`` must be a freshly initialized HPE-Li template, not the final
    Clean reference. Its COPY is warmed up from scratch using native attacker
    Adam before selecting recipients. Source ``epoch > warmup`` is preserved:
    warmup=10 means epochs0..10 are clean, followed by89 alternating epochs.
    ``doses`` is a predeclared full-TRAIN vector; it extends the source to the
    comparison's paired continuous-pose payload without TEST calibration.
    ``selection_callback(indices)`` may return a full-TRAIN dose vector matching
    the caller's ordinary explicit-recipient poisoning manifest. That resolved
    vector is authenticated in checkpoints and must match exactly on resume.

    Dataset iteration streams individual frames; no 240k-frame tensor bank is
    allocated. Selection uses the last clean epoch's online pre-update Smooth
    L1 scores, as actual author trainer.py, not its MAE comment or fresh TEST
    scores. Independent CSI frames cannot reproduce source raw-time spacing.
    """
    if not isinstance(config, BackTimeConfig) or config.fit_epochs <= config.warmup + 1:
        raise ValueError("streaming BackTime requires epochs after native clean warmup")
    if getattr(train_dataset, "split", "training") != "training":
        raise ValueError("BackTime accepts the official TRAIN Dataset only")
    if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64 or any(c not in "0123456789abcdef" for c in dataset_sha256.lower()):
        raise ValueError("audited TRAIN dataset SHA256 is required")
    n = len(train_dataset)
    n_poison = _integer(n_poison, "n_poison")
    if n < config.batch_size or not n_poison < n:
        raise ValueError("full TRAIN needs at least one batch and both poison/clean frames")
    doses = torch.as_tensor(doses, dtype=torch.float32).detach().cpu().clone()
    if doses.shape != (n,) or not torch.isfinite(doses).all() or torch.any(doses <= 0) or torch.any(doses > 1):
        raise ValueError("predeclared full-TRAIN positive paired doses required")
    devices = {p.device for p in model_init.parameters()}
    if len(devices) != 1:
        raise ValueError("fresh surrogate template must use one device")
    device = next(iter(devices))
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    initial_hash = _sha(model_init.state_dict())
    binding = {"implementation": IMPLEMENTATION, "config": asdict(config), "dataset_sha256": dataset_sha256,
               "n_train": n, "n_poison": n_poison, "doses_sha256": _sha(doses), "initial_model_sha256": initial_hash}
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(config.seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)
        epoch, history, recipients, selection_scores = 0, [], None, None
        online_scores = torch.full((n,), float("nan"))
        surrogate_updates, generator_updates = 0, 0
        state = None
        if resume_state is not None:
            state = _cpu(resume_state)
            claimed = state.pop("sha256", None)
            if claimed != _sha(state) or state.get("version") != IMPLEMENTATION or _sha(state.get("binding")) != _sha(binding):
                raise ValueError("BackTime streaming resume source/model/TRAIN/config checksum mismatch")
            epoch = state["epoch"]
            if not isinstance(epoch, int) or not 0 <= epoch <= config.fit_epochs:
                raise ValueError("invalid BackTime streaming resume epoch")
            graph = state["generator"]
            generator = _GraphGenerator(config, graph["features"], graph["variables"],
                                        float(graph["train_mean"]), float(graph["train_std"])).to(device)
        else:
            # Source picks attacked variables before constructing/training GNN.
            variables = np.sort(rng.choice(config.n_variables, config.n_attacked_variables, replace=False))
            full_features, mean, std = _stream_features(train_dataset, config, progress)
            generator = _GraphGenerator(config, full_features[variables], variables, mean, std).to(device)
        surrogate = deepcopy(model_init).to(device)
        surrogate.train()
        for parameter in surrogate.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        surrogate_optim = torch.optim.Adam(surrogate.parameters(), lr=config.surrogate_lr)
        generator_optim = torch.optim.Adam(generator.parameters(), lr=config.attack_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optim, milestones=[20, 40], gamma=.9)
        if state is not None:
            generator.load_state_dict(state["generator"], strict=True)
            surrogate.load_state_dict(state["surrogate"], strict=True)
            surrogate_optim.load_state_dict(state["surrogate_optimizer"])
            generator_optim.load_state_dict(state["generator_optimizer"])
            _optimizer_to(surrogate_optim, device)
            _optimizer_to(generator_optim, device)
            scheduler.load_state_dict(state["scheduler"])
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"], device)
            online_scores, selection_scores = state["online_scores"], state["selection_scores"]
            doses = state["resolved_doses"]
            if doses.shape != (n,) or not torch.isfinite(doses).all() or torch.any(doses <= 0) or torch.any(doses > 1):
                raise ValueError("invalid restored BackTime paired doses")
            recipients, history = state["recipients"], state["history"]
            surrogate_updates, generator_updates = state["surrogate_updates"], state["generator_updates"]
            if recipients is not None:
                recipients = np.asarray(recipients, dtype=np.int64)
                if len(recipients) != n_poison or len(np.unique(recipients)) != n_poison or np.any(recipients < 0) or np.any(recipients >= n):
                    raise ValueError("invalid restored BackTime poison recipients")
                if selection_callback:
                    mapped = selection_callback(recipients.copy())
                    if mapped is not None and not torch.equal(torch.as_tensor(mapped, dtype=torch.float32).cpu(), doses):
                        raise ValueError("restored BackTime dose plan differs from recipient manifest")

        def checkpoint():
            if save_state:
                payload = {"version": IMPLEMENTATION, "binding": binding, "epoch": epoch,
                           "generator": _cpu(generator.state_dict()), "surrogate": _cpu(surrogate.state_dict()),
                           "surrogate_optimizer": _cpu(surrogate_optim.state_dict()),
                           "generator_optimizer": _cpu(generator_optim.state_dict()), "scheduler": _cpu(scheduler.state_dict()),
                           "online_scores": online_scores.clone(), "recipients": None if recipients is None else recipients.tolist(),
                           "selection_scores": None if selection_scores is None else selection_scores.clone(),
                           "resolved_doses": doses.clone(),
                           "history": deepcopy(history), "numpy_rng": deepcopy(rng.bit_generator.state),
                           "torch_rng": torch.get_rng_state().clone(),
                           "cuda_rng": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
                           "surrogate_updates": surrogate_updates, "generator_updates": generator_updates}
                save_state({**payload, "sha256": _sha(payload)})

        checkpoint()
        for current in range(epoch, config.fit_epochs):
            active = current > config.warmup
            if active and recipients is None:
                if online_scores.shape != (n,) or not torch.isfinite(online_scores).all():
                    raise ValueError("complete clean warmup online TRAIN scores required before adaptive selection")
                # Source accumulates a pool until len(valid)>2K. On independent
                # frames no spacing applies, hence up to 2K+1 ranked candidates.
                order = torch.argsort(online_scores, descending=True, stable=True).numpy()[:min(n, 2*n_poison + 1)]
                recipients = np.sort(rng.choice(order, n_poison, replace=False)).astype(np.int64)
                selection_scores = online_scores.clone()
                if selection_callback:
                    mapped = selection_callback(recipients.copy())
                    if mapped is not None:
                        mapped = torch.as_tensor(mapped, dtype=torch.float32).detach().cpu().clone()
                        if mapped.shape != (n,) or not torch.isfinite(mapped).all() or torch.any(mapped <= 0) or torch.any(mapped > 1):
                            raise ValueError("recipient callback must return a positive full-TRAIN paired dose vector")
                        doses = mapped
                if progress:
                    progress({"phase": "adaptive_train_selection", "epoch_index": current,
                              "n_poison": n_poison, "score": "online_pre_update_clean_SmoothL1", "test_used": False})
                checkpoint()
            mask = torch.zeros(n, dtype=torch.bool)
            if recipients is not None:
                mask[recipients] = True
            # Source surrogate training follows fresh shuffled TRAIN order;
            # poisoning uses the frozen generator for that epoch (no G update).
            generator.train()
            surrogate.train()
            order = rng.permutation(n)
            next_online = torch.empty(n)
            train_sum, train_n = 0., 0
            for start in range(0, n, config.batch_size):
                indices = order[start:start + config.batch_size]
                xb, yb = _stream_batch(train_dataset, indices, config, device)
                db, mb = doses[indices].to(device), mask[indices].to(device)
                with torch.no_grad():
                    if active and bool(mb.any()):
                        xb[mb] = _apply(generator, xb[mb], db[mb])[0]
                        yb[mb] = _target(target_fn, yb[mb], db[mb])
                surrogate_optim.zero_grad(set_to_none=True)
                pred = _prediction(surrogate, xb, yb)
                losses = F.smooth_l1_loss(pred, yb, reduction="none").flatten(1).mean(1)
                next_online[indices] = losses.detach().cpu()
                loss = losses.mean()
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in surrogate.parameters()):
                    raise ValueError("nonfinite full-TRAIN BackTime surrogate loss/gradient")
                surrogate_optim.step()
                surrogate_updates += 1
                train_sum += float(loss.detach()) * len(indices)
                train_n += len(indices)
            online_scores = next_online
            attack_sum, norm_sum, attack_n = 0., 0., 0
            if active:
                # Source iterates sorted selected timestamps one by one. Pose
                # horizon1 has no overlapping future-pattern windows to batch.
                # Keep train mode (dropout/BN) while freezing surrogate weights.
                with _frozen(surrogate, eval_mode=False):
                    for index in recipients:
                        xb, yb = _stream_batch(train_dataset, [index], config, device)
                        db = doses[[index]].to(device)
                        generator_optim.zero_grad(set_to_none=True)
                        attacked, perturbation = _apply(generator, xb, db)
                        pred = _prediction(surrogate, attacked, yb)
                        expected = _target(target_fn, yb, db)
                        attack_loss = F.mse_loss(pred[..., list(config.target_joints), :], expected[..., list(config.target_joints), :])
                        norm_loss = perturbation.sum(-1).abs().mean()
                        loss = attack_loss + config.lam_norm * norm_loss
                        loss.backward()
                        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in generator.parameters()):
                            raise ValueError("nonfinite full-TRAIN BackTime generator loss/gradient")
                        generator_optim.step()
                        generator_updates += 1
                        attack_sum += float(attack_loss.detach())
                        norm_sum += float(norm_loss.detach())
                        attack_n += 1
                scheduler.step()
            epoch = current + 1
            record = {"epoch": epoch, "epoch_index": current, "phase": "alternating" if active else "clean_warmup",
                      "surrogate_smooth_l1": train_sum / train_n,
                      "generator_target_mse": attack_sum / attack_n if attack_n else None,
                      "generator_dc_sum": norm_sum / attack_n if attack_n else None,
                      "surrogate_updates": surrogate_updates, "generator_updates": generator_updates}
            history.append(record)
            checkpoint()
            if progress:
                progress(deepcopy(record))
        frozen = FrozenBackTimeTrigger(config, generator, {
            "fit_profile": "full_TRAIN_streaming_author_schedule_CSI_pose_adaptation",
            "n_fit": n, "n_poisoned_fit": n_poison, "poison_indices": recipients.tolist(),
            "selection": "online clean warmup SmoothL1; source-style top2K+1 pool sampled once; independent frames no fake timestamp spacing",
            "clean_warmup_epochs": config.warmup + 1,
            "fit_epochs": config.fit_epochs, "surrogate_updates": surrogate_updates,
            "generator_updates": generator_updates, "history": history,
            "fresh_surrogate_template_unchanged": True, "initial_model_sha256": initial_hash,
            "dataset_sha256": dataset_sha256, "test_used": False,
            "single_pose_horizon": 1, "soft_identification_weight": 1.,
        }, selection_scores=selection_scores)
    if _sha(model_init.state_dict()) != initial_hash:
        raise ValueError("fresh surrogate template changed during streaming BackTime fit")
    return frozen
