"""Independent TrojanNN/BackdoorBench adaptation for normalized MM-Fi CSI.

This is *not* an exact image attack or an RF-native attack.  A clean HPE model
is needed for white-box trigger synthesis; the poisoned victim must afterwards
be initialized afresh by the caller.  No CSI/labels from TEST are used here.

Algorithm provenance: BackdoorBench ``attack/trojannn.py`` and its
``config/attack/trojannn/preactresnet18.yaml`` at the pinned commit below.
The selected units are INPUT units of the chosen affine module, ranked by
absolute outgoing weights.  PGD optimizes a universal masked input, not a
per-example adversarial perturbation.  Image/PIL/uint8 operations are deliberately
replaced by float32 CSI addition and clipping; there is no integer wraparound.

The bottom-right 3-by-3 frequency/packet mask on all antennas matches the
selected MM-Fi BadNets experiment geometry.  Its location is not TEST-tuned.
PGD eps=.3 is a radius around the initial WHITE patch, not a cap on the actual
injected residual (the learned patch is in [.7,1]).  Dose scales this fixed
additive patch.  This substantial distinction must accompany reported results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import math
from typing import Callable

import numpy as np
import torch
from torch import nn


SOURCE_COMMIT = "f02e3534645f0ee63d6848653062cd6c0d6c400d"
SOURCE_SHA256 = "b5cbcf7771dc4c31a3cc8610510ad6a58d2b493e75f68e1908b945a457617807"
SOURCE_CONFIG_SHA256 = "e33f69471d27f95658701b6c26f2597ff637faed732d2be375caead3285ab307"
IMPLEMENTATION = "independent-backdoorbench-trojannn-mmfi-float-v1"


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class TrojanNNConfig:
    layer_name: str = "regression.7"
    parameter_name: str = "regression.7.weight"
    num_neurons: int = 2
    target_value: float = 100.0
    steps: int = 1000
    alpha: float = 0.1
    eps: float = 0.3
    tolerance: float = 1e-3
    shape: tuple[int, int, int] = (3, 114, 10)
    patch_shape: tuple[int, int] = (3, 3)

    def __post_init__(self):
        if not isinstance(self.layer_name, str) or not self.layer_name:
            raise ValueError("layer_name is required")
        if self.parameter_name != self.layer_name + ".weight":
            raise ValueError("parameter_name must be the selected module's weight")
        object.__setattr__(self, "num_neurons", _integer(self.num_neurons, "num_neurons"))
        object.__setattr__(self, "steps", _integer(self.steps, "steps"))
        if len(self.shape) != 3 or len(self.patch_shape) != 2:
            raise ValueError("shape/patch_shape must contain three/two dimensions")
        shape = tuple(_integer(v, "shape dimension") for v in self.shape)
        patch = tuple(_integer(v, "patch dimension") for v in self.patch_shape)
        if any(p > s for p, s in zip(patch, shape[1:])):
            raise ValueError("patch does not fit CSI frame")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "patch_shape", patch)
        for key in ("target_value", "alpha", "eps", "tolerance"):
            if not math.isfinite(float(getattr(self, key))):
                raise ValueError(f"{key} must be finite")
        if self.alpha <= 0 or not 0 <= self.eps <= 1 or self.tolerance < 0:
            raise ValueError("alpha must be positive; eps in [0,1]; tolerance nonnegative")

    def mask(self) -> torch.Tensor:
        value = torch.zeros(self.shape, dtype=torch.float32)
        frequency, packets = self.patch_shape
        value[:, -frequency:, -packets:] = 1.0
        return value


def most_connected_neurons(weight: torch.Tensor, count: int = 2) -> torch.Tensor:
    """Rank INPUT units by absolute weight mass to all output units.

    The stable tie-break is ascending input index.  Conv spatial axes are summed
    before output channels; grouped convolutions are rejected by the optimizer.
    """
    count = _integer(count, "count")
    value = weight.detach().abs()
    if value.ndim not in (2, 4) or not torch.isfinite(value).all():
        raise ValueError("selection requires finite Linear/Conv2d weights")
    if count > value.shape[1]:
        raise ValueError("more selected neurons than input units")
    if value.ndim == 4:
        value = value.sum(dim=(2, 3))
    return torch.argsort(value.sum(dim=0), descending=True, stable=True)[:count]


def _array_sha(value: torch.Tensor) -> str:
    array = value.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


class TrojanNNTrigger:
    baseline_name = "TrojanNN-CSI adapted"
    requires_deferred_injection = False
    requires_event_metadata = False
    dose_semantics = "clip(x + dose * fixed_float_patch, 0, 1); PGD eps is NOT residual cap"

    def __init__(self, config: TrojanNNConfig, pattern: torch.Tensor, diagnostics: dict):
        self.config = config
        self.shape = config.shape
        value = torch.as_tensor(pattern).detach().cpu().to(torch.float32).clone()
        mask = config.mask().bool()
        if tuple(value.shape) != self.shape or not torch.isfinite(value).all():
            raise ValueError("invalid optimized trigger shape/values")
        if torch.any(value[~mask] != 0) or torch.any(value[mask] < 1.0 - config.eps - 1e-6) or torch.any(value[mask] > 1):
            raise ValueError("trigger violates fixed-mask/PGD projection contract")
        self._pattern = value
        self._diagnostics = deepcopy(diagnostics)

    @property
    def pattern(self) -> np.ndarray:
        return self._pattern.numpy().copy()

    def inject(self, csi, dose, eps=None) -> np.ndarray:
        """Dataset interface; ``eps`` is ignored, not an injection-strength knob."""
        value = np.asarray(csi)
        if value.shape != self.shape or np.iscomplexobj(value):
            raise ValueError(f"expected real normalized CSI {self.shape}")
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError("CSI must be finite and in [0,1]")
        dose = float(dose)
        if not math.isfinite(dose) or not 0 <= dose <= 1:
            raise ValueError("dose must be finite and in the registered [0,1] range")
        if eps is not None and (not math.isfinite(float(eps)) or float(eps) < 0):
            raise ValueError("compatibility eps must be finite and nonnegative")
        out = value.astype(np.float32, copy=True)
        if dose == 0:
            return out
        return np.clip(out + dose * self._pattern.numpy(), 0, 1).astype(np.float32)

    def apply_tensor(self, csi: torch.Tensor, dose: float) -> torch.Tensor:
        if not csi.is_floating_point() or tuple(csi.shape[-3:]) != self.shape or csi.ndim not in (3, 4):
            raise ValueError("expected floating CHW/BCHW CSI with matching shape")
        if not torch.isfinite(csi).all() or torch.any(csi < 0) or torch.any(csi > 1):
            raise ValueError("CSI must be finite and in [0,1]")
        dose = float(dose)
        if not math.isfinite(dose) or not 0 <= dose <= 1:
            raise ValueError("dose must be finite and in the registered [0,1] range")
        if dose == 0:
            return csi.clone()
        patch = self._pattern.to(device=csi.device, dtype=csi.dtype)
        return (csi + dose * patch).clamp(0, 1)

    def provenance(self) -> dict:
        return {
            "implementation": IMPLEMENTATION,
            "source_commit": SOURCE_COMMIT,
            "source_url": f"https://github.com/SCLBD/BackdoorBench/blob/{SOURCE_COMMIT}/attack/trojannn.py",
            "source_sha256_lf": SOURCE_SHA256,
            "source_config_sha256_lf": SOURCE_CONFIG_SHA256,
            "config": asdict(self.config),
            "mask_sha256": _array_sha(self.config.mask()),
            "pattern_sha256": _array_sha(self._pattern),
            "dose_semantics": self.dose_semantics,
            "model_access": "white-box matching clean reference for trigger synthesis; fresh victim required",
            "image_to_csi_adaptation": "fixed 3x3 frequency/packet float patch; no PIL, resize or uint8 wraparound",
            "diagnostics": deepcopy(self._diagnostics),
        }

    def state_dict(self) -> dict:
        return {"version": IMPLEMENTATION, "config": asdict(self.config),
                "pattern": self._pattern.clone(), "diagnostics": deepcopy(self._diagnostics),
                "pattern_sha256": _array_sha(self._pattern)}

    @classmethod
    def from_state_dict(cls, state: dict) -> "TrojanNNTrigger":
        if state.get("version") != IMPLEMENTATION:
            raise ValueError("incompatible TrojanNN trigger version")
        trigger = cls(TrojanNNConfig(**state["config"]), state["pattern"], state["diagnostics"])
        if state.get("pattern_sha256") != _array_sha(trigger._pattern):
            raise ValueError("TrojanNN pattern digest mismatch")
        return trigger


def optimize_trigger(model: nn.Module, config: TrojanNNConfig | None = None,
                     device=None, progress: Callable[[dict], None] | None = None) -> TrojanNNTrigger:
    """Synthesize a fixed universal patch without changing the clean model.

    Place ``model`` on the desired device before calling.  Its parameters,
    buffers, parameter gradients, training flags and requires-grad flags are
    preserved.  Registered hooks are removed even on exceptions.  Nothing is
    selected using examples, labels, validation or TEST performance.
    """
    config = config or TrojanNNConfig()
    modules = dict(model.named_modules())
    if config.layer_name not in modules:
        raise ValueError("selected layer not found in clean model")
    layer = modules[config.layer_name]
    if not isinstance(layer, (nn.Linear, nn.Conv2d)) or (isinstance(layer, nn.Conv2d) and layer.groups != 1):
        raise ValueError("selected layer must be Linear or ungrouped Conv2d")
    parameters = list(model.parameters())
    if not parameters or any(p.device != parameters[0].device or p.dtype != torch.float32 for p in parameters):
        raise ValueError("clean model must have float32 parameters on a single device")
    actual_device = parameters[0].device
    if device is not None and torch.device(device) != actual_device:
        raise ValueError("move clean model to the requested device before optimization")
    selected = most_connected_neurons(layer.weight, config.num_neurons).to(actual_device)
    original_modes = [(module, module.training) for module in model.modules()]
    original_requires = [(parameter, parameter.requires_grad) for parameter in parameters]
    captured = []

    def capture(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise ValueError("selected layer has no tensor input")
        captured.append(inputs[0])

    def loss_for(pattern):
        captured.clear()
        model(pattern)
        if len(captured) != 1:
            raise ValueError("selected layer must be called exactly once per forward")
        activation = captured[0]
        if activation.ndim < 2 or activation.shape[1] != layer.weight.shape[1]:
            raise ValueError("selected layer input does not match weight input units")
        loss = (activation.index_select(1, selected) - config.target_value).square().sum()
        if not torch.isfinite(loss):
            raise ValueError("non-finite TrojanNN neuron loss")
        return loss

    hook = None
    try:
        model.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        hook = layer.register_forward_pre_hook(capture)
        mask = config.mask().to(actual_device).unsqueeze(0)
        initial = mask.clone()
        pattern = initial.clone()
        trace = []
        first_loss = None
        completed = 0
        with torch.enable_grad():
            for step in range(1, config.steps + 1):
                pattern.requires_grad_(True)
                loss = loss_for(pattern * mask)
                scalar = float(loss.detach())
                if first_loss is None:
                    first_loss = scalar
                gradient, = torch.autograd.grad(loss, pattern)
                if not torch.isfinite(gradient).all():
                    raise ValueError("non-finite TrojanNN input gradient")
                with torch.no_grad():
                    proposed = pattern - config.alpha * gradient.sign()
                    delta = (proposed - initial).clamp(-config.eps, config.eps)
                    pattern = ((initial + delta).clamp(0, 1) * mask).detach()
                completed = step
                if step == 1 or step % 100 == 0 or scalar < config.tolerance or step == config.steps:
                    item = {"step": step, "loss_before_update": scalar}
                    trace.append(item)
                    if progress:
                        progress(dict(item))
                if scalar < config.tolerance:
                    break
        with torch.no_grad():
            final_loss = float(loss_for(pattern))
        diagnostics = {"selected_neurons": selected.cpu().tolist(), "steps_run": completed,
                       "initial_loss": first_loss, "final_loss": final_loss, "trace": trace}
        return TrojanNNTrigger(config, pattern[0], diagnostics)
    finally:
        if hook is not None:
            hook.remove()
        for parameter, requires_grad in original_requires:
            parameter.requires_grad_(requires_grad)
        for module, training in original_modes:
            module.training = training
