"""Independent Adap-Patch adaptation for normalized MM-Fi amplitude / pose.

Algorithm reference: Unispac/Circumventing-Backdoor-Defenses, ICLR 2023,
commit e7c070344a5c6004d6fbcdf0720e2d60ec808953, config.py and
poison_tool_box/adaptive_patch.py. No upstream source or PNG is redistributed.
The complete native GTSRB trigger preset is used: four individual training
triggers and two sequential, stronger testing triggers. The CIFAR preset is
not used because its configured phoenix_corner2_32.png is absent at that pin.
The dataset remains MM-Fi, NOT GTSRB.

Declared adaptations: RGB channels map to three antennas; marks are bilinear
and masks nearest resized from 32x32 to 114x10. Injection is in normalized
float CSI, with no PNG quantization, second normalization, or comparison cap.
The common paired dose scales native alpha; the caller applies pose targets
only to poisoned identities and retains true poses for covers. This is an
exploratory digital CSI/pose adaptation, not an image reproduction or an
equal-modified-data / equal-distortion comparison.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import math
import os
from pathlib import Path
from urllib.request import Request, urlopen
import uuid

import numpy as np
import torch
from torch.nn import functional as F


IMPLEMENTATION = "adap-patch-mmfi-pose-gtsrb-source-preset-v1"
SOURCE_COMMIT = "e7c070344a5c6004d6fbcdf0720e2d60ec808953"
SOURCE_REPOSITORY = "https://github.com/Unispac/Circumventing-Backdoor-Defenses"
RAW_ROOT = "https://raw.githubusercontent.com/Unispac/Circumventing-Backdoor-Defenses/" + SOURCE_COMMIT
TRAIN_NAMES = ("phoenix_corner_32.png", "firefox_corner_32.png",
               "badnet_patch4_32.png", "trojan_square_32.png")
TRAIN_ALPHAS = (.5, .2, .5, .3)
TEST_NAMES = ("firefox_corner_32.png", "trojan_square_32.png")
ASSET_SHA256 = {
    "phoenix_corner_32.png": "e94807ec4b0d64135f40524ee2533c672642f7dd0ee3fe75fcce6edd4434a36b",
    "firefox_corner_32.png": "74f526d75a0fe15235e9b72d5733b77dd8cc330542ded39c0bad1b43fb01246d",
    "badnet_patch4_32.png": "c313b50a4416f5ab6b100acabf2ca047eb19f4f1303ecc75130a52be8d36ddef",
    "mask_badnet_patch4_32.png": "ac8058be64fcac9f50f7bf350eb69c22a10c9c8b47bd5e888aff7c891bc9e2db",
    "trojan_square_32.png": "3c3fa22c50566aac14c484b0c809ef0ed2ee8e43465671bf1221dd5660a47ca3",
}


def source_config():
    """JSON-ready provenance and fixed, pre-TEST source/adaptation choices."""
    return {
        "implementation": IMPLEMENTATION,
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "source_files": ["config.py", "poison_tool_box/adaptive_patch.py"],
        "source_preset": "gtsrb",
        "dataset_is_not_gtsrb": True,
        "cifar_preset_excluded": "configured phoenix_corner2_32.png absent from pinned tree",
        "train_names": list(TRAIN_NAMES), "train_alphas": list(TRAIN_ALPHAS),
        "test_names": list(TEST_NAMES), "test_alphas": [1., 1.],
        "source_shape": [3, 32, 32], "shape": [3, 114, 10],
        "mark_resize": "bilinear_align_corners_false", "mask_resize": "nearest",
        "explicit_mask": "mask_badnet_patch4_32.png",
        "fallback_mask": "any_original_RGB_channel_greater_than_zero",
        "train_assignment": "sorted_identity_rank: floor(rank * 4 / count)",
        "test_composition": "sequential firefox, then trojan",
        "injection_domain": "normalized float amplitude CSI; no renormalization or image quantization",
        "dose_policy": "native_alpha_times_common_paired_dose",
        "cover_rate": .2, "cover_labels": "unchanged true poses",
        "comparison_cap": None, "asset_sha256": dict(ASSET_SHA256),
        "license_note": "no upstream license found; independent code, runtime asset retrieval only",
    }


def _decode_png(payload, name):
    from PIL import Image
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format != "PNG" or image.size != (32, 32):
                raise ValueError("expected PNG with dimensions 32x32")
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except Exception as exc:
        raise ValueError(f"Invalid pinned PNG {name}: {exc}") from exc
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.)


def _verified_payload(payload, name):
    if hashlib.sha256(payload).hexdigest() != ASSET_SHA256[name]:
        raise ValueError(f"Pinned asset SHA256 mismatch: {name}; cache was not overwritten")
    return _decode_png(payload, name)


def _asset_path(directory, name):
    target = Path(directory) / name
    if target.is_symlink():
        raise ValueError(f"Pinned asset must not be a symbolic link: {target}")
    return target


def prepare_assets(directory):
    """Download missing pinned assets only; never overwrite an existing file.

    All responses are SHA/geometry checked before entering the cache. Hardlink
    publication is atomic and refuses replacement, including concurrent runs.
    Invalid existing assets fail closed; the caller must inspect them.
    """
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("Asset cache directory must not be a symbolic link")
    root.mkdir(parents=True, exist_ok=True)
    for name in ASSET_SHA256:
        target = _asset_path(root, name)
        if target.exists():
            _verified_payload(target.read_bytes(), name)
            continue
        request = Request(f"{RAW_ROOT}/triggers/{name}", headers={"User-Agent": "MMFi-AdapPatch-SourceAudit/1"})
        with urlopen(request, timeout=60) as response:
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError(f"Unexpected oversized pinned asset: {name}")
        _verified_payload(payload, name)
        temporary = root / f".{name}.{uuid.uuid4().hex}.part"
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                _verified_payload(target.read_bytes(), name)
        finally:
            temporary.unlink(missing_ok=True)
    return load_assets(root)


def load_assets(directory):
    """Read-only verification of every asset, including the explicit patch mask."""
    decoded = {}
    for name in ASSET_SHA256:
        target = _asset_path(directory, name)
        decoded[name] = _verified_payload(target.read_bytes(), name)
    training = []
    for name, alpha in zip(TRAIN_NAMES, TRAIN_ALPHAS):
        mark = decoded[name]
        explicit_name = "mask_" + name
        mask = (decoded[explicit_name][0] if explicit_name in decoded
                else (mark > 0).any(dim=0).float())
        training.append({"name": name, "mark": mark.clone(), "mask": mask.clone(), "alpha": alpha})
    by_name = {entry["name"]: entry for entry in training}
    testing = [{**deepcopy(by_name[name]), "alpha": 1.} for name in TEST_NAMES]
    return {"train": training, "test": testing, "asset_sha256": dict(ASSET_SHA256)}


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def rank_groups(count, k=4):
    """Exact integer equivalent of source's rank < (j+1)*(count/k)."""
    count = _positive_integer(count, "count", 0)
    k = _positive_integer(k, "k")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(count, dtype=np.int64) * k // count


def cover_group_plan(n, poison_ids, poison_doses, *, seed=42, cover_rate=.2):
    """Preserve common poisons/doses; sample disjoint clean-label covers.

    The cover RNG is independent of the common poison RNG state. Its paired
    dose is fixed U(.2,1); cover poses stay true in the caller's dataset wrapper.
    Cover identities and doses are sorted jointly before native rank grouping.
    """
    n = _positive_integer(n, "n")
    seed = _positive_integer(seed, "seed", 0)
    ids = np.asarray(poison_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or ids.dtype.kind == "b":
        raise ValueError("poison_ids must be a one-dimensional integer array")
    ids = ids.astype(np.int64, copy=True)
    doses = np.asarray(poison_doses, dtype=np.float64)
    if doses.shape != ids.shape or not np.isfinite(doses).all() or ((doses < .2) | (doses > 1.)).any():
        raise ValueError("poison_doses must match poison_ids and be finite in [.2,1]")
    if len(np.unique(ids)) != len(ids) or ((ids < 0) | (ids >= n)).any():
        raise ValueError("poison_ids must be unique and inside the training dataset")
    if isinstance(cover_rate, (bool, np.bool_)) or not np.isfinite(cover_rate) or not 0 <= cover_rate <= 1:
        raise ValueError("cover_rate must be finite in [0,1]")
    n_cover = math.floor(float(cover_rate) * n)
    if n_cover + len(ids) > n:
        raise ValueError("Insufficient unpoisoned training identities for requested cover rate")
    order = np.argsort(ids, kind="stable")
    ids, doses = ids[order], doses[order].copy()
    available = np.setdiff1d(np.arange(n, dtype=np.int64), ids, assume_unique=True)
    rng = np.random.default_rng(seed)
    cover_ids = rng.choice(available, size=n_cover, replace=False)
    cover_doses = rng.uniform(.2, 1., size=n_cover)
    cover_order = np.argsort(cover_ids, kind="stable")
    return {
        "poison_ids": ids, "poison_doses": doses, "poison_groups": rank_groups(len(ids)),
        "cover_ids": cover_ids[cover_order], "cover_doses": cover_doses[cover_order],
        "cover_groups": rank_groups(n_cover),
    }


def _tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


class AdapPatchCSI:
    """Fixed native train/test trigger composition, without learned components."""

    def __init__(self, assets, shape=(3, 114, 10)):
        if tuple(shape) != (3, 114, 10):
            raise ValueError("This MM-Fi adaptation is locked to shape (3,114,10)")
        self.shape = tuple(shape)
        self._train = self._resize_entries(assets["train"], TRAIN_NAMES, TRAIN_ALPHAS)
        self._test = self._resize_entries(assets["test"], TEST_NAMES, (1., 1.))
        self.asset_sha256 = deepcopy(assets.get("asset_sha256", {}))

    def _resize_entries(self, entries, names, alphas):
        if len(entries) != len(names):
            raise ValueError("Adap-Patch source preset trigger count mismatch")
        output = []
        for entry, name, alpha in zip(entries, names, alphas):
            if entry.get("name") != name or entry.get("alpha") != alpha:
                raise ValueError("Adap-Patch source preset trigger name/alpha mismatch")
            mark = torch.as_tensor(entry["mark"]).detach().cpu().float().clone()
            mask = torch.as_tensor(entry["mask"]).detach().cpu().float().clone()
            if tuple(mark.shape) != (3, 32, 32) or tuple(mask.shape) != (32, 32):
                raise ValueError("Expected source mark (3,32,32) and mask (32,32)")
            if not torch.isfinite(mark).all() or not torch.isfinite(mask).all() or (mark < 0).any() or (mark > 1).any() or (mask < 0).any() or (mask > 1).any():
                raise ValueError("Source marks/masks must be finite in [0,1]")
            mark = F.interpolate(mark[None], size=self.shape[1:], mode="bilinear", align_corners=False)[0]
            mask = F.interpolate(mask[None, None], size=self.shape[1:], mode="nearest")[0]
            if not (mask > 0).any():
                raise ValueError(f"Trigger mask vanished during declared nearest resize: {name}")
            output.append({"name": name, "alpha": alpha, "mark": mark, "mask": mask})
        return output

    def _validate(self, x, dose):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError("CSI input must be a floating-point torch tensor")
        if x.ndim not in (3, 4) or tuple(x.shape[-3:]) != self.shape:
            raise ValueError("CSI input must have shape CHW or BCHW with (3,114,10)")
        batch = x.unsqueeze(0) if x.ndim == 3 else x
        if not torch.isfinite(batch).all() or (batch < 0).any() or (batch > 1).any():
            raise ValueError("Normalized CSI must be finite in [0,1]")
        d = torch.as_tensor(dose, device=x.device, dtype=x.dtype)
        if d.ndim == 0:
            d = d.expand(len(batch))
        if tuple(d.shape) != (len(batch),) or not torch.isfinite(d).all() or (d < 0).any() or (d > 1).any():
            raise ValueError("dose must be scalar or length-B, finite in [0,1]")
        return batch, d.reshape(-1, 1, 1, 1)

    @staticmethod
    def _blend(x, dose, entry):
        mark, mask = entry["mark"].to(x), entry["mask"].to(x)
        mixed = x + entry["alpha"] * dose * mask * (mark - x)
        return torch.where(dose == 0, x, mixed)

    def inject_train(self, x, dose, group):
        batch, d = self._validate(x, dose)
        groups = torch.as_tensor(group, device=x.device)
        if groups.dtype == torch.bool or groups.is_floating_point() or groups.is_complex():
            raise ValueError("group must contain integers in [0,3]")
        if groups.ndim == 0:
            groups = groups.expand(len(batch))
        if tuple(groups.shape) != (len(batch),) or (groups < 0).any() or (groups > 3).any():
            raise ValueError("group must be scalar or length-B, in [0,3]")
        result = batch.clone()
        for index, entry in enumerate(self._train):
            selected = groups == index
            if selected.any():
                result[selected] = self._blend(batch[selected], d[selected], entry)
        return result[0] if x.ndim == 3 else result

    def inject_test(self, x, dose):
        batch, d = self._validate(x, dose)
        result = batch.clone()
        for entry in self._test:
            result = self._blend(result, d, entry)
        return result[0] if x.ndim == 3 else result

    def state_summary(self):
        def summarize(entry):
            return {"name": entry["name"], "alpha": entry["alpha"],
                    "mark_sha256_float32": _tensor_digest(entry["mark"]),
                    "mask_sha256_float32": _tensor_digest(entry["mask"]),
                    "mask_nonzero": int((entry["mask"] > 0).sum().item())}
        return {"implementation": IMPLEMENTATION, "source_preset": "gtsrb",
                "shape": list(self.shape), "asset_sha256": deepcopy(self.asset_sha256),
                "train": [summarize(entry) for entry in self._train],
                "test": [summarize(entry) for entry in self._test]}
