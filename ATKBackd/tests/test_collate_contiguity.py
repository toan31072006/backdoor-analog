"""MM-Fi batches must be viewable regardless of NumPy or trigger layout."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import attack.payload as payload  # noqa: E402
from attack.poison import PoisonedDataset, collate  # noqa: E402
from attack.traditional import BadNetsTrigger, BlendedTrigger  # noqa: E402
from data_utils.feeder import MMFI  # noqa: E402
from models.hpeli import HPELiNet  # noqa: E402
from third_party.backdoorbench.blended import blendedImageAttack  # noqa: E402
from third_party.backdoorbench.patch import AddMaskPatchTrigger  # noqa: E402


def _with_layout(array, layout):
    if layout == "c":
        return np.ascontiguousarray(array)
    if layout == "fortran":
        return np.asfortranarray(array)
    if layout == "channels_last":
        # A CHW view over the HWC storage used by image-trigger operators.
        return np.moveaxis(np.ascontiguousarray(np.moveaxis(array, 0, -1)), -1, 0)
    raise ValueError(layout)


class _LayoutMMFIBase(Dataset):
    """Importable spawn fixture with actual MMFI normalization, no disk data."""

    normalize = staticmethod(MMFI.normalize)

    def __init__(self, layout="fortran"):
        self.items = [{"csi": index, "kpt": index, "frame_idx": 0}
                      for index in range(4)]
        antenna, frequency, packet = np.indices((3, 114, 10), dtype=np.float32)
        self.raws = []
        for index in range(len(self.items)):
            raw = (np.float32(0.15 + index * 0.02) + antenna * np.float32(0.14)
                   + frequency * np.float32(0.001) + packet * np.float32(0.0071))
            raw = _with_layout(raw, layout)
            # Evaluation caches return read-only input. Injection and collation
            # must allocate their own result without changing source values.
            raw.setflags(write=False)
            self.raws.append(raw)

    def __len__(self):
        return len(self.items)

    def load_raw(self, index):
        return self.raws[index]

    @staticmethod
    def load_pose(index, frame_idx=0):
        del frame_idx
        joint = np.arange(17, dtype=np.float32)
        pose = np.stack((joint * 0.07, (joint % 3) * 0.1, (joint % 5) * 0.12), axis=-1)
        return np.asfortranarray((pose + index * 0.01)[None])

    def __getitem__(self, index):
        return {"csi": self.normalize(self.load_raw(index)),
                "pose": self.load_pose(index)}


@pytest.fixture(autouse=True)
def _restore_skeleton_config():
    original = "mmfi" if payload.N_JOINTS == 17 else "person-in-wifi-3d"
    yield
    payload.set_skeleton_config(original)


def _dataset(method, layout="fortran", mode="train", rho=0.5, dose=1.0):
    trigger_class = BadNetsTrigger if method == "badnets" else BlendedTrigger
    return PoisonedDataset(
        _LayoutMMFIBase(layout), trigger_class(seed=42), mode=mode, rho=rho,
        dose_min=1.0, dose_max=1.0, fixed_dose=dose, eps=0.185,
        pivot=1, theta_max_deg=40.0, seed=42, dataset="mmfi",
    )


def _source_expected(trigger, raw, dose):
    """Expected amplitudes from the pinned operators, preserving CHW axes."""
    image = np.moveaxis(raw, 0, -1)
    pattern = np.moveaxis(trigger.pattern, 0, -1)
    if isinstance(trigger, BadNetsTrigger):
        replacement = np.moveaxis(AddMaskPatchTrigger(pattern)(image), -1, 0)
        alpha = min(dose * trigger.patch_opacity, 1.0)
        expected = raw.copy()
        expected[trigger.mask] = ((1.0 - alpha) * raw[trigger.mask]
                                  + alpha * replacement[trigger.mask])
    else:
        expected = np.moveaxis(blendedImageAttack(pattern, min(dose * 0.185, 1.0))(image), -1, 0)
    return MMFI.normalize(expected)


def _assert_batch(batch, samples):
    csi = batch["csi"]
    # This is the operation that rejects the reported stride pattern. Testing
    # it before the flag assertion also retains the original exception message.
    flat = csi.view(len(samples), -1)
    assert csi.is_contiguous()
    assert csi.dtype == torch.float32
    assert tuple(csi.shape) == (len(samples), 3, 114, 10)
    expected = np.stack([sample["csi"] for sample in samples])
    np.testing.assert_array_equal(csi.numpy(), expected)
    np.testing.assert_array_equal(flat.numpy(), expected.reshape(len(samples), -1))
    assert set(batch) == set(samples[0])
    for key in batch:
        dtype = torch.long if key == "poisoned" else torch.float32
        assert batch[key].dtype == dtype
        numpy_dtype = np.int64 if key == "poisoned" else np.float32
        np.testing.assert_array_equal(batch[key].numpy(),
                                      np.asarray([sample[key] for sample in samples], dtype=numpy_dtype))


@pytest.mark.parametrize("layout", ["c", "fortran", "channels_last"])
def test_collate_preserves_all_fields_with_noncontiguous_csi(layout):
    base = _LayoutMMFIBase(layout)
    samples = []
    for index in range(len(base)):
        pose = base.load_pose(index)
        samples.append({"csi": base.normalize(base.load_raw(index)), "pose": pose,
                        "target": pose + 0.05, "clean_pose": pose.copy(),
                        "dose": index * 0.25, "poisoned": index % 2})
    if layout != "c":
        assert not samples[0]["csi"].flags.c_contiguous
    before = [{key: value.copy() if isinstance(value, np.ndarray) else value
               for key, value in sample.items()} for sample in samples]
    _assert_batch(collate(samples), samples)
    for original, sample in zip(before, samples):
        for key in sample:
            np.testing.assert_array_equal(sample[key], original[key])


@pytest.mark.parametrize("method", ["badnets", "blended"])
@pytest.mark.parametrize("layout", ["c", "fortran", "channels_last"])
@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_mmfi_vendor_trigger_train_batches_are_viewable(method, layout, rho):
    dataset = _dataset(method, layout=layout, rho=rho)
    raw_before = [raw.copy() for raw in dataset.base.raws]
    samples = [dataset[index] for index in range(len(dataset))]
    for index, sample in enumerate(samples):
        expected = (_source_expected(dataset.trig, raw_before[index], 1.0)
                    if index in dataset.poison_idx else MMFI.normalize(raw_before[index]))
        np.testing.assert_array_equal(sample["csi"], expected)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), num_workers=0,
                                 collate_fn=collate)))
    _assert_batch(batch, samples)
    for original, raw in zip(raw_before, dataset.base.raws):
        np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("method", ["badnets", "blended"])
@pytest.mark.parametrize("dose", [0.0, 0.4, 1.0])
def test_mmfi_vendor_trigger_evaluation_doses_keep_values_and_layout(method, dose):
    dataset = _dataset(method, mode="trigger@dose", dose=dose)
    samples = [dataset[index] for index in range(len(dataset))]
    for index, sample in enumerate(samples):
        np.testing.assert_array_equal(sample["csi"],
                                      _source_expected(dataset.trig, dataset.base.raws[index], dose))
    _assert_batch(collate(samples), samples)


@pytest.mark.parametrize("method", ["badnets", "blended"])
def test_spawned_mmfi_vendor_trigger_mixed_batches_are_viewable(method):
    dataset = _dataset(method, rho=0.5)
    assert len(dataset.poison_idx) == 2
    samples = [dataset[index] for index in range(len(dataset))]
    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=1,
                        multiprocessing_context="spawn", timeout=30, collate_fn=collate)
    batches = list(loader)
    assert len(batches) == 1
    _assert_batch(batches[0], samples)


@pytest.mark.parametrize("method", ["badnets", "blended"])
def test_mmfi_vendor_trigger_batch_runs_actual_hpeli_forward(method):
    dataset = _dataset(method, rho=1.0)
    batch = collate([dataset[index] for index in range(len(dataset))])
    model = HPELiNet(num_keypoints=17, subcarrier_num=114, dataset="mmfi").eval()
    with torch.inference_mode():
        pose, features = model(batch["csi"])
    assert tuple(pose.shape) == (len(dataset), 1, 17, 3)
    assert tuple(features.shape) == (len(dataset), 128)
    assert torch.isfinite(pose).all()
    assert torch.isfinite(features).all()
