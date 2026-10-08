"""Spawned loaders must preserve MM-Fi poison geometry without global state."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import attack.payload as payload  # noqa: E402
from attack.poison import PoisonedDataset, collate as poison_collate  # noqa: E402
from train_rf_backdoor import _TrainingPairs  # noqa: E402


class _SpawnMMFIBase(Dataset):
    """Importable worker fixture; no filesystem data or model is needed."""

    def __init__(self):
        self.items = [{"csi": index, "kpt": index, "frame_idx": 0} for index in range(4)]

    def __len__(self):
        return len(self.items)

    def load_raw(self, index):
        return np.full((3, 114, 10), 0.3 + 0.1 * index, dtype=np.float32)

    def load_pose(self, index, frame_idx=0):
        joint = np.arange(17, dtype=np.float32)
        pose = np.stack((joint * 0.07, (joint % 3) * 0.1, (joint % 5) * 0.12), axis=-1)
        return (pose + index * 0.01)[None]

    @staticmethod
    def normalize(raw):
        return np.array(raw, dtype=np.float32, copy=True)

    def __getitem__(self, index):
        return {"csi": self.load_raw(index), "pose": self.load_pose(index)}


class _IdentityTrigger:
    """Isolate pose labels from trigger details and numerical perturbations."""

    @staticmethod
    def inject(csi, dose, eps=0.185):
        return np.array(csi, dtype=np.float32, copy=True)


def _expected_pose(base):
    poses = np.stack([base.load_pose(index) for index in range(len(base))])
    targets = poses.copy()
    angle = np.deg2rad(40.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    pivot = poses[..., 1, :]
    for joint in (2, 3):
        targets[..., joint, :] = (poses[..., joint, :] - pivot) @ rotation.T + pivot
    return torch.from_numpy(targets), torch.from_numpy(poses)


@pytest.mark.parametrize("dataset_kind", ["ordinary_poison", "rf_training_pairs"])
def test_spawned_worker_preserves_mmfi_targets_after_parent_geometry_changes(dataset_kind):
    original_dataset = "mmfi" if payload.N_JOINTS == 17 else "person-in-wifi-3d"
    try:
        payload.set_skeleton_config("mmfi")
        base = _SpawnMMFIBase()
        if dataset_kind == "ordinary_poison":
            dataset = PoisonedDataset(
                base, _IdentityTrigger(), mode="train", rho=1.0,
                dose_min=1.0, dose_max=1.0, pivot=1, theta_max_deg=40.0,
                dose_mode="linear", axis=(0.0, 0.0, 1.0), seed=42, dataset="mmfi",
            )
        else:
            cfg = {"pivot": 1, "theta_max_deg": 40.0, "dose_mode": "linear", "payload_axis": [0.0, 0.0, 1.0]}
            dataset = _TrainingPairs(base, {index: 0 for index in range(len(base))}, cfg)
        # A worker imports this default independently; mutating the parent
        # also proves labels are owned by the dataset rather than live globals.
        payload.set_skeleton_config("person-in-wifi-3d")
        assert payload.descendants(1) == []
        collate_fn = poison_collate if dataset_kind == "ordinary_poison" else None
        local = list(DataLoader(dataset, batch_size=len(base), num_workers=0, collate_fn=collate_fn))[0]
        spawned = list(DataLoader(
            dataset, batch_size=len(base), num_workers=1,
            multiprocessing_context="spawn", timeout=30, collate_fn=collate_fn,
        ))[0]
        expected, clean = _expected_pose(base)
        torch.testing.assert_close(local["pose"], expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(spawned["pose"], expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(local["pose"], spawned["pose"], rtol=0, atol=0)
        assert not torch.equal(expected[..., [2, 3], :], clean[..., [2, 3], :])
        unchanged = [joint for joint in range(17) if joint not in (2, 3)]
        torch.testing.assert_close(spawned["pose"][..., unchanged, :], clean[..., unchanged, :], rtol=0, atol=0)
    finally:
        payload.set_skeleton_config(original_dataset)
