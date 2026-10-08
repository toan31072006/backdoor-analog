import hashlib
import json
import numpy as np
from torch.utils.data import Dataset
from attack.payload import (make_target_pose, set_skeleton_config, descendants,
                            PWIF3D_JOINT_NAMES)

class PoisonedDataset(Dataset):
    def __init__(self, base, trigger, mode='train', rho=0.1,
                 dose_min=0.2, dose_max=1.0, eps=0.3,
                 pivot=7, theta_max_deg=60.0, dose_mode='linear',
                 axis=(0.0, 0.0, 1.0), fixed_dose=None, seed=0, select='uniform',
                 dataset='person-in-wifi-3d', dose_coupling='paired',
                 cover_ratio=0.0):
        """
        Args:
            dataset: 'person-in-wifi-3d' or 'mmfi' - configures skeleton structure
        """
        # Configure skeleton structure based on dataset
        set_skeleton_config(dataset)

        self.base = base
        # Learned triggers (TSBA) must be injected on the collated torch batch
        # so gradients can reach their generator.  Do not retain that module in
        # this Dataset: loader workers may fork after it moves to CUDA.
        self.defer_trigger = bool(
            getattr(trigger, 'requires_deferred_injection', False))
        self.trig = None if self.defer_trigger else trigger
        self.mode = mode
        self.eps = eps
        self.pivot = pivot
        self.target_joints = tuple(descendants(pivot))
        self.pivot_name = (PWIF3D_JOINT_NAMES[pivot]
                           if dataset == 'person-in-wifi-3d' else None)
        self.target_joint_names = (
            [PWIF3D_JOINT_NAMES[j] for j in self.target_joints]
            if dataset == 'person-in-wifi-3d' else None)
        self.theta_max = np.deg2rad(theta_max_deg)
        self.dose_mode = dose_mode
        self.axis = axis
        self.fixed_dose = fixed_dose
        self.select = select
        self.dataset = dataset
        self.seed = int(seed)
        self.rho_requested = float(rho)
        self.dose_min = float(dose_min)
        self.dose_max = float(dose_max)
        self.dose_coupling = str(dose_coupling)
        self.cover_ratio = float(cover_ratio)
        if self.dose_coupling not in ('paired', 'shuffled'):
            raise ValueError('dose_coupling must be paired or shuffled')
        if self.defer_trigger and self.dose_coupling != 'paired':
            raise ValueError('shuffled coupling is only supported for fixed triggers')
        if not 0.0 <= self.cover_ratio <= 1.0:
            raise ValueError('cover_ratio must be in [0, 1]')
        if mode == 'train' and rho + self.cover_ratio > 1.0:
            raise ValueError('poison and clean-label cover fractions cannot exceed 1')
        if self.cover_ratio and not callable(getattr(trigger, 'noise_inject', None)):
            raise ValueError('clean-label covers require trigger.noise_inject')
        if not 0.0 <= self.dose_min <= self.dose_max <= 1.0:
            raise ValueError(
                'dose range must satisfy 0 <= dose_min <= dose_max <= 1, '
                f'got [{dose_min}, {dose_max}]')

        rng = np.random.default_rng(seed)
        n = len(base)
        self.n_total = n
        if mode == 'train':
            if not 0.0 <= rho <= 1.0:
                raise ValueError(f'rho must be in [0, 1], got {rho}')
            # Algorithm 1 in the paper uses floor(rho * N), not rounding.
            n_pois = int(np.floor(rho * n))
            idx = self._select_poison(rng, n, n_pois, select)
            doses = rng.uniform(dose_min, dose_max, size=len(idx))
            self.poison_plan = tuple(
                (int(i), float(d)) for i, d in zip(idx.tolist(), doses.tolist()))
            self.poison_idx = {i for i, _ in self.poison_plan}
            self.dose_of = dict(self.poison_plan)
            # Same poison indices and dose marginals; change ONLY the pairing.
            label_doses = doses.copy()
            if self.dose_coupling == 'shuffled':
                np.random.default_rng(np.random.SeedSequence(
                    [self.seed, 7043])).shuffle(label_doses)
            self.payload_dose_of = dict(zip(map(int, idx), map(float, label_doses)))
            self.cover_idx = set()
            if self.cover_ratio:
                available = np.array([i for i in range(n) if i not in self.poison_idx])
                cover_rng = np.random.default_rng(np.random.SeedSequence(
                    [self.seed, 7044]))
                self.cover_idx = set(map(int, cover_rng.choice(
                    available, int(np.floor(n * self.cover_ratio)), replace=False)))
        else:
            self.poison_plan = tuple()
            self.poison_idx = set()
            self.dose_of = {}
            self.payload_dose_of = {}
            self.cover_idx = set()
        self.n_poison = len(self.poison_idx)
        self.n_cover = len(self.cover_idx)

    def manifest(self):
        """Private audit record; never returned by the victim DataLoader."""
        samples = []
        for i, d in self.poison_plan:
            entry = {'index': i, 'dose': d}
            if self.dose_coupling == 'shuffled':
                entry['payload_dose'] = self.payload_dose_of[i]
            item = self.base.items[i]
            if isinstance(item, dict):
                if 'csi' in item:
                    entry['csi_id'] = str(item['csi'])
                if 'kpt' in item:
                    entry['pose_id'] = str(item['kpt'])
                if 'frame_idx' in item:
                    entry['frame_idx'] = int(item['frame_idx'])
            samples.append(entry)
        plan_bytes = json.dumps(
            [[i, d] for i, d in self.poison_plan],
            separators=(',', ':'), ensure_ascii=True).encode('utf-8')
        record = {
            'schema': 4,
            'seed': self.seed,
            'selection': self.select,
            'rho_requested': self.rho_requested,
            'n_total': self.n_total,
            'n_poison': self.n_poison,
            'dose_min': self.dose_min,
            'dose_max': self.dose_max,
            'pivot': int(self.pivot),
            'pivot_name': self.pivot_name,
            'target_joints': list(self.target_joints),
            'target_joint_names': self.target_joint_names,
            'theta_max_deg': float(np.rad2deg(self.theta_max)),
            'payload_axis': [float(v) for v in np.asarray(self.axis)],
            'dose_mode': self.dose_mode,
            'poison_plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
            'samples': samples,
        }
        if self.dose_coupling != 'paired' or self.cover_ratio:
            record.update(schema=5, dose_coupling=self.dose_coupling,
                          cover_ratio=self.cover_ratio, n_cover=self.n_cover,
                          cover_indices=sorted(self.cover_idx))
            assignment = [[i, d, self.payload_dose_of[i]]
                          for i, d in self.poison_plan]
            record['coupling_sha256'] = hashlib.sha256(json.dumps(
                assignment, separators=(',', ':')).encode('utf-8')).hexdigest()
        return record

    def _select_poison(self, rng, n, n_pois, select):
        """Which training samples to poison.
        'uniform'  : random subset (standard).
        'diverse'  : farthest-point sampling over poses -> better pose coverage, which
                     can plant the same backdoor at lower rho (a poison-efficiency study).
        """
        if n_pois <= 0:
            return np.array([], int)
        if select == 'uniform':
            return rng.choice(n, size=n_pois, replace=False)
        if select == 'diverse':
            print(f'[poison] diverse FPS sampling: loading {n} poses...', flush=True)
            poses = []
            for i in range(n):
                it = self.base.items[i]
                if 'frame_idx' in it:
                    p = self.base.load_pose(it['kpt'], it['frame_idx'])
                else:
                    p = self.base.load_pose(it['kpt'])
                poses.append(p.reshape(-1))
            poses = np.stack(poses)
            if not np.isfinite(poses).all():
                raise ValueError('diverse poison selection received NaN/Inf poses')
            print(f'[poison] FPS selecting {n_pois} from {n}...', flush=True)
            chosen = [int(rng.integers(n))]
            selected = np.zeros(n, dtype=bool)
            selected[chosen[0]] = True
            d = np.linalg.norm(poses - poses[chosen[0]], axis=1)
            d[selected] = -np.inf
            for _ in range(n_pois - 1):
                nxt = int(np.argmax(d))
                chosen.append(nxt)
                selected[nxt] = True
                d = np.minimum(d, np.linalg.norm(poses - poses[nxt], axis=1))
                d[selected] = -np.inf
            print(f'[poison] FPS done.', flush=True)
            return np.array(chosen, int)
        raise ValueError(f'unknown select policy {select}')

    def _inject(self, name_csi_raw, dose):
        return self.trig.inject(name_csi_raw, dose, eps=self.eps)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = self.base.items[i]
        raw = self.base.load_raw(it['csi'])
        # MMFI items carry 'frame_idx'; PersonInWiFi3D items do not
        if 'frame_idx' in it:
            pose = self.base.load_pose(it['kpt'], it['frame_idx'])
        else:
            pose = self.base.load_pose(it['kpt'])
        clean_pose = pose.copy() if self.defer_trigger else None

        if self.mode == 'train':
            if i in self.poison_idx:
                d = self.dose_of[i]
                if not self.defer_trigger:
                    raw = self._inject(raw, d)
                pose = make_target_pose(pose, self.pivot, self.payload_dose_of[i], self.theta_max,
                                        self.dose_mode, self.axis,
                                        target_joints=self.target_joints)
                poisoned = 1
            else:
                d = 0.0
                poisoned = 0
                if i in self.cover_idx:
                    # Noise-mode covers keep their original pose labels.
                    raw = self.trig.noise_inject(
                        raw, seed=int(np.random.SeedSequence(
                            [self.seed, i, 7045]).generate_state(1)[0]),
                        eps=self.eps, dose=1.0)
            csi = self.base.normalize(raw)
            # The fixed-trigger paper path exposes an ordinary supervised pair
            # only.  Attack metadata is available solely to the explicitly
            # white-box learned-trigger baseline.
            out = {'csi': csi, 'pose': pose}
            if self.defer_trigger:
                out.update(clean_pose=clean_pose, dose=float(d),
                           poisoned=poisoned)
            return out

        if self.mode == 'clean':
            return {'csi': self.base.normalize(raw), 'pose': pose,
                    'target': pose, 'dose': 0.0}

        if self.mode == 'trigger@dose':
            d = self.fixed_dose if self.fixed_dose is not None else 1.0
            target = make_target_pose(pose, self.pivot, d, self.theta_max,
                                      self.dose_mode, self.axis,
                                      target_joints=self.target_joints)
            raw_t = raw if self.defer_trigger else self._inject(raw, d)
            return {'csi': self.base.normalize(raw_t),
                    'pose': pose,            # true (clean) pose
                    'target': target,        # attacker-intended pose at this dose
                    'dose': float(d)}

        raise ValueError(self.mode)

def collate(batch):
    import torch
    out = {}
    # NumPy stacking can preserve Fortran/HWC-derived CSI strides. HPELi's
    # view-based SK blocks need contiguous NCHW; this changes storage only,
    # not CSI values, trigger operators, payloads, or training configuration.
    out['csi'] = torch.from_numpy(np.stack([b['csi'] for b in batch])).float().contiguous()
    out['pose'] = torch.from_numpy(np.stack([b['pose'] for b in batch])).float()
    if 'target' in batch[0]:
        out['target'] = torch.from_numpy(np.stack([b['target'] for b in batch])).float()
    if 'dose' in batch[0]:
        out['dose'] = torch.tensor([b['dose'] for b in batch]).float()
    if 'poisoned' in batch[0]:
        out['poisoned'] = torch.tensor([b['poisoned'] for b in batch]).long()
    if 'clean_pose' in batch[0]:
        out['clean_pose'] = torch.from_numpy(
            np.stack([b['clean_pose'] for b in batch])).float()
    return out
