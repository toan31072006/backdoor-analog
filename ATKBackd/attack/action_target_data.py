"""Data-only adapter for the separate Cross/Nod target-action experiment."""
from __future__ import annotations

import numpy as np
from attack.poison import PoisonedDataset
from attack.action_targets import target_pose


class ActionTargetDataset(PoisonedDataset):
    def __init__(self, base, trigger, cfg, mode='train', fixed_dose=None,
                 cache_targets=True):
        if mode not in ('train', 'clean', 'trigger@dose'):
            raise ValueError('Unknown action-target dataset mode')
        if (cfg['poison_select'] != 'uniform' or cfg['dose_coupling'] != 'paired'
                or getattr(trigger, 'requires_deferred_injection', False)):
            raise ValueError('Only the fixed data-only A1 operator with uniform paired doses')
        super().__init__(base, trigger, mode=mode, rho=cfg['rho'],
            dose_min=cfg['dose_min'], dose_max=cfg['dose_max'], eps=cfg['eps'],
            pivot=cfg['pivot'], theta_max_deg=cfg['theta_max_deg'],
            dose_mode='linear', axis=cfg['payload_axis'], fixed_dose=fixed_dose,
            seed=cfg['seed'], select='uniform', dataset='mmfi', dose_coupling='paired')
        self.payload = cfg['action_payload']
        self.target_joints = tuple(self.payload['target_joints'])
        self.target_joint_names = None
        self._target_cache = {}
        if mode == 'train' and cache_targets:
            # Poison IDs and doses are fixed: precompute only their labels,
            # not CSI. This is ~3MiB, not a new sampler or label augmentation.
            for index, dose in self.poison_plan:
                self._target_cache[index] = target_pose(self.clean_pose(index), dose, self.payload)

    def clean_pose(self, index):
        item = self.base.items[index]
        return self.base.load_pose(item['kpt'], item['frame_idx'])

    def manifest(self):
        record = super().manifest()
        for name in ('pivot', 'pivot_name', 'payload_axis', 'theta_max_deg'):
            record.pop(name, None)
        record.update(schema=101, payload=self.payload, target_joints=list(self.target_joints),
                      profile='new_target_actions_not_motion_prior_sensitivity')
        return record

    def __getitem__(self, index):
        item = self.base.items[index]
        raw = self.base.load_raw(item['csi'])
        truth = self.clean_pose(index)
        if self.mode == 'train':
            dose = self.dose_of.get(index, 0.)
            if index in self.poison_idx:
                raw = self.trig.inject(raw, dose, eps=self.eps)
                label = self._target_cache.get(index)
                if label is None:
                    label = target_pose(truth, dose, self.payload)
            else:
                label = truth
            # The victim loss cannot see dose, poison flags or action labels.
            return {'csi': self.base.normalize(raw), 'pose': np.array(label, copy=True)}
        dose = 0. if self.mode == 'clean' else float(self.fixed_dose)
        target = target_pose(truth, dose, self.payload)
        raw = raw if dose == 0 else self.trig.inject(raw, dose, eps=self.eps)
        return {'csi': self.base.normalize(raw), 'pose': truth,
                'target': target, 'dose': dose}
