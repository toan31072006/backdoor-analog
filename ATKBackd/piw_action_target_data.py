"""PiW native poison indices/doses and amplitude trigger, NEW action labels."""
from __future__ import annotations

import numpy as np

from attack.piw_action_targets import specification, target_pose
from piw_a1_adapter import EPS, PoisonedPairs


class ActionTargetPairs(PoisonedPairs):
    def __init__(self, base, trigger, action):
        super().__init__(base, trigger)
        self.spec = specification(action)
        self.selected = self._selected
        for name in ('pivot', 'target_joints', 'payload_axis', 'theta_max_deg', 'dose_mode'):
            self._manifest.pop(name, None)
        self._manifest.update(action=action, payload=self.spec,
            trigger='native PiWA1Trigger; post-normalization amplitude only')

    def clean_pose(self, index):
        # Geometry inspection loads TRAIN labels only, not CSI or TEST labels.
        return self.base.load_pose(self.items[index]['kpt'])

    def __getitem__(self, index):
        item = self.base[index]
        csi, pose = np.asarray(item['csi']), np.asarray(item['pose'])
        if index in self.selected:
            dose = self.selected[index]
            csi = self.trigger.inject(csi, dose, eps=EPS)
            pose = target_pose(pose, dose, self.spec).astype(np.float32)
        return dict(csi=csi, pose=pose)
