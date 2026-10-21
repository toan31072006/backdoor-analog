"""Interface-only wrapper of the pinned author WiFlow MM-Fi 3D model.

This is not a new WiFlow variant. The vendored author forward is called
unchanged; the added feature return is its axial-attention map pooled over
time. No original experiment factory/manifest is extended by this module.
"""
from __future__ import annotations

import torch

from ._vendor.wiflow_mmfi import CSIPoseEstimationModel


class WiFlowMMFiNet(CSIPoseEstimationModel):
    def __init__(self, num_keypoints=17, num_coor=3, num_person=1,
                 subcarrier_num=114, dataset='mmfi', pretrained=False):
        if (num_keypoints, num_coor, num_person, subcarrier_num, dataset) != (17, 3, 1, 114, 'mmfi'):
            raise ValueError('WiFlow probe supports only MM-Fi single-person 17-joint 3D')
        if pretrained is not False:
            raise ValueError('No external checkpoint is loaded by this scratch probe')
        super().__init__(dropout=0.3)  # Exact source constructor default.

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1:] != (3, 114, 10) or not len(x):
            raise ValueError('WiFlow MM-Fi expects a nonempty B x 3 x 114 x 10 tensor')
        captured = []

        def feature_hook(_module, _args, output):
            captured.append(output)

        # A temporary hook avoids reimplementing/reordering the source forward.
        handle = self.att.register_forward_hook(feature_hook)
        try:
            pose = super().forward(x)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (len(x), 64, 17, 10):
            raise RuntimeError('Pinned author attention geometry changed')
        features = captured[0].mean(dim=-1).flatten(1)
        return pose.unsqueeze(1), features
