"""Pinned WiFlow architecture with PiW3D-specific input and joint geometry.

The author MM-Fi model uses 342=3*114 input channels and 272=17*16 width
before four stride-two blocks. PiW3D needs 540=3*180 and 224=14*16. Its
temporal, convolutional, axial-attention and decoder components otherwise
remain the pinned author's implementation. This is an adaptation trained from
scratch, not an unchanged author model or transferred MM-Fi checkpoint.
"""
from __future__ import annotations

import torch
from torch import nn

from ._vendor.wiflow_mmfi import CSIPoseEstimationModel, TemporalBlock


class WiFlowPiW3DNet(CSIPoseEstimationModel):
    def __init__(self, num_keypoints: int = 14, num_coor: int = 3,
                 num_person: int = 1, subcarrier_num: int = 180,
                 dataset: str = 'person-in-wifi-3d', pretrained: bool = False):
        if (num_keypoints, num_coor, num_person, subcarrier_num, dataset) != (
                14, 3, 1, 180, 'person-in-wifi-3d'):
            raise ValueError('PiW3D WiFlow requires one person, 14 3D joints and 180 CSI rows')
        if pretrained is not False:
            raise ValueError('PiW3D WiFlow is a scratch adaptation; no pretrained weights')
        super().__init__(dropout=0.3)
        self.num_keypoints, self.num_coor = num_keypoints, num_coor
        self.num_person, self.subcarrier_num = num_person, subcarrier_num
        self.dataset, self.pretrained = dataset, False
        self.tcn = TemporalBlock(num_inputs=540, num_channels=[540, 306, 288],
                                 kernel_size=3, dropout=0.3, attention_type='none')
        self.tcn_proj = nn.Sequential(
            nn.Conv1d(288, 224, kernel_size=1, bias=False),
            nn.BatchNorm1d(224), nn.SiLU(inplace=True))
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 180, 20) or len(x) == 0:
            raise ValueError('PiW3D WiFlow expects nonempty B x 3 x 180 x 20 CSI')
        captured: list[torch.Tensor] = []

        def feature_hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> None:
            captured.append(output)

        handle = self.att.register_forward_hook(feature_hook)
        try:
            pose = super().forward(x)
        finally:
            handle.remove()
        if pose.shape != (len(x), 14, 3) or len(captured) != 1 or (
                captured[0].shape != (len(x), 64, 14, 20)):
            raise RuntimeError('PiW3D WiFlow output geometry changed')
        features = captured[0].mean(dim=-1).flatten(1)
        return pose.unsqueeze(1), features
