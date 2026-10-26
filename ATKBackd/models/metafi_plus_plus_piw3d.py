"""MetaFi++ architecture adapted to one-person Person-in-WiFi-3D.

This is a scratch, dataset-specific adaptation of the MM-Fi baseline in
``metafi_plus_plus.py``. The shared ResNet34 antenna encoder, channel
transformer and coordinate decoder are retained. Resizing to 112 x 32 makes
each antenna branch 14 x 4 after the encoder, so concatenating three branches
produces the transformer's required 14 x 12 grid. No MM-Fi checkpoint or
published PiW MetaFi++ result is implied by this adapter.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet34
from torchvision.transforms import Resize

from .metafi_channel_transformer import MetaFiChannelTransformer
from .metafi_plus_plus import metafi_init


class MetaFiPlusPlusPiW3DNet(nn.Module):
    """Map finite-size B x 3 x 180 x 20 CSI to B x 1 x 14 x 3 pose."""

    def __init__(self, num_keypoints: int = 14, num_coor: int = 3,
                 num_person: int = 1, subcarrier_num: int = 180,
                 dataset: str = 'person-in-wifi-3d', pretrained: bool = False):
        super().__init__()
        if (num_keypoints, num_coor, num_person, subcarrier_num, dataset) != (
                14, 3, 1, 180, 'person-in-wifi-3d'):
            raise ValueError('PiW3D MetaFi++ requires one person, 14 3D joints and 180 CSI rows')
        if pretrained is not False:
            raise ValueError('PiW3D MetaFi++ is a scratch adaptation; no pretrained weights')
        self.num_keypoints, self.num_coor = num_keypoints, num_coor
        self.num_person, self.subcarrier_num = num_person, subcarrier_num
        self.dataset, self.pretrained = dataset, False

        backbone = resnet34(weights=None)
        self.encoder_conv1_p1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.encoder_bn1_p1 = backbone.bn1
        self.encoder_relu_p1 = backbone.relu
        self.encoder_maxpool_p1 = backbone.maxpool  # source defines but does not execute it
        self.encoder_layer1_p1 = backbone.layer1
        self.encoder_layer2_p1 = backbone.layer2
        self.encoder_layer3_p1 = backbone.layer3
        self.encoder_layer4_p1 = backbone.layer4
        self.tf = MetaFiChannelTransformer(joints=14, channels=512)
        self.decode = nn.Sequential(
            nn.Conv2d(512, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, bias=False),
            nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.bn1 = nn.BatchNorm1d(3)
        self.bn2 = nn.BatchNorm2d(512)
        self._resize = Resize((112, 32), antialias=True)
        self._pose_pool = nn.AvgPool2d((1, 12), stride=1)
        self.apply(metafi_init)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 180, 20) or len(x) == 0:
            raise ValueError('PiW3D MetaFi++ expects nonempty B x 3 x 180 x 20 CSI')
        branches = [self._resize(x[:, i:i + 1]) for i in range(3)]
        for layer in (self.encoder_conv1_p1, self.encoder_bn1_p1,
                      self.encoder_relu_p1, self.encoder_layer1_p1,
                      self.encoder_layer2_p1, self.encoder_layer3_p1,
                      self.encoder_layer4_p1):
            branches = [layer(branch) for branch in branches]
        joined = torch.cat(branches, dim=3)
        if tuple(joined.shape[-2:]) != (14, 12):
            raise RuntimeError('PiW3D MetaFi++ encoder geometry changed')
        encoded = self.tf(self.bn2(joined))
        features = encoded.mean(dim=3).mean(dim=2)
        coordinates = self.bn1(self._pose_pool(self.decode(encoded)).squeeze(-1))
        pose = coordinates.transpose(1, 2).reshape(len(x), 1, 14, 3)
        return pose, features
