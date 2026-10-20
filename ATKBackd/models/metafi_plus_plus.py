"""MetaFi++ MM-Fi backbone, independently implemented from its architecture.

Reference: cseeyangchen/DT-Pose, pinned by metafi_sources.json. This is the
MetaFi++ baseline, not the two-stage DT-Pose MAE model. Shared ResNet34 antenna
branches, channel attention and the original decoder are retained. In
particular the final BatchNorm1d is AFTER the decoder ReLU and can produce
negative coordinates; removing that ReLU is unnecessary and is not done here.

This module does not redistribute the authors' unlicensed source. It uses the
torchvision ResNet34 implementation (torchvision BSD-3-Clause license).
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.transforms import Resize

from .metafi_channel_transformer import MetaFiChannelTransformer


def metafi_init(module: nn.Module) -> None:
    """DT-Pose baseline's post-construction Conv/BN affine initialization."""
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MetaFiPlusPlusNet(nn.Module):
    """MM-Fi input Bx3x114xT -> (Bx1x17x3 pose, Bx512 features).

    ``pretrained=False`` is a fully offline scratch initialization. The source
    constructs ImageNet-pretrained ResNet34s and subsequently reinitializes
    every Conv and BN affine parameter; their BN *running statistics* are not
    reset. We explicitly do not inherit those statistics in scratch mode.
    ``pretrained=True`` is opt-in and may download torchvision ImageNet weights;
    it retains those running statistics before the same Conv/BN initialization.
    Only the one ResNet actually used by the published forward is constructed.
    """

    def __init__(self, num_keypoints: int = 17, num_coor: int = 3,
                 num_person: int = 1, subcarrier_num: int = 114,
                 dataset: str = "mmfi", pretrained: bool = False):
        super().__init__()
        if dataset not in {"mmfi", "mmfi-csi"}:
            raise ValueError("MetaFi++ integration is validated only for MM-Fi")
        if (num_keypoints, num_coor, num_person, subcarrier_num) != (17, 3, 1, 114):
            raise ValueError("MetaFi++ MM-Fi requires 17 joints, 3 coordinates, 1 person, 114 subcarriers")
        if not isinstance(pretrained, bool):
            raise ValueError("pretrained must be an explicit bool (False is offline scratch)")
        self.num_keypoints, self.num_coor = num_keypoints, num_coor
        self.num_person, self.dataset = num_person, dataset
        self.subcarrier_num = subcarrier_num
        self.pretrained = bool(pretrained)
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder_conv1_p1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        self.encoder_bn1_p1 = backbone.bn1
        self.encoder_relu_p1 = backbone.relu
        # Source defines maxpool but does not execute it; retain the attribute.
        self.encoder_maxpool_p1 = backbone.maxpool
        self.encoder_layer1_p1 = backbone.layer1
        self.encoder_layer2_p1 = backbone.layer2
        self.encoder_layer3_p1 = backbone.layer3
        self.encoder_layer4_p1 = backbone.layer4
        self.tf = MetaFiChannelTransformer(joints=17, channels=512)
        self.decode = nn.Sequential(
            nn.Conv2d(512, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, bias=False),
            nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.bn1 = nn.BatchNorm1d(3)
        self.bn2 = nn.BatchNorm2d(512)
        self._resize = Resize((136, 32), antialias=True)
        self._pose_pool = nn.AvgPool2d((1, 12), stride=1)
        self.apply(metafi_init)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] != 114 or x.shape[3] < 1:
            raise ValueError("MetaFi++ expects B x 3 x 114 x T, T >= 1")
        # The order of shared BN calls matches source (each stage processes
        # antennas 1/2/3 before moving to the next stage). Do not batch-fold
        # antennas: that would alter training-time batch statistics.
        branches = [self._resize(x[:, i:i + 1]) for i in range(3)]
        for layer in (self.encoder_conv1_p1, self.encoder_bn1_p1,
                      self.encoder_relu_p1, self.encoder_layer1_p1,
                      self.encoder_layer2_p1, self.encoder_layer3_p1,
                      self.encoder_layer4_p1):
            branches = [layer(branch) for branch in branches]
        encoded = self.tf(self.bn2(torch.cat(branches, dim=3)))
        features = encoded.mean(dim=3).mean(dim=2)
        coordinates = self.bn1(self._pose_pool(self.decode(encoded)).squeeze(-1))
        pose = coordinates.transpose(1, 2).reshape(len(x), 1, 17, 3)
        return pose, features
