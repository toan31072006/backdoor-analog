"""HPE-Li scratch WiPose head, independently adapted from DT-Pose geometry.

Reuses this repository's existing SKUnit parameters but independently implements
the original DT-Pose attention reshape semantics for this isolated adapter.
This is not the DT-Pose pretrained encoder/decoder.  Historical SKUnit is not
modified, and its attention transpose correction remains unchanged elsewhere.
"""
import torch
from torch import nn

from models.sk_network import SKUnit


class WiPoseSKUnit(SKUnit):
    """Retain author's active attention reshapes without changing old models."""
    def forward(self, csi):
        value = self.conv1(csi)
        attention = self.conv2_sk[0]
        batch = value.shape[0]
        branches = torch.cat([branch(value) for branch in attention.convs], dim=1)
        branches = branches.view(batch, attention.M, branches.shape[2],
                                 attention.output_dim, branches.shape[3])
        combined = branches.sum(dim=1)
        pooled = combined.mean(dim=3)
        # This reshape deliberately preserves the original author's convention.
        # Do not replace it with the corrected transpose used by other backbones.
        pooled = pooled.reshape(batch, pooled.shape[2], pooled.shape[1])
        reduced = attention.fc(pooled)
        weights = torch.cat([head(reduced) for head in attention.fcs], dim=1)
        weights = weights.reshape(batch, attention.M, attention.output_dim,
                                  attention.dim1, 1)
        weights = attention.softmax(weights).reshape(
            batch, attention.M, attention.dim1, attention.output_dim, 1)
        selected = (branches * weights).sum(dim=1).transpose(1, 2)
        return self.conv2_sk[2](self.conv2_sk[1](selected))


class WiPoseHPELi(nn.Module):
    def __init__(self):
        super().__init__()
        self.skunit1 = WiPoseSKUnit(3, 64, 64, dim1=90, dim2=5,
                             pool_dim='freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.skunit2 = WiPoseSKUnit(64, 128, 128, dim1=45, dim2=2,
                             pool_dim='freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.pool = nn.AvgPool2d((2, 2))
        self.regression = nn.Sequential(
            nn.Conv2d(128, 64, (3, 1), (2, 1)), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 1), (2, 1)), nn.ReLU(),
            nn.Conv2d(32, 16, (3, 1)), nn.ReLU(),
            nn.Flatten(), nn.Linear(32, 36))

    def forward(self, csi):
        if csi.ndim != 4 or tuple(csi.shape[1:]) != (3, 90, 5):
            raise ValueError('WiPose HPE-Li requires Bx3x90x5 input')
        if not torch.is_floating_point(csi):
            raise ValueError('WiPose HPE-Li requires floating point CSI')
        encoded = self.pool(self.skunit1(csi))
        encoded = self.pool(self.skunit2(encoded))
        features = encoded.mean(dim=(2, 3))
        return self.regression(encoded).reshape(csi.shape[0], 1, 18, 2), features


def hpeli_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def build_wipose_hpeli():
    model = WiPoseHPELi()
    model.apply(hpeli_init)
    return model
