import torch
import torch.nn as nn
from models.sk_network import SKUnit


class HPELiNet(nn.Module):
    def __init__(self, num_keypoints=14, num_coor=3, subcarrier_num=180,
                 num_person=1, dataset='person-in-wifi-3d'):
        super().__init__()
        self.num_keypoints, self.num_coor = num_keypoints, num_coor
        self.num_person, self.dataset = num_person, dataset
        num_lay = 64

        # For Person-in-WiFi-3D: subcarrier_num=180, packets=20
        # For MMFI:              subcarrier_num=114, packets=10
        # SKUnit dim1 = subcarrier_num (height), dim2 = n_pkt (width, unused in pool)
        self.skunit1 = SKUnit(3, num_lay, num_lay, dim1=subcarrier_num, dim2=10,
                              pool_dim='freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.skunit2 = SKUnit(num_lay, num_lay * 2, num_lay * 2,
                              dim1=subcarrier_num // 2, dim2=8,
                              pool_dim='freq-chan', M=1, G=64, r=4, stride=1, L=32)

        # Compute the regression geometry analytically.  A dummy pass through
        # the real SKUnits would mutate BatchNorm running statistics during
        # model construction and consume RNG state in temporary Conv layers.
        # Person-in-WiFi-3D: 180 sub × 20 pkt | MMFI: 114 sub × 10 pkt.
        n_pkt = 10 if subcarrier_num == 114 else 20
        height, width = subcarrier_num // 4, n_pkt // 4
        height = (height - 3) // 2 + 1
        height = (height - 3) // 2 + 1
        height = height - 2
        if height <= 0 or width <= 0:
            raise ValueError(
                f'unsupported HPELi input: {subcarrier_num} subcarriers, '
                f'{n_pkt} packets')
        flat_size = 16 * height * width

        self.regression = nn.Sequential(
            nn.Conv2d(128, 64, (3, 1), (2, 1), 0), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 1), (2, 1), 0), nn.ReLU(),
            nn.Conv2d(32, 16, (3, 1), (1, 1), 0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, num_keypoints * num_coor * num_person))

        # Pre-build pool — avoid creating nn.AvgPool2d inside forward() every call
        self._pool = nn.AvgPool2d((2, 2))

    def forward(self, x):
        b = x.shape[0]
        x = self._pool(self.skunit1(x))
        out1 = self._pool(self.skunit2(x))
        fea = out1.mean(3).mean(2)
        x = self.regression(out1)
        x = x.reshape(b, self.num_person, self.num_keypoints, self.num_coor)
        return x, fea


def hpeli_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
