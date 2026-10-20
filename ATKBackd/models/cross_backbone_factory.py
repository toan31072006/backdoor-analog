"""Explicit MM-Fi extension; the historical HPE-Li factory stays unchanged."""
from __future__ import annotations

BACKBONES = ('metafiplusplus', 'graphposefi')


def build_backbone(name, num_keypoints=17, num_coor=3, num_person=1,
                   subcarrier_num=114, dataset='mmfi', pretrained=False):
    if name not in BACKBONES:
        raise ValueError(f'Unsupported cross-backbone victim: {name!r}')
    if (dataset != 'mmfi' or num_keypoints != 17 or num_coor != 3
            or num_person != 1 or subcarrier_num != 114):
        raise ValueError('This adapter is validated only for MM-Fi single-person 17-joint 3D CSI')
    if pretrained is not False:
        raise ValueError('The frozen cross-backbone study trains from scratch; no pretrained weights')
    if name == 'metafiplusplus':
        from models.metafi_plus_plus import MetaFiPlusPlusNet
        return MetaFiPlusPlusNet(num_keypoints=17, num_coor=3, num_person=1,
                                subcarrier_num=114, dataset='mmfi', pretrained=False)
    from models.graphpose_fi import GraphPoseFiNet, graphpose_init
    model = GraphPoseFiNet(num_keypoints=17, num_coor=3, num_person=1,
                           dataset='mmfi', pretrained=False)
    model.apply(graphpose_init)
    return model
