"""Scoped scratch model factory for one-person Person-in-WiFi-3D."""
from __future__ import annotations

BACKBONES = ('metafiplusplus', 'wiflow')


def build_piw3d_backbone(name: str, num_keypoints: int = 14, num_coor: int = 3,
                         num_person: int = 1, subcarrier_num: int = 180,
                         dataset: str = 'person-in-wifi-3d', pretrained: bool = False):
    if name not in BACKBONES:
        raise ValueError(f'Unsupported PiW3D backbone: {name!r}')
    kwargs = dict(num_keypoints=num_keypoints, num_coor=num_coor,
                  num_person=num_person, subcarrier_num=subcarrier_num,
                  dataset=dataset, pretrained=pretrained)
    if name == 'metafiplusplus':
        from .metafi_plus_plus_piw3d import MetaFiPlusPlusPiW3DNet
        return MetaFiPlusPlusPiW3DNet(**kwargs)
    from .wiflow_piw3d import WiFlowPiW3DNet
    return WiFlowPiW3DNet(**kwargs)
