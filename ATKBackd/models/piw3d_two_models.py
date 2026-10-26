"""Entry point for the two scratch PiW3D backbone adaptations.

Both models consume B x 3 x 180 x 20 and return (B x 1 x 14 x 3 pose,
auxiliary features). They are dataset adaptations of the named architectures,
not author-reported PiW3D implementations or MM-Fi checkpoint transfers.
"""
from __future__ import annotations

from .piw3d_backbone_factory import BACKBONES, build_piw3d_backbone


def build_piw3d_model(backbone: str):
    return build_piw3d_backbone(backbone)


__all__ = ['BACKBONES', 'build_piw3d_model']
