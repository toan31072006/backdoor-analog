"""Compatibility view of the canonical MMFi skeleton in ``attack.payload``.

New code should import from ``attack.payload`` directly. Keeping this module as
a re-export prevents the formerly duplicated COCO-style topology from silently
diverging again.
"""

from attack.payload import MMFI_EDGES, _build_tree


N_JOINTS = 17
ROOT = 0


PARENT, CHILDREN, ADJ = _build_tree(MMFI_EDGES, N_JOINTS, ROOT)


def descendants(pivot):
    """Return all joints distal to pivot joint."""
    out = []
    stack = list(CHILDREN[pivot])
    while stack:
        j = stack.pop()
        out.append(j)
        stack.extend(CHILDREN[j])
    return sorted(out)


# Names follow the branch structure used by DT-Pose; A/B avoids asserting a
# left/right convention that is not encoded in the data loader.
JOINT_NAMES = {
    0: 'pelvis', 1: 'leg-A-1', 2: 'leg-A-2', 3: 'leg-A-3',
    4: 'leg-B-1', 5: 'leg-B-2', 6: 'leg-B-3',
    7: 'spine', 8: 'thorax', 9: 'neck', 10: 'head',
    11: 'arm-A-1', 12: 'arm-A-2', 13: 'arm-A-3',
    14: 'arm-B-1', 15: 'arm-B-2', 16: 'arm-B-3',
}
