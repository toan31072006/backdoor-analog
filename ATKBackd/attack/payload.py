import numpy as np

# ========== Person-in-WiFi-3D (14 joints) ==========
# DT-Pose's GCN adjacency is cyclic and must not be mistaken for a kinematic
# parent tree.  Retain it for model-graph provenance only.
PWIF3D_MODEL_GRAPH_EDGES = [
    (0, 1), (0, 2), (2, 5), (3, 0), (4, 2), (5, 7), (6, 3),
    (7, 3), (8, 4), (9, 5), (10, 6), (11, 7), (12, 9), (13, 11),
]
PWIF3D_JOINT_NAMES = (
    'neck', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'left_hip', 'right_elbow', 'right_hip',
    'left_hand', 'left_knee', 'right_hand', 'right_knee',
    'left_ankle', 'right_ankle',
)
# Explicit tree used by the manuscript geometry configs.  The extra (5, 7)
# hip-to-hip edge in PWIF3D_MODEL_GRAPH_EDGES closes a torso cycle and is a GCN
# relation, not a parent-child bone, so it is deliberately excluded here.
PWIF3D_KINEMATIC_EDGES = [
    (0, 1),
    (0, 2), (2, 4), (4, 8), (2, 5), (5, 9), (9, 12),
    (0, 3), (3, 6), (6, 10), (3, 7), (7, 11), (11, 13),
]
PWIF3D_EDGES = PWIF3D_KINEMATIC_EDGES

# ========== MMFI (17 joints) ==========
# Official topology, taken verbatim from DT-Pose
# (model/model.py :: generate_adjacency_matrix, dataset='mmfi-csi'):
#     0-1-2-3        right leg          0-4-5-6        left leg
#     0-7-8          pelvis -> thorax
#     8-9-10         neck -> head       8-11-12-13     one arm
#                                       8-14-15-16     other arm
# The previous COCO-style graph here rooted at 5 left joints 0-4 in a separate
# component (only 12/17 reachable), so no pivot could ever rotate them.
MMFI_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

# Global configuration (will be set by set_skeleton_config())
N_JOINTS = 14
ROOT = 0
_CURRENT_EDGES = PWIF3D_EDGES


def set_skeleton_config(dataset='person-in-wifi-3d'):
    """Set global skeleton configuration based on dataset."""
    global N_JOINTS, ROOT, _CURRENT_EDGES, PARENT, CHILDREN, ADJ

    if dataset == 'mmfi':
        N_JOINTS = 17
        ROOT = 0  # pelvis — the hub in DT-Pose's mmfi-csi graph
        _CURRENT_EDGES = MMFI_EDGES
    else:  # default to person-in-wifi-3d
        N_JOINTS = 14
        ROOT = 0
        _CURRENT_EDGES = PWIF3D_EDGES

    # Rebuild tree with new configuration
    PARENT, CHILDREN, ADJ = _build_tree(_CURRENT_EDGES, N_JOINTS, ROOT)


def _build_tree(edges, n, root):
    if len(edges) != n - 1:
        raise ValueError(
            f'kinematic graph must contain n-1 edges, got {len(edges)} for n={n}')
    adj = {i: [] for i in range(n)}
    for a, b in edges:
        if a not in adj or b not in adj or a == b:
            raise ValueError(f'invalid kinematic edge {(a, b)}')
        adj[a].append(b); adj[b].append(a)
    parent = {root: None}; order = [root]; seen = {root}
    qi = 0
    while qi < len(order):
        u = order[qi]; qi += 1
        for v in adj[u]:
            if v not in seen:
                seen.add(v); parent[v] = u; order.append(v)
    if len(seen) != n:
        raise ValueError(
            f'kinematic graph is disconnected: reached {len(seen)} of {n} joints')
    children = {i: [] for i in range(n)}
    for v, p in parent.items():
        if p is not None:
            children[p].append(v)
    return parent, children, adj


# Initialize with default Person-in-WiFi-3D configuration
PARENT, CHILDREN, ADJ = _build_tree(_CURRENT_EDGES, N_JOINTS, ROOT)


def descendants(pivot):
    out = []
    stack = list(CHILDREN[pivot])
    while stack:
        j = stack.pop(); out.append(j); stack.extend(CHILDREN[j])
    return sorted(out)


def terminal_chains():
    leaves = [j for j in range(N_JOINTS) if not CHILDREN[j]]
    chains = []
    for leaf in leaves:
        path = [leaf]; p = PARENT[leaf]
        while p is not None:
            path.append(p); p = PARENT[p]
        path = path[::-1]                              # root ... leaf
        # report the distal 3-joint segment (pivot, mid, distal) when available
        seg = path[-3:] if len(path) >= 3 else path
        chains.append({'leaf': leaf, 'root_path': path,
                       'pivot': seg[0], 'segment': seg,
                       'rotated_joints': descendants(seg[0])})
    return chains

# --------------------------------------------------------------------------- rotation
def _rodrigues(axis, theta):
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        raise ValueError('rotation axis must be non-zero')
    axis = axis / norm
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def g_dose(dose, theta_max=np.deg2rad(60.0), mode='linear'):
    """Monotone dose -> rotation-angle map. dose in [0,1]."""
    dose = float(np.clip(dose, 0.0, 1.0))
    if mode == 'linear':
        return theta_max * dose
    if mode == 'sqrt':
        return theta_max * np.sqrt(dose)
    if mode == 'quad':
        return theta_max * dose ** 2
    raise ValueError(mode)


def rotate_subchain(pose, pivot, theta, axis=(0.0, 0.0, 1.0), target_joints=None):
    pose = np.array(pose, float, copy=True)
    # Datasets may be unpickled in spawned workers whose module-global tree
    # defaults to PiW3D. An explicit, persisted branch makes labels independent
    # of worker imports or a different dataset configured in the same process.
    rot_js = descendants(pivot) if target_joints is None else list(target_joints)
    if any(j == pivot or j < 0 or j >= pose.shape[-2] for j in rot_js):
        raise ValueError('target_joints must be valid non-pivot pose indices')
    if not rot_js:
        return pose
    R = _rodrigues(axis, theta)
    pj = pose[..., pivot, :]                            # (...,3) pivot position
    for j in rot_js:
        rel = pose[..., j, :] - pj
        pose[..., j, :] = (rel @ R.T) + pj
    return pose


def make_target_pose(pose, pivot, dose, theta_max=np.deg2rad(60.0),
                     mode='linear', axis=(0.0, 0.0, 1.0), target_joints=None):
    return rotate_subchain(pose, pivot, g_dose(dose, theta_max, mode), axis,
                           target_joints=target_joints)


def subchain_bone_lengths(pose, pivot):
    js = [pivot] + descendants(pivot)
    L = []
    for j in js:
        p = PARENT[j]
        if p is not None and p in js:
            L.append(np.linalg.norm(pose[..., j, :] - pose[..., p, :], axis=-1))
    return np.array(L)


def all_bone_lengths(pose):
    """Lengths of every edge in the active payload kinematic tree."""
    pose = np.asarray(pose)
    return np.stack([
        np.linalg.norm(pose[..., a, :] - pose[..., b, :], axis=-1)
        for a, b in _CURRENT_EDGES
    ], axis=-1)
