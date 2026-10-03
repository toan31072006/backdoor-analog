def build_model(name, num_keypoints=14, num_coor=3, num_person=1,
                subcarrier_num=180, dataset='person-in-wifi-3d', pretrained=False):
    """
    Build model with proper configuration for dataset.

    Args:
        name: Model name ('hpeli', 'metafiplusplus', 'graphposefi')
        num_keypoints: Number of keypoints (14 for PWIF3D, 17 for MMFI)
        num_coor: Coordinates per keypoint (default 3 for xyz)
        num_person: Number of people (default 1)
        subcarrier_num: Number of CSI subcarriers (default 180)
        dataset: Dataset name ('person-in-wifi-3d' or 'mmfi')
        pretrained: Whether to use pretrained weights
    """
    name = name.lower()

    # Auto-adjust num_keypoints based on dataset if not explicitly set
    if dataset == 'mmfi' and num_keypoints == 14:
        num_keypoints = 17
        print(f"[build_model] Auto-adjusted num_keypoints to 17 for MMFI dataset")

    if name == 'hpeli':
        from models.hpeli import HPELiNet, hpeli_init
        m = HPELiNet(num_keypoints, num_coor, subcarrier_num, num_person, dataset)
        m.apply(hpeli_init)
        return m
    if name == 'metafiplusplus':
        raise ValueError(
            "Victim 'metafiplusplus' has been removed: its decoder ends in a ReLU, so "
            "every predicted coordinate is clamped to >= 0 while MMFi ground truth "
            "contains negative x/y. That inflates its clean error systematically and, "
            "because the ASR thresholds scale with the clean floor, distorts any "
            "cross-victim comparison. Fix the output activation before reinstating it.")
    if name == 'graphposefi':
        raise ValueError(
            "Victim 'graphposefi' has been removed pending a topology decision: its "
            "GraFormer adjacency still uses the old COCO-style MMFi graph, not DT-Pose's "
            "mmfi-csi topology that attack/payload.py now follows, so the rotated "
            "sub-chain is not a connected sub-chain in the model's own graph.")
    raise ValueError(f"Unknown model: {name}. Only 'hpeli' is available.")
