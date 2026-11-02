"""Actual HPE-Li CPU B32 updates with synthetic CSI; not a CUDA/data efficacy test."""
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ATKBackd'))


def assets_fixture():
    from attack.adap_patch_mmfi import TRAIN_NAMES, TEST_NAMES
    train = []
    for group, alpha in enumerate((.5, .2, .5, .3)):
        mark = torch.full((3, 32, 32), .15 + .2 * group)
        mask = torch.zeros((32, 32))
        mask[group * 6:group * 6 + 6, 20:30] = 1.
        train.append(dict(name=TRAIN_NAMES[group], mark=mark, mask=mask, alpha=alpha))
    by_name = {entry['name']: entry for entry in train}
    return dict(train=train, test=[dict(by_name[name], alpha=1.) for name in TEST_NAMES])


def test_real_hpeli_b32_two_sgd_updates_with_poison_cover_and_clean():
    from attack.adap_patch_mmfi import AdapPatchCSI
    from attack.payload import make_target_pose, set_skeleton_config
    from models.factory import build_model
    from train_backdoor import _victim_update

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(42)
        trigger = AdapPatchCSI(assets_fixture())
        x = torch.rand((32, 3, 114, 10), dtype=torch.float32)
        clean_x = x.clone()
        clean_pose = torch.rand((32, 1, 17, 3), dtype=torch.float32)
        labels = clean_pose.clone()
        groups = torch.arange(4)
        doses = torch.tensor([.2, .4, .6, 1.])
        # Four changed-label poison, four clean-label covers, 24 clean rows.
        x[:4] = trigger.inject_train(x[:4], doses, groups)
        x[4:8] = trigger.inject_train(x[4:8], doses, groups)
        set_skeleton_config('mmfi')
        for i, dose in enumerate(doses.tolist()):
            changed = make_target_pose(
                clean_pose[i].numpy(), 1, dose, np.deg2rad(40.), 'linear',
                (0., 0., 1.), target_joints=(2, 3),
            )
            labels[i] = torch.from_numpy(changed)
        assert torch.equal(labels[4:], clean_pose[4:])
        assert not torch.equal(labels[:4], clean_pose[:4])
        assert not torch.equal(x[:8], clean_x[:8])
        assert torch.equal(x[8:], clean_x[8:])
        assert torch.equal(trigger.inject_test(clean_x, 0.), clean_x)
        assert x.is_contiguous()

        model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                            dataset='mmfi', pretrained=False).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9,
                                    weight_decay=0.)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        for _ in range(2):
            loss = _victim_update(model, optimizer, x, labels)
            assert np.isfinite(float(loss))
            assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                       if p.grad is not None)
        assert any(not torch.equal(before[name], p.detach())
                   for name, p in model.named_parameters())
        model.eval()
        with torch.no_grad():
            output = model(trigger.inject_test(clean_x, 1.))
        if isinstance(output, (list, tuple)):
            output = output[0]
        assert output.shape == (32, 1, 17, 3)
        assert torch.isfinite(output).all()
    finally:
        torch.set_num_threads(previous_threads)
