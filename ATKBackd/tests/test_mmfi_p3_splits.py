"""Check the P3 launcher changes only dataset split, not the A1 recipe."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_a1_three_tables as p1  # noqa: E402
import run_mmfi_p3_splits as p3  # noqa: E402
from mmfi_s2_split import MMFIS2  # noqa: E402


def test_p3_splits_keep_p1_a1_operator_and_training_recipe(tmp_path):
    plan = p3.build_plan(tmp_path / 'home', tmp_path / 'out', device='cpu')
    assert [c['id'] for c in plan['cells']] == [
        'p3_s3_clean', 'p3_s3_a1', 'p3_s2_clean', 'p3_s2_a1',
        'p3_s1_clean', 'p3_s1_a1']
    assert plan['contract']['selection'].startswith('fixed final epoch49')
    for cell in plan['cells']:
        cfg = cell['cfg']
        old = p1._configuration(tmp_path / 'home', 'cpu', 0, 42, cell['key'])
        expected = dict(old, mmfi_protocol='protocol3',
                        mmfi_setting=cell['setting'],
                        confirmation_profile=p3.PROFILE)
        assert cfg == expected
        assert cfg['model'] == 'hpeli'
        assert cfg['trigger'] == 'md_multicarrier'
        assert cfg['rho'] == (0. if cell['key'] == 'clean' else .1)
        assert cfg['eps'] == .185
        assert cfg['batch_size'] == 32 and cfg['epochs'] == 50
    assert not (tmp_path / 'out').exists()  # pure dry plan


def test_launch_refuses_missing_preflight_and_gpu_smoke(tmp_path):
    plan = p3.build_plan(tmp_path / 'home', tmp_path / 'out', device='cpu')
    p3._verify_saved_plan(plan)
    p3._verify_saved_plan(plan)  # JSON round-trip must preserve the plan
    try:
        p3.launch(plan, jobs=1)
    except ValueError as exc:
        assert 'Missing gate' in str(exc)
    else:
        raise AssertionError('Launch should require completed run gates')


def test_cpu_preflight_checks_real_trigger_and_pose_sample(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    root = home / 'datasets' / 'Compress'
    for subject in ('S01', 'S05'):
        sequence = root / 'E01' / subject / 'A01'
        wifi = sequence / 'wifi-csi'
        wifi.mkdir(parents=True)
        np.save(sequence / 'ground_truth.npy',
                np.ones((1, 17, 3), np.float32))
        np.save(wifi / 'frame001_processed.npy',
                np.full((3, 114, 10), .5, np.float32))
    actions = home / 'actions'
    actions.mkdir()
    np.save(actions / 'data_bend.npy',
            np.random.default_rng(7).normal(
                size=(1, 3, 30, 25, 1)).astype(np.float32))
    monkeypatch.setattr(p3, 'EXPECTED_ACTION_SHA256',
                        p3._sha_file(actions / 'data_bend.npy'))
    monkeypatch.setattr(p3, 'EXPECTED_SEQUENCES',
                        {name: (1, 1) for name in p3.SETTINGS})
    monkeypatch.setattr(p3, '_dataset',
                        lambda cfg, split: MMFIS2(split, cfg['dataset_root']))
    plan = p3.build_plan(home, tmp_path / 'out', device='cpu')
    p3.preflight(plan)
    assert (tmp_path / 'out' / 'preflight.json').is_file()
