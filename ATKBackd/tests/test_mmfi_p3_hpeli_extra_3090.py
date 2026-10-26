"""Contract checks for the new six-victim HPE-Li P3 extension."""
from pathlib import Path

import run_mmfi_p3_hpeli_extra_3090 as run


def test_six_cells_keep_completed_p3_victim_recipe(tmp_path):
    home = tmp_path / 'data'
    plan = run.build_plan(home, home / 'runs' / 'fresh')
    assert len(plan['cells']) == 6
    assert {row['setting'] for row in plan['cells']} == {'s1', 's2', 's3'}
    assert {row['method'] for row in plan['cells']} == {
        'smooth_lf_csi', 'tsba_adapted'}
    for row in plan['cells']:
        cfg = row['cfg']
        assert cfg['model'] == 'hpeli'
        assert cfg['mmfi_protocol'] == 'protocol3'
        assert cfg['mmfi_setting'] == row['setting']
        assert (cfg['epochs'], cfg['batch_size'], cfg['lr'], cfg['rho']) == (
            50, 32, .001, .1)
        assert cfg['target_joints'] == [2, 3]
        assert cfg['dose_grid'] == [0., .2, .4, .6, .8, 1.]
        assert Path(row['folder']).is_relative_to(home / 'runs' / 'fresh')


def test_s2_training_can_use_official_cross_subject_adapter(tmp_path):
    from mmfi_s2_split import MMFIS2
    home = tmp_path / 'data'
    row = run.cell(run.build_plan(home, home / 'runs' / 'fresh'),
                   's2', 'tsba_adapted')
    engine = run.extra.private_engine(row['cfg'], row['method'])
    engine.MMFI = MMFIS2
    assert engine.MMFI is MMFIS2
    assert engine.build_trigger(row['cfg']).requires_deferred_injection
