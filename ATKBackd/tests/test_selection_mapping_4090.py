"""Synthetic/CPU contract tests; never full MM-Fi efficacy evidence."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('selection_mapping', REPO/'remote_tools/run_selection_mapping_4090.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class Toy:
    def __init__(self, n=20):
        self.items = [dict(csi=str(i), kpt=str(i), frame_idx=0) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def load_pose(self, name, frame_idx):
        rng = np.random.default_rng(int(name))
        return rng.normal(size=(1,17,3))


def arrays():
    gt = np.random.default_rng(42).normal(size=(4,17,3))
    target = np.stack([gt.copy() for _ in range(6)])
    for i, d in enumerate(study.GRID):
        target[i,:,2:4,0] += d
    return gt, target


def test_selection_is_native_and_doses_shared():
    from attack.poison import PoisonedDataset
    base = Toy()
    selection = study.shared_selection(base, 5)
    study.validate_selection(selection, 20, 5)
    native = PoisonedDataset(base, None, rho=.25, seed=42, pivot=1, dataset='mmfi')
    assert list(native.poison_plan) == list(zip(selection['uniform'], selection['doses']))
    for name in ('uniform','diverse'):
        # Tiny fixture uses five indices; production paired() deliberately pins rho=.1.
        cfg = dict(poison_select=name, dose_mode='linear')
        short = study.shared_selection(base, 2)
        pairs = study.paired(base, None, cfg, short)
        assert list(pairs.poison_plan) == list(zip(short[name], short['doses']))


def test_factorial_configs_change_only_two_factors():
    configs = {key:study.configuration(Path('D:/fixture_home'), key) for key in study.CELLS}
    normalized = [{k:v for k,v in c.items() if k not in ('poison_select','dose_mode')} for c in configs.values()]
    assert all(c == normalized[0] for c in normalized)
    assert len(configs) == 6
    for cfg in configs.values():
        assert cfg['rho'] == .1 and cfg['epochs'] == 50 and cfg['seed'] == 42
        assert cfg['num_workers'] == 0 and cfg['trigger'] == 'md_multicarrier'
        assert cfg['eps'] == .185 and cfg['target_joints'] == [2,3]


def test_still_predictor_ndce_one():
    gt, target = arrays()
    result = study.metrics(gt, np.stack([gt]*6), target)
    assert result['summary']['mean_ndce'] == pytest.approx(1)
    assert result['dose_rows'][0]['ndce'] is None
    assert result['summary']['nt_drift_mm'] == 0
    assert result['summary']['mean_dce_mm'] == pytest.approx(600)


def test_perfect_displacement_zero_control_error():
    gt, target = arrays()
    result = study.metrics(gt, target, target)
    assert result['summary']['mean_dce_mm'] == 0
    assert result['summary']['mean_ndce'] == 0
    assert result['summary']['target_pck_pct'] == 100


def test_reversed_displacement_ndce_two():
    gt, target = arrays()
    pred = gt[None]-(target-gt[None])
    assert study.metrics(gt, pred, target)['summary']['mean_ndce'] == pytest.approx(2)


@pytest.mark.parametrize('damage', ['duplicate','out_of_range','dose','uniform_order'])
def test_selection_rejects_tampering(damage):
    selection = study.shared_selection(Toy(), 5)
    if damage == 'duplicate':
        selection['diverse'][1] = selection['diverse'][0]
    elif damage == 'out_of_range':
        selection['diverse'][0] = 20
    elif damage == 'dose':
        selection['doses'][0] += .0001
    else:
        selection['uniform'].reverse()
    with pytest.raises(ValueError):
        study.validate_selection(selection, 20, 5)


def test_metrics_zero_requested_rejected():
    gt, target = arrays()
    target[1] = gt
    with pytest.raises(ValueError, match='zero requested'):
        study.metrics(gt, target, target)


def test_atomic_immutable_write(tmp_path):
    path = tmp_path/'record.json'
    study.write(path, {'a':1}, immutable=True)
    study.write(path, {'a':1}, immutable=True)
    with pytest.raises(ValueError):
        study.write(path, {'a':2}, immutable=True)
    assert study.read(path) == {'a':1}


def test_inspect_does_not_write(tmp_path, capsys):
    root = tmp_path/'runs/mmfi_selection_mapping_s42_v1'
    study.main(['--mode','inspect','--data-home',str(tmp_path/'home'),'--outdir',str(root)])
    assert not root.exists()
    assert 'six fresh' in capsys.readouterr().out


def test_capacity_commands_are_two_disposable_cases(tmp_path):
    commands = study.commands(tmp_path, ['uniform_sqrt','diverse_quad'], smoke=True)
    assert len(commands) == 2
    assert all('_smoke' in command for _,command,_ in commands)


def test_real_process_queue_overlap(tmp_path):
    code = ('import pathlib,time,sys,json; p=pathlib.Path(sys.argv[1]); '
            'start=time.time(); time.sleep(1.3); p.write_text(json.dumps([start,time.time()]))')
    commands = [(str(i),[sys.executable,'-B','-c',code,str(tmp_path/f'{i}.json')],tmp_path/f'{i}.log') for i in range(3)]
    study.run_queue(commands)
    a,b,c = [json.loads((tmp_path/f'{i}.json').read_text()) for i in range(3)]
    assert max(a[0],b[0]) < min(a[1],b[1])
    assert c[0] >= min(a[1],b[1])


def test_queue_failure_drains_peer_no_third_job(tmp_path):
    marker = tmp_path/'peer_done'
    peer = [sys.executable,'-B','-c',f'import time,pathlib; time.sleep(1.5); pathlib.Path({str(marker)!r}).touch()']
    third = tmp_path/'must_not_exist'
    commands = [('fail',[sys.executable,'-B','-c','raise SystemExit(7)'],tmp_path/'fail.log'),
                ('peer',peer,tmp_path/'peer.log'),
                ('pending',[sys.executable,'-B','-c',f'import pathlib; pathlib.Path({str(third)!r}).touch()'],tmp_path/'third.log')]
    with pytest.raises(RuntimeError, match='drained'):
        study.run_queue(commands)
    assert marker.exists() and not third.exists()


def test_real_hpeli_cpu_step():
    import torch
    from models.factory import build_model
    from train_backdoor import _victim_update
    torch.set_num_threads(2)
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114, dataset='mmfi', pretrained=False)
    opt = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    before = next(model.parameters()).detach().clone()
    loss = _victim_update(model, opt, torch.rand(32,3,114,10), torch.rand(32,1,17,3))
    assert np.isfinite(loss) and not torch.equal(before,next(model.parameters()))


def test_checkpoint_binding_rng_and_sgd_guards():
    import copy
    import torch
    value = dict(binding={'cell':'uniform_linear'}, epoch=10,
        model={'weight':torch.ones(2)}, initial_model_sha256='abc',
        optimizer={'param_groups':[dict(lr=.001,momentum=.9,weight_decay=0.)]},
        rng=dict(python=1,numpy=2,torch=3,cuda=[torch.ones(1)]))
    study.check_checkpoint(value, {'cell':'uniform_linear'})
    for mutate in ('epoch','rng','sgd','model','binding'):
        broken = copy.deepcopy(value)
        if mutate == 'epoch':
            broken['epoch'] = 50
        elif mutate == 'rng':
            broken['rng']['cuda'] = []
        elif mutate == 'sgd':
            broken['optimizer']['param_groups'][0]['lr'] = .01
        elif mutate == 'model':
            broken['model']['weight'][0] = float('nan')
        else:
            broken['binding']['cell'] = 'diverse_linear'
        with pytest.raises(ValueError):
            study.check_checkpoint(broken, {'cell':'uniform_linear'})


def test_epoch_rng_resume_exact_cpu():
    import copy
    import random
    import torch
    from train_backdoor import _rng_state, _restore_rng_state, _victim_update
    # Real optimizer/RNG serialization; synthetic linear regressor, not efficacy.
    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3,3)
        def forward(self,x):
            return self.linear(x), None
    torch.manual_seed(42)
    model = Regressor()
    optimizer = torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    x,y = torch.rand(8,17,3),torch.rand(8,17,3)
    _victim_update(model,optimizer,x,y)
    state = copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=_rng_state()))
    expected_rng = (random.random(),np.random.rand(),torch.rand(1))
    _victim_update(model,optimizer,x,y)
    expected = copy.deepcopy(model.state_dict())
    resumed = Regressor()
    resumed_opt = torch.optim.SGD(resumed.parameters(),lr=.001,momentum=.9)
    resumed.load_state_dict(state['model'])
    resumed_opt.load_state_dict(state['optimizer'])
    _restore_rng_state(state['rng'])
    observed_rng = (random.random(),np.random.rand(),torch.rand(1))
    assert expected_rng[0] == observed_rng[0] and expected_rng[1] == observed_rng[1]
    assert torch.equal(expected_rng[2],observed_rng[2])
    _victim_update(resumed,resumed_opt,x,y)
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in expected.items())
