"""Frozen three-table orchestration; no benchmark data or GPU training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_a1_three_tables as runner
import run_proposed_ablation as ablation
import run_a1_native_comparison as native
import train_backdoor as trainer


@pytest.fixture
def matrix(tmp_path):
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'new', 'cpu', 4)


def test_39_cells_13_immutable_and_26_new(matrix):
    assert matrix['seeds'] == [42, 43, 44]
    assert len(matrix['cells']) == 39
    assert len({c['method_key'] for c in matrix['cells']}) == 39
    assert sum(c['origin'] == 'fresh' for c in matrix['cells']) == 26
    for seed in runner.SEEDS:
        cells = [c for c in matrix['cells'] if c['seed'] == seed]
        assert [c['key'] for c in cells] == list(runner.KEYS)
        a1 = next(c for c in cells if c['key'] == 'a1')
        assert a1['tables'] == ['T1', 'T2', 'T3']
        assert len(cells) == 13
        assert not any('blended' in c['key'] for c in cells)
    runner._validate_manifest(matrix)


def test_seed42_routes_existing_scientific_victims(matrix):
    for cell in matrix['cells'][:13]:
        folder = Path(cell['ckpt_dir'])
        if cell['key'] in runner.ABLATION_KEYS:
            assert folder.parent.name == 'mmfi_proposed_factorial_ablation_s42_v1'
        elif cell['key'].startswith('rho'):
            assert folder.parent.name == 'mmfi_a1_dose_poison_s42_v1'
        elif cell['key'] == 'clean':
            assert folder.parent.name == 'mmfi_tables_s42_20261009_003044'
        else:
            assert folder.parent.name == 'mmfi_a1_native_comparison_s42_v3'
        assert folder not in Path(cell['artifact_dir']).parents
        assert cell['origin'] == 'reused_seed42'


def test_operator_and_recipe_identical_except_experiment_seed(matrix):
    for cell in matrix['cells']:
        cfg, key, seed = cell['cfg'], cell['key'], cell['seed']
        if key in dict(runner.BASELINE_METHODS):
            expected = native._method_config(Path(matrix['data_home']), 'cpu', 4, key)
            expected['seed'] = seed
            assert cfg == expected
            assert cfg['num_workers'] == (0 if key in runner.STOCHASTIC_CELLS else 4)
        assert cfg['epochs'] == cfg['victim_epochs'] == 50
        assert cfg['optimizer'] == 'sgd' and cfg['lr'] == .001
        assert cfg['batch_size'] == 32 and cfg['momentum'] == .9
        assert cfg['mmfi_split_seed'] == 0 and cfg['mmfi_random_ratio'] == .8
        assert cfg['target_joints'] == [2, 3] and cfg['seed'] == seed
        assert not any(k.startswith('comparison_peak_') for k in cfg)
    for seed in runner.SEEDS:
        configs = ablation._configs(Path(matrix['data_home']), 'cpu', 4, seed, factorial=True)
        cells = {c['key']: c for c in matrix['cells'] if c['seed'] == seed}
        for key, expected in zip(('a1', 'a2', 'a3', 'a4'), configs):
            assert cells[key]['cfg'] == expected
        assert cells['clean']['cfg']['rho'] == 0
        for key, rho in (('rho0p2', .2), ('rho0p4', .4)):
            wanted = dict(configs[0], rho=rho, confirmation_profile='mmfi_a1_dose_poison_v1')
            assert cells[key]['cfg'] == wanted


def test_optional_retrain_all_requires_new_paths(tmp_path):
    plan = runner.build_matrix(tmp_path/'data', tmp_path/'new', retrain_seed42=True)
    assert all(c['origin'] == 'fresh' for c in plan['cells'])
    assert len({c['ckpt_dir'] for c in plan['cells']}) == 39
    runner._validate_manifest(plan)


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_workers_validated(tmp_path, value):
    with pytest.raises(ValueError, match='num_workers'):
        runner.build_matrix(tmp_path, tmp_path/'out', num_workers=value)


@pytest.mark.parametrize('mutation', ['seed', 'operator', 'rho', 'path', 'source', 'tables'])
def test_canonical_plan_rejects_tampering_even_after_rehash(matrix, mutation):
    cell = matrix['cells'][15]
    if mutation == 'seed':
        cell['cfg']['seed'] = 42
    elif mutation == 'operator':
        cell['cfg']['badnets_patch_opacity'] = .2
    elif mutation == 'rho':
        cell['cfg']['rho'] = .4
    elif mutation == 'path':
        cell['ckpt_dir'] += '_other'
    elif mutation == 'source':
        matrix['sources']['train_backdoor.py'] = '0'*64
    else:
        cell['tables'] = ['T4']
    matrix['plan_sha256'] = runner._plan_hash(matrix)
    with pytest.raises(ValueError, match='Scientific|canonical'):
        runner._validate_manifest(matrix)


def test_dry_run_no_data_cuda_or_training(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('Dry run touched data/GPU/training')
    monkeypatch.setattr(runner, '_preflight', forbidden)
    monkeypatch.setattr(runner, '_schedule', forbidden)
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', forbidden)
    args = ['--data-home', str(tmp_path/'absent'), '--outdir', str(tmp_path/'out'),
            '--devices', 'cuda:999', '--dry-run']
    assert runner.main(args) == 0
    assert runner.main(args) == 0  # resume same plan, no duplicate directory
    assert not list(tmp_path.rglob('checkpoint.pt'))
    assert not (tmp_path/'absent').exists()


def test_source_change_rejects_resume(tmp_path, monkeypatch):
    args = ['--data-home', str(tmp_path/'data'), '--outdir', str(tmp_path/'out'), '--dry-run']
    assert runner.main(args) == 0
    original = runner._source_provenance()
    monkeypatch.setattr(runner, '_source_provenance', lambda: dict(original, source='changed'))
    with pytest.raises(ValueError, match='Scientific'):
        runner.main(args)


def test_status_is_readonly_and_no_gpu(matrix, tmp_path, monkeypatch, capsys):
    root = Path(matrix['outdir'])
    root.mkdir()
    runner._atomic_json(root/runner.MANIFEST, matrix)
    path = Path(matrix['cells'][0]['artifact_dir'])
    path.mkdir(parents=True)
    (path/'evaluation.json').write_text('{}')
    monkeypatch.setattr(runner, '_preflight', lambda *a: pytest.fail('status needs no GPU'))
    assert runner.main(['--data-home', matrix['data_home'], '--outdir', str(root), '--status']) == 0
    assert 'Total: 1/39' in capsys.readouterr().out


def _poison_fixture(cfg, n=80):
    count = int(np.floor(n*cfg['rho']))
    rng = np.random.default_rng(cfg['seed'])
    indices = rng.choice(n, count, replace=False)
    doses = rng.uniform(.2, 1., count)
    payload = doses.copy()
    if cfg.get('dose_coupling', 'paired') == 'shuffled':
        np.random.default_rng(np.random.SeedSequence([cfg['seed'], 7043])).shuffle(payload)
    samples = [dict(index=int(i), dose=float(d), payload_dose=float(pd))
               for i,d,pd in zip(indices,doses,payload)]
    plan = [[int(i),float(d)] for i,d in zip(indices,doses)]
    digest = runner._sha_json(plan)
    ratio = cfg.get('clean_label_cover_ratio', cfg.get('wanet_cover_ratio', 0.))
    covers = []
    if ratio:
        available = np.array(sorted(set(range(n))-set(map(int,indices))))
        covers = sorted(map(int, np.random.default_rng(np.random.SeedSequence(
            [cfg['seed'],7044])).choice(available, int(np.floor(n*ratio)), replace=False)))
    poison = dict(samples=samples, n_total=n, n_poison=count, seed=cfg['seed'],
        config_fingerprint=trainer._config_fingerprint(cfg), selection='uniform',
        dose_min=.2, dose_max=1., rho_requested=cfg['rho'], pivot=1,
        target_joints=[2,3], theta_max_deg=40., payload_axis=[0.,0.,1.],
        dose_mode='linear', poison_plan_sha256=digest, n_cover=len(covers),
        cover_ratio=ratio, cover_indices=covers, dose_coupling=cfg.get('dose_coupling','paired'))
    res = dict(n_total=n,n_poison=count,n_cover=len(covers),poison_plan_sha256=digest,
               training_contract='ordinary_erm',attacker_access='data_only')
    return res,poison


@pytest.mark.parametrize('seed,key', [(42,'clean'),(43,'a1'),(44,'a2'),(43,'wanet_source'),
                                     (44,'fiba'),(43,'rho0p4'),(44,'bpp_source')])
def test_exact_poison_sampler_and_native_covers(matrix, monkeypatch, seed, key):
    cfg = next(c['cfg'] for c in matrix['cells'] if c['seed']==seed and c['key']==key)
    res,poison = _poison_fixture(cfg)
    monkeypatch.setitem(runner.EXPECTED_COUNTS,'train',80)
    audit = runner._audit_poison(cfg,res,poison)
    assert audit['n_poison'] == int(np.floor(80*cfg['rho']))
    if poison['samples']:
        poison['samples'][0]['payload_dose'] += .001
        with pytest.raises(ValueError,match='sampler'):
            runner._audit_poison(cfg,res,poison)


def test_cover_identity_not_just_count_is_audited(matrix,monkeypatch):
    cfg = next(c['cfg'] for c in matrix['cells'] if c['key']=='wanet_source')
    res,poison = _poison_fixture(cfg)
    monkeypatch.setitem(runner.EXPECTED_COUNTS,'train',80)
    poison['cover_indices'] = list(reversed(poison['cover_indices']))
    with pytest.raises(ValueError,match='cover'):
        runner._audit_poison(cfg,res,poison)


def test_reused_cell_never_calls_train(matrix,monkeypatch):
    cell = matrix['cells'][1]  # seed42 A1
    calls=[]
    class Lock:
        def __enter__(self): return None
        def __exit__(self,*a): return False
    monkeypatch.setattr(runner,'_cell_lock',lambda *a:Lock())
    monkeypatch.setattr(runner,'_current_inputs',lambda *a:{'current':True})
    monkeypatch.setattr(runner,'_read_completed',lambda *a:{'cfg':cell['cfg']})
    monkeypatch.setattr(runner,'_audit_run',lambda *a:{})
    monkeypatch.setattr(runner,'_atomic_json',lambda *a:None)
    monkeypatch.setattr(trainer,'train',lambda *a,**kw:pytest.fail('reused seed42 retrained'))
    import a1_three_tables_eval as evaluator
    monkeypatch.setattr(evaluator,'evaluate_cell',lambda *a: calls.append(a) or {'key':'a1'})
    assert runner._execute_cell(cell)=={'key':'a1'}
    assert calls[0][0]['verified_inputs']=={'current':True}


def test_export_requires_all_cells_before_reporting(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'_export',lambda *a:pytest.fail('incomplete report'))
    with pytest.raises(ValueError,match='All 39'):
        runner.main(['--data-home',str(tmp_path/'data'),'--outdir',str(tmp_path/'out'),
                     '--export-only'])


def test_output_cannot_overlap_immutable_sources(tmp_path):
    data=tmp_path/'data'
    output=data/'runs/mmfi_proposed_factorial_ablation_s42_v1'
    with pytest.raises(ValueError,match='overlap'):
        runner.main(['--data-home',str(data),'--outdir',str(output),'--dry-run'])
    assert not output.exists()


@pytest.mark.parametrize('options', [['--devices','cuda:0','cuda:0'],
    ['--devices','cuda:-1'], ['--cells','s43__blended'], ['--dry-run','--export-only']])
def test_invalid_cli_prevents_launch(tmp_path,options):
    with pytest.raises(ValueError):
        runner.main(['--data-home',str(tmp_path/'data'),'--dry-run',*options])


def test_scheduler_uses_three_slots_and_only_planned_children(matrix,tmp_path,monkeypatch):
    import subprocess
    processes=[]
    active=[]
    peak=[]
    class Child:
        def __init__(self, command, **kwargs):
            self.command=command
            self.polls=0
            self.pid=500+len(processes)
            key=command[command.index('--_cell')+1]
            self.cell=next(c for c in matrix['cells'] if c['method_key']==key)
            processes.append(self)
            active.append(self)
            peak.append(len(active))
        def poll(self):
            self.polls+=1
            if self.polls==1:
                return None
            if self in active:
                active.remove(self)
                folder=Path(self.cell['artifact_dir'])
                folder.mkdir(parents=True)
                (folder/'evaluation.json').write_text('{}')
            return 0
    monkeypatch.setattr(subprocess,'Popen',Child)
    monkeypatch.setattr(runner.time,'sleep',lambda *a:None)
    selected=[c['method_key'] for c in matrix['cells'][:5]]
    runner._schedule(matrix,tmp_path/'manifest.json',['cuda:0','cuda:1','cuda:2'],selected)
    assert len(processes)==5 and max(peak)==3
    assert [p.cell['method_key'] for p in processes]==selected
    assert [p.cell['cfg']['device'] for p in processes[:3]]==['cuda:0','cuda:1','cuda:2']


def test_failed_child_cleans_only_its_own_other_children(matrix,tmp_path,monkeypatch):
    processes=[]
    class Child:
        def __init__(self,command,**kwargs):
            self.pid=len(processes)
            self.terminated=False
            processes.append(self)
        def poll(self): return 1 if self.pid==0 else None
        def terminate(self): self.terminated=True
        def wait(self,timeout): return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    with pytest.raises(RuntimeError,match='failed'):
        runner._schedule(matrix,tmp_path/'manifest.json',['cuda:0','cuda:1','cuda:2'])
    assert len(processes)==3
    assert not processes[0].terminated
    assert all(p.terminated for p in processes[1:])


def test_preflight_reports_bad_remapped_cuda_id(matrix,monkeypatch):
    def invalid(*a): raise AssertionError('Invalid device id')
    monkeypatch.setattr(runner,'_check_inputs',invalid)
    with pytest.raises(ValueError,match='logical CUDA ID'):
        runner._preflight(matrix,['cuda:3'])


@pytest.mark.parametrize('key',['a1','clean'])
def test_actual_new_cell_50_epoch_route_and_checkpoint_reuse(tmp_path,monkeypatch,key):
    """Tiny CPU routing test, not a full-MM-Fi efficacy experiment."""
    import torch
    import models.factory
    import run_peak_confirmation as confirmation
    import a1_three_tables_eval as evaluator

    class TinyData:
        def __init__(self,split,data_root,**kw):
            self.data_root=str(data_root)
            count=50 if split=='training' else 3
            self.items=[dict(csi=str(Path(data_root)/split/f'{i}.npy'),
                kpt=str(Path(data_root)/split/'pose.npy'),frame_idx=i) for i in range(count)]
            self.raw=np.random.default_rng(9).uniform(.1,.8,(3,114,10)).astype(np.float32)
            self.poses=np.random.default_rng(19).normal(0,.2,(50,1,17,3)).astype(np.float32)
        def __len__(self): return len(self.items)
        def load_raw(self,path): return self.raw.copy()
        def normalize(self,raw): return np.clip(raw,0,1).astype(np.float32)
        def load_pose(self,path,frame_idx): return self.poses[frame_idx].copy()

    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head=torch.nn.Linear(3*114*10,17*3)
        def forward(self,csi):
            return self.head(csi.flatten(1)).reshape(-1,1,17,3),None

    counts={'train':50,'eval':3}
    monkeypatch.setattr(runner,'EXPECTED_COUNTS',counts)
    monkeypatch.setattr(confirmation,'EXPECTED_COUNTS',counts)
    monkeypatch.setattr(evaluator,'COUNTS',counts)
    monkeypatch.setattr(trainer,'MMFI',TinyData)
    monkeypatch.setattr(trainer,'build_model',lambda *a,**kw:TinyVictim())
    monkeypatch.setattr(models.factory,'build_model',lambda *a,**kw:TinyVictim())
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *a:None)
    def dataset(cfg):
        from run_peak_confirmation import _pair_ids
        train=trainer._load_dataset(cfg,'training')
        test=trainer._load_dataset(cfg,'test')
        records={name:dict(n=len(ds),pair_ids_sha256=runner._sha_json(_pair_ids(ds)))
                 for name,ds in [('train',train),('eval',test)]}
        return test,records,_pair_ids(test)
    monkeypatch.setattr(evaluator,'_dataset_record',dataset)
    data=tmp_path/'data'
    action=data/'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action,np.random.default_rng(23).normal(size=(1,3,30,25,1)).astype(np.float32))
    matrix=runner.build_matrix(data,tmp_path/'out','cpu',0)
    cell=next(c for c in matrix['cells'] if c['seed']==43 and c['key']==key)
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        record=runner._execute_cell(cell)
        checkpoint=torch.load(Path(cell['ckpt_dir'])/'checkpoint.pt',weights_only=False)
        assert checkpoint['epoch']==49 and checkpoint['cfg']['seed']==43
        assert (record['metrics']['target_pck_pct'] is None) == (key=='clean')
        assert len(record['dose_rows'])==(0 if key=='clean' else 6)
        assert (Path(cell['artifact_dir'])/'per_sample.npz').is_file()
        old_hash=runner._file_sha(Path(cell['ckpt_dir'])/'checkpoint.pt')
        monkeypatch.setattr(trainer,'build_model',lambda *a,**kw:pytest.fail('retrained'))
        assert runner._execute_cell(cell)==record
        assert runner._file_sha(Path(cell['ckpt_dir'])/'checkpoint.pt')==old_hash
    finally:
        torch.set_num_threads(threads)
