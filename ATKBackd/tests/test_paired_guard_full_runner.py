"""Single frozen-key full confirmation; no official data or CUDA required."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paired_guard_full as runner
import run_paired_guard_drafts as drafts
import train_backdoor as trainer
import frozen_full_contract as contract
import learned_carrier_fit as fitter
from attack.trigger import build_trigger_by_name
from attack.learned_carrier import TrainableCarrier, write_artifact


def forbidden(*args, **kwargs):
    pytest.fail('Must not open data, fit, import victim weights, or probe CUDA')


@pytest.fixture
def source(tmp_path):
    matrix = drafts.build_matrix(tmp_path / 'data', tmp_path / 'screen', 'cpu', 0)
    cell = matrix['cells'][2]
    cfg = copy.deepcopy(cell['cfg'])
    folder = Path(cell['ckpt_dir'])
    folder.mkdir(parents=True)
    runner._atomic_json(folder.parent / 'paired_guard.resolved.json', matrix)
    action = Path(cfg['action_npy'])
    action.parent.mkdir(parents=True)
    # Moving skeletons give a nondegenerate source operator for real injection.
    np.save(action, np.random.default_rng(17).normal(size=(1, 3, 30, 25, 1)).astype(np.float32) * .01)
    fit = dict(variant='paired_guard', recipe_sha256=cell['recipe_sha256'],
        official_test_loaded=False, external_draft_holdout_loaded=False,
        no_initialization_weight_transfer=True, utility_gate_required=True,
        selected_utility_gate_passed=False, no_eligible_candidate=True,
        action_sha256=runner._file_sha(action))
    runner._atomic_json(folder / 'fitting.json', fit)
    original = build_trigger_by_name('micro_dropper', cfg)
    module = TrainableCarrier(original, cfg, 'paired_guard')
    digest = write_artifact(folder / 'learned_trigger.json', module, cell['recipe_sha256'], provenance=fit)
    cfg.update(lc_recipe_sha256=cell['recipe_sha256'],
        lc_artifact_path=str(folder / 'learned_trigger.json'), lc_artifact_sha256=digest,
        lc_fitting_sha256=runner._file_sha(folder / 'fitting.json'))
    runner._atomic_json(folder / 'prepared_cfg.json', dict(schema=1, cfg=cfg,
        recipe_sha256=cell['recipe_sha256'], cfg_fingerprint=trainer._config_fingerprint(cfg),
        artifact_sha256=digest, fitting_sha256=cfg['lc_fitting_sha256'],
        action_file_sha256=runner._file_sha(action)))
    # Deliberately invalid old victim files: confirmation must never read them.
    (folder / 'checkpoint.pt').write_bytes(b'NEVER IMPORT OLD VICTIM')
    (folder / 'results.json').write_bytes(b'NEVER IMPORT OLD METRICS')
    return folder


def _res(cfg):
    result = dict(clean_mpjpe=.177763, clean_pampjpe=.104351, dose_grid=runner.GRID,
        tmpjpe=[.1, .2, .25, .3, .33, .358170],
        clean_to_target_tmpjpe=[.1, .3, .35, .4, .45, .5], reference_dose=1.,
        target_joints=[2, 3], training_contract='ordinary_erm', attacker_access='data_only',
        config_fingerprint=trainer._config_fingerprint(cfg), seed=42,
        n_total=10, n_poison=1, n_cover=0)
    for threshold, value in zip((.5, .4, .3, .2, .1), (.8836, .8183, .7124, .5414, .2300)):
        result[f'clean_pck@{threshold:.1f}'] = value
    return result


def _args(source, out, *extra):
    return ['--source-cell', str(source), '--outdir', str(out), '--device', 'cpu', '--num-workers', '0', *extra]


def test_full_plan_freezes_same_key_changes_only_victim_scope(source, tmp_path):
    data = runner.read_frozen_source(source)
    plan = runner.build_plan(data, tmp_path / 'full', 'cpu', 0)
    cfg = plan['cfg']
    assert cfg['epochs'] == cfg['victim_epochs'] == 50 and cfg['seed'] == 42
    assert cfg['rho'] == .1 and cfg['method_draft'] is False
    assert not any(k.startswith('draft_') for k in cfg)
    assert cfg['lc_artifact_sha256'] == data['cfg']['lc_artifact_sha256']
    assert cfg['confirmation_source']['no_eligible_candidate'] is True
    assert plan['metrics_contract']['source_trigger_refitted'] is False
    assert plan['metrics_contract']['all_train_frames'] is True
    assert plan['metrics_contract']['all_test_frames'] is True
    assert len(plan['metrics_contract']['dose_grid']) == 6


def test_runtime_ids_workers_do_not_change_plan_but_science_does(source, tmp_path):
    data = runner.read_frozen_source(source)
    a = runner.build_plan(data, tmp_path / 'full', 'cpu', 0)
    b = runner.build_plan(data, tmp_path / 'full', 'cuda:0', 4)
    assert a['plan_sha256'] == b['plan_sha256']
    b['cfg']['epochs'] = 15
    assert runner._plan_sha(b) != a['plan_sha256']


def test_snapshot_preserves_exact_bytes_never_copies_old_victim(source, tmp_path):
    data = runner.read_frozen_source(source)
    out = tmp_path / 'full'
    runner._snapshot_source(data, out)
    runner._snapshot_source(data, out)
    assert {p.name for p in (out / 'source_frozen').iterdir()} == set(runner.SOURCE_FILES)
    for name in runner.SOURCE_FILES:
        assert (out / 'source_frozen' / name).read_bytes() == (source / name).read_bytes()
    plan = runner.build_plan(data, out, 'cpu', 0)
    contract.validate_frozen_full_source_files(plan['cfg'])


def test_snapshot_tamper_is_not_overwritten(source, tmp_path):
    data = runner.read_frozen_source(source)
    out = tmp_path / 'full'
    runner._snapshot_source(data, out)
    target = out / 'source_frozen' / 'fitting.json'
    target.write_bytes(b'TAMPER')
    with pytest.raises(ValueError, match='snapshot changed'):
        runner._snapshot_source(data, out)
    assert target.read_bytes() == b'TAMPER'


@pytest.mark.parametrize('name', runner.SOURCE_FILES)
def test_source_artifact_tamper_rejected_before_full_training(source, name):
    target = source / name
    if name == 'prepared_cfg.json':
        prepared = json.loads(target.read_text())
        prepared['cfg']['lr'] = .01
        runner._atomic_json(target, prepared)
    else:
        target.write_bytes(target.read_bytes() + b' ')
    with pytest.raises(ValueError):
        runner.read_frozen_source(source)


def test_dry_run_has_no_data_gpu_training_fitting_access(source, tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(trainer, 'MMFI', forbidden)
    monkeypatch.setattr(fitter, 'prepare_learned_cell', forbidden)
    monkeypatch.setattr(runner, '_validate_single_device', forbidden)
    out = tmp_path / 'full'
    assert runner.main(_args(source, out, '--dry-run')) == 0
    assert (out / 'paired_guard_full.resolved.json').exists()
    assert not (out / 'source_frozen').exists()
    assert not (out / 'checkpoint.pt').exists()


def test_only_one_fresh_full_call_then_verified_cache_export_and_resume(source, tmp_path, monkeypatch):
    calls = []
    def fake_train(cfg, ckpt_dir=None):
        contract.validate_frozen_full_source_files(cfg)
        calls.append(copy.deepcopy(cfg))
        assert not (Path(ckpt_dir) / 'checkpoint.pt').exists()
        result = _res(cfg)
        trainer._save_cached_result(ckpt_dir, cfg, result)
        print('synthetic full victim completion')
        return None, result
    monkeypatch.setattr(trainer, 'train', fake_train)
    monkeypatch.setattr(fitter, 'prepare_learned_cell', forbidden)
    out = tmp_path / 'full'
    assert runner.main(_args(source, out)) == 0
    assert len(calls) == 1 and calls[0]['epochs'] == 50
    report = json.loads((out / 'main_metrics.json').read_text())
    row = report['row']
    assert row['tmpjpe_mm'] == pytest.approx(358.170)
    assert row['clean_pck_0.1_pct'] == pytest.approx(23.)
    assert row['selected_utility_gate_passed'] is False and row['no_eligible_candidate'] is True
    assert len(json.loads((out / 'dose_response.json').read_text())['rows']) == 6
    assert (out / 'console.log').read_text().count('synthetic full victim completion') == 1
    assert runner.main(_args(source, out)) == 0
    # Resume dispatches only the same full config; trainer itself owns caching.
    assert len(calls) == 2 and calls[0] == calls[1]


def test_full_training_result_cache_is_checked_before_export(source, tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, 'train', lambda *a, **k: (None, {}))
    out = tmp_path / 'full'
    with pytest.raises(ValueError, match='missing valid completed cache'):
        runner.main(_args(source, out))
    assert not (out / 'main_metrics.json').exists()


@pytest.mark.parametrize('target', ['same', 'parent', 'nested'])
def test_output_must_not_overlap_historical_source(source, target):
    out = {'same': source, 'parent': source.parent, 'nested': source / 'new'}[target]
    with pytest.raises(ValueError, match='separate, non-nested'):
        runner.main(_args(source, out, '--dry-run'))


def test_fresh_refuses_existing_results_without_deletion(source, tmp_path):
    out = tmp_path / 'full'
    out.mkdir()
    sentinel = out / 'sentinel'
    sentinel.write_bytes(b'KEEP')
    with pytest.raises(FileExistsError):
        runner.main(_args(source, out, '--fresh'))
    assert sentinel.read_bytes() == b'KEEP'


def test_changed_source_code_plan_fails_closed(source, tmp_path):
    out = tmp_path / 'full'
    runner.main(_args(source, out, '--dry-run'))
    path = out / 'paired_guard_full.resolved.json'
    plan = json.loads(path.read_text())
    plan['source_sha256']['train_backdoor.py'] = 'a' * 64
    runner._atomic_json(path, plan)
    with pytest.raises(ValueError, match='plan/source/code changed'):
        runner.main(_args(source, out, '--dry-run'))


@pytest.mark.parametrize('device', ['cuda:3:0', 'cuda:-1', 'cuda', 'cpu:0'])
def test_invalid_device_fails_before_source_read(tmp_path, device, monkeypatch):
    monkeypatch.setattr(runner, 'read_frozen_source', forbidden)
    with pytest.raises(ValueError, match='logical cuda'):
        runner.main(['--source-cell', 'absent', '--outdir', str(tmp_path), '--device', device])


def test_launcher_cli_matches_single_arm_parser():
    script = (Path(__file__).resolve().parents[2] / 'remote_linux/12_run_paired_guard_full.sh').read_text()
    assert '--source-cell' in script and '--data-home' in script
    assert 'run_paired_guard_full.py' in script
    parsed = runner.parse_args(['--data-home', 'data', '--source-cell', 'source', '--outdir', 'out', '--device', 'cuda:0'])
    assert parsed.device == 'cuda:0'
    assert not any(word in script for word in ('pkill', 'rm -', 'git push', 'export CUDA_VISIBLE_DEVICES'))


def test_single_device_preflight_explains_mask_without_scheduler_flags(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    runner._validate_single_device('cuda:0')
    with pytest.raises(ValueError) as error:
        runner._validate_single_device('cuda:3')
    assert '--device cuda:0' in str(error.value)
    assert '--devices' not in str(error.value)
    assert 'CUDA_VISIBLE_DEVICES' in str(error.value)


def test_cpu_preflight_does_not_probe_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    runner._validate_single_device('cpu')


def test_real_tiny_full_train_eval_freeze_cache_and_resume(source, tmp_path, monkeypatch):
    """Run the real 50-epoch trainer on tiny synthetic uncapped split parents."""
    split_calls, updates = [], []
    class TinyMMFI:
        def __init__(self, **kwargs):
            split_calls.append(kwargs['split'])
            self.items = [dict(csi=str(i), kpt=str(i), frame_idx=i)
                          for i in range(20 if kwargs['split'] == 'training' else 7)]
            self.pose = np.random.default_rng(3).normal(size=(1, 17, 3)).astype(np.float32) * .15
        def __len__(self):
            return len(self.items)
        def load_raw(self, path):
            return np.full((3, 114, 10), .4 + int(path) * .001, dtype=np.float32)
        def normalize(self, value):
            return np.asarray(value, dtype=np.float32)
        def load_pose(self, path, frame_idx):
            return self.pose.copy()
    class TinyHPE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.randn(1, 17, 3) * .1)
            self.weight = torch.nn.Parameter(torch.ones(1, 17, 3) * .01)
        def forward(self, x):
            return self.bias.unsqueeze(0) + x.mean((1, 2, 3))[:, None, None, None] * self.weight.unsqueeze(0), None
    real_update = trainer._victim_update
    def update(*args):
        updates.append(1)
        return real_update(*args)
    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *a, **k: TinyHPE())
    monkeypatch.setattr(trainer, '_victim_update', update)
    monkeypatch.setattr(fitter, 'prepare_learned_cell', forbidden)
    out = tmp_path / 'full'
    assert runner.main(_args(source, out)) == 0
    assert split_calls == ['training', 'test']
    assert len(updates) == 50
    saved = json.loads((out / 'results.json').read_text())
    assert saved['res']['n_total'] == 20 and saved['res']['n_poison'] == 2
    assert all(np.isfinite(saved['res'][name]) for name in ('clean_mpjpe', 'clean_pampjpe'))
    assert len(saved['res']['tmpjpe']) == 6
    checkpoint = torch.load(out / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 49
    assert runner.main(_args(source, out)) == 0
    assert split_calls == ['training', 'test'] and len(updates) == 50
