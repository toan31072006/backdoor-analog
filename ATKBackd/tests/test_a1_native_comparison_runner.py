"""Native baselines/A1 reference isolation; no real MM-Fi data or CUDA."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_a1_native_comparison as runner
import run_proposed_ablation as ablation
import train_backdoor as trainer


KEYS = ['badnets', 'wanet_source', 'ftrojan', 'fiba', 'sig_source', 'bpp_source']


def _forbidden(*args, **kwargs):
    pytest.fail('Planning accessed dataset/reference/CUDA/training')


@pytest.fixture
def sources(monkeypatch):
    snapshot = {'fixture.py': 'a' * 64}
    monkeypatch.setattr(runner, '_source_provenance', lambda: dict(snapshot))
    return snapshot


@pytest.fixture
def matrix(tmp_path, sources):
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'out', 'cpu', 4)


def test_six_baselines_only_a1_never_scheduled(matrix):
    assert [cell['method_key'] for cell in matrix['cells']] == KEYS
    assert all(cell['dependencies'] == [] for cell in matrix['cells'])
    assert 'blended' not in KEYS and not any(c['method_key'] in ('a1', 'full', 'proposed') for c in matrix['cells'])
    assert matrix['metrics_contract']['reuse_a1'] is True
    assert matrix['metrics_contract']['train_a1'] is False
    assert matrix['metrics_contract']['comparison_scope'] == 'selected_native_csi_pose_adaptations'
    assert 'no superiority over it is claimed' in matrix['metrics_contract']['excluded_methods']['blended']
    assert not any(key.startswith('blended') for key in matrix['sources']['references']['backdoorbench'])
    runner._validate_manifest(matrix)


def test_full_a1_scientific_protocol_and_native_operators(matrix):
    for cell in matrix['cells']:
        cfg = cell['cfg']
        for key, expected in dict(seed=42, rho=0.1, epochs=50, victim_epochs=50,
                model='hpeli', optimizer='sgd', lr=0.001, momentum=0.9,
                weight_decay=0.0, batch_size=32, pretrained=False,
                data_parallel=False, pivot=1, target_joints=[2, 3],
                theta_max_deg=40.0, payload_axis=[0.0, 0.0, 1.0],
                dose_min=0.2, dose_max=1.0, dose_coupling='paired',
                dose_grid=runner.GRID, victim_loss='mpjpe',
                training_protocol='ordinary_erm', attacker_access='data_only',
                mmfi_protocol='protocol1', mmfi_setting='s1',
                mmfi_random_ratio=0.8, mmfi_split_seed=0,
                eps=0.185, measure_dose_control=True).items():
            assert cfg[key] == expected, (cell['method_key'], key)
        assert not any(k.startswith('comparison_peak_') for k in cfg)
        assert not any(k.startswith('method_') for k in cfg)
        assert 'confirmation_profile' not in cfg
        assert 'loader_persistent_workers' not in cfg
    assert matrix['metrics_contract']['full_split_counts'] == dict(train=133056, eval=33264)
    assert matrix['metrics_contract']['matches_noise_budget'] is False
    assert matrix['metrics_contract']['shared_peak_ceiling'] is False


def test_source_knobs_and_cover_counts_are_not_optimized(matrix):
    cfgs = {c['method_key']: c['cfg'] for c in matrix['cells']}
    assert cfgs['badnets']['badnets_patch_subcarriers'] == cfgs['badnets']['badnets_patch_packets'] == 3
    assert cfgs['badnets']['badnets_patch_opacity'] == 1.0
    assert cfgs['wanet_source']['wanet_grid_size'] == 4
    assert cfgs['wanet_source']['wanet_strength'] == 0.5
    assert cfgs['wanet_source']['wanet_cover_ratio'] == 0.2
    assert cfgs['fiba']['fiba_alpha'] == 0.15 and cfgs['fiba']['fiba_beta'] == 0.1
    assert cfgs['fiba']['clean_label_cover_ratio'] == 0.1
    assert cfgs['ftrojan']['ftrojan_magnitude_255'] == 20.0
    assert cfgs['sig_source']['sig_delta_255'] == 20.0
    assert cfgs['sig_source']['sig_frequency'] == 6
    assert cfgs['bpp_source']['bpp_levels'] == 32
    assert cfgs['bpp_source']['bpp_dithering'] is False
    assert cfgs['bpp_source']['clean_label_cover_ratio'] == 0.2
    for key, cfg in cfgs.items():
        assert cfg['num_workers'] == (0 if key in runner.STOCHASTIC_CELLS else 4)


def test_only_attack_specific_a1_keys_are_removed(matrix, tmp_path):
    a1 = ablation._configs(tmp_path / 'data', 'cpu', 4, 42, factorial=True)[0]
    baseline = matrix['cells'][0]['cfg']
    for key, value in a1.items():
        if key not in ('confirmation_profile', 'trigger') and not key.startswith('method_'):
            assert baseline[key] == value


def test_configs_are_independent(matrix):
    matrix['cells'][0]['cfg']['target_joints'].append(10)
    assert all(c['cfg']['target_joints'] == [2, 3] for c in matrix['cells'][1:])
    assert matrix['metrics_contract']['target_joints'] == [2, 3]


@pytest.mark.parametrize('workers', [-1, True, 1.5])
def test_invalid_worker_counts_rejected(tmp_path, sources, workers):
    with pytest.raises(ValueError, match='num_workers'):
        runner.build_matrix(tmp_path / 'data', tmp_path / 'out', num_workers=workers)


def test_plan_never_reads_data_or_reference(tmp_path, sources, monkeypatch):
    import a1_frozen_reference
    monkeypatch.setattr(trainer, '_load_dataset', _forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', _forbidden)
    monkeypatch.setattr(trainer, 'train', _forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', _forbidden)
    monkeypatch.setattr(a1_frozen_reference, 'load_a1_reference', _forbidden)
    runner.build_matrix(tmp_path / 'missing', tmp_path / 'output', 'cuda:77', 4,
                        a1_reference=tmp_path / 'missing_reference')


@pytest.mark.parametrize('edit', ['rho', 'target', 'operator', 'profile', 'projection', 'drop', 'reorder', 'metrics'])
def test_rehashed_scientific_tampering_rejected(matrix, edit):
    if edit == 'rho':
        matrix['cells'][0]['cfg']['rho'] = 0.4
    elif edit == 'target':
        matrix['cells'][0]['cfg']['target_joints'] = [5, 6]
    elif edit == 'operator':
        matrix['cells'][0]['cfg']['badnets_patch_packets'] = 8
    elif edit == 'profile':
        matrix['cells'][0]['cfg']['traditional_comparison_profile'] = 'other'
    elif edit == 'projection':
        matrix['cells'][0]['cfg']['comparison_peak_budget'] = 'original_postclip_linf_v1'
    elif edit == 'drop':
        matrix['cells'].pop()
    elif edit == 'reorder':
        matrix['cells'].reverse()
    else:
        matrix['metrics_contract']['rho'] = 0.2
    matrix['plan_sha256'] = runner._plan_fingerprint(matrix)
    with pytest.raises(ValueError):
        runner._validate_manifest(matrix)


def test_source_drift_is_rejected_even_if_rehashed(matrix, monkeypatch):
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture.py': 'b' * 64})
    with pytest.raises(ValueError, match='source changed'):
        runner._validate_manifest(matrix)


def test_dry_run_inert_and_resume_same_plan(tmp_path, sources, monkeypatch):
    import a1_frozen_reference
    monkeypatch.setattr(a1_frozen_reference, 'load_a1_reference', _forbidden)
    monkeypatch.setattr(runner, '_check_inputs', _forbidden)
    monkeypatch.setattr(runner, 'run_matrix', _forbidden)
    argv = ['--data-home', str(tmp_path / 'data'), '--devices', 'cuda:0', 'cuda:1', 'cuda:2', '--dry-run']
    assert runner.main(argv) == 0
    folder = tmp_path / 'data' / 'runs' / runner.FOLDER
    plan = json.loads((folder / runner.MANIFEST).read_text(encoding='utf-8'))
    assert [c['method_key'] for c in plan['cells']] == KEYS
    assert not any(p.is_dir() for p in folder.iterdir())
    assert runner.main(argv) == 0


def test_fresh_does_not_overwrite_existing_manifest(tmp_path, sources):
    argv = ['--data-home', str(tmp_path / 'data'), '--dry-run']
    runner.main(argv)
    with pytest.raises(FileExistsError, match='non-empty'):
        runner.main(argv + ['--fresh'])


@pytest.mark.parametrize('devices', [['cuda:0', 'cuda:0'], ['cuda:-1'], ['0']])
def test_bad_runtime_devices_rejected(tmp_path, sources, devices):
    with pytest.raises(ValueError, match='distinct logical'):
        runner.main(['--data-home', str(tmp_path), '--devices', *devices, '--dry-run'])


def test_worker_executes_only_requested_planned_baseline(matrix, tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(matrix), encoding='utf-8')
    seen = []
    monkeypatch.setattr(runner, '_run_cell', lambda c: seen.append(c['method_key']))
    assert runner.main(['--_cell', 'sig_source', '--_manifest', str(path)]) == 0
    assert seen == ['sig_source']
    with pytest.raises(ValueError, match='unplanned'):
        runner.main(['--_cell', 'full', '--_manifest', str(path)])


def test_worker_source_guard_precedes_training(matrix, tmp_path, monkeypatch):
    matrix['sources']['source_sha256']['fixture.py'] = 'b' * 64
    matrix['plan_sha256'] = runner._plan_fingerprint(matrix)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(matrix), encoding='utf-8')
    monkeypatch.setattr(runner, '_run_cell', _forbidden)
    with pytest.raises(ValueError, match='source changed'):
        runner.main(['--_cell', 'badnets', '--_manifest', str(path)])


def test_main_preflights_a1_and_schedules_only_baselines(tmp_path, sources, monkeypatch):
    import a1_frozen_reference
    events = []
    monkeypatch.setattr(a1_frozen_reference, 'load_a1_reference', lambda path: events.append('reference') or {})
    monkeypatch.setattr(a1_frozen_reference, 'validate_a1_reference', lambda ref, matrix, audit: events.append(('validate', audit)))
    monkeypatch.setattr(a1_frozen_reference, 'validate_a1_input_record', lambda ref, record: events.append('identity'))
    monkeypatch.setattr(runner, '_collect_input_record', lambda cell: {})
    monkeypatch.setattr(runner, '_check_inputs', lambda matrix, devices: events.append('inputs'))
    def run(matrix, path, devices, requested, worker_script):
        events.append(('run', [c['method_key'] for c in matrix['cells']], devices, requested))
        assert worker_script.name == 'run_a1_native_comparison.py'
    monkeypatch.setattr(runner, 'run_matrix', run)
    monkeypatch.setattr(runner, '_complete', lambda c: False)
    assert runner.main(['--data-home', str(tmp_path), '--devices', 'cpu', '--cells', 'badnets']) == 0
    assert events[:4] == ['reference', ('validate', None), 'inputs', 'identity']
    assert events[-1] == ('run', KEYS, ['cpu'], ['badnets'])


def test_invalid_a1_reference_stops_before_gpu_training(tmp_path, sources, monkeypatch):
    import a1_frozen_reference
    monkeypatch.setattr(a1_frozen_reference, 'load_a1_reference', lambda path: {})
    def reject(*args):
        raise ValueError('A1 scientific contract differs')
    monkeypatch.setattr(a1_frozen_reference, 'validate_a1_reference', reject)
    monkeypatch.setattr(runner, '_check_inputs', _forbidden)
    monkeypatch.setattr(runner, 'run_matrix', _forbidden)
    with pytest.raises(ValueError, match='A1 scientific'):
        runner.main(['--data-home', str(tmp_path), '--devices', 'cpu'])


def test_selected_cells_do_not_export_incomplete_report(tmp_path, sources, monkeypatch):
    import a1_frozen_reference
    monkeypatch.setattr(a1_frozen_reference, 'load_a1_reference', lambda path: {})
    monkeypatch.setattr(a1_frozen_reference, 'validate_a1_reference', lambda *args: None)
    monkeypatch.setattr(a1_frozen_reference, 'validate_a1_input_record', lambda *args: None)
    monkeypatch.setattr(runner, '_collect_input_record', lambda cell: {})
    monkeypatch.setattr(runner, '_check_inputs', lambda *args: None)
    monkeypatch.setattr(runner, 'run_matrix', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, '_complete', lambda c: False)
    monkeypatch.setattr(runner, 'export_summary', _forbidden)
    assert runner.main(['--data-home', str(tmp_path), '--devices', 'cpu', '--cells', 'fiba']) == 0


def test_changed_dataset_identity_stops_before_baseline_training(tmp_path, sources, monkeypatch):
    import a1_frozen_reference
    monkeypatch.setattr(a1_frozen_reference, 'load_a1_reference', lambda path: {})
    monkeypatch.setattr(a1_frozen_reference, 'validate_a1_reference', lambda *args: None)
    monkeypatch.setattr(runner, '_check_inputs', lambda *args: None)
    monkeypatch.setattr(runner, '_collect_input_record', lambda cell: {'train': 'changed'})
    def reject(reference, record):
        raise ValueError('TRAIN identity differs from A1')
    monkeypatch.setattr(a1_frozen_reference, 'validate_a1_input_record', reject)
    monkeypatch.setattr(runner, 'run_matrix', _forbidden)
    with pytest.raises(ValueError, match='TRAIN identity'):
        runner.main(['--data-home', str(tmp_path), '--devices', 'cpu'])


def test_limitations_do_not_claim_equal_noise_or_general_superiority():
    notice = '\n'.join(runner._limitations())
    assert 'does not match realized Linf, L2' in notice
    assert 'no claim of superiority over all backdoor attacks' in notice
    assert 'Target-PCK' in notice and 'workers=0' in notice
    assert 'Blended is omitted' in notice and 'previous results are retained' in notice
    assert 'aliases negative f=4' in notice and 'with replacement' in notice


def test_markdown_has_only_agreed_three_main_metrics(tmp_path, matrix, monkeypatch):
    rows = [dict(method_key=key, method=label, t1_mpjpe_mm=150.0,
                 clean_mpjpe_mm=174.0, clean_pampjpe_mm=104.0)
            for key, label in runner.METHODS]
    rows.append(dict(rows[0], method_key='a1', method='Proposed A1 (reused)'))
    report = dict(status='a1_native_comparison_complete', rows=rows,
                  dose_response=[], limitations=runner._limitations())
    monkeypatch.setattr(runner, 'build_summary', lambda plan: report)
    runner.export_summary(matrix, tmp_path)
    text = (tmp_path / 'a1_native_comparison.md').read_text(encoding='utf-8')
    assert '| Method | T-MPJPE@1 (mm) | Clean MPJPE (mm) | Clean PA-MPJPE (mm) |' in text
    assert '| BppAttack-CSI (author-code quantization variant) |' in text and '| Proposed A1 (reused) |' in text
    assert '| Blended' not in text
    assert 'Target-PCK is not implemented' in text
    with (tmp_path / 'a1_native_comparison.csv').open(encoding='utf-8', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 7


def test_summary_only_alias():
    assert runner.parse_args(['--summary-only']).export_only is True
    assert runner.parse_args(['--export-only']).export_only is True


def test_three_slots_queue_six_baselines_without_a1(matrix, tmp_path, monkeypatch):
    import run_mmfi_tables as scheduler
    launches = []
    class CompletedChild:
        def __init__(self, command, **kwargs):
            self.pid = len(launches) + 100
            launches.append(command)
        def poll(self):
            return 0
    monkeypatch.setattr(scheduler.subprocess, 'Popen', CompletedChild)
    monkeypatch.setattr(scheduler, '_complete', lambda c: True)
    monkeypatch.setattr(scheduler.time, 'sleep', lambda seconds: None)
    runner.run_matrix(matrix, tmp_path / 'manifest.json',
        ['cuda:0', 'cuda:1', 'cuda:2'], worker_script=Path(runner.__file__))
    assert [cmd[cmd.index('--_cell') + 1] for cmd in launches] == KEYS
    assert [c['cfg']['device'] for c in matrix['cells']] == ['cuda:0', 'cuda:1', 'cuda:2'] * 2
    assert all(Path(cmd[2]).name == 'run_a1_native_comparison.py' for cmd in launches)


def test_actual_six_baseline_cpu_training_and_audited_reference_export(tmp_path, sources, monkeypatch):
    """Only integration plumbing: 50 synthetic frames, NOT MM-Fi results."""
    import a1_frozen_reference as frozen
    import run_traditional_comparison as traditional
    import run_peak_confirmation as confirmation
    from mmfi_tables import metric_row, dose_rows

    model_initializations = []
    class TinyMMFI:
        def __init__(self, split, data_root, **kwargs):
            assert split in ('training', 'test')
            self.split, self.data_root = split, str(data_root)
            n = 50 if split == 'training' else 3
            self.items = [dict(csi=str(Path(data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(data_root) / split / 'ground_truth.npy'), frame_idx=i)
                for i in range(n)]
            self.raw = np.random.default_rng(7).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(11).normal(0, .2, (n, 1, 17, 3)).astype(np.float32)
        def __len__(self):
            return len(self.items)
        def load_raw(self, path):
            return self.raw.copy()
        @staticmethod
        def normalize(raw):
            return np.clip(raw, 0, 1).astype(np.float32)
        def load_pose(self, path, frame_idx):
            return self.poses[frame_idx].copy()

    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 51)
            model_initializations.append({k: v.clone() for k, v in self.state_dict().items()})
        def forward(self, csi):
            return self.head(csi.mean((-2, -1))).reshape(-1, 1, 17, 3), None

    counts = dict(train=50, eval=3)
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(traditional, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(confirmation, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *args, **kwargs: TinyVictim())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *args: None)
    monkeypatch.setattr(torch.backends.cudnn, 'is_available', lambda: False)
    data, output = tmp_path / 'synthetic', tmp_path / 'comparison'
    action = data / 'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(17).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(data, output, 'cpu', 0)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        poison_plans = []
        for cell in matrix['cells']:
            runner._run_cell(cell)
            folder = Path(cell['ckpt_dir'])
            checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
            assert checkpoint['epoch'] == 49
            assert checkpoint['cfg']['measure_dose_control'] is True
            poison = json.loads((folder / 'poison_manifest.json').read_text(encoding='utf-8'))
            poison_plans.append(poison['samples'])
            expected_covers = 10 if cell['method_key'] in ('wanet_source', 'bpp_source') else 5 if cell['method_key'] == 'fiba' else 0
            assert poison.get('n_cover', 0) == expected_covers
            if expected_covers:
                assert checkpoint['trigger']['cover_draws'] == expected_covers * 50
                assert 'trigger_optimizer' not in checkpoint
        assert len(model_initializations) == 6
        assert all(plan == poison_plans[0] for plan in poison_plans)
        for initialized in model_initializations[1:]:
            for key in initialized:
                torch.testing.assert_close(initialized[key], model_initializations[0][key], rtol=0, atol=0)
        audit = runner.audit_common_inputs(matrix)
        cell = matrix['cells'][0]
        result, provenance = runner._verified_cache(cell)
        row = metric_row(cell, result)
        row.update(t1_mpjpe_mm=row.pop('tmpjpe_mm'), n_poison=5, n_cover=0,
            train_samples=50, eval_samples=3, epochs=50, rho=.1,
            mean_dose_control_error_mm=sum(result['dose_control_error'][1:]) * 1000 / 5,
            mean_positive_dose_tmpjpe_mm=sum(result['tmpjpe'][1:]) * 1000 / 5,
            cfg_fingerprint=provenance['cfg_fingerprint'],
            train_pair_ids_sha256=audit['inputs']['badnets']['train']['pair_ids_sha256'],
            eval_pair_ids_sha256=audit['inputs']['badnets']['eval']['pair_ids_sha256'],
            poison_plan_sha256=audit['poison_plan_sha256']['badnets'])
        series, _ = dose_rows(cell, result)
        reference = dict(row=row, dose_response=[dict(r, dose_control_error_mm=1000 * error)
            for r, error in zip(series, result['dose_control_error'])],
            cfg=ablation._configs(data, 'cpu', 0, 42, factorial=True)[0],
            audit=dict(audit['inputs']['badnets'],
                       poison_plan_sha256=audit['poison_plan_sha256']['badnets']),
            provenance={'synthetic_reference': True}, source_sha256={})
        # Exercise the real cross-config/input/poison validator; only the disk
        # loader is replaced because frozen production A1 bytes are not fixtures.
        monkeypatch.setattr(frozen, 'load_a1_reference', lambda path: copy.deepcopy(reference))
        report = runner.export_summary(matrix, output)
        assert len(report['rows']) == 7 and len(report['dose_response']) == 42
        assert report['rows'][-1]['result_origin'] == 'frozen_ablation_reference'
        assert report['rows'][-1]['method_key'] == 'a1'
        assert len({r['poison_plan_sha256'] for r in report['rows']}) == 1
        assert not (output / 'a1').exists() and not (output / 'proposed').exists()
        for name in ('a1_native_comparison.csv', 'a1_native_comparison.json',
                     'a1_native_comparison.md', 'dose_response.csv', 'dose_response.json'):
            assert (output / name).is_file()
        monkeypatch.setattr(trainer, 'build_model', lambda *args: pytest.fail('finished baseline retrained'))
        runner._run_cell(matrix['cells'][0])
        assert runner.build_summary(matrix)['rows'] == report['rows']
        path = output / 'badnets/poison_manifest.json'
        poison = json.loads(path.read_text(encoding='utf-8'))
        poison['samples'][0]['dose'] = 1.0
        path.write_text(json.dumps(poison), encoding='utf-8')
        with pytest.raises(ValueError, match='poison'):
            runner.build_summary(matrix)
    finally:
        torch.set_num_threads(previous_threads)
