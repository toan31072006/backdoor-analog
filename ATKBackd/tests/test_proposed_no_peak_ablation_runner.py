"""Local planning/operator/report tests for the explicitly unbounded profile.

No benchmark training is launched. Synthetic operator frames and cache metadata
are fixtures, never scientific results or imported historical evaluations.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_proposed_ablation as runner
import train_backdoor as trainer


KEYS = ['full', 'w_o_coupling', 'w_o_auxiliary']
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def forbidden(*args, **kwargs):
    pytest.fail('A planning/report test attempted training or unexpected data/CUDA access')


def args(data_home, output=None, *extra):
    result = ['--data-home', str(data_home), '--devices', 'cpu',
              '--num-workers', '0', '--no-peak', '--dry-run']
    if output is not None:
        result.extend(['--outdir', str(output)])
    return result + list(extra)


def stored(output):
    return json.loads((output / 'proposed_ablation.resolved.json').read_text(encoding='utf-8'))


@pytest.fixture
def sources(monkeypatch):
    hashes = {'fixture.py': 'a' * 64}
    monkeypatch.setattr(runner, '_source_provenance', lambda: dict(hashes))
    return hashes


@pytest.fixture
def matrix(tmp_path, sources):
    return runner.build_matrix(tmp_path / 'missing_data', tmp_path / 'new',
                               'cpu', 0, no_peak=True)


def test_new_profile_has_only_three_ordered_unbounded_arms(matrix):
    assert runner.NO_PEAK_PROFILE == 'mmfi_proposed_no_peak_ablation_v1'
    assert matrix['confirmation_profile'] == runner.NO_PEAK_PROFILE
    assert matrix['status'] == 'FULL_MMFI_ABLATION_SINGLE_SEED'
    assert matrix['seed'] == 42 and matrix['fresh_results_only'] is True
    assert [c['method_key'] for c in matrix['cells']] == KEYS
    assert [c['cfg']['trigger'] for c in matrix['cells']] == [
        'md_multicarrier', 'md_multicarrier', 'micro_dropper']
    assert [c['cfg']['dose_coupling'] for c in matrix['cells']] == [
        'paired', 'shuffled', 'paired']
    assert matrix['metrics_contract']['peak_bound_required_keys'] == []
    assert matrix['metrics_contract']['full_split_counts'] == {'train': 133056, 'eval': 33264}
    for cell in matrix['cells']:
        assert cell['cfg']['confirmation_profile'] == runner.NO_PEAK_PROFILE
        assert cell['cfg']['epochs'] == cell['cfg']['victim_epochs'] == 50
        assert cell['cfg']['rho'] == .4 and cell['cfg']['eps'] == .185
        assert cell['cfg']['seed'] == 42 and cell['cfg']['measure_dose_control'] is True
        assert cell['cfg']['optimizer'] == 'sgd' and cell['cfg']['lr'] == .001
        assert cell['dependencies'] == [] and cell['tables'] == []
        assert not any(k.startswith('draft_') for k in cell['cfg'])
    runner._validate_manifest(matrix)


def test_new_full_equals_previous_no_peak_diagnostic_without_other_science_changes(
        matrix, tmp_path, sources):
    legacy = runner.build_matrix(tmp_path / 'missing_data', tmp_path / 'legacy', 'cpu', 0)
    old_no_peak = next(c['cfg'] for c in legacy['cells'] if c['method_key'] == 'w_o_peak')
    old_auxiliary = next(c['cfg'] for c in legacy['cells'] if c['method_key'] == 'w_o_auxiliary')
    new_full, no_coupling, no_auxiliary = [c['cfg'] for c in matrix['cells']]
    assert new_full == dict(old_no_peak, confirmation_profile=runner.NO_PEAK_PROFILE)
    assert no_auxiliary == dict(old_auxiliary, confirmation_profile=runner.NO_PEAK_PROFILE)
    assert no_coupling == dict(new_full, dose_coupling='shuffled')
    assert all(c['cfg']['method_carrier_seed'] == 42 for c in matrix['cells'][:2])


def test_legacy_default_is_still_four_arms_with_its_original_peak_requirements(tmp_path, sources):
    legacy = runner.build_matrix(tmp_path / 'missing_data', tmp_path / 'legacy', 'cpu', 0)
    assert legacy['confirmation_profile'] == runner.PROFILE
    assert [c['method_key'] for c in legacy['cells']] == KEYS + ['w_o_peak']
    assert [c['cfg']['trigger'] for c in legacy['cells']] == [
        'md_multicarrier_peak_matched', 'md_multicarrier_peak_matched',
        'micro_dropper', 'md_multicarrier']
    assert legacy['metrics_contract']['peak_bound_required_keys'] == KEYS
    runner._validate_manifest(legacy)


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_future_replications_do_not_change_frozen_carrier_seed(tmp_path, sources, seed):
    matrix = runner.build_matrix(tmp_path / 'data', tmp_path / f's{seed}',
                                 'cpu', 0, seed=seed, no_peak=True)
    assert matrix['seed'] == seed
    assert all(c['cfg']['seed'] == seed for c in matrix['cells'])
    assert all(c['cfg']['method_carrier_seed'] == 42 for c in matrix['cells'][:2])
    runner._validate_manifest(matrix)


@pytest.mark.parametrize('change', ['profile', 'order', 'extra_arm', 'missing_arm',
    'projection', 'peak_contract', 'epochs', 'rho', 'coupling', 'carrier_seed'])
def test_manifest_rejects_profile_arm_projection_and_science_tampering(matrix, change):
    altered = copy.deepcopy(matrix)
    if change == 'profile':
        altered['confirmation_profile'] = runner.PROFILE
    elif change == 'order':
        altered['cells'].reverse()
    elif change == 'extra_arm':
        altered['cells'].append(copy.deepcopy(altered['cells'][0]))
    elif change == 'missing_arm':
        altered['cells'].pop()
    elif change == 'projection':
        altered['cells'][0]['cfg']['trigger'] = 'md_multicarrier_peak_matched'
    elif change == 'peak_contract':
        altered['metrics_contract']['peak_bound_required_keys'] = ['full']
    else:
        key, value = dict(epochs=('epochs', 15), rho=('rho', .1),
            coupling=('dose_coupling', 'shuffled'),
            carrier_seed=('method_carrier_seed', 43))[change]
        altered['cells'][0]['cfg'][key] = value
    altered['plan_sha256'] = runner._plan_fingerprint(altered)
    with pytest.raises(ValueError):
        runner._validate_manifest(altered)


def test_no_peak_dry_run_is_inert_and_defaults_to_its_own_output_folder(
        tmp_path, sources, monkeypatch):
    monkeypatch.setattr(runner, '_check_inputs', forbidden)
    monkeypatch.setattr(runner, 'run_matrix', forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', forbidden)
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', forbidden)
    data = tmp_path / 'data'
    assert runner.main(args(data, None, '--devices', 'cuda:999')) == 0
    output = data / 'runs/mmfi_proposed_no_peak_ablation_s42_v1'
    assert stored(output)['confirmation_profile'] == runner.NO_PEAK_PROFILE
    assert [c['method_key'] for c in stored(output)['cells']] == KEYS
    assert not (data / 'datasets').exists()
    assert not (data / 'runs/mmfi_proposed_ablation_s42_v1').exists()
    assert {p.name for p in output.iterdir()} == {'run.lock', 'proposed_ablation.resolved.json'}


def test_real_cli_no_peak_dry_run_requires_no_data_or_gpu(tmp_path):
    output = tmp_path / 'cli'
    result = subprocess.run([sys.executable, str(Path(runner.__file__).resolve()),
        *args(tmp_path / 'missing', output, '--devices', 'cuda:999')],
        capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [c['method_key'] for c in stored(output)['cells']] == KEYS
    assert stored(output)['confirmation_profile'] == runner.NO_PEAK_PROFILE
    assert not (tmp_path / 'missing').exists()


def test_existing_peak_manifest_cannot_be_reused_as_no_peak_results(tmp_path, sources):
    data, output = tmp_path / 'data', tmp_path / 'existing'
    legacy_args = [a for a in args(data, output) if a != '--no-peak']
    assert runner.main(legacy_args) == 0
    before = (output / 'proposed_ablation.resolved.json').read_bytes()
    with pytest.raises(ValueError):
        runner.main(args(data, output))
    assert (output / 'proposed_ablation.resolved.json').read_bytes() == before


def test_no_peak_manifest_cannot_be_reused_as_peak_results(tmp_path, sources):
    data, output = tmp_path / 'data', tmp_path / 'existing'
    assert runner.main(args(data, output)) == 0
    before = (output / 'proposed_ablation.resolved.json').read_bytes()
    with pytest.raises(ValueError):
        runner.main([a for a in args(data, output) if a != '--no-peak'])
    assert (output / 'proposed_ablation.resolved.json').read_bytes() == before


def test_unchanged_no_peak_plan_resumes_metadata_without_new_training(tmp_path, sources):
    data, output = tmp_path / 'data', tmp_path / 'resume'
    assert runner.main(args(data, output)) == 0
    previous = stored(output)['plan_sha256']
    assert runner.main(args(data, output, '--devices', 'cuda:7', '--num-workers', '8')) == 0
    assert stored(output)['plan_sha256'] == previous
    assert all(c['cfg']['device'] == 'cuda:7' and c['cfg']['num_workers'] == 8
               for c in stored(output)['cells'])


def test_no_peak_input_record_uses_the_new_profile(matrix, tmp_path, monkeypatch):
    class TinySplit:
        def __init__(self, split, n):
            self.data_root = str(tmp_path / 'frames')
            self.items = [dict(csi=str(Path(self.data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(self.data_root) / split / 'pose.npy'), frame_idx=i)
                for i in range(n)]

        def __len__(self):
            return len(self.items)

    cell = copy.deepcopy(matrix['cells'][0])
    action = tmp_path / 'action.npy'
    action.write_bytes(b'synthetic reference metadata only')
    cell['cfg']['action_npy'] = str(action)
    datasets = {'training': TinySplit('training', 5), 'test': TinySplit('test', 3)}
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 5, 'eval': 3})
    monkeypatch.setattr(trainer, '_load_dataset', lambda cfg, split: datasets[split])
    monkeypatch.setattr(trainer.np, 'load', forbidden)
    record = runner._collect_input_record(cell)
    assert record['profile'] == runner.NO_PEAK_PROFILE
    assert record['config_fingerprint'] == trainer._config_fingerprint(cell['cfg'])


def test_new_operators_are_native_unbounded_multicarrier_not_peak_projection(matrix, tmp_path):
    from attack.method_drafts import MDMultiCarrierTrigger, MDPeakMatchedMultiCarrierTrigger
    from attack.trigger import MicroDopplerTrigger

    action = tmp_path / 'reference.npy'
    np.save(action, np.random.default_rng(12).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    for cell in matrix['cells']:
        operator = trainer.build_trigger(dict(cell['cfg'], action_npy=str(action)))
        if cell['method_key'] == 'w_o_auxiliary':
            assert type(operator) is MicroDopplerTrigger
        else:
            assert type(operator) is MDMultiCarrierTrigger
            assert not isinstance(operator, MDPeakMatchedMultiCarrierTrigger)


@pytest.fixture
def synthetic_report(matrix, monkeypatch):
    """Only cache-shaped numbers; does not read or pretend to run a victim."""
    audit = dict(inputs={}, poison_plan_sha256={}, action_file_sha256='c' * 64)
    for cell in matrix['cells']:
        key = cell['method_key']
        audit['inputs'][key] = dict(train=dict(pair_ids_sha256='a' * 64),
                                    eval=dict(pair_ids_sha256='b' * 64))
        audit['poison_plan_sha256'][key] = 'd' * 64

    def cache(cell):
        result = dict(training_contract='ordinary_erm', attacker_access='data_only',
            dose_coupling=cell['cfg']['dose_coupling'], n_total=133056, n_poison=53222,
            poison_plan_sha256='d' * 64, dose_grid=GRID, reference_dose=1.0,
            clean_mpjpe=.18, clean_pampjpe=.10,
            tmpjpe=[.12, .13, .14, .15, .16, .17],
            clean_to_target_tmpjpe=[.12, .16, .21, .27, .33, .40],
            dose_control_error=[0.0, .02, .03, .04, .05, .06])
        result.update({f'clean_pck@{t:.1f}': .55 for t in [.5, .4, .3, .2, .1]})
        result['asr_curve'] = [dict(dose_control_error_mean=value)
                               for value in result['dose_control_error']]
        return result, dict(cfg_fingerprint=trainer._config_fingerprint(cell['cfg']))

    monkeypatch.setattr(runner, 'audit_common_inputs', lambda m: audit)
    monkeypatch.setattr(runner, '_verified_cache', cache)
    return matrix, audit


def test_summary_keeps_no_peak_profile_and_clean_only_pck_semantics(synthetic_report):
    matrix, _ = synthetic_report
    report = runner.build_summary(matrix)
    assert report['status'] == 'proposed_ablation_complete'
    assert report['profile'] == runner.NO_PEAK_PROFILE
    assert [r['method_key'] for r in report['rows']] == KEYS
    assert len(report['dose_response']) == 18
    assert all(r['mean_dose_control_error_mm'] == pytest.approx(40.0)
               and r['clean_pck_0.2_pct'] == pytest.approx(55.0) for r in report['rows'])


def test_export_labels_pck_as_clean_and_does_not_call_training(
        synthetic_report, tmp_path, monkeypatch):
    matrix, _ = synthetic_report
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(runner, 'build_distortion', lambda *a: dict(
        profile=runner.NO_PEAK_PROFILE, plan_sha256=matrix['plan_sha256'], rows=[]))
    output = tmp_path / 'report'
    output.mkdir()
    report = runner.export_summary(matrix, output)
    assert report['profile'] == runner.NO_PEAK_PROFILE
    exported = json.loads((output / 'ablation_summary.json').read_text())
    assert exported['profile'] == runner.NO_PEAK_PROFILE
    assert exported['rows'][0]['clean_pck_0.2_pct'] == pytest.approx(55.0)
    assert json.loads((output / 'dose_response.json').read_text())['profile'] == runner.NO_PEAK_PROFILE
    text = (output / 'ablation_summary.md').read_text().lower()
    assert 'clean pck@0.2' in text and 'untriggered' in text
    assert ('relative' in text or 'reference joint distance' in text) and '20 mm' in text
    assert 'four fresh victims' not in text


def test_no_peak_distortion_allows_native_excess_as_a_diagnostic(
        matrix, tmp_path, monkeypatch):
    """Unbounded does not mean silently projecting back to Original."""
    class TinyCSI:
        def __init__(self):
            self.data_root = str(tmp_path / 'frames')
            self.items = [dict(csi=str(Path(self.data_root) / f'csi_{i}.npy'),
                kpt=str(Path(self.data_root) / 'pose.npy'), frame_idx=i) for i in range(3)]

        def __len__(self):
            return len(self.items)

        def load_raw(self, path):
            return np.full((3, 114, 10), .5, dtype=np.float32)

        @staticmethod
        def normalize(raw):
            return np.clip(raw, 0, 1).astype(np.float32)

    class FixedOperator:
        def __init__(self, amplitude):
            self.amplitude = amplitude

        def inject(self, raw, dose, eps):
            return (raw + self.amplitude * dose).astype(np.float32)

    record = dict(profile=runner.NO_PEAK_PROFILE)
    action = tmp_path / 'action.npy'
    action.write_bytes(b'fixed synthetic reference')
    for cell in matrix['cells']:
        cell['cfg']['action_npy'] = str(action)
    audit = dict(inputs={'full': record}, action_file_sha256=runner._file_sha(action))
    monkeypatch.setattr(runner, '_collect_input_record', lambda cell: record)
    monkeypatch.setattr(trainer, '_load_dataset', lambda *a: TinyCSI())
    monkeypatch.setattr(trainer, 'build_trigger', lambda cfg: FixedOperator(
        .05 if cfg['trigger'] == 'micro_dropper' else .20))
    monkeypatch.setattr(runner, 'trigger_state_sha256', lambda operator:
        ('a' if operator.amplitude == .05 else 'b') * 64)
    monkeypatch.setattr(trainer, 'train', forbidden)
    report = runner.build_distortion(matrix, audit)
    assert report['profile'] == runner.NO_PEAK_PROFILE
    assert len(report['rows']) == 18
    assert all(row['peak_bound_required'] is False for row in report['rows'])
    full = next(row for row in report['rows'] if row['method_key'] == 'full' and row['dose'] == 1.0)
    assert full['paired_linf_violations'] == 3
    assert full['linf'] > full['original_linf']


def test_real_three_cell_cpu_route_with_synthetic_data_keeps_actual_50_epochs(
        tmp_path, monkeypatch):
    """Code integration only: tiny frames/model, no MM-Fi benchmark claims."""
    import torch

    class TinyMMFI:
        def __init__(self, split, data_root, **kwargs):
            self.data_root = str(data_root)
            count = 10 if split == 'training' else 3
            self.items = [dict(csi=str(Path(data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(data_root) / split / 'pose.npy'), frame_idx=i,
                name=f'{split}_{i}') for i in range(count)]
            self.raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(19).normal(0, .2, (10, 1, 17, 3)).astype(np.float32)

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
            self.head = torch.nn.Linear(3 * 114 * 10, 17 * 3)

        def forward(self, csi):
            return self.head(csi.flatten(1)).reshape(-1, 1, 17, 3), None

    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 10, 'eval': 3})
    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *a, **k: TinyVictim())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a: None)
    monkeypatch.setattr(torch.backends.cudnn, 'is_available', lambda: False)
    data, output = tmp_path / 'synthetic_data', tmp_path / 'synthetic_cpu_route'
    action = data / 'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(data, output, 'cpu', 0, no_peak=True, distortion_samples=3)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for cell in matrix['cells']:
            runner._run_cell(cell)
            folder = Path(cell['ckpt_dir'])
            checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False, map_location='cpu')
            assert checkpoint['epoch'] == 49 and checkpoint['cfg']['epochs'] == 50
            assert checkpoint['cfg']['confirmation_profile'] == runner.NO_PEAK_PROFILE
            assert json.loads((folder / 'ablation_inputs.json').read_text())['profile'] == runner.NO_PEAK_PROFILE
            cache = json.loads((folder / 'eval_cache.json').read_text())
            assert cache['trained_epochs'] == 50
            assert cache['res']['dose_grid'] == GRID and len(cache['res']['dose_control_error']) == 6
        report = runner.export_summary(matrix, output)
        assert report['profile'] == runner.NO_PEAK_PROFILE
        assert [r['method_key'] for r in report['rows']] == KEYS
        assert len(report['dose_response']) == 18
        assert len(set(report['audit']['poison_plan_sha256'].values())) == 1
        assert all(r['train_samples'] == 10 and r['eval_samples'] == 3 for r in report['rows'])
        distortion = json.loads((output / 'input_distortion.json').read_text())
        assert distortion['profile'] == runner.NO_PEAK_PROFILE
        assert len(distortion['rows']) == 18
        assert all(r['peak_bound_required'] is False for r in distortion['rows'])
        # Original remains a reference for diagnostics, not an imposed ceiling.
        full_d1 = next(r for r in distortion['rows'] if r['method_key'] == 'full' and r['dose'] == 1)
        assert full_d1['paired_linf_violations'] > 0
        monkeypatch.setattr(trainer, 'build_model', forbidden)
        runner._run_cell(matrix['cells'][0])
        assert runner.build_summary(matrix)['rows'] == report['rows']
    finally:
        torch.set_num_threads(previous_threads)
