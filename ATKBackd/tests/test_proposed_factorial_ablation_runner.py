"""Fixed-rho, unbounded 2x2 Proposed ablation; synthetic fixtures only.

No MM-Fi benchmark is launched. The tiny CPU integration tests routing, not
attack efficacy; all four fixtures still execute the actual 50-epoch budget.
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


KEYS = ['full', 'w_o_coupling', 'w_o_auxiliary', 'w_o_both']
TRIGGERS = ['md_multicarrier', 'md_multicarrier', 'micro_dropper', 'micro_dropper']
COUPLINGS = ['paired', 'shuffled', 'paired', 'shuffled']
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def forbidden(*args, **kwargs):
    pytest.fail('Planning or a synthetic metadata audit attempted data/CUDA/training access')


def args(data_home, output=None, *extra):
    result = ['--data-home', str(data_home), '--devices', 'cpu', '--num-workers', '0',
              '--factorial', '--dry-run']
    if output is not None:
        result.extend(['--outdir', str(output)])
    return result + list(extra)


def stored(output):
    return json.loads((output / 'proposed_ablation.resolved.json').read_text(encoding='utf-8'))


@pytest.fixture
def sources(monkeypatch):
    values = {'fixture.py': 'a' * 64}
    monkeypatch.setattr(runner, '_source_provenance', lambda: dict(values))
    return values


@pytest.fixture
def matrix(tmp_path, sources):
    return runner.build_matrix(tmp_path / 'missing_data', tmp_path / 'factorial',
                               'cpu', 0, factorial=True)


def test_factorial_is_exact_four_ordered_cells_at_rho_point_one(matrix):
    assert runner.FACTORIAL_PROFILE == 'mmfi_proposed_factorial_ablation_v1'
    assert matrix['confirmation_profile'] == runner.FACTORIAL_PROFILE
    assert matrix['status'] == 'FULL_MMFI_ABLATION_SINGLE_SEED'
    assert matrix['dataset'] == 'mmfi' and matrix['seed'] == 42
    assert matrix['fresh_results_only'] is True
    assert [c['method_key'] for c in matrix['cells']] == KEYS
    assert [c['cfg']['trigger'] for c in matrix['cells']] == TRIGGERS
    assert [c['cfg']['dose_coupling'] for c in matrix['cells']] == COUPLINGS
    contract = matrix['metrics_contract']
    assert contract['rho'] == .1 and contract['peak_bound_required_keys'] == []
    assert contract['full_split_counts'] == {'train': 133056, 'eval': 33264}
    assert contract['dose_grid'] == GRID and contract['seeds'] == [42]
    for cell in matrix['cells']:
        cfg = cell['cfg']
        assert cfg['confirmation_profile'] == runner.FACTORIAL_PROFILE
        assert cfg['rho'] == .1 and cfg['seed'] == 42 and cfg['eps'] == .185
        assert cfg['epochs'] == cfg['victim_epochs'] == 50
        assert cfg['optimizer'] == 'sgd' and cfg['lr'] == .001
        assert cfg['batch_size'] == 32 and cfg['measure_dose_control'] is True
        assert cfg['dose_grid'] == GRID and cfg['dose_min'] == .2 and cfg['dose_max'] == 1.0
        assert cfg['target_joints'] == [2, 3] and cfg['pivot'] == 1
        assert not any(k.startswith('draft_') or 'subset' in k for k in cfg)
        assert cell['dependencies'] == [] and cell['tables'] == []
    runner._validate_manifest(matrix)


def test_two_by_two_changes_only_coupling_and_carrier(matrix):
    full, shuffled, single, both = [c['cfg'] for c in matrix['cells']]
    assert shuffled == dict(full, dose_coupling='shuffled')
    assert both == dict(single, dose_coupling='shuffled')
    ordinary_full = {k: v for k, v in full.items() if not k.startswith('method_') and k != 'trigger'}
    ordinary_single = {k: v for k, v in single.items() if not k.startswith('method_') and k != 'trigger'}
    assert ordinary_full == ordinary_single
    assert full['method_carrier_seed'] == shuffled['method_carrier_seed'] == 42
    assert full['method_carrier_sub_mode'] == shuffled['method_carrier_sub_mode'] == 3
    assert full['method_carrier_time_mode'] == shuffled['method_carrier_time_mode'] == 1


def test_factorial_full_is_previous_native_full_with_only_profile_and_rho_changed(
        matrix, tmp_path, sources):
    previous = runner.build_matrix(tmp_path / 'missing_data', tmp_path / 'old',
                                   'cpu', 0, no_peak=True)
    for new, old in zip(matrix['cells'][:3], previous['cells']):
        assert new['cfg'] == dict(old['cfg'], rho=.1,
                                  confirmation_profile=runner.FACTORIAL_PROFILE)
    assert all(c['cfg']['rho'] == .4 for c in previous['cells'])
    assert previous['confirmation_profile'] == runner.NO_PEAK_PROFILE


def test_factorial_alone_and_combined_with_no_peak_resolve_the_same_plan(tmp_path, sources):
    data, output = tmp_path / 'data', tmp_path / 'same'
    implicit = runner.build_matrix(data, output, 'cpu', 0, factorial=True)
    explicit = runner.build_matrix(data, output, 'cpu', 0, factorial=True, no_peak=True)
    assert implicit['plan_sha256'] == explicit['plan_sha256']
    assert runner.main(args(data, output)) == 0
    before = stored(output)['plan_sha256']
    assert runner.main(args(data, output, '--no-peak')) == 0
    assert stored(output)['plan_sha256'] == before


@pytest.mark.parametrize('change', ['rho', 'epochs', 'projection', 'coupling',
    'peak_contract', 'profile', 'order', 'missing_both', 'duplicate', 'carrier_seed'])
def test_factorial_manifest_rejects_editable_hash_tampering(matrix, change):
    changed = copy.deepcopy(matrix)
    if change == 'rho':
        changed['cells'][0]['cfg']['rho'] = .4
    elif change == 'epochs':
        changed['cells'][0]['cfg']['epochs'] = 15
    elif change == 'projection':
        changed['cells'][0]['cfg']['trigger'] = 'md_multicarrier_peak_matched'
    elif change == 'coupling':
        changed['cells'][3]['cfg']['dose_coupling'] = 'paired'
    elif change == 'peak_contract':
        changed['metrics_contract']['peak_bound_required_keys'] = ['full']
    elif change == 'profile':
        changed['confirmation_profile'] = runner.NO_PEAK_PROFILE
    elif change == 'order':
        changed['cells'].reverse()
    elif change == 'missing_both':
        changed['cells'].pop()
    elif change == 'duplicate':
        changed['cells'].append(copy.deepcopy(changed['cells'][0]))
    else:
        changed['cells'][0]['cfg']['method_carrier_seed'] = 43
    changed['plan_sha256'] = runner._plan_fingerprint(changed)
    with pytest.raises(ValueError):
        runner._validate_manifest(changed)


def test_cli_factorial_dry_run_is_inert_and_has_separate_default_folder(
        tmp_path, sources, monkeypatch):
    monkeypatch.setattr(runner, '_check_inputs', forbidden)
    monkeypatch.setattr(runner, 'run_matrix', forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', forbidden)
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', forbidden)
    data = tmp_path / 'data'
    assert runner.main(args(data, None, '--devices', 'cuda:999')) == 0
    output = data / 'runs/mmfi_proposed_factorial_ablation_s42_v1'
    assert stored(output)['confirmation_profile'] == runner.FACTORIAL_PROFILE
    assert [c['method_key'] for c in stored(output)['cells']] == KEYS
    assert not (data / 'datasets').exists()
    assert not (data / 'runs/mmfi_proposed_no_peak_ablation_s42_v1').exists()


def test_real_cli_factorial_dry_run_needs_no_data_or_gpu(tmp_path):
    output = tmp_path / 'cli'
    result = subprocess.run([sys.executable, str(Path(runner.__file__).resolve()),
        *args(tmp_path / 'missing', output, '--devices', 'cuda:999')],
        capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [c['method_key'] for c in stored(output)['cells']] == KEYS
    assert stored(output)['metrics_contract']['rho'] == .1
    assert not (tmp_path / 'missing').exists()


@pytest.mark.parametrize('old_flag', [None, '--no-peak'])
def test_factorial_cannot_import_prior_profile_manifest_or_results(tmp_path, sources, old_flag):
    data, output = tmp_path / 'data', tmp_path / 'existing'
    old_args = [a for a in args(data, output) if a != '--factorial']
    if old_flag:
        old_args.append(old_flag)
    assert runner.main(old_args) == 0
    before = (output / 'proposed_ablation.resolved.json').read_bytes()
    with pytest.raises(ValueError):
        runner.main(args(data, output))
    assert (output / 'proposed_ablation.resolved.json').read_bytes() == before


@pytest.fixture
def audited_matrix(tmp_path, sources, monkeypatch):
    """Exact poison RNG metadata, fifty fake TRAIN identities, no array reads."""
    from attack.poison import PoisonedDataset

    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 50, 'eval': 3})
    matrix = runner.build_matrix(tmp_path / 'data', tmp_path / 'audited',
                                 'cpu', 0, factorial=True)

    class Base:
        items = [dict(csi=f'train/csi_{i}.npy', kpt='train/pose.npy', frame_idx=i)
                 for i in range(50)]

        def __len__(self):
            return len(self.items)

    for cell in matrix['cells']:
        cfg = cell['cfg']
        folder = Path(cell['ckpt_dir'])
        folder.mkdir(parents=True)
        record = dict(schema=1, profile=runner.FACTORIAL_PROFILE,
            config_fingerprint=trainer._config_fingerprint(cfg),
            action_file_sha256='c' * 64,
            train=dict(n=50, pair_ids_sha256='a' * 64),
            eval=dict(n=3, pair_ids_sha256='b' * 64))
        (folder / 'ablation_inputs.json').write_text(json.dumps(record), encoding='utf-8')
        poison = PoisonedDataset(Base(), object(), rho=.1, dose_min=.2,
            dose_max=1.0, eps=.185, pivot=1, theta_max_deg=40.0,
            seed=42, dataset='mmfi', dose_coupling=cfg['dose_coupling']).manifest()
        poison['config_fingerprint'] = trainer._config_fingerprint(cfg)
        (folder / 'poison_manifest.json').write_text(json.dumps(poison), encoding='utf-8')
    return matrix


def test_factorial_audit_shares_indices_and_doses_but_intervenes_on_payload_pairing(audited_matrix):
    audit = runner.audit_common_inputs(audited_matrix)
    assert list(audit['inputs']) == KEYS
    assert len(set(audit['poison_plan_sha256'].values())) == 1
    assignments = audit['payload_assignment_sha256']
    assert assignments['full'] == assignments['w_o_auxiliary']
    assert assignments['w_o_coupling'] == assignments['w_o_both']
    assert assignments['full'] != assignments['w_o_both']


@pytest.mark.parametrize('change', ['rho', 'paired_payload', 'marginal', 'count', 'input_profile'])
def test_factorial_poison_audit_refuses_wrong_rho_or_wrong_payload_intervention(audited_matrix, change):
    folder = Path(audited_matrix['cells'][3]['ckpt_dir'])
    if change == 'input_profile':
        path = folder / 'ablation_inputs.json'
        data = json.loads(path.read_text())
        data['profile'] = runner.NO_PEAK_PROFILE
    else:
        path = folder / 'poison_manifest.json'
        data = json.loads(path.read_text())
        if change == 'rho':
            data['rho_requested'] = .4
        elif change == 'paired_payload':
            for sample in data['samples']:
                sample['payload_dose'] = sample['dose']
        elif change == 'marginal':
            data['samples'][0]['payload_dose'] = .23
        else:
            data['n_poison'] = 20
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        runner.audit_common_inputs(audited_matrix)


def test_factorial_uses_native_operators_without_any_peak_wrapper(matrix, tmp_path):
    from attack.method_drafts import MDMultiCarrierTrigger, MDPeakMatchedMultiCarrierTrigger
    from attack.trigger import MicroDopplerTrigger

    action = tmp_path / 'reference.npy'
    np.save(action, np.random.default_rng(12).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    for cell in matrix['cells']:
        trigger = trainer.build_trigger(dict(cell['cfg'], action_npy=str(action)))
        if cell['cfg']['trigger'] == 'micro_dropper':
            assert type(trigger) is MicroDopplerTrigger
        else:
            assert type(trigger) is MDMultiCarrierTrigger
            assert not isinstance(trigger, MDPeakMatchedMultiCarrierTrigger)


def test_real_four_cell_factorial_cpu_route_executes_50_epochs_at_rho_point_one(tmp_path, monkeypatch):
    """Only code integration: fifty synthetic TRAIN frames, three TEST frames."""
    import torch

    class TinyMMFI:
        def __init__(self, split, data_root, **kwargs):
            self.data_root = str(data_root)
            count = 50 if split == 'training' else 3
            self.items = [dict(csi=str(Path(data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(data_root) / split / 'pose.npy'), frame_idx=i,
                name=f'{split}_{i}') for i in range(count)]
            self.raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(19).normal(0, .2, (50, 1, 17, 3)).astype(np.float32)

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

    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 50, 'eval': 3})
    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *a, **k: TinyVictim())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a: None)
    monkeypatch.setattr(torch.backends.cudnn, 'is_available', lambda: False)
    data, output = tmp_path / 'synthetic_data', tmp_path / 'synthetic_factorial'
    action = data / 'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(data, output, 'cpu', 0, factorial=True, distortion_samples=3)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for cell in matrix['cells']:
            runner._run_cell(cell)
            folder = Path(cell['ckpt_dir'])
            checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False, map_location='cpu')
            assert checkpoint['epoch'] == 49 and checkpoint['cfg']['epochs'] == 50
            assert checkpoint['cfg']['confirmation_profile'] == runner.FACTORIAL_PROFILE
            assert checkpoint['cfg']['rho'] == .1
            record = json.loads((folder / 'ablation_inputs.json').read_text())
            assert record['profile'] == runner.FACTORIAL_PROFILE
            poison = json.loads((folder / 'poison_manifest.json').read_text())
            assert poison['n_total'] == 50 and poison['n_poison'] == 5
            assert poison['rho_requested'] == .1
        report = runner.export_summary(matrix, output)
        assert report['status'] == 'proposed_ablation_complete'
        assert report['profile'] == runner.FACTORIAL_PROFILE
        assert [r['method_key'] for r in report['rows']] == KEYS
        assert all(r['rho'] == .1 and r['epochs'] == 50 for r in report['rows'])
        assert len(report['dose_response']) == 24
        audit = report['audit']
        assert len(set(audit['poison_plan_sha256'].values())) == 1
        assignments = audit['payload_assignment_sha256']
        assert assignments['full'] == assignments['w_o_auxiliary']
        assert assignments['w_o_coupling'] == assignments['w_o_both']
        assert assignments['full'] != assignments['w_o_both']
        distortion = json.loads((output / 'input_distortion.json').read_text())
        assert distortion['profile'] == runner.FACTORIAL_PROFILE
        assert len(distortion['rows']) == 24
        assert all(r['peak_bound_required'] is False for r in distortion['rows'])
        markdown = (output / 'ablation_summary.md').read_text().lower()
        assert 'rho=.1' in markdown or 'rho=0.1' in markdown
        assert 'rho=.4' not in markdown and 'rho=0.4' not in markdown
        assert 'clean pck@0.2' in markdown and 'untriggered' in markdown
        monkeypatch.setattr(trainer, 'build_model', forbidden)
        runner._run_cell(matrix['cells'][3])
        assert runner.build_summary(matrix)['rows'] == report['rows']
    finally:
        torch.set_num_threads(previous_threads)
