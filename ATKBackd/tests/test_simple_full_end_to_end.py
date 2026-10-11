"""Real 50-epoch CPU ERM for four Proposed-only arms, peak budget and resume.

The MM-Fi interface is synthetic and tiny; these are execution checks, not
measurements or scientific outcomes on the official MM-Fi dataset.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_peak_confirmation as confirmation
import run_simple_full as runner
import simple_full_contract as contract
import train_backdoor as trainer
from attack.peak_budget import PeakBudgetTrigger


KEYS = ['additive', 'single', 'shared_links', 'frequency_only']
REFERENCE_KEYS = ['badnets', 'blended', 'wanet_source', 'ftrojan', 'fiba', 'proposed']


def forbidden(*args, **kwargs):
    pytest.fail('A completed or invalid cache attempted fresh training')


def test_real_full_four_proposed_victims_frozen_baselines_peak_exports_and_resume(
        tmp_path, monkeypatch):
    initial_models, updates, split_calls, model_configs = [], [], [], []
    current = [None]
    dataset_state = {'reverse_test': False, 'drop_train': False}

    class TinyMMFI:
        def __init__(self, split, data_root, **kwargs):
            assert split in ('training', 'test')
            assert kwargs['protocol'] == 'protocol1' and kwargs['setting'] == 's1'
            split_calls.append(split)
            self.data_root = str(data_root)
            n = 30 if split == 'training' else 3
            self.items = [dict(csi=str(Path(data_root) / split / f'csi_{i}.npy'),
                kpt=str(Path(data_root) / split / 'ground_truth.npy'), frame_idx=i)
                for i in range(n)]
            if split == 'test' and dataset_state['reverse_test']:
                self.items.reverse()
            if split == 'training' and dataset_state['drop_train']:
                self.items.pop()
            self.raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(19).normal(0, .2, (n, 1, 17, 3)).astype(np.float32)

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
            initial_models.append({key: value.clone() for key, value in self.state_dict().items()})

        def forward(self, csi):
            return self.head(csi.mean((-2, -1))).reshape(-1, 1, 17, 3), None

    real_update = trainer._victim_update
    def update(*args, **kwargs):
        updates.append(current[0])
        return real_update(*args, **kwargs)

    counts = {'train': 30, 'eval': 3}
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(contract, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(confirmation, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    def build_candidate_model(model_name, *args, **kwargs):
        cfg = next(cell['cfg'] for cell in matrix['cells'] if cell['method_key'] == current[0])
        model_configs.append(dict(cfg))
        assert model_name == cfg['model']
        assert cfg['trigger'] == 'simple_carrier', 'Never build/train a traditional baseline'
        assert cfg['simple_variant'] in KEYS
        return TinyVictim()
    monkeypatch.setattr(trainer, 'build_model', build_candidate_model)
    monkeypatch.setattr(trainer, '_victim_update', update)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a: None)
    monkeypatch.setattr(torch.backends.cudnn, 'is_available', lambda: False)

    data, output = tmp_path / 'data', tmp_path / 'simple_full'
    action = data / 'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(data, output, 'cpu', 0, distortion_samples=3)
    reference_rows = [dict(status=runner.STATUS, method_key=key,
        method=f'Unchanged frozen {key}', seed=42, epochs=50, rho=.1,
        train_samples=30, eval_samples=3, t1_mpjpe_mm=150.0 + i,
        clean_mpjpe_mm=174.0 + i / 10, clean_pampjpe_mm=104.0 + i / 10,
        **{metric: 90.0 - j * 10 - i / 100 for j, metric in enumerate(runner.PCK_METRICS)},
        cfg_fingerprint=f'{i + 1:064x}') for i, key in enumerate(REFERENCE_KEYS)]
    reference_doses = [dict(status=runner.STATUS, method_key=key, method=f'Unchanged frozen {key}',
        seed=42, d=dose, no_trigger_tmpjpe_mm=400.0, triggered_tmpjpe_mm=150.0 + i,
        improvement_mm=250.0 - i, baseline_source='clean_to_target_tmpjpe')
        for i, key in enumerate(REFERENCE_KEYS) for dose in runner.GRID]
    reference_bundle = dict(rows=reference_rows, dose_response=reference_doses,
        inputs={}, poison_plans={}, configs={}, provenance={'frozen_files_sha256': 'c' * 64})
    reference_snapshot = json.dumps(reference_bundle, sort_keys=True)
    monkeypatch.setattr(runner, '_reference_bundle', lambda cfg: reference_bundle)
    first_identity, first_poison = [], []
    def validate_frozen_identity(bundle, record, poison_manifest=None):
        assert bundle is reference_bundle
        identity = [record['train'], record['eval'], record['action_file_sha256']]
        if first_identity:
            if identity != first_identity[0]:
                raise ValueError('Candidate changed the frozen TRAIN/TEST/action identity')
        else:
            first_identity.append(identity)
        if poison_manifest is not None:
            samples = poison_manifest['samples']
            if first_poison:
                if samples != first_poison[0]:
                    raise ValueError('Candidate changed frozen poison identities or doses')
            else:
                first_poison.append(samples)
    monkeypatch.setattr(runner, 'validate_candidate_compatibility', validate_frozen_identity)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        poison_plans = []
        for cell in matrix['cells']:
            current[0] = cell['method_key']
            folder = Path(cell['ckpt_dir'])
            assert not folder.exists(), 'Every arm starts with a fresh victim'
            runner._run_cell(cell)
            assert runner._complete(cell)
            checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=False, map_location='cpu')
            assert checkpoint['epoch'] == 49
            assert checkpoint['cfg']['epochs'] == checkpoint['cfg']['victim_epochs'] == 50
            assert checkpoint['cfg']['rho'] == .1
            assert checkpoint['cfg']['dose_min'] == .2 and checkpoint['cfg']['dose_max'] == 1.0
            assert 'trigger_optimizer' not in checkpoint
            record = json.loads((folder / 'simple_inputs.json').read_text(encoding='utf-8'))
            assert record['profile'] == runner.PROFILE
            assert record['config_fingerprint'] == trainer._config_fingerprint(cell['cfg'])
            assert record['train']['n'] == 30 and record['eval']['n'] == 3
            assert not (folder / 'draft_subsets.json').exists()
            manifest = json.loads((folder / 'poison_manifest.json').read_text(encoding='utf-8'))
            assert manifest.get('n_cover', 0) == 0
            assert manifest['n_poison'] == 3
            poison_plans.append(manifest['samples'])
            doses = [sample['dose'] for sample in manifest['samples']]
            assert len(set(doses)) == 3 and all(.2 <= dose <= 1.0 for dose in doses)

            # Exercise the actual wrapped operators independently of the HPE
            # and measure the historical peak ceiling on final float32 input.
            trigger = trainer.build_trigger(cell['cfg'])
            assert isinstance(trigger, PeakBudgetTrigger)
            assert 'simple_relative_l2' not in cell['cfg']
            raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
            for dose in runner.GRID:
                triggered, audit = trigger.inject_with_audit(raw, dose, eps=cell['cfg']['eps'])
                assert audit['violation'] is False
                assert audit['peak'] <= audit['ceiling']
                assert 'l2_ceiling' not in audit and 'l2_violation' not in audit
                assert np.isfinite(triggered).all()
                assert triggered.min() >= 0 and triggered.max() <= 1
                if dose == 0:
                    np.testing.assert_array_equal(triggered, raw)

        assert len(initial_models) == len(model_configs) == 4
        for initial in initial_models[1:]:
            for key in initial:
                torch.testing.assert_close(initial[key], initial_models[0][key], rtol=0, atol=0)
        assert all(updates.count(key) == 50 for key in KEYS)
        assert len(poison_plans) == 4 and all(plan == poison_plans[0] for plan in poison_plans)
        assert set(split_calls) == {'training', 'test'}

        monkeypatch.setattr(trainer, 'build_model', forbidden)
        monkeypatch.setattr(trainer, '_victim_update', forbidden)
        monkeypatch.setattr(TinyVictim, 'forward', forbidden)
        report = runner.export_summary(matrix, output)
        assert [row['method_key'] for row in report['rows']] == KEYS
        assert len(report['dose_response']) == 24
        assert report['frozen_reference_rows'] == reference_rows
        assert report['frozen_reference_dose_response'] == reference_doses
        assert json.dumps(reference_bundle, sort_keys=True) == reference_snapshot
        assert all(row['train_samples'] == 30 and row['eval_samples'] == 3
                   and row['epochs'] == 50 for row in report['rows'])
        assert len({report['audit']['poison_plan_sha256'][key]
                    for key in KEYS}) == 1
        for key in KEYS:
            assert [row['d'] for row in report['dose_response'] if row['method_key'] == key] == runner.GRID
        assert all((output / name).is_file() for name in (
            'main_metrics.csv', 'main_metrics.json', 'main_metrics.md',
            'dose_response.csv', 'dose_response.json', 'comparisons.csv',
            'comparisons.json', 'input_distortion.csv', 'input_distortion.json'))
        with (output / 'main_metrics.csv').open(encoding='utf-8', newline='') as handle:
            merged_rows = list(csv.DictReader(handle))
        assert [row['method_key'] for row in merged_rows] == REFERENCE_KEYS + KEYS
        with (output / 'dose_response.csv').open(encoding='utf-8', newline='') as handle:
            merged_doses = list(csv.DictReader(handle))
        assert len(merged_doses) == 60
        assert {row['method_key'] for row in merged_doses} == set(REFERENCE_KEYS + KEYS)
        comparison = json.loads((output / 'comparisons.json').read_text(encoding='utf-8'))
        assert len(comparison['rows']) == 8
        assert {row['reference'] for row in comparison['rows']} == {'proposed', 'blended'}
        assert {row['method_key'] for row in comparison['rows']} == set(KEYS)
        by_key = {row['method_key']: row for row in report['rows'] + reference_rows}
        for row in comparison['rows']:
            for metric in runner.MAIN_METRICS:
                assert row[f'delta_{metric}'] == pytest.approx(
                    by_key[row['method_key']][metric] - by_key[row['reference']][metric])
            assert 'winner' not in row
        text = (output / 'main_metrics.md').read_text(encoding='utf-8').lower()
        assert any(phrase in text for phrase in ('single seed', 'single-seed', 'one seed'))
        assert 'no winner is selected' in text
        distortion = json.loads((output / 'input_distortion.json').read_text(encoding='utf-8'))
        assert len(distortion['rows']) == 24
        assert len(distortion['common_pair_ids']) == 3
        assert distortion['hpe_forward_used'] is False
        assert all(row['peak_violations'] == 0 and 'l2_violations' not in row
                   for row in distortion['rows'])
        assert all(row['relative_l2'] == 0 and row['linf'] == 0
                   for row in distortion['rows'] if row.get('dose', row.get('d')) == 0)

        # Resume must accept the real completion without any second victim.
        completed = matrix['cells'][3]
        runner._run_cell(completed)
        assert len(initial_models) == 4 and len(updates) == 200
        assert runner.build_summary(matrix)['rows'] == report['rows']

        # Read-only references are frozen in the passport, not silently rebound.
        reference_bundle['provenance']['frozen_files_sha256'] = 'd' * 64
        try:
            with pytest.raises(ValueError):
                runner._complete(completed)
            with pytest.raises(ValueError):
                runner._run_cell(completed)
        finally:
            reference_bundle['provenance']['frozen_files_sha256'] = 'c' * 64
        assert len(initial_models) == 4 and len(updates) == 200

        # Reordered identities and incomplete split parents must fail before
        # the old metrics could be accepted or audit bytes rewritten.
        input_path = Path(completed['ckpt_dir']) / 'simple_inputs.json'
        before = input_path.read_bytes()
        for state_key in ('reverse_test', 'drop_train'):
            dataset_state[state_key] = True
            try:
                with pytest.raises(ValueError):
                    runner._run_cell(completed)
                assert input_path.read_bytes() == before
            finally:
                dataset_state[state_key] = False

        cache_path = Path(completed['eval_cache'])
        original_cache = cache_path.read_bytes()
        for mutation in ('incomplete_epochs', 'stale_fingerprint', 'missing_dose'):
            cache = json.loads(original_cache)
            if mutation == 'incomplete_epochs':
                cache['trained_epochs'] = 49
            elif mutation == 'stale_fingerprint':
                cache['cfg_fingerprint'] = '0' * 64
            else:
                cache['res']['tmpjpe'].pop()
            cache_path.write_text(json.dumps(cache), encoding='utf-8')
            try:
                with pytest.raises(ValueError):
                    runner.build_summary(matrix)
                with pytest.raises(ValueError):
                    runner._complete(completed)
                with pytest.raises(ValueError):
                    runner._run_cell(completed)
            finally:
                cache_path.write_bytes(original_cache)

        poison_path = Path(completed['ckpt_dir']) / 'poison_manifest.json'
        original_poison = poison_path.read_bytes()
        poison = json.loads(original_poison)
        poison['samples'][0]['dose'] = 1.0
        poison_path.write_text(json.dumps(poison), encoding='utf-8')
        try:
            with pytest.raises(ValueError):
                runner.build_summary(matrix)
        finally:
            poison_path.write_bytes(original_poison)

        # Frozen source action bytes are also part of the resume identity.
        action_bytes = action.read_bytes()
        np.save(action, np.random.default_rng(24).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
        try:
            with pytest.raises(ValueError):
                runner._run_cell(completed)
        finally:
            action.write_bytes(action_bytes)
        assert runner.build_summary(matrix)['rows'] == report['rows']
    finally:
        torch.set_num_threads(previous_threads)
