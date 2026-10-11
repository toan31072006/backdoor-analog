"""Real 50-epoch CPU ERM, fixed operators, dual budgets, export and resume.

The MM-Fi interface is synthetic and tiny; these are execution checks, not
measurements or scientific outcomes on the official MM-Fi dataset.
"""
from __future__ import annotations

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
from attack.learned_carrier import DualBudgetTrigger


KEYS = ['clean', 'blended', 'proposed', 'additive', 'single',
        'shared_links', 'frequency_only']


def forbidden(*args, **kwargs):
    pytest.fail('A completed or invalid cache attempted fresh training')


def test_real_full_seven_method_training_budgets_exports_and_fail_closed_resume(
        tmp_path, monkeypatch):
    initial_models, updates, split_calls = [], [], []
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
    monkeypatch.setattr(trainer, 'build_model', lambda *a, **k: TinyVictim())
    monkeypatch.setattr(trainer, '_victim_update', update)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a: None)
    monkeypatch.setattr(torch.backends.cudnn, 'is_available', lambda: False)

    data, output = tmp_path / 'data', tmp_path / 'simple_full'
    action = data / 'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(size=(1, 3, 30, 25, 1)).astype(np.float32))
    matrix = runner.build_matrix(data, output, 'cpu', 0, distortion_samples=3)
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
            assert checkpoint['cfg']['rho'] == (0.0 if current[0] == 'clean' else .1)
            assert checkpoint['cfg']['dose_min'] == .2 and checkpoint['cfg']['dose_max'] == 1.0
            assert 'trigger_optimizer' not in checkpoint
            record = json.loads((folder / 'simple_inputs.json').read_text(encoding='utf-8'))
            assert record['profile'] == runner.PROFILE
            assert record['config_fingerprint'] == trainer._config_fingerprint(cell['cfg'])
            assert record['train']['n'] == 30 and record['eval']['n'] == 3
            assert not (folder / 'draft_subsets.json').exists()
            manifest = json.loads((folder / 'poison_manifest.json').read_text(encoding='utf-8'))
            assert manifest.get('n_cover', 0) == 0
            if current[0] == 'clean':
                assert manifest['n_poison'] == 0 and manifest['samples'] == []
            else:
                assert manifest['n_poison'] == 3
                poison_plans.append(manifest['samples'])
                doses = [sample['dose'] for sample in manifest['samples']]
                assert len(set(doses)) == 3 and all(.2 <= dose <= 1.0 for dose in doses)

            # Exercise the actual wrapped operators independently of the HPE
            # and measure both ceilings on precisely the final float32 input.
            trigger = trainer.build_trigger(cell['cfg'])
            assert isinstance(trigger, DualBudgetTrigger)
            raw = np.random.default_rng(9).uniform(.1, .8, (3, 114, 10)).astype(np.float32)
            for dose in runner.GRID:
                triggered, audit = trigger.inject_with_audit(raw, dose, eps=cell['cfg']['eps'])
                assert audit['violation'] is False and audit['l2_violation'] is False
                assert audit['peak'] <= audit['ceiling']
                assert audit['l2'] <= audit['l2_ceiling']
                assert np.isfinite(triggered).all()
                assert triggered.min() >= 0 and triggered.max() <= 1
                if dose == 0:
                    np.testing.assert_array_equal(triggered, raw)

        assert len(initial_models) == 7
        for initial in initial_models[1:]:
            for key in initial:
                torch.testing.assert_close(initial[key], initial_models[0][key], rtol=0, atol=0)
        assert all(updates.count(key) == 50 for key in KEYS)
        assert len(poison_plans) == 6 and all(plan == poison_plans[0] for plan in poison_plans)
        assert set(split_calls) == {'training', 'test'}

        monkeypatch.setattr(trainer, 'build_model', forbidden)
        monkeypatch.setattr(trainer, '_victim_update', forbidden)
        monkeypatch.setattr(TinyVictim, 'forward', forbidden)
        report = runner.export_summary(matrix, output)
        assert [row['method_key'] for row in report['rows']] == KEYS
        assert len(report['dose_response']) == 42
        assert all(row['train_samples'] == 30 and row['eval_samples'] == 3
                   and row['epochs'] == 50 for row in report['rows'])
        assert len({report['audit']['poison_plan_sha256'][key]
                    for key in KEYS if key != 'clean'}) == 1
        for key in KEYS:
            assert [row['d'] for row in report['dose_response'] if row['method_key'] == key] == runner.GRID
        clean_doses = [row for row in report['dose_response'] if row['method_key'] == 'clean']
        assert all(row['triggered_tmpjpe_mm'] == row['no_trigger_tmpjpe_mm']
                   and row['improvement_mm'] == 0 for row in clean_doses)
        assert report['rows'][0]['t1_mpjpe_mm'] == clean_doses[-1]['no_trigger_tmpjpe_mm']
        assert all((output / name).is_file() for name in (
            'main_metrics.csv', 'main_metrics.json', 'main_metrics.md',
            'dose_response.csv', 'dose_response.json', 'comparisons.csv',
            'comparisons.json', 'input_distortion.csv', 'input_distortion.json'))
        comparison = json.loads((output / 'comparisons.json').read_text(encoding='utf-8'))
        assert len(comparison['rows']) == 8
        assert {row['reference'] for row in comparison['rows']} == {'proposed', 'blended'}
        assert {row['method_key'] for row in comparison['rows']} == set(KEYS[3:])
        by_key = {row['method_key']: row for row in report['rows']}
        for row in comparison['rows']:
            for metric in runner.MAIN_METRICS:
                assert row[f'delta_{metric}'] == pytest.approx(
                    by_key[row['method_key']][metric] - by_key[row['reference']][metric])
            assert 'winner' not in row
        text = (output / 'main_metrics.md').read_text(encoding='utf-8').lower()
        assert any(phrase in text for phrase in ('single seed', 'single-seed', 'one seed'))
        assert 'no winner is selected' in text
        distortion = json.loads((output / 'input_distortion.json').read_text(encoding='utf-8'))
        assert len(distortion['rows']) == 42
        assert len(distortion['common_pair_ids']) == 3
        assert distortion['hpe_forward_used'] is False
        assert all(row['peak_violations'] == row['l2_violations'] == 0
                   for row in distortion['rows'])
        assert all(row['relative_l2'] == 0 and row['linf'] == 0
                   for row in distortion['rows'] if row.get('dose', row.get('d')) == 0)

        # Resume must accept the real completion without any second victim.
        completed = matrix['cells'][3]
        runner._run_cell(completed)
        assert len(initial_models) == 7 and len(updates) == 350
        assert runner.build_summary(matrix)['rows'] == report['rows']

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
