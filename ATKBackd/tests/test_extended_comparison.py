"""Full extended protocol and actual synthetic train/export; never MM-Fi data."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_traditional_comparison as runner
import run_peak_confirmation as confirmation
import train_backdoor as trainer

KEYS = ['badnets', 'blended', 'wanet_source', 'ftrojan', 'fiba', 'proposed']


@pytest.mark.parametrize('mode', ['shared_peak', 'native'])
def test_extended_plan_has_six_rows_source_defaults_no_ccai_and_common_task(tmp_path, mode, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Dry planning attempted dataset/model/trigger/benchmark work')
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer, 'build_model', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', forbidden)
    matrix = runner.build_matrix(tmp_path/'missing', tmp_path/'run', 'cuda:3', 4,
        baseline_set='extended', budget_mode=mode)
    assert [c['method_key'] for c in matrix['cells']] == KEYS
    assert 'ccai2026_backdoorrf' not in KEYS and len(KEYS) == 6
    assert 'INFOCOM/POR' in matrix['metrics_contract']['por_role']
    assert matrix['traditional_comparison_profile'] == f'mmfi_extended_{mode}_comparison_v1'
    for cell in matrix['cells']:
        cfg = cell['cfg']
        assert cfg['rho'] == .1 and cfg['epochs'] == 50 and cfg['seed'] == 42
        assert (cfg['dose_min'], cfg['dose_max'], cfg['dose_coupling']) == (.2, 1, 'paired')
        assert cfg['training_protocol'] == 'ordinary_erm' and cfg['attacker_access'] == 'data_only'
        assert bool(cfg.get('comparison_peak_budget')) == (mode == 'shared_peak')
    ft, fiba, proposed = [c['cfg'] for c in matrix['cells'][3:]]
    assert ft['ftrojan_magnitude_255'] == 20 and ft['ftrojan_window_size'] == 32
    assert ft['ftrojan_positions_32'] == [[31,31], [15,15]] and ft['ftrojan_channels'] == [1,2]
    assert fiba['fiba_alpha'] == .15 and fiba['fiba_beta'] == .1 and fiba['fiba_cross_ratio'] == 1
    assert fiba['clean_label_cover_ratio'] == .1 and fiba['num_workers'] == 0
    assert proposed['trigger'] == 'md_multicarrier_peak_matched'
    runner._validate_manifest(matrix)
    altered = copy.deepcopy(matrix)
    altered['cells'][3]['cfg']['ftrojan_magnitude_255'] = 25
    altered['plan_sha256'] = runner._plan_fingerprint(altered)
    with pytest.raises(ValueError):
        runner._validate_manifest(altered)
    altered = copy.deepcopy(matrix)
    altered['cells'].pop(3)
    altered['plan_sha256'] = runner._plan_fingerprint(altered)
    with pytest.raises(ValueError, match='fixed order'):
        runner._validate_manifest(altered)


@pytest.mark.parametrize('mode', ['shared_peak', 'native'])
def test_real_six_cell_train_export_same_poison_doses_covers_and_six_evaluations(tmp_path, monkeypatch, mode):
    class TinyMMFI:
        def __init__(self, split, data_root, **kwargs):
            self.data_root = str(data_root)
            n = 30 if split == 'training' else 3
            self.items = [dict(csi=str(Path(data_root)/split/f'{i}.npy'),
                kpt=str(Path(data_root)/split/'ground_truth.npy'), frame_idx=i) for i in range(n)]
            self.inputs = np.random.default_rng(9).uniform(.1, .8, (n, 3, 114, 10)).astype(np.float32)
            self.poses = np.random.default_rng(19).normal(0, .2, (n, 1, 17, 3)).astype(np.float32)
        def __len__(self):
            return len(self.items)
        def load_raw(self, path):
            return self.inputs[int(Path(path).stem)].copy()
        def load_pose(self, path, frame_idx):
            return self.poses[frame_idx].copy()
        @staticmethod
        def normalize(value):
            return np.clip(value, 0, 1).astype(np.float32)

    class TinyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 51)
        def forward(self, csi):
            return self.head(csi.mean((-2,-1))).reshape(-1,1,17,3), None

    counts = dict(train=30, eval=3)
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(confirmation, 'EXPECTED_COUNTS', counts)
    monkeypatch.setattr(trainer, 'MMFI', TinyMMFI)
    monkeypatch.setattr(trainer, 'build_model', lambda *args, **kwargs: TinyVictim())
    data, output = tmp_path/'data', tmp_path/'run'
    action = data/'actions/data_bend.npy'
    action.parent.mkdir(parents=True)
    np.save(action, np.random.default_rng(23).normal(size=(1,3,30,25,1)).astype(np.float32))
    matrix = runner.build_matrix(data, output, 'cpu', 0, baseline_set='extended',
                                 budget_mode=mode, distortion_samples=3)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        manifests = []
        for cell in matrix['cells']:
            runner._run_cell(cell)
            path = Path(cell['ckpt_dir'])
            manifests.append(json.loads((path/'poison_manifest.json').read_text()))
            blob = torch.load(path/'checkpoint.pt', map_location='cpu', weights_only=False)
            assert blob['epoch'] == 49
            if cell['method_key'] == 'fiba':
                inputs = json.loads((path/'comparison_inputs.json').read_text())
                assert len(inputs['fiba_reference_sha256']) == 64
                state = blob['trigger']
                native = state.get('native', state)
                assert native['cover_draws'] == 150
        assert len({m['poison_plan_sha256'] for m in manifests}) == 1
        assert manifests[2]['n_cover'] == 6 and manifests[4]['n_cover'] == 3
        report = runner.export_summary(matrix, output)
        assert [r['method_key'] for r in report['rows']] == KEYS
        assert len(report['dose_response']) == 36 and len(report['distortion']['rows']) == 36
        assert len({r['common_pair_ids_sha256'] for r in report['distortion']['rows']}) == 1
        if mode == 'shared_peak':
            assert len(report['distortion']['peak_audits']) == 108
            assert len(report['distortion']['cover_peak_audits']) == 6
            assert all(not r['violation'] for r in report['distortion']['peak_audits'])
        assert not any('ccai' in r['method_key'] for r in report['rows'])
    finally:
        torch.set_num_threads(threads)
