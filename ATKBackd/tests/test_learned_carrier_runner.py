"""Isolated runner contracts: inert fixtures, no official data or real training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_learned_carrier_drafts as runner
import train_backdoor as trainer
from attack.learned_carrier import TrainableCarrier, write_artifact
from attack.poison import PoisonedDataset
from attack.trigger import MicroDopplerTrigger
from data_utils.draft_subset import apply_draft_subset
from mmfi_tables import config_fingerprint


def forbidden(*args, **kwargs):
    pytest.fail('This operation must not load data, fit, probe CUDA or train')


class TinyDataset:
    def __init__(self, root, n=24):
        self.data_root = str(root)
        self.items = [dict(csi=str(root / f'csi_{i}.npy'),
            kpt=str(root / f'pose_{i}.npy'), frame_idx=i) for i in range(n)]
        self.raw = np.random.default_rng(1).uniform(.1, .8, (3, 114, 10)).astype(np.float32)

    def __len__(self):
        return len(self.items)

    def load_raw(self, path):
        return self.raw.copy()

    def normalize(self, value):
        return np.clip(value, 0, 1).astype(np.float32)

    def enable_evaluation_cache(self):
        return True


class Identity:
    def inject(self, csi, dose, eps=0):
        return csi.copy()


@pytest.fixture
def matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture': 'a' * 64})
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'out', 'cpu', 0,
        epochs=3, train_samples=10, eval_samples=4, distortion_samples=4)


def _prepared_fixture(cell):
    cfg = copy.deepcopy(cell['cfg'])
    action = Path(cfg['action_npy'])
    action.parent.mkdir(parents=True, exist_ok=True)
    if not action.exists():
        np.save(action, np.ones((1, 3, 30, 25, 1), dtype=np.float32))
    folder = Path(cell['ckpt_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    if cfg.get('lc_variant'):
        provenance = dict(recipe_sha256=cell['recipe_sha256'],
            variant=cfg['lc_variant'], official_test_loaded=False,
            external_draft_holdout_loaded=False)
        runner._atomic_json(folder / 'fitting.json', provenance)
        cfg.update(lc_recipe_sha256=cell['recipe_sha256'],
            lc_fitting_sha256=runner._file_sha(folder / 'fitting.json'))
        if cfg['lc_variant'] == 'selection':
            cfg.update(lc_poison_indices=[1], lc_selection='stratified_training_pose_error',
                lc_selection_sha256=runner._sha_json([1]))
        else:
            base = MicroDopplerTrigger(n_ant=3, n_sub=114, n_pkt=10, seed=42, zero_mean=True)
            pattern = np.random.default_rng(2).normal(size=(3, 114, 10))
            base.m_zm = (pattern - pattern.mean()) / np.sqrt(np.mean((pattern - pattern.mean()) ** 2))
            base.m = base.m_zm.astype(np.complex128)
            module = TrainableCarrier(base, cfg, cfg['lc_variant'])
            artifact = folder / 'learned_trigger.json'
            digest = write_artifact(artifact, module, cell['recipe_sha256'])
            cfg.update(lc_artifact_path=str(artifact), lc_artifact_sha256=digest)
    runner._atomic_json(folder / 'prepared_cfg.json', dict(cfg=cfg,
        recipe_sha256=cell['recipe_sha256'], cfg_fingerprint=config_fingerprint(cfg),
        action_file_sha256=runner._file_sha(action)))
    return dict(cell, cfg=cfg)


def _completed_fixture(matrix):
    datasets = {}
    for requested in matrix['cells']:
        cell = _prepared_fixture(requested)
        cfg, key = cell['cfg'], cell['method_key']
        parent = TinyDataset(Path(cfg['dataset_root']))
        train = apply_draft_subset(parent, cfg, 'train')
        holdout = apply_draft_subset(parent, cfg, 'test')
        datasets[key] = holdout
        fingerprint = config_fingerprint(cfg)
        poison = PoisonedDataset(train, Identity(), mode='train', rho=cfg['rho'],
            seed=42, dataset='mmfi', pivot=1, theta_max_deg=40,
            explicit_indices=cfg.get('lc_poison_indices')).manifest()
        poison['config_fingerprint'] = fingerprint
        res = dict(clean_mpjpe=.025, clean_pampjpe=.015, dose_grid=runner.GRID,
            reference_dose=1., tmpjpe=[.025, .03, .035, .04, .045, .05],
            clean_to_target_tmpjpe=[.025, .04, .055, .07, .085, .10],
            target_joints=[2, 3], config_fingerprint=fingerprint,
            draft_action_sha256=runner._file_sha(cfg['action_npy']),
            poison_plan_sha256=poison['poison_plan_sha256'],
            training_contract='ordinary_erm', attacker_access='data_only',
            dose_coupling='paired', n_total=len(train), n_poison=poison['n_poison'])
        for threshold in (.5, .4, .3, .2, .1):
            res[f'clean_pck@{threshold:.1f}'] = threshold + .4
        runner._atomic_json(cell['eval_cache'], dict(result_schema=trainer._RESULT_SCHEMA,
            cfg=cfg, cfg_fingerprint=fingerprint, trained_epochs=cfg['epochs'], res=res))
        runner._atomic_json(Path(cell['ckpt_dir']) / 'poison_manifest.json', poison)
        runner._atomic_json(Path(cell['ckpt_dir']) / 'draft_subsets.json',
            dict(status=runner.STATUS, config_fingerprint=fingerprint,
                action_file_sha256=runner._file_sha(cfg['action_npy']),
                train=train.draft_subset_manifest(), eval=holdout.draft_subset_manifest()))
    return datasets


def test_plan_retains_every_direction_and_common_protocol(matrix):
    assert [c['method_key'] for c in matrix['cells']] == [m[0] for m in runner.METHODS]
    assert len(matrix['cells']) == 9
    assert matrix['DRAFT_ONLY'] is True and matrix['status'] == runner.STATUS
    for cell in matrix['cells']:
        cfg = cell['cfg']
        assert cfg['seed'] == 42 and cfg['epochs'] == 3 and cfg['lr'] == .001
        assert cfg['lc_fit_seed'] == 4242
        assert cfg['optimizer'] == 'sgd' and cfg['momentum'] == .9
        assert cfg['weight_decay'] == 0 and cfg['batch_size'] == 32
        assert cfg['rho'] == (0 if cell['method_key'] == 'clean' else .1)
        assert cfg['draft_profile'] == runner.PROFILE
        assert cfg['draft_eval_source'] == 'training_holdout'
        assert cfg['lc_relative_l2'] == .1 and cfg['lc_reference_eps'] == .185
        assert cfg['dose_min'] == .2 and cfg['dose_max'] == 1
        assert cfg['dose_coupling'] == 'paired' and cfg['target_joints'] == [2, 3]
        assert cfg['pretrained'] is False and cfg['attacker_access'] == 'data_only'
    runner._validate_manifest(matrix)


def test_plan_runtime_is_ignored_but_fitting_science_is_bound(matrix):
    altered = copy.deepcopy(matrix)
    for cell in altered['cells']:
        cell['cfg'].update(device='cuda:3', num_workers=8)
    assert runner._plan_sha(altered) == matrix['plan_sha256']
    runner._validate_manifest(altered)
    altered['cells'][3]['cfg']['lc_rounds'] += 1
    assert runner._plan_sha(altered) != matrix['plan_sha256']
    with pytest.raises(ValueError, match='fingerprint'):
        runner._validate_manifest(altered)


def test_source_drift_is_rejected(matrix, monkeypatch):
    monkeypatch.setattr(runner, '_source_provenance', lambda: {'fixture': 'b' * 64})
    with pytest.raises(ValueError, match='source'):
        runner._validate_manifest(matrix)


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_bad_fit_seed_fails_before_data_access(tmp_path, value):
    with pytest.raises(ValueError, match='lc_fit_seed'):
        runner.build_matrix(tmp_path, tmp_path / 'out', fit_options={'lc_fit_seed': value})


def test_dry_run_never_loads_data_cuda_fits_or_trains(tmp_path, monkeypatch):
    import learned_carrier_fit as fitting
    monkeypatch.setattr(runner, '_check_inputs', forbidden)
    monkeypatch.setattr(runner, 'run_matrix', forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', forbidden)
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(fitting, 'prepare_learned_cell', forbidden)
    out = tmp_path / 'dry'
    assert runner.main(['--data-home', str(tmp_path / 'absent'), '--outdir', str(out), '--dry-run']) == 0
    manifest = json.loads((out / 'learned_carrier.resolved.json').read_text())
    assert len(manifest['cells']) == 9
    assert not (out / 'clean').exists()


def test_fresh_does_not_delete_existing_results(tmp_path):
    out = tmp_path / 'occupied'
    out.mkdir()
    old = out / 'old.json'
    old.write_text('user result')
    with pytest.raises(FileExistsError, match='never deletes'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(out), '--fresh', '--dry-run'])
    assert old.read_text() == 'user result'


def test_prepared_artifact_config_and_action_are_strict(matrix):
    requested = matrix['cells'][3]
    completed = _prepared_fixture(requested)
    assert runner._prepared_cell(requested)['cfg'] == completed['cfg']
    changed = dict(completed['cfg'], theta_max_deg=10)
    with pytest.raises(ValueError, match='immutable'):
        runner._validated_prepared(requested, changed)
    artifact = Path(completed['cfg']['lc_artifact_path'])
    artifact.write_text(artifact.read_text() + '\n')
    with pytest.raises(ValueError, match='artifact changed'):
        runner._prepared_cell(requested)


def test_selection_hash_is_checked_separately(matrix):
    cell = matrix['cells'][-1]
    cfg = _prepared_fixture(cell)['cfg']
    assert runner._validated_prepared(cell, cfg)['lc_poison_indices'] == [1]
    with pytest.raises(ValueError, match='selection identities'):
        runner._validated_prepared(cell, dict(cfg, lc_poison_indices=[2]))


def test_summary_retains_all_rows_six_doses_and_separate_selection(matrix):
    _completed_fixture(matrix)
    report = runner.build_summary(matrix)
    assert len(report['rows']) == 9 and len(report['dose_response']) == 54
    assert report['rows'][3]['t1_mpjpe_mm'] == 50
    assert report['rows'][3]['mean_positive_dose_tmpjpe_mm'] == 40
    assert report['rows'][3]['i1_improvement_mm'] == 50
    assert report['rows'][-1]['selection_ablation'] is True
    assert all(runner._complete(c) for c in matrix['cells'])
    uniform = [r['poison_plan_sha256'] for r in report['rows'][1:-1]]
    assert len(set(uniform)) == 1
    assert report['rows'][-1]['poison_plan_sha256'] != uniform[0]


def test_mismatched_uniform_poison_plan_is_not_comparison(matrix):
    _completed_fixture(matrix)
    cell = runner._prepared_cell(matrix['cells'][2])
    record_path = Path(cell['ckpt_dir']) / 'poison_manifest.json'
    record = json.loads(record_path.read_text())
    record['samples'][0]['dose'] = .3
    import hashlib
    plan = [[s['index'], s['dose']] for s in record['samples']]
    record['poison_plan_sha256'] = hashlib.sha256(json.dumps(plan,
        separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    runner._atomic_json(record_path, record)
    with pytest.raises(ValueError, match='same poison'):
        runner.build_summary(matrix)


def test_partial_export_fails_without_fitting_or_training(matrix, monkeypatch):
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(runner, 'run_matrix', forbidden)
    with pytest.raises(ValueError, match='prepared config'):
        runner.build_summary(matrix)


def test_export_only_missing_cells_never_invokes_fitting(matrix, monkeypatch):
    import learned_carrier_fit as fitting
    monkeypatch.setattr(fitting, 'prepare_learned_cell', forbidden)
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(runner, '_check_inputs', forbidden)
    monkeypatch.setattr(runner, 'run_matrix', forbidden)
    monkeypatch.setattr(runner, 'build_matrix', lambda *args, **kw: copy.deepcopy(matrix))
    with pytest.raises(ValueError, match='Missing valid completed cells'):
        runner.main(['--data-home', 'missing', '--outdir', str(Path(matrix['cells'][0]['ckpt_dir']).parent),
            '--devices', 'cpu', '--export-only'])


def test_export_keeps_all_rows_and_marks_explicit_missing_controls(matrix, monkeypatch):
    matrix['metrics_contract']['clean_probe'] = 'missing_by_explicit_skip'
    matrix['metrics_contract']['distortion'] = 'missing_by_explicit_skip'
    matrix['plan_sha256'] = runner._plan_sha(matrix)
    _completed_fixture(matrix)
    monkeypatch.setattr(runner, 'build_distortion', forbidden)
    monkeypatch.setattr(runner, 'build_clean_probes', forbidden)
    outdir = Path(matrix['cells'][0]['ckpt_dir']).parent
    runner.export_summary(matrix, outdir)
    import csv
    with (outdir / 'draft_summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert rows[-1]['method_key'] == 'lc_selection'
    with (outdir / 'dose_response.csv').open(newline='') as handle:
        assert len(list(csv.DictReader(handle))) == 54
    assert 'missing_by_explicit_skip' in (outdir / 'draft_summary.md').read_text()


def test_clean_probe_uses_one_verified_rho_zero_checkpoint_for_every_candidate(matrix, monkeypatch):
    import torch
    import models.factory as factory
    datasets = _completed_fixture(matrix)
    clean = runner._prepared_cell(matrix['cells'][0])
    checkpoint_path = Path(clean['ckpt_dir']) / 'checkpoint.pt'
    torch.save(dict(epoch=clean['cfg']['epochs'] - 1,
        cfg_fingerprint=config_fingerprint(clean['cfg']), model={'fake': torch.tensor(1.)}), checkpoint_path)
    loaded = []
    class TinyVictim:
        def load_state_dict(self, state):
            loaded.append(state)
        def to(self, device):
            return self
        def eval(self):
            return self
    monkeypatch.setattr(factory, 'build_model', lambda *args, **kw: TinyVictim())
    monkeypatch.setattr(trainer, '_load_dataset', lambda *args: datasets['clean'])
    monkeypatch.setattr(trainer, 'build_trigger', lambda cfg: Identity())
    def predict(model, loader, device, **kwargs):
        target = np.full((4, 17, 3), .1)
        prediction = np.zeros((4, 17, 3)) if loader.dataset.mode == 'clean' else np.full((4, 17, 3), .02)
        return prediction, target, target, None
    monkeypatch.setattr(trainer, '_predict', predict)
    report = runner.build_clean_probes(matrix, 'cpu')
    assert len(loaded) == 1
    assert len(report['rows']) == 8
    assert len({r['clean_checkpoint_sha256'] for r in report['rows']}) == 1
    assert [r['method_key'] for r in report['rows']] == [c['method_key'] for c in matrix['cells'][1:]]
    assert all(r['clean_victim_improvement_mm'] > 0 for r in report['rows'])


def test_input_distortion_no_hpe_forward_and_zero_budget_violation(matrix, monkeypatch):
    datasets = _completed_fixture(matrix)
    audit = runner.build_summary(matrix)['audit']
    monkeypatch.setattr(trainer, '_load_dataset', lambda cfg, split: datasets[next(
        c['method_key'] for c in matrix['cells'] if c['cfg']['trigger'] == cfg['trigger']
        and c['cfg'].get('lc_variant') == cfg.get('lc_variant')
        and c['cfg']['rho'] == cfg['rho'])])
    monkeypatch.setattr(trainer, 'train', forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', lambda cfg: Identity())
    report = runner.build_distortion(matrix, audit, n=4)
    assert report['no_hpe_forward'] is True
    assert len(report['rows']) == 54
    assert all(row['relative_l2'] == 0 and row['l2_violations'] == 0
        and row['peak_violations'] == 0 for row in report['rows'])


@pytest.mark.parametrize('devices', [['cuda:0', 'cuda:0'], ['cuda:-1'], ['gpu:0']])
def test_bad_device_lists_fail_before_resolution(tmp_path, devices):
    with pytest.raises(ValueError, match='Devices'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(tmp_path / 'bad'),
            '--devices', *devices, '--dry-run'])


def test_shell_wrapper_forwards_python_devices_and_does_not_push():
    wrapper = (Path(__file__).resolve().parents[2] / 'remote_linux/09_run_learned_carrier_drafts.sh').read_text()
    assert '--python)' in wrapper and '--data-home)' in wrapper
    assert '"${DOSE_FORWARDED[@]}"' in wrapper
    assert wrapper.index('DOSE_PY="${DOSE_PY:-${DOSE_DATA_HOME}') > wrapper.index('\ndone\n')
    assert 'git push' not in wrapper and 'rm -' not in wrapper
