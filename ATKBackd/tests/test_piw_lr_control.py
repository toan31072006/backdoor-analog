"""LR-only control parity and exact-resume contracts; no efficacy claim."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import piw_v2_adapter as adapter
import run_piw_hpeli_clean as base
import run_piw_hpeli_v2 as previous
import run_piw_hpeli_lr_control as control
import test_piw_v2 as shared


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, 'runner', control)
    return shared.dataset.__wrapped__(tmp_path, monkeypatch)


def test_historical_source_unchanged_and_only_lr_changes(tmp_path):
    historical = hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest()
    assert historical == 'ac3ac345a73340f9538354fad0fdd6a6b80517467e1387c2928066f0a2a305de'
    old = previous.configuration(tmp_path/'home', tmp_path/'old', device='cpu')
    new = control.configuration(tmp_path/'home', tmp_path/'new', device='cpu')
    administrative = {'profile', 'status', 'outdir'}
    changes = {k: (old[k], new[k]) for k in old if k not in administrative and old[k] != new[k]}
    assert changes == {'lr': (.01, .001)}
    assert set(old) == set(new)
    assert old['target'] == new['target'] == adapter.target_contract()
    assert new['drop_last'] is True
    assert 'run_piw_hpeli_v2.py' in control.SOURCE_NAMES
    assert 'run_piw_hpeli_lr_control.py' in control.SOURCE_NAMES


def test_unchanged_training_and_evaluation_implementation():
    def functions(path):
        tree = ast.parse(Path(path).read_text(encoding='utf-8'))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    class Labels(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                value = node.value.replace('[PiW LR control', '[PiW v2')
                value = value.replace('frozen body-axis implementation', 'v2 implementation')
                value = value.replace('historical profiles cannot be interchanged', 'v1 and v2 cannot be interchanged')
                value = value.replace('Unexpected LR control prediction schema', 'Unexpected v2 prediction schema')
                return ast.copy_location(ast.Constant(value=value), node)
            return node

    old, new = functions(previous.__file__), functions(control.__file__)
    for name in ('datasets_for', 'prepare', 'seed_execution', 'optimizer_for', 'checkpoint',
                 'step', 'train', 'infer', 'atomic_npz', 'evaluate', 'artifact_hashes',
                 'replay_equal', 'verify_complete', 'cpu_preflight', 'wait_for_files', 'smoke_worker'):
        assert ast.dump(Labels().visit(old[name])) == ast.dump(Labels().visit(new[name])), name


def test_poison_targets_initialization_and_sampler_are_identical(dataset):
    old_cfg = previous.configuration(dataset['data_home'], Path(dataset['outdir']).with_name('old'),
                                     dataset['dataset_root'], dataset['action_npy'], 'cpu', 2)
    old_context, old_ds, old_poison, _, _ = previous.prepare(old_cfg)
    new_context, new_ds, new_poison, _, _ = control.prepare(dataset)
    assert old_context == new_context
    for i in range(len(old_poison)):
        for key in ('csi', 'pose'):
            assert np.array_equal(old_poison[i][key], new_poison[i][key])
    previous.seed_execution(old_cfg)
    old_model = base.build_model()
    old_order = torch.randperm(len(old_ds['training']))
    old_rng = torch.get_rng_state().clone()
    control.seed_execution(dataset)
    new_model = base.build_model()
    new_order = torch.randperm(len(new_ds['training']))
    for key in old_model.state_dict():
        assert torch.equal(old_model.state_dict()[key], new_model.state_dict()[key])
    assert torch.equal(old_order, new_order)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert previous.optimizer_for(old_model, old_cfg).param_groups[0]['lr'] == .01
    assert control.optimizer_for(new_model, dataset).param_groups[0]['lr'] == .001


def test_lr_control_full_toy_run_resume_and_cross_profile_rejection(dataset, monkeypatch):
    m = shared.manifest(dataset)
    # Evaluation must fail before both victims are at the fixed final epoch.
    with pytest.raises(FileNotFoundError):
        control.evaluate(m)
    control.train(m, 'clean')
    full = control.checkpoint(m, 'clean', final=True)
    cfg2 = control.configuration(dataset['data_home'], Path(dataset['outdir']).with_name('resume'),
                                  dataset['dataset_root'], dataset['action_npy'], 'cpu', 2)
    other = shared.manifest(cfg2)
    original = base.atomic_checkpoint

    def interrupt(path, blob):
        original(path, blob)
        if blob['epoch'] == 2:
            raise RuntimeError('simulated interruption after atomic checkpoint')

    monkeypatch.setattr(base, 'atomic_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='interruption'):
        control.train(other, 'clean')
    monkeypatch.setattr(base, 'atomic_checkpoint', original)
    control.train(other, 'clean')
    resumed = control.checkpoint(other, 'clean', final=True)
    for key in full['model']:
        assert torch.equal(full['model'][key], resumed['model'][key])
    assert torch.equal(full['rng']['torch'], resumed['rng']['torch'])
    assert full['train_loss'] == resumed['train_loss']
    with pytest.raises(FileNotFoundError):
        control.evaluate(m)
    control.train(m, 'a1')
    control.evaluate(m)
    result = control.verify_complete(m)
    assert result['status'] == control.STATUS
    assert len(result['a1_metrics']['dose_rows']) == 6
    assert control.load_manifest(Path(dataset['outdir'])/control.MANIFEST) == m
    # A v2 validator must not accept a v3 checkpoint, even if paths coincide.
    with pytest.raises(ValueError, match='Foreign'):
        previous.checkpoint(m, 'a1', final=True)
    path = Path(dataset['outdir'])/'evaluation.json'
    bad = copy.deepcopy(result); bad['a1_metrics']['mainmetrics']['target_pck_pct'] += 1
    base.atomic_json(path, bad)
    with pytest.raises(ValueError, match='replay'):
        control.verify_complete(m)
    base.atomic_json(path, result)
    bad = copy.deepcopy(m); bad['cfg']['lr'] = .01
    base.atomic_json(Path(dataset['outdir'])/control.MANIFEST, bad)
    with pytest.raises(ValueError, match='source/runtime/config'):
        control.load_manifest(Path(dataset['outdir'])/control.MANIFEST)


def test_no_old_output_or_receipt_reuse(dataset):
    with pytest.raises(ValueError, match='Historical'):
        control.configuration(dataset['data_home'],
                              Path(dataset['outdir']).with_name('piw_hpeli_body_axis_s42_v2'))
    old_cfg = previous.configuration(dataset['data_home'], dataset['outdir'],
                                     dataset['dataset_root'], dataset['action_npy'], 'cpu', 2)
    base.atomic_json(control.receipt_path(dataset, 'preflight'), dict(kind='preflight', status='PASS',
        plan_sha256=base.object_sha(previous.plan(old_cfg))))
    with pytest.raises(ValueError, match='different source/config/runtime'):
        control.receipt(dataset, 'preflight')
    assert not Path(dataset['outdir']).exists()


def test_control_dry_run_is_metadata_only(tmp_path):
    path = tmp_path/'no_output'
    code = subprocess.run([sys.executable, '-B', control.__file__, '--data-home', str(tmp_path/'missing'),
        '--outdir', str(path)], capture_output=True, text=True, timeout=30)
    assert code.returncode == 0, code.stderr
    assert 'DRY_RUN_ONLY' in code.stdout and not path.exists()
    payload = json.loads(code.stdout.split('\n', 1)[1])
    assert payload['cfg']['lr'] == .001
    assert payload['cfg']['target']['version'] == 'piw_body_hip_axis_v2'
    assert payload['cfg']['drop_last'] is True


def test_actual_paired_comparison_from_completed_toy_runs(dataset, monkeypatch):
    import compare_piw_lr_control as comparison
    old_cfg = previous.configuration(dataset['data_home'], Path(dataset['outdir']).with_name('v2_reference'),
                                     dataset['dataset_root'], dataset['action_npy'], 'cpu', 2)
    monkeypatch.setattr(shared, 'runner', previous)
    old = shared.manifest(old_cfg)
    for role in ('clean', 'a1'):
        previous.train(old, role)
    previous.evaluate(old)
    monkeypatch.setattr(shared, 'runner', control)
    new = shared.manifest(dataset)
    for role in ('clean', 'a1'):
        control.train(new, role)
    control.evaluate(new)
    # Actual saved artifacts from two complete controlled runs, not mocked metrics.
    report = comparison.compare_completed(old_cfg['outdir'], dataset['outdir'])
    assert report['n_samples'] == 3 and report['seed'] == 42
    assert len(report['rows']) == 7 and len(report['controls']) == 10
    assert old['poison'] == new['poison']
    assert old['trigger'] == new['trigger']
