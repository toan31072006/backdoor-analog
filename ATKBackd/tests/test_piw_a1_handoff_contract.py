"""CPU/fixture-only handoff contracts; no remote tasks or real-data efficacy.

The expensive completed Clean/A1 fixture is shared within this module.  Each
test restores only its own temporary A1 publication artifacts; the Clean
reference and raw inputs must remain byte- and timestamp-identical throughout.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_piw_hpeli_a1_runner as support

runner, clean = support.runner, support.clean


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    patcher = pytest.MonkeyPatch()
    fixture = support.fixture.__wrapped__(
        tmp_path_factory.mktemp('piw_a1_handoff'), patcher)
    try:
        manifest = support.complete(fixture)
        root = Path(manifest['cfg']['outdir'])
        snapshot = {str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob('*') if path.is_file()}
        yield dict(fixture=fixture, manifest=manifest, snapshot=snapshot,
                   clean_tree=support.tree(fixture['clean']),
                   input_tree=support.tree(fixture['root']))
    finally:
        patcher.undo()


@pytest.fixture
def state(completed):
    fixture, snapshot = completed['fixture'], completed['snapshot']
    root = fixture['out']
    yield completed
    assert support.tree(fixture['clean']) == completed['clean_tree']
    assert support.tree(fixture['root']) == completed['input_tree']
    for path in root.rglob('*'):
        if path.is_file() and str(path.relative_to(root)) not in snapshot:
            path.unlink()
    for relative, data in snapshot.items():
        path = root / relative
        if not path.is_file() or path.read_bytes() != data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def forbid_training_and_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Publication recovery must not construct/infer/train a model or use CUDA')
    monkeypatch.setattr(runner, 'build_model', forbidden)
    monkeypatch.setattr(clean, 'build_model', forbidden)
    monkeypatch.setattr(clean, '_setup_execution', forbidden)
    monkeypatch.setattr(runner, '_infer', forbidden)
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)


def publication_paths(manifest):
    root, folder = Path(manifest['cfg']['outdir']), runner.cell_dir(manifest)
    return folder / 'evaluation.json', root / 'a1_result.json'


def test_preflight_smoke_checks_real_private_pair_phase_identity_and_metrics(state, monkeypatch):
    manifest, fixture = state['manifest'], state['fixture']
    _, datasets, poisoned, trigger = runner.prepare(manifest)
    before = support.tree(fixture['out'])

    def forbidden(*args, **kwargs):
        pytest.fail('Preflight must not initialize CUDA or construct an optimizer')
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(clean, '_setup_execution', forbidden)
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    report = runner.preflight_smoke(datasets, poisoned, trigger)
    assert report['actual_poison_pair'] is True
    assert report['phase_unchanged'] is True
    assert report['dose_zero_identity'] is True
    assert report['six_dose_metrics'] is True
    json.dumps(report, allow_nan=False)
    assert support.tree(fixture['out']) == before


def test_preflight_prints_counts_hash_but_never_private_poison_samples(state, monkeypatch, capsys):
    fixture, manifest = state['fixture'], state['manifest']
    capsys.readouterr()
    before = support.tree(fixture['out'])
    monkeypatch.setattr(clean, 'cpu_model_smoke', lambda: {'finite': True})
    monkeypatch.setattr(torch.cuda, 'is_available',
                        lambda: pytest.fail('Preflight CUDA probe'))
    monkeypatch.setattr(torch.optim, 'AdamW',
                        lambda *a, **k: pytest.fail('Preflight optimizer update'))
    assert runner.main(support.cli(fixture, '--preflight')) == 0
    printed = capsys.readouterr().out
    assert '"n_total": 12' in printed and '"n_poison": 1' in printed
    assert manifest['poison']['poison_plan_sha256'] in printed
    assert '"samples"' not in printed
    for sample in manifest['poison']['samples']:
        assert sample['name'] not in printed
    assert support.tree(fixture['out']) == before


@pytest.mark.parametrize('missing_index', [0, 1])
def test_valid_partial_publication_repairs_without_model_or_inference(state, monkeypatch, missing_index):
    manifest = state['manifest']
    paths = publication_paths(manifest)
    missing = paths[missing_index]
    missing.unlink()
    assert runner.verify_complete(manifest) is None
    survivor = runner.verify_complete(manifest, allow_partial=True)
    assert survivor['checkpoint_epoch'] == 199
    npz = runner.cell_dir(manifest) / 'a1_predictions.npz'
    original_npz = npz.read_bytes()
    forbid_training_and_inference(monkeypatch)
    assert runner.recover_partial_completion(manifest) is True
    assert runner.verify_complete(manifest) == survivor
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert npz.read_bytes() == original_npz
    assert (Path(manifest['cfg']['outdir']) / 'a1_pilot.md').is_file()
    assert (Path(manifest['cfg']['outdir']) / 'a1_six_doses.csv').is_file()
    assert (Path(manifest['cfg']['outdir']) / 'a1_diagnostics.json').is_file()


def test_repeated_repair_interruption_preserves_npz_and_retries_publication(state, monkeypatch):
    manifest = state['manifest']
    root = Path(manifest['cfg']['outdir'])
    _, missing = publication_paths(manifest)
    missing.unlink()
    for name in ('a1_pilot.md', 'a1_six_doses.csv', 'a1_diagnostics.json'):
        (root / name).unlink()
    npz = runner.cell_dir(manifest) / 'a1_predictions.npz'
    original_npz = npz.read_bytes()
    original_atomic = runner.atomic_json
    calls = []

    def interrupt_after_write(path, value):
        original_atomic(path, value)
        calls.append(Path(path))
        raise RuntimeError('simulated interruption after the first repair JSON write')

    forbid_training_and_inference(monkeypatch)
    monkeypatch.setattr(runner, 'atomic_json', interrupt_after_write)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        runner.recover_partial_completion(manifest)
    assert len(calls) == 1
    assert npz.read_bytes() == original_npz
    # The second attempt sees valid paired completion JSONs, but must still
    # restore reports.  Interrupt again, now after diagnostics publication.
    with pytest.raises(RuntimeError, match='simulated interruption'):
        runner.recover_partial_completion(manifest)
    assert len(calls) == 2
    assert npz.read_bytes() == original_npz
    monkeypatch.setattr(runner, 'atomic_json', original_atomic)
    runner.recover_partial_completion(manifest)
    assert runner.verify_complete(manifest) is not None
    assert npz.read_bytes() == original_npz
    for name in ('a1_pilot.md', 'a1_six_doses.csv', 'a1_diagnostics.json'):
        assert (root / name).is_file()


def test_invalid_partial_npz_cannot_be_repaired_or_re_evaluated(state, monkeypatch):
    manifest = state['manifest']
    _, missing = publication_paths(manifest)
    missing.unlink()
    npz = runner.cell_dir(manifest) / 'a1_predictions.npz'
    npz.write_bytes(npz.read_bytes() + b'\n')
    before = support.tree(Path(manifest['cfg']['outdir']))
    forbid_training_and_inference(monkeypatch)
    with pytest.raises(ValueError, match='artifact fingerprint'):
        runner.verify_complete(manifest, allow_partial=True)
    with pytest.raises(ValueError, match='artifact fingerprint'):
        runner.recover_partial_completion(manifest)
    assert not missing.exists()
    assert support.tree(Path(manifest['cfg']['outdir'])) == before


def test_no_completion_metadata_never_invents_results(state, monkeypatch):
    manifest = state['manifest']
    for path in publication_paths(manifest):
        path.unlink()
    before = support.tree(Path(manifest['cfg']['outdir']))
    forbid_training_and_inference(monkeypatch)
    assert runner.verify_complete(manifest, allow_partial=True) is None
    assert runner.recover_partial_completion(manifest) is False
    assert support.tree(Path(manifest['cfg']['outdir'])) == before
