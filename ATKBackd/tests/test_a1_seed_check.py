"""Read-only seed diagnostic tests, without real data, GPU or training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_a1_three_tables as check
import run_a1_three_tables as frozen


@pytest.fixture
def existing(tmp_path):
    matrix = frozen.build_matrix(tmp_path / 'data', tmp_path / 'out',
                                 retrain_seed42=True)
    root = Path(matrix['outdir'])
    root.mkdir(parents=True)
    manifest = root / frozen.MANIFEST
    manifest.write_text(json.dumps(matrix), encoding='utf-8')
    return matrix, manifest


def _snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).rglob('*') if path.is_file()}


def _artifacts(matrix, seeds=(42, 43)):
    for cell in matrix['cells']:
        if cell['seed'] in seeds:
            artifact = Path(cell['artifact_dir']) / 'evaluation.json'
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{}', encoding='utf-8')


def _forbidden(*args, **kwargs):
    pytest.fail('Read-only seed check attempted a mutation/training/GPU operation')


def test_default_seeds():
    assert check.parse_args(['--outdir', 'saved']).seeds == [42, 43]


@pytest.mark.parametrize('extra', [[], ['--seeds', '42', '42'],
                                  ['--seeds', '41'], ['--seeds'],
                                  ['--dry-run'], ['--export-only'],
                                  ['--jobs-per-gpu', '2']])
def test_invalid_options(extra):
    argv = [] if not extra else ['--outdir', 'saved', *extra]
    with pytest.raises(SystemExit):
        check.parse_args(argv)


def test_all_selected_verified_once_and_seed44_untouched(existing, monkeypatch, capsys):
    matrix, _ = existing
    _artifacts(matrix)
    calls = []
    monkeypatch.setattr(check.parallel, '_completed',
                        lambda cell: calls.append(cell['method_key']) or True)
    monkeypatch.setattr(check.parallel, '_schedule', _forbidden)
    monkeypatch.setattr(frozen, '_preflight', _forbidden)
    monkeypatch.setattr(frozen, '_cell_lock', _forbidden)
    monkeypatch.setattr(frozen, '_export', _forbidden)
    before = _snapshot(Path(matrix['outdir']))
    assert check.main(['--outdir', matrix['outdir']]) == 0
    assert len(calls) == len(set(calls)) == 26
    assert all(not name.startswith('s44__') for name in calls)
    assert _snapshot(Path(matrix['outdir'])) == before
    output = capsys.readouterr().out
    assert 'seed 42: VERIFIED 13/13' in output
    assert 'seed 43: VERIFIED 13/13' in output
    assert 'ALL_SELECTED_SEEDS_VERIFIED' in output
    assert not Path(matrix['data_home']).exists()


def test_requested_seed44_is_verified_without_seed42_or43(existing, monkeypatch, capsys):
    matrix, _ = existing
    _artifacts(matrix, seeds=(44,))
    calls = []
    monkeypatch.setattr(check.parallel, '_completed',
                        lambda cell: calls.append(cell['seed']) or True)
    assert check.main(['--outdir', matrix['outdir'], '--seeds', '44']) == 0
    assert calls == [44] * 13
    assert 'ALL_SELECTED_SEEDS_VERIFIED' in capsys.readouterr().out


def test_missing_is_not_verified_or_audited(existing, monkeypatch, capsys):
    matrix, _ = existing
    monkeypatch.setattr(check.parallel, '_completed', _forbidden)
    before = _snapshot(Path(matrix['outdir']))
    assert check.main(['--outdir', matrix['outdir']]) == 1
    assert _snapshot(Path(matrix['outdir'])) == before
    output = capsys.readouterr().out
    assert output.count('MISSING s') == 26
    assert 'VERIFIED 0/13' in output
    assert 'ALL_SELECTED_SEEDS_VERIFIED' not in output


@pytest.mark.parametrize('failure', ['false', 'value', 'runtime', 'file'])
def test_invalid_cell_does_not_hide_other_cells(existing, monkeypatch, capsys, failure):
    matrix, _ = existing
    _artifacts(matrix)
    calls = []

    def completed(cell):
        calls.append(cell['method_key'])
        if cell['method_key'] == 's42__a1':
            if failure == 'false':
                return False
            raise {'value': ValueError, 'runtime': RuntimeError,
                   'file': FileNotFoundError}[failure]('binding invalid')
        return True

    monkeypatch.setattr(check.parallel, '_completed', completed)
    assert check.main(['--outdir', matrix['outdir']]) == 1
    assert len(calls) == 26
    output = capsys.readouterr().out
    assert 'INVALID s42__a1' in output
    assert 'seed 42: VERIFIED 12/13; MISSING 0; INVALID 1' in output
    assert 'seed 43: VERIFIED 13/13' in output
    assert 'ALL_SELECTED_SEEDS_VERIFIED' not in output


@pytest.mark.parametrize('mutation', ['hash', 'source', 'rehashed_config', 'outdir'])
def test_bad_plan_never_reaches_audit(existing, monkeypatch, capsys, mutation):
    matrix, manifest = existing
    if mutation == 'hash':
        matrix['plan_sha256'] = '0' * 64
    elif mutation == 'source':
        matrix['sources']['train_backdoor.py'] = '0' * 64
        matrix['plan_sha256'] = frozen._plan_hash(matrix)
    elif mutation == 'rehashed_config':
        matrix['cells'][0]['cfg']['epochs'] = 15
        matrix['plan_sha256'] = frozen._plan_hash(matrix)
    else:
        # Valid canonical plan at a different root must not be accepted as a copy.
        other = frozen.build_matrix(Path(matrix['data_home']), manifest.parent / 'other',
                                    retrain_seed42=True)
        matrix = other
    manifest.write_text(json.dumps(matrix), encoding='utf-8')
    before = _snapshot(manifest.parent)
    monkeypatch.setattr(check.parallel, '_completed', _forbidden)
    assert check.main(['--outdir', str(manifest.parent)]) == 1
    assert '[seed-check] ERROR:' in capsys.readouterr().err
    assert _snapshot(manifest.parent) == before


def test_missing_manifest_does_not_create_output(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'absent'
    monkeypatch.setattr(check.parallel, '_completed', _forbidden)
    assert check.main(['--outdir', str(root)]) == 1
    assert not root.exists()
    assert 'Existing manifest required' in capsys.readouterr().err


def test_exact_seed_cell_set_required(existing, monkeypatch, capsys):
    matrix, manifest = existing
    matrix['cells'] = [cell for cell in matrix['cells']
                       if cell['method_key'] != 's43__a4']
    manifest.write_text(json.dumps(matrix), encoding='utf-8')
    monkeypatch.setattr(frozen, '_validate_manifest', lambda matrix: None)
    monkeypatch.setattr(check.parallel, '_completed', _forbidden)
    assert check.main(['--outdir', str(manifest.parent)]) == 1
    assert 'exact 13 planned cells' in capsys.readouterr().err


def test_bytecode_writes_disabled():
    assert sys.dont_write_bytecode is True
