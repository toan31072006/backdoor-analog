"""Frozen-reference plumbing tests; no victim or dataset is loaded."""
import copy
import hashlib
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a1_frozen_reference as reference


def test_pinned_records_folder_and_zip_are_identical_without_extraction(tmp_path, monkeypatch):
    data = {'proposed_ablation.resolved.json': b'{"x":1}',
            'full/eval_cache.json': b'{"y":2}'}
    monkeypatch.setattr(reference, 'RECORD_SHA256', {
        n: hashlib.sha256(b).hexdigest() for n, b in data.items()})
    folder = tmp_path / 'original'
    for name, content in data.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    archive = tmp_path / 'reference.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name, content in data.items():
            z.writestr('original/' + name, content)
    assert reference._records(folder) == reference._records(archive) == data
    assert sorted(p.name for p in tmp_path.iterdir()) == ['original', 'reference.zip']
    (folder / 'full/eval_cache.json').write_bytes(b'{"y":3}')
    with pytest.raises(ValueError, match='frozen reference changed'):
        reference._records(folder)


def test_zip_duplicate_manifest_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, 'RECORD_SHA256', {'proposed_ablation.resolved.json': 'unused'})
    path = tmp_path / 'duplicate.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('a/proposed_ablation.resolved.json', b'{}')
        z.writestr('b/proposed_ablation.resolved.json', b'{}')
    with pytest.raises(ValueError, match='exactly one'):
        reference._records(path)


@pytest.fixture
def shared():
    cfg = {k: 'frozen-' + k for k in reference.COMMON_KEYS}
    inputs = dict(train=dict(n=133056, pair_ids_sha256='a'*64),
                  eval=dict(n=33264, pair_ids_sha256='b'*64),
                  action_file_sha256='c'*64, poison_plan_sha256='d'*64)
    frozen = dict(cfg=cfg, audit=inputs)
    matrix = dict(cells=[dict(method_key='test_operator', cfg=copy.deepcopy(cfg))])
    audit = dict(inputs={'test_operator': copy.deepcopy(inputs)},
                 poison_plan_sha256={'test_operator': 'd'*64})
    return frozen, matrix, audit


def test_validate_runtime_and_operator_differences_do_not_relabel_reference(shared):
    frozen, matrix, audit = shared
    original = copy.deepcopy(frozen)
    matrix['cells'][0]['cfg'].update(trigger='other_native_operator',
        num_workers=0, loader_persistent_workers=False, clean_label_cover_ratio=.2,
        traditional_comparison_profile='new_comparison')
    reference.validate_a1_reference(frozen, matrix)
    reference.validate_a1_reference(frozen, matrix, audit)
    assert frozen == original


@pytest.mark.parametrize('key', reference.COMMON_KEYS)
def test_validate_rejects_every_changed_scientific_setting(shared, key):
    frozen, matrix, audit = shared
    matrix['cells'][0]['cfg'][key] = 'changed'
    with pytest.raises(ValueError, match='scientific settings differ'):
        reference.validate_a1_reference(frozen, matrix, audit)


@pytest.mark.parametrize('name', ['train', 'eval', 'action_file_sha256', 'poison'])
def test_validate_rejects_different_inputs_or_poison(shared, name):
    frozen, matrix, audit = shared
    if name == 'poison':
        audit['poison_plan_sha256']['test_operator'] = 'e'*64
    else:
        audit['inputs']['test_operator'][name] = {'different': True}
    with pytest.raises(ValueError, match='differ from frozen A1'):
        reference.validate_a1_reference(frozen, matrix, audit)


def test_validate_forbids_external_peak_projection(shared):
    frozen, matrix, audit = shared
    matrix['cells'][0]['cfg']['comparison_peak_budget'] = 'original_postclip_linf_v1'
    with pytest.raises(ValueError, match='must not project'):
        reference.validate_a1_reference(frozen, matrix, audit)


def test_preflight_identity_requires_all_original_hashes(shared):
    frozen, _, audit = shared
    record = audit['inputs']['test_operator']
    reference.validate_a1_input_record(frozen, record)
    for name in ('train', 'eval', 'action_file_sha256'):
        incomplete = dict(record)
        incomplete.pop(name)
        with pytest.raises(ValueError, match='identities differ'):
            reference.validate_a1_input_record(frozen, incomplete)


def test_actual_user_artifact_if_available_is_read_only():
    archive = Path(r'C:\Users\Admin\Downloads\mmfi_proposed_factorial_ablation_s42_v1_results.zip')
    if not archive.is_file():
        pytest.skip('Local user artifact is deliberately not vendored in the repository')
    stat = archive.stat()
    frozen = reference.load_a1_reference(archive)
    assert frozen['row']['method_key'] == 'a1_reference'
    assert frozen['row']['t1_mpjpe_mm'] == pytest.approx(155.3979218006134)
    assert frozen['row']['clean_mpjpe_mm'] == pytest.approx(174.68443512916565)
    assert frozen['row']['clean_pampjpe_mm'] == pytest.approx(104.599646961329)
    assert frozen['cfg']['confirmation_profile'] == reference.PROFILE
    assert frozen['provenance']['plan_sha256'] == reference.PLAN
    assert len(frozen['dose_response']) == 6
    assert archive.stat().st_mtime_ns == stat.st_mtime_ns
