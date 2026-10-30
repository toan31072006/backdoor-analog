"""Offline checks for downloading the author's exact PSBA action assets.

All network responses are in-memory fixtures, and all NPY fixtures are tiny.
These tests do not verify a full public file checksum or run an experiment.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'remote_tools' / 'Get-PSBAMotionPriors.py'


@pytest.fixture
def module():
    name = '_test_psba_motion_prior_download'
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def npy_bytes(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=True)
    return stream.getvalue()


@pytest.fixture
def tiny(module, monkeypatch):
    array = np.arange(1, 121, dtype='<f4').reshape(2, 3, 5, 4, 1)
    payload = npy_bytes(array)
    records = deepcopy(module.FILES)
    for record in records.values():
        record.update(bytes=len(payload), shape=list(array.shape))
    monkeypatch.setattr(module, 'FILES', records)
    return module, payload, array


class Response:
    def __init__(self, payload, *, status=206, headers=None):
        self.status = status
        self.headers = headers or {}
        self.stream = io.BytesIO(payload)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def read(self, amount):
        return self.stream.read(amount)


def opener_for(payload, *, status=206, start=0, total=None, headers=None):
    total = len(payload) + start if total is None else total
    values = {
        'Content-Type': 'application/octet-stream',
        'Content-Range': f'bytes {start}-{total - 1}/{total}',
    }
    if headers is not None:
        values.update(headers)
    response = Response(payload, status=status, headers=values)
    calls = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return response

    return opener, calls, response


def forbid_network(*args, **kwargs):
    raise AssertionError('No request was expected')


def paths(root, action='cross'):
    target = root / f'data_{action}.npy'
    partial = root / f'data_{action}.npy.download.part'
    return target, Path(str(target) + '.provenance.json'), partial, Path(str(partial) + '.source.json')


def create_partial(module, root, payload, *, action='cross'):
    _, _, partial, ownership = paths(root, action)
    partial.write_bytes(payload)
    ownership.write_text(json.dumps(module.source(action)), encoding='utf-8')
    return partial, ownership


def test_official_asset_id_size_and_shape_are_pinned(module):
    assert module.FILES['cross']['id'] == '1irgTzLOMkpfOQjNHI9omf56iJIeHq3dk'
    assert module.FILES['cross']['bytes'] == 225720128
    assert module.FILES['cross']['shape'] == [1254, 3, 300, 25, 2]
    assert module.FILES['nod']['id'] == '1aGItQm8dSMk0FweZQnAdX7IuAaKHQx6A'
    assert module.FILES['nod']['bytes'] == 237600128
    assert module.FILES['nod']['shape'] == [1320, 3, 300, 25, 2]
    assert module.FILES['bend']['id'] == '11EAVcLT0iZk2YeWPEXoSkfP66mVxi8uS'
    assert module.FILES['bend']['bytes'] == 227520128
    assert module.FILES['bend']['shape'] == [1264, 3, 300, 25, 2]


def test_source_links_to_author_and_does_not_invent_published_hash(module):
    value = module.source('cross')
    assert value['source_project'] == 'https://qichenzheng.github.io/psba-website/'
    assert '1ZdUKi9FpC3q-5CZmOx5lXOMyOEIcFJTU' in value['source_folder']
    assert value['file_id'] in value['source_url']
    assert 'sha256' not in value
    assert 'published_sha256' not in value


def test_validate_tiny_numeric_array_measures_sha256(tiny, tmp_path):
    module, payload, array = tiny
    path = tmp_path / 'input.npy'
    path.write_bytes(payload)
    value = module.validate(path, 'cross')
    assert value == dict(sha256=hashlib.sha256(payload).hexdigest(),
                         shape=list(array.shape), dtype='<f4', bytes=len(payload))


@pytest.mark.parametrize('variant', ['float64', 'object', 'shape', 'fortran', 'nan', 'inf', 'zero'])
def test_validate_rejects_unsafe_or_foreign_array(tiny, tmp_path, variant):
    module, _, array = tiny
    value = array.copy()
    if variant == 'float64':
        value = value.astype(np.float64)
    elif variant == 'object':
        value = value.astype(object)
    elif variant == 'shape':
        value = value.reshape(3, 2, 5, 4, 1)
    elif variant == 'fortran':
        value = np.asfortranarray(value)
    elif variant == 'nan':
        value.flat[4] = np.nan
    elif variant == 'inf':
        value.flat[4] = np.inf
    elif variant == 'zero':
        value.fill(0)
    payload = npy_bytes(value)
    module.FILES['cross']['bytes'] = len(payload)
    path = tmp_path / 'foreign.npy'
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        module.validate(path, 'cross')


@pytest.mark.parametrize('extra', [-1, 1])
def test_validate_requires_exact_official_byte_count(tiny, tmp_path, extra):
    module, payload, _ = tiny
    path = tmp_path / 'input.npy'
    path.write_bytes(payload[:-1] if extra < 0 else payload + b'x')
    with pytest.raises(ValueError, match='expected .* bytes'):
        module.validate(path, 'cross')


def test_fresh_download_has_bounded_range_and_measured_source_receipt(tiny, tmp_path, capsys):
    module, payload, array = tiny
    opener, calls, response = opener_for(payload)
    result = module.download('cross', tmp_path, opener=opener)
    target, receipt_path, partial, ownership = paths(tmp_path)
    assert result == target
    assert target.read_bytes() == payload
    assert not partial.exists()
    assert json.loads(ownership.read_text()) == module.source('cross')
    receipt = json.loads(receipt_path.read_text())
    assert receipt['sha256'] == hashlib.sha256(payload).hexdigest()
    assert receipt['shape'] == list(array.shape)
    assert receipt['origin'] == 'official_psba_download'
    assert receipt['semantic_approved'] is True
    assert receipt['semantic_label'] == 'crossing hands in front'
    assert 'measured' in receipt['sha256_status']
    assert 'no author-published SHA256' in receipt['sha256_status']
    assert 'published_sha256' not in receipt
    request, timeout = calls[0]
    assert len(calls) == 1
    assert request.get_header('Range') == f'bytes=0-{len(payload) - 1}'
    assert module.FILES['cross']['id'] in request.full_url
    assert request.full_url.startswith('https://drive.usercontent.google.com/download?')
    assert timeout == 60
    assert response.closed
    assert 'DOWNLOADED_AND_VALIDATED cross' in capsys.readouterr().out


def test_first_request_may_accept_http_200(tiny, tmp_path):
    module, payload, _ = tiny
    opener, _, _ = opener_for(payload, status=200)
    assert module.download('cross', tmp_path, opener=opener).read_bytes() == payload


def test_resume_requires_matching_source_and_exact_remaining_range(tiny, tmp_path):
    module, payload, _ = tiny
    offset = 147
    _, ownership = create_partial(module, tmp_path, payload[:offset])
    original_source = ownership.read_bytes()
    opener, calls, _ = opener_for(payload[offset:], start=offset)
    target = module.download('cross', tmp_path, opener=opener)
    assert calls[0][0].get_header('Range') == f'bytes={offset}-{len(payload) - 1}'
    assert target.read_bytes() == payload
    assert ownership.read_bytes() == original_source


def test_complete_owned_partial_is_validated_without_network(tiny, tmp_path):
    module, payload, _ = tiny
    create_partial(module, tmp_path, payload)
    assert module.download('cross', tmp_path, opener=forbid_network).read_bytes() == payload


def test_empty_owned_partial_can_continue_without_deleting_user_file(tiny, tmp_path):
    module, payload, _ = tiny
    _, ownership = create_partial(module, tmp_path, b'')
    original_source = ownership.read_bytes()
    opener, calls, _ = opener_for(payload)
    target = module.download('cross', tmp_path, opener=opener)
    assert target.read_bytes() == payload
    assert calls[0][0].get_header('Range') == f'bytes=0-{len(payload)-1}'
    assert ownership.read_bytes() == original_source


def test_resume_cannot_restart_or_overwrite_when_server_ignores_range(tiny, tmp_path):
    module, payload, _ = tiny
    original = payload[:100]
    partial, _ = create_partial(module, tmp_path, original)
    opener, _, _ = opener_for(payload, status=200)
    with pytest.raises(ValueError, match='honor resume'):
        module.download('cross', tmp_path, opener=opener)
    assert partial.read_bytes() == original
    assert not paths(tmp_path)[0].exists()


@pytest.mark.parametrize('content_range', ['', 'garbage', 'bytes 1-607/608',
                                         'bytes 0-606/608', 'bytes 0-607/609'])
def test_bad_content_range_is_rejected_before_writing(tiny, tmp_path, content_range):
    module, payload, _ = tiny
    if content_range.startswith('bytes'):
        # Generate bad values relative to fixture length, rather than relying
        # on a specific NumPy header length.
        index = ['bytes 1-607/608', 'bytes 0-606/608', 'bytes 0-607/609'].index(content_range)
        content_range = [f'bytes 1-{len(payload)-1}/{len(payload)}',
                         f'bytes 0-{len(payload)-2}/{len(payload)}',
                         f'bytes 0-{len(payload)-1}/{len(payload)+1}'][index]
    opener, _, _ = opener_for(payload, headers={'Content-Range': content_range})
    with pytest.raises(ValueError, match='unexpected/partial range'):
        module.download('cross', tmp_path, opener=opener)
    assert not paths(tmp_path)[0].exists()
    assert not paths(tmp_path)[2].exists()


@pytest.mark.parametrize('status', [403, 404, 429, 500])
def test_error_http_status_does_not_write_file(tiny, tmp_path, status):
    module, payload, _ = tiny
    opener, _, _ = opener_for(payload, status=status)
    with pytest.raises(ValueError):
        module.download('cross', tmp_path, opener=opener)
    assert not paths(tmp_path)[0].exists()
    assert not paths(tmp_path)[2].exists()


def test_html_login_or_quota_page_is_never_saved_as_action(tiny, tmp_path):
    module, payload, _ = tiny
    opener, _, _ = opener_for(b'<html>Login or quota</html>', total=len(payload),
                              headers={'Content-Type': 'text/html; charset=utf-8'})
    with pytest.raises(ValueError, match='login/quota/confirmation'):
        module.download('cross', tmp_path, opener=opener)
    assert not paths(tmp_path)[0].exists()
    assert not paths(tmp_path)[2].exists()


def test_non_npy_payload_with_binary_content_type_is_not_promoted(tiny, tmp_path):
    module, payload, _ = tiny
    opener, _, _ = opener_for(b'X' * len(payload))
    with pytest.raises(ValueError):
        module.download('cross', tmp_path, opener=opener)
    assert not paths(tmp_path)[0].exists()
    assert not paths(tmp_path)[1].exists()


def test_truncated_download_retains_owned_partial_for_resume(tiny, tmp_path):
    module, payload, _ = tiny
    opener, _, _ = opener_for(payload[:177], total=len(payload))
    with pytest.raises(ValueError, match='expected .* bytes'):
        module.download('cross', tmp_path, opener=opener)
    target, receipt, partial, ownership = paths(tmp_path)
    assert partial.read_bytes() == payload[:177]
    assert ownership.exists()
    assert not target.exists()
    assert not receipt.exists()


def test_oversized_response_cannot_be_promoted(tiny, tmp_path):
    module, payload, _ = tiny
    opener, _, _ = opener_for(payload + b'extra', total=len(payload))
    with pytest.raises(ValueError, match='exceeds official size'):
        module.download('cross', tmp_path, opener=opener)
    assert not paths(tmp_path)[0].exists()
    assert not paths(tmp_path)[1].exists()


def test_interrupted_network_keeps_partial_and_source_receipt(tiny, tmp_path):
    module, payload, _ = tiny
    response = Response(payload, headers={
        'Content-Type': 'application/octet-stream',
        'Content-Range': f'bytes 0-{len(payload)-1}/{len(payload)}',
    })
    read_calls = []

    def failing_read(amount):
        read_calls.append(amount)
        if len(read_calls) > 1:
            raise OSError('fixture disconnection')
        return payload[:120]

    response.read = failing_read
    with pytest.raises(OSError, match='fixture disconnection'):
        module.download('cross', tmp_path, opener=lambda request, timeout: response)
    target, receipt, partial, ownership = paths(tmp_path)
    assert partial.read_bytes() == payload[:120]
    assert json.loads(ownership.read_text()) == module.source('cross')
    assert not target.exists()
    assert not receipt.exists()
    assert response.closed


def test_existing_unsourced_action_is_never_overwritten_or_auto_approved(tiny, tmp_path):
    module, payload, _ = tiny
    target, receipt, _, _ = paths(tmp_path)
    target.write_bytes(payload)
    with pytest.raises(FileExistsError, match='without official-download provenance'):
        module.download('cross', tmp_path, opener=forbid_network)
    assert target.read_bytes() == payload
    assert not receipt.exists()


def test_existing_valid_sourced_action_is_verified_without_network(tiny, tmp_path):
    module, payload, _ = tiny
    opener, _, _ = opener_for(payload)
    target = module.download('cross', tmp_path, opener=opener)
    receipt = paths(tmp_path)[1].read_bytes()
    assert module.download('cross', tmp_path, opener=forbid_network) == target
    assert paths(tmp_path)[1].read_bytes() == receipt


@pytest.mark.parametrize('field,value', [('file_id', 'foreign'), ('source_project', 'https://foreign.invalid/'),
                                         ('origin', 'manual'), ('semantic_approved', False),
                                         ('sha256', '0' * 64), ('shape', [1])])
def test_foreign_existing_receipt_is_not_rewritten(tiny, tmp_path, field, value):
    module, payload, _ = tiny
    opener, _, _ = opener_for(payload)
    target = module.download('cross', tmp_path, opener=opener)
    receipt_path = paths(tmp_path)[1]
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    original = receipt_path.read_bytes()
    with pytest.raises(ValueError, match='provenance'):
        module.download('cross', tmp_path, opener=forbid_network)
    assert receipt_path.read_bytes() == original
    assert target.read_bytes() == payload


def test_unowned_partial_is_not_resumed(tiny, tmp_path):
    module, payload, _ = tiny
    partial = paths(tmp_path)[2]
    partial.write_bytes(payload[:90])
    with pytest.raises(FileExistsError, match='Unowned partial'):
        module.download('cross', tmp_path, opener=forbid_network)
    assert partial.read_bytes() == payload[:90]
    assert not paths(tmp_path)[3].exists()


def test_foreign_partial_ownership_is_not_changed(tiny, tmp_path):
    module, payload, _ = tiny
    partial, ownership = create_partial(module, tmp_path, payload[:90])
    ownership.write_text('{"file_id":"foreign"}', encoding='utf-8')
    original = ownership.read_bytes()
    with pytest.raises(ValueError, match='Existing receipt differs'):
        module.download('cross', tmp_path, opener=forbid_network)
    assert ownership.read_bytes() == original
    assert partial.read_bytes() == payload[:90]


def test_orphan_receipt_is_not_replaced(tiny, tmp_path):
    module, _, _ = tiny
    receipt = paths(tmp_path)[1]
    receipt.write_text('{"foreign":true}', encoding='utf-8')
    with pytest.raises(FileExistsError, match='Orphan provenance'):
        module.download('cross', tmp_path, opener=forbid_network)
    assert receipt.read_text() == '{"foreign":true}'


def test_partial_cannot_exceed_pinned_official_size(tiny, tmp_path):
    module, payload, _ = tiny
    partial, _ = create_partial(module, tmp_path, payload + b'x')
    with pytest.raises(ValueError, match='Partial file exceeds'):
        module.download('cross', tmp_path, opener=forbid_network)
    assert partial.read_bytes() == payload + b'x'


def test_write_new_is_idempotent_but_never_replaces_different_receipt(module, tmp_path):
    path = tmp_path / 'receipt.json'
    module.write_new(path, dict(value=1))
    original = path.read_bytes()
    module.write_new(path, dict(value=1))
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='no overwrite'):
        module.write_new(path, dict(value=2))
    assert path.read_bytes() == original


def test_default_cli_only_verifies_cross_and_nod_without_download(tiny, tmp_path, monkeypatch, capsys):
    module, _, _ = tiny
    calls = []
    monkeypatch.setattr(module, 'validate', lambda path, action: calls.append(('validate', action)) or {})
    monkeypatch.setattr(module, 'validate_receipt', lambda path, action, measured: calls.append(('receipt', action)))
    monkeypatch.setattr(module, 'download', forbid_network)
    module.main(['--outdir', str(tmp_path)])
    assert calls == [('validate', 'cross'), ('receipt', 'cross'),
                     ('validate', 'nod'), ('receipt', 'nod')]
    assert 'no environment installation, GPU use, or training' in capsys.readouterr().out


def test_cli_download_requires_explicit_switch_and_defaults_to_two_missing_priors(tiny, tmp_path, monkeypatch):
    module, _, _ = tiny
    calls = []
    monkeypatch.setattr(module, 'download', lambda action, outdir: calls.append((action, outdir)))
    module.main(['--outdir', str(tmp_path), '--download'])
    assert calls == [('cross', str(tmp_path)), ('nod', str(tmp_path))]


@pytest.mark.parametrize('priors', [['cross', 'cross'], ['foreign']])
def test_cli_rejects_duplicate_or_unknown_prior_before_work(tiny, tmp_path, monkeypatch, priors):
    module, _, _ = tiny
    monkeypatch.setattr(module, 'download', forbid_network)
    with pytest.raises(SystemExit) as error:
        module.main(['--outdir', str(tmp_path), '--download', '--priors', *priors])
    assert error.value.code == 2
