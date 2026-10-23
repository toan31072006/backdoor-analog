"""Synthetic input/model/preparation contracts; no real dataset/GPU claim."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from wipose_data import (WiPoseDataset, dataset_audit, next_frame_name,
                         read_frame, read_names)
from prepare_wipose import convert_record, prepare


def write_frame(path, csi=None, pose=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if csi is None:
        csi = np.arange(1350, dtype=np.float64).reshape(3, 90, 5) + 1
    if pose is None:
        pose = np.arange(36, dtype=np.float64).reshape(2, 18) + 1
    np.savez(path, CSI=csi, SkeletonPoints=pose)


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / 'processed'
    root.mkdir()
    for prefix, names in (
        ('Train', ['walk_001-frame003', 'walk_001-frame001', 'walk_001-frame002']),
        ('Test', ['bend_002-frame017', 'bend_002-frame020'])):
        (root / f'{prefix}_data_list.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
        for name in names:
            write_frame(root / f'{prefix}_Amplitude_DWT' / f'{name}.npz')
    return root


def test_official_filter_order_shapes_and_values(dataset_root):
    data = WiPoseDataset(dataset_root, 'training')
    assert [item['name'] for item in data.items] == ['walk_001-frame001', 'walk_001-frame002']
    sample = data[0]
    assert sample['csi'].shape == (3, 90, 5)
    assert sample['pose'].shape == (1, 18, 2)
    assert sample['csi'].dtype == sample['pose'].dtype == np.float32
    assert sample['csi'].min() == 0 and sample['csi'].max() == 1
    np.testing.assert_allclose(sample['pose'][0], (np.arange(36).reshape(2,18)+1).T*.001)
    test = WiPoseDataset(dataset_root, 'validation')
    assert [item['name'] for item in test.items] == ['bend_002-frame017', 'bend_002-frame020']


def test_next_frame_preserves_author_zfill():
    assert next_frame_name('walk_001-frame009') == 'walk_001-frame010'
    assert next_frame_name('walk_001-frame0999') == 'walk_001-frame1000'


@pytest.mark.parametrize('value', ['../walk_001-frame001', 'walk_001-frame001.npz',
    'walk_001-frame1', 'unknown_001-frame001', '/walk_001-frame001'])
def test_unsafe_identity_rejected(tmp_path, value):
    path = tmp_path / 'list.txt'
    path.write_text(value, encoding='utf-8')
    with pytest.raises(ValueError):
        read_names(path)


@pytest.mark.parametrize('value', ['', 'walk_001-frame001\n\nwalk_001-frame002',
                                     'walk_001-frame001\nwalk_001-frame001'])
def test_empty_blank_duplicate_lists_rejected(tmp_path, value):
    path = tmp_path / 'list.txt'
    path.write_text(value, encoding='utf-8')
    with pytest.raises(ValueError):
        read_names(path)


@pytest.mark.parametrize('kind', ['constant', 'nan', 'complex', 'shape', 'object', 'badpose'])
def test_unsafe_numeric_arrays_rejected(tmp_path, kind):
    path = tmp_path / 'bad.npz'
    csi = np.arange(1350, dtype=float).reshape(3,90,5)
    pose = np.arange(36, dtype=float).reshape(2,18)
    if kind == 'constant': csi.fill(1)
    if kind == 'nan': csi[0,0,0] = np.nan
    if kind == 'complex': csi = csi.astype(complex)
    if kind == 'shape': csi = csi.reshape(3,45,10)
    if kind == 'object': csi = csi.astype(object)
    if kind == 'badpose': pose = pose.T
    write_frame(path, csi, pose)
    with pytest.raises(ValueError):
        read_frame(path)


def test_missing_file_and_schema_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_frame(tmp_path / 'missing.npz')
    np.savez(tmp_path / 'bad.npz', something=np.zeros(3))
    with pytest.raises(ValueError):
        read_frame(tmp_path / 'bad.npz')


def test_audit_binds_all_data_and_is_read_only(dataset_root):
    before = {path.relative_to(dataset_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in dataset_root.rglob('*') if path.is_file()}
    audit = dataset_audit(dataset_root)
    assert audit['splits']['training']['official_count'] == 3
    assert audit['splits']['training']['retained_count'] == 2
    assert len(audit['splits']['training']['files']) == 3
    json.dumps(audit, allow_nan=False)
    assert before == {path.relative_to(dataset_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in dataset_root.rglob('*') if path.is_file()}
    path = dataset_root / 'Train_Amplitude_DWT/walk_001-frame003.npz'
    write_frame(path, pose=np.arange(36.).reshape(2,18)+2)
    assert dataset_audit(dataset_root)['fingerprint'] != audit['fingerprint']


def test_audit_rejects_missing_filtered_final_frame(dataset_root):
    (dataset_root / 'Train_Amplitude_DWT/walk_001-frame003.npz').unlink()
    with pytest.raises(FileNotFoundError):
        dataset_audit(dataset_root)


def test_degenerate_pck_scale_fails_before_training(dataset_root):
    path = dataset_root / 'Train_Amplitude_DWT/walk_001-frame003.npz'
    pose = np.arange(36.).reshape(2,18)
    pose[:,8] = pose[:,5]
    write_frame(path, pose=pose)
    with pytest.raises(ValueError, match='PCK reference scale'):
        read_frame(path)
    with pytest.raises(ValueError, match='PCK reference scale'):
        dataset_audit(dataset_root)


def test_audit_rejects_identity_overlap(dataset_root):
    name = 'walk_001-frame001'
    (dataset_root / 'Test_data_list.txt').write_text(name, encoding='utf-8')
    write_frame(dataset_root / f'Test_Amplitude_DWT/{name}.npz')
    with pytest.raises(ValueError, match='overlap'):
        dataset_audit(dataset_root)


def test_dtpose_preparation_is_exact_active_branch():
    csi = np.arange(1350.).reshape(5,30,3,3)
    points = np.arange(54.).reshape(3,18)
    converted, schema, transform = convert_record(dict(CSI=csi, SkeletonPoints=points))
    assert schema == 'dtpose' and 'no DWT' in transform
    np.testing.assert_array_equal(converted['CSI'], csi.transpose(3,2,1,0).reshape(3,90,5))
    np.testing.assert_array_equal(converted['SkeletonPoints'], points[:2])


@pytest.mark.parametrize('shape', [(18,4), (4,18), (72,), (1,72)])
def test_official_complex_amplitude_and_joint_order(shape):
    csi = np.arange(1350.).reshape(5,30,3,3) + 2j
    points = np.arange(72.).reshape(18,4)
    supplied = points.T if shape == (4,18) else points.reshape(shape)
    converted, schema, _ = convert_record(dict(csi_serial=csi, jointsVector=supplied))
    assert schema == 'official'
    np.testing.assert_array_equal(converted['CSI'], abs(csi).transpose(3,2,1,0).reshape(3,90,5))
    np.testing.assert_array_equal(converted['SkeletonPoints'], points[:,:2].T)


@pytest.mark.parametrize('record', [dict(), dict(CSI=np.ones((5,30,3,3)), SkeletonPoints=np.ones(54)),
    dict(CSI=np.ones((5,30,3,3), complex), SkeletonPoints=np.ones(54)),
    dict(csi_serial=np.ones((5,30,3,3)), jointsVector=np.ones(36))])
def test_bad_raw_schema_rejected(record):
    with pytest.raises(ValueError):
        convert_record(record)


def raw_fixture(root, with_lists=True):
    from scipy.io import savemat
    for prefix, names in (('Train', ['walk_001-frame001', 'walk_001-frame002']),
                          ('Test', ['bend_002-frame001'])):
        folder = root / prefix
        folder.mkdir(parents=True)
        if with_lists:
            (root / f'{prefix}_data_list.txt').write_text('\n'.join(names)+'\n', encoding='utf-8')
        for name in names:
            savemat(folder / f'{name}.mat', dict(CSI=np.arange(1350.).reshape(5,30,3,3),
                                                SkeletonPoints=np.arange(54.).reshape(3,18)))


def test_preparation_full_fixture_no_raw_writes(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'processed'
    raw_fixture(raw)
    before = {path: path.read_bytes() for path in raw.rglob('*') if path.is_file()}
    manifest = prepare(raw, output)
    assert manifest['raw_schema'] == 'dtpose'
    assert manifest['inputs'] == dataset_audit(output)
    assert manifest['inputs']['splits']['training']['retained_count'] == 1
    assert (output / 'preparation.json').is_file()
    assert before == {path: path.read_bytes() for path in before}
    with pytest.raises(FileExistsError):
        prepare(raw, output)


def test_missing_original_lists_need_explicit_generation(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'processed'
    raw_fixture(raw, with_lists=False)
    with pytest.raises(FileNotFoundError, match='generate-lists'):
        prepare(raw, output)
    assert not output.exists()
    result = prepare(raw, output, generate_lists=True)
    assert 'order adaptation' in result['list_provenance']['training']['origin']


def test_prepare_nested_roots_rejected(tmp_path):
    raw_fixture(tmp_path / 'raw')
    with pytest.raises(ValueError, match='nonnested'):
        prepare(tmp_path / 'raw', tmp_path / 'raw/processed')


def test_explicit_resume_after_interruption_never_rewrites_valid_frame(tmp_path, monkeypatch):
    import prepare_wipose as module
    raw, output = tmp_path / 'raw', tmp_path / 'processed'
    raw_fixture(raw)
    original = module.convert_record
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interrupted CPU conversion')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'convert_record', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        prepare(raw, output)
    first = output / 'Train_Amplitude_DWT/walk_001-frame001.npz'
    before = (first.read_bytes(), first.stat().st_mtime_ns)
    intent = json.loads((output / 'preparation.inprogress.json').read_text(encoding='utf-8'))
    assert len(intent['raw_files']) == 3
    assert set(intent['sources']) == {'prepare_wipose.py', 'wipose_data.py'}
    monkeypatch.setattr(module, 'convert_record', original)
    with pytest.raises(FileExistsError):
        prepare(raw, output)
    result = prepare(raw, output, resume=True)
    assert result['intent_fingerprint'] == intent['fingerprint']
    assert (first.read_bytes(), first.stat().st_mtime_ns) == before
    final = output / 'preparation.json'
    final_before = (final.read_bytes(), final.stat().st_mtime_ns)
    assert prepare(raw, output, resume=True) == result
    assert (final.read_bytes(), final.stat().st_mtime_ns) == final_before


def test_resume_rejects_unbound_existing_directory(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'processed'
    raw_fixture(raw)
    output.mkdir()
    with pytest.raises(ValueError, match='no bound'):
        prepare(raw, output, resume=True)


def test_resume_requires_existing_output(tmp_path):
    with pytest.raises(FileNotFoundError, match='existing bound'):
        prepare(tmp_path / 'raw', tmp_path / 'processed', resume=True)


@pytest.mark.parametrize('change', ['raw', 'list', 'schema', 'source', 'intent', 'final', 'frame'])
def test_resume_rejects_changed_binding_or_outputs(tmp_path, monkeypatch, change):
    import prepare_wipose as module
    from scipy.io import savemat
    raw, output = tmp_path / 'raw', tmp_path / 'processed'
    raw_fixture(raw)
    prepare(raw, output)
    options = {}
    if change == 'raw':
        savemat(raw / 'Train/walk_001-frame001.mat', dict(CSI=np.arange(1350.).reshape(5,30,3,3)+1,
                                                    SkeletonPoints=np.arange(54.).reshape(3,18)))
    elif change == 'list':
        (raw / 'Train_data_list.txt').write_text('walk_001-frame002\nwalk_001-frame001\n', encoding='utf-8')
    elif change == 'schema':
        options['schema'] = 'dtpose'
    elif change == 'source':
        original = module.file_sha
        monkeypatch.setattr(module, 'file_sha', lambda path: '0'*64 if str(path).endswith('wipose_data.py') else original(path))
    elif change == 'intent':
        path = output / 'preparation.inprogress.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['raw_root'] = 'other'
        path.write_text(json.dumps(value), encoding='utf-8')
    elif change == 'final':
        path = output / 'preparation.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['raw_schema'] = 'official'
        path.write_text(json.dumps(value), encoding='utf-8')
    elif change == 'frame':
        write_frame(output / 'Train_Amplitude_DWT/walk_001-frame001.npz', pose=np.ones((2,18)))
    with pytest.raises(ValueError, match='binding changed|metadata differs|resume validation'):
        prepare(raw, output, resume=True, **options)


def test_partial_saving_file_is_preserved_and_not_overwritten(tmp_path, monkeypatch):
    import prepare_wipose as module
    raw, output = tmp_path / 'raw', tmp_path / 'processed'
    raw_fixture(raw)
    original = module._publish_exclusive
    def interrupted(path, writer):
        if str(path).endswith('.npz'):
            Path(str(path) + '.saving').write_bytes(b'partial interrupted data')
            raise RuntimeError('simulated interrupted publication')
        return original(path, writer)
    monkeypatch.setattr(module, '_publish_exclusive', interrupted)
    with pytest.raises(RuntimeError, match='publication'):
        prepare(raw, output)
    monkeypatch.setattr(module, '_publish_exclusive', original)
    partial = output / 'Train_Amplitude_DWT/walk_001-frame001.npz.saving'
    before = partial.read_bytes()
    with pytest.raises(FileExistsError, match='partial .saving'):
        prepare(raw, output, resume=True)
    assert partial.read_bytes() == before


def test_complex_nonfinite_components_cannot_be_hidden_by_amplitude():
    csi = np.arange(1350.).reshape(5,30,3,3).astype(complex)
    csi[0,0,0,0] = complex(1, np.nan)
    with pytest.raises(ValueError, match='Nonfinite'):
        convert_record(dict(csi_serial=csi, jointsVector=np.arange(72.)))


def test_model_shapes_backward_and_initialization():
    import torch
    from models.wipose_hpeli import build_wipose_hpeli
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_wipose_hpeli()
    inputs = torch.randn(2,3,90,5)
    predicted, features = model(inputs)
    assert predicted.shape == (2,1,18,2) and features.shape == (2,128)
    predicted.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    for layer in model.modules():
        if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            torch.testing.assert_close(layer.weight, torch.ones_like(layer.weight))
            torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))
    with pytest.raises(ValueError):
        model(torch.zeros(2,3,90,10))


def test_original_author_active_forward_backward_parity_when_available():
    """Inspect only source-pinned model definitions; no script main is executed."""
    import torch
    from models.wipose_hpeli import build_wipose_hpeli
    upstream = Path('E:/wifi/_audit_dt_pose')
    if not upstream.is_dir():
        pytest.skip('Optional original-author local checkout not available')
    pins = json.loads((HERE / 'models/wipose_sources.json').read_text(encoding='utf-8'))
    namespace = dict(torch=torch, nn=torch.nn, time=time)
    for relative in ('model/hpeli/SK_network.py', 'model/hpeli/hpeli.py'):
        source = (upstream / relative).read_text(encoding='utf-8')
        assert hashlib.sha256(source.replace('\r\n','\n').encode()).hexdigest() == pins['sha256_git_blob_lf'][relative]
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), relative, 'exec'), namespace)
    torch.set_num_threads(2)
    adapted = build_wipose_hpeli().eval()
    author = namespace['hpelinet'](18, 2, 90, dataset='wipose').eval()
    state = adapted.state_dict()
    assert set(state) - {'pool'} <= set(author.state_dict())
    author.load_state_dict(state, strict=False)
    one = torch.randn(2,3,90,5, requires_grad=True)
    two = one.detach().clone().requires_grad_(True)
    actual, feature = adapted(one)
    expected, reference_feature = author(two)
    torch.testing.assert_close(actual[:,0], expected, rtol=0, atol=0)
    torch.testing.assert_close(feature, reference_feature, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(one.grad, two.grad, rtol=0, atol=0)
    for name, parameter in adapted.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(author.named_parameters())[name].grad,
                                   rtol=0, atol=0)
