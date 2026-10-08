"""CPU-only checks of pinned BackdoorBench operators and CSI adaptations.

These tests use synthetic arrays and resolved configurations, without training,
datasets, or network access.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from attack.traditional import (  # noqa: E402
    BadNetsTrigger, BlendedTrigger, build_traditional_trigger,
)
from third_party.backdoorbench.blended import blendedImageAttack  # noqa: E402
from third_party.backdoorbench.patch import AddMaskPatchTrigger  # noqa: E402


PINNED_COMMIT = "f02e3534645f0ee63d6848653062cd6c0d6c400d"
VENDORED = HERE / "third_party" / "backdoorbench"
UPSTREAM_PATHS = {
    "badnets": "utils/bd_img_transform/patch.py",
    "blended": "utils/bd_img_transform/blended.py",
}
UPSTREAM_SHA256 = {
    "badnets": "b6eb898626e80cc852156324a65d0a1b1fc694e9b3c19792eaa2336085f3720d",
    "blended": "e0820614234ebbc856405982ae7c543224b1dc750391b572baafa56ec98b56b5",
}
IMPLEMENTATIONS = {
    "badnets": "backdoorbench-mask-patch-v1",
    "blended": "backdoorbench-blend-v1",
}


def _csi(shape=(3, 5, 4)):
    # Each axis has a different contribution. A transpose error or mixing
    # antenna channels cannot hide behind a constant or symmetric fixture.
    antenna, frequency, packet = np.indices(shape)
    return (0.073 + 0.191 * antenna + 0.029 * frequency
            + 0.0071 * packet).astype(np.float32)


def _hwc(csi):
    return np.transpose(csi, (1, 2, 0))


def _chw(image):
    return np.transpose(image, (2, 0, 1))


def _manifest():
    return json.loads((VENDORED / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_upstream_patch_zero_is_transparent_and_nonzero_overwrites_float_hwc():
    image = _hwc(_csi()).copy()
    original = image.copy()
    pattern = np.zeros_like(image)
    pattern[1, 3, 0] = 0.31729
    pattern[4, 0, 1] = 0.64183
    pattern[2, 1, 2] = 1.0
    pattern_before = pattern.copy()
    operator = AddMaskPatchTrigger(pattern)

    result = operator(image)
    expected = np.where(pattern == 0, image, pattern)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[pattern == 0], original[pattern == 0])
    np.testing.assert_array_equal(result[pattern != 0], pattern[pattern != 0])
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(pattern, pattern_before)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, image)
    assert float(result[1, 3, 0]) == float(pattern[1, 3, 0])


def test_upstream_all_zero_patch_is_an_independent_exact_identity():
    image = _hwc(_csi())
    result = AddMaskPatchTrigger(np.zeros_like(image))(image)
    np.testing.assert_array_equal(result, image)
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize("alpha", [0.0, 0.074, 0.185, 0.4, 1.0])
def test_upstream_blended_is_exact_float_hwc_convex_mix(alpha):
    image = _hwc(_csi()).copy()
    pattern = np.random.default_rng(42).uniform(0.0, 1.0, image.shape).astype(np.float32)
    image_before, pattern_before = image.copy(), pattern.copy()

    result = blendedImageAttack(pattern, alpha)(image)
    np.testing.assert_array_equal(result, (1.0 - alpha) * image + alpha * pattern)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(pattern, pattern_before)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, image)
    assert not np.shares_memory(result, pattern)


def test_badnets_default_is_a_fixed_white_eight_by_three_patch():
    trigger = BadNetsTrigger(seed=42)
    expected_mask = np.zeros((3, 114, 10), dtype=bool)
    expected_mask[:, -8:, -3:] = True
    np.testing.assert_array_equal(trigger.mask, expected_mask)
    np.testing.assert_array_equal(trigger.pattern, expected_mask.astype(np.float32))
    np.testing.assert_array_equal(BadNetsTrigger(seed=43).pattern, trigger.pattern)
    assert isinstance(trigger.operator, AddMaskPatchTrigger)

    csi = _csi(trigger.shape)
    # Use a unit-range asymmetric fixture even for the larger MMFi shape.
    csi /= float(csi.max())
    result = trigger.inject(csi, dose=1.0, eps=0.185)
    np.testing.assert_array_equal(result[trigger.mask], np.ones(72, dtype=np.float32))
    np.testing.assert_array_equal(result[~trigger.mask], csi[~trigger.mask])


@pytest.mark.parametrize("dose", [1.0, 0.4])
def test_badnets_adapter_matches_direct_operator_without_antenna_mixing(dose):
    csi = _csi()
    original = csi.copy()
    trigger = BadNetsTrigger(n_ant=3, n_sub=5, n_pkt=4, seed=42,
                            patch_subcarriers=2, patch_packets=2,
                            patch_start=(1, 2), antennas=(1, 2))
    direct = _chw(AddMaskPatchTrigger(_hwc(trigger.pattern))(_hwc(csi)))
    expected = csi.copy()
    expected[trigger.mask] = ((1.0 - dose) * csi[trigger.mask]
                             + dose * direct[trigger.mask])

    result = trigger.inject(csi, dose=dose, eps=0.185)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(trigger.operator(_hwc(csi)), _hwc(direct))
    np.testing.assert_array_equal(result[0], csi[0])
    np.testing.assert_array_equal(result[~trigger.mask], csi[~trigger.mask])
    np.testing.assert_array_equal(csi, original)
    assert result.shape == csi.shape
    assert result.dtype == np.float32
    assert not np.shares_memory(result, csi)
    assert not np.shares_memory(result, trigger.pattern)


@pytest.mark.parametrize("dose", [1.0, 0.4])
def test_blended_adapter_matches_direct_operator_without_antenna_mixing(dose):
    csi = _csi()
    original = csi.copy()
    trigger = BlendedTrigger(n_ant=3, n_sub=5, n_pkt=4, seed=42)
    expected_pattern = np.random.default_rng(42).uniform(0.0, 1.0, csi.shape).astype(np.float32)
    np.testing.assert_array_equal(trigger.pattern, expected_pattern)
    expected = _chw(blendedImageAttack(_hwc(expected_pattern), dose * 0.185)(_hwc(csi)))

    result = trigger.inject(csi, dose=dose, eps=0.185)
    np.testing.assert_array_equal(result, expected)
    for antenna in range(3):
        np.testing.assert_array_equal(result[antenna],
                                      (1.0 - dose * 0.185) * csi[antenna]
                                      + dose * 0.185 * expected_pattern[antenna])
    np.testing.assert_array_equal(csi, original)
    assert result.shape == csi.shape
    assert result.dtype == np.float32
    assert not np.shares_memory(result, csi)
    assert not np.shares_memory(result, trigger.pattern)


@pytest.mark.parametrize("cls,operator", [
    (BadNetsTrigger, AddMaskPatchTrigger),
    (BlendedTrigger, blendedImageAttack),
])
def test_adapters_call_the_vendored_operator_with_hwc_arrays(monkeypatch, cls, operator):
    original_add_trigger = operator.add_trigger
    inputs = []

    def record(self, image):
        inputs.append(image.copy())
        return original_add_trigger(self, image)

    monkeypatch.setattr(operator, "add_trigger", record)
    csi = _csi()
    kwargs = {"n_sub": 5, "n_pkt": 4}
    if cls is BadNetsTrigger:
        kwargs.update(patch_subcarriers=2, patch_packets=2)
    trigger = cls(**kwargs)
    trigger.inject(csi, dose=0.4, eps=0.185)
    assert len(inputs) == 1
    np.testing.assert_array_equal(inputs[0], _hwc(csi))


@pytest.mark.parametrize("eps", [0.0, 0.001, 0.185, 1.0, 2.0])
@pytest.mark.parametrize("dose", [0.4, 1.0])
def test_badnets_epsilon_does_not_scale_patch_opacity(eps, dose):
    csi = _csi((3, 114, 10))
    csi /= float(csi.max())
    trigger = BadNetsTrigger()
    np.testing.assert_array_equal(trigger.inject(csi, dose=dose, eps=eps),
                                  trigger.inject(csi, dose=dose, eps=0.185))


@pytest.mark.parametrize("opacity,dose", [(0.0, 1.0), (0.25, 0.4), (0.5, 1.0), (1.0, 3.0)])
def test_badnets_optional_opacity_and_dose_saturation(opacity, dose):
    csi = _csi()
    trigger = BadNetsTrigger(n_sub=5, n_pkt=4, patch_subcarriers=2,
                            patch_packets=2, patch_opacity=opacity)
    alpha = min(dose * opacity, 1.0)
    expected = csi.copy()
    expected[trigger.mask] = ((1.0 - alpha) * csi[trigger.mask]
                             + alpha * trigger.pattern[trigger.mask])
    np.testing.assert_array_equal(trigger.inject(csi, dose=dose, eps=0.185), expected)


@pytest.mark.parametrize("opacity", [-0.1, 1.01, np.nan, np.inf, -np.inf])
def test_badnets_rejects_invalid_patch_opacity(opacity):
    with pytest.raises(ValueError):
        BadNetsTrigger(patch_opacity=opacity)


@pytest.mark.parametrize("cls", [BadNetsTrigger, BlendedTrigger])
def test_adapter_zero_dose_is_exact_identity_with_independent_storage(cls):
    csi = np.random.default_rng(7).uniform(0.0, 1.0, (3, 114, 10)).astype(np.float32)
    trigger = cls(seed=42)
    result = trigger.inject(csi, dose=0.0, eps=0.185)
    np.testing.assert_array_equal(result, csi)
    assert not np.shares_memory(result, csi)
    assert not np.shares_memory(result, trigger.pattern)
    result[0, 0, 0] = 0.0
    assert csi[0, 0, 0] != 0.0


def test_blended_zero_epsilon_identity_and_saturation():
    csi = _csi()
    trigger = BlendedTrigger(n_sub=5, n_pkt=4)
    result = trigger.inject(csi, dose=1.0, eps=0.0)
    np.testing.assert_array_equal(result, csi)
    assert not np.shares_memory(result, csi)
    saturated = trigger.inject(csi, dose=3.0, eps=0.5)
    np.testing.assert_array_equal(saturated, trigger.pattern)
    assert not np.shares_memory(saturated, trigger.pattern)


@pytest.mark.parametrize("eps", [-0.1, np.nan, np.inf, -np.inf])
def test_badnets_still_validates_epsilon(eps):
    with pytest.raises(ValueError):
        BadNetsTrigger().inject(np.zeros((3, 114, 10), dtype=np.float32), 1.0, eps)


def test_source_manifest_binds_both_operators_to_the_pinned_upstream_commit():
    manifest = _manifest()
    assert manifest["schema"] == 1
    assert manifest["commit"] == PINNED_COMMIT
    assert "BackdoorBench" in manifest["repository"]
    assert manifest["license"]
    assert (VENDORED / "LICENSE").is_file()
    assert (VENDORED / "NOTICE.md").is_file()
    for method, path in UPSTREAM_PATHS.items():
        source = manifest["files"][path]
        assert source["sha256"] == UPSTREAM_SHA256[method]
        assert PINNED_COMMIT in source["url"]
        assert source["url"].endswith(path)
        assert (VENDORED / source["local_file"]).is_file()
        assert source["operator"] in {"AddMaskPatchTrigger", "blendedImageAttack"}


@pytest.mark.parametrize("method", ["badnets", "blended"])
def test_matrix_cells_bind_implementation_commit_and_source_digests(tmp_path, method):
    import run_mmfi_tables as runner

    matrix = runner.build_matrix(tmp_path / "absent_data", tmp_path / "runs", "cpu", 0)
    cell = next(cell for cell in matrix["cells"] if cell["method_key"] == method)
    cfg = cell["cfg"]
    source = _manifest()["files"][UPSTREAM_PATHS[method]]
    assert cfg[f"{method}_implementation"] == IMPLEMENTATIONS[method]
    assert cfg[f"{method}_source_commit"] == PINNED_COMMIT
    assert cfg[f"{method}_source_sha256"] == source["sha256"]
    assert cfg[f"{method}_operator_sha256"] == _sha256(VENDORED / source["local_file"])
    assert cfg[f"{method}_adapter_sha256"] == _sha256(HERE / "attack" / "traditional.py")
    assert not (tmp_path / "absent_data").exists()
    assert not (tmp_path / "runs").exists()
    assert isinstance(build_traditional_trigger(method, cfg),
                      BadNetsTrigger if method == "badnets" else BlendedTrigger)


@pytest.mark.parametrize("method", ["badnets", "blended"])
@pytest.mark.parametrize("field,value", [
    ("implementation", "unknown-implementation-v999"),
    ("source_commit", "0" * 40),
    ("source_sha256", "0" * 64),
    ("operator_sha256", "0" * 64),
    ("adapter_sha256", "0" * 64),
])
def test_factory_rejects_unknown_versions_and_mismatched_source_metadata(method, field, value):
    cfg = {"experiment_name": "mmfi", f"{method}_{field}": value}
    with pytest.raises(ValueError):
        build_traditional_trigger(method, cfg)


@pytest.mark.parametrize('method', ['badnets', 'blended'])
def test_direct_trainer_materializes_source_before_hash_and_rejects_legacy_cache(tmp_path, method):
    import train_backdoor as trainer

    raw = {'experiment_name': 'mmfi', 'model': 'hpeli',
           'trigger': method, 'epochs': 50}
    current = trainer._resolve_training_config(raw)
    # Simulate the same recipe before integration: no source or new white-patch
    # metadata. This used to hash identically despite different trigger code.
    legacy = {key: value for key, value in current.items()
              if not key.startswith(method + '_')}
    blob = {'result_schema': trainer._RESULT_SCHEMA,
            'cfg_fingerprint': trainer._config_fingerprint(legacy),
            'trained_epochs': 50, 'cfg': legacy, 'res': {'legacy_result': True}}
    cache = tmp_path / 'eval_cache.json'
    cache.write_text(json.dumps(blob), encoding='utf-8')
    before = cache.read_bytes()
    assert trainer._config_fingerprint(current) != blob['cfg_fingerprint']
    assert trainer._load_cached_result(tmp_path, current) is None
    assert cache.read_bytes() == before


@pytest.mark.parametrize('method', ['badnets', 'blended'])
def test_direct_trainer_rejects_stale_source_markers_before_loading_data(method):
    import train_backdoor as trainer

    cfg = {'experiment_name': 'mmfi', 'model': 'hpeli', 'trigger': method,
           method + '_source_commit': 'stale-independent-version'}
    with pytest.raises(ValueError, match='source metadata mismatch'):
        trainer._resolve_training_config(cfg)


def test_proposed_wanet_and_rf_configs_do_not_receive_changed_baseline_markers():
    import train_backdoor as trainer

    for name in ('micro_dropper', 'wanet', 'infocom2025_por', 'ccai2026_backdoorrf'):
        cfg = trainer._resolve_training_config(
            {'experiment_name': 'mmfi', 'model': 'hpeli', 'trigger': name})
        assert not any(key.startswith(('badnets_', 'blended_')) for key in cfg)
    assert trainer._CHECKPOINT_SCHEMA == 9
    assert trainer._RESULT_SCHEMA == 10


def test_main_source_entries_match_vendor_manifest_and_disclose_third_party_adapter():
    import run_mmfi_tables as runner

    manifest = _manifest()
    for method, path in UPSTREAM_PATHS.items():
        source = runner.SOURCES[method]
        assert source['commit'] == manifest['commit']
        assert source['repository'] == manifest['repository']
        assert source['license'] == manifest['license']
        assert source['source_file'] == path
        assert source['source_sha256'] == manifest['files'][path]['sha256']
        assert 'Third-party' in '; '.join(source['limitations'])
        assert 'Calls vendored' in source['implementation']
