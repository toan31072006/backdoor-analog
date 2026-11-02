"""Independent CPU/synthetic operator tests; not full MM-Fi/CUDA efficacy."""
from copy import deepcopy
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F


SPEC = importlib.util.spec_from_file_location(
    "adap_patch_operator_under_test",
    Path(__file__).resolve().parents[1] / "ATKBackd" / "attack" / "adap_patch_mmfi.py",
)
op = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = op
SPEC.loader.exec_module(op)


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def assets():
    generator = torch.Generator().manual_seed(19)
    training = []
    for index, (name, alpha) in enumerate(zip(op.TRAIN_NAMES, op.TRAIN_ALPHAS)):
        mask = torch.zeros(32, 32)
        mask[2 + index:19 + index, 3 + index:26] = 1.
        training.append({"name": name, "alpha": alpha,
                         "mark": torch.rand(3, 32, 32, generator=generator), "mask": mask})
    by_name = {item["name"]: item for item in training}
    return {"train": training,
            "test": [{**deepcopy(by_name[name]), "alpha": 1.} for name in op.TEST_NAMES],
            "asset_sha256": {"fixture_only": "not-author-assets"}}


def test_source_preset_is_complete_gtsrb_not_missing_cifar():
    config = op.source_config()
    assert config["source_preset"] == "gtsrb"
    assert config["dataset_is_not_gtsrb"] is True
    assert config["commit"] == "e7c070344a5c6004d6fbcdf0720e2d60ec808953"
    assert config["train_alphas"] == [.5, .2, .5, .3]
    assert config["test_names"] == ["firefox_corner_32.png", "trojan_square_32.png"]
    assert config["test_alphas"] == [1., 1.]
    assert "phoenix_corner2_32.png absent" in config["cifar_preset_excluded"]
    assert config["comparison_cap"] is None
    json.dumps(config, allow_nan=False)
    config["asset_sha256"].clear()
    assert len(op.ASSET_SHA256) == 5


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 7, 11, 13, 24057, 48114])
def test_rank_groups_exact_source_boundary(count):
    expected = []
    for rank in range(count):
        expected.append(next(j for j in range(4) if rank < (j + 1) * (count / 4)))
    assert np.array_equal(op.rank_groups(count), np.asarray(expected, dtype=np.int64))


def test_cover_plan_preserves_poison_pairing_and_is_disjoint():
    ids = np.array([8, 1, 19, 3], dtype=np.int64)
    doses = np.array([.2, .4, .6, 1.])
    original_ids, original_doses = ids.copy(), doses.copy()
    first = op.cover_group_plan(25, ids, doses)
    second = op.cover_group_plan(25, ids, doses)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert np.array_equal(first["poison_ids"], np.sort(ids))
    assert dict(zip(first["poison_ids"], first["poison_doses"])) == dict(zip(ids, doses))
    assert len(first["cover_ids"]) == 5
    assert len(set(first["cover_ids"]) & set(ids)) == 0
    assert len(np.unique(first["cover_ids"])) == 5
    assert np.all(np.diff(first["cover_ids"]) > 0)
    assert np.all((first["cover_doses"] >= .2) & (first["cover_doses"] <= 1.))
    assert np.array_equal(first["cover_groups"], op.rank_groups(5))
    assert np.array_equal(ids, original_ids) and np.array_equal(doses, original_doses)


def test_full_p3s3_count_budget_is_reportable_not_equal_modified_data():
    count = 240570
    ids = np.arange(24057, dtype=np.int64)
    plan = op.cover_group_plan(count, ids, np.full(len(ids), .6))
    assert len(plan["poison_ids"]) == 24057
    assert len(plan["cover_ids"]) == 48114
    assert len(plan["poison_ids"]) + len(plan["cover_ids"]) == 72171


@pytest.mark.parametrize("ids,doses", [([0, 0], [.5, .5]), ([10], [.5]), ([-1], [.5]),
                                        ([1.5], [.5]), ([1], [.1]), ([1], [float("nan")]),
                                        ([1], [.3, .4]), ([[1]], [[.5]])])
def test_bad_poison_plans_rejected(ids, doses):
    with pytest.raises(ValueError):
        op.cover_group_plan(10, ids, doses)


def test_cover_plan_empty_and_insufficient():
    plan = op.cover_group_plan(3, np.empty(0, dtype=np.int64), [], cover_rate=0.)
    assert all(len(value) == 0 for value in plan.values())
    with pytest.raises(ValueError, match="Insufficient"):
        op.cover_group_plan(3, [0, 1], [.5, .5], cover_rate=.9)


def resized(entry):
    mark = F.interpolate(entry["mark"][None], (114, 10), mode="bilinear", align_corners=False)[0]
    mask = F.interpolate(entry["mask"][None, None], (114, 10), mode="nearest")[0]
    return mark, mask


@pytest.mark.parametrize("group", range(4))
def test_train_blend_matches_native_formula_after_explicit_resize(assets, group):
    trigger = op.AdapPatchCSI(assets)
    x = torch.linspace(0, 1, 3 * 114 * 10).reshape(3, 114, 10)
    original = x.clone()
    dose = .6
    entry = assets["train"][group]
    mark, mask = resized(entry)
    expected = x + entry["alpha"] * torch.tensor(dose, dtype=x.dtype) * mask * (mark - x)
    result = trigger.inject_train(x, dose, group)
    assert torch.equal(result, expected)
    assert torch.equal(x, original)
    assert torch.equal(result[:, mask[0] == 0], x[:, mask[0] == 0])
    assert (result >= 0).all() and (result <= 1).all()


def test_test_composes_both_native_strong_triggers_in_order(assets):
    trigger = op.AdapPatchCSI(assets)
    x = torch.full((2, 3, 114, 10), .4)
    doses = torch.tensor([.2, 1.])
    expected = x.clone()
    for entry in assets["test"]:
        mark, mask = resized(entry)
        expected = expected + doses[:, None, None, None] * mask * (mark - expected)
    result = trigger.inject_test(x, doses)
    assert torch.equal(result, expected)
    backwards = x.clone()
    for entry in reversed(assets["test"]):
        mark, mask = resized(entry)
        backwards += doses[:, None, None, None] * mask * (mark - backwards)
    assert not torch.equal(result, backwards)


def test_zero_dose_exact_identity_including_signed_zero(assets):
    trigger = op.AdapPatchCSI(assets)
    x = torch.rand(4, 3, 114, 10)
    x[:, :, 0, 0] = -0.
    for actual in (trigger.inject_train(x, 0., torch.arange(4)), trigger.inject_test(x, 0.)):
        assert actual.numpy().tobytes() == x.numpy().tobytes()
        assert actual.data_ptr() != x.data_ptr()


def test_vectorized_groups_and_doses_equal_individual_calls(assets):
    trigger = op.AdapPatchCSI(assets)
    x = torch.rand(4, 3, 114, 10, dtype=torch.float64)
    doses = torch.tensor([0., .2, .7, 1.], dtype=torch.float64)
    groups = torch.tensor([3, 2, 1, 0])
    result = trigger.inject_train(x, doses, groups)
    assert result.dtype == x.dtype
    for row in range(4):
        assert torch.equal(result[row], trigger.inject_train(x[row], doses[row], groups[row]))


def test_input_gradient_and_no_learned_trigger_parameters(assets):
    trigger = op.AdapPatchCSI(assets)
    x = torch.full((2, 3, 114, 10), .3, requires_grad=True)
    result = trigger.inject_train(x, [.4, 1.], [0, 3])
    result.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(not entry["mark"].requires_grad and not entry["mask"].requires_grad
               for entry in trigger._train + trigger._test)


@pytest.mark.parametrize("dose", [-.1, 1.1, float("nan"), float("inf"), [.5, .6]])
def test_bad_doses_rejected(assets, dose):
    with pytest.raises(ValueError, match="dose"):
        op.AdapPatchCSI(assets).inject_test(torch.rand(3, 114, 10), dose)


@pytest.mark.parametrize("group", [-1, 4, .2, True, [0, 1]])
def test_bad_groups_rejected(assets, group):
    with pytest.raises(ValueError, match="group"):
        op.AdapPatchCSI(assets).inject_train(torch.rand(3, 114, 10), .5, group)


def test_invalid_inputs_and_source_geometry_rejected(assets):
    trigger = op.AdapPatchCSI(assets)
    for x in (torch.zeros(3, 114, 10, dtype=torch.int64), torch.zeros(3, 32, 32),
              torch.full((3, 114, 10), float("nan")), torch.full((3, 114, 10), 1.01)):
        with pytest.raises(ValueError):
            trigger.inject_test(x, .5)
    with pytest.raises(ValueError, match="locked"):
        op.AdapPatchCSI(assets, shape=(3, 32, 32))
    assets["train"][0]["mask"].zero_()
    with pytest.raises(ValueError, match="vanished"):
        op.AdapPatchCSI(assets)


def test_summary_is_stable_json_and_not_mutable_by_caller(assets):
    trigger = op.AdapPatchCSI(assets)
    before = trigger.state_summary()
    json.dumps(before, allow_nan=False)
    assets["train"][0]["mark"].zero_()
    assets["train"][0]["mask"].zero_()
    assert trigger.state_summary() == before
    assert before["train"][0]["mask_nonzero"] > 0
    before["asset_sha256"].clear()
    assert trigger.state_summary()["asset_sha256"]


def png_payload(value=100, *, size=(32, 32), mode="RGB"):
    stream = BytesIO()
    Image.new(mode, size, color=value).save(stream, format="PNG")
    return stream.getvalue()


def fake_asset_pins(monkeypatch):
    payloads = {name: png_payload(index + 40) for index, name in enumerate(op.ASSET_SHA256)}
    for name, payload in payloads.items():
        monkeypatch.setitem(op.ASSET_SHA256, name, hashlib.sha256(payload).hexdigest())
    return payloads


def test_runtime_asset_download_is_pinned_and_cache_is_immutable(tmp_path, monkeypatch):
    payloads = fake_asset_pins(monkeypatch)
    requests = []
    def fake_open(request, timeout):
        requests.append(request.full_url)
        assert timeout == 60
        assert op.SOURCE_COMMIT in request.full_url
        return BytesIO(payloads[request.full_url.rsplit("/", 1)[1]])
    monkeypatch.setattr(op, "urlopen", fake_open)
    first = op.prepare_assets(tmp_path)
    assert len(requests) == 5
    mtimes = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    assert not any(path.name.endswith(".part") for path in tmp_path.iterdir())
    second = op.prepare_assets(tmp_path)
    assert len(requests) == 5
    assert {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()} == mtimes
    assert torch.equal(first["train"][0]["mark"], second["train"][0]["mark"])
    target = tmp_path / op.TRAIN_NAMES[0]
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        op.prepare_assets(tmp_path)
    assert target.read_bytes() == b"tampered"


def test_load_assets_uses_explicit_mask_channel_not_nonblack_fallback(tmp_path, monkeypatch):
    payloads = fake_asset_pins(monkeypatch)
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
    loaded = op.load_assets(tmp_path)
    patch = loaded["train"][2]
    mask_rgb = op._decode_png(payloads["mask_badnet_patch4_32.png"], "mask")
    assert torch.equal(patch["mask"], mask_rgb[0])
    assert not torch.equal(patch["mask"], (patch["mark"] > 0).any(0).float())
    for index in (0, 1, 3):
        entry = loaded["train"][index]
        assert torch.equal(entry["mask"], (entry["mark"] > 0).any(0).float())


def test_bad_geometry_is_rejected_even_with_matching_sha(monkeypatch):
    name = op.TRAIN_NAMES[0]
    payload = png_payload(size=(31, 32))
    monkeypatch.setitem(op.ASSET_SHA256, name, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="32x32"):
        op._verified_payload(payload, name)


def test_bad_download_is_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(op, "urlopen", lambda *args, **kwargs: BytesIO(b"not pinned"))
    with pytest.raises(ValueError, match="SHA256"):
        op.prepare_assets(tmp_path)
    assert list(tmp_path.iterdir()) == []
