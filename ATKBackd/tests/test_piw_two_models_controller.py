"""Controller scheduling/result-binding tests without PiW data or CUDA."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "remote_linux/piw3d_two_models/controller.py"
spec = importlib.util.spec_from_file_location("piw_two_models_controller", SOURCE)
assert spec and spec.loader
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_six_baselines_plus_a1_clean_with_three_slots_per_gpu():
    waves = controller.waves()
    assert [len(wave) for wave in waves] == [6, 6, 4]
    assert controller.JOBS_PER_GPU == 3
    assert len({cell["id"] for wave in waves for cell in wave}) == 16
    for index, wave in enumerate(waves):
        for gpu in ("cuda:0", "cuda:1"):
            assigned = [cell for cell in wave if cell["device"] == gpu]
            assert len(assigned) == (3 if index < 2 else 2)
            assert {cell["backbone"] for cell in assigned} == {"metafiplusplus", "wiflow"}
    for index in (0, 1):
        per_gpu = [sum(cell['backbone'] == 'metafiplusplus'
                       for cell in waves[index] if cell['device'] == gpu)
                   for gpu in ('cuda:0', 'cuda:1')]
        assert sorted(per_gpu) == [1, 2]
    assert [sum(cell['backbone'] == 'metafiplusplus'
                for cell in waves[index] if cell['device'] == 'cuda:0')
            for index in (0, 1)] == [2, 1]


@pytest.mark.parametrize('wave_index,workers', [(0, 6), (1, 6), (2, 4)])
def test_smoke_barrier_matches_live_wave_size(tmp_path, monkeypatch, wave_index, workers):
    seen = []

    class FakeProcess:
        pid = 12345

        def __init__(self, command, *, stdout, stderr, env):
            seen.append(int(env['PIW_SMOKE_COUNT']))
            name = env['PIW_SMOKE_CELL']
            root = Path(env['PIW_SMOKE_BARRIER_DIR'])
            for phase in ('ready', 'done'):
                (root / phase / f'{name}.json').write_text('{}', encoding='utf-8')
            stdout.write('GPU_SMOKE_WORKER_PASS\n')

        def wait(self):
            return 0

    monkeypatch.setattr(controller.subprocess, 'Popen', FakeProcess)
    args = SimpleNamespace(outdir=tmp_path, repo_dir=tmp_path,
                           dataset_root=tmp_path, action_npy=tmp_path)
    wave = controller.waves()[wave_index]
    result = controller.run_wave(args, wave, mode='smoke', wave_index=wave_index,
                                 attempt='a' * 32)
    assert seen == [workers] * workers
    assert len(result['cells']) == workers


def test_completed_worker_preflight_is_reused_with_bound_input_stats(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    action = tmp_path / 'action.npy'
    action.write_bytes(b'authentic-action-placeholder')
    out = tmp_path / 'run'
    out.mkdir()
    receipt = out / 'preflight_receipt.json'
    receipt.write_text(json.dumps(dict(status='PASS', plan_sha256='plan',
        input_fingerprint='content', input_stat_fingerprint='stats')),
        encoding='utf-8')
    (out / 'plan.json').write_text('{}', encoding='utf-8')
    controller.require_prepare_output(out)

    def fake_worker(args, mode, *, capture):
        if mode == '--preflight-only':
            report = dict(status='PASS_EXISTING', receipt_path=str(receipt),
                          receipt_sha256=controller.file_sha(receipt),
                          input_fingerprint='content', plan_sha256='plan')
        elif mode == '--check-input-stats':
            report = dict(plan_sha256='plan', input_stat_fingerprint='stats')
        else:
            raise AssertionError(mode)
        return SimpleNamespace(stdout=json.dumps(report))

    monkeypatch.setattr(controller, 'run_worker', fake_worker)
    monkeypatch.setattr(controller, 'paths', lambda args: (tmp_path, dataset, action, out))
    result = controller.preflight(SimpleNamespace(outdir=out))
    assert result['input_stat_fingerprint'] == 'stats'

    (out / 'smoke.log').write_text('unexpected', encoding='utf-8')
    with pytest.raises(ValueError, match='completed worker preflight'):
        controller.require_prepare_output(out)


def test_result_export_requires_complete_bound_artifacts(tmp_path):
    gate = {"preflight": {"plan_sha256": "test-science-plan"}}
    for backbone in controller.BACKBONES:
        for method in ("clean", "a1", "wanet_source", "ftrojan", "fiba",
                       "bpp_source", "smooth_lf_csi", "tsba_adapted"):
            folder = tmp_path / "attacks" / backbone / "seed42" / method
            folder.mkdir(parents=True)
            (folder / "checkpoint.pt").write_bytes(b"ckpt")
            (folder / "predictions.npz").write_bytes(b"paired")
            metrics = dict(clean_mpjpe_mm=100., clean_pampjpe_mm=60.)
            if method != "clean":
                metrics.update(t1_mpjpe_mm=140., target_pck_pct=55., mean_dce_mm=70.)
            report = dict(status="COMPLETE", backbone=backbone, method=method,
                          plan_sha256="test-science-plan",
                          checkpoint_sha256=controller.file_sha(folder / "checkpoint.pt"),
                          predictions_sha256=controller.file_sha(folder / "predictions.npz"))
            report.update(clean_metrics=metrics) if method == "clean" else report.update(
                metrics=dict(mainmetrics=metrics))
            (folder / "evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    summary = controller.export_results(tmp_path, gate)
    assert len(summary["rows"]) == 16
    assert sum(row["method"] == "clean" for row in summary["rows"]) == 2
    assert (tmp_path / "attack_results.json").is_file()

    (tmp_path / "attacks/wiflow/seed42/a1/predictions.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Invalid PiW evaluation binding"):
        controller.export_results(tmp_path, gate)
