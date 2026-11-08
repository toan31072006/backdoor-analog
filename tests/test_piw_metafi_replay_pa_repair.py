"""Local synthetic repair correctness; no real Van artifacts or training."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'remote_tools/Repair-PiWMetaFiReplayPA.py'


@pytest.fixture
def repair():
    spec = importlib.util.spec_from_file_location('_pa_repair_test', SCRIPT)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def metrics(pa):
    return dict(mainmetrics=dict(clean_pampjpe_mm=pa, t1_mpjpe_mm=150.,
        target_pck_pct=50., clean_mpjpe_mm=160., mean_dce_mm=80.),
        dose_rows=[dict(dose=0., dce_mm=0.), dict(dose=1., dce_mm=90.)],
        controls=dict(unit='mm', target=[11, 13]))


def sample(repair, tmp_path):
    path = tmp_path / 'evaluation.json'
    report = dict(method='smooth_lf_csi', plan_sha256='synthetic', metrics=metrics(repair.OLD))
    original = (json.dumps(report, indent=3) + '\n').encode()
    path.write_bytes(original)
    protected = tmp_path / 'checkpoint.pt'; protected.write_bytes(b'SYNTHETIC_NOT_A_MODEL')
    return path, report, original, {protected: repair.sha(protected)}


def test_confirmed_values_are_one_binary64_ulp(repair):
    assert repair.NEW - repair.OLD == math.ulp(repair.OLD) == 1.4210854715202004e-14
    assert round(repair.NEW, 2) == round(repair.OLD, 2) == 81.46


def test_default_readonly_and_no_input_mutation(repair, tmp_path):
    path, report, original, protected = sample(repair, tmp_path)
    before = copy.deepcopy(report)
    assert repair.apply_correction(path, report, metrics(repair.NEW), protected) is False
    assert path.read_bytes() == original and report == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['checkpoint.pt', 'evaluation.json']


def test_backup_before_single_scalar_replacement_and_exact_replay(repair, tmp_path):
    path, report, original, protected = sample(repair, tmp_path)
    expected = copy.deepcopy(report); expected['metrics'] = metrics(repair.NEW)
    assert repair.apply_correction(path, report, metrics(repair.NEW), protected, apply=True)
    assert repair.read(path) == expected
    assert (tmp_path / ('evaluation.before_' + repair.TAG + '.json')).read_bytes() == original
    proposal = repair.read(tmp_path / (repair.TAG + '.proposal.json'))
    applied = repair.read(tmp_path / (repair.TAG + '.applied.json'))
    assert applied == proposal and proposal['ulps'] == 1
    assert proposal['replacement_sha256'] == repair.sha(path)
    assert repair.digest(repair.read(path)['metrics']) == repair.digest(metrics(repair.NEW))
    assert all(repair.sha(p) == value for p, value in protected.items())
    files_before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert repair.apply_correction(path, expected, metrics(repair.NEW), protected, apply=True) is False
    assert files_before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('kind', ['other_metric', 'two_ulp', 'other_pa', 'nonfinite', 'tsba', 'dose', 'schema', 'minus_zero'])
def test_any_unconfirmed_change_refused_without_writes(repair, tmp_path, kind):
    path, report, original, protected = sample(repair, tmp_path)
    actual = metrics(repair.NEW)
    if kind == 'other_metric': actual['mainmetrics']['t1_mpjpe_mm'] = math.nextafter(150., math.inf)
    if kind == 'two_ulp': actual['mainmetrics']['clean_pampjpe_mm'] = math.nextafter(repair.NEW, math.inf)
    if kind == 'other_pa': report['metrics']['mainmetrics']['clean_pampjpe_mm'] = 90.; actual['mainmetrics']['clean_pampjpe_mm'] = math.nextafter(90., math.inf)
    if kind == 'nonfinite': actual['mainmetrics']['clean_pampjpe_mm'] = float('nan')
    if kind == 'tsba': report['method'] = 'tsba_adapted'
    if kind == 'dose': actual['dose_rows'][1]['dce_mm'] += 1.
    if kind == 'schema': actual['controls']['new_key'] = 'unexpected'
    if kind == 'minus_zero': actual['dose_rows'][0]['dce_mm'] = -0.
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        repair.apply_correction(path, report, actual, protected, apply=True)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('*pa_replay*'))


def test_tampered_checkpoint_refused(repair, tmp_path):
    path, report, original, protected = sample(repair, tmp_path)
    next(iter(protected)).write_bytes(b'CHANGED')
    with pytest.raises(ValueError, match='Protected'):
        repair.apply_correction(path, report, metrics(repair.NEW), protected, apply=True)
    assert path.read_bytes() == original and not list(tmp_path.glob('*pa_replay*'))


def test_existing_foreign_backup_preserved(repair, tmp_path):
    path, report, original, protected = sample(repair, tmp_path)
    backup = tmp_path / ('evaluation.before_' + repair.TAG + '.json')
    backup.write_bytes(b'historical evidence')
    with pytest.raises(ValueError, match='backup/audit'):
        repair.apply_correction(path, report, metrics(repair.NEW), protected, apply=True)
    assert path.read_bytes() == original and backup.read_bytes() == b'historical evidence'


def test_existing_changed_json_refused(repair, tmp_path):
    path, report, _, protected = sample(repair, tmp_path)
    path.write_text('{}')
    with pytest.raises(ValueError, match='changed during'):
        repair.apply_correction(path, report, metrics(repair.NEW), protected, apply=True)
    assert path.read_text() == '{}'


def test_lock_main_schema_hash_checks_and_complete_synthetic_flow(repair, tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'; tools = checkout / 'remote_tools'; tools.mkdir(parents=True)
    ctrl = tools / 'piw_metafi_remaining_3090'; ctrl.mkdir()
    (ctrl / 'controller.py').write_text('from contextlib import contextmanager\n@contextmanager\ndef lock(root):\n    yield\n')
    gate_code = {}
    for key, name in {'runner': 'piw_metafi_two_extra_4090.py',
            'helper': 'Start-PiWMetaFiTwoExtra4090.ps1', 'orchestration': 'piw_metafi_four_attacks_4090.py',
            'a1_sidecar': 'piw_metafi_a1_4090.py'}.items():
        path = tools / name; path.write_text('# SYNTHETIC ONLY\n'); gate_code[key] = repair.sha(path)
    root = tmp_path / repair.PROFILE; root.mkdir()
    snapshot = root / 'source_snapshot/ATKBackd'; snapshot.mkdir(parents=True)
    source = snapshot / 'fixture.py'; source.write_text('# SYNTHETIC ONLY\n')
    clean_root = tmp_path / 'clean'; clean_folder = clean_root / repair.CELL / 'clean'
    clean_folder.mkdir(parents=True)
    (clean_folder / 'checkpoint.pt').write_bytes(b'SYNTHETIC_CLEAN_NOT_A_MODEL')
    y = np.zeros((2586, 14, 3), dtype=np.float32)
    names = [str(i) for i in range(2586)]
    np.savez_compressed(clean_folder / 'predictions.npz', truth=y, names=np.array(names))
    ref = dict(run=str(clean_root), files={name: repair.sha(clean_folder / name)
        for name in ('checkpoint.pt', 'predictions.npz')})
    plan = dict(profile=repair.PROFILE, cfg=dict(methods=['tsba_adapted', 'smooth_lf_csi'],
        seed=42, epochs=200, final_epoch=199, eval_doses=[0., .2, .4, .6, .8, 1.]),
        sources={'fixture.py': repair.sha(source)}, clean_reference=ref,
        inputs={'splits': {'test_data': {'ordered_names': names}}},
        native_records={m: {'fixture': True} for m in ('tsba_adapted', 'smooth_lf_csi')})
    plan['plan_sha256'] = repair.digest(plan)
    (root / 'plan.json').write_text(json.dumps(plan))
    (root / 'prelaunch_gate.json').write_text(json.dumps(dict(plan_sha256=plan['plan_sha256'], code=gate_code)))
    grid = np.repeat(y[:, None], 6, axis=1)
    for method in ('tsba_adapted', 'smooth_lf_csi'):
        folder = root / repair.CELL / method; folder.mkdir(parents=True)
        (folder / 'checkpoint.pt').write_bytes(b'SYNTHETIC_OWN_NOT_A_MODEL')
        np.savez_compressed(folder / 'predictions.npz', truth=y, names=names, doses=[0., .2, .4, .6, .8, 1.],
            predictions=y, trigger_predictions=grid, reference_predictions=y, reference_trigger_predictions=grid, targets=grid)
        report = dict(profile=repair.PROFILE, status='COMPLETE', method=method,
            backbone='metafiplusplus', seed=42, epoch=199, plan_sha256=plan['plan_sha256'],
            poison=plan['native_records'][method], clean_reference=ref,
            checkpoint_sha256=repair.sha(folder / 'checkpoint.pt'),
            predictions_sha256=repair.sha(folder / 'predictions.npz'),
            clean_checkpoint_sha256=ref['files']['checkpoint.pt'],
            metrics=metrics(repair.OLD if method == 'smooth_lf_csi' else repair.NEW))
        (folder / 'evaluation.json').write_text(json.dumps(report))
    fake = SimpleNamespace(measure_predictions=lambda *a: metrics(repair.NEW), target_pose=lambda truth, dose: truth)
    monkeypatch.setitem(sys.modules, 'piw_v2_adapter', fake)
    # Exercise the UNCHANGED original strict replay checker before/after the
    # derived-scalar canonicalization; its hash comparison is not patched.
    spec = importlib.util.spec_from_file_location('_original_two_extra_replay',
        SCRIPT.parent / 'piw_metafi_two_extra_4090.py')
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    q = legacy.queue_module()
    base = SimpleNamespace(read=repair.read, sha=repair.sha, digest=repair.digest)
    legacy_args = SimpleNamespace(out=root, clean_run=clean_root,
        scientific_worker=SimpleNamespace(measure_predictions=fake.measure_predictions,
            target_pose=fake.target_pose, DOSES=(0., .2, .4, .6, .8, 1.)))
    with pytest.raises(ValueError, match='exactly replay'):
        legacy.replay_evaluation(q, legacy_args, base, plan, {'binding': ref}, 'smooth_lf_csi')
    args = ['--run', str(root), '--checkout', str(checkout)]
    initial = {p: repair.sha(p) for p in root.rglob('*') if p.is_file()}
    assert repair.main(args) == 0
    assert all(repair.sha(p) == h for p, h in initial.items())
    assert repair.main(args + ['--apply']) == 0
    changed = [p for p, h in initial.items() if repair.sha(p) != h]
    assert changed == [root / repair.CELL / 'smooth_lf_csi/evaluation.json']
    assert legacy.replay_evaluation(q, legacy_args, base, plan, {'binding': ref}, 'smooth_lf_csi')['status'] == 'COMPLETE'
    assert legacy.replay_evaluation(q, legacy_args, base, plan, {'binding': ref}, 'tsba_adapted')['status'] == 'COMPLETE'
    assert repair.main(args + ['--apply']) == 0
    source.write_text('# mutated')
    with pytest.raises(ValueError, match='source changed'):
        repair.main(args)


def test_real_legacy_os_lock_cannot_be_bypassed(repair, tmp_path):
    checkout = SCRIPT.parents[1]
    controller = checkout / 'remote_tools/piw_metafi_remaining_3090/controller.py'
    spec = importlib.util.spec_from_file_location('_real_legacy_os_lock_test', controller)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    root = tmp_path / repair.PROFILE; root.mkdir()
    with base.lock(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            repair.main(['--run', str(root), '--checkout', str(checkout), '--apply'])
    assert not list(root.glob('*.json'))


def test_native_numpy_npz_layout_reproduces_pa_only_roundoff(tmp_path, monkeypatch):
    """Actual native numerical functions; invented poses, not Van's arrays."""
    import io
    monkeypatch.syspath_prepend(str(SCRIPT.parents[1] / 'ATKBackd'))
    from piw_v2_adapter import measure_predictions
    rng = np.random.default_rng(42)
    truth = rng.normal(size=(64, 14, 3)).astype(np.float32)
    grid = rng.normal(size=(64, 6, 3, 14)).astype(np.float32).swapaxes(-2, -1)
    ref = rng.normal(size=(64, 6, 3, 14)).astype(np.float32).swapaxes(-2, -1)
    old = measure_predictions(truth, grid[:, 0], grid, ref[:, 0], ref)
    stream = io.BytesIO()
    np.savez_compressed(stream, truth=truth, grid=grid, ref=ref,
        pred=grid[:, 0], reference=ref[:, 0])
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as p:
        new = measure_predictions(p['truth'], p['pred'], p['grid'], p['reference'], p['ref'])
        np.testing.assert_array_equal(p['grid'], grid)
    pa_old = old['mainmetrics'].pop('clean_pampjpe_mm')
    pa_new = new['mainmetrics'].pop('clean_pampjpe_mm')
    assert old == new  # all other metrics identical
    assert abs(pa_new-pa_old) <= 4*math.ulp(pa_old)
    # Depending on numerical libraries the synthetic PA may match exactly;
    # the literal Van mismatch and its one-ULP repair have separate exact tests.
