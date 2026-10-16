"""T2 protocol/reuse/export tests. Synthetic data only; no full victim training."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_a1_dose_poison as runner
from run_mmfi_tables import _plan_fingerprint
from train_backdoor import _config_fingerprint, _RESULT_SCHEMA


@pytest.fixture
def matrix(tmp_path):
    return runner.build_matrix(tmp_path / 'data', tmp_path / 'out')


def test_only_two_new_victims_and_all_six_main_doses(matrix):
    runner._validate_manifest(matrix)
    assert [(c['method_key'], c['cfg']['rho']) for c in matrix['cells']] == [('rho0p2', .2), ('rho0p4', .4)]
    assert matrix['metrics_contract']['reused_cells'] == ['full', 'w_o_coupling']
    assert matrix['metrics_contract']['main_doses'] == runner.GRID
    assert matrix['metrics_contract']['train_reference'] is False
    for cell in matrix['cells']:
        cfg = cell['cfg']
        assert cfg['trigger'] == 'md_multicarrier'
        assert cfg['epochs'] == cfg['victim_epochs'] == 50
        assert cfg['optimizer'] == 'sgd' and cfg['lr'] == .001
        assert cfg['rho'] in (.2, .4) and cfg['dose_coupling'] == 'paired'
        assert cfg['dose_grid'] == runner.GRID
        assert not any(k.startswith(('comparison_peak', 'lc_', 'draft_')) for k in cfg)


def test_plan_round_trip_and_runtime_device_resume(matrix):
    restored = json.loads(json.dumps(matrix))
    runner._validate_manifest(restored)
    restored['cells'][1]['cfg'].update(device='cuda:7', num_workers=0)
    assert _plan_fingerprint(restored) == restored['plan_sha256']
    runner._validate_manifest(restored)


@pytest.mark.parametrize('key,value', [
    ('rho', .1), ('trigger', 'md_multicarrier_peak_matched'), ('epochs', 15),
    ('victim_epochs', 15), ('lr', .01), ('dose_coupling', 'shuffled'),
    ('seed', 43), ('dose_min', 1.), ('pivot', 0), ('method_carrier_seed', 99),
    ('comparison_peak_budget', 'original_postclip_linf_v1'),
])
def test_rehashed_manifest_cannot_change_scientific_recipe(matrix, key, value):
    matrix['cells'][0]['cfg'][key] = value
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    with pytest.raises(ValueError, match='configuration|contract'):
        runner._validate_manifest(matrix)


def test_rehashed_manifest_cannot_schedule_reference(matrix):
    matrix['cells'].append(dict(matrix['cells'][0], method_key='full'))
    matrix['plan_sha256'] = _plan_fingerprint(matrix)
    with pytest.raises(ValueError, match='exactly'):
        runner._validate_manifest(matrix)


@pytest.mark.parametrize('workers', [-1, True, 1.5])
def test_invalid_workers_rejected(tmp_path, workers):
    with pytest.raises(ValueError, match='nonnegative integer'):
        runner.build_matrix(tmp_path, tmp_path / 'out', num_workers=workers)


def test_config_is_exact_original_a1_except_rho_profile_and_runtime(matrix):
    data_home = Path(matrix['cells'][0]['cfg']['dataset_root']).parent.parent
    original = runner._configs(data_home, 'cuda:0', 4, 42, factorial=True)[0]
    for cell in matrix['cells']:
        runner._check_against_a1(cell['cfg'], dict(cfg=original))
        changed = {k for k in set(original) | set(cell['cfg']) if original.get(k) != cell['cfg'].get(k)}
        assert changed == {'rho', 'confirmation_profile'}
    original['eps'] = .2
    with pytest.raises(ValueError, match='eps'):
        runner._check_against_a1(matrix['cells'][0]['cfg'], dict(cfg=original))


def test_pinned_shuffled_records_folder_zip_and_tamper(tmp_path, monkeypatch):
    values = {'w_o_coupling/eval_cache.json': b'{"res":1}',
              'w_o_coupling/ablation_inputs.json': b'{"input":2}'}
    monkeypatch.setattr(runner, 'COUPLING_RECORDS', {k: hashlib.sha256(v).hexdigest() for k, v in values.items()})
    folder = tmp_path / 'source'
    for name, value in values.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('source/proposed_ablation.resolved.json', '{}')
        for name, value in values.items():
            z.writestr('source/' + name, value)
    assert runner._pinned_coupling_records(folder) == runner._pinned_coupling_records(archive) == values
    (folder / 'w_o_coupling/eval_cache.json').write_bytes(b'{}')
    with pytest.raises(ValueError, match='reference changed'):
        runner._pinned_coupling_records(folder)


def test_duplicate_frozen_zip_manifest_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'COUPLING_RECORDS', {'w_o_coupling/eval_cache.json': 'unused'})
    path = tmp_path / 'duplicates.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('a/proposed_ablation.resolved.json', '{}')
        z.writestr('b/proposed_ablation.resolved.json', '{}')
    with pytest.raises(ValueError, match='one original'):
        runner._pinned_coupling_records(path)


def _poison(cfg):
    n = runner.EXPECTED_COUNTS['train']
    count = int(np.floor(n * cfg['rho']))
    rng = np.random.default_rng(42)
    ids = rng.choice(n, size=count, replace=False)
    doses = rng.uniform(.2, 1., size=count)
    labels = doses.copy()
    if cfg['dose_coupling'] == 'shuffled':
        np.random.default_rng(np.random.SeedSequence([42, 7043])).shuffle(labels)
    samples = [dict(index=int(i), dose=float(d), payload_dose=float(pd)) for i, d, pd in zip(ids, doses, labels)]
    return dict(samples=samples, config_fingerprint=_config_fingerprint(cfg),
        n_total=n, n_poison=count, rho_requested=cfg['rho'], seed=42,
        selection='uniform', dose_min=.2, dose_max=1., pivot=1,
        target_joints=[2, 3], theta_max_deg=40., payload_axis=[0., 0., 1.],
        dose_mode='linear', dose_coupling=cfg['dose_coupling'], n_cover=0,
        poison_plan_sha256=runner._sha_json([[s['index'], s['dose']] for s in samples]))


def _result(cfg, poison):
    errors = [0., .012, .014, .016, .018, .02]
    return dict(clean_mpjpe=.17, clean_pampjpe=.104, **{f'clean_pck@{d:.1f}': .6 for d in (.1, .2, .3, .4, .5)},
        dose_grid=list(runner.GRID), tmpjpe=[.11, .12, .13, .14, .15, .16],
        clean_to_target_tmpjpe=[.11, .15, .2, .25, .3, .4],
        dose_control_error=errors, asr_curve=[dict(dose_control_error_mean=e) for e in errors],
        training_contract='ordinary_erm', attacker_access='data_only',
        dose_coupling=cfg['dose_coupling'], n_total=poison['n_total'], n_poison=poison['n_poison'],
        n_cover=0, poison_plan_sha256=poison['poison_plan_sha256'])


@pytest.mark.parametrize('rho,coupling', [(.1, 'paired'), (.1, 'shuffled'), (.2, 'paired'), (.4, 'paired')])
def test_seeded_poison_plan_and_payload_are_verified(matrix, monkeypatch, rho, coupling):
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', dict(train=100, eval=25))
    cfg = dict(matrix['cells'][0]['cfg'], rho=rho, dose_coupling=coupling)
    poison = _poison(cfg)
    result = _result(cfg, poison)
    assert runner._audit_poison(cfg, result, poison) == poison['poison_plan_sha256']
    changed = copy.deepcopy(poison)
    changed['samples'][0]['dose'] += .00001
    with pytest.raises(ValueError, match='identities/doses'):
        runner._audit_poison(cfg, result, changed)
    changed = copy.deepcopy(poison)
    changed['samples'][0]['payload_dose'] += .001
    with pytest.raises(ValueError, match='identities/doses'):
        runner._audit_poison(cfg, result, changed)


def test_new_result_cover_or_partial_count_rejected(matrix, monkeypatch):
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', dict(train=100, eval=25))
    cfg = matrix['cells'][0]['cfg']
    poison = _poison(cfg)
    for key, value in [('n_cover', 1), ('n_total', 99), ('attacker_access', 'white_box'), ('dose_coupling', 'shuffled')]:
        res = dict(_result(cfg, poison), **{key: value})
        with pytest.raises(ValueError, match='protocol differs'):
            runner._audit_poison(cfg, res, poison)


def test_grid_dce_excludes_zero_and_errors_are_mm(matrix, monkeypatch):
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', dict(train=100, eval=25))
    cell = matrix['cells'][0]
    res = _result(cell['cfg'], _poison(cell['cfg']))
    row, series = runner._series(cell, res, .2, 'fresh')
    assert [row[k] for k in runner.GRID_DOSE_COLUMNS] == pytest.approx([110, 120, 130, 140, 150, 160])
    assert row['clean_mpjpe_mm'] == pytest.approx(170)
    assert row['mean_positive_dose_control_error_mm'] == pytest.approx(16)
    assert [r['dose_control_error_mm'] for r in series] == pytest.approx([0, 12, 14, 16, 18, 20])
    assert [r['d'] for r in series] == runner.GRID
    res['asr_curve'][2]['dose_control_error_mean'] = .8
    with pytest.raises(ValueError, match='curve differs'):
        runner._series(cell, res, .2, 'fresh')


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', dict(train=100, eval=25))
    matrix = runner.build_matrix(tmp_path / 'data', tmp_path / 'out')
    ref_cfg = runner._configs(tmp_path / 'data', 'cuda:0', 4, 42, factorial=True)[0]
    audit = dict(train=dict(n=100, pair_ids_sha256='a' * 64),
                 eval=dict(n=25, pair_ids_sha256='b' * 64), action_file_sha256='c' * 64)
    full_cell = dict(method_key='full', label='Full', cfg=ref_cfg)
    _, full_curves = runner._series(full_cell, _result(ref_cfg, _poison(ref_cfg)), .1, 'frozen')
    ref = dict(cfg=ref_cfg, row=dict(clean_mpjpe_mm=170.), dose_response=full_curves,
               audit=audit, provenance=dict(reference='original'), source_sha256=runner._ablation_sources())
    shuffled_curves = [dict(r, method_key='w_o_coupling', method='w/o Coupling, rho=0.1') for r in full_curves]
    coupling = dict(dose_response=shuffled_curves, provenance=dict(reference='shuffled'))
    monkeypatch.setattr(runner, '_preflight', lambda matrix: (ref, coupling))
    import train_backdoor
    monkeypatch.setattr(train_backdoor, 'build_trigger', lambda cfg: object())
    monkeypatch.setattr(runner, 'trigger_state_sha256', lambda trigger: 'd' * 64)
    for cell in matrix['cells']:
        folder, cfg = Path(cell['ckpt_dir']), cell['cfg']
        folder.mkdir(parents=True)
        poison = _poison(cfg)
        res = _result(cfg, poison)
        fp = _config_fingerprint(cfg)
        files = {
            'eval_cache.json': dict(result_schema=_RESULT_SCHEMA, cfg=cfg, cfg_fingerprint=fp, trained_epochs=50, res=res),
            'results.json': dict(method_key=cell['method_key'], cfg=cfg, cfg_fingerprint=fp, res=res),
            'poison_manifest.json': poison,
            'ablation_inputs.json': dict(audit, schema=1, profile=runner.PROFILE, config_fingerprint=fp),
        }
        for name, payload in files.items():
            (folder / name).write_text(json.dumps(payload), encoding='utf-8')
        (folder / 'config.resolved.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return matrix, ref, coupling


def test_summary_has_three_rates_and_complete_control_curves(completed):
    matrix, _, _ = completed
    report = runner.build_summary(matrix)
    assert report['status'] == 't2_dose_poison_complete_single_seed'
    assert [r['rho'] for r in report['rows']] == [.1, .2, .4]
    assert len(report['dose_response']) == 24
    for key in ('rho0p1', 'rho0p2', 'rho0p4', 'w_o_coupling'):
        assert report['mean_positive_dose_control_error_mm'][key] == pytest.approx(16)
    assert report['rows'][0]['result_origin'] == 'immutable_existing_factorial_A1'
    assert len(report['provenance']) == 2
    for row in report['rows']:
        assert set(runner.GRID_COLUMNS) <= set(row)


def test_summary_rejects_yaml_or_result_edits(completed):
    matrix, _, _ = completed
    cell = matrix['cells'][0]
    path = Path(cell['ckpt_dir']) / 'results.json'
    value = json.loads(path.read_text())
    value['res']['clean_mpjpe'] = .001
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='YAML/results differ'):
        runner.build_summary(matrix)


def test_summary_rejects_wrong_input_identity(completed):
    matrix, _, _ = completed
    path = Path(matrix['cells'][1]['ckpt_dir']) / 'ablation_inputs.json'
    value = json.loads(path.read_text())
    value['eval']['pair_ids_sha256'] = 'f' * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='eval identities differ'):
        runner.build_summary(matrix)


def test_export_includes_full_six_dose_grid_and_both_plots(completed):
    matrix, _, _ = completed
    outdir = Path(matrix['cells'][0]['ckpt_dir']).parent
    runner.export_summary(matrix, outdir)
    md = (outdir / 'table2_dose_poison.md').read_text(encoding='utf-8')
    assert 'd=0 | d=.2 | d=.4 | d=.6 | d=.8 | d=1' in md
    assert 'Mean positive-dose DCE' in md
    assert (outdir / 'tmpjpe_vs_dose.png').stat().st_size > 1000
    assert (outdir / 'dose_control_error.png').stat().st_size > 1000
    data = json.loads((outdir / 'dose_response.json').read_text())
    assert len(data['rows']) == 24
    csv = (outdir / 'table2_dose_poison.csv').read_text()
    for name in runner.GRID_DOSE_COLUMNS:
        assert name in csv.splitlines()[0]


def test_plot_failure_does_not_write_complete_marker(completed, monkeypatch):
    import matplotlib.pyplot as plt
    def fail(*args, **kwargs):
        raise OSError('synthetic plot failure')
    monkeypatch.setattr(plt, 'subplots', fail)
    matrix, _, _ = completed
    outdir = Path(matrix['cells'][0]['ckpt_dir']).parent
    with pytest.raises(OSError, match='plot failure'):
        runner.export_summary(matrix, outdir)
    assert not (outdir / 'table2_dose_poison.json').exists()


def test_only_audited_old_to_current_sources_are_accepted(completed, monkeypatch):
    matrix, ref, _ = completed
    # Build a synthetic but internally consistent frozen shuffled metadata set.
    historical = copy.deepcopy(ref)
    historical['source_sha256'] = runner._ablation_sources()
    for name, (old, current) in runner.SOURCE_COMPATIBILITY.items():
        assert historical['source_sha256'][name] == current
        historical['source_sha256'][name] = old
    cfg = dict(ref['cfg'], dose_coupling='shuffled')
    poison = _poison(cfg)
    res = _result(cfg, poison)
    fp = _config_fingerprint(cfg)
    records = {
        'w_o_coupling/eval_cache.json': json.dumps(dict(cfg=cfg, res=res, cfg_fingerprint=fp, trained_epochs=50, result_schema=_RESULT_SCHEMA)).encode(),
        'w_o_coupling/results.json': json.dumps(dict(cfg=cfg, res=res, cfg_fingerprint=fp)).encode(),
        'w_o_coupling/ablation_inputs.json': json.dumps(dict(ref['audit'], config_fingerprint=fp)).encode(),
        'w_o_coupling/poison_manifest.json': json.dumps(poison).encode(),
        'w_o_coupling/config.resolved.yaml': yaml.safe_dump(cfg).encode(),
    }
    historical['audit']['poison_plan_sha256'] = poison['poison_plan_sha256']
    monkeypatch.setattr(runner, 'load_a1_reference', lambda path: historical)
    monkeypatch.setattr(runner, '_pinned_coupling_records', lambda path: records)
    full, shuffled = runner.load_references('no_disk')
    assert len(shuffled['dose_response']) == 6
    current = runner._ablation_sources()
    current['train_backdoor.py'] = 'f' * 64
    monkeypatch.setattr(runner, '_ablation_sources', lambda: current)
    with pytest.raises(ValueError, match='source differs.*train_backdoor.py'):
        runner.load_references('no_disk')


def test_dry_run_never_reads_old_runs_or_opens_data(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry run tried data/CUDA/reference access')
    for name in ('_preflight', '_check_inputs', 'run_matrix', 'load_a1_reference'):
        monkeypatch.setattr(runner, name, forbidden)
    out = tmp_path / 'dry'
    assert runner.main(['--data-home', str(tmp_path), '--outdir', str(out), '--dry-run']) == 0
    manifest = json.loads((out / runner.MANIFEST).read_text())
    assert len(manifest['cells']) == 2
    assert sorted(p.name for p in out.iterdir()) == [runner.MANIFEST, 'run.lock']


def test_existing_changed_manifest_refused_before_training(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    runner.main(['--data-home', str(tmp_path), '--outdir', str(out), '--dry-run'])
    with pytest.raises(ValueError, match='differs'):
        runner.main(['--data-home', str(tmp_path), '--outdir', str(out),
                     '--a1-reference', str(tmp_path / 'other'), '--dry-run'])


def test_main_schedules_only_two_new_rates(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, '_preflight', lambda matrix: None)
    monkeypatch.setattr(runner, '_check_inputs', lambda *args: None)
    monkeypatch.setattr(runner, 'run_matrix', lambda matrix, *args, **kw: calls.append([c['method_key'] for c in matrix['cells']]))
    monkeypatch.setattr(runner, '_complete', lambda cell: True)
    monkeypatch.setattr(runner, 'export_summary', lambda *args: {})
    assert runner.main(['--data-home', str(tmp_path), '--outdir', str(tmp_path / 'out')]) == 0
    assert calls == [['rho0p2', 'rho0p4']]


def test_unknown_worker_never_trains(matrix, tmp_path, monkeypatch):
    path = tmp_path / runner.MANIFEST
    path.write_text(json.dumps(matrix))
    calls = []
    monkeypatch.setattr(runner, '_train_cell', lambda cell: calls.append(cell))
    with pytest.raises(ValueError, match='Unknown T2'):
        runner.main(['--_cell', 'full', '--_manifest', str(path)])
    assert calls == []


@pytest.mark.parametrize('devices', [['cuda:0', 'cuda:0'], ['cuda:00'], ['bad']])
def test_bad_logical_devices_rejected(tmp_path, devices):
    with pytest.raises(ValueError, match='distinct logical'):
        runner.main(['--data-home', str(tmp_path), '--dry-run', '--devices', *devices])
