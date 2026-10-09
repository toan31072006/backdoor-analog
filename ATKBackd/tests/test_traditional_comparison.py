"""Dose-adapted comparison planning/audits; no real dataset or CUDA."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_traditional_comparison as runner
import train_backdoor as trainer


KEYS = ['badnets', 'blended', 'wanet_source', 'proposed']
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _forbidden(*args, **kwargs):
    pytest.fail('Planning/distortion attempted training, CUDA or HPE access')


@pytest.fixture
def sources(monkeypatch):
    record = {'fixture.py': 'a' * 64}
    monkeypatch.setattr(runner, '_source_provenance', lambda: dict(record))
    return record


@pytest.fixture
def matrix(tmp_path, sources):
    return runner.build_matrix(tmp_path / 'missing_data', tmp_path / 'comparison', 'cpu', 4)


def test_plan_contains_all_four_methods_fixed_order_and_full_protocol(matrix):
    assert [c['method_key'] for c in matrix['cells']] == KEYS
    assert matrix['seed'] == 42 and matrix['fresh_results_only'] is True
    assert matrix['metrics_contract']['full_split_counts'] == {'train': 133056, 'eval': 33264}
    assert matrix['metrics_contract']['matches_noise_budget'] is False
    assert matrix['metrics_contract']['compared_methods'] == KEYS
    assert matrix['metrics_contract']['comparison_group'] == 'continuous_dose_task'
    for cell in matrix['cells']:
        cfg = cell['cfg']
        for key, value in dict(seed=42, rho=0.1, epochs=50, victim_epochs=50,
                optimizer='sgd', lr=0.001, momentum=0.9, batch_size=32,
                weight_decay=0.0, pivot=1, target_joints=[2, 3],
                theta_max_deg=40.0, payload_axis=[0.0, 0.0, 1.0],
                victim_loss='mpjpe', dose_coupling='paired',
                dose_grid=GRID, dose_min=0.2, dose_max=1.0, pretrained=False,
                data_parallel=False, attacker_access='data_only',
                training_protocol='ordinary_erm', loader_persistent_workers=False).items():
            assert cfg[key] == value, (cell['method_key'], key)
        assert cfg['mmfi_protocol'] == 'protocol1' and cfg['mmfi_setting'] == 's1'
        assert cfg['mmfi_split_seed'] == 0 and cfg['mmfi_random_ratio'] == 0.8
        assert cell['tables'] == [] and cell['dependencies'] == []
        assert not any(k.startswith('draft_') or k == 'method_draft' for k in cfg)
    runner._validate_manifest(matrix)


def test_source_settings_have_explicit_minimal_dose_adaptation_and_common_task(matrix):
    badnets, blend, wanet, proposed = [c['cfg'] for c in matrix['cells']]
    assert badnets['badnets_patch_subcarriers'] == badnets['badnets_patch_packets'] == 3
    assert badnets['badnets_patch_opacity'] == 1.0 and badnets['badnets_pattern'] == 'white'
    assert blend['eps'] == 0.2
    assert wanet['trigger'] == 'wanet_source' and wanet['num_workers'] == 0
    assert wanet['wanet_grid_size'] == 4 and wanet['wanet_strength'] == 0.5
    assert wanet['wanet_grid_rescale'] == 1.0 and wanet['wanet_cover_ratio'] == 0.2
    for cfg in (badnets, blend, wanet, proposed):
        assert cfg['dose_min'] == 0.2 and cfg['dose_max'] == 1.0
        assert cfg['dose_coupling'] == 'paired' and cfg['dose_grid'] == GRID
    assert proposed['eps'] == 0.185
    assert proposed['trigger'] == 'md_multicarrier_peak_matched'
    assert all(c['group'] == 'continuous_dose_task' for c in matrix['cells'])
    adaptation = matrix['metrics_contract']['dose_adaptation']
    assert adaptation['badnets'] == 'patch opacity=d'
    assert adaptation['blended'] == 'alpha=0.2*d'
    assert '0.5*d*N/H' in adaptation['wanet'] and 'native covers at d=1' in adaptation['wanet']


def test_source_defaults_attribution_is_explicit_not_original_author_claim(matrix):
    references = matrix['sources']['references']
    assert references['backdoorbench_commit'] == 'f02e3534645f0ee63d6848653062cd6c0d6c400d'
    assert references['badnets_patch_reference']['path'].endswith('generate_white_square.py')
    assert references['blended_default_reference']['path'].endswith('default.yaml')
    assert references['wanet_author_code'].startswith('https://github.com/VinAIResearch/')
    assert matrix['cells'][0]['cfg']['badnets_operator_sha256']
    assert matrix['cells'][1]['cfg']['blended_operator_sha256']
    assert matrix['cells'][2]['cfg']['wanet_adapter_sha256']


def test_build_does_not_read_dataset_construct_trigger_train_or_use_cuda(tmp_path, sources, monkeypatch):
    monkeypatch.setattr(trainer, '_load_dataset', _forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', _forbidden)
    monkeypatch.setattr(trainer, 'train', _forbidden)
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', _forbidden)
    runner.build_matrix(tmp_path / 'missing', tmp_path / 'output', 'cuda:3', 8)


def test_mutable_configs_do_not_alias(matrix):
    matrix['cells'][0]['cfg']['target_joints'].append(9)
    matrix['cells'][0]['cfg']['dose_grid'][0] = 0.2
    for cell in matrix['cells'][1:]:
        assert cell['cfg']['target_joints'] == [2, 3]
        assert cell['cfg']['dose_grid'] == GRID
    assert runner.GRID == GRID


def test_metric_contract_edits_cannot_mutate_canonical_protocol(matrix):
    matrix['metrics_contract']['dose_grid'][1] = 0.1
    matrix['metrics_contract']['full_split_counts']['train'] = 20
    assert runner.GRID == GRID
    assert runner.EXPECTED_COUNTS == {'train': 133056, 'eval': 33264}
    matrix['plan_sha256'] = runner._plan_fingerprint(matrix)
    with pytest.raises(ValueError, match='metric contract changed'):
        runner._validate_manifest(matrix)


@pytest.mark.parametrize('change', [
    'rho', 'loss', 'epochs', 'patch', 'blend_alpha', 'fixed_dose_regression',
    'wanet_workers', 'persistent_workers', 'drop_method', 'order', 'changed_group', 'draft',
    'metric_contract', 'references', 'dataset_path', 'output_path'])
def test_rehashed_altered_scientific_plan_is_rejected(matrix, change):
    if change == 'rho':
        matrix['cells'][0]['cfg']['rho'] = 0.4
    elif change == 'loss':
        matrix['cells'][0]['cfg']['victim_loss'] = 'mse'
    elif change == 'epochs':
        matrix['cells'][0]['cfg']['epochs'] = 15
    elif change == 'patch':
        matrix['cells'][0]['cfg']['badnets_patch_subcarriers'] = 8
    elif change == 'blend_alpha':
        matrix['cells'][1]['cfg']['eps'] = 0.185
    elif change == 'fixed_dose_regression':
        matrix['cells'][1]['cfg']['dose_min'] = 1.0
    elif change == 'wanet_workers':
        matrix['cells'][2]['cfg']['num_workers'] = 4
    elif change == 'persistent_workers':
        matrix['cells'][0]['cfg']['loader_persistent_workers'] = True
    elif change == 'drop_method':
        matrix['cells'].pop(0)
    elif change == 'order':
        matrix['cells'].reverse()
    elif change == 'changed_group':
        matrix['cells'][3]['group'] = 'fixed_endpoint'
    elif change == 'draft':
        matrix['cells'][0]['cfg']['draft_train_samples'] = 20000
    elif change == 'metric_contract':
        matrix['metrics_contract']['matches_noise_budget'] = True
    elif change == 'references':
        matrix['sources']['references'] = {}
    elif change == 'dataset_path':
        matrix['cells'][2]['cfg']['dataset_root'] += '_different'
    else:
        matrix['cells'][2]['ckpt_dir'] += '_different'
    matrix['plan_sha256'] = runner._plan_fingerprint(matrix)
    with pytest.raises(ValueError):
        runner._validate_manifest(matrix)


def test_unrehased_modified_manifest_rejected(matrix):
    matrix['cells'][0]['cfg']['lr'] = 0.01
    with pytest.raises(ValueError, match='fingerprint'):
        runner._validate_manifest(matrix)


def test_old_v1_fixed_endpoint_profile_cannot_be_imported_even_rehashed(matrix):
    matrix['traditional_comparison_profile'] = 'mmfi_traditional_source_comparison_v1'
    matrix['plan_sha256'] = runner._plan_fingerprint(matrix)
    with pytest.raises(ValueError, match='v3 MM-Fi'):
        runner._validate_manifest(matrix)


def test_source_change_refuses_resume_and_export(matrix, sources):
    sources['fixture.py'] = 'b' * 64
    with pytest.raises(ValueError, match='source changed'):
        runner._validate_manifest(matrix)
    with pytest.raises(ValueError, match='source changed'):
        runner.build_summary(matrix)


@pytest.mark.parametrize('value', [True, 0, -1, 1.5])
def test_distortion_samples_are_positive_integer(tmp_path, sources, value):
    with pytest.raises(ValueError):
        runner.build_matrix(tmp_path, tmp_path / 'out', distortion_samples=value)


def _cli(tmp_path, *extra):
    return ['--data-home', str(tmp_path / 'missing'), '--outdir', str(tmp_path / 'out'),
            '--devices', 'cpu', *extra]


def test_dry_run_writes_only_isolated_plan(tmp_path, sources, monkeypatch):
    monkeypatch.setattr(trainer, 'train', _forbidden)
    monkeypatch.setattr(trainer, '_load_dataset', _forbidden)
    monkeypatch.setattr(runner, '_check_inputs', _forbidden)
    assert runner.main(_cli(tmp_path, '--dry-run', '--fresh')) == 0
    output = tmp_path / 'out'
    assert {p.name for p in output.iterdir()} == {'run.lock', 'traditional_comparison.resolved.json'}
    matrix = json.loads((output / 'traditional_comparison.resolved.json').read_text())
    runner._validate_manifest(matrix)


def test_same_plan_resume_allows_runtime_device_workers_only(tmp_path, sources):
    assert runner.main(_cli(tmp_path, '--dry-run')) == 0
    path = tmp_path / 'out/traditional_comparison.resolved.json'
    before = json.loads(path.read_text())
    arguments = _cli(tmp_path, '--dry-run', '--num-workers', '7')
    index = arguments.index('cpu')
    arguments[index] = 'cuda:3'
    assert runner.main(arguments) == 0
    after = json.loads(path.read_text())
    assert before['plan_sha256'] == after['plan_sha256']
    assert all(c['cfg']['device'] == 'cuda:3' for c in after['cells'])
    assert [c['cfg']['num_workers'] for c in after['cells']] == [7, 7, 0, 7]
    runner._validate_manifest(after)


@pytest.mark.parametrize('fresh', [False, True])
def test_refuses_nonempty_unrelated_output_without_overwriting(tmp_path, sources, fresh):
    output = tmp_path / 'out'
    output.mkdir()
    previous = output / 'eval_cache.json'
    previous.write_text('historical content')
    with pytest.raises((ValueError, FileExistsError)):
        runner.main(_cli(tmp_path, '--dry-run', *(['--fresh'] if fresh else [])))
    assert previous.read_text() == 'historical content'
    assert not (output / 'traditional_comparison.resolved.json').exists()


def test_refuses_edited_source_or_distortion_budget_in_existing_output(tmp_path, sources):
    runner.main(_cli(tmp_path, '--dry-run'))
    with pytest.raises(ValueError, match='settings/source changed'):
        runner.main(_cli(tmp_path, '--dry-run', '--distortion-samples', '32'))
    sources['fixture.py'] = 'c' * 64
    with pytest.raises(ValueError, match='source changed'):
        runner.main(_cli(tmp_path, '--dry-run'))


@pytest.mark.parametrize('artifact', ['checkpoint.pt', 'eval_cache.json'])
def test_unaudited_historical_cache_is_not_imported(matrix, artifact):
    cell = matrix['cells'][0]
    folder = Path(cell['ckpt_dir'])
    folder.mkdir(parents=True)
    (folder / artifact).write_text('historical')
    with pytest.raises(ValueError, match='historical caches'):
        runner._bind_inputs(cell, {'profile': runner.PROFILE})
    assert (folder / artifact).read_text() == 'historical'


def test_bound_inputs_can_resume_but_changed_identity_fails(matrix):
    cell = matrix['cells'][0]
    record = dict(profile=runner.PROFILE, action_file_sha256='a' * 64)
    runner._bind_inputs(cell, record)
    runner._bind_inputs(cell, record)
    with pytest.raises(ValueError, match='identity changed'):
        runner._bind_inputs(cell, dict(record, action_file_sha256='b' * 64))


@pytest.fixture
def audited_matrix(matrix, monkeypatch):
    monkeypatch.setattr(runner, 'EXPECTED_COUNTS', {'train': 20, 'eval': 10})
    for cell in matrix['cells']:
        key, cfg = cell['method_key'], cell['cfg']
        folder = Path(cell['ckpt_dir'])
        folder.mkdir(parents=True)
        fingerprint = trainer._config_fingerprint(cfg)
        inputs = dict(schema=1, profile=runner.PROFILE, config_fingerprint=fingerprint,
            action_file_sha256='a' * 64,
            train=dict(n=20, pair_ids_sha256='b' * 64),
            eval=dict(n=10, pair_ids_sha256='c' * 64))
        doses = [0.3, 0.8]
        samples = [dict(index=i, dose=d, csi_id=f'csi_{i}', pose_id=f'pose_{i}', frame_idx=i)
                   for i, d in zip([1, 7], doses)]
        plan_sha = runner._sha_json([[s['index'], s['dose']] for s in samples])
        cover = [0, 2, 3, 4] if key == 'wanet_source' else []
        poison = dict(schema=5, config_fingerprint=fingerprint, rho_requested=0.1,
            n_total=20, n_poison=2, seed=42, selection='uniform',
            dose_min=cfg['dose_min'], dose_max=1.0, poison_plan_sha256=plan_sha,
            samples=samples, n_cover=len(cover), cover_indices=cover,
            cover_ratio=0.2 if cover else 0.0, pivot=1, target_joints=[2, 3],
            theta_max_deg=40.0, payload_axis=[0.0, 0.0, 1.0], dose_mode='linear')
        (folder / 'comparison_inputs.json').write_text(json.dumps(inputs))
        (folder / 'poison_manifest.json').write_text(json.dumps(poison))
    matrix['metrics_contract']['full_split_counts'] = runner.EXPECTED_COUNTS
    matrix['plan_sha256'] = runner._plan_fingerprint(matrix)
    return matrix


def test_audit_proves_identical_multidose_poison_ids_and_dose_values_for_all_four(audited_matrix):
    audit = runner.audit_common_inputs(audited_matrix)
    plans = audit['poison_plan_sha256']
    assert len({plans[k] for k in KEYS}) == 1
    assert audit['counts']['wanet_source'] == dict(n_poison=2, n_cover=4)
    assert audit['counts']['blended'] == dict(n_poison=2, n_cover=0)


@pytest.mark.parametrize('change', ['split', 'action', 'poison_id', 'dose', 'cover_overlap', 'cover_count'])
def test_common_audit_refuses_changed_ids_doses_or_covers(audited_matrix, change):
    cell = audited_matrix['cells'][2]
    folder = Path(cell['ckpt_dir'])
    name = 'comparison_inputs.json' if change in ('split', 'action') else 'poison_manifest.json'
    path = folder / name
    value = json.loads(path.read_text())
    if change == 'split':
        value['eval']['pair_ids_sha256'] = 'd' * 64
    elif change == 'action':
        value['action_file_sha256'] = 'd' * 64
    elif change == 'poison_id':
        value['samples'][0]['csi_id'] = 'different_frame'
    elif change == 'dose':
        value['samples'][0]['dose'] = 0.7
    elif change == 'cover_overlap':
        value['cover_indices'][0] = 1
    else:
        value['n_cover'] = 3
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.audit_common_inputs(audited_matrix)


def _fake_cached(cell):
    key = cell['method_key']
    poison = json.loads((Path(cell['ckpt_dir']) / 'poison_manifest.json').read_text())
    t1 = 0.120 if key == 'badnets' else 0.140
    res = dict(training_contract='ordinary_erm', attacker_access='data_only',
        dose_coupling='paired', dose_grid=GRID, reference_dose=1.0,
        n_total=20, n_poison=2, n_cover=poison['n_cover'],
        poison_plan_sha256=poison['poison_plan_sha256'],
        clean_mpjpe=0.176, clean_pampjpe=0.104,
        tmpjpe=[0.110, 0.112, 0.114, 0.116, 0.118, t1],
        clean_to_target_tmpjpe=[0.110, 0.160, 0.220, 0.280, 0.340, 0.400])
    res.update({f'clean_pck@{threshold:.1f}': 0.8 for threshold in (0.5, 0.4, 0.3, 0.2, 0.1)})
    return res, dict(cfg_fingerprint=trainer._config_fingerprint(cell['cfg']))


def test_summary_retains_stronger_baseline_row_all_four_common_task_and_six_doses(audited_matrix, monkeypatch):
    monkeypatch.setattr(runner, '_verified_cache', _fake_cached)
    report = runner.build_summary(audited_matrix)
    assert [r['method_key'] for r in report['rows']] == KEYS
    assert report['rows'][0]['t1_mpjpe_mm'] == 120.0
    assert report['rows'][3]['t1_mpjpe_mm'] == 140.0
    assert report['rows'][0]['i1_improvement_mm'] == pytest.approx(280.0)
    assert report['rows'][0]['no_trigger_t1_mpjpe_mm'] == 400.0
    assert all(r['comparison_group'] == 'continuous_dose_task' for r in report['rows'])
    assert report['rows'][0]['mean_positive_dose_tmpjpe_mm'] == pytest.approx(116.0)
    assert len(report['dose_response']) == 24
    for key in KEYS:
        assert [r['d'] for r in report['dose_response'] if r['method_key'] == key] == GRID
    assert all('asr' not in key.lower() for row in report['rows'] for key in row)
    assert any('NOT equal realized peak usage or L2' in s for s in report['limitations'])


@pytest.mark.parametrize('change', ['cover', 'dosegrid', 'contract', 'poison'])
def test_summary_refuses_results_metadata_disagreeing_with_audit(audited_matrix, monkeypatch, change):
    def wrong_cache(cell):
        res, source = _fake_cached(cell)
        res[{'cover': 'n_cover', 'dosegrid': 'dose_grid', 'contract': 'training_contract',
             'poison': 'poison_plan_sha256'}[change]] = {'cover': 15, 'dosegrid': [0, .5, 1],
                'contract': 'target_weighted', 'poison': 'd' * 64}[change]
        return res, source
    monkeypatch.setattr(runner, '_verified_cache', wrong_cache)
    with pytest.raises(ValueError, match='cached results disagree'):
        runner.build_summary(audited_matrix)


def test_distortion_measurement_uses_same_ids_all_six_doses_and_no_hpe(audited_matrix, monkeypatch):
    import eval.distortion as distortion
    records = {c['method_key']: json.loads((Path(c['ckpt_dir']) /
               'comparison_inputs.json').read_text()) for c in audited_matrix['cells']}
    monkeypatch.setattr(runner, '_collect_input_record', lambda cell: records[cell['method_key']])
    monkeypatch.setattr(trainer, 'train', _forbidden)
    monkeypatch.setattr(trainer, 'build_model', _forbidden)

    class Dataset:
        data_root = '/fake'
        items = [dict(csi=f'/fake/csi_{i}', kpt=f'/fake/pose_{i}', frame_idx=i) for i in range(10)]
        def __len__(self):
            return len(self.items)

    class Trigger:
        key = 'fixed'

    ds, calls = Dataset(), []
    monkeypatch.setattr(trainer, '_load_dataset', lambda cfg, split: ds)
    monkeypatch.setattr(trainer, 'build_trigger', lambda cfg: Trigger())
    monkeypatch.setattr(runner, '_peak_audit', lambda trigger, dataset, indices, dose, eps, **kwargs:
        [dict(ceiling=.2*dose, peak=.2*dose, violation=False, shrunk=False) for i in indices])

    def measure(cfg, n, dose, split, trig, dataset):
        assert dataset is ds and split == 'test' and cfg['num_workers'] == 0
        calls.append((cfg['trigger'], cfg['eps'], dose, n))
        return dict(relative_l2=.1*dose, rms=.05*dose, max_abs=.2*dose,
            snr_db=float('inf') if dose == 0 else 20.0, n_samples=n, eps=cfg['eps'])

    monkeypatch.setattr(distortion, 'measure', measure)
    report = runner.build_distortion(audited_matrix)
    assert report['hpe_forward_used'] is False and report['matches_noise_budget'] is False
    assert report['shared_peak_ceiling'] is True
    assert len(report['peak_audits']) == 240 and len(report['cover_peak_audits']) == 10
    assert len(report['rows']) == 24 and len(calls) == 24
    assert len({r['common_pair_ids_sha256'] for r in report['rows']}) == 1
    assert all(r['n_samples'] == 10 for r in report['rows'])
    assert report['rows'][0]['snr_db'] is None and report['rows'][0]['snr_db_is_infinite'] is True
    assert calls[6][1] == .2 and calls[18][1] == .185
    for key in KEYS:
        assert [r['dose'] for r in report['rows'] if r['method_key'] == key] == GRID
    json.dumps(report, allow_nan=False)


def test_source_wanet_fixed_key_hash_ignores_cover_rng_progress():
    from attack.wanet_source import WaNetSourceTrigger
    trigger = WaNetSourceTrigger()
    before = runner._trigger_state_sha256(trigger)
    trigger.noise_inject(np.full((3, 114, 10), .5, dtype=np.float32), seed=42)
    assert runner._trigger_state_sha256(trigger) == before


def test_distortion_export_rejects_changed_training_time_action_before_measure(audited_matrix, monkeypatch):
    import eval.distortion as distortion
    records = {c['method_key']: json.loads((Path(c['ckpt_dir']) /
               'comparison_inputs.json').read_text()) for c in audited_matrix['cells']}
    for record in records.values():
        record['action_file_sha256'] = 'e' * 64
    monkeypatch.setattr(runner, '_collect_input_record', lambda cell: records[cell['method_key']])
    monkeypatch.setattr(distortion, 'measure', _forbidden)
    monkeypatch.setattr(trainer, 'build_trigger', _forbidden)
    with pytest.raises(ValueError, match='identity changed'):
        runner.build_distortion(audited_matrix)


def test_audit_rejects_one_methods_valid_but_different_dose_plan_after_rehash(audited_matrix):
    cell = audited_matrix['cells'][1]
    path = Path(cell['ckpt_dir']) / 'poison_manifest.json'
    poison = json.loads(path.read_text())
    poison['samples'][0]['dose'] = 0.6
    poison['poison_plan_sha256'] = runner._sha_json(
        [[s['index'], s['dose']] for s in poison['samples']])
    path.write_text(json.dumps(poison))
    with pytest.raises(ValueError, match='identical poison indices AND dose values'):
        runner.audit_common_inputs(audited_matrix)


def test_export_retains_all_four_methods_and_writes_complete_six_dose_csv_json(audited_matrix, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, '_verified_cache', _fake_cached)
    monkeypatch.setattr(runner, 'export_distortion', lambda matrix, output: {'rows': [], 'hpe_forward_used': False})
    outdir = tmp_path / 'exported'
    report = runner.export_summary(audited_matrix, outdir)
    with (outdir / 'traditional_comparison.csv').open(newline='', encoding='utf-8') as handle:
        main_rows = list(csv.DictReader(handle))
    with (outdir / 'dose_response.csv').open(newline='', encoding='utf-8') as handle:
        dose_rows = list(csv.DictReader(handle))
    dose_json = json.loads((outdir / 'dose_response.json').read_text())
    assert [r['method_key'] for r in main_rows] == KEYS
    assert len(dose_rows) == len(dose_json['rows']) == 24
    assert dose_json['rows'] == report['dose_response']
    assert dose_json['profile'] == runner.PROFILE
    for key in KEYS:
        assert [float(r['d']) for r in dose_rows if r['method_key'] == key] == GRID
    text = (outdir / 'traditional_comparison.md').read_text()
    assert 'paired U(0.2,1)' in text and 'd=0,0.2,0.4,0.6,0.8,1' in text
    assert 'Proposed-fixed' not in text and 'separate task' not in text


def test_source_provenance_covers_model_trigger_and_vendored_operators():
    provenance = runner._source_provenance()
    required = ['run_traditional_comparison.py', 'attack/wanet_source.py',
        'attack/traditional.py', 'attack/trigger.py', 'train_backdoor.py',
        'models/hpeli.py', 'models/sk_network.py', 'attack/payload.py',
        'attack/poison.py', 'third_party/backdoorbench/patch.py',
        'third_party/backdoorbench/blended.py']
    assert all(len(provenance[path]) == 64 for path in required)


def test_shared_peak_policy_is_identical_for_all_four_and_proposed_is_peak_candidate(matrix):
    assert matrix['metrics_contract']['shared_peak_ceiling'] is True
    for cell in matrix['cells']:
        cfg = cell['cfg']
        assert cfg['comparison_peak_budget'] == 'original_postclip_linf_v1'
        assert cfg['comparison_peak_reference_eps'] == .185
        assert cfg['comparison_peak_matches_l2'] is False
        assert len(cfg['comparison_peak_adapter_sha256']) == 64
    assert matrix['cells'][-1]['cfg']['trigger'] == 'md_multicarrier_peak_matched'


def test_native_control_is_separate_profile_without_projection(tmp_path, sources):
    native = runner.build_matrix(tmp_path / 'data', tmp_path / 'native', 'cpu', 0,
                                 budget_mode='native')
    assert native['traditional_comparison_profile'] == runner.NATIVE_PROFILE
    assert native['metrics_contract']['shared_peak_ceiling'] is False
    assert all('comparison_peak_budget' not in cell['cfg'] for cell in native['cells'])
    runner._validate_manifest(native)


@pytest.mark.parametrize('key,value', [('comparison_peak_reference_eps', .2),
    ('comparison_peak_budget', 'nominal_epsilon'), ('comparison_peak_matches_l2', True)])
def test_rehashed_peak_budget_change_is_rejected(matrix, key, value):
    matrix['cells'][0]['cfg'][key] = value
    matrix['plan_sha256'] = runner._plan_fingerprint(matrix)
    with pytest.raises(ValueError):
        runner._validate_manifest(matrix)


def test_native_and_shared_peak_cannot_resume_each_other(tmp_path, sources):
    runner.main(_cli(tmp_path, '--dry-run'))
    with pytest.raises(ValueError, match='settings/source changed'):
        runner.main(_cli(tmp_path, '--dry-run', '--budget-mode', 'native'))
