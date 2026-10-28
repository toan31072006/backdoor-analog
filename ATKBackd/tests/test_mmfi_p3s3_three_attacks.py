"""CPU-only contract tests; no full datasets, CUDA jobs or remote launch."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_mmfi_p3_s3_three_attacks_4090 as run


COMMON = (
    'model', 'seed', 'experiment_name', 'mmfi_protocol', 'mmfi_setting',
    'mmfi_split_seed', 'mmfi_random_ratio', 'optimizer', 'lr', 'momentum',
    'weight_decay', 'batch_size', 'epochs', 'victim_epochs', 'num_workers',
    'pretrained', 'data_parallel', 'rho', 'pivot', 'eps', 'theta_max_deg',
    'payload_axis', 'target_joints', 'dose_mode', 'dose_min', 'dose_max',
    'dose_grid', 'dose_coupling', 'victim_loss', 'poison_select',
)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'sources', lambda: {'runner.py': 'a' * 64})
    home = tmp_path / 'host'
    home.mkdir()
    return run.build_plan(home, home / 'runs' / 'new_three_attacks')


def save_plan(plan):
    run.write(Path(plan['outdir']) / 'plan.json', plan, immutable=True)


@pytest.mark.parametrize('method', run.METHODS)
def test_all_three_configs_preserve_frozen_a1_p3s3_victim(tmp_path, method):
    import run_mmfi_p3_splits as p3
    cfg = run.configuration(tmp_path, method)
    reference = p3._configuration(tmp_path, 'cuda:0', 's3', 'a1')
    assert {k: cfg.get(k) for k in COMMON} == {k: reference.get(k) for k in COMMON}
    assert (cfg['optimizer'], cfg['lr'], cfg['momentum'], cfg['weight_decay']) == (
        'sgd', .001, .9, 0.)
    assert (cfg['epochs'], cfg['victim_epochs'], cfg['batch_size']) == (50, 50, 32)
    assert cfg['num_workers'] == 0
    assert cfg['strict_resume'] is True
    assert cfg['ckpt_every'] == 1
    assert cfg['mmfi_protocol'] == 'protocol3' and cfg['mmfi_setting'] == 's3'
    assert cfg['rho'] == .1 and cfg['dose_coupling'] == 'paired'
    assert cfg['target_joints'] == [2, 3] and cfg['dose_grid'] == list(run.GRID)
    assert not any(k.startswith('comparison_peak_') for k in cfg)


def test_badnets_and_sig_are_existing_fixed_native_operators(tmp_path):
    from train_backdoor import build_trigger
    patch_cfg = run.configuration(tmp_path, 'badnets')
    patch = build_trigger(patch_cfg)
    assert patch_cfg['badnets_patch_subcarriers'] == 3
    assert patch_cfg['badnets_patch_packets'] == 3
    assert patch_cfg['badnets_patch_opacity'] == 1.
    assert patch_cfg['badnets_pattern'] == 'white'
    assert patch.mask.sum() == 3 * 3 * 3
    assert patch.patch_start == (111, 7)
    sine_cfg = run.configuration(tmp_path, 'sig_source')
    sine = build_trigger(sine_cfg)
    assert sine_cfg['sig_delta_255'] == 40.
    assert sine_cfg['sig_frequency'] == 6 and sine_cfg['sig_axis'] == 'packet'
    assert sine_cfg['clean_label_cover_ratio'] == 0.
    expected = np.broadcast_to((40 / 255) * np.sin(2 * np.pi * np.arange(10) * 6 / 10), (3, 114, 10))
    np.testing.assert_array_equal(sine.pattern, expected)
    for trigger in (patch, sine):
        x = np.full((3, 114, 10), .5, np.float32)
        np.testing.assert_array_equal(trigger.inject(x, 0., .185), x)
        assert not np.array_equal(trigger.inject(x, 1., .185), x)


def test_iad_extra_mask_compute_does_not_extend_victim_budget(tmp_path):
    cfg = run.configuration(tmp_path, 'input_aware')
    assert cfg['input_aware']['mask_warmup_epochs'] == 25
    assert cfg['input_aware']['mask_frozen_during_victim_training'] is True
    assert cfg['input_aware']['upstream_commit'] == '1750e4ff0cc2371b243f1fd0ee18971f7e49fe78'
    assert cfg['epochs'] == cfg['victim_epochs'] == 50
    assert cfg['attacker_access'] == 'white_box_generator_and_victim_training'
    assert cfg['clean_label_cover_ratio'] == .1


def test_unknown_method_is_not_silently_substituted(tmp_path):
    with pytest.raises(ValueError, match='three approved'):
        run.configuration(tmp_path, 'wanet_source')


def test_plan_freezes_scope_counts_and_serial_queue(plan):
    assert plan['methods'] == ['input_aware', 'badnets', 'sig_source']
    assert plan['jobs_per_gpu'] == 1 and plan['device'] == 'cuda:0'
    assert plan['contract']['counts'] == {'training': 240570, 'test': 80190}
    assert plan['contract']['final_epoch'] == 49
    assert plan['contract']['batch_size'] == 32 and plan['contract']['drop_last'] is False
    assert plan['contract']['amp'] is False
    assert plan['contract']['test_checkpoint_selection'] is False
    assert plan['contract']['scratch'] is True
    assert plan['contract']['status'].startswith('EXPLORATORY_SINGLE_SEED')
    value = dict(plan)
    expected = value.pop('plan_sha256')
    assert run.digest(value) == expected


@pytest.mark.parametrize('relative', ['runs', 'datasets/new', 'runs/mmfi_p3_hpeli_s42_v1',
    'runs/mmfi_p3_hpeli_s42_v1/new', 'runs/mmfi_p3_hpeli_extra_s42_3090_v1',
    'runs/mmfi_three_tables_4090_win_fresh_v1'])
def test_protected_output_roots_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        run.build_plan(tmp_path, tmp_path / relative)


@pytest.mark.parametrize('device,jobs', [('cpu', 1), ('cuda:1', 1), ('cuda:0', 2)])
def test_changed_execution_policy_rejected(tmp_path, device, jobs):
    with pytest.raises(ValueError, match='serial'):
        run.build_plan(tmp_path, tmp_path / 'runs/new', device, jobs)


@pytest.mark.parametrize('kind', ['source', 'runtime', 'config', 'method_order'])
def test_saved_plan_rejects_source_runtime_config_and_scope_drift(plan, kind):
    save_plan(plan)
    changed = copy.deepcopy(plan)
    if kind == 'source': changed['sources']['runner.py'] = 'b' * 64
    elif kind == 'runtime': changed['runtime']['python'] = 'different'
    elif kind == 'config': changed['cfg']['badnets']['lr'] = .1
    else: changed['methods'].reverse()
    with pytest.raises(ValueError, match='Plan/source/runtime'):
        run.load_plan(changed)


class TinyFrames:
    def __init__(self, n=100):
        self.items = [dict(name=f'E1_S01_A01_{i}', csi=f'csi/{i}.npy',
            kpt='pose/sequence.npy', frame_idx=i) for i in range(n)]
    def __len__(self): return len(self.items)
    def load_raw(self, path):
        index = int(Path(path).stem)
        return np.full((3, 114, 10), .25 + index / 1000, np.float32)
    @staticmethod
    def load_pose(path, frame_idx):
        return np.arange(51, dtype=np.float32).reshape(1, 17, 3) / 100
    @staticmethod
    def normalize(raw):
        return np.asarray(raw, dtype=np.float32).copy()
    def __getitem__(self, index):
        pose = np.arange(51, dtype=np.float32).reshape(1, 17, 3) / 100
        return dict(csi=np.full((3, 114, 10), .25 + index / 1000, np.float32),
                    pose=pose, name=self.items[index]['name'])


def test_exact_paired_poison_draws_are_shared_and_cross_ids_disjoint(tmp_path):
    base = TinyFrames()
    records, pairs_by_method = {}, {}
    for method in run.METHODS:
        pairs = run.paired_data(base, run.Deferred(), run.configuration(tmp_path, method))
        records[method], cross = run.poison_summary(pairs, method)
        pairs_by_method[method] = pairs
        assert pairs.n_poison == 10
        assert all(.2 <= d <= 1 for _, d in pairs.poison_plan)
        assert not cross.intersection(pairs.poison_idx)
        assert len(cross) == (10 if method == 'input_aware' else 0)
        assert run.poison_summary(pairs, method) == (records[method], cross)
    assert len({r['common_poison_sha256'] for r in records.values()}) == 1
    assert len({r['poison_plan_sha256'] for r in records.values()}) == 1
    expected = pairs_by_method['input_aware'].poison_plan
    assert all(p.poison_plan == expected for p in pairs_by_method.values())
    a1_cfg = run.configuration(tmp_path, 'badnets')
    reference = run.paired_data(base, run.Deferred(), a1_cfg)
    assert reference.poison_plan == expected


def test_training_rows_and_collate_keep_physical32_and_clean_cross_labels(tmp_path):
    pairs = run.paired_data(TinyFrames(), run.Deferred(), run.configuration(tmp_path, 'input_aware'))
    _, cross = run.poison_summary(pairs, 'input_aware')
    data = run.TrainingRows(pairs, cross)
    indices = sorted(pairs.poison_idx)[:4] + sorted(cross)[:4]
    indices += [i for i in range(len(pairs)) if i not in set(indices)][:32 - len(indices)]
    batch = run.train_collate([data[i] for i in indices])
    assert batch['csi'].shape == (32, 3, 114, 10)
    assert batch['cross'].dtype == torch.bool
    assert not (batch['cross'] & batch['poisoned'].bool()).any()
    torch.testing.assert_close(batch['pose'][batch['cross']], batch['clean_pose'][batch['cross']])


def test_wddm_desktop_is_not_misclassified_as_training():
    output = '\n'.join(['12, C:\\Windows\\explorer.exe', '13, C:\\App\\Chrome.exe',
        '14, C:\\App\\Code.exe', '15, C:\\App\\python.exe', '16, [Not Found]'])
    assert run.active_compute(output, True, 15) == [(16, '[Not Found]')]
    assert run.active_compute(output, True, 999)[0][0] == 15
    assert len(run.active_compute(output, False, 999)) == 5


@pytest.mark.parametrize('text', ['n/a, python.exe', '12', '12, python.exe, extra'])
def test_unidentifiable_gpu_query_fails_closed(text):
    with pytest.raises(ValueError, match='Cannot identify'):
        run.active_compute(text, True, 999)


def test_require_idle_does_not_stop_other_jobs(monkeypatch):
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='123, C:\\Python\\python.exe\n')
    monkeypatch.setattr(run.subprocess, 'run', query)
    with pytest.raises(ValueError, match='no process stopped'):
        run.require_idle()
    assert len(calls) == 1 and calls[0][0] == 'nvidia-smi'


def test_immutable_record_refuses_changes_and_unfinished_write(tmp_path):
    path = tmp_path / 'record.json'
    run.write(path, {'value': 1}, immutable=True)
    run.write(path, {'value': 1}, immutable=True)
    with pytest.raises(ValueError, match='immutable'):
        run.write(path, {'value': 2}, immutable=True)
    (tmp_path / 'other.json.saving').touch()
    with pytest.raises(ValueError, match='Unfinished'):
        run.write(tmp_path / 'other.json', {'value': 1})


def valid_checkpoint(method='badnets', epoch=49):
    from train_backdoor import _rng_state
    cfg = {'trigger': method, 'epochs': 50}
    binding = {'profile': run.PROFILE, 'plan_sha256': 'a' * 64, 'method': method,
               'config_sha256': run.digest(cfg)}
    rng = _rng_state()
    rng['cuda'] = [torch.zeros(16, dtype=torch.uint8)]
    blob = dict(binding=binding, phase='victim', epoch=epoch,
                mask_epoch=24 if method == 'input_aware' else -1,
                cfg=cfg, model={'weight': torch.ones(1)},
                optimizer={'state': {}, 'param_groups': [{'lr': .001}]}, rng=rng,
                trigger=None, trigger_optimizers={}, schedulers={})
    if method == 'input_aware':
        blob['trigger'] = {'weight': torch.ones(1), '_mask_frozen': torch.tensor(True)}
        blob['trigger_optimizers'] = {k: {'state': {}, 'param_groups': [{}]} for k in ('mask', 'generator')}
        blob['schedulers'] = {k: {'last_epoch': 25 if k == 'mask' else epoch + 1} for k in ('mask', 'generator')}
    return blob, binding


@pytest.mark.parametrize('method', run.METHODS)
def test_checkpoint_has_required_resume_state(method):
    blob, binding = valid_checkpoint(method)
    run.validate_checkpoint(blob, binding, method)


@pytest.mark.parametrize('missing', ['model', 'optimizer', 'rng'])
def test_checkpoint_missing_core_state_rejected(missing):
    blob, binding = valid_checkpoint()
    blob.pop(missing)
    with pytest.raises(ValueError, match='state'):
        run.validate_checkpoint(blob, binding, 'badnets')


@pytest.mark.parametrize('missing', ['python', 'numpy', 'torch', 'cuda'])
def test_checkpoint_missing_any_rng_stream_rejected(missing):
    blob, binding = valid_checkpoint()
    blob['rng'].pop(missing)
    with pytest.raises(ValueError, match='RNG'):
        run.validate_checkpoint(blob, binding, 'badnets')


@pytest.mark.parametrize('missing', ['trigger', 'trigger_optimizers', 'schedulers'])
def test_iad_checkpoint_needs_both_generators_optimizers_and_schedulers(missing):
    blob, binding = valid_checkpoint('input_aware')
    blob.pop(missing)
    with pytest.raises(ValueError, match='Input-Aware'):
        run.validate_checkpoint(blob, binding, 'input_aware')


@pytest.mark.parametrize('field,value', [('plan_sha256', 'other'), ('execution', {'gpu': 'other'}),
    ('poison', {'n_poison': 1}), ('config_sha256', 'other')])
def test_resume_rejects_foreign_source_execution_poison_or_config_binding(field, value):
    blob, binding = valid_checkpoint()
    changed = dict(binding, **{field: value})
    with pytest.raises(ValueError, match='binding'):
        run.validate_checkpoint(blob, changed, 'badnets')


@pytest.mark.parametrize('method,phase,epoch,mask_epoch', [
    ('badnets', 'mask', -1, 0), ('badnets', 'victim', 49.5, -1),
    ('badnets', 'victim', True, -1), ('badnets', 'victim', 49, 24),
    ('input_aware', 'victim', 0, 0), ('input_aware', 'mask', 0, 0),
    ('input_aware', 'mask', -1, 25), ('input_aware', 'victim', 50, 24)])
def test_impossible_checkpoint_phase_or_epoch_rejected(method, phase, epoch, mask_epoch):
    blob, binding = valid_checkpoint(method)
    blob.update(phase=phase, epoch=epoch, mask_epoch=mask_epoch)
    with pytest.raises(ValueError):
        run.validate_checkpoint(blob, binding, method)


def gate_fixture(plan, monkeypatch):
    root = Path(plan['outdir'])
    root.mkdir(parents=True)
    inputs = {'input_stat_fingerprint': 'input-stats', 'poison': {m: {'n_poison': 24057} for m in run.METHODS}}
    run.write(root / 'inputs.json', inputs)
    execution = {'gpu_name': 'NVIDIA GeForce RTX 4090', 'device': 'cuda:0'}
    monkeypatch.setattr(run, 'verify_inputs', lambda p: inputs)
    monkeypatch.setattr(run, 'gpu', lambda: execution)
    records = {}
    for method in run.METHODS:
        path = root / 'smoke' / f'{method}.json'
        run.write(path, dict(plan_sha256=plan['plan_sha256'], method=method,
            execution=execution, real_input=True, batch_size=32, steps=2,
            poison=inputs['poison'][method], loss={'pose_loss': .1}))
        records[method] = {'path': path.relative_to(root).as_posix(), 'sha256': run.sha(path)}
    gate = dict(plan_sha256=plan['plan_sha256'], inputs_sha256=run.sha(root / 'inputs.json'),
                execution=execution, jobs_per_gpu=1, records=records)
    run.write(root / 'gpu_gate.json', gate)
    return root, inputs, gate


def test_gate_accepts_only_all_three_real_physical32_smokes(plan, monkeypatch):
    _, inputs, _ = gate_fixture(plan, monkeypatch)
    assert run.gate_check(plan) == inputs


@pytest.mark.parametrize('field,value', [('plan_sha256', 'wrong'), ('jobs_per_gpu', 2),
    ('inputs_sha256', 'wrong'), ('execution', {'gpu_name': 'other'})])
def test_gate_rejects_foreign_plan_inputs_execution_or_concurrency(plan, monkeypatch, field, value):
    root, _, gate = gate_fixture(plan, monkeypatch)
    gate[field] = value
    run.write(root / 'gpu_gate.json', gate)
    with pytest.raises(ValueError, match='gate'):
        run.gate_check(plan)


@pytest.mark.parametrize('field,value', [('batch_size', 16), ('steps', 0),
    ('real_input', False), ('method', 'other'), ('plan_sha256', 'other'),
    ('poison', {'n_poison': 0})])
def test_gate_rejects_incomplete_or_unmatched_smoke_even_with_updated_file_hash(plan, monkeypatch, field, value):
    root, _, gate = gate_fixture(plan, monkeypatch)
    smoke = root / gate['records']['badnets']['path']
    row = run.read(smoke)
    row[field] = value
    run.write(smoke, row)
    gate['records']['badnets']['sha256'] = run.sha(smoke)
    run.write(root / 'gpu_gate.json', gate)
    with pytest.raises(ValueError, match='smoke'):
        run.gate_check(plan)


def test_gate_cannot_omit_any_approved_method(plan, monkeypatch):
    root, _, gate = gate_fixture(plan, monkeypatch)
    gate['records'].pop('input_aware')
    run.write(root / 'gpu_gate.json', gate)
    with pytest.raises(ValueError, match='three smokes'):
        run.gate_check(plan)


def test_inspect_is_read_only_and_never_starts_training(plan, monkeypatch, capsys):
    home = Path(plan['home'])
    prior = home / 'prior.npy'
    prior.write_bytes(b'prior')
    plan['cfg']['badnets']['action_npy'] = str(prior)
    monkeypatch.setattr(run, 'verify_bundle', lambda: None)
    monkeypatch.setattr(run, 'build_plan', lambda *a: plan)
    calls = []
    monkeypatch.setattr(run.subprocess, 'run', lambda command, **kw:
        calls.append(command) or SimpleNamespace(stdout='idle GPU', stderr=''))
    monkeypatch.setattr(run, 'child', lambda *a: pytest.fail('Inspect started a worker'))
    monkeypatch.setattr(run, 'gpu', lambda: pytest.fail('Inspect initialized CUDA'))
    before = sorted(str(p.relative_to(home)) for p in home.rglob('*'))
    run.main(['--mode', 'inspect', '--data-home', str(home)])
    assert sorted(str(p.relative_to(home)) for p in home.rglob('*')) == before
    assert not Path(plan['outdir']).exists()
    assert calls == [['nvidia-smi']]
    assert '"full_training_started": false' in capsys.readouterr().out


def main_mocks(plan, monkeypatch):
    import run_mmfi_tables
    import shutil
    monkeypatch.setattr(run, 'verify_bundle', lambda: None)
    monkeypatch.setattr(run, 'build_plan', lambda *a: plan)
    monkeypatch.setattr(run_mmfi_tables, '_cell_lock', lambda *a: nullcontext())
    monkeypatch.setattr(run, 'require_idle', lambda: None)
    monkeypatch.setattr(run, 'gate_check', lambda p: {'verified': True})
    monkeypatch.setattr(shutil, 'disk_usage', lambda p: SimpleNamespace(free=10 * 2**30))


def test_check_mode_checks_gate_without_any_training(plan, monkeypatch, capsys):
    main_mocks(plan, monkeypatch)
    monkeypatch.setattr(run, 'child', lambda *a: pytest.fail('Check launched training'))
    run.main(['--mode', 'check', '--data-home', plan['home']])
    assert 'PRELAUNCH_OK; no full training' in capsys.readouterr().out


def test_serial_launch_waits_each_child_and_skips_verified_complete(plan, monkeypatch):
    main_mocks(plan, monkeypatch)
    done = {'badnets'}
    calls = []
    monkeypatch.setattr(run, 'completed', lambda p, m: {'complete': True} if m in done else None)
    def child(p, work, method):
        assert work == 'train'
        calls.append(method)
        done.add(method)
    monkeypatch.setattr(run, 'child', child)
    monkeypatch.setattr(run, 'export', lambda p: calls.append('export'))
    run.main(['--mode', 'launch', '--data-home', plan['home']])
    assert calls == ['input_aware', 'sig_source', 'export']


def test_failed_child_stops_queue_without_retry_or_later_job(plan, monkeypatch):
    main_mocks(plan, monkeypatch)
    calls = []
    monkeypatch.setattr(run, 'completed', lambda p, m: None)
    def child(p, work, method):
        calls.append(method)
        raise RuntimeError('failed worker')
    monkeypatch.setattr(run, 'child', child)
    monkeypatch.setattr(run, 'export', lambda p: pytest.fail('Incomplete run exported'))
    with pytest.raises(RuntimeError, match='failed worker'):
        run.main(['--mode', 'launch', '--data-home', plan['home']])
    assert calls == ['input_aware']


def test_child_command_always_pins_scope_device_and_one_job(plan):
    args = run.command(plan, 'train', 'badnets')
    assert args[args.index('--device') + 1] == 'cuda:0'
    assert args[args.index('--jobs-per-gpu') + 1] == '1'
    assert args[args.index('--mode') + 1] == 'cell'
    assert args[args.index('--work') + 1] == 'train'
    assert args[args.index('--method') + 1] == 'badnets'


def input_receipt_fixture(plan, monkeypatch):
    root = Path(plan['outdir'])
    data = Path(plan['home']) / 'datasets' / 'Compress'
    data.mkdir(parents=True)
    sample = data / 'frame.npy'
    sample.write_bytes(b'input frame')
    prior = Path(plan['home']) / 'actions' / 'data_bend.npy'
    prior.parent.mkdir()
    prior.write_bytes(b'authentic prior fixture')
    prior_sha = run.sha(prior)
    monkeypatch.setattr(run, 'ACTION_SHA', prior_sha)
    plan['cfg']['badnets']['action_npy'] = str(prior)
    save_plan(plan)
    files = [{'path': 'frame.npy', 'bytes': sample.stat().st_size, 'sha256': run.sha(sample)}]
    receipt = dict(plan_sha256=plan['plan_sha256'], action_sha256=prior_sha,
        dataset_root=str(data), files=files, full_content_sha256=run.digest(files),
        input_stat_fingerprint=run.input_stats(data, files), poison={})
    run.write(root / 'inputs.json', receipt)
    return root, sample, prior, receipt


def test_input_receipt_validates_existing_inputs_without_training(plan, monkeypatch):
    _, _, _, receipt = input_receipt_fixture(plan, monkeypatch)
    monkeypatch.setattr(run, 'child', lambda *a: pytest.fail('Receipt check launched a worker'))
    assert run.verify_inputs(plan) == receipt


@pytest.mark.parametrize('change', ['prior', 'dataset_size', 'dataset_mtime', 'content_record', 'plan'])
def test_input_receipt_rejects_prior_dataset_metadata_and_plan_drift(plan, monkeypatch, change):
    root, sample, prior, receipt = input_receipt_fixture(plan, monkeypatch)
    if change == 'prior': prior.write_bytes(b'changed prior')
    elif change == 'dataset_size': sample.write_bytes(b'different sized data')
    elif change == 'dataset_mtime':
        import os
        stat = sample.stat()
        os.utime(sample, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    elif change == 'content_record':
        receipt['files'][0]['sha256'] = 'changed'
        run.write(root / 'inputs.json', receipt)
    else:
        plan['runtime']['python'] = 'different'
    with pytest.raises(ValueError):
        run.verify_inputs(plan)


def completion_fixture(plan, method='badnets'):
    root = Path(plan['outdir'])
    folder = root / method
    folder.mkdir(parents=True)
    inputs = {'input_stat_fingerprint': 'input-stats', 'poison': {m: {'n_poison': 24057} for m in run.METHODS}}
    execution = {'gpu_name': 'NVIDIA GeForce RTX 4090', 'device': 'cuda:0'}
    run.write(root / 'inputs.json', inputs)
    run.write(root / 'gpu_gate.json', {'execution': execution})
    blob, _ = valid_checkpoint(method)
    blob['cfg'] = plan['cfg'][method]
    blob['binding'] = run.checkpoint_binding(plan, method, inputs, execution, inputs['poison'][method])
    torch.save(blob, folder / 'checkpoint.pt')
    (folder / 'per_sample.npz').write_bytes(b'per-sample fixture')
    report = dict(status='complete', method=method, plan_sha256=plan['plan_sha256'],
        checkpoint_epoch=49, frames=80190, checkpoint_sha256=run.sha(folder / 'checkpoint.pt'),
        per_sample_sha256=run.sha(folder / 'per_sample.npz'))
    run.write(folder / 'evaluation.json', report)
    return folder, blob, report


@pytest.mark.parametrize('method', run.METHODS)
def test_completion_requires_bound_real_final49_weights_and_state(plan, method):
    _, _, report = completion_fixture(plan, method)
    assert run.completed(plan, method) == report


@pytest.mark.parametrize('change', ['partial_epoch', 'missing_optimizer', 'missing_rng',
                                  'cfg_drift', 'method_drift', 'mask_not_complete'])
def test_completion_report_cannot_replace_final_checkpoint_or_resume_state(plan, change):
    method = 'input_aware' if change == 'mask_not_complete' else 'badnets'
    folder, blob, report = completion_fixture(plan, method)
    if change == 'partial_epoch': blob['epoch'] = 48
    elif change == 'missing_optimizer': blob.pop('optimizer')
    elif change == 'missing_rng': blob.pop('rng')
    elif change == 'cfg_drift': blob['cfg'] = dict(blob['cfg'], lr=.1)
    elif change == 'method_drift': blob['binding']['method'] = 'sig_source'
    else: blob['mask_epoch'] = 23
    torch.save(blob, folder / 'checkpoint.pt')
    report['checkpoint_sha256'] = run.sha(folder / 'checkpoint.pt')
    run.write(folder / 'evaluation.json', report)
    with pytest.raises(ValueError):
        run.completed(plan, method)


def test_native_resume_continues_final_epoch_and_atomically_retains_full_state(plan, monkeypatch):
    """Toy CPU orchestration only; no HPE training, real data or CUDA execution."""
    import run_mmfi_p3_splits as p3
    import train_backdoor
    from train_backdoor import build_trigger
    real_rng_state = train_backdoor._rng_state
    def cpu_fixture_rng():
        state = real_rng_state()
        state['cuda'] = [torch.zeros(16, dtype=torch.uint8)]
        return state
    monkeypatch.setattr(train_backdoor, '_rng_state', cpu_fixture_rng)
    root = Path(plan['outdir'])
    root.mkdir(parents=True)
    base = TinyFrames()
    cfg = plan['cfg']['badnets']
    pairs = run.paired_data(base, build_trigger(cfg), cfg)
    poison, _ = run.poison_summary(pairs, 'badnets')
    inputs = {'input_stat_fingerprint': 'toy-inputs', 'poison': {'badnets': poison}}
    run.write(root / 'inputs.json', inputs)
    execution = {'gpu_name': 'mock GPU execution'}
    binding = run.checkpoint_binding(plan, 'badnets', inputs, execution, poison)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    blob, _ = valid_checkpoint('badnets', epoch=48)
    blob.update(binding=binding, cfg=cfg, model=model.state_dict(), optimizer=optimizer.state_dict())
    folder = root / 'badnets'
    folder.mkdir()
    torch.save(blob, folder / 'checkpoint.pt')
    monkeypatch.setattr(run, 'verify_inputs', lambda p: inputs)
    monkeypatch.setattr(run, 'gpu', lambda: execution)
    monkeypatch.setattr(run, 'make_components', lambda *a: (model, optimizer, build_trigger(cfg), {}, {}))
    monkeypatch.setattr(p3, '_dataset', lambda *a: base)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    observed = []
    monkeypatch.setattr(run, 'step', lambda method, model, opt, trigger, opts, batch, donor:
        observed.append(batch['csi'].shape[0]) or {'pose_loss': .1})
    evaluations = []
    monkeypatch.setattr(run, 'evaluate', lambda p, m: evaluations.append(m))
    run.train_cell(plan, 'badnets')
    assert observed == [32, 32, 32, 4]
    assert evaluations == ['badnets']
    saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert saved['epoch'] == 49 and saved['phase'] == 'victim' and saved['mask_epoch'] == -1
    assert saved['binding'] == binding
    assert not (folder / 'checkpoint.pt.saving').exists()
    assert saved['model'] and saved['optimizer'] and saved['rng']
