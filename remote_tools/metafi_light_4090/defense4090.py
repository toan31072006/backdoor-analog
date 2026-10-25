"""Windows/one-GPU light MetaFi++ A1 defense from the authenticated Vast snapshot.

NoDef, FT-10 and CLP run serially. FT-10/B32/lr1e-4 is a NEW adaptation, not
the historical FT-100/B256 recipe. Original Linux artifacts remain read-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import capacity4090 as capacity
import portable_source as source

PROFILE = 'metafiplusplus_mmfi_windows_light_a1_defense_s42_v2_nc'
METHODS = ('none', 'ft', 'clp', 'nc')
ROOT = Path(__file__).resolve().parent
FT_EPOCHS = 10
FT_LR = 1e-4
FT_BATCH = 32


def activate(repo):
    os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    sys.path.insert(0, str(Path(repo) / 'ATKBackd'))
    for name in ('a1_nc_bank', 'neural_cleanse_pose'):
        module = f'defend.{name}'
        path = ROOT.parents[1] / 'ATKBackd' / 'defend' / f'{name}.py'
        spec = importlib.util.spec_from_file_location(module, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Missing NC adaptation source: {path}')
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module] = loaded
        spec.loader.exec_module(loaded)


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def preserve_clean_bank(folder, bank):
    """Bind every resumed repair to the same trusted-clean selection."""
    path = Path(folder) / 'clean_bank.json'
    if path.exists():
        if json_read(path) != bank:
            raise ValueError('Saved trusted-clean bank differs; preserve this run and inspect it')
    else:
        atomic(path, bank)


def runtime():
    return dict(python=platform.python_version(), system=platform.system(),
        packages={name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'scipy', 'PyYAML')})


def tool_sources():
    result = {name: source.sha(ROOT / name) for name in
              ('defense4090.py', 'capacity4090.py', 'portable_source.py')}
    for name in ('a1_nc_bank.py', 'neural_cleanse_pose.py'):
        result[name] = source.sha(ROOT.parents[1] / 'ATKBackd' / 'defend' / name)
    return result


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-home', type=Path, default=Path.home() / 'backdooranalog')
    p.add_argument('--archive', type=Path)
    p.add_argument('--outdir', type=Path)
    p.add_argument('--source-root', type=Path)
    p.add_argument('--jobs-per-gpu', type=int, choices=(1,), default=1,
                   help='One exclusive job on the RTX 4090; no GPU sharing')
    modes = p.add_mutually_exclusive_group()
    for key in ('dry-run', 'preflight', 'gpu-smoke', 'launch', 'status', 'check-only'):
        modes.add_argument('--' + key, action='store_true')
    a = p.parse_args(argv)
    a.data_home = a.data_home.resolve()
    a.outdir = (a.outdir or a.data_home / 'runs/metafiplusplus_light_a1_defense4090_s42_v2_nc').resolve()
    a.source_root = (a.source_root or a.data_home / 'imports/metafiplusplus_s42_snapshot_20261025').resolve()
    a.attack_root = a.source_root
    a.repo = ROOT.parents[1]
    if a.archive is None:
        candidates = list((ROOT / 'inputs').glob('metafiplusplus_clean_a1_s42_20261024_164753.tar.gz'))
        candidates += list((Path.home() / 'Downloads').glob('metafiplusplus_clean_a1_s42_20261024_164753.tar.gz'))
        archives = sorted(candidates,
                          key=lambda path: path.stat().st_mtime, reverse=True)
        a.archive = archives[0] if archives else None
    if a.archive: a.archive = a.archive.resolve()
    for root in (a.source_root, a.repo, a.data_home / 'datasets', a.data_home / 'actions'):
        if a.outdir == root or a.outdir.is_relative_to(root) or a.outdir in root.parents:
            raise ValueError('Defense output must be a NEW separate directory')
    return a


def native_hashes():
    import run_vast_seed42
    return run_vast_seed42.source_hashes()


def plan(a):
    import vast_seed42_defense_adapter as adapter
    from dataclasses import asdict
    from defend.neural_cleanse_pose import NCConfig
    ft = adapter.recipe('metafiplusplus', 'ft')
    ft.update(epochs=FT_EPOCHS, lr=FT_LR)
    return dict(profile=PROFILE, seed=42, backbone='metafiplusplus', data_home=str(a.data_home),
        source_root=str(a.source_root), attack_root=str(a.attack_root),
        source_archive_sha256=source.ARCHIVE_SHA256,
        outdir=str(a.outdir), runtime=runtime(), tools=tool_sources(),
        science_sources=native_hashes(), methods=list(METHODS), physical_repair_batch=FT_BATCH,
        evaluation_batch=32, activation_offload='none', jobs_per_gpu=1,
        recipes={'none': {}, 'ft': ft, 'clp': adapter.recipe('metafiplusplus', 'clp'),
                 'nc': asdict(NCConfig())},
        serial_order=list(METHODS),
        source_policy='Original final49 Clean/A1 immutable; all repairs restart independently from A1',
        migration='New platform/data-root binding; not exact resume of Linux FT',
        no_automatic_retry=True, no_amp=True, no_batch_fallback=True)


def checkpoint_model(a, key='a1'):
    import torch
    from vast_seed42_adapter import build_model
    model = build_model('metafiplusplus', num_keypoints=17, num_coor=3, num_person=1,
                        subcarrier_num=114, dataset='mmfi', pretrained=False)
    if key not in ('a1', 'clean'): raise ValueError('Unknown source checkpoint')
    blob = torch.load(a.attack_root / f'metafiplusplus/seed42/{key}/checkpoint.pt', map_location='cpu', weights_only=False)
    if blob['epoch'] != 49: raise ValueError('Source is not final49')
    model.load_state_dict(blob['model'], strict=True)
    return model, blob.get('trigger')


def verify_preflight(a, current_plan, full=False):
    from vast_seed42_preflight import check_receipt
    p = json_read(a.outdir / 'preflight.json')
    if (p.get('status') != 'PASS' or p.get('plan_sha256') != source.digest(current_plan)
            or p.get('sha256') != source.digest({k: v for k, v in p.items() if k != 'sha256'})):
        raise ValueError('Preflight/source/runtime/path changed; do not reuse this output')
    for key in ('clean', 'a1'):
        if source.sha(a.attack_root / f'metafiplusplus/seed42/{key}/checkpoint.pt') != p['source']['cells'][key]['checkpoint_sha256']:
            raise ValueError('Original source checkpoint changed')
    if source.authenticate(a.attack_root, a.repo / 'ATKBackd', load_weights=False) != p['source']:
        raise ValueError('Imported Clean/A1 source artifacts changed')
    data = json_read(a.outdir / 'dataset_content.json')
    if data['binding_sha256'] != p['dataset_binding_sha256']:
        raise ValueError('Local input content binding differs')
    check_receipt(data, full_hash=full)
    return p


def hardware():
    import torch
    from run_piw_hpeli_clean import _setup_execution
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
    if torch.cuda.device_count() != 1: raise ValueError('This launcher requires one visible GPU')
    props = torch.cuda.get_device_properties(0)
    if 'RTX 4090' not in props.name:
        raise ValueError('This frozen light profile requires an RTX 4090')
    return dict(name=props.name, uuid=str(getattr(props, 'uuid', None)), total_memory=props.total_memory,
                capability=[props.major, props.minor], backend=_setup_execution('cuda:0'),
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def verify_smoke(a, current_plan):
    p = verify_preflight(a, current_plan)
    record = json_read(a.outdir / 'gpu_smoke.json')
    if (record.get('status') != 'PASS' or record.get('preflight_sha256') != p['sha256']
            or record.get('hardware') != hardware()
            or record.get('sha256') != source.digest({k: v for k, v in record.items() if k != 'sha256'})):
        raise ValueError('GPU capacity/source/runtime gate is missing or differs')
    if record.get('jobs_per_gpu') != 1 or len(record.get('waves', [])) != len(METHODS):
        raise ValueError('Incomplete exclusive serial capacity smoke')
    for wave, method in zip(record['waves'], METHODS):
        capacity.validate_wave(wave, method)
    return record


def ensure_gpu_not_used_by_other_python():
    """Allow Windows display apps, refuse other Python CUDA workloads."""
    import csv
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=30)
    for row in csv.reader(query.stdout.splitlines()):
        if len(row) != 2: continue
        pid, name = int(row[0].strip()), Path(row[1].strip()).name.lower()
        if pid != os.getpid() and (name.startswith('python') or name.startswith('ipython')):
            raise ValueError(f'Other Python GPU process PID={pid}; inspect it, no process stopped')


def cell_binding(a, method, preflight):
    return source.digest(dict(profile=PROFILE, preflight_sha256=preflight['sha256'], method=method,
                              device='cuda:0', original_source='a1-final49', independent_repair=True))


def completed(a, method, preflight):
    from defend.a1_defense_metrics import enrich_evaluation
    import numpy as np
    folder = a.outdir / 'metafiplusplus/seed42' / method
    path = folder / 'evaluation.json'
    if not path.exists(): return None
    r = json_read(path)
    if (r.get('profile') != PROFILE or r.get('binding_sha256') != cell_binding(a, method, preflight)
            or r.get('record_sha256') != source.digest({k: v for k, v in r.items() if k != 'record_sha256'})):
        raise ValueError('Defense record identity or binding differs')
    if method == 'nc' and r.get('status') == 'nc_diagnostic_only':
        expected = {'clean_bank.json', 'nc_partitions.json', 'nc_stage/nc_diagnostic.json'}
        if set(r.get('artifacts', {})) != expected:
            raise ValueError('NC diagnostic artifact set differs')
        for name, sha in r['artifacts'].items():
            if source.sha(folder / name) != sha:
                raise ValueError('NC diagnostic artifact changed: ' + name)
        return r
    if r.get('status') != 'complete':
        raise ValueError('Defense record is not a completed evaluation')
    expected = {'per_sample.npz'} | ({'repair.pt'} if method != 'none' else set())
    if method not in ('none', 'clp'): expected.add('clean_bank.json')
    if method == 'nc':
        expected |= {'nc_partitions.json', 'nc_stage/nc_diagnostic.json',
                     'nc_stage/nc_receipt.json', 'nc_stage/nc_unlearned.pt'}
    if set(r['artifacts']) != expected: raise ValueError('Missing defense artifact')
    for name, sha in r['artifacts'].items():
        if source.sha(folder / name) != sha: raise ValueError('Defense artifact changed: ' + name)
    with np.load(folder / 'per_sample.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    ids = [[str(csi), str(pose), int(frame)] for csi, pose, frame in arrays['pair_ids'].tolist()]
    if (len(ids) != 33264 or source.digest(ids) != r['test_ids_sha256']
            or not np.array_equal(arrays['sample_indices'], np.arange(33264))):
        raise ValueError('Ordered TEST identities differ')
    if source.digest(ids) != preflight['source']['cells']['a1']['inputs']['eval']['pair_ids_sha256']:
        raise ValueError('Saved TEST identities no longer match original ordered source')
    replay = enrich_evaluation(dict(metrics=r['metrics'], diagnostics=r['diagnostics'],
                                    dose_rows=r['dose_rows'], arrays=arrays))
    # New outputs canonicalize array layouts before first enrichment; no
    # historical rounding/score rewrite or broad replay tolerance is used.
    if replay['metrics'] != r['metrics'] or replay['dose_rows'] != r['dose_rows']:
        raise ValueError('New defense metrics do not replay exactly')
    if method != 'none':
        import torch
        blob = torch.load(folder / 'repair.pt', map_location='cpu', weights_only=False)
        if blob.get('binding_sha256') != r['binding_sha256'] or blob.get('complete') is not True:
            raise ValueError('Repair checkpoint is incomplete or differently bound')
    return r


def execute(a, method, preflight):
    import numpy as np
    import torch
    import vast_seed42_defense_adapter as adapter
    from train_backdoor import _load_dataset
    from eval_a1_target_pck import _restore_trigger
    from run_peak_confirmation import _pair_ids
    from run_a1_three_tables import _cell_lock
    from run_piw_hpeli_clean import _setup_execution
    from defend.a1_defense_eval import select_clean_bank, clean_loaders, evaluate_defense
    from defend.a1_defense_metrics import enrich_evaluation
    folder = a.outdir / 'metafiplusplus/seed42' / method
    with _cell_lock(folder):
        cached = completed(a, method, preflight)
        if cached: return cached
        _setup_execution('cuda:0'); torch.set_num_threads(2)
        cfg = source.local_config(preflight['source'], a.data_home)
        model, trigger_state = checkpoint_model(a)
        trigger = _restore_trigger(cfg, trigger_state)
        model = model.to('cuda:0')
        train = _load_dataset(cfg, 'training'); test = _load_dataset(cfg, 'validation')
        source.bind_local_ids(preflight['source'], a.data_home, train, test)
        fit = selection = None
        bank = None
        if method not in ('none', 'clp'):
            bank = select_clean_bank(_pair_ids(train), 42, .05, .2)
            preserve_clean_bank(folder, bank)
            fit, selection = clean_loaders(train, bank, cfg, 42, batch_size=FT_BATCH)
        binding = cell_binding(a, method, preflight)
        print(f'[4090 defense] {method}: independent A1 final49; light FT10/B32/lr1e-4; TEST deferred', flush=True)
        if method == 'ft':
            from defend.a1_repair_ops import train_repair
            recipe = adapter.recipe('metafiplusplus', 'ft')
            recipe.update(epochs=FT_EPOCHS, lr=FT_LR)
            model, repaired = train_repair(model, fit, selection, 'ft', recipe,
                                            'cuda:0', folder / 'repair.pt', binding, 42)
            repaired = dict(repaired, architecture=adapter.architecture_audit(model, 'metafiplusplus', 'ft'))
        elif method == 'nc':
            from defend.a1_nc_bank import run as run_nc
            from defend.a1_repair_ops import _atomic_save, _state

            clean_model, _ = checkpoint_model(a, 'clean')
            model, nc_receipt, sections = run_nc(model, clean_model, train, cfg, bank,
                device='cuda:0', folder=folder / 'nc_stage', binding=binding)
            del clean_model
            atomic(folder / 'nc_partitions.json', sections)
            if model is None:
                files = ('clean_bank.json', 'nc_partitions.json', 'nc_stage/nc_diagnostic.json')
                record = dict(profile=PROFILE, status='nc_diagnostic_only', method='nc', seed=42,
                    binding_sha256=binding, nc_diagnostic=nc_receipt,
                    source_checkpoint_sha256=preflight['source']['cells']['a1']['checkpoint_sha256'],
                    artifacts={name: source.sha(folder / name) for name in files},
                    statement='TRAIN-only inversion did not certify a trigger; no repaired model or TEST row')
                record['record_sha256'] = source.digest(record)
                atomic(folder / 'evaluation.json', record)
                print('[4090 defense] NC diagnostic only: ' + nc_receipt['status'], flush=True)
                return completed(a, method, preflight)
            repaired = dict(complete=True, method='nc_pose_adapted_unlearning',
                            nc_receipt=nc_receipt, nc_partitions=sections)
            _atomic_save(dict(binding_sha256=binding, complete=True,
                              selected_model_state=_state(model), metadata=repaired),
                         folder / 'repair.pt')
        else:
            model, repaired = adapter.repair(model, 'metafiplusplus', method, fit, selection,
                'cuda:0', folder / 'repair.pt', binding, 42, activation_offload='none')
        if repaired.get('complete') is not True: raise ValueError('Repair incomplete')
        if method == 'clp':
            from defend.a1_repair_ops import _atomic_save, _state
            _atomic_save(dict(binding_sha256=binding, complete=True, selected_model_state=_state(model),
                              metadata=repaired), folder / 'repair.pt')
        measured = evaluate_defense(model, test, trigger, cfg, 'cuda:0')
        measured['arrays'] = {k: np.ascontiguousarray(v) for k, v in measured['arrays'].items()}
        measured = enrich_evaluation(measured)
        ids = _pair_ids(test)
        measured['arrays']['pair_ids'] = np.asarray(ids, dtype=str)
        buffer = io.BytesIO(); np.savez(buffer, **measured.pop('arrays'))
        path = folder / 'per_sample.npz'; temp = folder / 'per_sample.tmp'
        temp.write_bytes(buffer.getvalue()); os.replace(temp, path)
        files = ['per_sample.npz'] + ([] if method == 'none' else ['repair.pt'])
        if bank is not None: files.append('clean_bank.json')
        if method == 'nc':
            files += ['nc_partitions.json', 'nc_stage/nc_diagnostic.json',
                      'nc_stage/nc_receipt.json', 'nc_stage/nc_unlearned.pt']
        record = dict(profile=PROFILE, status='complete', method=method, seed=42,
            binding_sha256=binding, repair=repaired, source_checkpoint_sha256=preflight['source']['cells']['a1']['checkpoint_sha256'],
            test_ids_sha256=source.digest(ids), artifacts={name: source.sha(folder / name) for name in files},
            platform=runtime(), migration='New Windows evaluation/repair; not Linux FT resume', **measured)
        if method == 'none':
            original = preflight['source']['cells']['a1']['original_metrics']
            record['original_linux_delta'] = {name: record['metrics'][new] - original[old]
                for name, new, old in [('T_mm', 't_mpjpe_mm', 't1_mpjpe_mm'),
                    ('TargetPCK_pp', 'target_pck_pct', 'target_pck_pct'),
                    ('Clean_mm', 'clean_mpjpe_mm', 'clean_mpjpe_mm')]}
        record['record_sha256'] = source.digest(record)
        atomic(folder / 'evaluation.json', record)
        result = completed(a, method, preflight)
        print('[4090 defense] COMPLETE ' + method, flush=True)
        return result


def export(a, records):
    columns = ('t_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm', 'clean_pampjpe_mm',
               'triggered_gt_mpjpe_mm', 'clean_target_mpjpe_mm', 'mean_positive_dce_mm')
    rows = [dict(Method=r['method'], Seed=42, **{k: r['metrics'][k] for k in columns})
            for r in records if r.get('status') == 'complete']
    with (a.outdir / 'defense_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# MetaFi++ / MM-Fi Windows light A1 defense seed42', '',
        'Exploratory new profile. NoDef and CLP are unchanged operators; FT uses 10 epochs, B32, lr1e-4 on trusted 5% clean TRAIN.',
        'All methods start independently from original A1 final49. One seed; no across-seed SD/CI.', '',
        '| Defense | T-MPJPE mm | Target-PCK % | Clean MPJPE 17j mm | Clean MPJPE target2j mm | Triggered-GT target2j mm | Mean DCE mm |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append('| ' + r['Method'] + ' | ' + ' | '.join(f'{r[k]:.2f}' for k in
            ('t_mpjpe_mm', 'target_pck_pct', 'clean_mpjpe_mm', 'clean_target_mpjpe_mm',
             'triggered_gt_mpjpe_mm', 'mean_positive_dce_mm')) + ' |')
    lines += ['', 'Defense success requires reduced triggered-to-true error while preserving clean utility; low Target-PCK alone is insufficient.']
    nc = next((r for r in records if r['method'] == 'nc'), None)
    if nc and nc.get('status') == 'nc_diagnostic_only':
        lines += ['', 'NC-adapted: TRAIN-only inversion did not certify a trigger. Diagnostic saved; no NC repair/TEST row.']
    (a.outdir / 'defense_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    a = get_args(argv)
    if a.status:
        print(json.dumps({m: dict(evaluation_file_present=(a.outdir / 'metafiplusplus/seed42'
                           / m / 'evaluation.json').exists(), completion_verified=False)
                           for m in METHODS}, indent=2))
        return 0
    if a.dry_run or not (a.preflight or a.gpu_smoke or a.launch or a.check_only):
        print('DRY_RUN_ONLY: MetaFi++ A1 NoDef, FT10/B32/lr1e-4, CLP, NC-adapted; one GPU serial; no data/CUDA/training/writes')
        print(json.dumps(dict(profile=PROFILE, archive=str(a.archive),
             expected_archive_sha256=source.ARCHIVE_SHA256, source_root=str(a.source_root),
             attack_root=str(a.attack_root), outdir=str(a.outdir), methods=METHODS,
             ft=dict(epochs=FT_EPOCHS, lr=FT_LR, physical_batch=FT_BATCH),
             official_test_selection=False), indent=2))
        return 0
    if a.preflight and not a.source_root.exists():
        if a.archive is None:
            raise FileNotFoundError('MetaFi++ source import is absent; copy the authenticated tar to Downloads or pass --archive')
        source.unpack(a.archive, a.source_root)
    if not a.repo.is_dir() or not a.attack_root.is_dir():
        raise FileNotFoundError('Missing complete MetaFi++ Clean+A1 tar import; preserve it and inspect the archive')
    activate(a.repo)
    current = plan(a)
    from run_a1_three_tables import _cell_lock
    if a.preflight:
        if a.outdir.is_dir() and not (a.outdir / 'preflight.json').exists():
            if any(path.name != 'run.lock' for path in a.outdir.iterdir()):
                raise ValueError('Nonempty output has no valid new-profile gate; preserve it and choose a NEW output')
        with _cell_lock(a.outdir):
            if (a.outdir / 'preflight.json').exists():
                verify_preflight(a, current, full=True); print('PREFLIGHT_PASS_EXISTING'); return 0
            original = source.authenticate(a.attack_root, a.repo / 'ATKBackd')
            from vast_seed42_preflight import inspect
            from train_backdoor import _load_dataset
            cfg = source.local_config(original, a.data_home)
            data = inspect(a.data_home)
            train = _load_dataset(cfg, 'training'); test = _load_dataset(cfg, 'validation')
            identities = source.bind_local_ids(original, a.data_home, train, test)
            atomic(a.outdir / 'dataset_content.json', data)
            preflight = dict(status='PASS', profile=PROFILE, plan_sha256=source.digest(current),
                source=original, import_archive_sha256=(source.sha(a.archive) if a.archive and a.archive.is_file() else None),
                dataset_binding_sha256=data['binding_sha256'], original_virtual_ids=identities,
                original_vast_raw_content_reverified=False)
            preflight['sha256'] = source.digest(preflight)
            atomic(a.outdir / 'plan.json', current); atomic(a.outdir / 'preflight.json', preflight)
            print('PREFLIGHT_PASS; original weights/config/IDs authenticated; local data bound; no full repair launched')
        return 0
    preflight = verify_preflight(a, current)
    if a.check_only:
        missing = [m for m in METHODS if completed(a, m, preflight) is None]
        print(json.dumps(dict(complete=not missing, missing=missing), indent=2))
        return int(bool(missing))
    if a.gpu_smoke:
        with _cell_lock(a.outdir):
            target = a.outdir / 'gpu_smoke.json'
            if target.exists():
                verify_smoke(a, current); print('GPU_SMOKE_PASS_EXISTING'); return 0
            ensure_gpu_not_used_by_other_python()
            available = capacity.memory_available()
            if available < 8 * 2**30:
                raise ValueError('Less than 8GiB host RAM available before serial B32 smoke')
            print(f'[4090 capacity] available RAM={available/2**30:.1f}GiB; serial one job', flush=True)
            record = dict(status='FAIL', preflight_sha256=preflight['sha256'], jobs_per_gpu=1,
                          hardware=hardware(), available_ram_bytes=available, waves=[])
            atomic(target, record)
            for method in METHODS:
                print('[4090 capacity wave] ' + method, flush=True)
                wave = capacity.run_wave(a.repo, method)
                record['waves'].append(wave); atomic(target, record)
                if not wave['complete']: print(json.dumps(wave, indent=2), flush=True)
                capacity.validate_wave(wave, method)
            record['status'] = 'PASS'; record['sha256'] = source.digest(record); atomic(target, record)
            print('GPU_SMOKE_PASS: serial B32 real-model gate; separate launch required', flush=True)
        return 0
    verify_smoke(a, current)
    with _cell_lock(a.outdir):
        records = []
        for method in METHODS:
            ensure_gpu_not_used_by_other_python()
            execute(a, method, preflight)
        for method in METHODS:
            result = completed(a, method, preflight)
            if result is None: raise ValueError('Incomplete defense: ' + method)
            records.append(result)
        verify_preflight(a, current, full=True)
        export(a, records)
        print('COMPLETE_METAFIPLUSPLUS_LIGHT_A1_DEFENSE_THREE_ROWS; originals unchanged', flush=True)
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print('[4090 defense] ERROR: ' + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
