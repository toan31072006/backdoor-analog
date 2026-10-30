"""Exploratory motion-prior sensitivity: HPE-Li/MM-Fi P1S1/A1/seed42.

Only the external skeleton prior changes. Each requested prior has a separate
fresh victim; cross/nod are the default two new rows. Existing bend results are
never guessed, imported or overwritten. No legacy cross/nod YAML is executed.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', PYTHONUTF8='1')
HERE = Path(__file__).resolve().parent
PROFILE = 'mmfi_p1s1_hpeli_a1_motion_priors_s42_serial_v1'
PLAN = 'motion_priors.plan.json'
PRIORS = ('bend', 'cross', 'nod')
GRID = [0., .2, .4, .6, .8, 1.]

from run_mmfi_badnets_two_models_4090 import (digest, read, write, sources,
    require_idle, gpu_signature, validate_checkpoint)


def private_module(name):
    spec = importlib.util.spec_from_file_location('_motion_priors_' + name,
                                                 HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_engine():
    """Original scientific trainer; only its checkpoint write becomes atomic."""
    engine = private_module('train_backdoor')
    original = engine._save_checkpoint
    def atomic_checkpoint(path, *args, **kwargs):
        path = Path(path)
        temporary = path.with_name(path.name + '.motion-prior.tmp')
        original(temporary, *args, **kwargs)
        os.replace(temporary, path)
    engine._save_checkpoint = atomic_checkpoint
    return engine


def evaluator(source_hashes):
    from vast_seed42_preflight import sha256
    engine = private_module('a1_three_tables_eval')
    def checked_sources():
        actual = {name: sha256(HERE / name) for name in source_hashes}
        if actual != source_hashes:
            raise ValueError('Scientific source changed after the immutable plan')
        return actual
    engine._source_hashes = checked_sources
    return engine


def configuration(home, prior, action):
    import run_a1_three_tables as frozen
    from train_backdoor import _resolve_training_config, _validate_training_contract
    if prior not in PRIORS:
        raise ValueError('Unknown motion prior')
    cfg = frozen._configuration(Path(home), 'cuda:0', 0, 42, 'a1')
    cfg['action_npy'] = str(Path(action).resolve())
    cfg['motion_prior_profile'] = PROFILE
    cfg['motion_prior_name'] = prior
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    return cfg


def action_record(path, prior, approve_labels=False):
    """Validate numeric content and bind label provenance, never infer labels."""
    import numpy as np
    from attack.trigger import velocity_profiles_from_skeleton
    from vast_seed42_preflight import sha256, EXPECTED_ACTION_SHA256
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing {prior} prior: {path}; supply the genuine action file')
    before = path.stat()
    values = np.load(path, allow_pickle=False, mmap_mode='r')
    if (values.ndim != 5 or values.shape[1] != 3 or values.shape[0] < 1
            or values.shape[2] < 2 or values.shape[3] != 25 or values.shape[4] < 1
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError('Action must be finite real numeric NTU N x 3 x T x 25 x M')
    file_hash = sha256(path)
    sidecar = Path(str(path) + '.provenance.json')
    if prior == 'bend':
        if file_hash != EXPECTED_ACTION_SHA256:
            raise ValueError('Bend must match the archived original action SHA256')
        provenance = dict(kind='archived_original_bend_sha256')
    elif sidecar.is_file():
        provenance = read(sidecar)
        if (provenance.get('semantic_approved') is not True
                or provenance.get('action') != prior or provenance.get('sha256') != file_hash):
            raise ValueError('Action provenance/semantic label/SHA256 mismatch')
        provenance = dict(kind='bound_sidecar', path=str(sidecar), sha256=sha256(sidecar),
                          record=provenance)
    elif approve_labels:
        provenance = dict(kind='explicit_user_action_label_attestation', action=prior)
    else:
        raise ValueError(f'{prior}: supply {sidecar.name} provenance or explicitly '
                         'attest the action labels with --approve-action-labels')
    first_person = values[..., 0]
    occupancy = np.count_nonzero(first_person, axis=(1, 2, 3)) / np.prod(first_person.shape[1:])
    selected = int(np.argmax(occupancy))
    skeleton = first_person[selected].transpose(1, 2, 0)
    valid_frames = int((np.abs(skeleton).sum((1, 2)) > 1e-6).sum())
    if valid_frames < 2:
        raise ValueError('Selected action has fewer than two valid frames')
    velocity, position, movers = velocity_profiles_from_skeleton(path, top_k=6)
    if (not np.isfinite(velocity).all() or not np.isfinite(position).all()
            or np.linalg.norm(velocity) <= 1e-12):
        raise ValueError('Selected action has degenerate radial motion')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Action changed during validation')
    return dict(path=str(path), sha256=file_hash, shape=list(values.shape),
        dtype=values.dtype.str, selected_instance=selected, valid_frames=valid_frames,
        moving_joints=list(map(int, movers)), selection='first-person maximum occupancy; native top6 speed',
        fps=30., los=[1., 0., 0.], provenance=provenance)


def trigger_record(cfg):
    import numpy as np
    from train_backdoor import build_trigger
    trigger = build_trigger(cfg)
    arrays = dict(base_complex=trigger.base.m, base_zero_mean=trigger.base.m_zm,
                  p0=trigger.p0, p1=trigger.p1, pattern=trigger.pattern)
    result = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.shape != (3, 114, 10) or not np.isfinite(value).all():
            raise ValueError('Invalid native multicarrier pattern')
        import hashlib
        result[name] = dict(shape=list(value.shape), dtype=value.dtype.str,
                           sha256=hashlib.sha256(value.tobytes()).hexdigest())
    sample = np.full((3, 114, 10), .5, dtype=np.float32)
    if (not np.array_equal(trigger.inject(sample.copy(), 0., cfg['eps']), sample)
            or not np.isfinite(trigger.inject(sample.copy(), 1., cfg['eps'])).all()
            or np.array_equal(trigger.inject(sample.copy(), 1., cfg['eps']), sample)):
        raise ValueError('Trigger must have dose0 identity and a finite nondegenerate dose1')
    return result


def build_plan(home, outdir, priors=('cross', 'nod'), actions=None, approve_labels=False):
    from run_a1_cross_backbones import runtime_versions
    import platform
    home, outdir = Path(home).resolve(), Path(outdir).resolve()
    if (not priors or len(set(priors)) != len(priors) or set(priors) - set(PRIORS)):
        raise ValueError('Choose distinct bend/cross/nod priors')
    if (outdir == home / 'runs' or home / 'runs' not in outdir.parents
            or outdir == home or outdir == Path(outdir.anchor)):
        raise ValueError('Outdir must be a dedicated directory beneath DataHome/runs')
    if outdir.exists() and not (outdir / PLAN).is_file() and any(outdir.iterdir()):
        raise ValueError('Output is nonempty without this profile plan; preserve historical artifacts')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
        raise ValueError('Use physical cuda:0 without CUDA_VISIBLE_DEVICES remapping')
    cells = []
    actions = actions or {}
    for prior in priors:
        action = Path(actions.get(prior) or home / 'actions' / f'data_{prior}.npy')
        cfg = configuration(home, prior, action)
        folder = outdir / prior / 'seed42' / 'a1'
        cells.append(dict(method_key=f'{prior}__s42__a1', key='a1', prior=prior,
            label=f'A1 / {prior}', seed=42, origin='fresh', tables=[], dependencies=[],
            cfg=cfg, run_dir=str(folder), ckpt_dir=str(folder),
            eval_cache=str(folder / 'eval_cache.json'), artifact_dir=str(outdir / 'evaluations' / prior),
            action=action_record(action, prior, approve_labels), trigger_arrays=trigger_record(cfg)))
    hashes = [c['action']['sha256'] for c in cells]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Different prior names cannot reuse the same action bytes')
    value = dict(schema=1, profile=PROFILE, data_home=str(home), outdir=str(outdir),
        device='cuda:0', max_workers=1, priors=list(priors), actions={c['prior']:c['action']['path'] for c in cells},
        approve_action_labels=bool(approve_labels), cells=cells, sources=sources(),
        runtime=dict(runtime_versions(), executable=str(Path(sys.executable).resolve()), system=platform.system()),
        contract=dict(dataset='MM-Fi', protocol='protocol1', setting='s1', split_seed=0,
            train=133056, test=33264, seed=42, epochs=50, final_epoch=49, rho=.1,
            poison_count=13305, paired_doses='uniform U(.2,1)', target_joints=[2,3], pivot=1,
            payload_axis=[0.,0.,1.], theta_max_deg=40., dose_grid=GRID, eps=.185,
            optimizer='SGD lr=.001 momentum=.9 weight_decay=0 physicalbatch32',
            intervention='only external motion prior; independent scratch victims; native A1 operator unchanged',
            clean_training=False, baseline_training=False,
            scope='exploratory single-seed motion-prior sensitivity, not target-action scenarios',
            clean_reference_controls='not included; no matching reference supplied',
            equal_realized_noise=False, uncertainty='not estimated from one seed'))
    value['plan_sha256'] = digest(value)
    return value


def verify_inputs(plan, *, full_hash=False):
    from vast_seed42_preflight import check_receipt
    if sources() != plan['sources']:
        raise ValueError('Scientific source changed after the immutable plan')
    receipt = read(Path(plan['outdir']) / 'inputs.json')
    if (receipt.get('data_home') != plan['data_home']
            or {k:v['n'] for k,v in receipt.get('splits', {}).items()} != dict(train=133056, eval=33264)):
        raise ValueError('Foreign/incomplete full-input receipt')
    check_receipt(receipt, full_hash=full_hash)
    for cell in plan['cells']:
        if action_record(cell['action']['path'], cell['prior'], plan['approve_action_labels']) != cell['action']:
            raise ValueError('Motion prior/provenance changed after Prepare')
        if trigger_record(cell['cfg']) != cell['trigger_arrays']:
            raise ValueError('Native trigger changed after Prepare')
    return receipt


def binding(plan, cell):
    from vast_seed42_preflight import sha256
    return dict(plan_sha256=plan['plan_sha256'], method_key=cell['method_key'],
        inputs_sha256=sha256(Path(plan['outdir']) / 'inputs.json'), gpu=gpu_signature())


def execution():
    import torch
    from run_piw_hpeli_clean import _setup_execution
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; no CPU fallback')
    gpu = torch.cuda.get_device_properties(0)
    if '4090' not in gpu.name or gpu.total_memory < 22 * 2**30:
        raise ValueError('This preparation profile requires Van RTX4090/24GiB')
    return _setup_execution('cuda:0')


def smoke(plan, cell):
    import numpy as np
    import torch
    from torch.utils.data import Subset
    from run_mmfi_p3_s3_three_attacks_4090 import paired_data
    cfg = cell['cfg']; context = execution()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    engine = training_engine()
    data = paired_data(engine._load_dataset(cfg, 'training'), engine.build_trigger(cfg), cfg)
    if len(data) != 133056 or data.n_poison != 13305:
        raise ValueError('Full TRAIN/poison count differs')
    ids = sorted(data.poison_idx)[:8] + [i for i in range(len(data)) if i not in data.poison_idx][:56]
    model = engine.build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                                dataset='mmfi', pretrained=False).to('cuda:0')
    _, optimizer = engine._build_optimizer(model, cfg, 'mmfi')
    parameter = next(p for p in model.parameters() if p.requires_grad)
    before = parameter.detach().clone(); losses = []
    for batch in engine._training_loader(Subset(data, ids), cfg):
        if tuple(batch['csi'].shape) != (32,3,114,10):
            raise ValueError('Smoke requires real TRAIN physical B32')
        losses.append(engine._victim_update(model, optimizer, batch['csi'].to('cuda:0'), batch['pose'].to('cuda:0')))
    updated = not torch.equal(before, parameter.detach())
    if len(losses) != 2 or not all(math.isfinite(x) for x in losses) or not updated:
        raise ValueError('B32 SGD smoke failed; no fallback')
    write(Path(cell['run_dir']) / 'gpu_smoke.json', dict(binding=binding(plan,cell),
        execution=context, physical_batch=32, optimizer_steps=2, real_train_input=True,
        poison_examples=8, parameter_updated=True, losses=losses), immutable=True)
    print(f"GPU_SMOKE_PASS {cell['prior']}; no full training", flush=True)


def gate_check(plan):
    verify_inputs(plan)
    for cell in plan['cells']:
        gate = read(Path(cell['run_dir']) / 'gpu_smoke.json')
        if (gate.get('binding') != binding(plan,cell) or gate.get('physical_batch') != 32
                or gate.get('optimizer_steps') != 2 or gate.get('real_train_input') is not True
                or gate.get('poison_examples') != 8 or gate.get('parameter_updated') is not True
                or len(gate.get('losses',[])) != 2
                or not all(math.isfinite(x) for x in gate['losses'])):
            raise ValueError('Missing/foreign B32 CUDA preparation gate')


def distortion(plan, cell):
    """Predefined first256 full TRAIN IDs; diagnostic only, never retune strength."""
    import numpy as np
    from eval.distortion import distortion_stats, _aggregate
    from train_backdoor import _load_dataset, build_trigger
    from run_peak_confirmation import _pair_ids
    data = _load_dataset(cell['cfg'], 'training'); trigger = build_trigger(cell['cfg'])
    rows=[]
    for dose in GRID:
        stats=[]
        for index in range(256):
            clean = np.asarray(data[index]['csi'], dtype=np.float32)
            hit = trigger.inject(clean.copy(), dose, cell['cfg']['eps'])
            stats.append(distortion_stats(clean, hit))
        metrics = _aggregate(stats)
        metrics['snr_db_infinite'] = math.isinf(metrics['snr_db'])
        if metrics['snr_db_infinite']: metrics['snr_db'] = None
        rows.append(dict(dose=dose, **metrics))
    value = dict(plan_sha256=plan['plan_sha256'], action=cell['action'],
        n=256, split='TRAIN', selection='first256 frozen ordered TRAIN identities',
        pair_ids_sha256=digest(_pair_ids(data)[:256]), domain='normalized model input',
        equal_realized_noise=False, trigger_arrays=cell['trigger_arrays'], rows=rows)
    write(Path(cell['run_dir']) / 'input_distortion.json', value, immutable=True)


def current_inputs(cell):
    import run_a1_three_tables as frozen
    current = frozen._current_inputs(cell)
    current['profile'] = PROFILE
    return current


def completed(plan, cell):
    from vast_seed42_preflight import sha256
    folder = Path(cell['run_dir']); marker = folder / 'complete.json'
    if not marker.is_file(): return None
    gate = read(folder / 'gpu_smoke.json'); owned = read(folder / 'execution_binding.json')
    expected = dict(plan_sha256=plan['plan_sha256'], method_key=cell['method_key'],
        inputs_sha256=sha256(Path(plan['outdir']) / 'inputs.json'), gpu=gate['binding']['gpu'])
    if owned != dict(binding=expected, execution=gate['execution']):
        raise ValueError('Completed run has a foreign execution binding')
    import run_a1_three_tables as frozen
    run=frozen._read_completed(cell); current=current_inputs(cell)
    frozen._audit_run(cell,run,current)
    record=evaluator(plan['sources']).load_evaluation(dict(cell,verified_inputs=current),cell['artifact_dir'])
    if (record is None or read(marker) != dict(binding=owned, status='complete',
            evaluation_sha256=record['record_sha256'], distortion_sha256=sha256(folder/'input_distortion.json'))):
        raise ValueError('Completed evaluation/distortion binding differs')
    return record


def run_cell(plan, cell):
    import torch
    import run_a1_three_tables as frozen
    from run_mmfi_tables import _cell_lock
    from vast_seed42_preflight import sha256
    verify_inputs(plan); expected=binding(plan,cell); context=execution()
    folder=Path(cell['run_dir']); owned=dict(binding=expected, execution=context)
    if context != read(folder/'gpu_smoke.json')['execution']:
        raise ValueError('CUDA execution differs from Prepare')
    with _cell_lock(folder):
        audit=folder/'execution_binding.json'; current=current_inputs(cell)
        if (folder/'checkpoint.pt').exists():
            if not audit.is_file() or read(audit) != owned:
                raise ValueError('Checkpoint lacks its original execution binding')
            validate_checkpoint(torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False),cell['cfg'])
        path=folder/'three_tables_inputs.json'
        if path.is_file():
            if read(path) != current: raise ValueError('Training inputs changed; refuse resume')
        elif any((folder/name).exists() for name in ('checkpoint.pt','eval_cache.json')):
            raise ValueError('Checkpoint/cache lacks training input audit')
        write(audit,owned,immutable=True); write(path,current,immutable=True)
        engine=training_engine(); model,_=engine.train(cell['cfg'],ckpt_dir=folder)
        del model,engine
        if current_inputs(cell) != current: raise ValueError('Training inputs/action changed')
        run=frozen._read_completed(cell); scientific=frozen._audit_run(cell,run,current)
        report=evaluator(plan['sources']).evaluate_cell(dict(cell,verified_inputs=current),cell['artifact_dir'],'cuda:0')
        distortion(plan,cell); verify_inputs(plan,full_hash=True)
        if binding(plan,cell) != expected: raise ValueError('GPU/inputs changed during run')
        write(folder/'scientific_audit.json',scientific)
        write(folder/'complete.json',dict(binding=owned,status='complete',
            evaluation_sha256=report['record_sha256'],distortion_sha256=sha256(folder/'input_distortion.json')),immutable=True)


def child(plan,cell,work):
    root=Path(plan['outdir']); logs=root/'logs'; logs.mkdir(parents=True,exist_ok=True)
    log=logs/f"{work}.{cell['prior']}.{time.time_ns()}.log"
    print(f"[{work}] {cell['prior']}: {log}",flush=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--data-home',plan['data_home'],
        '--outdir',plan['outdir'],'--mode','worker','--_prior',cell['prior'],'--_work',work,
        '--priors',*plan['priors']]
    for name,path in plan['actions'].items(): cmd += ['--'+name+'-action',path]
    if plan['approve_action_labels']: cmd += ['--approve-action-labels']
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run(cmd,cwd=HERE,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"{cell['prior']} {work} failed; no retry/later job: {log}")


def export(plan):
    verify_inputs(plan); rows=[]
    if 'bend' not in plan['priors']:
        rows.append(dict(prior='bend',status='historical_unverified_not_imported',
                         metrics=None, action_sha256=None))
    for cell in plan['cells']:
        record=completed(plan,cell)
        rows.append(dict(prior=cell['prior'],status='complete' if record else 'pending',
            metrics=record['metrics'] if record else None,
            action_sha256=cell['action']['sha256']))
    value=dict(profile=PROFILE,plan_sha256=plan['plan_sha256'],rows=rows,
        bend_reference=('fresh victim in this run' if 'bend' in plan['priors'] else
                        'not imported; run --priors bend cross nod in a NEW output to train all3'),
        limitations=['Single seed42; no uncertainty estimated.',
            'Motion priors are not pose targets. Same epsilon does not imply equal realized distortion.',
            'No clean-trained+trigger reference was supplied; same-victim no-trigger controls are in evaluations.'])
    write(Path(plan['outdir'])/'motion_prior_results.json',value)
    lines=['# Motion-prior sensitivity / HPE-Li MM-Fi P1S1 / A1 seed42','',
        '| Prior | Status | T-MPJPE mm | Target-PCK % | DCE mm | Clean MPJPE mm | Clean PA mm |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['metrics']; numbers=[f'{m[k]:.2f}' for k in ('t1_mpjpe_mm','target_pck_pct','mean_dce_mm','clean_mpjpe_mm','clean_pampjpe_mm')] if m else ['--']*5
        lines.append('| '+' | '.join([row['prior'],row['status'],*numbers])+' |')
    lines += ['',*value['limitations'],'']
    from eval_a1_target_pck import _atomic_bytes
    _atomic_bytes(Path(plan['outdir'])/'motion_prior_results.md','\n'.join(lines).encode('utf-8'))
    print('\n'.join(lines))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home',required=True); parser.add_argument('--outdir',required=True)
    parser.add_argument('--device',choices=['cuda:0'],default='cuda:0')
    parser.add_argument('--mode',required=True,choices=['audit','prepare','check','launch','status','export','worker'])
    parser.add_argument('--priors',nargs='+',choices=PRIORS,default=['cross','nod'])
    for name in PRIORS: parser.add_argument('--'+name+'-action')
    parser.add_argument('--approve-action-labels',action='store_true')
    parser.add_argument('--_prior',choices=PRIORS); parser.add_argument('--_work',choices=['smoke','run'])
    args=parser.parse_args(argv)
    plan=build_plan(args.data_home,args.outdir,args.priors,
        {p:getattr(args,p+'_action') for p in PRIORS if getattr(args,p+'_action')},args.approve_action_labels)
    root=Path(plan['outdir']); path=root/PLAN
    if args.mode=='status':
        if path.is_file() and read(path)!=plan: raise ValueError('Plan/source/runtime changed; original checkout required')
        for cell in plan['cells']:
            folder=Path(cell['run_dir'])
            print(f"{cell['prior']}: checkpoint={(folder/'checkpoint.pt').is_file()}, completion_marker={(folder/'complete.json').is_file()} (existence only)")
        logs = (p for p in (root/'logs').glob('*.log')
                if p.name.startswith(('run.', 'smoke.')))
        for log in sorted(logs,key=lambda p:p.stat().st_mtime)[-2:]:
            from collections import deque
            print(f'LOG={log}')
            with log.open(encoding='utf-8',errors='replace') as stream: print(''.join(deque(stream,maxlen=12)))
        return
    if args.mode not in ('audit','prepare') and (not path.is_file() or read(path)!=plan):
        raise ValueError('Missing/mismatched immutable plan; Audit/Prepare in original checkout first')
    from run_mmfi_tables import _cell_lock
    if args.mode=='worker':
        if not args._prior or not args._work: raise ValueError('Private worker requires prior/work')
        cell=next(c for c in plan['cells'] if c['prior']==args._prior)
        with _cell_lock(Path(plan['data_home'])/'runs/.mmfi_motion_prior_gpu0_worker'):
            require_idle()
            gate_check(plan) if args._work=='run' else verify_inputs(plan)
            (smoke if args._work=='smoke' else run_cell)(plan,cell)
        return
    with _cell_lock(root):
        if args.mode=='export': return export(plan)
        if args.mode in ('audit','prepare'):
            write(path,plan,immutable=True)
            if (root/'inputs.json').is_file(): verify_inputs(plan)
            else:
                from vast_seed42_preflight import inspect
                write(root/'inputs.json',inspect(plan['data_home']),immutable=True)
            if args.mode=='audit':
                print('AUDITED_ONLY: full input/action/trigger CPU audit; no CUDA or full training',flush=True); return
        require_idle()
        if args.mode=='prepare':
            for cell in plan['cells']:
                if not (Path(cell['run_dir'])/'gpu_smoke.json').is_file(): child(plan,cell,'smoke')
            gate_check(plan)
            print('PREPARED_ONLY: full P1S1/action audit + B32 CUDA smoke; no full training',flush=True); return
        gate_check(plan)
        if args.mode=='check':
            print('VERIFIED_PRELAUNCH_GATE; no training launched',flush=True); return
        for cell in plan['cells']:
            if completed(plan,cell) is None:
                require_idle(); child(plan,cell,'run')
                if completed(plan,cell) is None: raise ValueError('Worker exited without authenticated completion')
        export(plan)
        print(f"COMPLETE {len(plan['cells'])}/{len(plan['cells'])} A1 priors; max one own GPU worker",flush=True)


if __name__=='__main__':
    main()
