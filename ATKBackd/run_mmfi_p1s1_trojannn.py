"""Isolated, single-seed TrojanNN CSI/pose adaptation: HPE-Li MM-Fi P1S1 only.

Prepare creates a universal model-informed trigger and a real B32 CUDA smoke.
Launch trains ONE independent scratch victim, fixed final49, then evaluates
six doses and the clean surrogate control. No historical artifacts are edited.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import ntpath
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib.metadata import version

os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  MKL_THREADING_LAYER='SEQUENTIAL', PYTHONUTF8='1')
from run_mmfi_p3_s3_three_attacks_4090 import (
    sha, digest, read, write, sources, atomic_checkpoint, paired_data, loader, seed,
)

PROFILE = 'mmfi_p1s1_hpeli_trojannn_adapted_s42_v1'
GRID = (0., .2, .4, .6, .8, 1.)
COUNTS = {'train': 133056, 'eval': 33264}
METHOD = 'trojannn_adapted'


def configuration(home):
    from run_a1_three_tables import _configuration
    cfg = _configuration(Path(home), 'cuda:0', 0, 42, 'a1')
    cfg = {k: v for k, v in cfg.items() if k != 'confirmation_profile'
           and not k.startswith('method_')}
    cfg.update(trigger=METHOD, attacker_access='white_box_clean_surrogate_for_trigger_only',
               trojannn_profile=PROFILE, ckpt_every=1)
    return cfg


def build_plan(home, outdir, clean_checkpoint, device='cuda:0', allow_concurrent_a1=False):
    from attack.trojannn_mmfi import TrojanNNConfig
    from dataclasses import asdict
    home, outdir, reference = (Path(p).resolve() for p in (home, outdir, clean_checkpoint))
    if (outdir.parent != home/'runs' or 'trojannn' not in outdir.name.lower()
            or reference.is_relative_to(outdir)):
        raise ValueError('Choose a dedicated trojannn directory directly under data-home/runs')
    if device != 'cuda:0':
        raise ValueError('This profile fixes one worker on cuda:0')
    if not reference.is_file():
        raise FileNotFoundError(f'Matching P1S1 Clean checkpoint required: {reference}')
    plan = dict(schema=1, profile=PROFILE, home=str(home), outdir=str(outdir), device=device,
        clean_checkpoint=str(reference), clean_sha256=sha(reference), cfg=configuration(home),
        operator=json.loads(json.dumps(asdict(TrojanNNConfig()))), sources=sources(),
        runtime=dict(python=platform.python_version(), platform=platform.system(),
                     packages={n:version(n) for n in ('torch','numpy','scipy','PyYAML')}),
        contract=dict(protocol='protocol1', setting='s1', split_seed=0, seed=42,
            model='hpeli', jobs_per_gpu=1, counts=COUNTS, rho=.1, n_poison=13305,
            physical_batch=32, victim_epochs=50, final_epoch=49, scratch=True,
            warm_start=False, loss='ordinary all-joint MPJPE', drop_last=False,
            optimizer='SGD', lr=.001, momentum=.9, weight_decay=0., amp=False,
            doses=list(GRID), target_joints=[2,3], pck_scale=[5,12], threshold=.2,
            status='EXPLORATORY_T1_EXTENSION_SINGLE_SEED',
            limitations=['BackdoorBench model-informed TrojanNN adapted to float CSI and pose/dose labels.',
                'Clean surrogate access is stronger than model-agnostic data-only poisoning.',
                'Trigger hyperparameters fixed before TEST; no checkpoint selection on TEST.',
                'Native patch intensity and realized noise are not equal-budget with A1.',
                'Historical Clean lacks training-time input hashes; present inputs are audited, not retrospectively proven.',
                'One seed; no across-seed SD or full-table completion claim.']))
    plan['execution_policy'] = dict(
        mode='trojannn_plus_exact_p3s3_a1' if allow_concurrent_a1 else 'exclusive',
        max_trojannn_workers=1, max_total_training_workers=2 if allow_concurrent_a1 else 1,
        authorized_a1_runner=ntpath.join(ntpath.dirname(str(home)), 'code', 'ATKBackd',
                                       'run_mmfi_p3_splits.py') if allow_concurrent_a1 else None,
        authorized_a1_outdirs=[str(home/'runs'/name) for name in (
            'mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1',
            'mmfi_p3s3_hpeli_a1_s42_4090_v1')] if allow_concurrent_a1 else [],
        note='Execution sharing only; A1 is never started, stopped, resumed, or modified here.')
    plan['plan_sha256'] = digest(plan)
    return plan


def clean_model(plan):
    import torch
    from models.factory import build_model
    blob = torch.load(plan['clean_checkpoint'], map_location='cpu', weights_only=False)
    expected = dict(experiment_name='mmfi', mmfi_protocol='protocol1', mmfi_setting='s1',
        mmfi_split_seed=0, mmfi_random_ratio=.8, model='hpeli', seed=42, rho=0.,
        epochs=50, victim_epochs=50, batch_size=32, optimizer='sgd', lr=.001,
        momentum=.9, weight_decay=0., pretrained=False, data_parallel=False,
        num_person=1, n_ant=3, n_sub=114, n_pkt=10, victim_loss='mpjpe',
        lr_scheduler=False)
    if (blob.get('epoch') != 49 or not blob.get('model')
            or any(blob.get('cfg',{}).get(k) != v for k,v in expected.items())
            or sha(plan['clean_checkpoint']) != plan['clean_sha256']):
        raise ValueError('Clean reference is not matching HPE-Li P1S1 seed42 final49 rho0 SGD')
    from train_backdoor import _config_fingerprint
    if blob.get('cfg_fingerprint') != _config_fingerprint(blob['cfg']):
        raise ValueError('Clean reference configuration fingerprint is not intact')
    model = build_model('hpeli', num_keypoints=17, subcarrier_num=114,
                        dataset='mmfi', pretrained=False)
    model.load_state_dict(blob['model'], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite Clean reference state')
    return model


def dataset(plan, split):
    from data_utils.feeder import MMFI
    return MMFI(split, plan['cfg']['dataset_root'], protocol='protocol1',
                setting='s1', random_ratio=.8, random_seed=0)


def audit(plan):
    from vast_seed42_preflight import inspect
    root = Path(plan['outdir'])
    if (root/'plan.json').exists():
        return verify_inputs(plan)
    if any(p.name != '.controller' for p in root.iterdir()):
        raise ValueError('Nonempty unbound output; preserve it and choose a NEW output')
    clean_model(plan)  # No forward/TEST use to construct the trigger.
    inputs = inspect(plan['home'])  # Full content, counts and split identity audit.
    from run_mmfi_p3_s3_three_attacks_4090 import Deferred
    pairs = paired_data(dataset(plan,'training'), Deferred(), plan['cfg'])
    poison = pairs.manifest()
    if poison['n_poison'] != 13305:
        raise ValueError('Expected floor(.1*133056)=13305 poisoned TRAIN frames')
    write(root/'plan.json', plan, immutable=True)
    write(root/'inputs.json', inputs, immutable=True)
    write(root/'poison.json', poison, immutable=True)
    print('FULL_P1S1_INPUT_AUDIT_PASS; no full training', flush=True)
    return inputs


def verify_inputs(plan):
    from vast_seed42_preflight import check_receipt
    root = Path(plan['outdir'])
    if read(root/'plan.json') != plan:
        raise ValueError('Source/config/runtime/reference changed; preserve run and use NEW output')
    inputs = read(root/'inputs.json')
    check_receipt(inputs)
    return inputs


def execution():
    from run_piw_hpeli_clean import _setup_execution
    return _setup_execution('cuda:0')


def parse_windows_commandline(text):
    """Strict subset emitted by our Windows launcher, not a shell interpreter."""
    token = r'(?:"[^"\r\n]*"|[^\s"\r\n]+)'
    if (not isinstance(text, str) or any(ord(c) < 32 for c in text)
            or not re.fullmatch(r'\s*'+token+r'(?:\s+'+token+r')*\s*', text)):
        raise ValueError('Cannot parse exact Windows command line')
    result = [m.group()[1:-1] if m.group().startswith('"') else m.group()
              for m in re.finditer(token, text)]
    if any(not value for value in result):
        raise ValueError('Cannot parse empty Windows argument')
    return result


def _windows_path(value):
    return ntpath.normcase(ntpath.normpath(str(value).replace('/', '\\')))


def _a1_environments(plan):
    return {_windows_path(ntpath.join(plan['home'], suffix)) for suffix in (
        r'envs\dose-backdoor\python.exe', r'envs\mmfi-p3\Scripts\python.exe')}


def a1_peer(record, plan):
    """Match only the historical A1-only launch, not arbitrary Python/SIG work.

    This matches a process record; authorized_a1_peers also validates its
    executable/venv ancestry. The guard prevents accidental contention and is
    not an adversarial process-authentication boundary.
    """
    if plan.get('execution_policy', {}).get('mode') != 'trojannn_plus_exact_p3s3_a1':
        return False
    try:
        if (type(record.get('ProcessId')) is not int or record['ProcessId'] <= 0
                or type(record.get('ParentProcessId')) is not int
                or record['ParentProcessId'] < 0
                or str(record.get('Name', '')).lower() != 'python.exe'):
            return False
        argv = parse_windows_commandline(record.get('CommandLine'))
        executable = record.get('ExecutablePath')
        if (len(argv) != 17 or argv[1] != '-u' or not executable
                or not ntpath.isabs(argv[0]) or not ntpath.isabs(executable)
                or ntpath.basename(argv[0]).lower() != 'python.exe'
                or ntpath.basename(executable).lower() != 'python.exe'
                or _windows_path(argv[0]) != _windows_path(executable)):
            return False
        expected_runner = ntpath.join(ntpath.dirname(plan['home']), 'code', 'ATKBackd',
                                      'run_mmfi_p3_splits.py')
        if _windows_path(argv[2]) != _windows_path(expected_runner):
            return False
        names = argv[3::2]
        if len(set(names)) != len(names):
            return False
        options = dict(zip(names, argv[4::2]))
        expected = {'--data-home', '--outdir', '--device', '--workers',
                    '--jobs-per-gpu', '--mode', '--_cell'}
        if set(options) != expected:
            return False
        allowed_outputs = {ntpath.join(plan['home'], 'runs', n) for n in (
            'mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1',
            'mmfi_p3s3_hpeli_a1_s42_4090_v1')}
        return (_windows_path(options['--data-home']) == _windows_path(plan['home'])
            and _windows_path(options['--outdir']) in {_windows_path(p) for p in allowed_outputs}
            and options['--device'] == 'cuda:0' and options['--workers'] == '0'
            and options['--jobs-per-gpu'] == '1' and options['--mode'] == 'launch'
            and options['--_cell'] == 'p3_s3_a1')
    except (ValueError, TypeError, KeyError):
        return False


def authorized_a1_peers(records, plan):
    """At most one A1 worker, including its ordinary Windows venv proxy chain."""
    candidates = {r['ProcessId']: r for r in (records or []) if a1_peer(r, plan)}
    if not candidates:
        return set()
    roots = [r for r in candidates.values() if r['ParentProcessId'] not in candidates]
    if len(roots) != 1:
        return set()
    root = roots[0]
    root_argv = parse_windows_commandline(root['CommandLine'])
    envs = _a1_environments(plan)
    if (_windows_path(root_argv[0]) not in envs
            or _windows_path(root['ExecutablePath']) != _windows_path(root_argv[0])):
        return set()
    # The same approved output must be used by every proxy/worker in the chain.
    root_options = dict(zip(root_argv[3::2], root_argv[4::2]))
    children = {}
    for record in candidates.values():
        children.setdefault(record['ParentProcessId'], []).append(record['ProcessId'])
        argv = parse_windows_commandline(record['CommandLine'])
        options = dict(zip(argv[3::2], argv[4::2]))
        if any((_windows_path(options[k]) != _windows_path(root_options[k]))
               if k in ('--data-home', '--outdir') else options[k] != root_options[k]
               for k in root_options):
            return set()
        if (_windows_path(argv[0]) not in envs
                and _windows_path(argv[0]) != _windows_path(record['ExecutablePath'])):
            return set()
    if any(len(ids) > 1 for parent, ids in children.items() if parent in candidates):
        return set()
    seen = set()
    current = root['ProcessId']
    while current not in seen:
        seen.add(current)
        descendants = children.get(current, [])
        if not descendants:
            break
        current = descendants[0]
    return seen if seen == set(candidates) else set()


def pending_experiment_conflicts(records, plan, own_pid):
    """Also see known workers still loading on CPU, before their CUDA allocation.

    This is a snapshot/coordination guard, not an OS resource reservation: a
    separately launched external job can always appear after this check.
    """
    records = {r.get('ProcessId'): r for r in (records or [])}
    exempt = authorized_a1_peers(list(records.values()), plan) | {own_pid}
    # A normal Windows venv may keep one launcher ancestor. Exempt only that
    # same exact invocation, not another TrojanNN output/worker or its children.
    current = records.get(own_pid)
    if current:
        try:
            own_args = parse_windows_commandline(current.get('CommandLine'))
            own_script = _windows_path(__file__)
            while (len(own_args) > 2 and own_args[1] == '-u'
                   and _windows_path(own_args[2]) == own_script):
                parent = records.get(current.get('ParentProcessId'))
                if not parent or parent['ProcessId'] in exempt:
                    break
                args = parse_windows_commandline(parent.get('CommandLine'))
                if (args[1:] != own_args[1:]
                        or str(parent.get('Name', '')).lower() != 'python.exe'):
                    break
                exempt.add(parent['ProcessId'])
                current = parent
        except (ValueError, TypeError, KeyError):
            pass
    conflicts = []
    for pid, record in records.items():
        if pid in exempt or str(record.get('Name', '')).lower() not in ('python.exe', 'pythonw.exe'):
            continue
        line = str(record.get('CommandLine') or '')
        try:
            args = parse_windows_commandline(line)
        except ValueError:
            args = []
        known = False
        for value in args[1:]:
            name = ntpath.basename(value).lower()
            if (name.endswith('.py') and name.startswith(('run_mmfi', 'run_piw', 'run_a1', 'run_hpeli'))
                    or name in ('train.py', 'train_backdoor.py', 'eval_offsets.py')
                    or (name == 'controller.py' and 'piw_metafi_remaining_3090' in value.lower())):
                known = True
                break
        if known:
            conflicts.append((pid, line))
    return conflicts


def classify_gpu(report, windows_names=None, own_pid=None, *, plan=None, windows_processes=None):
    """Fail closed for unknown/compute names; known Windows desktop peers only.

    Process-name checks prevent accidental contention, not adversarial spoofing.
    """
    desktop = {'explorer.exe','dwm.exe','chrome.exe','msedge.exe','msedgewebview2.exe',
        'code.exe','lockapp.exe','searchhost.exe','startmenuexperiencehost.exe',
        'shellexperiencehost.exe','shellhost.exe','textinputhost.exe','systemsettings.exe',
        'applicationframehost.exe','windowsterminal.exe','taskmgr.exe','snippingtool.exe',
        'crossdeviceresume.exe','copilot.exe','copilotapp.exe','copilotapphost.exe',
        'phoneexperiencehost.exe','teams.exe','ms-teams.exe','teamviewer.exe','logonui.exe',
        'anydesk.exe','system'}
    permitted = authorized_a1_peers(windows_processes, plan) if plan else set()
    conflicts = []
    for row in csv.reader(report.splitlines()):
        if not row: continue
        if len(row)!=2 or not row[0].strip().isdigit():
            raise ValueError(f'Cannot parse GPU process: {row}')
        pid, name = int(row[0]), row[1].strip()
        if pid == own_pid: continue
        if windows_names is not None and pid in permitted: continue
        if windows_names is not None:
            name = windows_names.get(pid, name)
            if ntpath.basename(name.replace('/','\\')).lower() in desktop: continue
        conflicts.append((pid,name))
    return conflicts


def require_idle(plan=None):
    report = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=30)
    names = None
    records = None
    if os.name == 'nt':
        script = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine) | ConvertTo-Json -Compress"
        result = subprocess.run(['powershell','-NoProfile','-Command',script],
            capture_output=True,encoding='utf-8',check=True,timeout=30)
        records = json.loads(result.stdout.lstrip('\ufeff'))
        if not isinstance(records,list): records=[records]
        names = {int(r['ProcessId']):r['Name'] for r in records}
    conflicts = classify_gpu(report.stdout,names,os.getpid(),plan=plan,windows_processes=records)
    if plan is not None:
        conflicts.extend(pending_experiment_conflicts(records,plan,os.getpid()))
    if conflicts:
        raise ValueError(f'Other/unknown GPU work active; no process stopped: {conflicts}')
    permitted = authorized_a1_peers(records, plan) if plan else set()
    if permitted:
        print(f'AUTHORIZED_A1_PEER PIDs={sorted(permitted)}; left running. '
              'At most one TrojanNN + one A1; no SIG/other workers.',flush=True)


def trigger_binding(plan, env):
    root = Path(plan['outdir'])
    return dict(plan_sha256=plan['plan_sha256'], execution=env,
        inputs_sha256=sha(root/'inputs.json'), poison_sha256=sha(root/'poison.json'),
        clean_sha256=plan['clean_sha256'])


def load_trigger(plan, env):
    import torch
    from attack.trojannn_mmfi import TrojanNNTrigger
    blob = torch.load(Path(plan['outdir'])/'trigger.pt', map_location='cpu', weights_only=False)
    if blob['binding'] != trigger_binding(plan,env):
        raise ValueError('Frozen trigger source/input/surrogate/execution binding changed')
    trigger = TrojanNNTrigger.from_state_dict(blob['trigger'])
    from dataclasses import asdict
    if (trigger.provenance() != blob['provenance']
            or json.loads(json.dumps(asdict(trigger.config))) != plan['operator']):
        raise ValueError('Frozen trigger provenance changed')
    return trigger


def components(plan, trigger):
    from models.factory import build_model
    from train_backdoor import _build_optimizer
    seed()  # Clean surrogate and trigger optimization never seed the poisoned victim state.
    model = build_model('hpeli',num_keypoints=17,subcarrier_num=114,
                        dataset='mmfi',pretrained=False).to('cuda:0').train()
    _, opt = _build_optimizer(model,plan['cfg'],'mmfi')
    pairs = paired_data(dataset(plan,'training'),trigger,plan['cfg'])
    if pairs.manifest() != read(Path(plan['outdir'])/'poison.json'):
        raise ValueError('Poison identities/doses/targets changed')
    return model,opt,pairs


def prepare(plan):
    import torch
    from attack.trojannn_mmfi import optimize_trigger
    from attack.poison import collate
    from train_backdoor import _victim_update
    audit(plan); require_idle(plan); env=execution(); root=Path(plan['outdir'])
    if (root/'gpu_gate.json').exists():
        gate_check(plan)
        print('ALREADY_PREPARED; frozen trigger/gate retained; no training',flush=True)
        return
    if not (root/'trigger.pt').exists():
        model=clean_model(plan).to('cuda:0').eval()
        trigger=optimize_trigger(model,device='cuda:0',
            progress=lambda r: print('[trigger]',json.dumps(r),flush=True))
        atomic_checkpoint(root/'trigger.pt',dict(binding=trigger_binding(plan,env),
            trigger=trigger.state_dict(),provenance=trigger.provenance()))
        del model
    trigger=load_trigger(plan,env)
    require_idle(plan)
    model,opt,pairs=components(plan,trigger)
    selected=sorted(pairs.poison_idx)[:4]
    selected += [i for i in range(len(pairs)) if i not in pairs.poison_idx][:28]
    batch=collate([pairs[i] for i in selected])
    if tuple(batch['csi'].shape)!=(32,3,114,10):
        raise ValueError('Physical B32 real TRAIN smoke required')
    before=next(model.parameters()).detach().clone()
    losses=[_victim_update(model,opt,batch['csi'].to('cuda:0'),batch['pose'].to('cuda:0'))
            for _ in range(2)]
    if (not all(__import__('math').isfinite(v) for v in losses)
            or torch.equal(before,next(model.parameters()))):
        raise ValueError('Real B32 smoke failed finite/update checks')
    write(root/'gpu_gate.json',dict(binding=trigger_binding(plan,env),
        trigger_sha256=sha(root/'trigger.pt'), batch_size=32, steps=2,
        real_train_input=True, poison_examples=4, losses=losses),immutable=True)
    print('PREPARED_ONLY: P1S1 audit + frozen trigger + B32 CUDA smoke; no full training',flush=True)


def gate_check(plan):
    verify_inputs(plan); env=execution(); root=Path(plan['outdir']); gate=read(root/'gpu_gate.json')
    if (gate['binding']!=trigger_binding(plan,env) or gate['batch_size']!=32
            or gate['steps']!=2 or gate['real_train_input'] is not True
            or gate['trigger_sha256']!=sha(root/'trigger.pt')):
        raise ValueError('Missing/foreign prelaunch CUDA gate')
    load_trigger(plan,env)
    return env


def checkpoint_binding(plan,env):
    return dict(**trigger_binding(plan,env), trigger_sha256=sha(Path(plan['outdir'])/'trigger.pt'))


def validate_checkpoint(blob,plan,env):
    if (blob.get('binding')!=checkpoint_binding(plan,env) or blob.get('cfg')!=plan['cfg']
            or type(blob.get('epoch')) is not int or not -1<=blob['epoch']<=49
            or any(not blob.get(k) for k in ('model','optimizer','rng'))
            or set(blob['rng'])!={'python','numpy','torch','cuda'} or blob['rng']['cuda'] is None):
        raise ValueError('Incomplete/foreign resume checkpoint; no epoch0 fallback')
    groups=blob['optimizer'].get('param_groups',[])
    if not groups or any(g.get('lr')!=.001 or g.get('momentum')!=.9
            or g.get('weight_decay')!=0. or 'betas' in g for g in groups):
        raise ValueError('Resume SGD recipe differs')


def train(plan,env):
    import torch
    from train_backdoor import _victim_update,_rng_state,_restore_rng_state
    root=Path(plan['outdir']); path=root/'checkpoint.pt'
    trigger=load_trigger(plan,env); model,opt,pairs=components(plan,trigger)
    start=0
    def save(epoch):
        atomic_checkpoint(path,dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),
            rng=_rng_state(),cfg=plan['cfg'],binding=checkpoint_binding(plan,env)))
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=False); validate_checkpoint(old,plan,env)
        model.load_state_dict(old['model'],strict=True); opt.load_state_dict(old['optimizer'])
        _restore_rng_state(old['rng']); start=old['epoch']+1
        print(f'RESUME next_epoch={start}; model/SGD/RNG restored',flush=True)
    else: save(-1)
    batches=loader(pairs,train=True)
    for epoch in range(start,50):
        t=time.monotonic(); total=0.; n=0
        for batch in batches:
            loss=_victim_update(model,opt,batch['csi'].to('cuda:0'),batch['pose'].to('cuda:0'))
            if not __import__('math').isfinite(loss): raise ValueError('Nonfinite training loss; no retry')
            total+=loss*len(batch['csi']); n+=len(batch['csi'])
        if n!=COUNTS['train']: raise ValueError('Not every official TRAIN frame was consumed')
        save(epoch)
        print(f'epoch {epoch}/49: loss={total/n:.8f} lr=1.00e-03 t={time.monotonic()-t:.1f}s',flush=True)
    print('TRAINED_NOT_EVALUATED final49',flush=True)


def evaluate(plan,env):
    import numpy as np
    import torch
    from models.factory import build_model
    from extra_attack_eval import _loader,measure_pair
    from eval import metrics as M
    from vast_seed42_preflight import check_receipt
    root=Path(plan['outdir']); path=root/'checkpoint.pt'
    old=torch.load(path,map_location='cpu',weights_only=False); validate_checkpoint(old,plan,env)
    if old['epoch']!=49: raise ValueError('Only fixed final49 may be evaluated')
    if (root/'evaluation.json').exists():
        return completed(plan)
    check_receipt(read(root/'inputs.json'),full_hash=True)
    trigger=load_trigger(plan,env); data=dataset(plan,'test')
    if len(data)!=COUNTS['eval']: raise ValueError('Full official TEST required')
    output={}; arrays={}; truth=None
    for key in ('clean_reference',METHOD):
        model=clean_model(plan) if key=='clean_reference' else build_model(
            'hpeli',num_keypoints=17,subcarrier_num=114,dataset='mmfi',pretrained=False)
        if key==METHOD: model.load_state_dict(old['model'],strict=True)
        model=model.to('cuda:0').eval()
        @torch.no_grad()
        def predict(mode,dose=0.):
            preds=[]; ys=[]; ts=[]
            for batch in _loader(data,trigger,plan['cfg'],mode,dose):
                pred=model(batch['csi'].to('cuda:0'))[0].cpu().numpy()[:,0]
                preds.append(pred); ys.append(batch['pose'].numpy()[:,0]); ts.append(batch['target'].numpy()[:,0])
            return np.concatenate(preds),np.concatenate(ys),np.concatenate(ts)
        clean,true,_=predict('clean'); rows=[]; predictions=[]; errors=[]; no_trigger=[]; dces=[]
        if truth is not None and not np.array_equal(truth,true): raise ValueError('Control TEST order differs')
        truth=true
        for dose in GRID:
            print(f'[evaluate] {key} dose={dose:g}',flush=True)
            pred,again,target=predict('trigger@dose',dose)
            if not np.array_equal(true,again) or (dose==0 and not np.array_equal(pred,clean)):
                raise ValueError('Evaluation order/dose0 identity differs')
            metrics,per=measure_pair(clean,pred,true,target)
            rows.append(dict(dose=dose,**metrics)); predictions.append(pred)
            errors.append(per['target_error_mm']); no_trigger.append(per['no_trigger_target_error_mm']); dces.append(per['dce_mm'])
        output[key]=dict(metrics=dict(t_mpjpe_mm=rows[-1]['t_mpjpe_mm'],
            target_pck_pct=rows[-1]['target_pck_pct'],
            mean_positive_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]])),
            clean_mpjpe_mm=float(M.mpjpe(clean,true).mean()*1000),
            clean_pampjpe_mm=float(M.pa_mpjpe(clean,true).mean()*1000)),dose_rows=rows)
        for name,value in dict(clean_prediction=clean,trigger_prediction=np.stack(predictions),
            target_error_mm=np.stack(errors),no_trigger_target_error_mm=np.stack(no_trigger),
            dce_mm=np.stack(dces)).items(): arrays[key+'__'+name]=value
        del model; torch.cuda.empty_cache()
    buffer=io.BytesIO(); np.savez_compressed(buffer,**arrays,true_pose=truth,
        pair_ids=np.asarray([i['name'] for i in data.items]),doses=np.asarray(GRID))
    sample=root/'per_sample.npz'
    if sample.exists():
        with np.load(sample,allow_pickle=False) as a,np.load(io.BytesIO(buffer.getvalue()),allow_pickle=False) as b:
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in b.files):
                raise ValueError('Existing partial evaluation differs; preserve artifacts')
    else:
        temporary=root/'per_sample.npz.saving'
        with temporary.open('xb') as stream: stream.write(buffer.getvalue())
        os.replace(temporary,sample)
    check_receipt(read(root/'inputs.json'),full_hash=True)
    row=dict(status='complete',profile=PROFILE,plan_sha256=plan['plan_sha256'],
        binding=checkpoint_binding(plan,env),epoch=49,frames=len(data),records=output,
        checkpoint_sha256=sha(path),per_sample_sha256=sha(sample),
        metrics_contract=dict(target=[2,3],pck_clean_gt_scale=[5,12],threshold=.2,
            endpoint=1.,DCE='mean five positive doses',units='mm / percent'))
    row['record_sha256']=digest(row)
    write(root/'evaluation.json',row,immutable=True)
    print('EVALUATION_COMPLETE',json.dumps(output[METHOD]['metrics']),flush=True)
    return row


def completed(plan):
    import torch
    root=Path(plan['outdir']); path=root/'evaluation.json'
    if not path.exists(): return None
    row=read(path)
    unsigned=dict(row); claimed=unsigned.pop('record_sha256',None)
    if (claimed!=digest(unsigned) or row.get('status')!='complete'
            or row.get('profile')!=PROFILE or row.get('plan_sha256')!=plan['plan_sha256']
            or row.get('epoch')!=49 or row.get('frames')!=COUNTS['eval']
            or read(root/'plan.json')!=plan
            or row.get('checkpoint_sha256')!=sha(root/'checkpoint.pt')
            or row.get('per_sample_sha256')!=sha(root/'per_sample.npz')):
        raise ValueError('Completed evaluation artifacts differ')
    gate=read(root/'gpu_gate.json'); env=gate['binding']['execution']
    if (gate['binding']!=trigger_binding(plan,env)
            or gate['trigger_sha256']!=sha(root/'trigger.pt')
            or row.get('binding')!=checkpoint_binding(plan,env)):
        raise ValueError('Completed evaluation input/trigger/gate binding differs')
    load_trigger(plan,env)
    saved=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(saved,plan,env)
    if saved['epoch']!=49: raise ValueError('Completed result requires real final49 checkpoint')
    return row


def export(plan):
    row=completed(plan)
    if row is None: raise ValueError('Final evaluation not complete')
    keys=('t_mpjpe_mm','target_pck_pct','mean_positive_dce_mm','clean_mpjpe_mm','clean_pampjpe_mm')
    lines=['# MM-Fi / HPE-Li / P1S1 / seed42 — exploratory TrojanNN adaptation','',
        '| Method | T-MPJPE mm | Target-PCK % | DCE mm | Clean MPJPE mm | Clean PA mm |',
        '|---|---:|---:|---:|---:|---:|']
    clean=row['records']['clean_reference']['metrics']
    lines.append(f"| Clean reference | — | — | — | {clean['clean_mpjpe_mm']:.2f} | {clean['clean_pampjpe_mm']:.2f} |")
    lines.append('| TrojanNN adapted | '+' | '.join(f'{row["records"][METHOD]["metrics"][k]:.2f}' for k in keys)+' |')
    lines+=['','Clean-reference + trigger and same-victim no-trigger target controls are in evaluation.json.',
            'Not an equal-noise comparison or the completed multi-seed paper table.']
    text='\n'.join(lines)+'\n'; path=Path(plan['outdir'])/'comparison.md'
    if path.exists() and path.read_text(encoding='utf-8')!=text: raise ValueError('Existing export differs')
    if not path.exists(): path.write_text(text,encoding='utf-8')
    print(text)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-home',type=Path,required=True)
    parser.add_argument('--outdir',type=Path)
    parser.add_argument('--clean-checkpoint',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--mode',choices=('prepare','check','launch','status','export'),required=True)
    parser.add_argument('--allow-concurrent-a1',action='store_true',
        help='Permit only the existing historical MM-Fi P3S3 A1-only peer; no SIG/other work.')
    args=parser.parse_args(argv)
    run_name = ('mmfi_p1s1_hpeli_trojannn_s42_with_a1_v1' if args.allow_concurrent_a1
                else 'mmfi_p1s1_hpeli_trojannn_s42_v1')
    out=args.outdir or args.data_home/'runs'/run_name
    plan=build_plan(args.data_home,out,args.clean_checkpoint,args.device,args.allow_concurrent_a1)
    root=Path(plan['outdir'])
    if args.mode=='status':
        import torch
        checkpoint=root/'checkpoint.pt'; epoch=None
        if checkpoint.is_file():
            epoch=torch.load(checkpoint,map_location='cpu',weights_only=False).get('epoch')
        print(json.dumps(dict(profile=PROFILE,checkpoint=checkpoint.is_file(),epoch=epoch,
            gate=(root/'gpu_gate.json').is_file(),result=completed(plan)),indent=2)); return
    if args.mode=='export': export(plan); return
    from run_mmfi_tables import _cell_lock
    with _cell_lock(Path(plan['home'])/'tools/trojannn_gpu_locks/cuda0'), _cell_lock(root/'.controller'):
        require_idle(plan)
        if shutil.disk_usage(plan['home']).free<5*2**30: raise ValueError('Need at least 5GiB free')
        if args.mode=='prepare': prepare(plan)
        elif args.mode=='check': gate_check(plan); print('VERIFIED_PRELAUNCH_GATE; no training launched')
        else:
            env=gate_check(plan)
            require_idle(plan); train(plan,env)
            require_idle(plan); evaluate(plan,env); export(plan)


if __name__=='__main__':
    try: main()
    except Exception as error:
        print(f'[MM-Fi P1S1 TrojanNN] ERROR: {error}',file=sys.stderr,flush=True)
        raise
