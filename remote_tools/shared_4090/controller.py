"""Explicit A1 + TrojanNN sharing with the one user-approved nscaling job.

Foreground SSH controller. No tasks, service, process termination, environment
installation, source/gate mutation, trigger optimization, or epoch-zero fallback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trojan_context as context

TROJAN_SHA = '0c335047a4869d0a354cf184ffb41ede04fd9e324452913372d31bfa9f565318'
A1_SHA = 'baffa80ec27a1a859dd424b5bc8620fcec5dc78488106a4abb8ab7d0afdc8c73'
PROFILE = 'van4090_a1_trojannn_with_explicit_nscaling_context_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_digest(value):
    unsigned = {k: v for k, v in value.items() if k != 'request_sha256'}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def a1_command(request):
    return [request['python'], '-u', str(Path(request['a1_repo'])/'ATKBackd/run_mmfi_p3_splits.py'),
        '--data-home', request['home'], '--outdir', request['a1_out'],
        '--device', 'cuda:0', '--workers', '0', '--jobs-per-gpu', '1',
        '--mode', 'launch', '--_cell', 'p3_s3_a1']


def make_request(home, trojan_repo, a1_repo, python_exe, records):
    home, trojan_repo, a1_repo, python_exe = [Path(p).resolve()
        for p in (home, trojan_repo, a1_repo, python_exe)]
    for path in (home, trojan_repo, a1_repo, python_exe):
        if str(path).startswith('\\\\') or any(c in str(path) for c in ('"', '\n', '\r', '\x00')):
            raise ValueError('Only ordinary local paths are supported')
    if a1_repo != (home.parent/'code').resolve():
        raise ValueError('A1Repo must remain the original sibling code checkout; no path substitution')
    trojan_runner = trojan_repo/'ATKBackd/run_mmfi_p1s1_trojannn.py'
    a1_runner = a1_repo/'ATKBackd/run_mmfi_p3_splits.py'
    if sha(trojan_runner) != TROJAN_SHA or sha(a1_runner) != A1_SHA:
        raise ValueError('Prepared scientific runner differs; no source/gate migration attempted')
    if not python_exe.is_file():
        raise ValueError('Use the existing dose-backdoor Python; no installation fallback')
    trojan_out = home/'runs/mmfi_p1s1_hpeli_trojannn_s42_with_a1_v1'
    a1_out = home/'runs/mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1'
    # Existing approved invocation only. A1 cannot silently start a fresh victim.
    if not (a1_out/'p3_s3/a1/checkpoint.pt').is_file():
        raise ValueError('Existing A1 checkpoint required; no epoch-zero fallback')
    approved = [r for r in records if r.get('ProcessId') in (26344, 44248)]
    if approved and len(approved) != 2:
        raise ValueError('Approved nscaling lineage is incomplete; inspect before launch')
    context.validate_nscaling(records, approved)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex
    execution_dir = home/'tools/shared4090/executions'/stamp
    request = dict(schema=1, profile=PROFILE, created_utc=datetime.now(timezone.utc).isoformat(),
        home=str(home), python=str(python_exe), trojan_repo=str(trojan_repo), a1_repo=str(a1_repo),
        trojan_out=str(trojan_out), a1_out=str(a1_out),
        clean_checkpoint=str(home/'runs/mmfi_three_tables_4090_win_fresh_v1/seed42/clean/checkpoint.pt'),
        approved_nscaling=approved, trojan_runner_sha256=sha(trojan_runner),
        a1_runner_sha256=sha(a1_runner),
        trojan_plan_sha256=sha(trojan_out/'plan.json'),
        trojan_trigger_sha256=sha(trojan_out/'trigger.pt'),
        trojan_gate_sha256=sha(trojan_out/'gpu_gate.json'),
        source_sha256={str(p): sha(p) for p in (Path(__file__).resolve(),
                          Path(context.__file__).resolve())},
        execution_dir=str(execution_dir), smoke_receipt=str(execution_dir/'trojan_shared_smoke.json'),
        policy=dict(max_model_training_workers=2, approved_external_jobs=1 if approved else 0,
            max_total_compute_jobs=3 if approved else 2,
            old_gate_certifies_shared_context=False, no_kill=True, no_retry=True,
            note='Explicit external nscaling exception, separate from the unchanged original plan/gate.'))
    request['request_sha256'] = request_digest(request)
    return request


A1_AUDIT = r'''
import hashlib,io,json,pathlib,sys,torch
repo,home,out=sys.argv[1:]
sys.path.insert(0,str(pathlib.Path(repo)/'ATKBackd'))
import run_mmfi_p3_splits as run
plan=run.build_plan(home,out,'cuda:0',0)
saved=json.loads(run._plan_file(out).read_text(encoding='utf-8'))
if saved!=plan: raise ValueError('A1 original source/plan/settings changed')
run._receipt(plan,'preflight.json'); run._receipt(plan,'gpu_smoke.json')
cell=next(c for c in plan['cells'] if c['id']=='p3_s3_a1')
cfg=cell['cfg']
if cfg.get('strict_resume') is not True: raise ValueError('A1 strict resume required')
path=pathlib.Path(cell['folder'])/'checkpoint.pt'
data=path.read_bytes(); blob=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
epoch=blob.get('epoch')
if type(epoch) is not int or not 0<=epoch<=49: raise ValueError('A1 saved epoch invalid')
if blob.get('cfg_fingerprint')!=run._config_fingerprint(cfg): raise ValueError('A1 checkpoint config mismatch')
if not blob.get('model') or not blob.get('optimizer') or not blob.get('rng'): raise ValueError('A1 resume state incomplete')
if set(blob['rng'])!={'python','numpy','torch','cuda'} or blob['rng']['cuda'] is None: raise ValueError('A1 exact RNG resume state required')
groups=blob['optimizer'].get('param_groups',[])
if not groups or any(g.get('lr')!=.001 or g.get('momentum')!=.9 or g.get('weight_decay')!=0. or 'betas' in g for g in groups): raise ValueError('A1 saved optimizer must retain SGD .001/.9/wd0')
print(json.dumps(dict(epoch=epoch,next_epoch=epoch+1,checkpoint_sha256=hashlib.sha256(data).hexdigest(),
    plan_sha256=plan['sha256'],complete=run._complete(cell))))
'''


def audit_a1(request):
    result = subprocess.run([request['python'], '-c', A1_AUDIT,
        request['a1_repo'], request['home'], request['a1_out']],
        capture_output=True, encoding='utf-8', check=True, timeout=180)
    return json.loads(result.stdout.strip().splitlines()[-1])


def child_phase(request_file, phase):
    return [sys.executable, '-u', str(Path(context.__file__).resolve()),
            '--request', str(request_file), '--phase', phase]


def a1_live(request, records):
    sys.path.insert(0, str(Path(request['trojan_repo'])/'ATKBackd'))
    import run_mmfi_p1s1_trojannn as run
    policy = {'home': request['home'], 'execution_policy': {'mode': 'trojannn_plus_exact_p3s3_a1'}}
    permitted = run.authorized_a1_peers(records, policy)
    for record in records:
        if record['ProcessId'] in permitted:
            argv = run.parse_windows_commandline(record['CommandLine'])
            opts = dict(zip(argv[3::2], argv[4::2]))
            if run._windows_path(opts['--outdir']) != run._windows_path(request['a1_out']):
                raise ValueError('A different A1 output is active; no duplicate A1 launch')
    return permitted


def drain_owned(owned):
    """Wait for owned peers without terminating them or launching replacements."""
    exits = {}
    while len(exits) < len(owned):
        for name, process in owned.items():
            code = process.poll()
            if code is not None and name not in exits:
                exits[name] = code
                print(f'EXIT {name}={code}; no process killed or retried',flush=True)
        if len(exits) < len(owned):
            time.sleep(10)
    return exits


def launch(request):
    root = Path(request['execution_dir'])
    root.mkdir(parents=True, exist_ok=False)
    request_file = root/'request.json'
    write_new_json(request_file, request)
    print('EXPLICIT_SHARED_CONTEXT: at most 2 model workers + '
          f'{1 if request["approved_nscaling"] else 0} approved external nscaling job. '
          'Snapshot guard, not an OS reservation; no process will be killed.',flush=True)
    print('EXECUTION_REQUEST='+str(request_file),flush=True)
    # Native Trojan check with narrow, explicit external-job exception. No train.
    subprocess.run(child_phase(request_file, 'check'), check=True)
    a1_state = audit_a1(request)
    write_new_json(root/'a1_audit.json', a1_state)
    print('A1_AUDIT', json.dumps(a1_state), flush=True)
    current = context.windows_processes()
    context.validate_nscaling(current, request['approved_nscaling'])
    existing = a1_live(request, current)
    owned = {}
    streams = []
    try:
        if existing:
            print(f'A1_ALREADY_RUNNING PIDs={sorted(existing)}; left unchanged', flush=True)
        elif not a1_state['complete']:
            a1_log = Path(request['a1_out'])/'p3_s3/a1'/('console.shared.'+root.name+'.log')
            stream = a1_log.open('x', encoding='utf-8')
            streams.append(stream)
            owned['a1'] = subprocess.Popen(a1_command(request), cwd=request['a1_repo'],
                stdout=stream, stderr=subprocess.STDOUT)
            print(f'A1_RESUME_CREATED PID={owned["a1"].pid}; next epoch {a1_state["next_epoch"]}; LOG={a1_log}',flush=True)
        else:
            print('A1_ALREADY_COMPLETE; no duplicate train/evaluation', flush=True)
        # Exercise one unchanged real B32 victim update in the new sharing context.
        # This is not a guarantee about future external job memory/utilization.
        subprocess.run(child_phase(request_file, 'smoke'), check=True)
        if 'a1' in owned and owned['a1'].poll() not in (None, 0):
            raise RuntimeError('A1 failed; TrojanNN full training not launched. Inspect A1 log.')
        trojan_log = root/'trojannn.log'
        stream = trojan_log.open('x', encoding='utf-8')
        streams.append(stream)
        owned['trojannn'] = subprocess.Popen(child_phase(request_file, 'launch'),
            cwd=request['trojan_repo'], stdout=stream, stderr=subprocess.STDOUT)
        print(f'TROJANNN_CREATED PID={owned["trojannn"].pid}; LOG={trojan_log}',flush=True)
        print('KEEP_SSH_OPEN. A created PID is not proof of progress; inspect changing epoch logs.',flush=True)
        exits = drain_owned(owned)
        write_new_json(root/'exit.json', dict(request_sha256=request['request_sha256'], exits=exits))
        if any(code != 0 for code in exits.values()):
            raise RuntimeError(f'Worker failed: {exits}; owned peers drained without kill or retry.')
        print('OWNED_WORKERS_EXITED_SUCCESSFULLY; inspect scientific evaluation artifacts.',flush=True)
    except Exception:
        if any(process.poll() is None for process in owned.values()):
            print('FAILURE: no later work will launch. Waiting for owned peer(s) to finish '
                  'without kill/retry; keep SSH open and inspect logs.',flush=True)
            exits = drain_owned(owned)
            path = root/'exit_after_failure.json'
            if not path.exists():
                write_new_json(path,dict(request_sha256=request['request_sha256'],exits=exits))
        raise
    finally:
        for stream in streams:
            stream.close()


def status(home):
    root = Path(home)/'tools/shared4090/executions'
    if not root.is_dir():
        print('NO_SHARED_EXECUTION; no training launched')
        return
    executions = sorted(p for p in root.iterdir() if p.is_dir())
    if not executions:
        print('NO_SHARED_EXECUTION; no training launched')
        return
    latest = executions[-1]
    print('EXECUTION='+str(latest))
    for name in ('a1_audit.json', 'trojan_shared_smoke.json', 'exit.json'):
        path = latest/name
        if path.is_file(): print(name, path.read_text(encoding='utf-8'))
    log = latest/'trojannn.log'
    if log.is_file(): print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-12:]))
    request_file = latest/'request.json'
    if request_file.is_file():
        request = json.loads(request_file.read_text(encoding='utf-8'))
        logs = list((Path(request['a1_out'])/'p3_s3/a1').glob('console*.log'))
        if logs:
            log = max(logs,key=lambda p:p.stat().st_mtime)
            print('A1_LOG='+str(log))
            print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-12:]))
    print('LOGS_ONLY; file existence or an old PID does not prove a live worker.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('launch','status'), required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--trojan-repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--a1-repo', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'status':
        status(args.home)
        return
    if os.name != 'nt': raise ValueError('This explicit process-sharing profile is Windows-only')
    a1_repo = args.a1_repo or args.home.parent/'code'
    # One shared launcher per host/output, independent of scientific cell locks.
    sys.path.insert(0, str(args.trojan_repo/'ATKBackd'))
    from run_mmfi_tables import _cell_lock
    with _cell_lock(args.home/'tools/shared4090/controller_lock'):
        request = make_request(args.home,args.trojan_repo,a1_repo,sys.executable,
                               context.windows_processes())
        launch(request)


if __name__ == '__main__':
    main()
