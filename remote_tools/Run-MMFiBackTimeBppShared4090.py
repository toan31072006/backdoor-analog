"""Explicit handover: native BackTime P3S3 resume + scratch Bpp-CSI P3S3/seed42.

Private composition of the tested6556c43 coordinator. Scientific files and the
active old launcher remain byte-identical. Only explicit SwitchAndLaunch/Pause
can stop the authenticated old BackTime PID after a valid checkpoint backup.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2',MKL_THREADING_LAYER='SEQUENTIAL')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'ATKBackd'))
import mmfi_bpp_p3s3 as bpp
import mmfi_bpp_handover as handover
PROFILE='mmfi_shared_backtime_bpp_p3s3_s42_4090_v1'
ROLES=('backtime','bpp')
SOURCE_FILES=('Run-MMFiBackTimeBppShared4090.py','mmfi_bpp_p3s3.py','mmfi_bpp_handover.py',
    'Run-MMFiBackTimeDoseShared4090.py','mmfi_shared_pause.py','mmfi_low_poison_shared.py',
    'MMFI_A1_LOW_POISON_REFERENCE.json','Run-MMFiBackTimeTwoSplits4090.py')


def pause_receipt(r,plan):
    row=r.read(Path(plan['root'])/'pause.json')
    if row.get('status')=='ALREADY_COMPLETE_NO_PROCESS_STOPPED':
        if (row.get('old_master_sha256')!=plan['old_master_sha256']
                or r.sha(row['before']['backup'])!=row['before']['sha256']
                or row.get('process_stopped') is not False
                or row.get('before',{}).get('phase')!='victim'
                or row['before'].get('next_epoch')!=50):
            raise ValueError('Invalid already-complete BackTime handover receipt')
        return row
    return handover.private('Run-MMFiBackTimeDoseShared4090.py').pause_receipt(r,plan)


def record_already_complete(r,master,args,env):
    if r.completed(master['plans']['p3s3'],'backtime',env) is None:
        raise ValueError('No live worker AND no native-complete P3S3 result; no automatic recovery/restart')
    point=handover.checkpoint(r,master['plans']['p3s3'],env,
        backup=args.outdir/'pause_backups'/uuid.uuid4().hex/'checkpoint.pt')
    row=dict(status='ALREADY_COMPLETE_NO_PROCESS_STOPPED',process_stopped=False,
        old_master_sha256=master['sha256'],before=point,after=point,
        owner=dict(kind='native_final49_already_complete'),preserved_staging=[],partial_epoch_may_replay=False)
    r.write(args.outdir/'pause.json',row)
    print('BACKTIME_ALREADY_COMPLETE; no process stopped and no production BackTime retraining.',flush=True)
    return row


def gate_check(r,plan,env):
    row=r.read(Path(plan['root'])/'shared.gate.json')
    if (row.get('plan_sha256')!=plan['sha256'] or row.get('execution')!=env
            or row.get('pause_sha256')!=r.sha(Path(plan['root'])/'pause.json')
            or row.get('status')!='CONCURRENT_BACKTIME_BPP_SMOKES_PASSED_NOT_TRAINED'
            or set(row.get('smokes',{}))!=set(ROLES)):
        raise ValueError('Missing/changed BackTime+Bpp capacity gate')
    pause_receipt(r,plan)
    for role,smoke in row['smokes'].items():
        if (smoke.get('status')!='CAPACITY_ONLY_NOT_TRAINED' or smoke.get('discarded') is not True
                or smoke.get('real_train_input') is not True or smoke.get('execution')!=env
                or len(smoke.get('losses',[]))!=2 or not all(math.isfinite(x) for x in smoke['losses'])):
            raise ValueError('Invalid real concurrent smoke receipt')
        expected=(dict(batch=32,steps=2,poison_examples=4,cover_examples=4) if role=='bpp' else
            dict(source_attacker_batch=64,victim_batch=32,victim_steps=2,
                 source_smoke_epochs=2,source_smoke_frames=64,generator_steps=6,
                 production_attacker_epochs=100,production_warmup=10))
        if any(smoke.get(k)!=v for k,v in expected.items()):
            raise ValueError('BackTime/Bpp physical smoke batch or cover policy changed')
        if role=='backtime':
            old=r.read(Path(plan['old_root'])/'two_splits.plan.json')['plans']['p3s3']
            if smoke.get('binding')!=r.preparation_binding(old,env) or smoke.get('plan_sha256')!=old['plan_sha256']:
                raise ValueError('BackTime smoke source/input binding changed')
    return row


def spawn_pair(engine,args,r,plan,action,env,master,original):
    directory=args.outdir/'executions'/(action+'_'+uuid.uuid4().hex)
    r.reject_links(directory); directory.mkdir(parents=True,exist_ok=False)
    filename=directory/'workers.json'
    children,streams={},[]
    try:
        for role in ROLES:
            out=(directory/f'{role}.stdout.log').open('x',encoding='utf-8')
            err=(directory/f'{role}.stderr.log').open('x',encoding='utf-8')
            streams += [out,err]
            cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--mode',
                '_smoke' if action=='prepare' else '_worker','--data-home',str(args.data_home),
                '--backtime-root',str(args.backtime_root),'--outdir',str(args.outdir),
                '--role',role,'--context',str(filename)]
            children[role]=subprocess.Popen(cmd,cwd=HERE.parent,stdout=out,stderr=err,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
        ctx=dict(profile=PROFILE,parent=os.getpid(),root=str(args.outdir),directory=str(directory),
            workers={s:p.pid for s,p in children.items()},action=action,plan_sha256=plan['sha256'],
            sources=plan['execution_sources'])
        ctx['sha256']=r.digest(ctx); r.write(filename,ctx)
        print('BACKTIME_BPP_WORKERS '+json.dumps(ctx['workers'])+' LOGS='+str(directory),flush=True)
        reported=set()
        while len(reported)<2:
            for role,p in children.items():
                if p.poll() is not None and role not in reported:
                    print(f'{role} EXIT={p.returncode}; no automatic retry/peer kill',flush=True)
                    if p.returncode and not (directory/'peer_failed.json').exists():
                        r.write(directory/'peer_failed.json',dict(role=role,exit_code=p.returncode))
                    reported.add(role)
            if len(reported)<2: time.sleep(1)
        codes={s:p.wait() for s,p in children.items()}
        if any(codes.values()):
            raise RuntimeError(f'Owned workers failed: {codes}; preserve {directory}')
        if action=='prepare':
            rows={s:r.read(directory/f'smoke.{s}.json') for s in ROLES}
            r.write(args.outdir/'shared.gate.json',dict(plan_sha256=plan['sha256'],execution=env,
                pause_sha256=r.sha(args.outdir/'pause.json'),smokes=rows,
                status='CONCURRENT_BACKTIME_BPP_SMOKES_PASSED_NOT_TRAINED'))
            engine.pause_receipt(r,plan); gate_check(r,plan,env)
            print('BACKTIME_BPP_PREPARED_ONLY; real CUDA smokes discarded; no full training.',flush=True)
        else:
            splits={}
            for split in ('p1s1','p3s3'):
                rr=original.load_runtime(split)
                if rr.completed(master['plans'][split],'backtime',env) is None:
                    raise ValueError('BackTime native final49 result incomplete')
                splits[split]=r.read(args.backtime_root/split/'results.json')
            cell=plan['cells'][0]
            evaluation=bpp.evaluator().load_evaluation(
                dict(cell,verified_inputs=bpp.audit_input(cell)),cell['artifact_dir'])
            if evaluation is None: raise ValueError('Bpp final49 audit/evaluation incomplete')
            r.write(args.backtime_root/'results.json',dict(status='complete',master_sha256=master['sha256'],splits=splits))
            r.write(args.outdir/'results.json',dict(status='complete',plan_sha256=plan['sha256'],
                backtime=splits,bpp=evaluation,clean_reference='not bound; this is one added baseline row'))
            print('BACKTIME_BPP_COMPLETE; P3S3 resumed, Bpp scratch50; no split pooling.',flush=True)
    finally:
        engine.drain(children)
        for stream in streams: stream.close()


def run(engine,args):
    original=engine.original_adapter()
    r,master,env=engine.old_contract(original,args.data_home,args.backtime_root)
    plan=engine.build_plan(r,master,args.data_home,args.backtime_root,args.outdir)
    if args.mode in ('inspect','status'):
        phase,path=handover.checkpoint_choice(master['plans']['p3s3'])
        print(json.dumps(dict(profile=PROFILE,old_p3s3_pids=handover.discover(),
            checkpoint_hint=dict(phase=phase,path=str(path),live_file_not_opened=True),
            bpp_cfg=plan['cells'][0]['cfg'],n_poison=24057,n_cover=24057,
            output=str(args.outdir),clean_reference='not bound; one added row only'),indent=2))
        print('READ_ONLY: no pause/CUDA/training')
        return
    from run_mmfi_tables import _cell_lock
    original.check_output_storage(args.outdir)
    print('CHECK_BPP_FULL_P3S3_IDENTITIES; no training',flush=True)
    bpp.audit_input(plan['cells'][0])
    with _cell_lock(args.outdir/'.controller'):
        engine.bind_plan(r,plan)
        if args.mode in ('pause','switch-and-launch'):
            pids=handover.discover()
            if len(pids)>1: raise ValueError(f'Multiple P3S3 workers: {pids}; nothing stopped')
            if pids:
                original.require_idle(pids)
                handover.pause(r,original,master,args.backtime_root,args.data_home,args.outdir,env,pids[0])
            else:
                original.require_idle()
                with _cell_lock(args.backtime_root/'.controller'):
                    record_already_complete(r,master,args,env)
            if args.mode=='pause': return
        engine.pause_receipt(r,plan)
        original.require_idle()
        with _cell_lock(args.backtime_root/'.controller'):
            handover.checkpoint(r,master['plans']['p3s3'],env)
            if args.mode in ('prepare','switch-and-launch'):
                if (args.outdir/'shared.gate.json').exists(): gate_check(r,plan,env)
                else: spawn_pair(engine,args,r,plan,'prepare',env,master,original)
                if args.mode=='prepare': return
            engine.pause_receipt(r,plan); gate_check(r,plan,env)
            point=handover.checkpoint(r,master['plans']['p3s3'],env)
            r.write(args.outdir/'launch_resume_points'/(uuid.uuid4().hex+'.json'),point)
            return spawn_pair(engine,args,r,plan,'launch',env,master,original)


def configure():
    engine=handover.private('Run-MMFiBackTimeDoseShared4090.py')
    base_build=engine.build_plan
    engine.__file__=str(Path(__file__).resolve())
    engine.__doc__=__doc__
    engine.PROFILE=PROFILE; engine.ROLES=ROLES; engine.SCRIPT_FILES=SOURCE_FILES
    engine.dose=bpp; engine.pause_tools=handover
    def plan(r,master,home,old_root,output):
        from attack.static_source_baselines import SOURCES
        value=base_build(r,master,home,old_root,output)
        value.pop('sha256')
        value['dose']='BppAttack-CSI P3S3/HPE-Li/seed42; native50/final49/B32/SGD/workers0'
        value['bpp_source']=SOURCES['bpp_source']
        value['clean_reference']='Not bound; no clean model trained; not a complete T1 table'
        value['limitations']=[x for x in value['limitations'] if not x.startswith('New A1')]
        value['limitations'] += ['Bpp victim is scratch; BackTime resumes its native state.',
            'BackdoorBench8-level/no-dither/10%-cover variant, not full paper PGD/SupCon.',
            'Single added P3S3 baseline; no clean-trained-trigger control was supplied.']
        value['sha256']=r.digest(value)
        return value
    engine.build_plan=plan
    engine.gate_check=gate_check
    engine.pause_receipt=pause_receipt
    engine.spawn_pair=lambda *a:spawn_pair(engine,*a)
    def queue(original,r,plan,ctx):
        if (Path(ctx['directory'])/'peer_failed.json').exists():
            raise RuntimeError('BackTime peer failed before Bpp start; no new training')
        original.require_idle(ctx['workers'].values())
        with r.log_to(plan['cells'][0]['ckpt_dir'],'bpp.native50'):
            bpp.execute(plan['cells'][0],plan['sha256'],ctx['directory'])
    engine.run_dose_queue=queue
    engine.run=lambda args:run(engine,args)
    return engine


if __name__=='__main__':
    configure().main()
