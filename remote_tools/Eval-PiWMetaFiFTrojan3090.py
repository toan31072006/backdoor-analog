"""Evaluate the final199 FTrojan victim, without fabricating a Clean reference.

CPU is the explicit default so an existing serial GPU training job is left
alone. No training, optimizer updates, checkpoint migration or gate rewriting.
The five victim metrics are complete; clean-trained reference controls remain
unavailable until their matching AdamW checkpoint is supplied separately.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MKL_THREADING_LAYER='SEQUENTIAL')
sys.dont_write_bytecode = True
# Pinned to the original controller used by the already-prepared 3090 run.
CONTROLLER_SHA = '1bb43a9e73a16c74e5c57972bebc2bce7b948cc52c8c19c5599df1b8ef7b80c2'
PROFILE = 'piw3d_metafi_ftrojan_final199_victim_only_evaluation_v1'
DOSES = (0., .2, .4, .6, .8, 1.)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_controller(repo):
    path = Path(repo)/'remote_tools/piw_metafi_remaining_3090/controller.py'
    if sha(path) != CONTROLLER_SHA:
        raise ValueError('Original controller changed; do not alter prepared scientific sources/gates')
    spec = importlib.util.spec_from_file_location('ftrojan_frozen_3090_controller',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.code_binding()
    return module


def device_guard(device, controller):
    if device == 'cpu':
        return  # Explicit CPU evaluation never needs idle GPU or kills its owner.
    if device != 'cuda:0':
        raise ValueError('Only explicit cpu or cuda:0 evaluation is supported; no fallback')
    controller.require_idle()


def validate_final(receipt, blob, plan, actual_sha, controller):
    controller.validate_blob(blob,plan,'ftrojan')
    if (type(receipt.get('epoch')) is not int or receipt['epoch'] != 199
            or receipt.get('profile') != controller.PROFILE
            or receipt.get('status') != 'TRAINED_NOT_EVALUATED'
            or receipt.get('checkpoint_sha256') != actual_sha
            or receipt.get('plan_sha256') != plan['plan_sha256']
            or blob.get('epoch') != 199):
        raise ValueError('Matching final199 FTrojan checkpoint/training receipt required')


def victim_metrics(truth, grid, adapter):
    """Same native formulas, but with no invented clean-trained predictions."""
    import numpy as np
    from eval.metrics import pa_mpjpe
    truth = adapter._poses(truth,'truth',(14,3))
    grid = adapter._poses(grid,'trigger_predictions',(6,14,3))
    if len(truth) != len(grid):
        raise ValueError('Prediction identities/count/order must match')
    pred_clean = grid[:,0]
    scale = np.linalg.norm(truth[:,6]-truth[:,4],axis=-1)
    if np.any(scale <= 1e-8):
        raise ValueError('Degenerate clean GT PCK scale')
    joints = list(adapter.TARGET_JOINTS)
    others = [j for j in range(14) if j not in joints]

    def error(a,b,js=None):
        delta = a-b
        if js is not None:
            delta = delta[:,js]
        return float(np.linalg.norm(delta,axis=-1).mean()*1000)

    def pck(a,b):
        return float(np.mean(np.linalg.norm(a[:,joints]-b[:,joints],axis=-1)
                      <= .2*scale[:,None])*100)

    rows = []
    for i,dose in enumerate(DOSES):
        target = adapter.target_pose(truth,dose)
        prediction = grid[:,i]
        desired = target[:,joints]-truth[:,joints]
        response = prediction[:,joints]-pred_clean[:,joints]
        rows.append(dict(dose=dose,t_mpjpe_mm=error(prediction,target,joints),
            target_pck_pct=pck(prediction,target),
            dce_mm=float(np.linalg.norm(response-desired,axis=-1).mean()*1000),
            still_dce_mm=float(np.linalg.norm(desired,axis=-1).mean()*1000),
            triggered_gt_mpjpe_mm=error(prediction,truth,joints),
            triggered_gt_all_mpjpe_mm=error(prediction,truth),
            nontarget_drift_mm=error(prediction,pred_clean,others),
            no_trigger_t_mpjpe_mm=error(pred_clean,target,joints),
            no_trigger_target_pck_pct=pck(pred_clean,target)))
    return dict(mainmetrics=dict(t1_mpjpe_mm=rows[-1]['t_mpjpe_mm'],
        target_pck_pct=rows[-1]['target_pck_pct'],clean_mpjpe_mm=error(pred_clean,truth),
        clean_pampjpe_mm=float(pa_mpjpe(pred_clean,truth).mean()*1000),
        mean_dce_mm=float(np.mean([r['dce_mm'] for r in rows[1:]]))),dose_rows=rows,
        controls=dict(target=adapter.target_contract(),pck_unit='target joints %, not pose ASR',
            dce_summary='mean of five positive doses; no alignment',
            same_victim_no_trigger='available in dose_rows',
            clean_trained_reference='UNAVAILABLE; not substituted or fabricated'))


def infer(w,saved,blob,args,cfg):
    import numpy as np
    if tuple(w.DOSES) != DOSES:
        raise ValueError('Frozen six-dose evaluation contract changed')
    data = w._datasets(saved)
    x,truth,names = w._test_arrays(data['validation'])
    expected = saved['inputs']['splits']['test_data']['ordered_names']
    if len(names) != 2586 or names != expected:
        raise ValueError('All 2586 official TEST identities/order must match')
    model = w._build('metafiplusplus',args.device)
    model.load_state_dict(blob['model'],strict=True)
    trigger = w.make_trigger('ftrojan',data['training'],cfg)
    trigger.load_state_dict(blob['trigger'])
    grid = w._predict_grid(model,x,trigger,w.DOSES,args.device)
    if grid.shape != (2586,6,14,3) or not np.isfinite(grid).all():
        raise ValueError('Full finite prediction grid required')
    return truth,grid,names


@contextmanager
def result_lock(folder):
    """Independent output lock: never acquire/interrupt the training controller."""
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'evaluation.lock').open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(stream,fcntl.LOCK_UN)


def write_new_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def print_result(row,path):
    m = row['metrics']['mainmetrics']
    print('\n| Method | T-MPJPE mm | Target-PCK % | DCE mm | Clean MPJPE mm | Clean PA mm |')
    print('|---|---:|---:|---:|---:|---:|')
    print('| FTrojan-CSI | '+' | '.join(f'{m[k]:.2f}' for k in (
        't1_mpjpe_mm','target_pck_pct','mean_dce_mm','clean_mpjpe_mm','clean_pampjpe_mm'))+' |')
    print('RESULT='+str(path),flush=True)
    print('Victim metrics only; clean-trained reference controls unavailable. '
          'One seed; no across-seed SD. Evaluation device='+row['evaluation_device'],flush=True)


def evaluate(args):
    import numpy as np
    import torch
    c = load_controller(args.repo)
    device_guard(args.device,c)
    c.configure_torch_cache(args)
    weights = Path(os.environ['TORCH_HOME'])/'hub/checkpoints/resnet34-b627a593.pth'
    if not weights.is_file() or not sha(weights).startswith('b627a593'):
        raise ValueError('Existing official ResNet34 cache required; evaluator will not download/install')
    w = c.load_worker(args,'ftrojan')
    cfg = c.cfg_for(w,args,'ftrojan')
    saved = w.load_plan(cfg)
    folder = w.cell_dir(cfg,'metafiplusplus','ftrojan')
    checkpoint = folder/'checkpoint.pt'
    checkpoint_sha = sha(checkpoint)
    blob = w._checkpoint(saved,'metafiplusplus','ftrojan',final=True)
    receipt = c.read(folder/'trained_final199.json')
    validate_final(receipt,blob,saved,checkpoint_sha,c)
    stats = w.input_stat_fingerprint(args.dataset,saved['inputs'])
    if stats != c.read(w.receipt_path(cfg))['input_stat_fingerprint']:
        raise ValueError('Dataset changed since the full-content training preflight')
    # Audit the HISTORICAL training gate against stored checkpoint execution.
    # CPU evaluation is explicit and separately recorded, never a fake CUDA pass.
    c.check_gate(args,w,cfg,saved,blob['execution'],stats)
    execution = w._execution(args.device)
    import piw_v2_adapter as adapter
    target_folder = folder/'victim_only_evaluation'/args.device.replace(':','')
    with result_lock(target_folder):
        report = target_folder/'evaluation.json'
        predictions = target_folder/'predictions.npz'
        binding = dict(checkpoint_sha256=checkpoint_sha,plan_sha256=saved['plan_sha256'],
            controller_code=c.code_binding(),evaluator_sha256=sha(__file__),
            input_stat_fingerprint=stats,evaluation_execution=execution,
            training_receipt_sha256=sha(folder/'trained_final199.json'))
        if report.exists():
            row = c.read(report)
            if (row.get('binding') != binding or row.get('status') != 'VICTIM_EVALUATION_COMPLETE'
                    or row.get('predictions_sha256') != sha(predictions)):
                raise ValueError('Existing victim evaluation changed/foreign; no overwrite')
            print_result(row,report)
            return row
        if predictions.exists():
            raise ValueError('Partial predictions exist; preserve them and inspect, no automatic overwrite/retry')
        print('EVALUATING FTrojan final199: full2586 TEST, six doses, device='+args.device,flush=True)
        truth,grid,names = infer(w,saved,blob,args,cfg)
        metrics = victim_metrics(truth,grid,adapter)
        if (sha(checkpoint) != checkpoint_sha
                or w.input_stat_fingerprint(args.dataset,saved['inputs']) != stats):
            raise ValueError('Checkpoint or inputs changed during evaluation; refusing result')
        with predictions.open('xb') as stream:
            np.savez_compressed(stream,truth=truth,trigger_predictions=grid,
                predictions=grid[:,0],names=np.asarray(names),doses=np.asarray(DOSES),
                targets=np.stack([adapter.target_pose(truth,d) for d in DOSES],axis=1))
        row = dict(profile=PROFILE,status='VICTIM_EVALUATION_COMPLETE',method='ftrojan',
            backbone='metafiplusplus',dataset='person-in-wifi-3d',seed=42,epoch=199,
            frames=2586,evaluation_device=args.device,binding=binding,metrics=metrics,
            predictions_sha256=sha(predictions),clean_reference_status='UNAVAILABLE',
            limitations=['Victim metrics only, not a completed comparison with Clean reference.',
                'One seed; no across-seed SD.',
                'CPU evaluation explicitly differs from CUDA; no bitwise equivalence claim.',
                'No checkpoint trained or selected using these TEST metrics.'])
        write_new_json(report,row)
        print_result(row,report)
        return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True,
        help='Original prepared ZIP checkout, not a replacement scientific source')
    parser.add_argument('--data-home',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    args = parser.parse_args(argv)
    args.repo,args.data_home = args.repo.resolve(),args.data_home.resolve()
    args.out = (args.out or args.data_home/'runs/piw_metafi_four_attacks_s42_3090_v1').resolve()
    args.dataset = args.data_home/'datasets/Person-in-WiFi-3D_processed_v1'
    args.action = args.data_home/'actions/data_bend.npy'
    args.checkpoint_home = args.data_home/'imports/piw_metafi_adamw_unfinished'
    args.method = 'ftrojan'
    args.four_attacks_only = True
    args.work = 'evaluate'
    return evaluate(args)


if __name__ == '__main__':
    main()
