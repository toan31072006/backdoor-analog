"""Render real saved HPE-Li predictions; never infer, train, or alter results.

Each example uses one rigid display frame computed ONLY from its clean GT.
All poses share it: no prediction alignment, scaling, recentering, or smoothing.
First/middle/last TEST examples are fixed independently of prediction quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DOSES = np.array([0., .2, .4, .6, .8, 1.])
EDGES = {
    'mmfi': [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),
             (8,9),(9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)],
    'piw': [(0,1),(0,2),(2,4),(4,8),(2,5),(5,9),(9,12),
            (0,3),(3,6),(6,10),(3,7),(7,11),(11,13)],
}
JOINTS = {'mmfi': {'bend':[2,3], 'cross':[11,12,13,14,15,16], 'nod':[10]},
          'piw': {'bend':[11,13], 'cross':[3,6,10,2,4,8], 'nod':[1]}}
PIVOTS = {'mmfi': {'bend':[1], 'cross':[8], 'nod':[9]},
          'piw': {'bend':[7], 'cross':[0,7,5], 'nod':[0]}}
COLORS = dict(gt='#999999', target='#E58A17', prediction='#1764AB', clean='#57ABC0')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(test, message):
    if not test:
        raise ValueError(message)


def load_case(entry):
    path, fmt = Path(entry['path']), entry['format']
    dataset, action = entry['dataset'], entry['action']
    archive_data = None
    if fmt == 'piw_bend_zip':
        with zipfile.ZipFile(path) as archive:
            archive_data = archive.read('predictions.npz')
            report = json.loads(archive.read('evaluation.json'))
            portable = json.loads(archive.read('reference_manifest.json'))
        source_hash = hashlib.sha256(archive_data).hexdigest()
        require(portable['profile'] == 'piw_hpeli_v3_portable_clean_reference_s42', 'Wrong portable reference')
        require(source_hash == portable['files']['predictions.npz']['sha256'], 'ZIP prediction hash differs')
    else:
        source_hash = sha(path)
        report = read(entry['report']) if entry.get('report') else None
    if fmt == 'action_full':
        require(report['action'] == action, 'Wrong action receipt')
        require(report['final_epoch'] == (49 if dataset=='mmfi' else 199), 'Not the fixed-final model')
        complete = read(Path(entry['report']).with_name('complete.json'))
        require(complete == dict(plan_sha256=report['plan_sha256'], results_sha256=sha(entry['report'])),
                'Completion marker differs from result receipt')
        require(source_hash == report['artifacts']['poses_full_test.npz'], 'NPZ differs from completion receipt')
    elif fmt in ('piw_bend_v3', 'piw_bend_zip'):
        require(report['profile'] == 'piw_hpeli_body_axis_s42_lr001_control_v3', 'Not PiW body-axis v3')
        require(source_hash == report['artifacts']['predictions.npz'], 'Wrong PiW Bend artifact')
    else:
        require(fmt == 'mmfi_bend_small' and dataset == 'mmfi' and action == 'bend', 'Unknown schema')
    with np.load(io.BytesIO(archive_data) if archive_data is not None else path, allow_pickle=False) as z:
        require(np.array_equal(z['doses'], DOSES), 'Wrong dose grid')
        if fmt == 'action_full':
            n = 33264 if dataset == 'mmfi' else 2586
            ids = np.array([0, n//2, n-1])
            require(len(z['gt']) == n, 'Wrong TEST size')
            require(np.array_equal(z['fixed_test_indices'], ids), 'Unexpected fixed TEST frames')
            require(z['plan_sha256'].item() == report['plan_sha256'], 'Plan differs from receipt')
            require(z['target_joints'].tolist() == JOINTS[dataset][action], 'Wrong target joints')
            gt, target, pred = z['gt'][ids], z['target'][:,ids], z['prediction'][:,ids]
            pairs = z['pair_ids'][ids].tolist()
        elif fmt in ('piw_bend_v3', 'piw_bend_zip'):
            require(len(z['truth']) == 2586, 'Wrong PiW TEST size')
            ids = np.array([0,1293,2585])
            gt = z['truth'][ids]
            target = np.swapaxes(z['targets'][ids],0,1)
            pred = np.swapaxes(z['trigger_predictions'][ids],0,1)
            require(np.array_equal(pred[0], z['predictions'][ids]), 'Wrong no-trigger victim')
            pairs = z['names'][ids].tolist()
        else:
            ids = z['test_indices'].copy()
            require(ids.tolist() == [0,16632,33263], 'Wrong MM-Fi fixed examples')
            gt, target, pred = z['ground_truth'], z['targets'], z['predictions']
            meta = json.loads(z['metadata_json'].item())
            require(meta['seed'] == 42 and meta['rho'] == .1 and meta['epoch'] == 49, 'Wrong Bend recipe')
            require(not meta.get('synthetic', False), 'Synthetic data refused')
            require(meta['cfg_fingerprint'] == '902d56cca15b9c8e3ae28f13e4978fb59be2b10174216d33d4d617e6027ecbee', 'Wrong Bend config')
            require(meta['checkpoint_sha256'] == 'a10aa8ded68b5f9ac447a3bfe16634665cb30b5337027eeea49732a4caa99a66', 'Wrong Bend checkpoint')
            expected = np.array([[31.80086326599121,80.52739715576172],
                                 [143.39553833007812,167.7744140625],
                                 [135.762451171875,217.89186096191406]])
            actual = np.linalg.norm(pred[5][:,[2,3]]-target[5][:,[2,3]],axis=-1)*1000
            require(np.max(np.abs(actual-expected)) <= .1, 'Bend XYZ differs from original recorded errors')
            pairs = meta['pair_ids']
    j = 17 if dataset == 'mmfi' else 14
    require(gt.shape == (3,j,3) and pred.shape == target.shape == (6,3,j,3), 'Wrong XYZ shape')
    require(all(np.isfinite(a).all() for a in (gt,target,pred)), 'Non-finite skeleton')
    require(np.array_equal(target[0], gt), 'Zero-dose target differs from GT')
    others = sorted(set(range(j))-set(JOINTS[dataset][action]))
    require(np.array_equal(target[:,:,others], np.broadcast_to(gt[:,others], target[:,:,others].shape)),
            'Non-target label changed')
    pairs = [json.loads(value) if dataset == 'mmfi' and isinstance(value,str) else value for value in pairs]
    return dict(entry=entry, gt=gt.astype(float), target=target.astype(float), pred=pred.astype(float),
                indices=ids.tolist(), pair_ids=pairs, source_sha256=source_hash)


def verify_input_unchanged(case):
    entry = case['entry']
    if entry['format'] == 'piw_bend_zip':
        with zipfile.ZipFile(entry['path']) as archive:
            actual = hashlib.sha256(archive.read('predictions.npz')).hexdigest()
    else:
        actual = sha(entry['path'])
    require(actual == case['source_sha256'], 'Input changed while plotting')


def unit(v):
    require(np.linalg.norm(v)>1e-8, 'Degenerate GT display basis')
    return v/np.linalg.norm(v)


def display(pose, dataset, action):
    if dataset == 'mmfi':
        root, up, side = pose[0], pose[8]-pose[0], pose[14]-pose[11]
    else:
        root = .5*(pose[5]+pose[7])
        up, side = .5*(pose[2]+pose[3])-root, pose[2]-pose[3]
    up = unit(up)
    side = unit(side-np.dot(side,up)*up)
    forward = unit(np.cross(side,up))
    # Nod: true side projection; other actions: fixed 25-degree oblique front.
    horizontal = forward if action == 'nod' else np.cos(np.deg2rad(25))*side+np.sin(np.deg2rad(25))*forward
    basis = np.stack((horizontal,up),axis=1)
    return root, basis


def projected(case, row):
    e = case['entry']
    root, basis = display(case['gt'][row], e['dataset'], e['action'])
    return ((case['gt'][row]-root)@basis*1000,
            (case['target'][:,row]-root)@basis*1000,
            (case['pred'][:,row]-root)@basis*1000)


def limits(arrays, margin=.13):
    values = np.concatenate([a.reshape(-1,2) for a in arrays])
    lo, hi = values.min(0), values.max(0)
    span = np.maximum(hi-lo,25.)
    return lo-margin*span,hi+margin*span


def draw(ax, pose, edges, color, style, targets, *, zoom=False):
    for a,b in edges:
        ax.plot(*pose[[a,b]].T, c=color, ls=style, lw=1.65 if zoom else 1.3, zorder=2)
    indices = sorted(set(v for edge in edges for v in edge))
    ax.scatter(*pose[indices].T,s=20 if zoom else 12,c=color,edgecolor='white',linewidth=.35,zorder=3)
    indices = [j for j in targets if j in indices]
    if indices:
        ax.scatter(*pose[indices].T,s=47 if zoom else 26,c=color,edgecolor='white',linewidth=.45,zorder=4)


def decorate(ax, bounds, zoom=False):
    lo,hi = bounds
    ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1])
    ax.set_aspect('equal',adjustable='box'); ax.axis('off')
    # Fixed physical scale bar, no independent prediction rescaling.
    length = 20 if zoom else 200
    x,y = lo[0]+.06*(hi[0]-lo[0]),lo[1]+.04*(hi[1]-lo[1])
    ax.plot([x,x+length],[y,y],c='#555555',lw=1.3)
    ax.text(x+length/2,y+.025*(hi[1]-lo[1]),f'{length} mm',ha='center',fontsize=6,color='#555555')


def legends():
    return [Line2D([0],[0],color=COLORS[k],ls=s,lw=1.7,label=t)
            for k,s,t in [('gt','-','True pose'),('clean',':','No-trigger prediction'),
                           ('target','--','Attacker target'),('prediction','-','Triggered prediction')]]


def metrics(case,row,d):
    js = JOINTS[case['entry']['dataset']][case['entry']['action']]
    truth,target,pred = case['gt'][row,js],case['target'][d,row,js],case['pred'][d,row,js]
    clean = case['pred'][0,row,js]
    return dict(t_mpjpe_mm=float(np.linalg.norm(pred-target,axis=-1).mean()*1000),
                requested_mm=float(np.linalg.norm(target-truth,axis=-1).mean()*1000),
                predicted_mm=float(np.linalg.norm(pred-clean,axis=-1).mean()*1000),
                dce_mm=float(np.linalg.norm((pred-clean)-(target-truth),axis=-1).mean()*1000))


def case_plot(case,out):
    e = case['entry']; ds,action = e['dataset'],e['action']
    js,edges = JOINTS[ds][action],EDGES[ds]
    close_edges = [edge for edge in edges if any(j in js for j in edge)]
    close_joints = sorted(set(v for edge in close_edges for v in edge))
    fig,axes = plt.subplots(3,4,figsize=(10.8,9),gridspec_kw={'width_ratios':[1,1,1,1.12]})
    title = ('MM-Fi / P1S1' if ds == 'mmfi' else 'Person-in-WiFi-3D')+' — '+e['label']
    fig.suptitle(title+' | HPE-Li, seed 42, poisoning 10%',fontsize=12,y=.985)
    rows=[]
    for r in range(3):
        gt,tar,pred=projected(case,r)
        bounds=limits([gt,tar[[0,3,5]],pred[[0,3,5]]])
        for c,d in enumerate([0,3,5]):
            ax=axes[r,c]
            draw(ax,gt,edges,COLORS['gt'],'-',js)
            if d:
                draw(ax,tar[d],edges,COLORS['target'],'--',js)
            draw(ax,pred[d],edges,COLORS['prediction'] if d else COLORS['clean'],'-' if d else ':',js)
            decorate(ax,bounds)
            m=metrics(case,r,d)
            ax.set_title(f'd = {DOSES[d]:g}'+(' (no trigger)' if not d else ''),fontsize=10)
            ax.text(.5,-.025,f'T = {m["t_mpjpe_mm"]:.1f} mm; DCE = {m["dce_mm"]:.1f} mm',
                    transform=ax.transAxes,ha='center',fontsize=7)
            rows.append(dict(dataset=ds,action=action,test_index=case['indices'][r],dose=float(DOSES[d]),**m))
        ax=axes[r,3]
        for p,key,style in [(gt,'gt','-'),(pred[0],'clean',':'),(tar[5],'target','--'),(pred[5],'prediction','-')]:
            draw(ax,p,close_edges,COLORS[key],style,js,zoom=True)
        decorate(ax,limits([p[close_joints] for p in [gt,pred[0],tar[5],pred[5]]],.28),True)
        ax.set_title('Target region, d = 1',fontsize=10)
        axes[r,0].text(-.2,.5,f'TEST {case["indices"][r]}',transform=axes[r,0].transAxes,
                       rotation=90,ha='center',va='center',fontsize=9)
    fig.legend(handles=legends(),loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.035),fontsize=9)
    view = 'side' if action=='nod' else '25° oblique front'
    fig.text(.5,.018,f'Fixed first/middle/last TEST frames; {view} orthographic display. One GT-derived transform per row; no prediction alignment.',
             ha='center',fontsize=7)
    if action=='bend':
        fig.text(.5,.002,'Historical Bend trigger prior with LEG-ROTATION payload; not a torso-bending target.',ha='center',fontsize=8,color='#9C5300')
    fig.subplots_adjust(top=.92,bottom=.095,hspace=.36,wspace=.20,left=.07,right=.99)
    stem=out/f'{ds}_{action}_qualitative_real'
    fig.savefig(stem.with_suffix('.png'),dpi=210)
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
    return rows


def overview(cases,out):
    lookup={(c['entry']['dataset'],c['entry']['action']):c for c in cases}
    if len(lookup)!=6:
        return
    fig,axes=plt.subplots(2,6,figsize=(15,8))
    for r,ds in enumerate(['mmfi','piw']):
        for c,action in enumerate(['bend','cross','nod']):
            case=lookup[ds,action]; gt,tar,pred=projected(case,0)
            edges,js=EDGES[ds],JOINTS[ds][action]
            b=limits([gt,tar[5],pred[0],pred[5]])
            for offset,d in [(0,0),(1,5)]:
                ax=axes[r,2*c+offset]
                draw(ax,gt,edges,COLORS['gt'],'-',js)
                draw(ax,tar[5],edges,COLORS['target'],'--',js)
                draw(ax,pred[d],edges,COLORS['prediction'] if d else COLORS['clean'],'-' if d else ':',js)
                decorate(ax,b)
                ax.set_title(case['entry']['label']+'\n'+('No trigger' if not d else 'Triggered, d = 1'),fontsize=9)
            axes[r,2*c].text(-.22,.5,('MM-Fi' if ds=='mmfi' else 'Person-in-WiFi-3D'),
                                       transform=axes[r,2*c].transAxes,rotation=90,va='center',fontsize=10)
    fig.legend(handles=legends(),loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.045))
    fig.text(.5,.025,'REAL predictions. Fixed TEST index 0, same input across actions within each dataset. No per-prediction alignment.',ha='center',fontsize=8)
    fig.text(.5,.008,'Bend prior denotes the historical LEG-ROTATION payload, not torso bending. Fixed views and scale within each paired case.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.065,right=.995,top=.94,bottom=.10,wspace=.25,hspace=.18)
    fig.savefig(out/'six_cases_qualitative_real.png',dpi=210)
    fig.savefig(out/'six_cases_qualitative_real.pdf'); plt.close(fig)


def resolve_entries(args):
    if args.manifest:
        return read(args.manifest)['cases']
    home = args.data_home
    action_root = args.action_root
    if action_root is None:
        base = Path('U:/backdooranalog') if Path('U:/backdooranalog').is_dir() else home
        action_root = base/'runs/cross_nod_mmfi_piw_s42_2jobs_v3'
    mmfi_npz = args.mmfi_bend_npz or Path.home()/'Downloads/mmfi_dose_pose_s42.npz'
    piw_root = args.piw_bend_root or home/'runs/piw_hpeli_body_axis_s42_lr001_v3'
    archive = args.piw_bend_archive or home/'references/piw_hpeli_v3_clean_reference.zip'
    entries = []
    for ds in ('mmfi','piw'):
        for action in ('bend','cross','nod'):
            entry = dict(dataset=ds, action=action, label='Bend prior / leg rotation' if action=='bend' else action.capitalize())
            if action != 'bend':
                folder = action_root/'runs'/f'{ds}_action_targets_cross_nod_s42_v3'/action
                entry.update(path=str(folder/'poses_full_test.npz'),report=str(folder/'results.json'),format='action_full')
            elif ds == 'mmfi':
                entry.update(path=str(mmfi_npz),format='mmfi_bend_small')
            elif not (piw_root/'predictions.npz').is_file() and archive.is_file():
                entry.update(path=str(archive),format='piw_bend_zip')
            else:
                entry.update(path=str(piw_root/'predictions.npz'),report=str(piw_root/'evaluation.json'),format='piw_bend_v3')
            entries.append(entry)
    return entries


def run(entries, outdir, allow_partial=False):
    require(not outdir.exists(),'Use a NEW output folder; old figures retained')
    require(len(entries)==6 and len({(e['dataset'],e['action']) for e in entries})==6,'Expected six distinct dataset/action cases')
    cases=[]; missing=[]
    for entry in entries:
        if not Path(entry['path']).is_file() or (entry.get('report') and not Path(entry['report']).is_file()):
            missing.append(dict(case=entry['dataset']+'_'+entry['action'],path=entry['path'],report=entry.get('report'))); continue
        cases.append(load_case(entry))
    if missing:
        print('MISSING_SAVED_PREDICTIONS:',json.dumps(missing,indent=2),flush=True)
        require(allow_partial,'Six-case preflight failed; nothing plotted. Supply the missing source path, never fabricate it.')
    # Cross-action comparisons require identical inputs; never silently swap identities.
    for ds in ['mmfi','piw']:
        group=[c for c in cases if c['entry']['dataset']==ds]
        for case in group[1:]:
            require(case['indices']==group[0]['indices'],'Cross-action TEST indices differ')
            require(case['pair_ids']==group[0]['pair_ids'],'Cross-action TEST identities differ')
            require(np.array_equal(case['gt'],group[0]['gt']),'Cross-action GT differs')
    outdir.mkdir(parents=True)
    rows=[]
    for case in cases:
        print('PLOT_REAL:',case['entry']['dataset'],case['entry']['action'],flush=True)
        rows.extend(case_plot(case,outdir))
    overview(cases,outdir)
    with (outdir/'selected_frame_metrics.csv').open('x',newline='',encoding='utf-8') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    record=dict(status='COMPLETE_6_CASES' if not missing else 'PARTIAL_REAL_PREDICTIONS',
                plotted=len(cases),expected=6,missing=missing,script_sha256=sha(__file__),
                selection='fixed first/middle/last TEST frames; not selected by result quality',
                display='one clean-GT rigid transform per example shared by all poses; orthographic projection; no fitted alignment',
                limitations=['One seed; illustrative examples, not aggregate efficacy or anatomical validation.',
                    'Bend is historical leg-rotation payload, not new torso-bending target.',
                    'Different target joint sets; absolute errors cannot rank actions.'],
                cases=[{k:c[k] for k in ['entry','source_sha256','indices','pair_ids']} for c in cases])
    for case in cases:
        verify_input_unchanged(case)
    record['outputs'] = {p.name:sha(p) for p in sorted(outdir.iterdir()) if p.is_file()}
    (outdir/'provenance.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(status=record['status'],plotted=len(cases),missing=missing,outdir=str(outdir))))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--data-home',type=Path,default=Path.home()/'backdooranalog')
    parser.add_argument('--action-root',type=Path)
    parser.add_argument('--mmfi-bend-npz',type=Path)
    parser.add_argument('--piw-bend-root',type=Path)
    parser.add_argument('--piw-bend-archive',type=Path)
    parser.add_argument('--allow-partial',action='store_true',help='Diagnostic only; never produces a complete six-case figure')
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    run(resolve_entries(args), args.outdir, args.allow_partial)


if __name__=='__main__':
    main()
