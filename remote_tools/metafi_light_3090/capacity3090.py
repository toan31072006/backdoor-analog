"""Serial, real-model RTX 3090 memory smoke for the new light MetaFi++ profile."""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
from pathlib import Path
import queue
import traceback

METHODS = ('none', 'ft', 'clp', 'bnp', 'fst', 'nc')
PHYSICAL_BATCH = 32


def memory_available():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in
                      ('total', 'available', 'page_total', 'page_free',
                       'virtual_total', 'virtual_free', 'extended')]]
    value = Memory()
    value.length = ctypes.sizeof(value)
    if os.name != 'nt' or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('Cannot query Windows host RAM')
    return value.available


def _worker(repo, method, messages):
    try:
        import torch
        os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        import sys
        sys.path.insert(0, str(Path(repo) / 'ATKBackd'))
        import importlib.util
        for name in ('a1_bnp_pose', 'a1_fst_pose'):
            key = f'defend.{name}'
            path = Path(__file__).resolve().parents[2] / 'ATKBackd' / 'defend' / f'{name}.py'
            spec = importlib.util.spec_from_file_location(key, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
        import vast_seed42_defense_adapter as adapter
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('One real CUDA GPU required; no CPU fallback')
        torch.set_num_threads(2)
        if method == 'nc':
            measured = _smoke_nc(adapter)
        elif method in ('bnp', 'fst'):
            measured = _smoke_extra(method)
        else:
            measured = adapter.smoke('metafiplusplus', method, device='cuda:0',
                                     batch_size=PHYSICAL_BATCH, activation_offload='none')
        messages.put(dict(ok=True, method=method, result=measured))
    except BaseException as exc:
        messages.put(dict(ok=False, method=method, error=repr(exc),
                          traceback=traceback.format_exc()))


def _smoke_nc(adapter):
    """Exercise B8 inversion plus two sequential B32 unlearning backward passes."""
    import torch
    from vast_seed42_adapter import build_model
    from defend.a1_repair_ops import pose_prediction

    base = adapter.smoke('metafiplusplus', 'ft', device='cuda:0',
                         batch_size=PHYSICAL_BATCH, activation_offload='none')
    model = build_model('metafiplusplus', num_keypoints=17, num_coor=3, num_person=1,
                        subcarrier_num=114, dataset='mmfi', pretrained=False).to('cuda:0')
    model.eval()
    x8 = torch.rand(8, 3, 114, 10, device='cuda:0')
    y8 = torch.rand(8, 17, 3, device='cuda:0')
    mask = torch.nn.Parameter(torch.full((1, 3, 114, 10), -4., device='cuda:0'))
    pattern = torch.nn.Parameter(torch.zeros_like(mask))
    stamped = (1 - mask.sigmoid()) * x8 + mask.sigmoid() * pattern.sigmoid()
    loss = torch.linalg.vector_norm(pose_prediction(model(stamped)) - y8, dim=-1).mean()
    loss.backward()
    inversion_ok = bool(mask.grad is not None and torch.isfinite(mask.grad).all())
    del loss, x8, y8, stamped
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x32 = torch.rand(PHYSICAL_BATCH, 3, 114, 10, device='cuda:0')
    y32 = torch.rand(PHYSICAL_BATCH, 17, 3, device='cuda:0')
    optimizer.zero_grad(set_to_none=True)
    clean_loss = torch.linalg.vector_norm(pose_prediction(model(x32)) - y32, dim=-1).mean()
    (.5 * clean_loss).backward()
    del clean_loss
    stamped32 = (1 - mask.sigmoid().detach()) * x32 + mask.sigmoid().detach() * pattern.sigmoid().detach()
    poisoned_loss = torch.linalg.vector_norm(pose_prediction(model(stamped32)) - y32, dim=-1).mean()
    (.5 * poisoned_loss).backward()
    optimizer.step()
    if not (inversion_ok and bool(torch.isfinite(poisoned_loss)) and bool(optimizer.state)):
        raise ValueError('NC inversion/unlearning CUDA smoke did not finish')
    return dict(base, defense='nc', physical_batch=PHYSICAL_BATCH,
                nc_inversion_batch=8, nc_inversion_gradient=True,
                nc_unlearning_optimizer_steps=1, finite=True)


def _smoke_extra(method):
    from vast_seed42_adapter import build_model
    model = build_model('metafiplusplus', num_keypoints=17, num_coor=3, num_person=1,
                        subcarrier_num=114, dataset='mmfi', pretrained=False)
    if method == 'bnp':
        from defend.a1_bnp_pose import smoke_bnp
        details = smoke_bnp('metafiplusplus', model, 'cuda:0', PHYSICAL_BATCH)
    else:
        from defend.a1_fst_pose import smoke_fst
        details = smoke_fst('metafiplusplus', model, 'cuda:0', PHYSICAL_BATCH)
    return dict(details, defense=method, execution={'device': 'cuda:0'})


def run_wave(repo, method):
    if method not in METHODS:
        raise ValueError('Unsupported light-defense smoke method')
    context = mp.get_context('spawn')
    messages = context.Queue()
    child = context.Process(target=_worker, args=(str(repo), method, messages))
    child.start()
    child.join()
    try:
        record = messages.get(timeout=2)
    except queue.Empty:
        record = dict(ok=False, method=method,
                      error=f'Worker exited {child.exitcode} without a traceback')
    messages.close()
    messages.join_thread()
    result = dict(method=method, exitcode=child.exitcode, record=record)
    result['complete'] = child.exitcode == 0 and record.get('ok') is True
    return result


def validate_wave(wave, method):
    if wave.get('complete') is not True or wave.get('method') != method or wave.get('exitcode') != 0:
        raise ValueError(f'Physical32 CUDA smoke failed for {method}; preserve report')
    result = wave['record'].get('result', {})
    if (result.get('backbone') != 'metafiplusplus' or result.get('defense') != method
            or result.get('physical_batch') != PHYSICAL_BATCH
            or result.get('finite') is not True
            or result.get('execution', {}).get('device') != 'cuda:0'):
        raise ValueError(f'Smoke lacks exact real-model CUDA evidence for {method}')
    if method == 'ft' and (result.get('optimizer_steps', 0) < 2
                           or result.get('optimizer_state_initialized') is not True):
        raise ValueError('FT smoke did not exercise the optimizer')
    if method == 'bnp' and result.get('n_layers', 0) < 1:
        raise ValueError('BNP smoke did not inspect hidden BN layers')
    if method == 'fst' and (result.get('head_reinitialized') is not True
                           or result.get('norm_projected') is not True):
        raise ValueError('FST smoke did not reinitialize and project the coordinate head')
    if method == 'nc' and (result.get('nc_inversion_batch') != 8
                           or result.get('nc_inversion_gradient') is not True
                           or result.get('nc_unlearning_optimizer_steps') != 1):
        raise ValueError('NC smoke did not exercise inversion and unlearning')
    return True
