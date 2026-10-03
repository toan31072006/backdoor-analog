"""
run_experiments.py
==================
Single-run (không sweep) backdoor experiments trên MMFI / Person-in-WiFi-3D.

Ma trận thực nghiệm
-------------------
  Victim models  : hpeli
  Paper scenario : bend                            (reference action)
  Triggers       : micro_dropper | tsba

  Mặc định paper: HPE-Li × bend × micro-Doppler × seeds {42,0,1} = 3 runs
  TSBA chỉ chạy khi truyền rõ: --triggers micro_dropper tsba

Parallel execution
------------------
  --parallel N   : spawn N worker processes trên cùng 1 GPU (shared VRAM)
  --parallel N --gpus 0 1 2   : round-robin N processes sang nhiều GPU

  Blackwell (RTX 5090 32 GB / B200 96 GB / GB200 192 GB):
    Mỗi run chiếm ~2–3 GB VRAM (batch=32, model nhỏ).
    RTX 5090 → --parallel 8
    B200/GB200 → --parallel 9 (tất cả 9 runs song song)

Sử dụng
-------
  # Sequential (debug):
  python run_experiments.py --dataset mmfi --models hpeli --scenarios bend

  # Parallel 9 runs (B200/GB200):
  python run_experiments.py --dataset mmfi --models hpeli --scenarios bend --parallel 1

  # Parallel trên nhiều GPU:
  python run_experiments.py --dataset mmfi --models hpeli --scenarios bend --parallel 3 --gpus 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import yaml

# Ensure wbackdoor root is on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── import train function ───────────────────────────────────────────────────
from train_backdoor import (train as _train_backdoor, _get_dataset_name,
                            _load_dataset, _uses_deferred_trigger, _trigger_eps,
                            _resolve_training_config,
                            _validate_training_contract,
                            _config_fingerprint)

# ── import trigger factory ─────────────────────────────────────────────────
from attack.trigger import build_trigger_by_name
from attack.poison import PoisonedDataset, collate
from attack.payload import set_skeleton_config
from models.factory import build_model

# ── Experiment matrix ──────────────────────────────────────────────────────
# Only HPE-Li is kept as the victim. MetaFi++ was dropped (its final ReLU
# clamps predicted coordinates to >= 0, but MMFi ground truth has negative
# x/y, so its clean error is systematically inflated); GraphPoseFi was
# dropped pending a topology decision (its graph prior still uses the old
# COCO MMFi adjacency, not DT-Pose's).
ALL_MODELS   = ['hpeli']
# Only the manuscript's validated reference action is advertised here. Legacy
# cross/nod YAMLs remain available for direct exploratory runs but intentionally
# fail the paper contract until their old attack-specific loss keys are migrated.
ALL_SCENARIOS = ['bend']
PAPER_SCENARIOS = ['bend']
# TSBA is a learned, sample-specific digital time-series baseline. SIG, Blended
# and WaNet remain dropped because their image-domain scales/warps are invalid
# for [0,1] CSI.
ALL_TRIGGERS  = ['micro_dropper', 'tsba']

# scenario label → config filename suffix (we keep a single ln file for bend)
_SCENARIO_FILE = {'bend': 'attack_ln', 'cross': 'attack_cross', 'nod': 'attack_nod'}

# dose_mode label → config folder per dataset
_MODEL_CONFIG_DIR = {
    # Person-in-WiFi-3D
    ('hpeli',       'pwif3d'): 'configs/hpeli',
    # MMFI
    ('hpeli',       'mmfi'):   'configs/mmfi',
}

# MMFI uses a single config folder (model specified inside yaml)
_MMFI_SCENARIO_FILE = {'bend': 'attack_bend', 'cross': 'attack_cross', 'nod': 'attack_nod'}


MM = 1000.0


def _apply_model_overrides(cfg: dict, model: str, dataset_name: str) -> dict:
    """Apply the intended hyperparameters per model and ensure MMFI uses the MMFI dataset."""
    if model == 'hpeli':
        cfg['batch_size'] = 32
        cfg['epochs'] = 50 if dataset_name == 'mmfi' else 200

    if model == 'hpeli' and dataset_name == 'mmfi':
        # DT-Pose MMFi HPELi optimization recipe.
        cfg['lr'] = 1e-3
        cfg['optimizer'] = 'sgd'
        cfg['momentum'] = 0.9
        cfg['weight_decay'] = 0.0
    elif model == 'hpeli':
        cfg['lr'] = 1e-3
        cfg['optimizer'] = 'adamw'
        cfg['weight_decay'] = 0.01

    if dataset_name == 'mmfi':
        cfg['experiment_name'] = 'mmfi'
        cfg['model'] = model
        root = cfg.get('dataset_root')
        if root:
            # Preserve the SSH-style path from the config when it is already valid.
            if not Path(root).exists():
                candidates = [
                    Path(root),
                    Path(root) / 'Compress',
                    _ROOT / '..' / 'MMFI' / 'Compress',
                    _ROOT / 'MMFI' / 'Compress',
                    _ROOT / '..' / 'MMFI',
                    _ROOT / 'MMFI',
                ]
                for cand in candidates:
                    if cand.exists():
                        cfg['dataset_root'] = str(cand)
                        break
    return cfg


# ── Skeleton figure helper ──────────────────────────────────────────────────
@torch.no_grad()
def _render_skeleton(model: torch.nn.Module, cfg: dict,
                     trigger_name: str, cell_dir: Path,
                     tag: str, trained_trigger=None) -> None:
    """
    Run one forward pass on clean + triggered data and save a 3D skeleton figure.
    """
    # Keep plotting optional for config validation/unit tests.
    from eval.vis_skeleton import save_skeleton_figure
    device       = cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    pivot        = cfg['pivot']
    dataset_name = _get_dataset_name(cfg)

    base_test = _load_dataset(cfg, 'validation' if dataset_name != 'mmfi' else 'test')
    trig      = (trained_trigger if trained_trigger is not None
                 else build_trigger_by_name(trigger_name, cfg))
    if _uses_deferred_trigger(trig):
        trig = trig.to(device)

    def _make_ds(mode, **kw):
        return PoisonedDataset(base_test, trig, mode=mode, pivot=pivot,
                               dataset=dataset_name, **kw)

    clean_loader = torch.utils.data.DataLoader(
        _make_ds('clean'), batch_size=32, collate_fn=collate, num_workers=0)

    atk_loader = torch.utils.data.DataLoader(
        _make_ds('trigger@dose', fixed_dose=1.0,
                 eps=cfg['eps'], theta_max_deg=cfg['theta_max_deg'],
                 dose_mode=cfg['dose_mode']),
        batch_size=32, collate_fn=collate, num_workers=0)

    model.eval()
    def _pred(loader):
        preds, trues, targets = [], [], []
        for b in loader:
            csi = b['csi'].to(device)
            if _uses_deferred_trigger(trig) and 'dose' in b:
                trig.eval()
                csi = trig.inject_tensor(
                    csi, b['dose'].to(device), eps=_trigger_eps(cfg, trig))
            out, _ = model(csi)
            preds.append(out.cpu().numpy())
            trues.append(b['pose'].numpy())
            if 'target' in b:
                targets.append(b['target'].numpy())
        P  = np.concatenate(preds)[:, 0]   # (N, K, 3)
        Tr = np.concatenate(trues)[:, 0]
        Tg = np.concatenate(targets)[:, 0] if targets else None
        return P, Tr, Tg

    Pc, Tc, _  = _pred(clean_loader)
    Pd, _,  Tg = _pred(atk_loader)

    if Tg is None:
        print('[vis] WARNING: attacker target is None — skeleton figure skipped')
        return

    save_skeleton_figure(
        Pc=Pc, Tc=Tc, Pd=Pd, Tg=Tg,
        pivot=pivot,
        outpath=cell_dir / 'skeleton_3d.png',
        title=tag,
        dataset=dataset_name,
    )


# ── Run one cell ────────────────────────────────────────────────────────────
def _prepare_run_config(model: str, trigger_name: str, base_cfg: dict,
                        device: str | None, epochs: int | None,
                        seed: int) -> dict:
    """Return the exact, validated configuration consumed by ``train()``."""
    cfg = copy.deepcopy(base_cfg)
    cfg['model'] = model
    cfg['seed'] = seed
    cfg['trigger'] = trigger_name
    if device:
        cfg['device'] = device
    if epochs is not None:
        cfg['epochs'] = epochs
    cfg = _resolve_training_config(cfg)
    _validate_training_contract(cfg)
    return cfg


def run_one(model: str, scenario: str, trigger_name: str,
            base_cfg: dict, device: str | None,
            epochs: int | None, outdir: Path,
            seed: int = 42) -> dict:
    """Train one backdoored model and return a result-row dict."""

    tag = f"{model}_{scenario}_{trigger_name}_mpjpe_s{seed}"
    cell_dir = outdir / tag
    cell_dir.mkdir(parents=True, exist_ok=True)

    cfg = _prepare_run_config(model, trigger_name, base_cfg,
                              device=device, epochs=epochs, seed=seed)

    print(f"\n{'='*60}", flush=True)
    print(f"  model={model}  scenario={scenario}  trigger={trigger_name}")
    print(f"  epochs={cfg['epochs']}  device={cfg.get('device','auto')}")
    print(f"{'='*60}", flush=True)

    trained_model, res = _train_backdoor(cfg, ckpt_dir=str(cell_dir))

    # Save per-run JSON
    run_json = cell_dir / 'result.json'
    run_json.write_text(json.dumps(res, indent=2), encoding='utf-8')

    # ── 3-D skeleton figure ──────────────────────────────────────────────
    # train() returns model=None on an evaluation-cache hit: the figure from the
    # first run is already in cell_dir, so there is nothing to redraw.
    if trained_model is None:
        fig = cell_dir / 'skeleton_3d.png'
        if fig.exists():
            print('[vis] cached run — keeping the existing skeleton figure')
        else:
            # The cache was written but the figure was not (e.g. the process was
            # killed in between). We no longer hold the weights, so it cannot be
            # redrawn without discarding the cache.
            print(f'[vis] WARNING: cached run and no {fig.name} on disk; delete '
                  f'{cell_dir / "eval_cache.json"} to retrain and redraw it')
    else:
        try:
            # Unwrap DataParallel if needed
            trained_trigger = getattr(trained_model, '_trained_trigger', None)
            vis_model = (trained_model.module if hasattr(trained_model, 'module')
                         else trained_model)
            _render_skeleton(vis_model, cfg, trigger_name, cell_dir, tag,
                             trained_trigger=trained_trigger)
        except Exception as e:
            print(f'[vis] WARNING: skeleton figure skipped ({e})')

    # Build flat result row (mirrors sweep.py _row format + extras)
    dr  = res['dose_response']
    asr = res['asr@ref']
    row = {
        'model':            model,
        'scenario':         scenario,
        'trigger':          trigger_name,
        'seed':             seed,
        'config_fingerprint': res.get('config_fingerprint', ''),
        'poison_plan_sha256': res.get('poison_plan_sha256', ''),
        'victim_loss':      res['victim_loss'],
        'training_contract': res['training_contract'],
        'attacker_access':  res['attacker_access'],
        'pivot':            cfg['pivot'],
        'theta_max_deg':    cfg['theta_max_deg'],
        'rho':              cfg['rho'],
        'poison_select':    res.get('poison_select', cfg.get('poison_select', 'diverse')),
        'n_poison':         res.get('n_poison', ''),
        'n_total':          res.get('n_total', ''),
        # clean accuracy
        'clean_mpjpe_mm':   round(res['clean_mpjpe'] * MM, 3),
        'clean_pampjpe_mm': round(res['clean_pampjpe'] * MM, 3),
        'clean_pck_0.5':    round(res['clean_pck@0.5'], 4),
        # attack effectiveness at max dose
        'displacement_mm':  round(res['displacement'][-1] * MM, 3),
        'tmpjpe_mm':        round(res['tmpjpe'][-1] * MM, 3),
        'tpampjpe_mm':      round(res['tpampjpe'][-1] * MM, 3),
        'clean_target_floor_mm': round(res['clean_target_floor'] * MM, 3),
        'delta_tmpjpe_mm': round(
            (res['tmpjpe'][-1] - res['clean_target_floor']) * MM, 3),
        'nontarget_mpjpe_mm': round(res['nontarget_mpjpe'][-1] * MM, 3),
        'plausibility':     round(res['plausibility'][-1], 4),
        # dose-response quality
        'spearman':         round(dr['spearman'], 4),
        'ramp_minus_step':  round(dr['ramp_minus_step'], 4),
        'schedule_mad':     round(res['schedule_shape']['mad'], 4),
        # conjunctive ASR
        'asr':              round(asr['asr'], 4),
        'frac_landed':      round(asr['frac_landed'], 4),
        'frac_preserved':   round(asr['frac_preserved'], 4),
        'frac_plausible':   round(asr.get('frac_plausible', float(asr['plausible'])), 4),
        'target_residual_ratio': round(asr['target_residual_ratio_mean'], 4),
        'target_floor_ratio': round(asr['target_floor_ratio_mean'], 4),
        'cosine_alignment': round(asr['cosine_alignment_mean'], 4),
        'effect_gain': round(asr['effect_gain_mean'], 4),
        'payload_cosine_alignment': round(
            asr['payload_cosine_alignment_mean'], 4),
        'payload_effect_gain': round(asr['payload_effect_gain_mean'], 4),
        'target_progress':   round(asr['target_progress_mean'], 4),
        'nontarget_ratio':   round(asr['nontarget_ratio_mean'], 4),
        'plausible':        asr['plausible'],
    }
    return row


# ── Worker for parallel execution ───────────────────────────────────────────
def _worker(wid: int, cells: list, base_cfgs: dict,
            device: str, epochs: int | None,
            poison_select: str | None,
            outdir: Path) -> None:
    """
    Spawn-safe worker: trains a shard of cells on `device`.

    Explicitly sets CUDA device context so multiple workers on the same GPU
    don't contend on the default context.
    """
    # Re-insert root path (needed in spawned process)
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    # Set CUDA device affinity for this worker process
    if device.startswith('cuda'):
        dev_idx = int(device.split(':')[1]) if ':' in device else 0
        torch.cuda.set_device(dev_idx)
        # Limit each worker to a fair VRAM slice via memory fraction
        # (comment out if using large models that need full VRAM)
        # torch.cuda.set_per_process_memory_fraction(0.95 / n_workers)

    rows = []
    for model, scenario, trigger, seed in cells:
        cfg = copy.deepcopy(base_cfgs[(model, scenario)])
        if poison_select is not None:
            cfg['poison_select'] = poison_select
        row = run_one(model, scenario, trigger, cfg,
                      device=device, epochs=epochs, outdir=outdir, seed=seed)
        rows.append(row)

    part = outdir / f'_part_{wid}.json'
    part.write_text(json.dumps(rows, indent=2), encoding='utf-8')


def _assert_complete_matrix(cells: list, rows: list[dict], failed=()) -> None:
    """Refuse partial or duplicated results before writing paper tables."""
    expected = Counter(cells)
    observed = Counter(
        (r.get('model'), r.get('scenario'), r.get('trigger'), r.get('seed'))
        for r in rows)
    if failed or observed != expected:
        missing = list((expected - observed).elements())
        unexpected = list((observed - expected).elements())
        raise RuntimeError(
            'parallel experiment matrix is incomplete; refusing to write '
            f'partial paper tables. failed_workers={list(failed)}, '
            f'missing={missing}, unexpected_or_duplicate={unexpected}')


# ── Save tables ─────────────────────────────────────────────────────────────
def save_tables(rows: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / 'results.json'
    json_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')

    csv_path = outdir / 'results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[run_experiments] Saved {len(rows)} rows → {csv_path}")


# ── Aggregate results (mean±std over seeds) ───────────────────────────────────
def aggregate_results(rows: list[dict], outdir: Path) -> list[dict]:
    """Group by (model, scenario, trigger) and compute mean±std over seeds.

    Metrics output in the aggregated table:
      - MPJPE↓  PA-MPJPE↓  PCK@0.5↑  (clean accuracy)
      - ASR↑  Landed↑  Preserved↑  Plausible↑  T-MPJPE↓  (attack)
      - Spearman↑  (dose-response quality)
    """
    from collections import defaultdict

    # Metrics that are rates (0–1 → printed as %)
    RATE_METRICS = ['asr', 'frac_landed', 'frac_preserved',
                    'frac_plausible', 'clean_pck_0.5']
    # Metrics already in mm
    MM_METRICS   = ['clean_mpjpe_mm', 'clean_pampjpe_mm',
                    'tmpjpe_mm', 'tpampjpe_mm', 'clean_target_floor_mm',
                    'delta_tmpjpe_mm', 'nontarget_mpjpe_mm']
    OTHER_METRICS = ['spearman', 'target_residual_ratio', 'target_floor_ratio',
                     'target_progress', 'nontarget_ratio', 'schedule_mad',
                     'cosine_alignment', 'effect_gain',
                     'payload_cosine_alignment', 'payload_effect_gain']
    ALL_METRICS   = RATE_METRICS + MM_METRICS + OTHER_METRICS

    groups: dict = defaultdict(list)
    for r in rows:
        key = (r['model'], r['scenario'], r['trigger'], r['victim_loss'],
               r['training_contract'], r['attacker_access'])
        groups[key].append(r)

    agg_rows: list[dict] = []
    for (model, scenario, trigger, victim_loss, training_contract,
         attacker_access), grp in sorted(groups.items()):
        arow: dict = {
            'model':    model,
            'scenario': scenario,
            'trigger':  trigger,
            'victim_loss': victim_loss,
            'training_contract': training_contract,
            'attacker_access': attacker_access,
            'n_seeds':  len(grp),
            'seeds':    str(sorted(g.get('seed', '') for g in grp)),
        }
        for m in ALL_METRICS:
            vals = [g[m] for g in grp if m in g]
            if not vals:
                continue
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=0))
            arow[f'{m}_mean'] = round(mean, 4)
            arow[f'{m}_std']  = round(std,  4)
            if m in RATE_METRICS:
                # Convert to % for readability
                arow[f'{m}_str'] = f'{mean * 100:.2f}±{std * 100:.2f}'
            else:
                arow[f'{m}_str'] = f'{mean:.3f}±{std:.3f}'
        agg_rows.append(arow)

    if not agg_rows:
        return agg_rows

    # Save aggregated CSV (flatten list columns)
    flat_rows = [{k: v for k, v in a.items() if not isinstance(v, list)}
                 for a in agg_rows]
    agg_path = outdir / 'results_agg.csv'
    with open(agg_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    # Pretty-print aggregated table
    n_seeds_label = agg_rows[0]['n_seeds']
    W = 155
    print(f'\n[aggregate] {len(agg_rows)} groups'
          f' (mean±std, N={n_seeds_label} seed(s)) → {agg_path}')
    print('\n' + '=' * W)
    print(
        f"{'Model':<18} {'Scenario':<8} {'N':>2}  "
        f"{'MPJPE↓ (mm)':>14} {'PA-MPJPE↓':>11} {'PCK@0.5↑ (%)':>14}  "
        f"{'ASR↑ (%)':>11} {'Landed↑':>10} {'Preserved↑':>12} "
        f"{'Plausible↑':>12} {'T-MPJPE↓ (mm)':>15}"
    )
    print('-' * W)
    for ar in agg_rows:
        print(
            f"{ar['model']:<18} {ar['scenario']:<8} {ar['n_seeds']:>2}  "
            f"{ar.get('clean_mpjpe_mm_str', '—'):>14} "
            f"{ar.get('clean_pampjpe_mm_str', '—'):>11} "
            f"{ar.get('clean_pck_0.5_str', '—'):>14}  "
            f"{ar.get('asr_str', '—'):>11} "
            f"{ar.get('frac_landed_str', '—'):>10} "
            f"{ar.get('frac_preserved_str', '—'):>12} "
            f"{ar.get('frac_plausible_str', '—'):>12} "
            f"{ar.get('tmpjpe_mm_str', '—'):>15}"
        )
    print('=' * W)

    return agg_rows


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description='Run MMFi backdoor experiments across models, scenarios, and triggers')
    ap.add_argument('--models',    nargs='+', default=['hpeli'],
                    choices=ALL_MODELS)
    ap.add_argument('--scenarios', nargs='+', default=PAPER_SCENARIOS,
                    choices=ALL_SCENARIOS)
    ap.add_argument('--triggers',  nargs='+', default=['micro_dropper'],
                    choices=ALL_TRIGGERS)
    ap.add_argument('--dataset',   default='mmfi', choices=['pwif3d', 'mmfi'],
                    help='Dataset to run experiments on (default: mmfi)')
    ap.add_argument('--epochs',    type=int, default=None)
    ap.add_argument('--poison-select', default=None, choices=['uniform', 'diverse'])
    ap.add_argument('--dataset-root', default=None,
                    help='Override dataset_root from the source YAML. Useful for '
                         'portable/server runs; quote Windows paths containing spaces.')
    ap.add_argument('--action-npy', default=None,
                    help='Override the micro-Doppler action skeleton path.')
    ap.add_argument('--rho', type=float, default=None,
                    help='Override poison rate. Use 0 for a paired clean control.')
    ap.add_argument('--eps', type=float, default=None,
                    help='Override fixed-trigger epsilon.')
    ap.add_argument('--pivot', type=int, default=None,
                    help='Override payload pivot joint.')
    ap.add_argument('--theta-max-deg', type=float, default=None,
                    help='Override maximum payload rotation in degrees.')
    ap.add_argument('--dose-mode', default=None,
                    choices=['linear', 'sqrt', 'quad'],
                    help='Override the dose-to-angle schedule.')
    ap.add_argument('--mmfi-setting', default=None, choices=['s1', 's3'],
                    help='Override MM-Fi setting (s1 main, s3 cross-environment).')
    ap.add_argument('--num-workers', type=int, default=None,
                    help='Override DataLoader workers. Start with 0 on Windows.')
    ap.add_argument('--device',    default=None,
                    help='cuda:0 | cpu (default: auto-detect)')
    ap.add_argument('--parallel',  type=int, default=1,
                    help='Number of concurrent runs on the SAME device. '
                         'RTX Pro 6000 (96 GB) → try 4 or 6. Default=1 (sequential).')
    ap.add_argument('--outdir',    default='experiments_out')
    ap.add_argument('--seeds',     type=int, nargs='+', default=[42, 0, 1],
                    help='Random seeds to run (one run per seed) for mean±std. '
                         'Default: 42 0 1, matching the manuscript.')
    ap.add_argument('--gpus', type=int, nargs='+', default=None,
                    help='[legacy] GPU ids for multi-GPU round-robin')
    ap.add_argument('--dry-run', action='store_true',
                    help='Resolve and validate every cell, write the experiment '
                         'matrix, then exit without loading data or using a GPU.')
    a = ap.parse_args()

    if a.rho is not None and not 0.0 <= a.rho <= 1.0:
        ap.error('--rho must be in [0, 1]')
    if a.eps is not None and a.eps < 0.0:
        ap.error('--eps must be non-negative')
    if a.pivot is not None:
        max_pivot = 16 if a.dataset == 'mmfi' else 13
        if not 0 <= a.pivot <= max_pivot:
            ap.error(f'--pivot must be in [0, {max_pivot}] for {a.dataset}')
    if a.theta_max_deg is not None and a.theta_max_deg < 0.0:
        ap.error('--theta-max-deg must be non-negative')
    if a.num_workers is not None and a.num_workers < 0:
        ap.error('--num-workers must be non-negative')

    # Resolve device
    if a.device:
        device = a.device
    elif a.gpus:
        device = f'cuda:{a.gpus[0]}'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    outdir = _ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load base configs ──────────────────────────────────────────────────
    base_cfgs: dict[tuple, dict] = {}
    for model in a.models:
        for scenario in a.scenarios:
            if a.dataset == 'mmfi':
                cfg_dir  = _ROOT / 'configs/mmfi'
                fname    = _MMFI_SCENARIO_FILE[scenario] + '.yaml'
            else:
                cfg_dir  = _ROOT / _MODEL_CONFIG_DIR[(model, 'pwif3d')]
                fname    = _SCENARIO_FILE[scenario] + '.yaml'

            cfg_path = cfg_dir / fname
            if not cfg_path.exists():
                raise FileNotFoundError(f"Config not found: {cfg_path}")
            cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

            # For MMFI, override model in config per iteration
            if a.dataset == 'mmfi':
                cfg = _apply_model_overrides(cfg, model, 'mmfi')
            else:
                cfg = _apply_model_overrides(cfg, model, 'person-in-wifi-3d')

            if a.poison_select is not None:
                cfg['poison_select'] = a.poison_select
            if a.dataset_root is not None:
                cfg['dataset_root'] = str(Path(a.dataset_root).expanduser())
            if a.action_npy is not None:
                cfg['action_npy'] = str(Path(a.action_npy).expanduser())
            if a.rho is not None:
                cfg['rho'] = float(a.rho)
            if a.eps is not None:
                cfg['eps'] = float(a.eps)
            if a.pivot is not None:
                cfg['pivot'] = int(a.pivot)
            if a.theta_max_deg is not None:
                cfg['theta_max_deg'] = float(a.theta_max_deg)
            if a.dose_mode is not None:
                cfg['dose_mode'] = a.dose_mode
            if a.mmfi_setting is not None:
                if a.dataset != 'mmfi':
                    ap.error('--mmfi-setting is only valid with --dataset mmfi')
                cfg['mmfi_setting'] = a.mmfi_setting
            if a.num_workers is not None:
                cfg['num_workers'] = int(a.num_workers)
            base_cfgs[(model, scenario)] = cfg

    # action_npy is now scenario-specific inside each config file
    # (configs/mmfi/attack_{bend,cross,nod}.yaml), so no runtime swap needed.

    # ── Build cells ────────────────────────────────────────────────────────
    # TSBA does not consume the action skeleton, so bend/cross/nod would be
    # byte-for-byte duplicate learned-trigger runs (same pose payload and seed).
    # Keep one bend-labelled reference instead of wasting 3x compute.
    if 'tsba' in a.triggers and 'bend' not in a.scenarios:
        raise ValueError('TSBA is action-skeleton independent; include scenario '
                         "'bend' for its single reference run")
    cells = [
        (model, scenario, trigger, seed)
        for model    in a.models
        for scenario in a.scenarios
        for trigger  in a.triggers
        for seed     in a.seeds
        if trigger != 'tsba' or scenario == 'bend'
    ]

    if a.dry_run:
        matrix = []
        for model, scenario, trigger, seed in cells:
            cfg = _prepare_run_config(
                model, trigger, base_cfgs[(model, scenario)],
                device=device, epochs=a.epochs, seed=seed)
            matrix.append({
                'model': model,
                'scenario': scenario,
                'trigger': trigger,
                'seed': seed,
                'config_fingerprint': _config_fingerprint(cfg),
                'config': cfg,
            })
        matrix_path = outdir / 'experiment_matrix.resolved.json'
        matrix_path.write_text(
            json.dumps(matrix, indent=2, default=str), encoding='utf-8')
        print(f'[dry-run] validated {len(matrix)} cell(s) -> {matrix_path}')
        for cell in matrix:
            cfg = cell['config']
            print(
                f"  {cell['model']}/{cell['scenario']}/{cell['trigger']} "
                f"seed={cell['seed']} loss={cfg['victim_loss']} "
                f"opt={cfg['optimizer']} rho={cfg['rho']} "
                f"fp={cell['config_fingerprint'][:12]}")
        return

    n_workers = min(a.parallel, len(cells))
    print(f"[run_experiments] {len(cells)} runs | device={device} | workers={n_workers}")

    # ── VRAM estimate (informational, no hard block) ───────────────────────
    if n_workers > 1 and torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # Memory per run: ~2 GB HPELi (batch=32)
        model_vram = {'hpeli': 2.0}
        # Use the heaviest model in the cell list for a conservative estimate
        models_in_run = list({m for m, _, _, _ in cells})
        est_per_run = max(model_vram.get(m, 3.0) for m in models_in_run)
        est_total   = est_per_run * n_workers
        n_gpus_used  = len(set(
            f"cuda:{a.gpus[i % len(a.gpus)]}" if a.gpus else device
            for i in range(n_workers)
        ))
        print(f"[run_experiments] GPU{'s' if n_gpus_used>1 else ''}: "
              f"{total_vram:.0f} GB × {n_gpus_used} | "
              f"~{est_per_run:.1f} GB/run × {n_workers} workers = "
              f"~{est_total:.0f} GB estimated")
        vram_available = total_vram * n_gpus_used
        if est_total > vram_available * 0.90:
            print(f"[run_experiments] WARNING: estimated {est_total:.0f} GB > "
                  f"90% of {vram_available:.0f} GB available — consider reducing --parallel")
        else:
            headroom = vram_available - est_total
            print(f"[run_experiments] VRAM headroom: ~{headroom:.0f} GB ✓")

    # ── Execute ────────────────────────────────────────────────────────────
    if n_workers <= 1:
        # Sequential — no subprocess overhead, easier debugging
        rows = []
        for model, scenario, trigger, seed in cells:
            cfg = copy.deepcopy(base_cfgs[(model, scenario)])
            if a.poison_select is not None:
                cfg['poison_select'] = a.poison_select
            row = run_one(model, scenario, trigger, cfg,
                          device=device, epochs=a.epochs,
                          outdir=outdir, seed=seed)
            rows.append(row)
    else:
        # ── Parallel: spawn N processes ───────────────────────────────────
        # spawn must be set before any CUDA context is created.
        # On Linux this is the default for PyTorch; on Windows it is required.
        import torch.multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # already set — fine

        # Assign GPU per worker (round-robin over --gpus list)
        gpu_list = a.gpus if a.gpus else [int(device.split(':')[1]) if ':' in device else 0]

        # Distribute cells round-robin across workers
        shards = [cells[i::n_workers] for i in range(n_workers)]

        procs = []
        for wid, shard in enumerate(shards):
            if not shard:
                continue
            worker_gpu  = gpu_list[wid % len(gpu_list)]
            worker_dev  = f'cuda:{worker_gpu}'
            p = mp.Process(
                target=_worker,
                args=(wid, shard, base_cfgs, worker_dev,
                      a.epochs, a.poison_select, outdir),
                daemon=False,
            )
            p.start()
            procs.append((wid, p))
            print(f"[run_experiments] Worker {wid} → {worker_dev} "
                  f"(pid={p.pid}) — {len(shard)} runs: "
                  f"{[f'{m}_{s}_{t}_s{sd}' for m,s,t,sd in shard]}")

        # Wait and collect — with timeout to avoid infinite hang
        WORKER_TIMEOUT = 60 * 60 * 6   # 6 hours max per worker
        failed = []
        for wid, p in procs:
            p.join(timeout=WORKER_TIMEOUT)
            if p.is_alive():
                print(f"[run_experiments] TIMEOUT: worker {wid} exceeded {WORKER_TIMEOUT//3600}h, killing...", flush=True)
                p.terminate()
                p.join(timeout=30)
                if p.is_alive():
                    p.kill()
                failed.append(wid)
            elif p.exitcode != 0:
                print(f"[run_experiments] ERROR: worker {wid} exited with code {p.exitcode}", flush=True)
                failed.append(wid)

        rows = []
        for wid, _ in procs:
            part = outdir / f'_part_{wid}.json'
            if part.exists():
                rows.extend(json.loads(part.read_text(encoding='utf-8')))
                part.unlink()

        _assert_complete_matrix(cells, rows, failed)

    # ── Save ───────────────────────────────────────────────────────────────
    save_tables(rows, outdir)
    aggregate_results(rows, outdir)

    # Per-seed detail table
    W2 = 120
    print("\n" + "═" * W2)
    print(
        f"{'Model':<18} {'Scenario':<8} {'Seed':>4}  "
        f"{'MPJPE↓':>8} {'PA-MPJPE↓':>10} {'PCK@0.5↑':>9}  "
        f"{'ASR↑':>7} {'Landed↑':>8} {'Preserved↑':>11} "
        f"{'Plausible↑':>11} {'T-MPJPE↓':>9}"
    )
    print("─" * W2)
    for r in sorted(rows,
                    key=lambda x: (x['model'], x['scenario'],
                                   x['trigger'], x.get('seed', 0))):
        print(
            f"{r['model']:<18} {r['scenario']:<8} {r.get('seed', 0):>4}  "
            f"{r['clean_mpjpe_mm']:>8.1f} {r['clean_pampjpe_mm']:>10.1f} "
            f"{r['clean_pck_0.5'] * 100:>9.2f}  "
            f"{r['asr'] * 100:>7.2f} {r['frac_landed'] * 100:>8.2f} "
            f"{r['frac_preserved'] * 100:>11.2f} "
            f"{r['frac_plausible'] * 100:>11.2f} "
            f"{r['tmpjpe_mm']:>9.1f}"
        )
    print("═" * W2)
    print(f"\nPer-seed results → {outdir}/results.csv")
    print(f"Aggregated mean±std → {outdir}/results_agg.csv")


if __name__ == '__main__':
    main()
