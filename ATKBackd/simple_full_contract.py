"""Fixed full-data contract for minimal Proposed hypotheses, not fitted drafts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROFILE = 'mmfi_simple_full_peak_v2'
STATUS = 'FULL_EXPLORATORY_SINGLE_SEED'
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EXPECTED_COUNTS = {'train': 133056, 'eval': 33264}
SOURCE_FILES = (
    'run_simple_full.py', 'simple_full_contract.py', 'simple_reference.py',
    'attack/simple_carrier.py', 'attack/peak_budget.py',
    'run_traditional_comparison.py', 'run_mmfi_tables.py', 'run_method_drafts.py',
    'run_peak_confirmation.py', 'train_backdoor.py', 'configs/mmfi/attack_bend.yaml',
    'attack/trigger.py', 'attack/method_drafts.py', 'attack/learned_carrier.py',
    'attack/traditional.py', 'attack/poison.py', 'attack/payload.py',
    'attack/skeleton_mmfi.py', 'data_utils/feeder.py', 'data_utils/draft_subset.py',
    'eval/distortion.py', 'eval/metrics.py', 'models/factory.py', 'models/hpeli.py',
    'models/sk_network.py', 'mmfi_tables.py', 'third_party/backdoorbench/blended.py',
    'third_party/backdoorbench/SOURCE_MANIFEST.json',
)

FIXED = dict(
    simple_full_profile=PROFILE, experiment_name='mmfi', num_person=1,
    mmfi_protocol='protocol1', mmfi_setting='s1', mmfi_random_ratio=.8,
    mmfi_split_seed=0, model='hpeli', seed=42, epochs=50, victim_epochs=50,
    optimizer='sgd', lr=.001, momentum=.9, weight_decay=0.0,
    batch_size=32, pretrained=False, data_parallel=False, lr_scheduler=False,
    loader_persistent_workers=False, strict_resume=True, victim_loss='mpjpe',
    training_protocol='ordinary_erm', attacker_access='data_only',
    threat_model='training_data_poisoning', poison_select='uniform',
    dose_coupling='paired', dose_mode='linear', dose_min=.2, dose_max=1.0,
    dose_grid=GRID, pivot=1, target_joints=[2, 3], theta_max_deg=40.0,
    payload_axis=[0.0, 0.0, 1.0], trigger_zero_mean=True,
    n_ant=3, n_sub=114, n_pkt=10, aoa_spread=.6, top_k=6,
    simple_reference_eps=.185,
)


def validate_simple_full_config(cfg):
    """Fail closed before cache acceptance or dataset construction.

    No learning/guard options, subset flags, imported checkpoints, scheduler
    changes, target-weighted victim losses or additional L2 ceilings belong here.
    The manifest additionally binds each arm to its canonical configuration.
    """
    for key, expected in FIXED.items():
        value = cfg.get(key)
        # bool is an int in Python, but is not a valid epoch/seed/budget.
        if value != expected or (isinstance(value, bool) != isinstance(expected, bool)):
            raise ValueError(f'Simple full profile requires {key}={expected!r}')
        if isinstance(expected, int) and not isinstance(expected, bool) and type(value) is not int:
            raise ValueError(f'Simple full profile requires integer {key}')
    forbidden = [k for k in cfg if k.startswith(('draft_', 'lc_', 'confirmation_',
        'comparison_peak_', 'traditional_comparison_', 'rf_'))
        or k in ('method_draft', 'lambda_target', 'lambda_nontarget', 'loss_norm',
                 'clean_label_cover_ratio', 'wanet_cover_ratio', 'simple_relative_l2')]
    if forbidden:
        raise ValueError(f'Simple full profile refuses fitting/subsets/other protocols: {sorted(forbidden)}')
    name = cfg.get('trigger')
    if name != 'simple_carrier':
        raise ValueError('Simple full peak profile trains Proposed variants only; baselines remain frozen')
    rho = cfg.get('rho')
    if isinstance(rho, bool) or rho != .1:
        raise ValueError('Simple full profile requires attack rho=.1')
    if cfg.get('eps') != .185:
        raise ValueError('Simple full profile does not tune epsilon')
    reference_run = cfg.get('simple_reference_run')
    if (not isinstance(reference_run, str) or not reference_run.strip()
            or not Path(reference_run).is_absolute()):
        raise ValueError('Simple full peak profile requires an absolute frozen reference-run path')
    from attack.simple_carrier import resolve_simple_config
    if resolve_simple_config(cfg) != cfg:
        raise ValueError('Resolve all simple carrier defaults before training')


def source_provenance():
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def validate_simple_full_bound_files(cfg, ckpt_dir):
    """Protect direct trainer/cache calls too: only run bound full recipes."""
    validate_simple_full_config(cfg)
    if ckpt_dir is None:
        raise ValueError('Simple full training requires the runner and a bound input passport')
    path = Path(ckpt_dir) / 'simple_inputs.json'
    if not path.is_file():
        raise ValueError('Simple full training requires its full input passport; use run_simple_full.py')
    record = json.loads(path.read_text(encoding='utf-8'))
    from train_backdoor import _config_fingerprint, build_trigger
    if (record.get('schema') != 1 or record.get('profile') != PROFILE
            or record.get('config_fingerprint') != _config_fingerprint(cfg)
            or record.get('scientific_source_sha256') != source_provenance()
            or record.get('action_file_sha256') != hashlib.sha256(Path(cfg['action_npy']).read_bytes()).hexdigest()
            or record.get('trigger_key_sha256') != build_trigger(cfg).fixed_key_sha256()):
        raise ValueError('Simple full input/action/source/trigger passport changed; use a NEW directory')
    for name, count in EXPECTED_COUNTS.items():
        if record.get(name, {}).get('n') != count:
            raise ValueError('Simple full passport does not describe uncapped official splits')
