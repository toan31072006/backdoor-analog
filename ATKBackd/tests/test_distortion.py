"""Regression tests for eval/distortion.py.

These exist because the first version of this tool measured TSBA with a freshly
initialised generator and reported the resulting ratio as if it characterised
the trained attack. It does not: a learned trigger's perturbation is entirely a
property of its trained weights. Tests 4-7 are the guard against that.

Run:  python tests/test_distortion.py     (no pytest needed)
      pytest tests/test_distortion.py     (also works)
"""
import os
import sys

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.distortion import (                                    # noqa: E402
    distortion_stats, _aggregate, load_trigger_for_measurement)
from train_backdoor import (build_trigger, _config_fingerprint,
                            _resolve_training_config)             # noqa: E402

def _tsba_cfg():
    return {
        'experiment_name': 'mmfi',
        'model': 'hpeli',
        'trigger': 'tsba',
        'tsba_hidden': 8,
        'tsba_eps': 0.1,
        'victim_loss': 'mpjpe',
        'pivot': 1,
        'theta_max_deg': 40.0,
        'rho': 0.4,
        'eps': 0.185,
        'dose_mode': 'linear',
        'seed': 42,
        'epochs': 50,
    }


def _fixed_cfg(action_npy):
    cfg = _tsba_cfg()
    cfg.update({
        'trigger': 'micro_dropper',
        'action_npy': str(action_npy),
        'top_k': 4,
        'trigger_zero_mean': True,
    })
    return cfg


@pytest.fixture
def action_npy(tmp_path):
    """A self-contained moving skeleton; no developer-specific path required."""
    rng = np.random.default_rng(123)
    skeleton = rng.normal(size=(2, 3, 50, 25, 1)).astype(np.float32)
    path = tmp_path / 'reference_action.npy'
    np.save(path, skeleton)
    return path


# ── 1. identical inputs give infinite SNR, not a large finite number ─────────
def test_zero_error_is_infinite_snr():
    x = np.random.RandomState(0).rand(3, 20, 10)
    s = distortion_stats(x, x)
    assert s['snr_db'] == float('inf'), f'expected inf, got {s["snr_db"]}'
    assert s['relative_l2'] == 0.0
    assert s['max_abs'] == 0.0


# ── 2. a known gain has the textbook relative_l2 and SNR ────────────────────
def test_known_gain_matches_theory():
    x = np.random.RandomState(0).rand(3, 20, 10)
    s = distortion_stats(x, x * 1.1)
    assert abs(s['relative_l2'] - 0.1) < 1e-9, s['relative_l2']
    assert abs(s['snr_db'] - 20.0) < 1e-6, s['snr_db']


# ── 3. dataset stats pool energy; max is a true max ─────────────────────────
def test_aggregate_pools_energy_not_ratios():
    rs = np.random.RandomState(0)
    big, small = rs.rand(100) * 10.0, rs.rand(100) * 0.1
    r1 = distortion_stats(big, big + 1.0)
    r2 = distortion_stats(small, small + 1.0)      # same error, tiny signal
    agg = _aggregate([r1, r2])

    mean_ratio = agg['mean_per_sample_relative_l2']
    assert agg['relative_l2'] < mean_ratio, (
        'pooled relative_l2 must not equal the mean of per-sample ratios; '
        f'pooled={agg["relative_l2"]} mean={mean_ratio}')
    assert abs(agg['max_abs'] - max(r1['max_abs'], r2['max_abs'])) < 1e-12, \
        'max_abs must be a true dataset maximum, not a mean of per-sample maxima'


# ── 4. a learned trigger is refused without its trained weights ─────────────
def test_learned_trigger_requires_checkpoint():
    cfg = _tsba_cfg()
    try:
        load_trigger_for_measurement(cfg)
    except SystemExit:
        return
    raise AssertionError('TSBA must not be measurable without --checkpoint')


# ── 5. the escape hatch works, and is explicit ──────────────────────────────
def test_allow_untrained_escape_hatch():
    cfg = _tsba_cfg()
    trig, epoch = load_trigger_for_measurement(cfg, allow_untrained=True)
    assert trig is not None and epoch is None


# ── 6. a fixed trigger needs no checkpoint ──────────────────────────────────
def test_fixed_trigger_needs_no_checkpoint(action_npy):
    trig, epoch = load_trigger_for_measurement(_fixed_cfg(action_npy))
    assert trig is not None and epoch is None


# ── 7. checkpoint loading: right weights, wrong config refused ──────────────
def test_checkpoint_load_and_fingerprint_guard(tmp_path):
    cfg = _tsba_cfg()
    resolved = _resolve_training_config(cfg)
    trained = build_trigger(resolved)
    with torch.no_grad():                           # pretend it was trained
        for parameter in trained.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'epoch': 49, 'model': {}, 'optimizer': {}, 'cfg': resolved,
                'cfg_fingerprint': _config_fingerprint(resolved),
                'trigger': trained.state_dict(),
                'trigger_optimizer': {}}, checkpoint)

    got, epoch = load_trigger_for_measurement(cfg, checkpoint=checkpoint)
    assert epoch == 49
    for expected, observed in zip(
            trained.state_dict().values(), got.state_dict().values()):
        assert torch.allclose(expected, observed), \
            'trained generator weights were not loaded'

    with pytest.raises(SystemExit):                 # config mismatch must refuse
        load_trigger_for_measurement(
            dict(cfg, pivot=6), checkpoint=checkpoint)

    fixed_checkpoint = tmp_path / 'fixed-trigger.pt'
    torch.save({'epoch': 10, 'model': {}, 'optimizer': {}, 'cfg': resolved,
                'cfg_fingerprint': _config_fingerprint(resolved)}, fixed_checkpoint)
    with pytest.raises(SystemExit):
        load_trigger_for_measurement(cfg, checkpoint=fixed_checkpoint)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e}')
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f'ERROR {t.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
