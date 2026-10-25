"""Read-only, paired A1 defense reporting across MM-Fi pose backbones.

This is an exploratory A1-only comparison. It never treats a low attacker
Target-PCK as evidence of pose recovery without the triggered-to-true error and
the same model's clean error on the *same two target joints*.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defend.a1_defense_metrics import enrich_evaluation


BACKBONES = ('hpeli', 'metafiplusplus', 'wiflow')
METAFI_WINDOWS_PROFILE = 'metafi_mmfi_windows_single_gpu_defense_s42_v1'
LIGHT_WINDOWS_PROFILES = {
    'hpeli': 'hpeli_mmfi_windows_light_a1_defense_s42_v3_bnp_fst_nc',
    'metafiplusplus': 'metafiplusplus_mmfi_windows_light_a1_defense_s42_v3_bnp_fst_nc',
    'wiflow': 'wiflow_mmfi_windows_light_a1_defense_s42_v3_bnp_fst_nc',
}
LEGACY_LIGHT_WINDOWS_PROFILES = {
    backbone: profile.replace('v3_bnp_fst_nc', 'v2_nc')
    for backbone, profile in LIGHT_WINDOWS_PROFILES.items()
}
FIELDS = ('clean_mpjpe_mm', 'clean_target_mpjpe_mm',
          'triggered_gt_mpjpe_mm', 'triggered_gt_minus_clean_target_mm',
          't_mpjpe_mm', 'target_pck_pct')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _ordered_ids(arrays):
    ids = arrays.get('pair_ids')
    if ids is None or ids.ndim != 2 or ids.shape[1] != 3 or ids.dtype.kind != 'U':
        raise ValueError('Full ordered TEST pair_ids (N,3) are required')
    indices = arrays.get('sample_indices')
    if (indices is None or indices.shape != (len(ids),)
            or not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(indices, np.arange(len(ids)))):
        raise ValueError('Full ordered TEST sample_indices are required')
    h = hashlib.sha256()
    for row in ids:
        h.update(json.dumps(row.tolist(), ensure_ascii=False,
                            separators=(',', ':')).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def load_verified_record(backbone, path):
    """Check record, paired NPZ and scalar replay without touching model/CUDA."""
    if backbone not in BACKBONES:
        raise ValueError(f'Unknown backbone {backbone}')
    path = Path(path)
    record = json.loads(path.read_text(encoding='utf-8'))
    checksum = record.get('record_sha256')
    if not isinstance(checksum, str) or checksum != _digest(
            {k: v for k, v in record.items() if k != 'record_sha256'}):
        raise ValueError(f'Invalid defense record checksum: {path}')
    # The historical Windows MetaFi sidecar used `method` for the repair name.
    # Its source checkpoint SHA binds that otherwise ambiguous legacy field
    # to the immutable A1 final-49 victim. New records use method=a1/defense=...
    if record.get('profile') == METAFI_WINDOWS_PROFILE:
        if backbone != 'metafiplusplus' or record.get('defense') is not None:
            raise ValueError(f'Unexpected legacy MetaFi record identity: {path}')
        defense = record.get('method')
        source_identity = record.get('source_checkpoint_sha256')
        if not isinstance(source_identity, str) or len(source_identity) != 64:
            raise ValueError(f'Legacy MetaFi source checkpoint binding missing: {path}')
    elif record.get('profile') in (LIGHT_WINDOWS_PROFILES.get(backbone),
                                  LEGACY_LIGHT_WINDOWS_PROFILES.get(backbone)):
        allowed = (('none', 'ft', 'clp', 'nc')
                   if record['profile'] == LEGACY_LIGHT_WINDOWS_PROFILES[backbone]
                   else ('none', 'ft', 'clp', 'bnp', 'fst', 'nc'))
        if record.get('method') not in allowed:
            raise ValueError(f'Unknown light defense method: {path}')
        defense = record['method']
        source_identity = record.get('source_checkpoint_sha256')
        if record.get('status') != 'complete':
            raise ValueError(f'NC diagnostic without repair is not a defense row: {path}')
        if not isinstance(source_identity, str) or len(source_identity) != 64:
            raise ValueError(f'Light defense source checkpoint binding missing: {path}')
    else:
        if record.get('method') != 'a1':
            raise ValueError(f'Not an A1 defense record: {path}')
        defense = record.get('defense')
        source_artifacts = record.get('binding', {}).get('source_artifacts')
        source_identity = _digest(source_artifacts) if source_artifacts else None
        if source_identity is None:
            source_identity = record.get('source_checkpoint_sha256')
    if (record.get('status') != 'complete' or record.get('seed') != 42
            or not isinstance(defense, str) or not defense
            or record.get('backbone', backbone) != backbone
            or not isinstance(source_identity, str)):
        raise ValueError(f'Not a completed seed42 A1 {backbone} defense: {path}')
    if record.get('diagnostics', {}).get('full_official_test') is not True:
        raise ValueError(f'Full official TEST evaluation required: {path}')
    sample_file = path.parent / record.get('per_sample_file', 'per_sample.npz')
    expected = (record.get('artifacts', {}).get(sample_file.name)
                or record.get('per_sample_sha256'))
    if sample_file.name != 'per_sample.npz' or not isinstance(expected, str):
        raise ValueError(f'Per-sample artifact binding missing: {path}')
    if _sha(sample_file) != expected:
        raise ValueError(f'Per-sample artifact changed: {sample_file}')
    with np.load(sample_file, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if len(arrays.get('clean_error_mm', ())) != 33264:
        raise ValueError(f'Expected 33264 full TEST samples: {path}')
    ids_sha256 = _ordered_ids(arrays)
    replay = enrich_evaluation(dict(metrics=record['metrics'],
        diagnostics=record['diagnostics'], dose_rows=record['dose_rows'], arrays=arrays))
    if replay['metrics'] != record['metrics'] or replay['dose_rows'] != record['dose_rows']:
        raise ValueError(f'Defense metrics do not replay from paired arrays: {path}')
    return dict(backbone=backbone, method='a1', defense=defense, seed=42,
                metrics=record['metrics'], pair_ids_sha256=ids_sha256,
                clean_scale_mm=np.asarray(arrays['clean_scale_mm'], dtype=np.float64),
                source_identity_sha256=source_identity,
                source_path=str(path.resolve()), record_sha256=checksum,
                per_sample_sha256=expected, profile=record.get('profile'))


def build_report(records, *, backbones=BACKBONES, required_defenses=None):
    """Require paired NoDef per model; never pool frames as independent seeds."""
    backbones = tuple(backbones)
    if not backbones or len(set(backbones)) != len(backbones) or set(backbones) - set(BACKBONES):
        raise ValueError('Choose unique known backbones')
    expected = tuple(required_defenses or ())
    if len(set(expected)) != len(expected) or 'none' in expected:
        raise ValueError('Required repair names must be unique and exclude none')
    index = {}
    identity = scale = None
    model_sources = {}
    for item in records:
        backbone, defense = item['backbone'], item['defense']
        if backbone not in backbones or (backbone, defense) in index:
            raise ValueError('Out-of-scope or duplicate backbone/defense record')
        if item.get('method') != 'a1' or item.get('seed') != 42 or not defense:
            raise ValueError('Only seed42 A1 named defense records are accepted')
        source = item.get('source_identity_sha256')
        if not isinstance(source, str) or not source:
            raise ValueError(f'Missing immutable A1 source identity in {backbone}/{defense}')
        if backbone in model_sources and model_sources[backbone] != source:
            raise ValueError(f'{backbone} defenses do not share the same A1 checkpoint')
        model_sources[backbone] = source
        m = item['metrics']
        for key in FIELDS:
            value = m.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Missing/nonfinite {key} in {backbone}/{defense}')
            if key != 'triggered_gt_minus_clean_target_mm' and value < 0:
                raise ValueError(f'Negative {key} in {backbone}/{defense}')
        if not 0 <= m['target_pck_pct'] <= 100:
            raise ValueError(f'Invalid Target-PCK in {backbone}/{defense}')
        gap = m['triggered_gt_mpjpe_mm'] - m['clean_target_mpjpe_mm']
        if abs(m['triggered_gt_minus_clean_target_mm'] - gap) > .05:
            raise ValueError(f'Same-target-joint recovery gap inconsistent in {backbone}/{defense}')
        if item['pair_ids_sha256'] != (identity or item['pair_ids_sha256']):
            raise ValueError('Three models/defenses do not share ordered TEST identities')
        identity = item['pair_ids_sha256']
        current_scale = np.asarray(item['clean_scale_mm'], dtype=np.float64)
        if scale is not None and (current_scale.shape != scale.shape
                or not np.allclose(current_scale, scale, atol=.002, rtol=0)):
            raise ValueError('Clean GT PCK scales differ between records')
        scale = current_scale
        index[backbone, defense] = item
    for backbone in backbones:
        if (backbone, 'none') not in index:
            raise ValueError(f'Missing paired NoDef A1 row for {backbone}')
        for defense in expected:
            if (backbone, defense) not in index:
                raise ValueError(f'Missing required {backbone}/{defense} row')
        if not any(b == backbone and d != 'none' for b, d in index):
            raise ValueError(f'No evaluated repair for {backbone}')
    rows = []
    for backbone in backbones:
        baseline = index[backbone, 'none']['metrics']
        defenses = sorted((d for b, d in index if b == backbone and d != 'none'))
        for defense in ('none', *defenses):
            item = index[backbone, defense]
            m = item['metrics']
            paired = {key + '_delta_vs_none': m[key] - baseline[key] for key in FIELDS}
            rows.append(dict(backbone=backbone, method='a1', defense=defense,
                             seed=42, **{key: m[key] for key in FIELDS},
                             **paired, record_sha256=item['record_sha256'],
                             per_sample_sha256=item['per_sample_sha256'],
                             source_path=item['source_path'], profile=item['profile']))
    return dict(schema=1, status='EXPLORATORY_A1_ONLY_THREE_MODEL_DEFENSE',
                backbones=list(backbones), seed=42, n_seeds=1, attack_methods=['a1'],
                excluded_attack_methods=['badnets'], target_joints=[2, 3],
                target_pck_threshold=.2, scale_joints=[5, 12],
                ordered_test_ids_sha256=identity,
                source_identity_sha256=model_sources, rows=rows,
                limitations=['A1-only defense comparison; no cross-attack resistance claim.',
                    'One seed; no across-seed standard deviation or confidence interval.',
                    'Clean MPJPE covers 17 joints; clean-target and triggered-GT errors cover the same two target joints.',
                    'The signed recovery gap is triggered-GT minus same-model clean-target error; it is never clipped.',
                    'A low attacker Target-PCK alone does not establish recovery of the correct pose.',
                    'This reporter checks saved record/NPZ hashes and metric replay; each source checkpoint and target definition still require the owning runner audit.'])


def _parse_record(spec):
    if '=' not in spec:
        raise argparse.ArgumentTypeError('Use BACKBONE=PATH for each --record')
    backbone, path = spec.split('=', 1)
    if backbone not in BACKBONES or not path:
        raise argparse.ArgumentTypeError('Use hpeli/metafiplusplus/wiflow=PATH')
    return backbone, Path(path)


def markdown_report(report):
    """Show pose recovery and clean sacrifice without mixing joint scopes."""
    if report.get('status') != 'EXPLORATORY_A1_ONLY_THREE_MODEL_DEFENSE':
        raise ValueError('Only completed A1-only three-model reports can be rendered')
    lines = [
        '# Exploratory A1-only defense comparison (MM-Fi P1S1, seed 42)',
        '',
        'One seed; BadNets excluded by the experiment owner. This table does not establish cross-attack defense resistance.',
        'Errors are in mm. Clean17 is all-joint MPJPE; CleanTarget2 and TriggeredGT2 use target joints [2,3].',
        'RecoveryGap = TriggeredGT2 - CleanTarget2 of the same model. A negative gap is retained.',
        '',
        '| Backbone | Defense | Clean17 ↓ | CleanTarget2 ↓ | TriggeredGT2 ↓ | RecoveryGap | T-MPJPE ↑ for defense | Target-PCK ↓ for defense |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in report['rows']:
        lines.append('| ' + ' | '.join((row['backbone'], row['defense'],
            f"{row['clean_mpjpe_mm']:.2f}", f"{row['clean_target_mpjpe_mm']:.2f}",
            f"{row['triggered_gt_mpjpe_mm']:.2f}",
            f"{row['triggered_gt_minus_clean_target_mm']:.2f}",
            f"{row['t_mpjpe_mm']:.2f}", f"{row['target_pck_pct']:.2f}")) + ' |')
    lines.extend(['', 'Paired changes below are repair minus NoDef for the same backbone and seed. Target-PCK changes are percentage points.', '',
        '| Backbone | Defense | ΔClean17 | ΔCleanTarget2 | ΔTriggeredGT2 | ΔRecoveryGap | ΔT-MPJPE | ΔTarget-PCK pp |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for row in report['rows']:
        if row['defense'] == 'none':
            continue
        lines.append('| ' + ' | '.join((row['backbone'], row['defense'],
            f"{row['clean_mpjpe_mm_delta_vs_none']:+.2f}",
            f"{row['clean_target_mpjpe_mm_delta_vs_none']:+.2f}",
            f"{row['triggered_gt_mpjpe_mm_delta_vs_none']:+.2f}",
            f"{row['triggered_gt_minus_clean_target_mm_delta_vs_none']:+.2f}",
            f"{row['t_mpjpe_mm_delta_vs_none']:+.2f}",
            f"{row['target_pck_pct_delta_vs_none']:+.2f}")) + ' |')
    lines.extend(['', 'Interpret recovery jointly: lower TriggeredGT2 and stable CleanTarget2/Clean17 support correct-pose recovery. Lower Target-PCK alone is insufficient.', ''])
    return '\n'.join(lines)


def _write_identical_or_new(path, payload):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Existing report differs; preserve it and choose a new path: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.saving')
    if temporary.exists():
        raise ValueError(f'Unfinished output exists: {temporary}')
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='append', type=_parse_record, required=True,
                        help='Repeat BACKBONE=/absolute/path/to/evaluation.json')
    parser.add_argument('--require-defense', action='append', default=[],
                        help='Require this named repair on every backbone')
    parser.add_argument('--out', type=Path, required=True,
                        help='New JSON report path; existing output must be identical')
    parser.add_argument('--markdown-out', type=Path,
                        help='Optional Markdown table path; existing output must be identical')
    args = parser.parse_args(argv)
    if args.markdown_out and args.out.resolve() == args.markdown_out.resolve():
        raise ValueError('JSON and Markdown outputs must use different paths')
    report = build_report([load_verified_record(backbone, path)
                           for backbone, path in args.record],
                          required_defenses=args.require_defense)
    payload = json.dumps(report, indent=2, ensure_ascii=False,
                         allow_nan=False).encode('utf-8') + b'\n'
    if args.markdown_out:
        markdown = markdown_report(report).encode('utf-8')
        if args.markdown_out.exists() and args.markdown_out.read_bytes() != markdown:
            raise ValueError('Existing Markdown report differs; choose a new path')
    _write_identical_or_new(args.out, payload)
    if args.markdown_out:
        _write_identical_or_new(args.markdown_out, markdown)
    print(f'{report["status"]}: {len(report["rows"])} checksum-checked, array-replayed rows -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
