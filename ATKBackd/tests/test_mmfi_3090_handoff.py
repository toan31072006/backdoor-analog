"""CPU checks for the P1S1 HPE-Li transfer used by the RTX 3090 defense."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'hpeli_3090_source', ROOT / 'remote_tools/hpeli_light_3090/portable_source.py')
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def make_zip(path, entries):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_handoff_requires_van_sha_before_extract(tmp_path):
    archive = tmp_path / 'source.zip'
    sha = make_zip(archive, {'three_tables.resolved.json': '{}'})
    target = tmp_path / 'import'
    with pytest.raises(ValueError, match='SHA256'):
        source.unpack(archive, target, '0' * 64)
    assert not target.exists()
    source.unpack(archive, target, sha)
    assert (target / 'three_tables.resolved.json').read_text() == '{}'
    with pytest.raises(FileExistsError):
        source.unpack(archive, target, sha)


def test_handoff_rejects_path_escape(tmp_path):
    archive = tmp_path / 'bad.zip'
    sha = make_zip(archive, {'three_tables.resolved.json': '{}', '../escape.txt': 'bad'})
    target = tmp_path / 'import'
    with pytest.raises(ValueError, match='Unsafe'):
        source.unpack(archive, target, sha)
    assert not target.exists()
    assert not (tmp_path / 'escape.txt').exists()


def test_import_rebases_only_frozen_root(tmp_path):
    manifest = {'outdir': r'C:\Users\Van\backdooranalog\runs\mmfi_three_tables_4090_win_fresh_v1'}
    path = manifest['outdir'] + r'\seed42\a1'
    assert source._local_path(tmp_path, path, manifest) == tmp_path / 'seed42' / 'a1'
    with pytest.raises(ValueError):
        source._local_path(tmp_path, r'C:\Users\Van\backdooranalog\runs\another\seed42\a1', manifest)
    with pytest.raises(ValueError):
        source.cell(tmp_path, 'p3_s1_a1')
