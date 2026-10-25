"""Small, data-free checks for the standalone WiFlow transfer archive."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'remote_linux/vast_seed42/archive_wiflow_for_transfer.py'
SPEC = importlib.util.spec_from_file_location('archive_wiflow_for_transfer', SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class FakeAttack:
    MANIFEST = 'vast_seed42_attack.resolved.json'

    def __init__(self, manifest, records):
        self.manifest = manifest
        self.records = records
        self.calls = []

    def load_manifest(self, path):
        assert path.is_file()
        return self.manifest

    def cell_for(self, manifest, name, device=None):
        assert manifest is self.manifest
        return {'name': name, 'device': device}

    def completed(self, cell, sources):
        assert sources == self.manifest['sources']
        self.calls.append(cell)
        return self.records[cell['name']]


def make_run(tmp_path, *, omit_last=False):
    run = tmp_path / 'run'
    (run / 'attacks').mkdir(parents=True)
    manifest = {'backbones': ['wiflow'], 'outdir': str(run / 'attacks'),
                'sources': {'frozen': 'hash'}, 'plan_sha256': 'plan'}
    (run / 'attacks' / FakeAttack.MANIFEST).write_text('{}', encoding='utf-8')
    tasks = [{'cell': 'wiflow__s42__' + key, 'preferred_device': f'cuda:{i % 2}'}
             for i, key in enumerate(archive.KEYS)]
    records = {task['cell']: {'status': 'complete', 'cell': task['cell']}
               for task in tasks}
    for key in archive.KEYS:
        cell = run / 'attacks/wiflow/seed42' / key
        cell.mkdir(parents=True)
        (cell / 'checkpoint.pt').write_bytes(('ckpt ' + key).encode())
    (run / 'attack_only_plan.json').write_text(json.dumps({
        'profile': archive.T1_PROFILE, 'attacks': manifest, 'tasks': tasks}), encoding='utf-8')
    (run / 'attack_only_prelaunch.json').write_text(json.dumps({'binding': 'input-hash'}), encoding='utf-8')
    ordered = [records[task['cell']] for task in tasks]
    if omit_last:
        ordered.pop()
    final = {'profile': archive.T1_PROFILE, 'status': archive.T1_STATUS,
             'seed': 42, 'n_seeds': 1, 'records': ordered,
             'source_binding': 'input-hash'}
    (run / 'T1_attack_results.json').write_text(json.dumps(final), encoding='utf-8')
    return run, FakeAttack(manifest, records)


def test_reject_missing_final_record(tmp_path):
    run, reader = make_run(tmp_path, omit_last=True)
    with pytest.raises(ValueError, match='differs from authenticated'):
        archive.authenticate_t1(run, reader)
    assert len(reader.calls) == 8


def test_authenticated_eight_record_run(tmp_path):
    run, reader = make_run(tmp_path)
    final, manifest = archive.authenticate_t1(run, reader)
    assert len(final['records']) == 8
    assert manifest['plan_sha256'] == 'plan'
    assert {call['device'] for call in reader.calls} == {'cuda:0', 'cuda:1'}


def test_package_full_reread_and_preserve_sources(tmp_path):
    run, _ = make_run(tmp_path)
    source = tmp_path / 'frozen'
    source.mkdir()
    (source / 'model.py').write_text('frozen source', encoding='utf-8')
    tools = tmp_path / 'tools'
    tools.mkdir()
    (tools / 'helper.py').write_text('continuation', encoding='utf-8')
    action = tmp_path / 'data_bend.npy'
    action.write_bytes(b'original motion prior')
    receipt = tmp_path / 'dataset_content.json'
    receipt.write_text('{"dataset": "MM-Fi"}', encoding='utf-8')
    groups = {'wiflow_run': run, 'frozen_v6_code': source, 'continuation_tools': tools,
              'original_action_prior': action, 'dataset_content_receipt': receipt}
    before = {str(p): archive.file_sha(p) for p in [
        action, receipt, source / 'model.py', tools / 'helper.py',
        run / 'attacks/wiflow/seed42/a1/checkpoint.pt']}
    output = tmp_path / 'output' / 'wiflow.zip'
    result = archive.package(groups, output, {'reason': 'test'}, progress=False)
    assert result['source_files_preserved']
    assert archive.verify_archive(output)['files'] == result['files']
    assert before == {name: archive.file_sha(name) for name in before}
    with zipfile.ZipFile(output) as z:
        members = set(z.namelist())
        assert 'wiflow_run/T1_attack_results.json' in members
        assert 'wiflow_run/attacks/wiflow/seed42/a1/checkpoint.pt' in members
        assert 'frozen_v6_code/model.py' in members
        assert 'continuation_tools/helper.py' in members
        assert 'original_action_prior/data_bend.npy' in members
        assert 'dataset_content_receipt/dataset_content.json' in members
        index = json.loads(z.read('_ARCHIVE_INDEX.json'))
        assert index['status'] == 'BACKUP_ONLY_WINDOWS_DEFENSE_NOT_YET_VALIDATED'
        assert all(len(item['sha256']) == 64 for item in index['files'].values())
    with pytest.raises(FileExistsError, match='Output/checksum exists'):
        archive.package(groups, output, {}, progress=False)
    damaged = tmp_path / 'damaged.zip'
    with zipfile.ZipFile(output) as original, zipfile.ZipFile(damaged, 'w') as changed:
        for name in original.namelist():
            data = original.read(name)
            if name == 'original_action_prior/data_bend.npy':
                data += b'changed'
            changed.writestr(name, data)
    with pytest.raises(ValueError, match='Archived member SHA256/size mismatch'):
        archive.verify_archive(damaged)
    output.with_suffix('.zip.sha256').write_text('wrong  wiflow.zip\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Whole ZIP SHA256 sidecar mismatch'):
        archive.verify_archive(output)
