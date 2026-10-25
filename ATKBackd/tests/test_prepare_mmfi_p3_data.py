"""Safety checks for the Windows MM-Fi P3 download/extraction handoff."""
import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest


HELPER_PATH = (Path(__file__).resolve().parents[2] / 'remote_tools' /
               'Prepare-MMFiP3Data.py')
SPEC = importlib.util.spec_from_file_location('prepare_mmfi_p3_data', HELPER_PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def make_archive(path, *, root='Compress'):
    with zipfile.ZipFile(path, 'w') as archive:
        for scene in ('E01', 'E02', 'E03', 'E04'):
            archive.writestr(f'{root}/{scene}/S01/A01/sample.bin', scene.encode())
    return path


def test_default_plan_does_not_download_or_write(tmp_path, monkeypatch, capsys):
    home = tmp_path / 'mmfi'
    monkeypatch.setattr(helper.subprocess, 'run',
                        lambda *a, **k: pytest.fail('plan attempted download'))
    assert helper.main(['--data-home', str(home)]) == 0
    assert not home.exists()
    assert '"train": false' in capsys.readouterr().out


def test_rejects_wrong_archive_root_before_extraction(tmp_path):
    archive = make_archive(tmp_path / 'wrong.zip', root='Other')
    home = tmp_path / 'mmfi'
    with pytest.raises(ValueError, match='Compress/'):
        helper.prepare_data(home, archive)
    assert not home.exists()


def test_verified_extract_resume_and_modified_file_refusal(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / 'data.zip')
    home = tmp_path / 'mmfi'
    monkeypatch.setattr(helper, 'validate_mmfi',
                        lambda root: {'checked_root': str(root)})

    first = helper.prepare_data(home, archive)
    sample = home / 'datasets' / 'Compress' / 'E01' / 'S01' / 'A01' / 'sample.bin'
    assert sample.read_bytes() == b'E01'
    assert first['dataset_root'] == str(home / 'datasets' / 'Compress')
    assert (home / 'dataset_receipt_p3_windows.json').is_file()
    with pytest.raises(FileExistsError, match='explicit --resume'):
        helper.prepare_data(home, archive)
    assert helper.prepare_data(home, archive, resume=True) == first

    sample.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='never overwrite'):
        helper.prepare_data(home, archive, resume=True)
    assert sample.read_bytes() == b'wrong'


@pytest.mark.parametrize('resume', [False, True])
def test_accepts_precreated_empty_datasets_directory(tmp_path, monkeypatch, resume):
    archive = make_archive(tmp_path / 'data.zip')
    home = tmp_path / 'mmfi'
    (home / 'datasets').mkdir(parents=True)
    monkeypatch.setattr(helper, 'validate_mmfi', lambda root: {'ok': True})

    receipt = helper.prepare_data(home, archive, resume=resume)
    assert receipt['dataset_root'] == str(home / 'datasets' / 'Compress')
    assert (home / 'datasets' / '.extraction.json').is_file()


def test_download_requires_flag_and_never_overwrites_existing_zip(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / 'data.zip')
    called = []
    monkeypatch.setattr(helper.subprocess, 'run',
                        lambda *a, **k: called.append((a, k)))
    with pytest.raises(FileExistsError, match='already exists'):
        helper.download_archive(archive, sys.executable)
    assert called == []
    helper.download_archive(archive, sys.executable, resume=True)
    assert called == []  # valid ZIP was kept; --continue was not run


def test_explicit_download_can_list_without_extracting(tmp_path, monkeypatch, capsys):
    home = tmp_path / 'mmfi'
    archive = home / 'downloads' / helper.ARCHIVE_NAME
    commands = []

    def fake_gdown(command, check):
        commands.append(command)
        assert check is True
        make_archive(archive)

    monkeypatch.setattr(helper.subprocess, 'run', fake_gdown)
    assert helper.main(['--data-home', str(home), '--mode', 'list',
                        '--download', '--tools-python', sys.executable]) == 0
    assert len(commands) == 1
    assert commands[0][:3] == [sys.executable, '-m', 'gdown']
    assert not (home / 'datasets').exists()
    assert 'LIST_ONLY' in capsys.readouterr().out
