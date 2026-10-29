"""Actual PowerShell parser and small-file importer regression; no download/GPU."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
IMPORT = ROOT/'remote_tools/Import-PiWMetaFiAdamWBackup.ps1'
LAUNCH = ROOT/'remote_tools/Start-PiWMetaFiRemaining3090.ps1'
DOC = ROOT/'remote_tools/PIW_METAFI_FOUR_ATTACKS_3090_VI.md'
DATA = ROOT/'remote_tools/Get-PiW3DProcessedData.ps1'
pytestmark = pytest.mark.skipif(os.name != 'nt' or not shutil.which('powershell'),reason='Windows PowerShell required')


def run_ps(command):
    return subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,timeout=60)


def test_scripts_and_all_new_documented_commands_parse():
    for text in [IMPORT.read_text(),DATA.read_text(),LAUNCH.read_text(),*re.findall(r'```powershell\s*\n(.*?)```',DOC.read_text(encoding='utf-8'),re.S)]:
        literal = text.replace("'","''")
        result = run_ps("$t=$null;$e=$null;[System.Management.Automation.Language.Parser]::ParseInput('"+literal+"',[ref]$t,[ref]$e)|Out-Null;if($e.Count){$e|Out-String|Write-Output;exit 1}")
        assert result.returncode == 0,result.stdout+result.stderr


def test_four_attack_ps_flag_uses_separate_output_and_hidden_single_controller():
    source = LAUNCH.read_text()
    assert '[switch]$FourAttacksOnly' in source
    assert "if ($FourAttacksOnly) { $Common += '--four-attacks-only' }" in source
    assert 'piw_metafi_four_attacks_s42_3090_v1' in source
    assert source.count('Start-Process -FilePath') == 1
    assert '-WindowStyle Hidden' in source
    assert 'Stop-Process' not in source
    assert 'pip install' not in source


def fake_importer(tmp_path,names_extra=()):
    # Only data hashes change. Execute all production path/extraction/idempotence logic.
    source = IMPORT.read_text()
    mapping = {
        'ftrojan':'9d2a713315f694183cd9512e9feb74fe4f44bb823fedf455fafbf5edddd7d093',
        'bpp_source':'4ec9d9ff57fa578a42f368764b9c110d2376e155d6d578d5831676a5fd2a4c5f',
        'fiba':'590394ff220b1b99dcd32db6416948e40fa6b49fb4f8934f7dfd9f3bbda98389',
        'smooth_lf_csi':'f918001926d5716067d7d321360e1d51e495342cab543204f1019df4c03b6cb2'}
    names = ['download.ps1','inventory.json','README.md','sources.tar.gz',
             'provenance/placements.json','provenance/plan.json',
             'provenance/preflight_receipt.json','provenance/prelaunch_gate.json']
    names += ['metafiplusplus/seed42/'+m+'/checkpoint.pt' for m in mapping]
    archive = tmp_path/'fixture.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name in [*names,*names_extra]:
            z.writestr(name,name.encode())
    for method,sha in mapping.items():
        content = ('metafiplusplus/seed42/'+method+'/checkpoint.pt').encode()
        source = source.replace(sha,hashlib.sha256(content).hexdigest())
    source = source.replace('4607f4e3524af9f0cfb1b719ad89e6fee0e23c1765a150508604726fe720d43d',hashlib.sha256(archive.read_bytes()).hexdigest())
    home = tmp_path/'new data home'
    cmd = "$ErrorActionPreference='Stop';$s=[ScriptBlock]::Create('"+source.replace("'","''")+"');& $s -DataHome '"+str(home).replace("'","''")+"' -ZipPath '"+str(archive).replace("'","''")+"'"
    return cmd,home


def test_actual_importer_extracts_and_rechecks_without_overwrite_or_training(tmp_path):
    cmd,home = fake_importer(tmp_path)
    for _ in range(2):
        result = run_ps(cmd)
        assert result.returncode == 0,result.stdout+result.stderr
        assert 'IMPORTED_AND_VERIFIED 4/4' in result.stdout
    checkpoints = list(home.glob('imports/**/checkpoint.pt'))
    assert len(checkpoints) == 4
    assert not (home/'tools').exists()
    checkpoints[0].write_bytes(b'preserve this different file')
    result = run_ps(cmd)
    assert result.returncode != 0
    assert 'Existing import differs' in result.stderr
    assert checkpoints[0].read_bytes() == b'preserve this different file'


def test_importer_rejects_path_traversal_before_extracting(tmp_path):
    cmd,home = fake_importer(tmp_path,['../escape.txt'])
    result = run_ps(cmd)
    assert result.returncode != 0
    assert 'Unexpected/duplicate archive member' in result.stderr
    assert not home.exists()
    assert not (tmp_path/'escape.txt').exists()


def test_dataset_downloader_pins_exact_mirror_and_never_overwrites_partial_data():
    source = DATA.read_text()
    assert '1tr98LAenpHUxhV3ibq91Qkjk6cxsftik' in source
    assert 'bdd348909f1ff7d67d5674c40637c24b14d75e6bd9274cc9b3d8cfc3f2a35cf1' in source
    assert '8577781802' in source
    assert '$ExtractBytes + 5GB' in source
    assert '[IO.Compression.ZipFile]::ExtractToDirectory' in source
    assert 'Remove-Item' not in source
    assert 'Start-Process' not in source
    assert 'No overwrite/extraction' in source


def test_real_large_zip_remaining_bytes_use_int64_math_overload():
    # Exercise the exact production AST expression with the actual >8 GB ZIP,
    # not the small fixture ZIP that masked the original PowerShell overload bug.
    path = str(DATA).replace("'","''")
    command = f"""
$ErrorActionPreference='Stop'
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{path}',[ref]$null,[ref]$null)
$nodes=@($ast.FindAll({{param($n) $n -is [System.Management.Automation.Language.InvokeMemberExpressionAst] -and $n.Member.Value -eq 'Max'}},$true))
if($nodes.Count -ne 1){{throw 'Expected exactly one remaining-byte Max expression'}}
$ExpectedBytes=[long]8577781802
foreach($HaveBytes in @([long]0,[long]1,[long]1073741824,[long]6442450944,[long]8577781801,[long]8577781802,[long]8577781803)){{
    $actual=Invoke-Expression $nodes[0].Extent.Text
    $expected=if($HaveBytes -lt $ExpectedBytes){{$ExpectedBytes-$HaveBytes}}else{{[long]0}}
    if($actual -ne $expected -or $actual -isnot [long]){{throw ('Wrong Int64 byte calculation at '+$HaveBytes)}}
    if(($actual + 4GB) -lt 4GB){{throw 'Safety margin overflow'}}
}}
'INT64_LARGE_ZIP_CHECK_PASSED'
"""
    result = run_ps(command)
    assert result.returncode == 0,result.stdout+result.stderr
    assert 'INT64_LARGE_ZIP_CHECK_PASSED' in result.stdout


@pytest.mark.parametrize('complete',[True,False])
def test_existing_dataset_is_never_downloaded_or_modified(tmp_path,complete):
    root = tmp_path/'dataset'
    (root/'train_data').mkdir(parents=True)
    train = root/'train_data/train_data_list.txt';train.write_text('unchanged')
    if complete:
        (root/'test_data').mkdir()
        (root/'test_data/test_data_list.txt').write_text('test')
    cmd = "& '"+str(DATA)+"' -DataHome '"+str(tmp_path/'new home')+"' -DatasetRoot '"+str(root)+"'"
    result = run_ps(cmd)
    assert result.returncode == (0 if complete else 1),result.stdout+result.stderr
    if complete:
        assert 'DATA_LISTS_FOUND_REQUIRES_PREFLIGHT' in result.stdout
    else:
        assert 'No overwrite/extraction' in result.stderr
    assert train.read_text() == 'unchanged'
    assert not (tmp_path/'new home').exists()


@pytest.mark.parametrize('malicious',[False,True])
def test_actual_dataset_extraction_and_traversal_refusal_on_small_verified_archive(tmp_path,malicious):
    archive = tmp_path/'small dataset.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('train_data/train_data_list.txt','train item')
        z.writestr('test_data/test_data_list.txt','test item')
        if malicious:
            z.writestr('../escape.txt','must not escape')
    source = DATA.read_text().replace(
        'bdd348909f1ff7d67d5674c40637c24b14d75e6bd9274cc9b3d8cfc3f2a35cf1',
        hashlib.sha256(archive.read_bytes()).hexdigest()).replace('8577781802',str(archive.stat().st_size))
    # Fixture disk requirement is zero; production retains its 5 GiB safety margin.
    source = source.replace('$ExtractBytes + 5GB','$ExtractBytes + 0')
    root = tmp_path/'new dataset'
    cmd = "$ErrorActionPreference='Stop';$s=[ScriptBlock]::Create('"+source.replace("'","''")+"');& $s -DataHome '"+str(tmp_path/'home')+"' -DatasetRoot '"+str(root)+"' -ZipPath '"+str(archive)+"'"
    result = run_ps(cmd)
    assert result.returncode == (1 if malicious else 0),result.stdout+result.stderr
    if malicious:
        assert not root.exists()
        assert 'Unsafe archive member' in result.stderr
    else:
        assert 'DATA_EXTRACTED_REQUIRES_PREFLIGHT' in result.stdout
        assert (root/'train_data/train_data_list.txt').read_text() == 'train item'
        assert (root/'test_data/test_data_list.txt').read_text() == 'test item'
    assert not (tmp_path/'escape.txt').exists()
