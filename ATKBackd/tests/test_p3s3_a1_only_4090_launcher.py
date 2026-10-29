"""CPU/PowerShell checks for the one-cell historical A1 handoff; no GPU run."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO/'remote_tools/Start-MMFiP3S3A1Only4090.ps1'
DOC = SCRIPT.with_name('MMFI_P3S3_A1_ONLY_4090_VI.md')
sys.path.insert(0,str(REPO/'ATKBackd'))


def test_original_a1_scientific_configuration_is_unchanged(tmp_path):
    import run_mmfi_p3_splits as p3
    plan = p3.build_plan(tmp_path/'data',tmp_path/'out','cuda:0',0)
    cells = [c for c in plan['cells'] if c['id'] == 'p3_s3_a1']
    assert len(cells) == 1
    cfg = cells[0]['cfg']
    expected = dict(seed=42,epochs=50,batch_size=32,lr=.001,momentum=.9,
        weight_decay=0.,rho=.1,mmfi_protocol='protocol3',mmfi_setting='s3',
        mmfi_split_seed=0,trigger='md_multicarrier',pivot=1,theta_max_deg=40.,
        payload_axis=[0.,0.,1.],target_joints=[2,3],dose_coupling='paired')
    for k,v in expected.items():
        assert cfg[k] == v,k
    assert not (tmp_path/'out').exists()


def test_launch_selects_one_original_cell_not_six_cell_controller():
    source = SCRIPT.read_text(encoding='utf-8')
    assert "@('--mode','launch','--_cell','p3_s3_a1')" in source
    assert "'--jobs-per-gpu','1'" in source
    assert source.count('Start-Process -FilePath') == 1
    assert 'Invoke-Legacy \'launch\'' not in source
    assert "Invoke-Legacy 'preflight'" in source
    assert "Invoke-Legacy 'gpu-smoke'" in source
    assert source.index("if ($Saved -ne $Current)") < source.index('Start-Process -FilePath')
    assert '-WindowStyle Hidden' in source
    assert 'Stop-Process' not in source
    assert 'Register-ScheduledTask' not in source


def test_scope_host_and_recipe_are_bound():
    source = SCRIPT.read_text(encoding='utf-8')
    for token in ('RTX 4090','physical_batch -ne 32','jobs_per_gpu -ne 1',
                  'wrapper_sha256','runner_sha256','host_binding','python_sha256',
                  'cfg.seed -ne 42','cfg.epochs -ne 50','cfg.rho -ne 0.1',
                  'cfg.lr -ne 0.001'):
        assert token in source


@pytest.mark.skipif(os.name != 'nt' or not shutil.which('powershell'),reason='Windows PowerShell required')
def test_script_and_all_documented_commands_parse():
    for content in (SCRIPT.read_text(encoding='utf-8'),
                    *re.findall(r'```powershell\s*\n(.*?)```',DOC.read_text(encoding='utf-8'),re.S)):
        # Send text as a PowerShell literal to the syntax parser, not an executor.
        literal = content.replace("'","''")
        command = "$t=$null;$e=$null;[System.Management.Automation.Language.Parser]::ParseInput('"+literal+"',[ref]$t,[ref]$e)|Out-Null;if($e.Count){$e|Out-String|Write-Output;exit 1}"
        result = subprocess.run(['powershell','-NoProfile','-Command',command],
                                capture_output=True,text=True,timeout=30)
        assert result.returncode == 0,result.stdout+result.stderr


@pytest.mark.skipif(os.name != 'nt' or not shutil.which('powershell'),reason='Windows PowerShell required')
@pytest.mark.parametrize('mode',['Inspect','Status'])
def test_read_only_modes_never_create_outputs(tmp_path,mode):
    home = tmp_path/'absent-data-home'
    result = subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass',
        '-File',str(SCRIPT),'-Mode',mode,'-DataHome',str(home),'-PythonExe',sys.executable],
        capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stdout+result.stderr
    assert 'only A1 / HPE-Li / MM-Fi P3S3 / seed42' in result.stdout
    assert not home.exists()


@pytest.mark.skipif(os.name != 'nt' or not shutil.which('powershell'),reason='Windows PowerShell required')
def test_native_python_runtime_probe_quotes_are_preserved():
    source = SCRIPT.read_text(encoding='utf-8')
    expression = next(line.strip() for line in source.splitlines() if '$Runtime = & $PythonExe -c ' in line)
    command = "$PythonExe='"+sys.executable.replace("'","''")+"';"+expression+';$Runtime'
    result = subprocess.run(['powershell','-NoProfile','-Command',command],
                            capture_output=True,text=True,timeout=60)
    combined = result.stdout+result.stderr
    assert 'SyntaxError' not in combined,combined
    assert 'NameError' not in combined,combined
    assert ('CUDA unavailable' in combined) or ('actual_gpu' in combined),combined
