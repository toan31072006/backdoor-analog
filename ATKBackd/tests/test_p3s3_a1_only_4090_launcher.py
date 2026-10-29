"""CPU/PowerShell checks for the one-cell historical A1 handoff; no GPU run."""
from pathlib import Path
import base64
import copy
import json
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
    assert '[switch]$AllowConcurrentSIG' in source
    assert 'maximum_sig_workers' in source
    assert 'mmfi_p3s3_hpeli_a1_s42_4090_with_sig_v1' in source


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
def test_authorized_sig_peer_policy_rejects_every_other_training_job():
    home = r'C:\Users\Van\backdooranalog'
    runner = r'C:\Users\Van\code\ATKBackd\run_mmfi_p3_s3_three_attacks_4090.py'
    output = home+r'\runs\mmfi_p3s3_three_attacks_s42_4090_v1'
    args = ['python.exe','-u',runner,'--data-home',home,'--outdir',output,
            '--device','cuda:0','--jobs-per-gpu','1','--mode','cell',
            '--work','train','--method','sig_source']
    plain = ' '.join(args)
    valid = [plain, ' '.join('"'+a+'"' for a in args)]
    invalid = [plain.replace('--method sig_source','--method '+m)
               for m in ('input_aware','badnets','sig_source_extra')]
    invalid += [plain.replace('--mode cell','--mode launch'),
        plain.replace('--work train','--work smoke'),
        plain.replace('--device cuda:0','--device cuda:1'),
        plain.replace('--jobs-per-gpu 1','--jobs-per-gpu 2'),
        plain.replace(output,output+'_other'),plain.replace(runner,runner.replace('code','other-code')),
        plain+' --method input_aware',plain.replace('--outdir '+output,'')]
    payload = base64.b64encode(json.dumps(dict(home=home,runner=runner,out=output,
        valid=valid,invalid=invalid)).encode()).decode()
    ps_path = str(SCRIPT).replace("'","''")
    command = f"""
$d=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload}'))|ConvertFrom-Json
$DataHome=$d.home;$SIGRunner=$d.runner;$SIGOutdir=$d.out
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{ps_path}',[ref]$null,[ref]$null)
$nodes=$ast.FindAll({{param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -in @('Test-ExactArgument','Test-AuthorizedSIG','Test-AnyDeskDesktopPath','Assert-GpuPolicy')}},$true)
foreach($n in $nodes){{Invoke-Expression $n.Extent.Text}}
foreach($line in $d.valid){{if(!(Test-AuthorizedSIG ([pscustomobject]@{{Name='python.exe';CommandLine=$line}}))){{throw 'Valid SIG rejected'}}}}
foreach($line in $d.invalid){{if(Test-AuthorizedSIG ([pscustomobject]@{{Name='python.exe';CommandLine=$line}})){{throw ('Wrong job accepted: '+$line)}}}}
$script:FakeRows=@();$script:FakeProcesses=@()
function nvidia-smi {{ $global:LASTEXITCODE=0; return $script:FakeRows }}
function Get-CimInstance {{param($ClassName,$Filter) $idValue=[int]($Filter -replace 'ProcessId=',''); return $script:FakeProcesses|Where-Object {{$_.ProcessId -eq $idValue}} }}
$AllowConcurrentSIG=$true
$script:FakeRows=@('14504, C:\\env\\python.exe')
$script:FakeProcesses=@([pscustomobject]@{{ProcessId=14504;Name='python.exe';CommandLine=$d.valid[0]}})
Assert-GpuPolicy
$AllowConcurrentSIG=$false;$blocked=$false
try{{Assert-GpuPolicy}}catch{{$blocked=$true}};if(!$blocked){{throw 'Default exclusive guard weakened'}}
$AllowConcurrentSIG=$true;$script:FakeProcesses[0].CommandLine=$d.invalid[0];$blocked=$false
try{{Assert-GpuPolicy}}catch{{$blocked=$true}};if(!$blocked){{throw 'Input-Aware was allowed'}}
$script:FakeRows=@('14504, C:\\env\\python.exe','40232, C:\\env\\python.exe')
$script:FakeProcesses=@([pscustomobject]@{{ProcessId=14504;Name='python.exe';CommandLine=$d.valid[0]}},[pscustomobject]@{{ProcessId=40232;Name='python.exe';CommandLine=$d.valid[0]}})
$blocked=$false;try{{Assert-GpuPolicy}}catch{{$blocked=$true}};if(!$blocked){{throw 'Two SIG workers were allowed'}}
$script:FakeRows=@('12724, C:\\Windows\\explorer.exe');Assert-GpuPolicy
$AllowConcurrentSIG=$false
$script:FakeRows=@('30540, '+(Join-Path ${{env:ProgramFiles(x86)}} 'AnyDesk\\AnyDesk.exe'));Assert-GpuPolicy
$script:FakeRows=@('30540, C:\\Temp\\AnyDesk.exe');$script:FakeProcesses=@();$blocked=$false
try{{Assert-GpuPolicy}}catch{{$blocked=$true}};if(!$blocked){{throw 'Arbitrary AnyDesk basename was allowed'}}
'SIG_POLICY_TESTS_PASSED'
"""
    result = subprocess.run(['powershell','-NoProfile','-Command',command],
                            capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stdout+result.stderr
    assert 'SIG_POLICY_TESTS_PASSED' in result.stdout


@pytest.mark.skipif(os.name != 'nt' or not shutil.which('powershell'),reason='Windows PowerShell required')
def test_desktop_gate_refresh_allows_only_pinned_wrapper_hash_change():
    old = dict(scope='a1_only_hpeli_mmfi_p3s3_s42_4090_v1',cell='p3_s3_a1',
        wrapper_sha256='9e6953da833683cc0968ddf3a76c3a28bfc777b589570103353bc8f1a5caf565',
        runner_sha256='unchanged-runner',data_home='same-home',outdir='same-output',
        files={'plan':'same-plan','preflight':'same-input-audit','gpu_smoke':'same-smoke'},
        execution_policy={'allow_concurrent_sig':True,'maximum_sig_workers':1},
        host_binding={'gpu':'same-gpu-uuid','runtime':'same-packages','python_sha256':'same-python'})
    current = copy.deepcopy(old)
    current['wrapper_sha256'] = 'new-wrapper'
    invalid = []
    for key in ('scope','cell','runner_sha256','data_home','outdir'):
        changed = copy.deepcopy(current);changed[key] = 'tampered';invalid.append([old,changed])
    for section,key in [('files','plan'),('files','preflight'),('files','gpu_smoke'),
                        ('execution_policy','allow_concurrent_sig'),
                        ('execution_policy','maximum_sig_workers'),
                        ('host_binding','gpu'),('host_binding','runtime'),('host_binding','python_sha256')]:
        changed = copy.deepcopy(current);changed[section][key] = 'tampered';invalid.append([old,changed])
    unknown = copy.deepcopy(old);unknown['wrapper_sha256'] = 'not-approved'
    invalid.append([unknown,current])
    crlf = copy.deepcopy(old)
    crlf['wrapper_sha256'] = '61fbb0788458880b9cd2f6dfc529d199f9cf034ce2694229e538221479ae4eae'.upper()
    payload = base64.b64encode(json.dumps(dict(valid=[[old,current],[crlf,current]],invalid=invalid)).encode()).decode()
    path = str(SCRIPT).replace("'","''")
    command = f"""
$d=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload}'))|ConvertFrom-Json
$ast=[System.Management.Automation.Language.Parser]::ParseFile('{path}',[ref]$null,[ref]$null)
$node=$ast.FindAll({{param($n)$n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Assert-DesktopGateUpgrade'}},$true)[0]
Invoke-Expression $node.Extent.Text
foreach($pair in $d.valid){{Assert-DesktopGateUpgrade $pair[0] $pair[1]}}
foreach($pair in $d.invalid){{$blocked=$false;try{{Assert-DesktopGateUpgrade $pair[0] $pair[1]}}catch{{$blocked=$true}};if(!$blocked){{throw 'Changed gate was accepted'}}}}
'DESKTOP_REFRESH_TESTS_PASSED'
"""
    result = subprocess.run(['powershell','-NoProfile','-Command',command],
                            capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stdout+result.stderr
    assert 'DESKTOP_REFRESH_TESTS_PASSED' in result.stdout
    source = SCRIPT.read_text(encoding='utf-8')
    refresh = source[source.index("if ($Mode -eq 'RefreshDesktopGate')"):source.index("if ($Saved -ne $Current)")]
    assert refresh.count('Assert-DesktopGateUpgrade') == 2
    assert "Invoke-Legacy 'gpu-smoke'" in refresh
    assert "Invoke-Legacy 'preflight'" not in refresh
    assert '[IO.File]::Copy($GatePath,$Backup,$false)' in refresh
    assert '[IO.File]::Replace' in refresh
    assert 'checkpoint.pt' in refresh
    assert 'Start-Process' not in refresh


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
