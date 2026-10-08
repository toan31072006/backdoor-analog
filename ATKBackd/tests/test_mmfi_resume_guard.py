"""A new table run must not overwrite mismatched audit records/checkpoints."""
import hashlib
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_backdoor as trainer
from run_mmfi_tables import main, _cell_lock


def _cfg():
    return trainer._resolve_training_config(dict(model='hpeli', experiment_name='mmfi',
        seed=42, lr=0.001, rho=0.4, eps=0.185, batch_size=32, epochs=50,
        pivot=1, theta_max_deg=40, device='cpu', num_workers=0, strict_resume=True))


@pytest.mark.parametrize('failure', ['config', 'over_budget'])
def test_strict_resume_refuses_before_loading_data_or_rewriting_metadata(tmp_path, monkeypatch, failure):
    cfg = _cfg()
    metadata = tmp_path / 'config.resolved.yaml'
    metadata.write_text('ORIGINAL_AUDIT_RECORD', encoding='utf-8')
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'cfg_fingerprint': 'foreign-config' if failure == 'config'
                else trainer._config_fingerprint(cfg),
                'epoch': 3 if failure == 'config' else 50}, checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(trainer, '_load_dataset', lambda *args: pytest.fail('loaded data before guard'))
    with pytest.raises(ValueError, match='strict resume'):
        trainer.train(cfg, ckpt_dir=tmp_path)
    assert metadata.read_text(encoding='utf-8') == 'ORIGINAL_AUDIT_RECORD'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('protocol', ['infocom2025_por', 'ccai2026_backdoorrf'])
def test_rf_protocol_cannot_silently_run_as_fixed_trigger_ordinary_poisoning(protocol, monkeypatch):
    cfg = _cfg()
    cfg['training_protocol'] = protocol
    monkeypatch.setattr(trainer, '_load_dataset', lambda *args: pytest.fail('wrong trainer loaded data'))
    with pytest.raises(ValueError, match='train_rf_backdoor'):
        trainer.train(cfg)


@pytest.mark.parametrize('device', ['cuda:', 'cuda:garbage', 'cuda:-1', 'cuda:00', 'cuda'])
def test_matrix_refuses_malformed_devices_before_creating_files(tmp_path, device):
    out = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='distinct logical'):
        main(['--data-home', str(tmp_path), '--outdir', str(out), '--devices', device, '--dry-run'])
    assert not out.exists()


def test_second_scheduler_refuses_an_already_running_matrix(tmp_path):
    out = tmp_path / 'matrix'
    with _cell_lock(out):
        with pytest.raises(RuntimeError, match='Another process'):
            main(['--data-home', str(tmp_path), '--outdir', str(out), '--devices', 'cpu', '--dry-run'])
        assert not (out / 'mmfi_matrix.resolved.json').exists()
