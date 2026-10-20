from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import a1_three_tables_report as original
from a1_backbone_report import export_reports
from test_a1_three_tables_report import make_records


@pytest.mark.parametrize('model,label', [('metafiplusplus', 'MetaFi++'), ('graphposefi', 'GraphPose-Fi')])
def test_reports_are_model_labeled_and_include_t1_dce(tmp_path, model, label):
    old = original.T1_METRICS
    report = export_reports(make_records(), tmp_path / model,
        plan_sha256='synthetic-plan', provenance=dict(backbone=model))
    assert original.T1_METRICS == old
    row = next(r for r in report['tables']['T1']['rows'] if r['key'] == 'a1')
    assert row['mean_dce_mm']['n'] == 3
    root = tmp_path / model
    assert 'mean_dce_mm_mean' in (root / 'table1_comparison.csv').read_text()
    text = (root / 'table1_comparison.tex').read_text()
    assert label in text and r'DCE $\downarrow$' in text
    assert r'\mbox{' in text
    matrix = (root / 'table2_dose_poison.tex').read_text()
    assert matrix.count('$d=') == 6
    assert f'tab:{model}-dose-poison' in matrix
    assert label in matrix
    assert 'minipage' not in matrix and 'mean DCE' not in matrix
    assert matrix.endswith('\\end{table}\n')
    ablation = (root / 'table3_ablation.tex').read_text()
    assert f'tab:{model}-ablation' in ablation and label in ablation
    assert (root / 'three_main_tables.json').is_file()


def test_incomplete_no_reports_or_directories(tmp_path):
    root = tmp_path / 'out'
    with pytest.raises(ValueError, match='39'):
        export_reports(make_records()[:-1], root, plan_sha256='toy',
                       provenance=dict(backbone='metafiplusplus'))
    assert not root.exists()
