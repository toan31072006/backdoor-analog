"""Backbone-labeled exports, retaining the original three-seed statistics."""
from __future__ import annotations

import math

from a1_backbone_adapter import _private_module


def export_reports(records, outdir, *, plan_sha256, provenance):
    # A private reporter preserves both the old HPE-Li source bytes and globals.
    engine = _private_module('a1_three_tables_report.py', '_cross_backbone_report_engine')
    engine.T1_METRICS = (*engine.T1_METRICS, 'mean_dce_mm')
    original_format, original_t2, original_t3 = engine._formatted, engine._render_t2, engine._render_t3
    label = {'metafiplusplus': 'MetaFi++', 'graphposefi': 'GraphPose-Fi'}[provenance['backbone']]

    def formatted(stats, *, latex=False, bold=False):
        value = original_format(stats, latex=latex, bold=bold)
        return r'\mbox{' + value + '}' if latex else value

    def render_t1(report):
        rows = report['tables']['T1']['rows']
        headers = ('Method', 'T-MPJPE ↓', 'Target-PCK ↑', 'Clean MPJPE ↓', 'Clean PA-MPJPE ↓', 'DCE ↓')
        caption = (f'{label}; full MM-Fi protocol1-s1, seeds42/43/44, final50epochs, '
            'mean +/- sample SD. Target metrics at dose1; DCE averages five positive doses. '
            'Errors in mm; Target-PCK in percent. Bold marks best observed targeted-method mean, '
            'not statistical significance; Clean is not attack-ranked.')
        md = [f'# T1: Comparison ({label})', '', caption, '',
              '| ' + ' | '.join(headers) + ' |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        tex = [r'\begin{table*}[t]', r'\centering',
            r'\caption{' + label + r' on MM-Fi protocol1-s1; seeds 42,43,44; fixed final epoch. '
            r'Mean $\pm$ sample SD. Target metrics use $d=1$; DCE averages five positive doses. '
            r'Errors in mm and Target-PCK in percent. Bold denotes best observed targeted-method mean, '
            r'not statistical significance; Clean is not attack-ranked.}',
            r'\label{tab:' + provenance['backbone'] + r'-comparison}',
            r'\small', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lrrrrr}', r'\toprule',
            r'Method & T-MPJPE $\downarrow$ & Target-PCK $\uparrow$ & Clean MPJPE $\downarrow$ & Clean PA-MPJPE $\downarrow$ & DCE $\downarrow$ \\',
            r'\midrule']
        for row in rows:
            flags = {name: row['key'] != 'clean' and row[name] is not None and math.isclose(
                row[name]['mean'], engine._best(rows, name, higher=name == 'target_pck_pct'),
                rel_tol=0, abs_tol=1e-12) for name in engine.T1_METRICS}
            md.append('| ' + row['method'] + ' | ' + ' | '.join(
                formatted(row[name], bold=flags[name]) for name in engine.T1_METRICS) + ' |')
            tex.append(row['method'] + ' & ' + ' & '.join(
                formatted(row[name], latex=True, bold=flags[name]) for name in engine.T1_METRICS) + r' \\')
            if row['key'] == 'clean':
                tex.append(r'\midrule')
        tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
        return '\n'.join(md) + '\n', '\n'.join(tex) + '\n'

    def render_t2(report):
        md, tex = original_t2(report)
        # Keep clean/DCE summaries in machine-readable exports and Markdown;
        # the compact paper matrix contains only the six dose rows.
        marker = r'\par\smallskip'
        if marker in tex:
            tex = tex.split(marker, 1)[0] + '\\end{table}\n'
        tex = tex.replace(r'\caption{', r'\caption{' + label + ': ', 1).replace(
            r'\label{tab:dose-poison}', r'\label{tab:' + provenance['backbone'] + '-dose-poison}')
        return md.replace('# T2:', f'# T2 ({label}):', 1), tex

    def render_t3(report):
        md, tex = original_t3(report)
        tex = tex.replace(r'\caption{', r'\caption{' + label + ': ', 1).replace(
            r'\label{tab:ablation}', r'\label{tab:' + provenance['backbone'] + '-ablation}')
        return md.replace('# T3:', f'# T3 ({label}):', 1), tex

    engine._formatted, engine._render_t1 = formatted, render_t1
    engine._render_t2, engine._render_t3 = render_t2, render_t3
    return engine.export_reports(records, outdir, plan_sha256=plan_sha256, provenance=provenance)
