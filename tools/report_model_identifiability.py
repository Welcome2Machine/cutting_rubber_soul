"""Read-only synthesis of the common-parameter/model audit; no FE imports."""
import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import shutil

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.render_solver_dashboard import atomic_write, shell, stamp
from run_support import write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def render(study):
    evidence = read(study/'evidence/evidence.json')
    history = read(study/'history_tau/result.json')
    if history['status'] != 'ended':
        raise RuntimeError('Final report requires an ended diagnostic, not a running snapshot')
    verified = {'source': 0, 'input': 0, 'evidence_input': 0}
    for name, expected in history['source_sha256'].items():
        if sha(study/'history_tau/source'/name) != expected or sha(ROOT/name) != expected:
            raise RuntimeError(f'Source changed: {name}')
        verified['source'] += 1
    for name, expected in history['inputs'].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f'Input changed: {name}')
        verified['input'] += 1
    for name, expected in evidence['inputs'].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f'Evidence input changed: {name}')
        verified['evidence_input'] += 1
    out = study/'review'; out.mkdir(exist_ok=True)
    summary = dict(generated=stamp(), verified=verified, new_loading_states=0, angles=[])
    with plt.rc_context({'font.size': 10, 'axes.grid': True, 'grid.alpha': .22}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True, layout='constrained')
        for ax, a in zip(axes, history['angles']):
            rows = []
            for row in a['steps']:
                if not row['passed']: break
                rows.append(row)
            x = [s['depth_mm'] for s in rows]
            ax.plot(x, [s['derivative_n'] for s in rows], color='#0072B2', marker='o', markevery=5,
                    markersize=3, label='Propagated history')
            ax.plot(x, [s['frozen_history_derivative_n'] for s in rows], color='#D55E00', linestyle='--',
                    label='Frozen history (conditional)')
            last = rows[-1] if rows else {}
            rejected = next((s for s in a['steps'] if not s['passed']), None)
            if rejected:
                ax.axvline(rejected['depth_mm'], color='#555555', linestyle=':', linewidth=1.4,
                           label='Derivative check stopped')
            title=f'{a["angle"]}° | qualified to {last["depth_mm"]:.2f} mm' if rows else f'{a["angle"]}° | no qualified prefix'
            if not rows: ax.text(.5, .4, 'No qualified derivative\n(not zero sensitivity)',ha='center',va='center',transform=ax.transAxes)
            ax.set(xlim=(0, .5), xlabel='Vertical depth (mm)', title=title)
            ax.axhline(0, color='#777777', linewidth=.5)
            summary['angles'].append(dict(angle=a['angle'], status=a['status'], qualified_states=len(rows),
                last_qualified_depth_mm=last.get('depth_mm'), derivative_n=last.get('derivative_n'),
                frozen_history_derivative_n=last.get('frozen_history_derivative_n'),
                failure_depth_mm=rejected['depth_mm'] if rejected else None,
                failed_checks=rejected if rejected else None,
                linear_call_count=len(a.get('linear_calls', [])), elapsed_s=a['elapsed_s'],
                error=a.get('error'), traceback=a.get('traceback')))
        axes[0].set_ylabel('dFz / d ln(tau) (N)')
        axes[0].legend(fontsize=8, loc='upper left')
        fig.savefig(out/'history_sensitivity.png', dpi=170, facecolor='white')
        fig.savefig(out/'history_sensitivity.svg', facecolor='white')
        plt.close(fig)
    write_json(out/'summary.json', summary)
    document = ROOT/'docs/model_identifiability_audit.md'
    content = markdown.markdown(document.read_text(encoding='utf-8'), extensions=['tables','fenced_code','sane_lists'])
    def link(match):
        target = match.group(1)
        if target.startswith(('http:', 'https:', '#', 'data:')): return match.group(0)
        return 'href="'+os.path.relpath(document.parent/target, out).replace('\\','/')+'"'
    content = re.sub(r'href="([^"]+)"', link, content)
    body = '<header><h1>参数选择与模型结构：本轮核查</h1><p>'+stamp()+'</p>'
    body += '<nav><a href="../history_tau/index.html">实际诊断日志</a><a href="../evidence/evidence.json">既有算例数值</a><a href="summary.json">新诊断数值</a><a href="../regressions.log">51项回归</a></nav></header>'
    body += '<section><h2>完整历史局部敏感性</h2><p>实线只显示连续通过两尺度、真实残差/反力、状态导数和几何检查的前缀；虚线是固定旧历史的单尺度对照偏导，未单独取得方向资格。不是新的Fz曲线，无平滑或外推；一条确定性路径/角度，没有实验置信区间。</p>'
    body += '<img src="history_sensitivity.png" style="width:100%" alt="三个角度的松弛时间敏感性；完整历史实线与固定历史虚线对比，失败深度有竖线且不向后外推。">'
    body += '<table><tr><th>角度</th><th>资格终态</th><th>有效状态数/末深度</th><th>历史导数 N</th><th>固定历史偏导 N</th><th>停止位置</th></tr>'
    for a in summary['angles']:
        body += '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in
            (a['angle'],a['status'],f'{a["qualified_states"]} / {a["last_qualified_depth_mm"]}',
             a['derivative_n'],a['frozen_history_derivative_n'],a['failure_depth_mm']))+'</tr>'
        if a['error']: body+='<tr><td colspan="6">'+escape(a['error'])+'；这是诊断终止，不是新增FE加载失败。</td></tr>'
    body += '</table><p>冻结源、原输入与实验汇总输入SHA-256复核：'+escape(str(verified))+'。新增加载接受态：0。</p></section>'
    body += '<section class="document">'+content+'</section>'
    atomic_write(out/'index.html', shell('参数选择与模型结构核查',body))
    shutil.copy2(document,out/'report.md')
    shutil.copy2(Path(__file__),out/'report_code.py')
    provenance=dict(audience='Local research audit HTML; not journal submission',
        data=[str(study/'history_tau/result.json'),str(study/'evidence/evidence.json')],
        data_sha256={str(p):sha(p) for p in (study/'history_tau/result.json',study/'evidence/evidence.json')},
        transformations='Only contiguous passed prefix plotted; frozen-history comparison explicitly conditional; no smoothing, extrapolation or fit.',
        uncertainty='No uncertainty interval; deterministic path derivative, two perturbation scales recorded in source.',
        versions={'matplotlib':matplotlib.__version__},figure_inches=[12,3.8],png_dpi=170,
        hashes={name:sha(out/name) for name in ('index.html','summary.json','history_sensitivity.png','history_sensitivity.svg','report.md','report_code.py')})
    write_json(out/'provenance.json',provenance)
    print(json.dumps({'verified':verified,'angles':[{k:v for k,v in a.items() if k!='failed_checks'} for a in summary['angles']]},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    render(parser.parse_args().study.resolve())
