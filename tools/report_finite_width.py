"""Report completed conditional width probes without extrapolating Fz curves."""
import argparse
from html import escape
import json
from pathlib import Path
import shutil
import numpy as np
from tools import run_common_material_study as host
from tools.render_solver_dashboard import atomic_write, shell, stamp
from prototype.material_case import load_force_csv
from run_support import write_json


def report(out):
    r = host.read(out/'result.json')
    if r['status'] != 'ended': raise ValueError('Completed diagnostics required')
    review = out/'review'; review.mkdir(exist_ok=True)
    for name, h in r['source_sha256'].items():
        if host.sha(out/'source'/name) != h: raise ValueError('Frozen source mismatch: '+name)
        if host.sha(host.ROOT/name) != h: raise ValueError('Executed source changed: '+name)
    for name, h in r['inputs'].items():
        if host.sha(Path(name)) != h: raise ValueError('Input mismatch: '+name)
    if host.sha(out/'plan.md') != r['plan_sha256']: raise ValueError('Frozen plan mismatch')
    rows = []
    for f in r['fields']:
        if f['status'] != 'completed': raise ValueError('Do not suppress failed diagnostic')
        v = f['directions'][1]; refined = f['directions'][2]
        ed, ef, _ = load_force_csv(host.ROOT/'docs'/host.EXPERIMENTS[f['angle']])
        rows.append(dict(angle=f['angle'], depth_mm=f['depth_mm'], baseline_fz_n=f['baseline_fz_n'],
            experiment_fz_n=float(np.interp(f['depth_mm'], ed, ef)),
            conditional_derivative_n=v['derivative_fz_n'],
            normalized_derivative_percent=100*v['derivative_fz_n']/f['baseline_fz_n'],
            refined_derivative_n=refined['derivative_fz_n'],
            layer_4_8_relative=f['layer_4_8_force_relative_change'], qualified=v['qualified']))
    checks = [c for f in r['fields'] for v in f['directions'] for c in v.get('checks', [])]
    result = dict(reviewed=stamp(), fields=len(rows), conditional_directions=sum(len(f['directions']) for f in r['fields']),
        qualified_primary_directions=sum(v['qualified'] for v in rows), nonlinear_checks=len(checks),
        max_residual_check_error=max(c['residual_relative'] for c in checks),
        max_force_check_error=max(c['force_relative'] for c in checks),
        max_layer_change=max(v['layer_4_8_relative'] for v in rows),
        source_hashes_checked=len(r['source_sha256']), input_hashes_checked=len(r['inputs']),
        new_loading_states=0, rows=rows,
        user_observations=dict(bottom='bonded or clamped; essentially no slip',
            transverse='blade-width direction, one-sided outward bulge less than about 1 mm at indentation about 1--3 mm',
            scope='qualitative estimate; angle/location/measurement uncertainty not resolved; not a 0.5 mm observation',
            conditional_conversion='For width20mm and symmetric center-fixed motion only: half-width average stretch <~1.1; not pointwise lambda33 cap'),
        decision='No plane-stress promotion or new cutting queue. Compatibility matters; these restricted modes do not isolate a large small-angle-only force reduction.',
        limitations=['No transverse friction or width-varying in-plane warping',
                     'Fixed prior histories: derivative is not a new full loading history',
                     '4-to-8-layer check is within restricted space, not full3D convergence',
                     'Free-bottom controls have linear residual checks only, not independent nonlinear directional qualification'])
    # Static scientific figure: signs and normalization stated; no inferred new forces.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained', sharey=True)
        for ax, depth in zip(axes, (.16, .5)):
            selected = [v for v in rows if v['depth_mm']==depth]
            pos = np.arange(3)
            ax.bar(pos, [v['normalized_derivative_percent'] for v in selected], color=['#0072B2', '#009E73', '#D55E00'], width=.58)
            for x, v in zip(pos, selected):
                ax.text(x, v['normalized_derivative_percent']-.18, f'{v["normalized_derivative_percent"]:.2f}%', ha='center', va='top')
            ax.set(xticks=pos, xticklabels=['0°', '33°', '57°'], ylim=(-7, .3),
                   xlabel='Inclination', title=f'Accepted depth = {depth:g} mm')
            ax.axhline(0, color='#333333', lw=.8); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        axes[0].set_ylabel('100 / Fz × dFz/dε (% per unit ε)')
        fig.suptitle('Local response to transverse constraint release\nFixed bottom; 4 layers; NOT finite force changes', fontsize=13)
        fig.savefig(review/'sensitivity.png', dpi=180); plt.close(fig)
    write_json(review/'review.json', result)
    write_json(review/'figure_manifest.json', dict(input_sha256=host.sha(out/'result.json'),
        plotted_rows=rows, normalization='100*(dFz/depsilon)/baselineFz',
        error_bars='none: deterministic conditional derivatives, not replicated measurements',
        matplotlib_version=matplotlib.__version__, png_sha256=host.sha(review/'sensitivity.png')))
    body = '<h1>有限宽检查完成：相容约束限制释放，但尚未找到小角度专属修正</h1><p>'+stamp()+'</p>'
    body += '<p>六个真实受载场均完成，21个条件方向，六个4层主方向全部通过两档新鲜三维残差检验；4→8层导数变化0.40–0.54%。原材料、接触、实验零点及物理验收保持，没有新增加载曲线。</p>'
    body += '<h2>0.5 mm同深度结果</h2><table><tr><th>角度</th><th>原Fz N</th><th>实验 N</th><th>dFz/dε N</th><th>归一化导数</th></tr>'
    for v in rows:
        if v['depth_mm'] == .5:
            body += f'<tr><td>{v["angle"]}°</td><td>{v["baseline_fz_n"]:.3f}</td><td>{v["experiment_fz_n"]:.3f}</td><td>{v["conditional_derivative_n"]:.3f}</td><td>{v["normalized_derivative_percent"]:.2f}% / 单位ε</td></tr>'
    body += '</table><p>ε是维持平面应变所需虚拟面外约束反力的释放比例；这里只验证ε=0附近的导数，不能把上表导数直接加到原Fz，或称为释放后的真实降力幅度。0/33/57°按原力归一化后的局部敏感性接近，未出现只显著降低小角度而保持大角度的方向。</p>'
    body += '<img src="sensitivity.png" style="width:100%;max-width:1100px" alt="两个深度的归一化条件导数；三角度均为负且幅度接近，不是新Fz曲线。">'
    body += '<h2>新实验约束</h2><p>用户确认底面粘接或夹紧、基本不能滑动；采用底面面外固定主诊断。用户估计下压1–3 mm时沿刀刃宽度方向单侧外鼓不到约1 mm。若20 mm宽度且中心近似不动、两侧近似对称，这对应半宽平均伸长小于约10%；不能转成每个材料点λ33≤1.1，也不是0.5 mm的实测值。角度、局部位置、刚体运动及误差尚未细化。</p>'
    body += '<p>底面面外自由控制会放大条件响应，但与已确认的装夹前提不符，不能用来降力。它仅作线性敏感性证据，没有独立完成非线性方向检验。平面应力中很大的局部厚度伸长也不等于实测外鼓；缺少跨宽相容条件时不能从它直接推算侧面位移。</p>'
    body += '<h2>决策与下一步</h2><p>不采用完全平面应力，也不按角度混合二维模型。当前检查保留面内位移跨宽一致，缺少跨宽面内翘曲和面外库仑摩擦；不能据此排除所有三维效应，也不能称为完整有限宽反力上下界。后续若实现有限宽候选，需要这两项共同进入方程，并在同一底面固定条件下检查力、面内滑移及侧面外鼓。没有通过这些资格前，不启动新的三角度长加载。</p>'
    body += '<p>本轮18项相关回归通过、无跳过；实际场三维/二维应力和切线一致。六场原历史重新验收通过；冻结源码和旧输入哈希重新核对。证据只能支持受限方向内的局部判断，不能把材料真实性或实验曲线复现称为已完成。</p>'
    body += '<p>依据：<a href="https://bleyerj.github.io/comet-fenicsx/tours/linear_problems/generalized_plane_strain/generalized_plane_strain.html">Bleyer广义平面应变教程正文</a>说明整体面外自由度与局部应力条件的区别；<a href="https://arxiv.org/abs/2410.22562">Ahmadi等平面应力论文摘要</a>说明积分点面外消去。两者均不直接验证本项目的有限宽黏弹切割。</p>'
    body += '<p><a href="review.json">审查数值</a> · <a href="../index.html">原始条件方向和HTML日志</a> · <a href="../plan.md">冻结方案</a></p>'
    body += '<details><summary>完整审查记录</summary><pre>'+escape(json.dumps(result, ensure_ascii=False, indent=2))+'</pre></details>'
    atomic_write(review/'index.html', shell('有限宽研究结果', body))
    shutil.copy2(__file__, review/'report_source.py')
    write_json(review/'report_provenance.json', dict(report_sha256=host.sha(review/'report_source.py'),
        physical_result_sha256=host.sha(out/'result.json'), scope='postprocessing only; original frozen physical files unchanged'))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    report(parser.parse_args().output.resolve())
