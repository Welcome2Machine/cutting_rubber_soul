"""Render independently hashed contact qualification evidence; no FE calls."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);args=parser.parse_args()
    out=args.study.resolve();sources={};rows=[];input_hashes={};candidate_hashes=[]
    for angle in (0,33,57):
        folder=out/f'angle{angle}';s=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if s['status']!='completed':raise ValueError('Incomplete qualification evidence')
        manifest=json.loads((folder/'candidate_sources.json').read_text(encoding='utf-8'));candidate_hashes.append(manifest)
        for name,h in manifest.items():
            if sha(folder/'source'/name)!=h:raise ValueError('Changed diagnostic snapshot')
        for path,h in s['inputs'].items():
            if path not in input_hashes and sha(path)!=h:raise ValueError('Changed original evidence: '+path)
            input_hashes[path]=h
        sources[str(folder/'summary.json')]=sha(folder/'summary.json')
        rows.extend(s['rows'])
    if any(m!=candidate_hashes[0] for m in candidate_hashes):raise ValueError('Unequal qualification code versions')
    tests=json.loads((out/'regressions.json').read_text(encoding='utf-8'))
    if not tests['passed'] or tests['skipped']:raise ValueError('Regression qualification incomplete')
    patches=[p for r in rows for p in r['local_contact']['patches']]
    statuses=Counter(p['contact_state'] for p in patches)
    tangents=[p for p in patches if 'tangent_verified' in p]
    table=[]
    for r in rows:
        p=r['local_contact']['patches'];table.append(dict(angle=r['angle'],depth_mm=r['depth_mm'],accepted=r['accepted'],
            smooth_vs_p1_max_um=1000*r['geometry']['maximum_same_parameter_p1_deviation_mm'],
            curve_blade_crossings=r['geometry']['intersection_count'],native_intersection=r['native_intersection'],
            active=sum(v['point_coulomb_metric'] is not None for v in p),sampled=len(p),
            active_tangents=sum(v.get('tangent_verified',False) for v in p),
            max_local_gap_change_um=1000*max(abs(v['gap_change_mm']) for v in p),
            unsupported_vv=r['local_contact']['unsupported_native_counts'].get('VV_not_assembled_by_local_candidate',0)))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=table[0]);writer.writeheader();writer.writerows(table)
    summary=dict(accepted_fields=sum(r['accepted'] for r in rows),unaccepted_fields=sum(not r['accepted'] for r in rows),
        patches=len(patches),contact_status_counts=dict(statuses),active_tangents=len(tangents),
        tangent_passes=sum(p['tangent_verified'] for p in tangents),
        max_tangent_relative_error=max(t['relative_error'] for p in tangents for t in p['tangent_trials']),
        max_force_balance_n=max(p['force_balance_n'] for p in patches),
        max_moment_balance_nmm=max(p['moment_balance_nmm'] for p in patches),
        max_rigid_rotation_slip_mm=max(abs(p['rigid_rotation_slip_mm']) for p in patches),
        max_rigid_rotation_friction_force_n=max(p['rigid_rotation_friction_force_n'] for p in patches),
        original_input_hashes_verified=len(input_hashes),source_summary_sha256=sources,
        candidate_snapshot_sha256=candidate_hashes[0],tests=tests,
        new_accepted_loading_states=0,long_queue_started=False,production_promoted=False,
        limitations=['Local positive chart only; no global native-stencil equivalence',
            'Geometry changes contact states; 64 samples are not exhaustive boundary coverage',
            'Static cubic-line intersection check is not whole-edge or curved-self-contact CCD',
            'Backward-Euler transported friction work is not an integrated path energy audit',
            'Fixed displacement smoothing unchanged; full temporal convergence remains unqualified'],
        initial_harness_error='Initial angle57 diagnostic omitted absent failed-trial depth metadata; qualified run uses rejected-load depth and checks prescribed blade movement. No new mechanical solve failed.')
    (out/'verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
        for ax,angle,label in zip(axes,(0,33,57),('0050','0050','failed')):
            row=next(r for r in rows if r['angle']==angle and r['label']==label)
            with np.load(out/f'angle{angle}'/f'geometry_{label}.npz') as z:
                u=z['parameters'];x=z['top_positions'];blade=z['blade_positions']
            c=CubicSpline(u,x,axis=0,bc_type='natural');center=np.array(row['geometry']['maximum_deviation_location']['point'])
            keep=np.flatnonzero(np.linalg.norm(x-center,axis=1)<.1)
            a,b=max(0,keep.min()-2),min(len(x)-1,keep.max()+2)
            q=np.concatenate([np.linspace(u[i],u[i+1],17) for i in range(a,b)])
            ax.fill(blade[:,0],blade[:,1],color='#d0d0d0',label='Native blade')
            ax.plot(x[a:b+1,0],x[a:b+1,1],'k.-',lw=1,ms=3,label='Native P1 surface')
            ax.plot(c(q)[:,0],c(q)[:,1],color='#0072B2',ls='--',lw=1.4,label='Cubic interpolation')
            ax.plot(*center,'x',color='#D55E00',ms=7,label='Max deviation location')
            ax.set(xlim=(center[0]-.045,center[0]+.045),ylim=(center[1]-.045,center[1]+.045),
                xlabel='Laboratory x (mm)',ylabel='Laboratory y (mm)',aspect='equal',
                title=f"{angle} deg, {row['depth_mm']:.3f} mm\n"+('Accepted state' if row['accepted'] else 'UNACCEPTED failed iterate'))
            ax.ticklabel_format(useOffset=False);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8,loc='lower left')
        fig.savefig(out/'geometry.png',dpi=180);fig.savefig(out/'geometry.svg');plt.close(fig)
    headers=['角度','行程mm','原状态','样条/P1最大偏差μm','静态交叉','有载局部点','切线通过','VV未覆盖']
    cells=[[r['angle'],r['depth_mm'],'接受' if r['accepted'] else '未收敛',f"{r['smooth_vs_p1_max_um']:.4f}",r['curve_blade_crossings'],f"{r['active']}/{r['sampled']}",r['active_tangents'],r['unsupported_vv']] for r in table]
    tab='<table><thead><tr>'+''.join('<th>'+v+'</th>' for v in headers)+'</tr></thead><tbody>'
    tab+=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in cells)+'</tbody></table>'
    content='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>客观接触历史：实现及实际场资格核查</title><style>body{max-width:1180px;margin:32px auto;padding:0 20px;color:#202124;font:16px/1.7 system-ui}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #bbb;text-align:right}th{background:#eef3f6}img{width:100%}code{background:#eee;padding:2px 5px}.decision{border-left:5px solid #a34b00;background:#fff6e8;padding:14px}a{color:#005ea8}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>客观接触历史：实现及实际场资格核查</h1>
<p class="decision">局部修正及保存场导数检查通过；完整FE接触尚未合格。新增加载态0，没有启动长队列，也没有新的Fz改善结论。默认材料、摩擦、边界、P1单元、CCD和物理残差验收均保持。</p>
<h2>已经实现</h2><p>跨样条节点的材料坐标历史；激活时刻之后的滑移；脱离清空、再接触重新激活；仅在整体平衡接受后提交历史。完整局部切线含投影、曲率度量、法向载荷、摩擦平滑及激活时刻导数。共同刚体转动不产生伪滑移，曲面有限间隙处的点切向力满足指定的正则库仑上限。</p>
<p>采用h=D/|c′|、β=h∇u、w=h(u−uₐ)、R<sub>f</sub>=μNf′(w)β。这里w是后向欧拉传输增量，并非有限曲线弧长；μNf′(w)w是该离散增量的非负耗散量，尚不是完整路径能量验收。单点/曲线路径采用相对速度上界检查；遇到多次激活/脱离或无法证明的间隙时拒绝，而不是漏过。</p>
<h2>实际保存场证据</h2>'''+tab+f'''
<p>7个已接受场用各自冻结源码和双历史重新验收；57°的0.135 mm场明确保留为未收敛场。64个局部重建中{statuses.get('persistent',0)+statuses.get('activated',0)}个有载、{statuses.get('inactive',0)}个未激活、{statuses.get('detached',0)}个脱离。32个有载点各做三档有限差分，全部满足相邻两档相对误差&lt;10⁻³；全体最大误差{summary['max_tangent_relative_error']:.3g}。</p>
<p>共同转动试验最大伪滑移{summary['max_rigid_rotation_slip_mm']:.3g} mm，最大摩擦力{summary['max_rigid_rotation_friction_force_n']:.3g} N；力与力矩闭合误差分别不超过{summary['max_force_balance_n']:.3g} N、{summary['max_moment_balance_nmm']:.3g} N·mm。这些量属于独立局部核，不是整体Fz。</p>
<h2>真实阻碍：几何与积分不能分开替换</h2>
<p>33°、0.5 mm的最大同参数位置偏差为1.35046 μm，相当于2.5 μm接触作用距离的54%。它是表面插值的向量距离，不等同法向间隙误差；64个局部点中最大间隙变化约0.426 μm。部分原生接触变为未激活或脱离。8个场静态样条/刀具线段交叉数均为0，但这不构成曲线整体CCD或自接触资格。</p>
<img src="geometry.png" alt="0度和33度0.5毫米接受场，以及57度0.135毫米未收敛场的刀尖局部原折线与样条表面；叉号标记最大同参数插值偏差位置。每图使用实验室坐标毫米及等比例坐标。">
<p>图中样条仅用于几何诊断；没有平滑实验数据或把未收敛场作为预测。自然样条与原P1边界不是相同几何。每个场还存在局部候选未组装的VV项及刀面端部接触；相邻EV还可能重复引用同一个点。将正EV新核逐项塞入原生组合，或删除剩余VV，都没有合格的全局积分依据。</p>
<h2>下一项实现与验收顺序</h2><ol>
<li>优先保留P1材料边界，刀具使用具有连续法向的解析圆鼻/刀面；对原材料边界建立固定、正权重的面积积分，避免为了平滑接触而移动橡胶表面。历史属于唯一材料积分点。</li>
<li>刀具—材料接触整体独立组装，不能与原生EV/VV反复计数。材料自接触及其CCD继续完整保留；解析刀具、折线碰撞保护的几何包络必须一致。</li>
<li>用完整接触力/切线、跨圆鼻连接处与接触区域边界、面积积分收敛和路径历史验收；旧解析接触路线曾失败，不因为再次使用解析几何而称其已修好。</li>
<li>合格后从零建立独立0/33°短路径，57°先过原失败位置及相邻增量。曲线没有改进则降低接触对幅值偏差的归因优先级；不通过增加PTC预算来替代上述检查。</li></ol>
<h2>证据与运行</h2><p>{tests['tests']}项回归通过，跳过{tests['skipped']}项；核对{len(input_hashes)}个原始证据路径及三个相同候选源码快照。首次57°诊断遇到失败快照不含depth_mm的工具字段错误，已改为读取拒绝增量元数据并核对刀具实际位移；不是新的力学失败。初次记录保存在上级目录，权威结果在qualified。</p>
<p><a href="verification.json">验证摘要JSON</a> · <a href="summary.csv">汇总CSV</a> · <a href="regressions.log">回归原始记录</a> · <a href="regressions.html">回归HTML</a></p>
<p><a href="angle0/run_status.html">0°运行日志</a> · <a href="angle33/run_status.html">33°运行日志</a> · <a href="angle57/run_status.html">57°运行日志</a></p>
<p><a href="../../../docs/contact_transport_qualification.md">详细模型与实施边界</a> · <a href="../../../references/contact_transport_notes.md">文献阅读记录</a></p></html>'''
    (out/'index.html').write_text(content,encoding='utf-8')
    (out/'regressions.html').write_text('<!doctype html><meta charset="utf-8"><title>接触回归</title><pre>'+html.escape((out/'regressions.log').read_text(encoding='utf-8'))+'</pre>',encoding='utf-8')
    shutil.copy2(__file__,out/'report_contact_transport.py')
    files=[out/f for f in ('index.html','geometry.png','geometry.svg','verification.json','summary.csv','report_contact_transport.py')]
    (out/'render_manifest.json').write_text(json.dumps({str(p.name):sha(p) for p in files},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_summary_sha256','candidate_snapshot_sha256','tests')},ensure_ascii=False))


if __name__=='__main__':main()
