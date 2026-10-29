"""Plot saved actual-cutting paths and finite-crack evidence; never starts FE."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
import numpy as np
from run_support import write_json


def connected_length(mask, y):
    """Reference length connected to the original top, requiring both edge ends."""
    order = np.argsort(y)[::-1]
    length = 0.
    for i,j in zip(order[:-1],order[1:]):
        if not (mask[i] and mask[j]):
            break
        length += y[i]-y[j]
    return float(length)


def curve_difference(da, fa, db, fb, end):
    """Exact integral for the plotted linear interpolants, including tiny bursts.

    Fz is an interval average; this compares displayed curves, not a newly
    reconstructed instantaneous force. No event shift or uniform resampling.
    """
    knots=np.unique(np.r_[0.,end,da[(da>0)&(da<end)],db[(db>0)&(db<end)]])
    a=np.interp(knots,da,fa);b=np.interp(knots,db,fb);delta=b-a
    def square_integral(value):
        return float(np.sum(np.diff(knots)*(value[:-1]**2+value[:-1]*value[1:]+value[1:]**2)/3))
    difference=square_integral(delta);reference=square_integral(a)
    imax=int(np.argmax(np.abs(delta)))
    return dict(relative_l2=float(np.sqrt(difference/max(reference,1e-24))),
        maximum_absolute_difference_n=float(abs(delta[imax])),
        maximum_difference_depth_mm=float(knots[imax]),knots=len(knots))


def summarize(directory):
    path = directory/'result.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    rows = result['steps']
    if not rows:
        raise ValueError(f'No accepted path: {directory}')
    if not result.get('accepted_prefix_audited') or not result['summary'].get('accepted_chord_ccd_verified'):
        raise ValueError(f'Plot requires a numerically audited, CCD-safe prefix: {directory}')
    d = np.array([r['depth_mm'] for r in rows])
    f = np.array([r['reaction_fz_n'] for r in rows])
    energy = {k: np.array([r['energy'][k] for r in rows]) for k in rows[0]['energy']}
    maximum = np.maximum.accumulate(f)
    drop = maximum-f
    index = int(np.argmax(drop))
    peak_index=int(np.argmax(f[:index+1]))
    dt = []
    # Plot only stored data; no problem construction or numerical solve here.
    for r in rows:
        with np.load(directory/r['state_file']) as state:
            dt.append(float(state['material_dt_s']))
    with np.load(directory/rows[-1]['state_file']) as z:
        ref = z['reference_xy']
        body_n = len(z['mass'])
        seam = np.flatnonzero(abs(ref[:body_n,0]-15.) < 1e-10)
        if len(seam) != len(z['released']):
            raise ValueError('This report requires the explicit centred straight seam')
        y = ref[seam,1]
        failed = z['history'] >= 2*result['config']['toughness_n_mm']/result['config']['strength_mpa']
        connected_failed = connected_length(failed,y)
        connected_released = connected_length(z['released'],y)
        opening = 2*(z['positions'][seam,0]-15.)
        final = dict(y=y.tolist(),opening_mm=opening.tolist(),history_mm=z['history'].tolist(),
                     released=z['released'].tolist(),fully_failed=failed.tolist())
    start = next((r['depth_mm'] for r in rows if r['insertion_events']),None)
    accepted = [c for r in rows for c in r.get('finite_increment_checks',[]) if c['accepted']]
    physical = energy['physical_balance_defect_n_mm']
    work = float(np.abs(energy['external_work_n_mm']).sum())
    k = energy['kinetic_energy_n_mm']
    peak = int(np.argmax(k))
    window = np.abs(d-d[peak]) <= .001 # +/-5 ms at the fixed 0.2 mm/s.
    reservoir = max(float(energy['mechanical_reservoir_n_mm'][peak]),1e-12)
    evidence = {}
    result_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    for kind in ('finite_audit','path_audit'):
        audit_path = directory.parent/(directory.name+'.'+kind+'.json')
        evidence[kind] = json.loads(audit_path.read_text()) if audit_path.exists() else None
        if evidence[kind] is not None:
            a=evidence[kind]
            if (a['result_sha256']!=result_sha or a['source_hashes']!=result['source_hashes']
                    or not a.get('source_unchanged')):
                raise ValueError(f'Audit identity differs: {audit_path}')
    events=[];old_failed=0
    for i,row in enumerate(rows):
        if row['insertion_events'] or row['fully_failed_nodes']>old_failed:
            mask=np.abs(d-d[i])<=.001
            events.append(dict(depth_mm=float(d[i]),new_fully_failed_nodes=row['fully_failed_nodes']-old_failed,
                inserted_nodes=sum(len(event['nodes']) for event in row['insertion_events']),
                window_halfwidth_s=.005,window_complete=bool(d[0]<=d[i]-.001 and d[-1]>=d[i]+.001),
                absolute_defect_n_mm=float(np.abs(physical[mask]).sum()),
                absolute_defect_over_event_reservoir=float(np.abs(physical[mask]).sum()/max(float(energy['mechanical_reservoir_n_mm'][i]),1e-12))))
        old_failed=row['fully_failed_nodes']
    summary = dict(name=directory.name,directory=str(directory.resolve()),
        result_sha256=result_sha,status=result['status'],
        last_depth_mm=float(d[-1]),accepted_states=len(rows),first_release_mm=start,
        released_nodes=rows[-1]['released_nodes'],fully_failed_nodes=rows[-1]['fully_failed_nodes'],
        connected_released_reference_length_mm=connected_released,
        connected_fully_failed_reference_length_mm=connected_failed,
        maximum_drawdown_n=float(drop[index]),drawdown_depth_mm=float(d[index]),
        drawdown_over_prior_peak=float(drop[index]/maximum[index]) if maximum[index]>0 else None,
        drawdown_diagnostic=dict(peak_depth_mm=float(d[peak_index]),trough_depth_mm=float(d[index]),
            peak_normal_fz_n=rows[peak_index]['reaction_normal_fz_n'],trough_normal_fz_n=rows[index]['reaction_normal_fz_n'],
            peak_friction_fz_n=rows[peak_index]['reaction_friction_fz_n'],trough_friction_fz_n=rows[index]['reaction_friction_fz_n'],
            new_insertion_during_drawdown=any(r['insertion_events'] for r in rows[peak_index:index+1]),
            fully_failed_node_change=rows[index]['fully_failed_nodes']-rows[peak_index]['fully_failed_nodes'],
            resolved_event_claim=False),
        absolute_defect_over_absolute_work=float(np.abs(physical).sum()/work) if work else None,
        peak_kinetic_n_mm=float(k[peak]),peak_kinetic_depth_mm=float(d[peak]),
        peak_window_absolute_defect_over_peak_reservoir=float(np.abs(physical[window]).sum()/reservoir),
        peak_window_complete=bool(d[0] <= d[peak]-.001 and d[-1] >= d[peak]+.001),
        minimum_dt_s=min(dt),max_residual_n=max(r['residual_n'] for r in rows),
        accepted_finite_increments=len(accepted),
        minimum_accepted_energy_margin_n_mm=min((c['energy_margin_n_mm'] for c in accepted),default=None),
        accepted_prefix_audited=result.get('accepted_prefix_audited',False),
        accepted_chord_ccd_verified=result['summary']['accepted_chord_ccd_verified'],
        producer_source_unchanged=result['source_unchanged'],error=result.get('error'),
        completed_2mm=bool(d[-1]>=2.-1e-12),
        physical_cutting_qualified=False,temporal_convergence_verified=False,mesh_convergence_verified=False,
        config=result['config'],evidence=evidence,final_seam=final,event_energy_windows=events)
    summary['cumulative_energy_n_mm']={key:float(value.sum()) for key,value in energy.items()
        if key not in ('mechanical_reservoir_n_mm','kinetic_energy_n_mm')}
    summary['final_kinetic_energy_n_mm']=float(k[-1])
    return summary, d, f, np.array(dt), energy


def build(directories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    output.mkdir(parents=True,exist_ok=False)
    cases = [summarize(p.resolve()) for p in directories]
    fig, axes = plt.subplots(3,1,figsize=(11,11),sharex=True,layout='constrained')
    for s,d,f,dt,e in cases:
        name=s['name']
        line=axes[0].plot(d,f,label=name,lw=1.25)[0]
        color=line.get_color()
        axes[1].plot(d,np.cumsum(np.abs(e['physical_balance_defect_n_mm'])),color=color)
        axes[2].semilogy(d,dt,color=color)
    axes[0].set_ylabel('Fz [N] (interval average)');axes[0].legend(ncol=2,fontsize=8)
    axes[0].set_title('Actual blade, independent histories; missing suffixes are not extrapolated')
    axes[1].set_ylabel('Cumulative |energy defect| [N mm]')
    axes[2].set_ylabel('Accepted physical dt [s]');axes[2].set_xlabel('Vertical blade travel [mm]')
    for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,2)
    fig.savefig(output/'force_energy_time.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for s,d,f,*_ in cases:ax.plot(d,f,lw=1.2,label=s['name'])
    ax.set(xlabel='Vertical blade travel [mm]',ylabel='Fz [N] (interval average)',xlim=(0,2),
           title='Finite-increment candidates: incomplete post-fracture paths')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.savefig(output/'fz.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5.5),layout='constrained')
    for s,*_ in cases:
        q=s['final_seam'];label=f"{s['name']} @ {s['last_depth_mm']:.3f} mm"
        axes[0].plot(q['opening_mm'],q['y'],'.-',label=label,lw=1)
        axes[1].plot(q['history_mm'],q['y'],'.-',lw=1)
    axes[0].set_xlabel('Current crack opening [mm]');axes[0].set_ylabel('Reference seam height [mm]')
    axes[1].set_xlabel('Maximum opening history [mm]');axes[1].set_ylabel('Reference seam height [mm]')
    axes[0].legend(fontsize=7);axes[0].set_title('Final saved states: different depths')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(output/'seam_opening.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,len(cases),figsize=(3.4*len(cases),5),layout='constrained',squeeze=False)
    for ax,(s,*_) in zip(axes[0],cases):
        folder=Path(s['directory']);raw=json.loads((folder/'result.json').read_text())
        with np.load(folder/raw['steps'][-1]['state_file']) as z:
            n=len(z['mass']);xy=z['positions'][:n];tri=z['triangles']
            ax.triplot(xy[:,0]-15.,xy[:,1],tri,color='#96a8b1',lw=.45)
            ax.triplot(15.-xy[:,0],xy[:,1],tri,color='#96a8b1',lw=.45)
            seam=np.flatnonzero(abs(z['reference_xy'][:n,0]-15.)<1e-10)
            for sign in (-1,1):
                ax.plot(sign*(xy[seam,0]-15.),xy[seam,1],color='#127a79',lw=1)
                ax.scatter(sign*(xy[seam[~z['released']],0]-15.),xy[seam[~z['released']],1],s=9,c='#cc643c')
            blade=z['positions'][n:];ax.plot(blade[:,0]-15.,blade[:,1],'.',ms=1.7,color='#3b4252')
        ax.set(xlim=(-1.2,1.2),ylim=(2.5,5.4),aspect='equal',xlabel='x - centre [mm]',
               title=f"{s['name']}\n{s['last_depth_mm']:.3f} mm")
    axes[0,0].set_ylabel('Deformed height [mm]')
    fig.suptitle('Saved deformed meshes; orange nodes remain tied. Different terminal depths.')
    fig.savefig(output/'deformed_seam.png',dpi=150);plt.close(fig)
    comparisons=[]
    families=dict(time_control={'increment_mm','energy_tolerance','energy_absolute_tolerance_n_mm'},
        body_and_seam_mesh={'seam_spacing_mm','local_spacing_mm','surface_spacing_mm','coarse_spacing_mm'})
    for family,keys in families.items():
        groups={}
        for case in cases:
            fixed=json.dumps({k:v for k,v in case[0]['config'].items() if k not in keys},sort_keys=True)
            groups.setdefault(fixed,[]).append(case)
        for group in groups.values():
            if len(group)<2:continue
            end=min(c[0]['last_depth_mm'] for c in group)
            for i,a in enumerate(group):
                for b in group[i+1:]:
                    comparisons.append(dict(family=family,first=a[0]['name'],second=b[0]['name'],common_window_mm=[0.,end],
                        **curve_difference(a[1],a[2],b[1],b[2],end)))
    bundle=dict(schema='softcut-finite-increment-comparison/1',loading_started=False,
        experimental_force_read=False,source=__file__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=[c[0] for c in cases],controlled_comparisons=comparisons,
        comparison_method='Exact squared integral and pointwise maximum of the plotted piecewise-linear interpolants on their union of knots; one family-wide common window; no event alignment; interval-average Fz, not an instantaneous-force reconstruction; not a convergence-order estimate')
    write_json(output/'comparison.json',bundle)
    def link(path):return quote(os.path.relpath(path,output).replace('\\','/'),safe='/')
    table=''
    for s,*_ in cases:
        e=s['evidence'];a=e['finite_audit'];p=e['path_audit']
        finite_status=('不适用' if s['config'].get('fracture_criterion')!='finite_increment' else
            '未附审查' if a is None else str(bool(a.get('passed') and a.get('primary_seed_increment_complete') and a.get('producer_source_matches_current'))))
        path_status='未附审查' if p is None else str(bool(p.get('passed') and p.get('producer_source_matches_current')))
        values=[s['name'],f"{s['last_depth_mm']:.6f} / {s['accepted_states']}",
            '-' if s['first_release_mm'] is None else f"{s['first_release_mm']:.6f}",
            f"{s['released_nodes']} / {s['fully_failed_nodes']}",
            f"{s['connected_fully_failed_reference_length_mm']:.4g}",
            f"{100*s['absolute_defect_over_absolute_work']:.5g}%",
            f"{s['maximum_drawdown_n']:.4g}",s['status'],
            finite_status,path_status]
        table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+f'<td><a href="{link(Path(s["directory"])/"result.json")}">JSON</a></td></tr>'
    comparison_table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        [r['family'],r['first']+' / '+r['second'],f"0–{r['common_window_mm'][1]:.6f}",
         f"{100*r['relative_l2']:.4g}%",f"{r['maximum_absolute_difference_n']:.4g}"] )+'</tr>' for r in comparisons)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>有限裂增量双条件：实际切割验证</title>
    <style>body{{font:16px system-ui;max-width:1400px;margin:32px auto;padding:24px;color:#172b36}}img{{max-width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:18px;background:#fff0d5}}table{{font-size:13px;width:100%}}pre{{white-space:pre-wrap}}a{{color:#126477}}</style>
    <h1>强度—能量双条件：实际刀具验证</h1>
    <p class="note">已实现冻结历史有限裂增量门控与动态内聚开口；下表逐项显示实际完成范围。
    未完成2 mm、未形成表面连通完全失效裂段、或未通过能量/时间/网格验证时，不能称为切割修复成功。
    内部孤立裂纹与表面切开分开统计，较平滑Fz不能替代裂纹几何证据。</p>
    <p>30×5×20 mm，居中0°镜像、预设直裂路；与原s=7 mm偏心基线不同。
    同一未测示例T=.2 MPa、Gc=.006 N/mm、ρ=1000 kg/m³、真实速度.2 mm/s；没有实验力反演、质量缩放或强制单调Fz。</p>
    <p>同一候选裂段须同时通过离散牵引强度与“冻结储能差−虚拟摩擦功≥Gc×参考面积”。
    虚拟场不覆盖真实历史，实际内聚功支付Gc一次。摩擦路径、节点面积、冻结历史与CC/CZM组合仍是待验证假设。</p>
    <p><a href="comparison.json">完整统计、配置和独立审查</a> · <a href="{link(Path(__file__).resolve().parents[3]/'docs/fracture_implementation.md')}">方法与理论边界（第22节）</a></p>
    <div style="overflow-x:auto"><table><tr><th>算例</th><th>末行程mm / 态数</th><th>首次释放mm</th><th>释放 / 全失效节点</th><th>表面连通全失效长度mm</th><th>Σ|缺口|/Σ|外功|</th><th>最大回落N</th><th>状态</th><th>双条件审查</th><th>真实轨迹审查</th><th>原始结果</th></tr>{table}</table></div>
    <img src="force_energy_time.png" alt="实际Fz、累计能量缺口与接受时间步">
    <h2>时间与网格敏感性：完整行程尚未通过</h2>
    <table><tr><th>控制族</th><th>配对</th><th>共同深度mm</th><th>曲线相对L2差</th><th>最大点差N</th></tr>{comparison_table}</table>
    <p>使用全部接受点的并集分段精确积分，保留极窄脉冲。小L2差仍可能伴随很大峰值差；
    未完整覆盖失稳事件和2 mm后缀时不能宣布时间收敛。逐释放/全失效事件±5 ms能量窗口与完整性另存JSON。</p>
    <img src="deformed_seam.png" alt="保存末态真实变形网格与未释放连接"><img src="seam_opening.png" alt="真实裂口与开口历史">
    <p>Fz为算法区间平均，按接受深度绘出，不做平滑或事件对齐。图中的不同末态不能作同深度裂路比较。
    表面连通全失效长度要求从原顶面起每条边的两个端点均完全失效；节点释放不等于整条裂面已经无牵引。
    峰动能±5 ms能量窗口以固定速度换算，窗口不完整时单独标false；它是诊断窗口，不是起裂定义。</p>
    <p>理论依据：<a href="https://doi.org/10.1016/S0997-7538(01)01184-6">Leguillon 2002</a>；
    <a href="https://doi.org/10.1016/j.finel.2019.103344">Doitrand等2020</a>；
    <a href="https://link.springer.com/article/10.1007/s10704-025-00866-y">含裂面摩擦的耦合准则2025</a>。
    原文的弹性/准静态论证不直接认证本项目的有限应变黏弹性快速失稳。</p></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    return bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();bundle=build(args.directories,args.output)
    print(json.dumps({'cases':len(bundle['cases']),'output':str(args.output.resolve()),'loading_started':False}))


if __name__=='__main__':main()
