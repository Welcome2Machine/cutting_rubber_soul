"""Read-only, audited 33-degree B/C interim review; writes offline HTML.

No FE imports, loading steps, calibration or queue mutations. Consumes completed
matched-normal paths and compares identical reference surface material nodes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.render_solver_dashboard import atomic_write, shell, stamp


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_files(expected):
    for name, value in expected.items():
        if digest(name) != value:
            raise ValueError('Evidence hash mismatch: '+name)


def pair_contract(b, c):
    for key in ('config','material','mesh_sha256','loading_rate','contact_formulation',
                'dhat','physical_force_tolerance_n','kinematics','quadrature_layout'):
        if b['signature'][key] != c['signature'][key]:
            raise ValueError('B/C physical/discretization mismatch: '+key)
    for key in ('angle','radius','order','increment'):
        if b['settings'][key] != c['settings'][key]:
            raise ValueError('B/C settings mismatch: '+key)
    if b['settings']['mode'] != 'normal' or c['settings']['mode'] != 'oblique':
        raise ValueError('Expected B normal and C oblique')
    qb=np.array([x['depth_mm'] for x in b['steps']])
    qc=np.array([x['depth_mm'] for x in c['steps']])
    if qb.shape!=qc.shape or not np.allclose(qb,qc,rtol=0,atol=1e-12):
        raise ValueError('Accepted time/depth grids differ; do not silently interpolate')
    if abs(qb[0])>1e-12 or abs(qb[-1]-.1)>1e-12 or np.any(np.diff(qb)<=0):
        raise ValueError('Require complete monotone q=0--.1 paths')
    np.testing.assert_allclose([x['material']['dt_s'] for x in b['steps']],
                               [x['material']['dt_s'] for x in c['steps']],rtol=0,atol=1e-12)


def forces(b,c,angle):
    theta=np.deg2rad(angle);co,si=np.cos(theta),np.sin(theta)
    rows=[]
    for x,y in zip(b['steps'],c['steps']):
        q=x['depth_mm'];mb,mc=x['motion'],y['motion']
        nb,nc=mb['normal_reaction_n'],mc['normal_reaction_n']
        tb,tc=mb['tangent_resistance_n'],mc['tangent_resistance_n']
        for s in (x,y):
            m=s['motion'];N=m['normal_reaction_n'];T=m['tangent_resistance_n']
            np.testing.assert_allclose(s['reaction_fz_n'],N*co+T*si,rtol=0,atol=1e-11)
            np.testing.assert_allclose(s['reaction_x_n'],N*si-T*co,rtol=0,atol=1e-11)
            np.testing.assert_allclose(m['actuator_power_nmm_s'],
                N*m['normal_speed_mm_s']+T*m['tangent_speed_mm_s'],rtol=0,atol=1e-11)
        dN=nc-nb;dT=tc-tb;df=y['reaction_fz_n']-x['reaction_fz_n']
        np.testing.assert_allclose(df,dN*co+dT*si,rtol=0,atol=1e-11)
        row=dict(q_mm=q,b_N_n=nb,c_N_n=nc,b_T_n=tb,c_T_n=tc,b_Fz_n=x['reaction_fz_n'],
            c_Fz_n=y['reaction_fz_n'],delta_N_fraction=dN/nb if q>0 and nb>.1 else None,
            delta_Fz_n=df,delta_N_projection_n=dN*co,delta_T_projection_n=dT*si,
            b_minJ=x['min_j'],c_minJ=y['min_j'],
            b_tip_offset_mm=x['tip_contact'].get('reference_offset_mm') if x['tip_contact']['status']=='unique' else None,
            c_tip_offset_mm=y['tip_contact'].get('reference_offset_mm') if y['tip_contact']['status']=='unique' else None)
        for prefix,s in (('b',x),('c',y)):
            footprint=s['blade_contact_footprint']
            row[prefix+'_support_reference_mm']=footprint['supported_reference_length_mm']
            row[prefix+'_support_current_mm']=footprint['supported_current_length_mm']
        rows.append(row)
    return rows


def surface_motion(first,second,b,c,angle,*,preparation_metric_tolerance=2e-12):
    theta=np.deg2rad(angle);es=np.array([np.cos(theta),-np.sin(theta)])
    en=np.array([np.sin(theta),np.cos(theta)])
    rows=[];end=None;shared={}
    with np.load(first/'states/0000.npz') as z:
        reference=z['reference_xy'][:int(z['base_body_nodes'])]
    ids=np.flatnonzero(np.isclose(reference@en,b['signature']['config']['height_mm'],atol=1e-10))
    s=reference[ids]@es-b['signature']['config']['drop_s_mm']
    order=np.argsort(s);ids=ids[order];s=s[order]
    for i,(xb,xc) in enumerate(zip(b['steps'],c['steps'])):
        with np.load(first/'states'/f'{i:04d}.npz') as zb,np.load(second/'states'/f'{i:04d}.npz') as zc:
            np.testing.assert_array_equal(zb['reference_xy'],zc['reference_xy'])
            ub=zb['positions'][ids]-reference[ids];uc=zc['positions'][ids]-reference[ids]
            if i==0:
                shared=dict(max_position_difference_mm=float(np.max(abs(zb['positions']-zc['positions']))),
                    max_material_metric_difference=float(np.max(abs(zb['material_metric']-zc['material_metric']))))
                if shared['max_position_difference_mm']>2e-12 or shared['max_material_metric_difference']>preparation_metric_tolerance:
                    raise ValueError('Different zero-depth preparation')
        difference=uc-ub;norm=np.linalg.norm(difference,axis=1)
        row=dict(q_mm=xb['depth_mm'],top_nodes=len(ids))
        for name,sel in (('all',np.ones(len(ids),dtype=bool)),('near1',abs(s)<=1),('far1',abs(s)>1)):
            row[name+'_max_mm']=float(norm[sel].max())
            row[name+'_node_rms_mm']=float(np.sqrt(np.mean(norm[sel]**2)))
        rows.append(row)
        end=dict(node_id=ids,s_mm=s,b_us_mm=ub@es,c_us_mm=uc@es,b_un_mm=ub@en,c_un_mm=uc@en,
                 delta_us_mm=difference@es,delta_un_mm=difference@en)
    return rows,end,shared


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def figure(out,rows,surface,end,angle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blue,orange,green='#0072B2','#D55E00','#007864'
    q=np.array([x['q_mm'] for x in rows])
    def arr(key):return np.array([np.nan if x[key] is None else x[key] for x in rows])
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'path','axes.grid':True,'grid.alpha':.18}):
        fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for a,key,title,unit in ((ax[0,0],'N','A  Normal reaction','N (N)'),
                                 (ax[0,1],'T','B  Tangential reaction','T (N)'),
                                 (ax[0,2],'Fz','C  Laboratory vertical force','Fz (N)')):
            a.plot(q,arr('b_'+key+'_n'),'-o',color=blue,ms=3,label=f'B{angle}: normal only')
            a.plot(q,arr('c_'+key+'_n'),'--s',color=orange,ms=3,label=f'C{angle}: normal + tangential')
            a.set(title=title,xlabel='Nominal normal depth q (mm)',ylabel=unit,ylim=(0,None));a.legend(fontsize=8)
        a=ax[1,0]
        for key,label,style,color in (('delta_Fz_n','Delta Fz','-',green),
                                      ('delta_N_projection_n','Delta N cos(theta)',':',blue),
                                      ('delta_T_projection_n','Delta T sin(theta)','--',orange)):
            a.plot(q,arr(key),style,color=color,label=label)
        a.set(title='D  B-to-C force difference',xlabel='Normal depth q (mm)',ylabel='Force difference (N)');a.legend(fontsize=8)
        a=ax[1,1]
        for prefix,label,style,color in (('b','B','-',blue),('c','C','--',orange)):
            a.plot(end['s_mm'],1000*end[prefix+'_us_mm'],style,color=color,label=label+' tangential motion')
        a.set(title='E  Full original top surface at q=0.1',xlabel='Reference distance s from initial contact (mm)',ylabel='Material tangential displacement (um)');a.legend(fontsize=8)
        a=ax[1,2]
        for prefix,style,color in (('b','-',blue),('c','--',orange)):
            a.plot(q,arr(prefix+'_minJ'),style,color=color,label=prefix.upper()+' min J')
        a.set(title='F  Local deformation indicator',xlabel='Normal depth q (mm)',ylabel='min J',ylim=(0,1.05));a.legend(fontsize=8)
        fig.suptitle(f'{angle} deg paired intact paths: identical normal depth and speed; numerical controls pending',fontsize=13)
        fig.savefig(out/'paired_response.png',dpi=150,facecolor='white');plt.close(fig)
        fig,ax=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
        for prefix,label,style,color in (('b','B','-',blue),('c','C','--',orange)):
            ax[0].plot(end['s_mm'],1000*end[prefix+'_un_mm'],style,color=color,label=label)
            ax[1].plot(q,1000*arr(prefix+'_tip_offset_mm'),style+'o',ms=3,color=color,label=label)
            ax[2].plot(q,1000*arr(prefix+'_support_current_mm'),style,color=color,label=label+' current length')
        ax[0].set(title='G  Near-contact normal surface motion',xlabel='Reference s (mm)',ylabel='Outward normal displacement (um)',xlim=(-.5,.5))
        ax[1].set(title='H  Pole identity (q < 0.02 unavailable)',xlabel='Normal depth q (mm)',
                  ylabel='Reference offset (um)',xlim=(0,.1))
        ax[2].set(title='I  Geometric proximity support',xlabel='Normal depth q (mm)',ylabel='Current supported surface length (um)',ylim=(0,None))
        for a in ax:a.legend(fontsize=9)
        fig.suptitle('Surface displacement is not accumulated slip; proximity support is not measured contact area',fontsize=12)
        fig.savefig(out/'surface_contact.png',dpi=150,facecolor='white');plt.close(fig)
    return {'matplotlib':matplotlib.__version__,'numpy':np.__version__,'format':'PNG','dpi':150,
        'purpose':'internal HTML mechanism review; no journal target','smoothing':'none',
        'uncertainty':'none; deterministic paths, controls pending',
        'surface_rms':'node-weighted on nonuniform trace, not area-weighted',
        'missing_tip':'NaN gaps; never connected across missing observations'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--study',type=Path,required=True)
    # The interpretation below belongs to the observed shallow 33-degree regime.
    # Do not reuse it automatically for a different angle or a deeper path.
    p.add_argument('--angle',type=int,choices=(33,),required=True);args=p.parse_args()
    study=args.study.resolve();angle=args.angle;out=study/f'pair{angle}_review'
    out.mkdir(exist_ok=True)
    state=read(study/'queue_status.json');manifest=read(study/'manifest.json')
    names=('A0_base',f'B{angle}_base',f'C{angle}_base');jobs={j['name']:j for j in state['jobs']}
    sources={str(Path(manifest['source'])/k):v for k,v in manifest['source_sha256'].items()}
    verified=dict(sources);verified.update(manifest['input_sha256']);results={};counts={}
    for name in names:
        j=jobs[name]
        if not (j.get('mechanics_completed') and j.get('evidence_verified')):
            raise ValueError('Await terminal audited path: '+name)
        folder=study/name;r=read(folder/'result.json');audit=read(folder/'audit.json')
        if not r['completed'] or not audit['passed'] or audit['states']!=len(r['steps']):
            raise ValueError('Terminal audit incomplete')
        evidence=read(folder/'evidence_hashes.json');verified.update(evidence)
        counts[name]=len(r['steps']);results[name]=r
        for x,y in zip(r['steps'],audit['rows']):
            if not y['passed']:raise ValueError('Failed accepted-state audit')
            for key in ('depth_mm','reaction_fz_n','min_j'):
                np.testing.assert_allclose(x[key],y[key],rtol=0,atol=1e-10)
    verify_files(verified)
    b,c=results[names[1]],results[names[2]];pair_contract(b,c)
    rows=forces(b,c,angle)
    surf,end,shared=surface_motion(study/names[1],study/names[2],b,c,angle)
    energies={}
    for name,r in (('B',b),('C',c)):
        steps=r['steps'];first,last=steps[0],steps[-1]
        energies[name]=dict(input_work_trapezoid_nmm=float(np.trapezoid(
            [s['motion']['generalized_resistance_n'] for s in steps],[s['depth_mm'] for s in steps])),
            stored_energy_increase_nmm=last['material']['stored_energy_n_mm']-first['material']['stored_energy_n_mm'],
            material_dissipation_postzero_nmm=last['material']['cumulative_viscous_dissipation_n_mm']-first['material']['cumulative_viscous_dissipation_n_mm'],
            friction_incremental_sum_postzero_nmm=sum(x['friction_work_n_mm'] for x in steps[1:]))
    e=rows[-1];projection_fraction=e['delta_T_projection_n']/e['delta_Fz_n']
    nchanges=[r['delta_N_fraction'] for r in rows if r['delta_N_fraction'] is not None]
    rreport=read(study/'study_report.json');controls=[x for x in rreport.get('controls',[]) if x['case'] in (f'B{angle}',f'C{angle}')]
    if max(abs(x) for x in nchanges)>.02 or e['delta_Fz_n']<=0 or controls:
        raise ValueError('New regime/control evidence: revise interim interpretation before regenerating report')
    result=dict(observed_jst=stamp(),angle=angle,decision='continue_existing_bounded_queue',
        reason='Useful force-projection and material-motion contrast; no proven solver defect or shear-induced normal unloading',
        sources_checked=len(sources),frozen_inputs_checked=len(manifest['input_sha256']),
        unique_file_hashes_checked=len(verified),terminal_audited_states=counts,
        audit_scope='Existing fresh FE audits rechecked against files; this analysis does not reassemble FE equations',
        shared_preparation=shared,endpoint=e,normal_change_fraction_range=[min(nchanges),max(nchanges)],
        endpoint_tangent_projection_fraction_of_delta_fz=projection_fraction,
        surface=surf,energy=energies,force_rows=rows,
        pair_controls_completed=len(controls),pair_controls_passed=len(controls)==6 and all(x['passed'] for x in controls),
        experimental_validation=False,no_automatic_depth_extension=True)
    write_csv(out/'force_comparison.csv',rows);write_csv(out/'surface_comparison.csv',surf)
    write_csv(out/'endpoint_surface.csv',[{key:(int(value[i]) if key=='node_id' else float(value[i]))
        for key,value in end.items()} for i in range(len(end['s_mm']))])
    result['figure_provenance']=figure(out,rows,surf,end,angle)
    result['analysis_code_sha256']={str(Path(__file__).resolve()):digest(__file__)}
    # Include every consumed result and checkpoint in the reproducible evidence list.
    atomic_write(out/'evidence_checked.json',json.dumps(verified,indent=2))
    atomic_write(out/'report.json',json.dumps(result,ensure_ascii=False,indent=2))
    bN,cN=e['b_N_n'],e['c_N_n'];bT,cT=e['b_T_n'],e['c_T_n'];bF,cF=e['b_Fz_n'],e['c_Fz_n']
    sr=surf[-1];Wb=energies['B']['input_work_trapezoid_nmm'];Wc=energies['C']['input_work_trapezoid_nmm']
    table=''.join(f'<tr><td>{label}</td><td>{v1:.6f}</td><td>{v2:.6f}</td></tr>' for label,v1,v2 in (
        ('法向反力 N / N',bN,cN),('切向反力 T / N',bT,cT),('竖直力 Fz / N',bF,cF),
        ('局部 min J',e['b_minJ'],e['c_minJ']),('输入功近似积分 / N·mm',Wb,Wc),
        ('参考支撑长度 / mm',e['b_support_reference_mm'],e['c_support_reference_mm']),
        ('当前支撑长度 / mm',e['b_support_current_mm'],e['c_support_current_mm'])))
    body=f'''<header><span class="pill">{angle}° · 匹配法向历史对照</span><h1>建议继续：已发现切向承载，尚未证明实验偏差改善</h1>
<p class="sub">核查时间：{result['observed_jst']} · 仅使用已完成并审计的路径</p><nav><a href="../status.html">实时状态</a><a href="../analysis.html">理论与计算方案</a><a href="report.json">完整数值证据</a></nav></header>
<section><h2>决定与理由</h2><p>保留既定队列，继续 57°成对路径和半步、积分、富集范围控制；本轮不修改本构、摩擦或求解器，也不自动加深。
{angle}°整条 0–0.1 mm 路径中，C 相对 B 的法向反力变化为 {100*min(nchanges):.3f}%–{100*max(nchanges):.3f}%，
没有出现明显法向减载；切向承载与表面运动却明显改变。这是有用的机制辨别结果。</p>
<div class="notice">目前成对数值控制完成 {len(controls)}/6。约 1% 的 N 差异低于预声明的单路径 2% 力门槛，不能据此认定物理增刚。
该 2% 是尚待核验的工程门槛，不是已经测得的误差条。min J 下降也需要控制核查。</div></section>
<section><h2>同一法向深度 q=0.1 mm 的结果</h2><div class="table"><table><thead><tr><th>量</th><th>B：纯法向</th><th>C：法向＋切向</th></tr></thead><tbody>{table}</tbody></table></div>
<p>N 仅增加 {100*(cN/bN-1):.3f}%，Fz 增加 {100*(cF/bF-1):.2f}%。
ΔFz={e['delta_Fz_n']:.6f} N，其中 ΔN·cosθ={e['delta_N_projection_n']:.6f} N，ΔT·sinθ={e['delta_T_projection_n']:.6f} N，
后者占差值 {100*projection_fraction:.2f}%。这是精确坐标分解，不是独立摩擦/材料因果贡献。</p>
<p>切向运动没有使本模型在该浅层窗口显著法向卸载，而是牵动材料并增加切向反力。
它提示“法向压入减小带来的降力，被切向承载部分抵消”可以作为弱角度效应的候选解释；仍需 57°及数值控制检验。</p>
<img src="paired_response.png" style="width:100%" alt="33度成对路径的N几乎重合，C的T和Fz增加；全顶面切向位移增加，C的局部min J更低。">
<p class="sub">曲线显示所有21个接受态，无平滑。q是初始表面法向深度，非实验室下行深度；不是同一实验室位移下的角度比较。</p></section>
<section><h2>确实改变了材料运动，不只是换坐标</h2><p>比较同一原始顶面上的 {sr['top_nodes']} 个材料点：端点 C−B 位移差最大 {1000*sr['all_max_mm']:.2f} μm；
即使在距初触点 |s|&gt;1 mm 的远区，最大差仍为 {1000*sr['far1_max_mm']:.2f} μm。
整顶面的节点等权 RMS 为 {1000*sr['all_node_rms_mm']:.2f} μm；非均匀采样下它不代表面积平均。</p>
<p>刀尖最低点对应的参考位置在端点分别为 {1000*e['b_tip_offset_mm']:.2f}/{1000*e['c_tip_offset_mm']:.2f} μm；
q&lt;0.02 mm 时两组均无唯一刀尖身份，保留缺失。参考点迁移与表面位移都不能直接称为累计界面滑程。</p>
<p>端点参考支撑长度变化 {100*(e['c_support_reference_mm']/e['b_support_reference_mm']-1):.2f}%，
而当前构型支撑长度变化 {100*(e['c_support_current_mm']/e['b_support_current_mm']-1):.2f}%。
参考长度与当前长度不能混用；这些区间只是距刀具小于 IPC dhat 的几何支撑，不是实测接触面积。
B端点的微小支撑断续也不能直接解读为真实脱离。</p>
<img src="surface_contact.png" style="width:100%" alt="近刀尖法向材料运动、带缺失值的刀尖参考位置，以及当前几何支撑长度对照。"></section>
<section><h2>功、局部变形与计算成本</h2><p>输入功近似积分增加 {100*(Wc/Wb-1):.2f}%；材料储能与黏性耗散也增加，说明总承载改变伴随真实场响应。
该功采用接受态梯形积分，含q=0速度切换端点误差；黏性耗散是后向Euler端点积分，不能据此声称精确能量平衡。</p>
<p>B/C min J 从 {e['b_minJ']:.4f} 变为 {e['c_minJ']:.4f}；不能因N近似相同就说局部场相同。
更低J提示需要重点检查C的时间与空间控制，但没有形成负J或已证明的求解器缺陷。</p>
<p>B/C本段续跑连同导出/审计分别 {jobs[names[1]]['wall_elapsed_s']/60:.1f}/{jobs[names[2]]['wall_elapsed_s']/60:.1f} 分钟，
均在原预算内，未出现因成本必须终止的证据。这不保证更深加载仍有同样速度。</p></section>
<section><h2>尚未解决的问题与下一步判据</h2><p>这是一组受控诊断，不是改进模型后重新贴合实验。
法向速度0.1 mm/s、不同实验室轨迹及仅0.1 mm深度，均不同于原始实验条件；不把本轮Fz变化叫作实验误差降低，也不能解释全部1–2 mm趋势。</p>
<p>下一步看57°是否仍主要增加T，或出现明显N变化；再用六条成对控制检验N/T/Fz与整顶面运动。
控制失败先辨别时间/积分/空间来源；只有结果足以支撑机理才考虑加深。
尚无证据支持现在按角度调材料、补水平力或改摩擦来拟合。</p></section>
<section><h2>证据与复现</h2><p>A0/B/C共 {sum(counts.values())} 个接受态的原终态重新组装审查通过，本轮复核审查与保存记录、全部检查点哈希；
另核对 {len(sources)} 份冻结源与 {len(manifest['input_sha256'])} 项冻结输入。共 {len(verified)} 个唯一文件哈希一致。
这次分析没有再次求平衡，也没有改写原始结果。共同零深度最大位置差 {shared['max_position_difference_mm']:.3g} mm，材料内变量差 {shared['max_material_metric_difference']:.3g}。</p>
<p><a href="force_comparison.csv">完整反力对照</a> · <a href="endpoint_surface.csv">端点全部材料点</a> · <a href="surface_comparison.csv">全路径表面差异</a> · <a href="evidence_checked.json">哈希清单</a></p>
<p><code>python -m tools.analyze_angle_path_pair --study outputs/angle_paths_20261018 --angle {angle}</code></p></section>'''
    atomic_write(out/'index.html',shell(f'{angle}°成对路径分析',body))
    print(json.dumps({k:result[k] for k in ('observed_jst','decision','endpoint','normal_change_fraction_range',
        'endpoint_tangent_projection_fraction_of_delta_fz','unique_file_hashes_checked')},ensure_ascii=False))


if __name__=='__main__':main()
