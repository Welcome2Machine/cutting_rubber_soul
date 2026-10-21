"""Bounded structural diagnostics; no FE loading or parameter fit.

fields: verified frozen q30 histories, conditional implicit response to mixing
two constitutive residuals at FIXED previous history (not a full-path derivative).
fixtures: response restrictions and persistent smooth-surface history checks.
report: measured data and HTML, preserving exported experimental zeros.
"""
import argparse
import csv
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert,allow_nan=False),encoding='utf-8')


def table_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as s:
        w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def snapshot(out):
    files=[Path(__file__),ROOT/'run_support.py',ROOT/'tests/test_material_structure.py',
           ROOT/'tests/test_surface_history.py']
    files += [p for directory in ('mechanics','prototype','tests') for p in (ROOT/directory).rglob('*.py')]
    files += list((ROOT/'prototype').glob('*.json'))
    manifest={}
    for p in files:
        name=p.relative_to(ROOT);dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if sha(p)!=sha(dest):raise ValueError('Diagnostic source changed; use a new output: '+str(name))
        else:shutil.copy2(p,dest)
        manifest[str(name)]=sha(dest)
    save(out/'diagnostic_sources.json',manifest)


def progress(out,event,**data):
    row=dict(time=time.strftime('%Y-%m-%d %H:%M:%S %z'),event=event,**data)
    with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)
    log=(out/'events.jsonl').read_text(encoding='utf-8')
    (out/'run_status.html').write_text('<!doctype html><meta charset="utf-8"><title>结构核查运行记录</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px system-ui}pre{white-space:pre-wrap}</style>'
        '<h1>结构核查运行记录</h1><p>固定数量的材料/保存场探针；不推进FE加载。</p><pre>'+html.escape(log)+'</pre>',encoding='utf-8')


def contact_scales(p,x,out,label):
    """Actual persistent EV kinematics only; unsupported stencils stay counted."""
    lag=p.lag(x);sx=x[p.surface];old=p.previous[p.surface];eps=float(p.friction.eps_v)
    rows=[];excluded={};excluded_capacity=0.
    for kind in ('vv','ev'):
        for s in getattr(lag,kind+'_collisions'):
            ids=np.array([i for i in s.vertex_ids(p.cmesh.edges,p.cmesh.faces) if i>=0],int)
            capacity=abs(float(s.weight*s.normal_force_magnitude*s.mu_s_effective_lagged))
            reason=None
            if kind!='ev':reason='VV'
            elif s.weight<=0:reason='signed_weight'
            else:
                y,z=sx[ids],old[ids];e,e0=y[2]-y[1],z[2]-z[1];L,L0=np.linalg.norm(e),np.linalg.norm(e0)
                if min(L,L0)<1e-14:reason='degenerate'
                else:
                    b,b0=float((y[0]-y[1])@e/L**2),float((z[0]-z[1])@e0/L0**2)
                    if not (0<b<1 and 0<b0<1):reason='cross_edge_history'
            if reason:
                excluded[reason]=excluded.get(reason,0)+1;excluded_capacity+=capacity;continue
            t,t0=e/L,e0/L0;theta=float(np.arctan2(t0[0]*t[1]-t0[1]*t[0],t0@t))
            gap=float(np.linalg.norm(y[0]-y[1]-b*e))
            native=float(t@s.relative_velocity((y-z).ravel()));objective=L*(b-b0)
            rows.append(dict(nodes=' '.join(map(str,p.surface[ids])),master='blade' if np.all(p.surface[ids[1:]]>=p.n) else 'body',
                capacity_n=capacity,gap_mm=gap,rotation_rad=theta,rotation_scale_over_epsilon=gap*abs(theta)/eps,
                native_slip_mm=native,objective_slip_mm=objective,slip_difference_over_epsilon=abs(native-objective)/eps))
    if rows:table_csv(out/f'contact_scales_{label}.csv',rows)
    capacity=sum(r['capacity_n'] for r in rows)
    return dict(eligible_count=len(rows),unsupported_counts=excluded,epsilon_u_mm=eps,
        eligible_absolute_stencil_capacity_n=capacity,unsupported_absolute_stencil_capacity_n=excluded_capacity,
        max_rotation_scale_over_epsilon=max((r['rotation_scale_over_epsilon'] for r in rows),default=0.),
        max_actual_slip_difference_over_epsilon=max((r['slip_difference_over_epsilon'] for r in rows),default=0.),
        eligible_capacity_share_difference_above_epsilon=sum(r['capacity_n'] for r in rows if r['slip_difference_over_epsilon']>=1)/max(capacity,1e-30),
        caveat='Persistent positive EV subset. Absolute stencil capacity is NOT a macroscopic force share; kinematic difference is not an equilibrated Fz error.')


def fields(study,out):
    # No project import before the original physics snapshot is installed.
    sys.path.insert(0,str(ROOT/'.runtime/ipc-python'));sys.path.insert(0,str(study/'source'))
    from tools import run_common_material_study as host
    from prototype.material_case import signature,load_force_csv
    from prototype.pardiso import PardisoDirection
    from mechanics.finite_strain import response
    from threadpoolctl import threadpool_limits
    manifest=host.read(study/'manifest.json');host.check(manifest,study)
    inputs={};rows=[];curves=[];stats=[]
    names={0:'1008n0a2__Fz__curve_1.csv',33:'1008n33a1__Fz__curve_2.csv'}
    start=time.monotonic()
    with threadpool_limits(1):
        for angle in (0,33):
            folder=study/f'soft_q30_{angle}';inputs.update(host.verify_case(folder))
            report=host.read(folder/'result.json');p=host.make(angle,report['job']['parameters'])
            if json.loads(json.dumps(signature(p)))!=report['signature']:raise ValueError('Physical signature mismatch')
            exp=ROOT/'docs'/names[angle];inputs[str(exp)]=sha(exp)
            ed,ef,speed=load_force_csv(exp)
            if speed!=p.loading_rate:raise ValueError('Experiment rate mismatch')
            depths=np.array([r['depth_mm'] for r in report['steps']])
            forces=np.array([r['reaction_fz_n'] for r in report['steps']])
            target=np.interp(depths,ed,ef);anchor=np.argmin(abs(depths-.25))
            selected=(depths>=.25-1e-10)&(depths<=.5+1e-10)
            offset=forces-target;inc=(forces-forces[anchor])-(target-target[anchor])
            stats.append(dict(angle=angle,window_mm=[.25,.5],sample_count=int(selected.sum()),
                zero_fe_n=float(forces[0]),zero_experiment_n=float(target[0]),
                offset_mean_n=float(offset[selected].mean()),offset_range_n=[float(offset[selected].min()),float(offset[selected].max())],
                anchored_increment_rmse_n=float(np.sqrt(np.mean(inc[selected]**2))),
                anchored_increment_max_abs_error_n=float(np.max(abs(inc[selected]))),
                fe_increment_n=float(forces[-1]-forces[anchor]),experiment_increment_n=float(target[-1]-target[anchor])))
            for d,f,e in zip(depths,forces,target):
                curves.append(dict(angle=angle,depth_mm=float(d),fe_n=float(f),experiment_n=float(e),
                    fe_increment_from_025_n=float(f-forces[anchor]),experiment_increment_from_025_n=float(e-target[anchor])))
            for depth in (.25,.4,.5):
                if time.monotonic()-start>600:raise TimeoutError('600 s diagnostic budget; no retry')
                progress(out,'field_begin',angle=angle,depth_mm=depth)
                i=next(i for i,r in enumerate(report['steps']) if abs(r['depth_mm']-depth)<1e-10)
                with np.load(folder/'states'/f'{i:04d}.npz',allow_pickle=False) as z:state=dict(z)
                fresh=p.audit_state(state,report['steps'][i]);x=state['positions']
                p.previous=state['previous_positions'];p.previous_material_metric=state['previous_material_metric']
                p.material_dt=float(state['material_dt_s']);base=p.material
                F=np.eye(2)+np.einsum('eai,eaJ->eiJ',(x[:p.n]-p.rest[:p.n])[p.mesh.triangles],p.grad)
                q=p.constitutive_response(F,tangent=False)
                lag=p.lag(x);_,g,H,_=p.evaluate(x,lag,projected=False)
                A=p.coupled_friction_tangent(x,lag,H)
                ge=base.moduli()[0];gv=base.moduli(True)[0]
                candidates=dict(log_j=replace(base,volumetric='paper2023'),
                    eq_alpha1=replace(base,mu_eq=(2*ge,),alpha_eq=(1.,)),
                    branch_alpha1=replace(base,mu_ov=(2*gv,),alpha_ov=(1.,)))
                fz=lambda v:float(-v.reshape(-1,2)[p.n:,1].sum())
                def material_gradient(y,material):
                    f=np.eye(2)+np.einsum('eai,eaJ->eiJ',(y[:p.n]-p.rest[:p.n])[p.mesh.triangles],p.grad)
                    c=response(f,p.previous_material_metric,p.material_dt,material,tangent=False,kinematics='plane_strain')
                    local=np.einsum('e,eiJ,eaJ->eai',p.material_integration_weights(),c['P'],p.grad).reshape(-1,6)
                    v=np.zeros(p.total);np.add.at(v,p.mesh.bulk_data[2].ravel(),local.ravel())
                    return v,c
                gb,_=material_gradient(x,base);probes={}
                for name,material in candidates.items():
                    progress(out,'conditional_begin',angle=angle,depth_mm=depth,candidate=name)
                    try:
                        gc,rc=material_gradient(x,material);partial=gc-gb
                        solver=PardisoDirection(8,runtime=str(ROOT/'.runtime/pardiso'))
                        try:du=solver.solve(A[p.free][:,p.free],-partial[p.free]);calls=list(solver.rows)
                        finally:solver.close()
                        direction=np.zeros(p.total);direction[p.free]=du;action=A@direction+partial
                        relative=float(np.linalg.norm(action[p.free])/max(np.linalg.norm(partial[p.free]),1e-30))
                        if relative>1e-8:raise RuntimeError('Implicit linear residual failed')
                        derivative=fz(action);trials=[]
                        for eps in (1e-3,3e-4,1e-4,1e-5,1e-6):
                            change=eps*direction.reshape(-1,2);safe=float(p.safe_step(x,change))
                            if safe<1-1e-10:
                                trials.append(dict(mixture_fraction=eps,safe_fraction=safe,status='geometry_limited'));continue
                            y=x+change;_,gy,_,minj=p.evaluate(y,p.lag(y),tangent=False)
                            alt,_=material_gradient(y,material);original,_=material_gradient(y,base)
                            gy+=eps*(alt-original)
                            actual=(fz(gy)-fz(g))/eps
                            trials.append(dict(mixture_fraction=eps,status='evaluated',safe_fraction=safe,min_J=float(minj),
                                finite_difference_fz_n=actual,relative_derivative_error=abs(actual-derivative)/max(abs(derivative),1e-20),
                                normalized_free_increment_error=float(np.linalg.norm((gy-g)[p.free]))/(eps*max(np.linalg.norm(partial[p.free]),1e-30))))
                        checks=[t['status']=='evaluated' and t['relative_derivative_error']<.01
                            and t['normalized_free_increment_error']<.01 for t in trials]
                        passed=any(a and b for a,b in zip(checks,checks[1:]))
                        saved=out/f'conditional_{angle}_{i:04d}_{name}.npz'
                        np.savez_compressed(saved,direction=direction,partial_residual=partial,positions=x,
                            previous_positions=p.previous,previous_material_metric=p.previous_material_metric,material_dt_s=p.material_dt)
                        probes[name]=dict(status='verified_conditional_derivative' if passed else 'unqualified_derivative',
                            candidate_material=material.to_dict(),dFz_dlambda_n=derivative,linear_relative_residual=relative,
                            direction_infinity_mm=float(abs(direction).max()),fixed_field_free_residual_change_n=float(np.linalg.norm(partial[p.free])),
                            stress_change_relative=float(np.linalg.norm(rc['P']-q['P'])/np.linalg.norm(q['P'])),
                            min_candidate_dissipation=float(rc['viscous_rate'].min()),trials=trials,linear_calls=calls,
                            evidence_file=saved.name,evidence_sha256=sha(saved))
                    except (ValueError,RuntimeError,FloatingPointError) as exc:
                        probes[name]=dict(status='failed_probe',error=repr(exc))
                    progress(out,'conditional_end',angle=angle,depth_mm=depth,candidate=name,status=probes[name]['status'])
                rows.append(dict(angle=angle,depth_mm=depth,fresh_audit=fresh,fz_n=fz(g),min_J=float(q['J'].min()),
                    original_material=base.to_dict(),probes=probes,contact_scales=contact_scales(p,x,out,f'{angle}_{i:04d}')))
                save(out/'fields.json',dict(rows=rows,inputs=inputs,curve_statistics=stats,new_accepted_states=0,
                    scope='Conditional derivative of R_base + lambda*(R_alt-R_base), prior histories fixed; not a rerun or parameter fit.',
                    physical_manifest=str(study/'manifest.json'),physical_manifest_sha256=sha(study/'manifest.json')))
    table_csv(out/'curves.csv',curves)
    table_csv(out/'conditional_summary.csv',[dict(angle=r['angle'],depth_mm=r['depth_mm'],candidate=name,
        status=v['status'],dFz_dlambda_n=v.get('dFz_dlambda_n'),linear_relative_residual=v.get('linear_relative_residual'))
        for r in rows for name,v in r['probes'].items()])


def fixtures(out):
    sys.path.insert(0,str(out/'source'))
    from mechanics.finite_strain import Material,response,elastic_response
    from prototype.surface_history import SurfaceChart,surface_contact
    ge,gv,K=.2898,.6762,23.34333333333331
    base=Material((ge,),(2.,),(gv,),(2.,),(3*K-2*ge)/(2*(3*K+ge)),1.7,viscous_flow='isochoric_maxwell')
    def dev(f,q):
        s=np.zeros((len(f),3,3));s[:,:2,:2]=q['P']@f.transpose(0,2,1)
        s[:,2,2]=q['plane_stress_residual'];s/=q['J'][:,None,None]
        return s-np.trace(s,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3
    materials=dict(same_shape=base,eq_alpha1=replace(base,mu_eq=(2*ge,),alpha_eq=(1.,)),
        branch_alpha1=replace(base,mu_ov=(2*gv,),alpha_ov=(1.,)))
    rows=[];hold=[]
    f=np.array([[[v,0],[0,1/v]] for v in (1.1,1.5,2.,3.)]+[[[1.,1.],[0.,1.]],[[.8,.2],[0.,.7]]])
    for name,m in materials.items():
        old=np.tile(np.eye(3),(len(f),1,1));instant=response(f,old,1e-10,m,kinematics='plane_strain',tangent=False)
        equilibrium=elastic_response(f,m,kinematics='plane_strain');target=dev(f,equilibrium)
        for i,(a,b) in enumerate(zip(np.linalg.norm(target,axis=(1,2)),np.linalg.norm(dev(f,instant),axis=(1,2)))):
            rows.append(dict(candidate=name,deformation_id=i,fully_relaxed_to_instant_dev_norm_ratio=float(a/b)))
        for j in range(300):
            q=response(f,old,1.7/10,m,kinematics='plane_strain',tangent=False);old=q['metric']
            if j in (0,9,49,99,299):
                hold.append(dict(candidate=name,time_over_tau=(j+1)/10,
                    max_relative_error_to_equilibrium=float(np.linalg.norm(dev(f,q)-target)/np.linalg.norm(target)),
                    minimum_dissipation=float(q['viscous_rate'].min())))
    u=np.linspace(-.01,.01,5);chart=SurfaceChart(u);nodes=np.column_stack((u,.02-np.sqrt(.02**2-u*u)))
    def configuration(a,gap=.0015):
        c=chart.geometry(nodes);t=c(a,1);t/=np.linalg.norm(t)
        return np.vstack((c(a)+gap*np.array([-t[1],t[0]]),nodes))
    kw=dict(weight=.01,dhat=.0025,stiffness=132.,mu=.35,epsilon=1e-5)
    old=configuration(-.002);angle=.7;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    rigid=surface_contact(chart,old@Q.T+[.2,-.3],old,**kw)
    crossing=[]
    for h in (1e-5,1e-6,1e-7,1e-8):
        a=surface_contact(chart,configuration(-h),old,**kw);b=surface_contact(chart,configuration(h),old,**kw)
        crossing.append(dict(parameter_offset=h,total_force_difference_n=float(np.linalg.norm(a['gradient']-b['gradient']))))
    actual=surface_contact(chart,configuration(.001),old,**kw)
    save(out/'fixtures.json',dict(endpoint_ratios=rows,holds=hold,contact=dict(
        rigid_slip_mm=rigid['slip_mm'],rigid_friction_force_norm_n=float(np.linalg.norm(rigid['friction_gradient'])),
        cross_knot=crossing,curved_finite_gap_metric=actual['slave_tangential_metric'],
        complete_native_contact_qualified=False,scope=actual['scope'])))
    table_csv(out/'endpoint_ratios.csv',rows)


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((out/'fields.json').read_text(encoding='utf-8'))
    fixture=json.loads((out/'fixtures.json').read_text(encoding='utf-8'))
    # Rendering can evolve without rewriting the measurement snapshot. It is
    # separately hashed and cannot advance or resume any physical state.
    frozen=json.loads((out/'diagnostic_sources.json').read_text(encoding='utf-8'))
    for name,digest in frozen.items():
        if sha(out/'source'/name)!=digest:raise ValueError('Measurement source changed: '+name)
    for name,digest in data['inputs'].items():
        if sha(name)!=digest:raise ValueError('Measurement input changed: '+name)
    for row in data['rows']:
        for probe in row['probes'].values():
            if 'evidence_file' in probe and sha(out/probe['evidence_file'])!=probe['evidence_sha256']:
                raise ValueError('Conditional direction changed')
    with (out/'curves.csv').open(encoding='utf-8-sig') as f:curves=list(csv.DictReader(f))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for k,angle in enumerate((0,33)):
        c=[r for r in curves if int(r['angle'])==angle]
        d=np.array([float(r['depth_mm']) for r in c]);f=np.array([float(r['fe_n']) for r in c]);e=np.array([float(r['experiment_n']) for r in c])
        ax[0,k].plot(d,e,color='#0072B2',label='Experiment, original zero')
        ax[0,k].plot(d,f,'--',color='#D55E00',label='q30 FE, accepted states')
        sel=d>=.25-1e-10;a=np.argmin(abs(d-.25))
        ax[1,k].plot(d[sel],e[sel]-e[a],color='#0072B2',label='Experiment')
        ax[1,k].plot(d[sel],f[sel]-f[a],'--',color='#D55E00',label='q30 FE')
        ax[0,k].set(title=f'{angle} degrees: absolute force',ylabel='Fz (N)',xlabel='Vertical depth (mm)')
        ax[1,k].set(title=f'{angle} degrees: increase after 0.25 mm',ylabel='Fz - Fz(0.25 mm) (N)',xlabel='Vertical depth (mm)')
    for a in ax.ravel():a.grid(alpha=.2);a.legend(fontsize=9);a.set_ylim(bottom=0)
    for row in ax:
        upper=max(a.get_ylim()[1] for a in row)
        for a in row:a.set_ylim(0,upper)
    fig.savefig(out/'force_structure.png',dpi=180);fig.savefig(out/'force_structure.svg');plt.close(fig)
    def celltable(headers,rows):
        return '<table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    stats=celltable(['角度','FE零深度 / N','0.25–0.5偏差均值 / N','增量RMSE / N','FE增量 / N','实验增量 / N'],
        [[s['angle']]+[f"{s[k]:.6f}" for k in ('zero_fe_n','offset_mean_n','anchored_increment_rmse_n','fe_increment_n','experiment_increment_n')] for s in data['curve_statistics']])
    measured=celltable(['角度','深度/mm','候选','条件 dFz/dλ / N','导数检查'],[[r['angle'],r['depth_mm'],name,
        f"{v['dFz_dlambda_n']:.6f}" if 'dFz_dlambda_n' in v else '未求得',v['status']] for r in data['rows'] for name,v in r['probes'].items()])
    scales=celltable(['角度','深度/mm','持续EV数量','最大g|Δθ|/ε','最大|滑移差|/ε','差值≥ε的子集承载权重占比'],
        [[r['angle'],r['depth_mm'],r['contact_scales']['eligible_count'],
          f"{r['contact_scales']['max_rotation_scale_over_epsilon']:.3f}",
          f"{r['contact_scales']['max_actual_slip_difference_over_epsilon']:.3f}",
          f"{100*r['contact_scales']['eligible_capacity_share_difference_above_epsilon']:.2f}%"] for r in data['rows']])
    contact=fixture['contact']
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>模型结构与实际承载场核查</title>
<style>body{max-width:1150px;margin:32px auto;padding:0 20px;font:17px/1.7 system-ui;color:#17202a}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd3d8;padding:7px;text-align:right}th{background:#edf3f7}img{width:100%}.note{border-left:4px solid #b35b00;background:#fff6e9;padding:12px}code{overflow-wrap:anywhere}</style>
<h1>模型结构与实际承载场核查 · 2026-10-21</h1>
<p>范围：共同材料、完整体压入、0°/33° q30 已接受场；不拟合实验、不新增FE加载。57°仍受完整接触切换资格限制。</p>
<p class="note">本轮判断：优先补齐接触运动学与跨特征历史，暂不继续材料拟合。全宽贴合已确认；0°后段形状接近不等于33°形状已解决。三个材料方向的局部导数均未朝纠正角度排序的方向变化，但这不能排除重新加载后的材料历史效应。</p>
<p>实验布置新依据：用户确认前0.5 mm刀刃全宽紧密贴合。部分宽度尚未接触降为低优先级，不采用有效宽度缩放。沿宽度压力均匀性与三维端部释放仍未确认。</p>
<p><a href="run_status.html">运行记录HTML</a> · <a href="fields.json">实际场/条件导数JSON</a> · <a href="fixtures.json">材料与接触夹具JSON</a> · <a href="curves.csv">未平移曲线及分析增量CSV</a> · <a href="conditional_summary.csv">条件导数CSV</a> · <a href="diagnostic_sources.json">诊断源码哈希</a> · <a href="verification.json">额外签名/回归检查</a></p>
<h2>1. 起点与形状不是同一个问题</h2><img src="force_structure.png" alt="0与33度实验和q30绝对力以及0.25mm以后增量"/>
<p>下排仅把0.25 mm作为分析锚点，不修改上排原始零点，不是修正后的预测。0°后段增量接近实验，33°仍明显偏大。FE零深度仅约0.04 N，不能把约1.3–1.7 N的后段差值直接称为模拟初始预载。</p>'''+stats+'''
<h2>2. 共同材料的结构限制</h2>
<p>同形NH平衡/黏性支路在无初始历史、瞬时施加并保持完整F、充分松弛条件下，偏应力端点比例等于G∞/G0。材料点覆盖剪切、等容伸长与压缩剪切；不同支路形状能解除固定比例，但这不是当前橡胶必须使用该模型的证据。</p>
<p>新增显式log_j体积律选项 U=K/4(J²−1−2lnJ)，与原quadratic_j具有相同小应变K，且保持等容Maxwell流动。它复用已有体积公式，不切换到论文2023的黏性流动。默认仍为quadratic_j；历史签名包含体积律。二次J体积压力有界不等于平面应变总能量在J→0时有界。</p>
<h2>3. 实际场的条件灵敏度</h2>
<p>每个场先用原冻结源码与准确双历史重新验收。然后取完整非对称物理切线，求 R_base+λ(R_alt−R_base) 在λ=0的局部平衡方向；自由DOF线性残差和五个微扰尺度重新装配校验，至少相邻两个尺度通过1%检查才标为通过。无PTC移位、无残差投影。最小扰动可能被接触反力差分噪声污染，全部数值保留。</p>
<p class="note">这些是旧历史固定时的条件导数，单位为N/混合系数。λ不是体积模量或Ogden指数；不能外推λ=1，不能当作替代材料的加载曲线。log_j行在原冻结代码中使用等价旧名称paper2023，但黏性流动仍为isochoric_maxwell。</p>'''+measured+'''
<h2>4. 接触修改的实际进展与边界</h2>
<p>下表使用实际相邻接受态，不是任意附加的刚体转动。仅统计正权重、旧新投影仍在同一边内部的EV。原生位移投影与材料坐标滑移的差包含转动与变形运输差异，不能把差值全部解释成纯转动伪滑移。VV、跨边和带符号项保留未覆盖标记；权重占比不是整把刀的Fz误差份额。</p>'''+scales+'''
<p>实际滑移差达到摩擦平滑尺度ε=10⁻⁵ mm的数倍，说明运动学差异在当前加载步中不能仅以“间隙很小”忽略。仍需完整一致的接触模型重新平衡，才可求它对Fz和角度差的因果影响。</p>
<p>新增独立C²材料表面坐标夹具，跨样条节点时保持材料参数，解析滑移梯度和功共轭力通过有限差分；共同刚体转动、总力和总力矩检查通过。内部节点不是额外的投影约束，投影搜索已排除这类虚假并列候选。</p>
<p>该夹具不是原生EV/VV修复：改变了表面表示，仅覆盖持续接触和正权重；未覆盖激活/脱离/再接触、自接触、完整FE切线和原P1几何等价性，禁止接入正式路径。</p>
<p>在弯曲有限间隙夹具中，功共轭滑移力与点切向库仑力存在曲率度量差异。需验证其间隙极限，不能因客观性通过就宣称完全物理等价。</p>
<pre>'''+html.escape(json.dumps(contact,ensure_ascii=False,indent=2))+'''</pre>
<h2>5. 文献依据与阅读范围</h2>
<p><a href="https://arxiv.org/html/1808.10420v2">Duong &amp; Sauer (2019)</a>：本轮核读引言及第3.1–3.2节，支持表面参数、前一相互作用点、跨单元历史的统一处理；本地夹具不是该论文完整摩擦锥算法的复现。</p>
<p><a href="https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-hyperelastic.htm">Abaqus超弹材料官方说明</a>：本轮核读可压缩性部分，支持平面应变受限区对体积响应敏感、需区别体积与剪切参数；不提供本项目CR参数识别。</p>
<p>因果边界：接触客观性缺口已有夹具证据，但对实际Fz偏差的份额未定；本构结构有限制，但缺少独立材料多模式与松弛数据；全宽贴合排除不了二维约束差异。没有把任一项宣布为唯一源头。</p></html>'''
    if (out/'regressions.log').exists():
        log=(out/'regressions.log').read_text(encoding='utf-8-sig')
        text=text.replace('</html>','<h2>6. 相关回归记录</h2><details><summary>测试原始输出</summary><pre>'+html.escape(log)+'</pre></details></html>')
    (out/'index.html').write_text(text,encoding='utf-8')
    save(out/'figure_manifest.json',dict(destination='Local research HTML, not a journal submission',
        figure_size_inches=[11,7],dpi=180,matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
        inputs={n:sha(out/n) for n in ('fields.json','fixtures.json','curves.csv')},
        transformations='Experiment interpolated to accepted FE depth grid. No smoothing or zero shift. Lower panels subtract each own force at 0.25 mm; analysis only.',
        replicates='One authorized trial per angle, no uncertainty estimate',
        figures={n:sha(out/n) for n in ('force_structure.png','force_structure.svg')}))


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('mode',choices=('fields','fixtures','report'))
    ap.add_argument('--study',type=Path,default=ROOT/'outputs/material_uncertainty_20261021')
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve()
    if a.mode in ('fields','fixtures') and (out/(a.mode+'.json')).exists():
        raise FileExistsError('Do not overwrite measured evidence; choose a new independent output')
    out.mkdir(parents=True,exist_ok=True)
    if a.mode=='report':
        rendering=out/'report_source';rendering.mkdir(exist_ok=True)
        shutil.copy2(__file__,rendering/Path(__file__).name)
        save(rendering/'manifest.json',{Path(__file__).name:sha(__file__),
            'scope':'Presentation only; original diagnostic sources and physical inputs verified separately'})
    else:snapshot(out)
    progress(out,a.mode+'_started')
    try:
        if a.mode=='fields':fields(a.study.resolve(),out)
        elif a.mode=='fixtures':fixtures(out)
        else:report(out)
    except Exception as exc:
        progress(out,a.mode+'_failed',error=repr(exc),traceback=traceback.format_exc());raise
    progress(out,a.mode+'_completed')


if __name__=='__main__':main()
