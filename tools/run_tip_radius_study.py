"""Bounded common-radius 0/33-degree study using the intact native IPC baseline.

Only nose radius changes. The incomplete transported-blade candidate is not
used or silently repaired. No parameter fitting, tolerance relaxation, automatic
cutback, failed-case retry, or cross-radius history reuse.
"""
import argparse
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from prototype.material_case import load_force_csv, signature
from prototype.initiation import local_to_lab, rounded_blade_distance
from tools import run_common_material_study as host
from tools.run_material_uncertainty import direction_trace
from tools.render_solver_dashboard import atomic_write, shell, stamp
from run_support import write_json

ROOT=Path(__file__).resolve().parents[1]
MODULE='tools.run_tip_radius_study'
RADII=(.02,.01,.04)
PARAMETERS=(.14,1.24,1.7)


def jobs():
    return [dict(name=f'r{round(1000*r):02d}_angle{a}',angle=a,radius_mm=r,
        parameters=list(PARAMETERS),stage='radius',target_mm=.5,increment=.01,
        seconds=3600,status='pending') for r in RADII for a in (0,33)]


def make(angle,parameters,*,radius,surface_spacing=.005):
    if angle not in (0,33) or radius not in RADII:
        raise ValueError('This bounded study supports 0/33 degrees and radii10/20/40 um')
    if surface_spacing not in (.005,.0025):
        raise ValueError('Explicit coarse/fine surface spacings only')
    ge,gv,tau=map(float,parameters)
    if (ge,gv,tau)!=PARAMETERS:raise ValueError('Common material must stay unchanged')
    p=host.ComputeMaterial(maxwell_shear=gv,relaxation_s=tau,loading_rate=.2,bulk_modulus=host.K,
        case='oblique',mu=.35,shear_modulus=ge,ogden_alpha=2.,dhat=.0025,local_spacing=.05,
        surface_spacing=surface_spacing,nose_facets=64,inclination=angle,tip_radius=radius,fine_halfwidth=3.,
        max_iterations=500,predictor='secant',hessian_mode='regularized',element='p1')
    p.stagnation_window=40;p.parallel_linear=None;p.ipc.set_num_threads(4)
    # Signature metadata only; native equations and kernels are unmodified.
    p.contact_control=dict(study='native-radius-only-v1' if surface_spacing==.005 else 'native-tip-mesh-v1',tip_radius_mm=radius,
        native_body_and_blade_contact=True,nose_facets=64,surface_spacing_mm=surface_spacing,
        reference_zero='analytic first tangency at same original material point',
        friction_smoothing_mm=1e-5)
    return p


def case(out,job):
    with direction_trace(out,job):
        factory=lambda a,p:make(a,p,radius=job['radius_mm'])
        targets=np.unique(np.r_[0.,np.arange(.01,job['target_mm']-1e-12,.01),job['target_mm']])
        return host.case(out,job,factory=factory,targets=targets)


def geometry_qualification():
    rows=[];baseline={}
    with threadpool_limits(1):
        for j in jobs():
            a=j['angle'];r=j['radius_mm'];p=make(a,j['parameters'],radius=r)
            contact=local_to_lab(np.array([[p.config.drop_s_mm,p.config.height_mm]]),p.config)
            penetration,normal=rounded_blade_distance(contact,p.config,p.options,0.)
            if abs(float(penetration[0]))>1e-12:raise ValueError('Radius changed first-touch zero')
            if a not in baseline:baseline[a]=(p.mesh.reference_xy.copy(),p.mesh.triangles.copy(),p.material.to_dict())
            b=baseline[a]
            if not np.array_equal(p.mesh.reference_xy,b[0]) or not np.array_equal(p.mesh.triangles,b[1]):
                raise ValueError('Changing radius changed rubber mesh')
            if p.material.to_dict()!=b[2]:raise ValueError('Material differs by radius')
            angle=np.deg2rad(p.config.blade_tip_angle_deg)
            sag=r*(1-np.cos((np.pi-angle)/(2*p.nose_facets)))
            if sag>.01*p.dhat:raise ValueError('Nose polygon exceeds1% of activation distance')
            rows.append(dict(angle=a,radius_mm=r,first_touch_gap_mm=-float(penetration[0]),
                normal=normal[0].tolist(),same_body_mesh=True,same_material=True,
                blade_thickness_mm=p.config.blade_thickness_mm,wedge_angle_deg=p.config.blade_tip_angle_deg,
                nose_sagitta_mm=float(sag),dhat_over_radius=p.dhat/r,
                body_nodes=p.n,body_elements=len(p.mesh.triangles),initial_force=p.reference_contact_check,
                signature=json.loads(json.dumps(signature(p)))))
    return dict(passed=True,rows=rows,limits='Geometry/factory check only; not FE or barrier/body-mesh convergence')


def render(out,state):
    body=f'<h1>刀尖半径：0°/33°完整橡胶对照</h1><p>{stamp()} · {escape(state["status"])}</p>'
    body+='<p>半径10/20/40 μm（不是直径）；99755厚度0.245 mm、楔角23°不变。共同G∞=.14/Gv=1.24/τ=1.7/K=23.3433/μ=.35，竖直0.2 mm/s。</p>'
    body+='<p>使用已完成0°/33°路径的原生IPC主线隔离刀尖影响；解析刀具候选的步内事件问题未修、未混入。保留原生接触已知运动学局限，结果是该模型内的几何敏感性，不是已校准物理真值。</p>'
    body+='<p><a href="analysis.html">力曲线/成对分析</a> · <a href="plan.html">方案与限制</a> · <a href="qualification.json">几何与零点核查</a> · <a href="manifest.json">冻结源/输入</a> · <a href="tests.html">回归记录</a></p>'
    body+='<table><tr><th>任务</th><th>半径μm</th><th>状态</th><th>接受mm</th><th>Fz N</th><th>证据</th></tr>'
    for j in state['jobs']:
        folder=out/j['name'];r=host.read(folder/'result.json') if (folder/'result.json').exists() else {};last=(r.get('steps') or [{}])[-1]
        if r:host.case_page(folder)
        body+=f'<tr><td>{j["name"]}</td><td>{1000*j["radius_mm"]:g}</td><td>{escape(j["status"])}</td><td>{last.get("depth_mm","—")}</td><td>{last.get("reaction_fz_n","—")}</td><td>'
        if r:body+=f'<a href="{j["name"]}/index.html">HTML日志</a> · <a href="{j["name"]}.ptc.jsonl">PTC方向</a>'
        body+='</td></tr>'
    body+='</table><p>各正式组最多3600 s求解+900 s准备/审查；失败继续下一组，不加预算、不缩步、不续别的半径历史。固定0.01 mm、500 PTC/40停滞、原1e−4 N/J/CCD/耗散验收。57°本轮不算。</p>'
    write_json(out/'queue_status.json',state);atomic_write(out/'index.html',shell('刀尖半径对照',body,refresh=state['status']=='running'))


def supervise(out,m,job,state):
    before=host.MODULE,host.render;host.MODULE,host.render=MODULE,render
    try:
        ok=host.supervise(out,m,job,state)
        folder=out/job['name']
        if (folder/'result.json').exists():
            r=host.read(folder/'result.json');actual=r.get('signature',{}).get('contact_control',{}).get('tip_radius_mm')
            if actual!=job['radius_mm']:raise ValueError('Actual radius does not match requested radius')
        return ok
    finally:host.MODULE,host.render=before


def pair_verdict(records):
    """Only two complete matching-angle curves establish a common-radius trend."""
    lookup={(r['radius_mm'],r['angle']):r for r in records if r.get('complete')}
    proposals=[]
    for radius in (.01,.04):
        pairs=[]
        for angle in (0,33):
            base=lookup.get((.02,angle));candidate=lookup.get((radius,angle))
            if base is None or candidate is None:continue
            pairs.append(dict(angle=angle,force_change05_n=candidate['fz05_n']-base['fz05_n'],
                relative_force_change05=candidate['fz05_n']/base['fz05_n']-1,
                rmse_change_n=candidate['rmse_01_05_n']-base['rmse_01_05_n']))
        qualified=len(pairs)==2
        proposals.append(dict(radius_mm=radius,paired_complete=qualified,pairs=pairs,
            lowers_both=qualified and all(p['force_change05_n']<0 for p in pairs),
            improves_both_rmse=qualified and all(p['rmse_change_n']<0 for p in pairs),
            promotion=False,reason='Even beneficial changes require independent space/barrier checks and measured edge geometry'))
    return proposals


def analysis(out,state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[];exports=[]
    for j in state['jobs']:
        folder=out/j['name']
        if not (folder/'audit.json').exists():continue
        r=host.read(folder/'result.json');a=host.read(folder/'audit.json')
        if not a['passed'] or a['states']!=len(r['steps']) or not r['steps']:continue
        host.verify_case(folder)
        ed,ef,_=load_force_csv(out/f'experiment{j["angle"]}.csv')
        d=np.array([s['depth_mm'] for s in r['steps']]);f=np.array([s['reaction_fz_n'] for s in r['steps']])
        complete=bool(r.get('completed') and d[-1]>=.5-1e-12)
        record=dict(name=j['name'],angle=j['angle'],radius_mm=j['radius_mm'],status=r['status'],
            complete=complete,accepted_states=len(d),last_depth_mm=float(d[-1]),last_fz_n=float(f[-1]),
            reference_curve_error_at_endpoint_n=float(f[-1]-np.interp(d[-1],ed,ef)),
            tip_contact_missing=sum(s.get('tip_contact',{}).get('status')!='unique' for s in r['steps']),
            footprint_errors=sum(s.get('blade_contact_footprint',{}).get('status')=='error' for s in r['steps']),
            min_j=min(s['min_j'] for s in r['steps']),rows=[dict(depth_mm=float(x),fz_n=float(y)) for x,y in zip(d,f)])
        if complete:
            grid=np.arange(10,51)/100;pts=np.array([.1,.25,.5])
            record.update(fz05_n=float(f[-1]),experiment05_n=float(np.interp(.5,ed,ef)),
                rmse_01_05_n=float(np.sqrt(np.mean((np.interp(grid,d,f)-np.interp(grid,ed,ef))**2))),
                simulation_secants_n_mm=(np.diff(np.interp(pts,d,f))/np.diff(pts)).tolist(),
                experiment_secants_n_mm=(np.diff(np.interp(pts,ed,ef))/np.diff(pts)).tolist())
        records.append(record)
        for x,y in zip(d,f):exports.append(dict(case=j['name'],angle=j['angle'],radius_mm=j['radius_mm'],depth_mm=float(x),fz_n=float(y),experiment_n=float(np.interp(x,ed,ef))))
    colors={.01:'#009E73',.02:'#D55E00',.04:'#7B4AB5'};styles={.01:('--','s'),.02:('-','o'),.04:('-.','^')}
    with plt.rc_context({'font.size':11}):
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharex=True,sharey=True,layout='constrained')
        for ax,angle in zip(axes,(0,33)):
            d,f,_=load_force_csv(out/f'experiment{angle}.csv');mask=d<=.5+1e-12
            ax.plot(d[mask],f[mask],color='#0072B2',label='Experiment (original zero)',lw=1.5)
            for r in records:
                if r['angle']!=angle:continue
                vals=r['rows'];ls,marker=styles[r['radius_mm']]
                ax.plot([v['depth_mm'] for v in vals],[v['fz_n'] for v in vals],ls=ls,marker=marker,
                    ms=3,markevery=max(1,len(vals)//8),color=colors[r['radius_mm']],
                    label=f"R={r['radius_mm']*1000:g} um | {r['status']} to {r['last_depth_mm']:.2f} mm")
            ax.set(title=f'{angle} degrees',xlabel='Vertical blade travel (mm)',xlim=(0,.5));ax.set_ylim(bottom=0)
            ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
        axes[0].set_ylabel('Fz (N)');fig.suptitle('Native intact model: radius only; common material, friction and wedge')
        fig.savefig(out/'force_comparison.png',dpi=160);plt.close(fig)
    result=dict(created=stamp(),records=records,pair_verdict=pair_verdict(records),zero_shifted=False,fracture=False,
        uncertified='No calibrated radius, body-mesh/barrier convergence or physical-contact objectivity qualification')
    write_json(out/'analysis.json',result)
    import csv
    with (out/'force_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['case','angle','radius_mm','depth_mm','fz_n','experiment_n'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(exports)
    body='<h1>刀尖半径敏感性：实际接受态</h1><p>'+stamp()+'</p><p>仅显示已审查曲线；短前缀不外推，失败不等于起裂。实验N和导出零点不改，深度μm转mm。相同半径用于0°和33°，未按角度选参数。浅层未独立认证无裂。</p>'
    body+='<img src="force_comparison.png" alt="0度与33度三个刀尖半径的已审查力曲线与各自实验对比" style="width:100%"><table><tr><th>任务</th><th>接受mm</th><th>Fz N</th><th>同深度偏差N</th><th>0.1–0.5 RMSE N</th></tr>'
    for r in records:body+=f'<tr><td>{r["name"]}</td><td>{r["last_depth_mm"]:.3f}</td><td>{r["last_fz_n"]:.6g}</td><td>{r["reference_curve_error_at_endpoint_n"]:+.6g}</td><td>{r.get("rmse_01_05_n","未完整")}</td></tr>'
    body+='</table><h2>共同半径判定</h2><pre>'+escape(json.dumps(result['pair_verdict'],ensure_ascii=False,indent=2))+'</pre><p>更低的Fz不是自动正确。半径/接触激活距离比、体网格和原生有限间隙摩擦局限仍需单独核查；无自动默认替换或进一步长算例。</p><p><a href="index.html">状态/日志</a> · <a href="analysis.json">JSON</a> · <a href="force_comparison.csv">CSV</a></p>'
    atomic_write(out/'analysis.html',shell('刀尖半径力曲线',body))


def prepare(out):
    if (out/'manifest.json').exists():raise ValueError('Refuse existing study manifest')
    tests=host.read(out/'tests.json')
    if not tests['passed'] or tests['skipped']:raise ValueError('Tests missing or failed')
    q=geometry_qualification();write_json(out/'qualification.json',q)
    inputs={str(out/'qualification.json'):host.sha(out/'qualification.json'),str(out/'tests.json'):host.sha(out/'tests.json')}
    # Actual native baseline field reaudits before any new loading.
    fresh=[]
    with threadpool_limits(1):
        for a in (0,33):
            folder=ROOT/host.REFERENCES[a];inputs.update(host.verify_case(folder));r=host.read(folder/'result.json')
            p=make(a,PARAMETERS,radius=.02)
            sig=json.loads(json.dumps(signature(p)));sig.pop('contact_control')
            if sig!=r['signature']:raise ValueError('Native control differs from existing baseline')
            for idx in (0,3,50):
                with np.load(folder/'states'/f'{idx:04d}.npz',allow_pickle=False) as z:
                    fresh.append(dict(angle=a,**p.audit_state(dict(z),r['steps'][idx])))
            shutil.copy2(ROOT/'docs'/host.EXPERIMENTS[a],out/f'experiment{a}.csv')
            inputs[str(out/f'experiment{a}.csv')]=host.sha(out/f'experiment{a}.csv')
    write_json(out/'baseline_fresh_audit.json',dict(passed=True,rows=fresh))
    inputs[str(out/'baseline_fresh_audit.json')]=host.sha(out/'baseline_fresh_audit.json')
    source=host.snapshot(out)
    for name in ('tools/run_tip_radius_study.py','tools/run_material_uncertainty.py','tests/test_tip_radius_study.py'):
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest);source[name]=host.sha(dest)
    tasks=jobs();m=dict(created=stamp(),python=sys.executable,runtime_path=str(ROOT/'.runtime/ipc-python'),
        pardiso_runtime=str(ROOT/'.runtime/pardiso'),source_sha256=source,inputs=inputs,jobs=tasks,
        material=list(PARAMETERS),radii_mm=list(RADII),angles=[0,33],
        policy='Native radius-only controls; own rest-to-.03 preflight; .5 target; no retry/cutback/automatic promotion',
        numerical_limits=dict(surface_spacing_mm=.005,dhat_mm=.0025,nose_facets=64,
            smallest_radius_to_surface_spacing=2,space_barrier_convergence=False))
    write_json(out/'manifest.json',m)
    atomic_write(out/'plan.html',shell('刀尖对照方案','<h1>半径敏感性方案</h1><p>用户授权改变锋利度，先检验0/33°降力。原20 μm、较尖10 μm、较钝40 μm，0–0.5 mm各角度独立加载。23°楔角/.245 mm厚度、材料、μ、边界、网格与原生接触形式不变。解析刀具候选未修复、不启用。</p><p>每组先独立计算到.03 mm并审查（600 s+900 s），只续同一组自己的历史。正式3600 s+900 s。失败继续下一组，零容差放宽、零自动重试。PARDISO8/IPC4/BLAS1。</p><p>相同初始材料切点与解析首触零点核查通过；半径改变会改变匹配圆弧的肩部位置，属于同一固定厚度/楔角几何的必然变化。当前dhat/R=.25/.125/.0625；半径10 μm只有两个表面边长，未作空间或接触层收敛，结果仅为筛查。保留原生摩擦客观性限制；不宣称新接触候选已修。</p><p>参考：<a href="https://doi.org/10.1016/j.engfracmech.2009.10.003">McCarthy等2010</a>。论文区分压入和起裂、研究圆鼻半径/楔角；不据此保证完整橡胶同位移反力随锋利度单调下降。当前半径非99755实测值。</p>'))
    state=dict(status='preflight',jobs=[]);failed=[]
    for job in tasks:
        pilot=dict(job,name='pilot_'+job['name'],stage='preflight',target_mm=.03,seconds=600)
        state['jobs'].append(pilot)
        if supervise(out,m,pilot,state):
            m['inputs'].update(host.verify_case(out/pilot['name']));job['resume']=str(out/pilot['name'])
        else:
            failed.append(job['name']);job.update(status='not_qualified',reason='Own .03 preflight failed')
        write_json(out/'manifest.json',m)
    state=dict(status='prepared' if len(failed)<len(tasks) else 'preflight_failed',jobs=tasks,preflights=state['jobs'])
    analysis(out,state);render(out,state)
    print(json.dumps(dict(prepared=state['status']=='prepared',failed_preflights=failed,sources=len(source),inputs=len(m['inputs']))),flush=True)


def worker(out):
    m=host.read(out/'manifest.json');host.check(m,out);state=host.read(out/'queue_status.json')
    if state['status']!='prepared':raise ValueError('Not qualified/prepared')
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),time=stamp()),f)
    state['status']='running';render(out,state)
    try:
        for job in state['jobs']:
            if job['status']=='not_qualified':continue
            try:supervise(out,m,job,state)
            except Exception as exc:job.update(status='worker_case_failed',error=repr(exc));traceback.print_exc()
            analysis(out,state);render(out,state)
        state['status']='ended'
    finally:render(out,state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','worker','case','report'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--job-json');a=parser.parse_args();out=a.output.resolve()
    if a.mode=='case':case(out,json.loads(a.job_json))
    elif a.mode=='prepare':prepare(out)
    elif a.mode=='worker':worker(out)
    else:analysis(out,host.read(out/'queue_status.json'))


if __name__=='__main__':main()
