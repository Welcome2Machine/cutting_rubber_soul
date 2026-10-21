"""Bounded common-material uncertainty experiment, without changing defaults.

Reuses audited soft/partition controls. Two missing combinations on 0/33 only;
57 loading remains gated on the unresolved complete contact-switch treatment.
"""
import argparse
from contextlib import contextmanager
from html import escape
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_reference import IPCProblem
from prototype.material_case import load_force_csv, signature
from run_support import compute_process_policy, save_npz, write_json
from tools import run_common_material_study as host
from tools.render_solver_dashboard import atomic_write, shell, stamp

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'tools.run_material_uncertainty'
G0 = 1.38 * .7
GROUPS = {'soft_q30': [G0*.3, G0*.7, 1.7],
          'soft_q50': [G0*.5, G0*.5, 1.7]}
OLD = ROOT/'outputs/common_material_fe_20261020'


def jobs():
    return [dict(name=f'{g}_{a}', group=g, stage='train', angle=a,
                 parameters=p, target_mm=.5, seconds=3600, status='pending')
            for g,p in GROUPS.items() for a in (0,33)]


@contextmanager
def direction_trace(out, job):
    """Observe actual locals; neither substitute directions nor alter acceptance."""
    lines, first = inspect.getsourcelines(IPCProblem.solve_friction)
    accept = first+next(i for i,s in enumerate(lines) if s.strip() == 'x = trial')
    reject = first+next(i for i,s in enumerate(lines) if 'print(f"FRICTION_PTC_REJECT' in s)
    updated = first+next(i for i,s in enumerate(lines) if s.strip() == 'break' and i>accept-first)
    seen=set(); latest={}; pending=None
    def local(frame,event,arg):
        nonlocal pending
        if event!='line': return local
        v=frame.f_locals
        if frame.f_lineno==updated and pending is not None:
            pending['shift_next']=float(v['shift']);emit(pending);pending=None
        if frame.f_lineno not in (accept,reject):return local
        accepted=frame.f_lineno==accept
        key=(v['depth'],v['iteration'],v['retry'],accepted)
        if key in seen:return local
        seen.add(key);p=v['self'];alpha=float(v['alpha'])
        row=dict(depth_mm=float(v['depth']),iteration=int(v['iteration']),retry=int(v['retry']),
            accepted=accepted,criterion=('merit' if v['merit_accept'] else 'model_error') if accepted else 'rejected',
            norm_n=float(v['norm']),trial_norm_n=float(v['nt']),ratio=float(v['nt']/v['norm']),
            shift_used=float(v['shift']),alpha=alpha,geometric_fraction=float(v['geometric_fraction']),
            tip_cap_alpha=.5*p.options.tip_radius_mm/float(v['maximum']),
            displacement_infinity_mm=alpha*float(v['maximum']),
            predicted_norm_n=float(np.linalg.norm(v['r']+alpha*v['linear_action'])),model_error_n=float(v['error']))
        latest.update(positions=v['x'],direction=v['direction'],previous_positions=p.previous,
            previous_material_metric=p.previous_material_metric,material_dt_s=p.material_dt,
            r=v['r'],shift=v['shift'],alpha=alpha,depth_mm=v['depth'])
        if accepted:pending=row
        else:row['shift_next']=max(.1,2*float(v['shift']));emit(row)
        return local
    def emit(row):
        with (out/(job['name']+'.ptc.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    def dispatch(frame,event,arg):
        return local if frame.f_code is IPCProblem.solve_friction.__code__ else None
    previous=sys.gettrace()
    try:
        sys.settrace(dispatch);yield
    finally:
        sys.settrace(previous)
        if pending is not None:emit(pending)
        if latest:save_npz(out/(job['name']+'.last_direction.npz'),**latest)


def case(out,job):
    with direction_trace(out,job):return host.case(out,job)


def supervise(out,m,job,state):
    before=host.MODULE,host.render
    host.MODULE,host.render=MODULE,render
    try:return host.supervise(out,m,job,state)
    finally:host.MODULE,host.render=before


def render(out,state):
    body=f'<h1>材料不确定性：降低瞬时刚度后减弱松弛</h1><p>{stamp()} · {escape(state["status"])}</p>'
    body+='<p>独立共同参数候选；未替换主线参数。0/33°各0–0.5 mm，原.01 mm步长、500 PTC/40停滞及全部物理验收。57°接触切换尚未解决，本队列不重开57°；没有三角度验证通过的结论。</p>'
    body+='<p><a href="assessment.html">依据与已有证据</a> · <a href="analysis.html">自动力曲线比较</a> · <a href="manifest.json">冻结参数/预算/哈希</a></p>'
    body+='<table><tr><th>算例</th><th>状态</th><th>深度mm</th><th>Fz N</th><th>日志</th></tr>'
    for j in state['jobs']:
        f=out/j['name'];r=host.read(f/'result.json') if (f/'result.json').exists() else {};z=(r.get('steps') or [{}])[-1]
        if r:host.case_page(f)
        body+=f'<tr><td>{j["name"]}</td><td>{escape(j["status"])}</td><td>{z.get("depth_mm","—")}</td><td>{z.get("reaction_fz_n","—")}</td><td>'
        if r:body+=f'<a href="{j["name"]}/index.html">HTML日志</a> · <a href="{j["name"]}.ptc.jsonl">实际PTC探针</a>'
        body+='</td></tr>'
    body+='</table><p>每正式case预先限定3600 s求解+900 s准备/审查；超时或失败进入下一组，无缩步/重试/自动扩展。日志保留实际方向、shift、预测/真实残差、几何与位移上限，另存最后方向及准确历史。</p>'
    atomic_write(out/'index.html',shell('材料不确定性诊断',body,refresh=state['status']=='running'))
    write_json(out/'queue_status.json',state)


def curve(folder):
    r=host.read(folder/'result.json');a=host.read(folder/'audit.json')
    if not a['passed'] or a['states']!=len(r['steps']) or not r['steps']:raise ValueError('No audited curve')
    return r,np.array([s['depth_mm'] for s in r['steps']]),np.array([s['reaction_fz_n'] for s in r['steps']])


def window_metrics(d,f,ed,ef):
    if d[0]>.1 or d[-1]<.5-1e-10:return None
    grid=np.arange(10,51)/100;err=np.interp(grid,d,f)-np.interp(grid,ed,ef)
    pts=np.array([.1,.25,.5]);fs=np.diff(np.interp(pts,d,f))/np.diff(pts);es=np.diff(np.interp(pts,ed,ef))/np.diff(pts)
    return dict(rmse_n=float(np.sqrt(np.mean(err**2))),fz05_n=float(np.interp(.5,d,f)),
        fe_secants_n_mm=fs.tolist(),experiment_secants_n_mm=es.tolist(),
        max_secant_relative_error=float(np.max(abs(fs/es-1))))


def analysis(out,m,state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained',sharey=True)
        for ax,a in zip(axs,(0,33)):
            ed,ef,_=load_force_csv(out/f'experiment{a}.csv');ax.plot(ed,ef,color='black',lw=2,label='Experiment (raw)')
            cases=[('baseline',Path(m['baselines'][str(a)])),
                   ('soft q10',Path(m['controls'][f'soft_{a}'])),
                   ('stiff q30',Path(m['controls'][f'partition_{a}']))]
            cases += [(j['group'],out/j['name']) for j in state['jobs'] if j['angle']==a and j['stage']=='train' and (out/j['name']/'audit.json').exists()]
            for (name,folder),color,ls in zip(cases,['#777777','#0072B2','#D55E00','#009E73','#CC79A7'],['--',':','-.','-','-']):
                try:r,d,f=curve(folder)
                except (ValueError,KeyError):continue
                ax.plot(d,f,color=color,ls=ls,label=name+(' [partial]' if d[-1]<.5-1e-10 else ''))
                rows.append(dict(name=name,angle=a,status=r['status'],depth_mm=float(d[-1]),metrics=window_metrics(d,f,ed,ef)))
            ax.set(title=f'{a} deg',xlabel='Depth (mm)',xlim=(0,.51),ylim=(0,15));ax.grid(alpha=.2);ax.legend(fontsize=8)
        axs[0].set_ylabel('Fz (N)');fig.suptitle('Audited FE histories vs experiment | no shifts, rescaling or extrapolation')
        fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    decision={}
    for g in GROUPS:
        found=[r for r in rows if r['name']==g and r['metrics']]
        decision[g]=dict(status='incomplete' if len(found)!=2 else 'screened_0_33_only',three_angle_validated=False)
        if len(found)==2:
            decision[g]['improves_amplitude_and_slope_vs_soft']=all(
                r['metrics']['rmse_n'] < next(z['metrics']['rmse_n'] for z in rows if z['angle']==r['angle'] and z['name']=='soft q10')
                and r['metrics']['max_secant_relative_error'] < next(z['metrics']['max_secant_relative_error'] for z in rows if z['angle']==r['angle'] and z['name']=='soft q10') for r in found)
    write_json(out/'analysis.json',dict(rows=rows,decision=decision,window_mm=[.1,.5],certified_intact=False))
    body='<h1>真实FE对照：幅值与斜率分别检验</h1><p>这里只显示已完成终态审查的曲线，未完成前缀不外推。共同参数候选不是厂家识别值。0°当前a2起裂未知，较早区间仍是假定未裂；57°未纳入新计算。</p><img src="comparison.png" style="width:100%" alt="0和33度共同材料候选与各自实验曲线">'
    body+='<table><tr><th>参数组</th><th>角度</th><th>深度mm</th><th>RMSE N</th><th>两段割线 N/mm（FE / 实验）</th></tr>'
    for r in rows:
        v=r['metrics'];body+=f'<tr><td>{r["name"]}</td><td>{r["angle"]}</td><td>{r["depth_mm"]:.3f}</td><td>{v["rmse_n"] if v else "—"}</td><td>{escape(str((v["fe_secants_n_mm"],v["experiment_secants_n_mm"]) if v else "未达到比较窗口"))}</td></tr>'
    body+='</table><pre>'+escape(json.dumps(decision,ensure_ascii=False,indent=2))+'</pre><p>通过0/33筛查也不自动推广到57°、更深区间或真实材料常数。失败仅说明该离散求解路径未完成。无自动追加计算。</p><p><a href="index.html">队列</a> · <a href="analysis.json">数值证据</a></p>'
    atomic_write(out/'analysis.html',shell('材料候选分析',body))


def prepare(out):
    if (out/'manifest.json').exists():raise ValueError('Already prepared')
    tests=host.read(out/'tests.json')
    if not tests['passed'] or tests['skipped']:raise ValueError('Regression qualification missing')
    inputs={str(out/'tests.json'):host.sha(out/'tests.json')};fresh=[]
    with threadpool_limits(1),compute_process_policy():
        for a in host.ANGLES:
            folder=ROOT/host.REFERENCES[a];inputs.update(host.verify_case(folder));r=host.read(folder/'result.json')
            p=host.make(a,host.BASE)
            if json.loads(json.dumps(signature(p)))!=r['signature']:raise ValueError('Baseline signature differs')
            for depth in (.25,.5):
                i=next(i for i,s in enumerate(r['steps']) if abs(s['depth_mm']-depth)<1e-10)
                with np.load(folder/'states'/f'{i:04d}.npz',allow_pickle=False) as z:
                    fresh.append(dict(angle=a,**p.audit_state(dict(z),r['steps'][i])))
            dest=out/f'experiment{a}.csv';shutil.copy2(ROOT/'docs'/host.EXPERIMENTS[a],dest);inputs[str(dest)]=host.sha(dest)
        for g in ('soft','partition'):
            for a in (0,33):inputs.update(host.verify_case(OLD/f'{g}_{a}'))
    source=host.snapshot(out)
    for name in ('tools/run_material_uncertainty.py','tests/test_material_uncertainty.py'):
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest);source[name]=host.sha(dest)
    tasks=jobs();m=dict(created=stamp(),source_sha256=source,inputs=inputs,jobs=tasks,
        python=sys.executable,runtime_path=str(ROOT/'.runtime/ipc-python'),pardiso_runtime=str(ROOT/'.runtime/pardiso'),
        baselines={str(a):str(ROOT/p) for a,p in host.REFERENCES.items()},
        controls={f'{g}_{a}':str(OLD/f'{g}_{a}') for g in ('soft','partition') for a in (0,33)},candidate_parameters=GROUPS,
        unchanged=dict(K=host.K,mu=.35,speed_mm_s=.2,tau_s=1.7,geometry=True,boundaries=True,physical_acceptance=True),
        budget=dict(solve_s=3600,setup_audit_s=900,preflight_solve_s=180),
        scope='4 independent 0/33 paths to .5; no 57 loading, no automatic extension or fit',
        caveat='Phenomenological uncertainty probes, not measured material constants; finite-gap objective friction remains unqualified')
    write_json(out/'manifest.json',m);write_json(out/'baseline_fresh_audit.json',dict(passed=True,rows=fresh))
    for group in GROUPS:
        first=next(j for j in tasks if j['group']==group)
        pilot=dict(first,name='pilot_'+first['name'],stage='preflight',target_mm=.01,seconds=180)
        if not supervise(out,m,pilot,dict(status='preparing',jobs=[pilot])):raise RuntimeError('Candidate preflight failed; no launch')
        m['inputs'].update(host.verify_case(out/pilot['name']));first['resume']=str(out/pilot['name'])
        write_json(out/'manifest.json',m)
    state=dict(status='prepared',jobs=tasks);analysis(out,m,state);render(out,state)
    print(json.dumps(dict(prepared=True,fresh_anchors=len(fresh),sources=len(source),inputs=len(m['inputs']))))


def worker(out):
    m=host.read(out/'manifest.json');host.check(m,out)
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),started=stamp()),f)
    state=dict(status='running',jobs=m['jobs']);render(out,state)
    for j in list(state['jobs']):
        try:
            if not j.get('resume'):
                pilot=dict(j,name='pilot_'+j['name'],stage='preflight',target_mm=.01,seconds=180,status='pending');state['jobs'].append(pilot)
                if not supervise(out,m,pilot,state):
                    j.update(status='not_qualified');continue
                j['resume']=str(out/pilot['name'])
            supervise(out,m,j,state)
        except Exception as exc:j.update(status='worker_case_failed',error=repr(exc));traceback.print_exc()
        analysis(out,m,state);render(out,state)
    state.update(status='finished',finished=stamp());analysis(out,m,state);render(out,state)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('phase',choices=('prepare','launch','worker','case','report'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--job-json');args=ap.parse_args();out=args.output.resolve()
    if args.phase=='prepare':prepare(out)
    elif args.phase=='case':case(out,json.loads(args.job_json))
    elif args.phase=='worker':worker(out)
    elif args.phase=='launch':
        m=host.read(out/'manifest.json')
        if len([j for j in m['jobs'] if j.get('resume')])!=2:raise ValueError('Two own preflights required')
        host.MODULE=MODULE;host.launch(out)
    else:
        m=host.read(out/'manifest.json');s=host.read(out/'queue_status.json');analysis(out,m,s);render(out,s)
