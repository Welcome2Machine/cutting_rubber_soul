"""Bounded qualification and independent loading of the rigid blade integral.

Common material unchanged. New native-history fields are conditional probes,
never resume prefixes. Only this study's identical preflight may resume. No
adaptive cutback, budget escalation or automatic parameter fit.
"""
import argparse
from contextlib import contextmanager
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.transported_blade import TransportedBladeProblem
from prototype.material_case import signature,load_force_csv
from tools import run_common_material_study as host
from tools.run_curve_contact_study import CurveObserver
from tools.run_material_uncertainty import direction_trace
from tools.render_solver_dashboard import shell,atomic_write,stamp
from run_support import write_json,compute_process_policy

ROOT=Path(__file__).resolve().parents[1]
MODULE='tools.run_transported_blade'


class ComputeTransported(TransportedBladeProblem):
    linear_direction=host.ComputeMaterial.linear_direction


def make(angle,parameters,*,spacing=.00125,order=3):
    ge,gv,tau=map(float,parameters)
    p=ComputeTransported(maxwell_shear=gv,relaxation_s=tau,loading_rate=.2,bulk_modulus=host.K,
        case='oblique',mu=.35,shear_modulus=ge,ogden_alpha=2.,dhat=.0025,local_spacing=.05,
        surface_spacing=.005,nose_facets=64,inclination=angle,tip_radius=.02,fine_halfwidth=3.,
        max_iterations=500,predictor='secant',hessian_mode='regularized',element='p1',
        contact_spacing=spacing,contact_order=order,guard_facets=256)
    p.stagnation_window=40;p.parallel_linear=None;p.ipc.set_num_threads(4)
    return p


def case(out,job):
    current={};original=host.TipContactTracker
    def factory(angle,parameters):
        p=make(angle,parameters);current['p']=p;return p
    host.TipContactTracker=lambda *a,**kw:CurveObserver(current['p'])
    try:
        with direction_trace(out,job):
            targets=np.r_[0.,np.arange(.01,job['target_mm']-1e-12,.01),job['target_mm']]
            return host.case(out,job,factory=factory,targets=np.unique(targets))
    finally:host.TipContactTracker=original


def render(out,state):
    body=f'<h1>P1边界与解析刀具：路径滑移验证</h1><p>{stamp()} · {escape(state["status"])}</p>'
    body+='<p>共同G∞=.14/Gv=1.24/τ=1.7/K=23.3433/μ=.35不变；竖直0.2 mm/s。全局刀具接触整体替换，完整原生材料自接触保留。没有断裂或按角度拟合。</p>'
    body+='<p>57°仅到0.16 mm验证原失败区及邻近增量；0/33°各到0.5 mm。固定.01 mm，500 PTC/40停滞，残差1e−4 N/J/CCD/真实历史验收不变。失败进入下一组，无自动扩展。</p>'
    body+='<table><tr><th>算例</th><th>状态</th><th>已接受行程mm</th><th>Fz N</th><th>证据</th></tr>'
    for j in state['jobs']:
        f=out/j['name'];r=host.read(f/'result.json') if (f/'result.json').exists() else {};last=(r.get('steps') or [{}])[-1]
        body+=f'<tr><td>{j["name"]}</td><td>{escape(j["status"])}</td><td>{last.get("depth_mm","—")}</td><td>{last.get("reaction_fz_n","—")}</td><td>'
        if r:
            host.case_page(f)
            body+=f'<a href="{j["name"]}/index.html">HTML日志</a> · <a href="{j["name"]}.ptc.jsonl">实际方向</a>'
        body+='</td></tr>'
    body+='</table><p><a href="qualification.html">整体资格与精度</a> · <a href="analysis.html">已审查前缀分析</a> · <a href="manifest.json">冻结方案</a></p>'
    body+='<p>替换后的接触积分不是原生IPC的代数等价形式；材料自接触的有限间隙转动限制仍在。完成加载不等于实验/空间/时间收敛。57°曾参与材料选择。</p>'
    atomic_write(out/'index.html',shell('解析刀具路径验证',body,refresh=state['status']=='running'));write_json(out/'queue_status.json',state)


def supervise(out,m,job,state):
    old=host.MODULE,host.render;host.MODULE,host.render=MODULE,render
    try:return host.supervise(out,m,job,state)
    finally:host.MODULE,host.render=old


def qualification(out):
    """Whole assembled tangent + independent spatial quadrature, fixed histories."""
    records=[];inputs={};deadline=time.monotonic()+600
    with threadpool_limits(1),compute_process_policy():
        for angle,folder,label in ((0,ROOT/'outputs/material_uncertainty_20261021/soft_q30_0','0050'),
                                   (33,ROOT/'outputs/material_uncertainty_20261021/soft_q30_33','0050'),
                                   (57,ROOT/'outputs/contact_time_paths_20261021/fixed_half_57','failed')):
            print(f'QUALIFY angle={angle} field={label}',flush=True)
            inputs.update(host.verify_case(folder));r=host.read(folder/'result.json')
            path=folder/'failed_trial.npz' if label=='failed' else folder/'states'/f'{label}.npz'
            inputs[str(path)]=host.sha(path)
            with np.load(path,allow_pickle=False) as z:s=dict(z)
            variants=[];grads=[];ps=[]
            for spacing,order in ((.0025,2),(.00125,3),(.000625,4)):
                if time.monotonic()>deadline:raise TimeoutError('600 s field qualification budget')
                p=make(angle,r['job']['parameters'],spacing=spacing,order=order)
                if not np.array_equal(s['reference_xy'],p.rest) or not np.array_equal(s['triangles'],p.mesh.triangles):raise ValueError('Saved geometry differs')
                if json.loads(json.dumps(p.material.to_dict()))!=r['signature']['material']:raise ValueError('Conditional material changed')
                p.previous=s['previous_positions'];p.previous_material_metric=s['previous_material_metric'];p.material_dt=float(s['material_dt_s'])
                data,D,gap=p._blade_data(s['positions']);gradient=np.asarray(D.T@data['gradient'].ravel())
                fz=float(-gradient.reshape(-1,2)[p.n:,1].sum());grads.append(gradient);ps.append(p)
                variants.append(dict(spacing_mm=spacing,order=order,fz_n=fz,minimum_gap_mm=gap,
                    active_quadrature=len(data['events']),maximum_slip_integration_error_mm=max((v['slip_quadrature_error_mm'] for v in data['events']),default=0.)))
            reference=grads[-1];p=ps[1];x=s['positions'];_,g,A,J=p.evaluate(x,projected=False)
            differences=[]
            for gradient in grads[:-1]:
                differences.append(dict(force_relative=float(np.linalg.norm((gradient-reference)[p.free])/max(np.linalg.norm(reference[p.free]),1e-15)),
                    fz_relative=float(abs((gradient-reference).reshape(-1,2)[p.n:,1].sum())/max(abs(reference.reshape(-1,2)[p.n:,1].sum()),1e-15))))
            rng=np.random.default_rng(125+angle);d=np.zeros_like(x);near=np.linalg.norm(x[:p.n]-x[p.n+2+p.nose_facets//2],axis=1)<.1
            d[:p.n][near]=rng.normal(size=(near.sum(),2));d[p.mesh.bottom_nodes]=0;d/=np.max(abs(d));action=A@d.ravel();errors=[]
            for h in (1e-7,1e-8,1e-9):
                gp=p.evaluate(x+h*d,tangent=False)[1];gm=p.evaluate(x-h*d,tangent=False)[1]
                errors.append(dict(h_mm=h,relative_error=float(np.linalg.norm((gp-gm)/(2*h)-action)/max(np.linalg.norm(action),1e-15))))
            tangent=any(a['relative_error']<1e-3 and b['relative_error']<1e-3 for a,b in zip(errors,errors[1:]))
            quad=differences[1]['force_relative']<.01 and differences[1]['fz_relative']<.005
            row=dict(angle=angle,native_state_accepted=label!='failed',variants=variants,differences_vs_finest=differences,
                coupled_tangent_trials=errors,tangent_passed=tangent,quadrature_passed=quad,min_J=J,
                conditional_new_free_residual_n=float(np.linalg.norm(g[p.free])),
                native_geometry_has_intersections=bool(p.ipc.has_intersections(p.guard_mesh,p._guard_positions(x))),
                scope='Fixed native old histories; new conditional force is NOT a new equilibrium, Fz curve or resume prefix')
            records.append(row);write_json(out/'field_qualification.json',dict(rows=records,inputs=inputs,passed=False))
    passed=all(r['tangent_passed'] and r['quadrature_passed'] and not r['native_geometry_has_intersections'] for r in records)
    result=dict(rows=records,inputs=inputs,passed=passed,spatial_candidate=dict(spacing_mm=.00125,order=3),
        limits='Three fixed fields only; independent loading/time convergence and experiment remain unqualified')
    write_json(out/'field_qualification.json',result)
    atomic_write(out/'qualification.html',shell('完整接触资格',
        '<h1>完整装配与接触积分检查</h1><p>固定旧历史的条件诊断，不是新平衡态。每场采用三档参考边界积分、三档方向导数扰动。中档相对最细档要求自由力差&lt;1%、Fz差&lt;0.5%；整体切线要求相邻两档相对误差&lt;0.1%。</p><pre>'+
        escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>'))
    return result


def analysis(out,m,state):
    rows=[]
    for j in state['jobs']:
        folder=out/j['name']
        if not (folder/'audit.json').exists():continue
        r=host.read(folder/'result.json');a=host.read(folder/'audit.json')
        if not a['passed'] or a['states']!=len(r['steps']) or not r['steps']:continue
        host.verify_case(folder);ed,ef,_=load_force_csv(out/f'experiment{j["angle"]}.csv')
        values=[dict(depth_mm=s['depth_mm'],fz_n=s['reaction_fz_n'],experiment_n=float(np.interp(s['depth_mm'],ed,ef)),
                     residual_n=s['residual_n'],iterations=s['iterations']) for s in r['steps']]
        rows.append(dict(case=j['name'],angle=j['angle'],stage=j['stage'],status=r['status'],values=values))
    write_json(out/'analysis.json',dict(rows=rows,unshifted_experiment=True,certified_intact=False))
    body='<h1>独立加载结果</h1><p>仅列通过新鲜历史审查的接受态；失败前缀不外推，机械失败不是断裂。力以原N值比较，57°实验零点0.51 N保留。</p>'
    body+='<table><tr><th>算例</th><th>行程mm</th><th>Fz N</th><th>实验N</th><th>误差N</th></tr>'
    for r in rows:
        v=r['values'][-1];body+=f'<tr><td>{r["case"]}</td><td>{v["depth_mm"]}</td><td>{v["fz_n"]:.6g}</td><td>{v["experiment_n"]:.6g}</td><td>{v["fz_n"]-v["experiment_n"]:.6g}</td></tr>'
    body+='</table><p><a href="analysis.json">逐步数值</a> · <a href="index.html">运行状态</a></p>'
    atomic_write(out/'analysis.html',shell('解析刀具加载对照',body))


def prepare(out):
    if (out/'manifest.json').exists():raise ValueError('Already prepared')
    tests=host.read(out/'tests.json')
    if not tests['passed'] or tests['skipped']:raise ValueError('Regression qualification missing')
    q=qualification(out)
    if not q['passed']:raise RuntimeError('Whole-interface qualification failed; no loading launch')
    inputs=dict(q['inputs']);inputs[str(out/'field_qualification.json')]=host.sha(out/'field_qualification.json')
    inputs[str(out/'tests.json')]=host.sha(out/'tests.json')
    for a in (0,33,57):
        file=out/f'experiment{a}.csv';shutil.copy2(ROOT/'docs'/host.EXPERIMENTS[a],file);load_force_csv(file);inputs[str(file)]=host.sha(file)
    source=host.snapshot(out)
    extras=['tools/run_transported_blade.py','tools/run_material_uncertainty.py','tools/run_curve_contact_study.py',
            'tests/test_transported_blade.py','tests/test_curve_contact.py']
    for name in extras:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest);source[name]=host.sha(dest)
    jobs=[dict(name=f'angle{a}',stage='path',angle=a,parameters=[.14,1.24,1.7],target_mm=.16 if a==57 else .5,
               increment=.01,seconds=1800 if a==57 else 3600,status='pending') for a in (57,0,33)]
    m=dict(created=stamp(),python=sys.executable,runtime_path=str(ROOT/'.runtime/ipc-python'),
        pardiso_runtime=str(ROOT/'.runtime/pardiso'),source_sha256=source,inputs=inputs,jobs=jobs,
        scope='Independent common-material knife-contact candidate, no production promotion',
        stop_policy='Original500PTC/40stagnation, no cutback/retry/extension; each failed case advances queue',
        physical_acceptance='Original force1e-4N, J, native complete self-contact and conservative whole-edge CCD, extra load-chord check')
    write_json(out/'manifest.json',m);state=dict(status='preflight',jobs=[])
    # All pilots independent. Failure means no new long queue, not extra budget.
    failures=[]
    for job in jobs:
        pilot=dict(job,name='pilot_'+job['name'],stage='preflight',target_mm=.01,seconds=300)
        state['jobs'].append(pilot)
        if not supervise(out,m,pilot,state):failures.append(pilot['name'])
        else:
            m['inputs'].update(host.verify_case(out/pilot['name']));job['resume']=str(out/pilot['name'])
        write_json(out/'manifest.json',m)
    if failures:
        state['status']='preflight_failed';analysis(out,m,state);render(out,state)
        raise RuntimeError('Independent preflight(s) failed, no long queue: '+','.join(failures))
    state=dict(status='prepared',jobs=jobs,preflights=state['jobs']);analysis(out,m,state);render(out,state)
    print(json.dumps(dict(prepared=True,sources=len(source),inputs=len(m['inputs']))))


def worker(out):
    m=host.read(out/'manifest.json');host.check(m,out)
    state=host.read(out/'queue_status.json')
    if state['status']!='prepared':raise ValueError('Study is not qualified/prepared')
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),time=stamp()),f)
    state['status']='running';render(out,state)
    try:
        for job in state['jobs']:
            try:supervise(out,m,job,state)
            except Exception as exc:job.update(status='worker_case_failed',error=repr(exc));traceback.print_exc()
            analysis(out,m,state);render(out,state)
        state['status']='ended'
    finally:render(out,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','worker','case'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--job-json');a=p.parse_args();out=a.output.resolve()
    if a.mode=='case':case(out,json.loads(a.job_json))
    elif a.mode=='prepare':prepare(out)
    else:worker(out)


if __name__=='__main__':main()
