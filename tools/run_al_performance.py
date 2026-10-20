"""Qualify and run the explicit multicore AL backend, with original fresh audits."""
import argparse
import contextlib
from html import escape
import json
from pathlib import Path
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from run_support import compute_process_policy,write_json,save_npz
from tools import run_augmented_blade as base
from tools.profile_al_resources import prepare
from tools.render_solver_dashboard import shell,atomic_write,stamp
from prototype.al_performance import FastAugmentedBladeProblem


def fast_factory(angle,*,coarse=False,spacing=.000625,runtime=None):
    p=base.make_problem(angle,coarse=coarse,spacing=spacing)
    p.__class__=FastAugmentedBladeProblem
    return p.configure_compute(material_workers=2,batch_size=4096,linear_threads=8,runtime=runtime)


def compare_record(p,a,b):
    values={key:abs(a['motion'][key]-b['motion'][key])/max(.1,abs(a['motion'][key]))
        for key in ('normal_reaction_n','tangent_resistance_n','generalized_resistance_n')}
    values['Fz']=abs(a['reaction_fz_n']-b['reaction_fz_n'])/max(.1,abs(a['reaction_fz_n']))
    return values


def replay(source,out,fast,target=.105):
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',completed=False,fast=fast,source_sha256=base.snapshot(out),
        input_sha256=base.verify_case(source,require_independent=True),target=target)
    p=None
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1 if fast else 2),compute_process_policy():
        try:
            p,x,old,r,i=prepare(source,target,factory=fast_factory if fast else base.make_problem)
            p.ipc.set_num_threads(2);tick=time.perf_counter()
            y,record=p.step(x,old,target,time.monotonic()+180)
            elapsed=time.perf_counter()-tick
            state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=y,
                previous_positions=p.previous,depth_mm=target,body_nodes=p.n,**p.checkpoint_data())
            save_npz(out/'state.npz',**state)
            independent=base.make_problem(r['settings']['angle'],coarse=r['settings']['coarse'],spacing=r['settings']['spacing'])
            audit=independent.audit_state(state,record)
            with np.load(source/'states'/f'{i+1:04d}.npz') as reference:
                dx=float(np.linalg.norm(reference['positions'][:p.n]-y[:p.n],axis=1).max())
                metric=float(np.max(abs(reference['material_metric']-p.material_metric)))
            forces=compare_record(p,r['steps'][i+1],record)
            gate=bool(max(forces.values())<=.002 and dx<=.0002 and metric<=1e-6)
            report.update(status='completed' if gate else 'accuracy_gate_failed',completed=gate,
                solve_s=elapsed,record=record,original_fresh_audit=audit,force_fractions=forces,
                max_body_displacement_difference_mm=dx,max_material_metric_difference=metric,
                compute_settings=getattr(p,'compute_settings',{'backend':'original'}),
                linear_calls=p.parallel_linear.rows if fast else [])
        except Exception as exc:
            report.update(status='failed',error=repr(exc));traceback.print_exc(file=log)
        finally:
            if fast and p is not None:p.close_compute()
    write_json(out/'result.json',report)
    atomic_write(out/'index.html',shell('计算后端增量对照','<h1>准确历史的单增量对照</h1><pre>'+escape(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','input_sha256','linear_calls')},ensure_ascii=False,indent=2))+'</pre><a href="log.html">HTML日志</a>'))
    atomic_write(out/'log.html',shell('增量日志','<pre>'+escape((out/'run.log').read_text(encoding='utf-8'))+'</pre>'))
    return report


def run_path(out,*,fast,depth=.02,resume=None,seconds=240):
    created=[]
    def factory(*args,**kwargs):
        p=fast_factory(*args,**kwargs);created.append(p);return p
    args=argparse.Namespace(output=out,angle=57.,coarse=False,spacing=.000625,
        depth=depth,increment=.005,seconds=seconds,resume=resume,phase='run',blas_threads=1 if fast else 2)
    try:
        r=base.run(args,problem_factory=factory if fast else None)
        if r['steps']:
            # Deliberately use ORIGINAL unsplit kernel and full P2 geometry.
            with threadpool_limits(2),compute_process_policy():
                original=base.make_problem(57,spacing=.000625)
                a=base.audit(original,out,r,destination=out/'audit_original.json')
            r['original_fresh_audit_passed']=bool(a['passed'])
        else:r['original_fresh_audit_passed']=False
        if fast:
            r['compute_linear_calls']=created[0].parallel_linear.rows if created else []
        write_json(out/'result.json',r);base.report_html(out,r)
        return r
    finally:
        for p in created:p.close_compute()


def dashboard(out,r):
    body='<h1>多核AL计算加速：实测与精度核验</h1><p>'+stamp()+' · '+escape(r['status'])+'</p>'
    body+='<div class="notice">独立性能对照，原正在运行的冻结队列不变。共同材料、接触、网格、加载步、双精度和验收门槛不变。材料2工作线程、分块4096，线性PARDISO8线程；不同阶段使用不同并行度，不把线程数当作加速倍数。原始未分批材料核与完整P2几何重新审查每个接受态。</div>'
    body+='<table><tr><th>项目</th><th>状态</th><th>计算/s</th><th>力分量最大差</th></tr>'
    for row in r.get('replays',[]):
        body+=f'<tr><td><a href="{row["name"]}/index.html">{row["name"]}</a></td><td>{row["status"]}</td><td>{row.get("solve_s","—")}</td><td>{row.get("force_max","—")}</td></tr>'
    body+='</table><p><a href="path_original/index.html">独立原后端短路径</a> · <a href="path_fast/index.html">独立多核短路径</a> · <a href="summary.json">完整比较与门槛</a></p>'
    body+='<p>单增量ABBA顺序比较q=.1→.105，固定准确前步历史；新短路径从静止独立到q=.02，均5态。对照门槛比原精度资格更严：反力差≤0.2%、位移差≤0.2 μm；原物理残差与几何门槛继续执行。局部提速不等于整条深路径提速或收敛保证。</p>'
    body+='<pre>'+escape(json.dumps({k:v for k,v in r.items() if k not in ('replays','source_sha256','input_sha256')},ensure_ascii=False,indent=2))+'</pre>'
    atomic_write(out/'index.html',shell('AL多核加速验证',body,refresh=r['status']=='running'))


def qualify(source,out):
    out.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',replays=[],source_sha256=base.snapshot(out),input_sha256=base.verify_case(source,require_independent=True))
    try:
        for name,fast in [('original_A',False),('fast_A',True),('fast_B',True),('original_B',False)]:
            result=replay(source,out/name,fast)
            row=dict(name=name,status=result['status'],passed=result['completed'],solve_s=result.get('solve_s'),
                force_max=max(result.get('force_fractions',{'none':float('inf')}).values()))
            r['replays'].append(row);write_json(out/'summary.json',r);dashboard(out,r)
        if all(x['passed'] for x in r['replays']):
            a=run_path(out/'path_original',fast=False)
            b=run_path(out/'path_fast',fast=True)
            control=base.compare(out/'path_original',out/'path_fast',.02)
            control['strict_force_passed']=max(max(x['force_fractions'].values()) for x in control['rows'])<=.002
            control['strict_motion_passed']=max(x['whole_top_max_difference_mm'] for x in control['rows'])<=.0002
            r['path_comparison']=control
            r['path_original_solve_s']=a['solve_elapsed_s'];r['path_fast_solve_s']=b['solve_elapsed_s']
            r['path_original_elapsed_s']=a['elapsed_s'];r['path_fast_elapsed_s']=b['elapsed_s']
            r['path_qualified']=bool(a['completed'] and b['completed'] and a['original_fresh_audit_passed'] and b['original_fresh_audit_passed'] and control['passed'] and control['strict_force_passed'] and control['strict_motion_passed'])
        original=[x['solve_s'] for x in r['replays'] if x['name'].startswith('original') and x['passed']]
        fast=[x['solve_s'] for x in r['replays'] if x['name'].startswith('fast') and x['passed']]
        if len(original)==len(fast)==2:
            r['replay_median_original_s']=float(np.median(original));r['replay_median_fast_s']=float(np.median(fast))
            r['replay_speedup']=float(np.median(original)/np.median(fast))
        r['passed']=bool(r.get('path_qualified') and r.get('replay_speedup',0)>1.1)
        r['status']='completed'
    except Exception as exc:r.update(status='failed',passed=False,error=repr(exc));traceback.print_exc()
    write_json(out/'summary.json',r);dashboard(out,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_sha256','input_sha256','path_comparison')},indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--phase',choices=('qualify','run'),required=True)
    a.add_argument('--source',type=Path);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--depth',type=float,default=.1);a.add_argument('--seconds',type=float,default=1800)
    a.add_argument('--resume',type=Path)
    args=a.parse_args()
    if args.phase=='qualify':qualify(args.source.resolve(),args.output.resolve())
    else:
        if not (0<args.depth<=.25 and 0<args.seconds<=3600):a.error('Bounded AL backend requires depth<=.25, seconds<=3600')
        run_path(args.output.resolve(),fast=True,depth=args.depth,resume=args.resume.resolve() if args.resume else None,seconds=args.seconds)
