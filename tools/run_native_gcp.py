"""Native blade IPC / body GCP qualification, independent paths and gated queue.

All depths are laboratory vertical d, speed .2 mm/s. Fixed increments, no
failed-step bisection or budget extension. Only OWN identical prefixes resume.
"""
import argparse
import contextlib
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from run_support import (source_hashes,write_json,save_npz,compute_process_policy,
                         background_child_options,classify_process_exit)
from prototype.native_geometric_contact import NativeGeometricProblem,POLICY
from prototype.ipc_reference import BASELINE,record_tip_contact,export_contact_table
from prototype.tip_contact import TipContactTracker
from tools.run_material_candidate import (signature as base_signature,audit,resume,
    make_problem as native_problem,load_force_csv)
from tools.render_solver_dashboard import shell,atomic_write,stamp

ROOT=Path(__file__).resolve().parents[1]
COMMON=dict(equilibrium_shear=.14,maxwell_shear=1.24,relaxation_s=1.7,
            loading_rate=.2,bulk_modulus=23.34333333333331,mu=.35)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_environment(manifest):
    """Resolve optional native libraries independently of a frozen source cwd."""
    env=os.environ.copy()
    env.update(PYTHONPATH=manifest['runtime_path'],
        SOFTMODELING_PARDISO_RUNTIME=manifest['pardiso_runtime'],
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    return env


def make(angle,*,fast=False,resources=None):
    p=NativeGeometricProblem(maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2,
        bulk_modulus=COMMON['bulk_modulus'],case='oblique',mu=.35,shear_modulus=.14,
        dhat=.0025,local_spacing=.05,surface_spacing=.005,nose_facets=64,
        inclination=angle,tip_radius=.02,fine_halfwidth=3.,max_iterations=500,
        predictor='secant',hessian_mode='regularized',element='p1')
    p.stagnation_window=40;p.ipc.set_num_threads(4)
    if fast:p.configure_compute(**(resources or dict(material_workers=2,linear_threads=8)))
    return p


def signature(p):
    return dict(base_signature(p),body_contact=POLICY,geometric_self_spec=p.geometric_self.spec,
                blade_contact='native improved-max; cross-body pairs only',
                ccd='original full body/body and blade/body',element='P1')


def snapshot(out):
    hashes=source_hashes(snapshot=out/'source')
    for name in ('prototype/ipc_baseline.json','tools/run_native_gcp.py',
                 'tools/run_material_candidate.py','tools/render_solver_dashboard.py'):
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dst);hashes[name]=digest(dst)
    return hashes


def verify(folder):
    folder=Path(folder);r=read(folder/'result.json');a=read(folder/'audit.json')
    if not a.get('passed') or a['states']!=len(r['steps']):raise ValueError('Incomplete fresh audit')
    files={str(folder/'result.json'):digest(folder/'result.json'),str(folder/'audit.json'):digest(folder/'audit.json')}
    for name,h in r['source_sha256'].items():
        path=folder/'source'/name
        if digest(path)!=h:raise ValueError('Frozen source differs: '+str(path))
        files[str(path)]=h
    for i,row in enumerate(a['rows']):
        path=folder/'states'/f'{i:04d}.npz'
        if not row.get('passed') or digest(path)!=row['checkpoint_sha256']:raise ValueError('Checkpoint hash differs')
        files[str(path)]=row['checkpoint_sha256']
    return files


def load_state(folder,index):
    with np.load(Path(folder)/'states'/f'{index:04d}.npz',allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def case_page(out,r):
    z=(r.get('steps') or [{}])[-1]
    body=f'<h1>原生刀具IPC＋本体GCP：{r.get("angle")}°</h1><p>{stamp()} · {escape(r["status"])}</p>'
    body+='<p>原生P1，共同Maxwell；竖直速度.2 mm/s。完整CCD/J、原残差与双历史；失败不缩步。材料/空间/滑移尚非实验验证。</p>'
    body+='<p>'+escape(f'接受态 {len(r.get("steps",[]))}；深度 {z.get("depth_mm")} mm；Fz {z.get("reaction_fz_n")} N；minJ {z.get("min_j")}')+'</p>'
    body+='<p>'+escape(r.get('error',''))+'</p><p><a href="log.html">运行日志</a> · <a href="result.json">状态JSON</a> · <a href="fz_depth.csv">Fz CSV</a> · <a href="../index.html">队列与分析</a></p>'
    if (out/'audit.json').exists():body+='<p><a href="audit.json">准确双历史独立审查</a></p>'
    atomic_write(out/'index.html',shell('原生刀具/本体GCP状态',body,refresh=r['status']=='running'))
    if (out/'run.log').exists():
        log=(out/'run.log').read_text(encoding='utf-8',errors='replace')
        atomic_write(out/'log.html',shell('运行日志', '<p>最近120000字符；完整过程保存在run.log。</p><pre>'+escape(log[-120000:])+'</pre>',refresh=r['status']=='running'))


def run_case(out,job,resources):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);(out/'states').mkdir()
    r=dict(status='initializing',completed=False,angle=job['angle'],target_mm=job['depth'],
        increment_mm=job['increment'],source_sha256=snapshot(out),steps=[],rejected_load_attempts=[],
        tip_contact_tracking=dict(errors=[]),load_policy='fixed-no-cutback',job=job,
        compute=resources,force_tolerance_n=BASELINE['force_tolerance_n'])
    write_json(out/'result.json',r);p=None;x=None;depth=None;start=time.monotonic()
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy() as policy:
        r['runtime_process_policy']=policy
        try:
            p=make(job['angle']);r['signature']=json.loads(json.dumps(signature(p)))
            r['reference_contact_check']=p.reference_contact_check
            r['mesh']=dict(nodes=p.n,elements=len(p.mesh.triangles))
            x=p.rest.copy();old=p.start
            if job.get('resume'):
                previous=read(Path(job['resume'])/'result.json')
                if previous['increment_mm']!=job['increment']:raise ValueError('Resume changed physical time schedule')
                x,old=resume(p,Path(job['resume']),out,r)
            p.configure_compute(**resources)
            tracker=TipContactTracker(p.rest,p.mesh.triangles,p.n,p.config,p.dhat,p.nose_facets)
            r['setup_s']=time.monotonic()-start;r['status']='running';tick=time.monotonic()
            deadline=tick+job['seconds'];r['solve_s']=0.
            targets=np.r_[0.,np.arange(job['increment'],job['depth']-1e-12,job['increment']),job['depth']]
            for depth in targets[targets>old+1e-12]:
                step_start=time.monotonic();x,row=p.step(x,old,float(depth),deadline)
                old=float(depth);row['wall_step_s']=time.monotonic()-step_start
                save_npz(out/'states'/f'{len(r["steps"]):04d}.npz',reference_xy=p.rest,
                    triangles=p.mesh.triangles,positions=x,previous_positions=p.previous,
                    depth_mm=old,body_nodes=p.n,**p.checkpoint_data())
                record_tip_contact(tracker,x,row,r);r['steps'].append(row);r['last_depth_mm']=old
                r['solve_s']=time.monotonic()-tick;write_json(out/'result.json',r);case_page(out,r)
                print(f'ACCEPT d_mm={old:.6f} Fz_N={row["reaction_fz_n"]:.9g} minJ={row["min_j"]:.8g} seconds={row["wall_step_s"]:.3f}',flush=True)
            r.update(status='completed',completed=True)
        except Exception as exc:
            r.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',completed=False,error=f'{type(exc).__name__}: {exc}')
            if 'tick' in locals():r['solve_s']=time.monotonic()-tick
            r['rejected_load_attempts'].append(dict(depth_mm=float(depth) if depth is not None else None,error=repr(exc),policy='fixed-no-cutback'))
            traceback.print_exc(file=log)
            if p is not None and x is not None and p.material_dt is not None:
                save_npz(out/'failed_trial.npz',positions=getattr(p,'trial',x),previous_positions=p.previous,
                    attempted_depth_mm=depth,reference_xy=p.rest,triangles=p.mesh.triangles,**p.checkpoint_data())
        finally:
            if p is not None:p.close_compute()
        try:
            audit_start=time.monotonic()
            if r['steps']:
                q=make(job['angle']) # unbatched original material, original SuperLU-independent residual
                audit(q,out,r,deadline=time.monotonic()+1800.)
            r['audit_s']=time.monotonic()-audit_start
        except Exception as exc:
            r.update(status='audit_failed',completed=False,audit_error=repr(exc));traceback.print_exc(file=log)
        r['elapsed_s']=time.monotonic()-start
    export_contact_table(out,r)
    with (out/'fz_depth.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['depth_mm','fz_n']);w.writerows((v['depth_mm'],v['reaction_fz_n']) for v in r['steps'])
    write_json(out/'result.json',r);write_json(out/'outcome.json',{k:r.get(k) for k in ('status','completed','last_depth_mm','elapsed_s','error','audit_error')})
    case_page(out,r);return r


def interval_distance(a,b):
    """Exact Hausdorff distance of closed 1D interval unions (including gaps)."""
    if not a or not b:return None if not a and not b else float('inf')
    def directed(a,b):
        probes=[x for pair in a for x in pair]
        probes += [(.5*(u[1]+v[0])) for u,v in zip(b,b[1:])
                   if any(lo<=.5*(u[1]+v[0])<=hi for lo,hi in a)]
        return max(min(max(lo-x,0.,x-hi) for lo,hi in b) for x in probes)
    return max(directed(a,b),directed(b,a))


def paired_gate(first,second):
    a,b=read(first/'result.json'),read(second/'result.json')
    if not a['completed'] or not b['completed']:return dict(passed=False,reason='Incomplete independent paths')
    verify(first);verify(second)
    if a['signature']!=b['signature']:raise ValueError('Pair changes physical specification')
    rows=[];unique_count=0;missing_both=0;mismatches=0
    for i,r in enumerate(a['steps']):
        matches=[(j,s) for j,s in enumerate(b['steps']) if abs(r['depth_mm']-s['depth_mm'])<1e-10]
        if len(matches)!=1:raise ValueError('Missing common accepted depth')
        j,s=matches[0];sa,sb=load_state(first,i),load_state(second,j)
        np.testing.assert_array_equal(sa['reference_xy'],sb['reference_xy']);np.testing.assert_array_equal(sa['triangles'],sb['triangles'])
        # Same trace; all original boundary nodes, including the sparse region.
        tri=sa['triangles'];edges,c=np.unique(np.sort(np.vstack([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0,return_counts=True)
        ids=np.unique(edges[c==1]);motion=float(np.linalg.norm(sa['positions'][ids]-sb['positions'][ids],axis=1).max())
        x,y=r['tip_contact'],s['tip_contact'];offset=None
        if x['status']=='unique' and y['status']=='unique':
            unique_count+=1;offset=abs(x['reference_offset_mm']-y['reference_offset_mm'])
        elif x['status']!=y['status']:mismatches+=1
        else:missing_both+=1
        nx,ny=x['nose_reference_intervals_mm'],y['nose_reference_intervals_mm']
        nose=interval_distance(nx,ny)
        rows.append(dict(depth_mm=r['depth_mm'],force_change_fraction=abs(r['reaction_fz_n']-s['reaction_fz_n'])/max(1.,abs(s['reaction_fz_n'])),
            boundary_max_difference_mm=motion,unique_tip_difference_mm=offset,
            nose_interval_distance_mm=nose if nose is None or np.isfinite(nose) else None,
            nose_availability_match=bool(nx)==bool(ny),
            tip_status_first=x['status'],tip_status_second=y['status']))
    finite=lambda key:[v[key] for v in rows if v[key] is not None]
    gates=dict(force_2pct=max(finite('force_change_fraction'))<=.02,
        entire_boundary_2um=max(finite('boundary_max_difference_mm'))<=.002,
        nose_interval_2um=all(v['nose_availability_match'] for v in rows) and bool(finite('nose_interval_distance_mm')) and max(finite('nose_interval_distance_mm'))<=.002,
        shared_unique_tip_5um=not finite('unique_tip_difference_mm') or max(finite('unique_tip_difference_mm'))<=.005,
        positive_j=min(v['min_j'] for v in a['steps']+b['steps'])>0,
        force_history_available=len(rows)==51,
        cost_gate=a['solve_s']<=3600 and b['solve_s']<=5400)
    # Classification switches forbid claiming a unique cut-point, but a stable
    # surface/nose interval can still qualify a FORCE-path continuation.
    return dict(passed=all(gates.values()),gates=gates,rows=rows,
        unique_tip_qualified=mismatches==0 and unique_count==len(rows),
        shared_unique_states=unique_count,both_missing_or_ambiguous_states=missing_both,
        classification_mismatches=mismatches,
        continuation_scope='Force/surface path; unique pole identity ONLY where qualified; intervals are geometric proximity, not slip')


def report_queue(out,state):
    body='<h1>原生P1／刀具IPC＋本体GCP：独立路径验证</h1><p>'+stamp()+' · '+escape(state['status'])+'</p>'
    body+='<p>33°→57°→0°，每个角度先做从零到.5 mm的.01与.005 mm独立路径。原方程/全CCD/J/双历史通过，力差≤2%（分母下限1 N）、全边界运动与刀鼻材料区间差≤2 μm、共有唯一刀尖身份差≤5 μm且成本合格后，才接自己的.01 mm历史至1和2 mm。缺失的刀尖身份不填值，力路径资格不等于滑移或空间/实验验证。无失败缩步或追加预算。</p>'
    body+='<table><tr><th>工况</th><th>状态</th><th>深度mm</th><th>Fz N</th><th>路径minJ</th><th>HTML</th></tr>'
    for job in state['jobs']:
        folder=out/job['name'];r=read(folder/'result.json') if (folder/'result.json').exists() else {};z=(r.get('steps') or [{}])[-1]
        body+=f'<tr><td>{escape(job["name"])}</td><td>{escape(job["status"])}</td><td>{z.get("depth_mm","—")}</td><td>{z.get("reaction_fz_n","—")}</td><td>{min((s["min_j"] for s in r.get("steps",[])),default="—")}</td><td>'+ (f'<a href="{job["name"]}/index.html">状态/日志</a>' if r else escape(job.get('reason','待运行')))+'</td></tr>'
        if r:case_page(folder,r)
    body+='</table><p><a href="preparation.json">原生基线、输入来源及资源实测</a> · <a href="analysis.html">终态/当前范围分析</a> · <a href="manifest.json">冻结方案</a></p>'
    for angle,gate in state.get('gates',{}).items():
        body+=f'<h2>{angle}°资格</h2><pre>'+escape(json.dumps({k:v for k,v in gate.items() if k!='rows'},ensure_ascii=False,indent=2))+'</pre>'
    atomic_write(out/'index.html',shell('原生刀具IPC / 本体GCP队列',body,refresh=state['status']=='running'))
    write_json(out/'queue_status.json',state)


def analyze(out,manifest,state):
    result=[]
    for job in state['jobs']:
        path=out/job['name']/'result.json'
        if not path.exists():continue
        r=read(path);rows=r.get('steps',[])
        if not rows:continue
        d=np.array([v['depth_mm'] for v in rows]);f=np.array([v['reaction_fz_n'] for v in rows])
        baseline=read(Path(manifest['baselines'][str(job['angle'])])/'result.json')
        bd=np.array([v['depth_mm'] for v in baseline['steps']]);bf=np.array([v['reaction_fz_n'] for v in baseline['steps']])
        comparison=dict(name=job['name'],status=job['status'],end_mm=float(d[-1]),fz_n=float(f[-1]),
            original_native_fz_n=float(np.interp(d[-1],bd,bf)),min_j=min(v['min_j'] for v in rows),
            max_gcp_pairs=max(v['geometric_self_contact']['active_pairs'] for v in rows),
            unique_tip_states=sum(v['tip_contact']['status']=='unique' for v in rows),states=len(rows),solve_s=r.get('solve_s'),
            accepted_only=True,experimental_validation=False)
        if job['angle']==57:
            ed,ef,_=load_force_csv(out/'experiment57.csv');observed=np.interp(d,ed,ef)
            comparison['experiment']=dict(rmse_n=float(np.sqrt(np.mean((f-observed)**2))),
                endpoint_n=float(observed[-1]),handling='original .51 N retained; selection trial, not independent validation')
        elif job['angle']==33:
            comparison['experiment_visual']=[dict(depth_mm=z,fe_n=float(np.interp(z,d,f)),visual_n=v,range_n=[v-1,v+1])
                for z,v in ((.5,8.5),(1.,17.1),(2.,31.5)) if z<=d[-1]+1e-12]
        result.append(comparison)
    write_json(out/'analysis.json',dict(rows=result,gates=state.get('gates',{}),experimental_validation=False))
    body='<h1>已接受范围分析</h1><p>33°仅历史截图估计范围（不是置信区间）；57°只用授权CSV；0°没有选定数值实验曲线。没有将失败试算当平衡力，没有跨越缺失深度外推。原生旧模型仅为计算对照。</p>'
    body+='<pre>'+escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre><p><a href="index.html">返回队列</a></p>'
    atomic_write(out/'analysis.html',shell('当前接受范围分析',body))


def worker(out):
    m=read(out/'manifest.json')
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),time=stamp()),f)
    for path,h in m['inputs'].items():
        if digest(path)!=h:raise ValueError('Input hash changed: '+path)
    for name,h in m['source_sha256'].items():
        if digest(out/'source'/name)!=h:raise ValueError('Source hash changed: '+name)
    state=dict(status='running',jobs=m['jobs'],gates={});report_queue(out,state)
    for job in state['jobs']:
        try:
            dependency=job.get('requires')
            if dependency and not state['gates'].get(str(job['angle']),{}).get('passed'):
                job.update(status='cancelled',reason='独立.5 mm数值门槛未通过');continue
            if job.get('resume') and not read(Path(job['resume'])/'result.json')['completed']:
                job.update(status='cancelled',reason='自己的前缀未完整完成');continue
            if job['kind']=='half' and not read(out/f'base{job["angle"]}'/'result.json')['completed']:
                job.update(status='cancelled',reason='该角度基础路径未完成，停止精度扩展');continue
            cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_native_gcp','case','--output',str(out),'--name',job['name']]
            env=runtime_environment(m)
            with (out/(job['name']+'.stdout.log')).open('wb') as stdout,(out/(job['name']+'.stderr.log')).open('wb') as stderr:
                proc=subprocess.Popen(cmd,cwd=out/'source',env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**background_child_options())
                job.update(status='running',pid=proc.pid,started=stamp());report_queue(out,state);tick=time.monotonic();timed=False
                while proc.poll() is None:
                    if time.monotonic()-tick>job['seconds']+2400:
                        timed=True;proc.terminate();proc.wait(timeout=30);break
                    report_queue(out,state);time.sleep(10)
            job.update(returncode=proc.returncode,wall_s=time.monotonic()-tick,status=classify_process_exit(proc.returncode,supervisor_timeout=timed))
            folder=out/job['name'];r=read(folder/'result.json')
            if not timed and r['steps']:verify(folder)
            job['status']=r['status'] if proc.returncode==0 and not timed else job['status']
            if job['kind']=='half':
                gate=paired_gate(out/f'base{job["angle"]}',folder);state['gates'][str(job['angle'])]=gate
                write_json(out/f'gate{job["angle"]}.json',gate)
        except Exception as exc:
            job.update(status='failed',error=repr(exc));traceback.print_exc()
        finally:analyze(out,m,state);report_queue(out,state)
    state.update(status='finished',finished=stamp());analyze(out,m,state);report_queue(out,state)


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    baselines={33:ROOT/'outputs/native_angles_2mm_20261016/01_angle33',
        57:ROOT/'outputs/native_angles_2mm_20261016/02_angle57',
        0:ROOT/'outputs/relaxation_partition_20261015/angles30_0_2mm/02_angle0'}
    inputs={};review=dict(baselines=[],resources={},status='preparing')
    with threadpool_limits(1),(out/'preparation.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        for angle,path in baselines.items():
            inputs.update(verify(path));r=read(path/'result.json')
            if not r['completed'] or abs(r['steps'][-1]['depth_mm']-2)>1e-10:raise ValueError('Baseline incomplete')
            p=native_problem(angle=angle,**COMMON);p.ipc.set_num_threads(4)
            s=load_state(path,len(r['steps'])-1);fresh=p.audit_state(s,r['steps'][-1])
            review['baselines'].append(dict(angle=angle,path=str(path),states=len(r['steps']),fresh_endpoint=fresh))
        # Measured native-P1 workload; this old state is a profiling field ONLY.
        p=make(33);s=load_state(baselines[33],25)
        p.previous=s['previous_positions'];p.previous_material_metric=s['previous_material_metric'];p.material_dt=float(s['material_dt_s'])
        from prototype.al_performance import ParallelMaterial,PardisoDirection
        x=s['positions'];ref=p.evaluate(x,p.lag(x),projected=False,with_metric=True)
        measurements=[]
        for count in (1,2,4):
            p.parallel_material=ParallelMaterial(count);samples=[]
            for _ in range(2):
                tick=time.perf_counter();v=p.evaluate(x,p.lag(x),projected=False,with_metric=True);samples.append(time.perf_counter()-tick)
                np.testing.assert_allclose(v[1],ref[1],rtol=1e-11,atol=1e-10)
            p.parallel_material.close();p.parallel_material=None
            measurements.append(dict(workers=count,median_s=float(np.median(samples))))
        A=p.coupled_friction_tangent(x,p.lag(x),ref[2])[p.free][:,p.free]+.1*ref[4][p.free][:,p.free]
        linear=[]
        for count in (1,4,8):
            solver=PardisoDirection(count);samples=[]
            try:
                for _ in range(2):
                    tick=time.perf_counter();v=solver.solve(A,-ref[1][p.free]);samples.append(time.perf_counter()-tick)
                    if np.linalg.norm(A@v+ref[1][p.free])>1e-8:raise ValueError('Benchmark linear residual')
                linear.append(dict(threads=count,median_s=float(np.median(samples))))
            finally:solver.close()
        resources=dict(material_workers=min(measurements,key=lambda a:a['median_s'])['workers'],
            linear_threads=min(linear,key=lambda a:a['median_s'])['threads'])
        review['resources']=dict(selected=resources,material=measurements,linear=linear,scope='native P1 fixed field; not full-path speedup')
    experiment=ROOT/'docs/1014n57a1__Fz__curve_0.csv';shutil.copy2(experiment,out/'experiment57.csv')
    inputs[str(out/'experiment57.csv')]=digest(out/'experiment57.csv')
    review['status']='prepared';write_json(out/'preparation.json',review)
    jobs=[]
    for angle in (33,57,0):
        jobs.extend([dict(name=f'base{angle}',angle=angle,kind='base',depth=.5,increment=.01,seconds=3600.,status='pending',resume=str(out/f'preflight{angle}')),
            dict(name=f'half{angle}',angle=angle,kind='half',depth=.5,increment=.005,seconds=5400.,status='pending')])
    for depth in (1.,2.):
        for angle in (33,57,0):
            predecessor=f'base{angle}' if depth==1 else f'd1_{angle}'
            jobs.append(dict(name=f'd{int(depth)}_{angle}',angle=angle,kind='continuation',depth=depth,increment=.01,
                seconds=10800. if depth==1 else 21600.,status='pending',requires='paired_05',resume=str(out/predecessor)))
    m=dict(created=stamp(),source_sha256=snapshot(out),inputs=inputs,resources=resources,jobs=jobs,
           baselines={str(k):str(v) for k,v in baselines.items()},runtime_path=str(ROOT/'.runtime/ipc-python'),
           pardiso_runtime=str(ROOT/'.runtime/pardiso'))
    write_json(out/'manifest.json',m);report_queue(out,dict(status='prepared_not_launched',jobs=jobs,gates={}))
    print(json.dumps(review,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','preflight','launch','worker','case'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--name');parser.add_argument('--angle',type=int,choices=(0,33,57))
    args=parser.parse_args();out=args.output.resolve()
    if args.action=='prepare':prepare(out);return
    m=read(out/'manifest.json')
    os.environ['SOFTMODELING_PARDISO_RUNTIME']=m['pardiso_runtime']
    if args.action=='worker':worker(out);return
    if args.action in ('case','preflight'):
        if args.action=='case':job=next(j for j in m['jobs'] if j['name']==args.name)
        else:job=dict(name=f'preflight{args.angle}',angle=args.angle,depth=.01,increment=.01,seconds=300.,kind='preflight')
        r=run_case(out/job['name'],job,m['resources']);print(json.dumps({k:r.get(k) for k in ('status','last_depth_mm','elapsed_s','error')}))
        sys.exit(0 if r['completed'] else 2)
    # Exactly one detached queue, only after actual fine-mesh preflights.
    if (out/'launch.json').exists() or (out/'worker_started.json').exists():raise ValueError('Already launched')
    for angle in (33,57,0):
        path=out/f'preflight{angle}';r=read(path/'result.json')
        if not r['completed'] or r['last_depth_mm']!=.01:raise ValueError('Actual fine preflight failed')
        m['inputs'].update(verify(path))
    for path,h in m['inputs'].items():
        if digest(path)!=h:raise ValueError('Input changed before launch')
    for name,h in m['source_sha256'].items():
        if digest(out/'source'/name)!=h:raise ValueError('Frozen source changed')
    write_json(out/'manifest.json',m)
    cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_native_gcp','worker','--output',str(out)]
    env=runtime_environment(m)
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        proc=subprocess.Popen(cmd,cwd=out/'source',env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**background_child_options())
    row=dict(worker_pid=proc.pid,started=stamp(),command=cmd);write_json(out/'launch.json',row);print(json.dumps(row))


if __name__=='__main__':main()
