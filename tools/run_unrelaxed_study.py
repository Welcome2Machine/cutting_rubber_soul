"""Frozen native IPC unrelaxed G0/K control; three bounded .5 mm paths only.

No old queue revival, cross-model history, material calibration or deep extension.
"""
import argparse
import contextlib
import csv
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.unrelaxed_control import UnrelaxedProblem
from prototype.ipc_reference import BASELINE, record_tip_contact, export_contact_table
from prototype.tip_contact import TipContactTracker
from tools.run_material_candidate import signature as base_signature, audit, resume, load_force_csv
from tools.run_material_queue import summarize_case
from tools.check_contact_transport import BASELINES, read, sha, inspect_case
from tools.render_solver_dashboard import shell, atomic_write, stamp
from run_support import source_hashes, write_json, save_npz, compute_process_policy, background_child_options, classify_process_exit

ROOT=Path(__file__).resolve().parents[1]
POLICY='native-ipc-fixed-G0-K-unrelaxed-v1'
MODULE='tools.run_unrelaxed_study'


def make(angle):
    p=UnrelaxedProblem(instantaneous_shear=1.38,bulk_modulus=23.34333333333331,
        loading_rate=.2,case='oblique',mu=.35,dhat=.0025,local_spacing=.05,
        surface_spacing=.005,nose_facets=64,inclination=angle,tip_radius=.02,
        fine_halfwidth=3.,max_iterations=500,predictor='secant',hessian_mode='regularized',element='p1')
    p.stagnation_window=40;p.ipc.set_num_threads(4)
    return p


def signature(p): return dict(base_signature(p),diagnostic_policy=POLICY)


def snapshot(out):
    hashes=source_hashes(snapshot=out/'source')
    for name in ('prototype/ipc_baseline.json','tools/run_unrelaxed_study.py',
                 'tools/run_material_candidate.py','tools/run_material_queue.py',
                 'tools/check_contact_transport.py','tools/render_solver_dashboard.py'):
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dst);hashes[name]=sha(dst)
    return hashes


def env(m):
    e=os.environ.copy();e.update(PYTHONPATH=m['runtime_path'],
        SOFTMODELING_PARDISO_RUNTIME=m['pardiso_runtime'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    return e


def case_page(out,r):
    z=(r.get('steps') or [{}])[-1]
    body=f'<h1>无松弛对照 {r["angle"]}°</h1><p>{stamp()} · {escape(r["status"])}</p>'
    body+='<p>G₀=1.38 MPa、K=23.343333 MPa；原生IPC、μ=.35、原刀具/网格/固定底面。只去材料松弛，摩擦历史仍演化。是机制对照，不是实验拟合模型。固定步长，失败不缩步。</p>'
    body+='<p>'+escape(f'接受态 {len(r.get("steps",[]))}；d={z.get("depth_mm")} mm；Fz={z.get("reaction_fz_n")} N；minJ={z.get("min_j")}；求解耗时={r.get("solve_s",0):.1f} s')+'</p>'
    body+='<p>'+escape(r.get('error','') or r.get('audit_error',''))+'</p><p><a href="log.html">日志</a> · <a href="result.json">状态</a> · <a href="fz_depth.csv">力CSV</a> · <a href="../index.html">队列</a></p>'
    if (out/'audit.json').exists():body+='<p><a href="audit.json">全部接受历史重新审查</a></p>'
    atomic_write(out/'index.html',shell('无松弛对照',body,refresh=r['status'] in ('running','auditing','initializing')))
    if (out/'run.log').exists():
        with (out/'run.log').open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-120000));log=f.read().decode('utf-8',errors='replace')
        atomic_write(out/'log.html',shell('运行日志','<p>最近120000字节；完整过程见run.log。</p><pre>'+escape(log)+'</pre>',refresh=r['status'] in ('running','auditing','initializing')))


def targets(depth,increment):
    if not 0<depth<=.5 or not 0<increment<=.01:raise ValueError('Bounded diagnostic only')
    return np.r_[0.,np.arange(increment,depth-1e-12,increment),depth]


def run_case(out,job):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);(out/'states').mkdir()
    r=dict(status='initializing',completed=False,angle=job['angle'],target_mm=job['target_mm'],
        settings=dict(increment=.01),source_sha256=snapshot(out),steps=[],rejected_load_attempts=[],
        tip_contact_tracking=dict(errors=[]),job=job,force_tolerance_n=BASELINE['force_tolerance_n'],
        diagnostic_policy=POLICY,compute=dict(material_workers=1,linear_threads=8,ipc_threads=4,blas_threads=1))
    write_json(out/'result.json',r);start=time.monotonic();p=None;x=None;depth=None
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy() as policy:
        r['runtime_process_policy']=policy
        try:
            p=make(job['angle']);r['signature']=json.loads(json.dumps(signature(p)))
            r['reference_contact_check']=p.reference_contact_check
            r['mesh']=dict(nodes=p.n,elements=len(p.mesh.triangles))
            x=p.rest.copy();old=p.start
            if job.get('resume'):
                if read(Path(job['resume'])/'result.json')['settings']['increment']!=.01:raise ValueError('Changed own-prefix schedule')
                x,old=resume(p,Path(job['resume']),out,r)
            p.configure_compute(linear_threads=8)
            tracker=TipContactTracker(p.rest,p.mesh.triangles,p.n,p.config,p.dhat,p.nose_facets)
            r.update(status='running',setup_s=time.monotonic()-start,solve_s=0.)
            write_json(out/'result.json',r);case_page(out,r);tick=time.monotonic();deadline=tick+job['seconds']
            for depth in targets(job['target_mm'],.01):
                if depth<=old+1e-12:continue
                step_start=time.monotonic();x,row=p.step(x,old,float(depth),deadline)
                old=float(depth);row['wall_step_s']=time.monotonic()-step_start
                save_npz(out/'states'/f'{len(r["steps"]):04d}.npz',reference_xy=p.rest,triangles=p.mesh.triangles,
                    positions=x,previous_positions=p.previous,depth_mm=old,body_nodes=p.n,**p.checkpoint_data())
                record_tip_contact(tracker,x,row,r);r['steps'].append(row)
                r.update(last_depth_mm=old,solve_s=time.monotonic()-tick)
                write_json(out/'result.json',r);case_page(out,r)
                print(f'ACCEPT depth_mm={old:.6f} Fz_N={row["reaction_fz_n"]:.9g} minJ={row["min_j"]:.9g} step_seconds={row["wall_step_s"]:.3f}',flush=True)
            r.update(status='completed',completed=True)
        except Exception as exc:
            r.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',completed=False,error=f'{type(exc).__name__}: {exc}')
            if 'tick' in locals():r['solve_s']=time.monotonic()-tick
            r['rejected_load_attempts'].append(dict(attempted_depth_mm=float(depth) if depth is not None else None,error=repr(exc)))
            traceback.print_exc(file=log)
            if p is not None and x is not None and p.material_dt is not None:
                save_npz(out/'failed_trial.npz',positions=getattr(p,'trial',x),previous_positions=p.previous,
                    reference_xy=p.rest,triangles=p.mesh.triangles,attempted_depth_mm=depth,**p.checkpoint_data())
        finally:
            if p is not None:
                r['linear_solve_statistics']=dict(calls=len(p.parallel_linear.rows),
                    fallback_count=sum(v['fallback_reason'] is not None for v in p.parallel_linear.rows),
                    total_s=sum(v['seconds'] for v in p.parallel_linear.rows)) if p.parallel_linear is not None else {}
                p.close_compute()
        terminal=r['status'];r['status']='auditing';write_json(out/'result.json',r);case_page(out,r)
        try:
            audit_start=time.monotonic()
            if r['steps']:audit(make(job['angle']),out,r,deadline=time.monotonic()+600.)
            r.update(audit_s=time.monotonic()-audit_start,status=terminal)
        except Exception as exc:
            r.update(status='audit_failed',completed=False,audit_error=repr(exc));traceback.print_exc(file=log)
        r['elapsed_s']=time.monotonic()-start
    export_contact_table(out,r)
    with (out/'fz_depth.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['depth_mm','fz_n']);w.writerows((z['depth_mm'],z['reaction_fz_n']) for z in r['steps'])
    write_json(out/'result.json',r);case_page(out,r)
    return r


def verify_case(folder):
    r=read(folder/'result.json');a=read(folder/'audit.json');hashes={}
    if not a['passed'] or a['states']!=len(r['steps']):raise ValueError('Unaudited prefix')
    for name,h in r['source_sha256'].items():
        p=folder/'source'/name
        if sha(p)!=h:raise ValueError('Source changed')
        hashes[str(p.resolve())]=h
    for i,row in enumerate(a['rows']):
        p=folder/'states'/f'{i:04d}.npz'
        if not row['passed'] or sha(p)!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        hashes[str(p.resolve())]=row['checkpoint_sha256']
    for name in ('result.json','audit.json'):hashes[str((folder/name).resolve())]=sha(folder/name)
    return hashes


def render_queue(out,m,state):
    body=f'<h1>固定G₀/K：无松弛三角度对照</h1><p>{stamp()} · {escape(state["status"])}</p>'
    body+='<p>仅0/33/57°各至.5 mm；独立从静止或自己的.01 mm预检恢复。固定.01 mm步长，原500次PTC和原验收；失败/超时转下一组，不缩步、不追加预算、不自动延伸。每组自动审查全部接受历史。</p>'
    body+='<table><tr><th>角度</th><th>状态</th><th>接受深度 mm</th><th>Fz N</th><th>求解预算 s</th><th>日志</th></tr>'
    for job in state['jobs']:
        folder=out/job['name'];r=read(folder/'result.json') if (folder/'result.json').exists() else {};z=(r.get('steps') or [{}])[-1]
        body+=f'<tr><td>{job["angle"]}</td><td>{escape(job["status"])}</td><td>{z.get("depth_mm","—")}</td><td>{z.get("reaction_fz_n","—")}</td><td>{job["seconds"]}</td><td>'+(f'<a href="{job["name"]}/index.html">状态/日志</a>' if r else '待运行')+'</td></tr>'
        if r:case_page(folder,r)
    body+='</table><p><a href="contact_review/index.html">历史接触/材料运动分析</a> · <a href="analysis.html">本轮自动比较</a> · <a href="qualification.html">启动资格</a> · <a href="manifest.json">冻结方案</a></p>'
    body+='<p>当前材料是因果诊断，不预期直接拟合实验。缺失刀尖身份不填零；完成不是空间/滑移/实验验证。57°参与过材料选参，33°仅截图范围，0°无选定数值实验曲线。</p>'
    atomic_write(out/'index.html',shell('无松弛对照队列',body,refresh=state['status']=='running'))
    write_json(out/'queue_status.json',state)


def analyze(out,m,state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];fig,axs=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True,layout='constrained')
    curves={}
    for ax,job in zip(axs,state['jobs']):
        angle=job['angle'];base=read(Path(m['baselines'][str(angle)])/'result.json')
        br=[z for z in base['steps'] if z['depth_mm']<=.5+1e-10];bd=np.array([z['depth_mm'] for z in br]);bf=np.array([z['reaction_fz_n'] for z in br])
        ax.plot(bd,bf,'--',color='#0072B2',label='Current Maxwell')
        folder=out/job['name']
        if (folder/'result.json').exists():
            r=read(folder/'result.json');rr=r['steps'];d=np.array([z['depth_mm'] for z in rr]);f=np.array([z['reaction_fz_n'] for z in rr])
            if len(rr):
                ax.plot(d,f,'-',color='#D55E00',label='Unrelaxed control')
                curves[angle]=(d,f)
                item=dict(angle=angle,status=job['status'],accepted_states=len(rr),end_mm=float(d[-1]),
                    min_j=min(z['min_j'] for z in rr),solve_s=r.get('solve_s'),prior_elapsed_s=r.get('prior_elapsed_s'),anchors=[])
                for a in (.25,.5):
                    if d[-1]>=a-1e-10:item['anchors'].append(dict(depth_mm=a,unrelaxed_n=float(np.interp(a,d,f)),maxwell_n=float(np.interp(a,bd,bf))))
                if d[-1]>=.5-1e-10:
                    k1=(np.interp(.25,d,f)-f[0])/.25;k2=(np.interp(.5,d,f)-np.interp(.25,d,f))/.25
                    bk1=(np.interp(.25,bd,bf)-bf[0])/.25;bk2=(bf[-1]-np.interp(.25,bd,bf))/.25
                    item['quarter_mm_secants_n_mm']=dict(unrelaxed=[k1,k2],maxwell=[bk1,bk2])
                    item['secant_growth_ratio']=dict(unrelaxed=k2/k1,maxwell=bk2/bk1)
                rows.append(item)
        if angle==57:
            ed,ef,_=load_force_csv(out/'experiment57.csv');mask=ed<=.5
            ax.plot(ed[mask],ef[mask],color='#009E73',lw=1,label='Experiment 1014 (selected)')
        if angle==33:ax.errorbar([.25,.5],[4.,8.5],yerr=[1.,1.],fmt='s',color='#009E73',label='Image readings (visual range)')
        ax.set(title=f'{angle} deg',xlabel='Vertical depth (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[0].set_ylabel('Fz (N)');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    ratios=[]
    if 0 in curves:
        for angle in (33,57):
            if angle not in curves:continue
            for d in (.25,.5):
                if min(curves[0][0][-1],curves[angle][0][-1])<d-1e-10:continue
                zero=np.interp(d,*curves[0]);incl=np.interp(d,*curves[angle])
                baseline=[]
                for a in (0,angle):
                    b=read(Path(m['baselines'][str(a)])/'result.json')['steps']
                    baseline.append(float(np.interp(d,[v['depth_mm'] for v in b],[v['reaction_fz_n'] for v in b])))
                ratios.append(dict(angle=angle,depth_mm=d,unrelaxed_reduction=1-incl/zero,maxwell_reduction=1-baseline[1]/baseline[0]))
    write_json(out/'analysis.json',dict(rows=rows,angle_reductions=ratios,scope='Accepted prefixes only; diagnostic, not fitted candidate',no_extrapolation=True))
    body='<h1>本轮对照：已有接受范围</h1><p>未完成的深度留空，不外推。三图共同纵轴；原始力未重新归零。诊断对照力偏大不自动判定失败，主要看曲率、角度差与材料运动。原Maxwell不是实验真值。</p><img src="comparison.png" style="max-width:100%" alt="三个角度无松弛与Maxwell及各自可用实验的反力比较">'
    body+='<p>33°误差条仅人工读图范围，非置信区间；57°为选参试次；0°没有数值实验对照。各角度试次不同，不构成配对实验角度比。</p><pre>'+escape(json.dumps(dict(rows=rows,angle_reductions=ratios),ensure_ascii=False,indent=2))+'</pre><p><a href="index.html">队列</a></p>'
    if (out/'control_contact.json').exists():body+='<p><a href="control_contact.json">接受路径接触承载/材料迁移锚点</a></p>'
    atomic_write(out/'analysis.html',shell('无松弛对照分析',body))


def prepare(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Do not overwrite an existing study')
    review=read(out/'contact_review/report.json')
    if review['status']!='completed' or review['new_loading_solves']!=0:raise ValueError('Saved-history audit incomplete')
    tests=read(out/'tests.json')
    if not tests['passed'] or tests['skipped']:raise ValueError('Required regressions not passed')
    inputs={}
    for c in review['cases']:inputs.update(c['input_sha256'])
    for name in ('contact_review/report.json','tests.json'):inputs[str(out/name)]=sha(out/name)
    shutil.copy2(ROOT/'docs/1014n57a1__Fz__curve_0.csv',out/'experiment57.csv')
    inputs[str(out/'experiment57.csv')]=sha(out/'experiment57.csv')
    jobs=[dict(name=f'angle{a}',angle=a,target_mm=.5,seconds=3600,status='pending',resume=str(out/f'pilot{a}')) for a in (0,33,57)]
    m=dict(policy=POLICY,created=stamp(),source_sha256=snapshot(out),inputs=inputs,jobs=jobs,
        runtime_path=str(ROOT/'.runtime/ipc-python'),pardiso_runtime=str(ROOT/'.runtime/pardiso'),
        baselines={str(a):str(ROOT/p) for a,p in BASELINES.items()},python=sys.executable,
        budget_rule='ceil(3*50*pilot_solve_seconds/2), clipped to [900,3600]; audit600 + setup300; NOT ETA',
        only_own_prefix=True,no_deep_extension=True)
    write_json(out/'manifest.json',m)
    qualify=[]
    for job in jobs:
        pilot=dict(name=f'pilot{job["angle"]}',angle=job['angle'],target_mm=.01,seconds=300,status='pending')
        cmd=[sys.executable,'-X','utf8','-u','-m',MODULE,'case','--output',str(out),'--job-json',json.dumps(pilot)]
        forced=None
        with (out/(pilot['name']+'.stdout.log')).open('wb') as stdout,(out/(pilot['name']+'.stderr.log')).open('wb') as stderr:
            try:
                code=subprocess.run(cmd,cwd=out/'source',env=env(m),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                    timeout=1200,**background_child_options()).returncode
            except subprocess.TimeoutExpired:code=-1;forced='preflight_supervisor_timeout'
        folder=out/pilot['name'];summary=summarize_case(folder,pilot,code,forced_status=forced)
        if summary['mechanics_completed']:
            inputs.update(verify_case(folder));pr=read(folder/'result.json')
            base=read(Path(m['baselines'][str(job['angle'])])/'result.json')
            if pr['signature']['mesh_sha256']!=base['signature']['mesh_sha256']:raise ValueError('Changed geometry/mesh')
            a=pr['signature']['config'].copy();b=base['signature']['config'].copy()
            for key in ('shear_modulus_n_mm2','poisson_ratio'):a.pop(key);b.pop(key)
            if a!=b:raise ValueError('Unintended model setting change')
            job['seconds']=int(np.clip(math.ceil(75*pr['solve_s']),900,3600))
        else:job.update(status='not_qualified',reason='Own .01 mm preflight failed; no retry')
        qualify.append(summary)
    m['inputs']=inputs;m['jobs']=jobs;write_json(out/'manifest.json',m)
    qualification=dict(pilots=qualify,regressions=tests,all_preflights_passed=all(r['mechanics_completed'] for r in qualify),
        budget_rule=m['budget_rule'],physical_residual_unchanged=True)
    write_json(out/'qualification.json',qualification)
    atomic_write(out/'qualification.html',shell('本轮资格','<h1>无松弛实现与真实细网格预检</h1><p>预检不是.5 mm结果。全部接受态独立审查；瞬时G/K固定、其余配置和网格与各自原生Maxwell严格核对。预算是上限，不是耗时预测。</p><pre>'+escape(json.dumps(qualification,ensure_ascii=False,indent=2))+'</pre>'))
    state=dict(status='prepared',jobs=jobs);analyze(out,m,state);render_queue(out,m,state)
    print(json.dumps(qualification))


def check_hashes(m,out):
    for name,h in m['source_sha256'].items():
        if sha(out/'source'/name)!=h:raise ValueError('Frozen source changed: '+name)
    for path,h in m['inputs'].items():
        if sha(path)!=h:raise ValueError('Frozen input changed: '+path)


def worker(out):
    m=read(out/'manifest.json')
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),time=stamp()),f)
    check_hashes(m,out)
    state=dict(status='running',jobs=m['jobs']);render_queue(out,m,state);contacts=[]
    for job in state['jobs']:
        if job['status']=='not_qualified':continue
        try:
            cmd=[sys.executable,'-X','utf8','-u','-m',MODULE,'case','--output',str(out),'--job-json',json.dumps(job)]
            with (out/(job['name']+'.stdout.log')).open('wb') as stdout,(out/(job['name']+'.stderr.log')).open('wb') as stderr:
                proc=subprocess.Popen(cmd,cwd=out/'source',env=env(m),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**background_child_options())
                job.update(status='running',pid=proc.pid,started=stamp());render_queue(out,m,state);tick=time.monotonic();timed=False
                while proc.poll() is None:
                    if time.monotonic()-tick>job['seconds']+900:
                        timed=True;proc.terminate();proc.wait(timeout=30);break
                    render_queue(out,m,state);time.sleep(10)
            summary=summarize_case(out/job['name'],job,proc.returncode,
                forced_status=classify_process_exit(proc.returncode,supervisor_timeout=timed) if timed or proc.returncode else None)
            job.update(status=summary['status'],summary=summary,wall_s=time.monotonic()-tick)
            r=read(out/job['name']/'result.json')
            if r['steps'] and summary['fresh_audit_passed']:
                verify_case(out/job['name'])
                try:
                    with (out/(job['name']+'.contact.log')).open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),threadpool_limits(1):
                        contacts.append(inspect_case(out/job['name'],job['name'],end=.5))
                    write_json(out/'control_contact.json',dict(cases=contacts,loading_solves=0))
                except Exception as exc:job['contact_analysis_error']=repr(exc)
        except Exception as exc:
            job.update(status='failed',error=repr(exc));traceback.print_exc()
        finally:
            try:analyze(out,m,state)
            except Exception as exc:job['analysis_error']=repr(exc)
            render_queue(out,m,state)
    state.update(status='finished',finished=stamp());analyze(out,m,state);render_queue(out,m,state)


def launch(out):
    m=read(out/'manifest.json');check_hashes(m,out)
    if (out/'launch.json').exists() or (out/'worker_started.json').exists():raise ValueError('Study already launched')
    if not any(j['status']=='pending' for j in m['jobs']):raise ValueError('No qualified jobs')
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m',MODULE,'worker','--output',str(out)],
            cwd=out/'source',env=env(m),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**background_child_options())
    value=dict(pid=p.pid,time=stamp(),source_count=len(m['source_sha256']),input_count=len(m['inputs']))
    write_json(out/'launch.json',value);print(json.dumps(value))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['prepare','case','worker','launch','report'])
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--job-json');args=ap.parse_args();out=args.output.resolve()
    if args.phase=='case':run_case(out/json.loads(args.job_json)['name'],json.loads(args.job_json))
    elif args.phase=='prepare':prepare(out)
    elif args.phase=='worker':worker(out)
    elif args.phase=='launch':launch(out)
    else:
        m=read(out/'manifest.json');s=read(out/'queue_status.json');analyze(out,m,s);render_queue(out,m,s)


if __name__=='__main__':main()
