"""Bounded common-Maxwell FE response study, with one joint proposal at most.

Every parameter set is shared by 0/33/57; each path owns its full histories.
The affine response fit proposes ONE independently re-solved candidate. It is
never presented as an FE force. No old queue restart or automatic cutbacks.
"""
import argparse
import contextlib
import csv
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
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from prototype.ipc_viscoelastic import ViscoelasticProblem
from prototype.ipc_reference import BASELINE, record_tip_contact, export_contact_table
from prototype.tip_contact import TipContactTracker
from prototype.pardiso import PardisoDirection
from prototype.material_case import make_problem, signature, audit, resume, load_force_csv
from tools.run_material_queue import summarize_case
from tools.render_solver_dashboard import shell, atomic_write, stamp
from run_support import (source_hashes, write_json, save_npz, compute_process_policy,
                         background_child_options, classify_process_exit)

ROOT=Path(__file__).resolve().parents[1]
MODULE='tools.run_common_material_study'
ANGLES=(0,33,57)
BASE=np.array([.14,1.24,1.7])
K=23.34333333333331
PROBES={'soft':BASE*np.array([.7,.7,1.]),
        'fast':BASE*np.array([1.,1.,.5]),
        'partition':np.array([1.38*.3,1.38*.7,1.7])}
REFERENCES={0:'outputs/relaxation_partition_20261015/angles30_0_2mm/02_angle0',
            33:'outputs/native_angles_2mm_20261016/01_angle33',
            57:'outputs/native_angles_2mm_20261016/02_angle57'}
EXPERIMENTS={0:'1008n0a2__Fz__curve_1.csv',33:'1008n33a1__Fz__curve_2.csv',57:'1014n57a1__Fz__curve_0.csv'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ComputeMaterial(ViscoelasticProblem):
    """Only replace the linear backend; same guarded FP64 physical equations."""
    def linear_direction(self,matrix,rhs):
        if self.parallel_linear is None:return super().linear_direction(matrix,rhs)
        value=self.parallel_linear.solve(matrix,rhs)
        residual=np.linalg.norm(matrix@value-rhs)
        if not np.isfinite(value).all() or not np.isfinite(residual) or residual>max(1e-8*np.linalg.norm(rhs),1e-10):
            raise RuntimeError('Physical linear residual failed')
        return value


def make(angle,parameters):
    ge,gv,tau=map(float,parameters)
    p=make_problem(angle=angle,equilibrium_shear=ge,maxwell_shear=gv,relaxation_s=tau,
                   loading_rate=.2,bulk_modulus=K,mu=.35)
    p.__class__=ComputeMaterial;p.parallel_linear=None;p.ipc.set_num_threads(4)
    return p


def schedule(end):
    if not np.isfinite(end) or not 0<end<=.8:raise ValueError('Bounded .8 mm maximum')
    return np.r_[0.,np.arange(.01,end-1e-12,.01),end]


def coordinates(parameters):
    ge,gv,tau=np.asarray(parameters,dtype=float)
    if not np.isfinite([ge,gv,tau]).all() or min(ge,gv,tau)<=0:raise ValueError('Positive parameters required')
    return np.array([np.log(ge+gv),np.log(ge/gv),np.log(tau)])


def decode(value):
    g0=np.exp(value[0]);fraction=1/(1+np.exp(-value[1]))
    return np.array([g0*fraction,g0*(1-fraction),np.exp(value[2])])


def joint_proposal(control,probes,target):
    """Convex-hull response approximation; NOT candidate equilibrium forces."""
    A=np.column_stack([np.asarray(probes[k])-control for k in PROBES])
    b=np.asarray(target)-control
    fit=minimize(lambda c:float(np.sum((A@c-b)**2)),np.zeros(3),
                 jac=lambda c:2*A.T@(A@c-b),bounds=[(0.,1.)]*3,
                 constraints=[dict(type='ineq',fun=lambda c:1.-c.sum(),jac=lambda c:-np.ones(3))],
                 method='SLSQP',options=dict(maxiter=200,ftol=1e-11))
    if not fit.success:raise RuntimeError('Bounded response proposal failed: '+fit.message)
    c=fit.x
    parameters=decode(coordinates(BASE)+sum(c[i]*(coordinates(PROBES[k])-coordinates(BASE)) for i,k in enumerate(PROBES)))
    return dict(weights=c.tolist(),parameters=parameters.tolist(),predicted_residual_norm=float(np.linalg.norm(A@c-b)),
                original_residual_norm=float(np.linalg.norm(b)),singular_values=np.linalg.svd(A,compute_uv=False).tolist(),
                scope='Finite-direction response interpolation, not FE forces or global identification',
                bounds='nonnegative weights, sum<=1; log G0/logit shear fraction/log tau within sampled tetrahedron')


def features(curves):
    grid=np.arange(10,51)/100
    values=[]
    for a in ANGLES:
        d,f=curves[a]
        if d[0]>.1 or d[-1]<.5-1e-10:raise ValueError('Complete .1--.5 comparison needed')
        v=np.interp(grid,d,f)
        values.extend(v/np.sqrt(3*len(grid)))
        # Increment-force constraints share N units and are independent of offsets.
        inc=np.interp([.1,.25,.5],d,f)
        values.extend(.5*np.diff(inc)/np.sqrt(6))
    return np.array(values)


def metrics(curves,experiment):
    result={};grid=np.arange(10,51)/100
    for a in ANGLES:
        d,f=curves[a];ed,ef=experiment[a]
        if d[-1]<.5-1e-10:raise ValueError('Missing target; no extrapolation')
        e=np.interp(grid,ed,ef);v=np.interp(grid,d,f);err=v-e
        points=np.array([.1,.25,.5]);es=np.diff(np.interp(points,ed,ef))/np.diff(points)
        fs=np.diff(np.interp(points,d,f))/np.diff(points)
        result[str(a)]=dict(rmse_n=float(np.sqrt(np.mean(err**2))),bias_n=float(err.mean()),
            max_abs_n=float(np.max(abs(err))),experimental_secants_n_mm=es.tolist(),fe_secants_n_mm=fs.tolist(),
            maximum_secant_relative_error=float(np.max(abs(fs/es-1))))
    return dict(angles=result,joint_rmse_n=float(np.sqrt(np.mean([v['rmse_n']**2 for v in result.values()]))),
                worst_angle_rmse_n=max(v['rmse_n'] for v in result.values()))


def promising(candidate,baseline):
    return (candidate['joint_rmse_n']<=.8*baseline['joint_rmse_n']
        and candidate['worst_angle_rmse_n']<=.8*baseline['worst_angle_rmse_n']
        and all(v['rmse_n']<=1.5 and v['max_abs_n']<=2.
                and v['maximum_secant_relative_error']<=.2 for v in candidate['angles'].values()))


def training_jobs(jobs,group):
    return [j for j in jobs if j.get('group')==group and j.get('stage')=='train']


def snapshot(out):
    hashes=source_hashes(snapshot=out/'source')
    for name in ('prototype/ipc_baseline.json','tools/run_common_material_study.py',
                 'tools/run_material_candidate.py','tools/run_material_queue.py','tools/render_solver_dashboard.py'):
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest);hashes[name]=sha(dest)
    return hashes


def verify_case(folder):
    r=read(folder/'result.json');a=read(folder/'audit.json');files={}
    if not a['passed'] or a['states']!=len(r['steps']) or len(a['rows'])!=len(r['steps']):raise ValueError('Incomplete audit')
    for name,h in r['source_sha256'].items():files[str(folder/'source'/name)]=h
    for i,row in enumerate(a['rows']):
        if not row['passed']:raise ValueError('Rejected audit state')
        files[str(folder/'states'/f'{i:04d}.npz')]=row['checkpoint_sha256']
    for name in ('result.json','audit.json'):files[str(folder/name)]=sha(folder/name)
    for path,h in files.items():
        if sha(path)!=h:raise ValueError('Changed case input: '+path)
    return files


def check(m,out):
    for name,h in m['source_sha256'].items():
        if sha(out/'source'/name)!=h:raise ValueError('Frozen source changed: '+name)
    for name,h in m['inputs'].items():
        if sha(name)!=h:raise ValueError('Study input changed: '+name)


def case(out,job):
    m=read(out/'manifest.json');check(m,out);folder=out/job['name'];folder.mkdir(exist_ok=False);(folder/'states').mkdir()
    for name in m['source_sha256']:
        dest=folder/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(out/'source'/name,dest)
    r=dict(status='initializing',completed=False,target_mm=job['target_mm'],steps=[],rejected_load_attempts=[],
           source_sha256=m['source_sha256'],settings=dict(increment=.01),job=job,
           tip_contact_tracking=dict(errors=[]),force_tolerance_n=BASELINE['force_tolerance_n'])
    write_json(folder/'result.json',r);began=time.monotonic();p=None;x=None;depth=None
    with (folder/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy() as policy:
        r['runtime_process_policy']=policy
        try:
            p=make(job['angle'],job['parameters']);r['signature']=json.loads(json.dumps(signature(p)))
            r['reference_contact_check']=p.reference_contact_check;r['mesh']=dict(nodes=p.n,elements=len(p.mesh.triangles))
            x=p.rest.copy();old=p.start
            if job.get('resume'):x,old=resume(p,Path(job['resume']),folder,r)
            p.parallel_linear=PardisoDirection(8)
            tracker=TipContactTracker(p.rest,p.mesh.triangles,p.n,p.config,p.dhat,p.nose_facets)
            r.update(status='running',setup_s=time.monotonic()-began);write_json(folder/'result.json',r)
            tick=time.monotonic();deadline=tick+job['seconds']
            for depth in schedule(job['target_mm']):
                if depth<=old+1e-12:continue
                step_tick=time.monotonic();x,row=p.step(x,old,float(depth),deadline);old=float(depth)
                row['wall_step_s']=time.monotonic()-step_tick
                save_npz(folder/'states'/f'{len(r["steps"]):04d}.npz',reference_xy=p.rest,triangles=p.mesh.triangles,
                    positions=x,previous_positions=p.previous,depth_mm=old,body_nodes=p.n,**p.checkpoint_data())
                record_tip_contact(tracker,x,row,r);r['steps'].append(row)
                r.update(last_depth_mm=old,solve_s=time.monotonic()-tick);write_json(folder/'result.json',r)
                print(f'ACCEPT depth_mm={old:.6f} Fz_N={row["reaction_fz_n"]:.9g} minJ={row["min_j"]:.9g} step_s={row["wall_step_s"]:.3f}',flush=True)
            r.update(status='completed',completed=True)
        except Exception as exc:
            r.update(status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed',completed=False,error=repr(exc));traceback.print_exc()
            if 'tick' in locals():r['solve_s']=time.monotonic()-tick
            r['rejected_load_attempts'].append(dict(depth_mm=None if depth is None else float(depth),error=repr(exc)))
            if p is not None and x is not None and p.material_dt is not None:
                save_npz(folder/'failed_trial.npz',positions=getattr(p,'trial',x),previous_positions=p.previous,
                    reference_xy=p.rest,triangles=p.mesh.triangles,**p.checkpoint_data())
        finally:
            if p is not None and p.parallel_linear is not None:
                r['linear_statistics']=dict(calls=len(p.parallel_linear.rows),
                    total_s=sum(v['seconds'] for v in p.parallel_linear.rows),
                    fallbacks=sum(v['fallback_reason'] is not None for v in p.parallel_linear.rows))
                p.parallel_linear.close();p.parallel_linear=None
        terminal=r['status'];r['status']='auditing';write_json(folder/'result.json',r)
        try:
            t=time.monotonic()
            if r['steps']:audit(make(job['angle'],job['parameters']),folder,r,deadline=t+600.)
            r.update(status=terminal,audit_s=time.monotonic()-t)
        except Exception as exc:r.update(status='audit_failed',completed=False,audit_error=repr(exc));traceback.print_exc()
        r['elapsed_s']=time.monotonic()-began
    export_contact_table(folder,r)
    with (folder/'fz_depth.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['depth_mm','fz_n']);w.writerows((s['depth_mm'],s['reaction_fz_n']) for s in r['steps'])
    write_json(folder/'result.json',r);return r


def child_env(m):
    env=os.environ.copy();env.update(PYTHONPATH=m['runtime_path'],SOFTMODELING_PARDISO_RUNTIME=m['pardiso_runtime'],
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');return env


def case_page(folder):
    if not (folder/'result.json').exists():return
    r=read(folder/'result.json');last=(r['steps'] or [{}])[-1]
    body='<h1>'+escape(folder.name)+'</h1><p>'+escape(stamp())+' · '+escape(r['status'])+'</p>'
    body+='<p>'+escape(str(dict(accepted=len(r['steps']),depth_mm=last.get('depth_mm'),fz_n=last.get('reaction_fz_n'),
                              min_j=last.get('min_j'),parameters=r['job']['parameters'])))+'</p>'
    body+='<p>自己的准确材料/摩擦历史，固定.01 mm、原500次PTC、原验收。共用参数，不按角度拟合。</p>'
    body+='<p>'+escape(r.get('error','') or r.get('audit_error',''))+'</p><p><a href="../index.html">队列</a> · <a href="result.json">状态JSON</a> · <a href="fz_depth.csv">力CSV（终态生成）</a></p>'
    if (folder/'run.log').exists():
        with (folder/'run.log').open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-60000));body+='<pre>'+escape(f.read().decode('utf-8',errors='replace'))+'</pre>'
    atomic_write(folder/'index.html',shell('共同材料算例日志',body,refresh=r['status'] in ('running','auditing','initializing')))


def render(out,state):
    body=f'<h1>共同材料跨角度可行性检验</h1><p>{stamp()} · {escape(state["status"])}</p>'
    body+='<p>0/33/57°等权；暂定较早训练窗口.1–.5 mm。三方向各三角度独立从静止加载；最多一个组合候选。组合响应近似只用于提案，最终必须真实FE重算。后段已裂不参与选参。</p>'
    body+='<p>0°为a2，机械首触/起裂未独立确认；57°初始.51 N保留。数值通过不等于未裂、空间/材料/滑移验证。已看过的数据仅作探索交叉检验。</p>'
    body+='<table><tr><th>任务</th><th>角度</th><th>状态</th><th>已接受深度mm</th><th>Fz N</th><th>日志</th></tr>'
    for job in state['jobs']:
        folder=out/job['name'];r=read(folder/'result.json') if (folder/'result.json').exists() else {};z=(r.get('steps') or [{}])[-1]
        body+=f'<tr><td>{escape(job["name"])}</td><td>{job["angle"]}</td><td>{escape(job["status"])}</td><td>{z.get("depth_mm","—")}</td><td>{z.get("reaction_fz_n","—")}</td><td>'+(f'<a href="{job["name"]}/index.html">状态/日志</a>' if r else '待运行')+'</td></tr>'
        if r:case_page(folder)
    body+='</table><p><a href="plan.html">边界、门槛、预算与资格</a> · <a href="analysis.html">自动联合分析</a> · <a href="manifest.json">冻结方案</a></p>'
    atomic_write(out/'index.html',shell('共同材料可行性',body,refresh=state['status']=='running'));write_json(out/'queue_status.json',state)


def supervise(out,m,job,state):
    check(m,out);folder=out/job['name']
    if folder.exists():raise ValueError('Refuse existing case')
    cmd=[m['python'],'-X','utf8','-u','-m',MODULE,'case','--output',str(out),'--job-json',json.dumps(job)]
    forced=None;tick=time.monotonic()
    with (out/(job['name']+'.stdout.log')).open('wb') as stdout,(out/(job['name']+'.stderr.log')).open('wb') as stderr:
        p=subprocess.Popen(cmd,cwd=out/'source',env=child_env(m),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**background_child_options())
        job.update(status='running',pid=p.pid,started=stamp(),command=cmd);render(out,state)
        while p.poll() is None:
            if time.monotonic()-tick>job['seconds']+900:
                forced='supervisor_timeout';p.terminate();p.wait(timeout=30);break
            render(out,state);time.sleep(10)
    if p.returncode and not forced:forced=classify_process_exit(p.returncode)
    summary=summarize_case(folder,job,p.returncode,forced_status=forced)
    job.update(status=summary['status'],summary=summary,wall_s=time.monotonic()-tick)
    if summary['fresh_audit_passed']:job['verified_input_count']=len(verify_case(folder))
    write_json(out/(job['name']+'.outcome.json'),job);render(out,state);return bool(summary['mechanics_completed'])


def run_job(out,m,job,state):
    try:
        if not job.get('resume'):
            pilot=dict(job,name='pilot_'+job['name'],stage='preflight',target_mm=.01,seconds=180,status='pending')
            state['jobs'].append(pilot)
            if not supervise(out,m,pilot,state):
                job.update(status='not_qualified',reason='Own .01 mm preflight failed');return False
            job['resume']=str(out/pilot['name'])
        return supervise(out,m,job,state)
    except Exception as exc:
        job.update(status='worker_case_failed',error=repr(exc));traceback.print_exc();render(out,state);return False


def curves_from(out,names):
    curves={}
    for a in ANGLES:
        r=read(out/names[a]/'result.json');rows=r['steps']
        curves[a]=(np.array([v['depth_mm'] for v in rows]),np.array([v['reaction_fz_n'] for v in rows]))
    return curves


def analysis(out,m,state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ex={a:load_force_csv(out/f'experiment{a}.csv')[:2] for a in ANGLES}
    bas={a:(np.array([s['depth_mm'] for s in read(Path(m['baselines'][str(a)])/'result.json')['steps']]),
            np.array([s['reaction_fz_n'] for s in read(Path(m['baselines'][str(a)])/'result.json')['steps']])) for a in ANGLES}
    sets={'baseline':metrics(bas,ex)};names={};curves={'baseline':bas}
    for group in list(PROBES)+['joint']:
        jobs=training_jobs(state['jobs'],group)
        if len(jobs)==3 and all(j.get('summary',{}).get('mechanics_completed') for j in jobs):
            ns={j['angle']:j['name'] for j in jobs};cc=curves_from(out,ns)
            sets[group]=metrics(cc,ex);names[group]=ns;curves[group]=cc
    write_json(out/'analysis.json',dict(status=state['status'],completed_sets=sets,
        only_complete_three_angle_sets=True,provisional_intact_window_mm=[.1,.5],new_fit_is_not_independent_validation=True))
    fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained',sharex=True,sharey=True)
    ymax=max(14.,1.08*max(float(f[d<=.5+1e-10].max()) for group in curves.values() for d,f in group.values()))
    for ax,a in zip(axs,ANGLES):
        ax.plot(*ex[a],c='black',lw=1.8,label='Experiment')
        for key,style in zip(curves,['--','-.',':','-','--']):ax.plot(*curves[key][a],ls=style,label=key)
        ax.set(title=f'{a} deg',xlabel='Depth (mm)',xlim=(0,.52),ylim=(0,ymax));ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[0].set_ylabel('Fz (N)');fig.suptitle('Actual complete FE sets only | raw force, no fitted shifts or scaling')
    fig.savefig(out/'comparison.png',dpi=155);plt.close(fig)
    body='<h1>共同参数：自动实际FE比较</h1><p>只比较三角度均完整且终态审查通过的参数组。未完成组留在队列表中，不用缺失深度外推。以下误差为工程筛查，不是统计置信区间或已认证未裂。</p><img src="comparison.png" style="width:100%" alt="三角度实际完整FE参数组与实验比较">'
    body+='<table><tr><th>参数组</th><th>等权联合RMSE N</th><th>0° RMSE</th><th>33° RMSE</th><th>57° RMSE</th><th>早期门槛</th></tr>'
    for k,v in sets.items():
        body+=f'<tr><td>{k}</td><td>{v["joint_rmse_n"]:.4f}</td>'+''.join(f'<td>{v["angles"][str(a)]["rmse_n"]:.4f}</td>' for a in ANGLES)+f'<td>{promising(v,sets["baseline"])}</td></tr>'
    body+='</table><p><a href="analysis.json">分角度斜率、幅值指标</a> · <a href="index.html">队列</a></p>'
    if (out/'proposal.json').exists():body+='<p><a href="proposal.json">有限方向响应提案（不是FE力）</a></p>'
    if state.get('decision'):body+='<pre>'+escape(json.dumps(state['decision'],ensure_ascii=False,indent=2))+'</pre>'
    if (out/'depth_check.json').exists():
        review=read(out/'depth_check.json')
        body+='<h2>.5–.8 mm迁移检查</h2><p>不使用这些深度重新选参；只是暂定较早窗口，非独立盲验证或未裂证明。</p><table><tr><th>角度</th><th>RMSE N</th><th>最大力差N</th></tr>'
        for row in review['rows']:body+=f'<tr><td>{row["angle"]}</td><td>{row["rmse_n"]:.4f}</td><td>{row["max_abs_n"]:.4f}</td></tr>'
        body+='</table><p><a href="depth_check.json">迁移检查证据</a></p>'
    atomic_write(out/'analysis.html',shell('共同材料实际FE分析',body));return sets,names,curves,ex


def worker(out):
    m=read(out/'manifest.json');check(m,out)
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),started=stamp()),f)
    state=dict(status='running',jobs=m['jobs']);render(out,state)
    for job in list(state['jobs']):
        run_job(out,m,job,state);analysis(out,m,state)
    sets,names,curves,ex=analysis(out,m,state)
    if all(k in sets for k in PROBES):
        try:
            proposal=joint_proposal(features(curves['baseline']),{k:features(curves[k]) for k in PROBES},features(ex))
            write_json(out/'proposal.json',proposal)
            distance={k:float(np.linalg.norm(coordinates(v)-coordinates(proposal['parameters']))) for k,v in PROBES.items()}
            near=min(distance,key=distance.get)
            if distance[near]<1e-5:state['proposal_reuses_complete_probe']=near
            elif proposal['predicted_residual_norm']<.95*proposal['original_residual_norm']:
                for a in ANGLES:
                    job=dict(name=f'joint_{a}',group='joint',stage='train',angle=a,parameters=proposal['parameters'],target_mm=.5,seconds=1800,status='pending')
                    state['jobs'].append(job);run_job(out,m,job,state);analysis(out,m,state)
            else:state['proposal_not_run']='Less than 5% predicted improvement; bounded diagnostic stops'
        except Exception as exc:state['proposal_error']=repr(exc)
    else:state['proposal_not_run']='Incomplete probe set; no missing response fitted'
    sets,names,curves,ex=analysis(out,m,state)
    eligible=[k for k in sets if k!='baseline' and promising(sets[k],sets['baseline'])]
    if eligible:
        winner=min(eligible,key=lambda k:sets[k]['joint_rmse_n'])
        state['decision']=dict(status='promising_early_joint',winner=winner,scope='Provisional early-window screen only; conditional .8 extension')
        for a in ANGLES:
            source=out/names[winner][a];parameters=read(source/'result.json')['job']['parameters']
            job=dict(name=f'check08_{a}',group=winner,stage='depth_check',angle=a,parameters=parameters,
                     target_mm=.8,seconds=1800,status='pending',resume=str(source))
            state['jobs'].append(job);run_job(out,m,job,state)
        # Test .5--.8 without selecting another set from these depths.
        checks=[j for j in state['jobs'] if j.get('stage')=='depth_check']
        if all(j.get('summary',{}).get('mechanics_completed') for j in checks):
            cc=curves_from(out,{j['angle']:j['name'] for j in checks});window=[]
            for a in ANGLES:
                d=np.arange(51,81)/100;err=np.interp(d,*cc[a])-np.interp(d,*ex[a])
                window.append(dict(angle=a,rmse_n=float(np.sqrt(np.mean(err**2))),max_abs_n=float(abs(err).max())))
            write_json(out/'depth_check.json',dict(status='completed',window_mm=[.5,.8],rows=window,
                note='Depth transfer check, not certified intact or independent blind validation; no refitting'))
        else:state['decision']['depth_check']='Incomplete; accepted prefixes preserved'
    else:
        state['decision']=dict(status='no_tested_complete_set_meets_joint_gate',
            conclusion='Bounded common sets do not establish a fit; not proof all parameters or this material family are impossible',
            next='Inspect measured response modes/constraints/discretization; no extra sweeps or budgets')
    state.update(status='finished',finished=stamp());analysis(out,m,state);render(out,state)


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Study already exists')
    tests=read(out/'tests.json')
    if not tests['passed'] or tests['skipped']:raise ValueError('Required regressions incomplete')
    inputs={str(out/'tests.json'):sha(out/'tests.json')};fresh=[]
    with threadpool_limits(1),compute_process_policy():
        for a in ANGLES:
            folder=ROOT/REFERENCES[a];r=read(folder/'result.json');au=read(folder/'audit.json')
            if not r['completed'] or not au['passed']:raise ValueError('Baseline incomplete')
            p=make(a,BASE)
            if json.loads(json.dumps(signature(p)))!=r['signature']:raise ValueError('Baseline signature changed')
            for name in ('result.json','audit.json'):inputs[str(folder/name)]=sha(folder/name)
            for i,row in enumerate(r['steps']):
                if row['depth_mm']>.8+1e-10:break
                path=folder/'states'/f'{i:04d}.npz';h=au['rows'][i]['checkpoint_sha256']
                if not au['rows'][i]['passed'] or sha(path)!=h:raise ValueError('Baseline state changed')
                inputs[str(path)]=h
                if any(abs(row['depth_mm']-v)<1e-10 for v in (.25,.5)):
                    with np.load(path,allow_pickle=False) as z:fresh.append(dict(angle=a,**p.audit_state({k:z[k] for k in z.files},row)))
            dest=out/f'experiment{a}.csv';shutil.copy2(ROOT/'docs'/EXPERIMENTS[a],dest)
            _,_,rate=load_force_csv(dest)
            if rate!=.2:raise ValueError('Speed mismatch')
            inputs[str(dest)]=sha(dest)
    source=snapshot(out)
    jobs=[dict(name=f'{k}_{a}',group=k,stage='train',angle=a,parameters=v.tolist(),target_mm=.5,seconds=1800,status='pending') for k,v in PROBES.items() for a in ANGLES]
    m=dict(created=stamp(),policy='bounded-common-material-FE-v1',source_sha256=source,inputs=inputs,
           baselines={str(a):str(ROOT/f) for a,f in REFERENCES.items()},jobs=jobs,python=sys.executable,
           runtime_path=str(ROOT/'.runtime/ipc-python'),pardiso_runtime=str(ROOT/'.runtime/pardiso'),
           baseline_parameters=BASE.tolist(),probe_parameters={k:v.tolist() for k,v in PROBES.items()},
           original_acceptance=True,compute=dict(material_threads=1,blas_threads=1,linear_threads=8,ipc_threads=4),
           scope='At most9 probe .5 paths,3 joint .5 paths,3 own-history conditional .8 extensions; all preflights .01',
           budgets=dict(preflight_solve=180,train_solve=1800,extension_solve=1800,audit=600,setup_allowance=300),
           joint_gate=dict(joint_rmse_ratio=.8,worst_angle_rmse_ratio=.8,each_rmse_n=1.5,each_max_abs_n=2.,maximum_secant_relative_error=.2))
    write_json(out/'manifest.json',m);write_json(out/'baseline_fresh_audit.json',dict(passed=True,states=fresh))
    # A real fine P1 path with the compute backend qualifies launch. Remaining
    # case-specific preflights are supervised dependencies of their own cases.
    first=jobs[0];pilot=dict(first,name='pilot_'+first['name'],stage='preflight',target_mm=.01,seconds=180,status='pending')
    state=dict(status='preparing',jobs=[pilot])
    if not supervise(out,m,pilot,state):raise RuntimeError('First actual fine preflight failed; no queue launch')
    inputs.update(verify_case(out/pilot['name']));first['resume']=str(out/pilot['name'])
    m['inputs']=inputs;m['jobs']=jobs;write_json(out/'manifest.json',m)
    plan='<h1>共同材料检验：预先限定的方案</h1><p>用户授权执行下一步；共同Maxwell/原生IPC/P1，G/K/τ候选跨角度相同，几何、摩擦、边界、物理验收一致。旧队列不恢复；每组从零或自己的同模型预检/完整前缀恢复。</p>'
    plan+='<p>三个真实FE方向：G0×.7（分配比及τ不变）；τ×.5（两G不变）；长期份额升到.3（G0和τ不变）。仅此有限范围，不是穷举材料可行域。每方向0/33/57各至.5 mm，固定.01 mm、原500次PTC及停滞规则，无自动缩步/追加预算。失败转下一组，接受态全部终态准确双历史审查。</p>'
    plan+='<p>三方向都完整时，以其真实FE力的有限方向响应近似，在采样参数四面体内用三个角度等权拟合幅值与增量力，最多提出一个组合；组合从零真实FE三角度重新计算，不把响应近似叫作新FE。若与已算顶点重合，直接复用该顶点；预测改善不足5%不新增组合。</p>'
    plan+='<p>暂定训练.1–.5 mm。早期工程门槛：联合RMSE和最差角度RMSE各下降至少20%；每角度RMSE≤1.5 N、最大差≤2 N；.1–.25/.25–.5两段割线相对误差≤20%。这些是预先选定工程筛查，不是实验噪声界限。57可改变，采用各角度绝对门槛而非保护原57误差。</p>'
    plan+='<p>只有至少一组真实三角度FE通过门槛，才取联合RMSE最小的一组，接其各自准确双历史至.8 mm；只检查.5–.8、不再选参。0°a2起裂/机械首触未独立确认，33估计1.2–1.3，57未知；所有未裂窗口都只是暂定。没有通过则停止本范围并记录，不宣称整个材料家族不可行。</p>'
    plan+='<p>每个正式case求解1800 s，审查600 s，准备余量300 s；预检求解180 s。串行运行、PARDISO8/IPC4/BLAS1，避免多个大非线性场争抢资源。到时限即推进，预算不是ETA。后台进程独立于终端，启动核查一次后脱离agent。</p><pre>'+escape(json.dumps(dict(tests=tests,fresh_baseline_anchors=fresh,first_preflight=pilot),ensure_ascii=False,indent=2))+'</pre>'
    atomic_write(out/'plan.html',shell('共同材料方案与资格',plan));state=dict(status='prepared',jobs=jobs)
    analysis(out,m,state);render(out,state);print(json.dumps(dict(prepared=True,baseline_anchor_count=len(fresh),first_preflight=pilot['summary'],source_count=len(source),input_count=len(inputs))))


def launch(out):
    m=read(out/'manifest.json');check(m,out)
    if (out/'launch.json').exists() or (out/'worker_started.json').exists():raise ValueError('Already launched')
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        p=subprocess.Popen([m['python'],'-X','utf8','-u','-m',MODULE,'worker','--output',str(out)],
            cwd=out/'source',env=child_env(m),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**background_child_options())
    value=dict(pid=p.pid,time=stamp());write_json(out/'launch.json',value);print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=('prepare','case','worker','launch','report'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--job-json');args=parser.parse_args();out=args.output.resolve()
    if args.phase=='case':case(out,json.loads(args.job_json))
    elif args.phase=='prepare':prepare(out)
    elif args.phase=='worker':worker(out)
    elif args.phase=='launch':launch(out)
    else:
        m=read(out/'manifest.json');s=read(out/'queue_status.json');analysis(out,m,s);render(out,s)
