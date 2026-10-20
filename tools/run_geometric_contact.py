"""Explicit GCP-body/AL-blade qualification and fixed-increment long paths.

No old path is resumed for production. A conditional failed-step replay is a
diagnostic only. Long runs resume only their own fresh qualified GCP prefix.
"""
import argparse
import ast
import contextlib
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.sparse import bmat,diags
from threadpoolctl import threadpool_limits
from prototype.geometric_self_contact import enable_geometric_self_contact,POLICY
from prototype.al_performance import ParallelMaterial,PardisoDirection
from tools.run_al_integration import factory
from tools import run_augmented_blade as base
from tools.render_solver_dashboard import shell,atomic_write,stamp
from run_support import write_json,save_npz,compute_process_policy,background_child_options,classify_process_exit

ROOT=Path(__file__).resolve().parents[1]


def settings(out,angle,depth,**kw):
    c=float(np.cos(np.deg2rad(angle)))
    values=dict(output=Path(out),angle=float(angle),depth=depth*c,increment=.005*c,
        lab_speed=.2,seconds=21600.,audit_seconds=1800.,order=5,spacing=.000625,
        surface_spacing=.0025,contact_basis='P1',blas_threads=1,ipc_threads=4,
        material_workers=4,linear_threads=8,coarse=False,resume=None,
        self_contact_policy=POLICY,fixed_load_steps=True,vertical_target_mm=depth)
    values.update(kw);return argparse.Namespace(**values)


def make(args,fast=True):
    p=factory(args.angle,coarse=args.coarse,spacing=args.spacing,order=args.order,
        contact_basis=args.contact_basis,surface_spacing=args.surface_spacing,fast=fast,
        material_workers=args.material_workers,linear_threads=args.linear_threads)
    enable_geometric_self_contact(p)
    p.loading_rate=args.lab_speed*np.cos(np.deg2rad(args.angle))
    if fast:p.direction_policy='adaptive';p.nested_contact_model=True;p.nonmonotone_window=3
    return p


def run(args):
    made=[]
    def candidate(*a,**kw):
        p=make(args,True);made.append(p);return p
    try:return base.run(args,problem_factory=candidate,audit_factory=lambda *a,**kw:make(args,False))
    finally:
        for p in made:p.close_compute()


def load_diagnostic(origin,args):
    r=json.loads((origin/'result.json').read_text(encoding='utf-8'))
    file=origin/'states'/f'{len(r["steps"])-1:04d}.npz'
    with np.load(file) as z:s={k:z[k].copy() for k in z.files}
    p=make(args);p.ipc.set_num_threads(args.ipc_threads)
    np.testing.assert_array_equal(p.rest,s['reference_xy'])
    p.previous=s['previous_positions'].copy();p.previous_material_metric=s['previous_material_metric'].copy()
    p.material_metric=s['material_metric'].copy();p.material_dt=float(s['material_dt_s'])
    p.material_dissipation=float(s['cumulative_viscous_dissipation_n_mm'])
    p._set_quadrature(s['al_edges'],s['al_fractions'],s['al_weights']);p.multipliers=s['al_multipliers'].copy()
    inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (origin/'result.json',file)}
    return p,s,r,inputs


def page(out,title,data):
    text=json.dumps(data,ensure_ascii=False,indent=2)
    atomic_write(out/'index.html',shell(title,f'<h1>{escape(title)}</h1><p>{stamp()}</p><pre>{escape(text)}</pre><a href="log.html">HTML日志</a>'))
    if (out/'run.log').exists():atomic_write(out/'log.html',shell(title+'日志','<pre>'+escape((out/'run.log').read_text(encoding='utf-8',errors='replace'))+'</pre>'))


def qualify(out):
    out.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',passed=False,source_sha256=base.snapshot(out),inputs_sha256={},cases=[],resources={})
    def save():
        write_json(out/'qualification.json',result)
        page(out,'GCP修复：原失败增量与资源实测',{k:v for k,v in result.items() if k not in ('source_sha256','inputs_sha256')})
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy():
        for angle,target,budget in ((57,.290,120.),(33,.110,90.)):
            origin=ROOT/f'outputs/mode_diagnosis_20261019/continuation/lab{angle}_d030'
            args=settings(out,angle,target)
            p,s,old,inputs=load_diagnostic(origin,args);result['inputs_sha256'].update(inputs)
            prev=float(s['depth_mm']);row=dict(angle=angle,target_d_mm=target,previous_q_mm=prev,budget_s=budget)
            result['cases'].append(row);tick=time.monotonic()
            try:
                # This old state is a conditional diagnostic, NEVER a GCP prefix.
                e,g,H,w=p._self_terms(s['positions'],True)
                row['predecessor_gcp']=dict(p.geometric_self.last)
                if e!=0 or np.linalg.norm(g)!=0 or H.nnz:raise ValueError('Predecessor is not unchanged by GCP')
                if angle==57:
                    p.previous=s['positions'].copy();p.previous_material_metric=s['material_metric'].copy()
                    p.material_dt=(args.depth-prev)/p.loading_rate
                    trial=p.affine_loading_candidate(s['positions'],prev,args.depth);p.prepare_quadrature(trial)
                    rows=[];reference=None
                    for workers in (1,2,4,8):
                        p.parallel_material.close();p.parallel_material=ParallelMaterial(workers)
                        samples=[]
                        for _ in range(2):
                            start=time.perf_counter();ev=p.evaluate(trial);samples.append(time.perf_counter()-start)
                        if reference is None:reference=ev[1].copy()
                        np.testing.assert_allclose(ev[1],reference,rtol=1e-12,atol=1e-10)
                        rows.append(dict(workers=workers,seconds=samples,median_s=float(np.median(samples))))
                    chosen=min(rows,key=lambda x:x['median_s'])['workers']
                    p.parallel_material.close();p.parallel_material=ParallelMaterial(chosen)
                    _,g,h,j=p.evaluate(trial);q,_=p.constraint_audit(trial);weights=np.repeat(p.edge_area,2)
                    residual=np.r_[g[p.free],weights*q['constraint'].ravel()]
                    matrix=bmat([[h[p.free][:,p.free],q['force_lambda'][p.free]],
                        [diags(weights)@q['constraint_u'][:,p.free],diags(weights)@q['constraint_lambda']]],format='csc')
                    matrix+=diags(np.r_[np.full(len(p.free),1e-9),np.zeros(len(weights))])
                    linear=[]
                    for threads in (4,8,12,16):
                        solve=PardisoDirection(threads);timings=[]
                        try:
                            for _ in range(2):
                                start=time.perf_counter();u=solve.solve(matrix,-residual);timings.append(time.perf_counter()-start)
                                if np.linalg.norm(matrix@u+residual)>max(1e-8*np.linalg.norm(residual),1e-10):raise ValueError('Physical linear residual')
                            linear.append(dict(threads=threads,seconds=timings,median_s=float(np.median(timings)),checks=solve.rows))
                        finally:solve.close()
                    best=min(linear,key=lambda x:x['median_s'])['threads']
                    result['resources']=dict(material=rows,linear=linear,material_workers=chosen,linear_threads=best,
                        ipc_threads=4,blas_threads=1,scope='One actual failed-increment matrix/assembly; not full-path speedup')
                    p.parallel_linear.close();p.parallel_linear=PardisoDirection(best)
                else:
                    resources=result['resources'];p.parallel_material.close();p.parallel_material=ParallelMaterial(resources['material_workers'])
                    p.parallel_linear.close();p.parallel_linear=PardisoDirection(resources['linear_threads'])
                # Profiling must not supply a different multiplier warm start.
                p._set_quadrature(s['al_edges'],s['al_fractions'],s['al_weights']);p.multipliers=s['al_multipliers'].copy()
                tick=time.monotonic();x,record=p.step(s['positions'].copy(),prev,args.depth,time.monotonic()+budget)
                row['solve_s']=time.monotonic()-tick
                state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,previous_positions=p.previous,
                    depth_mm=args.depth,body_nodes=p.n,**p.checkpoint_data())
                np.testing.assert_array_equal(state['previous_positions'],s['positions'])
                np.testing.assert_array_equal(state['previous_material_metric'],s['material_metric'])
                save_npz(out/f'replay{angle}.npz',**state)
                auditor=make(args,False);auditor.ipc.set_num_threads(4)
                fresh=auditor.audit_state(state,record)
                row.update(passed=True,record=record,audit=fresh)
            except Exception as exc:
                row.update(passed=False,error=repr(exc),solve_s=time.monotonic()-tick)
                if hasattr(p,'rejected_al_state'):save_npz(out/f'failed{angle}.npz',**p.rejected_al_state)
                traceback.print_exc(file=log)
            finally:p.close_compute()
            save()
    for file,digest in result['inputs_sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('Changed diagnostic input')
    result.update(status='finished',passed=all(r.get('passed') for r in result['cases']));save()
    print(json.dumps(dict(passed=result['passed'],cases=[{k:v for k,v in r.items() if k in ('angle','solve_s','passed','error')} for r in result['cases']])))
    return result


def queue_page(out,state):
    body='<h1>GCP完整自接触／无断裂长线</h1><p>'+stamp()+' · '+escape(state['status'])+'</p>'
    body+='<p>只有通过原失败增量资格的角度，才从自己的新GCP预检历史推进至竖直下压2 mm；未通过项明确取消。竖直速度0.2 mm/s，固定步长0.005 mm；失败不缩步、不增加预算。共同材料、刀具、刀具摩擦和验收阈值保持。GCP自摩擦是本项目正则库仑模型；自摩擦系数沿用0.35，尚无独立测量。数值通过不等于实验验证。</p>'
    body+='<table><tr><th>角度</th><th>状态</th><th>竖直深度/mm</th><th>Fz/N</th><th>minJ</th><th>记录</th></tr>'
    for job in state['jobs']:
        folder=out/job['name'];r={};z={}
        if (folder/'result.json').exists():
            r=json.loads((folder/'result.json').read_text(encoding='utf-8'));z=(r.get('steps') or [{}])[-1]
            base.report_html(folder,r)
        d=z.get('depth_mm',0)/np.cos(np.deg2rad(job['angle']))
        links=(f'<a href="{job["name"]}/index.html">状态</a> · <a href="{job["name"]}/log.html">日志</a>' if r else escape(job.get('reason','尚未启动')))
        body+=f'<tr><td>{job["angle"]}°</td><td>{escape(r.get("status",job["status"]))}</td><td>{d:.6g}</td><td>{z.get("reaction_fz_n","—")}</td><td>{z.get("min_j","—")}</td><td>{links}</td></tr>'
    body+='</table><p><a href="../index.html">本轮资格、失败证据及资源基准</a> · <a href="manifest.json">冻结来源与参数</a> · <a href="queue_status.json">进程／终态核查</a></p>'
    atomic_write(out/'index.html',shell('GCP长线状态',body,refresh=state['status']=='running'))


def worker(out):
    m=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    with (out/'worker_started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),time=stamp()),f)
    for file,digest in m['inputs_sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen input differs: '+file)
    state=dict(status='running',jobs=m['jobs']);write_json(out/'queue_status.json',state)
    env=os.environ.copy();env.update(PYTHONPATH=m['runtime_path'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    # Serial cases avoid doubling the mesh/history memory footprint on 32 GB.
    # Each case uses measured threaded assembly and PARDISO configurations.
    for job in state['jobs']:
        if job['status']=='not_qualified':continue
        cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_geometric_contact','run','--output',str(out/job['name']),
             '--angle',str(job['angle']),'--depth','2','--resume',job['resume'],
             '--material-workers',str(m['resources']['material_workers']),'--linear-threads',str(m['resources']['linear_threads'])]
        with (out/(job['name']+'.stdout.log')).open('wb') as stdout,(out/(job['name']+'.stderr.log')).open('wb') as stderr:
            p=subprocess.Popen(cmd,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**background_child_options())
            job.update(status='running',pid=p.pid,command=cmd,started=stamp());start=time.monotonic();forced=False
            while p.poll() is None:
                write_json(out/'queue_status.json',state);queue_page(out,state)
                if time.monotonic()-start>21600+1800+600:
                    p.terminate();p.wait(timeout=30);forced=True;break
                time.sleep(15)
        job.update(status=classify_process_exit(p.returncode,supervisor_timeout=forced),returncode=p.returncode,wall_s=time.monotonic()-start)
        try:
            r=json.loads((out/job['name']/'result.json').read_text(encoding='utf-8'))
            if r['source_sha256']!=m['source_sha256']:raise ValueError('Physical source changed')
            base.verify_case(out/job['name'],require_independent=True)
            if not forced and p.returncode==0:job['status']=r['status']
            job.update(audit_verified=True,completed=bool(r['completed'] and not forced and p.returncode==0))
        except Exception as exc:job.update(audit_verified=False,verification_error=repr(exc))
        write_json(out/'queue_status.json',state);queue_page(out,state)
    state.update(status='finished',finished=stamp());write_json(out/'queue_status.json',state);queue_page(out,state)


def launch(out,qualification):
    q=json.loads((qualification/'qualification.json').read_text(encoding='utf-8'))
    qualified=[c['angle'] for c in q['cases'] if c.get('passed')]
    if q['status']!='finished' or not qualified:raise ValueError('No angle passed failed-increment qualification')
    if out.exists():
        # Resume preparation only; never revive a launched or stopped worker.
        if (out/'launch.json').exists() or (out/'worker_started.json').exists():raise ValueError('Queue was already launched')
        if not all((out.parent/f'preflight{angle}/result.json').exists() for angle in qualified):
            raise ValueError('Incomplete preparation requires a new output directory')
        sources={n:hashlib.sha256((out/'source'/n).read_bytes()).hexdigest() for n in q['source_sha256']}
    else:
        out.mkdir(parents=True,exist_ok=False);sources=base.snapshot(out)
    changed={n for n in set(sources)|set(q['source_sha256']) if sources.get(n)!=q['source_sha256'].get(n)}
    if changed-{'tools/run_geometric_contact.py'}:raise ValueError('Physical source changed after qualification')
    if changed:
        # An orchestration-only change may gate each angle independently.
        # Verify all constructors, FE execution and qualification functions by
        # AST, in addition to byte-identical mechanics/material/contact files.
        def core(file):
            tree=ast.parse(file.read_text(encoding='utf-8'))
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in ('launch','worker','queue_page')}
        if core(out/'source/tools/run_geometric_contact.py')!=core(qualification/'source/tools/run_geometric_contact.py'):
            raise ValueError('Qualification/run constructors changed')
    resources=q['resources'];inputs=dict(q['inputs_sha256'])
    for name,digest in q['source_sha256'].items():inputs[str(qualification/'source'/name)]=digest
    for name,digest in sources.items():inputs[str(out/'source'/name)]=digest
    inputs[str(qualification/'qualification.json')]=hashlib.sha256((qualification/'qualification.json').read_bytes()).hexdigest()
    for angle in qualified:
        prefix=out.parent/f'preflight{angle}'
        args=settings(prefix,angle,.01,seconds=240.,audit_seconds=180.,
            material_workers=resources['material_workers'],linear_threads=resources['linear_threads'])
        r=(json.loads((prefix/'result.json').read_text(encoding='utf-8')) if (prefix/'result.json').exists() else run(args))
        if not r['completed'] or r['source_sha256']!=sources:raise ValueError('Fresh GCP preflight failed')
        inputs.update(base.verify_case(prefix,require_independent=True))
        inputs[str(prefix/'result.json')]=hashlib.sha256((prefix/'result.json').read_bytes()).hexdigest()
        origin=ROOT/f'outputs/mode_diagnosis_20261019/continuation/lab{angle}_d030'
        old=json.loads((origin/'result.json').read_text(encoding='utf-8'));checks=[]
        normalized=json.loads(json.dumps(r['signature']))
        for key in ('config','material','loading_rate','motion','mesh_sha256'):
            if normalized[key]!=old['signature'][key]:raise ValueError('Preflight physical comparison mismatch: '+key)
        for i,z in enumerate(r['steps']):
            j=next(j for j,y in enumerate(old['steps']) if abs(y['depth_mm']-z['depth_mm'])<1e-12)
            file=origin/'states'/f'{j:04d}.npz';inputs[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
            with np.load(file) as a,np.load(prefix/'states'/f'{i:04d}.npz') as b:
                np.testing.assert_array_equal(a['reference_xy'],b['reference_xy'])
                delta=float(np.linalg.norm(a['positions']-b['positions'],axis=1).max())
            force=abs(z['reaction_fz_n']-old['steps'][j]['reaction_fz_n'])/max(.1,abs(old['steps'][j]['reaction_fz_n']))
            if z['geometric_self_contact']['active_pairs'] or force>.02 or delta>.002:
                raise ValueError('GCP shallow zero-contact force/displacement gate failed')
            checks.append(dict(q_mm=z['depth_mm'],force_fraction=force,max_nodal_displacement_difference_mm=delta))
        check=prefix/'shallow_comparison.json';write_json(check,dict(passed=True,rows=checks,
            scope='Zero-GCP shallow equations only; prior time/integration checks do not qualify later active self-contact'))
        inputs[str(check)]=hashlib.sha256(check.read_bytes()).hexdigest()
    jobs=[(dict(name=f'GCP{a}_d2',angle=a,status='pending',resume=str(out.parent/f'preflight{a}')) if a in qualified
           else dict(name=f'GCP{a}_d2',angle=a,status='not_qualified',reason='原失败增量未通过，未启动长线')) for a in (57,33)]
    m=dict(source=str(out/'source'),source_sha256=sources,inputs_sha256=inputs,resources=resources,jobs=jobs,
        runtime_path=str(ROOT/'.runtime/ipc-python')+os.pathsep+str(out/'source'),
        stop_policy='fixed .005 mm vertical increments; no cutback or budget extensions',
        solve_budget_s=21600,audit_budget_s=1800,setup_allowance_s=600,
        qualification_path=str(qualification),qualified_angles=qualified,
        orchestration_only_changes=sorted(changed),
        algorithm='frozen-bulk inner PDAS; 3-iterate nonmonotone merit capped at 2x residual; original acceptance')
    binary=Path(ROOT/'.runtime/ipc-python/ipctk.cp310-win_amd64.pyd')
    m['inputs_sha256'][str(binary)]=hashlib.sha256(binary.read_bytes()).hexdigest()
    for file,digest in inputs.items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('Input changed before launch')
    write_json(out/'manifest.json',m);state=dict(status='prepared',jobs=jobs);write_json(out/'queue_status.json',state);queue_page(out,state)
    env=os.environ.copy();env.update(PYTHONPATH=m['runtime_path'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    cmd=[sys.executable,'-X','utf8','-u','-m','tools.run_geometric_contact','worker','--output',str(out)]
    with (out/'worker.stdout.log').open('wb') as stdout,(out/'worker.stderr.log').open('wb') as stderr:
        p=subprocess.Popen(cmd,cwd=m['source'],env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,**background_child_options())
    record=dict(pid=p.pid,command=cmd,started=stamp());write_json(out/'launch.json',record);print(json.dumps(record))


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('phase',choices=('qualify','launch','worker','run'))
    a.add_argument('--output',type=Path,required=True);a.add_argument('--qualification',type=Path)
    a.add_argument('--angle',type=float,choices=(33.,57.),default=57.);a.add_argument('--depth',type=float,default=2.)
    a.add_argument('--resume',type=Path);a.add_argument('--material-workers',type=int,default=4);a.add_argument('--linear-threads',type=int,default=8)
    v=a.parse_args();out=v.output.resolve()
    if v.phase=='qualify':qualify(out)
    elif v.phase=='launch':launch(out,v.qualification.resolve())
    elif v.phase=='worker':worker(out)
    else:
        if not 0<v.depth<=2:a.error('Current qualified scope is vertical depth <= 2 mm')
        r=run(settings(out,v.angle,v.depth,resume=v.resume,material_workers=v.material_workers,linear_threads=v.linear_threads))
        if not r['completed']:sys.exit(2)


if __name__=='__main__':main()
