"""Shared G0/q/tau full-history derivatives; no new FE loading.

The old tau-only diagnostic remains frozen with its original result files.
"""
import argparse
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
from mechanics.finite_strain import response
from prototype.history_sensitivity import (PARAMETERS, common_material, metric_perturb,
    metric_direction, previous_friction_action, assemble_piola)
from prototype.material_case import signature
from prototype.pardiso import PardisoDirection
from tools import run_common_material_study as host
from tools.render_solver_dashboard import atomic_write, shell, stamp
from run_support import write_json

ROOT=Path(os.environ.get('SOFTMODELING_WORKSPACE',Path(__file__).resolve().parents[1]))
host.ROOT=ROOT
SCALES=(1e-4,3e-5)
CHECK_SCALES=(1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6)
BUDGET=1800


def fz(p,g): return -g.reshape(-1,2)[p.n:,1].sum()


def render(out,result):
    write_json(out/'result.json',result)
    body='<h1>共同G₀ / q / τ：解析摩擦历史偏导</h1><p>'+stamp()+' · '+result['status']+'</p>'
    body+='<p>沿保存平衡路径传播参数方向；没有新FE加载、修改材料默认或放宽物理验收。每角度1800 s，两套材料差分链及相邻两档实际扰动资格。</p>'
    body+='<table><tr><th>深度 mm</th><th>Fz N</th><th>dFz/dlogG₀</th><th>dFz/dlogit(q)</th><th>dFz/dlogτ</th><th>方向资格</th></tr>'
    for row in result.get('steps',[]):
        values=[row['depth_mm'],row['fz_n'],*[v['derivative_n'] for v in row['parameters'].values()],row['passed']]
        body+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>'
    body+='</table><p>未通过行是尝试记录，不是有效导数，不得用于拟合或补原加载前缀。</p>'
    if result.get('error'):body+='<pre>'+escape(result['error'])+'</pre>'
    log=out/'run.log'
    if log.exists():body+='<details open><summary>运行记录</summary><pre>'+escape(log.read_text(encoding='utf-8')[-18000:])+'</pre></details>'
    body+='<p><a href="result.json">全部数值与尺度检查</a></p>'
    atomic_write(out/'index.html',shell('共同材料全历史诊断',body,refresh=result['status']=='running'))


def material_call(p,F,mat,M,dt):
    return response(F,M,dt,mat,tangent=False,kinematics='plane_strain')


def linearize(p,x,M,dx,dM,keys,solver):
    """Columns ordered (parameter, differentiation scale); no old-state FD."""
    mat=p.material; previous=p.previous; previous_M=p.previous_material_metric;dt=p.material_dt
    lag=p.lag(x);_,g,H,_=p.evaluate(x,lag,projected=False)
    A=p.coupled_friction_tangent(x,lag,H)
    hf=p.friction.hessian(lag,p.cmesh,(x-previous)[p.surface],p.ipc.PSDProjectionMethod.NONE)
    friction=previous_friction_action(p,hf,dx)
    F=np.eye(2)+np.einsum('eai,eaJ->eiJ',(x[:p.n]-p.mesh.reference_xy)[p.mesh.triangles],p.grad)
    base=material_call(p,F,mat,previous_M,dt)
    np.testing.assert_allclose(base['metric'],M,rtol=1e-9,atol=1e-11)
    partial=[]
    for j,key in enumerate(keys):
        e=SCALES[j%2]
        plus=material_call(p,F,common_material(mat,key,e),metric_perturb(previous_M,dM[j],e),dt)
        minus=material_call(p,F,common_material(mat,key,-e),metric_perturb(previous_M,dM[j],-e),dt)
        partial.append(assemble_piola(p,(plus['P']-minus['P'])/(2*e))+friction[:,j])
    b=np.column_stack(partial)
    D=np.zeros_like(b);D[p.free]=solver.solve(A[p.free][:,p.free],-b[p.free])
    action=A@D+b
    linear=np.linalg.norm(action[p.free],axis=0)/np.maximum(np.linalg.norm(b[p.free],axis=0),1e-30)
    if np.max(linear)>1e-8:raise RuntimeError('Physical linear residual failed')
    forces=np.array([fz(p,action[:,j]) for j in range(len(keys))])
    new_M=[];projection=[]
    for j,key in enumerate(keys):
        e=SCALES[j%2];dd=D[:,j].reshape(-1,2)
        dF=np.einsum('eai,eaJ->eiJ',dd[:p.n][p.mesh.triangles],p.grad)
        plus=material_call(p,F+e*dF,common_material(mat,key,e),metric_perturb(previous_M,dM[j],e),dt)['metric']
        minus=material_call(p,F-e*dF,common_material(mat,key,-e),metric_perturb(previous_M,dM[j],-e),dt)['metric']
        raw=(plus-minus)/(2*e);dm=metric_direction(M,plus,minus,e)
        projection.append(float(np.linalg.norm(raw-dm)/max(np.linalg.norm(raw),1e-12)));new_M.append(dm)
    return dict(g=g,A=A,b=b,D=D,action=action,forces=forces,new_M=np.stack(new_M),projection=projection,linear=linear)


def qualify(p,x,dx,dM,keys,data,deadline):
    mat=p.material;previous=p.previous;previous_M=p.previous_material_metric
    base_chord=float(p.safe_step(previous,x-previous));records={}
    def trial(j,eps):
        D=data['D'][:,j].reshape(-1,2);old_D=dx[:,j].reshape(-1,2)
        safe=min(float(p.safe_step(x,eps*D)),float(p.safe_step(x,-eps*D)),
                 float(p.safe_step(previous,eps*old_D)),float(p.safe_step(previous,-eps*old_D)))
        row=dict(eps=eps,chain=j%2,endpoint_safe=safe,passed=False,maximum_displacement_mm=float(np.max(abs(eps*D))))
        if safe<1-1e-10:return row
        responses=[]
        try:
            for sign in (1,-1):
                p.material=common_material(mat,keys[j],sign*eps)
                p.previous=previous+sign*eps*old_D
                p.previous_material_metric=metric_perturb(previous_M,dM[j],sign*eps)
                y=x+sign*eps*D;responses.append(p.evaluate(y,p.lag(y),tangent=False)[1])
        finally:p.material=mat;p.previous=previous;p.previous_material_metric=previous_M
        plus,minus=responses;pred=data['action'][:,j]
        absolute=max(np.linalg.norm((plus-data['g']-eps*pred)[p.free]),np.linalg.norm((minus-data['g']+eps*pred)[p.free]))
        relative=absolute/max(eps*np.linalg.norm(data['b'][p.free,j]),1e-10)
        actual=fz(p,(plus-minus)/(2*eps));ferr=abs(actual-data['forces'][j])/max(abs(data['forces'][j]),1e-5)
        whole=min(float(p.safe_step(previous+eps*old_D,x+eps*D-previous-eps*old_D)),
                  float(p.safe_step(previous-eps*old_D,x-eps*D-previous+eps*old_D)))
        row.update(residual_remainder_n=float(absolute),residual_relative=float(relative),force_relative=float(ferr),
            actual_derivative_n=float(actual),whole_chord_safe=whole,
            passed=bool(relative<.02 and ferr<.02 and (base_chord<1-1e-10 or whole>=1-1e-10)))
        return row
    for pair in range(len(keys)//2):
        j=2*pair;key=keys[j]
        chain=dict(force=float(abs(data['forces'][j]-data['forces'][j+1])/max(abs(data['forces'][j+1]),1e-5)),
            displacement=float(np.linalg.norm(data['D'][:,j]-data['D'][:,j+1])/max(np.linalg.norm(data['D'][:,j+1]),1e-10)),
            metric=float(np.linalg.norm(data['new_M'][j]-data['new_M'][j+1])/max(np.linalg.norm(data['new_M'][j+1]),1e-10)))
        checks=[];passed=False
        if max(chain.values())<.02 and max(data['projection'][j:j+2])<.01:
            for m,eps in enumerate(CHECK_SCALES):
                if time.monotonic()>deadline:raise TimeoutError('Preset diagnostic budget')
                try:check=trial(j+m%2,eps)
                except (FloatingPointError,RuntimeError,ValueError) as exc:check=dict(eps=eps,chain=m%2,passed=False,error=repr(exc))
                checks.append(check)
                if len(checks)>1 and checks[-2]['passed'] and checks[-1]['passed']:passed=True;break
        records[key]=dict(derivative_n=float(data['forces'][j+1]),derivatives_n=data['forces'][j:j+2].tolist(),
            chain_relative=chain,material_projection_relative=data['projection'][j:j+2],checks=checks,passed=passed)
    return records,base_chord


def preflight(out):
    root=ROOT/'outputs/model_identifiability_audit_20261021/history_tau'
    result=dict(status='running',fields=[],new_loading_states=0,inputs={});write_json(out/'preflight.json',result)
    tick=time.monotonic()
    with threadpool_limits(1):
        for angle,index in ((33,17),(57,0)):
            folder=ROOT/host.REFERENCES[angle];record=host.read(folder/'result.json')['steps'][index]
            paths=[folder/'states'/f'{index:04d}.npz',root/f'failed_direction_{angle}_{index:04d}.npz']
            for path in paths:result['inputs'][str(path)]=host.sha(path)
            with np.load(paths[0],allow_pickle=False) as z:s=dict(z)
            with np.load(paths[1],allow_pickle=False) as z:old=dict(z)
            p=host.make(angle,host.BASE);p.ipc.set_num_threads(2);audit=p.audit_state(s,record)
            p.previous=s['previous_positions'];p.previous_material_metric=s['previous_material_metric'];p.material_dt=float(s['material_dt_s'])
            dx=old['previous_dx'].reshape(2,-1).T;dm=old['previous_dM'];solver=PardisoDirection(4)
            try:
                data=linearize(p,s['positions'],s['material_metric'],dx,dm,['log_tau']*2,solver)
                checks,chord=qualify(p,s['positions'],dx,dm,['log_tau']*2,data,tick+240)
            finally:solver.close()
            result['fields'].append(dict(angle=angle,index=index,audit=audit,parameters=checks,base_chord_safe=chord,
                scope='Conditional fixture with frozen old previous derivatives; not new full-history prefix'))
    result.update(status='completed',elapsed_s=time.monotonic()-tick,passed=all(v['passed'] for f in result['fields'] for v in f['parameters'].values()))
    write_json(out/'preflight.json',result)
    if not result['passed']:raise RuntimeError('Preflight qualification failed; full diagnostic not launched')


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);sources=host.snapshot(out)
    for name in ('tools/probe_history_sensitivity.py','tests/test_history_sensitivity.py'):
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest);sources[name]=host.sha(dest)
    shutil.copy2(ROOT/'docs/joint_history_sensitivity.md',out/'plan.md')
    write_json(out/'manifest.json',dict(created=stamp(),source_sha256=sources,parameters=list(PARAMETERS),
        differentiation_scales=SCALES,qualification_scales=CHECK_SCALES,budget_per_angle_s=BUDGET,physical_defaults_unchanged=True))
    preflight(out);print('PREPARED '+str(out),flush=True)


def worker(out,angle):
    case=out/f'angle{angle}';case.mkdir(exist_ok=False)
    for name,h in host.read(out/'manifest.json')['source_sha256'].items():
        if host.sha(out/'source'/name)!=h:raise ValueError('Frozen source changed: '+name)
    result=dict(angle=angle,status='running',started=stamp(),steps=[],new_loading_states=0)
    log=(case/'run.log').open('w',encoding='utf-8');tick=time.monotonic();solver=None
    def note(message):print(stamp()+' '+message,file=log,flush=True);render(case,result)
    try:
        with threadpool_limits(1):
            folder=ROOT/host.REFERENCES[angle];result['inputs']=host.verify_case(folder);old=host.read(folder/'result.json')
            p=host.make(angle,host.BASE);p.ipc.set_num_threads(2)
            if json.loads(json.dumps(signature(p)))!=old['signature']:raise ValueError('Baseline signature mismatch')
            keys=[key for key in PARAMETERS for _ in SCALES]
            px=p.rest.copy();pm=np.tile(np.eye(3),(len(p.mesh.triangles),1,1));depth_old=p.start
            dx=np.zeros((p.total,len(keys)));dm=np.zeros((len(keys),*pm.shape));solver=PardisoDirection(4)
            for index,record in enumerate(old['steps']):
                if record['depth_mm']>.5+1e-10:break
                if time.monotonic()>tick+BUDGET:raise TimeoutError('Preset per-angle budget')
                step_start=time.monotonic()
                with np.load(folder/'states'/f'{index:04d}.npz',allow_pickle=False) as z:s=dict(z)
                np.testing.assert_array_equal(s['previous_positions'],px);np.testing.assert_array_equal(s['previous_material_metric'],pm)
                np.testing.assert_allclose(float(s['material_dt_s']),(record['depth_mm']-depth_old)/p.loading_rate,rtol=1e-12)
                audit=p.audit_state(s,record);x=s['positions'];M=s['material_metric']
                p.previous=px;p.previous_material_metric=pm;p.material_dt=float(s['material_dt_s'])
                data=linearize(p,x,M,dx,dm,keys,solver);checks,chord=qualify(p,x,dx,dm,keys,data,tick+BUDGET)
                row=dict(index=index,depth_mm=record['depth_mm'],fz_n=record['reaction_fz_n'],fresh_audit=audit,
                         parameters=checks,base_chord_safe=chord,linear_relative=data['linear'].tolist(),
                         passed=all(v['passed'] for v in checks.values()),elapsed_s=time.monotonic()-step_start)
                result['steps'].append(row);note(f'DEPTH {row["depth_mm"]:.3f} passed={row["passed"]} seconds={row["elapsed_s"]:.2f} derivatives='+str([v['derivative_n'] for v in checks.values()]))
                if not row['passed']:
                    np.savez_compressed(case/'rejected_direction.npz',direction=data['D'],previous_dx=dx,previous_dM=dm,index=index)
                    raise RuntimeError('Derivative qualification failed; do not propagate rejected direction')
                if any(abs(record['depth_mm']-z)<1e-10 for z in (.1,.25,.5)):
                    np.savez_compressed(case/f'anchor_{index:04d}.npz',direction=data['D'],previous_dx=dx,previous_dM=dm,index=index)
                dx,dm=data['D'],data['new_M'];px,pm=x,M;depth_old=record['depth_mm']
            result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());note('STOP '+repr(exc))
    finally:
        if solver is not None:result['linear_calls']=solver.rows;solver.close()
        result.update(ended=stamp(),elapsed_s=time.monotonic()-tick);log.close();render(case,result)


def run_workers(out):
    if not host.read(out/'preflight.json')['passed']:raise RuntimeError('Preflight required')
    env=os.environ.copy();env.update(SOFTMODELING_WORKSPACE=str(ROOT),PYTHONPATH=str(out/'source')+os.pathsep+str(ROOT/'.runtime/ipc-python'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SOFTMODELING_PARDISO_RUNTIME=str(ROOT/'.runtime/pardiso'))
    children=[];status=dict(started=stamp(),status='running',workers=[])
    for angle in (0,33,57):
        stdout=(out/f'worker_{angle}.log').open('w',encoding='utf-8');stderr=(out/f'worker_{angle}.stderr.log').open('w',encoding='utf-8')
        command=[sys.executable,'-B','-X','utf8','-m','tools.probe_history_sensitivity','worker','--output',str(out),'--angle',str(angle)]
        proc=subprocess.Popen(command,cwd=out/'source',env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        children.append((proc,stdout,stderr,time.monotonic()));status['workers'].append(dict(angle=angle,pid=proc.pid,command=command))
    write_json(out/'queue.json',status)
    try:
        while any(p.poll() is None for p,_,_,_ in children):
            for p,_,_,start in children:
                if p.poll() is None and time.monotonic()-start>BUDGET+60:p.terminate()
            time.sleep(.5)
    finally:
        for (p,stdout,stderr,_),row in zip(children,status['workers']):
            if p.poll() is None:p.terminate()
            row['exit_code']=p.wait();stdout.close();stderr.close()
        status.update(status='ended',ended=stamp());write_json(out/'queue.json',status)
    print('ALL_DIAGNOSTICS_ENDED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--angle',type=int,choices=[0,33,57])
    args=parser.parse_args();out=args.output.resolve()
    if args.mode=='prepare':prepare(out)
    elif args.mode=='run':run_workers(out)
    else:worker(out,args.angle)
