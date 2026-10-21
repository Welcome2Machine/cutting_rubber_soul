"""Bounded, fixed-history implicit parameter sensitivities at accepted fields.

These are conditional derivatives, never replacement equilibrium curves or
full-path material sensitivities. The old material AND friction histories stay
fixed. Parameter and displacement perturbations are checked together.
"""
import argparse
from dataclasses import replace
from html import escape
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from prototype.material_case import signature
from prototype.pardiso import PardisoDirection
from tools import run_common_material_study as host
from tools.render_solver_dashboard import atomic_write, shell, stamp
from run_support import write_json

ROOT=Path(__file__).resolve().parents[1]
FACTORS=('ge','gv','tau','bulk','friction')


def parameter_state(material,config,key,eta):
    """Positive log perturbations; shear changes keep the bulk modulus fixed."""
    ge,k=material.moduli();gv=material.moduli(True)[0];tau=material.relaxation_s
    mu=config.friction_coefficient
    if key=='ge':ge*=np.exp(eta)
    elif key=='gv':gv*=np.exp(eta)
    elif key=='tau':tau*=np.exp(eta)
    elif key=='bulk':k*=np.exp(eta)
    elif key=='friction':mu*=np.exp(eta)
    else:raise ValueError(key)
    mat=replace(material,mu_eq=(2*ge/material.alpha_eq[0],),
        mu_ov=(2*gv/material.alpha_ov[0],),relaxation_s=tau,
        poisson=(3*k-2*ge)/(2*(3*k+ge)))
    return mat,replace(config,friction_coefficient=mu)


def render(out,result):
    body='<h1>小角度幅值偏差：共同参数条件探针</h1><p>'+stamp()+'</p>'
    body+='<p>固定原上一接受步的材料与摩擦历史，解完整非对称切线的隐式方向。数值为 dFz/dln(parameter)，单位N；正值表示该参数局部增大时Fz增大。不是重新加载曲线，不能外推大幅调参或证明57°全路径被保持。</p>'
    body+='<table><tr><th>角度</th><th>深度mm</th><th>基准Fz N</th><th>因素</th><th>条件导数N</th><th>方向验证</th></tr>'
    for field in result['fields']:
        for key,p in field['probes'].items():
            body+=f'<tr><td>{field["angle"]}</td><td>{field["depth_mm"]}</td><td>{field["fz_n"]:.5f}</td><td>{key}</td><td>{p["derivative_n"]:.6g}</td><td>{p["qualified"]}</td></tr>'
    body+='</table><p>每个扰动均重建原生接触及正常承载，不用固定法向力替代真实摩擦切线；保留原边界与所有自接触。此页不报告新的加载完成。</p>'
    body+='<pre>'+escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>'
    atomic_write(out/'index.html',shell('共同参数条件探针',body))
    write_json(out/'probe.json',result)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    source=host.snapshot(out)
    import shutil
    target=out/'source/tools/probe_angle_factors.py';shutil.copy2(__file__,target)
    source['tools/probe_angle_factors.py']=host.sha(target)
    result=dict(created=stamp(),status='running',fields=[],source_sha256=source,inputs={},
        scope='Conditional same-history derivatives; NOT full-path sensitivities',budget_s=900)
    start=time.monotonic();render(out,result)
    try:
        with threadpool_limits(1):
            for angle in (0,33,57):
                folder=ROOT/host.REFERENCES[angle];result['inputs'].update(host.verify_case(folder))
                report=host.read(folder/'result.json');p=host.make(angle,host.BASE)
                if json.loads(json.dumps(signature(p)))!=report['signature']:raise ValueError('Baseline signature mismatch')
                for depth in (.25,.5,1.):
                    if time.monotonic()-start>900:raise TimeoutError('Fixed diagnostic budget')
                    print(f'FIELD angle={angle} depth={depth}',flush=True)
                    idx=next(i for i,s in enumerate(report['steps']) if abs(s['depth_mm']-depth)<1e-10)
                    with np.load(folder/'states'/f'{idx:04d}.npz',allow_pickle=False) as z:state=dict(z)
                    audited=p.audit_state(state,report['steps'][idx]);x=state['positions']
                    p.previous=state['previous_positions'];p.previous_material_metric=state['previous_material_metric']
                    p.material_dt=float(state['material_dt_s']);mat=p.material;config=p.config
                    lag=p.lag(x);_,g,H,_=p.evaluate(x,lag,projected=False)
                    A=p.coupled_friction_tangent(x,lag,H)
                    def fz(v):return float(-v.reshape(-1,2)[p.n:,1].sum())
                    def residual(y,key,eta):
                        try:
                            p.material,p.config=parameter_state(mat,config,key,eta)
                            return p.evaluate(y,p.lag(y),tangent=False)[1]
                        finally:p.material,p.config=mat,config
                    eps=1e-4
                    partial=np.column_stack([(residual(x,key,eps)-residual(x,key,-eps))/(2*eps) for key in FACTORS])
                    solver=PardisoDirection(8)
                    try:du=solver.solve(A[p.free][:,p.free],-partial[p.free]);linear=list(solver.rows)
                    finally:solver.close()
                    direction=np.zeros((p.total,len(FACTORS)));direction[p.free]=du
                    action=A@direction+partial
                    field=dict(angle=angle,depth_mm=depth,fz_n=fz(g),fresh_audit=audited,linear=linear,probes={})
                    for j,key in enumerate(FACTORS):
                        relative=float(np.linalg.norm(action[p.free,j])/max(np.linalg.norm(partial[p.free,j]),1e-30))
                        if relative>1e-8:raise RuntimeError('Linear residual failed')
                        derivative=fz(action[:,j]);trials=[]
                        for eta in (1e-3,3e-4,1e-4):
                            change=eta*direction[:,j].reshape(-1,2);safe=float(p.safe_step(x,change))
                            if safe<1-1e-10:
                                trials.append(dict(eta=eta,safe=safe,passed=False));continue
                            gy=residual(x+change,key,eta);actual=(fz(gy)-fz(g))/eta
                            ferr=abs(actual-derivative)/max(abs(derivative),1e-4)
                            rerr=float(np.linalg.norm((gy-g-eta*action[:,j])[p.free])/max(eta*np.linalg.norm(partial[p.free,j]),1e-12))
                            trials.append(dict(eta=eta,safe=safe,actual_n=actual,force_relative_error=ferr,
                                residual_relative_error=rerr,passed=bool(ferr<.02 and rerr<.02)))
                        field['probes'][key]=dict(derivative_n=derivative,linear_relative=relative,trials=trials,
                            qualified=bool(any(a['passed'] and b['passed'] for a,b in zip(trials,trials[1:]))))
                        print(f'PROBE {angle}/{depth}/{key} derivative={derivative:.7g} qualified={field["probes"][key]["qualified"]}',flush=True)
                    result['fields'].append(field);render(out,result)
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['elapsed_s']=time.monotonic()-start;render(out,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output.resolve())
