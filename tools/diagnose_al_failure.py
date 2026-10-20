"""Read-only exact-history AL failure diagnostics; never commit trial histories."""
import argparse
import contextlib
import hashlib
import json
from html import escape
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.sparse import bmat, diags, save_npz
from scipy.sparse.linalg import splu, spsolve
from threadpoolctl import threadpool_limits

from run_support import write_json, compute_process_policy
from tools import run_augmented_blade as base
from tools.run_al_performance import fast_factory
from tools.render_solver_dashboard import shell, atomic_write


def restore(p, state):
    p.previous=state['previous_positions'].copy()
    p.previous_material_metric=state['previous_material_metric'].copy()
    p.material_metric=state['material_metric'].copy()
    p.material_dt=float(state['material_dt_s'])
    p.material_dissipation=float(state['cumulative_viscous_dissipation_n_mm'])
    p._set_quadrature(state['al_edges'],state['al_fractions'],state['al_weights'])
    p.multipliers=state['al_multipliers'].copy()
    return state['positions'].copy()


def system(p,x):
    _,g,h,j=p.evaluate(x)
    q,checks=p.constraint_audit(x);weights=np.repeat(p.edge_area,2)
    r=np.r_[g[p.free],weights*q['constraint'].ravel()]
    A=bmat([[h[p.free][:,p.free],q['force_lambda'][p.free]],
        [diags(weights)@q['constraint_u'][:,p.free],diags(weights)@q['constraint_lambda']]],format='csc')
    return A,r,h,q,checks,j


def linear_checks(p,A,r,h,out):
    rows=[];nf=len(p.free);scale=np.r_[np.maximum(abs(h.diagonal()[p.free]),1.),np.zeros(len(r)-nf)]
    for shift in (0.,1e-5,1e-3,.1,1.):
        M=A+diags(shift*scale+np.r_[np.full(nf,1e-9),np.zeros(len(r)-nf)])
        if shift==0:save_npz(out/'mixed_matrix.npz',M);np.save(out/'mixed_rhs.npy',-r)
        for mode in ('superlu','pardiso','equilibrated_superlu'):
            tick=time.perf_counter()
            if mode=='equilibrated_superlu':
                rr=1/np.maximum(np.asarray(abs(M).max(axis=1).toarray()).ravel(),1e-30)
                N=diags(rr)@M
                cc=1/np.maximum(np.asarray(abs(N).max(axis=0).toarray()).ravel(),1e-30)
                lu=splu((N@diags(cc)).tocsc());du=cc*lu.solve(-rr*r)
                corrections=[]
                for _ in range(3):
                    e=M@du+r;corrections.append(float(np.linalg.norm(e)))
                    if np.linalg.norm(e)<=max(1e-8*np.linalg.norm(r),1e-10):break
                    du-=cc*lu.solve(rr*e)
            elif mode=='pardiso':
                # Raw library result, without production fallback, for diagnosis.
                old=p.parallel_linear.set_threads(8)
                try:du=p.parallel_linear.solver.solve(M.tocsr(),-r)
                finally:p.parallel_linear.set_threads(old)
            else:du=spsolve(M,-r)
            error=M@du+r;limit=max(1e-8*np.linalg.norm(r),1e-10)
            backward=np.linalg.norm(error,np.inf)/max(np.linalg.norm(r,np.inf)+float(abs(M).sum(axis=1).max())*np.linalg.norm(du,np.inf),1e-300)
            row=dict(shift=shift,backend=mode,seconds=time.perf_counter()-tick,
                residual_n=float(np.linalg.norm(error)),limit_n=limit,passed=bool(np.isfinite(du).all() and np.linalg.norm(error)<=limit),
                normwise_backward_error=float(backward),max_displacement_direction_mm=float(abs(du[:nf]).max()),
                max_multiplier_direction_mpa=float(abs(du[nf:]).max()),slope=float(r@(A@du)))
            if mode=='equilibrated_superlu':row['refinement_residuals_n']=corrections
            rows.append(row)
            print('LINEAR',json.dumps(row),flush=True)
    return rows


def constraint_rank(p,q):
    # Rows without a multiplier derivative are pure kinematic constraints.
    D=q['constraint_lambda'].tocsr();pure=np.flatnonzero(np.asarray(abs(D).sum(axis=1)).ravel()<1e-14)
    C=q['constraint_u'][pure][:,p.free].tocsr()
    norms=np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
    normalized=diags(1/np.maximum(norms,1e-30))@C
    G=(normalized@normalized.T).toarray()
    eig,vec=np.linalg.eigh(G)
    worst=vec[:,0] if len(eig) else np.empty(0)
    order=np.argsort(abs(worst))[-12:][::-1]
    return dict(pure_constraints=len(pure),zero_rows=int(np.count_nonzero(norms<1e-14)),
        normalized_gram_eigenvalues=eig.tolist(),
        worst_mode=[dict(multiplier_row=int(pure[i]),body_edge=int(p.contact_edges[pure[i]//2]),
            component='normal' if pure[i]%2==0 else 'tangent',amplitude=float(worst[i])) for i in order])


def tangent_check(p,x,A,r):
    rng=np.random.default_rng(719);nf=len(p.free)
    d=rng.normal(size=len(r));d[:nf]*=1e-3;d[nf:]*=.1
    direction=np.zeros(p.total);direction[p.free]=d[:nf];direction=direction.reshape(-1,2)
    lam=p.multipliers.copy();w=np.repeat(p.edge_area,2);pred=A@d;rows=[]
    for eps in (1e-3,1e-4,1e-5):
        residuals=[]
        for sign in (-1,1):
            p.multipliers=lam+sign*eps*d[nf:].reshape(-1,2)
            trial=x+sign*eps*direction
            g=p.evaluate(trial,tangent=False)[1];c=p.contact_data(trial)
            residuals.append(np.r_[g[p.free],w*c['constraint'].ravel()])
        fd=(residuals[1]-residuals[0])/(2*eps)
        rows.append(dict(epsilon=eps,relative_error=float(np.linalg.norm(fd-pred)/np.linalg.norm(pred)),
            force_error_n=float(np.linalg.norm(fd[:nf]-pred[:nf])),constraint_error_n=float(np.linalg.norm(fd[nf:]-pred[nf:]))))
    p.multipliers=lam
    return rows


def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',source_sha256=base.snapshot(out),input_sha256=base.verify_case(source,require_independent=True),cases=[])
    script=Path(__file__).resolve();report['diagnostic_sha256']=hashlib.sha256(script.read_bytes()).hexdigest()
    (out/'source/tools'/script.name).write_bytes(script.read_bytes())
    r=json.loads((source/'result.json').read_text(encoding='utf-8'))
    p=None
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy():
        try:
            p=fast_factory(r['settings']['angle'],coarse=r['settings']['coarse'],spacing=r['settings']['spacing']);p.ipc.set_num_threads(2)
            for label,file in [('accepted',source/'states'/f'{len(r["steps"])-1:04d}.npz'),('failed',source/'failed_trial.npz')]:
                dest=out/label;dest.mkdir();report['input_sha256'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
                with np.load(file,allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
                x=restore(p,state);tick=time.perf_counter();A,rhs,h,q,checks,j=system(p,x)
                row=dict(name=label,assembly_s=time.perf_counter()-tick,matrix_shape=list(A.shape),nnz=A.nnz,
                    min_j=j,force_residual_n=float(np.linalg.norm(rhs[:len(p.free)])),checks=checks,
                    dt_s=p.material_dt,traction_range_mpa=[float(p.multipliers.min()),float(p.multipliers.max())],
                    rank=constraint_rank(p,q),tangent=tangent_check(p,x,A,rhs))
                row['linear']=linear_checks(p,A,rhs,h,dest)
                report['cases'].append(row);write_json(out/'result.json',report)
            report['status']='completed'
        except Exception as exc:report.update(status='failed',error=repr(exc));traceback.print_exc(file=log)
        finally:
            if p is not None:p.close_compute()
    write_json(out/'result.json',report)
    display={k:v for k,v in report.items() if k not in ('source_sha256','input_sha256')}
    atomic_write(out/'index.html',shell('C57失败态核查','<h1>C57 原接受态与失败试算的混合系统</h1><p>固定准确前步历史，仅重新装配、切线与线性代数检查；失败试算不是平衡态。</p><pre>'+escape(json.dumps(display,ensure_ascii=False,indent=2))+'</pre><a href="log.html">日志</a>'))
    atomic_write(out/'log.html',shell('失败态核查日志','<pre>'+escape((out/'run.log').read_text(encoding='utf-8'))+'</pre>'))
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']),error=report.get('error'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source.resolve(),args.output.resolve())
