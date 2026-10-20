"""Read-only component/Jacobian audit of an exact-history failed path iterate.

Frozen-contact evaluations are diagnostics, never accepted physical states.
No material history or load path is advanced by this tool.
"""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from tools.replay_path_increment import sha, load


def components(p, x, *, fixed_lag=None, fixed_normal=None):
    c=p.collisions(x) if fixed_normal is None else fixed_normal
    lag=p.lag(x) if fixed_lag is None else fixed_lag
    bulk=np.zeros(p.total)
    r,_,dofs,_,_=p.material_bulk((x[:p.n]-p.mesh.reference_xy).ravel(),with_tangent=False)
    np.add.at(bulk,dofs.ravel(),r.ravel())
    normal=np.zeros(p.total);friction=np.zeros(p.total)
    normal[p.surface_dofs]=p.barrier.gradient(c,p.cmesh,x[p.surface])
    friction[p.surface_dofs]=p.friction.gradient(lag,p.cmesh,(x-p.previous)[p.surface])
    return {'bulk':bulk,'normal':normal,'friction':friction,'total':bulk+normal+friction}


def feature_keys(c):
    keys=[]
    for s in c.vv_collisions:
        keys.append(('vv',int(s.vertex0_id),int(s.vertex1_id)))
    for s in c.ev_collisions:
        keys.append(('ev',int(s.edge_id),int(s.vertex_id)))
    if len(c.ee_collisions) or len(c.fv_collisions):raise ValueError('Expected planar contacts')
    return set(keys)


def embed(p, matrix):
    m=matrix.tocoo()
    return coo_matrix((m.data,(p.surface_dofs[m.row],p.surface_dofs[m.col])),shape=(p.total,p.total)).tocsc()


def component_audit(p,x,out):
    lag=p.lag(x);normal=p.collisions(x);keys=feature_keys(normal)
    _,g,H,j,M=p.evaluate(x,lag,projected=False,with_metric=True)
    A=p.coupled_friction_tangent(x,lag,H)
    parts=components(p,x)
    np.testing.assert_allclose(parts['total'],g,atol=1e-12,rtol=1e-12)
    method=p.ipc.PSDProjectionMethod.NONE
    hn=embed(p,p.barrier.hessian(normal,p.cmesh,x[p.surface],method))
    hf=embed(p,p.friction.hessian(lag,p.cmesh,(x-p.previous)[p.surface],method))
    hb=H-hn-hf;coupling=A-H
    matrices={'bulk':hb,'normal':hn,'friction':hf+coupling,'total':A}
    r=g[p.free];norm=float(np.linalg.norm(r));rows=[]
    free_nodes=np.unique(p.free//2)
    ng=np.linalg.norm(g.reshape(-1,2)[free_nodes],axis=1)
    order=free_nodes[np.argsort(ng)[::-1]]
    energy=np.sum(g[p.free]**2)
    top=[]
    for n in order[:20]:
        top.append(dict(node=int(n),reference_xy=p.rest[n].tolist(),current_xy=x[n].tolist(),
            **{k:v.reshape(-1,2)[n].tolist() for k,v in parts.items()}))
    surface_nodes=np.intersect1d(free_nodes,p.surface[p.surface<p.n])
    surface_dofs=np.intersect1d(p.free,(2*surface_nodes[:,None]+np.arange(2)).ravel())
    summary=dict(residual_n=norm,min_j=float(j),normal_features=len(keys),
        residual_squared_fraction_top10=float(np.sum(g.reshape(-1,2)[order[:10]]**2)/energy),
        residual_squared_fraction_surface=float(np.sum(g[surface_dofs]**2)/energy),
        component_norms_n={k:float(np.linalg.norm(v[p.free])) for k,v in parts.items()},top_nodes=top)
    np.savez_compressed(out/'components.npz',positions=x,free=p.free,surface=p.surface,**parts)
    aa=A[p.free][:,p.free];mm=M[p.free][:,p.free]+diags(np.full(len(p.free),1e-8))
    directions={}
    for name,shift in [('newton',0.),('ptc',.1)]:
        d=spsolve(aa+shift*mm,-r)
        directions[name]=d
    # Contact-local residual direction is independent of the Newton solve.
    d=np.zeros(p.total);d[(2*order[:10,None]+np.arange(2)).ravel()]=g[(2*order[:10,None]+np.arange(2)).ravel()]
    directions['hotspot']=d[p.free]
    for name,d in directions.items():
        direction=np.zeros(p.total);direction[p.free]=d/max(np.max(np.abs(d)),1e-30)
        direction=direction.reshape(x.shape)
        actions={k:v@direction.ravel() for k,v in matrices.items()}
        for h in (1e-6,1e-7,1e-8):
            if min(p.safe_step(x,h*direction),p.safe_step(x,-h*direction))<1.-1e-10:
                rows.append(dict(direction=name,h_mm=h,status='geometry_rejected'));continue
            plus=components(p,x+h*direction);minus=components(p,x-h*direction)
            frozen_plus=components(p,x+h*direction,fixed_lag=lag,fixed_normal=normal)
            frozen_minus=components(p,x-h*direction,fixed_lag=lag,fixed_normal=normal)
            kp=feature_keys(p.collisions(x+h*direction));km=feature_keys(p.collisions(x-h*direction))
            detail={}
            for k in parts:
                fd=(plus[k]-minus[k])/(2*h);action=actions[k]
                fd0=(frozen_plus[k]-frozen_minus[k])/(2*h)
                frozen_action=({'bulk':hb,'normal':hn,'friction':hf,'total':H}[k])@direction.ravel()
                detail[k]=dict(fd_norm=float(np.linalg.norm(fd[p.free])),
                    error_norm=float(np.linalg.norm((fd-action)[p.free])),
                    relative_error=float(np.linalg.norm((fd-action)[p.free])/max(np.linalg.norm(fd[p.free]),1e-30)),
                    one_sided_disagreement=float(np.linalg.norm((plus[k]+minus[k]-2*parts[k])[p.free])/h),
                    frozen_relative_error=float(np.linalg.norm((fd0-frozen_action)[p.free])/max(np.linalg.norm(fd0[p.free]),1e-30)))
            rows.append(dict(direction=name,h_mm=h,status='evaluated',
                feature_changes_plus=len(keys^kp),feature_changes_minus=len(keys^km),components=detail))
            print('COMPONENT_FD '+json.dumps(rows[-1]),flush=True)
    summary['derivatives']=rows
    # Examine the actual finite Newton/PTC trial, not only infinitesimal probes.
    trials=[]
    for name in ('newton','ptc'):
        d=directions[name];full=np.zeros(p.total);full[p.free]=d;full=full.reshape(x.shape)
        _,alpha,safe=p._ptc_trial(x,full,float(np.max(np.abs(d))))
        for fraction in (1.,.25,.0625):
            a=alpha*fraction;trial=x+a*full
            fresh=components(p,trial);frozen=components(p,trial,fixed_lag=lag,fixed_normal=normal)
            k=feature_keys(p.collisions(trial))
            linear={n:mat@(a*full.ravel()) for n,mat in matrices.items()}
            errors={n:float(np.linalg.norm((fresh[n]-parts[n]-linear[n])[p.free])) for n in parts}
            trials.append(dict(direction=name,alpha=float(a),geometric_fraction=float(safe),
                fresh_norm_n=float(np.linalg.norm(fresh['total'][p.free])),
                frozen_norm_n=float(np.linalg.norm(frozen['total'][p.free])),
                feature_changes=len(keys^k),model_errors_n=errors,
                friction_refresh_n=float(np.linalg.norm((fresh['friction']-frozen['friction'])[p.free]))))
    summary['finite_trials']=trials
    return summary


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--trial',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise ValueError('Refuse overwrite')
    a.output.mkdir(parents=True)
    from threadpoolctl import threadpool_limits
    from run_support import compute_process_policy,write_json
    from tools.run_angle_path_study import make_problem,signature
    started=time.monotonic();report={'status':'starting','scope':'failed trial; no equilibrium claim; no advancement'}
    inputs={};sources={}
    with (a.output/'run.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),threadpool_limits(2),compute_process_policy():
        try:
            r=json.loads((a.source/'result.json').read_text(encoding='utf-8'))
            audit=json.loads((a.source/'audit.json').read_text(encoding='utf-8'))
            for path in (a.source/'result.json',a.source/'audit.json',a.trial):inputs[str(path.resolve())]=sha(path)
            for n,h in r['source_sha256'].items():
                if n.startswith(('prototype/','mechanics/')):
                    if sha(n)!=h:raise ValueError('Original physical source mismatch: '+n)
                    sources[n]=h
            for n in ['tools/diagnose_path_coupling.py','tools/run_angle_path_study.py','tools/replay_path_increment.py','prototype/compact_p2_assembly.py']:
                sources[n]=sha(n)
            for n in sources:
                dest=a.output/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(n,dest)
            s=r['settings'];p=make_problem(s['angle'],s['mode'],radius=s['radius'],order=s['order'],compact_assembly=True)
            if json.loads(json.dumps(signature(p)))!=r['signature']:raise ValueError('Signature differs')
            z=load(a.trial)
            prev=z['previous_positions']
            i=next(i for i in range(len(r['steps'])-1,-1,-1) if np.array_equal(load(a.source/'states'/f'{i:04d}.npz')['positions'],prev))
            path=a.source/'states'/f'{i:04d}.npz';inputs[str(path.resolve())]=sha(path)
            if sha(path)!=audit['rows'][i]['checkpoint_sha256']:raise ValueError('Accepted checkpoint mismatch')
            accepted=load(path);fresh=p.audit_state(accepted,r['steps'][i])
            np.testing.assert_array_equal(z['previous_material_metric'],accepted['material_metric'])
            dt=float(z['material_dt_s']);depth=r['steps'][i]['depth_mm']+dt*p.loading_rate
            np.testing.assert_allclose(z['positions'][p.n:],p.prescribed_blade_positions(depth),rtol=0,atol=1e-12)
            p.previous=prev.copy();p.previous_material_metric=z['previous_material_metric'].copy();p.material_dt=dt
            report.update(previous_fresh_audit=fresh,previous_depth_mm=r['steps'][i]['depth_mm'],depth_mm=depth,dt_s=dt,
                diagnostic=component_audit(p,z['positions'],a.output),status='complete')
        except Exception as exc:
            import traceback
            report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        finally:
            report.update(input_sha256=inputs,source_sha256=sources,elapsed_s=time.monotonic()-started)
            report['input_hashes_unchanged']=all(sha(n)==h for n,h in inputs.items())
            report['source_hashes_unchanged']=all(sha(n)==h for n,h in sources.items())
            write_json(a.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','elapsed_s','input_hashes_unchanged','source_hashes_unchanged')}))


if __name__=='__main__':main()
