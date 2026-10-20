"""Bounded AL resource profiling on exact saved histories; no queue mutation."""
import argparse
import contextlib
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import time
import numpy as np
from scipy.sparse import bmat,diags,save_npz
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits,threadpool_info
from run_support import compute_process_policy,write_json
from tools.run_augmented_blade import make_problem,verify_case,snapshot


def prepare(source,target,*,factory=make_problem):
    r=json.loads((source/'result.json').read_text(encoding='utf-8'))
    i=next(i for i,s in enumerate(r['steps']) if abs(s['depth_mm']-target)<1e-12)-1
    if i<1:raise ValueError('Need two prior states')
    p=factory(r['settings']['angle'],coarse=r['settings']['coarse'],spacing=r['settings']['spacing'])
    states=[]
    for index in (i-1,i):
        with np.load(source/'states'/f'{index:04d}.npz') as z:states.append({k:z[k].copy() for k in z.files})
    older,s=states;old=float(s['depth_mm']);x=s['positions'].copy()
    p.material_metric=s['material_metric'].copy();p.material_dissipation=float(s['cumulative_viscous_dissipation_n_mm'])
    p.previous=x.copy();p.previous_material_metric=p.material_metric.copy();p.material_dt=(target-old)/p.loading_rate
    p.seed_predictor_history([(float(older['depth_mm']),older['positions']),(old,x)])
    return p,x,old,r,i


def profile(source,out,target):
    out.mkdir(parents=True,exist_ok=False)
    inputs=verify_case(source,require_independent=True)
    report=dict(status='initializing',input_sha256=inputs,source_sha256=snapshot(out),target=target,
        scope='One actual AL increment and its initial mixed system; no production modification')
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(2),compute_process_policy() as policy:
        p,x,old,r,i=prepare(source,target);p.ipc.set_num_threads(2)
        trial=p.affine_loading_candidate(x,old,target);p.prepare_quadrature(trial)
        _,g,h,j=p.evaluate(trial);q,check=p.constraint_audit(trial);w=np.repeat(p.edge_area,2)
        residual=np.r_[g[p.free],w*q['constraint'].ravel()]
        A=bmat([[h[p.free][:,p.free],q['force_lambda'][p.free]],
            [diags(w)@q['constraint_u'][:,p.free],diags(w)@q['constraint_lambda']]],format='csc')
        A+=diags(np.r_[np.full(len(p.free),1e-9),np.zeros(len(w))])
        save_npz(out/'mixed_matrix.npz',A);np.save(out/'mixed_rhs.npy',-residual)
        tick=time.perf_counter();du=spsolve(A,-residual);linear=time.perf_counter()-tick
        report['initial_system']=dict(shape=list(A.shape),nnz=A.nnz,superlu_s=linear,
            residual_n=float(np.linalg.norm(A@du+residual)),limit_n=max(1e-8*np.linalg.norm(residual),1e-10))
        # A new step resets predictor/quadrature and uses the original history.
        prof=cProfile.Profile();tick=time.perf_counter();prof.enable()
        value,record=p.step(x,old,target,time.monotonic()+180)
        prof.disable();elapsed=time.perf_counter()-tick
        state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=value,
            previous_positions=p.previous,depth_mm=target,body_nodes=p.n,**p.checkpoint_data())
        fresh=p.audit_state(state,record)
        stream=io.StringIO();pstats.Stats(prof,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(60)
        (out/'profile.txt').write_text(stream.getvalue(),encoding='utf-8')
        prof.dump_stats(out/'profile.pstats')
        stats=[]
        for (file,line,name),(primitive,total,self_time,cumulative,callers) in pstats.Stats(prof).stats.items():
            if name in ('evaluate','material_bulk','response','contact_data','_self_terms','safe_step','geometry_audit','compact_matrices','spsolve','_equilibrate','minimum_j','convex_edge_gap_bounds'):
                stats.append(dict(file=file,line=line,name=name,calls=total,self_s=self_time,cumulative_s=cumulative))
        report.update(status='completed',solve_s=elapsed,record=record,audit=fresh,profile=stats,process_policy=policy,threadpools=threadpool_info())
    write_json(out/'profile.json',report)
    print(json.dumps({k:report[k] for k in ('status','solve_s','initial_system','profile')},indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--target',type=float,default=.105)
    args=a.parse_args();profile(args.source.resolve(),args.output.resolve(),args.target)
