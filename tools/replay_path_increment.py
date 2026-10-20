"""Bounded exact-history replay; never changes or resumes the production queue.

The derivative audit uses a saved failed iterate, labelled as such. Solver
comparisons always start from the last accepted state and its original history.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def derivative_audit(p, x):
    from scipy.sparse.linalg import spsolve
    lag=p.lag(x)
    _,g,H,_=p.evaluate(x,lag,projected=False)
    A=p.coupled_friction_tangent(x,lag,H)[p.free][:,p.free]
    r=g[p.free]
    du=spsolve(A,-r)
    directions={'newton':du}
    rng=np.random.default_rng(1349)
    # A deterministic local perturbation, independent of the Newton direction.
    blade=p.prescribed_blade_positions(float(p._audit_depth))
    pole=blade[np.argmin(blade@p.en)]
    local=np.linalg.norm(x[:p.n]-pole,axis=1)<1.
    v=rng.standard_normal(p.total);v[2*p.n:]=0
    v[:2*p.n]*=np.repeat(local,2)
    directions['local_random']=v[p.free]
    rows=[]
    for name,d in directions.items():
        if not np.any(d):raise ValueError('Empty derivative probe direction: '+name)
        d=d/max(float(np.max(np.abs(d))),1e-30)
        full=np.zeros(p.total);full[p.free]=d;full=full.reshape(x.shape)
        analytic=A@d
        for h in (1e-6,1e-7,1e-8):
            safe=min(p.safe_step(x,h*full),p.safe_step(x,-h*full))
            if safe<1.-1e-10:
                rows.append(dict(direction=name,h_mm=h,status='geometry_rejected',safe_fraction=float(safe)));continue
            gp=p.evaluate(x+h*full,p.lag(x+h*full),tangent=False)[1][p.free]
            gm=p.evaluate(x-h*full,p.lag(x-h*full),tangent=False)[1][p.free]
            fd=(gp-gm)/(2*h)
            rows.append(dict(direction=name,h_mm=h,status='evaluated',
                relative_error=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(fd),1e-30)),
                fd_norm=float(np.linalg.norm(fd)),analytic_norm=float(np.linalg.norm(analytic)),
                one_sided_disagreement=float(np.linalg.norm((gp-g[p.free])/h-(g[p.free]-gm)/h)/max(np.linalg.norm(fd),1e-30))))
    return dict(residual_n=float(np.linalg.norm(r)),linear_relative_residual=float(np.linalg.norm(A@du+r)/np.linalg.norm(r)),rows=rows)


def main():
    a=argparse.ArgumentParser(__doc__)
    a.add_argument('--source',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--depth',type=float,required=True)
    a.add_argument('--seconds',type=float,default=150.)
    a.add_argument('--variant',choices=('tangent','assembly','threads','compact','model-error','residual-merit','newton','contact-refresh','contact-refresh-model'),required=True)
    a.add_argument('--ipc-threads',type=int,default=None,help='Explicit native TBB limit; BLAS is independently limited to 2')
    a.add_argument('--predictor',choices=('secant','affine'),default='secant')
    args=a.parse_args()
    if args.output.exists():raise ValueError('Refuse to overwrite a diagnostic')
    args.output.mkdir(parents=True)
    from threadpoolctl import threadpool_limits
    from run_support import compute_process_policy,save_npz,write_json
    from tools.run_angle_path_study import make_problem,signature
    report=dict(status='starting',variant=args.variant,predictor=args.predictor,
        scope='one increment; same original friction and material histories; no load cutback',
        requested_depth_mm=args.depth,budget_s=args.seconds)
    started=time.monotonic();solve_start=None
    with (args.output/'run.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),threadpool_limits(2),compute_process_policy():
        try:
            original=json.loads((args.source/'result.json').read_text(encoding='utf-8'))
            audit=json.loads((args.source/'audit.json').read_text(encoding='utf-8'))
            records=original['steps']
            target_index=next((i for i,r in enumerate(records) if abs(r['depth_mm']-args.depth)<1e-10),None)
            previous_index=target_index-1 if target_index is not None else len(records)-1
            if previous_index<1:raise ValueError('Need two prior accepted states')
            target_depth=float(records[target_index]['depth_mm']) if target_index is not None else args.depth
            if target_depth<=records[previous_index]['depth_mm']:raise ValueError('Target must follow accepted state')
            inputs={args.source/'result.json':sha(args.source/'result.json'),args.source/'audit.json':sha(args.source/'audit.json')}
            sources={n:h for n,h in original['source_sha256'].items() if n.startswith(('prototype/','mechanics/'))}
            for n,h in sources.items():
                if sha(n)!=h:raise ValueError('Physical source differs: '+n)
            for n in list(sources)+['tools/replay_path_increment.py','tools/run_angle_path_study.py']:
                dest=args.output/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(n,dest)
            report['source_sha256']={n:sha(n) for n in list(sources)+['tools/replay_path_increment.py','tools/run_angle_path_study.py']}
            s=original['settings'];p=make_problem(s['angle'],s['mode'],radius=s['radius'],order=s['order'])
            report['native_ipc_threads_initial']=p.ipc.get_num_threads()
            if args.ipc_threads is not None:
                if args.ipc_threads<1:raise ValueError('Positive native thread count required')
                p.ipc.set_num_threads(args.ipc_threads)
            report['native_ipc_threads']=p.ipc.get_num_threads()
            if json.loads(json.dumps(signature(p)))!=original['signature']:raise ValueError('Physical signature differs')
            report['signature']=signature(p);states=[]
            for i in (previous_index-1,previous_index):
                path=args.source/'states'/f'{i:04d}.npz'
                inputs[path]=sha(path)
                if inputs[path]!=audit['rows'][i]['checkpoint_sha256']:raise ValueError('Checkpoint hash differs')
                z=load(path);p.audit_state(z,records[i]);states.append(z)
            older,previous=states;x=previous['positions'].copy();old=float(previous['depth_mm'])
            p.previous=x.copy();p.material_metric=previous['material_metric'].copy()
            p.material_dissipation=float(previous['cumulative_viscous_dissipation_n_mm'])
            p.seed_predictor_history([(float(older['depth_mm']),older['positions']),(old,x)])
            report.update(previous_depth_mm=old,replay_depth_mm=target_depth,dt_s=(target_depth-old)/p.loading_rate)
            p.predictor=args.predictor;p.stagnation_window=40 if args.predictor=='secant' else 0
            if args.variant=='threads':
                from scipy.sparse.linalg import spsolve
                p.previous=previous['previous_positions'].copy()
                p.previous_material_metric=previous['previous_material_metric'].copy()
                p.material_dt=float(previous['material_dt_s'])
                native=p.ipc.get_num_threads();rows=[];r0=None
                for threads in (native,2,2,native):
                    p.ipc.set_num_threads(threads);tick=time.perf_counter();lag=p.lag(x)
                    _,g,H,_,M=p.evaluate(x,lag,projected=False,with_metric=True)
                    assembly=time.perf_counter()-tick;tick=time.perf_counter()
                    A=p.coupled_friction_tangent(x,lag,H)[p.free][:,p.free]
                    du=spsolve(A+.1*M[p.free][:,p.free],-g[p.free])
                    direction=np.zeros(p.total);direction[p.free]=du;direction=direction.reshape(x.shape)
                    safe=p.safe_step(x,direction);other=time.perf_counter()-tick
                    if r0 is None:r0=g.copy()
                    np.testing.assert_allclose(r0,g,rtol=0,atol=1e-12)
                    rows.append(dict(native_threads=threads,assembly_s=assembly,direction_and_ccd_s=other,
                        total_s=assembly+other,residual_difference_n=float(np.max(np.abs(r0-g))),safe_fraction=float(safe)))
                p.ipc.set_num_threads(native)
                report.update(status='thread_check_complete',rows=rows,scope='same accepted field, original assembly; no new load accepted')
            elif args.variant=='assembly':
                from prototype.compact_p2_assembly import CompactP2AssemblyMixin
                n='prototype/compact_p2_assembly.py';shutil.copy2(n,args.output/'source'/n)
                report['source_sha256'][n]=sha(n)
                p.previous=previous['previous_positions'].copy()
                p.previous_material_metric=previous['previous_material_metric'].copy()
                p.material_dt=float(previous['material_dt_s'])
                original_class=type(p);candidate_class=type('CompactReplay',(CompactP2AssemblyMixin,original_class),{})
                lag=p.lag(x);times={'original':[],'compact':[]};metrics=[]
                for repeat in range(4):
                    results={}
                    for name,cls in [('original',original_class),('compact',candidate_class)][::(1 if repeat%2==0 else -1)]:
                        p.__class__=cls;tick=time.perf_counter()
                        results[name]=p.evaluate(x,lag,projected=False,with_metric=True)
                        times[name].append(time.perf_counter()-tick)
                    b,c=results['original'],results['compact']
                    # Native parallel contact reduction can differ at roundoff,
                    # even for repeated calls of the unchanged implementation.
                    np.testing.assert_allclose(b[1],c[1],rtol=0,atol=1e-12)
                    np.testing.assert_allclose([b[0],b[3]],[c[0],c[3]],rtol=0,atol=1e-12)
                    errors={'residual_max_n':float(np.max(np.abs(b[1]-c[1])))}
                    for idx,label in ((2,'exact'),(4,'metric')):
                        delta=b[idx]-c[idx]
                        errors[label]=float(np.max(np.abs(delta.data),initial=0.))
                        if errors[label]>1e-9:raise ValueError('Assembly changed: '+label)
                    metrics.append(errors)
                report.update(status='assembly_checked',times_s=times,matrix_errors=metrics,
                    median_speedup=float(np.median(times['original'])/np.median(times['compact'])),
                    assembly_entries_original=int(len(p.mesh.triangles)*144),
                    assembly_entries_compact=int(len(p.linear_elements)*36+len(p.active_elements)*144))
            elif args.variant=='tangent':
                if target_index is not None:raise ValueError('Tangent mode expects failed target')
                path=args.source/'failed_trial.npz';inputs[path]=sha(path);z=load(path)
                np.testing.assert_array_equal(z['previous_positions'],x)
                np.testing.assert_array_equal(z['previous_material_metric'],p.material_metric)
                np.testing.assert_allclose(z['positions'][p.n:],p.prescribed_blade_positions(target_depth),rtol=0,atol=1e-12)
                p.previous_material_metric=z['previous_material_metric'].copy();p.material_dt=float(z['material_dt_s'])
                np.testing.assert_allclose(p.material_dt,(target_depth-old)/p.loading_rate,rtol=0,atol=1e-14)
                p._audit_depth=target_depth
                report.update(derivative=derivative_audit(p,z['positions']),status='audit_complete',equilibrium=False)
            else:
                if args.variant=='compact':
                    from prototype.compact_p2_assembly import CompactP2AssemblyMixin
                    p.__class__=type('CompactReplay',(CompactP2AssemblyMixin,type(p)),{})
                    n='prototype/compact_p2_assembly.py';shutil.copy2(n,args.output/'source'/n)
                    report['source_sha256'][n]=sha(n)
                elif args.variant in ('contact-refresh','contact-refresh-model'):
                    from prototype.contact_refresh import ContactRefreshMixin
                    from prototype.compact_p2_assembly import CompactP2AssemblyMixin
                    p.__class__=type('ReplayContactRefresh',(ContactRefreshMixin,CompactP2AssemblyMixin,type(p)),{})
                    p.contact_inner_merit=args.variant=='contact-refresh'
                    for n in ('prototype/contact_refresh.py','prototype/compact_p2_assembly.py'):
                        shutil.copy2(n,args.output/'source'/n);report['source_sha256'][n]=sha(n)
                elif args.variant=='newton':
                    from prototype.residual_newton import ResidualNewtonMixin
                    p.__class__=type('ReplayNewton',(ResidualNewtonMixin,type(p)),{})
                    n='prototype/residual_newton.py';dest=args.output/'source'/n
                    shutil.copy2(n,dest);report['source_sha256'][n]=sha(n)
                else:p.friction_globalization=args.variant
                solve_start=time.monotonic()
                value,record=p.step(x,old,target_depth,solve_start+args.seconds)
                report['solve_elapsed_s']=time.monotonic()-solve_start
                path=args.output/'accepted.npz'
                save_npz(path,reference_xy=p.rest,triangles=p.mesh.triangles,positions=value,
                    previous_positions=p.previous,depth_mm=target_depth,body_nodes=p.n,**p.checkpoint_data())
                fresh=p.audit_state(load(path),record)
                np.testing.assert_array_equal(p.previous,x)
                np.testing.assert_array_equal(p.previous_material_metric,previous['material_metric'])
                report.update(status='accepted',record=record,fresh_audit=fresh,checkpoint_sha256=sha(path))
                if target_index is not None:
                    path=args.source/'states'/f'{target_index:04d}.npz';inputs[path]=sha(path)
                    if inputs[path]!=audit['rows'][target_index]['checkpoint_sha256']:raise ValueError('Target hash differs')
                    target=load(path);baseline=records[target_index]
                    top=np.flatnonzero(np.isclose(p.mesh.local_sn[:,1],p.config.height_mm,atol=1e-10))
                    report['difference_from_original']=dict(fz_n=record['reaction_fz_n']-baseline['reaction_fz_n'],
                        whole_top_max_mm=float(np.linalg.norm(value[top]-target['positions'][top],axis=1).max()),
                        all_nodes_max_mm=float(np.linalg.norm(value-target['positions'],axis=1).max()),
                        material_metric_max=float(np.max(np.abs(p.material_metric-target['material_metric']))),
                        min_j=record['min_j']-baseline['min_j'])
            report['input_sha256']={str(k.resolve()):h for k,h in inputs.items()}
            for path,h in inputs.items():
                if sha(path)!=h:raise ValueError('Input changed during replay')
        except Exception as exc:
            report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
            if solve_start is not None:report['solve_elapsed_s']=time.monotonic()-solve_start
            if solve_start is not None and hasattr(p,'trial'):
                save_npz(args.output/'failed_trial.npz',positions=p.trial,previous_positions=p.previous,**p.checkpoint_data())
            print(traceback.format_exc(),flush=True)
        finally:
            if 'inputs' in locals():
                report['input_sha256']={str(k.resolve()):h for k,h in inputs.items()}
                changed=[str(k) for k,h in inputs.items() if sha(k)!=h]
                report['input_hashes_unchanged']=not changed
                if changed:report.update(status='failed',input_hash_errors=changed)
            report['wall_elapsed_s']=time.monotonic()-started
            write_json(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('signature','source_sha256','input_sha256','traceback','record','fresh_audit')},ensure_ascii=False))


if __name__=='__main__':main()
