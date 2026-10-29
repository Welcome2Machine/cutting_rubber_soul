"""Component, directional-derivative and linear-algebra audit of archived roots."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu, onenormest, LinearOperator

from numerical_runtime import numerical_runtime, add_thread_arguments, thread_arguments
from run_support import write_json
from .frozen_scene import load_scene
from virtual_path_geometry import native_partitioned_chord


def contact_signature(v, x):
    c = v.collisions(x)
    return {kind: sorted(list(q.vertex_ids(v.cmesh.edges, v.cmesh.faces)) for q in getattr(c, kind))
            for kind in ('vv_collisions', 'ev_collisions')}


def inspect(case, output, endpoint, polish_iterations=0):
    output.mkdir(parents=True, exist_ok=False)
    p, base, v, before, x, info = load_scene(case, endpoint)
    v.solver_diagnostics = True
    def covered(value):
        v.virtual_chord_certificate=native_partitioned_chord(v,value,256)
        return v.virtual_chord_certificate['complete']
    v.accepted_chord_is_safe=covered
    v.phase_seconds = {}
    v.set_primary_increment(x, x-v.previous)
    lag = v.lag(x)
    _, g, h, j, metric = v.evaluate(x, lag, projected=False, with_metric=True)
    v.update_constraint_free(x, g)
    free = v.free.copy(); r = g[free]
    a = v.coupled_friction_tangent(x, lag, h)[free][:, free].tocsc()
    m = metric[free][:, free].tocsc()
    rbulk, _, dofs, _, _ = v.material_bulk((x[:v.n]-v.mesh.reference_xy).ravel(), with_tangent=False)
    bulk = np.zeros(v.total); np.add.at(bulk, dofs.ravel(), rbulk.ravel())
    normal = np.zeros(v.total); normal[v.surface_dofs] = v.barrier.gradient(v.collisions(x),v.cmesh,x[v.surface])
    friction = np.zeros(v.total); friction[v.surface_dofs] = v.friction.gradient(lag,v.cmesh,v.step_increment(x)[v.surface])
    cohesive = g-bulk-normal-friction
    k = free[np.argmax(np.abs(r))]
    rows = 1/np.maximum(np.asarray(abs(a).max(axis=1).toarray()).ravel(), 1e-30)
    columns = 1/np.maximum(np.asarray(abs(diags(rows)@a).max(axis=0).toarray()).ravel(), 1e-30)
    started = time.perf_counter(); lu = splu(a); factor_s = time.perf_counter()-started
    inv = LinearOperator(a.shape, matvec=lu.solve, rmatvec=lambda q:lu.solve(q,trans='T'))
    cond = float(onenormest(a)*onenormest(inv))
    scaled = (diags(rows)@a@diags(columns)).tocsc(); slu = splu(scaled)
    directions = {'newton':lu.solve(-r), 'equilibrated_newton':columns*slu.solve(-rows*r)}
    for _ in range(3):
        q = directions['equilibrated_newton']; q += columns*slu.solve(rows*(-r-a@q))
    for shift in (1e-6, 1e-3, .1, 1.):
        directions[f'ptc_{shift:g}'] = splu(a+shift*m).solve(-r)
    report = dict(diagnostic_chord_policy='geometry partition with conservative unresolved-resolution exit', schema='softcut-root-mechanism/1', scene=info, min_j=float(j),
        full_residual_n=float(np.sqrt(2)*np.linalg.norm(r)), half_force_tolerance_n=v.force_tolerance_n,
        maximum_residual=dict(dof=int(k),node=int(k//2),component='xy'[k%2],position_mm=x[k//2].tolist(),
            total_n=float(g[k]),bulk_n=float(bulk[k]),normal_n=float(normal[k]),friction_n=float(friction[k]),
            cohesive_n=float(cohesive[k]),body_force_n=0.,inertia_in_virtual_equation=False),
        active_constraints=dict(free_dofs=free.tolist(),mirror_active_nodes=np.flatnonzero(v.active_contact).tolist(),
            tied_nodes=v.seam[~v.released].tolist(),supports=v.mesh.bottom_nodes.tolist()),
        matrix=dict(shape=a.shape,nnz=a.nnz,one_norm_condition_estimate=cond,
            condition_is_estimate=True,factorization_s=factor_s,
            row_scale_range=[float(rows.min()),float(rows.max())],
            column_scale_range=[float(columns.min()),float(columns.max())]),directions={},derivatives=[])
    write_json(output/'partial.json',report)
    original_contact = contact_signature(v,x)
    report['contact_signature'] = original_contact
    for name, q in directions.items():
        d = np.zeros(v.total); d[free]=q; d=d.reshape(x.shape)
        started=time.perf_counter(); local=v.safe_step(x,d)
        local_s=time.perf_counter()-started
        started=time.perf_counter(); trial,alpha,safe=v._ptc_trial(x,d,float(np.max(np.abs(q))))
        chord_s=time.perf_counter()-started
        v.set_primary_increment(trial,trial-v.previous)
        _,gt,_,jt=v.evaluate(trial,v.lag(trial),tangent=False)
        report['directions'][name]=dict(max_update_mm=float(np.max(np.abs(q))),
            original_linear_residual_n=float(np.linalg.norm(a@q+r)),
            relative_backward_error=float(np.linalg.norm(a@q+r)/(np.linalg.norm(r)+onenormest(a)*np.linalg.norm(q))),
            residual_slope=float(r@(a@q)),local_ccd_j_fraction=float(local),full_chord_alpha=float(alpha),
            local_ccd_s=local_s,full_chord_s=chord_s,trial_full_residual_n=float(np.sqrt(2)*np.linalg.norm(gt[free])),
            trial_min_j=float(jt),contact_changed=contact_signature(v,trial)!=original_contact,
            full_chord_certificate=v.virtual_chord_certificate)
    write_json(output/'partial.json',report)
    rng=np.random.default_rng(118)
    vectors={'newton':directions['newton'],'random':rng.normal(size=len(free))}
    for name,q in vectors.items():
        q=q/np.max(np.abs(q));d=np.zeros(v.total);d[free]=q;d=d.reshape(x.shape)
        exact=a@q
        for step in (1e-5,1e-6,1e-7,1e-8,1e-9,1e-10):
            gp=[];signatures=[]
            for sign in (-1,1):
                trial=x+sign*step*d;v.set_primary_increment(trial,trial-v.previous)
                gp.append(v.evaluate(trial,v.lag(trial),tangent=False)[1][free])
                signatures.append(contact_signature(v,trial))
            fd=(gp[1]-gp[0])/(2*step)
            report['derivatives'].append(dict(direction=name,step_mm=step,
                relative_error=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-30)),
                error_n_per_mm=float(np.linalg.norm(fd-exact)),
                contact_changed=any(s!=original_contact for s in signatures)))
    write_json(output/'partial.json',report)
    # Fixed 20 original Newton steps from the saved endpoint, no continuation/history changes.
    value=x.copy();report['polish']=[]
    for iteration in range(polish_iterations):
        v.set_primary_increment(value,value-v.previous)
        lag=v.lag(value);_,g,h,j=v.evaluate(value,lag,projected=False);v.update_constraint_free(value,g)
        residual=g[v.free];norm=float(np.sqrt(2)*np.linalg.norm(residual))
        rec=dict(iteration=iteration,residual_n=norm,min_j=float(j));report['polish'].append(rec)
        write_json(output/'partial.json',report)
        if norm<=1e-4:break
        jac=v.coupled_friction_tangent(value,lag,h)[v.free][:,v.free].tocsc()
        q=splu(jac).solve(-residual);d=np.zeros(v.total);d[v.free]=q;d=d.reshape(value.shape)
        _,alpha,_=v._ptc_trial(value,d,float(np.max(np.abs(q))))
        if alpha==0:
            rec['failure']='no_certified_chord_step';break
        for cut in range(24):
            trial=value+alpha*d;v.set_primary_increment(trial,trial-v.previous)
            _,gt,_,jt=v.evaluate(trial,v.lag(trial),tangent=False)
            nt=float(np.sqrt(2)*np.linalg.norm(gt[v.free]))
            if jt>0 and nt<norm:break
            alpha*=.5
        else:rec['failure']='no_original_residual_decrease';break
        rec.update(alpha=float(alpha),backtracks=cut);value=trial
    v.set_primary_increment(value,value-v.previous)
    _,g,_,j=v.evaluate(value,v.lag(value),tangent=False);v.update_constraint_free(value,g)
    norm=float(np.sqrt(2)*np.linalg.norm(g[v.free]));ccd=v.accepted_chord_is_safe(value)
    report['polished_endpoint']=dict(residual_n=norm,min_j=float(j),full_chord_safe=ccd,
        accepted=bool(norm<=1e-4 and j>0 and ccd),original_equation=True)
    report['phase_seconds']=v.phase_seconds
    np.savez_compressed(output/'fields.npz',before=before,original_after=x,polished_after=value,
        free=free,residual=r,matrix_data=a.data,matrix_indices=a.indices,matrix_indptr=a.indptr,
        bulk=bulk,normal=normal,friction=friction,cohesive=cohesive,**directions)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--endpoint',choices=['production','diagnostic'],default='production')
    parser.add_argument('--polish-iterations',type=int,default=0)
    add_thread_arguments(parser);args=parser.parse_args()
    if not 0<=args.polish_iterations<=20:raise ValueError('Bounded 0..20 polish iterations required')
    with numerical_runtime(**thread_arguments(args)) as runtime:
        report=inspect(args.case,args.output,args.endpoint,args.polish_iterations);report['thread_runtime']=runtime
    write_json(args.output/'diagnostic.json',report)
    print(json.dumps(report['polished_endpoint']))


if __name__=='__main__':main()
