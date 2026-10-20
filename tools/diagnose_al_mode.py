"""Bounded exact-history displacement-mode and self-contact audit.

P1-restricted increments and component-deleted tangents are algebraic probes,
NOT changed equilibria or admissible material-history transfers.
"""
import argparse
import contextlib
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.sparse import coo_matrix, block_diag, eye, diags
from threadpoolctl import threadpool_limits
from run_support import compute_process_policy, write_json
from tools.run_al_integration import factory
from tools.run_augmented_blade import snapshot
from tools.diagnose_al_failure import restore, system
from tools.audit_mainline import nonlinear_split
from tools.render_solver_dashboard import shell, atomic_write


def p1_prolongation(p):
    """P1 correction subspace in the existing P2 state, including multipliers."""
    base=p.enrichment.base_nodes
    rows=list(range(2*base)); cols=rows.copy(); vals=[1.]*len(rows)
    for i,edge in enumerate(p.enrichment.interior_edges):
        for axis in range(2):
            for node in edge:
                rows.append(2*(base+i)+axis);cols.append(2*int(node)+axis);vals.append(.5)
    T=coo_matrix((vals,(rows,cols)),shape=(2*p.n,2*base)).tocsr()
    return block_diag((T[p.free][:,p.free[p.free<2*base]],eye(2*len(p.multipliers))),format='csc')


def self_audit(p,x,direction):
    c=p.ipc.NormalCollisions();c.use_area_weighting=True
    c.collision_set_type=p.ipc.NormalCollisions.IMPROVED_MAX_APPROX
    sx=x[p.body_surface];c.build(p.self_mesh,sx,p.dhat)
    adjacent={tuple(sorted(e)) for e in np.asarray(p.self_mesh.edges)}
    grouped={'adjacent_vv':[], 'other_vv':[]}
    for v in c.vv_collisions:
        key=tuple(sorted((int(v.vertex0_id),int(v.vertex1_id))))
        grouped['adjacent_vv' if key in adjacent else 'other_vv'].append(v)
    rows=[];ds=direction[p.body_surface].ravel()
    for name in ('adjacent_vv','other_vv','ev'):
        z=p.ipc.NormalCollisions();z.use_area_weighting=True
        z.collision_set_type=p.ipc.NormalCollisions.IMPROVED_MAX_APPROX
        if name=='ev':z.ev_collisions=c.ev_collisions
        else:z.vv_collisions=grouped[name]
        g=p.barrier.gradient(z,p.self_mesh,sx)
        h=p.barrier.hessian(z,p.self_mesh,sx,p.ipc.PSDProjectionMethod.NONE)
        rows.append(dict(component=name,count=len(z),energy_n_mm=float(p.barrier(z,p.self_mesh,sx)),
            gradient_norm_n=float(np.linalg.norm(g)),directional_curvature_n_mm=float(ds@(h@ds))))
    rows_total=sum(row['energy_n_mm'] for row in rows)
    np.testing.assert_allclose(rows_total,p.barrier(c,p.self_mesh,sx),atol=1e-13,rtol=1e-12)
    return dict(groups=rows,body_intersection=bool(p.ipc.has_intersections(p.self_mesh,sx)))


def run(source,failed,out,seconds=240):
    out.mkdir(parents=True,exist_ok=False)
    origin=json.loads((source/'result.json').read_text(encoding='utf-8'));s=origin['settings']
    with np.load(failed,allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    result=dict(status='running',scope='Exact-history fixed-field algebraic diagnostic; no accepted load advance',
        settings=s,input_sha256={str(failed.resolve()):hashlib.sha256(failed.read_bytes()).hexdigest()},source_sha256=snapshot(out))
    for name in ('tools/diagnose_al_mode.py','tools/audit_mainline.py','tools/diagnose_al_failure.py'):
        b=Path(name).read_bytes();(out/'source'/name).write_bytes(b);result['source_sha256'][name]=hashlib.sha256(b).hexdigest()
    p=None;began=time.monotonic();deadline=began+seconds
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),threadpool_limits(1),compute_process_policy():
        try:
            p=factory(s['angle'],coarse=s['coarse'],spacing=s['spacing'],order=s['order'],contact_basis=s['contact_basis'],surface_spacing=s['surface_spacing'])
            p.ipc.set_num_threads(2)
            with np.load(source/'states'/'0000.npz',allow_pickle=False) as z:np.testing.assert_array_equal(p.rest,z['reference_xy'])
            x=restore(p,state);target=float(state['attempted_depth_mm'])
            np.testing.assert_allclose(x[p.n:],p.prescribed_blade_positions(target),rtol=0,atol=p.blade_boundary_roundoff_tolerance(target))
            np.testing.assert_array_equal(x[p.mesh.bottom_nodes],p.rest[p.mesh.bottom_nodes])
            ids=[i for i,row in enumerate(origin['steps']) if abs(row['depth_mm']-(target-p.material_dt*p.loading_rate))<1e-10]
            if len(ids)!=1:raise ValueError('No unique exact predecessor')
            prior=source/'states'/f'{ids[0]:04d}.npz'
            with np.load(prior,allow_pickle=False) as z:
                np.testing.assert_array_equal(p.previous,z['positions']);np.testing.assert_array_equal(p.previous_material_metric,z['material_metric'])
            result['input_sha256'][str(prior.resolve())]=hashlib.sha256(prior.read_bytes()).hexdigest()
            A,r,h,q,checks,j=system(p,x);nf=len(p.free);w=np.repeat(p.edge_area,2)
            result.update(q_mm=target,previous_index=ids[0],dt_s=p.material_dt,min_j=j,residual_n=float(np.linalg.norm(r[:nf])))
            regularizer=diags(np.r_[np.full(nf,1e-9),np.zeros(len(r)-nf)])
            d=p.linear_direction(A+regularizer,-r)
            full=np.zeros(p.total);full[p.free]=d[:nf];full=full.reshape(-1,2)
            result['self_contact']=self_audit(p,x,full)
            sh=p._self_terms(x,True)[2]
            from prototype.curve_contact import blocks
            blade_h=p.D.T@blocks(q['jacobian'])@p.D;bulk_h=h-sh-blade_h
            result['directional_curvatures_n_mm']={k:float(full.ravel()@(v@full.ravel())) for k,v in [('bulk',bulk_h),('body_self',sh),('blade_geometric',blade_h)]}
            base=p.enrichment.base_nodes;edge=p.enrichment.interior_edges
            bubble=full[base:p.n]-full[edge].mean(axis=1)
            order=np.argsort(np.linalg.norm(full[:p.n],axis=1))[::-1][:30]
            result['direction']=dict(max_mm=float(abs(full).max()),nonaffine_mid_norm_mm=float(np.linalg.norm(bubble)),
                all_mid_norm_mm=float(np.linalg.norm(full[base:p.n])),max_nonaffine_mid_mm=float(abs(bubble).max()),
                top_nodes=[dict(id=int(i),reference_xy=p.rest[i].tolist(),current_xy=x[i].tolist(),direction_mm=full[i].tolist(),midpoint=bool(i>=base)) for i in order])
            # Per-element remainder, assembled nodal norm and local norms differ.
            re,k,dofs,_,_=p.material_bulk((x[:p.n]-p.rest[:p.n]).ravel(),with_tangent=True)
            rt=p.material_bulk((x[:p.n]+full[:p.n]-p.rest[:p.n]).ravel(),with_tangent=False)[0]
            remainder=rt-re-np.einsum('eij,ej->ei',k,full.ravel()[dofs])
            local=np.linalg.norm(remainder,axis=1);order_e=np.argsort(local)[::-1][:30]
            result['bulk_local_remainder']=dict(norm_n=float(np.linalg.norm(local)),
                active_fraction_squared=float(np.sum(local[p.active_elements]**2)/np.sum(local**2)),
                top_elements=[dict(id=int(i),reference_center=p.rest[p.mesh.triangles[i]].mean(axis=0).tolist(),
                    remainder_n=float(local[i]),active=bool(i in p.active_elements)) for i in order_e])
            T=p1_prolongation(p);reduced=T.T@(A+regularizer)@T
            dp=T@p.linear_direction(reduced.tocsc(),-T.T@r)
            # Only tangent changes: residual/constitutive histories stay physical.
            from scipy.sparse import bmat
            no_self=A-bmat([[sh[p.free][:,p.free],None],[None,coo_matrix((len(w),len(w)))]],format='csc')
            dn=p.linear_direction(no_self+regularizer,-r)
            result['probes']=[];lam=p.multipliers.copy()
            for name,dv in [('full_newton',d),('p1_correction_subspace',dp),('self_tangent_removed',dn)]:
                if time.monotonic()>deadline:raise TimeoutError('Mode audit budget')
                ff=np.zeros(p.total);ff[p.free]=dv[:nf];ff=ff.reshape(-1,2)
                amax=min(1.,.01/max(abs(dv[:nf]).max(),1e-30),p.safe_step(x,ff))
                row=dict(name=name,original_linear_residual_n=float(np.linalg.norm(A@dv+r)),extent_mm=float(abs(ff).max()),safe_alpha=float(amax),trials=[])
                result['probes'].append(row)
                for fraction in (1.,.25,.0625,.00390625):
                    alpha=amax*fraction
                    try:
                        p.multipliers=lam+alpha*dv[nf:].reshape(-1,2)
                        _,g,_,tj=p.evaluate(x+alpha*ff,tangent=False);qq=p.contact_data(x+alpha*ff)
                        rr=np.r_[g[p.free],w*qq['constraint'].ravel()]
                    finally:p.multipliers=lam.copy()
                    row['trials'].append(dict(alpha=float(alpha),residual_n=float(np.linalg.norm(rr)),min_j=tj,
                        prediction_error_n=float(np.linalg.norm(rr-r-alpha*(A@dv)))))
                print(json.dumps(row),flush=True);write_json(out/'result.json',result)
            result.update(status='completed',elapsed_s=time.monotonic()-began)
        except Exception as exc:
            result.update(status='failed',error=repr(exc),elapsed_s=time.monotonic()-began);traceback.print_exc(file=log)
        finally:
            if p is not None:p.close_compute()
    write_json(out/'result.json',result)
    display={k:v for k,v in result.items() if k not in ('source_sha256','input_sha256')}
    atomic_write(out/'index.html',shell('困难位移模式核查','<h1>困难位移模式核查</h1><p>固定失败场与准确历史；没有新增平衡态。</p><pre>'+escape(json.dumps(display,ensure_ascii=False,indent=2))+'</pre><a href="log.html">运行日志</a>'))
    atomic_write(out/'log.html',shell('诊断日志','<pre>'+escape((out/'run.log').read_text(encoding='utf-8'))+'</pre>'))
    print(json.dumps({k:result.get(k) for k in ('status','elapsed_s','error')}))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source',type=Path,required=True)
    a.add_argument('--failed',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--seconds',type=float,default=240);v=a.parse_args();run(v.source,v.failed,v.output,v.seconds)
