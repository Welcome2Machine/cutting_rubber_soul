"""Saved-state spatial audit for MINI contact diagnostics.

No new loading is performed. Fresh acceptance uses each exact saved friction
history. J is actual local volume ratio, not projected pressure. Keep local
statistics beside full-domain statistics; small pathological regions matter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_reference import IPCProblem
from prototype.quadratic import quadratic_minimum
from run_support import compute_process_policy, source_hashes, write_json


def force_shape(report, horizon=1.):
    """Fixed physical window, never normalize a short prefix as a full curve."""
    rows=report.get('steps',[])
    if not rows:
        return dict(status='no_accepted_states',horizon_mm=horizon)
    d=np.array([r['depth_mm'] for r in rows]);f=np.array([r['reaction_fz_n'] for r in rows])
    if not np.isfinite([d,f]).all() or np.any(np.diff(d)<=0) or abs(d[0])>1e-12:
        raise ValueError('Shape comparison needs finite ordered zero-origin accepted depths')
    if d[-1]<horizon-1e-10:
        return dict(status='insufficient_depth',horizon_mm=horizon,last_depth_mm=float(d[-1]))
    half,end=np.interp([horizon/2,horizon],d,f)-f[0]
    if half<=0 or end<=0:
        return dict(status='nonpositive_force_increments',horizon_mm=horizon)
    return dict(status='available',horizon_mm=horizon,zero_force_n=float(f[0]),
        half_depth_force_increment_n=float(half),end_force_increment_n=float(end),
        half_depth_ratio=float(half/end),first_half_stiffness_n_mm=float(2*half/horizon),
        second_half_stiffness_n_mm=float(2*(end-half)/horizon),
        stiffness_growth_ratio=float((end-half)/half),
        interpolation='linear within accepted force states only; no contact interpolation')


def problem_arguments(report):
    c,m,s=report['config'],report['mesh_settings'],report['simplifications']
    extra=m.get('interior_refinement',{})
    return dict(mu=c['friction_coefficient'],dhat=report['dhat_mm'],
        local_spacing=m['local_spacing_mm'],surface_spacing=m['surface_spacing_mm'],
        fine_halfwidth=m['fine_halfwidth_mm'],nose_facets=s['nose_facets'],
        inclination=c['inclination_deg'],tip_radius=s['tip_radius_mm'],
        shear_modulus=c['shear_modulus_n_mm2'],ogden_alpha=c['ogden_alpha'],
        predictor=report['predictor'],hessian_mode=report['hessian_mode'],
        element=report.get('discretization',{}).get('element','p1'),
        quadrature_order=report.get('discretization',{}).get('quadrature_order',4),
        interior_refinement_levels=extra.get('levels',0),
        interior_refinement_radius=extra.get('radius_mm',.15),
        interior_spacing=m.get('interior_coarsening',{}).get('spacing_mm'))


def spatial_audit(p,x):
    el=p.enrichment; nv=p.nodal_volume
    u=(x[:p.n]-p.mesh.reference_xy).ravel()
    nv.bulk(u,with_tangent=False)
    J=nv.last['J'].reshape(len(el.area),-1)
    pressure=nv.last['pressure_quadrature'].reshape(J.shape)
    mismatch=J-1-pressure/nv.K
    uv=u.reshape(-1,2)
    vertex_F=np.eye(2)+np.einsum('eai,eaJ->eiJ',uv[el.elements[:,:3]],el.linear_grad)
    vertex_J=np.linalg.det(vertex_F)
    lower=quadratic_minimum(np.linalg.det(el.deformation(u,el.vertex_grad)))
    worst=int(np.argmin(lower)); coords=p.mesh.local_sn[el.elements[:,:3]]
    sample=np.einsum('qa,eaJ->eqJ',el.points,coords)
    distance=np.linalg.norm(sample-[p.config.drop_s_mm,p.config.height_mm],axis=-1)
    weights=el.area[:,None]*el.weights[None,:]
    stats=[]
    for radius in (.05,.15,.5,None):
        keep=np.ones(J.shape,bool) if radius is None else distance<=radius
        w=weights*keep; total=w.sum()
        if total==0:continue
        stats.append(dict(reference_radius_mm=radius,reference_area_mm2=float(total),
            quadrature_min_j=float(J[keep].min()),
            mean_j=float((w*J).sum()/total),
            projection_error_rms=float(np.sqrt((w*mismatch**2).sum()/total)),
            projection_error_max=float(np.abs(mismatch[keep]).max()),
            area_j_below_09_mm2=float(w[J<.9].sum()),
            area_abs_projection_error_above_005_mm2=float(w[np.abs(mismatch)>.05].sum()),
            bubble_j_rms=float(np.sqrt((w*(J-vertex_J[:,None])**2).sum()/total))))
    return dict(exact_min_j=float(lower.min()),worst_element=worst,
        worst_centroid_local_sn_mm=coords[worst].mean(axis=0).tolist(),
        worst_vertex_only_j=float(vertex_J[worst]),
        worst_bubble_coefficient_mm=uv[el.elements[worst,3]].tolist(),
        max_bubble_coefficient_mm=float(np.linalg.norm(uv[el.base_nodes:],axis=1).max()),
        pressure_mass_residual_mm3=nv.last['pressure_constraint_residual_mm3'],regions=stats)


def audit(run):
    run=Path(run); report=json.loads((run/'result.json').read_text(encoding='utf-8'))
    p=IPCProblem(**problem_arguments(report)); p.ipc.set_num_threads(2)
    rows=[]; previous=p.rest.copy()
    for i,step in enumerate(report['steps']):
        path=run/'states'/f'{i:04d}.npz'
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['reference_xy'],p.rest)
            np.testing.assert_array_equal(z['previous_positions'],previous)
            for key,value in p.checkpoint_data().items():np.testing.assert_array_equal(z[key],value)
            x=z['positions'].copy();p.previous=z['previous_positions'].copy()
        fresh=p.acceptance(x,step['depth_mm'],0,0)
        for key in ('reaction_fz_n','residual_n','min_j','friction_work_n_mm'):
            if abs(fresh[key]-step[key])>1e-8:raise AssertionError(f'Fresh mismatch: {key}')
        row=dict(depth_mm=step['depth_mm'],fresh_acceptance=fresh,
                 checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if p.element=='mini':row['spatial']=spatial_audit(p,x)
        rows.append(row);previous=x
    return dict(run=str(run.resolve()),run_status=report['status'],accepted=rows,force_shape_1mm=force_shape(report),
                status='passed',source_sha256=source_hashes(packages=('prototype','mechanics')))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with compute_process_policy('compute') as policy,threadpool_limits(2):
        result=audit(args.run)
    result['process_policy']=policy;write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],states=len(result['accepted']))),flush=True)


if __name__=='__main__':main()
