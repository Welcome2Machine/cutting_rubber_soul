"""Read-only actual-state virtual work and material verification.

Runs against a case's verified frozen source, never advances an FE state.
The accepted displacement increment is a virtual field, not a claim that
endpoint power is a finite-step energy balance or a causal force partition.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def virtual_work(p, state):
    from mechanics.finite_strain import elastic_response
    if p.material.volumetric != 'quadratic_j':
        raise ValueError('This explicit volume split requires quadratic_j')
    x = state['positions']
    p.previous = state['previous_positions']
    p.previous_material_metric = state['previous_material_metric']
    p.material_dt = float(state['material_dt_s'])
    delta_d = p.material_dt*p.loading_rate
    v = (x-p.previous)/delta_d
    np.testing.assert_allclose(v[p.n:], np.tile([0., -1.], (len(x)-p.n, 1)), atol=2e-10)
    np.testing.assert_allclose(v[p.mesh.bottom_nodes], 0., atol=1e-12)
    F = np.eye(2)+np.einsum('eai,eaJ->eiJ',
        (x[:p.n]-p.rest[:p.n])[p.mesh.triangles], p.grad)
    q = p.constitutive_response(F, tangent=False)
    eq = elastic_response(F, p.material, tangent=False, kinematics='plane_strain')
    J = q['J']; K = p.material.moduli()[1]
    pv = K*(J*(J-1))[:, None, None]*np.linalg.inv(F).transpose(0, 2, 1)
    D = np.einsum('eai,eaJ->eiJ', v[:p.n][p.mesh.triangles], p.grad)
    volumes = p.material_integration_weights()
    works = {}
    for name, P in (('equilibrium_iso', eq['P']-pv), ('volume', pv),
                    ('maxwell', q['P']-eq['P'])):
        works[name] = float(np.einsum('e,eiJ,eiJ->', volumes, P, D))
    sx = x[p.surface]; c = p.collisions(x); lag = p.lag(x)
    ng = p.barrier.gradient(c, p.cmesh, sx).reshape(-1, 2)
    fg = p.friction.gradient(lag, p.cmesh, (x-p.previous)[p.surface]).reshape(-1, 2)
    works['normal_contact'] = float(np.sum(ng*v[p.surface]))
    works['friction'] = float(np.sum(fg*v[p.surface]))
    _, g, _, _ = p.evaluate(x, lag, tangent=False)
    fz = float(-g.reshape(-1, 2)[p.n:, 1].sum())
    free_work = float(g[p.free]@v.ravel()[p.free])
    np.testing.assert_allclose(sum(works.values()), fz+free_work, rtol=1e-10, atol=1e-9)
    # Reference-volume statistics; minJ is not the whole sample's volume loss.
    active_nodes = set()
    for name in ('vv_collisions', 'ev_collisions', 'ee_collisions', 'fv_collisions'):
        for s in getattr(c, name, []):
            ids = np.array([i for i in s.vertex_ids(p.cmesh.edges, p.cmesh.faces) if i >= 0])
            full = p.surface[ids]
            if np.any(full >= p.n): active_nodes.update(int(i) for i in full if i < p.n)
    bearing = np.any(np.isin(p.mesh.triangles, list(active_nodes)), axis=1)
    stats = {}
    for name, mask in (('all', np.ones(len(F), bool)), ('contact_adjacent_elements', bearing)):
        w = volumes[mask]
        stats[name] = dict(elements=int(mask.sum()), reference_volume_mm3=float(w.sum()),
            min_J=float(J[mask].min()), weighted_mean_J=float(w@J[mask]/w.sum()),
            reference_volume_fraction_below={str(t):float(w@(J[mask]<t)/w.sum()) for t in (.9,.8,.5)},
            volumetric_energy_nmm=float(w@(K/2*(J[mask]-1)**2)))
    # Independently verify actual material states, including the most compressed.
    ids = np.unique(np.r_[np.argsort(J)[:8], np.flatnonzero(bearing)[::max(1,int(bearing.sum())//8)],
                           np.linspace(0, len(F)-1, 8).astype(int)])
    checks = material_checks(F[ids], p.previous_material_metric[ids], p.material_dt, p.material)
    return dict(fz_n=fz, virtual_work_per_depth_n=works, free_residual_work_n=free_work,
                closure_error_n=sum(works.values())-fz-free_work,
                compression=stats, material_checks=checks,contact_checks=contact_checks(p,x),
                caveat='Endpoint virtual work on the actual accepted increment; not a causal parameter attribution or finite-step energy balance.')


def contact_checks(p,x):
    """Actual loaded geometry, plus an imposed common rotation at fixed gap.

    Only the positive-weight, persistent-EV domain is compared with the local
    repair. Unsupported interactions are counted, never silently discarded.
    Conditional force changes do NOT constitute a new equilibrium.
    """
    from prototype.ipc_reference import BASELINE
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('prototype._physical_candidate',root/'prototype/contact_controls.py')
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    sx=x[p.surface];delta=(x-p.previous)[p.surface];lag=p.lag(x)
    rot=np.array([[0.,-1.],[1.,0.]])
    fg=p.friction.gradient(lag,p.cmesh,delta).reshape(-1,2)
    ng=p.barrier.gradient(p.collisions(x),p.cmesh,sx).reshape(-1,2)
    moments=dict(normal_nmm=float(np.sum(ng*(sx@rot.T))),friction_nmm=float(np.sum(fg*(sx@rot.T))))
    rotations=[]
    for angle in (.001,.01):
        Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        y=x@Q.T;du=(y-x)[p.surface];rlag=p.lag(y)
        force=p.friction.gradient(rlag,p.cmesh,du)
        rotations.append(dict(angle_rad=angle,friction_work_nmm=float(force@du.ravel()),
                              friction_gradient_norm_n=float(np.linalg.norm(force))))
    counts={};rows=[];excluded=0.;eligible=0.;difference=np.zeros_like(sx)
    for kind in ('vv','ev'):
        for s in getattr(lag,kind+'_collisions'):
            ids=np.array([i for i in s.vertex_ids(p.cmesh.edges,p.cmesh.faces) if i>=0],int)
            capacity=abs(float(s.weight*s.normal_force_magnitude*s.mu_s_effective_lagged))
            reason=None
            if kind!='ev':reason='VV_requires_surface_chart'
            elif s.weight<=0:reason='signed_correction_not_positive_contact_area'
            else:
                try:
                    repaired=candidate.objective_edge_contact(sx[ids],p.previous[p.surface[ids]],
                        weight=float(s.weight),dhat=p.dhat,
                        stiffness=BASELINE['barrier_stiffness_mpa']*p.config.width_mm,
                        mu=p.config.friction_coefficient,epsilon=float(p.friction.eps_v))
                except ValueError as exc:reason=str(exc)
            if reason:
                counts[reason]=counts.get(reason,0)+1;excluded+=capacity;continue
            eligible+=capacity
            native=p.friction.gradient(s,delta[ids].ravel()).reshape(3,2)
            corrected=repaired['friction_gradient'].reshape(3,2)
            # Verify normal capacity independently against the frozen library.
            np.testing.assert_allclose(repaired['normal_capacity_n'],s.weight*s.normal_force_magnitude,rtol=1e-8,atol=1e-8)
            np.add.at(difference,ids,corrected-native)
            rows.append(dict(nodes=p.surface[ids].tolist(),capacity_n=capacity,
                friction_force_change_norm_n=float(np.linalg.norm(corrected-native)),
                native_friction_moment_nmm=float(np.sum(native*(sx[ids]@rot.T))),
                repaired_friction_moment_nmm=float(np.sum(corrected*(sx[ids]@rot.T)))))
    return dict(actual_native_total_moments=moments,common_rotation_at_loaded_geometry=rotations,
        local_repair_eligible_pairs=len(rows),unsupported_counts=counts,
        eligible_absolute_stencil_capacity_n=eligible,unsupported_absolute_stencil_capacity_n=excluded,
        conditional_partial_blade_force_change_xy_n=(-difference[p.surface>=p.n].sum(axis=0)).tolist(),
        maximum_local_native_friction_moment_nmm=max((abs(r['native_friction_moment_nmm']) for r in rows),default=0.),
        maximum_local_repaired_friction_moment_nmm=max((abs(r['repaired_friction_moment_nmm']) for r in rows),default=0.),
        full_contact_repair_qualified=False,
        caveat='Absolute stencil capacities are NOT macroscopic force fractions. Partial conditional change is not an equilibrated Fz prediction. Cross-edge/VV/signed combinations remain unsupported.')


def material_checks(F, old, dt, material):
    from mechanics.finite_strain import response
    r = response(F, old, dt, material, kinematics='plane_strain')
    Q = np.array([[np.cos(.37),-np.sin(.37)], [np.sin(.37),np.cos(.37)]])
    rotated = response(Q@F, old, dt, material, tangent=False, kinematics='plane_strain')
    stress_error = float(np.linalg.norm(rotated['P']-Q@r['P'])/max(np.linalg.norm(r['P']),1e-30))
    np.testing.assert_allclose(rotated['P'], Q@r['P'], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(rotated['metric'],r['metric'],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(rotated['energy'],r['energy'],rtol=1e-9,atol=1e-10)
    D = np.random.default_rng(20261021).normal(size=F.shape)
    D /= np.linalg.norm(D, axis=(1,2))[:,None,None]
    action = np.einsum('eij,ej->ei',r['tangent'],D.reshape(-1,4))
    errors = []
    for h in (1e-5, 3e-6, 1e-6):
        plus = response(F+h*D,old,dt,material,tangent=False,kinematics='plane_strain')
        minus = response(F-h*D,old,dt,material,tangent=False,kinematics='plane_strain')
        fd = ((plus['P']-minus['P'])/(2*h)).reshape(-1,4)
        errors.append(dict(h=h,relative_error=float(np.linalg.norm(fd-action)/np.linalg.norm(action))))
    if min(e['relative_error'] for e in errors)>2e-5:raise AssertionError('Actual-state material tangent mismatch')
    if r['viscous_rate'].min() < -1e-10:raise AssertionError('Negative material dissipation')
    return dict(points=len(F),rigid_rotation_stress_relative_error=stress_error,
                tangent_direction_errors=errors,min_dissipation_density=float(r['viscous_rate'].min()),
                max_metric_determinant_error=float(np.max(abs(np.linalg.det(r['metric'])-1))),passed=True)


def volume_sensitivity(p,state,out,label):
    """Conditional implicit derivative wrt log(K), with the OLD history fixed.

    NOT the derivative of an independently rerun material history. The entire
    fresh nonsymmetric contact tangent is used, no PSD projection/PTC shift.
    Tiny first-order trial fields certify this local derivative, not a new
    equilibrium or a material-parameter fit.
    """
    from prototype.pardiso import PardisoDirection
    from run_support import save_npz
    root=Path(__file__).resolve().parents[1];x=state['positions'];K=p.material.moduli()[1]
    p.previous=state['previous_positions'];p.previous_material_metric=state['previous_material_metric']
    p.material_dt=float(state['material_dt_s'])
    def volume_gradient(y):
        F=np.eye(2)+np.einsum('eai,eaJ->eiJ',(y[:p.n]-p.rest[:p.n])[p.mesh.triangles],p.grad)
        J=np.linalg.det(F);P=K*(J*(J-1))[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
        local=np.einsum('e,eiJ,eaJ->eai',p.material_integration_weights(),P,p.grad).reshape(-1,6)
        result=np.zeros(p.total);np.add.at(result,p.mesh.bulk_data[2].ravel(),local.ravel())
        return result
    lag=p.lag(x);_,g,H,_=p.evaluate(x,lag,projected=False)
    A=p.coupled_friction_tangent(x,lag,H);partial=volume_gradient(x)
    solver=PardisoDirection(8,runtime=str(root/'.runtime/pardiso'))
    try:
        du=solver.solve(A[p.free][:,p.free],-partial[p.free])
        calls=list(solver.rows)
    finally:solver.close()
    direction=np.zeros(p.total);direction[p.free]=du;action=A@direction+partial
    rel=float(np.linalg.norm(action[p.free])/max(np.linalg.norm(partial[p.free]),1e-30))
    if rel>1e-8:raise RuntimeError('Conditional sensitivity linear residual failed')
    fz=lambda vec:float(-vec.reshape(-1,2)[p.n:,1].sum())
    derivative=fz(action);trials=[]
    for eps in (1e-4,1e-5,1e-6):
        change=eps*direction.reshape(-1,2)
        safe=float(p.safe_step(x,change))
        if safe<1.-1e-10:
            trials.append(dict(log_K_change=eps,safe_fraction=safe,status='geometry_limited'));continue
        y=x+change
        _,gy,_,minj=p.evaluate(y,p.lag(y),tangent=False)
        # At fixed previous isochoric history, changing K affects only the
        # current equilibrium volumetric term; this is exact for this material.
        gy+=np.expm1(eps)*volume_gradient(y)
        actual=(fz(gy)-fz(g))/eps
        trials.append(dict(log_K_change=eps,status='evaluated',min_J=minj,
            finite_difference_fz_n=actual,relative_force_derivative_error=abs(actual-derivative)/max(abs(derivative),1e-20),
            normalized_free_increment_error=float(np.linalg.norm((gy-g)[p.free]))/(eps*max(np.linalg.norm(partial[p.free]),1e-30))))
    saved=out/f'volume_direction_{label}.npz'
    save_npz(saved,positions=x,previous_positions=p.previous,previous_material_metric=p.previous_material_metric,
             material_dt_s=p.material_dt,direction_logK=direction,partial_residual_logK=partial)
    return dict(dFz_dlogK_n=derivative,original_K_mpa=K,linear_relative_residual=rel,
        direction_infinity_mm=float(np.max(abs(direction))),linear_calls=calls,trials=trials,
        probe_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
        scope='Conditional infinitesimal equilibrium response with previous material/friction history fixed; no new accepted state or full-history parameter derivative.')


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--volume-sensitivity',action='store_true')
    args=ap.parse_args();study=args.study.resolve();out=args.output.resolve()
    root=Path(__file__).resolve().parents[1]
    out.mkdir(parents=True,exist_ok=True)
    sources={}
    for name in ('tools/audit_physical_balance.py','prototype/contact_controls.py',
                 'tests/test_objective_edge_contact.py'):
        saved=out/'source'/name;saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,saved)
        sources[name]=hashlib.sha256(saved.read_bytes()).hexdigest()
    # No project imports before installing the frozen source path.
    sys.path.insert(0,str(root/'.runtime/ipc-python'));sys.path.insert(0,str(study/'source'))
    from tools import run_common_material_study as host
    from prototype.material_case import signature
    from run_support import write_json
    from threadpoolctl import threadpool_limits
    manifest=host.read(study/'manifest.json');host.check(manifest,study)
    inputs={};rows=[]
    with threadpool_limits(1):
        for angle in (0,33):
            folder=study/f'soft_q30_{angle}';inputs.update(host.verify_case(folder))
            report=host.read(folder/'result.json');p=host.make(angle,report['job']['parameters'])
            assert json.loads(json.dumps(signature(p)))==report['signature']
            for depth in (.1,.25,.4,.5):
                i=next(i for i,r in enumerate(report['steps']) if abs(r['depth_mm']-depth)<1e-10)
                with np.load(folder/'states'/f'{i:04d}.npz',allow_pickle=False) as z:state=dict(z)
                fresh=p.audit_state(state,report['steps'][i])
                row=dict(angle=angle,depth_mm=depth,fresh_audit=fresh,**virtual_work(p,state))
                if args.volume_sensitivity and depth in (.4,.5):
                    row['conditional_volume_sensitivity']=volume_sensitivity(p,state,out,f'{angle}_{depth:.2f}')
                rows.append(row)
                print(json.dumps(row),flush=True)
    write_json(out/'balance.json',dict(new_loading=False,rows=rows,input_sha256=inputs,
        diagnostic_source_sha256=sources,frozen_physical_source_manifest=str(study/'manifest.json'),
        frozen_physical_source_manifest_sha256=host.sha(study/'manifest.json')))


if __name__=='__main__':main()
