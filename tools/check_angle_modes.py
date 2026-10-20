"""Bounded mechanism audit of completed analytic-contact paths.

No material fitting or new production path. Work partitions are evaluated on
the actual last-increment virtual displacement; they are not causal fractions.
Constraint releases retain the exact old histories and are conditional endpoints.
"""
import argparse
import contextlib
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits
from mechanics.finite_strain import response, elastic_response
from prototype.curve_contact import CurveBladeProblem
from prototype.ipc_reference import BASELINE, LoadStepFailure
from prototype.model_config import _rotation_vectors, local_to_lab
from run_support import compute_process_policy, write_json
from tools.run_curve_contact_study import make_problem, curve_signature, load, read, sha, verify_manifest


def prepare(p, state):
    p.previous = state['previous_positions'].copy()
    p.previous_material_metric = state['previous_material_metric'].copy()
    p.material_metric = state['material_metric'].copy()
    p.material_dt = float(state['material_dt_s'])


def deformation(p, x):
    return np.eye(2) + np.einsum('eai,eaJ->eiJ',
        (x[:p.n]-p.rest[:p.n])[p.mesh.triangles], p.grad)


def tensor_parts(F, q, material):
    """Exact current PE alpha2 stress/energy split, independently reassembled."""
    if material.alpha_eq != (2.,) or material.volumetric != 'quadratic_j':
        raise ValueError('This diagnostic split requires quadratic-J alpha2 equilibrium')
    eq = elastic_response(F, replace(material, mu_ov=(), alpha_ov=()), tangent=False)
    J = q['J']; K = material.moduli()[1]
    pv = (K*J*(J-1))[:, None, None]*np.linalg.inv(F).transpose(0, 2, 1)
    stresses = dict(volume=pv, equilibrium_shear=eq['P']-pv, maxwell=q['P']-eq['P'])
    ev = K/2*(J-1)**2
    energies = dict(volume=ev, equilibrium_shear=eq['energy']-ev, maxwell=q['overstress_energy'])
    np.testing.assert_allclose(sum(stresses.values()), q['P'], atol=2e-13, rtol=2e-13)
    # Independent elastic and history kernels use different eigendecompositions;
    # allow their roundoff at strongly distorted states, not a force tolerance change.
    np.testing.assert_allclose(sum(energies.values()), q['energy'], atol=1e-10, rtol=2e-12)
    return stresses, energies


def basis(p, slide=False):
    """Admissible displacement basis; slider moves along the ROTATED bottom."""
    rows = list(p.free); cols = list(range(len(rows))); vals = [1.]*len(rows)
    anchor = None
    if slide:
        es, en = _rotation_vectors(p.config.inclination_deg)
        bottom = p.mesh.bottom_nodes
        origin = local_to_lab(np.array([[p.config.drop_s_mm, 0.]]), p.config)[0]
        anchor = int(bottom[np.argmin(np.linalg.norm(p.rest[bottom]-origin, axis=1))])
        for node in bottom:
            if node == anchor: continue
            column = len(p.free)+(len(vals)-len(p.free))//2
            rows.extend([2*int(node), 2*int(node)+1]); cols.extend([column, column]); vals.extend(es)
    count = max(cols)+1
    return coo_matrix((vals, (rows, cols)), shape=(p.total, count)).tocsc(), anchor


def inspect_state(p, state, record):
    fresh = p.audit_state(state, record); prepare(p, state)
    x = state['positions']; F = deformation(p, x); q = p.material_response(x)
    stresses, energies = tensor_parts(F, q, p.material)
    area, grad, _ = p.mesh.bulk_data; volume = area*p.config.width_mm
    step = p.material_dt*p.loading_rate
    velocity = (x-p.previous)/step
    dF = np.einsum('eai,eaJ->eiJ', velocity[:p.n][p.mesh.triangles], grad)
    work = {k: volume*np.einsum('eiJ,eiJ->e', value, dF) for k, value in stresses.items()}
    data, D, _ = p._blade_data(x)
    power = {k: float(v.sum()) for k, v in work.items()}
    power['blade_normal'] = float(data['normal_gradient'].ravel()@(D@velocity.ravel()))
    power['blade_friction'] = float(data['friction_gradient'].ravel()@(D@velocity.ravel()))
    c = p._self_collisions(x); sx = x[p.body_surface]
    sg = p.barrier.gradient(c, p.self_mesh, sx)
    lag = p.lag(x)
    if lag is not None:
        sg += p.friction.gradient(lag, p.self_mesh, (x-p.previous)[p.body_surface])
    power['self_contact'] = float(sg@velocity[p.body_surface].ravel())
    _, g, _, _ = p.evaluate(x, tangent=False)
    total = sum(power.values()); residual_bound = np.linalg.norm(g[p.free])*np.linalg.norm(velocity.ravel()[p.free])
    if abs(total-float(g@velocity.ravel())) > 2e-7: raise RuntimeError('Work partition assembly mismatch')
    if abs(total-record['reaction_fz_n']) > residual_bound+2e-7: raise RuntimeError('Work/reaction closure failed')
    # Spatial Hencky strain, expressed in the specimen normal/tangent frame.
    b = F@F.transpose(0, 2, 1); eig, Q = np.linalg.eigh(b)
    h = (Q*(.5*np.log(eig))[:, None, :])@Q.transpose(0, 2, 1)
    es, en = _rotation_vectors(p.config.inclination_deg); R = np.column_stack([es, en])
    hs = R.T@h@R
    centroid = x[:p.n][p.mesh.triangles].mean(axis=1)
    tip = x[p.n:][np.argmin(x[p.n:, 1])]
    distance = np.linalg.norm(centroid-tip, axis=1)
    stress = q['P']@F.transpose(0, 2, 1)/q['J'][:, None, None]
    sigma_out = q['plane_stress_residual']/q['J']
    regions = []
    for lo, hi in ((0., .25), (.25, 1.), (1., np.inf)):
        mask = (distance>=lo)&(distance<hi)
        regions.append(dict(radius_mm=[lo, None if np.isinf(hi) else hi],
            reference_area_mm2=float(area[mask].sum()),
            energy_n_mm={k: float(volume[mask]@v[mask]) for k, v in energies.items()},
            virtual_work_n={k: float(v[mask].sum()) for k, v in work.items()}))
    near = distance<1.; weights = area[near]/area[near].sum()
    row = dict(angle_deg=p.config.inclination_deg, depth_mm=record['depth_mm'], fresh_audit=fresh,
        reaction=dict(total_n=record['reaction_fz_n'], normal_n=record['reaction_normal_fz_n'], friction_n=record['reaction_friction_fz_n']),
        work_n=power, work_sum_n=total, work_residual_bound_n=float(residual_bound),
        energy_n_mm={k: float(volume@v) for k, v in energies.items()}, regions=regions,
        near_1mm=dict(reference_area_mm2=float(area[near].sum()),
            normal_log_strain_rms=float(np.sqrt(weights@hs[near, 1, 1]**2)),
            shear_log_strain_rms=float(np.sqrt(weights@(2*hs[near, 0, 1])**2)),
            tangential_log_strain_rms=float(np.sqrt(weights@hs[near, 0, 0]**2)),
            out_of_plane_stress_rms_mpa=float(np.sqrt(weights@sigma_out[near]**2))),
        min_J=float(q['J'].min()), max_stretch=float(np.sqrt(eig.max())),
        compressed_reference_area_mm2={str(t): float(area[q['J']<t].sum()) for t in (.95, .8, .5)},
        kinetic_scope='Actual previous-to-current displacement divided by depth increment; instantaneous virtual work, not integrated energy fractions')
    fields = dict(reference_xy=p.rest[:p.n], positions=x[:p.n], triangles=p.mesh.triangles,
        blade=x[p.n:], J=q['J'], normal_log_strain=hs[:, 1, 1], shear_log_strain=2*hs[:, 0, 1],
        out_of_plane_stress_mpa=sigma_out, maxwell_energy_mpa=energies['maxwell'],
        cauchy=stress, contact_tip=tip)
    return row, fields


def plane_stress_methods(p):
    """Bind an explicit diagnostic PS material; same constitutive law/history."""
    from types import MethodType
    def material_response(self, x, *, tangent=False):
        return response(deformation(self, x), self.previous_material_metric, self.material_dt,
                        self.material, tangent=tangent, kinematics='plane_stress')
    def material_bulk(self, u, *, with_tangent=True):
        areas, gradients, dofs = self.mesh.bulk_data
        F = np.eye(2)+np.einsum('eai,eaJ->eiJ',u.reshape(-1,2)[self.mesh.triangles],gradients)
        q = response(F,self.previous_material_metric,self.material_dt,self.material,
                     tangent=with_tangent,kinematics='plane_stress')
        if np.max(q['plane_stress_scaled_residual'])>1e-8:raise RuntimeError('Local plane-stress check failed')
        volumes=areas*self.config.width_mm
        r=np.einsum('e,eiJ,eaJ->eai',volumes,q['P'],gradients).reshape(-1,6)
        k=None
        if with_tangent:
            k=np.einsum('e,eaJ,eiJkL,ebL->eaibk',volumes,gradients,q['tangent'].reshape(-1,2,2,2,2),
                        gradients,optimize=True).reshape(-1,6,6)
        return r,k,dofs,float(volumes@q['incremental_potential']),float(q['J'].min())
    p.material_response=MethodType(material_response,p);p.material_bulk=MethodType(material_bulk,p)


def conditional_newton(p, x, depth, B, deadline):
    """Full nonsymmetric Jacobian, residual Armijo, original CCD/J and tolerance."""
    checks=[]
    for it in range(45):
        if time.monotonic()>deadline:raise TimeoutError('Conditional constraint wall budget')
        _,g,A,J=p.evaluate(x,projected=False);r=np.asarray(B.T@g).ravel();norm=np.linalg.norm(r)
        print(f'CONSTRAINT iteration={it} residual_N={norm:.10g} min_J={J:.9g}',flush=True)
        if norm<=BASELINE['force_tolerance_n']:return x,it,checks
        Ar=(B.T@A@B).tocsc();du=spsolve(Ar,-r)
        error=float(np.linalg.norm(Ar@du+r))
        if not np.isfinite(du).all() or error>max(1e-9,1e-7*norm):raise LoadStepFailure('Constraint linear residual failed')
        direction=(B@du).reshape(-1,2)
        if it==0:
            unit=direction/max(np.max(np.abs(direction)),1e-30);exact=A@unit.ravel()
            for h in (3e-7,1e-7):
                if min(p.safe_step(x,h*unit),p.safe_step(x,-h*unit))<1:raise LoadStepFailure('Constraint derivative geometry unsafe')
                fd=(p.evaluate(x+h*unit,tangent=False)[1]-p.evaluate(x-h*unit,tangent=False)[1])/(2*h)
                checks.append(float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-30)))
            if max(checks)>2e-3:raise LoadStepFailure('Constraint tangent directional check failed')
        alpha=min(1.,.99*p.safe_step(x,direction))
        for back in range(22):
            if time.monotonic()>deadline:raise TimeoutError('Conditional constraint line-search budget')
            trial=x+alpha*direction
            if np.array_equal(trial,x):raise LoadStepFailure('Constraint step cannot be represented')
            _,gt,_,jt=p.evaluate(trial,tangent=False);rt=np.linalg.norm(B.T@gt)
            if jt>0 and rt<=(1-1e-4*alpha)*norm:
                x=trial;break
            alpha*=.5
        else:raise LoadStepFailure('Constraint residual line search failed')
    raise LoadStepFailure('Conditional constraint iteration budget')


def release(p, s, record, kind, out, budget):
    prepare(p,s);x=s['positions'].copy();depth=float(s['depth_mm'])
    if kind=='plane_stress':plane_stress_methods(p)
    B,anchor=basis(p,slide=kind=='bottom_slider')
    result=dict(kind=kind,angle_deg=p.config.inclination_deg,depth_mm=depth,seconds_budget=budget,
        status='running',baseline_fz_n=record['reaction_fz_n'],anchor_bottom_node=anchor,
        scope='Conditional endpoint with original PE/clamped pre-step histories; NOT a new full loading path or experimentally justified boundary')
    started=time.monotonic()
    try:
        value,it,checks=conditional_newton(p,x,depth,B,started+budget)
        accepted=p.acceptance(value,depth,it+1,0)
        _,g,_,_=p.evaluate(value,tangent=False);fresh_norm=float(np.linalg.norm(B.T@g))
        if fresh_norm>BASELINE['force_tolerance_n']:raise RuntimeError('Fresh released residual failed')
        es,en=_rotation_vectors(p.config.inclination_deg)
        ub=value[p.mesh.bottom_nodes]-p.rest[p.mesh.bottom_nodes]
        if kind=='bottom_slider':
            if np.max(np.abs(ub@en))>1e-11 or np.max(np.abs(value[anchor]-p.rest[anchor]))>1e-11:
                raise RuntimeError('Rotated bottom constraints failed')
        elif not np.array_equal(value[p.mesh.bottom_nodes],p.rest[p.mesh.bottom_nodes]):
            raise RuntimeError('Clamped bottom changed')
        q=p.material_response(value)
        if kind=='plane_stress' and np.max(q['plane_stress_scaled_residual'])>1e-8:raise RuntimeError('PS terminal residual failed')
        result.update(status='accepted',fresh_reduced_residual_n=fresh_norm,record=accepted,
            tangent_relative_errors=checks,delta_fz_n=accepted['reaction_fz_n']-record['reaction_fz_n'],
            relative_force_change=accepted['reaction_fz_n']/record['reaction_fz_n']-1,
            lambda_out_range=[float(q['lambda_out'].min()),float(q['lambda_out'].max())],
            inplane_min_J=float(np.linalg.det(deformation(p,value)).min()),
            max_bottom_tangent_motion_mm=float(np.max(np.abs(ub@es))))
        candidate={k:v for k,v in s.items() if k!='cumulative_viscous_dissipation_n_mm'}
        candidate.update(positions=value,material_metric=q['metric'])
        np.savez_compressed(out,**candidate)
        result['checkpoint_sha256']=sha(out)
    except (LoadStepFailure,TimeoutError) as exc:
        result.update(status='failed',error=str(exc))
    result['elapsed_s']=time.monotonic()-started
    return result


def verify_sources(manifest):
    verify_manifest(manifest)
    root=Path(__file__).resolve().parents[1]
    # The calculation imports working-tree modules only after proving equality
    # to the completed run's frozen physical/runtime source.
    for rel,digest in manifest['source_sha256'].items():
        if rel.startswith(('prototype/','mechanics/')) or rel in ('run_support.py','tools/run_curve_contact_study.py'):
            if sha(root/rel)!=digest:raise RuntimeError('Working source differs from frozen run: '+rel)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',type=Path,default=Path('outputs/curve_contact_depth_20261015'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phase',choices=['fields','constraints'],required=True)
    ap.add_argument('--seconds',type=float,default=120.)
    args=ap.parse_args();manifest=read(args.study/'manifest.json');verify_sources(manifest)
    queue=read(args.study/'queue_status.json')
    if queue['status']!='finished' or not all(j.get('fresh_audit_passed') for j in queue['jobs']):
        raise RuntimeError('All six terminal audited paths are required')
    args.output.mkdir(exist_ok=True,parents=True)
    report=dict(status='running',phase=args.phase,script_sha256=sha(__file__),
        frozen_sources=manifest['source_sha256'],inputs={},rows=[],
        scope='Same material at both angles; no fracture, fit, production launch or default change')
    def save():write_json(args.output/(args.phase+'.json'),report)
    with threadpool_limits(limits=2),compute_process_policy() as policy:
        report['runtime_policy']=policy
        for angle in (30,60):
            job=next(j for j in queue['jobs'] if j['angle']==angle and j['variant']=='base')
            folder=Path(job['output']);r=read(folder/'result.json')
            for depth in ((.1,.25,.5) if args.phase=='fields' else (.25,.5)):
                index=next(i for i,s in enumerate(r['steps']) if abs(s['depth_mm']-depth)<1e-10)
                path=folder/'states'/f'{index:04d}.npz';digest=sha(path)
                if digest!=r['checkpoint_sha256'][path.name]:raise RuntimeError('Changed checkpoint')
                report['inputs'][str(path)]=digest;s=load(path);record=r['steps'][index]
                if args.phase=='fields':
                    p=make_problem(job)
                    if curve_signature(p)!=r['signature']:raise RuntimeError('Candidate signature mismatch')
                    row,fields=inspect_state(p,s,record)
                    np.savez_compressed(args.output/f'fields_{angle}_{depth:.2f}.npz',**fields)
                    row['checkpoint_sha256']=digest;report['rows'].append(row);save()
                    print('FIELD_DONE',angle,depth,flush=True)
                else:
                    for kind in ('bottom_slider','plane_stress'):
                        p=make_problem(job);p.audit_state(s,record)
                        name=f'{kind}_{angle}_{depth:.2f}'
                        with (args.output/(name+'.log')).open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                            row=release(p,s,record,kind,args.output/(name+'.npz'),args.seconds)
                        report['rows'].append(row);save()
                        print('RELEASE_DONE',json.dumps(row),flush=True)
    report['status']='completed';save()


if __name__=='__main__':main()
