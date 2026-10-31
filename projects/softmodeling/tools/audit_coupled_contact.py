"""Independent assembly, history/trajectory and entry audit for coupled windows."""
import argparse,json,time
from pathlib import Path
import numpy as np
from mechanics.discrete_gradient import frozen_response
from mechanics.finite_strain import response
from coupled_contact import problem
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from .run_coupled_contact import sources
from .audit_precrack_entry import strict_contacts,check_entry_flag
from . import run_zero_gap_cut as base
from .frozen_scene import digest
from .face_witness import native
from run_support import write_json


def independent_balance(p,s):
    x=s['positions'];old=s['previous_positions'];du=s['increment'];h=float(s['material_dt_s'])
    grad=p.mesh.bulk_data[1];tri=p.mesh.triangles;volume=p.material_integration_weights()
    F=lambda z:np.einsum('eai,eaJ->eiJ',z[tri],grad)
    F1,F0=F(x),F(old);A=s['previous_material_metric']
    oldq=frozen_response(F0,A,p.config.shear_modulus_n_mm2,p.material.mu_ov[0],p.config.bulk_modulus_n_mm2,tangent=False)
    if p.material_layer=='maxwell':q=response(F1,A,h,p.material,tangent=False,kinematics='plane_strain')
    else:q=frozen_response(F1,A,p.config.shear_modulus_n_mm2,p.material.mu_ov[0],p.config.bulk_modulus_n_mm2,tangent=False)
    bulk=np.zeros_like(x);local=volume[:,None,None]*np.einsum('eiJ,eaJ->eai',q['P'],grad)
    np.add.at(bulk,tri.ravel(),local.reshape(-1,2))
    n=np.zeros_like(x);n[p.surface]=p.barrier.gradient(p.collisions(x),p.cmesh,x[p.surface]).reshape(-1,2)
    f=np.zeros_like(x);lag=p.lag(x)
    if lag is not None:f[p.surface]=p.friction.gradient(lag,p.cmesh,du[p.surface]).reshape(-1,2)
    # Pure precrack has zero traction on every exposed face; intact faces are
    # exact ties. Check this property rather than drop arbitrary cohesive forces.
    cq=p.cohesive_response(x);assert np.max(abs(cq['traction'][p.active_faces]),initial=0.)==0.
    np.testing.assert_array_equal(s['active_faces'],p.precracked_faces)
    np.testing.assert_array_equal(s['history'],s['previous_history'])
    v=s['velocity'];v0=s['previous_velocity'];moment=p.mass_dof.reshape(-1,2)*(v-v0)
    residual=moment/h+bulk+n+f
    force=float(np.sqrt(2)*np.linalg.norm(residual.ravel()[p.free]))
    if force>1e-4:raise ValueError(f'Independent coupled residual {force:.17g} N')
    np.testing.assert_allclose(s['impulse_bulk'],-h*bulk,rtol=1e-8,atol=2e-12)
    for name,z in [('normal',n),('friction',f)]:np.testing.assert_allclose(s['impulse_'+name],-h*z,rtol=1e-10,atol=1e-14)
    np.testing.assert_allclose(du[:p.n],h*v[:p.n],rtol=0,atol=1e-14)
    Ub=2*float(volume@(q['energy']-oldq['energy']))
    Dv=2*h*float(volume@q['viscous_rate']) if p.material_layer=='maxwell' else 0.
    dK=float(p.mass@np.sum(v[:p.n]**2-v0[:p.n]**2,axis=1))
    Un=2*float(p.barrier(p.collisions(x),p.cmesh,x[p.surface])-p.barrier(p.collisions(old),p.cmesh,old[p.surface]))
    W=2*float(np.sum((n+f)[p.n:]*du[p.n:]));Df=2*float(np.sum(f*du))
    Rb=2*float(np.sum(bulk*du))-Ub-Dv;Rn=2*float(np.sum(n*du))-Un
    Rm=float(p.mass@np.sum((v[:p.n]-v0[:p.n])**2,axis=1))
    active=p.active_contact.copy();active[p.seam[~p.released]]=False
    Rc=-2*float(residual[:p.n,0][active]@du[:p.n,0][active])
    eqwork=2*float(residual.ravel()[p.free]@du.ravel()[p.free])
    physical=W-Ub-Un-Df-Dv-dK
    closure=physical-Rb-Rn-Rm-Rc
    # Free residual work is the only allowed algebraic remainder, not a hidden
    # energy source. The independent material assembly may differ at roundoff.
    np.testing.assert_allclose(closure,-eqwork,rtol=1e-4,atol=1e-10)
    return dict(residual_n=force,impulse_residual_n_s=h*force,
        independent_external_work_n_mm=W,independent_friction_n_mm=Df,
        independent_bulk_delta_n_mm=Ub,independent_maxwell_n_mm=Dv,
        independent_barrier_delta_n_mm=Un,independent_kinetic_delta_n_mm=dK,
        net_physical_remainder_n_mm=physical,numerical_bulk_n_mm=Rb,numerical_barrier_n_mm=Rn,
        numerical_inertia_n_mm=Rm,numerical_constraint_n_mm=Rc,algebraic_residual_work_n_mm=closure,
        absolute_numerical_including_equation_work_n_mm=sum(abs(z) for z in (Rb,Rn,Rm,Rc,closure)))


@runtime_entry
def inspect(case,out,seconds=500,*,runtime=None):
    case=Path(case);out=Path(out);start=time.monotonic()
    if out.exists() or not 0<seconds<=540:raise ValueError('New bounded audit required')
    raw=json.loads((case/'result.json').read_text());report=dict(passed=False,source_hashes=sources(),result_sha256=digest(case/'result.json'),auditor_sha256=digest(Path(__file__)),thread_runtime=runtime,states=[])
    try:
        if raw['source_hashes']!=sources():raise ValueError('Numerical source identity mismatch')
        p=problem(raw['config'],raw['material_layer'])
        with np.load(case/'parent_state.npz',allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
        if digest(case/'parent_state.npz')!=raw['parent']['state_sha256']:raise ValueError('Parent identity mismatch')
        previous=dict(time_s=raw['initial_time_s'],depth_mm=raw['initial_depth_mm'],state_file='parent_state.npz')
        for row in raw['steps']:
            if time.monotonic()-start>seconds:raise TimeoutError('Independent window audit budget')
            file=case/row['state_file']
            if digest(file)!=row['state_sha256']:raise ValueError('State hash mismatch')
            with np.load(file,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
            for a,b in [('previous_positions','positions'),('previous_velocity','velocity'),('previous_material_metric','material_metric'),('previous_history','history'),('previous_active_faces','active_faces')]:np.testing.assert_array_equal(s[a],parent[b])
            p.active_faces=s['active_faces'].copy();p.previous_active_faces=s['previous_active_faces'].copy()
            check=base.audit(p,s,row)
            if not check['accepted_chord_ccd_safe']:raise ValueError('Full chord not safe')
            balance=independent_balance(p,s)
            contacts=strict_contacts(p,s['positions'],raw['config']['precrack_mm']);check_entry_flag(contacts,row['blade_entered_surface_channel'])
            if contacts and 'first_entry_bracket' not in report:report['first_entry_bracket']=dict(previous_nonentry=previous,first_entry=dict(time_s=row['time_s'],depth_mm=row['depth_mm'],state_file=row['state_file']))
            if not contacts:previous=dict(time_s=row['time_s'],depth_mm=row['depth_mm'],state_file=row['state_file'])
            report['states'].append(dict(state_file=row['state_file'],time_s=row['time_s'],depth_mm=row['depth_mm'],strict_contacts=contacts,**balance))
            parent=s
        report.update(passed=bool(raw['steps']),strict_entry_verified=any(s['strict_contacts'] for s in report['states']),
            complete_requested_window=raw['status']=='completed_window',new_gc_n_mm=0.,no_activation=True,
            time_converged=False,mesh_converged=False,event_complete=False,
            max_residual_n=max((s['residual_n'] for s in report['states']),default=None),
            absolute_numerical_including_equation_work_n_mm=sum(s['absolute_numerical_including_equation_work_n_mm'] for s in report['states']))
    except (ValueError,RuntimeError,AssertionError,TimeoutError,KeyError) as exc:report['error']=str(exc)
    report.update(elapsed_s=time.monotonic()-start,source_unchanged=sources()==raw['source_hashes'])
    report['passed'] &= report['source_unchanged'];write_json(out,native(report));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--seconds',type=float,default=500);add_thread_arguments(p);a=p.parse_args()
    q=inspect(a.case,a.out,a.seconds,**thread_arguments(a));print(json.dumps({k:v for k,v in q.items() if k not in ('states','source_hashes','thread_runtime')}))
if __name__=='__main__':main()
