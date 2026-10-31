"""Same-parent, same-step root errors and explicitly distinct IPC transitions.

The Jacobian correction is a fixed local linearization estimate, not an error
bound. Conditional roots are never appended to the original physical history.
"""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve
from coupled_contact import problem
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from .run_coupled_contact import sources,restore_parent
from .coupled_audit_contract import bind_case
from . import run_zero_gap_cut as base
from . import run_fracture_route as route
from .audit_precrack_entry import strict_contacts
from .face_witness import native,accepted_fields,array_hash,trial_fields
from .frozen_scene import digest
from run_support import write_json


def read_state(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def contact_signature(p,x):
    lag=p.lag(x);records=[];du=p.step_increment(x);h=p.material_dt
    if lag is not None:
        for kind in ('vv_collisions','ev_collisions'):
            for c in getattr(lag,kind):
                ids=np.asarray([i for i in c.vertex_ids(p.cmesh.edges,p.cmesh.faces) if i>=0]);nodes=p.surface[ids]
                vrel=c.relative_velocity((du[nodes]/h).ravel())
                speed=float(np.linalg.norm(np.asarray(c.tangent_basis).T@vrel))
                records.append(dict(kind=kind,nodes=nodes.tolist(),native_normal_magnitude= float(c.normal_force_magnitude),native_weight=float(c.weight),weighted_normal_load_n=2*float(c.weight*c.normal_force_magnitude),
                    tangent_speed_mm_s=speed,regularization_ratio=speed/p.friction_velocity_epsilon,
                    friction_branch='regularized' if speed<p.friction_velocity_epsilon else 'sliding',
                    group='blade' if np.any(nodes>=p.n) and np.any(nodes<p.n) else 'self'))
    records.sort(key=lambda r:(r['kind'],r['nodes']))
    active=p.active_contact.copy();active[p.seam[~p.released]]=False
    return dict(ipc_force_norm_n=float(np.sqrt(2)*np.linalg.norm(p.normal_response(x)[0])),
        contacts=records,strict_contacts=strict_contacts(p,x,p._diagnostic_config['precrack_mm']),
        mirror_active_nodes=np.flatnonzero(active).tolist())


def local_correction(p,x):
    lag=p.lag(x);_,g,H,_=p.evaluate(x,lag,projected=False)
    J=p.coupled_friction_tangent(x,lag,H)[p.free][:,p.free];r=g[p.free]
    delta=spsolve(J,-r);dv=np.zeros(p.total);dv[p.free]=delta/p.material_dt
    massnorm=float(np.sqrt(np.sum(p.mass_dof*dv**2)))
    return dict(scope='fixed local Jacobian and active set estimate; not a strict global upper bound',
        full_force_residual_n=float(np.sqrt(2)*np.linalg.norm(r)),
        linear_relative_residual=float(np.linalg.norm(J@delta+r)/max(np.linalg.norm(r),1e-300)),
        correction_mass_velocity_sqrt_nmm=massnorm,
        correction_max_velocity_mm_s=float(np.max(abs(dv))),
        free_residual_impulse_mass_norm=float(p.material_dt*np.sqrt(np.sum(r*r/p.mass_dof[p.free]))))


@runtime_entry
def run(case,audit,state_index,out,seconds=450,*,tolerances=(1e-7,1e-9),runtime=None):
    case=Path(case);out=Path(out);start=time.monotonic();deadline=start+seconds
    if not 0<seconds<=500:raise ValueError('Bounded local diagnosis only')
    q=json.loads((case/'result.json').read_text());a=json.loads(Path(audit).read_text());bind_case(case,q,a)
    if sources()!=q['source_hashes']:raise ValueError('Original equations must match diagnostic sources')
    row=q['steps'][state_index];s=read_state(case/row['state_file'])
    parentfile=case/(q['steps'][state_index-1]['state_file'] if state_index else 'parent_state.npz');parent=read_state(parentfile)
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(parentfile,out/'parent.npz');shutil.copy2(case/row['state_file'],out/'original.npz')
    report=dict(scope='same accepted parent, material history, physical dt and target; conditional roots only',
        source_hashes=sources(),case_result_sha256=digest(case/'result.json'),parent_sha256=digest(parentfile),
        config=q['config'],layer=q['material_layer'],dt_s=row['dt_s'],target_time_s=row['time_s'],target_depth_mm=row['depth_mm'],
        thread_runtime=runtime,roots=[],comparisons=[])
    p=problem(q['config'],q['material_layer'],q['solve_tolerance_n']);p._diagnostic_config=q['config']
    p.active_faces=s['active_faces'].copy();p.previous_active_faces=s['previous_active_faces'].copy();base.audit(p,s,row)
    report['original']=dict(correction=local_correction(p,s['positions']),signature=contact_signature(p,s['positions']))
    results=[]
    for tol in tolerances:
        if time.monotonic()>deadline-10:break
        p=problem(q['config'],q['material_layer'],tol);p._diagnostic_config=q['config'];x=restore_parent(p,parent)
        oldreleased=p.released.copy();before=array_hash(accepted_fields(p,x));begin=time.monotonic()
        try:
            y,rec=p.step(x,float(parent['depth_mm']),float(s['depth_mm']),min(deadline,time.monotonic()+100))
            state=route.snapshot(p,y,oldreleased,float(s['depth_mm']));name=f'root_{tol:.0e}.npz';np.savez_compressed(out/name,**state)
            n=p.n;dv=state['velocity'][:n]-s['velocity'][:n]
            report['roots'].append(dict(tolerance_n=tol,status='accepted',elapsed_s=time.monotonic()-begin,state_file=name,state_sha256=digest(out/name),
                record=rec,correction=local_correction(p,y),signature=contact_signature(p,y),
                original_mass_velocity_difference=float(np.sqrt(p.mass@np.sum(dv*dv,axis=1)))))
            results.append((tol,state,rec))
        except (RuntimeError,ValueError,TimeoutError,AssertionError) as exc:
            assert before==array_hash(accepted_fields(p,x))
            for field in ('released','active_contact','active_faces'):
                setattr(p,field,getattr(p,'failed_'+field,getattr(p,field)).copy())
            p.rebuild_contact();p.set_free()
            fields=trial_fields(p,getattr(p,'failed_positions',x));energy=None
            try:energy=p.energy_ledger(fields['positions'])
            except (RuntimeError,ValueError,FloatingPointError):pass
            name=f'rejected_{tol:.0e}.npz';np.savez_compressed(out/name,**fields)
            report['roots'].append(dict(tolerance_n=tol,status='rejected',error=str(exc),rollback_verified=True,elapsed_s=time.monotonic()-begin,
                state_file=name,state_sha256=digest(out/name),energy=energy))
        write_json(out/'result.json',native(report))
    for (t0,s0,r0),(t1,s1,r1) in zip(results[:-1],results[1:]):
        n=len(s1['mass']);dv=s1['velocity'][:n]-s0['velocity'][:n]
        comp=dict(tolerances_n=[t0,t1],mass_velocity_difference=float(np.sqrt(s1['mass']@np.sum(dv*dv,axis=1))),
            max_velocity_difference_mm_s=float(np.max(np.linalg.norm(dv,axis=1))),fz_difference_n=r1['reaction_fz_n']-r0['reaction_fz_n'])
        for name in ('normal','friction'):
            comp[name+'_impulse_difference_n_s']=float(np.sqrt(2)*np.linalg.norm(s1['impulse_'+name]-s0['impulse_'+name]))
        report['comparisons'].append(comp)
    report.update(elapsed_s=time.monotonic()-start,source_unchanged=sources()==report['source_hashes'])
    write_json(out/'result.json',native(report));return report


@runtime_entry
def scan(case,audit,out,seconds=450,*,runtime=None):
    case=Path(case);out=Path(out);start=time.monotonic()
    q=json.loads((case/'result.json').read_text());a=json.loads(Path(audit).read_text());bind_case(case,q,a)
    if sources()!=q['source_hashes']:raise ValueError('Original equation identity mismatch')
    p=problem(q['config'],q['material_layer'],q['solve_tolerance_n']);p._diagnostic_config=q['config'];records=[]
    for i,row in enumerate(q['steps']):
        if time.monotonic()-start>seconds:raise TimeoutError('Bounded transition scan')
        s=read_state(case/row['state_file']);p.active_faces=s['active_faces'].copy();p.previous_active_faces=s['previous_active_faces'].copy();base.audit(p,s,row)
        records.append(dict(index=i,time_s=a['states'][i]['time_s'],depth_mm=row['depth_mm'],state_sha256=digest(case/row['state_file']),
            signature=contact_signature(p,s['positions']),correction=local_correction(p,s['positions']) if 1.890<=row['time_s']<=1.892 else None))
    result=dict(result_sha256=digest(case/'result.json'),source_hashes=sources(),thread_runtime=runtime,records=records,elapsed_s=time.monotonic()-start,
        definitions=dict(ipc='nonzero native barrier gradient (not hard impact)',strict='native interior face identity excluding tip/top/corners',
            friction='native tangent speed divided by velocity regularization epsilon',mirror='active unilateral seam constraints excluding intact ties'))
    write_json(out,native(result));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--audit',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--state-index',type=int);p.add_argument('--seconds',type=float,default=450);add_thread_arguments(p);a=p.parse_args()
    kw=thread_arguments(a)
    q=run(a.case,a.audit,a.state_index,a.out,a.seconds,**kw) if a.state_index is not None else scan(a.case,a.audit,a.out,a.seconds,**kw)
    print(json.dumps({k:v for k,v in q.items() if k in ('elapsed_s','comparisons','source_unchanged')}))
if __name__=='__main__':main()
