"""Fixed-step conditional FE window; no history splice into a production path."""
import argparse,csv,json,shutil,sys,time
from pathlib import Path
import numpy as np
from coupled_contact import problem
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from . import run_precrack_entry as parent_route
from . import run_fracture_route as route
from . import run_zero_gap_cut as driver
from .face_witness import native,accepted_fields,array_hash,trial_fields,rejection_measurement
from .frozen_scene import digest
from run_support import write_json
from step_control import failure_kind
from release_dynamics import DEFECT_KEYS


def sources():
    s=parent_route.sources();root=Path(__file__).resolve().parents[1]
    for name in ('coupled_contact.py','tools/run_coupled_contact.py'):
        s[name]=digest(root/name)
    return s


def restore_parent(p,s):
    for key in ('reference_xy','mass','triangles'):
        np.testing.assert_array_equal(s[key],{'reference_xy':p.rest,'mass':p.mass,'triangles':p.mesh.triangles}[key])
    np.testing.assert_array_equal(s['active_faces'],p.precracked_faces)
    np.testing.assert_array_equal(s['history'],p.history)
    for key in ('velocity','material_metric','history','released','active_contact','active_faces'):
        setattr(p,key,s[key].copy())
    p.rebuild_contact();p.set_free();p.previous=s['positions'].copy()
    p.previous_velocity=p.velocity.copy();p.previous_history=p.history.copy();p.previous_material_metric=p.material_metric.copy()
    p.previous_active_faces=p.active_faces.copy();p._accepted_predictor_history=[]
    return s['positions'].copy()


def rejected_fields(p,failed):
    """Serialize the actual trial graph after step() has rolled the object back."""
    saved={k:getattr(p,k).copy() for k in ('released','active_contact','active_faces')}
    try:
        for k in saved:setattr(p,k,getattr(p,'failed_'+k,getattr(p,k)).copy())
        p.rebuild_contact();p.set_free()
        fields=trial_fields(p,failed);energy=None
        try:energy=p.energy_ledger(failed)
        except (RuntimeError,ValueError,FloatingPointError):pass
        return fields,energy
    finally:
        for k,v in saved.items():setattr(p,k,v)
        p.rebuild_contact();p.set_free()


def finish(out,result):
    rows=result['steps'];energies=[r['energy'] for r in rows]
    sums={k:sum(e[k] for e in energies) for k in energies[0]} if energies else {}
    for k in ('mechanical_reservoir_n_mm','kinetic_energy_n_mm'):sums.pop(k,None)
    result['summary']=dict(accepted_states=len(rows),last_depth_mm=rows[-1]['depth_mm'] if rows else result['initial_depth_mm'],
        completed_window=result['status']=='completed_window',completed_2mm=False,
        blade_entered=any(r['blade_entered_surface_channel'] for r in rows),new_activation=False,new_connected_crack_growth=False,
        independent_entry_verified=False,event_complete=False,time_converged=False,mesh_converged=False,physical_prediction_validated=False,
        ledger_n_mm=sums,absolute_numerical_work_n_mm=sum(sum(abs(e[k]) for k in (*DEFECT_KEYS,'friction_time_defect_n_mm')) for e in energies),
        max_residual_n=max((r['residual_n'] for r in rows),default=None),
        max_speed_mm_s=max((r['maximum_speed_mm_s'] for r in rows),default=None),
        terminal_kinetic_energy_n_mm=energies[-1]['kinetic_energy_n_mm'] if rows else None)
    write_json(out/'result.json',native(result))
    if rows:
        keys=['depth_mm','time_s','dt_s','reaction_fz_n','reaction_normal_fz_n','reaction_friction_fz_n','maximum_speed_mm_s','mass_velocity_norm_sqrt_nmm','blade_entered_surface_channel','blade_normal_impulse_n_s','blade_friction_impulse_n_s']
        with (out/'curves.csv').open('w') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
        with (out/'energy.csv').open('w') as f:
            w=csv.DictWriter(f,['time_s',*energies[0]]);w.writeheader();w.writerows(dict(time_s=r['time_s'],**r['energy']) for r in rows)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Coupled IPC conditional window</h1><p>Conditional integrator/material comparison from an audited parent, not a production suffix.</p><pre>'+json.dumps(result['summary'],indent=2)+'</pre><a href="result.json">Raw evidence</a>')


@runtime_entry
def run(parent,parent_audit,state_index,preregister,out,dt,layer,seconds=600,*,runtime=None):
    parent=Path(parent);out=Path(out);audit=Path(parent_audit);pre=Path(preregister)
    if not 0<seconds<=2400:raise ValueError('Loading budget <=2400 s')
    limits=json.loads(pre.read_text())['coupled_window']
    if dt not in limits['dt_s'] or state_index!=limits['parent_state_index']:raise ValueError('Outside preregistered window')
    raw=json.loads((parent/'result.json').read_text());proof=json.loads(audit.read_text())
    if not proof['passed'] or proof['result_sha256']!=digest(parent/'result.json') or proof['source_hashes']!=raw['source_hashes']:raise ValueError('Parent source/audit identity mismatch')
    row=raw['steps'][state_index];file=parent/row['state_file']
    if digest(file)!=row['state_sha256']:raise ValueError('Parent state identity changed')
    with np.load(file,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(file,out/'parent_state.npz')
    shutil.copy2(audit,out/'parent_audit.json');shutil.copy2(pre,out/'preregistration.json')
    write_json(out/'parent_record.json',row)
    solve_tolerance=limits.get('solve_tolerance_n',1e-4)
    p=problem(raw['config'],layer,solve_tolerance);x=restore_parent(p,s);depth=float(row['depth_mm']);t0=float(row['time_s'])
    source=sources();start=time.monotonic();deadline=start+seconds-5
    result=dict(schema='softcut-coupled-contact-window/1',config=raw['config'],source_hashes=source,source_unchanged=False,
        interpreter=sys.executable,execution_identity=driver.execution_identity(),thread_runtime=runtime,
        conditional_comparison=True,production_continuation=False,material_layer=layer,solve_tolerance_n=solve_tolerance,physical_acceptance_n=1e-4,
        parent=dict(directory=str(parent.resolve()),state_file=row['state_file'],state_sha256=digest(file),result_sha256=digest(parent/'result.json'),source_hashes=raw['source_hashes'],audit_sha256=digest(audit)),
        preregistration_sha256=digest(pre),initial_depth_mm=depth,initial_time_s=t0,dt_requested_s=dt,window_s=limits['window_s'],
        initial_barrier_energy_n_mm=2*float(p.barrier(p.collisions(x),p.cmesh,x[p.surface])),contact_conversion_work_n_mm=0.,
        contact_model='unchanged native IPC including initial barrier energy; exact mirror constraint; no added hard blade impulse',
        steps=[],attempts=[],status='running')
    count=int(round(limits['window_s']/dt))
    if abs(count*dt-limits['window_s'])>1e-12:raise ValueError('Window must contain whole steps')
    for i in range(count):
        if time.monotonic()>deadline:result['status']='global_compute_budget_exhausted';break
        target=result['initial_depth_mm']+p.loading_rate*(i+1)*dt
        before=array_hash(accepted_fields(p,x));oldreleased=p.released.copy()
        p.global_compute_deadline=deadline
        try:
            y,rec=p.step(x,depth,target,min(deadline,time.monotonic()+raw['config']['base']['attempt_seconds']))
            x=y;depth=target
            rec.update(time_s=t0+(i+1)*dt,dt_s=p.material_dt)
            route.observables(p,x,rec)
            state=route.snapshot(p,x,oldreleased,depth);name=f'state_{i:05d}.npz';np.savez_compressed(out/name,**state)
            rec.update(state_file=name,state_sha256=digest(out/name));result['steps'].append(native(rec))
            write_json(out/'progress.json',dict(accepted=len(result['steps']),depth_mm=depth,time_s=rec['time_s']))
        except (RuntimeError,ValueError,TimeoutError,AssertionError,FloatingPointError) as exc:
            if array_hash(accepted_fields(p,x))!=before:raise RuntimeError('Failed coupled step committed history') from exc
            failed=getattr(p,'failed_positions',x);fields,energy=rejected_fields(p,failed)
            np.savez_compressed(out/'rejected_trial.npz',**fields)
            if array_hash(accepted_fields(p,x))!=before:raise RuntimeError('Rejection serialization changed accepted state')
            result['status']='rejected_fixed_step'
            result['attempts'].append(dict(from_mm=depth,target_mm=target,dt_s=p.material_dt,error=str(exc),failure_kind=failure_kind(exc,deadline),
                rollback_verified=True,parent_hash=before,parent_state_file=result['steps'][-1]['state_file'] if result['steps'] else 'parent_state.npz',rejection_measurement=rejection_measurement(exc,energy,p),trial_file='rejected_trial.npz',trial_sha256=digest(out/'rejected_trial.npz'),energy=energy,
                nonlinear_summary=getattr(p,'force_globalization_summary',{})))
            break
    else:result['status']='completed_window'
    result.update(elapsed_s=time.monotonic()-start,source_unchanged=source==sources())
    finish(out,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('--parent-audit',required=True,type=Path);p.add_argument('--state-index',type=int,required=True);p.add_argument('--preregister',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--dt',required=True,type=float);p.add_argument('--layer',choices=['frozen_metric','maxwell'],default='frozen_metric');p.add_argument('--seconds',type=float,default=600);add_thread_arguments(p);a=p.parse_args()
    q=run(a.parent,a.parent_audit,a.state_index,a.preregister,a.out,a.dt,a.layer,a.seconds,**thread_arguments(a));print(json.dumps(q['summary']))
if __name__=='__main__':main()
