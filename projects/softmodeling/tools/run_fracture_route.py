"""Own-initial-state face/nodal fracture-route controls and independent audits.

No archived history transfer, experimental force input, or automatic queue.
The intact FFM remains in run_zero_gap_cut; this continuous cohesive route has
no additional virtual energy gate and no upfront Gc debit.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from run_support import write_json
from . import run_zero_gap_cut as driver
from .frozen_scene import digest
from .audit_zero_gap_cut import blade_contacts
from .audit_contact_time import integrate
from face_cut import FaceCutProblem,eligible_nodes
from release_dynamics import check_energy_gate,DEFECT_KEYS,contact_velocity,instantaneous_friction
from prototype.ipc_reference import LoadStepFailure,IPCProblem
from step_control import execute_step,failure_kind,physical_cutback,reduced_increment
from endpoint_friction import velocity_increment


def sources():
    result=driver.hashes();root=Path(__file__).resolve().parents[1]
    for name in ('face_cut.py','face_cohesive_quadrature.py','release_dynamics.py','endpoint_friction.py',
                 'stiff_contact.py','tools/run_fracture_route.py'):
        result[name]=digest(root/name)
    return result


def problem(config):
    if set(config)!={'schema','base','interface','precrack_mm','quadrature_order','friction_time'} or config['schema']!='softcut-fracture-route/1':
        raise ValueError('Explicit route configuration required')
    if config['base'].get('fracture_criterion')!='local_or_neighbourhood' or config['base']['insertion_rule']!='local_strength':
        raise ValueError('Continuous route cannot silently combine FFM or a second threshold')
    p=driver.problem(config['base'])
    if config['interface']=='face':
        # Factory preserves all physical/geometry arguments of the original
        # EnergyCutProblem. Change only the explicitly selected interface class.
        p.__class__=FaceCutProblem
        p.initialize_faces(config['quadrature_order'],config['precrack_mm'],config['friction_time'])
    elif config['interface']=='node':
        if not 0<=config['precrack_mm']<p.config.height_mm:raise ValueError('Invalid precrack')
        y=p.rest[p.seam,1];tip=p.config.height_mm-config['precrack_mm']
        if np.min(abs(y-tip))>1e-10:raise ValueError('Precrack tip must be a seam vertex')
        p.released=y>tip+1e-10
        p.history[p.released]=2*p.toughness/p.strength;p.previous_history=p.history.copy()
        p.rebuild_contact();p.set_free()
        if config['friction_time']=='endpoint_friction':
            from endpoint_friction import install
            install(p)
        elif config['friction_time']!='midpoint':raise ValueError('Unknown friction time rule')
    else:raise ValueError('Choose explicit face or node interface')
    return p


def snapshot(p,x,old_released,depth):
    state=driver.snapshot(p,x,old_released,depth)
    fields,_=momentum(p,x)
    state.update(fields)
    if isinstance(p,FaceCutProblem):
        state.update(active_faces=p.active_faces,previous_active_faces=p.previous_active_faces,
            precracked_faces=p.precracked_faces,face_weights=p.rule['weights_mm2'],face_shape=p.rule['shape'])
    return state


def momentum(p,x):
    """Independent assembly of impulse balance, including exact reactions."""
    if not isinstance(p,FaceCutProblem):
        from stiff_contact import momentum_ledger
        return momentum_ledger(p,x)
    h=p.material_dt;du=p.step_increment(x);shape=x.shape
    r,_,dofs,_,_=p.material_bulk((x[:p.n]-p.mesh.reference_xy).ravel(),with_tangent=False)
    bulk=np.zeros(p.total);np.add.at(bulk,dofs.ravel(),r.ravel())
    coh=np.zeros(p.total);coh[2*p.seam]=p.face_force(p.cohesive_algorithmic(x))[0]
    normal=np.zeros(p.total);normal[p.surface_dofs]=p.normal_response(x)[0]
    friction=np.zeros(p.total);fu=velocity_increment(p,du) if p.friction_time=='endpoint_friction' else du
    lag=p.lag(x)
    if lag is not None:friction[p.surface_dofs]=p.friction.gradient(lag,p.cmesh,fu[p.surface])
    delta=p.mass_dof*(p.endpoint_velocity(x)-p.previous_velocity).ravel()
    residual=delta+h*(bulk+coh+normal+friction)
    constrained=np.setdiff1d(np.arange(p.total),p.free);reaction=np.zeros(p.total);reaction[constrained]=residual[constrained]
    defect=residual-reaction
    norm=float(np.sqrt(2)*np.linalg.norm(defect[p.free])/h)
    if norm>1e-4:raise RuntimeError('Independent face impulse balance exceeds original force gate')
    fields={f'impulse_{name}':(-h*f).reshape(shape) for name,f in [('bulk',bulk),('cohesive',coh),('normal',normal),('friction',friction)]}
    fields.update(impulse_constraint=reaction.reshape(shape),momentum_change=delta.reshape(shape),impulse_balance_defect=defect.reshape(shape))
    return fields,dict(equivalent_force_residual_n=norm,scope='half-domain vectors; full reflected residual norm')


def restore_audit(p,state,record):
    if isinstance(p,FaceCutProblem):
        np.testing.assert_array_equal(state['precracked_faces'],p.precracked_faces)
        np.testing.assert_array_equal(state['face_weights'],p.rule['weights_mm2'])
        np.testing.assert_array_equal(state['face_shape'],p.rule['shape'])
        if np.any(state['previous_active_faces']&~state['active_faces']):raise ValueError('Face healing')
        p.active_faces=state['active_faces'].copy();p.previous_active_faces=state['previous_active_faces'].copy();p.face_events=record.get('face_events',[])
        np.testing.assert_array_equal(state['released'],eligible_nodes(p.active_faces))
    result=driver.audit(p,state,record)
    check_energy_gate(p,record['energy'])
    fields,result['momentum']=momentum(p,state['positions'])
    for key,value in fields.items():np.testing.assert_allclose(value,state[key],rtol=1e-10,atol=1e-14)
    return result


def observables(p,x,rec):
    contacts=blade_contacts(p,x)
    if isinstance(p,FaceCutProblem):
        q=p.cohesive_response(x,p.history)
        failed=p.active_faces&np.all(q['history']>=2*p.toughness/p.strength,axis=1)
        length=p.connected_length(failed)
    else:
        from .report_finite_increment import connected_length
        failed=p.history>=2*p.toughness/p.strength
        length=connected_length(failed,p.rest[p.seam,1])
    interior=[]
    for c in contacts:
        y=p.rest[c['body_nodes'],1]
        # Exclude the original top and crack tip/corner. Contact identity alone
        # at either endpoint cannot certify entry into an opened channel.
        if c['body_reference_group']=='seam' and np.all(y<p.config.height_mm-1e-9) and np.all(y>p.config.height_mm-length+1e-9) and np.all(x[c['body_nodes'],0]>p.config.drop_s_mm+1e-10):
            interior.append(c)
    vel=contact_velocity(p,p.velocity);gf=instantaneous_friction(p,x,vel).reshape(-1,2)
    gn=p.barrier.gradient(p.collisions(x),p.cmesh,x[p.surface]).reshape(-1,2)
    rec.update(surface_connected_failed_length_mm=float(length),blade_contact_objects=contacts,
        blade_entered_surface_channel=bool(interior),blade_entry_contacts=interior,
        instantaneous_normal_fz_n=-2*float(gn[p.surface>=p.n,1].sum()),
        instantaneous_friction_fz_n=-2*float(gf[p.n:,1].sum()),
        maximum_speed_mm_s=float(np.linalg.norm(p.velocity[:p.n],axis=1).max()),
        maximum_speed_node=int(np.argmax(np.linalg.norm(p.velocity[:p.n],axis=1))),
        mass_velocity_norm_sqrt_nmm=float(np.sqrt(p.mass@np.sum(p.velocity[:p.n]**2,axis=1))))
    rec['instantaneous_fz_n']=rec['instantaneous_normal_fz_n']+rec['instantaneous_friction_fz_n']
    du=p.step_increment(x);fu=velocity_increment(p,du) if getattr(p,'friction_time',getattr(p,'conditional_integrator','midpoint'))=='endpoint_friction' else du
    lag=p.lag(x);gfr=np.zeros(2*len(p.surface)) if lag is None else p.friction.gradient(lag,p.cmesh,fu[p.surface])
    rec['blade_normal_impulse_n_s']=p.material_dt*rec['reaction_normal_fz_n']
    rec['blade_friction_impulse_n_s']=-2*p.material_dt*float(gfr.reshape(-1,2)[p.surface>=p.n,1].sum())


def finish(out,result):
    rows=result['steps'];energies=[r['energy'] for r in rows]
    result['summary']=dict(last_depth_mm=rows[-1]['depth_mm'] if rows else -.02,accepted_states=len(rows),
        completed_2mm=bool(rows) and rows[-1]['depth_mm']>=2-1e-12,
        surface_connected=any(r['surface_connected_failed_length_mm']>0 for r in rows),
        blade_entered=any(r['blade_entered_surface_channel'] for r in rows),
        initiation_predicted=bool(rows) and result['config']['precrack_mm']==0 and any(r['released_nodes'] for r in rows),
        event_complete=False,energy_converged=False,time_converged=False,mesh_converged=False,independent_material_parameters=False,
        ledger_n_mm={k:sum(e[k] for e in energies) for k in energies[0]} if energies else {},
        absolute_numerical_work_n_mm=sum(sum(abs(e.get(k,0.)) for k in (*DEFECT_KEYS,'friction_time_defect_n_mm')) for e in energies),
        max_residual_n=max((r['residual_n'] for r in rows),default=None),
        actual_dt_range_s=[min((r['dt_s'] for r in rows),default=None),max((r['dt_s'] for r in rows),default=None)])
    write_json(out/'result.json',result)
    if rows:
        keys=['depth_mm','time_s','dt_s','reaction_fz_n','reaction_normal_fz_n','reaction_friction_fz_n','instantaneous_fz_n','instantaneous_normal_fz_n','instantaneous_friction_fz_n','maximum_speed_mm_s','mass_velocity_norm_sqrt_nmm','surface_connected_failed_length_mm','blade_entered_surface_channel','blade_normal_impulse_n_s','blade_friction_impulse_n_s']
        with (out/'curves.csv').open('w') as f:
            writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in rows)
        with (out/'energy.csv').open('w') as f:
            writer=csv.DictWriter(f,['depth_mm',*energies[0]]);writer.writeheader();writer.writerows(dict(depth_mm=r['depth_mm'],**r['energy']) for r in rows)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fracture route control</title><h1>Own-initial-state actual geometry control</h1><p>Precrack is a diagnostic input, not intact-material initiation. No temporal or mesh qualification is inferred from step acceptance.</p><pre>'+json.dumps(result['summary'],indent=2)+'</pre><a href="result.json">Full evidence</a>')


@runtime_entry
def run(config,out,seconds,*,stop_depth=None,runtime=None):
    if not 30<=seconds<=3000:raise ValueError('Loading internal budget must be 30..3000 s')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+seconds-10;source=sources();base=config['base']
    target=base['target_mm'] if stop_depth is None else stop_depth
    if not 0<target<=base['target_mm']<=2:raise ValueError('Invalid actual travel')
    p=problem(config);x=p.rest.copy();depth=p.start;increment=base['increment_mm']
    result=dict(schema='softcut-fracture-route-result/1',config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),source_hashes=source,
        execution_identity=driver.execution_identity(),interpreter=sys.executable,thread_runtime=runtime,seconds=seconds,requested_stop_depth_mm=target,
        status='running',steps=[],attempts=[],no_experimental_force_input=True,
        initial=dict(full_mass_n_s2_mm=float(2*p.mass.sum()),minimum_node_mass_n_s2_mm=float(p.mass.min()),geometry='30x5x20 mm centred s=15 mm mirror mode I',
            precrack_mm=config['precrack_mm'],preexisting_gc_excluded_n_mm=getattr(p,'initial_fracture_energy_n_mm',float(p.toughness*p.seam_area[p.released].sum()))))
    write_json(out/'manifest.json',result)
    try:
        while depth<target-1e-12:
            if time.monotonic()>=deadline:raise TimeoutError('Global route loading budget')
            next_depth=min(target,depth+increment);old=p.released.copy();before=time.monotonic()
            try:value,rec=execute_step(p,x,depth,next_depth,min(deadline,before+base['attempt_seconds']),deadline)
            except (LoadStepFailure,TimeoutError) as exc:
                kind=failure_kind(exc,deadline)
                result['attempts'].append(dict(from_mm=depth,target_mm=next_depth,dt_s=(next_depth-depth)/p.loading_rate,failure_kind=kind,error=str(exc),elapsed_s=time.monotonic()-before))
                if not physical_cutback(kind) or next_depth-depth<=base['minimum_increment_mm']+4*np.spacing(max(1.,abs(depth))):raise
                increment=reduced_increment(p,base,next_depth-depth,kind);continue
            x,depth=value,next_depth
            rec.update(time_s=(depth-p.start)/p.loading_rate,dt_s=p.material_dt)
            observables(p,x,rec)
            state=snapshot(p,x,old,depth);name=f'state_{len(result["steps"]):05d}.npz';np.savez_compressed(out/name,**state)
            rec.update(state_file=name,state_sha256=digest(out/name));result['steps'].append(rec)
            write_json(out/'progress.json',dict(depth_mm=depth,accepted=len(result['steps']),surface_connected_mm=rec['surface_connected_failed_length_mm'],blade_entered=rec['blade_entered_surface_channel']))
            print('ACCEPTED '+json.dumps({k:rec[k] for k in ('depth_mm','reaction_fz_n','residual_n','released_nodes')}),flush=True)
            increment=max(base['minimum_increment_mm'],min(base['increment_mm'],increment*driver.accepted_step_growth(base,rec)))
        result['status']='completed_2mm' if target==2 else 'completed_pilot'
    except (LoadStepFailure,TimeoutError,RuntimeError,ValueError,FloatingPointError) as exc:
        result.update(status='stopped',error=str(exc),failure_kind=('interface_topology_handoff_unresolved' if 'Unfunded face-contact' in str(exc) else failure_kind(exc,deadline)),next_increment_mm=increment)
        if hasattr(p,'failed_positions'):
            fields=dict(positions=p.failed_positions,increment=p.failed_increment,previous_positions=p.previous,previous_velocity=p.previous_velocity,
                previous_history=p.previous_history,previous_material_metric=p.previous_material_metric,released=p.failed_released,active_contact=p.failed_active_contact,
                material_dt_s=p.material_dt,target_depth_mm=next_depth,failed_stage=p.failed_stage)
            if isinstance(p,FaceCutProblem):fields.update(active_faces=p.failed_active_faces,previous_active_faces=p.previous_active_faces)
            np.savez_compressed(out/'failed_trial.npz',**fields)
    result.update(source_unchanged=source==sources(),elapsed_s=time.monotonic()-start)
    finish(out,result);return result


@runtime_entry
def audit(directory,out,seconds,*,runtime=None):
    if not 1<=seconds<=540:raise ValueError('Independent audit internal budget <=540 s')
    directory=Path(directory);out=Path(out)
    if out.exists():raise FileExistsError('Preserve previous audit')
    start=time.monotonic();deadline=start+seconds;raw=json.loads((directory/'result.json').read_text());source=sources()
    report=dict(schema='softcut-fracture-route-audit/1',result_sha256=digest(directory/'result.json'),source_hashes=source,thread_runtime=runtime,passed=False,audits=[])
    try:
        if raw['source_hashes']!=source or not raw['source_unchanged']:raise ValueError('Producer source differs')
        p=problem(raw['config']);previous=dict(positions=p.rest,velocity=p.velocity,material_metric=p.material_metric,history=p.history,released=p.released,depth_mm=p.start)
        if isinstance(p,FaceCutProblem):previous['active_faces']=p.active_faces.copy()
        for rec in raw['steps']:
            if time.monotonic()>=deadline:raise TimeoutError('Independent route audit budget')
            path=directory/rec['state_file']
            if path.resolve().parent!=directory.resolve() or digest(path)!=rec['state_sha256']:raise ValueError('State identity changed')
            with np.load(path,allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
            for a,b in [('previous_positions','positions'),('previous_velocity','velocity'),('previous_material_metric','material_metric'),('previous_history','history'),('previous_released','released')]:
                np.testing.assert_array_equal(state[a],previous[b])
            if isinstance(p,FaceCutProblem):np.testing.assert_array_equal(state['previous_active_faces'],previous['active_faces'])
            np.testing.assert_array_equal(state['mass'],p.mass)
            np.testing.assert_allclose(state['material_dt_s'],(state['depth_mm']-previous['depth_mm'])/p.loading_rate,rtol=0,atol=1e-12)
            fresh=restore_audit(p,state,rec)
            if not fresh['accepted_chord_ccd_safe']:raise ValueError('Accepted temporal CCD failed')
            fresh['independent_contact']=integrate(p,state,deadline)
            report['audits'].append(fresh);previous=state
        report['passed']=bool(raw['steps']) and all(a['independent_contact']['converged'] for a in report['audits'])
    except (RuntimeError,ValueError,AssertionError,OSError,TimeoutError) as exc:report['error']=str(exc)
    report.update(source_unchanged=source==sources(),elapsed_s=time.monotonic()-start)
    report['passed'] &= report['source_unchanged'];write_json(out,report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['run','audit']);parser.add_argument('input',type=Path)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--seconds',type=float,required=True);parser.add_argument('--stop-depth',type=float);add_thread_arguments(parser);a=parser.parse_args()
    if a.mode=='run':result=run(json.loads(a.input.read_text()),a.out,a.seconds,stop_depth=a.stop_depth,**thread_arguments(a))
    else:result=audit(a.input,a.out,a.seconds,**thread_arguments(a))
    print(json.dumps(result.get('summary',{k:result.get(k) for k in ('passed','error','elapsed_s')})),flush=True)

if __name__=='__main__':main()
