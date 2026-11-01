"""Own-initial-state continuous face growth on the coupled BE/IPC equation."""
import argparse,json,time
from pathlib import Path
import numpy as np
from coupled_cohesive import problem as core_problem
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from . import run_fracture_route as route
from .run_coupled_contact import sources as window_sources
from .audit_coupled_contact import independent_balance
from .audit_precrack_entry import strict_contacts,check_entry_flag
from .audit_face_witness import inspect_one
from .coupled_audit_contract import close
from .frozen_scene import digest
from .face_witness import native
from .face_activation_chain import transition
from run_support import write_json
BASE_FINISH=route.finish


def problem(config):
    c=config.copy();control=c.pop('coupled_control')
    if set(control)-{'material_layer','solve_tolerance_n','depth_step_caps','maximum_consecutive_rejections','nonlinear_error_policy'}:raise ValueError('Unknown coupled controls')
    p=core_problem(c,control['material_layer'],control['solve_tolerance_n'])
    if 'nonlinear_error_policy' in control:
        from nonlinear_accuracy import validate_policy
        p.nonlinear_error_policy=validate_policy(control['nonlinear_error_policy'])
    caps=control.get('depth_step_caps',[])
    if caps:
        ends=[v['through_mm'] for v in caps]
        if any(b<=a for a,b in zip([p.start,*ends[:-1]],ends)) or ends[-1]<c['base']['target_mm'] or any(v['max_dt_s']<=0 for v in caps):raise ValueError('Invalid prescribed depth caps')
        def next_depth(depth,increment,target):
            cap=next(v for v in caps if depth<v['through_mm']-1e-12)
            return min(target,depth+increment,cap['through_mm'],depth+p.loading_rate*cap['max_dt_s'])
        p.next_loading_depth=next_depth
    p.maximum_consecutive_rejections=control.get('maximum_consecutive_rejections',20)
    if not 1<=p.maximum_consecutive_rejections<=20:raise ValueError('Bounded local refinement cap required')
    return p


def sources():
    q=window_sources();root=Path(__file__).resolve().parents[1]
    for name in ('coupled_cohesive.py','nonlinear_accuracy.py','tools/run_coupled_growth.py'):q[name]=digest(root/name)
    return q


def finish(out,result):
    BASE_FINISH(out,result)
    result.update(solver_model='nonlinear_coupled_BE_IPC_continuous_P1_face',conditional_comparison=False,
        own_initial_state=True,old_production_continuation=False,diagnostic_geometry=result['config']['initial_slit_halfwidth_mm']>0,
        time_schedule=dict(kind='recorded_physical_cutbacks',maximum_dt_s=result['config']['base']['increment_mm']/.2,
            provenance='Existing deterministic physical nonlinear/energy cutback controller; no cutback for compute/virtual failures',
            end_times_s=[r['time_s'] for r in result['steps']]),
        root_and_handoff_gates_distinct=True)
    write_json(Path(out)/'result.json',native(result))


def install():route.problem=problem;route.sources=sources;route.finish=finish


@runtime_entry
def audit(directory,out,seconds=500,*,runtime=None):
    directory=Path(directory);out=Path(out);start=time.monotonic();deadline=start+seconds
    if not 0<seconds<=540 or out.exists():raise ValueError('New bounded independent audit required')
    raw=json.loads((directory/'result.json').read_text());source=sources()
    report=dict(schema='softcut-coupled-growth-audit/1',passed=False,result_sha256=digest(directory/'result.json'),source_hashes=source,
        auditor_sha256=digest(Path(__file__)),thread_runtime=runtime,states=[],witnesses=[])
    try:
        if raw['source_hashes']!=source or not raw['source_unchanged']:raise ValueError('Source mismatch')
        p=problem(raw['config'])
        previous=dict(positions=p.rest.copy(),velocity=p.velocity.copy(),material_metric=p.material_metric.copy(),history=p.history.copy(),
            released=p.released.copy(),active_contact=p.active_contact.copy(),active_faces=p.active_faces.copy(),depth_mm=p.start)
        with np.load(directory/'initial_state.npz',allow_pickle=False) as z:
            for key in z.files:np.testing.assert_array_equal(z[key],previous[key])
        parent_ref=dict(file='initial_state.npz',sha256=digest(directory/'initial_state.npz'))
        previous_time=0.;entry_previous=dict(time_s=0.,depth_mm=p.start,state_file='initial_state.npz')
        for row in raw['steps']:
            if time.monotonic()>deadline:raise TimeoutError('Independent growth audit budget')
            file=directory/row['state_file']
            if file.resolve().parent!=directory.resolve() or digest(file)!=row['state_sha256']:raise ValueError('State identity mismatch')
            with np.load(file,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
            for k in ('positions','velocity','material_metric','history','released','active_faces'):
                np.testing.assert_array_equal(s['previous_'+k],previous[k])
            np.testing.assert_array_equal(s['mass'],p.mass)
            h=float(s['material_dt_s']);depth=float(s['depth_mm']);t=(depth-p.start)/p.loading_rate
            close(h,(depth-previous['depth_mm'])/p.loading_rate,'Independent depth/time')
            close(row['dt_s'],h,'Production dt');close(row['time_s'],t,'Production time');close(t-previous_time,h,'Clock chain')
            if h<=0 or h>raw['config']['base']['increment_mm']/p.loading_rate+1e-12:raise ValueError('Invalid physical cutback schedule')
            if hasattr(p,'next_loading_depth') and depth>p.next_loading_depth(float(previous['depth_mm']),raw['config']['base']['increment_mm'],raw['requested_stop_depth_mm'])+1e-12:raise ValueError('Declared local time cap violated')
            activation_chain=transition(directory,previous,s,row,parent_ref,float(previous['depth_mm']),raw['config'],source,p.seam)
            fresh=route.restore_audit(p,s,row)
            if not fresh['accepted_chord_ccd_safe']:raise ValueError('Full trajectory unsafe')
            balance=independent_balance(p,s)
            failed=s['active_faces']&np.all(s['history']>=2*p.toughness/p.strength,axis=1)
            length=p.connected_length(failed);contacts=strict_contacts(p,s['positions'],length)
            check_entry_flag(contacts,row['blade_entered_surface_channel'])
            now=dict(time_s=t,depth_mm=depth,state_file=row['state_file'])
            if contacts and 'first_entry_bracket' not in report:report['first_entry_bracket']=dict(previous_nonentry=entry_previous,first_entry=now)
            if not contacts:entry_previous=now
            if np.any(s['active_faces']&~previous['active_faces']) and 'first_activation_bracket' not in report:
                report['first_activation_bracket']=dict(previous_time_s=previous_time,time_s=t,depth_mm=depth,new_faces=np.flatnonzero(s['active_faces']&~previous['active_faces']).tolist())
            if np.any(s['history'][~p.precracked_faces]>previous['history'][~p.precracked_faces]) and 'first_new_damage' not in report:report['first_new_damage']=now
            report['states'].append(dict(**now,dt_s=h,state_sha256=digest(file),min_j=fresh['min_j'],strict_contacts=contacts,
                activation_chain=activation_chain,connected_length_mm=length,active_faces=int(s['active_faces'].sum()),**balance))
            previous=s;previous_time=t;parent_ref=dict(file=row['state_file'],sha256=row['state_sha256'])
        complete=abs(previous['depth_mm']-raw['requested_stop_depth_mm'])<=1e-12
        if complete!=(raw['status'] in ('completed_pilot','completed_2mm')):raise ValueError('False completion label')
        refs=[ref for row in raw['steps'] for ref in row.get('activation_witnesses',[])]
        index=directory/'witnesses/representatives.json'
        if index.exists():
            for v in json.loads(index.read_text())['signatures'].values():refs.extend(v[k] for k in ('first','worst','latest') if v.get(k))
        for ref in {r['sha256']:r for r in refs}.values():report['witnesses'].append(inspect_one(directory,ref,source,deadline,factory=problem))
        report.update(passed=bool(report['states']),complete_requested_path=complete,terminal_depth_mm=float(previous['depth_mm']),
            terminal_time_s=previous_time,initial_state_sha256=digest(directory/'initial_state.npz'),
            new_fracture_dissipation_n_mm=sum(s['independent_fracture_n_mm'] for s in report['states']),
            absolute_numerical_work_n_mm=sum(s['absolute_numerical_including_equation_work_n_mm'] for s in report['states']),
            time_converged=False,mesh_converged=False,complete_event_verified=False)
    except (RuntimeError,ValueError,AssertionError,OSError,TimeoutError,KeyError) as exc:report['error']=str(exc)
    report.update(elapsed_s=time.monotonic()-start,source_unchanged=source==sources());report['passed'] &= report['source_unchanged']
    write_json(out,native(report));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','audit']);p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--stop-depth',type=float);add_thread_arguments(p);a=p.parse_args()
    if a.mode=='run':
        if not 30<=a.seconds<=2400:raise ValueError('Loading internal limit <=2400 s')
        install();q=route.run(json.loads(a.input.read_text()),a.out,a.seconds,stop_depth=a.stop_depth,**thread_arguments(a))
    else:q=audit(a.input,a.out,a.seconds,**thread_arguments(a))
    print(json.dumps(q.get('summary',{k:q.get(k) for k in ('passed','error','terminal_depth_mm','elapsed_s')})))
if __name__=='__main__':main()
