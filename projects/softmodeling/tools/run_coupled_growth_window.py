"""Conditional same-parent softening windows; never production continuations.

The immutable parent must come from this exact coupled-growth kernel/config.
Only the registered physical time partition varies. Observers retain activation
and rejection witnesses without changing the model's decisions or histories.
"""
import argparse,contextlib,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from . import run_coupled_contact as window
from . import run_coupled_growth as growth
from . import run_fracture_route as route
from .audit_coupled_contact import independent_balance
from .audit_face_witness import inspect_one
from .audit_precrack_entry import strict_contacts,check_entry_flag
from .coupled_audit_contract import TimeChain,input_identity
from .face_witness import WitnessStore,qualification,native
from .frozen_scene import digest
from face_cut import eligible_nodes
from step_control import failure_kind
from run_support import write_json


def sources():
    q=growth.sources();q['tools/run_coupled_growth_window.py']=digest(Path(__file__));return q


def candidate(config,layer,tolerance):
    control=config['coupled_control']
    if control['material_layer']!=layer or control['solve_tolerance_n']!=tolerance:
        raise ValueError('Conditional window cannot change parent material or numerical root control')
    return growth.problem(config)


def restore_parent(p,s):
    for key,value in [('reference_xy',p.rest),('mass',p.mass),('triangles',p.mesh.triangles),('precracked_faces',p.precracked_faces),('face_weights',p.rule['weights_mm2'])]:
        np.testing.assert_array_equal(s[key],value)
    np.testing.assert_array_equal(s['released'],eligible_nodes(s['active_faces']))
    if np.any(p.precracked_faces&~s['active_faces']) or np.any(s['history']<p.history):raise ValueError('Parent history inconsistent with its own initial state')
    np.testing.assert_allclose(s['positions'][p.n:],p.prescribed_blade_positions(float(s['depth_mm'])),rtol=0,atol=p.blade_boundary_roundoff_tolerance(float(s['depth_mm'])))
    for key in ('velocity','material_metric','history','released','active_contact','active_faces'):setattr(p,key,s[key].copy())
    p.previous=s['positions'].copy();p.previous_velocity=p.velocity.copy();p.previous_material_metric=p.material_metric.copy()
    p.previous_history=p.history.copy();p.previous_active_faces=p.active_faces.copy();p.face_events=[]
    p.rebuild_contact();p.set_free();p._accepted_predictor_history=[]
    return s['positions'].copy()


@contextlib.contextmanager
def selected_driver(out):
    """Process-local adapters; each worker constructs its own problem and store."""
    out=Path(out);saved={name:getattr(window,name) for name in ('problem','restore_parent','sources','finish')}
    context=dict(rejected=[],index=0)
    def factory(config,layer,tolerance):
        p=candidate(config,layer,tolerance)
        store=WitnessStore(out,dict(config=config,source_hashes=sources(),execution_identity=window.driver.execution_identity()))
        step=p.step
        def observed(x,old,depth,deadline):
            parent=out/('parent_state.npz' if context['index']==0 else f'state_{context["index"]-1:05d}.npz')
            store.begin(p,x,dict(parent=dict(file=parent.name,sha256=digest(parent)),from_depth_mm=old,target_depth_mm=depth,physical_dt_s=(depth-old)/p.loading_rate))
            try:
                y,rec=step(x,old,depth,deadline);store.accepted(rec);context['index']+=1;return y,rec
            except (RuntimeError,ValueError,TimeoutError,AssertionError,FloatingPointError) as exc:
                context['rejected'].append(store.rejected(p,x,exc,failure_kind(exc,p.global_compute_deadline)));raise
        p.step=observed;return p
    def finish(out,result):
        if context['rejected'] and result['attempts']:result['attempts'][-1]['witness']=context['rejected'][-1]
        saved['finish'](out,result)
        result['summary'].update(qualification(result['steps'],result['config']['precrack_mm']))
        result['summary']['new_activation']=result['summary']['model_activation_occurred']
        result['summary']['new_connected_crack_growth']=result['summary']['new_connected_failed_length_mm']>0
        result['scope']='conditional softening time comparison from own-path parent; not a physical suffix'
        write_json(Path(out)/'result.json',native(result))
        (Path(out)/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Conditional coupled softening window</h1><p>No production continuation or full-cutting qualification.</p><pre>'+json.dumps(result['summary'],indent=2)+'</pre>')
    window.problem=factory;window.restore_parent=restore_parent;window.sources=sources;window.finish=finish
    try:yield
    finally:
        for name,value in saved.items():setattr(window,name,value)


def run(parent,parent_audit,state_index,preregister,out,dt,seconds=600,**runtime):
    raw=json.loads((Path(parent)/'result.json').read_text())
    if raw['source_hashes']!=growth.sources():raise ValueError('Conditional parent must use exactly the current growth equations')
    with selected_driver(out):
        return window.run(parent,parent_audit,state_index,preregister,out,dt,raw['config']['coupled_control']['material_layer'],seconds,**runtime)


@runtime_entry
def audit(case,out,seconds=500,*,runtime=None):
    case=Path(case);out=Path(out);start=time.monotonic();deadline=start+seconds
    if out.exists() or not 0<seconds<=540:raise ValueError('New bounded audit required')
    raw=json.loads((case/'result.json').read_text());report=dict(schema='softcut-coupled-audit/2',passed=False,
        source_hashes=sources(),result_sha256=digest(case/'result.json'),input_identity=input_identity(raw),
        auditor_sha256=digest(Path(__file__)),thread_runtime=runtime,states=[],witnesses=[])
    try:
        if raw['source_hashes']!=sources():raise ValueError('Source mismatch')
        p=candidate(raw['config'],raw['material_layer'],raw['solve_tolerance_n'])
        with np.load(case/'parent_state.npz',allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
        if digest(case/'parent_state.npz')!=raw['parent']['state_sha256']:raise ValueError('Parent identity mismatch')
        planfile=case/'preregistration.json'
        if digest(planfile)!=raw['preregistration_sha256']:raise ValueError('Plan changed')
        clock=TimeChain(raw,parent,p.loading_rate,p.start,json.loads(planfile.read_text())['coupled_window'])
        restore_parent(p,parent)
        initial_failed=parent['active_faces']&np.all(parent['history']>=2*p.toughness/p.strength,axis=1)
        report['entry_present_at_parent']=bool(strict_contacts(p,parent['positions'],p.connected_length(initial_failed)))
        previous=dict(time_s=clock.t0,depth_mm=raw['initial_depth_mm'],state_file='parent_state.npz')
        for row in raw['steps']:
            if time.monotonic()>deadline:raise TimeoutError('Conditional softening audit budget')
            file=case/row['state_file']
            if file.resolve().parent!=case.resolve() or digest(file)!=row['state_sha256']:raise ValueError('State identity changed')
            with np.load(file,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
            for key in ('positions','velocity','material_metric','history','released','active_faces'):
                np.testing.assert_array_equal(s['previous_'+key],parent[key])
            tick=clock.advance(s,row);fresh=route.restore_audit(p,s,row)
            if not fresh['accepted_chord_ccd_safe']:raise ValueError('Full chord unsafe')
            balance=independent_balance(p,s)
            failed=s['active_faces']&np.all(s['history']>=2*p.toughness/p.strength,axis=1)
            contacts=strict_contacts(p,s['positions'],p.connected_length(failed));check_entry_flag(contacts,row['blade_entered_surface_channel'])
            if contacts and not report['entry_present_at_parent'] and 'first_entry_bracket' not in report:report['first_entry_bracket']=dict(previous_nonentry=previous,first_entry=dict(**tick,state_file=row['state_file']))
            if np.any(s['active_faces']&~parent['active_faces']) and 'first_activation_bracket' not in report:
                report['first_activation_bracket']=dict(previous=previous,first=dict(**tick,state_file=row['state_file']),new_faces=np.flatnonzero(s['active_faces']&~parent['active_faces']).tolist())
            report['states'].append(dict(state_file=row['state_file'],state_sha256=digest(file),**tick,**balance,strict_contacts=contacts,
                min_j=fresh['min_j'],accepted_chord_ccd_safe=True,maximum_speed_mm_s=float(np.linalg.norm(s['velocity'][:p.n],axis=1).max()),
                mass_velocity_norm_sqrt_nmm=float(np.sqrt(p.mass@np.sum(s['velocity'][:p.n]**2,axis=1)))))
            previous=dict(**tick,state_file=row['state_file']);parent=s
        report['time_chain']=clock.finish()
        activation=[ref for row in raw['steps'] for ref in row.get('activation_witnesses',[])]
        if len(activation)!=sum(len(row.get('face_events',[])) for row in raw['steps']):raise ValueError('Missing activation witnesses')
        refs=list(activation);index=case/'witnesses/representatives.json'
        if index.exists():
            for item in json.loads(index.read_text())['signatures'].values():refs.extend(item[k] for k in ('first','worst','latest') if item.get(k))
        for ref in {r['sha256']:r for r in refs}.values():report['witnesses'].append(inspect_one(case,ref,sources(),deadline,factory=growth.problem))
        report.update(passed=bool(report['states']),activation_count=len(activation),activation_witness_verified=bool(activation),
            complete_requested_window=report['time_chain']['complete_requested_window'],
            absolute_numerical_including_equation_work_n_mm=sum(s['absolute_numerical_including_equation_work_n_mm'] for s in report['states']),
            new_gc_n_mm=sum(s['independent_fracture_n_mm'] for s in report['states']),
            max_residual_n=max((s['residual_n'] for s in report['states']),default=None),time_converged=False,mesh_converged=False,event_complete=False)
    except (ValueError,RuntimeError,AssertionError,TimeoutError,KeyError,OSError) as exc:report['error']=str(exc)
    report['source_unchanged']=sources()==raw['source_hashes'];report['passed'] &= report['source_unchanged']
    report['elapsed_s']=time.monotonic()-start;write_json(out,native(report));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','audit']);p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=500);p.add_argument('--parent-audit',type=Path);p.add_argument('--state-index',type=int);p.add_argument('--preregister',type=Path);p.add_argument('--dt',type=float);add_thread_arguments(p);a=p.parse_args()
    if a.mode=='run':q=run(a.input,a.parent_audit,a.state_index,a.preregister,a.out,a.dt,a.seconds,**thread_arguments(a))
    else:q=audit(a.input,a.out,a.seconds,**thread_arguments(a))
    print(json.dumps(q.get('summary',{k:q.get(k) for k in ('passed','error','activation_count','new_gc_n_mm','elapsed_s')})))
if __name__=='__main__':main()
