"""Bounded independent mechanical chunks plus complete clock/topology validation.

A chunk is never a path certificate. Composition requires exact, disjoint full
coverage, immutable inputs and the same auditor/kernel, including all witnesses.
The producer and its existing whole-window auditor remain unchanged.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from . import branch_coupled_window as branch
from . import run_coupled_growth_window as base
from .coupled_audit_contract import TimeChain,input_identity
from .event_resolution import bind_reference,enforce
from .frozen_scene import digest
from .face_witness import native
from run_support import write_json


def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def validate_coverage(parts,count):
    identity=parts[0]['binding'];cursor=0
    for part in sorted(parts,key=lambda p:p['range'][0]):
        if not part['passed'] or part['binding']!=identity:raise ValueError('Failed or mismatched audit chunk')
        a,b=part['range']
        if a!=cursor or b<=a or b>count or len(part['states'])!=b-a:raise ValueError('Missing/duplicate/out-of-range audit coverage')
        cursor=b
    if cursor!=count:raise ValueError('Incomplete mechanical audit coverage')
    if sum(bool(p['witnesses_checked']) for p in parts)!=1:raise ValueError('Complete witness pass required exactly once')
    return identity


def inspect(case,out,seconds,first,stop,runtime):
    start=time.monotonic();deadline=start+seconds;case=Path(case);out=Path(out)
    if out.exists() or not 0<seconds<=540:raise ValueError('New bounded audit output required')
    raw=json.loads((case/'result.json').read_text());stop=min(stop,len(raw['steps']))
    if not 0<=first<stop:raise ValueError('Nonempty audit range required')
    identity=base.sources();report=dict(schema='softcut-coupled-mechanical-chunk/1',passed=False,range=[first,stop],states=[],witnesses=[],witnesses_checked=False,thread_runtime=runtime)
    try:
        if raw['source_hashes']!=identity:raise ValueError('Numerical source mismatch')
        p=base.candidate(raw['config'],raw['material_layer'],raw['solve_tolerance_n'])
        parent=read(case/'parent_state.npz')
        if digest(case/'parent_state.npz')!=raw['parent']['state_sha256']:raise ValueError('Parent identity mismatch')
        if digest(case/'preregistration.json')!=raw['preregistration_sha256']:raise ValueError('Plan changed')
        plan=json.loads((case/'preregistration.json').read_text())['coupled_window'];clock=TimeChain(raw,parent,p.loading_rate,p.start,plan)
        resolution=bind_reference(plan,raw['dt_requested_s']) if 'event_resolution' in plan else None
        base.restore_parent(p,parent)
        failed=parent['active_faces']&np.all(parent['history']>=2*p.toughness/p.strength,axis=1)
        report['entry_present_at_parent']=bool(base.strict_contacts(p,parent['positions'],p.connected_length(failed)))
        parent_ref=dict(file='parent_state.npz',sha256=digest(case/'parent_state.npz'));previous=dict(time_s=clock.t0,depth_mm=raw['initial_depth_mm'],state_file='parent_state.npz')
        old_record=json.loads((case/'parent_record.json').read_text());state_hashes=[]
        for index,row in enumerate(raw['steps']):
            if time.monotonic()>deadline:raise TimeoutError('Mechanical chunk budget')
            path=case/row['state_file']
            if path.resolve().parent!=case.resolve() or digest(path)!=row['state_sha256']:raise ValueError('State identity changed')
            s=read(path);state_hashes.append(row['state_sha256'])
            for key in ('positions','velocity','material_metric','history','released','active_faces'):np.testing.assert_array_equal(s['previous_'+key],parent[key])
            activation=base.transition(case,parent,s,row,parent_ref,previous['depth_mm'],raw['config'],identity,p.seam)
            tick=clock.advance(s,row)
            if first<=index<stop:
                fresh=base.route.restore_audit(p,s,row)
                if not fresh['accepted_chord_ccd_safe']:raise ValueError('Full accepted chord unsafe')
                balance=base.independent_balance(p,s)
                if resolution is not None:
                    old=dict(parent,blade_force_components_n=[old_record[k] for k in ('reaction_normal_fz_n','reaction_friction_fz_n')])
                    rec=p.acceptance(s['positions'],float(s['depth_mm']),0,0)
                    control=enforce(p,s['positions'],rec,old,resolution)
                    for group in ('values','allowances','ratios'):
                        for k,v in control[group].items():np.testing.assert_allclose(v,row['event_resolution'][group][k],rtol=1e-8,atol=1e-13)
                    balance['event_resolution']=control
                failed=s['active_faces']&np.all(s['history']>=2*p.toughness/p.strength,axis=1)
                contacts=base.strict_contacts(p,s['positions'],p.connected_length(failed));base.check_entry_flag(contacts,row['blade_entered_surface_channel'])
                report['states'].append(dict(state_file=row['state_file'],state_sha256=row['state_sha256'],**tick,**balance,strict_contacts=contacts,activation_chain=activation,min_j=fresh['min_j'],accepted_chord_ccd_safe=True,maximum_speed_mm_s=float(np.linalg.norm(s['velocity'][:p.n],axis=1).max()),mass_velocity_norm_sqrt_nmm=float(np.sqrt(p.mass@np.sum(s['velocity'][:p.n]**2,axis=1)))))
            previous=dict(**tick,state_file=row['state_file']);parent=s;parent_ref=dict(file=row['state_file'],sha256=row['state_sha256']);old_record=row
        report['time_chain']=clock.finish()
        activations=[ref for row in raw['steps'] for ref in row.get('activation_witnesses',[])]
        if len(activations)!=sum(len(row.get('face_events',[])) for row in raw['steps']):raise ValueError('Missing activation witnesses')
        report['activation_count']=len(activations)
        if first==0:
            refs=list(activations);path=case/'witnesses/representatives.json'
            if path.exists():
                for item in json.loads(path.read_text())['signatures'].values():refs.extend(item[k] for k in ('first','worst','latest') if item.get(k))
            for ref in {v['sha256']:v for v in refs}.values():report['witnesses'].append(base.inspect_one(case,ref,identity,deadline,factory=base.growth.problem))
            report['witnesses_checked']=True
        report['binding']=dict(result_sha256=digest(case/'result.json'),input_identity=input_identity(raw),source_hashes=identity,auditor_sha256=digest(Path(__file__)),state_hashes=state_hashes,representatives_sha256=digest(case/'witnesses/representatives.json') if (case/'witnesses/representatives.json').exists() else None)
        report['passed']=len(report['states'])==stop-first
    except (ValueError,RuntimeError,AssertionError,TimeoutError,KeyError,OSError) as exc:report['error']=str(exc)
    report['source_unchanged']=base.sources()==raw['source_hashes'];report['passed'] &= report['source_unchanged'];report['elapsed_s']=time.monotonic()-start
    write_json(out,native(report));return report


@runtime_entry
def run(case,out,first,stop,seconds=500,*,runtime=None):
    # Reuse all branch/predictor/numerical-policy identity checks, then perform
    # the explicit ranged auditor inside the original source context.
    saved=base.audit
    base.audit=lambda case,out,seconds:inspect(case,out,seconds,first,stop,runtime)
    try:return branch.audit(case,out,seconds)
    finally:base.audit=saved


def compose(case,chunk_files,out):
    case=Path(case);out=Path(out)
    if out.exists():raise ValueError('New composition output required')
    raw=json.loads((case/'result.json').read_text());parts=[json.loads(Path(f).read_text()) for f in chunk_files];parts.sort(key=lambda p:p['range'][0])
    binding=validate_coverage(parts,len(raw['steps']))
    if binding['result_sha256']!=digest(case/'result.json') or binding['input_identity']!=input_identity(raw):raise ValueError('Stale chunk/result identity')
    if binding['auditor_sha256']!=digest(Path(__file__)):raise ValueError('Different chunk auditor')
    if digest(case/'parent_state.npz')!=raw['parent']['state_sha256'] or digest(case/'preregistration.json')!=raw['preregistration_sha256']:raise ValueError('Changed parent/plan')
    refs=raw['branch_control']['predictor_states']
    if len(refs)!=2 or any(digest(case/ref['copied_file'])!=ref['state_sha256'] for ref in refs):raise ValueError('Changed predictor states')
    if 'numerical_policy_parent' in raw and digest(case/'numerical_policy_parent.json')!=raw['numerical_policy_parent']['manifest_sha256']:raise ValueError('Changed numerical policy declaration')
    states=[row for part in parts for row in part['states']]
    if len(binding['state_hashes'])!=len(states):raise ValueError('Incomplete bound state hash list')
    for row,check,h in zip(raw['steps'],states,binding['state_hashes']):
        if row['state_sha256']!=h or check['state_sha256']!=h or digest(case/row['state_file'])!=h or row['state_file']!=check['state_file']:raise ValueError('Changed audited state')
    witnesses=[z for p in parts for z in p['witnesses']]
    for z in witnesses:
        # The complete chunk hashes its representative index and all activation
        # references are in bound result; inspect_one checked every witness hash.
        if not z['passed']:raise ValueError('Failed witness')
    index=case/'witnesses/representatives.json'
    if (digest(index) if index.exists() else None)!=binding['representatives_sha256']:raise ValueError('Changed witness index')
    refs=[ref for row in raw['steps'] for ref in row.get('activation_witnesses',[])]
    if index.exists():
        for item in json.loads(index.read_text())['signatures'].values():refs.extend(item[k] for k in ('first','worst','latest') if item.get(k))
    for ref in refs:
        if digest(case/ref['file'])!=ref['sha256']:raise ValueError('Changed audited witness')
    if any(p['time_chain']!=parts[0]['time_chain'] for p in parts):raise ValueError('Different reconstructed clocks')
    with branch.source_context() as source:
        if source!=binding['source_hashes']:raise ValueError('Source changed after audit')
    report=dict(schema='softcut-coupled-audit/2',passed=True,**{k:binding[k] for k in ('source_hashes','result_sha256','input_identity','auditor_sha256')},states=states,witnesses=witnesses,time_chain=parts[0]['time_chain'],entry_present_at_parent=parts[0]['entry_present_at_parent'],complete_requested_window=parts[0]['time_chain']['complete_requested_window'],activation_count=parts[0]['activation_count'],activation_witness_verified=bool(parts[0]['activation_count']),absolute_numerical_including_equation_work_n_mm=sum(s['absolute_numerical_including_equation_work_n_mm'] for s in states),new_gc_n_mm=sum(s['independent_fracture_n_mm'] for s in states),max_residual_n=max(s['residual_n'] for s in states),source_unchanged=True,time_converged=False,mesh_converged=False,event_complete=False,composition=dict(scope='Every state mechanically checked exactly once; full clock/topology/hash chain in every chunk; witnesses once',parts=[dict(file=str(f),sha256=digest(Path(f))) for f in chunk_files]))
    previous=dict(time_s=raw['initial_time_s'],depth_mm=raw['initial_depth_mm'],state_file='parent_state.npz')
    for s,row in zip(states,raw['steps']):
        now={k:s[k] for k in ('time_s','depth_mm','dt_s','state_file')}
        if s['strict_contacts'] and not report['entry_present_at_parent'] and 'first_entry_bracket' not in report:report['first_entry_bracket']=dict(previous_nonentry=previous,first_entry=now)
        previous=now
    write_json(out,native(report));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','compose'));p.add_argument('case');p.add_argument('--out',required=True);p.add_argument('--first',type=int);p.add_argument('--stop',type=int);p.add_argument('--seconds',type=float,default=500);p.add_argument('--parts',nargs='*');add_thread_arguments(p);a=p.parse_args()
    q=run(a.case,a.out,a.first,a.stop,a.seconds,**thread_arguments(a)) if a.mode=='run' else compose(a.case,a.parts,a.out)
    print(json.dumps({k:q.get(k) for k in ('passed','error','range','elapsed_s','complete_requested_window')}))
if __name__=='__main__':main()
