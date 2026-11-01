"""Bounded suffix audit reusing a hash-bound, independently audited prefix.

All inherited state bytes are rechecked. New states get full fresh mechanical,
CCD/J, history, clock and activation audits; all retained witnesses are replayed.
This avoids reassembling a long immutable prefix for every staged continuation.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from . import run_coupled_growth as growth
from . import run_fracture_route as route
from .audit_coupled_contact import independent_balance
from .audit_precrack_entry import strict_contacts,check_entry_flag
from .audit_face_witness import inspect_one
from .face_activation_chain import transition
from .continue_coupled_growth import controller_history
from .coupled_audit_contract import close
from .frozen_scene import digest
from .face_witness import native
from run_support import write_json


@runtime_entry
def audit(case,parent,parent_audit,out,seconds=500,*,resume_partial=False,runtime=None):
    case=Path(case);parent=Path(parent);out=Path(out);start=time.monotonic();deadline=start+seconds
    if out.exists() or not 0<seconds<=540:raise ValueError('New bounded continuation audit required')
    raw=json.loads((case/'result.json').read_text());old=json.loads((parent/'result.json').read_text());proof=json.loads(Path(parent_audit).read_text());source=growth.sources();helper=digest(Path(__file__))
    report=dict(schema='softcut-coupled-growth-audit/1',passed=False,result_sha256=digest(case/'result.json'),source_hashes=source,auditor_sha256=helper,thread_runtime=runtime,states=[],witnesses=[],
        verification_mode='All prefix state hashes rechecked; independent mechanical prefix proof reused; suffix and all retained witnesses freshly assembled')
    try:
        partial=False
        if resume_partial:
            if (case.resolve()!=parent.resolve() or proof.get('passed') or proof.get('schema')!='softcut-coupled-growth-audit/1'
                    or proof.get('error')!='Independent growth audit budget' or not proof.get('source_unchanged')
                    or proof.get('auditor_sha256')!=digest(Path(growth.__file__))):
                raise ValueError('Only an original same-source growth audit stopped by its budget may be completed')
            n=len(proof['states'])
            if not 0<n<len(raw['steps']):raise ValueError('Incomplete nonempty checked prefix required')
            old=copy.deepcopy(raw);old['steps']=old['steps'][:n];partial=True
            # The old JSON stays failed. These derived fields only initialize
            # the suffix checker after rebinding the virgin state and prefix.
            initial_problem=growth.problem(raw['config'])
            initial=dict(positions=initial_problem.rest,velocity=initial_problem.velocity,material_metric=initial_problem.material_metric,history=initial_problem.history,released=initial_problem.released,active_contact=initial_problem.active_contact,active_faces=initial_problem.active_faces)
            with np.load(case/'initial_state.npz',allow_pickle=False) as z:
                for key in z.files:np.testing.assert_array_equal(z[key],initial[key])
            proof['initial_state_sha256']=digest(case/'initial_state.npz')
            proof['terminal_time_s']=(old['steps'][-1]['depth_mm']-initial_problem.start)/initial_problem.loading_rate
            close(proof['terminal_time_s'],proof['states'][-1]['time_s'],'Checked prefix clock')
            report['verification_mode']='Complete original budget-stopped audit: rebind initial state and every checked NPZ; assemble remaining states and ALL witnesses. Original partial report remains failed.'
        else:n=len(old['steps'])
        if (not proof['passed'] and not partial) or proof['result_sha256']!=digest(parent/'result.json') or proof['source_hashes']!=source or len(proof['states'])!=n:raise ValueError('Unbound parent audit')
        if raw['source_hashes']!=source or old['source_hashes']!=source or not raw['source_unchanged']:raise ValueError('Source mismatch')
        if not raw.get('own_initial_state') or raw.get('conditional_comparison') or raw['config']!=old['config']:raise ValueError('Not same own-initial-state path')
        if not partial:
            link=raw['continuation']
            if link['parent_result_sha256']!=proof['result_sha256'] or link['parent_audit_sha256']!=digest(Path(parent_audit)) or link['inherited_states']!=n:raise ValueError('Wrong continuation parent')
        if raw['steps'][:n]!=old['steps']:raise ValueError('Inherited records changed')
        if digest(case/'initial_state.npz')!=proof['initial_state_sha256'] or digest(parent/'initial_state.npz')!=proof['initial_state_sha256']:raise ValueError('Initial state identity mismatch')
        for row,checked in zip(old['steps'],proof['states']):
            if time.monotonic()>deadline:raise TimeoutError('Prefix hash budget')
            path=case/row['state_file']
            if path.resolve().parent!=case.resolve() or row['state_file']!=checked['state_file'] or row['state_sha256']!=checked['state_sha256'] or digest(path)!=checked['state_sha256']:raise ValueError('Inherited state bytes changed')
        p=growth.problem(raw['config']);controller_history(p,raw,terminal_accuracy_audit=True)
        report['continuation_eligible']=raw['status'].startswith('completed') or raw.get('failure_kind')=='global_compute_budget_exhausted'
        report['terminal_failure_not_accepted']=raw.get('failure_kind')
        report['states']=copy.deepcopy(proof['states']);report['inherited_independent_proof']=dict(directory=str(parent.resolve()),result_sha256=proof['result_sha256'],audit_file=str(Path(parent_audit).resolve()),audit_sha256=digest(Path(parent_audit)),states=n)
        for key in ('first_entry_bracket','first_activation_bracket','first_new_damage'):
            if key in proof:report[key]=copy.deepcopy(proof[key])
        with np.load(case/old['steps'][-1]['state_file'],allow_pickle=False) as z:previous={k:z[k].copy() for k in z.files}
        previous_time=proof['terminal_time_s'];parent_ref=dict(file=old['steps'][-1]['state_file'],sha256=old['steps'][-1]['state_sha256']);entry_previous=dict(time_s=previous_time,depth_mm=float(previous['depth_mm']),state_file=parent_ref['file'])
        for row in raw['steps'][n:]:
            if time.monotonic()>deadline:raise TimeoutError('Fresh continuation audit budget')
            file=case/row['state_file']
            if file.resolve().parent!=case.resolve() or digest(file)!=row['state_sha256']:raise ValueError('State identity mismatch')
            with np.load(file,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
            for key in ('positions','velocity','material_metric','history','released','active_faces'):np.testing.assert_array_equal(s['previous_'+key],previous[key])
            np.testing.assert_array_equal(s['mass'],p.mass)
            h=float(s['material_dt_s']);depth=float(s['depth_mm']);t=(depth-p.start)/p.loading_rate
            close(h,(depth-previous['depth_mm'])/p.loading_rate,'Independent depth/time');close(row['dt_s'],h,'Production dt');close(row['time_s'],t,'Production time');close(t-previous_time,h,'Clock chain')
            if h<=0 or h>raw['config']['base']['increment_mm']/p.loading_rate+1e-12:raise ValueError('Invalid physical schedule')
            if hasattr(p,'next_loading_depth') and depth>p.next_loading_depth(float(previous['depth_mm']),raw['config']['base']['increment_mm'],raw['requested_stop_depth_mm'])+1e-12:raise ValueError('Time cap violation')
            chain=transition(case,previous,s,row,parent_ref,float(previous['depth_mm']),raw['config'],source,p.seam)
            fresh=route.restore_audit(p,s,row)
            if not fresh['accepted_chord_ccd_safe']:raise ValueError('Unsafe full chord')
            balance=independent_balance(p,s);failed=s['active_faces']&np.all(s['history']>=2*p.toughness/p.strength,axis=1);length=p.connected_length(failed);contacts=strict_contacts(p,s['positions'],length);check_entry_flag(contacts,row['blade_entered_surface_channel'])
            now=dict(time_s=t,depth_mm=depth,state_file=row['state_file'])
            if contacts and 'first_entry_bracket' not in report:report['first_entry_bracket']=dict(previous_nonentry=entry_previous,first_entry=now)
            if not contacts:entry_previous=now
            if np.any(s['active_faces']&~previous['active_faces']) and 'first_activation_bracket' not in report:report['first_activation_bracket']=dict(previous_time_s=previous_time,time_s=t,depth_mm=depth,new_faces=np.flatnonzero(s['active_faces']&~previous['active_faces']).tolist())
            if np.any(s['history'][~p.precracked_faces]>previous['history'][~p.precracked_faces]) and 'first_new_damage' not in report:report['first_new_damage']=now
            report['states'].append(dict(**now,dt_s=h,state_sha256=digest(file),min_j=fresh['min_j'],strict_contacts=contacts,activation_chain=chain,connected_length_mm=length,active_faces=int(s['active_faces'].sum()),**balance))
            previous=s;previous_time=t;parent_ref=dict(file=row['state_file'],sha256=row['state_sha256'])
        complete=abs(float(previous['depth_mm'])-raw['requested_stop_depth_mm'])<=1e-12
        if complete!=(raw['status'] in ('completed_pilot','completed_2mm')):raise ValueError('False completion label')
        refs=[ref for row in raw['steps'] for ref in row.get('activation_witnesses',[])];index=case/'witnesses/representatives.json'
        if index.exists():
            for v in json.loads(index.read_text())['signatures'].values():refs.extend(v[k] for k in ('first','worst','latest') if v.get(k))
        for ref in {r['sha256']:r for r in refs}.values():report['witnesses'].append(inspect_one(case,ref,source,deadline,factory=growth.problem))
        report.update(passed=bool(report['states']),complete_requested_path=complete,terminal_depth_mm=float(previous['depth_mm']),terminal_time_s=previous_time,initial_state_sha256=proof['initial_state_sha256'],
            new_fracture_dissipation_n_mm=sum(s['independent_fracture_n_mm'] for s in report['states']),absolute_numerical_work_n_mm=sum(s['absolute_numerical_including_equation_work_n_mm'] for s in report['states']),time_converged=False,mesh_converged=False,complete_event_verified=False)
    except (RuntimeError,ValueError,AssertionError,OSError,TimeoutError,KeyError) as exc:report['error']=str(exc)
    report.update(source_unchanged=source==growth.sources() and helper==digest(Path(__file__)),elapsed_s=time.monotonic()-start);report['passed'] &= report['source_unchanged'];write_json(out,native(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('parent');p.add_argument('parent_audit');p.add_argument('--out',required=True);p.add_argument('--resume-partial',action='store_true');p.add_argument('--seconds',type=float,default=500);add_thread_arguments(p);a=p.parse_args();q=audit(a.case,a.parent,a.parent_audit,a.out,a.seconds,resume_partial=a.resume_partial,**thread_arguments(a));print(json.dumps({k:q.get(k) for k in ('passed','error','elapsed_s','terminal_depth_mm')}))
