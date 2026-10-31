"""Explicit conditional re-evaluation of one archived attempt, never continuation.

Requires an original complete accepted-prefix audit bound to the parent result.
Current equations independently reassemble the chosen state. Both producer and
current hashes remain recorded; this does not relabel a new witness as an
original unsaved trajectory, or certify a history from the new virgin state.
"""
import argparse,json,shutil,sys,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from run_support import write_json
from step_control import execute_step,failure_kind
from . import run_endpoint_face_route as route
from . import run_fracture_route as base
from .face_witness import WitnessStore,native
from .frozen_scene import digest


@runtime_entry
def run(case,parent_audit,out,seconds=120,*,mode='activation',runtime=None):
    case=Path(case);out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+seconds
    raw=json.loads((case/'result.json').read_text());a=json.loads(Path(parent_audit).read_text())
    if not 0<seconds<=540 or not a['passed'] or a['result_sha256']!=digest(case/'result.json'):raise ValueError('Original audited parent required')
    if a['source_hashes']!=raw['source_hashes'] or not a['source_unchanged']:raise ValueError('Original audit source identity')
    if mode=='activation':
        event=next(i for i,row in enumerate(raw['steps']) if row.get('face_events'));index=event-1;target=raw['steps'][event]['depth_mm']
    elif mode=='negative_work':
        attempt=next(row for row in reversed(raw['attempts']) if 'Negative endpoint friction work' in row['error']);index=next(i for i,row in enumerate(raw['steps']) if row['depth_mm']==attempt['from_mm']);target=attempt['target_mm']
    else:raise ValueError('Unknown conditional attempt')
    if index<0:raise ValueError('Expected a saved event predecessor')
    rec=raw['steps'][index];parent=case/rec['state_file']
    if digest(parent)!=rec['state_sha256']:raise ValueError('Parent state changed')
    with np.load(parent,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
    p=route.problem(raw['config']);fresh=base.restore_audit(p,s,rec)
    if not fresh['passed'] or not fresh['accepted_chord_ccd_safe']:raise ValueError('Current equation revalidation failed')
    for name in ('velocity','material_metric','history','released','active_contact','active_faces'):setattr(p,name,s[name].copy())
    p.rebuild_contact();p.set_free();p._accepted_predictor_history=[]
    for row in raw['steps'][max(0,index-1):index+1]:
        path=case/row['state_file']
        if digest(path)!=row['state_sha256']:raise ValueError('Predictor identity')
        with np.load(path,allow_pickle=False) as z:p._accepted_predictor_history.append((float(row['depth_mm']),z['positions'].copy()))
    shutil.copyfile(parent,out/'parent_state.npz');source=route.sources()
    result=dict(schema='softcut-conditional-face-attempt/1',config=raw['config'],source_hashes=source,
        original_source_hashes=raw['source_hashes'],original_result_sha256=digest(case/'result.json'),original_audit_sha256=digest(Path(parent_audit)),
        original_state_sha256=digest(parent),current_parent_revalidation=fresh,runner_sha256=digest(Path(__file__)),execution_identity=base.driver.execution_identity(),thread_runtime=runtime,interpreter=sys.executable,
        production_suffix_qualified=False,original_unsaved_trial_replayed=False,steps=[],attempts=[],mode=mode)
    store=WitnessStore(out,dict(config=raw['config'],source_hashes=source,execution_identity=result['execution_identity'],runtime=runtime))
    store.begin(p,s['positions'],dict(parent=dict(file='parent_state.npz',sha256=digest(out/'parent_state.npz')),from_depth_mm=rec['depth_mm'],target_depth_mm=target,physical_dt_s=(target-rec['depth_mm'])/p.loading_rate,conditional_revalidation=True))
    old=p.released.copy()
    try:
        x,row=execute_step(p,s['positions'],rec['depth_mm'],target,deadline,deadline)
        store.accepted(row);base.observables(p,x,row);row.update(time_s=(target-p.start)/p.loading_rate,dt_s=p.material_dt)
        state=base.snapshot(p,x,old,target);np.savez_compressed(out/'state_00000.npz',**state)
        row.update(state_file='state_00000.npz',state_sha256=digest(out/'state_00000.npz'));result['steps']=[row]
        result['conditional_endpoint_audit']=base.restore_audit(route.problem(raw['config']),state,row)
        result['status']='conditional_attempt_accepted'
    except (RuntimeError,ValueError,TimeoutError) as exc:
        kind=failure_kind(exc,deadline);w=store.rejected(p,s['positions'],exc,kind)
        result.update(status='conditional_attempt_rejected',error=str(exc),failure_kind=kind)
        result['attempts']=[dict(error=str(exc),failure_kind=kind,witness=w)]
    result.update(source_unchanged=source==route.sources(),elapsed_s=time.monotonic()-start)
    write_json(out/'result.json',native(result));print(json.dumps({k:result.get(k) for k in ('status','error','elapsed_s')}));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--parent-audit',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--mode',choices=['activation','negative_work'],required=True);p.add_argument('--seconds',type=float,default=120);add_thread_arguments(p);a=p.parse_args()
    run(a.case,a.parent_audit,a.out,a.seconds,mode=a.mode,**thread_arguments(a))
if __name__=='__main__':main()
