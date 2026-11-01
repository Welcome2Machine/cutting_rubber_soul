"""Conditional branching of an audited coupled window, with exact history/predictor.

This is not production continuation. Parent equations remain hash-checked;
helper identity is appended to the child's source identity and to its audit.
"""
import argparse,contextlib,json,shutil
from pathlib import Path
import numpy as np
from numerical_runtime import add_thread_arguments,thread_arguments
from . import run_coupled_growth_window as growth_window
from . import run_coupled_contact as window
from .continue_coupled_growth import restore
from .coupled_audit_contract import bind_case
from .frozen_scene import digest
from run_support import write_json


@contextlib.contextmanager
def source_context():
    original=growth_window.sources
    identity=original();identity['tools/branch_coupled_window.py']=digest(Path(__file__))
    growth_window.sources=lambda:identity.copy()
    try:yield identity
    finally:growth_window.sources=original


def run(parent,parent_audit,state_index,preregister,out,dt,seconds=600,**runtime):
    parent=Path(parent);out=Path(out);raw=json.loads((parent/'result.json').read_text());audit=json.loads(Path(parent_audit).read_text())
    own=raw.get('own_initial_state',False) and not raw.get('conditional_comparison',True)
    if own:
        from . import run_coupled_growth as growth
        if (not audit['passed'] or not audit.get('complete_requested_path')
                or audit['result_sha256']!=digest(parent/'result.json')
                or audit['source_hashes']!=raw['source_hashes'] or raw['source_hashes']!=growth.sources()
                or len(audit['states'])!=len(raw['steps'])):
            raise ValueError('Complete same-source own-initial parent audit required')
        layer=raw['config']['coupled_control']['material_layer']
    else:
        bind_case(parent,raw,audit)
        if raw['source_hashes']!=growth_window.sources() or not raw['conditional_comparison']:
            raise ValueError('Audited same-equation conditional parent required')
        layer=raw['material_layer']
    if state_index<1:raise ValueError('Two accepted predictor states required')
    records=[];refs=[]
    for row in raw['steps'][state_index-1:state_index+1]:
        if digest(parent/row['state_file'])!=row['state_sha256']:raise ValueError('Predictor state identity changed')
        with np.load(parent/row['state_file'],allow_pickle=False) as z:records.append((row['depth_mm'],z['positions'].copy()))
        refs.append({k:row[k] for k in ('state_file','state_sha256','depth_mm','time_s')})
    if len(records)!=2:raise ValueError('Branch state index outside parent')
    with source_context():
        with growth_window.selected_driver(out):
            window.restore_parent=lambda p,s:restore(p,s,records)
            result=window.run(parent,parent_audit,state_index,preregister,out,dt,layer,seconds,**runtime)
        for i,ref in enumerate(refs):
            ref['copied_file']=f'predictor_parent_{i}.npz';shutil.copy2(parent/ref['state_file'],out/ref['copied_file'])
        result['branch_control']=dict(parent_own_initial_state=own,parent_result_sha256=digest(parent/'result.json'),parent_audit_sha256=digest(Path(parent_audit)),
            predictor_states=refs,restore='Complete saved physical fields, exact contact topology and two accepted predictor states',
            physical_history_transplant=False,production_continuation=False)
        write_json(out/'result.json',result)
    return result


def audit(case,out,seconds=500,**runtime):
    case=Path(case);raw=json.loads((case/'result.json').read_text());refs=raw['branch_control']['predictor_states']
    if len(refs)!=2:raise ValueError('Two bound predictors required')
    for ref in refs:
        if Path(ref['copied_file']).name!=ref['copied_file'] or digest(case/ref['copied_file'])!=ref['state_sha256']:raise ValueError('Copied predictor identity mismatch')
    with np.load(case/'parent_state.npz') as parent,np.load(case/refs[0]['copied_file']) as prev,np.load(case/refs[1]['copied_file']) as final:
        np.testing.assert_array_equal(prev['positions'],parent['previous_positions'])
        for k in parent.files:np.testing.assert_array_equal(parent[k],final[k])
    with source_context():return growth_window.audit(case,out,seconds,**runtime)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','audit']);p.add_argument('case');p.add_argument('--parent-audit');p.add_argument('--state-index',type=int);p.add_argument('--preregister');p.add_argument('--out',required=True);p.add_argument('--dt',type=float);p.add_argument('--seconds',type=float,default=500);add_thread_arguments(p);a=p.parse_args()
    if a.mode=='run':q=run(a.case,a.parent_audit,a.state_index,a.preregister,a.out,a.dt,a.seconds,**thread_arguments(a))
    else:q=audit(a.case,a.out,a.seconds,**thread_arguments(a))
    print(json.dumps({k:q.get(k) for k in ('status','passed','error','elapsed_s')}))
