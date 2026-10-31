"""Explicit, same-equation/configuration continuation of an audited growth path.

The entire immutable prefix and its witnesses remain available to the ordinary
independent path audit. The loading controller is replayed, not reset. No
conditional window is accepted as a production parent.
"""
import argparse,copy,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from numerical_runtime import runtime_entry,add_thread_arguments,thread_arguments
from . import run_coupled_growth as growth
from . import run_fracture_route as route
from .run_coupled_growth_window import restore_parent
from .face_witness import WitnessStore,accepted_fields,array_hash,native
from .frozen_scene import digest
from step_control import execute_step,failure_kind,physical_cutback,reduced_increment
from prototype.ipc_reference import LoadStepFailure
from run_support import write_json


def controller_increment(p,raw):
    base=raw['config']['base'];inc=base['increment_mm'];depth=p.start;attempts=iter(raw['attempts']);pending=next(attempts,None)
    for row in raw['steps']:
        while pending is not None and abs(pending['from_mm']-depth)<=1e-12:
            if not physical_cutback(pending['failure_kind']):raise ValueError('Nonphysical rejection cannot precede accepted suffix')
            inc=reduced_increment(p,base,pending['target_mm']-pending['from_mm'],pending['failure_kind'])
            pending=next(attempts,None)
        expected=p.next_loading_depth(depth,inc,raw['requested_stop_depth_mm']) if hasattr(p,'next_loading_depth') else min(depth+inc,raw['requested_stop_depth_mm'])
        if abs(expected-row['depth_mm'])>1e-12:raise ValueError('Cannot reconstruct exact accepted controller schedule')
        inc=max(base['minimum_increment_mm'],min(base['increment_mm'],inc*route.driver.accepted_step_growth(base,row)))
        depth=row['depth_mm']
    if pending is not None:
        # A global timeout can occur after physical cutbacks at the final parent.
        while pending is not None:
            if abs(pending['from_mm']-depth)>1e-12:raise ValueError('Unordered terminal attempts')
            if physical_cutback(pending['failure_kind']):inc=reduced_increment(p,base,pending['target_mm']-pending['from_mm'],pending['failure_kind'])
            elif pending['failure_kind']!='global_compute_budget_exhausted':raise ValueError('Unresolved nonbudget failure is not resumable')
            pending=next(attempts,None)
    if 'next_increment_mm' in raw and abs(inc-raw['next_increment_mm'])>1e-14:raise ValueError('Saved/replayed controller mismatch')
    return inc


def restore(p,state,records):
    x=restore_parent(p,state)
    # Preserve the actual two-state predictor history. Older states need not
    # satisfy newly active unilateral constraints: live predictors keep them,
    # and their candidates still undergo the original boundary/CCD/J checks.
    if not records or len(records)>2:raise ValueError('One or two accepted predictor states required')
    for depth,pos in records:
        if not np.isfinite(pos).all() or pos.shape!=p.rest.shape:raise ValueError('Invalid predictor shape')
        np.testing.assert_array_equal(pos[p.mesh.bottom_nodes],p.rest[p.mesh.bottom_nodes])
        np.testing.assert_allclose(pos[p.n:],p.prescribed_blade_positions(depth),rtol=0,atol=p.blade_boundary_roundoff_tolerance(depth))
    np.testing.assert_array_equal(records[-1][1],x)
    if records[-1][0]!=float(state['depth_mm']):raise ValueError('Predictor parent depth mismatch')
    p._accepted_predictor_history=[(float(d),y.copy()) for d,y in records]
    expected={k:state[k] for k in accepted_fields(p,x)}
    if array_hash(accepted_fields(p,x))!=array_hash(expected):raise ValueError('Physical history restore mismatch')
    return x


@runtime_entry
def run(parent,parent_audit,out,stop_depth,seconds=1800,*,runtime=None):
    if not 30<=seconds<=2400:raise ValueError('Bounded continuation requires 30..2400 seconds')
    parent=Path(parent);out=Path(out);start=time.monotonic();source=growth.sources();helper=digest(Path(__file__))
    original=json.loads((parent/'result.json').read_text());proof=json.loads(Path(parent_audit).read_text())
    if original['source_hashes']!=source or not original['source_unchanged'] or not original.get('own_initial_state') or original.get('conditional_comparison'):
        raise ValueError('Only same-source own-initial-state physical parent can continue')
    if not proof['passed'] or proof['result_sha256']!=digest(parent/'result.json') or len(proof['states'])!=len(original['steps']):raise ValueError('Full bound independent parent audit required')
    signature=hashlib.sha256(json.dumps(original['config'],sort_keys=True).encode()).hexdigest()
    if signature!=original['config_sha256']:raise ValueError('Configuration identity mismatch')
    if original['status'] not in ('completed_pilot','completed_2mm') and original.get('failure_kind')!='global_compute_budget_exhausted':raise ValueError('Only completed stage or global budget stop can continue')
    p=growth.problem(original['config']);base=original['config']['base'];increment=controller_increment(p,original)
    if not original['steps'][-1]['depth_mm']<stop_depth<=min(1.,base['target_mm']):raise ValueError('Explicit increasing round scope <=1 mm required')
    records=[]
    for row in original['steps'][-2:]:
        file=parent/row['state_file']
        if digest(file)!=row['state_sha256']:raise ValueError('Parent checkpoint changed')
        with np.load(file,allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        records.append((row['depth_mm'],state['positions'].copy()))
    x=restore(p,state,records);depth=float(state['depth_mm']);out.mkdir(parents=True,exist_ok=False)
    store=WitnessStore(out,dict(config=original['config'],source_hashes=source,execution_identity=route.driver.execution_identity(),runtime=runtime))
    for file in [parent/'initial_state.npz',*[parent/r['state_file'] for r in original['steps']],*(parent/'witnesses').rglob('*.npz')]:
        dest=out/file.relative_to(parent);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
    index=parent/'witnesses/representatives.json'
    if index.exists():
        saved=json.loads(index.read_text());store.signatures=copy.deepcopy(saved['signatures']);store.omitted_signatures=saved['omitted_new_signatures'];shutil.copyfile(index,out/'witnesses/representatives.json')
    result=copy.deepcopy(original)
    for key in ('error','failure_kind','next_increment_mm','summary'):result.pop(key,None)
    result.update(status='running',requested_stop_depth_mm=stop_depth,seconds=seconds,thread_runtime=runtime,
        continuation=dict(parent_directory=str(parent.resolve()),parent_result_sha256=digest(parent/'result.json'),parent_audit_sha256=digest(Path(parent_audit)),
            helper_sha256=helper,inherited_states=len(original['steps']),restored_accepted_array_sha256=array_hash(accepted_fields(p,x)),
            restored_controller_increment_mm=increment,predictor_state_depths_mm=[d for d,_ in records],physical_configuration_changed=False),
        execution_identity=route.driver.execution_identity())
    write_json(out/'manifest.json',native(result));deadline=start+seconds-10;consecutive=0
    parent_ref=dict(file=result['steps'][-1]['state_file'],sha256=result['steps'][-1]['state_sha256'])
    try:
        while depth<stop_depth-1e-12:
            if time.monotonic()>=deadline:raise TimeoutError('Global continuation loading budget')
            target=p.next_loading_depth(depth,increment,stop_depth) if hasattr(p,'next_loading_depth') else min(stop_depth,depth+increment)
            old=p.released.copy();before=time.monotonic();store.begin(p,x,dict(parent=parent_ref,from_depth_mm=depth,target_depth_mm=target,physical_dt_s=(target-depth)/p.loading_rate))
            try:y,row=execute_step(p,x,depth,target,min(deadline,before+base['attempt_seconds']),deadline)
            except (LoadStepFailure,TimeoutError) as exc:
                kind=failure_kind(exc,deadline);witness=store.rejected(p,x,exc,kind)
                result['attempts'].append(dict(from_mm=depth,target_mm=target,dt_s=(target-depth)/p.loading_rate,failure_kind=kind,error=str(exc),elapsed_s=time.monotonic()-before,witness=witness));consecutive+=1
                if consecutive>=p.maximum_consecutive_rejections:raise TimeoutError('Local physical refinement computation cap reached') from exc
                if not physical_cutback(kind) or target-depth<=base['minimum_increment_mm']+4*np.spacing(max(1.,abs(depth))):raise
                increment=reduced_increment(p,base,target-depth,kind);continue
            consecutive=0;store.accepted(row);x,depth=y,target;row.update(time_s=(depth-p.start)/p.loading_rate,dt_s=p.material_dt)
            route.observables(p,x,row);name=f'state_{len(result["steps"]):05d}.npz';np.savez_compressed(out/name,**route.snapshot(p,x,old,depth))
            row.update(state_file=name,state_sha256=digest(out/name));result['steps'].append(row);parent_ref=dict(file=name,sha256=row['state_sha256'])
            write_json(out/'progress.json',dict(depth_mm=depth,accepted=len(result['steps']),inherited=len(original['steps']),surface_connected_mm=row['surface_connected_failed_length_mm']))
            print('ACCEPTED '+json.dumps({k:row[k] for k in ('depth_mm','reaction_fz_n','fully_failed_faces')}),flush=True)
            increment=max(base['minimum_increment_mm'],min(base['increment_mm'],increment*route.driver.accepted_step_growth(base,row)))
        result['status']='completed_pilot'
    except (RuntimeError,ValueError,TimeoutError,FloatingPointError) as exc:
        result.update(status='stopped',error=str(exc),failure_kind=failure_kind(exc,deadline),next_increment_mm=increment)
    result.update(elapsed_s=time.monotonic()-start,source_unchanged=source==growth.sources() and helper==digest(Path(__file__)))
    growth.finish(out,result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('parent');ap.add_argument('parent_audit');ap.add_argument('--out',required=True);ap.add_argument('--stop-depth',type=float,required=True);ap.add_argument('--seconds',type=float,default=1800);add_thread_arguments(ap);args=ap.parse_args()
    run(args.parent,args.parent_audit,args.out,args.stop_depth,args.seconds,**thread_arguments(args))
