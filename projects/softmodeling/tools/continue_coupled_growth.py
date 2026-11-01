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


def stage_records(raw):
    """Recover stage targets without rewriting any accepted/rejected record.

    Legacy single-stage paths are supported. Legacy continuations require the
    hash-bound parent result; otherwise their earlier truncations are unknown.
    """
    if 'continuation_stages' in raw:return copy.deepcopy(raw['continuation_stages'])
    prior=[];begin=attempt_begin=0;link=raw.get('continuation')
    if link:
        file=Path(link['parent_directory'])/'result.json'
        if digest(file)!=link['parent_result_sha256']:raise ValueError('Legacy stage parent identity mismatch')
        parent=json.loads(file.read_text());begin=len(parent['steps']);attempt_begin=len(parent['attempts'])
        if raw['steps'][:begin]!=parent['steps'] or raw['attempts'][:attempt_begin]!=parent['attempts']:
            raise ValueError('Legacy stage prefix changed')
        prior=stage_records(parent)
    prior.append(dict(stage_index=len(prior),requested_stop_depth_mm=raw['requested_stop_depth_mm'],
        inherited_states=begin,state_range=[begin,len(raw['steps'])],attempt_range=[attempt_begin,len(raw['attempts'])],
        config_sha256=raw['config_sha256'],source_hashes=raw['source_hashes'],
        parent_result_sha256=link['parent_result_sha256'] if link else None,
        parent_audit_sha256=link['parent_audit_sha256'] if link else None,
        initial_controller_increment_mm=link['restored_controller_increment_mm'] if link else raw['config']['base']['increment_mm'],
        final_controller_increment_mm=raw.get('next_increment_mm'),
        status=raw['status'],record_origin='legacy hash-bound reconstruction'))
    return prior


def controller_history(p,raw):
    base=raw['config']['base'];inc=base['increment_mm'];depth=p.start;end=attempt_end=0
    stages=stage_records(raw)
    for order,stage in enumerate(stages):
        begin,stop=stage['state_range'];ab,ae=stage['attempt_range']
        if stage['stage_index']!=order or begin!=end or ab!=attempt_end or not begin<=stop<=len(raw['steps']) or not ab<=ae<=len(raw['attempts']):
            raise ValueError('Invalid or incomplete ordered stage ranges')
        if stage['inherited_states']!=begin or stage['config_sha256']!=raw['config_sha256'] or stage['source_hashes']!=raw['source_hashes']:
            raise ValueError('Stage input/prefix identity mismatch')
        if abs(inc-stage['initial_controller_increment_mm'])>1e-14:raise ValueError('Stage controller handoff mismatch')
        target=stage['requested_stop_depth_mm']
        if target<=depth:raise ValueError('Stage target must advance')
        if begin:
            parent_ref=stage.get('parent_state')
            if parent_ref and parent_ref!={k:raw['steps'][begin-1][k] for k in ('state_file','state_sha256')}:
                raise ValueError('Stage accepted parent identity mismatch')
            if not stage.get('parent_result_sha256') or not stage.get('parent_audit_sha256'):raise ValueError('Unbound inherited stage')
        attempts=iter(raw['attempts'][ab:ae]);pending=next(attempts,None)
        for row in raw['steps'][begin:stop]:
            while pending is not None and abs(pending['from_mm']-depth)<=1e-12:
                if not physical_cutback(pending['failure_kind']):raise ValueError('Nonphysical rejection cannot precede accepted suffix')
                inc=reduced_increment(p,base,pending['target_mm']-pending['from_mm'],pending['failure_kind']);pending=next(attempts,None)
            expected=p.next_loading_depth(depth,inc,target) if hasattr(p,'next_loading_depth') else min(depth+inc,target)
            if abs(expected-row['depth_mm'])>1e-12:raise ValueError('Cannot reconstruct exact accepted controller schedule')
            inc=max(base['minimum_increment_mm'],min(base['increment_mm'],inc*route.driver.accepted_step_growth(base,row)));depth=row['depth_mm']
        while pending is not None:
            if abs(pending['from_mm']-depth)>1e-12:raise ValueError('Unordered terminal attempts')
            if physical_cutback(pending['failure_kind']):inc=reduced_increment(p,base,pending['target_mm']-pending['from_mm'],pending['failure_kind'])
            elif pending['failure_kind']!='global_compute_budget_exhausted':raise ValueError('Unresolved nonbudget failure is not resumable')
            pending=next(attempts,None)
        if stage['status'].startswith('completed') and abs(depth-target)>1e-12:raise ValueError('Stage endpoint mismatch')
        saved=stage.get('final_controller_increment_mm')
        if saved is not None and abs(inc-saved)>1e-14:raise ValueError('Saved/replayed stage controller mismatch')
        stage['final_controller_increment_mm']=inc
        end=stop;attempt_end=ae
    if end!=len(raw['steps']) or attempt_end!=len(raw['attempts']):raise ValueError('Stage ledger does not cover complete prefix')
    if abs(stages[-1]['requested_stop_depth_mm']-raw['requested_stop_depth_mm'])>1e-12:raise ValueError('Final stage target mismatch')
    if 'next_increment_mm' in raw and abs(inc-raw['next_increment_mm'])>1e-14:raise ValueError('Saved/replayed controller mismatch')
    return inc,stages


def controller_increment(p,raw):return controller_history(p,raw)[0]


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


def replay_saved_step(parent,index,out,seconds=90):
    """Restart an actual accepted parent and reproduce its already saved child.

    This is a diagnostic, never a suffix. Covers nonzero Maxwell/cohesive
    history and graph changes that a small split-path regression may not reach.
    """
    parent=Path(parent);out=Path(out);raw=json.loads((parent/'result.json').read_text())
    if raw['source_hashes']!=growth.sources() or not 2<=index<len(raw['steps']):raise ValueError('Same-source saved child and two predecessors required')
    p=growth.problem(raw['config']);records=[]
    for row in raw['steps'][index-2:index]:
        file=parent/row['state_file']
        if digest(file)!=row['state_sha256']:raise ValueError('Changed replay parent')
        with np.load(file,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
        records.append((row['depth_mm'],s['positions']))
    x=restore(p,s,records);old=p.released.copy();target=raw['steps'][index];deadline=time.monotonic()+seconds
    y,row=execute_step(p,x,records[-1][0],target['depth_mm'],deadline,deadline)
    actual=route.snapshot(p,y,old,target['depth_mm']);np.savez_compressed(out,**actual)
    with np.load(parent/target['state_file'],allow_pickle=False) as z:
        differences={k:float(np.max(np.abs(np.asarray(actual[k],float)-np.asarray(z[k],float)),initial=0)) for k in z.files}
        exact=all(np.array_equal(actual[k],z[k]) for k in z.files)
    return dict(index=index,parent_sha256=raw['steps'][index-1]['state_sha256'],reference_state_sha256=target['state_sha256'],
        replay_state_sha256=digest(out),all_saved_arrays_bitwise_equal=exact,array_maximum_absolute_differences=differences,
        original_events=target.get('face_events',[]),replayed_events=row.get('face_events',[]),
        parent_nonzero_material_history=bool(np.any(s['material_metric']!=np.eye(s['material_metric'].shape[-1]))),
        parent_new_cohesive_history=bool(np.any(s['history'][~p.precracked_faces]>0)))


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
    p=growth.problem(original['config']);base=original['config']['base'];increment,stages=controller_history(p,original)
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
    stages.append(dict(stage_index=len(stages),requested_stop_depth_mm=stop_depth,
        inherited_states=len(original['steps']),state_range=[len(original['steps']),len(original['steps'])],
        attempt_range=[len(original['attempts']),len(original['attempts'])],config_sha256=signature,source_hashes=source,
        parent_result_sha256=digest(parent/'result.json'),parent_audit_sha256=digest(Path(parent_audit)),
        parent_state={k:original['steps'][-1][k] for k in ('state_file','state_sha256')},
        restored_accepted_array_sha256=array_hash(accepted_fields(p,x)),
        predictor_states=[{k:r[k] for k in ('depth_mm','state_file','state_sha256')} for r in original['steps'][-2:]],
        initial_controller_increment_mm=increment,final_controller_increment_mm=None,status='running',record_origin='explicit continuation'))
    result['continuation_stages']=stages
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
    stages[-1].update(state_range=[stages[-1]['inherited_states'],len(result['steps'])],attempt_range=[stages[-1]['attempt_range'][0],len(result['attempts'])],final_controller_increment_mm=increment,status=result['status'])
    result.update(elapsed_s=time.monotonic()-start,source_unchanged=source==growth.sources() and helper==digest(Path(__file__)))
    growth.finish(out,result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('parent');ap.add_argument('parent_audit');ap.add_argument('--out',required=True);ap.add_argument('--stop-depth',type=float,required=True);ap.add_argument('--seconds',type=float,default=1800);add_thread_arguments(ap);args=ap.parse_args()
    run(args.parent,args.parent_audit,args.out,args.stop_depth,args.seconds,**thread_arguments(args))
