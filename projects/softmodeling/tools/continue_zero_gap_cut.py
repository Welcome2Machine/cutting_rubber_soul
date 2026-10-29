"""Explicit bounded continuation of a fully audited, identical cutting model.

New directory, same configuration/source/physical history, no automatic queue.
Parent states are copied byte-for-byte. Every original and appended state is
audited again. This does not turn the parent's budget exhaustion into completion.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
import numpy as np
from numerical_runtime import runtime_entry, add_thread_arguments, thread_arguments
from run_support import write_json
from . import run_zero_gap_cut as driver
from .audit_zero_gap_cut import inspect as inspect_path
from .audit_finite_increment import inspect as inspect_finite
from prototype.ipc_reference import LoadStepFailure
from step_control import execute_step, failure_kind, physical_cutback, reduced_increment


def restore(p, state, records):
    """Restore committed physical variables, never a failed trial."""
    p.material_metric=state['material_metric'].copy()
    p.velocity=state['velocity'].copy();p.history=state['history'].copy()
    p.released=state['released'].copy();p.active_contact=state['active_contact'].copy()
    p.rebuild_contact();p.set_free()
    try:
        p.seed_predictor_history(records)
        p.resume_predictor_policy='last_two_if_current_constraints_admit_them'
    except ValueError:
        # A currently closed unilateral node may have been open in the earlier
        # accepted state. That old state is physical but is not an admissible
        # constrained secant seed for the next solve. Its histories are retained.
        p.seed_predictor_history(records[-1:])
        p.resume_predictor_policy='last_only_after_constraint_change'
    return state['positions'].copy(),float(state['depth_mm'])


@runtime_entry
def continue_run(parent, output, seconds, *, runtime=None):
    if not np.isfinite(seconds) or not 30<=seconds<=3600:
        raise ValueError('Explicit continuation budget must be 30..3600 s')
    parent=parent.resolve();source=driver.hashes();begin=time.monotonic()
    helper_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_path=parent/'result.json';original=json.loads(original_path.read_text(encoding='utf-8'))
    if (original['source_hashes']!=source or not original['source_unchanged']
            or not original['steps'] or original['steps'][-1]['depth_mm']>=original['config']['target_mm']-1e-12):
        raise ValueError('Continuation requires an incomplete identical-source accepted prefix')
    if original.get('failure_kind') != 'global_compute_budget_exhausted' or 'next_increment_mm' not in original:
        raise ValueError('Continuation requires global budget stop and preserved physical controller state')
    signature=hashlib.sha256(json.dumps(original['config'],sort_keys=True).encode()).hexdigest()
    if signature!=original['config_sha256']:
        raise ValueError('Configuration signature differs')
    threads=dict(ipc_threads=runtime['requested']['ipc_threads'],blas_threads=runtime['requested']['blas_threads'])
    parent_audit=inspect_path(parent,min(180.,seconds*.2),**threads)
    if not parent_audit['passed'] or not parent_audit['producer_source_matches_current']:
        raise ValueError('Parent path must pass fresh independent audit')
    finite_audit=None
    if original['config'].get('fracture_criterion')=='finite_increment':
        finite_audit=inspect_finite(parent,min(180.,seconds*.2),**threads)
        if not (finite_audit['passed'] and finite_audit['primary_seed_increment_complete']
                and finite_audit['producer_source_matches_current']):
            raise ValueError('Parent finite-crack decisions must pass fresh audit')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'parent_path_audit.json',parent_audit)
    if finite_audit is not None:write_json(output/'parent_finite_audit.json',finite_audit)
    result=copy.deepcopy(original);prefix_count=len(result['steps'])
    for rec in result['steps']:
        names=[rec['state_file']]+[c['evidence_file'] for c in rec.get('finite_increment_checks',[]) if c['accepted']]
        for name in names:
            path=(parent/name).resolve()
            if path.parent!=parent or path.suffix!='.npz':raise ValueError('Invalid checkpoint path')
            shutil.copyfile(path,output/name)
    result.update(status='running',audits=[],budget_s=seconds,interpreter=sys.executable,thread_runtime=runtime,
        native_ipc_threads=threads['ipc_threads'],blas_threads=threads['blas_threads'],
        started_utc=datetime.now(timezone.utc).isoformat(),source_hashes=source,
        continuation=dict(parent_directory=str(parent),parent_result_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
            parent_status=original['status'],parent_elapsed_s=original['elapsed_s'],
            inherited_accepted_states=prefix_count,source_sha256=helper_sha,
            inherited_history_verified=True,configuration_changed=False))
    for name in ('error','audit_error','summary','accepted_prefix_audited','failure_kind','timeout_scope'):result.pop(name,None)
    write_json(output/'manifest.json',result)
    p=driver.problem(result['config']);p.ipc.set_num_threads(threads['ipc_threads'])
    with np.load(output/result['steps'][-1]['state_file']) as data:state={k:data[k].copy() for k in data.files}
    history=[]
    for rec in result['steps'][-2:]:
        with np.load(output/rec['state_file']) as data:history.append((rec['depth_mm'],data['positions'].copy()))
    x,depth=restore(p,state,history);config=result['config']
    result['continuation']['predictor_policy']=p.resume_predictor_policy
    increment=original['next_increment_mm'];deadline=begin+seconds
    solve_deadline=deadline-max(15.,min(180.,seconds*.2))
    try:
        while depth<config['target_mm']-1e-12:
            if time.monotonic()>=solve_deadline:raise TimeoutError('Reserved continuation audit budget')
            result['next_increment_mm']=increment
            target=min(config['target_mm'],depth+increment);old_released=p.released.copy();attempt=time.monotonic()
            try:
                value,rec=execute_step(p,x,depth,target,min(solve_deadline,attempt+config.get('attempt_seconds',seconds)),solve_deadline)
            except (LoadStepFailure,TimeoutError) as exc:
                failure=p.write_failure_evidence(output,len(result['attempts'])) if hasattr(p,'write_failure_evidence') else None
                kind=failure_kind(exc,solve_deadline)
                result['attempts'].append(dict(failure_kind=kind,physical_dt_s=(target-depth)/p.loading_rate,
                    physical_cutback_allowed=physical_cutback(kind),from_mm=depth,target_mm=target,status='rejected',error=str(exc),elapsed_s=time.monotonic()-attempt,virtual_failure=failure))
                if not physical_cutback(kind) or target-depth<=config['minimum_increment_mm']+4*np.spacing(max(1.,abs(depth))):raise
                increment=reduced_increment(p,config,target-depth,kind)
                continue
            x,depth=value,target
            if hasattr(p,'write_step_evidence'):p.write_step_evidence(output,len(result['steps']),rec)
            state=driver.snapshot(p,x,old_released,depth);name=f'state_{len(result["steps"]):05d}.npz'
            np.savez_compressed(output/name,**state)
            rec.update(state_file=name,state_sha256=hashlib.sha256((output/name).read_bytes()).hexdigest())
            result['steps'].append(rec)
            print('ACCEPTED '+json.dumps({k:rec[k] for k in ('depth_mm','reaction_fz_n','released_nodes','fully_failed_nodes')}),flush=True)
            write_json(output/'progress.json',dict(depth_mm=depth,accepted=len(result['steps']),inherited_accepted=prefix_count,elapsed_s=time.monotonic()-begin))
            increment=max(config['minimum_increment_mm'],min(config['increment_mm'],increment*driver.accepted_step_growth(config,rec)))
            result['next_increment_mm']=increment
        result['status']='completed_2mm_control' if config['target_mm']==2 else 'completed_short_control'
    except (TimeoutError,LoadStepFailure,RuntimeError,FloatingPointError) as exc:
        kind=failure_kind(exc,solve_deadline)
        result.update(status='budget_exhausted' if kind=='global_compute_budget_exhausted' else 'failed',
            error=str(exc),failure_kind=kind,next_increment_mm=increment)
        if hasattr(p,'failed_positions'):
            np.savez_compressed(output/'failed_trial.npz',positions=p.failed_positions,increment=p.failed_increment,
                previous_positions=p.previous,previous_material_metric=p.previous_material_metric,
                material_dt_s=p.material_dt,previous_velocity=p.previous_velocity,previous_history=p.previous_history,
                released=p.failed_released,active_contact=p.failed_active_contact,
                previous_barrier_energy_n_mm=p.previous_barrier_energy,failed_stage=p.failed_stage,target_depth_mm=target)
    result.update(source_unchanged=source==driver.hashes() and helper_sha==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_s=time.monotonic()-begin,accepted_prefix_audited=False)
    driver.report(output,result)
    remaining=max(1.,deadline-time.monotonic())
    audit=inspect_path(output,min(600.,remaining),**threads)
    write_json(output/'continuation_path_audit.json',audit)
    result['audits']=audit['audits'];result['accepted_prefix_audited']=audit['passed']
    result['elapsed_s']=time.monotonic()-begin
    result['source_unchanged']=source==driver.hashes() and helper_sha==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # This audit's input hash refers to the preliminary result, not the final
    # rewritten metadata. An external sidecar can bind the final result bytes.
    result['continuation']['preliminary_result_audit_sha256']=audit['result_sha256']
    result['continuation']['appended_accepted_states']=len(result['steps'])-prefix_count
    driver.report(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=float,required=True)
    add_thread_arguments(parser)
    args=parser.parse_args()
    result=continue_run(args.parent,args.out,args.seconds,**thread_arguments(args))
    write_json(args.out/'result.json',result)
    print(json.dumps(result['summary']))


if __name__=='__main__':main()
