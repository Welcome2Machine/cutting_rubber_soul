"""Bounded load-homotopy trace on a checked frozen candidate.

Auxiliary loads guide root finding only. They are zeroed in finally; no energy
or physical history is accepted under a nonzero load. The original straight
anchor chord and all local CCD constraints remain mandatory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numerical_runtime import numerical_runtime, add_thread_arguments, thread_arguments
from prototype.finite_increment_cut import FrozenVirtualCrack
from prototype.force_globalization import solve
from prototype.ipc_reference import LoadStepFailure
from run_support import write_json
from virtual_friction_work import virtual_work, interval_decision
from .frozen_scene import load_scene
from .diagnose_finite_increment import virtual_state
from .virtual_work_audit import audit_work
from .run_zero_gap_cut import hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=480.)
    parser.add_argument('--corrector', choices=['newton', 'newton_ptc'], default='newton')
    add_thread_arguments(parser)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 500:
        raise ValueError('Explicit branch budget 1..500 s required')
    args.output.mkdir(parents=True, exist_ok=False)
    begin=time.monotonic();deadline=begin+args.seconds;source=hashes()
    with numerical_runtime(**thread_arguments(args)) as runtime:
        p, base, archived, before, _, scene = load_scene(args.case, 'production')
        v=FrozenVirtualCrack(p,before,archived.previous_material_metric,archived.previous_history,
            p.released,archived.virtual_removed,p.ffm_epsilon)
        v.ffm_chord_partition='geometry';v.ffm_chord_max_queries=256
        v.set_primary_increment(before,np.zeros_like(before))
        initial=v.evaluate(before,v.lag(before),tangent=False)[1]
        load=np.zeros(v.total);load[v.base_free]=initial[v.base_free]
        x=before.copy();fraction=1.;step=.05
        report=dict(schema='softcut-frozen-load-branch/1',scene=scene,runtime=runtime,
            source_hashes=source,diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            loading_started=False,physical_history_committed=False,stages=[],status='unknown',
            auxiliary_load_is_solver_only=True,unchanged_anchor_and_straight_path=True,
            corrector=args.corrector)
        try:
            while fraction>0 and len(report['stages'])<200:
                if time.monotonic()>=deadline:raise TimeoutError('Branch budget')
                target=max(0.,fraction-step);v.virtual_solver_load=target*load
                v.set_primary_increment(x,x-v.previous);v.force_tolerance_n=1e-6/np.sqrt(2)
                v.max_iterations=40;start=time.monotonic();row=dict(from_fraction=fraction,target_fraction=target,accepted=False)
                report['stages'].append(row)
                try:
                    try:
                        value,_=solve(v,x.copy(),scene['physical_seed']['depth_mm'],min(deadline,start+15.))
                        row['method']='newton'
                    except (LoadStepFailure,TimeoutError) as exc:
                        if args.corrector!='newton_ptc':
                            raise
                        row['newton_error']=str(exc)
                        seed=v.trial.copy();v.max_iterations=80
                        v.set_primary_increment(seed,seed-v.previous)
                        value,_=v.solve_friction(seed,scene['physical_seed']['depth_mm'],min(deadline,start+30.))
                        row['method']='newton_then_ptc'
                    _,g,_,j=v.evaluate(value,v.lag(value),tangent=False);v.update_constraint_free(value,g)
                    residual=np.sqrt(2)*float(np.linalg.norm(g[v.free]))
                    if residual>1e-6 or j<=0 or not v.accepted_chord_is_safe(value):
                        raise LoadStepFailure('Auxiliary original-force/CCD gate')
                    row.update(accepted=True,auxiliary_residual_n=residual,
                        original_residual_n=float(np.sqrt(2)*np.linalg.norm((g+target*load)[v.free])),
                        min_j=float(j),certificate=v.virtual_chord_certificate)
                    x,fraction=value,target;step=min(.1,step*1.4)
                    np.savez_compressed(args.output/f'stage_{len(report["stages"]):03d}.npz',positions=x,fraction=fraction)
                except (LoadStepFailure,TimeoutError) as exc:
                    row.update(error=str(exc),certificate=v.virtual_chord_certificate)
                    path=args.output/f'failed_stage_{len(report["stages"]):03d}.npz'
                    np.savez_compressed(path,positions=v.trial,accepted_positions=x,
                        virtual_anchor=v.previous,auxiliary_load=v.virtual_solver_load,
                        frozen_metric=v.previous_material_metric,frozen_history=v.previous_history,
                        removed=v.virtual_removed,target_fraction=target)
                    row.update(failure_file=path.name,failure_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    step=.5*(fraction-target)
                    if step<1e-6:raise LoadStepFailure('Load branch step below 1e-6; no absence-of-root claim')
                row['elapsed_s']=time.monotonic()-start
                report['remaining_fraction']=fraction
                write_json(args.output/'diagnostic.json',report)
        except (LoadStepFailure,TimeoutError) as exc:
            report['error']=str(exc)
        finally:
            v.virtual_solver_load[:]=0.
        v.set_primary_increment(x,x-v.previous)
        report['final_state']=virtual_state(v,x)
        report['remaining_fraction']=fraction
        if fraction==0 and report['final_state']['residual_n']<=1e-4 and report['final_state']['min_j']>0 and report['final_state']['chord_safe']:
            report['status']='equilibrated'
            work=virtual_work(v,before,x,deadline);audit=audit_work(v,before,x,deadline)
            report.update(work=work,independent_work=audit)
            if work['work_n_mm'] is not None:
                report['decision']=interval_decision(base.stored(before)-v.stored(x),p.toughness*float(p.seam_area[v.virtual_removed].sum()),work['work_n_mm'],work['empirical_uncertainty_n_mm'],work['converged'])
        np.savez_compressed(args.output/'final.npz',before=before,positions=x,load=load,
            frozen_metric=v.previous_material_metric,frozen_history=v.previous_history,removed=v.virtual_removed)
    report['source_unchanged']=source==hashes();report['elapsed_s']=time.monotonic()-begin
    write_json(args.output/'diagnostic.json',report)
    print(json.dumps(dict(status=report['status'],fraction=fraction,final=report['final_state'],error=report.get('error'))))


if __name__=='__main__':main()
