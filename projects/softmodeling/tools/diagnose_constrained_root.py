"""Bounded original-equation test of a chord-constrained Newton search."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import numerical_runtime,add_thread_arguments,thread_arguments
from prototype.ipc_reference import LoadStepFailure
from run_support import write_json
from virtual_constrained_root import solve_constrained,signed_chord_gap
from virtual_friction_work import virtual_work,interval_decision
from .frozen_scene import load_scene
from .diagnose_finite_increment import virtual_state
from .virtual_work_audit import audit_work
from .run_zero_gap_cut import hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gap-factor',type=float,default=10.);parser.add_argument('--ccd-tolerance',type=float)
    add_thread_arguments(parser);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);source=hashes()
    with numerical_runtime(**thread_arguments(args)) as runtime:
        p,base,v,before,after,scene=load_scene(args.case,'production')
        original_tolerance=v.ffm_ccd_tolerance
        tolerance=original_tolerance if args.ccd_tolerance is None else args.ccd_tolerance
        if not 0<tolerance<=original_tolerance or not 1<=args.gap_factor<=100:
            raise ValueError('CCD may only tighten; numerical gap factor must be 1..100')
        v.ffm_ccd_tolerance=tolerance;v.virtual_narrow_phase_ccd=v.ipc.TightInclusionCCD(tolerance=tolerance,
            max_iterations=v.ipc.TightInclusionCCD.DEFAULT_MAX_ITERATIONS,
            conservative_rescaling=v.ipc.TightInclusionCCD.DEFAULT_CONSERVATIVE_RESCALING)
        v.ffm_chord_partition='geometry';v.ffm_chord_max_queries=256
        gap,gradient,feature=signed_chord_gap(before,after,(49,50),48)
        rng=np.random.default_rng(842);direction=np.zeros(v.total);direction[v.free]=rng.normal(size=len(v.free));direction=direction.reshape(after.shape);direction/=np.linalg.norm(direction)
        eps=1e-6;fd=(signed_chord_gap(before,after+eps*direction,(49,50),48)[0]-signed_chord_gap(before,after-eps*direction,(49,50),48)[0])/(2*eps)
        derivative=float(np.sum(gradient*direction));helper=Path(__file__).resolve().parents[1]/'virtual_constrained_root.py'
        report=dict(scene=scene,runtime=runtime,status='unknown',original_ccd_tolerance=original_tolerance,
            diagnostic_ccd_tolerance=tolerance,search_gap_factor=args.gap_factor,source_hashes=source,solver_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
            diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            gap_derivative=dict(analytic=derivative,central_difference=fd,absolute_error=abs(fd-derivative)),
            physical_history_committed=False,loading_started=False)
        try:
            x,trace=solve_constrained(v,after,scene['physical_seed']['depth_mm'],time.monotonic()+120,gap_floor=args.gap_factor*tolerance)
            report.update(status='equilibrated',endpoint=virtual_state(v,x))
            work=virtual_work(v,before,x,time.monotonic()+60);audit=audit_work(v,before,x,time.monotonic()+60)
            report.update(work=work,independent_work=audit,
                work_agreement=work['work_n_mm'] is not None and abs(work['work_n_mm']-audit['work_n_mm'])<=work['empirical_uncertainty_n_mm']+audit['empirical_uncertainty_n_mm']+1e-7)
            if work['work_n_mm'] is not None:
                report['decision']=interval_decision(base.stored(before)-v.stored(x),p.toughness*float(p.seam_area[v.virtual_removed].sum()),work['work_n_mm'],work['empirical_uncertainty_n_mm'],work['converged'])
        except (LoadStepFailure,TimeoutError,ValueError) as exc:
            x=v.trial.copy();report.update(error=str(exc),endpoint=virtual_state(v,x))
        report['trace']=getattr(v,'constrained_root_trace',[])
        np.savez_compressed(args.output/'fields.npz',before=before,after=x,original_after=after,
            frozen_metric=v.previous_material_metric,frozen_history=v.previous_history,removed=v.virtual_removed)
    report['source_unchanged']=source==hashes() and report['solver_sha256']==hashlib.sha256(helper.read_bytes()).hexdigest()
    write_json(args.output/'diagnostic.json',report)
    print(json.dumps(dict(status=report['status'],endpoint=report['endpoint'],decision=report.get('decision'),derivative=report['gap_derivative'])))


if __name__=='__main__':main()
