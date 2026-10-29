"""Deterministic diagnostic multistarts on one archived candidate, never insertion."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from numerical_runtime import numerical_runtime,add_thread_arguments,thread_arguments
from run_support import write_json
from .frozen_scene import load_scene
from .diagnose_finite_increment import virtual_state
from prototype.finite_increment_cut import FrozenVirtualCrack
from prototype.ipc_reference import LoadStepFailure
from virtual_path_probe import efficient_chord
from virtual_friction_work import virtual_work,interval_decision


def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--branch-seconds',type=float,default=90);add_thread_arguments(parser);args=parser.parse_args()
    if not 1<=args.branch_seconds<=100:raise ValueError('Each diagnostic branch <=100 s')
    args.output.mkdir(parents=True,exist_ok=False)
    with numerical_runtime(**thread_arguments(args)) as runtime:
        p,base,archived,before,after,scene=load_scene(args.case,'production')
        blend=before+.95*(after-before);ids=np.flatnonzero(archived.virtual_removed);shift=.2*p.ffm_epsilon # replaced below by reference seam scale
        spacing=float(np.median(np.diff(p.rest[p.seam,1])));shift=.2*spacing
        seeds=[('original_before',before.copy()),('saved_trial_blend',blend.copy())]
        for sign in (-1,1):
            seed=blend.copy();seed[p.seam[ids],0]=np.maximum(p.config.drop_s_mm,seed[p.seam[ids],0]+sign*shift)
            seeds.append((f'opening_predictor_{sign:+d}',seed))
        schedule=Path(__file__).parents[1]/'virtual_path_probe.py';schedule_hash=hashlib.sha256(schedule.read_bytes()).hexdigest()
        report=dict(scene=scene,runtime=runtime,branches=[],seed_order=[n for n,_ in seeds],branch_budget_s=args.branch_seconds,
            predictor_shift_mm=shift,predictor_is_solver_only=True,physical_insertion_permitted=False,
            schedule_source_sha256=schedule_hash,diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        for index,(name,seed) in enumerate(seeds):
            start=time.monotonic();deadline=start+args.branch_seconds
            v=FrozenVirtualCrack(p,before,archived.previous_material_metric,archived.previous_history,p.released,archived.virtual_removed,p.ffm_epsilon)
            v.ffm_chord_max_queries=256
            def covered(x):
                v.virtual_chord_certificate=efficient_chord(v,x,256)
                return v.virtual_chord_certificate['complete']
            v.accepted_chord_is_safe=covered
            # A failed start is shortened along the same anchor ray; no histories change.
            halvings=0
            while not covered(seed) and halvings<12:
                seed=before+.5*(seed-before);halvings+=1
            rec=dict(name=name,seed_halvings=halvings,accepted=False);report['branches'].append(rec)
            np.savez_compressed(args.output/f'seed_{index}.npz',positions=seed)
            try:
                value,g,norm=v.equilibrium(seed,scene['physical_seed']['depth_mm'],deadline)
                state=virtual_state(v,value);rec.update(accepted=True,state=state)
                work=virtual_work(v,before,value,time.monotonic()+30)
                rec['work']=work
                if work['work_n_mm'] is not None:
                    rec['energy_decision']=interval_decision(base.stored(before)-v.stored(value),p.toughness*float(p.seam_area[ids].sum()),work['work_n_mm'],work['empirical_uncertainty_n_mm'],work['converged'])
            except (LoadStepFailure,TimeoutError,ValueError) as exc:
                value=v.trial.copy();rec.update(error=str(exc),state=virtual_state(v,value))
            rec.update(elapsed_s=time.monotonic()-start,solver=getattr(v,'virtual_solver_method',None),phases=v.virtual_solver_phases,
                last_chord_certificate=v.virtual_chord_certificate)
            np.savez_compressed(args.output/f'branch_{index}.npz',positions=value,before=before,metric=v.previous_material_metric,history=v.previous_history,removed=v.virtual_removed)
            write_json(args.output/'diagnostic.json',report)
        report['schedule_source_unchanged']=schedule_hash==hashlib.sha256(schedule.read_bytes()).hexdigest()
    write_json(args.output/'diagnostic.json',report)
    print(json.dumps([dict(name=q['name'],accepted=q['accepted'],residual=q['state']['residual_n'],energy=q.get('energy_decision')) for q in report['branches']]))


if __name__=='__main__':main()
