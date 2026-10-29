"""Independent fixed-blade auxiliary-load release; NEVER a production criterion.

See fracture_implementation section 25.2 for the predeclared mechanics and its
initial-friction limitation. Accepted process equilibria, not Newton iterates,
form this path. Auxiliary work is retained even if it dominates the ledger.
"""
import hashlib
import time
from pathlib import Path
import numpy as np
from prototype.finite_increment_cut import FrozenVirtualCrack
from prototype.ipc_reference import LoadStepFailure
from virtual_friction_work import virtual_work
from .virtual_work_audit import audit_work
from .frozen_scene import load_scene
from .run_zero_gap_cut import hashes
from run_support import write_json


def auxiliary_work(load_before, load_after, start, end):
    """Full mirror-domain trapezoidal work; loads are half-domain DOF vectors."""
    return float((load_before+load_after)@(end-start).ravel())


def inspect(case, output, intervals, seconds):
    if intervals not in (8,16,32) or not 1 <= seconds <= 500:
        raise ValueError('Predeclared 8/16/32 equal release intervals; 1..500 seconds')
    output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();deadline=began+seconds;source=hashes()
    p,base,old,before,_,scene=load_scene(case,'production')
    host_history={k:getattr(p,k).copy() for k in ('history','material_metric','velocity','released')}
    v=FrozenVirtualCrack(p,before,old.previous_material_metric,old.previous_history,
        p.released,old.virtual_removed,p.ffm_epsilon)
    v.set_primary_increment(before,np.zeros_like(before))
    gradient=v.evaluate(before,v.lag(before),tangent=False)[1]
    load=np.zeros(v.total);load[v.free]=gradient[v.free]
    x=before.copy();initial_storage=v.stored(x)
    report=dict(schema='softcut-controlled-virtual-release/1',scene=scene,source_hashes=source,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        intervals=intervals,control='F_aux(lambda)=(1-lambda)*initial_free_storage_gradient',
        initial_friction_reset_explicit=True,initial_balance_difference_carried_by_auxiliary_load=True,
        numerical_iterates_are_not_process_states=True,physical_history_committed=False,
        production_crack_decision_permitted=False,steps=[],status='incomplete',
        initial_storage_n_mm=initial_storage,
        candidate_gc_n_mm=p.toughness*float(p.seam_area[v.virtual_removed].sum()))
    w_aux=friction=independent_friction=0.;fraction=0.
    try:
        for i in range(1,intervals+1):
            target=i/intervals
            if time.monotonic()>=deadline:raise TimeoutError('Controlled release budget')
            v.previous=x.copy();v.trial=x.copy();v.set_primary_increment(x,np.zeros_like(x))
            v.friction=v.ipc.FrictionPotential(p.ffm_epsilon*(target-fraction))
            v.virtual_solver_load=(1-target)*load
            value,_,norm=v.equilibrium(x,scene['physical_seed']['depth_mm'],
                min(deadline,time.monotonic()+40.))
            # Equilibrium gate includes this declared auxiliary load. It is NOT
            # an original zero-auxiliary endpoint until lambda reaches one.
            if norm>1e-4:raise LoadStepFailure('Auxiliary force gate')
            applied=v.virtual_solver_load.copy();v.virtual_solver_load[:]=0.
            storage=v.stored(value)
            work=virtual_work(v,x,value,deadline)
            audit=audit_work(v,x,value,deadline)
            agreement=bool(work['converged'] and audit['converged'] and
                abs(work['work_n_mm']-audit['work_n_mm']) <=
                work['empirical_uncertainty_n_mm']+audit['empirical_uncertainty_n_mm']+1e-7)
            if not agreement:raise LoadStepFailure('Unresolved process friction work')
            aux=auxiliary_work((1-fraction)*load,applied,x,value)
            w_aux+=aux;friction+=work['work_n_mm'];independent_friction+=audit['work_n_mm']
            row=dict(lambda_start=fraction,lambda_end=target,auxiliary_residual_n=norm,
                auxiliary_work_n_mm=aux,storage_n_mm=storage,friction=work,independent_friction=audit,
                balance_defect_without_gc_n_mm=w_aux-(storage-initial_storage)-friction,
                chord_certificate=v.virtual_chord_certificate)
            path=output/f'process_{i:03d}.npz'
            np.savez_compressed(path,positions=value,previous_positions=x,auxiliary_load=applied,
                frozen_metric=v.previous_material_metric,frozen_history=v.previous_history,
                removed=v.virtual_removed,process_parameter=target)
            row.update(state_file=path.name,state_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            report['steps'].append(row);x=value;fraction=target
            report['last_parameter']=fraction
            write_json(output/'diagnostic.json',report)
        report['status']='completed_auxiliary_process'
    except (TimeoutError,LoadStepFailure,ValueError,AssertionError) as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        path=output/'failed_process.npz'
        np.savez_compressed(path,positions=v.trial,previous_accepted=x,anchor=v.previous,
            auxiliary_load=v.virtual_solver_load,frozen_metric=v.previous_material_metric,
            frozen_history=v.previous_history,removed=v.virtual_removed,target_parameter=target)
        report.update(failed_state_file=path.name,failed_state_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    finally:
        v.virtual_solver_load[:]=0.
    delta_storage=v.stored(x)-initial_storage
    defect=w_aux-delta_storage-friction
    report['ledger']=dict(external_blade_support_work_n_mm=0.,kinetic_change_n_mm=0.,
        maxwell_dissipation_n_mm=0.,candidate_fracture_dissipation_n_mm=0.,
        auxiliary_work_n_mm=w_aux,storage_change_n_mm=delta_storage,
        friction_n_mm=friction,independent_friction_n_mm=independent_friction,
        balance_defect_without_gc_n_mm=defect,
        balance_defect_if_gc_added_n_mm=defect-report['candidate_gc_n_mm'])
    report.update(last_parameter=fraction,elapsed_s=time.monotonic()-began,source_unchanged=hashes()==source,
        host_history_unchanged=all(np.array_equal(getattr(p,k),value) for k,value in host_history.items()))
    write_json(output/'diagnostic.json',report)
    return report
