"""Separate physical resolution from incomplete numerical decisions.

Wall time is a computation limit, never evidence that a virtual slit is stable.
These types are shared by initial loading and explicit continuation.
"""
import time
import numpy as np
from prototype.ipc_reference import LoadStepFailure


class VirtualRootIncomplete(LoadStepFailure):
    pass


class VirtualWorkUnresolved(LoadStepFailure):
    pass


class GlobalComputeBudgetExhausted(TimeoutError):
    pass


def failure_kind(error, global_deadline):
    from prototype.zero_gap_cut import EnergyResolutionRequired
    if isinstance(error, GlobalComputeBudgetExhausted) or time.monotonic() >= global_deadline:
        return 'global_compute_budget_exhausted'
    if isinstance(error, VirtualRootIncomplete):
        return 'virtual_candidate_root_incomplete'
    if isinstance(error, VirtualWorkUnresolved):
        return 'virtual_work_unresolved'
    if isinstance(error, EnergyResolutionRequired):
        return 'physical_energy_resolution'
    if isinstance(error, TimeoutError):
        # An incomplete physical solve is not a demonstrated instability either.
        return 'physical_solve_budget_exhausted'
    if isinstance(error, LoadStepFailure):
        return 'physical_nonlinear_failure'
    return 'unclassified_failure'


def physical_cutback(kind):
    return kind in ('physical_nonlinear_failure', 'physical_energy_resolution')


def reduced_increment(p, config, attempted, kind):
    if not physical_cutback(kind):
        raise ValueError('Incomplete computation cannot change physical dt')
    increment = max(config['minimum_increment_mm'], attempted/2)
    if config.get('dynamic_cutback', False) and kind == 'physical_nonlinear_failure':
        softening = float(np.max(2*p.seam_area*p.strength/(2*p.toughness/p.strength)))
        increment = max(config['minimum_increment_mm'],
            min(increment, p.loading_rate*.25*np.sqrt(p.mass.min()/softening)))
    return increment


def execute_step(p, x, old, target, attempt_deadline, global_deadline):
    """Preserve the four-argument model API; this limit is only computation state."""
    p.global_compute_deadline = global_deadline
    try:
        return p.step(x, old, target, attempt_deadline)
    finally:
        del p.global_compute_deadline
