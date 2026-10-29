import time
import numpy as np
import pytest
from prototype.finite_increment_cut import FrozenVirtualCrack
from prototype.ipc_reference import LoadStepFailure
from prototype.zero_gap_cut import EnergyResolutionRequired
from step_control import (VirtualRootIncomplete, VirtualWorkUnresolved,
    GlobalComputeBudgetExhausted, execute_step, failure_kind, physical_cutback,
    reduced_increment)
from tools.run_zero_gap_cut import problem
from test_finite_increment import config


@pytest.mark.parametrize('error,kind,cutback', [
    (LoadStepFailure('force'), 'physical_nonlinear_failure', True),
    (EnergyResolutionRequired('energy'), 'physical_energy_resolution', True),
    (VirtualRootIncomplete('root'), 'virtual_candidate_root_incomplete', False),
    (VirtualWorkUnresolved('work'), 'virtual_work_unresolved', False),
    (GlobalComputeBudgetExhausted('global'), 'global_compute_budget_exhausted', False),
    (TimeoutError('physical wall time'), 'physical_solve_budget_exhausted', False)])
def test_failure_classification_does_not_confuse_compute_with_dt(error, kind, cutback):
    assert failure_kind(error, time.monotonic()+60) == kind
    assert physical_cutback(kind) is cutback
    p = problem(config())
    cfg = dict(config(), dynamic_cutback=True)
    if cutback:
        new = reduced_increment(p, cfg, .01, kind)
        assert new <= .005
        if kind == 'physical_energy_resolution':
            assert new == .005  # no unrelated inertial jump
    else:
        with pytest.raises(ValueError, match='cannot change physical dt'):
            reduced_increment(p, cfg, .01, kind)


def test_virtual_timeout_rolls_back_whole_step_with_same_dt(monkeypatch):
    p = problem(config())
    saved = {k:getattr(p,k).copy() for k in ('history','material_metric','velocity','released','active_contact')}
    original = p.propose_insertion
    monkeypatch.setattr(p, 'propose_insertion', lambda x,t,d,l: original(x,np.ones(len(p.seam)),d,l))
    calls = []
    def fail(v, x, d, deadline):
        calls.append((p.material_dt, v.previous.copy(), v.previous_material_metric.copy()))
        raise TimeoutError('controlled virtual solver delay')
    monkeypatch.setattr(FrozenVirtualCrack, 'equilibrium', fail)
    target = p.start+.0002
    with pytest.raises(VirtualRootIncomplete):
        execute_step(p,p.rest,p.start,target,time.monotonic()+20,time.monotonic()+100)
    assert len(calls) == 3
    for dt,anchor,metric in calls:
        assert dt == pytest.approx(.001)
        np.testing.assert_array_equal(anchor,calls[0][1])
        np.testing.assert_array_equal(metric,calls[0][2])
    for k,v in saved.items():
        np.testing.assert_array_equal(getattr(p,k),v)
    assert not p.ffm_pending and p.ffm_failure['metadata']['candidate_status'] == 'unknown'
    assert not hasattr(p,'global_compute_deadline')


def test_completed_decisions_survive_machine_delay_without_changing_physical_path(monkeypatch):
    original_equilibrium = FrozenVirtualCrack.equilibrium
    clock = time.monotonic
    outputs = []
    for delayed in (False,True):
        p = problem(config())
        original = p.propose_insertion
        offset = [0.]
        calls = [0]
        def root(v,x,d,limit):
            calls[0] += 1
            if delayed and calls[0] == 1:
                offset[0] += 30.  # exceeds physical attempt allowance, not global cap
                raise TimeoutError('same work on slower machine')
            return original_equilibrium(v,x,d,limit)
        with monkeypatch.context() as patch:
            patch.setattr(time,'monotonic',lambda:clock()+offset[0])
            patch.setattr(FrozenVirtualCrack,'equilibrium',root)
            patch.setattr(p,'propose_insertion',lambda x,t,d,l:original(x,np.ones(len(p.seam)),d,l))
            x,old=p.rest.copy(),p.start
            path=[]
            for step in range(2):
                target=old+.0002
                x,rec=execute_step(p,x,old,target,time.monotonic()+20,time.monotonic()+100)
                path.append((target,p.material_dt,rec['reaction_fz_n'],x.copy(),p.history.copy(),p.material_metric.copy()))
                old=target
            outputs.append(path)
    for fast,slow in zip(*outputs):
        for a,b in zip(fast,slow):
            np.testing.assert_array_equal(a,b)
