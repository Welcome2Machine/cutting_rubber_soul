import copy
import pytest
from tools.event_partition import Partition,verify

CONTROL=dict(maximum_bisections=12,recovery='original_physical_gates_only',maximum_step_growth=False)

def test_bisection_keeps_outer_clock_and_is_independently_replayed():
    plan=dict(dt_s=[1.],window_s=2.,adaptive_dyadic=CONTROL)
    p=Partition([1.,2.],CONTROL)
    assert p.decision('rejected','physical_energy_resolution')
    p.decision('accepted');p.decision('accepted');p.decision('accepted')
    a,complete=verify(p.saved(),plan,1.)
    assert complete and a.tolist()==[.5,1.,2.]
    bad=copy.deepcopy(p.saved());bad['trace'][0]['failure_kind']='physical_solver_accuracy_incomplete'
    with pytest.raises(ValueError):verify(bad,plan,1.)

def test_unknown_numerical_accuracy_does_not_change_physical_dt():
    p=Partition([1.],CONTROL);before=p.pending[0]
    assert not p.decision('rejected','physical_solver_accuracy_incomplete')
    assert p.pending[0]==before and not p.accepted
