"""Analytic memory/limits for the reduced mechanics diagnostic, not FE validation."""
import numpy as np
import pytest
from tools.check_force_history_mechanism import maxwell_history,positive_branch_interval


def test_ramp_matches_analytic_memory_for_irregular_timesteps():
    t=np.array([0.,.01,.05,.2,.7,1.3,2.])
    for tau in (.001,.1,1.,100.):
        np.testing.assert_allclose(maxwell_history(t,3*t,tau),
            -3*tau*np.expm1(-t/tau),rtol=1e-13,atol=1e-14)


def test_hold_relaxes_existing_memory_without_new_load():
    t=np.array([0.,1.,2.,3.]);f=np.array([0.,5.,5.,5.])
    h=maxwell_history(t,f,.5)
    np.testing.assert_allclose(h[2:]/h[1:-1],np.exp(-2),rtol=1e-14)
    # A decaying multiplier of the full force is already wrong during ramp.
    assert h[1]>5*np.exp(-2)


def test_elastic_and_fast_relaxation_limits_and_tare():
    t=np.linspace(0,1,101);f=7+3*t**2
    np.testing.assert_allclose(maxwell_history(t,f,1e12),f-f[0],rtol=1e-11,atol=1e-11)
    assert maxwell_history(t,f,1e-10).max()<1e-8
    np.testing.assert_allclose(maxwell_history(t,f,.4),maxwell_history(t,f-7,.4))


@pytest.mark.parametrize('time,force,tau',[
    ([0,0],[0,1],1),([0,1],[0,1],0),([0,1],[0,np.nan],1),([0,1],[0],1)])
def test_invalid_histories_rejected(time,force,tau):
    with pytest.raises(ValueError):maxwell_history(time,force,tau)


def test_positive_branch_range_intersection_and_zero_branch():
    np.testing.assert_allclose(positive_branch_interval([1,2],[1,2],[2,5],[4,8]),[1.5,3])
    assert positive_branch_interval([1,2],[1,2],[4,2],[5,3]) is None
    assert positive_branch_interval([1,2],[0,1],[2,2],[3,4]) is None
    assert positive_branch_interval([1,2],[0,1],[1,3],[1,4])==[1.,2.]
