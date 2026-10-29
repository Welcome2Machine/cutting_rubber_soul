"""Unresolved quadrature cannot supply a no-fracture certificate."""
import time
import numpy as np
import pytest
from virtual_friction_work import integrate_power, interval_decision


def test_large_unconverged_work_remains_unknown():
    q=interval_decision(1.,.01,100.,90.,False)
    assert q['status']=='unknown'


def test_interval_must_separate_zero_for_either_decision():
    assert interval_decision(1.,.1,.8,.2,True)['status']=='unknown'
    assert interval_decision(1.,.1,.2,.01,True)['status']=='accepted'
    assert interval_decision(1.,.1,1.,.01,True)['status']=='rejected'


def test_partition_resolves_a_narrow_peak_missed_by_global_gauss():
    center=.47453;width=1e-8
    def power(s):return width/((s-center)**2+width**2)
    exact=np.arctan((1-center)/width)+np.arctan(center/width)
    nodes,weights=np.polynomial.legendre.leggauss(129)
    coarse=weights@np.array([power(s) for s in (nodes+1)/2])/2
    assert abs(coarse-exact)>1
    cuts={0.,1.,center}
    for i in range(28):cuts.update((max(0.,center-width*2**i),min(1.,center+width*2**i)))
    q=integrate_power(power,sorted(cuts),deadline=time.monotonic()+5)
    assert q['converged'] and not q['uncertainty_is_rigorous_bound']
    assert q['work_n_mm']==pytest.approx(exact,rel=1e-7)


def test_expired_budget_is_unknown_not_partial_integral():
    q=integrate_power(lambda s:1.,[0.,.5,1.],deadline=time.monotonic()-1)
    assert not q['converged'] and q['work_n_mm'] is None


def test_invalid_partition_or_negative_power_rejected():
    with pytest.raises(ValueError):integrate_power(lambda s:1.,[.1,1.],deadline=time.monotonic()+2)
    with pytest.raises(ValueError):integrate_power(lambda s:-1.,[0.,1.],deadline=time.monotonic()+2)


def test_unresolved_production_candidate_does_not_commit_or_skip(monkeypatch):
    import prototype.finite_increment_cut as module
    from tools.run_zero_gap_cut import problem
    from test_finite_increment import config
    p=problem(config());p.material_dt=.025
    original=p.released.copy()
    def equilibrium(v,seed,depth,deadline):
        g=np.zeros(v.total);g[2*v.seam[-1]]=-.3*v.seam_area[-1]
        v.virtual_solver_method='test'
        return seed.copy(),g,0.
    monkeypatch.setattr(module.FrozenVirtualCrack,'equilibrium',equilibrium)
    monkeypatch.setattr(module.FrozenVirtualCrack,'stored',lambda v,x:0. if v.virtual_removed.any() else 1.)
    monkeypatch.setattr(module,'virtual_work',lambda *args:dict(work_n_mm=100.,empirical_uncertainty_n_mm=1.,converged=False))
    with pytest.raises(module.LoadStepFailure,match='candidate unknown'):
        p.propose_insertion(p.rest,np.full(len(p.seam),.3),0,time.monotonic()+5)
    np.testing.assert_array_equal(p.released,original)
    assert not p.ffm_pending and not p.ffm_checks
    assert p.ffm_failure['metadata']['stage']=='quadrature'
    assert p.ffm_failure['metadata']['candidate_status']=='unknown'
