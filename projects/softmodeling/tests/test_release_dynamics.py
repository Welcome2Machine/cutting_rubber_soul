import time
import numpy as np
import pytest
from test_energy_cut import candidate
from tools.run_zero_gap_cut import problem, audit, snapshot
from release_dynamics import activate, instantaneous_friction, contact_velocity, check_energy_gate
from prototype.zero_gap_cut import EnergyResolutionRequired, extrinsic_response


def test_conditional_activation_preserves_state_mass_and_pays_gc_only_through_law():
    p = problem(candidate()); x = p.rest.copy()
    old = [p.velocity.copy(), p.material_metric.copy(), p.history.copy(), p.mass.copy()]
    event = activate(p, x, [48])
    assert p.released[48] and p.damage_enabled
    assert event['upfront_fracture_debit_n_mm'] == event['auxiliary_work_n_mm'] == 0
    for a,b in zip(old, (p.velocity,p.material_metric,p.history,p.mass)):
        np.testing.assert_array_equal(a,b)
    assert not p.propose_insertion(x, np.full(len(p.seam), 1e6), 0, time.monotonic()+1).any()
    with pytest.raises(ValueError, match='virgin'): activate(p,x,[48])
    cumulative=0.; history=0.
    for opening in (0., .02, .01, .04, .08, .02, .1):
        oldd=extrinsic_response(0.,history,p.strength,p.toughness)['dissipated']
        q=extrinsic_response(opening,history,p.strength,p.toughness)
        cumulative+=p.seam_area[48]*(q['dissipated']-oldd);history=q['history']
    assert cumulative == pytest.approx(event['candidate_gc_capacity_n_mm'])


def test_topology_energy_jump_rejected_and_rolled_back(monkeypatch):
    p=problem(candidate());old=p.released.copy();original=p.barrier
    class ChangedBarrier:
        def __call__(self,*args): return original(*args)+float(p.released[48])
        def gradient(self,*args): return original.gradient(*args)
    # Disable friction construction only in this isolated energy-event test.
    monkeypatch.setattr('release_dynamics.instantaneous_friction',lambda p,*a:np.zeros(p.total))
    p.barrier=ChangedBarrier()
    with pytest.raises(ValueError,match='Unfunded'):activate(p,p.rest,[48])
    np.testing.assert_array_equal(p.released,old)


def test_native_velocity_friction_handoff_is_nonzero_and_scale_invariant():
    p=problem(candidate());x=p.rest.copy()
    # Undeformed specimen, blade just inside the IPC activation zone (no overlap).
    x[p.n:]=p.prescribed_blade_positions(-.0014)
    x[p.n:,0]+=.003
    v=contact_velocity(p,p.velocity)
    v[:p.n,0]=.03;v[p.mesh.bottom_nodes]=0
    g=instantaneous_friction(p,x,v)
    assert np.linalg.norm(g)>1e-6
    for h in (1.,1e-2,1e-5,1e-8):
        np.testing.assert_allclose(instantaneous_friction(p,x,v,h),g,rtol=2e-12,atol=1e-12)
    np.testing.assert_array_equal(instantaneous_friction(p,x,np.zeros_like(v)),np.zeros(p.total))
    np.testing.assert_array_equal(p.velocity,np.zeros_like(p.rest))


def test_real_energy_gate_remains_strict():
    p=problem(dict(candidate(),energy_tolerance=.001,energy_normalization='mechanical_reservoir'))
    from release_dynamics import DEFECT_KEYS
    energy={k:0. for k in DEFECT_KEYS};energy['mechanical_reservoir_n_mm']=1.
    energy['bulk_endpoint_defect_n_mm']=.00100001
    with pytest.raises(EnergyResolutionRequired):check_energy_gate(p,energy)


def test_conditional_control_keeps_real_inertia_and_failed_step_history(monkeypatch):
    p=problem(candidate());x=p.rest.copy();activate(p,x,[48])
    saved=[p.velocity.copy(),p.history.copy(),p.material_metric.copy(),p.released.copy()]
    def fail(*a): raise TimeoutError('controlled numerical timeout')
    monkeypatch.setattr(p,'solve_active',fail)
    with pytest.raises(TimeoutError):p.step(x,p.start,p.start+1e-6,time.monotonic()+2)
    for a,b in zip(saved,(p.velocity,p.history,p.material_metric,p.released)):
        np.testing.assert_array_equal(a,b)
    assert p.material_dt == pytest.approx(1e-6/p.loading_rate)
