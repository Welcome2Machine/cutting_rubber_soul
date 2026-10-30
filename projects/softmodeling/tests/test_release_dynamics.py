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


def test_native_midpoint_friction_energy_identity_can_hide_spurious_reversal():
    """Known failure mechanism, not a claim that the production method is fixed.

    A frozen native contact reduced to one dissipative velocity coordinate has
    m*v_dot=-F(v), with v*F(v)>0. Its continuous solution cannot cross zero.
    The original theta=.5 rule can reverse v while closing its energy identity.
    """
    from scipy.optimize import brentq
    from prototype.ipc_reference import IPCProblem
    p=problem(candidate());x=p.rest.copy()
    x[p.n:]=p.prescribed_blade_positions(-.0014);x[p.n:,0]+=.003
    lag=IPCProblem.lag(p,x)
    potential=p.ipc.FrictionPotential(p.friction_velocity_epsilon)
    direction=np.zeros_like(x);direction[:p.n,0]=1.
    direction[p.mesh.bottom_nodes]=0.
    d=direction[p.surface]
    mass=float(p.mass_dof@(direction.ravel()**2))
    def drag(v):return float(potential.gradient(lag,p.cmesh,v*d)@d.ravel())
    v0=1.;assert drag(.5)>0
    h=10*mass*v0/drag(.5)
    residual=lambda v1:mass*(v1-v0)/h+drag((v0+v1)/2)
    v1=brentq(residual,-v0,0.,xtol=1e-15,rtol=1e-14)
    assert v1<-.9*v0
    assert abs(residual(v1))<1e-8  # stricter than the unchanged physical force gate
    average=(v0+v1)/2
    kinetic=.5*mass*(v1*v1-v0*v0)
    work=h*drag(average)*average
    assert work>=0 and abs(kinetic+work)<1e-12
    # The contradiction is temporal accuracy, not a force root or an energy gap.
    for speed in (.01,.1,1.): assert speed*drag(speed)>0
