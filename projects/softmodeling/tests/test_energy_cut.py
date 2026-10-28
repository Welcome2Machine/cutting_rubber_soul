import time
import numpy as np
import pytest
from mechanics.discrete_gradient import discrete_response, frozen_energy, cohesive_discrete_response
from prototype.zero_gap_cut import extrinsic_response
from prototype.energy_cut import connected_patch_mask, nonlocal_tensile_mask, barrier_time_quadrature
from tools.run_zero_gap_cut import problem, snapshot, audit
from test_zero_gap_cut import config


def candidate():
    return dict(config(), time_integrator='discrete_gradient', nonlinear_solver='newton_lm',
                barrier_quadrature=3, barrier_quadrature_rule='lobatto', insertion_rule='local_strength')


def test_interior_gauss_can_remove_barrier_root_but_lobatto_keeps_divergence():
    # m=h=x0=1, v0=-3, phi=-log(x). Momentum residual is
    # 2*(x1+2)-sum(w/(1-s+s*x1)). With Gauss3 the contact term <=11/3,
    # while inertia >=4: NO admissible root. This is not a solver issue.
    s, w = barrier_time_quadrature(3, 'gauss')
    assert w@(1/(1-s)) == pytest.approx(11/3)
    grid = np.linspace(0, 1, 1001)
    assert np.min(2*(grid+2)-np.sum(w[:, None]/(1-s[:, None]+s[:, None]*grid), axis=0)) > 0
    from scipy.optimize import brentq
    s, w = barrier_time_quadrature(3, 'lobatto')
    residual = lambda x: 2*(x+2)-w@(1/(1-s+s*x))
    root = brentq(residual, 1e-12, 1.)
    assert 0 < root < 1 and abs(residual(root)) < 1e-9
    for n in (2, 3, 5):
        s,w = barrier_time_quadrature(n, 'lobatto')
        assert s[0] == 0 and s[-1] == 1 and w[-1] > 0
        for power in range(2*n-2):
            assert w@s**power == pytest.approx(1/(power+1))


def test_nonlocal_tension_rejects_isolated_peak_and_pure_compression():
    y=np.arange(5.)*.1; a=np.ones(5); r=np.zeros(5,bool)
    assert not nonlocal_tensile_mask(np.full(5,-3.),r,y,a,.2,.3).any()
    assert not nonlocal_tensile_mask(np.array([0,0,.3,0,0]),r,y,a,.2,.3).any()
    np.testing.assert_array_equal(nonlocal_tensile_mask(np.full(5,.3),r,y,a,.2,.3),y>0)


def test_released_residual_is_not_used_as_intact_neighbourhood_stress():
    y=np.arange(5.)*.1; a=np.ones(5); r=np.array([True,True,False,True,True])
    t=np.array([0.,0.,.3,0.,0.])
    assert not nonlocal_tensile_mask(t,r,y,a,.2,.3).any()
    np.testing.assert_array_equal(nonlocal_tensile_mask(t,r,y,a,.2,.3,intact_only=True),~r)


def test_blade_refinement_keeps_body_and_all_physical_parameters():
    p=problem(candidate());fine=problem(dict(candidate(),blade_surface_spacing_mm=.05))
    np.testing.assert_array_equal(p.mesh.reference_xy,fine.mesh.reference_xy)
    np.testing.assert_array_equal(p.mesh.triangles,fine.mesh.triangles)
    np.testing.assert_array_equal(p.mass,fine.mass)
    assert p.material==fine.material and p.strength==fine.strength and p.toughness==fine.toughness
    assert len(fine.rest)>len(p.rest)


def test_negative_friction_trial_rejected_for_cutback_not_accepted(monkeypatch):
    from prototype.zero_gap_cut import ZeroGapCutProblem
    from prototype.ipc_reference import LoadStepFailure
    p=problem(candidate())
    def fail(*args):raise RuntimeError('Negative IPC friction incremental work')
    monkeypatch.setattr(ZeroGapCutProblem,'acceptance',fail)
    with pytest.raises(LoadStepFailure,match='retry smaller physical step'):
        p.acceptance(p.rest,0,0,0)


@pytest.mark.parametrize('scale', [1., 1e-3, 1e-6])
def test_bulk_discrete_work_tangent_and_objectivity(scale):
    rng = np.random.default_rng(52)
    f0 = np.eye(2)+.15*rng.normal(size=(8, 2, 2))
    f1 = f0+scale*.1*rng.normal(size=(8, 2, 2))
    a = np.tile(np.diag([1.2, 1/1.2, 1.]), (8, 1, 1))
    q = discrete_response(f0, f1, a, .14, 1.24, 23.3433333)
    np.testing.assert_allclose(np.sum(q['P']*(f1-f0), axis=(1, 2)), q['energy']-q['previous_energy'], atol=2e-14)
    eps = 1e-6 if scale > 1e-4 else 1e-8
    for j, d in enumerate(np.eye(4).reshape(4, 2, 2)):
        plus = discrete_response(f0, f1+eps*d, a, .14, 1.24, 23.3433333, tangent=False)['P']
        minus = discrete_response(f0, f1-eps*d, a, .14, 1.24, 23.3433333, tangent=False)['P']
        np.testing.assert_allclose(q['tangent'][:, :, j], ((plus-minus)/(2*eps)).reshape(-1, 4), rtol=2e-4, atol=2e-5)
    rotation = np.array([[.8, -.6], [.6, .8]])
    rotated = discrete_response(rotation@f0, rotation@f1, a, .14, 1.24, 23.3433333, tangent=False)
    np.testing.assert_allclose(rotated['P'], rotation@q['P'], atol=2e-8)
    stress = q['P']@(.5*(f0+f1)).transpose(0, 2, 1)
    np.testing.assert_allclose(stress, stress.transpose(0, 2, 1), atol=2e-13)


def test_exact_cohesive_work_through_unload_reload_and_failure():
    history = 0.
    previous = 0.
    for opening in [.01, .03, .005, .02, .045, .1, 0.]:
        q0 = extrinsic_response(previous, history, .2, .006)
        q1 = extrinsic_response(opening, history, .2, .006)
        alg = cohesive_discrete_response(previous, opening, history, .2, .006)
        change = q1['stored']+q1['dissipated']-q0['stored']-q0['dissipated']
        assert alg['traction']*(opening-previous) == pytest.approx(change, abs=1e-16)
        eps = 1e-7
        if opening > eps:
            plus = cohesive_discrete_response(previous, opening+eps, history, .2, .006)['traction']
            minus = cohesive_discrete_response(previous, opening-eps, history, .2, .006)['traction']
            assert alg['tangent'] == pytest.approx((plus-minus)/(2*eps), rel=2e-7, abs=1e-9)
        history, previous = float(q1['history']), opening


def test_connected_patch_blocks_isolated_deep_release_and_keeps_compression_signed():
    y = np.arange(6.)
    released = np.zeros(6, bool)
    area = np.ones(6)
    t = np.array([0., 10., 0., 0., 0., 0.])
    assert not connected_patch_mask(t, released, y, area, .2, 2.).any()
    t[-2:] = [.3, .3]
    np.testing.assert_array_equal(np.flatnonzero(connected_patch_mask(t, released, y, area, .2, 2.)), [5])
    t[-1] = -.3
    assert not connected_patch_mask(t, released, y, area, .2, 2.).any()


def test_actual_candidate_free_step_audit_and_midpoint_velocity():
    cfg = candidate()
    p = problem(cfg)
    released = p.released.copy()
    x, rec = p.step(p.rest.copy(), p.start, -.015, time.monotonic()+20)
    assert audit(problem(cfg), snapshot(p, x, released, -.015), rec)['passed']
    assert abs(rec['energy']['physical_balance_defect_n_mm']) < 1e-10
    p.previous_velocity[:p.n, 1] = 2.
    p.set_primary_increment(x, np.zeros_like(x)) if np.array_equal(x, p.previous) else None
    v = p.endpoint_velocity(x)
    np.testing.assert_allclose(v[:p.n], 2*p.step_increment(x)[:p.n]/p.material_dt-p.previous_velocity[:p.n])


def test_full_discrete_work_balance_is_not_a_fitted_ledger():
    p = problem(candidate())
    p.material_dt = .02
    x = p.rest.copy()
    x[:p.n, 1] -= .01*p.rest[:p.n, 1]/5
    p.previous_barrier_energy = float(p.barrier(p.collisions(p.previous), p.cmesh, p.previous[p.surface]))
    e = p.energy_ledger(x)
    g = p.evaluate(x, p.lag(x), tangent=False)[1]
    expected = -2*float(g[p.free]@(x-p.previous).ravel()[p.free])
    assert abs(expected) > 1e-6
    assert e['algebraic_balance_residual_n_mm'] == pytest.approx(expected, abs=2e-10)
    assert abs(e['bulk_endpoint_defect_n_mm']) < 1e-10
    assert e['viscous_endpoint_dissipation_n_mm'] > 0


@pytest.mark.parametrize('theta', [.5, .525, .55, 1.])
def test_theta_numerical_dissipation_is_explicit_and_work_conjugate(theta):
    p=problem(dict(candidate(),kinematic_theta=theta))
    p.material_dt=.01
    x=p.rest.copy();x[:p.n,1]-=.001*p.rest[:p.n,1]
    p.previous_velocity[:p.n,1]=.01*p.rest[:p.n,1]
    e=p.energy_ledger(x)
    v=p.endpoint_velocity(x)
    expected=(2*theta-1)*float(p.mass@np.sum((v[:p.n]-p.previous_velocity[:p.n])**2,axis=1))
    assert e['backward_euler_inertia_loss_n_mm']==pytest.approx(expected,abs=1e-18)
    g=p.evaluate(x,p.lag(x),tangent=False)[1]
    residual=-2*float(g[p.free]@(x-p.previous).ravel()[p.free])
    assert e['algebraic_balance_residual_n_mm']==pytest.approx(residual,abs=2e-10)


def test_candidate_assembled_jacobian_with_real_blade_contact():
    p = problem(candidate())
    p.material_dt = .001
    x = p.rest.copy()
    # A coherent contact configuration just before nominal first touch.
    p.previous[p.n:] = p.prescribed_blade_positions(-.0015)
    x[p.n:] = p.prescribed_blade_positions(-.0014)
    # Avoid the exact VV/EV transition at the perfectly centred nose; that
    # nonsmooth switching point has no unique two-sided finite-difference J.
    x[p.n:, 0] += .003
    p.previous[p.n:, 0] += .003
    p.previous_velocity = np.zeros_like(x)
    lag = p.lag(x)
    _, g, h, _ = p.evaluate(x, lag, projected=False)
    h = p.coupled_friction_tangent(x, lag, h)
    direction = np.zeros_like(x)
    node = p.seam[-1]
    direction[node, 1] = .3
    direction[p.n:, 1] = -.2
    eps = 1e-6
    plus = p.evaluate(x+eps*direction, p.lag(x+eps*direction), tangent=False)[1]
    minus = p.evaluate(x-eps*direction, p.lag(x-eps*direction), tangent=False)[1]
    np.testing.assert_allclose(h@direction.ravel(), (plus-minus)/(2*eps), rtol=3e-4, atol=2e-3)
