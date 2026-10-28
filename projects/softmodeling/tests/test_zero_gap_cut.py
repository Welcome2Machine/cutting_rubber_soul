import json
import time
from pathlib import Path
import numpy as np
import pytest
from prototype.zero_gap_cut import extrinsic_response
from tools.run_zero_gap_cut import problem, audit, snapshot, accepted_step_growth


def config():
    return json.loads((Path(__file__).parents[1]/'configs/zero_gap_dynamic_control.json').read_text())


def test_extrinsic_envelope_work_unloading_and_no_virgin_spring():
    t, gc = .2, .006
    end = 2*gc/t
    u = np.linspace(0, end, 1001)
    q = extrinsic_response(u, np.zeros_like(u), t, gc)
    np.testing.assert_allclose(q['traction'], t*(1-u/end), atol=1e-15)
    assert np.trapezoid(q['traction'], u) == pytest.approx(gc)
    np.testing.assert_allclose(q['stored']+q['dissipated'], t*u-.5*t/end*u*u)
    unload = extrinsic_response(.01, .03, t, gc)
    assert unload['dissipated'] == pytest.approx(.003)
    assert unload['traction'] == pytest.approx(.1/3)
    assert extrinsic_response(0., 0., t, gc)['stored'] == 0
    assert extrinsic_response(.1, .1, t, gc)['dissipated'] == gc


@pytest.mark.parametrize('u,k,t,gc', [(-.1,0,.2,.006),(0,-1,.2,.006),(0,0,0,.006),(0,0,.2,-1)])
def test_invalid_extrinsic_states_rejected(u,k,t,gc):
    with pytest.raises(ValueError):
        extrinsic_response(u,k,t,gc)


def test_actual_geometry_mass_ties_and_two_dimensional_inertia():
    p = problem(config())
    assert p.config.drop_s_mm == 15
    assert p.rest[:p.n, 0].min() == 15
    assert 2*p.mass.sum() == pytest.approx(3e-6)
    assert 2*p.material_integration_weights().sum() == pytest.approx(3000)
    assert not np.intersect1d(p.free, 2*p.seam).size
    assert np.intersect1d(p.free, 2*p.seam+1).size > 0
    p.material_dt = .01
    x = p.rest.copy()
    e, g, h, j = p.evaluate(x, p.lag(x), projected=False)
    assert np.linalg.norm(g[p.free]) < 1e-10
    assert j == pytest.approx(1)
    # Both inertia components have the same physical nodal mass.
    np.testing.assert_array_equal(p.mass_dof[:2*p.n:2], p.mass_dof[1:2*p.n:2])
    assert p.energy_ledger(x)['algebraic_balance_residual_n_mm'] == pytest.approx(0, abs=1e-12)


def test_no_contact_loading_step_fresh_audit_and_rejected_history_rollback(monkeypatch):
    p = problem(config())
    initial = p.rest.copy()
    old_released = p.released.copy()
    x, rec = p.step(initial, p.start, -.015, time.monotonic()+20)
    state = snapshot(p, x, old_released, -.015)
    assert audit(problem(config()), state, rec)['passed']
    assert rec['energy']['physical_balance_defect_n_mm'] == pytest.approx(0, abs=1e-10)
    old = (p.released.copy(), p.history.copy(), p.material_metric.copy(), p.velocity.copy())
    def fail(*args):
        p.released[-1] = True
        raise TimeoutError('test reject after insertion')
    monkeypatch.setattr(p, 'solve_active', fail)
    with pytest.raises(TimeoutError):
        p.step(x, -.015, -.01, time.monotonic()+20)
    for saved, actual in zip(old, (p.released, p.history, p.material_metric, p.velocity)):
        np.testing.assert_array_equal(saved, actual)


def test_only_released_seam_edges_are_collision_surfaces():
    p = problem(config())
    def internal_edges():
        edges = p.surface[np.asarray(p.cmesh.edges)]
        return edges[np.all(np.isin(edges, p.seam), axis=1)]
    assert not len(internal_edges())
    p.released[-1] = True
    p.rebuild_contact()
    assert len(internal_edges()) == 1
    np.testing.assert_array_equal(np.sort(internal_edges()[0]), np.sort(p.seam[-2:]))
    p.released[-1] = False
    p.rebuild_contact()
    assert not len(internal_edges())


def test_energy_decomposition_is_a_work_identity_not_a_physical_qualification():
    p = problem(config())
    p.material_dt = .02
    x = p.rest.copy()
    y = p.mesh.local_sn[:, 1]
    x[:p.n, 1] -= .01*y/5
    # An arbitrary nonequilibrated displacement must retain a nonzero residual
    # work, rather than manufacturing energy closure by a residual bucket.
    p.previous_barrier_energy = float(p.barrier(p.collisions(p.previous), p.cmesh, p.previous[p.surface]))
    q = p.energy_ledger(x)
    g = p.evaluate(x, p.lag(x), tangent=False)[1]
    expected = -2*float(g[p.free]@(x-p.previous).ravel()[p.free])
    assert abs(expected) > 1e-6
    assert q['algebraic_balance_residual_n_mm'] == pytest.approx(expected, abs=1e-10)


def test_hard_contact_return_step_loss_is_separate_from_material_dissipation():
    p = problem(config())
    p.material_dt = 1e-5
    p.released[-1] = True
    p.previous_history[-1] = .1
    p.rebuild_contact()
    p.previous = p.rest.copy()
    p.previous[p.seam[-1], 0] += .002
    p.previous_velocity[p.seam[-1], 0] = -1000
    p.previous_barrier_energy = float(p.barrier(p.collisions(p.previous), p.cmesh, p.previous[p.surface]))
    x = p.rest.copy()
    p.active_contact[p.seam[-1]] = True
    p.set_free()
    q = p.energy_ledger(x)
    g = p.evaluate(x, p.lag(x), tangent=False)[1]
    expected = -2*float(g[p.free]@(x-p.previous).ravel()[p.free])
    assert q['hard_contact_endpoint_loss_n_mm'] > 0
    assert q['algebraic_balance_residual_n_mm'] == pytest.approx(expected, abs=1e-10)


def test_primary_increment_preserves_sub_ulp_inertial_force_updates():
    p = problem(config())
    p.material_dt = 1e-9
    node = np.flatnonzero((p.rest[:p.n, 0] > 18) & (p.rest[:p.n, 1] > 1))[0]
    x = p.rest.copy()
    inc = np.zeros_like(x)
    p.set_primary_increment(x, inc)
    g0 = p.evaluate(x, p.lag(x), tangent=False)[1]
    inc[node, 0] = 1e-16
    # Absolute positions cannot represent this update at x~20 mm.
    np.testing.assert_array_equal(x, p.previous+inc)
    p.set_primary_increment(x, inc)
    g1 = p.evaluate(x, p.lag(x), tangent=False)[1]
    expected = p.mass[node]*1e-16/p.material_dt**2
    assert expected > 1e-8
    assert g1[2*node]-g0[2*node] == pytest.approx(expected, rel=1e-12)


def test_increment_work_normalization_has_turning_point_pathology():
    # Exact BE update of m=k=u0=1, v0=0. This is an indicator counterexample,
    # not cutting validation: a good indicator must vanish as dt -> 0.
    ratios, reservoir_ratios = [], []
    for dt in (.01, .005, .0025):
        u1 = 1/(1+dt*dt)
        v1 = -dt/(1+dt*dt)
        dk, du = .5*v1*v1, .5*u1*u1-.5
        loss = -dk-du
        ratios.append(loss/(abs(dk)+abs(du)))
        reservoir_ratios.append(loss/(.25+.25*u1*u1+.25*v1*v1))
    assert min(ratios) > .33
    assert reservoir_ratios[-1]/reservoir_ratios[-2] == pytest.approx(.25, rel=1e-4)


def test_constant_traction_unloading_energy_and_same_loading_envelope():
    t,gc=.2,.006
    opening=np.linspace(0,.06,1001)
    a=extrinsic_response(opening,0,t,gc,'secant')
    b=extrinsic_response(opening,0,t,gc,'constant_traction')
    np.testing.assert_allclose(a['traction'],b['traction'],atol=1e-15)
    np.testing.assert_allclose(a['stored']+a['dissipated'],b['stored']+b['dissipated'],atol=1e-15)
    assert b['dissipated'][-1] == pytest.approx(gc)
    unload=extrinsic_response(np.array([.03,.02,.01,0]),.03,t,gc,'constant_traction')
    np.testing.assert_allclose(unload['traction'],.1)
    np.testing.assert_allclose(unload['tangent'][1:],0)
    np.testing.assert_allclose(unload['dissipated'],gc/4)
    np.testing.assert_allclose(unload['stored'],.1*np.array([.03,.02,.01,0]))
    for u,old in [(.01,.03),(.04,.03)]:
        eps=1e-7
        left=extrinsic_response(u-eps,old,t,gc,'constant_traction')
        right=extrinsic_response(u+eps,old,t,gc,'constant_traction')
        gradient=((right['stored']+right['dissipated'])-(left['stored']+left['dissipated']))/(2*eps)
        assert gradient == pytest.approx(extrinsic_response(u,old,t,gc,'constant_traction')['traction'],rel=1e-8)


def test_resolved_step_does_not_lock_at_iteration_target():
    cfg=dict(iteration_target=6,energy_tolerance=1e-4,energy_normalization='mechanical_reservoir')
    for iterations in (5,6,7,15):
        assert accepted_step_growth(cfg,dict(iterations=iterations,energy_reservoir_error_ratio=1e-6)) > 1
    assert accepted_step_growth(cfg,dict(iterations=6,energy_reservoir_error_ratio=.99)) < 1


def test_temporal_ccd_rejection_does_not_commit_history(monkeypatch):
    from prototype.ipc_reference import LoadStepFailure
    p=problem(config())
    old=(p.history.copy(),p.material_metric.copy(),p.velocity.copy())
    monkeypatch.setattr(p,'accepted_chord_is_safe',lambda x:False)
    with pytest.raises(LoadStepFailure,match='Accepted-state motion'):
        p.step(p.rest.copy(),p.start,-.015,time.monotonic()+20)
    for before,after in zip(old,(p.history,p.material_metric,p.velocity)):
        np.testing.assert_array_equal(before,after)
    assert p.failed_stage == 'accepted_motion'


def test_timeout_before_newton_saves_this_predictor_not_stale_trial(monkeypatch):
    p=problem(config())
    p.trial=p.rest.copy()+10
    candidate=p.rest.copy()
    candidate[p.n:]=p.prescribed_blade_positions(-.015)
    monkeypatch.setattr(p,'_loading_predictor',lambda *a:candidate.copy())
    with pytest.raises(TimeoutError):
        p.step(p.rest.copy(),p.start,-.015,time.monotonic()-1)
    np.testing.assert_array_equal(p.failed_positions,candidate)
    np.testing.assert_array_equal(p.previous+p.failed_increment,candidate)
    assert p.failed_stage == 'force_solve'
