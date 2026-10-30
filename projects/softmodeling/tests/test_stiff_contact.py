import time
import numpy as np
import pytest
from scipy.optimize import brentq
from test_energy_cut import candidate
from tools.run_zero_gap_cut import problem, snapshot, audit
from prototype.ipc_reference import IPCProblem
from stiff_contact import install, impulse_ledger
from release_dynamics import activate


def contact_fixture():
    p = problem(candidate()); x = p.rest.copy()
    x[p.n:] = p.prescribed_blade_positions(-.0014)
    x[p.n:, 0] += .003
    return p, x


def test_native_backward_euler_friction_has_no_scalar_reversal_and_separate_loss():
    p, x = contact_fixture()
    lag = IPCProblem.lag(p, x)
    potential = p.ipc.FrictionPotential(p.friction_velocity_epsilon)
    d = np.zeros_like(x); d[:p.n, 0] = 1.; d[p.mesh.bottom_nodes] = 0.
    mass = float(p.mass_dof@(d.ravel()**2))
    def drag(v):
        return float(potential.gradient(lag, p.cmesh, v*d[p.surface])@d[p.surface].ravel())
    v0 = 1.; h = 10*mass/drag(.5)
    v1 = brentq(lambda v: mass*(v-v0)/h+drag(v), 0., v0, xtol=1e-15)
    assert 0 <= v1 < .1*v0
    kinetic = .5*mass*(v1*v1-v0*v0)
    physical = h*drag(v1)*v1
    numerical = .5*mass*(v1-v0)**2
    assert physical >= 0 and numerical > 0
    assert abs(kinetic+physical+numerical) < 1e-12
    assert abs(kinetic+physical) > .1*mass  # must NOT label the loss as friction


def test_complete_endpoint_force_jacobian_direction_includes_geometry():
    p, x = contact_fixture(); install(p, 'endpoint_be')
    p.material_dt = 1e-4
    p.previous = x.copy(); p.previous[p.n:, 1] += 1e-5
    p.previous_velocity = np.zeros_like(x)
    p.friction = p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
    x[:p.n, 0] += 1e-6
    x[p.mesh.bottom_nodes] = p.rest[p.mesh.bottom_nodes]
    p.set_primary_increment(x, x-p.previous)
    lag = p.lag(x); _, g, H, _ = p.evaluate(x, lag, tangent=True, projected=False)
    H = p.coupled_friction_tangent(x, lag, H)
    rng = np.random.default_rng(403)
    direction = np.zeros(p.total); direction[p.free] = rng.normal(size=len(p.free))
    direction /= np.linalg.norm(direction)
    e = 1e-7
    xp, xm = x+e*direction.reshape(-1,2), x-e*direction.reshape(-1,2)
    derivative = (p.evaluate(xp, p.lag(xp), tangent=False)[1]-p.evaluate(xm, p.lag(xm), tangent=False)[1])/(2*e)
    assert np.linalg.norm(derivative-H@direction)/np.linalg.norm(derivative) < 2e-4
    assert np.linalg.norm(g) > 0


def test_endpoint_complete_step_impulse_audit_and_timeout_rolls_back(monkeypatch):
    p = problem(candidate()); x = p.rest.copy(); activate(p, x, [])
    install(p, 'endpoint_be')
    x, rec = p.step(x, p.start, -.015, time.monotonic()+20)
    fields, ledger = impulse_ledger(p, x)
    assert ledger['equivalent_force_residual_n'] <= 1e-4
    assert ledger['blade_work_n_mm'] == pytest.approx(rec['energy']['external_work_n_mm'])
    assert ledger['numerical_kinetic_loss_n_mm'] == pytest.approx(rec['energy']['backward_euler_inertia_loss_n_mm'])
    state = snapshot(p, x, p.released.copy(), -.015)
    fresh = install(problem(candidate()), 'endpoint_be')
    assert audit(fresh, state, rec)['passed']
    saved = [p.velocity.copy(), p.material_metric.copy(), p.history.copy(), p.released.copy()]
    def fail(*a): raise TimeoutError('controlled compute budget')
    monkeypatch.setattr(p, 'solve_active', fail)
    with pytest.raises(TimeoutError): p.step(x, -.015, -.014, time.monotonic()+1)
    for old, new in zip(saved, (p.velocity, p.material_metric, p.history, p.released)):
        np.testing.assert_array_equal(old, new)
