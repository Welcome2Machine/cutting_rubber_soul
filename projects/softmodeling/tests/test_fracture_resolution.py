import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from prototype.force_globalization import directions, solve
from prototype.extrinsic_topology import partition, transfer_release, lumped_mass
from prototype.cohesive_dynamics import advance
from prototype.cohesive_mixed import MixedModeLaw
from prototype.material import bulk
from prototype.model_config import FEMConfig, FEMMesh
from tools.verify_network_compliance import square_mesh
from tools.run_cohesive_cut import read_config
from tools.audit_fracture_energy import screen
from tools.verify_fracture_resolution import edge_metric


def test_newton_and_lm_descent_for_indefinite_nonsymmetric_jacobian():
    a = csc_matrix([[0., 2.], [-3., -5.]])
    r = np.array([1., -2.])
    values = list(directions(a, r))
    assert values[0][0] == 'newton'
    assert len(values) == 7
    np.testing.assert_allclose(a@values[0][1], -r)
    assert all(float(r@(a@p)) < 0 for _, p in values)


def test_lm_survives_singular_newton_without_claiming_equilibrium():
    a = csc_matrix([[1., 0.], [0., 0.]])
    values = list(directions(a, np.ones(2)))
    assert values and all(name.startswith('lm') for name, _ in values)
    assert all(np.linalg.norm(a@p+1.) >= 1. for _, p in values)


def test_newton_host_calls_geometry_gate_and_leaves_accepted_history():
    class Host:
        free = np.array([0]); total = 2; max_iterations = 30
        history = np.array([.123]); calls = 0
        def lag(self, x): return None
        def evaluate(self, x, lag, projected=False, tangent=True):
            z = x.ravel()[0]
            return 0., np.array([z*z-2., 0.]), csc_matrix([[2*z, 0.], [0., 1.]]), 1.
        def coupled_friction_tangent(self, x, lag, h): return h
        def _ptc_trial(self, x, direction, maximum):
            self.calls += 1
            return x+.5*direction, .5, .5
    host = Host()
    x, _ = solve(host, np.array([[1., 0.]]), 0., time.monotonic()+5.)
    assert abs(x[0, 0]**2-2.) <= 1e-4
    assert host.calls > 0
    np.testing.assert_array_equal(host.history, [.123])


@pytest.mark.parametrize('cells', [1, 2, 4])
def test_zero_gap_bulk_element_force_energy_and_tangent_are_identical(cells):
    config = FEMConfig(length_mm=1., height_mm=1., width_mm=2., inclination_deg=0.)
    original = square_mesh(cells, config)
    tied = partition(original.reference_xy, original.triangles)
    # Smooth non-affine motion: this is stronger than a constant-strain patch.
    x = original.reference_xy
    displacement = np.column_stack((.03*x[:, 0]*x[:, 1], -.01*x[:, 1]+.02*x[:, 0]**2))
    empty = np.empty(0, int)
    new = FEMMesh(tied.reference, tied.reference.copy(), tied.triangles,
                  np.empty((0, 2), int), np.empty(0), empty, empty, np.zeros(len(tied.reference), int))
    a = bulk(original, displacement.ravel(), config)
    b = bulk(new, displacement[tied.original_vertex_ids].ravel(), config)
    for i in (0, 1):
        np.testing.assert_allclose(a[i], b[i], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a[3:], b[3:], rtol=1e-13, atol=1e-13)
    assert len(tied.reference) == len(x)
    np.testing.assert_array_equal(tied.reference[tied.triangles], x[original.triangles])


def test_release_preserves_positions_velocity_mass_momentum_and_kinetic_energy():
    config = FEMConfig(length_mm=1., height_mm=1., inclination_deg=0.)
    mesh = square_mesh(2, config)
    old = partition(mesh.reference_xy, mesh.triangles)
    # Complete vertical cut; the center fan must split, the two faces coincide.
    new = partition(mesh.reference_xy, mesh.triangles, [(1, 4), (4, 7)])
    assert len(new.reference) == len(old.reference)+3
    x = old.reference@np.array([[1., .1], [.2, 1.1]])
    v = np.random.default_rng(7).normal(size=x.shape)
    x1, v1 = transfer_release(old, new, x, v)
    m0, m1 = (lumped_mass(t, .001, 2.) for t in (old, new))
    np.testing.assert_array_equal(x[old.triangles], x1[new.triangles])
    np.testing.assert_allclose(m0.sum(), m1.sum(), atol=1e-18)
    np.testing.assert_allclose(m0@v, m1@v1, atol=1e-18)
    np.testing.assert_allclose(m0@(v*v).sum(1), m1@(v1*v1).sum(1), atol=1e-18)
    with pytest.raises(ValueError, match='Healing'):
        transfer_release(new, old, x1, v1)


def test_single_interior_edge_release_keeps_connected_tip_fans():
    config = FEMConfig(length_mm=1., height_mm=1., inclination_deg=0.)
    mesh = square_mesh(3, config)
    t = partition(mesh.reference_xy, mesh.triangles, [(5, 10)])
    # Linear elements cannot open an isolated one-edge crack with both tips tied.
    assert len(t.reference) == len(mesh.reference_xy)


@pytest.mark.parametrize('damping', [0., .02])
def test_inertial_fracture_energy_closes_without_assigning_remainder(damping):
    law = MixedModeLaw(normal_stiffness_mpa_per_mm=50., compression_stiffness_mpa_per_mm=50.)
    u = v = k = work = dissipated = viscous = 0.
    remainders = []
    for step in range(1, 501):
        row = advance(law, displacement=u, velocity=v, history=k,
                      support_before=(step-1)*.0008, support_after=step*.0008,
                      dt=.004, mass=.001, spring=1., damping=damping)
        u, v, k = (row[name] for name in ('displacement', 'velocity', 'history'))
        remainders.append(row['energy_remainder'])
        work += row['work']; dissipated += row['fracture']; viscous += row['damping']
        assert row['fracture'] >= -1e-14
    assert row['damage'] == 1.
    assert max(map(abs, remainders)) < 1e-11
    assert abs(sum(remainders)) < 1e-10
    assert dissipated == pytest.approx(law.toughness_n_per_mm, abs=1e-12)
    assert work == pytest.approx(.5*.001*v*v+.5*(u-.4)**2+dissipated+viscous, abs=1e-10)


def test_inertial_coarse_step_and_bad_history_are_rejected():
    law = MixedModeLaw()
    inputs = dict(displacement=0., velocity=0., history=0., support_before=0.,
                  support_after=.2, dt=1., mass=.001, spring=1.)
    with pytest.raises(ValueError, match='Reduce dt'):
        advance(law, **inputs)
    inputs.update(dt=.001, displacement=.01)
    with pytest.raises(ValueError, match='Accepted history'):
        advance(law, **inputs)


def test_solver_configuration_is_explicit_and_legacy_defaults_unchanged(tmp_path):
    root = Path(__file__).resolve().parents[1]
    c = json.loads((root/'configs/fracture_network_finite_faces.json').read_text())
    p = tmp_path/'c.json'
    c['nonlinear_solver'] = 'newton_lm'; p.write_text(json.dumps(c))
    assert read_config(p)['nonlinear_solver'] == 'newton_lm'
    for invalid in ('unbounded', None, 3):
        bad = copy.deepcopy(c); bad['nonlinear_solver'] = invalid
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError): read_config(p)


def test_signed_energy_cancellation_cannot_pass_absolute_screen():
    base = dict(external_work_n_mm=1., equilibrium_stored_change_n_mm=0.,
                nonequilibrium_stored_change_n_mm=0., damageable_cohesive_stored_change_n_mm=0.,
                compression_stored_change_n_mm=0., viscous_dissipation_n_mm=0.,
                fracture_dissipation_n_mm=0., friction_work_n_mm=0., ipc_barrier_stored_change_n_mm=0.)
    result = screen([dict(base, unaccounted_energy_n_mm=.1), dict(base, unaccounted_energy_n_mm=-.1)])
    assert result['signed_remainder_n_mm'] == 0.
    assert result['absolute_relative_remainder'] == pytest.approx(.1)
    assert not result['numerical_screen_passed']
    assert not screen([])['numerical_screen_passed']
    with pytest.raises(ValueError, match='Nonfinite'):
        screen([dict(base, unaccounted_energy_n_mm=float('nan'))])


@pytest.mark.parametrize('cells', [8, 16, 32])
def test_fixed_edge_direction_bias_does_not_disappear_by_size_refinement(cells):
    rows = edge_metric(cells, 'aligned')
    assert next(r['cost_ratio'] for r in rows if r['angle_deg'] == 135) == pytest.approx(np.sqrt(2.))
    assert next(r['cost_ratio'] for r in rows if r['angle_deg'] == 45) == pytest.approx(1.)
