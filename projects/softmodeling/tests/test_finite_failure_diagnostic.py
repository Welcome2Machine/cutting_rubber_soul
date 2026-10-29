import json
import time

import numpy as np
import pytest

from tools.diagnose_finite_increment import inspect, restore_failed_seed, virtual_state
from tools.run_zero_gap_cut import problem
from prototype.finite_increment_cut import FrozenVirtualCrack
from test_finite_increment import config


def test_failed_seed_keeps_attempt_dt_and_sub_coordinate_increment():
    p = problem(config())
    delta = np.zeros_like(p.rest)
    delta.ravel()[p.free[-1]] = 1e-16
    fields = dict(positions=p.rest.copy(), increment=delta, previous_positions=p.rest.copy(),
        previous_velocity=p.velocity.copy(), previous_material_metric=p.material_metric.copy(),
        previous_history=p.history.copy(), released=p.released.copy(),
        active_contact=p.active_contact.copy(), material_dt_s=1e-12,
        previous_barrier_energy_n_mm=0., target_depth_mm=p.start)
    x, depth = restore_failed_seed(p, fields)
    np.testing.assert_array_equal(p.step_increment(x), delta)
    assert p.material_dt == 1e-12 and depth == p.start
    # Replacing the raw increment with x-previous would erase this force.
    assert np.sqrt(2)*np.linalg.norm(p.evaluate(x, p.lag(x), tangent=False)[1][p.free]) > 1e-4


def test_diagnostic_rejects_changed_producer_before_creating_output(tmp_path):
    case = tmp_path/'case'
    case.mkdir()
    (case/'result.json').write_text(json.dumps(dict(source_hashes={}, source_unchanged=True)))
    with pytest.raises(ValueError, match='unchanged producer'):
        inspect(case, tmp_path/'diagnostic', 10)
    assert not (tmp_path/'diagnostic').exists()


@pytest.mark.parametrize('seconds', [0, 601, float('nan')])
def test_diagnostic_enforces_user_audit_ceiling(tmp_path, seconds):
    with pytest.raises(ValueError, match='1..600'):
        inspect(tmp_path, tmp_path/'out', seconds)


def test_virtual_diagnostic_checks_original_force_without_history_commit():
    p = problem(config())
    p.material_dt = .025
    before = {k: getattr(p, k).copy() for k in ('previous', 'history', 'material_metric', 'velocity')}
    v = FrozenVirtualCrack(p, p.rest, p.material_metric, p.history, p.released,
                          np.zeros_like(p.released), p.ffm_epsilon)
    x, _, _ = v.equilibrium(p.rest, p.start, time.monotonic()+10)
    q = virtual_state(v, x)
    assert q['residual_n'] < 1e-10 and q['min_j'] > 0 and q['chord_safe']
    v.virtual_solver_load[v.base_free[0]] = 1.
    with pytest.raises(ValueError, match='artificial load'):
        virtual_state(v, x)
    for k, value in before.items():
        np.testing.assert_array_equal(getattr(p, k), value)


def test_event_ledger_preserves_whole_intervals_and_does_not_sum_state_energy():
    from tools.report_finite_increment import event_energy_window
    keys = ('external_work_n_mm bulk_stored_delta_n_mm cohesive_stored_delta_n_mm '
        'barrier_delta_n_mm kinetic_delta_n_mm fracture_dissipation_n_mm '
        'viscous_endpoint_dissipation_n_mm friction_work_n_mm '
        'physical_balance_defect_n_mm backward_euler_inertia_loss_n_mm '
        'bulk_endpoint_defect_n_mm cohesive_endpoint_defect_n_mm '
        'barrier_endpoint_defect_n_mm contact_topology_defect_n_mm '
        'hard_contact_endpoint_loss_n_mm').split()
    energy = {k: np.array([1., 2., 3.]) for k in keys}
    energy['kinetic_energy_n_mm'] = np.array([10., 12., 15.])
    energy['mechanical_reservoir_n_mm'] = np.ones(3)*1e9
    d = np.array([.1, .2, .3])
    q = event_energy_window(d, [.5, .5, .5], energy, .19, .02)
    assert q['window_complete'] and q['accepted_intervals'] == 2
    assert q['selected_depth_window_mm'] == pytest.approx([.1, .3])
    assert q['initial_kinetic_n_mm'] == 10 and q['final_kinetic_n_mm'] == 15
    assert q['energy_n_mm']['kinetic_delta_n_mm'] == 5
    assert 'mechanical_reservoir_n_mm' not in q['energy_n_mm']
    assert not event_energy_window(d, [.5]*3, energy, .3, .02)['window_complete']


def test_virtual_friction_homotopy_restores_original_problem_even_on_timeout(monkeypatch):
    import prototype.finite_increment_cut as module
    p = problem(dict(config(), ffm_virtual_solver='friction_continuation'))
    p.material_dt = .025
    v = FrozenVirtualCrack(p, p.rest, p.material_metric, p.history, p.released,
                          np.zeros_like(p.released), p.ffm_epsilon)
    friction, maximum = v.friction, v.max_iterations
    old = {k: getattr(p, k).copy() for k in ('previous', 'history', 'material_metric', 'velocity')}
    def fail(*args):
        raise TimeoutError('bounded solver stage')
    monkeypatch.setattr(module, 'solve', fail)
    monkeypatch.setattr(v, 'solve_friction', fail)
    with pytest.raises(TimeoutError):
        v.friction_continued_equilibrium(p.rest, p.start, time.monotonic()+10)
    assert v.friction is friction and v.max_iterations == maximum
    assert not np.any(v.virtual_solver_load)
    for k, value in old.items():
        np.testing.assert_array_equal(getattr(p, k), value)


@pytest.mark.parametrize('solver,method', [
    ('friction_continuation', 'friction_continued_equilibrium'),
    ('lagged_newton', 'lagged_newton_equilibrium')])
def test_auxiliary_solver_return_cannot_bypass_original_force_gate(monkeypatch, solver, method):
    p = problem(dict(config(), ffm_virtual_solver=solver))
    p.material_dt = .025
    v = FrozenVirtualCrack(p, p.rest, p.material_metric, p.history, p.released,
                          np.zeros_like(p.released), p.ffm_epsilon)
    bad = p.rest.copy()
    bad.ravel()[v.free[-1]] += .001
    def fail(*args):
        raise TimeoutError('direct stage')
    monkeypatch.setattr(v, 'solve_friction', fail)
    monkeypatch.setattr(v, method, lambda *args: bad)
    with pytest.raises(RuntimeError, match='force/CCD gate'):
        v.equilibrium(p.rest, p.start, time.monotonic()+10)


def test_successful_virtual_homotopy_has_original_friction_and_zero_physical_time():
    p = problem(dict(config(), ffm_virtual_solver='friction_continuation'))
    p.material_dt = .025
    v = FrozenVirtualCrack(p, p.rest, p.material_metric, p.history, p.released,
                          np.zeros_like(p.released), p.ffm_epsilon)
    friction = v.friction
    x = v.friction_continued_equilibrium(p.rest, p.start, time.monotonic()+10)
    assert v.friction is friction and v.material_dt == .025
    assert all(q['accepted'] for q in v.virtual_solver_phases)
    assert virtual_state(v, x)['residual_n'] < 1e-10
    np.testing.assert_array_equal(p.history, np.zeros_like(p.history))


def test_failed_virtual_context_survives_rollback_without_committing_histories(tmp_path, monkeypatch):
    p = problem(config())
    p.material_dt = .025
    old = p.released.copy()
    def fail(*args):
        raise TimeoutError('virtual pre-equilibrium')
    monkeypatch.setattr(FrozenVirtualCrack, 'equilibrium', fail)
    with pytest.raises(TimeoutError):
        p.propose_insertion(p.rest, np.ones(len(p.seam)), p.start, time.monotonic()+10)
    metadata = p.write_failure_evidence(tmp_path, 0)
    assert metadata['stage'] == 'before' and metadata['candidate_status'] == 'unknown'
    assert metadata['shared_scan_budget']['scope'] == 'pre_equilibrium_and_all_candidates'
    assert metadata['shared_scan_budget']['completed_candidates'] == []
    with np.load(tmp_path/metadata['evidence_file'], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['previous_released'], old)
        np.testing.assert_array_equal(data['virtual_trial'], p.rest)
        np.testing.assert_array_equal(data['frozen_metric'], p.material_response(p.rest)['metric'])
    np.testing.assert_array_equal(p.released, old)


def test_expired_shared_scan_archives_unattempted_candidate(tmp_path, monkeypatch):
    import prototype.finite_increment_cut as module
    p = problem(config())
    p.material_dt = .025
    old = p.history.copy(), p.released.copy()
    monotonic = time.monotonic
    def expired_candidates(*args):
        monkeypatch.setattr(module.time, 'monotonic', lambda: monotonic()+100)
        return [np.array([0])]
    monkeypatch.setattr(module, 'finite_patches', expired_candidates)
    with pytest.raises(TimeoutError, match='shared scan budget before candidate solve'):
        p.propose_insertion(p.rest, np.ones(len(p.seam)), p.start, time.monotonic()+10)
    evidence = p.write_failure_evidence(tmp_path, 0)
    assert evidence['seam_indices'] == [0] and evidence['solver_phases'] == []
    assert evidence['candidate_status'] == 'unknown'
    assert evidence['shared_scan_budget']['remaining_s'] == 0
    assert evidence['shared_scan_budget']['candidate_count'] == 1
    with np.load(tmp_path/evidence['evidence_file'], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['virtual_trial'], p.rest)
    np.testing.assert_array_equal(p.history, old[0])
    np.testing.assert_array_equal(p.released, old[1])


@pytest.mark.parametrize('mode', ['wide_smoothing', 'lagged_friction', 'lagged_newton'])
def test_probe_restores_original_force_functions_and_checks_the_original_root(mode):
    from tools.diagnose_finite_increment import probe_equilibrium
    p = problem(config()); p.material_dt = .025
    v = FrozenVirtualCrack(p, p.rest, p.material_metric, p.history, p.released,
                          np.zeros_like(p.released), p.ffm_epsilon)
    old = v.friction, v.lag, v.coupled_friction_tangent
    x, _, residual = probe_equilibrium(v, p.rest, p.start, time.monotonic()+10, mode)
    assert residual < 1e-10
    assert (v.friction, v.lag, v.coupled_friction_tangent) == old
    np.testing.assert_array_equal(x, p.rest)
    np.testing.assert_array_equal(p.material_metric, v.previous_material_metric)


@pytest.mark.parametrize('mode', ['wide_smoothing', 'lagged_friction'])
def test_probe_restores_force_functions_on_failure(mode, monkeypatch):
    import tools.diagnose_finite_increment as diagnostic
    p = problem(config()); p.material_dt = .025
    v = FrozenVirtualCrack(p, p.rest, p.material_metric, p.history, p.released,
                          np.zeros_like(p.released), p.ffm_epsilon)
    old = v.friction, v.lag, v.coupled_friction_tangent
    def fail(*args):
        raise TimeoutError('probe stage')
    monkeypatch.setattr(diagnostic, 'solve', fail)
    monkeypatch.setattr(v, 'solve_friction', fail)
    with pytest.raises(TimeoutError):
        diagnostic.probe_equilibrium(v, p.rest, p.start, time.monotonic()+10, mode)
    assert (v.friction, v.lag, v.coupled_friction_tangent) == old


def test_lagged_newton_production_restores_force_and_preserves_host_history():
    p = problem(dict(config(), ffm_virtual_solver='lagged_newton'))
    p.material_dt = .025
    v = FrozenVirtualCrack(p, p.rest, p.material_metric, p.history, p.released,
                          np.zeros_like(p.released), p.ffm_epsilon)
    old = v.friction, v.lag, v.coupled_friction_tangent
    history = p.history.copy()
    x = v.lagged_newton_equilibrium(p.rest, p.start, time.monotonic()+10)
    assert (v.friction, v.lag, v.coupled_friction_tangent) == old
    assert virtual_state(v, x)['residual_n'] < 1e-10
    np.testing.assert_array_equal(p.history, history)


def test_lagged_newton_restores_functions_after_inner_exception(monkeypatch):
    p = problem(dict(config(), ffm_virtual_solver='lagged_newton'))
    v = FrozenVirtualCrack(p, p.rest, p.material_metric, p.history, p.released,
                          np.zeros_like(p.released), p.ffm_epsilon)
    old = v.friction, v.lag, v.coupled_friction_tangent
    def fail(*args):
        raise ValueError('invalid predictor')
    monkeypatch.setattr(v, 'solve_friction', fail)
    with pytest.raises(ValueError, match='invalid predictor'):
        v.lagged_newton_equilibrium(p.rest, p.start, time.monotonic()+10)
    assert (v.friction, v.lag, v.coupled_friction_tangent) == old
