import numpy as np
import pytest

from src.analysis.validated_material_motion import validated_local_slopes


def run(y, support=None, edges=None):
    if y.ndim == 1: y = y[:, None, None]
    return validated_local_slopes(y, np.arange(len(y))*.01,
        np.ones(y.shape[:2], bool) if support is None else support,
        motion_step_valid=edges)


def test_curved_endpoint_derivative_is_at_current_time_and_uses_no_future():
    t = np.arange(31)*.01
    y = 3.+4.*t+30.*t*t
    r = run(y)
    assert r['source_kind'][-1, 0] == 2
    assert r['one_sided'][-1, 0]
    assert r['slope'][-1, 0, 0] == pytest.approx(4+60*t[-1], abs=1.e-8)
    # Prefix boundary derivative is correct without any later samples.
    p = run(y[:21])
    assert p['slope'][-1, 0, 0] == pytest.approx(4+60*t[20], abs=1.e-8)


def test_invalid_edges_and_short_reacquisitions_are_not_bridged():
    t = np.arange(31)*.01
    y = 7*t
    y[27:] += 30.
    edges = np.ones((31, 1), bool)
    edges[27] = False
    r = run(y, edges=edges)
    assert not r['valid'][-1, 0]
    assert r['valid'][26, 0]
    assert r['slope'][26, 0, 0] == pytest.approx(7., abs=1.e-8)


def test_endpoint_recovery_does_not_turn_unsupported_points_into_observations():
    y = np.arange(31)*.01
    support = np.ones((31, 1), bool)
    support[-1] = False
    r = run(y, support=support)
    assert not r['valid'][-1, 0]
    assert r['source_kind'][-1, 0] == 0


def test_discontinuous_final_jump_is_rejected():
    y = np.arange(31)*.01
    y[-1] += 15.
    r = run(y)
    assert not r['valid'][-1, 0]


def test_centered_results_are_preserved_and_uncertainty_is_not_zero():
    t = np.arange(31)*.01
    y = np.stack([4*t, -6*t], axis=-1)[:, None]
    r = run(y)
    assert r['source_kind'][15, 0] == 1
    assert np.allclose(r['slope'][15, 0], [4., -6.])
    assert np.all(r['velocity_sigma'][r['valid']] > 0)
    assert r['velocity_sigma'][-1, 0] > r['velocity_sigma'][15, 0]
