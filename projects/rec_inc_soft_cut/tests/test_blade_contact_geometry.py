import numpy as np

from src.analysis.blade_contact_geometry import blade_contact_geometry


def test_exclusion_preserves_boundary_uncertainty_and_contact_proximity():
    mask = np.ones((100, 100), np.uint8)
    mask[:60, 40:61] = 2
    points = np.array([[50., 30.], [40., 30.], [38., 50.], [64., 50.], [50., 64.], [-3., 5.]])
    result = blade_contact_geometry(mask, points, [50., 59.])
    np.testing.assert_array_equal(result['in_opaque_core'], [True, False, False, False, False, False])
    np.testing.assert_array_equal(result['routing_branch'], [0, 0, 1, 2, 3, 0])
    assert result['contact_proximity_candidate'][2:4].all()
    assert result['signed_blade_gap_px'][0] < 0 and np.isnan(result['signed_blade_gap_px'][-1])
    np.testing.assert_allclose(result['down_or_right_tangent_xy'][2:4], [[0, 1], [0, 1]], atol=1e-8)


def test_no_tip_cannot_create_contact_and_missing_blade_has_no_fake_boundary():
    points = np.array([[20., 30.], [40., 50.]])
    result = blade_contact_geometry(np.ones((100, 100), np.uint8), points, [np.nan, np.nan])
    assert not result['geometry_supported'] and not result['contact_proximity_candidate'].any()
    assert np.isnan(result['signed_blade_gap_px']).all()
    assert not result['routing_branch'].any()
