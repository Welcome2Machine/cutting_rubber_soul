import cv2
import numpy as np

from src.analysis.material_coordinate_contact import (
    MaterialCoordinateContactConfig, MaterialCoordinateContactEstimator)


def make_estimator(model='affine'):
    s = np.arange(-72., 72.1, 6.)
    boundary = np.column_stack((100. + s, np.full(len(s), 60.)))
    depths = np.array([1.5, 7.5, 19.5, 31.5, 49.5, 73.5, 109.5])
    seed = np.stack([boundary + [0., d] for d in depths], axis=1).reshape(-1, 2)
    reference = dict(boundary_xy=boundary, surface_s_px=s,
                     tangent=np.array([1., 0.]), inward_normal=np.array([0., 1.]), contact_s_px=0.)
    config = MaterialCoordinateContactConfig(reference_half_windows_px=(48.,),
        maximum_depths_px=(120.,), models=(model,), maximum_roi_centers=1,
        quadratic_regularization=0.)
    return MaterialCoordinateContactEstimator(seed, np.arange(len(seed)), reference, 1., config), seed


def test_known_shear_rotation_address_and_exact_contact_closure():
    estimator, seed = make_estimator()
    angle = .4
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    F = R @ np.array([[1.1, .6], [0., .8]])
    translation = np.array([70., 40.])
    current = (seed - [100., 60.]) @ F.T + translation
    tip = np.array([13.5, 0.]) @ F.T + translation
    result = estimator.estimate_frame(current, np.ones(len(seed), bool), tip)
    assert abs(result['reference_s_px'] - 13.5) < 1.e-8
    assert result['contact_compatible']
    assert np.allclose(result['normal_fiber_xy'][0], tip, atol=1.e-9)
    assert result['closure_rms_px'] < 1.e-8
    # The determinant follows the actual shear/stretch, not an imposed value 1.
    assert abs(result['min_jacobian'] - .88) < 1.e-8


def test_advancing_tip_in_static_material_retains_no_contact_comparison():
    estimator, seed = make_estimator()
    result = estimator.estimate_frame(seed, np.ones(len(seed), bool), [100., 85.])
    assert np.isfinite(result['reference_s_px'])
    assert not result['contact_compatible']
    assert result['unconstrained_rms_px'] < 1.e-8
    assert result['closure_rms_px'] > 4.
    assert result['closure_penalty_px'] > 4.
    assert result['constraint_objective_increase'] > 0


def test_unsupported_points_are_not_contact_evidence():
    estimator, seed = make_estimator()
    support = np.ones(len(seed), bool)
    support[::5] = False
    altered = seed.copy()
    altered[~support] = [9000., -9000.]
    a = estimator.estimate_frame(seed, support, [112., 60.])
    b = estimator.estimate_frame(altered, support, [112., 60.])
    assert a['reference_s_px'] == b['reference_s_px']
    assert np.allclose(a['profile_best_score_px'], b['profile_best_score_px'])
    assert not np.any(np.isin(b['support_ids'], np.flatnonzero(~support)))


def test_known_quadratic_deformation_recovers_curved_material_fiber():
    estimator, seed = make_estimator('quadratic')
    s, d = seed[:, 0] - 100., seed[:, 1] - 60.
    current = np.column_stack((100. + s + .002 * d ** 2, 60. + d + .001 * s ** 2))
    tip = np.array([112., 60.144])
    result = estimator.estimate_frame(current, np.ones(len(seed), bool), tip)
    assert result['reference_s_px'] == 12.
    assert result['contact_compatible']
    depth = result['normal_fiber_depth_px']
    expected = np.column_stack((112. + .002 * depth ** 2, 60.144 + depth))
    assert np.allclose(result['normal_fiber_xy'], expected, atol=1.e-8)


def test_curved_fiber_collision_is_not_endpoint_chord_collision():
    estimator, _ = make_estimator('quadratic')
    beta = np.zeros((1, 6, 2))
    beta[0, 0] = [100., 60.]
    beta[0, 1] = [24., 0.]
    beta[0, 2] = [-1.2, 24.]
    beta[0, 5] = [.576, 0.]
    blade = np.zeros((190, 220), np.uint8)
    blade[108:113, 102:105] = 1
    distance = cv2.distanceTransform(blade, cv2.DIST_L2, 5)
    result = estimator._fiber_geometry(beta, np.array([0.]), 0., 100.,
                                       'quadratic', np.array([100., 60.]), distance)
    assert result['penetration'][0] == 0
    chord = np.linspace(result['fiber'][0, 0], result['fiber'][0, -1], 101)
    assert estimator._sample(distance, chord).max() > 0


def test_tip_uncertainty_widens_interval_without_moving_best_address():
    estimator, seed = make_estimator()
    a = estimator.estimate_frame(seed, np.ones(len(seed), bool), [112., 60.], tip_uncertainty_px=0.)
    b = estimator.estimate_frame(seed, np.ones(len(seed), bool), [112., 60.], tip_uncertainty_px=4.)
    assert a['reference_s_px'] == b['reference_s_px']
    assert b['low_s_px'] < a['low_s_px']
    assert b['high_s_px'] > a['high_s_px']


def test_search_extent_minimum_is_retained_but_not_called_identified():
    estimator, seed = make_estimator()
    result = estimator.estimate_frame(seed, np.ones(len(seed), bool), [172., 60.])
    assert result['reference_s_px'] == 72.
    assert result['profile_minimum_at_extent']
    assert not result['contact_compatible']
