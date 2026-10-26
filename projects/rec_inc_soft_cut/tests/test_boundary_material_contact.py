"""Analytic material maps and geometric failure cases, without optical flow."""
from dataclasses import replace

import numpy as np

from src.analysis.boundary_material_contact import (
    BoundaryMaterialContactConfig, BoundaryMaterialContactEstimator, project_to_path,
)


def fixture(config=None):
    s, d = np.meshgrid(np.arange(-110., 111., 5.), np.arange(0., 31., 3.))
    seed = np.column_stack((s.ravel(), d.ravel()))
    stations = np.arange(-150., 151., 2.)
    reference = dict(boundary_xy=np.column_stack((stations, np.zeros(len(stations)))),
                     surface_s_px=stations, tangent=np.array([1., 0.]),
                     inward_normal=np.array([0., 1.]), contact_s_px=0.)
    estimator = BoundaryMaterialContactEstimator(seed, np.arange(len(seed)), reference, 1., config)
    return estimator, seed


def straight_path(gap=0.):
    r = np.arange(-300., 300.01, 1.)
    return dict(xy=np.column_stack((r, np.zeros(len(r)))), arc=r, tip_gap=gap)


def test_depth_shear_is_removed_before_inverting_surface_address():
    estimator, seed = fixture()
    s, d = seed.T
    true_address, stretch = 17.25, 1.3
    # The bulk normal column is sheared by 30 px at d=30. The required
    # surface address remains 17.25; it is not the nearest interior label.
    current = np.column_stack((stretch*(s-true_address)+.7*d+.01*d*d+.002*s*d, d))
    result = estimator.estimate_frame(current, np.ones(len(seed), bool), np.zeros(2),
                                      path=straight_path(), return_profiles=True)
    assert result['state'] == 1
    assert abs(result['reference_s_px']-true_address) < 1.e-8
    assert abs(result['estimated_surface_stretch']-stretch) < 1.e-8
    assert result['low_s_px'] <= true_address <= result['high_s_px']
    assert result['contact_compatible']


def test_curved_boundary_recovers_known_material_root():
    estimator, seed = fixture()
    s, d = seed.T
    true_address, radius = -12.5, 350.
    r = 1.15*(s-true_address)+.45*d+.004*d*d
    theta = r/radius
    # Concentric material layers have an exact closest point on the circle.
    # Their surface arc coordinate contains shear and normal curvature terms.
    current = np.column_stack(((radius-d)*np.sin(theta),
                               radius-(radius-d)*np.cos(theta)))
    arc = np.arange(-300., 300.01, .5)
    path = dict(xy=np.column_stack((radius*np.sin(arc/radius),
                                   radius*(1.-np.cos(arc/radius)))), arc=arc, tip_gap=0.)
    result = estimator.estimate_frame(current, np.ones(len(seed), bool), np.zeros(2), path=path)
    assert result['state'] == 1
    assert abs(result['reference_s_px']-true_address) < .02
    assert result['low_s_px'] <= true_address <= result['high_s_px']
    assert result['contact_compatible']


def test_static_material_and_advancing_tip_retains_estimate_but_not_contact():
    estimator, seed = fixture()
    for penetration in (15., 35.):
        result = estimator.estimate_frame(seed, np.ones(len(seed), bool), np.array([0., penetration]),
                                          path=straight_path(gap=penetration))
        assert result['state'] == 1
        assert np.isfinite(result['reference_s_px'])
        assert abs(result['reference_s_px']) < 1.e-8
        assert not result['contact_compatible']
        assert result['boundary_tip_gap_px'] == penetration


def test_projection_detects_nonlocal_equal_distance_and_configured_separation():
    # U-shaped path: two spatially close segments have remote material arcs.
    path = np.array([[-80., 0.], [80., 0.], [80., 10.], [-80., 10.]])
    arc = np.array([-160., 0., 10., 170.])
    point = np.array([[0., 5.]])
    r, foot, gap, alternate, clamp = project_to_path(path, arc, point)
    assert gap[0] == alternate[0] == 5.
    assert not clamp[0]
    excluded = project_to_path(path, arc, point, ambiguity_arc_separation_px=400.)
    assert np.isinf(excluded[3][0])


def test_ambiguous_supports_are_excluded_from_the_material_fit():
    estimator, seed = fixture()
    r = np.arange(-150., 151., 1.)
    lower = np.column_stack((r, np.zeros(len(r))))
    upper = np.column_stack((r[::-1], np.full(len(r), 10.)))
    path_xy = np.vstack((lower, [[150., 5.]], upper))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))]-150.
    # Every point lies midway between the two long branches; no material
    # address can be assigned by closest-boundary projection alone.
    current = np.column_stack((seed[:, 0], np.full(len(seed), 5.)))
    result = estimator.estimate_frame(current, np.ones(len(seed), bool), np.zeros(2),
        path=dict(xy=path_xy, arc=arc, tip_gap=0.))
    assert result['state'] == 0
    assert not len(result['support_ids'])


def test_actual_path_end_clamps_are_rejected_without_splitting_at_tip():
    estimator, seed = fixture()
    short_r = np.arange(-90., 91.)
    path = dict(xy=np.column_stack((short_r, np.zeros(len(short_r)))), arc=short_r, tip_gap=0.)
    result = estimator.estimate_frame(seed, np.ones(len(seed), bool), np.zeros(2),
                                      path=path, return_profiles=True)
    assert result['state'] == 1
    ids = np.unique(np.concatenate([p['support_ids'] for p in result['profiles']]))
    assert np.all(abs(seed[ids, 0]) <= 90.)
    assert any(seed[ids, 0] < -6.) and any(seed[ids, 0] > 6.)
    probe = project_to_path(path['xy'], path['arc'], np.array([[-100., 5.], [100., 5.], [0., 5.]]))
    assert probe[4].tolist() == [True, True, False]


def test_depth_order_reversal_cannot_pass_surface_only_stretch_check():
    # Each depth layer has broad s support. Its scalar projection reverses
    # order at depth 12.5 even though surface stretch equals +1.
    cfg = replace(BoundaryMaterialContactConfig(), maximum_depths_px=(36.,))
    estimator, seed = fixture(cfg)
    s, d = seed.T
    current = np.column_stack(((1.-.08*d)*s, d))
    result = estimator.estimate_frame(current, np.ones(len(seed), bool), np.zeros(2),
                                      path=straight_path())
    assert result['state'] == 0


def test_root_outside_visible_reference_surface_is_not_clamped_into_marker():
    estimator, seed = fixture()
    current = seed.copy()
    current[:, 0] -= 180.
    result = estimator.estimate_frame(current, np.ones(len(seed), bool), np.zeros(2),
                                      path=straight_path())
    assert result['state'] == 0
    assert np.isnan(result['reference_s_px'])
    assert np.isnan(result['reference_marker_xy']).all()
