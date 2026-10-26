import cv2
import numpy as np
import pytest

from src.analysis.loading_tip_trends import (
    LoadingTrendConfig, _visible_to_target, current_groups, current_material_regions,
    estimate_frame_trends, finalize_series, local_motion_fit,
    current_interface_regions, interface_point_groups, estimate_interface_frame_trends,
)


def scene():
    labels = np.ones((180, 180), np.uint8)
    labels[:91, 80:101] = 2
    return labels, np.array([90., 90.])


def gap_at(labels, points):
    distance = cv2.distanceTransform((labels != 2).astype(np.uint8), cv2.DIST_L2, 5)
    xy = np.rint(points).astype(int)
    return distance[xy[:, 1], xy[:, 0]]


def frame(points, velocity, *, valid=None, labels=None, extra=None):
    default_labels, tip = scene()
    labels = default_labels if labels is None else labels
    points = np.asarray(points, float).reshape(-1, 2)
    velocity = np.asarray(velocity, float).reshape(-1, 2)
    valid = np.ones(len(points), bool) if valid is None else valid
    return estimate_frame_trends(points, velocity, valid, gap_at(labels, points),
        np.ones(len(points), bool), tip, labels, extra=extra)


def series(y, *, times=None, material=None, blade=None):
    y = np.asarray(y, float)
    n = len(y)
    values = np.full((n, 4, 2), np.nan)
    values[np.isfinite(y), 0, 0], values[:, 0, 1] = 0., y
    state = np.zeros((n, 4), np.uint8)
    state[np.isfinite(y), 0] = 4
    trend = dict(relative_trend_xy=values, state=state,
        method=np.full((n, 4), 'unavailable', dtype='<U40'),
        material_pixel_count=np.full((n, 4), 100),
        interval_low_y=values[..., 1]-1., interval_high_y=values[..., 1]+1.)
    if material is not None:
        trend['material_pixel_count'][:, 0] = material
    times = np.arange(n)*.0125 if times is None else np.asarray(times)
    blade = np.full(n, 2) if blade is None else blade
    return finalize_series(trend, times, blade)


def test_current_groups_follow_current_side_and_gap_without_initial_layer_ids():
    points = np.array([[84, 85], [72, 85], [96, 85], [111, 85], [89, 86], [91, 86], [15, 15]], float)
    gaps = np.array([6, 18, 6, 21, -1, np.nan, 3.])
    groups = current_groups(points, [90, 90], gaps, np.ones(7, bool))
    np.testing.assert_array_equal(groups, [0, 1, 2, 3, -1, -1, -1])
    # Reordering persistent material IDs cannot alter their current categories.
    order = np.array([3, 0, 6, 2, 5, 1, 4])
    reordered = current_groups(points[order], [90, 90], gaps[order], np.ones(7, bool))
    np.testing.assert_array_equal(reordered, groups[order])
    moved = points[:4].copy()
    moved[0, 0] = 96
    assert current_groups(moved, [90, 90], gaps[:4], np.ones(4, bool))[0] == 2


def test_current_targets_are_material_pixels_and_empty_band_has_no_proxy_target():
    labels, tip = scene()
    result = current_material_regions(labels, tip)
    assert (result['pixel_count'] > 6).all()
    xy = result['target_xy'].astype(int)
    assert (labels[xy[:, 1], xy[:, 0]] == 1).all()
    np.testing.assert_array_equal(current_groups(result['target_xy'], tip,
        gap_at(labels, result['target_xy']), np.ones(4, bool)), np.arange(4))
    distance = cv2.distanceTransform((labels != 2).astype(np.uint8), cv2.DIST_L2, 5)
    yy, xx = np.indices(labels.shape)
    labels[(xx < tip[0]) & (distance < 12) & (labels == 1)] = 0
    absent = current_material_regions(labels, tip)
    assert absent['pixel_count'][0] == 0
    assert np.isnan(absent['target_xy'][0]).all()
    out = frame([[74, 75], [75, 82]], [[0, -3], [0, -3]], labels=labels)
    assert out['state'][0] == 0
    assert out['method'][0] == 'no_material_in_band'
    assert np.isnan(out['relative_trend_xy'][0]).all()


def test_two_real_local_points_supply_conditional_trend_without_four_point_gate():
    out = frame([[74, 70], [75, 83]], [[2, -4], [2, -6]])
    assert out['state'][0] == 2
    assert out['method'][0] == 'few_supported_local_points'
    assert out['observation_count'][0] == 2
    np.testing.assert_allclose(out['relative_trend_xy'][0], [2, -5])
    assert out['interval_low_y'][0] < -6 and out['interval_high_y'][0] > -4


def test_forecasts_cannot_fill_motion_and_short_image_evidence_is_labeled():
    points = np.array([[74, 66], [74, 75], [73, 83], [75, 92]], float)
    forecasts = frame(points, np.tile([500, -800], (4, 1)), valid=np.zeros(4, bool))
    assert not forecasts['state'].any()
    assert np.isnan(forecasts['relative_trend_xy']).all()
    labels, _ = scene()
    extra = dict(xy=points, velocity=np.tile([1, -3], (4, 1)), quality=np.ones(4)*.7,
                 gap_px=gap_at(labels, points))
    out = frame([], [], extra=extra)
    assert out['state'][0] == 3
    assert out['method'][0] == 'short_image_assisted'
    assert out['observation_count'][0] == 0 and out['estimate_count'][0] == 4
    np.testing.assert_allclose(out['relative_trend_xy'][0], [1, -3])


def test_dense_cluster_does_not_outvote_spatially_independent_material():
    clustered = np.tile([73., 73.], (80, 1))
    independent = np.array([[73., 61.], [73., 85.], [73., 97.]])
    out = frame(np.vstack([clustered, independent]),
                np.vstack([np.tile([0, 60.], (80, 1)), np.tile([0, -5.], (3, 1))]))
    assert out['state'][0] == 4
    assert out['independent_cells'][0] >= 3
    assert out['relative_trend_xy'][0, 1] < 0


def test_nonuniform_affine_motion_estimate_recovers_target_and_rejects_far_extrapolation():
    points = np.array([[x, y] for x in [54., 60., 66.] for y in [68., 78., 88.]])
    target = np.array([74., 78.])
    matrix = np.array([[.1, -.03], [.2, .04]])
    velocity = [1., -3.] + (points-target)@matrix.T
    out = local_motion_fit(points, velocity, np.ones(len(points)), target)
    assert out['valid']
    np.testing.assert_allclose(out['velocity'], [1, -3], atol=.25)
    assert out['low_y'] < -3 < out['high_y']
    distant = local_motion_fit(points, velocity, np.ones(len(points)), [120, 78])
    assert not distant['valid']
    collinear = local_motion_fit(points[:3], velocity[:3], np.ones(3), target)
    assert not collinear['valid']


def test_local_model_does_not_mix_opposite_blade_side_observations():
    labels, tip = scene()
    target = current_material_regions(labels, tip)['target_xy'][0]
    left = np.array([[x, y] for x in [54., 60., 66.] for y in [68., 78., 88.]])
    lv = [1., -3.] + (left-target)@np.array([[.1, -.03], [.2, .04]]).T
    baseline = frame(left, lv)
    right = np.array([[106., 68], [108, 78], [107, 88], [109, 96]])
    combined = frame(np.vstack([left, right]), np.vstack([lv, np.tile([400, 800], (4, 1))]))
    assert baseline['state'][0] == 2
    assert baseline['method'][0] == 'same_side_local_affine'
    np.testing.assert_allclose(combined['relative_trend_xy'][0], baseline['relative_trend_xy'][0])
    assert combined['relative_trend_xy'][2, 1] == pytest.approx(800)
    barrier = np.zeros(labels.shape, bool)
    barrier[40:90, 60:65] = True
    np.testing.assert_array_equal(_visible_to_target(np.array([[40, 50], [75, 80]]), np.array([74, 78]), barrier), [False, True])


def test_only_short_observed_bounded_gaps_are_bridged():
    short = series([-5, np.nan, -6])
    assert short['state'][1, 0] == 1
    assert short['method'][1, 0] == 'bounded_time_bridge'
    assert short['relative_trend_xy'][1, 0, 1] == pytest.approx(-5.5)
    assert np.isnan(short['unsmoothed_trend_xy'][1, 0, 1])
    long = series([-5, np.nan, -6], times=[0, .1, .2])
    assert long['state'][1, 0] == 0
    assert np.isnan(long['relative_trend_xy'][1, 0]).all()
    absent = series([np.nan, np.nan, np.nan])
    assert not absent['state'].any()
    assert np.isnan(absent['relative_trend_xy'][:, :, 1]).all()


@pytest.mark.parametrize('material,blade', [([100, 0, 100], [2, 2, 2]), ([100, 100, 100], [2, 0, 2])])
def test_missing_material_or_blade_reference_prevents_interpolation(material, blade):
    out = series([-5, np.nan, -6], material=material, blade=blade)
    assert out['state'][1, 0] == 0
    assert np.isnan(out['relative_trend_xy'][1, 0]).all()


def test_strong_sign_switch_is_neither_bridged_nor_smoothed_away():
    missing_transition = series([-7, np.nan, 7])
    assert missing_transition['state'][1, 0] == 0
    observed_transition = series([7, -7, 7])
    # A supported brief direction reversal can be a release/slip event. Retain
    # it for inspection rather than replacing it with a three-row median.
    assert observed_transition['relative_trend_xy'][1, 0, 1] == pytest.approx(-7)


def test_estimated_blade_reference_downgrades_state_and_absent_reference_removes_trend():
    out = series([-5, -5, -5], blade=[2, 1, 0])
    assert out['state'][0, 0] == 4
    assert out['state'][1, 0] == 2
    assert out['relative_trend_xy'][1, 0, 1] == pytest.approx(-5)
    assert out['state'][2, 0] == 0
    assert np.isnan(out['relative_trend_xy'][2, 0]).all()


def interface_scene():
    labels = np.zeros((300, 240), np.uint8)
    labels[:251, 110:131] = 2
    labels[40:, :110] = 1
    labels[155:, 131:] = 1
    return labels, np.array([120., 250.])


def test_full_interface_includes_upstream_sparse_material_but_not_detached_surface():
    labels, tip = interface_scene()
    # A detached upper right surface has 20 px of air and must stay outside.
    labels[40:130, 151:] = 1
    regions = current_interface_regions(labels, tip)
    points = np.array([[104, 55], [88, 55], [136, 165], [150, 165], [155, 55], [80, 285], [120, 240]])
    groups, segments = interface_point_groups(points, regions)
    np.testing.assert_array_equal(groups, [0, 1, 2, 3, -1, -1, -1])
    assert segments[0] == 4 and segments[2] == 1
    # The preexisting near-tip selection remains narrow for its separate plot.
    old = current_groups(points, tip, gap_at(labels, points), np.ones(len(points), bool))
    assert old[0] == -1
    assert (regions['group_map'][labels != 1] == -1).all()


def test_shadow_sized_gap_is_conditional_and_layer_width_starts_at_material():
    labels, tip = interface_scene()
    labels[170:210, 131:144] = 0
    regions = current_interface_regions(labels, tip)
    points = np.array([[144, 185], [155, 185], [156, 185], [173, 185]], float)
    groups, segments = interface_point_groups(points, regions)
    np.testing.assert_array_equal(groups, [2, 2, 3, 3])
    state, _ = __import__('src.analysis.blade_contact_geometry', fromlist=['sample_pixels']).sample_pixels(regions['geometry_state_map'], points)
    assert (state == 1).all()
    assert regions['surface_gap_px'][185, 1] == 14
    out = estimate_interface_frame_trends(points, np.tile([0., -5.], (4, 1)), np.ones(4, bool),
        np.ones(4, bool), tip, labels, regions=regions)
    assert out['state'][2] == 2 and out['state'][3] == 2
    assert out['segment_surface_to_blade_gap_px'][2, segments[0]] == 14


def test_disconnected_free_surface_does_not_become_inner_band():
    labels, tip = interface_scene()
    # A narrow material lip beside the knife is separated from a second patch.
    labels[120:150, 80:105] = 0
    regions = current_interface_regions(labels, tip)
    groups, _ = interface_point_groups(np.array([[107, 135], [78, 135]], float), regions)
    np.testing.assert_array_equal(groups, [0, -1])


def test_segmented_trends_keep_opposing_upstream_and_tip_movement():
    labels, tip = interface_scene()
    near = np.array([[104, 220], [104, 226], [104, 236], [104, 244]], float)
    upstream = np.array([[104, 60], [104, 74], [104, 84], [104, 92]], float)
    points = np.vstack([near, upstream])
    velocities = np.vstack([np.tile([0, -7], (4, 1)), np.tile([0, 7], (4, 1))])
    out = estimate_interface_frame_trends(points, velocities, np.ones(8, bool), np.ones(8, bool), tip, labels)
    assert out['segment_relative_trend_xy'][0, 0, 1] == -7
    assert out['segment_relative_trend_xy'][0, 3, 1] == 7
    assert np.isnan(out['segment_relative_trend_xy'][0, 2]).all()
    # Missing groups are unavailable rather than zero or a neighboring value.
    assert out['state'][2] == 0 and np.isnan(out['relative_trend_xy'][2]).all()


def test_full_interface_spatial_balance_does_not_let_dense_tip_swamp_sparse_upstream():
    labels, tip = interface_scene()
    dense = np.array([[x, y] for x in np.arange(99., 110., .5) for y in np.arange(233., 249., .5)])
    sparse = np.array([[104., y] for y in [55., 80., 105., 130., 155., 180.]])
    points = np.vstack([dense, sparse])
    velocities = np.vstack([np.tile([0, -9], (len(dense), 1)), np.tile([0, 5], (len(sparse), 1))])
    out = estimate_interface_frame_trends(points, velocities, np.ones(len(points), bool),
        np.ones(len(points), bool), tip, labels)
    assert out['relative_trend_xy'][0, 1] > 0
    assert out['segment_relative_trend_xy'][0, 0, 1] < 0


@pytest.mark.parametrize('short_image', [False, True])
def test_tiny_material_sliver_keeps_observed_trend_with_real_point_anchor(short_image):
    labels, tip = interface_scene()
    labels[:, 131:] = 0
    labels[245, 132:137] = 1
    point = np.array([[134.25, 245.]])
    regions = current_interface_regions(labels, tip)
    assert regions['segment_pixel_count'][2, 0] == 5
    assert np.isnan(regions['segment_target_xy'][2, 0]).all()
    if short_image:
        points, velocity, valid = np.empty((0, 2)), np.empty((0, 2)), np.zeros(0, bool)
        extra = dict(xy=point, velocity=np.array([[0., -3.]]), quality=np.ones(1))
    else:
        points, velocity, valid = point, np.array([[0., -3.]]), np.ones(1, bool)
        extra = None
    out = estimate_interface_frame_trends(points, velocity, valid, np.ones(len(points), bool),
        tip, labels, extra=extra, regions=regions)
    assert out['segment_state'][2, 0] == 2
    np.testing.assert_array_equal(out['segment_target_xy'][2, 0], point[0])
    np.testing.assert_array_equal(out['segment_relative_trend_xy'][2, 0], [0., -3.])
    # The caller's region geometry is not mutated by the conditional anchor.
    assert np.isnan(regions['segment_target_xy'][2, 0]).all()
