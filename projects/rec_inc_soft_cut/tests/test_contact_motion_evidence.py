import json

import numpy as np
import pytest

from src.analysis.contact_motion_evidence import (
    ContactMotionConfig, analyze_contact_motion, supported_local_slopes,
)


def _arguments(positions, times):
    length, count = positions.shape[:2]
    return dict(times_s=times, frame_indices=np.arange(length),
        tip_xy=np.column_stack((np.full(length, 70.), 50.+10.*times)),
        tip_supported=np.ones(length, bool), depth_mm=times,
        contact_frame=20, reversal_frame=length-2,
        reference_surface_center=[40., 40.], reference_surface_tangent=[1., 0.],
        reference_inward_normal=[0., 1.], point_ids=np.arange(count)+101)


def test_physical_irregular_timestamps_and_support_breaks_cannot_be_bridged():
    raw_frames = np.flatnonzero(np.isin(np.arange(150) % 5, [0, 3]))
    times = raw_frames/200.
    velocity = np.array([[4., -3.], [-5., 2.]])
    p = 10.+times[:, None, None]*velocity[None]
    support = np.ones(p.shape[:2], bool)
    support[20, 0] = False
    p[20, 0] = [1e5, -1e5]  # retained forecast must not contribute to either fit
    edge = np.ones(support.shape, bool)
    edge[35, 1] = False
    result = supported_local_slopes(p, times, support, motion_step_valid=edge,
        half_window_s=.075, maximum_gap_s=.04)
    for point in range(2):
        found = result['slope'][:, point][result['valid'][:, point]]
        assert len(found) > 20
        np.testing.assert_allclose(found, np.tile(velocity[point], (len(found), 1)), atol=1e-8)
    assert not result['valid'][19:22, 0].any()
    assert not result['valid'][34:36, 1].any()
    assert not result['valid'][[0, -1]].any()
    # The fit uses seconds derived from acquisition, not imported 12-fps time.
    assert np.nanmedian(result['slope'][:, 0, 0]) == pytest.approx(4.)


def test_measured_blade_rotation_cancels_rigid_motion_and_is_never_assumed():
    times = np.arange(81)/100.
    theta = .4*times
    tip = np.column_stack((np.full(len(times), 70.), 50.+2*times))
    local = np.array([[-20., 2.], [-15., 8.], [15., 20.], [20., 25.]])
    rotation = np.array([[[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] for a in theta])
    p = tip[:, None]+np.einsum('tij,nj->tni', rotation, local)
    support = np.ones(p.shape[:2], bool)
    args = _arguments(p, times)
    args.update(tip_xy=tip, reference_depth_px=[2., 8., 20., 25.])
    cfg = ContactMotionConfig(minimum_group_points=1)
    summary, arrays = analyze_contact_motion(p, support, **args, config=cfg,
        blade_angle_rad=theta, blade_angle_supported=np.ones(len(times), bool))
    valid = arrays['relative_velocity_valid']
    assert np.max(np.linalg.norm(arrays['relative_velocity_xy_px_s'][valid], axis=1)) < .003
    assert summary['blade_rotation_correction'] == 'measured'
    unchanged, translation_only = analyze_contact_motion(p, support, **args, config=cfg)
    assert 'translation-only' in unchanged['blade_rotation_correction']
    assert np.nanmedian(np.linalg.norm(translation_only['relative_velocity_xy_px_s'], axis=2)) > 5.
    with pytest.raises(ValueError, match='explicit'):
        analyze_contact_motion(p, support, **args, blade_angle_rad=theta)


def test_changing_local_membership_does_not_create_centroid_velocity():
    times = np.arange(81)/100.
    points = np.array([[5., 50.], [25., 50.], [45., 50.], [65., 50.], [85., 50.]])
    p = np.tile(points, (len(times), 1, 1))
    args = _arguments(p, times)
    args.update(tip_xy=np.column_stack((40.+30*times, np.full(len(times), 50.))),
                reference_depth_px=np.full(len(points), 5.))
    _, arrays = analyze_contact_motion(p, np.ones(p.shape[:2], bool), **args,
        config=ContactMotionConfig(near_tip_radius_px=32., minimum_group_points=1))
    values = arrays['group_x_median']
    np.testing.assert_allclose(values[np.isfinite(values)], -30., atol=1e-8)
    assert np.any((arrays['group_membership_jaccard_previous'] > 0)
                  & (arrays['group_membership_jaccard_previous'] < 1))


def _recovering_patch():
    times = np.arange(121)/100.
    reference = np.array([[x, y] for y in (42., 46., 50.) for x in (36., 40., 44.)])
    center = reference.mean(0)
    stretch = 1+np.where(times < .6, np.maximum(times-.2, 0)*.75,
                         np.maximum(1.-times, 0)*.75)
    p = center+(reference-center)[None]*stretch[:, None, None]
    return times, p


def test_fixed_patch_recovery_is_a_candidate_during_loading_not_fracture_or_energy():
    times, p = _recovering_patch()
    support = np.ones(p.shape[:2], bool)
    args = _arguments(p, times)
    cfg = ContactMotionConfig(half_window_s=.035, minimum_candidate_duration_s=.03)
    summary, arrays = analyze_contact_motion(p, support, **args, config=cfg)
    assert summary['candidate_events']
    event = summary['first_geometric_relaxation_candidate']
    assert 58 <= event['first_evidence_source_frame'] <= 70
    assert not event['supplied_opening_evidence_overlaps']
    assert summary['physical_fracture_onset'] is None and summary['energy_release'] is None
    assert not arrays['patch_relaxation_candidate'][:55].any()
    assert all(len(patch['member_point_ids']) >= 6 for patch in summary['patches'])
    # A support break prevents a recovery fit from spanning hidden material.
    support[58:105] = False
    hidden, hidden_arrays = analyze_contact_motion(p, support, **args, config=cfg)
    assert not hidden_arrays['patch_relaxation_candidate'][58:105].any()
    assert hidden['candidate_events'] == []
    json.dumps(summary, allow_nan=False)


def test_externally_supplied_opening_and_proximity_stay_separate_from_kinematics():
    times = np.arange(61)/100.
    p = np.tile([[40., 42.], [45., 42.], [40., 46.], [45., 46.]], (len(times), 1, 1))
    args = _arguments(p, times)
    support = np.ones(p.shape[:2], bool)
    opening = np.zeros(len(times), np.int8)
    opening[25:] = 1
    contact = np.ones(p.shape[:2], np.int8)
    tangents = np.tile([0., 1.], p.shape[:2]+(1,))
    gap = np.full(p.shape[:2], 3.)
    before = p.copy()
    original_tip_support = args['tip_supported'].copy()
    cfg = ContactMotionConfig(contact_evidence_kind='geometric_proximity',
        contact_evidence_provenance='material/blade mask within 3 px',
        opening_evidence_provenance='synthetic image observation', minimum_group_points=1)
    summary, arrays = analyze_contact_motion(p, support, **args, config=cfg,
        observed_contact=contact, contact_tangent_xy=tangents,
        image_opening_observed=opening, blade_boundary_distance_px=gap, um_per_px=6.5)
    assert summary['first_supplied_image_opening_source_frame'] == 25
    assert summary['physical_fracture_onset'] is None
    assert not summary['candidate_events']
    assert summary['contact_evidence']['kind'] == 'geometric_proximity'
    np.testing.assert_array_equal(arrays['blade_boundary_distance_px'], gap)
    np.testing.assert_array_equal(p, before)
    np.testing.assert_array_equal(args['tip_supported'], original_tip_support)
    assert np.nanmedian(arrays['contact_candidate_local_tangent_relative_px_s']) == pytest.approx(-10.)
    np.testing.assert_allclose(arrays['group_y_median_um_s'], arrays['group_y_median']*6.5, equal_nan=True)
    json.dumps(summary, allow_nan=False)
    with pytest.raises(ValueError, match='provenance'):
        analyze_contact_motion(p, support, **args, image_opening_observed=opening)


def test_multiple_relaxing_blocks_require_disjoint_fixed_material_ids():
    times, p = _recovering_patch()
    p = np.concatenate((p, p+[-32., 0.]), axis=1)
    summary, arrays = analyze_contact_motion(p, np.ones(p.shape[:2], bool),
        **_arguments(p, times), config=ContactMotionConfig(half_window_s=.035,
            minimum_candidate_duration_s=.03, near_tip_radius_px=80.))
    assert summary['multi_patch_relaxation_intervals']
    assert max(arrays['disjoint_relaxing_patch_count']) >= 2
    first_two = summary['patches'][:2]
    assert set(first_two[0]['member_point_ids']).isdisjoint(first_two[1]['member_point_ids'])
    assert summary['physical_fracture_onset'] is None
