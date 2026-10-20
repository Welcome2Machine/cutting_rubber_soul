"""Meaningful checks for material identity, P1 transfer and screenshot provenance."""
import numpy as np
import pandas as pd
import pytest

from tools.analyze_prefracture_behavior import (P1Sampler, chord_summary, conditional_reference_points, fixed_cohort, validate_image_estimates,
                                               region_contrasts, region_summary)


def test_affine_p1_field_and_outside_are_preserved():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    sampler = P1Sampler(xy, np.array([[0, 1, 2], [0, 2, 3]]))
    points = np.array([[.2, .1], [.2, .8], [2., .2]])
    matrix = np.array([[.7, -.4], [.3, .2]])
    nodal = xy @ matrix.T + [3., -2.]
    values = sampler.sample(nodal, sampler.locate(points))
    np.testing.assert_allclose(values[:2], (points @ matrix.T + [3., -2.])[:2], atol=1e-14)
    assert np.isnan(values[2]).all()  # no nearest-node extrapolation


def test_fixed_cohort_excludes_restarted_and_invalid_points():
    rows = []
    for frame in [0, 1, 2]:
        for name in ['continuous', 'restarted', 'missing']:
            rows.append(dict(frame_id=frame, point_id=name, reference_frame_id=1 if name == 'restarted' and frame == 2 else 0,
                             valid=0 if name == 'missing' and frame == 1 else 1, x_px=1., y_px=2.))
    assert fixed_cohort(pd.DataFrame(rows), [0, 1, 2]) == ['continuous']


def test_ambiguous_point_identity_rejected():
    table = pd.DataFrame([dict(frame_id=0, point_id='a', reference_frame_id=0, valid=1, x_px=1., y_px=2.)] * 2)
    with pytest.raises(ValueError, match='Ambiguous'):
        fixed_cohort(table, [0])


def test_region_differences_cancel_common_translation_but_are_not_pointwise_strain():
    regions = np.array(['near_left', 'near_left', 'far_left', 'far_left', 'near_right', 'far_right'])
    values = np.array([[0., 4.], [2., 8.], [3., 2.], [4., 4.], [0., 9.], [2., 5.]])
    first = region_contrasts(region_summary(values, regions), 1)
    second = region_contrasts(region_summary(values + [100., -50.], regions), 1)
    assert first == second == dict(left=3., right=4.)






def test_chords_reject_rigid_motion_as_strain_and_detect_deformation():
    xy = np.array([[0., 0.], [1., 0.], [0., 2.], [1., 2.]])
    edges = np.array([[0, 1], [2, 3]])
    theta = .7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rigid = xy @ rotation.T + [20., -6.]
    assert chord_summary(xy, rigid, edges)['median'] == pytest.approx(1., abs=1e-14)
    stretched = xy * [1.3, .7]
    assert chord_summary(xy, stretched, edges)['median'] == pytest.approx(1.3)


def test_chords_cannot_use_missing_or_coincident_material_endpoints():
    with pytest.raises(ValueError, match='Distinct'):
        chord_summary(np.zeros((2, 2)), np.zeros((2, 2)), np.array([[0, 1]]))
    with pytest.raises(ValueError, match='observed'):
        chord_summary(np.array([[0., 0.], [1., 0.]]), np.array([[0., 0.], [np.nan, 0.]]), np.array([[0, 1]]))




def test_conditional_pixel_metric_is_explicit_and_transforms_both_axes():
    angle = np.deg2rad(55.)
    meta = dict(reference=dict(initial_surface_direction_image=[np.cos(angle), np.sin(angle)],
                               initial_surface_center_image_px=[0., 0.],
                               estimated_contact_blade_source_xy_px=[0., 0.]),
                coordinates=dict(conditional_image_scale_mm_px=.01))
    config = dict(drop_s_mm=7., height_mm=5., inclination_deg=60.)
    aspect = np.tan(angle) / np.tan(np.deg2rad(60.))
    _, _, image_basis, _ = conditional_reference_points(np.array([[1., 2.]]), meta, config, pixel_aspect=aspect)
    np.testing.assert_allclose(image_basis[0], [np.cos(np.deg2rad(60.)), np.sin(np.deg2rad(60.))], atol=1e-14)
    # The input mapping remains an explicit hypothesis, not a fitted/calibrated observation.
    with pytest.raises(ValueError, match='Positive'):
        conditional_reference_points(np.array([[1., 2.]]), meta, config, pixel_aspect=0.)


def test_force_reference_accepts_only_explicit_image_readings():
    image = pd.DataFrame(dict(depth_mm=[1., .5], force_n=[14.3, 6.6], lower_n=[13.5, 6.1], upper_n=[15.6, 7.5]))
    result = validate_image_estimates(image)
    assert result.depth_mm.tolist() == [.5, 1.]
    assert result.force_n.tolist() == [6.6, 14.3]
    with pytest.raises(ValueError, match='sensor channels'):
        validate_image_estimates(pd.DataFrame(dict(V01_N=[1., 2.], V02_N=[3., 4.])))


def test_image_reference_keeps_ranges_and_rejects_inconsistent_values():
    image = pd.DataFrame(dict(depth_mm=[.5, 1.], force_n=[6.6, 14.3], lower_n=[6.1, 13.5], upper_n=[7.5, 15.6]))
    image.loc[0, 'upper_n'] = 6.
    with pytest.raises(ValueError, match='visual ranges'):
        validate_image_estimates(image)
