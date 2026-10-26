"""Physical-definition checks for original-surface forward transport."""
from dataclasses import replace

import numpy as np
import pytest

from src.analysis.surface_observability import (
    SurfaceObservabilityConfig, contact_from_material_surface, map_material_surface,
)


def grid(x=(20, 100), y=(30, 70), step=5):
    xx, yy = np.meshgrid(np.arange(x[0], x[1]+1, step), np.arange(y[0], y[1]+1, step))
    return np.column_stack((xx.ravel(), yy.ravel())).astype(float)


def inputs(points, surface):
    return dict(point_ids=np.arange(len(points))+1000, point_side=np.zeros(len(points), int),
                surface_side=np.zeros(len(surface), int),
                observation_mask=np.ones((500, 500), bool),
                config=SurfaceObservabilityConfig(neighbor_counts=(32, 48), contact_tolerance_px=.5,
                                                   ambiguity_tolerance_px=.25))


@pytest.mark.parametrize('matrix,translation', [
    (np.eye(2), np.array([12., 7.])),
    (np.array([[1.8, .9], [.3, .55]]), np.array([25., 13.])),
    (np.array([[.8, -.6], [.6, .8]]), np.array([80., 10.])),
])
def test_rigid_rotation_and_large_affine_recover_original_address(matrix, translation):
    ref = grid()
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    current = ref @ matrix.T+translation
    tip = np.array([57., 30.]) @ matrix.T+translation
    result = contact_from_material_surface(ref, current, np.ones(len(ref), bool), surface,
                                           tip, **inputs(ref, surface))
    np.testing.assert_allclose(result['surface']['mapped_xy_px'], surface @ matrix.T+translation, atol=1.e-10)
    assert result['state'] == 'unique_candidate'
    assert abs(result['candidates'][0]['s_estimate_px']-27.) < 1.e-9
    assert result['evidence'] == 'observed-supported'
    assert result['contact_observed'] is False
    assert result['new_fracture_surface_identity_assigned'] is False


def test_deep_only_fit_is_extrapolation_not_observed_surface():
    ref = grid(y=(40, 75))
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    result = contact_from_material_surface(ref, ref, np.ones(len(ref), bool), surface,
                                           [60., 30.], **inputs(ref, surface))
    assert result['state'] == 'unique_candidate'
    assert result['evidence'] == 'model-extrapolated'
    assert np.nanmin(result['surface']['hull_extrapolation_px']) >= 10.-1.e-9
    assert not np.any(result['surface']['evidence'] == 'observed-supported')
    assert np.isnan(result['surface']['surface_heldout_rms_px']).all()


def test_opposite_side_observations_are_not_mixed():
    base = grid()
    ref = np.vstack((base, base))
    current = np.vstack((base+[2, 1], base+[150, 100]))
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    options = inputs(ref, surface)
    options['point_side'][len(base):] = 1
    result = map_material_surface(ref, current, np.ones(len(ref), bool), surface, **options)
    np.testing.assert_allclose(result['mapped_xy_px'], surface+[2, 1], atol=1.e-10)
    assert all(max(ids) < 1000+len(base) for ids in result['support_ids'])


def test_no_observations_preserves_missing_addresses():
    ref = grid()
    surface = np.array([[30., 30.], [90., 30.]])
    result = contact_from_material_surface(ref, ref, np.zeros(len(ref), bool), surface,
                                           [60., 30.], **inputs(ref, surface))
    assert result['state'] == 'unavailable'
    assert result['candidates'] == []
    assert np.isnan(result['surface']['mapped_xy_px']).all()


def test_two_contact_minima_remain_distinct_material_intervals():
    ref = grid(y=(30, 50))
    surface = np.vstack((np.column_stack((np.arange(20, 101, 5), np.full(17, 30.))),
                         [[100., 35.], [100., 40.], [100., 45.]],
                         np.column_stack((np.arange(100, 19, -5), np.full(17, 50.)))))
    options = inputs(ref, surface)
    options['config'] = replace(options['config'], contact_tolerance_px=11., ambiguity_tolerance_px=.1)
    result = contact_from_material_surface(ref, ref, np.ones(len(ref), bool), surface, [60., 40.], **options)
    assert result['state'] == 'ambiguous'
    assert len(result['candidates']) == 2
    np.testing.assert_allclose([c['s_estimate_px'] for c in result['candidates']], [40., 140.], atol=1.e-8)
    assert result['candidates'][0]['s_interval_px'][1] < result['candidates'][1]['s_interval_px'][0]


def test_shadow_and_unknown_side_are_conditional_and_blade_occlusion_is_unavailable():
    ref = grid()
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    options = inputs(ref, surface)
    options['observation_mask'][:35] = False
    result = map_material_surface(ref, ref, np.ones(len(ref), bool), surface, **options)
    assert not np.any(result['evidence'] == 'observed-supported')
    assert np.isfinite(result['mapped_xy_px']).any()
    options = inputs(ref, surface)
    del options['point_side'], options['surface_side']
    result = map_material_surface(ref, ref, np.ones(len(ref), bool), surface, **options)
    assert np.all(result['evidence'] == 'model-extrapolated')
    options = inputs(ref, surface)
    blade = np.zeros((500, 500), bool)
    blade[29:32, 50:71] = True
    result = map_material_surface(ref, ref, np.ones(len(ref), bool), surface, blade_mask=blade, **options)
    assert np.isnan(result['mapped_xy_px'][(surface[:, 0] >= 50) & (surface[:, 0] <= 70)]).all()


def test_missing_surface_gap_does_not_produce_fabricated_contact():
    ref = grid()
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    options = inputs(ref, surface)
    blade = np.zeros((500, 500), bool)
    blade[29:32, 50:71] = True
    result = contact_from_material_surface(ref, ref, np.ones(len(ref), bool), surface,
                                           [60., 30.], blade_mask=blade, **options)
    assert result['state'] == 'unavailable'
    assert not result['candidates']


def test_sparse_surface_edge_cannot_bridge_blade_interior():
    ref = grid()
    surface = np.array([[40., 30.], [80., 30.]])
    blade = np.zeros((500, 500), bool)
    blade[29:32, 58:63] = True
    result = contact_from_material_surface(ref, ref, np.ones(len(ref), bool), surface,
                                           [57.5, 30.], blade_mask=blade, **inputs(ref, surface))
    assert np.isfinite(result['surface']['mapped_xy_px']).all()
    assert not result['surface']['segment_valid'][0]
    assert result['state'] == 'unavailable'


def test_surface_depth_motion_mismatch_is_not_hidden_by_good_deep_fit():
    ref = grid()
    current = ref.copy()
    current[ref[:, 1] == 30, 0] += 15.
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    result = map_material_surface(ref, current, np.ones(len(ref), bool), surface, **inputs(ref, surface))
    assert np.all(result['evidence'] == 'unavailable')
    assert all(reason in ('heldout_or_fit_failure', 'surface_depth_prediction_mismatch') for reason in result['reason'])


def test_strong_tangential_compression_returns_address_range_not_stopped_slip():
    ref = grid()
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    current = ref*[.01, 1.]+[100., 0.]
    result = contact_from_material_surface(ref, current, np.ones(len(ref), bool), surface,
                                           [100.6, 30.], **inputs(ref, surface))
    assert result['state'] == 'ambiguous'
    assert result['candidates'][0]['compression_unidentifiable']
    assert np.diff(result['candidates'][0]['s_interval_px'])[0] > 40.


def test_rank_deficiency_and_recycled_identity_fail_explicitly():
    ref = np.column_stack((np.arange(20, 101, 2), np.full(41, 30.)))
    surface = ref[::2]
    options = inputs(ref, surface)
    result = map_material_surface(ref, ref, np.ones(len(ref), bool), surface, **options)
    assert np.all(result['evidence'] == 'unavailable')
    options['point_ids'][1] = options['point_ids'][0]
    with pytest.raises(ValueError, match='unique immutable'):
        map_material_surface(ref, ref, np.ones(len(ref), bool), surface, **options)


def test_reference_row_reordering_preserves_physical_address():
    ref = grid()
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    order = np.random.default_rng(4).permutation(len(ref))
    options = inputs(ref, surface)
    options['point_ids'] = options['point_ids'][order]
    result = contact_from_material_surface(ref[order], ref[order]+[4, 5], np.ones(len(ref), bool),
                                           surface, [61., 35.], **options)
    assert abs(result['candidates'][0]['s_estimate_px']-27.) < 1.e-8


def test_quadratic_bending_truth_is_recovered_without_relaxing_fit_gates():
    ref = grid(step=3)
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    def bending(points):
        result = points.copy()+[6., 4.]
        result[:, 1] += .12*(points[:, 0]-60.)**2
        return result
    options = inputs(ref, surface)
    options['config'] = replace(options['config'], neighbor_counts=(48,))
    affine = map_material_surface(ref, bending(ref), np.ones(len(ref), bool), surface, **options)
    options['config'] = replace(options['config'], model='surface-quadratic')
    curve = contact_from_material_surface(ref, bending(ref), np.ones(len(ref), bool),
                                          surface, bending(np.array([[60., 30.]]))[0], **options)
    np.testing.assert_allclose(curve['surface']['mapped_xy_px'], bending(surface), atol=1.e-10)
    assert curve['state'] == 'unique_candidate'
    assert abs(curve['candidates'][0]['s_estimate_px']-30.) < 1.e-9
    assert np.nanmax(curve['surface']['surface_heldout_rms_px']) < 1.e-9
    assert np.nanmax(curve['surface']['heldout_rms_px']) < 1.e-9
    assert np.sum(affine['evidence'] != 'unavailable') < len(surface)


def test_surface_quadratic_contains_affine_and_is_rotation_equivariant():
    ref = grid(step=3)
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    matrix = np.array([[1.4, .6], [.2, .65]])
    theta = np.deg2rad(35.)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    offset = np.array([120., 40.])
    reference = ref @ rotation.T+offset
    current = (ref @ matrix.T+[3., 4.]) @ rotation.T+offset
    original_surface = surface @ rotation.T+offset
    options = inputs(ref, surface)
    options['config'] = replace(options['config'], model='surface-quadratic')
    actual = map_material_surface(reference, current, np.ones(len(ref), bool), original_surface, **options)
    expected = (surface @ matrix.T+[3., 4.]) @ rotation.T+offset
    np.testing.assert_allclose(actual['mapped_xy_px'], expected, atol=1.e-10)
    np.testing.assert_allclose(actual['jacobian_determinant'], np.linalg.det(matrix), atol=1.e-10)


def test_quadratic_band_excludes_deep_inconsistent_motion():
    ref = grid(step=3)
    current = ref.copy()
    current[ref[:, 1] > 48] += [70., 80.]
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    options = inputs(ref, surface)
    options['config'] = replace(options['config'], model='surface-quadratic')
    result = map_material_surface(ref, current, np.ones(len(ref), bool), surface, **options)
    np.testing.assert_allclose(result['mapped_xy_px'], surface, atol=1.e-10)
    for used in result['support_ids']:
        assert np.all(ref[np.array(used)-1000, 1] <= 48.)


def test_quadratic_missing_depth_support_never_falls_back_to_affine_observation():
    ref = np.column_stack((np.arange(20, 101, 2), np.full(41, 30.)))
    surface = ref[::2]
    options = inputs(ref, surface)
    options['config'] = replace(options['config'], model='surface-quadratic')
    result = map_material_surface(ref, ref, np.ones(len(ref), bool), surface, **options)
    assert np.all(result['evidence'] == 'unavailable')
    assert np.all(result['failure_category'] == 'model_identification_failure')
    assert result['model'] == 'surface-quadratic'
    missing = map_material_surface(ref, ref, np.zeros(len(ref), bool), surface, **options)
    assert np.all(missing['failure_category'] == 'insufficient_image_support')


def test_quadratic_preserves_layer_validation_and_blade_rejection():
    ref = grid(step=3)
    current = ref.copy()
    current[ref[:, 1] == 30, 0] += 15.
    surface = np.column_stack((np.arange(30, 91, 2), np.full(31, 30.)))
    options = inputs(ref, surface)
    options['config'] = replace(options['config'], model='surface-quadratic')
    mismatch = map_material_surface(ref, current, np.ones(len(ref), bool), surface, **options)
    assert not np.any(mismatch['evidence'] == 'observed-supported')
    assert np.any(mismatch['failure_category'] == 'independent_model_prediction_failure')
    blade = np.zeros((500, 500), bool)
    blade[29:32, 50:71] = True
    blocked = contact_from_material_surface(ref, ref, np.ones(len(ref), bool), surface,
                                            [60., 30.], blade_mask=blade, **options)
    assert blocked['state'] == 'unavailable'
    assert np.isnan(blocked['surface']['mapped_xy_px'][(surface[:, 0] >= 50) & (surface[:, 0] <= 70)]).all()


def curve_inputs(points, surface):
    options = inputs(points, surface)
    options['config'] = replace(options['config'], model='surface-curve', neighbor_counts=(12, 20, 32))
    return options


def straight_strip(offset=0.):
    surface = np.column_stack((np.arange(20., 101., 2.), np.full(41, 30.)))
    return surface+[0., offset], surface


def bent_displacement(points):
    return np.column_stack((.15*(points[:, 0]-60.)+4., .04*(points[:, 0]-60.)**2+8.))


def test_curve_recovers_bent_surface_from_collinear_material_ids():
    ref, surface = straight_strip()
    current = ref+bent_displacement(ref)
    expected = surface+bent_displacement(surface)
    result = contact_from_material_surface(ref, current, np.ones(len(ref), bool), surface,
                                           expected[17], **curve_inputs(ref, surface))
    np.testing.assert_allclose(result['surface']['mapped_xy_px'], expected, atol=1.e-10)
    assert result['state'] == 'unique_candidate'
    assert abs(result['candidates'][0]['s_estimate_px']-34.) < 1.e-9
    assert result['evidence'] == 'model-extrapolated'
    assert not result['contact_observed'] and not result['topology_verified']
    assert np.isnan(result['surface']['jacobian_determinant']).all()
    assert '1D boundary' in result['surface']['jacobian_definition']
    assert np.nanmax(result['surface']['heldout_rms_px']) < 1.e-9
    assert np.isnan(result['surface']['surface_heldout_rms_px']).all()  # no depth-held-out test exists for 1D
    np.testing.assert_allclose(result['surface']['curve_heldout_rms_px'], result['surface']['heldout_rms_px'])


def test_curve_thin_strip_uses_displacement_and_reports_actual_reference_offsets():
    ref, surface = straight_strip(offset=1.5)
    result = map_material_surface(ref, ref+bent_displacement(ref), np.ones(len(ref), bool),
                                  surface, **curve_inputs(ref, surface))
    # Fitting current x/y directly would incorrectly add the 1.5 px strip offset.
    np.testing.assert_allclose(result['mapped_xy_px'], surface+bent_displacement(surface), atol=1.e-10)
    np.testing.assert_allclose(result['maximum_reference_surface_distance_px'], 1.5)
    assert all(np.allclose(offset, 1.5) for offset in result['reference_surface_offset_px'])
    assert np.all(result['evidence'] == 'model-extrapolated')
    assert 'depth gradient is unmeasured' in result['thin_strip_assumption']


def test_curve_does_not_use_deep_ids_or_relabel_them_as_surface():
    edge, surface = straight_strip(offset=1.)
    ref = np.vstack((edge, grid(y=(40, 75))))
    current = ref+[4., 8.]
    current[len(edge):] += [120., 80.]
    result = map_material_surface(ref, current, np.ones(len(ref), bool), surface,
                                  **curve_inputs(ref, surface))
    np.testing.assert_allclose(result['mapped_xy_px'], surface+[4., 8.], atol=1.e-10)
    assert all(max(used) < 1000+len(edge) for used in result['support_ids'])
    assert not result['new_fracture_surface_identity_assigned']


@pytest.mark.parametrize('gap_type', ['unsupported', 'missing_rows', 'blade', 'visibility'])
def test_curve_missing_original_surface_chain_is_not_bridged(gap_type):
    ref, surface = straight_strip()
    support = np.ones(len(ref), bool)
    missing = (ref[:, 0] >= 48.) & (ref[:, 0] <= 62.)
    if gap_type == 'missing_rows':
        ref, support = ref[~missing], support[~missing]
    elif gap_type == 'unsupported':
        support[missing] = False
    options = curve_inputs(ref, surface)
    if gap_type == 'blade':
        blade = np.zeros((500, 500), bool)
        blade[29:32, 48:63] = True
        options['blade_mask'] = blade
    elif gap_type == 'visibility':
        options['observation_mask'][29:32, 48:63] = False
    result = contact_from_material_surface(ref, ref, support, surface, [55., 30.], **options)
    assert result['state'] == 'unavailable'
    in_gap = (surface[:, 0] >= 48.) & (surface[:, 0] <= 62.)
    assert np.isnan(result['surface']['mapped_xy_px'][in_gap]).all()
    assert np.isfinite(result['surface']['mapped_xy_px'][surface[:, 0] <= 40.]).all()
    assert not np.any(result['surface']['segment_valid'][np.flatnonzero(in_gap)[:-1]])


def test_curve_sparse_query_vertices_never_join_different_support_components():
    ref, _ = straight_strip()
    surface = np.array([[20., 30.], [40., 30.], [80., 30.], [100., 30.]])
    support = ~((ref[:, 0] >= 48.) & (ref[:, 0] <= 62.))
    result = contact_from_material_surface(ref, ref, support, surface, [60., 30.], **curve_inputs(ref, surface))
    assert np.isfinite(result['surface']['mapped_xy_px']).all()
    assert not result['surface']['segment_valid'][1]
    assert result['state'] == 'unavailable'


def test_curve_with_two_distinct_material_addresses_at_same_location_is_ambiguous():
    ref, surface = straight_strip()
    current = np.column_stack((np.full(len(ref), 60.), 30.+.04*(ref[:, 0]-60.)**2))
    result = contact_from_material_surface(ref, current, np.ones(len(ref), bool), surface,
                                           [60., 46.], **curve_inputs(ref, surface))
    assert result['state'] == 'ambiguous'
    assert len(result['candidates']) == 2
    np.testing.assert_allclose([item['s_estimate_px'] for item in result['candidates']], [20., 60.], atol=1.e-9)
    assert not result['contact_observed']


def test_curve_strong_compression_reports_ambiguity_not_zero_slip():
    ref, surface = straight_strip()
    current = ref*[.01, 1.]+[100., 0.]
    result = contact_from_material_surface(ref, current, np.ones(len(ref), bool), surface,
                                           [100.6, 30.], **curve_inputs(ref, surface))
    assert result['state'] == 'ambiguous'
    assert result['candidates'][0]['compression_unidentifiable']
    assert np.diff(result['candidates'][0]['s_interval_px'])[0] > 40.


def test_curve_independent_identity_validation_rejects_large_local_outlier():
    ref, surface = straight_strip()
    current = ref.copy()
    current[20] += [0., 40.]
    options = curve_inputs(ref, surface)
    options['config'] = replace(options['config'], neighbor_counts=(12,))
    result = contact_from_material_surface(ref, current, np.ones(len(ref), bool), surface,
                                           [60., 30.], **options)
    assert result['state'] == 'unavailable'
    assert result['surface']['failure_category'][20] == 'independent_model_prediction_failure'
    far = surface[:, 0] < 35.
    np.testing.assert_allclose(result['surface']['mapped_xy_px'][far], surface[far], atol=1.e-10)


def test_curve_identity_reordering_and_known_side_labels_preserve_address():
    edge, surface = straight_strip()
    ref = np.vstack((edge, edge))
    current = np.vstack((edge+[4., 8.], edge+[100., 90.]))
    options = curve_inputs(ref, surface)
    options['point_side'][len(edge):] = 1
    order = np.random.default_rng(9).permutation(len(ref))
    options['point_side'] = options['point_side'][order]
    options['point_ids'] = options['point_ids'][order]
    result = contact_from_material_surface(ref[order], current[order], np.ones(len(ref), bool),
                                           surface, [64., 38.], **options)
    assert result['state'] == 'unique_candidate'
    assert abs(result['candidates'][0]['s_estimate_px']-40.) < 1.e-9
    assert all(max(used) < 1000+len(edge) for used in result['surface']['support_ids'])


def test_curve_deep_only_or_too_few_independent_ids_stays_unavailable():
    edge, surface = straight_strip()
    for ref in (edge+[0., 5.], edge[18:24]):
        result = contact_from_material_surface(ref, ref, np.ones(len(ref), bool), surface,
                                               [60., 30.], **curve_inputs(ref, surface))
        assert result['state'] == 'unavailable'
        assert np.isnan(result['surface']['mapped_xy_px']).all()


def test_curve_coarse_query_edge_does_not_bridge_robustly_rejected_identity():
    ref = np.column_stack((np.arange(10., 131., 5.), np.full(25, 50.)))
    surface = ref[::2].copy()
    current = ref.copy()
    current[9, 1] += 1.7  # x=55: between the accepted query nodes x=50 and 60
    result = map_material_surface(ref, current, np.ones(len(ref), bool), surface,
                                  **curve_inputs(ref, surface))
    j = int(np.flatnonzero(surface[:, 0] == 50.)[0])
    assert np.isfinite(result['mapped_xy_px'][j:j+2]).all()
    assert not result['segment_valid'][j]
    assert result['segment_reason'][j] == 'rejected_surface_identity_gap'
    assert 1009 in result['curve_rejected_support_ids'][j]+result['curve_rejected_support_ids'][j+1]


def test_curve_coarse_polyline_does_not_replace_known_bend_with_straight_chord():
    ref, _ = straight_strip()
    surface = ref[[0, -1]]
    result = map_material_surface(ref, ref+bent_displacement(ref), np.ones(len(ref), bool), surface,
                                  **curve_inputs(ref, surface))
    assert np.isfinite(result['mapped_xy_px']).all()
    assert not result['segment_valid'][0]
    assert result['segment_reason'][0] == 'surface_polyline_resolution_or_support_error'
