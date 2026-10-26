import numpy as np
import pytest

from src.analysis.loading_surface_sliding import FixedMaterialSurfaceProjector, build_loading_surface_sliding


def fixture():
    x, y = np.meshgrid(np.arange(20., 181., 6.), np.arange(104., 189., 6.))
    points = np.column_stack([x.ravel(), y.ravel()])
    reference = dict(boundary_xy=np.column_stack([np.arange(0., 201., 4.), np.full(51, 100.)]),
        surface_s_px=np.arange(-100., 101., 4.), tangent=np.array([1., 0.]),
        inward_normal=np.array([0., 1.]), contact_s_px=0.)
    return points, reference


def saved():
    return dict(frame_indices=np.array([7, 8, 9]), state=np.array([1, 0, 0]),
        reason=np.array(['ok', 'original_surface_mismatch', 'no_support']),
        current_marker_xy=np.array([[100., 100.], [106., 130.], [112., 160.]]),
        reference_marker_xy=np.array([[100., 100.], [np.nan, np.nan], [np.nan, np.nan]]),
        s_um=np.array([0., np.nan, np.nan]), low_um=np.array([-5., np.nan, np.nan]),
        high_um=np.array([5., np.nan, np.nan]), raw_reference_s_um=np.array([0., 30., 60.]))


def test_large_affine_map_projects_interior_address_conditionally_and_keeps_depth():
    points, reference = fixture()
    projector = FixedMaterialSurfaceProjector(points, np.arange(len(points)), reference, 5.)
    affine = np.array([[1.4, .2], [.12, .8]])
    offset = np.array([-30., 12.])
    current = points @ affine.T + offset
    original = np.array([112., 142.])
    result = projector.estimate_frame(current, np.ones(len(points), bool), affine @ original + offset,
                                     branch_labels=np.ones(len(points), int))
    assert result['state'] == 2
    assert result['s_um'] == pytest.approx(60.)
    assert result['normal_offset_px'] == pytest.approx(42.)
    np.testing.assert_allclose(result['reference_marker_xy'], [112., 100.], atol=1e-6)
    np.testing.assert_allclose(result['raw_inverse_material_xy'], original, atol=1e-6)


def test_all_original_layers_and_sparse_outer_ids_are_allowed():
    points, reference = fixture()
    projector = FixedMaterialSurfaceProjector(points, np.arange(len(points))+1000, reference, 5.)
    support = (points[:, 1] >= 148) & (np.arange(len(points)) % 3 == 0)
    result = projector.estimate_frame(points, support, [112., 164.])
    assert result['state'] == 2
    assert result['normal_offset_px'] == pytest.approx(64.)
    assert np.all(result['support_ids'] >= 1000)


def test_blade_occupied_and_unsupported_material_never_produces_address():
    points, reference = fixture()
    projector = FixedMaterialSurfaceProjector(points, np.arange(len(points)), reference, 5.)
    result = projector.estimate_frame(points, np.zeros(len(points), bool), [112., 140.])
    assert result['state'] == 0
    result = projector.estimate_frame(points, np.ones(len(points), bool), [112., 140.],
                                     blade_mask=np.ones((240, 240), bool))
    assert result['state'] == 0


def test_collapsed_noninvertible_patch_is_unknown():
    points, reference = fixture()
    projector = FixedMaterialSurfaceProjector(points, np.arange(len(points)), reference, 5.)
    current = points.copy(); current[:, 1] = 130.
    result = projector.estimate_frame(current, np.ones(len(points), bool), [112., 130.])
    assert result['state'] == 0


def test_loading_adapter_restores_projection_without_calling_it_supported_contact():
    points, reference = fixture()
    old = saved()
    old_copy = {k: v.copy() for k, v in old.items()}
    p = np.tile(points, (3, 1, 1))
    support = np.ones(p.shape[:2], bool); support[2] = False
    result = build_loading_surface_sliding(old, [7, 8, 9], old['current_marker_xy'],
        surface_reference=reference, um_per_px=5., seed_points=points,
        point_ids=np.arange(len(points)), positions=p, supported=support)
    np.testing.assert_array_equal(result['state'], [2, 2, 0])
    np.testing.assert_allclose(result['migration_um'][:2], [0., 30.], atol=1e-6)
    assert np.isnan(result['migration_um'][2])
    assert result['evidence_kind'][1] == 'material_block_projected_surface_address'
    np.testing.assert_array_equal(result['current_marker_xy'], old['current_marker_xy'])
    np.testing.assert_allclose(result['address_initial_xy'][1], [106., 100.], atol=1e-6)
    assert not result['original_surface_identity_proven'].any()
    for key in old:
        np.testing.assert_array_equal(old[key], old_copy[key])


def test_frame_labels_are_aligned_and_changed_tip_does_not_reuse_old_identity():
    _, reference = fixture()
    old = saved()
    tips = old['current_marker_xy'][[0, 2]].copy(); tips[0] += [10., 0.]
    result = build_loading_surface_sliding(old, [7, 9], tips, surface_reference=reference, um_per_px=5.)
    assert not result['state'].any()
    assert np.isnan(result['migration_um']).all()
    with pytest.raises(ValueError, match='Every loading frame'):
        build_loading_surface_sliding(old, [7, 11], tips, surface_reference=reference, um_per_px=5.)


def test_interpolated_tip_or_uncertain_blade_cannot_claim_direct_surface_support():
    _, reference = fixture()
    old = saved(); old['state'][0] = 2
    for options in (dict(tip_state=np.array([1, 2, 2])),
                    dict(blade_geometry_uncertain=np.array([True, False, False]))):
        result = build_loading_surface_sliding(old, [7, 8, 9], old['current_marker_xy'],
            surface_reference=reference, um_per_px=5., **options)
        assert result['state'][0] == 2
        assert result['strict_state'][0] == 2


def test_missing_baseline_does_not_fabricate_relative_migration():
    _, reference = fixture()
    old = saved(); old['state'][0] = 0
    result = build_loading_surface_sliding(old, [7, 8, 9], old['current_marker_xy'],
        surface_reference=reference, um_per_px=5.)
    assert np.isnan(result['migration_um']).all()
    assert not result['metadata']['first_contact_baseline_available']


def test_competing_material_maps_use_fixed_id_continuity_without_averaging_branches():
    points, reference = fixture()
    ids = np.arange(len(points))
    projector = FixedMaterialSurfaceProjector(points, ids, reference, 5.)
    branch = (np.rint((points[:, 1]-104.)/6.).astype(int) % 2) + 1
    current = points.copy(); current[branch == 2, 0] += 24.
    previous = dict(state=2, s_um=-60., branch_label=2, support_ids=ids[branch == 2])
    result = projector.estimate_frame(current, np.ones(len(points), bool), [112., 140.],
        branch_labels=branch, previous=previous)
    assert result['state'] == 2
    assert result['branch_label'] == 2
    assert result['s_um'] == pytest.approx(-60.)
    assert result['low_um'] < -60. and result['high_um'] > 60.
    absent = projector.estimate_frame(current, np.zeros(len(points), bool), [112., 140.],
        branch_labels=branch, previous=previous)
    assert absent['state'] == 0


def test_continuity_prior_cannot_override_actual_abrupt_supported_change():
    points, reference = fixture()
    projector = FixedMaterialSurfaceProjector(points, np.arange(len(points)), reference, 5.)
    previous = dict(state=2, s_um=1000., branch_label=1, support_ids=np.arange(len(points)))
    result = projector.estimate_frame(points, np.ones(len(points), bool), [112., 140.],
        branch_labels=np.ones(len(points), int), previous=previous)
    assert result['s_um'] == pytest.approx(60.)
