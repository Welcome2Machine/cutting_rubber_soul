import numpy as np

from src.analysis.fixed_surface_origin import FixedSurfaceOriginTracker


def fixture():
    x, d = np.meshgrid(np.arange(20., 181., 8.), np.arange(2., 67., 8.))
    seed = np.column_stack((x.ravel(), 40.+d.ravel()))
    surface = np.column_stack((np.arange(0., 201., 2.), np.full(101, 40.)))
    ref = dict(boundary_xy=surface, surface_s_px=surface[:, 0]-100.,
               tangent=np.array([1., 0.]), inward_normal=np.array([0., 1.]))
    return seed, np.arange(len(seed))+1000, ref


def step(tracker, points, time, support=None, path=None, tip=(100., 130.), labels=None):
    if support is None:
        support = np.ones(len(points), bool)
    arc = None if path is None else np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return tracker.step(points, support, tip, labels, path, arc, time)


def test_rigid_and_affine_preserve_exact_material_origin():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    step(tracker, seed, 0., path=ref['boundary_xy'])
    A = np.array([[1.2, .25], [.15, .8]])
    b = np.array([3., -8.])
    result = step(tracker, seed@A.T+b, .1, path=ref['boundary_xy']@A.T+b)
    expected = np.array([100., 40.])@A.T+b
    assert np.allclose(result['raw_xy'], expected, atol=1.e-7)
    assert np.allclose(result['xy'], expected, atol=1.e-7)
    assert result['state'] == 1 and result['on_boundary']
    assert result['holdout_px'] < 1.e-7


def test_static_material_does_not_follow_descending_blade():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    for i in range(8):
        result = step(tracker, seed, i*.1, path=ref['boundary_xy'], tip=(100., 20.+i*10.))
        assert np.allclose(result['xy'], [100., 40.], atol=1.e-7)
        assert not result['contact_address_used']


def test_occlusion_recovery_keeps_ids_and_exposes_prediction():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    start = step(tracker, seed, 0., path=ref['boundary_xy'])
    for i in range(1, 6):
        result = step(tracker, seed, i*.1, support=np.zeros(len(seed), bool), path=ref['boundary_xy'])
        assert result['state'] == 4
        assert result['age_since_observation_s'] == i*.1
        assert np.allclose(result['xy'], [100., 40.], atol=1.e-7)
        assert result['sigma_px'] > start['sigma_px']
        assert np.array_equal(result['fixed_neighbourhood_ids'], start['fixed_neighbourhood_ids'])
    recovered = step(tracker, seed+np.array([2., 0.]), .6, path=ref['boundary_xy']+np.array([2., 0.]))
    assert recovered['state'] == 1
    assert np.allclose(recovered['xy'], [102., 40.], atol=1.e-7)
    assert set(recovered['support_ids']) <= set(start['fixed_neighbourhood_ids'])


def test_shallow_band_artificial_holdout_reports_extrapolation():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    result = step(tracker, seed+np.array([3., 4.]), 0., support=seed[:, 1] >= 58.,
                  path=ref['boundary_xy']+np.array([3., 4.]))
    assert result['state'] == 2
    assert result['minimum_initial_depth_px'] >= 18.
    assert np.allclose(result['raw_xy'], [103., 44.], atol=1.e-7)
    assert result['holdout_px'] < 1.e-7


def test_normal_contour_repair_does_not_reassign_tangential_identity():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    shifted_surface = ref['boundary_xy']+np.array([0., 6.])
    result = step(tracker, seed, 0., path=shifted_surface)
    assert np.allclose(result['raw_xy'], [100., 40.], atol=1.e-7)
    assert np.allclose(result['xy'], [100., 46.], atol=1.e-7)
    assert result['state'] == 2  # A six-pixel geometric repair is not direct support.
    # A remote contour endpoint cannot replace P0 with a nearby different ID.
    remote = np.array([[130., 40.], [180., 40.]])
    result = step(tracker, seed, .1, path=remote)
    assert not result['on_boundary']
    assert np.allclose(result['xy'], [100., 40.], atol=1.e-7)


def test_blade_crossing_and_illegal_side_switch_are_rejected():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, -20.)
    labels = np.ones((180, 240), np.uint8)
    labels[:100, 96:105] = 2
    first = step(tracker, seed, 0., path=ref['boundary_xy'], labels=labels, tip=(100., 100.))
    assert first['committed_blade_side'] == 1
    result = step(tracker, seed+np.array([40., 0.]), .1,
                  path=ref['boundary_xy']+np.array([40., 0.]), labels=labels, tip=(100., 100.))
    assert result['blade_crossing_rejected'] or result['side_change_rejected']
    assert result['state'] == 4
    assert result['xy'][0] < 96.
    assert set(result['same_id_increment_support_ids']) <= set(first['fixed_neighbourhood_ids'])


def test_available_unrelated_ids_cannot_replace_lost_origin_neighbourhood():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    first = step(tracker, seed, 0., path=ref['boundary_xy'])
    unrelated = ~np.isin(ids, first['fixed_neighbourhood_ids'])
    assert unrelated.any()
    result = step(tracker, seed+np.array([25., 0.]), .1, support=unrelated, path=ref['boundary_xy'])
    assert result['state'] == 4
    assert len(result['support_ids']) == 0
    assert np.allclose(result['raw_xy'], first['raw_xy'])


def test_identical_material_evidence_is_invariant_to_arbitrary_tip_path():
    seed, ids, ref = fixture()
    a = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    b = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    for i in range(5):
        shift = np.array([i*.7, i*.25])
        ra = step(a, seed+shift, i*.1, path=ref['boundary_xy']+shift, tip=(100., i*20.))
        rb = step(b, seed+shift, i*.1, path=ref['boundary_xy']+shift, tip=(200.-i*20., 150.))
        assert np.allclose(ra['raw_xy'], rb['raw_xy'])
        assert np.allclose(ra['xy'], rb['xy'])
        assert ra['state'] == rb['state']


def test_folded_contour_does_not_resolve_ambiguous_material_branch():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    # Two nonlocal contour branches are one pixel above/below the free P0.
    # Both satisfy the normal correction constraint; closest distance cannot
    # distinguish their material identity.
    folded = np.array([[0., 39.], [200., 39.], [200., 41.], [0., 41.]])
    result = step(tracker, seed, 0., path=folded)
    assert result['geometry_ambiguous']
    assert not result['on_boundary']
    assert result['geometry_alternative_arc_separation_px'] >= 24.
    assert result['geometry_alternative_distance_difference_px'] <= 2.
    assert np.allclose(result['xy'], result['raw_xy'])
    assert np.allclose(result['raw_xy'], [100., 40.], atol=1.e-7)


def test_adjacent_short_contour_segments_are_not_ambiguous_branches():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    dense = np.column_stack((np.arange(0., 201., .25), np.full(804, 41.)))
    result = step(tracker, seed, 0., path=dense)
    assert not result['geometry_ambiguous']
    assert result['on_boundary']
    assert np.allclose(result['xy'], [100., 41.], atol=1.e-7)


def test_local_increment_is_not_dragged_by_remote_deformation():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    step(tracker, seed, 0., path=ref['boundary_xy'])
    radius = np.linalg.norm(seed-np.array([100., 40.]), axis=1)
    delta = .5+4.*np.clip((radius-22.)/18., 0., 1.)
    current = seed+np.column_stack((delta, np.zeros(len(seed))))
    support = np.ones(len(seed), bool)
    local = tracker._increment(current, support, .1)
    broad = tracker._increment_legacy(current, support, .1)
    expected = np.array([100.5, 40.])
    assert local is not None and broad is not None
    assert np.linalg.norm(local['xy']-expected) < np.linalg.norm(broad['xy']-expected)
    assert local['current_radius'] == 18.
    assert local['holdout'] < 1.e-7
    assert local['hull_extrapolation'] <= 12.


def test_unsupported_increment_is_prediction_not_supported_propagation():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    first = step(tracker, seed, 0.)
    x = np.rint((seed[:, 0]-20.)/8.).astype(int)
    d = np.rint((seed[:, 1]-42.)/8.).astype(int)
    current = seed+np.column_stack((8.*(2*((x+d) % 2)-1), np.zeros(len(seed))))
    result = step(tracker, current, .1)
    assert result['state'] == 4
    assert len(result['same_id_increment_support_ids']) == 0
    assert np.allclose(result['raw_xy'], first['raw_xy'])
    assert result['sigma_px'] > first['sigma_px']


def test_loss_of_motion_evidence_holds_position_and_expands_uncertainty():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    step(tracker, seed, 0.)
    moving = step(tracker, seed+np.array([2., 0.]), .1)
    hidden = step(tracker, seed+np.array([4., 0.]), .2, support=np.zeros(len(seed), bool))
    assert hidden['state'] == 4
    assert np.allclose(hidden['raw_xy'], moving['raw_xy'])
    assert hidden['sigma_px'] >= moving['sigma_px']+2.


def test_moving_blade_covering_old_coordinate_does_not_lock_same_side_motion():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, -20.)
    labels = np.ones((180, 240), np.uint8)
    labels[:100, 96:105] = 2
    step(tracker, seed, 0., labels=labels, tip=(100., 100.), path=ref['boundary_xy'])
    for i in (1, 2, 3):
        labels = np.ones((180, 240), np.uint8)
        labels[:100, 82-4*i:105] = 2  # New knife silhouette covers previous P0.
        shift = np.array([-4.*i, 0.])
        result = step(tracker, seed+shift, i*.1, labels=labels,
                      tip=(100., 100.), path=ref['boundary_xy']+shift)
        assert result['previous_position_covered_by_current_blade']
        assert not result['blade_crossing_rejected']
        assert not result['side_change_rejected']
        assert result['state'] != 4
        assert np.allclose(result['xy'], [80.-4.*i, 40.], atol=1.e-7)


def test_near_edge_free_map_can_receive_normal_correction_before_blade_gate():
    seed, ids, ref = fixture()
    # A vertical original surface, with material to its left.
    depth = seed[:, 1]-40.
    seed = np.column_stack((100.-depth, seed[:, 0]))
    ref['boundary_xy'] = np.column_stack((np.full(101, 100.), np.arange(0., 201., 2.)))
    ref['tangent'], ref['inward_normal'] = np.array([0., 1.]), np.array([-1., 0.])
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    first_labels = np.ones((240, 180), np.uint8)
    first_labels[:150, 104:115] = 2
    step(tracker, seed, 0., labels=first_labels, tip=(110., 150.), path=ref['boundary_xy'])
    labels = np.ones_like(first_labels)
    labels[:150, 99:115] = 2
    result = step(tracker, seed, .1, labels=labels, tip=(110., 150.),
                  path=ref['boundary_xy']+np.array([-2., 0.]))
    assert result['raw_hidden_by_blade']
    assert result['on_boundary'] and not result['hidden_by_blade']
    assert result['geometry_correction_px'] <= 2.+1.e-7
    assert np.allclose(result['raw_xy'], [100., 100.], atol=1.e-7)
    assert np.allclose(result['xy'], [98., 100.], atol=1.e-7)
    assert not result['blade_crossing_rejected']


def test_occluded_origin_keeps_latent_fixed_id_motion_without_claiming_visibility():
    seed, ids, ref = fixture()
    tracker = FixedSurfaceOriginTracker(seed, ids, ref, 0.)
    step(tracker, seed, 0.)
    labels = np.ones((180, 240), np.uint8)
    labels[:100, 98:115] = 2
    result = step(tracker, seed+np.array([1., 0.]), .1, labels=labels, tip=(110., 100.))
    assert result['hidden_by_blade']
    assert not result['image_supported']
    assert not result['independent_position_observation']
    assert result['state'] != 4
    assert np.allclose(result['raw_xy'], [101., 40.], atol=1.e-7)
    assert not result['blade_crossing_rejected']
