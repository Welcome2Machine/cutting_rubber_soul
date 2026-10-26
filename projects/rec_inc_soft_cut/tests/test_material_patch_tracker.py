"""Ground-truth geometric tests; consistency alone is not tracking truth."""
import cv2
import numpy as np

from src.tracking.material_patch_tracker import (MaterialPatchConfig, MaterialPatchTracker,
                                                masked_gray, verify_material_candidates)


def texture(shape=(180, 220)):
    rng = np.random.default_rng(181)
    image = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (3, 3), .7)
    return np.clip((image-.5)*1.7+.5, .02, .98)


def warped(image, affine, offset, gain=1., bias=0.):
    result = cv2.warpAffine(image, np.column_stack((affine, offset)),
                            (image.shape[1], image.shape[0]))
    return np.clip(result*gain+bias, 0, 1)


def test_known_affine_and_brightness():
    image = texture()
    points = np.array([[65, 55], [100, 60], [80, 105]], np.float32)
    mask = np.ones(image.shape, bool)
    tracker = MaterialPatchTracker(image, mask, points)
    affine = np.array([[1.08, .04], [-.025, .95]], np.float32)
    offset = np.array([-3., 2.])
    target = warped(image, affine, offset, 1.18, -.04)
    result = tracker.step(target, mask)
    truth = points @ affine.T + offset
    assert result['supported'].all(), result
    assert np.max(np.linalg.norm(result['positions']-truth, axis=1)) < .3
    assert np.max(np.abs(result['affine']-affine)) < .04


def test_partial_blade_mask_keeps_surface_ids_supported():
    image = texture()
    points = np.array([[70, 81], [105, 81]], np.float32)
    material = np.ones(image.shape, bool)
    material[:80] = False
    tracker = MaterialPatchTracker(image, material, points)
    target = warped(image, np.eye(2), [1.2, .6])
    blade = np.zeros(image.shape, bool)
    blade[:80] = True
    target[blade] = 1.
    result = tracker.step(target, material, blade_mask=blade)
    assert result['supported'].all(), result
    assert np.max(np.linalg.norm(result['positions']-points-[1.2, .6], axis=1)) < .35
    np.testing.assert_array_equal(result['point_ids'], np.arange(2))


def test_cumulative_stretch_exceeds_initial_template_limit():
    image = texture((210, 280))
    points = np.array([[80, 75], [100, 110], [125, 90]], np.float32)
    mask = np.ones(image.shape, bool)
    tracker = MaterialPatchTracker(image, mask, points,
        MaterialPatchConfig(keyframe_interval=5, reference_check_interval=3))
    errors = []
    supports = []
    for step in range(1, 37):
        fraction = step/36
        affine = np.array([[1+1.15*fraction, .15*fraction],
                           [-.04*fraction, 1-.22*fraction]], np.float32)
        offset = np.array([-85*fraction, 15*fraction])
        target = warped(image, affine, offset, 1+.1*fraction, -.02*fraction)
        result = tracker.step(target, mask)
        truth = points @ affine.T + offset
        errors.append(np.linalg.norm(result['positions']-truth, axis=1).max())
        supports.append(result['supported'].mean())
    assert min(supports) == 1, supports
    assert max(errors) < 1., errors
    assert np.linalg.svd(result['cumulative_affine'], compute_uv=False)[:, 0].min() > 2.


def test_full_occlusion_retains_ids_and_can_recover():
    image = texture()
    points = np.array([[70, 70], [100, 95]], np.float32)
    mask = np.ones(image.shape, bool)
    tracker = MaterialPatchTracker(image, mask, points)
    first = tracker.step(warped(image, np.eye(2), [1., 0]), mask)
    assert first['supported'].all()
    hidden = tracker.step(np.full_like(image, .8), np.zeros_like(mask))
    assert not hidden['supported'].any()
    assert hidden['inferred'].all()
    assert np.isfinite(hidden['positions']).all()
    np.testing.assert_array_equal(hidden['point_ids'], [0, 1])
    recovered = tracker.step(warped(image, np.eye(2), [3., 0]), mask)
    assert recovered['supported'].all(), recovered
    assert np.max(np.linalg.norm(recovered['positions']-points-[3., 0], axis=1)) < .35
    assert not recovered['gap_age'].any()


def test_textureless_does_not_become_measured_or_update_reference():
    image = np.full((100, 100), .5, np.float32)
    points = np.array([[50., 50.]], np.float32)
    mask = np.ones(image.shape, bool)
    tracker = MaterialPatchTracker(image, mask, points)
    result = tracker.step(image, mask)
    assert not result['supported'].any()
    assert tracker.recent.age[0] == 0
    np.testing.assert_allclose(result['positions'], points)


def test_wrong_neural_proposal_does_not_replace_self_match():
    image = texture()
    points = np.array([[70., 70.], [115., 95.]], np.float32)
    mask = np.ones(image.shape, bool)
    tracker = MaterialPatchTracker(image, mask, points)
    result = tracker.step(warped(image, np.eye(2), [1.1, -.7]), mask,
                          initial_positions=points+[25., 15.])
    assert result['supported'].all(), result
    assert np.max(np.linalg.norm(result['positions']-points-[1.1, -.7], axis=1)) < .3


def test_local_unmasked_moving_glint_does_not_replace_material_motion():
    image = texture()
    points = np.array([[80., 80.], [130., 100.]], np.float32)
    mask = np.ones(image.shape, bool)
    tracker = MaterialPatchTracker(image, mask, points)
    target = warped(image, np.eye(2), [1.2, .7])
    target[75:84, 84:88] = 1.
    target[94:103, 134:138] = 1.
    result = tracker.step(target, mask)
    assert result['supported'].all(), result
    assert np.max(np.linalg.norm(result['positions']-points-[1.2, .7], axis=1)) < .45
    assert np.all(result['photometric_inlier_fraction'] < 1.)


def test_blade_occluded_center_is_forecast_even_with_visible_surrounding_texture():
    image = texture()
    points = np.array([[80., 80.]], np.float32)
    mask = np.ones(image.shape, bool)
    tracker = MaterialPatchTracker(image, mask, points)
    blade = np.zeros(image.shape, bool)
    blade[79:82, 79:82] = True
    result = tracker.step(image, mask, blade_mask=blade)
    assert not result['supported'].any()
    assert result['inferred'].all()
    np.testing.assert_allclose(result['positions'], points)


def test_opposite_tool_sides_do_not_form_one_affine_patch():
    image = texture()
    mask = np.ones(image.shape, bool)
    blade = np.zeros(image.shape, bool)
    blade[:, 100:104] = True
    reference = image.copy()
    reference[blade] = 1.
    points = np.array([[96., 80.], [108., 110.]], np.float32)
    tracker = MaterialPatchTracker(reference, mask, points, blade_mask=blade)
    left = warped(image, np.eye(2), [.3, .65])
    right = warped(image, np.eye(2), [-.25, -2.2])
    target = left.copy()
    target[:, 104:] = right[:, 104:]
    target[blade] = 1.
    result = tracker.step(target, mask, blade_mask=blade)
    truth = points + [[.3, .65], [-.25, -2.2]]
    assert result['supported'].all(), result
    assert np.max(np.linalg.norm(result['positions']-truth, axis=1)) < .45
    # Even the larger support must not borrow opposite-side material evidence.
    assert np.all(result['valid_fraction'] < .8)


def test_reverse_affine_cannot_borrow_pixels_from_another_atlas_id():
    from src.tracking.material_patch_tracker import _Image, _Bank, _fit
    image = texture()
    source = _Image(image, np.ones(image.shape, bool))
    points = np.array([[50., 80.], [150., 80.]], np.float32)
    identity = np.repeat(np.eye(2, dtype=np.float32)[None], 2, axis=0)
    bank = _Bank(source, points, identity, 0, 65)
    corrupted_atlas = bank.image.gray.copy()
    corrupted_atlas[:, 65:] = 1-corrupted_atlas[:, 65:]
    other = _Image(corrupted_atlas, bank.image.mask)
    config = MaterialPatchConfig()
    inverse_large_compression = np.array([[[2.8, 0], [0, 1]]], np.float32)
    bounds = np.column_stack((bank.slot[:1]+2, bank.slot[:1]+bank.side-3))
    kwargs = dict(reference=source, p=points[:1], q0=bank.atlas_points[:1],
                  affine0=inverse_large_compression, config=config, radius=19, iterations=1)
    own_a = _fit(target=bank.image, target_bounds=bounds, **kwargs)
    own_b = _fit(target=other, target_bounds=bounds, **kwargs)
    np.testing.assert_array_equal(own_a['positions'], own_b['positions'])
    np.testing.assert_array_equal(own_a['relative_affine'], own_b['relative_affine'])
    np.testing.assert_array_equal(own_a['valid_fraction'], own_b['valid_fraction'])
    # This fixture would leak into the next ID if only the narrow zero-mask
    # tile border were used. Verify the test actually exercises that failure.
    leaky_a = _fit(target=bank.image, **kwargs)
    leaky_b = _fit(target=other, **kwargs)
    assert np.max(np.abs(leaky_a['relative_affine']-leaky_b['relative_affine'])) > 1e-3


def test_reverse_center_outside_own_atlas_tile_cannot_be_accepted():
    from src.tracking.material_patch_tracker import _Image, _fit
    image = texture()
    source = _Image(image, np.ones(image.shape, bool))
    p = np.array([[100., 80.]], np.float32)
    result = _fit(source, source, p, p, np.eye(2, dtype=np.float32)[None],
                  MaterialPatchConfig(), 10, target_bounds=np.array([[2, 2, 62, 62]]))
    assert not result['accepted'].any()


def test_fast_blade_ray_skip_is_exact_for_subpixels_edges_and_outside_image():
    from src.tracking.material_patch_tracker import _Image, _sample, _unblocked_support
    rng = np.random.default_rng(627)
    blade = np.zeros((151, 173), np.float32)
    blade[20:105, 72:81] = 1.
    blade[100:106, 74:79] = .13
    blade[0, 0] = 1.
    blade[-1, -1] = .4
    image = _Image(np.full_like(blade, .5), np.ones_like(blade), blade)
    centers = rng.uniform([-20, -20], [195, 175], (200, 2)).astype(np.float32)
    samples = centers[:, None] + rng.uniform(-45, 45, (200, 121, 2)).astype(np.float32)
    fractions = np.linspace(.125, .875, 7, dtype=np.float32)
    rays = centers[:, None, None] + fractions[None, :, None, None]*(samples-centers[:, None])[:, None]
    original = ~(_sample(blade, rays.reshape(len(centers), -1, 2)).reshape(len(centers), 7, -1) > .25).any(1)
    np.testing.assert_array_equal(_unblocked_support(image, centers, samples), original)


def test_fast_blade_ray_skip_is_exact_for_empty_rectangles_between_atlas_tiles():
    from src.tracking.material_patch_tracker import _Image, _Bank, _sample, _unblocked_support
    image = texture()
    blade = np.zeros_like(image)
    blade[35:145, 101:105] = 1.
    source = _Image(image, np.ones_like(image), blade)
    points = np.array([[45., 60.], [90., 80.], [145., 105.]], np.float32)
    bank = _Bank(source, points, np.repeat(np.eye(2, dtype=np.float32)[None], 3, axis=0), 0, 65)
    rng = np.random.default_rng(348)
    centers = bank.atlas_points
    samples = centers[:, None]+rng.uniform(-31, 31, (3, 121, 2)).astype(np.float32)
    fractions = np.linspace(.125, .875, 7, dtype=np.float32)
    rays = centers[:, None, None]+fractions[None, :, None, None]*(samples-centers[:, None])[:, None]
    original = ~(_sample(bank.image.blade, rays.reshape(3, -1, 2)).reshape(3, 7, -1) > .25).any(1)
    np.testing.assert_array_equal(_unblocked_support(bank.image, centers, samples), original)


def test_forecast_in_opaque_blade_is_labelled_without_edge_projection_or_id_loss():
    image = texture()
    material = np.ones(image.shape, bool)
    blade = np.zeros(image.shape, bool)
    blade[:, 48:64] = True
    point = np.array([[46., 80.]], np.float32)
    tracker = MaterialPatchTracker(image, material, point, blade_mask=blade)
    tracker.velocity[:] = [4., 0.]
    result = tracker.step(np.full_like(image, .5), np.zeros_like(material), blade_mask=blade)
    assert not result['supported'].any()
    assert result['blade_occluded'].all()
    assert result['inferred'].all()
    np.testing.assert_array_equal(result['point_ids'], [0])
    # Position is a retained latent prediction, not a point invented on the edge.
    np.testing.assert_array_equal(result['positions'], [[50., 80.]])
    assert tracker.recent.age[0] == 0


def test_explicit_trusted_core_covers_a_semantic_blade_hole():
    image = texture()
    material = np.ones(image.shape, bool)
    point = np.array([[100., 80.]], np.float32)
    tracker = MaterialPatchTracker(image, material, point)
    core = np.zeros_like(material)
    core[:, 94:108] = True
    result = tracker.step(image, material, blade_mask=np.zeros_like(material), blade_core_mask=core)
    assert not result['supported'].any()
    assert result['blade_occluded'].all()
    np.testing.assert_array_equal(result['positions'], point)


def test_uncertain_blade_boundary_allows_supported_material_contact():
    image = texture()
    material = np.ones(image.shape, bool)
    blade = np.zeros_like(material)
    blade[:, 100:115] = True
    image[blade] = 1.
    point = np.array([[98.7, 80.]], np.float32)
    tracker = MaterialPatchTracker(image, material, point, blade_mask=blade)
    result = tracker.step(image, material, blade_mask=blade)
    assert result['supported'].all(), result
    assert result['blade_boundary_uncertain'].all()
    assert not result['blade_occluded'].any()
    np.testing.assert_allclose(result['positions'], point, atol=.05)


def test_temporal_blade_path_check_allows_tip_routes_and_gap_recovery():
    from src.tracking.material_patch_tracker import _crosses_opaque_core
    core = np.zeros((100, 100), bool)
    core[:60, 48:54] = True
    start = np.array([[42, 40], [42, 70], [42, 40], [42, 40]], np.float32)
    end = np.array([[60, 40], [60, 70], [60, 40], [43, 42]], np.float32)
    # First crosses the persistent shaft; second goes below the tip; third
    # reappears after an observation gap; fourth remains on the material side.
    blocked = _crosses_opaque_core(core, start, end, np.array([True, True, False, True]))
    np.testing.assert_array_equal(blocked, [True, False, False, False])


def test_blade_invalid_winner_does_not_discard_an_admissible_alternative(monkeypatch):
    import src.tracking.material_patch_tracker as module
    image = texture()
    material = np.ones(image.shape, bool)
    blade = np.zeros_like(material)
    blade[:, 95:106] = True
    tracker = MaterialPatchTracker(image, material, [[90., 80.]],
        config=MaterialPatchConfig(use_lk_prior=False), blade_mask=blade)
    calls = []

    def staged_fit(reference, target, p, q0, affine0, config, radius, iterations=None, target_bounds=None):
        count=len(p);number=len(calls);calls.append(number)
        # A high-score candidate appears on the tool; the larger support offers
        # a slightly weaker image match on material. Reverse returns correctly.
        positions=np.array(q0,np.float32,copy=True)
        if target_bounds is None:
            positions[:]=[100.,80.] if number==0 else [90.4,80.]
        score=np.full(count,.99 if number==0 else .95)
        return dict(positions=positions,relative_affine=np.repeat(np.eye(2,dtype=np.float32)[None],count,0),
            accepted=np.ones(count,bool),score=score,correlation=np.full(count,.99),
            residual=np.full(count,.01),raw_residual=np.full(count,.01),
            photometric_inlier_fraction=np.ones(count),valid_fraction=np.ones(count))

    monkeypatch.setattr(module,'_fit',staged_fit)
    result=tracker.step(image,material,blade_mask=blade)
    assert result['candidate_rejected_by_blade'].all()
    assert result['supported'].all()
    assert not result['blade_occluded'].any()
    np.testing.assert_allclose(result['positions'],[[90.4,80.]],atol=1e-5)


def test_dimensions_and_nonfinite_points_rejected():
    import pytest
    image = texture()
    mask = np.ones(image.shape, bool)
    with pytest.raises(ValueError, match='finite'):
        MaterialPatchTracker(image, mask, [[np.nan, 2.]])
    tracker = MaterialPatchTracker(image, mask, [[50., 50.]])
    with pytest.raises(ValueError, match='dimensions'):
        tracker.step(image[:50], mask[:50])


def test_masked_gray_excludes_bright_and_nonfinite_exterior_from_initial_blur():
    image = texture()
    material = np.zeros(image.shape, bool)
    material[30:130, 40:150] = True
    dark, bright, missing = [image.copy() for _ in range(3)]
    dark[~material], bright[~material], missing[~material] = 0., 1000., np.nan
    expected, support = masked_gray(dark, material)
    for altered in (bright, missing):
        actual, valid = masked_gray(altered, material)
        np.testing.assert_array_equal(valid, support)
        np.testing.assert_allclose(actual, expected, atol=0)
    assert not np.any(expected[~material])
    assert support[30, 80]  # retain measurable material up to the mask edge


def test_masked_gray_constant_material_has_no_dark_boundary_halo():
    image = np.zeros((40, 40), np.uint8)
    material = np.zeros(image.shape, bool)
    material[8:32, 8:32] = True
    image[material] = 128
    result, valid = masked_gray(image, material)
    np.testing.assert_allclose(result[valid], 128/255., atol=1.e-7)
    assert not valid[~material].any()
    empty, unsupported = masked_gray(image, np.zeros_like(material))
    assert not empty.any() and not unsupported.any()


def test_candidate_verification_known_translation_preserves_external_ids():
    image = texture()
    points = np.array([[65., 55.], [100., 65.], [80., 105.]], np.float32)
    offset = np.array([2.4, -.7])
    mask = np.ones(image.shape, bool)
    truth = points+offset
    result = verify_material_candidates(image, warped(image, np.eye(2), offset),
        points, truth+[.4, -.25], mask, mask, point_ids=[105, 4, 701])
    assert result['accepted'].all(), result
    assert not result['ambiguous'].any()
    assert np.linalg.norm(result['positions']-truth, axis=1).max() < .15
    assert result['cycle_error'].max() < .2
    np.testing.assert_array_equal(result['point_ids'], [105, 4, 701])


def test_candidate_verification_known_affine_gain_and_bias():
    image = texture()
    points = np.array([[65., 55.], [100., 60.], [80., 105.]], np.float32)
    affine = np.array([[1.08, .04], [-.025, .95]], np.float32)
    offset = np.array([-3., 2.])
    mask = np.ones(image.shape, bool)
    truth = points@affine.T+offset
    result = verify_material_candidates(image, warped(image, affine, offset, 1.18, -.04),
        points, truth+[.2, -.2], mask, mask)
    assert result['accepted'].all(), result
    assert np.linalg.norm(result['positions']-truth, axis=1).max() < .2
    assert np.max(abs(result['relative_affine']-affine)) < .04


def test_candidate_image_verification_rejects_blade_centers_without_relocating_ids():
    image = texture()
    points = np.array([[80., 80.], [125., 95.]], np.float32)
    mask = np.ones(image.shape, bool)
    blade = np.zeros_like(mask)
    blade[76:85, 76:85] = True
    result = verify_material_candidates(image, image, points, points, mask, mask,
                                        current_blade=blade, point_ids=[52, 93])
    np.testing.assert_array_equal(result['accepted'], [False, True])
    np.testing.assert_array_equal(result['positions'][0], points[0])
    np.testing.assert_array_equal(result['point_ids'], [52, 93])
    assert result['selected_proposal'][0] == -1


def test_boundary_candidate_matching_is_invariant_to_excluded_blade_brightness():
    image = texture()
    mask = np.ones(image.shape, bool)
    mask[:80] = False
    point = np.array([[80., 82.]], np.float32)
    target = warped(image, np.eye(2), [.3, .4])
    outputs = []
    for brightness in (0., 1000.):
        source, current = image.copy(), target.copy()
        source[~mask], current[~mask] = brightness, brightness
        outputs.append(verify_material_candidates(source, current, point, point+[.3, .4],
                                                  mask, mask))
    assert outputs[0]['accepted'].all()
    for key in ('positions', 'accepted', 'relative_affine', 'cycle_error', 'residual'):
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])


def test_periodic_texture_distinct_competitive_positions_remain_ambiguous():
    tile = texture((180, 32))
    image = np.tile(tile, (1, 7))
    mask = np.ones(image.shape, bool)
    point = np.array([[80., 80.]], np.float32)
    proposals = np.stack([point, point+[32., 0]])
    result = verify_material_candidates(image, image, point, proposals, mask, mask)
    assert result['ambiguous'].all(), result
    assert not result['accepted'].any()
    assert result['cycle_error'][0] < .01  # reverse consistency alone is insufficient


def test_duplicate_proposals_for_same_solution_are_not_ambiguity():
    image = texture()
    point = np.array([[80., 80.]], np.float32)
    mask = np.ones(image.shape, bool)
    result = verify_material_candidates(image, image, point,
        np.stack([point, point+[.3, -.2]]), mask, mask)
    assert result['accepted'].all()
    assert not result['ambiguous'].any()
    np.testing.assert_allclose(result['positions'], point, atol=.04)


def test_candidate_correction_bound_prevents_distant_initialization_reassignment():
    image = texture()
    point = np.array([[80., 80.]], np.float32)
    proposal = point+[2.5, 0]
    mask = np.ones(image.shape, bool)
    result = verify_material_candidates(image, image, point, proposal, mask, mask,
                                        maximum_correction_px=.2)
    assert not result['accepted'].any()
    np.testing.assert_array_equal(result['positions'], proposal)


def test_recent_reference_match_does_not_certify_inconsistent_initial_reference():
    initial = texture()
    recent = np.random.default_rng(1031).uniform(.05, .95, initial.shape).astype(np.float32)
    recent = cv2.GaussianBlur(recent, (3, 3), .7)
    current = warped(recent, np.eye(2), [1., .5])
    point = np.array([[80., 80.]], np.float32)
    proposal = point+[1., .5]
    mask = np.ones(initial.shape, bool)
    initial_result = verify_material_candidates(initial, current, point, proposal, mask, mask)
    recent_result = verify_material_candidates(recent, current, point, proposal, mask, mask)
    assert not initial_result['accepted'].any()
    assert recent_result['accepted'].all()
    np.testing.assert_array_equal(initial_result['point_ids'], recent_result['point_ids'])


def test_candidate_validation_rejects_duplicate_ids_and_keeps_missing_proposals_unobserved():
    import pytest
    image = texture()
    point = np.array([[65., 55.], [100., 65.]], np.float32)
    mask = np.ones(image.shape, bool)
    with pytest.raises(ValueError, match='unique'):
        verify_material_candidates(image, image, point, point, mask, mask, point_ids=[4, 4])
    proposal = point.copy()
    proposal[0] = np.nan
    result = verify_material_candidates(image, image, point, proposal, mask, mask,
                                        point_ids=[4, 105])
    np.testing.assert_array_equal(result['accepted'], [False, True])
    assert np.isnan(result['positions'][0]).all()
    np.testing.assert_array_equal(result['point_ids'], [4, 105])


def test_one_sided_boundary_centers_use_material_pixels_without_projection():
    image = texture()
    mask = np.zeros(image.shape, bool)
    mask[80:] = True
    current_mask = np.zeros_like(mask)
    current_mask[82:] = True
    points = np.array([[65., 79.5], [100., 80.], [135., 80.5]], np.float32)
    truth = points+[1.4, 2.]
    target = warped(image, np.eye(2), [1.4, 2.])
    legacy = verify_material_candidates(image, target, points, truth+[.1, .1], mask, current_mask)
    result = verify_material_candidates(image, target, points, truth+[.1, .1], mask, current_mask,
                                        boundary_mode='one-sided')
    assert not legacy['accepted'].any()  # the old eroded-center admission excludes all three
    assert result['accepted'].all()
    assert np.linalg.norm(result['positions']-truth, axis=1).max() < .3
    assert result['positions'][0, 1] < 82.  # not projected onto the semantic surface
    assert result['center_extrapolated'][0]
    assert result['diagnostics']['attempted'].all()
    assert not result['diagnostics']['failure_flags'].any()


def test_one_sided_compressed_curved_surface_has_known_geometric_error_bound():
    image = texture()
    yy, xx = np.indices(image.shape, dtype=np.float32)
    curvature, compression = .0005, .7
    mx = xx-2
    my = 80+(yy-80-curvature*(mx-90)**2)/compression
    target = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
    # Deliberately conservative semantic masks, without donating exterior pixels.
    mask = yy >= 81
    current_mask = yy >= (80+curvature*(xx-2-90)**2+.5)
    points = np.array([[70., 80.], [90., 80.], [110., 80.]], np.float32)
    truth = points.copy()
    truth[:, 0] += 2
    truth[:, 1] += curvature*(points[:, 0]-90)**2
    affine = np.tile(np.eye(2, dtype=np.float32), (3, 1, 1))
    affine[:, 1, 0] = 2*curvature*(points[:, 0]-90)
    affine[:, 1, 1] = compression
    result = verify_material_candidates(image, target, points, truth+[.2, .1], mask, current_mask,
        initial_affine=affine, boundary_mode='one-sided')
    assert result['accepted'].all()
    assert np.linalg.norm(result['positions']-truth, axis=1).max() < .35
    assert np.max(abs(result['relative_affine'][:, 1, 1]-compression)) < .06


def test_one_sided_matching_does_not_read_unknown_or_bright_exterior():
    image = texture()
    mask = np.zeros(image.shape, bool)
    mask[81:] = True
    point = np.array([[80., 80.]], np.float32)
    target = warped(image, np.eye(2), [.3, .4])
    results = []
    for exterior in (0., 1000., np.nan):
        source, current = image.copy(), target.copy()
        source[~mask] = current[~mask] = exterior
        results.append(verify_material_candidates(source, current, point, point+[.3, .4],
            mask, mask, boundary_mode='one-sided'))
    assert results[0]['accepted'].all()
    for result in results[1:]:
        for key in ('positions', 'accepted', 'relative_affine', 'cycle_error', 'residual'):
            np.testing.assert_array_equal(result[key], results[0][key])


def test_one_sided_blade_boundary_permission_requires_explicit_core_and_never_enters_core():
    image = texture()
    mask = np.ones(image.shape, bool)
    blade = np.zeros_like(mask)
    blade[:, 100:110] = True
    core = np.zeros_like(mask)
    core[:, 102:108] = True
    points = np.array([[99.6, 80.], [103., 110.]], np.float32)
    strict = verify_material_candidates(image, image, points, points, mask, mask,
        reference_blade=blade, current_blade=blade, boundary_mode='one-sided')
    relaxed = verify_material_candidates(image, image, points, points, mask, mask,
        reference_blade=blade, current_blade=blade, reference_blade_core=core,
        current_blade_core=core, boundary_mode='one-sided')
    assert not strict['accepted'].any()
    np.testing.assert_array_equal(relaxed['accepted'], [True, False])
    np.testing.assert_allclose(relaxed['positions'], points, atol=.02)
    assert not relaxed['diagnostics']['attempted'][:, :, 1].any()


def test_one_sided_ray_visibility_does_not_skip_a_thin_opaque_core():
    from src.tracking.material_patch_tracker import _Image, _unblocked_support
    gray = np.ones((100, 100), np.float32)
    core = np.zeros_like(gray)
    core[:, 41] = 1
    image = _Image(gray, gray, core)
    center = np.array([[40., 50.]], np.float32)
    samples = np.array([[[60., 50.], [20., 50.]]], np.float32)
    image.strict_visibility = True
    np.testing.assert_array_equal(_unblocked_support(image, center, samples), [[False, True]])


def test_one_sided_unknown_texture_is_not_promoted_by_relaxed_center_admission():
    image = texture()
    rng = np.random.default_rng(984)
    target = cv2.GaussianBlur(rng.uniform(.02, .98, image.shape).astype(np.float32), (3, 3), .7)
    mask = np.zeros(image.shape, bool)
    mask[80:] = True
    points = np.column_stack((np.linspace(35, 185, 24), np.full(24, 79.5))).astype(np.float32)
    result = verify_material_candidates(image, target, points, points, mask, mask,
                                        boundary_mode='one-sided')
    assert not result['accepted'].any()  # zero false accepts in this declared 24-point negative case
    diag = result['diagnostics']
    assert diag['attempted'].all()
    assert np.isfinite(diag['correlation']).all()
    assert (diag['failure_flags'] & 64).any()


def test_failed_fit_diagnostics_distinguish_geometry_texture_and_reverse_checks():
    image = np.full((100, 100), .5, np.float32)
    mask = np.ones(image.shape, bool)
    points = np.array([[40., 50.], [60., 50.]], np.float32)
    proposals = points.copy()
    proposals[1] = [-4., 50.]
    result = verify_material_candidates(image, image, points, proposals, mask, mask,
                                        boundary_mode='one-sided')
    diag = result['diagnostics']
    assert not result['accepted'].any()
    assert diag['attempted'][:, :, 0].all()
    assert not diag['attempted'][:, :, 1].any()
    assert (diag['failure_flags'][:, :, 0] & 32).all()  # not enough texture
    assert (diag['failure_flags'][:, :, 1] & 4).all()  # geometric admission
    assert np.isfinite(diag['correlation'][:, :, 0]).all()
    assert np.isnan(diag['correlation'][:, :, 1]).all()
    assert np.isnan(diag['reverse_correlation']).all()  # forward fit never qualified


def test_one_sided_boundary_band_cannot_be_used_as_arbitrary_recovery_radius():
    import pytest
    image = texture()
    mask = np.zeros(image.shape, bool)
    mask[80:] = True
    point = np.array([[80., 77.5]], np.float32)
    result = verify_material_candidates(image, image, point, point, mask, mask,
                                        boundary_mode='one-sided', center_tolerance_px=2.)
    assert not result['accepted'].any()
    assert not result['diagnostics']['attempted'].any()
    np.testing.assert_array_equal(result['positions'], point)
    for tolerance in (-1., 2.1, np.nan):
        with pytest.raises(ValueError, match='between 0 and 2'):
            verify_material_candidates(image, image, point, point, mask, mask,
                boundary_mode='one-sided', center_tolerance_px=tolerance)
    with pytest.raises(ValueError, match='masked pixels'):
        verify_material_candidates(image, image, point, point, mask, mask,
            boundary_mode='one-sided', boundary_aware=False)


def test_verifier_reverse_uses_per_id_atlas_bounds_with_different_image_sizes():
    image = texture()
    reference = image[40:120, 40:140]
    point = np.array([[40., 40.]], np.float32)
    proposal = point+[40., 40.]
    kwargs = dict(reference_frame=reference, current_frame=image, reference_points=point,
        proposals=proposal, reference_mask=np.ones(reference.shape, bool),
        current_mask=np.ones(image.shape, bool), boundary_mode='one-sided')
    normal = verify_material_candidates(**kwargs, reference_bounds=[[2, 2, 97, 77]])
    assert normal['accepted'].all()
    np.testing.assert_allclose(normal['positions'], proposal, atol=.02)
    narrow = verify_material_candidates(**kwargs, reference_bounds=[[38, 38, 42, 42]])
    assert not narrow['accepted'].any()
    assert (narrow['diagnostics']['failure_flags'] & 4096).all()
    assert (narrow['diagnostics']['reverse_failure_flags'] & 16).all()


def test_prepared_verification_snapshots_are_bitwise_equivalent_and_skip_reprocessing(monkeypatch):
    import src.tracking.material_patch_tracker as module
    image = texture()
    target = warped(image, np.eye(2), [.7, .3])
    mask = np.zeros(image.shape, bool)
    mask[80:] = True
    point = np.array([[70., 79.5], [100., 82.], [130., 90.]], np.float32)
    options = dict(boundary_mode='one-sided')
    expected = verify_material_candidates(image, target, point, point+[.7, .3], mask, mask, **options)
    source = module.prepare_verification_image(image, mask, **options)
    current = module.prepare_verification_image(target, mask, **options)
    monkeypatch.setattr(module, 'prepare_verification_image', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('Cached verification must not repeat image processing')))
    actual = verify_material_candidates(image, target, point, point+[.7, .3], mask, mask,
        prepared_reference=source, prepared_current=current, **options)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[key], value)
        elif key == 'diagnostics':
            for name in value:
                np.testing.assert_array_equal(actual[key][name], value[name])
        else:
            assert actual[key] == value


def test_prepared_snapshot_owns_readonly_pixels_and_rejects_different_modes():
    import pytest
    from src.tracking.material_patch_tracker import prepare_verification_image
    image = texture()
    mask = np.ones(image.shape, bool)
    blade = np.zeros_like(mask)
    blade[:30] = True
    snapshot = prepare_verification_image(image, mask, blade=blade, boundary_mode='one-sided')
    gray, material, barrier = snapshot.image.gray.copy(), snapshot.geometry[0].copy(), snapshot.geometry[1].copy()
    image[:], mask[:], blade[:] = 0., False, True
    np.testing.assert_array_equal(snapshot.image.gray, gray)
    np.testing.assert_array_equal(snapshot.geometry[0], material)
    np.testing.assert_array_equal(snapshot.geometry[1], barrier)
    for array in (snapshot.image.gray, snapshot.image.gx, *snapshot.geometry):
        with pytest.raises(ValueError, match='read-only'):
            array.flat[0] = 0
    points = np.array([[80., 80.]])
    with pytest.raises(ValueError, match='snapshot'):
        verify_material_candidates(image, image, points, points, mask, mask,
            prepared_reference=snapshot, prepared_current=snapshot)  # wrong mode
