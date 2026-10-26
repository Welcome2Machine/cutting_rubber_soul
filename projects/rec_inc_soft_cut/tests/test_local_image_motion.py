import cv2
import numpy as np

from src.analysis.local_image_motion import LocalImageMotionConfig, estimate_local_image_motion, _path_crosses


def texture(seed=10, shape=(220, 260)):
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 1, shape).astype(np.float32)
    raw = cv2.GaussianBlur(raw, (0, 0), 1.)
    return np.uint8(np.clip(125+65*raw/raw.std(), 0, 255))


def test_upward_translation_is_recovered_under_gain_and_smooth_shadow():
    a = texture()
    transform = np.float32([[1, 0, 6.25], [0, 1, -4.5]])
    b = cv2.warpAffine(a, transform, a.shape[::-1]).astype(np.float32)
    illumination = np.linspace(.40, .95, a.shape[1])[None]
    b = np.uint8(np.clip(b*illumination+18, 0, 255))
    labels = np.ones(a.shape, np.uint8)
    out = estimate_local_image_motion(a, b, labels, labels, [130, 120], [130, 122], dt_s=.03,
        config=LocalImageMotionConfig(max_points=100, radius_px=65, affine_rescue=False))
    assert out['valid'].sum() > 75
    np.testing.assert_allclose(np.median(out['displacement_xy'][out['valid']], axis=0), [6.25, -4.5], atol=.15)
    np.testing.assert_allclose(np.median(out['relative_displacement_xy'][out['valid']], axis=0), [6.25, -6.5], atol=.15)
    assert np.median(out['fb_error_px'][out['valid']]) < .2


def test_affine_deformation_recovers_local_motion_without_rigid_neighbor_constraint():
    a = texture(12)
    transform = np.array([[1.08, .09, -14], [-.06, 1.09, -4]], np.float32)
    b = cv2.warpAffine(a, transform, a.shape[::-1])
    labels = np.ones(a.shape, np.uint8)
    out = estimate_local_image_motion(a, b, labels, labels, [130, 120], [130, 120],
        config=LocalImageMotionConfig(max_points=80, radius_px=60))
    observed = out['source_xy'][out['valid']]
    expected = observed@transform[:, :2].T + transform[:, 2]
    error = np.linalg.norm(expected-out['current_xy'][out['valid']], axis=1)
    assert len(error) > 50
    assert np.median(error) < .25
    assert np.quantile(error, .9) < .65


def test_tool_pixels_are_not_seeds_and_cross_tool_matches_are_rejected():
    a = texture(13)
    labels0 = np.ones(a.shape, np.uint8)
    labels1 = labels0.copy()
    labels0[10:170, 120:131] = 2
    labels1[10:172, 120:131] = 2
    b = cv2.warpAffine(a, np.float32([[1, 0, 17], [0, 1, 0]]), a.shape[::-1])
    a[labels0 == 2] = 245
    b[labels1 == 2] = 245
    seeds = np.array([[125, 70], [113, 90], [115, 120], [160, 110], [170, 135]], np.float32)
    out = estimate_local_image_motion(a, b, labels0, labels1, [125, 170], [125, 172], seed_xy=seeds,
        config=LocalImageMotionConfig(radius_px=120, max_displacement_px=25, affine_rescue=False))
    assert not out['valid'][0]
    assert not out['valid'][1:3].any()
    # Large unsupported jumps may be rejected elsewhere too; do not turn a
    # missing short-time measurement into an assertion of stationary material.
    core = cv2.erode((labels0 == 2).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    np.testing.assert_array_equal(_path_crosses(core, seeds[1:3], seeds[1:3]+[17, 0]), [True, True])
    auto = estimate_local_image_motion(a, b, labels0, labels1, [125, 170], [125, 172],
        config=LocalImageMotionConfig(max_points=50, affine_rescue=False))
    xy = np.rint(auto['source_xy']).astype(int)
    assert np.all(labels0[xy[:, 1], xy[:, 0]] == 1)


def test_textureless_images_do_not_produce_confident_stationary_motion():
    a = np.full((120, 140), 100, np.uint8)
    labels = np.ones(a.shape, np.uint8)
    out = estimate_local_image_motion(a, a, labels, labels, [70, 70], [70, 70])
    assert len(out['source_xy']) == 0
    explicit = estimate_local_image_motion(a, a, labels, labels, [70, 70], [70, 70],
                                           seed_xy=np.array([[60, 60], [80, 80]]))
    assert not explicit['valid'].any()
    assert not explicit['confidence'].any()


def test_interface_seed_mask_tracks_upper_regions_outside_tip_circle():
    a = texture(44, (320, 260))
    b = cv2.warpAffine(a, np.float32([[1, 0, 0], [0, 1, 2]]), a.shape[::-1])
    labels = np.ones(a.shape, np.uint8)
    labels[:281, 120:141] = 2
    a[labels == 2] = b[labels == 2] = 240
    region = np.zeros(a.shape, bool)
    region[25:260, 82:118] = True
    out = estimate_local_image_motion(a, b, labels, labels, [130, 280], [130, 280],
        seed_region_mask=region, config=LocalImageMotionConfig(radius_px=35,
            max_points=120, vertical_bin_px=48, affine_rescue=False))
    good = out['valid']
    upper = good & (out['source_xy'][:, 1] < 100)
    assert upper.sum() >= 15
    xy = np.rint(out['source_xy']).astype(int)
    assert region[xy[:, 1], xy[:, 0]].all()
    np.testing.assert_allclose(np.median(out['displacement_xy'][upper], axis=0), [0, 2], atol=.2)
    assert len(np.unique(np.floor((280-out['source_xy'][good, 1])/48))) >= 5


def test_opposite_sides_can_have_opposite_vertical_motion():
    a = texture(15)
    b = a.copy()
    left = cv2.warpAffine(a, np.float32([[1, 0, 0], [0, 1, -3]]), a.shape[::-1])
    right = cv2.warpAffine(a, np.float32([[1, 0, 0], [0, 1, 4]]), a.shape[::-1])
    b[:, :120], b[:, 140:] = left[:, :120], right[:, 140:]
    labels = np.ones(a.shape, np.uint8)
    labels[:180, 120:141] = 2
    a[labels == 2] = b[labels == 2] = 240
    seeds = np.array([[85, 90], [90, 120], [95, 140], [165, 90], [170, 120], [175, 140]], np.float32)
    out = estimate_local_image_motion(a, b, labels, labels, [130, 180], [130, 180], seed_xy=seeds,
                                       config=LocalImageMotionConfig(affine_rescue=False))
    assert out['valid'].all()
    np.testing.assert_allclose(out['displacement_xy'][:3, 1], -3, atol=.2)
    np.testing.assert_allclose(out['displacement_xy'][3:, 1], 4, atol=.2)


def test_moving_blade_does_not_impose_its_motion_on_stationary_adjacent_material():
    a = texture(22)
    b = a.copy()
    labels0 = np.ones(a.shape, np.uint8)
    labels1 = labels0.copy()
    labels0[:155, 120:141] = 2
    labels1[:161, 120:141] = 2
    blade_texture = texture(30)
    shifted_blade = cv2.warpAffine(blade_texture, np.float32([[1, 0, 0], [0, 1, 6]]), a.shape[::-1])
    a[labels0 == 2] = blade_texture[labels0 == 2]
    b[labels1 == 2] = shifted_blade[labels1 == 2]
    seeds = np.array([[114, 80], [115, 110], [113, 135], [147, 90], [148, 120]], np.float32)
    out = estimate_local_image_motion(a, b, labels0, labels1, [130, 155], [130, 161], seed_xy=seeds)
    assert out['valid'].sum() >= 3
    assert np.max(np.linalg.norm(out['displacement_xy'][out['valid']], axis=1)) < .4
    np.testing.assert_allclose(np.median(out['relative_displacement_xy'][out['valid'], 1]), -6, atol=.2)


def test_replaced_texture_is_unknown_while_surviving_neighbors_remain_measurable():
    a = texture(23)
    b = cv2.warpAffine(a, np.float32([[1, 0, 2], [0, 1, -2]]), a.shape[::-1])
    b[75:115, 85:125] = texture(24)[75:115, 85:125]
    labels = np.ones(a.shape, np.uint8)
    seeds = np.array([[100, 95], [155, 90], [160, 120], [145, 145]], np.float32)
    out = estimate_local_image_motion(a, b, labels, labels, [130, 120], [130, 120], seed_xy=seeds)
    assert not out['valid'][0]
    assert out['valid'][1:].all()
    np.testing.assert_allclose(out['displacement_xy'][1:], np.tile([2, -2], (3, 1)), atol=.2)
