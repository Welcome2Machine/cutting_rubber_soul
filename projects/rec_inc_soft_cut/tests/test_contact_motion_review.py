"""Display contracts that prevent misleading material/contact interpretations."""
import cv2
import numpy as np
import pytest

from src.analysis.contact_motion_review import (
    _arrow_vector, _marker_geometry, _mask_material_ink, _phase_runs, _sample_mask,
    render_contact_motion_review,
)


def test_current_marker_uses_registered_tip_not_distant_inverse_material():
    address = dict(state=np.array([1, 0]), reference_marker_xy=np.array([[91., 82.], [90., 81.]]),
                   raw_inverse_material_xy=np.array([[73., 42.], [75., 44.]]),
                   current_marker_xy=np.array([[500., 500.], [600., 600.]]))
    current, reference, raw, state = _marker_geometry([12., 25.], address, 0)
    np.testing.assert_array_equal(current, [12., 25.])
    np.testing.assert_array_equal(reference, [91., 82.])
    np.testing.assert_array_equal(raw, [73., 42.])
    assert state == 1
    current, reference, raw, state = _marker_geometry([13., 26.], address, 1)
    np.testing.assert_array_equal(current, [13., 26.])
    assert np.isnan(reference).all()
    np.testing.assert_array_equal(raw, [75., 44.])


def test_material_ink_is_occluded_including_strokes_but_coordinates_are_retained():
    blade = np.zeros((40, 60), bool)
    blade[0:31, 25:36] = True
    points = np.array([[30., 20.], [20., 20.], [np.nan, 10.]])
    before = points.copy()
    ink = np.full((40, 60, 3), 210, np.uint8)
    result = _mask_material_ink(ink, blade, [0., 0.], (60, 40), (60, 40), [0, 0])
    assert not result[blade].any()
    assert (result[~blade] == 210).all()
    np.testing.assert_array_equal(_sample_mask(blade, points), [True, False, False])
    np.testing.assert_array_equal(points, before)


def test_depth_curves_do_not_bridge_unknowns_branches_or_reversal():
    frames = np.arange(10)
    depth = np.array([-.1, 0., .4, .8, 1., 1.2, 1., .8, .6, .4])
    values = np.array([0., 0., 1., np.nan, 2., 3., 4., 5., 6., 7.])
    branch = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    pieces = [(rows.tolist(), phase) for rows, phase in _phase_runs(frames, depth, values, 1, 5, branch)]
    assert pieces == [([1, 2], 0), ([4, 5], 0), ([6], 1), ([7, 8, 9], 1)]


def test_near_zero_residuals_do_not_create_emphatic_arrows_or_overflow_masks():
    assert _arrow_vector([1e-5, -1e-5], 10.) is None
    assert _arrow_vector([np.nan, 2.], 10.) is None
    vector = _arrow_vector([30., 40.], 10.)
    assert 19. < np.linalg.norm(vector) <= 20.
    np.testing.assert_allclose(vector/np.linalg.norm(vector), [.6, .8])
    with np.errstate(all='raise'):
        result = _sample_mask(np.ones((5, 5), bool), [[1e30, 1e30], [2., 3.]])
    np.testing.assert_array_equal(result, [False, True])


def _fixture(tmp_path):
    source = tmp_path/'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 12., (128, 96))
    if not writer.isOpened():
        pytest.skip('OpenCV MJPG encoder unavailable')
    for t in range(7):
        frame = np.full((96, 128, 3), 70, np.uint8)
        frame[:, 57:69] = 220
        frame[70:] = 95+t
        writer.write(frame)
    writer.release()
    points = np.array([[48., 32.], [40., 42.], [76., 32.], [84., 42.], [62., 40.]])
    positions = np.repeat(points[None], 7, axis=0)
    support = np.ones((7, 5), bool)
    support[:, 4] = False
    groups = np.tile([0, 1, 2, 3, -1], (7, 1))
    motion = dict(group_index=groups, point_ids=np.arange(5),
        group_y_median_um_s=np.tile([10., -20., 30., -40.], (7, 1)),
        group_y_count=np.ones((7, 4), int)*4,
        relative_velocity_xy_px_s=np.tile([0., 1.], (7, 5, 1)),
        relative_velocity_valid=support.copy(), disjoint_relaxing_patch_count=np.array([0, 0, 0, 1, 0, 0, 0]))
    address = dict(state=np.array([0, 2, 1, 1, 0, 0, 0]), s_um=np.array([np.nan, 0, 30, 60, np.nan, np.nan, np.nan]),
        reference_marker_xy=np.tile([86., 61.], (7, 1)),
        raw_inverse_material_xy=np.tile([90., 65.], (7, 1)),
        raw_surface_normal_offset_px=np.ones(7)*10.)
    surface = dict(boundary_xy=np.array([[20., 20.], [60., 45.], [100., 70.]]),
                   surface_s_px=np.array([0., 50., 100.]), contact_s_px=np.array(50.))
    blade = np.zeros((96, 128), bool)
    blade[:68, 57:69] = True
    kwargs = dict(frame_indices=np.arange(7), depth_mm=np.array([-.1, 0., 1., 2., 3., 1., -.1]),
        tip_xy=np.tile([63., 67.], (7, 1)), point_ids=np.arange(5), motion=motion, address=address,
        surface_reference=surface, blade_mask_reader=lambda _: blade,
        contact_frame=1, reversal_frame=4, um_per_px=2., codec='MJPG')
    return source, positions, support, kwargs


def test_full_saved_sequence_render_and_playback_fps_does_not_change_measurements(tmp_path):
    source, positions, support, kwargs = _fixture(tmp_path)
    before = positions.copy()
    first = render_contact_motion_review(source, positions, support, tmp_path/'review12.avi',
        overview_path=tmp_path/'overview.png', fps=12., **kwargs)
    second = render_contact_motion_review(source, positions, support, tmp_path/'review24.avi', fps=24., **kwargs)
    assert first['rendered_frames'] == second['rendered_frames'] == 7
    assert first['dimensions'] == [1600, 1000]
    assert first['per_frame_blade_occluded_ids'] == [1]*7
    assert first['motion_display_scale_um_s'] == second['motion_display_scale_um_s']
    assert first['camera_origin_xy'] == second['camera_origin_xy']
    assert first['fracture_classified'] is False
    assert len(first['overview_snapshots']) == 6
    np.testing.assert_array_equal(positions, before)
    image = cv2.imread(str(tmp_path/'overview_2mm.png'))
    assert image.shape == (1000, 1600, 3)
    caps = [cv2.VideoCapture(str(tmp_path/f'review{fps}.avi')) for fps in (12, 24)]
    try:
        for _ in range(7):
            decoded = [cap.read() for cap in caps]
            assert all(item[0] for item in decoded)
            np.testing.assert_array_equal(decoded[0][1], decoded[1][1])
    finally:
        for cap in caps:
            cap.release()


def test_renderer_rejects_mismatched_material_id_order(tmp_path):
    source, positions, support, kwargs = _fixture(tmp_path)
    kwargs['motion']['point_ids'] = np.arange(5)[::-1]
    with pytest.raises(ValueError, match='point IDs'):
        render_contact_motion_review(source, positions, support, tmp_path/'invalid.avi', **kwargs)
