import json

import cv2
import numpy as np
import pytest

from src.analysis.material_tracking_review import (
    MaterialReviewConfig, _clip_segments, render_material_review, summarize_material_tracks,
)


def test_light_summary_retains_forecasts_but_excludes_them_from_supported_motion():
    xy = np.array([[[0., 0.]], [[1., 0.]], [[50., 0.]], [[51., 0.]]])
    support = np.array([[1], [1], [0], [1]], bool)
    summary = summarize_material_tracks(xy, support, depth_mm=[0, .2, .4, .6],
                                        times_s=[0, .1, .2, .3], point_ids=[17])
    assert summary['loading']['predicted_finite_fraction'] == .25
    assert summary['loading_motion']['all_finite_jump_count'] == 1
    assert summary['loading_motion']['supported_only_jump_count'] == 0
    assert summary['loading_motion']['supported_edge_count'] == 1
    assert summary['loading_motion']['supported_only_speed_px_s']['median'] == pytest.approx(10)
    assert summary['point_ids'] == [17]
    json.dumps(summary, allow_nan=False)


def test_support_gap_survives_missing_depth_and_missing_positions():
    xy = np.zeros((40, 1, 2))
    support = np.ones((40, 1), bool)
    support[3:35] = False
    xy[10] = np.nan
    depth = np.r_[[-.3, -.2, -.1], np.linspace(0, 3, 37)]
    depth[20] = np.nan
    summary = summarize_material_tracks(xy, support, depth_mm=depth)
    assert summary['support_gaps']['maximum_samples'] == 32
    assert summary['support_gaps']['points_gt30'] == 1
    assert summary['loading']['unlocated_fraction'] == pytest.approx(1/37)
    json.dumps(summary, allow_nan=False)


def test_batched_trails_clip_crossing_lines_without_nan_or_overflow():
    segments = np.array([[[-20., 15.], [120., 15.]], [[-10., -10.], [-4., -4.]],
                         [[np.nan, 1.], [10., 10.]], [[50., -1000.], [50., 1000.]]])
    clipped, ids = _clip_segments(segments, 100, 30)
    np.testing.assert_array_equal(ids, [0, 3])
    np.testing.assert_array_equal(clipped, [[[0, 15], [99, 15]], [[50, 0], [50, 29]]])


def test_single_review_preserves_timeline_and_black_point_panel(tmp_path):
    video = tmp_path / 'shadow.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 12., (96, 80))
    if not writer.isOpened():
        pytest.skip('MP4 encoder unavailable')
    for value in range(5):
        writer.write(np.full((80, 96, 3), 90+value*4, np.uint8))
    writer.release()
    tracks = np.tile(np.array([[30., 30.], [55., 45.]]), (3, 1, 1))
    tracks[:, 0, 0] += np.arange(3)
    before = tracks.copy()
    support = np.array([[1, 1], [1, 0], [1, 0]], bool)
    config = MaterialReviewConfig(crop_size=(64, 48), panel_size=(160, 120), trail_length=2)
    output, overview = tmp_path / 'review.mp4', tmp_path / 'overview.png'
    rendered = render_material_review(video, tracks, output, frame_indices=[0, 2, 4],
        depth_mm=[0, 1.5, 3.], tip_xy=np.tile([48., 40.], (3, 1)), supported=support,
        point_ids=[20, 27], config=config, overview_path=overview)
    np.testing.assert_array_equal(tracks, before)
    assert rendered['rendered_frames'] == 3
    assert rendered['dimensions'] == [320, 272]
    assert overview.is_file()
    cap = cv2.VideoCapture(str(output))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
    ok, frame = cap.read()
    cap.release()
    assert ok
    left, right = frame[84:204, :160], frame[84:204, 160:]
    assert np.median(left) > 75 and np.median(right) < 5
    assert np.count_nonzero(np.max(right, axis=2) > 50) > 20
    with pytest.raises(ValueError):
        render_material_review(video, tracks, video, frame_indices=[0, 2, 4], depth_mm=[0, 1.5, 3.],
            tip_xy=np.tile([48., 40.], (3, 1)), supported=support, point_ids=[20, 27], config=config)


def test_full_cycle_summary_separates_loading_from_same_depth_on_withdrawal():
    frames = np.arange(10, 100, 10)
    depth = np.array([-.5, 0., 1., np.nan, 3., 2., 1., -.2, np.nan])
    xy = np.zeros((9, 1, 2))
    xy[:, 0, 0] = [0, 1, 2, 3, 4, 100, 101, 102, 103]
    support = np.ones((9, 1), bool)
    support[3:8] = False
    report = summarize_material_tracks(xy, support, frame_indices=frames, depth_mm=depth,
        contact_frame=20, reversal_frame=50, times_s=frames/100.)
    assert report['phase']['source'] == 'source_landmarks'
    assert report['loading']['frames'] == 4
    assert report['withdrawal']['frames'] == 4
    assert report['per_frame']['phase'] == ['precontact']+['loading']*4+['withdrawal']*4
    assert report['support_gaps']['maximum_samples'] == 2
    assert report['phase_support_gaps']['withdrawal']['maximum_samples'] == 3
    assert report['full_video_support_gaps']['maximum_samples'] == 5
    loading_bin = next(row for row in report['depth_bins'] if row['lower_mm'] == 1.)
    withdrawal_bin = next(row for row in report['phase_depth_bins']['withdrawal'] if row['lower_mm'] == 1.)
    assert loading_bin['frames'] == withdrawal_bin['frames'] == 1
    assert loading_bin['supported_fraction'] == 1.
    assert withdrawal_bin['supported_fraction'] == 0.
    # The large step crosses the declared reversal boundary; all-video keeps it.
    assert report['loading_motion']['all_finite_jump_count'] == 0
    assert report['phase_motion']['withdrawal']['all_finite_jump_count'] == 0
    assert report['all_video_motion']['all_finite_jump_count'] == 1
    json.dumps(report, allow_nan=False)


def test_explicit_phase_masks_override_depth_and_leave_unknown_frames_unclassified():
    xy = np.zeros((5, 1, 2))
    support = np.ones((5, 1), bool)
    masks = dict(loading=np.array([1, 1, 0, 0, 0], bool),
                 withdrawal=np.array([0, 0, 0, 1, 1], bool))
    report = summarize_material_tracks(xy, support, depth_mm=np.arange(5.), phase_masks=masks)
    assert report['phase']['source'] == 'explicit_phase_masks'
    assert report['per_frame']['phase'] == ['loading', 'loading', 'unclassified', 'withdrawal', 'withdrawal']
    unknown = summarize_material_tracks(xy, support)
    assert unknown['loading']['frames'] == 0
    assert unknown['phases']['unclassified']['frames'] == 5
    with pytest.raises(ValueError, match='overlap'):
        summarize_material_tracks(xy, support, phase_masks=dict(loading=np.ones(5, bool), **{'withdrawal': masks['withdrawal']}))


def test_full_view_retains_edge_ids_and_tip_camera_follows_complete_retreat(tmp_path):
    video = tmp_path / 'full_cycle.mp4'
    length = 43
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 12., (128, 160))
    if not writer.isOpened():
        pytest.skip('MP4 encoder unavailable')
    for _ in range(length):
        writer.write(np.full((160, 128, 3), 110, np.uint8))
    writer.release()
    tracks = np.tile([[2., 2.], [125., 2.], [2., 157.], [125., 157.], [64., 130.]], (length, 1, 1))
    support = np.ones((length, 5), bool)
    support[22:, 4] = False
    tips = np.column_stack([np.full(length, 64.),
        np.r_[np.linspace(45, 130, 22), np.linspace(126, 45, 20), np.nan]])
    depth = np.r_[[-.3, -.2, -.1], np.linspace(0, 3, 19), np.linspace(2.8, -.4, 20), np.nan]
    arguments = dict(frame_indices=np.arange(length), depth_mm=depth, tip_xy=tips,
        supported=support, point_ids=np.arange(5), contact_frame=3, reversal_frame=21)
    full = render_material_review(video, tracks, tmp_path/'full.mp4', **arguments,
        config=MaterialReviewConfig(view_mode='full', crop_size=(64, 48), panel_size=(160, 120),
                                    point_radius=1, trail_length=0),
        overview_path=tmp_path/'full_overview.png')
    assert full['camera_mode'] == 'fixed_full_source_image'
    assert full['crop_size'] == [128, 160]
    assert full['image_viewport'] == dict(size=[96, 120], offset=[32, 0])
    assert full['saved_point_count'] == 5
    assert full['per_frame_points_on_view'] == [5]*length
    assert full['final_depth_mm'] is None
    assert full['axis_range_mm'][0] < 0
    final = full['overview_snapshots'][-1]
    assert final['source_frame'] == length-1 and final['depth_mm'] is None
    assert final['phase'] == 'withdrawal' and 'final' in final['roles']
    # Full view keeps corner IDs, aspect ratio and the final unsupported ID.
    cap = cv2.VideoCapture(str(tmp_path/'full.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, length-1)
    ok, image = cap.read()
    cap.release()
    assert ok
    right = image[84:204, 160:]
    for x, y in [(33, 1), (126, 1), (33, 118), (126, 118), (80, 97)]:
        assert right[max(0, y-3):min(120, y+4), x-3:x+4].max() > 45
    tip = render_material_review(video, tracks, tmp_path/'tip.mp4', **arguments,
        config=MaterialReviewConfig(crop_size=(64, 48), panel_size=(160, 120), trail_length=0))
    peak = next(row for row in tip['overview_snapshots'] if 'peak_or_loading_end' in row['roles'])
    last = tip['overview_snapshots'][-1]
    assert peak['camera_origin_xy'][1] > last['camera_origin_xy'][1]+30
    assert peak['camera_origin_xy'][0] == last['camera_origin_xy'][0]
    assert tip['rendered_frames'] == full['rendered_frames'] == length
    assert tip['final_depth_mm'] is None
    json.dumps(full, allow_nan=False)
    json.dumps(tip, allow_nan=False)


def test_blade_occlusion_hides_filled_hollow_points_and_crossing_trails_without_mutating_ids(tmp_path):
    source = tmp_path/'shadow_mask_test.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 12., (96, 80))
    if not writer.isOpened():
        pytest.skip('MP4 encoder unavailable')
    for _ in range(5):
        writer.write(np.full((80, 96, 3), 100, np.uint8))
    writer.release()
    p = np.tile([[48., 35.], [48., 48.], [20., 25.], [80., 60.], [1e30, 1e30]], (3, 1, 1))
    p[1:, 2, 0] = 75.  # This trail crosses the opaque blade.
    supported = np.tile([True, False, True, True, False], (3, 1))
    ids = np.array([18, 29, 40, 51, 62])
    before, support_before, ids_before = p.copy(), supported.copy(), ids.copy()
    calls = []

    def blade_mask(source_frame):
        calls.append(source_frame)
        result = np.zeros((80, 96), bool)
        if source_frame < 4:
            result[:, 35:61] = True
        return result

    args = dict(frame_indices=[0, 2, 4], depth_mm=[0., 1.5, 0.], tip_xy=np.full((3, 2), np.nan),
        supported=supported, point_ids=ids, contact_frame=0, reversal_frame=2,
        config=MaterialReviewConfig(view_mode='full', panel_size=(160, 120), trail_length=2))
    plain_path, masked_path = tmp_path/'plain.mp4', tmp_path/'masked.mp4'
    render_material_review(source, p, plain_path, **args)
    # An enormous but finite off-screen forecast must not overflow mask indices.
    with np.errstate(invalid='raise', over='raise'):
        result = render_material_review(source, p, masked_path, **args, blade_mask_reader=blade_mask)
    assert calls == [0, 2, 4]
    np.testing.assert_array_equal(p, before)
    np.testing.assert_array_equal(supported, support_before)
    np.testing.assert_array_equal(ids, ids_before)
    assert result['saved_point_count'] == 5
    assert result['per_frame_points_on_view'] == [4, 4, 4]
    assert result['per_frame_blade_occluded_ids'] == [2, 2, 0]

    def decode(path, frame_number):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        cap.release()
        assert ok
        return frame[84:204]

    plain, masked = decode(plain_path, 1), decode(masked_path, 1)
    # Actual MP4 pixels: both kinds of marker and the crossing trail disappear
    # within the blade. The left panel keeps the original grayscale texture.
    assert plain[20:90, 160+65:160+95].max() > 100
    assert masked[20:90, 160+65:160+95].max() < 15
    assert np.ptp(masked[20:90, 65:95], axis=2).max() < 12
    assert 85 < np.median(masked[20:90, 65:95]) < 110
    assert masked[85:100, 160+120:160+138].max() > 80  # non-occluded ID remains
    final = decode(masked_path, 2)
    assert final[48:59, 160+75:160+86].max() > 80  # same filled ID reappears
    assert final[67:78, 160+75:160+86].max() > 80  # same hollow ID reappears
