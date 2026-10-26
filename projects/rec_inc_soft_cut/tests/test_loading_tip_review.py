"""Loading trend display preserves the distinction between data and estimates."""
import cv2
import numpy as np
import pytest

from src.analysis.loading_tip_review import (
    _arrow_vector, _shared_scale, _state_runs, _interface_camera, render_loading_tip_review,
)


def test_source_classes_never_connect_unknowns_or_short_bridges():
    depth = np.arange(9)/3
    values = np.array([1., 2., 3., 4., 5., np.nan, 6., 7., 8.])
    states = np.array([4, 4, 2, 3, 0, 2, 1, 1, 4])
    runs = [(rows.tolist(), source) for rows, source in _state_runs(depth, values, states)]
    assert runs == [([0, 1], 4), ([2, 3], 2), ([6, 7], 1), ([8], 4)]


def test_shared_scale_ignores_unknown_placeholders_and_small_arrows_disappear():
    value = np.ones((3, 4, 2))*10.
    state = np.ones((3, 4), int)*4
    state[0, 0] = 0
    value[0, 0] = 1e20
    assert _shared_scale(value, state) == 10.
    assert _arrow_vector([1e-9, 1e-9], 10.) is None
    assert _arrow_vector([np.nan, 10.], 10.) is None
    arrow = _arrow_vector([30., 40.], 10.)
    np.testing.assert_allclose(arrow/np.linalg.norm(arrow), [.6, .8])
    assert 23. < np.linalg.norm(arrow) <= 24.


def _fixture(tmp_path):
    source = tmp_path/'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 12., (128, 96))
    if not writer.isOpened():
        pytest.skip('OpenCV MJPG encoder unavailable')
    for t in range(6):
        frame = np.full((96, 128, 3), 70+t, np.uint8)
        frame[:70, 58:69] = 220
        writer.write(frame)
    writer.release()
    points = np.array([[48., 32.], [40., 42.], [76., 32.], [84., 42.], [63., 30.]])
    positions = np.tile(points, (6, 1, 1))
    support = np.ones((6, 5), bool)
    support[:, 4] = False
    groups = np.tile([0, 1, 2, 3, -1], (6, 1))
    relative = np.tile([[10., 20.], [-20., -20.], [10., 30.], [-10., -40.]], (6, 1, 1))
    states = np.tile([4, 2, 3, 1], (6, 1))
    states[4, 1] = 0
    relative[4, 1] = np.nan
    trend = dict(group_index=groups, point_ids=np.arange(5), relative_trend_xy=relative,
        direct_trend_xy=relative.copy(), legacy_trend_y=np.ones((6, 4)), state=states,
        observation_count=np.ones((6, 4), int)*5, estimate_count=np.ones((6, 4), int)*2,
        tip_reference_state=np.array([2, 2, 1, 2, 2, 2]), um_per_px=2.,
        blade_geometry_uncertain=np.array([False, False, False, False, True, False]),
        interval_low_y=relative[..., 1]-2., interval_high_y=relative[..., 1]+2.)
    blade = np.zeros((96, 128), bool)
    blade[:70, 58:69] = True
    kwargs = dict(frame_indices=np.arange(6), depth_mm=np.array([0., .5, 1., 1.5, 2., 3.]),
                  tip_xy=np.tile([63., 69.], (6, 1)), point_ids=np.arange(5), trend=trend,
                  blade_mask_reader=lambda _: blade, codec='MJPG')
    return source, positions, support, kwargs


def test_six_frame_render_masks_blade_keeps_ids_and_is_independent_of_playback_rate(tmp_path):
    source, positions, support, kwargs = _fixture(tmp_path)
    before = positions.copy()
    first = render_loading_tip_review(source, positions, support, tmp_path/'review12.avi',
                                     overview_path=tmp_path/'overview.png', fps=12., **kwargs)
    second = render_loading_tip_review(source, positions, support, tmp_path/'review24.avi', fps=24., **kwargs)
    assert first['rendered_frames'] == second['rendered_frames'] == 6
    assert first['dimensions'] == [1600, 1000]
    assert first['phase'] == 'loading_only'
    assert first['per_frame_blade_occluded_ids'] == [1]*6
    assert first['camera_origin_xy'] == second['camera_origin_xy']
    assert first['display_scale_px_s'] == second['display_scale_px_s']
    assert first['original_material_identity_estimated'] is False
    assert first['region_geometry_uncertain_frames'] == 1
    assert first['all_saved_ids_retained'] is True
    assert first['source_state_counts_by_group'][1] == [1, 0, 5, 0, 0]
    assert len(first['overview_snapshots']) == 4
    np.testing.assert_array_equal(before, positions)
    image = cv2.imread(str(tmp_path/'overview_0mm.png'))
    assert image.shape == (1000, 1600, 3)
    # Source (63,30) is deep in the blade and projects to (1197,246).
    # The right pure-point panel must contain only the dim blade silhouette.
    patch = image[236:254, 1188:1205]
    assert int(patch.max()) < 45
    caps = [cv2.VideoCapture(str(tmp_path/f'review{fps}.avi')) for fps in (12, 24)]
    try:
        for _ in range(6):
            decoded = [cap.read() for cap in caps]
            assert all(ok for ok, _ in decoded)
            np.testing.assert_array_equal(decoded[0][1], decoded[1][1])
        assert all(not cap.read()[0] for cap in caps)
    finally:
        for cap in caps:
            cap.release()


def test_renderer_rejects_wrong_id_order_and_unsupported_finite_state(tmp_path):
    source, positions, support, kwargs = _fixture(tmp_path)
    kwargs['trend']['point_ids'] = np.arange(5)[::-1]
    with pytest.raises(ValueError, match='point IDs'):
        render_loading_tip_review(source, positions, support, tmp_path/'bad.avi', **kwargs)
    kwargs['trend']['point_ids'] = np.arange(5)
    kwargs['trend']['relative_trend_xy'][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite relative'):
        render_loading_tip_review(source, positions, support, tmp_path/'bad.avi', **kwargs)


def test_geometry_uncertainty_is_optional_and_must_align_when_supplied(tmp_path):
    source, positions, support, kwargs = _fixture(tmp_path)
    kwargs['trend']['blade_geometry_uncertain'] = np.zeros(5, bool)
    with pytest.raises(ValueError, match='geometry uncertainty'):
        render_loading_tip_review(source, positions, support, tmp_path/'bad_geometry.avi', **kwargs)
    del kwargs['trend']['blade_geometry_uncertain']
    result = render_loading_tip_review(source, positions, support, tmp_path/'old_schema.avi', **kwargs)
    assert result['region_geometry_uncertain_frames'] == 0


def test_full_interface_camera_includes_sparse_material_far_above_tip():
    tips = np.array([[500., 550.], [500., 552.], [500., 554.]])
    offsets = np.array([[-20., -350.], [-18., -200.], [22., -20.], [30., 35.]])
    positions = tips[:, None, :]+offsets[None, :, :]
    _, origins, crop, framing = _interface_camera(positions, np.ones((3, 4), bool),
        np.tile([0, 1, 2, 3], (3, 1)), tips, np.arange(3), 1024, 768)
    local = positions-origins[:, None, :]
    assert np.all((local >= 0) & (local < np.asarray(crop)))
    assert crop[1] > 400
    assert framing['selected_relative_bounds_xy'][0][1] <= -350


def _extended_fixture(tmp_path):
    source, positions, support, kwargs = _fixture(tmp_path)
    velocity = np.zeros((6, 4, 2, 2))
    velocity[..., 1] = 30.
    velocity[:, :, 1, 1] = -30.
    target = np.empty_like(velocity)
    for group in range(4):
        target[:, group, 0] = positions[0, group]
        target[:, group, 1] = positions[0, group]+[0., 10.]
    kwargs['trend'].update(segment_relative_trend_xy=velocity,
        segment_state=np.full((6, 4, 2), 4), segment_target_xy=target)
    kwargs['slip'] = dict(migration_um=np.arange(6)*20., state=np.array([4, 4, 2, 2, 0, 2]),
        interval_low_um=np.arange(6)*20.-10, interval_high_um=np.arange(6)*20.+10,
        initial_surface_xy=np.array([[10., 10.], [50., 50.], [90., 90.]]),
        address_initial_xy=np.column_stack([np.arange(6)*4.+40, np.arange(6)*4.+40]),
        initial_contact_xy=np.array([40., 40.]), reference_frame_index=0)
    return source, positions, support, kwargs


def test_combined_review_uses_local_segments_and_separate_initial_address(tmp_path):
    source, positions, support, kwargs = _extended_fixture(tmp_path)
    before = positions.copy()
    result = render_loading_tip_review(source, positions, support, tmp_path/'extended.avi',
        overview_path=tmp_path/'combined.png', **kwargs)
    assert result['rendered_frames'] == 6
    assert result['dimensions'] == [1800, 1200]
    assert result['per_frame_local_segment_arrows'] == [8]*6
    assert result['per_frame_blade_occluded_ids'] == [1]*6
    assert result['surface_sliding_state_counts'] == [1, 0, 3, 0, 2]
    assert result['initial_address_marker'] == 'original configuration inset only'
    assert result['original_material_identity_estimated'] is True
    image = cv2.imread(str(tmp_path/'combined_0mm.png'))
    assert image.shape == (1200, 1800, 3)
    # Deep blade pixels remain unpainted in the expanded pure-point panel.
    assert int(image[210:222, 1334:1341].max()) < 45
    np.testing.assert_array_equal(positions, before)


def test_combined_review_rejects_nonlocal_or_unaligned_measurements(tmp_path):
    source, positions, support, kwargs = _extended_fixture(tmp_path)
    kwargs['trend']['segment_relative_trend_xy'] = np.zeros((6, 4, 2))
    with pytest.raises(ValueError, match='Segment trends'):
        render_loading_tip_review(source, positions, support, tmp_path/'bad_segment.avi', **kwargs)
    kwargs['trend']['segment_relative_trend_xy'] = np.zeros((6, 4, 2, 2))
    kwargs['slip']['migration_um'] = np.ones(5)
    with pytest.raises(ValueError, match='Sliding field'):
        render_loading_tip_review(source, positions, support, tmp_path/'bad_slip.avi', **kwargs)


def test_full_interface_does_not_require_or_invent_old_tip_comparison_fields(tmp_path):
    source, positions, support, kwargs = _extended_fixture(tmp_path)
    del kwargs['trend']['direct_trend_xy']
    del kwargs['trend']['legacy_trend_y']
    original_keys = set(kwargs['trend'])
    result = render_loading_tip_review(source, positions, support, tmp_path/'interface_schema.avi', **kwargs)
    assert result['rendered_frames'] == 6
    assert result['direct_unfilled_coverage_by_group'] is None
    assert result['legacy_initial_layer_coverage_by_group'] is None
    assert set(kwargs['trend']) == original_keys
