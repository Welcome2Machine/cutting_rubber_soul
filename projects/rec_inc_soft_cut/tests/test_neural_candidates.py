"""Network adapters must preserve coordinates, time limits and evidence semantics."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.tracking.neural_candidates import CoTrackerCandidates, TAPNextCandidates, _inputs


def test_input_identity_and_cutoff_are_not_silently_repaired():
    frames = np.zeros((3, 71, 103, 3), np.uint8)
    query = [[0, 0, 0], [2, 102, 70]]
    _, _, ids, indices = _inputs(frames, query, [89, 17], [8, 12, 15], 15)
    np.testing.assert_array_equal(ids, [89, 17])
    np.testing.assert_array_equal(indices, [8, 12, 15])
    for kwargs in (dict(point_ids=[3, 3]), dict(frame_indices=[8, 7, 15]), dict(cutoff_frame=14)):
        defaults = dict(point_ids=[89, 17], frame_indices=[8, 12, 15], cutoff_frame=15)
        defaults.update(kwargs)
        with pytest.raises(ValueError):
            _inputs(frames, query, **defaults)


@pytest.mark.parametrize('query', [[[.5, 1, 1]], [[0, 102.5, 1]], [[3, 1, 1]], [[0, np.nan, 1]], []])
def test_invalid_queries_are_rejected(query):
    with pytest.raises(ValueError):
        _inputs(np.zeros((3, 71, 103, 3), np.uint8), query, None, None, None)


def test_rgb_uint8_contract_prevents_silent_color_and_scale_assumptions():
    with pytest.raises(ValueError, match='RGB uint8'):
        _inputs(np.zeros((3, 20, 20), np.uint8), [[0, 1, 1]], None, None, None)
    with pytest.raises(ValueError, match='RGB uint8'):
        _inputs(np.zeros((3, 20, 20, 3), float), [[0, 1, 1]], None, None, None)


def test_resize_inverse_query_order_and_no_observation_promotion(monkeypatch):
    torch = pytest.importorskip('torch')
    tracker = CoTrackerCandidates.__new__(CoTrackerCandidates)
    tracker.device = torch.device('cpu')
    tracker.processing_shape = (64, 96)
    tracker.iterations, tracker.maximum_frames = 6, 10
    captured = {}

    def model(video, queries, iters):
        captured.update(video=video, queries=queries, iters=iters)
        positions = queries[:, None, :, 1:].expand(-1, 3, -1, -1).clone()
        positions[..., 0] += 95 / 102  # exactly one native input pixel
        score = torch.full(positions.shape[:3], .8)
        return positions, score, score + .01, None

    tracker.model = model
    monkeypatch.setitem(sys.modules, 'cotracker.models.core.model_utils', SimpleNamespace(
        get_points_on_a_grid=lambda count, shape, device: torch.zeros((1, count*count, 2), device=device)))
    frames = np.zeros((3, 71, 103, 3), np.uint8)
    frames[..., 0] = 240
    queries = np.array([[2, 102, 70], [0, 8.2, 9.1]])
    result = tracker.track(frames, queries, point_ids=[99, 10], frame_indices=[30, 31, 32], cutoff_frame=32)
    np.testing.assert_allclose(captured['queries'][0, 0].numpy(), [2, 95, 63])
    assert captured['video'].shape == (1, 3, 3, 64, 96)
    assert captured['video'][0, 0, 0, 0, 0] == 240
    np.testing.assert_allclose(result['positions'][2, 0], [102, 70])
    np.testing.assert_allclose(result['positions'][0, 1], [8.2, 9.1])
    np.testing.assert_allclose(result['positions'][2, 1], [9.2, 9.1], atol=1e-5)
    assert np.isnan(result['positions'][:2, 0]).all()
    assert not result['visibility'].any() and not result['observed'].any()
    assert result['query_anchored'].sum() == 2
    assert result['visibility_score'][0, 1] == pytest.approx(.8)
    np.testing.assert_array_equal(result['point_ids'], [99, 10])
    assert result['metadata']['uses_future_within_supplied_window']
    tracker.maximum_frames = 2
    with pytest.raises(ValueError, match='maximum_frames'):
        tracker.track(frames, queries)


def test_tapnext_xy_yx_half_pixel_mapping_and_causal_state_reset():
    torch = pytest.importorskip('torch')
    tracker = TAPNextCandidates.__new__(TAPNextCandidates)
    tracker.device, tracker.maximum_frames = torch.device('cpu'), 8
    captured = []

    def model(video, query_points, state):
        step = 0 if state is None else state[0] + 1
        q = query_points if state is None else state[1]
        captured.append((video.shape, None if query_points is None else query_points.clone(), step))
        yx = q[..., 1:].clone()
        yx[..., 0] += step * 256 / 71
        logits = torch.full((1, 1, q.shape[1], 1), -1.)
        return yx[:, None], None, logits, (step, q)

    tracker.model = model
    frames = np.zeros((3, 71, 103, 3), np.uint8)
    q = [[0, 102, 70], [0, 8.2, 9.1]]
    result = tracker.track(frames, q, point_ids=[81, 5], frame_indices=[4, 5, 6], cutoff_frame=6)
    np.testing.assert_allclose(captured[0][1][0, 0].numpy(), [0, 70.5*256/71, 102.5*256/103])
    assert captured[0][0] == (1, 1, 256, 256, 3)
    assert captured[1][1] is None
    np.testing.assert_allclose(result['positions'][2], [[102, 72], [8.2, 11.1]], atol=1e-5)
    assert not result['visibility'].any() and not result['observed'].any()
    assert np.isnan(result['confidence_score']).all()
    assert not result['metadata']['uses_future_within_supplied_window']
    # Future image changes do not influence the model's earlier invocation.
    tracker.track(frames, q)
    assert captured[3][2] == 0
    with pytest.raises(ValueError, match='first-frame'):
        tracker.track(frames, [[1, 1, 1]])


def test_unsigned_descending_frame_ids_do_not_wrap_into_a_valid_window():
    with pytest.raises(ValueError, match='strictly increasing'):
        _inputs(np.zeros((3, 20, 20, 3), np.uint8), [[0, 1, 1]], None,
                np.array([4, 2, 9], dtype=np.uint64), None)
