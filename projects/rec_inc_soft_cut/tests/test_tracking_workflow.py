import json
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from src.workflows import tracking as batch


@pytest.mark.parametrize('final_depth', [0.3, np.nan])
def test_full_video_exports_withdrawal_and_both_views(monkeypatch, tmp_path, final_depth):
    name = '0930l30a1'
    folder = tmp_path/'result'/name
    (folder/'tracking').mkdir(parents=True)
    (folder/'plan.json').write_text(json.dumps(dict(frame_end=4, display_frames=5,
        full_video_frames=5, full_video=True, query_scope='full_material',
        contact_frame=1, reversal_frame=3, contact_to_reversal_mm=3.)))
    (folder/'tracking/complete.json').write_text(json.dumps(dict(seconds=1., code_sha256='frozen')))
    np.savez(folder/'tracking/tracks.npz', source_frame_indices=np.arange(5),
        positions=np.zeros((5, 2, 2)), supported=np.ones((5, 2), bool), point_ids=np.array([4, 8]))
    np.savez(folder/'geometry.npz', tips_px=np.zeros((5, 2)),
        depth_mm=np.array([-.1, 0., 1.5, 3., final_depth]))
    data = tmp_path/'data/videos/0930'
    data.mkdir(parents=True)
    np.savez(data/f'{name}_alignment.npz', source_indices=np.array([0, 3, 5, 8, 10]))
    # Rendering uses the prepared case's clock, including any acquisition-clock
    # correction, rather than reconstructing times from a hard-coded 0930 path.
    physical_times = np.array([0., .030, .050, .080, .100])
    np.savez(folder/'frame_mapping.npz', physical_time_s=physical_times)
    monkeypatch.setattr(batch, 'ROOT', tmp_path)
    render = Mock(side_effect=lambda video, positions, output, **kwargs: dict(output_path=str(output)))
    summary = Mock(return_value=dict(loading_frames=3, withdrawal_frames=1))
    module = SimpleNamespace(render_material_review=render, summarize_material_tracks=summary,
                             MaterialReviewConfig=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, 'src.analysis.material_tracking_review', module)
    from src.workflows import tracking as validation
    monkeypatch.setattr(validation, 'encode_review_video', lambda _: dict(codec='h264'))
    result = batch.render(name, folder, SimpleNamespace(target_mm=3.))
    assert result['frames'] == 5 and result['full_video']
    assert result['actual_final_depth_mm'] == (None if np.isnan(final_depth) else final_depth)
    assert render.call_count == 2
    assert render.call_args_list[0].kwargs['config'].view_mode == 'full'
    for call in render.call_args_list:
        np.testing.assert_array_equal(call.kwargs['frame_indices'], np.arange(5))
        assert call.kwargs['contact_frame'] == 1 and call.kwargs['reversal_frame'] == 3
    assert summary.call_args.kwargs['reversal_frame'] == 3
    np.testing.assert_allclose(summary.call_args.kwargs['times_s'], physical_times)


from src.workflows.tracking import validate_cohort, mask_fingerprint


def test_preserves_fixed_ids_and_all_frames():
    validate_cohort(np.array([[20., 30.], [40., 50.]]), np.array([19, 81]), np.arange(31, 38))


@pytest.mark.parametrize('ids,frames', [
    (np.array([1, 1]), np.arange(4)),
    (np.array([1, 2]), np.array([0, 1, 3, 4])),
    (np.array([1, 2]), np.array([0, 1, 1, 2])),
    (np.array([1, 2]), np.array([-1, 0, 1])),
    (np.array([1, 2]), np.array([0.])),
])
def test_rejects_identity_or_timeline_changes(ids, frames):
    with pytest.raises(ValueError):
        validate_cohort(np.zeros((2, 2)), ids, frames)


def test_nonfinite_seeds_are_not_silently_replaced():
    with pytest.raises(ValueError):
        validate_cohort(np.array([[np.nan, 2.]]), np.array([0]), np.arange(3))


def test_mask_cache_fingerprint_follows_only_consumed_masks(tmp_path):
    for index in range(3):
        (tmp_path / f'f{index:06d}.png').write_bytes(bytes([index]))
    original = mask_fingerprint(tmp_path, np.arange(2))
    (tmp_path / 'f000002.png').write_bytes(b'not in the tracked timeline')
    assert mask_fingerprint(tmp_path, np.arange(2)) == original
    (tmp_path / 'f000001.png').write_bytes(b'changed material or blade mask')
    assert mask_fingerprint(tmp_path, np.arange(2)) != original


def test_preflight_requires_only_active_segmentation_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, 'ROOT', tmp_path)
    segmentation = tmp_path/'models/segmentation/clean_roi_20260909/checkpoint_best.pt'
    segmentation.parent.mkdir(parents=True)
    segmentation.write_bytes(b'local model fixture; no inference in preflight')
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path/'empty_home'))
    monkeypatch.setattr(batch.importlib.util, 'find_spec', lambda _: object())
    monkeypatch.setattr(batch.shutil, 'which', lambda _: 'ffmpeg')
    report = batch.preflight([], tracker='independent')
    assert report['ok'] and report['flow_checkpoint'] is None
    assert report['gpu_required_for'] == ('prepare',)
    with pytest.raises(ValueError, match='independent'):
        batch.preflight([], tracker='cotracker')
