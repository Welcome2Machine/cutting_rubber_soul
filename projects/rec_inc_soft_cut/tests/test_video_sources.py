import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.video_sources as sources
from src.workflows.video_import import selected_frame
from src.tracking.shadow_normalization import MaterialShadowNormalizer
from src.workflows.tracking_inputs import session_for_input


def test_source_routing_preserves_case_and_rejects_ambiguity(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, 'VIDEO_ROOT', tmp_path)
    folder = tmp_path/'20261005'
    folder.mkdir()
    (folder/'1005L60C1up.mp4').touch()
    assert sources.video_directory('1005L60C1up') == folder
    assert sources.video_directory('0930l60a1') == tmp_path/'0930'
    with pytest.raises(FileNotFoundError):
        sources.video_directory('1014missing')
    second = tmp_path/'other'
    second.mkdir()
    (second/'1005L60C1up.mp4').touch()
    with pytest.raises(ValueError, match='Ambiguous'):
        sources.video_directory('1005L60C1up')


@pytest.mark.parametrize('name', ['../1005a', '1005a.mp4', '1005:a', '1005a/b', '1005a\\b'])
def test_source_name_cannot_escape_root(name):
    with pytest.raises(ValueError):
        sources.validate_case_names([name])


def test_import_sampling_has_exact_recorded_source_indices():
    for count in (10, 11, 12, 13, 14, 4835):
        indices = np.array([i for i in range(count) if selected_frame(i)])
        expected = np.arange(len(indices))//2*5 + np.arange(len(indices))%2*3
        np.testing.assert_array_equal(indices, expected)


def test_unknown_grid_cannot_silently_use_old_preprocessing(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, 'VIDEO_ROOT', tmp_path)
    folder = tmp_path/'20261005'
    folder.mkdir()
    (folder/'1008n30a1.mp4').touch()
    (folder/'1008n30a1.json').write_text(json.dumps(dict(grid_0915_verified=False)))
    with pytest.raises(ValueError, match='Unverified'):
        sources.input_profile('1008n30a1')
    profile = dict(grid_enabled=False, overlay_enabled=False, acquisition_fps=200.,
                   acquisition_clock_status='unconfirmed')
    (folder/'1008n30a1_profile.json').write_text(json.dumps(profile))
    assert sources.input_profile('1008n30a1') == profile


def test_grid_free_preprocessing_preserves_texture_at_former_grid_lines():
    image = np.random.default_rng(4).integers(10, 250, (768, 1024, 3), dtype=np.uint8)
    processor = MaterialShadowNormalizer(grid_enabled=False)
    np.testing.assert_array_equal(processor.grid(image), image)
    assert not processor.exclude_grid.any()
    # A nonmaterial image must remain pixel-identical, including former grid rows.
    actual, _ = processor.process(image, np.zeros(image.shape[:2], np.uint8))
    np.testing.assert_array_equal(actual, image)
    session = dict(model=object(), config=dict(preprocessing=dict(
        grid_suppression=dict(enabled=True), overlay_redaction=dict(enabled=True))))
    configured = session_for_input(session, dict(grid_enabled=False, overlay_enabled=False))
    assert not configured['config']['preprocessing']['grid_suppression']['enabled']
    assert session['config']['preprocessing']['grid_suppression']['enabled']
    assert configured['model'] is session['model']


@pytest.mark.parametrize('failed_frame,permanent', [(3, False), (4, True), (3, True)])
def test_import_recovers_exact_frame_without_dropping_or_duplicating(monkeypatch, tmp_path, failed_frame, permanent):
    import cv2
    from src.workflows import video_import as importer
    source = tmp_path/'1014fixture.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 30, (1024, 768))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((768, 1024, 3), 20+15*i, np.uint8))
    writer.release()
    capture = cv2.VideoCapture
    failed_once = []
    class OneInterruptedRead:
        def __init__(self, path):
            self.cap = capture(path)
            self.source = str(path) == str(source)
        def read(self):
            if self.source and (permanent or not failed_once) and int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == failed_frame:
                failed_once.append(True)
                return False, None
            return self.cap.read()
        def __getattr__(self, key):
            return getattr(self.cap, key)
    monkeypatch.setattr(importer.cv2, 'VideoCapture', OneInterruptedRead)
    if permanent and failed_frame == 3:
        with pytest.raises(ValueError, match='truncated'):
            importer.import_one(source, tmp_path/'destination', lambda _: None)
        assert not (tmp_path/'destination/1014fixture.mp4').exists()
        return
    result = importer.import_one(source, tmp_path/'destination', lambda _: None)
    assert result['frames'] == 4 and result['source_sequentially_decoded_frames'] == 10-int(permanent)
    assert result['decode_recoveries'] == ([] if permanent else [dict(source_frame=3, attempt=1)])
    assert result['unreadable_unselected_source_frames'] == ([4] if permanent else [])
    with np.load(tmp_path/'destination/1014fixture_alignment.npz') as data:
        np.testing.assert_array_equal(data['source_indices'], [0, 3, 5, 8])
    cap = capture(result['output'])
    for expected in (20, 65, 95, 140):
        ok, frame = cap.read()
        assert ok and abs(float(frame.mean())-expected) < 4
    assert not cap.read()[0]
    cap.release()
