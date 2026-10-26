"""Saved estimates survive presentation without refitting or hidden gap filling."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.io import sha
from src.workflows import gallery


def make_manifest(tmp_path, **arrays):
    source = tmp_path/'estimate.npz'
    values = dict(frames=np.array([10, 12, 14, 16]), depth_mm=np.array([0., .3, .6, .9]),
                  offset_um=np.array([0., 10., np.nan, 25.]), available=np.array([1, 1, 0, 1], bool))
    values.update(arrays)
    np.savez(source, **values)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(cases=[dict(name='sample', group='Custom <group>', cutoff_um=1000,
        estimate='estimate.npz', prepared='prepared', tracks='tracks', measurements='measurements')])), encoding='utf-8')
    return manifest, source


def test_display_values_are_used_without_completing_other_gaps(tmp_path):
    manifest, source = make_manifest(tmp_path, display_offset_completed_um=np.array([0., 9., 17., 26.]),
                                    display_estimate_kind=np.array([1, 1, 2, 0]))
    digest = sha(source)
    data = gallery.series(gallery.load_cases(manifest)[0])
    np.testing.assert_array_equal(data['values'], [0., 9., 17., np.nan])
    np.testing.assert_array_equal(data['inferred'], [False, False, True, False])
    assert data['field'] == 'display_offset_completed_um'
    assert sha(source) == digest


def test_raw_availability_and_first_cutoff_crossing_are_respected(tmp_path):
    manifest, _ = make_manifest(tmp_path, depth_mm=np.array([0., .5, 1.1, .8]),
                               offset_um=np.array([0., 10., 20., 30.]), available=np.array([1, 0, 1, 1]))
    data = gallery.series(gallery.load_cases(manifest)[0])
    np.testing.assert_array_equal(data['frames'], [10, 12])
    np.testing.assert_array_equal(data['values'], [0., np.nan])


def test_offset_availability_overrides_legacy_flag_and_finite_predictions(tmp_path):
    manifest, source = make_manifest(tmp_path, offset_um=np.array([0., 10., 20., 30.]),
        available=np.ones(4, bool), transport_offset_available=np.array([1, 0, 1, 0], bool))
    digest = sha(source)
    data = gallery.series(gallery.load_cases(manifest)[0])
    np.testing.assert_array_equal(data['values'], [0., np.nan, 20., np.nan])
    assert sha(source) == digest


def test_material_frames_align_by_identity_not_row_number():
    np.testing.assert_array_equal(gallery.rows_at(np.array([0, 10, 12, 14]), np.array([10, 14])), [1, 3])
    with pytest.raises(ValueError, match='lacks saved'):
        gallery.rows_at(np.array([0, 10, 12, 14]), np.array([10, 11]))
    with pytest.raises(ValueError, match='increasing'):
        gallery.rows_at(np.array([0, 10, 10]), np.array([10]))


def test_display_interpolation_does_not_cross_missing_samples_or_extrapolate():
    frames, values = np.array([0, 2, 4, 6]), np.array([0., 10., np.nan, 30.])
    assert gallery.interpolation_value(frames, values, 1) == 5.
    for frame in (-1, 3, 4, 5, 7):
        assert np.isnan(gallery.interpolation_value(frames, values, frame))


def test_existing_output_rejects_silently_changed_estimates(tmp_path):
    manifest, source = make_manifest(tmp_path)
    case = gallery.load_cases(manifest)[0]
    out = tmp_path/'out'
    gallery.prepare(case, out)
    with np.load(source) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['offset_um'][1] += 1
    np.savez(source, **arrays)
    with pytest.raises(FileExistsError, match='different inputs'):
        gallery.prepare(case, out)


def test_plot_and_index_are_static_english_and_do_not_change_sources(tmp_path):
    manifest, source = make_manifest(tmp_path)
    out = tmp_path/'out'
    digest = sha(source)
    assert gallery.main(['--manifest', str(manifest), '--out', str(out), '--stage', 'plot']) == 0
    assert gallery.main(['--manifest', str(manifest), '--out', str(out), '--stage', 'index']) == 0
    svg = (out/'sample/curve.svg').read_text(encoding='utf-8')
    assert 'Displacement (mm)' in svg and 'Slip (μm)' in svg
    page = (out/'index.html').read_text(encoding='utf-8')
    assert '<script' not in page and 'sample/curve.png' in page and 'Custom &lt;group&gt;' in page
    assert '<video' not in page  # No broken video element for a plot-only export.
    assert sha(source) == digest
    saved = json.loads((out/'sample/calculation.json').read_text(encoding='utf-8'))
    assert saved['last_saved_slip_um'] == 25 and not saved['estimates_modified']


def test_video_validation_checks_decode_size_rate_and_duration(tmp_path, monkeypatch):
    path = tmp_path/'fake.mp4'
    path.write_bytes(b'test container')
    class Capture:
        def get(self, key):
            return {gallery.cv2.CAP_PROP_FRAME_COUNT: 12, gallery.cv2.CAP_PROP_FPS: 3,
                    gallery.cv2.CAP_PROP_FRAME_WIDTH: 1024, gallery.cv2.CAP_PROP_FRAME_HEIGHT: 384}[key]
        def read(self):
            return True, None
        def set(self, *args):
            pass
        def release(self):
            pass
    monkeypatch.setattr(gallery.cv2, 'VideoCapture', lambda _: Capture())
    info = gallery.video_info(path, expected_fps=3, expected_size=(1024, 384), expected_duration=4)
    assert info['first_and_last_frame_decoded']
    with pytest.raises(ValueError, match='duration'):
        gallery.video_info(path, expected_duration=5)
    with pytest.raises(ValueError, match='dimensions'):
        gallery.video_info(path, expected_size=(1280, 480))
    with pytest.raises(ValueError, match='rate'):
        gallery.video_info(path, expected_fps=12)
    original_get = Capture.get
    monkeypatch.setattr(Capture, 'get', lambda self, key:
        1 if key == gallery.cv2.CAP_PROP_FRAME_COUNT else original_get(self, key))
    decoded = iter([(True, None), (False, None)])
    monkeypatch.setattr(Capture, 'read', lambda self: next(decoded))
    assert gallery.video_info(path)['frames'] == 1  # First and last are the same decoded frame.
    monkeypatch.setattr(Capture, 'read', lambda self: (False, None))
    with pytest.raises(ValueError, match='cannot be decoded'):
        gallery.video_info(path)


def test_manifest_rejects_unsafe_or_duplicate_case_names(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    rows = json.loads(manifest.read_text())
    rows['cases'][0]['name'] = '../sample'
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='plain directory'):
        gallery.load_cases(manifest)
    rows['cases'][0]['name'] = 'sample'
    rows['cases'].append(dict(rows['cases'][0], name='Sample'))
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='Duplicate'):
        gallery.load_cases(manifest)
