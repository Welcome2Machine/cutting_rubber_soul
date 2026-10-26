"""Reference identity and bounded evaluation tests without neural downloads."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.workflows import optimize, reliability


def reference_check(positions, accepted, ids=(51, 4, 90)):
    return dict(positions=np.asarray(positions, float), accepted=np.asarray(accepted, bool),
                point_ids=np.asarray(ids))


def test_identity_consensus_requires_two_supported_consistent_same_id_positions():
    original = reference_check([[0, 0], [10, 0], [20, 0]], [True, True, False])
    recent = reference_check([[.4, 0], [15, 0], [20, 0]], [True, True, True])
    before = original['positions'].copy()
    result = reliability.identity_consensus(original, recent, [False, True, False])
    np.testing.assert_array_equal(result['identity_consistent'], [True, False, False])
    np.testing.assert_array_equal(result['recovered'], [True, False, False])
    np.testing.assert_array_equal(result['recent_only'], [False, True, True])
    np.testing.assert_array_equal(result['point_ids'], [51, 4, 90])
    np.testing.assert_array_equal(result['positions'], before)
    np.testing.assert_array_equal(original['positions'], before)


def test_consensus_does_not_nearest_neighbour_relabel_swapped_material_ids():
    original = reference_check([[0, 0], [10, 0], [20, 0]], [True]*3)
    recent = reference_check([[10, 0], [0, 0], [20, 0]], [True]*3)
    result = reliability.identity_consensus(original, recent, [True]*3)
    np.testing.assert_array_equal(result['identity_consistent'], [False, False, True])
    recent['point_ids'] = recent['point_ids'][[1, 0, 2]]
    with pytest.raises(ValueError, match='identit'):
        reliability.identity_consensus(original, recent, [True]*3)


def test_missing_or_nonfinite_reference_never_becomes_recovered():
    original = reference_check([[np.nan, 0], [10, 0], [20, 0]], [True, False, True])
    recent = reference_check([[0, 0], [10, 0], [np.nan, 0]], [True]*3)
    result = reliability.identity_consensus(original, recent, [False]*3)
    assert not result['identity_consistent'].any()
    assert not result['recovered'].any()


@pytest.mark.parametrize('tolerance', [0, -1, np.nan, np.inf])
def test_consensus_requires_finite_positive_agreement_tolerance(tolerance):
    check = reference_check([[0, 0], [10, 0], [20, 0]], [True]*3)
    with pytest.raises(ValueError):
        reliability.identity_consensus(check, check, [True]*3, tolerance)


def test_consensus_rejects_duplicate_ids_and_broadcast_support():
    check = reference_check([[0, 0], [10, 0], [20, 0]], [True]*3)
    with pytest.raises(ValueError):
        reliability.identity_consensus(check, check, True)
    duplicated = dict(check, point_ids=np.array([51, 51, 90]))
    with pytest.raises(ValueError):
        reliability.identity_consensus(duplicated, duplicated, [True]*3)


class Capture:
    def __init__(self, available=10):
        self.index, self.available, self.read_indices, self.released = 0, available, [], False

    def set(self, prop, value):
        self.index = int(value)

    def read(self):
        self.read_indices.append(self.index)
        if self.index >= self.available:
            return False, None
        image = np.full((8, 10, 3), [self.index, 40, 90], np.uint8)
        self.index += 1
        return True, image

    def release(self):
        self.released = True


def test_network_window_reads_exact_prefix_and_preserves_crop_and_rgb(monkeypatch):
    capture = Capture()
    monkeypatch.setattr(reliability.cv2, 'VideoCapture', lambda _: capture)
    result = reliability._window('unused', 2, 4, np.array([1, 2]), np.array([5, 6]))
    assert result.shape == (3, 4, 4, 3)
    np.testing.assert_array_equal(result[:, 0, 0], [[90, 40, 2], [90, 40, 3], [90, 40, 4]])
    assert capture.read_indices == [2, 3, 4]
    assert capture.released


def test_decode_failure_releases_capture_without_filling_future_frames(monkeypatch):
    capture = Capture(available=4)
    monkeypatch.setattr(reliability.cv2, 'VideoCapture', lambda _: capture)
    with pytest.raises(ValueError, match='decode'):
        reliability._window('unused', 2, 5, np.array([1, 2]), np.array([5, 6]))
    assert capture.read_indices == [2, 3, 4]
    assert capture.released


@pytest.mark.parametrize('first,last,lower,upper', [
    (-1, 3, [0, 0], [5, 5]), (4, 3, [0, 0], [5, 5]),
    (1, 3, [-1, 0], [5, 5]), (1, 3, [0, 0], [11, 5]),
    (1, 3, [4, 2], [4, 6]),
])
def test_network_window_rejects_invalid_or_silently_truncated_crop(monkeypatch, first, last, lower, upper):
    capture = Capture()
    monkeypatch.setattr(reliability.cv2, 'VideoCapture', lambda _: capture)
    with pytest.raises(ValueError):
        reliability._window('unused', first, last, np.asarray(lower), np.asarray(upper))


def valid_config():
    return dict(cases=[], network_backends=['cotracker3', 'tapnextpp'], maximum_identity_points=24,
                network_crop_side_px=320, original_surface_band_px=8., identity_agreement_px=1.5,
                mechanics_weight=.01, mechanics_bulk_shear_ratios=[2., 20.],
                geometry_evidence='results/geometry', reference_lag_frames=12,
                pilot_depths_mm=[.5, 1.], maximum_nodes=100, near_tip_radius_px=100.)


@pytest.mark.parametrize('key,value', [
    ('network_backends', ['unknown']), ('network_backends', ['cotracker3', 'cotracker3']),
    ('identity_agreement_px', np.nan), ('original_surface_band_px', -1),
    ('network_crop_side_px', 0), ('maximum_identity_points', 0),
    ('mechanics_weight', np.inf), ('mechanics_bulk_shear_ratios', [0, 20]),
])
def test_reliability_configuration_fails_before_models_or_outputs(key, value):
    config = valid_config()
    config[key] = value
    with pytest.raises(ValueError):
        reliability.validate_config(config)


def test_reliability_check_only_never_runs_models_or_creates_results(tmp_path, monkeypatch):
    config = tmp_path/'config.json'
    config.write_text(json.dumps(valid_config()))
    monkeypatch.setattr(reliability, 'run', lambda *a: pytest.fail('Models must not run'))
    out = tmp_path/'outputs'
    assert optimize.main(['--config', str(config), '--out', str(out), '--stage', 'reliability', '--check-only']) == 0
    assert not out.exists()


def test_reliability_check_only_rejects_unknown_backend(tmp_path):
    config = valid_config()
    config['network_backends'] = ['not_a_model']
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        optimize.main(['--config', str(path), '--stage', 'reliability', '--check-only'])


def test_real_evaluation_uses_first_depth_crossing_without_reentry_or_future_reads(tmp_path, monkeypatch):
    td, md, prep = (tmp_path/k for k in ('tracks', 'measurements', 'prepared'))
    prep.mkdir()
    np.savez(prep/'input.npz', scale=np.array(100.))
    seeds = np.array([[1000., 1000.], [1100., 1000.]])  # deliberately outside the tip ROI
    tracks = dict(point_ids=np.array([8, 2]), seed_points=seeds,
                  positions=np.tile(seeds, (16, 1, 1)), source_frame_indices=np.arange(16))
    guard = dict(source_frames=np.arange(10, 16), tips_xy=np.c_[np.zeros(6), [0., 4., 9., 11., -8., np.nan]],
                 state=np.full(6, 2))
    observations = dict(point_ids=tracks['point_ids'], frame_indices=np.arange(16), supported=np.ones((16, 2), bool))
    fixtures = {'blade_reference_guard.npz': guard, 'tracks.npz': tracks,
                'measurement_support.npz': observations,
                'initial_surface_extended.npz': {'boundary_xy': seeds}}
    monkeypatch.setattr(reliability, 'load_npz', lambda path: fixtures[Path(path).name])
    monkeypatch.setattr(reliability, 'sha', lambda _: 'fixture')
    read = []
    def frame(video, index):
        read.append(index)
        return np.zeros((20, 20, 3), np.uint8)
    monkeypatch.setattr(reliability, '_frame', frame)
    monkeypatch.setattr(reliability, '_mask', lambda *a: np.ones((20, 20), np.uint8))
    monkeypatch.setattr(reliability, '_window', lambda *a: pytest.fail('No near-surface queries'))
    out = tmp_path/'results'
    out.mkdir()
    audit = tmp_path/'contact_audit.json'
    audit.write_text(json.dumps({'accepted': True, 'selected_contact_frame': 10}))
    case = dict(name='synthetic_case', tracks=str(td), measurements=str(md), prepared=str(prep),
                role='development_only', stop_mm=1., contact_audit=str(audit))
    config = dict(valid_config(), reference_lag_frames=1, pilot_depths_mm=[.5, 1., 1.5])
    records = reliability.evaluate_case(case, config, out, networks={})
    assert [r['frame'] for r in records] == [11, 12]
    assert [r['actual_depth_mm'] for r in records] == [.4, .9]
    assert max(read) == 12
    assert all(not r['identity'] for r in records)
    assert (out/'synthetic_case'/'provenance.json').is_file()


def valid_identity_config():
    # Deliberately independent of legacy mechanics and recent-window settings.
    return dict(cases=[], original_surface_band_px=8., initial_neighbourhood_px=180.,
                identity_boundary_tolerance_px=1.5, identity_network_crop_side_px=512,
                identity_network_max_frames=1200, identity_blade_core_erosion_px=2,
                identity_prefix_stride=1, identity_agreement_px=1.5)


def test_identity_cohort_keeps_all_initial_surface_ids_and_explicit_regressions():
    seeds = np.c_[np.arange(50), np.zeros(50)].astype(float)
    seeds = np.vstack((seeds, [[500., 500.], [20., 40.]]))
    ids = np.arange(52)*17+103  # identities are not row indices
    surface = np.c_[np.arange(60), np.zeros(60)]
    config = dict(valid_identity_config(), maximum_identity_points=2)
    selected = reliability.identity_cohort(seeds, surface, np.array([25., 0.]),
        config, regression_ids=[ids[50]], point_ids=ids)
    np.testing.assert_array_equal(selected, np.arange(51))
    # The distant regression is retained despite failing both geometry tests;
    # no old 24-point cap or current/recent observation filter may trim the set.
    assert ids[50] in ids[selected]
    assert ids[51] not in ids[selected]


def test_identity_cohort_is_coordinate_translation_and_id_order_invariant():
    seeds = np.array([[0., 0.], [30., 2.], [30., 30.], [600., 600.]])
    surface = np.array([[0., 0.], [30., 0.]])
    tip = np.array([0., 0.])
    ids = np.array([105, 7, 40, 13])
    config = valid_identity_config()
    selected = reliability.identity_cohort(seeds, surface, tip, config, [13], ids)
    np.testing.assert_array_equal(ids[selected], [105, 7, 13])
    order = np.array([3, 1, 2, 0])
    offset = np.array([900., -80.])
    shifted = reliability.identity_cohort(seeds[order]+offset, surface+offset, tip+offset,
        config, [13], ids[order])
    assert set(ids[order][shifted]) == set(ids[selected])


def test_identity_cohort_rejects_unknown_regression_instead_of_silently_dropping_it():
    seeds = np.array([[0., 0.], [30., 0.]])
    with pytest.raises(ValueError):
        reliability.identity_cohort(seeds, seeds, seeds[0], valid_identity_config(),
                                    regression_ids=[999], point_ids=np.array([4, 8]))


def test_identity_configuration_is_independent_of_neural_mechanics_ablation():
    reliability.validate_identity_config(valid_identity_config())


@pytest.mark.parametrize('key,value', [
    ('original_surface_band_px', 0), ('initial_neighbourhood_px', np.inf),
    ('identity_agreement_px', np.nan), ('identity_agreement_px', -1.),
    ('identity_boundary_tolerance_px', 0), ('identity_boundary_tolerance_px', 2.1),
    ('identity_network_crop_side_px', 127), ('identity_network_crop_side_px', 1025),
    ('identity_network_crop_side_px', 512.5), ('identity_network_max_frames', 1),
    ('identity_network_max_frames', 2001), ('identity_network_max_frames', 1200.5),
    ('identity_blade_core_erosion_px', -1), ('identity_blade_core_erosion_px', 3),
    ('identity_blade_core_erosion_px', .5), ('identity_prefix_stride', 2),
])
def test_identity_configuration_rejects_unbounded_or_skipped_frame_work(key, value):
    config = valid_identity_config()
    config[key] = value
    with pytest.raises(ValueError):
        reliability.validate_identity_config(config)


def test_identity_regression_evidence_must_belong_to_the_declared_case(tmp_path):
    case = tmp_path/'chosen'
    case.mkdir()
    config = dict(valid_identity_config(), regression_evidence=str(tmp_path), cases=[{'name': 'chosen'}])
    path = case/'summary.json'
    path.write_text(json.dumps([{'case': 'chosen', 'frame': 12, 'reference_frame': 0}]))
    reliability.validate_identity_config(config)  # one case/endpoint is a legitimate bounded rerun
    for records in ([], [{'case': 'different_case', 'frame': 12, 'reference_frame': 0}],
                    [{'case': 'chosen', 'frame': 12, 'reference_frame': 12}],
                    [{'case': 'chosen', 'frame': 12}]):
        path.write_text(json.dumps(records))
        with pytest.raises(ValueError):
            reliability.validate_identity_config(config)
    path.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        reliability.validate_identity_config(config)


def test_identity_check_only_does_not_run_models_or_create_outputs(tmp_path, monkeypatch):
    config = tmp_path/'identity.json'
    config.write_text(json.dumps(valid_identity_config()))
    monkeypatch.setattr(reliability, 'run_identity', lambda *a: pytest.fail('Check-only must not execute identity models'))
    out = tmp_path/'outputs'
    assert optimize.main(['--config', str(config), '--stage', 'identity', '--out', str(out), '--check-only']) == 0
    assert not out.exists()


def test_identity_stage_routes_only_identity_and_records_unchanged_production(tmp_path, monkeypatch):
    config = valid_identity_config()
    path = tmp_path/'identity.json'
    path.write_text(json.dumps(config))
    out = tmp_path/'results'/'identity'
    monkeypatch.setattr(optimize, 'ROOT', tmp_path)
    monkeypatch.setattr(reliability, 'run', lambda *a: pytest.fail('Identity must not run legacy reliability ablation'))
    calls = []
    monkeypatch.setattr(reliability, 'run_identity', lambda supplied, destination: calls.append((supplied, destination)))
    assert optimize.main(['--config', str(path), '--stage', 'identity', '--out', str(out)]) == 0
    assert calls == [(config, out.resolve())]
    evidence = json.loads((out/'execution.json').read_text())
    assert evidence['stage'] == 'identity'
    assert evidence['optical_flow_default_changed'] is False
    assert evidence['source_results_modified'] is False
    assert evidence['model_training'] is False
    assert 'src/tracking/identity_chain.py' in evidence['source_sha256']
