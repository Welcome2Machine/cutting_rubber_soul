"""Track identities, missing-data rules and independent streaming validation."""
import csv
import gzip
import json

import numpy as np
import pytest

from src.exchange.observations import export_tracks
from src.exchange.validate import _displacement_roundoff_bound, validate_case, validate_root


def fixture_case(tmp_path):
    name = 'sample'
    source = tmp_path / 'tracks' / name
    measure = tmp_path / 'measurements' / name
    (source / 'tracking').mkdir(parents=True)
    measure.mkdir(parents=True)
    p = np.array([[[10, 10], [30, 20]], [[11, 12], [31, 22]],
                  [[12, 14], [32, 24]], [[13, 16], [33, 26]]], np.float32)
    raw = np.array([[1, 1], [0, 1], [1, 1], [1, 1]], bool)
    on_blade = np.zeros((4, 2), bool)
    on_blade[2, 1] = True
    good = raw & ~on_blade
    ids, frames = np.array([7, 9]), np.arange(4)
    np.savez(source / 'tracking/tracks.npz', positions=p, seed_points=p[0], point_ids=ids,
        supported=raw, visible=raw, position_kind=np.where(raw, 0, 2),
        source_frame_indices=frames, correlation=np.ones((4, 2)), cycle_error=np.zeros((4, 2)),
        selected_reference=np.zeros((4, 2), int), reference_age=np.ones((4, 2), int),
        ambiguous=np.zeros((4, 2), bool), blade_occluded=on_blade)
    np.savez(source / 'queries.npz', seed_points=p[0], point_ids=ids, is_core_point=[True, False])
    np.savez(measure / 'measurement_support.npz', supported=good, point_ids=ids,
             frame_indices=frames, motion_step_valid=good)
    np.savez(measure / 'blade_contact_geometry.npz', point_ids=ids, frame_indices=frames,
             on_blade=on_blade, in_image=np.ones((4, 2), bool), material_pixel=~on_blade)
    return dict(name=name, tracks_root='tracks', measurements_root='measurements')


def test_predictions_and_rejected_matches_do_not_become_observations(tmp_path):
    cfg = fixture_case(tmp_path)
    out = tmp_path / 'out'
    result = export_tracks(tmp_path, cfg, out, np.arange(4), 'precontact', .01, np.array([20, 10]))
    with (out / 'tracks.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    by_key = {(int(r['frame_id']), r['original_seed_point_id']): r for r in rows}
    first = by_key[0, '7']
    assert first['valid'] == '1' and first['confidence'] == '' and first['fb_error_px'] == ''
    assert first['quality_source'] == 'seed_initialization'
    forecast = by_key[1, '7']
    assert forecast['valid'] == '0' and forecast['x_px'] == '' and forecast['x_pred_px'] == '11'
    rejected = by_key[2, '9']
    assert rejected['track_status'] == 'occluded' and rejected['valid'] == '0'
    assert rejected['x_px'] == rejected['x_pred_px'] == ''
    assert rejected['x_raw_match_px'] == '32'
    resumed = by_key[2, '7']
    assert resumed['point_id'] == '7:s1' and resumed['parent_point_id'] == '7:s0'
    assert resumed['reference_kind'] == 'deformed' and resumed['reference_frame_id'] == '2'
    assert resumed['ux_image_est_mm'] == '0' and resumed['motion_step_valid'] == '0'
    continuation = by_key[3, '7']
    assert continuation['x_ref_px'] == resumed['x_ref_px']
    assert float(continuation['ux_image_est_mm']) == pytest.approx(.01)
    assert float(continuation['uy_image_est_mm']) == pytest.approx(-.02)
    assert all(r['x_lab_mm'] == r['ux_mm'] == '' for r in rows)
    assert result['exported_rows'] == 8 and result['resumed_segments'] == 2
    assert result['reidentified_observations'] == 0


def test_omitted_gap_still_starts_new_material_segment(tmp_path):
    cfg = fixture_case(tmp_path)
    out = tmp_path / 'out'
    export_tracks(tmp_path, cfg, out, np.array([0, 2, 3]), 'unknown', None, np.array([20, 10]))
    with (out / 'tracks.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    resumed = next(r for r in rows if r['frame_id'] == '2' and r['original_seed_point_id'] == '7')
    assert resumed['point_id'] == '7:s1' and resumed['reference_frame_id'] == '2'
    assert resumed['ux_image_est_mm'] == ''


def test_inconsistent_saved_support_is_rejected(tmp_path):
    cfg = fixture_case(tmp_path)
    path = tmp_path / 'measurements/sample/measurement_support.npz'
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays['supported'][1, 0] = True
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='exclusion rules'):
        export_tracks(tmp_path, cfg, tmp_path / 'out', np.arange(4), 'unknown', None, np.array([20, 10]))


def test_coverage_counts_final_mutually_exclusive_statuses(tmp_path):
    cfg = fixture_case(tmp_path)
    geometry_path = tmp_path / 'measurements/sample/blade_contact_geometry.npz'
    with np.load(geometry_path) as z:
        arrays = {k: z[k] for k in z.files}
    # A forecast may carry both an opaque-blade flag and an out-of-frame flag;
    # the final status precedence is out_of_frame, not two separate outcomes.
    arrays['in_image'][1, 0] = False
    arrays['on_blade'][1, 0] = True
    np.savez(geometry_path, **arrays)
    out = tmp_path / 'out'
    result = export_tracks(tmp_path, cfg, out, np.arange(4), 'unknown', None, np.array([20, 10]))
    with (out / 'coverage.csv').open(encoding='utf-8') as stream:
        coverage = list(csv.DictReader(stream))
    row = next(r for r in coverage if r['frame_id'] == '1' and r['roi_ref'] == 'near_left')
    assert row['out_of_frame_points'] == '1'
    assert row['occluded_points'] == row['rejected_points'] == '0'
    assert row['forecast_points'] == '1'
    assert result['status_counts']['out_of_frame'] == 1
    assert result['roi']['near_left']['occluded_points'] == 0
    assert 'may overlap' in result['coverage_count_definition']


@pytest.fixture
def exported(tmp_path):
    name = 'sample'
    config = fixture_case(tmp_path)
    source, measure = tmp_path/'tracks'/name, tmp_path/'measurements'/name
    frames = np.arange(4)
    out = tmp_path/'exchange'/name
    export_tracks(tmp_path, config, out, frames, 'unknown', .01, np.array([20., 10.]))
    metadata = dict(case_id=name, coordinates=dict(conditional_image_scale_mm_px=.01),
                    provenance=dict(source_paths=dict(tracks=str(source/'tracking/tracks.npz'),
                                                       measurements_dir=str(measure))))
    (out/'metadata.json').write_text(json.dumps(metadata))
    with (out/'frames.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', 'frame_id', 'time_s', 'depth_mm', 'depth_video_estimate_mm',
                         'geometry_supported', 'geometry_interpolated', 'blade_tip_status'])
        for f in frames:
            writer.writerow([name, f, f/12, '', f/10, 1, 0, 'rigid_inferred'])
    (out/'events.csv').write_text('case_id,event,frame_start,frame_end,evidence_kind,notes\n')
    return out


def mutate(out, change):
    path = out/'tracks.csv'
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream); columns = reader.fieldnames; rows = list(reader)
    change(rows)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def test_streaming_reference_gap_status_and_source_checks(exported):
    result = validate_case(exported, chunk_size=3)
    assert result['passed'] and result['track_rows'] == 8
    assert result['observation_segments'] == 4
    assert result['resumed_segments'] == 2
    assert result['coverage_rows'] == 16
    assert result['source_comparison']['sample_rows'] == 8
    assert not result['physical_truth_validated']


def test_gzip_exchange_matches_plain_validation_and_detects_tampering(exported):
    plain = validate_case(exported, chunk_size=3)
    path = exported/'tracks.csv'
    with gzip.open(str(path)+'.gz', 'wb') as stream:
        stream.write(path.read_bytes())
    path.unlink()
    meta = json.loads((exported/'metadata.json').read_text())
    meta['files'] = {'tracks': 'tracks.csv.gz'}
    (exported/'metadata.json').write_text(json.dumps(meta))
    assert validate_case(exported, chunk_size=3) == plain
    with gzip.open(str(path)+'.gz', 'rt') as stream:
        content = stream.read()
    with gzip.open(str(path)+'.gz', 'wt') as stream:
        stream.write(content.replace('0.01', '0.09'))
    with pytest.raises(ValueError):
        validate_case(exported, chunk_size=3)


def test_gzip_export_has_identical_csv_payload(tmp_path):
    config = fixture_case(tmp_path)
    plain, compressed = tmp_path/'plain', tmp_path/'gzip'
    args = (np.arange(4), 'unknown', .01, np.array([20., 10.]))
    a = export_tracks(tmp_path, config, plain, *args)
    b = export_tracks(tmp_path, config, compressed, *args, compress=True)
    assert a == b
    assert gzip.open(compressed/'tracks.csv.gz', 'rb').read() == (plain/'tracks.csv').read_bytes()


def test_exchange_rejects_table_outside_case(exported):
    meta = json.loads((exported/'metadata.json').read_text())
    meta['files'] = {'tracks': '../tracks.csv.gz'}
    (exported/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='within the exchange case'):
        validate_case(exported)


def test_prediction_cannot_appear_in_valid_position_columns(exported):
    mutate(exported, lambda rows: rows[2].update(x_px='11'))
    with pytest.raises(ValueError, match='invalid main coordinates'):
        validate_case(exported, chunk_size=3)


def test_cross_chunk_duplicate_is_rejected(exported):
    mutate(exported, lambda rows: rows.insert(3, rows[2].copy()))
    with pytest.raises(ValueError, match='duplicate frame/point across chunks'):
        validate_case(exported, chunk_size=3)


def test_same_identity_cannot_change_reference(exported):
    mutate(exported, lambda rows: rows[7].update(x_ref_px='31', ux_image_est_mm='.02'))
    with pytest.raises(ValueError, match='reference-frame current point|reference changed'):
        validate_case(exported, chunk_size=3)


def test_source_comparison_catches_consistent_csv_coordinate_tampering(exported):
    mutate(exported, lambda rows: rows[3].update(x_px='31.5', ux_image_est_mm='.015'))
    with pytest.raises(ValueError, match='current coordinates differ from saved source'):
        validate_case(exported, chunk_size=3)


def test_root_writes_failure_report_for_forbidden_missing_value(exported):
    mutate(exported, lambda rows: rows[3].update(confidence='NaN'))
    report = validate_root(exported.parent, ['sample'], chunk_size=3)
    assert not report['passed']
    stored = json.loads((exported.parent/'validation.json').read_text())
    assert stored['cases'][0]['error_type'] == 'ValueError'
    assert 'nonfinite' in stored['cases'][0]['error']


def test_resumed_measurement_cannot_silently_continue_initial_identity(exported):
    def change(rows):
        for i in (4, 6):
            rows[i].update(point_id='7:s0', reference_frame_id='0', reference_kind='unknown',
                           x_ref_px='10', y_ref_px='10', parent_point_id='',
                           ux_image_est_mm=str((float(rows[i]['x_px'])-10)*.01),
                           uy_image_est_mm=str(-(float(rows[i]['y_px'])-10)*.01))
    mutate(exported, change)
    with pytest.raises(ValueError, match='unsupported gap resumes'):
        validate_case(exported, chunk_size=3)


def test_coverage_must_match_all_track_rows(exported):
    path = exported/'coverage.csv'
    text = path.read_text()
    text = text.replace('sample,0,near_left,1,1,1,1,', 'sample,0,near_left,1,1,0,0,')
    path.write_text(text)
    with pytest.raises(ValueError, match='counts disagree'):
        validate_case(exported, chunk_size=3)


def add_capture_estimate(out, bad=False):
    path = out/'frames.csv'
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames), list(reader)
    fields += ['original_avi_frame_id', 'capture_time_s', 'capture_time_estimate_s']
    for i, row in enumerate(rows):
        avi = (0, 3, 5, 8)[i]
        row.update(original_avi_frame_id=avi, capture_time_s='',
                   capture_time_estimate_s=str(avi/200 + (.005 if bad and i == 2 else 0)))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    metadata_path = out/'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['source'] = dict(capture_rate_status='camera_overlay_supported_CFR_estimate', camera_reported_fps=200)
    metadata_path.write_text(json.dumps(metadata))


def test_estimated_capture_clock_matches_avi_mapping(exported):
    add_capture_estimate(exported)
    result = validate_case(exported, chunk_size=3)
    assert result['frame_source_checks']['all_capture_time_estimates_match_avi_200fps']
    assert not result['frame_source_checks']['exact_capture_clock_validated']


def test_mismatched_estimated_capture_clock_is_rejected(exported):
    add_capture_estimate(exported, bad=True)
    with pytest.raises(ValueError, match='Capture-time estimate differs'):
        validate_case(exported, chunk_size=3)


def test_parallel_validation_merges_success_and_failure_in_requested_order(exported):
    result = validate_root(exported.parent, ['missing', 'sample'], chunk_size=3, workers=2)
    assert not result['passed']
    assert result['workers'] == 2
    assert [r['case_id'] for r in result['cases']] == ['missing', 'sample']
    assert result['cases'][1]['passed']
    assert json.loads((exported.parent/'validation.json').read_text()) == result


def test_parallel_validation_rejects_unsupported_worker_count(exported):
    with pytest.raises(ValueError, match='workers must'):
        validate_root(exported.parent, workers=4)


def test_float32_large_displacement_roundoff_is_bounded_without_hiding_point01_pixel_error():
    # Real failure at n60a1 frame1365/ID757: source dtype operations, followed
    # by nine-significant-digit CSV, must not be mistaken for a bad trajectory.
    xy = np.array([[463.444366, 556.574707]])
    reference = np.array([[466.039825, 254.047806]])
    supplied = np.array([[-.0173636023, -2.02390313]])
    scale = .006689992985514707
    expected = (xy-reference)*[scale, -scale]
    bound = _displacement_roundoff_bound(xy, reference, supplied, scale, np.dtype('float32'))
    assert np.all(np.abs(supplied-expected) <= bound)
    assert bound.max()/scale < .0001
    changed = supplied.copy()
    changed[0, 1] += .01*scale
    changed_bound = _displacement_roundoff_bound(xy, reference, changed, scale, np.dtype('float32'))
    assert np.any(np.abs(changed-expected) > changed_bound)
