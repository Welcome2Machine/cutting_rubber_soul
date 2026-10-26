"""Source/optional evidence must survive export without fabricated measurements."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.exchange import source as exchange
from src.exchange.export import coverage_by_depth, export_frames, selected_indices
from src.exchange.observations import export_optional


@pytest.fixture
def case(tmp_path, monkeypatch):
    name = "0930example"
    source = tmp_path / "data/videos/0930"
    source.mkdir(parents=True)
    video = source / f"{name}.mp4"
    video.write_bytes(b"source fixture; decoder is stubbed")
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    imported = dict(width=100, height=80, frames=6, fps=12, source_header_fps=30,
                    output_sha256=digest, original_dimensions_preserved=True,
                    source={"path": "unavailable_original.avi"})
    video.with_suffix(".json").write_text(json.dumps(imported))
    root = tmp_path / "results/tracks"
    folder = root / name
    folder.mkdir(parents=True)
    frames = np.arange(5)
    source_ids = np.array([0, 3, 5, 8, 10, 13])
    np.savez(source / f"{name}_alignment.npz", source_indices=source_ids, frame_indices=np.arange(6))
    np.savez(folder / "frame_mapping.npz", frame_indices=frames, source_indices=source_ids[:5],
             physical_time_s=source_ids[:5] / 200., physical_acquisition_fps=200.)
    tips = np.column_stack((np.full(6, 10.), np.arange(6) * 4.))
    supported = np.array([True, True, True, False, True, True])
    entry = tips[:, 1] - tips[1, 1]
    np.savez(folder / "geometry.npz", tips_px=tips, supported=supported,
             geometry_interpolated=~supported, depth_mm=entry * .25, entry_px=entry)
    geometry = dict(contact_frame=1, reversal_frame=4, contact_to_reversal_mm=3,
                    contact_to_reversal_px=12, surface_angle_deg=50., surface_tangent=[.6, .8])
    (folder / "geometry.json").write_text(json.dumps(geometry))
    (folder / "plan.json").write_text(json.dumps(dict(name=name, source_sha256=digest,
                                                       display_frames=4, query_start_frame=0)))
    np.savez(folder / "queries.npz", seed_points=np.array([[3., 4.], [5., 7.]]),
             point_ids=np.array([13, 15]), source_frame_indices=frames)
    monkeypatch.setattr(exchange, "_probe_video", lambda path: dict(width_px=100, height_px=80,
                         frame_count=6, fps_nominal=12., probe_method="test fixture"))
    return tmp_path, dict(name=name, tracks_root=str(root), stroke_status="assumed"), folder


def test_clocks_and_conditional_depth_are_not_promoted_to_physical_measurements(case):
    repo, config, _ = case
    result = exchange.inspect_case(repo, config)
    np.testing.assert_allclose(result["time_s"], np.arange(5) / 12.)
    np.testing.assert_allclose(result["avi_header_time_s"], [0, .1, 5/30, 8/30, 10/30])
    assert not np.allclose(result["time_s"], result["legacy_assumed_capture_time_s"])
    assert result["metadata"]["coordinates"]["pixel_to_lab_mapping"] is None
    assert result["metadata"]["source"]["physical_capture_time_verified"] is False
    assert result["metadata"]["depth"]["core_depth_mm_available"] is False
    assert result["metadata"]["reference"]["last_separated_frame"] is None
    assert result["metadata"]["reference"]["first_contact_evidence_frame"] is None
    assert result["metadata"]["blade"]["supported_status"] == "rigid_inferred"
    np.testing.assert_array_equal(result["display_mask"], [True, True, True, True, False])
    assert not result["geometry_supported"][3] and result["geometry_interpolated"][3]
    # Conditional forecast values remain available without becoming observations.
    assert result["depth_estimate_mm"][3] == 2.
    assert result["json_evidence"]["checks"]["tracks_data_read"] is False
    json.dumps(result["metadata"], allow_nan=False)
    json.dumps(result["json_evidence"], allow_nan=False)


def test_stale_source_and_mismatched_frame_mapping_are_rejected(case):
    repo, config, folder = case
    np.savez(folder / "frame_mapping.npz", frame_indices=np.arange(5),
             source_indices=np.array([0, 2, 5, 8, 10]))
    with pytest.raises(ValueError, match="AVI frame mappings disagree"):
        exchange.inspect_case(repo, config)
    video = repo / "data/videos/0930/0930example.mp4"
    video.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="SHA-256"):
        exchange.inspect_case(repo, config)


def test_a_stale_geometry_scale_is_not_silently_recomputed(case):
    repo, config, folder = case
    geometry_path = folder / "geometry.json"
    geometry = json.loads(geometry_path.read_text())
    geometry["contact_to_reversal_px"] = 24.
    geometry_path.write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match="conditional depth disagrees"):
        exchange.inspect_case(repo, config)


def test_later_batch_source_is_taken_from_saved_plan(case):
    repo, config, folder = case
    old = repo / 'data/videos/0930'
    new = repo / 'data/videos/20261014'
    old.rename(new)
    plan_path = folder / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan['video'] = str(new / '0930example.mp4')
    plan_path.write_text(json.dumps(plan))
    result = exchange.inspect_case(repo, config)
    assert result['paths']['source_video'] == str(new / '0930example.mp4')
    assert result['json_evidence']['checks']['source_mapping_matches_import']


def _frame_info():
    frames = np.arange(4)
    return dict(case_id='0930l30a1', frames=frames, contact_frame=1, reversal_frame=3,
        display_mask=np.ones(4, bool), tips=np.array([[10., 20.], [10., 21.],
                                                    [np.nan, np.nan], [10., 25.]]),
        geometry_supported=np.array([1, 1, 0, 0], bool),
        geometry_interpolated=np.array([0, 0, 0, 1], bool),
        time_s=frames / 12, avi_frames=np.array([0, 3, 5, 8]),
        avi_header_time_s=np.array([0., .1, 1/6, 8/30]),
        legacy_assumed_capture_time_s=np.array([0., .015, .025, .04]),
        depth_estimate_mm=np.array([-.1, 0., np.nan, 1.2]),
        entry_px=np.array([-1., 0., np.nan, 4.]))


def _read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def test_unknown_tip_depth_and_onset_are_not_fabricated(tmp_path):
    info = _frame_info()
    export_frames(info, np.arange(4), tmp_path)
    rows = _read(tmp_path / 'frames.csv')
    missing = rows[2]
    assert missing['blade_tip_status'] == 'unavailable'
    assert missing['blade_tip_x_px'] == missing['blade_tip_y_px'] == ''
    assert missing['depth_video_estimate_mm'] == missing['blade_entry_px'] == ''
    assert missing['within_user_analysis_cutoff'] == ''
    assert all(row['depth_mm'] == row['depth_uncertainty_mm'] == '' for row in rows)
    assert all(row['fracture_state'] == 'unknown' for row in rows)
    assert rows[1]['blade_tip_status'] == 'rigid_inferred'
    assert rows[3]['blade_tip_status'] == 'predicted'
    assert rows[1]['within_user_analysis_cutoff'] == '1'
    assert rows[3]['within_user_analysis_cutoff'] == '0'
    assert all(row['capture_time_s'] == '' for row in rows)
    events = _read(tmp_path / 'events.csv')
    assert {event['event'] for event in events} == {'first_contact', 'loading_end'}
    assert all(event['evidence_kind'] in {'visible', 'manual', 'inferred'} for event in events)
    assert all(event['frame_start'] == event['frame_end'] == '' for event in events)
    assert all(event['estimation_method'] for event in events)
    assert not any('fracture' in event['event'] for event in events)


def test_missing_depth_has_coverage_bin_and_keeps_observation_counts(tmp_path):
    export_frames(_frame_info(), np.arange(4), tmp_path)
    with (tmp_path / 'coverage.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame_id', 'roi_ref', 'valid_points', 'total_points'])
        writer.writeheader()
        writer.writerows([
            dict(frame_id=0, roi_ref='near_left', valid_points=4, total_points=5),
            dict(frame_id=1, roi_ref='near_left', valid_points=3, total_points=5),
            dict(frame_id=2, roi_ref='near_left', valid_points=2, total_points=5),
            dict(frame_id=2, roi_ref='far_right', valid_points=7, total_points=8),
            dict(frame_id=3, roi_ref='near_left', valid_points=1, total_points=5),
        ])
    bins = coverage_by_depth(tmp_path)
    indexed = {(row['depth_bin'], row['roi_ref']): row for row in bins}
    unknown_near = indexed['unknown depth', 'near_left']
    assert unknown_near['valid_points'] == 2 and unknown_near['total_points'] == 5
    assert unknown_near['valid_fraction'] == pytest.approx(.4)
    assert indexed['unknown depth', 'far_right']['valid_fraction'] == pytest.approx(7/8)
    assert indexed['unknown depth', 'near_right']['valid_fraction'] is None
    assert sum(row['total_points'] for row in bins) == 28
    assert sum(row['valid_points'] for row in bins) == 17


def test_stop_depth_takes_first_crossing_prefix_and_never_picks_later_returns():
    info = dict(frames=np.arange(8), display_mask=np.ones(8, bool), contact_frame=1,
        reversal_frame=7, depth_estimate_mm=np.array([-.1, 0., .5, np.nan, 1.01, .9, 1.3, .8]))
    selected = selected_indices(info, stop_mm=1.)
    np.testing.assert_array_equal(selected, np.arange(5))
    # The unknown middle frame is retained, and the observed first crossing
    # is included. Later below-threshold frames do not re-enter the prefix.
    assert 3 in selected and 4 in selected and 5 not in selected and 7 not in selected
    np.testing.assert_array_equal(selected_indices(info), np.arange(8))


def test_no_crossing_does_not_invent_endpoint_and_reversal_limits_export():
    info = dict(frames=np.arange(6), display_mask=np.ones(6, bool), contact_frame=1,
        reversal_frame=3, depth_estimate_mm=np.array([-.2, 0., np.nan, .4, 1.2, 1.5]))
    np.testing.assert_array_equal(selected_indices(info, stop_mm=1.), np.arange(4))


def test_force_keeps_sensor_clock_units_and_all_samples(tmp_path):
    directory = tmp_path / "data/force/0930"
    directory.mkdir(parents=True)
    content = '\n'.join([
        '#BeginHeader,6', 'サンプリング周期,2ms', 'データ数,3',
        'チャンネル番号,,(1)HA-V01,(1)HA-V02,(1)HA-V03', '単位,,N,N,V',
        '#EndHeader,日時(μs),V01,V02,V03',
        '0日 00時00分00秒,998000,1.0,-2.0,0.3',
        '0日 00時00分01秒,000000,1.1,-2.1,0.4',
        '0日 00時00分01秒,002000,1.2,-2.2,0.5',
        '#BeginMark,3', 'no,data,after,marker,allowed', '#EndMark'])
    source = directory / "test.CSV"
    source.write_bytes(content.encode('cp932'))
    out = tmp_path / "exchange"
    result = export_optional(tmp_path, "test", out, np.array([0], dtype=int))
    rows = _read(out / "force_channels.csv")
    assert len(rows) == 3
    assert [float(row['time_force_s']) for row in rows] == [.998, 1.0, 1.002]
    assert rows[1]['sensor_microseconds_text'] == '000000'
    assert rows[1]['V02_N'] == '-2.1'
    assert 'Fz_N' not in rows[0]
    assert result['force']['all_intervals_2ms']
    assert result['force']['video_synchronization'] is None
    assert (out / 'source_evidence/force/test.CSV').read_bytes() == source.read_bytes()
    json.dumps(result, allow_nan=False)


def test_contours_split_at_state_occlusion_crop_and_bad_edge(tmp_path):
    directory = tmp_path / 'results/20261008_named_dual_contact/test'
    directory.mkdir(parents=True)
    # State transition, invalid edge, crop and uncertain blade band must all
    # prevent a single apparently observed polyline through the missing area.
    states = np.array([2, 2, 1, 1, 1, 0, 2, 3, 2])
    edges = np.ones(9, bool)
    edges[2] = False
    np.savez(directory / 'geometry.npz', frames=[5], closed_offsets=[0, 9],
             closed_xy=np.column_stack([np.arange(9), np.zeros(9)]),
             closed_state=states, closed_edge_valid=edges)
    out = tmp_path / 'exchange'
    result = export_optional(tmp_path, 'test', out, np.array([0, 5]))
    rows = _read(out / 'contours.csv')
    assert [int(r['source_vertex_index']) for r in rows] == [0, 1, 2, 3, 4, 6, 7, 8]
    groups = [[int(r['source_vertex_index']) for r in rows if r['component_id'] == c]
              for c in dict.fromkeys(r['component_id'] for r in rows)]
    assert groups == [[0, 1], [2], [3, 4], [6], [7], [8]]
    assert all(not r['point_id'] and not r['reference_s_rel_mm'] for r in rows)
    assert rows[-2]['visibility'] == 'uncertain'
    assert result['contours']['components_are_open']


def test_model_data_filtered_without_claiming_observed_identity(tmp_path):
    directory = tmp_path / 'results/20261008_named_dual_contact/test'
    directory.mkdir(parents=True)
    np.savez(directory / 'MMT-v1.npz', frames=[3, 5, 7], method_version='v1',
             method_identifier='MMT-v1', depth_mm=[.1, .2, .3],
             offset_um=[10., 20., 30.], sigma_um=[5., 8., 12.],
             transport_prediction_only=[False, True, False],
             transport_absolute_used=[True, False, False],
             transport_velocity_used=[False, False, True])
    out = tmp_path / 'exchange'
    result = export_optional(tmp_path, 'test', out, np.array([0, 3, 5]))
    rows = _read(out / 'model_contact_estimates.csv')
    assert [r['frame_id'] for r in rows] == ['3', '5']
    assert rows[0]['method_name'] == 'Mixed-model transport'
    assert 'depth_mm' not in rows[0]
    assert [float(r['depth_video_estimate_mm']) for r in rows] == [.1, .2]
    assert rows[1]['estimate_status'] == 'prediction_only'
    assert all(r['contact_confirmed'] == '0' and not r['point_id'] for r in rows)
    assert result['counts']['model_contact_estimates'] == 2
    assert result['contact_models']['methods'][0]['prediction_only_rows'] == 1
    json.dumps(result, allow_nan=False)
