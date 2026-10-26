"""Export saved material observations without promoting forecasts or gap identities.

The underlying tracker is unchanged. Exchange IDs identify continuous observed
segments, while ``original_seed_point_id`` records the tracker's unverified
cross-gap identity. A resumed segment begins a new deformed reference.
"""
from __future__ import annotations

import csv
import gzip
import math
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from src.io import sha as _sha256


TRACK_COLUMNS = (
    'case_id', 'frame_id', 'point_id', 'reference_frame_id', 'reference_kind',
    'x_ref_px', 'y_ref_px', 'x_px', 'y_px', 'point_kind', 'track_status', 'valid',
    'confidence', 'fb_error_px', 'position_uncertainty_px', 'invalid_reason',
    'x_pred_px', 'y_pred_px', 'raw_tracking_supported', 'measurement_supported',
    'original_seed_point_id', 'original_seed_x_px', 'original_seed_y_px',
    'parent_point_id', 'quality_source', 'roi_ref', 'roi_current',
    'x_raw_match_px', 'y_raw_match_px', 'ux_image_est_mm', 'uy_image_est_mm',
    'x_ref_lab_mm', 'y_ref_lab_mm', 'x_lab_mm', 'y_lab_mm', 'ux_mm', 'uy_mm',
    'selected_reference', 'reference_age_steps', 'motion_step_valid',
)
COVERAGE_COLUMNS = (
    'case_id', 'frame_id', 'roi_ref', 'total_points', 'raw_supported_points',
    'valid_points', 'valid_fraction', 'forecast_points', 'occluded_points',
    'rejected_points', 'out_of_frame_points', 'new_observation_segments',
    'reidentified_points',
)


def _load(path, keys):
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in keys}


def _root(repo, value):
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _number(value):
    value = float(value)
    return format(value, '.9g') if math.isfinite(value) else ''


def export_tracks(repo: Path, case_config: dict, out: Path,
                  frame_ids: np.ndarray, reference_kind: str,
                  scale_mm_px: float | None, initial_contact_xy: np.ndarray, *, compress=False) -> dict:
    """Write all fixed spatial samples for the requested saved source frames.

    ``frame_ids`` addresses the selected source MP4, not the acquisition AVI.
    ``valid`` means saved local image support plus saved semantic exclusions;
    it does not establish initial material identity, mechanical contact, or a
    calibrated physical error. No coordinate is smoothed or corrected here.
    """
    repo, out = Path(repo), Path(out)
    name = case_config['name']
    source = _root(repo, case_config['tracks_root']) / name
    measure = _root(repo, case_config['measurements_root']) / name
    # NPZ is compressed per-array. Do not decompress large affine/nearest-edge
    # arrays which are not used by this observational export.
    tracks = _load(source / 'tracking/tracks.npz', (
        'seed_points', 'point_ids', 'positions', 'supported', 'source_frame_indices',
        'visible', 'position_kind', 'blade_occluded', 'ambiguous', 'correlation',
        'cycle_error', 'selected_reference', 'reference_age'))
    query = _load(source / 'queries.npz', ('seed_points', 'point_ids', 'is_core_point'))
    measured = _load(measure / 'measurement_support.npz', (
        'supported', 'frame_indices', 'point_ids', 'motion_step_valid'))
    geometry = _load(measure / 'blade_contact_geometry.npz', (
        'point_ids', 'frame_indices', 'in_image', 'material_pixel', 'on_blade'))
    requested = np.asarray(frame_ids)
    if (requested.ndim != 1 or not len(requested)
            or not np.issubdtype(requested.dtype, np.integer)
            or requested[0] != 0 or np.any(np.diff(requested) <= 0)):
        raise ValueError('frame_ids must start at reference frame 0 and increase strictly')
    if reference_kind not in ('precontact', 'deformed', 'unknown'):
        raise ValueError('Unknown reference_kind')
    if scale_mm_px is not None and (not np.isfinite(scale_mm_px) or scale_mm_px <= 0):
        raise ValueError('scale_mm_px must be positive, finite, or None')
    seeds = np.asarray(tracks['seed_points'])
    ids = tracks['point_ids']
    p = tracks['positions']
    raw = tracks['supported']
    mapping = tracks['source_frame_indices']
    support = measured['supported']
    m_frames = measured['frame_indices']
    n = len(ids)
    if (seeds.shape != (n, 2) or p.shape != (len(mapping), n, 2)
            or raw.shape != p.shape[:2] or not np.isfinite(seeds).all()
            or not np.isfinite(p).all() or len(np.unique(ids)) != n):
        raise ValueError('Invalid track layout or material IDs')
    for other in (query, measured, geometry):
        if not np.array_equal(ids, other['point_ids']):
            raise ValueError('Point IDs differ across saved sources')
    if not np.array_equal(seeds, query['seed_points']):
        raise ValueError('Seed coordinates differ across saved sources')
    if not np.array_equal(p[0], seeds):
        raise ValueError('First positions are not the declared reference seeds')
    if not np.array_equal(m_frames, geometry['frame_indices']):
        raise ValueError('Geometry and measurement frame mappings disagree')
    track_row = {int(frame): i for i, frame in enumerate(mapping)}
    measure_row = {int(frame): i for i, frame in enumerate(m_frames)}
    if not all(int(frame) in track_row and int(frame) in measure_row for frame in requested):
        raise ValueError('Requested frames missing from saved track or measurement arrays')
    if support.shape != (len(m_frames), n):
        raise ValueError('Invalid measurement support shape')
    aligned = np.asarray([track_row[int(f)] for f in m_frames])
    expected_support = (raw[aligned] & geometry['in_image']
                        & geometry['material_pixel'] & ~geometry['on_blade'])
    if not np.array_equal(support, expected_support):
        raise ValueError('Saved measurement support disagrees with declared exclusion rules')
    if not np.array_equal(tracks['visible'], raw):
        raise ValueError('Visible and supported flags disagree')
    if np.any((tracks['position_kind'] == 0) != raw):
        raise ValueError('Position kind and supported flags disagree')

    contact = np.asarray(initial_contact_xy, float)
    if contact.shape != (2,) or not np.isfinite(contact).all():
        raise ValueError('Initial contact position required for fixed ROI grouping')
    core = np.asarray(query['is_core_point'], bool)
    left = seeds[:, 0] < contact[0]
    roi = np.where(core, np.where(left, 'near_left', 'near_right'),
                   np.where(left, 'far_left', 'far_right'))
    roi_names = ('near_left', 'near_right', 'far_left', 'far_right')
    roi_masks = {label: roi == label for label in roi_names}
    roi_text = roi.tolist()
    original_ids = [str(int(i)) for i in ids]
    seed_text = [(_number(x), _number(y)) for x, y in seeds]
    reference_text = seed_text.copy()
    segment_number = np.zeros(n, np.int32)
    segment_ids = [f'{pid}:s0' for pid in original_ids]
    parent_ids = [''] * n
    reference_frame = np.zeros(n, np.int64)
    reference_xy = seeds.copy()
    reference_kinds = np.full(n, reference_kind, dtype=object)
    previous_good = support[measure_row[0]].copy()
    previous_frame = 0
    totals = Counter()
    status_counts = Counter()
    reason_counts = Counter()
    roi_stats = {label: Counter() for label in roi_names}
    fb_values = []
    all_requested = set(map(int, requested))
    out.mkdir(parents=True, exist_ok=True)
    track_path = out / ('tracks.csv.gz' if compress else 'tracks.csv')
    track_stream = gzip.open(track_path, 'wt', encoding='utf-8', newline='', compresslevel=1) if compress else \
        track_path.open('w', encoding='utf-8', newline='', buffering=2**20)
    with track_stream as stream, \
            (out / 'coverage.csv').open('w', encoding='utf-8', newline='') as coverage:
        writer = csv.writer(stream)
        cov_writer = csv.writer(coverage)
        writer.writerow(TRACK_COLUMNS)
        cov_writer.writerow(COVERAGE_COLUMNS)
        for frame_value in requested:
            frame = int(frame_value)
            tr, mr = track_row[frame], measure_row[frame]
            position = p[tr]
            raw_good = raw[tr]
            good = support[mr]
            # Include missing observations in omitted frames when deciding
            # continuity, without inventing rows at unrequested frames.
            if frame == 0:
                restarted = np.zeros(n, bool)
            else:
                intervening = (m_frames >= previous_frame) & (m_frames < frame)
                continuous = support[intervening].all(0) if intervening.any() else previous_good
                restarted = good & ~continuous
            for i in np.flatnonzero(restarted):
                parent_ids[i] = segment_ids[i]
                segment_number[i] += 1
                segment_ids[i] = f'{original_ids[i]}:s{segment_number[i]}'
                reference_frame[i] = frame
                reference_xy[i] = position[i]
                reference_text[i] = (_number(position[i, 0]), _number(position[i, 1]))
                reference_kinds[i] = 'deformed'
            forecast = ~raw_good
            hidden = ~good & (geometry['on_blade'][mr] | tracks['blade_occluded'][tr])
            outside = ~good & ~geometry['in_image'][mr]
            rejected = ~good & ~hidden & ~outside & raw_good
            statuses = np.full(n, 'tracked', dtype=object)
            statuses[forecast] = 'predicted'
            statuses[rejected] = 'rejected'
            statuses[hidden] = 'occluded'
            statuses[outside] = 'out_of_frame'
            reasons = np.full(n, '', dtype=object)
            reasons[forecast] = 'local_image_checks_failed'
            reasons[forecast & tracks['ambiguous'][tr]] = 'ambiguous_image_match'
            reasons[rejected] = 'outside_semantic_material'
            reasons[hidden] = 'blade_occlusion'
            reasons[outside] = 'out_of_frame'
            edge = measured['motion_step_valid'][mr] & good & ~restarted
            if frame == 0:
                edge[:] = False
            quality_source = 'seed_initialization' if frame == 0 else 'local_image_checks'
            correlation = tracks['correlation'][tr]
            cycle_error = tracks['cycle_error'][tr]
            if frame > 0:
                fb_values.append(cycle_error[good & np.isfinite(cycle_error)])
            displacement = position - reference_xy
            # Match NumPy scalar promotion used by the original per-point
            # calculation, including environments with NumPy 1.x promotion.
            if scale_mm_px is not None:
                scaled_dtype = np.asarray(displacement[0, 0] * scale_mm_px).dtype
                ux_values = displacement[:, 0].astype(scaled_dtype) * scale_mm_px
                uy_values = (-displacement[:, 1]).astype(scaled_dtype) * scale_mm_px
            selected_reference = tracks['selected_reference'][tr].tolist()
            reference_age = tracks['reference_age'][tr].tolist()
            rows = []
            for i in range(n):
                valid = bool(good[i])
                observed = bool(raw_good[i])
                quality_available = frame > 0 and observed
                conf = _number(correlation[i]) if quality_available else ''
                fb = _number(cycle_error[i]) if frame > 0 else ''
                x, y = (_number(position[i, 0]), _number(position[i, 1]))
                xp, yp = (x, y) if forecast[i] else ('', '')
                rawx, rawy = (x, y) if observed and not valid else ('', '')
                ux = _number(ux_values[i]) if valid and scale_mm_px is not None else ''
                uy = _number(uy_values[i]) if valid and scale_mm_px is not None else ''
                row = (name, frame, segment_ids[i], int(reference_frame[i]), reference_kinds[i],
                    *reference_text[i],
                    x if valid else '', y if valid else '', 'rubber_face', statuses[i], int(valid),
                    conf, fb, '', reasons[i], xp, yp, int(observed), int(valid),
                    original_ids[i], *seed_text[i],
                    parent_ids[i], quality_source, roi_text[i], '', rawx, rawy, ux, uy,
                    '', '', '', '', '', '',
                    selected_reference[i] if frame > 0 else '',
                    reference_age[i] if frame > 0 else '', int(edge[i]))
                rows.append(row)
            writer.writerows(rows)
            totals.update(rows=n, valid=int(good.sum()), raw_supported=int(raw_good.sum()),
                          forecast=int(forecast.sum()), new_observation_segments=int(restarted.sum()))
            status_counts.update(statuses.tolist())
            reason_counts.update(reason for reason in reasons.tolist() if reason)
            for label, mask in roi_masks.items():
                total, accepted = int(mask.sum()), int((good & mask).sum())
                counts = dict(total_points=total, raw_supported_points=int((raw_good & mask).sum()),
                    valid_points=accepted, forecast_points=int((forecast & mask).sum()),
                    occluded_points=int(((statuses == 'occluded') & mask).sum()),
                    rejected_points=int(((statuses == 'rejected') & mask).sum()),
                    out_of_frame_points=int(((statuses == 'out_of_frame') & mask).sum()),
                    new_observation_segments=int((restarted & mask).sum()), reidentified_points=0)
                roi_stats[label].update(counts)
                cov_writer.writerow((name, frame, label, total, counts['raw_supported_points'], accepted,
                    _number(accepted / total) if total else '', counts['forecast_points'],
                    counts['occluded_points'], counts['rejected_points'], counts['out_of_frame_points'],
                    counts['new_observation_segments'], 0))
            previous_good, previous_frame = good.copy(), frame

    roi_info = {}
    for label, mask in roi_masks.items():
        bounds = ([seeds[mask].min(0).tolist(), seeds[mask].max(0).tolist()] if mask.any() else None)
        counts = dict(roi_stats[label])
        counts['valid_fraction'] = counts['valid_points'] / counts['total_points'] if counts['total_points'] else None
        roi_info[label] = dict(initial_point_count=int(mask.sum()), seed_bounds_px=bounds, **counts)
    fb_array = np.concatenate(fb_values) if fb_values else np.empty(0)
    return dict(case_id=name, frame_count=len(requested), initial_point_count=n,
        exported_rows=totals['rows'], valid_observations=totals['valid'],
        raw_supported_observations=totals['raw_supported'], forecast_rows=totals['forecast'],
        observation_segment_count=n+totals['new_observation_segments'],
        resumed_segments=totals['new_observation_segments'], reidentified_observations=0,
        valid_fraction=totals['valid'] / totals['rows'], status_counts=dict(status_counts),
        invalid_reason_counts=dict(reason_counts), roi=roi_info,
        fb_error_px_quantiles=(dict(zip(('median', 'p95', 'maximum'),
            map(float, np.quantile(fb_array, [.5, .95, 1.])))) if len(fb_array) else None),
        position_uncertainty_assessed=False, physical_identity_validated=False,
        trajectories_changed=False, source_frame_sampling='all supplied frames, no point subsampling',
        original_reference_frame_id=0,
        tracking_columns=list(TRACK_COLUMNS), coverage_columns=list(COVERAGE_COLUMNS),
        coverage_count_definition='occluded_points, rejected_points and out_of_frame_points count mutually exclusive final track_status values. forecast_points counts raw_tracking_supported=0 and may overlap those status counts; it must not be added to them as a partition.',
        roi_definition='Fixed initial seed x versus initial contact x; near=original dense core, far=appended sparse coverage. Not current contact-side labels.',
        segment_definition='Exchange IDs denote continuous measurement-supported segments. After any unsupported interval a new deformed reference starts. Parent IDs record bookkeeping only; original seed identity remains unverified.',
        reference_frames_present=bool(all(int(f) in all_requested for f in reference_frame)))


FORCE_COLUMNS = [
    "case_id", "sample_id", "time_force_s", "sensor_time_text",
    "sensor_microseconds_text", "V01_N", "V02_N", "V03_V", "valid", "invalid_reason",
]
MODEL_COLUMNS = [
    "case_id", "frame_id", "method_name", "method_name_zh", "method_version",
    "method_identifier", "depth_video_estimate_mm", "offset_um", "sensitivity_sigma_um",
    "reference_s_px", "baseline_s_px", "um_per_px", "estimate_status",
    "prediction_only", "absolute_position_valid", "absolute_position_used", "transport_rate_used",
    "side_count", "side_disagreement", "side_incompatible",
    "left_rate_supported", "right_rate_supported", "left_rate_used", "right_rate_used", "left_support_count",
    "right_support_count", "left_surface_gap_px", "right_surface_gap_px",
    "left_model", "right_model", "left_reason", "right_reason",
    "point_id", "evidence", "contact_confirmed",
]
CONTOUR_COLUMNS = [
    "case_id", "frame_id", "contour_id", "component_id", "vertex_order",
    "contour_kind", "x_px", "y_px", "visibility", "evidence", "point_id",
    "reference_s_rel_mm", "source_vertex_index", "source_state", "reason",
]




def _evidence(repo: Path, source: Path, out: Path, result: dict, label: str) -> None:
    """Copy a saved source and retain a source-relative path plus digest."""
    destination = out / "source_evidence" / label / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    result["sources_sha256"][source.relative_to(repo).as_posix()] = _sha256(source)
    result["evidence_files"].append(destination.relative_to(out).as_posix())


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else ""
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_row(writer, row):
    writer.writerow({key: _cell(value) for key, value in row.items()})


def _force(repo: Path, case_id: str, out: Path, result: dict) -> None:
    source = repo / "data" / "force" / "0930" / f"{case_id}.CSV"
    if not source.exists():
        result["missing_items"].append("force_channels: no saved sensor CSV for this case")
        return
    lines = source.read_text(encoding="cp932").splitlines()
    end_header = next((i for i, line in enumerate(lines) if line.startswith("#EndHeader")), None)
    if end_header is None:
        raise ValueError(f"Missing NR-600 header boundary: {source}")
    header = list(csv.reader(lines[:end_header + 1]))
    units = next((row[2:5] for row in header if row and row[0] == "単位"), None)
    if units != ["N", "N", "V"]:
        raise ValueError(f"Unknown force channel units in {source}: {units!r}")
    channels = next((row[2:5] for row in header if row and row[0] == "チャンネル番号"), None)
    if channels != ["(1)HA-V01", "(1)HA-V02", "(1)HA-V03"]:
        raise ValueError(f"Unknown force channel order in {source}: {channels!r}")
    instrument = {row[0]: row[1:] for row in header if row}
    destination = out / "force_channels.csv"
    times, invalid = [], 0
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FORCE_COLUMNS)
        writer.writeheader()
        for record in csv.reader(lines[end_header + 1:]):
            if not record or not record[0].strip():
                continue
            if record[0].startswith("#"):
                break
            if len(record) != 5:
                raise ValueError(f"Unexpected sensor row shape in {source}: {record!r}")
            components = re.fullmatch(r"(\d+)日\s*(\d+)時(\d+)分(\d+)秒", record[0])
            if components is None:
                raise ValueError(f"Unrecognised sensor timestamp in {source}: {record[0]!r}")
            day, hour, minute, second = map(int, components.groups())
            if not record[1].isdigit() or not 0 <= int(record[1]) < 1000000:
                raise ValueError(f"Unrecognised microseconds in {source}: {record[1]!r}")
            time = day * 86400 + hour * 3600 + minute * 60 + second + int(record[1]) / 1e6
            reasons = []
            if times and time <= times[-1]:
                reasons.append("non_increasing_sensor_time")
            values = []
            for label, value in zip(("V01", "V02", "V03"), record[2:]):
                try:
                    number = float(value)
                except ValueError:
                    number = np.nan
                values.append(number)
                if not np.isfinite(number):
                    reasons.append(f"nonfinite_{label}")
            row = dict(case_id=case_id, sample_id=len(times), time_force_s=time,
                       sensor_time_text=record[0], sensor_microseconds_text=record[1],
                       V01_N=values[0], V02_N=values[1], V03_V=values[2],
                       valid=not reasons, invalid_reason=";".join(reasons))
            _write_row(writer, row)
            times.append(time)
            invalid += bool(reasons)
    _evidence(repo, source, out, result, "force")
    manifest = source.parent / "import_manifest.json"
    if manifest.exists():
        _evidence(repo, manifest, out, result, "force")
    dt = np.diff(times)
    declared = instrument.get("データ数", [None])[0]
    declared = int(declared) if declared is not None else None
    result["files"]["force_channels"] = destination.name
    result["counts"]["force_samples"] = len(times)
    result["force"] = dict(
        sample_count=len(times), declared_sample_count=declared,
        declared_count_matches=declared == len(times), invalid_sample_count=int(invalid),
        first_sensor_time_s=times[0] if times else None, last_sensor_time_s=times[-1] if times else None,
        strictly_increasing_time=bool(len(times) > 0 and np.all(dt > 0)),
        interval_min_s=float(dt.min()) if dt.size else None,
        interval_max_s=float(dt.max()) if dt.size else None,
        all_intervals_2ms=bool(dt.size > 0 and np.allclose(dt, .002, rtol=0, atol=1e-9)),
        time_source="NR-600 stored relative sensor clock; day/hour/minute/second plus microseconds",
        nominal_sampling_period=instrument.get("サンプリング周期"),
        actual_sampling_period=instrument.get("実サンプリング周期"),
        collection_start_text=instrument.get("収集開始日時"),
        axis_mapping=None, video_synchronization=None, Fz_available=False,
        channel_units=dict(V01="N", V02="N", V03="V"),
        instrument_processing={key: instrument.get(key) for key in (
            "計測機能設定", "計測機能", "スムージング", "ポイント数", "スケーリング",
            "計測値1", "計測値2", "表示値1", "表示値2")},
        notes=["Recorded channel output is preserved; no additional filtering, taring or sign change.",
               "Instrument header declares averaging and smoothing; these are not unfiltered sensor voltages.",
               "V01/V02 force axes and polarity are unconfirmed. V03 purpose and machine-depth calibration are unknown.",
               "Sensor time has not been synchronized with video; it must not be joined to video by sample number."])
    result["columns"]["force_channels"] = dict(
        time_force_s="Seconds on the original sensor relative clock, independent of video.",
        sample_id="Zero-based original data-row number; no temporal resampling.",
        sensor_time_text="Original relative day/hour/minute/second text.",
        sensor_microseconds_text="Original subsecond text; leading zeroes retained in CSV.",
        V01_N="Recorded V01 in N; axis and sign unconfirmed.",
        V02_N="Recorded V02 in N; axis and sign unconfirmed.",
        V03_V="Recorded V03 in V; purpose unconfirmed.",
        valid="Finite channel values and increasing sensor time; does not establish axis identity or synchronization.")
    result["missing_items"].extend([
        "Fz mapping/polarity/tare: force channel axes are not confirmed",
        "Force-video synchronization and machine-depth mapping are not established"])


def _model_value(arrays, key, index, side=None):
    if key not in arrays:
        return None
    value = arrays[key]
    if value.ndim:
        value = value[index]
    if side is not None:
        value = value[side]
    return value.item() if np.ndim(value) == 0 else value


def _models(repo: Path, case_id: str, out: Path, frame_ids: set[int], result: dict) -> None:
    directory = repo / "results" / "20261008_named_dual_contact" / case_id
    sources = [directory / f"{name}.npz" for name in ("MMT-v1", "OBT-v1")]
    sources = [source for source in sources if source.exists()]
    if not sources:
        result["missing_items"].append("named_model_contact_estimates: no saved dual-method result for this case")
        return
    destination = out / "model_contact_estimates.csv"
    total = 0
    methods = []
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MODEL_COLUMNS)
        writer.writeheader()
        for source in sources:
            report_path = source.with_suffix(".json")
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            name = report.get("method", {})
            with np.load(source, allow_pickle=False) as arrays:
                frames = arrays["frames"]
                if len(np.unique(frames)) != len(frames):
                    raise ValueError(f"Duplicate model frame ID in {source}")
                count = predictions = 0
                selected_frames = []
                for i, frame in enumerate(frames):
                    if int(frame) not in frame_ids:
                        continue
                    prediction = bool(arrays["transport_prediction_only"][i])
                    row = dict(case_id=case_id, frame_id=int(frame),
                        method_name=name.get("english_name", "Mixed-model transport" if source.stem.startswith("MMT") else "Objective boundary transport"),
                        method_name_zh=name.get("chinese_name", "混合模型输运法" if source.stem.startswith("MMT") else "客观边界输运法"),
                        method_version=str(arrays["method_version"]), method_identifier=str(arrays["method_identifier"]),
                        depth_video_estimate_mm=_model_value(arrays, "depth_mm", i), offset_um=_model_value(arrays, "offset_um", i),
                        sensitivity_sigma_um=_model_value(arrays, "sigma_um", i),
                        reference_s_px=_model_value(arrays, "reference_s_px", i), baseline_s_px=_model_value(arrays, "baseline_s_px", i),
                        um_per_px=_model_value(arrays, "um_per_px", i), estimate_status="prediction_only" if prediction else "conditional_model_estimate",
                        prediction_only=prediction, absolute_position_valid=_model_value(arrays, "absolute_valid", i),
                        absolute_position_used=_model_value(arrays, "transport_absolute_used", i),
                        transport_rate_used=_model_value(arrays, "transport_velocity_used", i),
                        side_count=_model_value(arrays, "transport_side_count", i),
                        side_disagreement=_model_value(arrays, "transport_side_disagreement", i),
                        side_incompatible=_model_value(arrays, "transport_side_incompatible", i),
                        point_id="", evidence="saved_material_motion_model_inference", contact_confirmed=0)
                    for side, side_name in enumerate(("left", "right")):
                        for column, key in (("rate_supported", "side_flux_valid"), ("rate_used", "transport_side_used"), ("support_count", "side_support_count"),
                                            ("surface_gap_px", "side_surface_gap_px"), ("model", "side_observation_model"),
                                            ("reason", "side_reason")):
                            row[f"{side_name}_{column}"] = _model_value(arrays, key, i, side)
                    _write_row(writer, row)
                    count += 1
                    predictions += prediction
                    selected_frames.append(int(frame))
                methods.append(dict(method_identifier=str(arrays["method_identifier"]), exported_rows=count,
                    prediction_only_rows=int(predictions), source_rows=len(frames),
                    first_exported_frame=min(selected_frames) if selected_frames else None,
                    last_exported_frame=max(selected_frames) if selected_frames else None))
                total += count
            _evidence(repo, source, out, result, "contact_models")
            if report_path.exists():
                _evidence(repo, report_path, out, result, "contact_models")
    result["files"]["model_contact_estimates"] = destination.name
    result["counts"]["model_contact_estimates"] = total
    result["contact_models"] = dict(methods=methods, contact_confirmed=False,
        notes=["All rows are conditional estimates of original-surface material address at the registered blade tip.",
               "Neither position support nor a geometrically small gap confirms physical contact or material identity.",
               "Sensitivity sigma is a model sensitivity scale, not a calibrated experimental confidence interval; zero at the imposed first-contact origin does not imply zero physical measurement error.",
               "Saved temporal fusion may use later samples within the original declared cutoff; frame filtering does not recompute fusion.",
               "The analysis cutoff was supplied externally; it is not an automatically measured fracture time.",
               "The two methods remain separate, with their saved qualification behavior and no winner selected."])
    result["columns"]["model_contact_estimates"] = dict(
        depth_video_estimate_mm="Copied unchanged from the saved model NPZ depth_mm array: registered image-vertical blade travel since proximity-based contact zero, scaled by approximate contact-to-reversal stroke. Conditional video estimate, not independently registered fixed-base laboratory depth.",
        offset_um="Saved inferred initial-surface material-address change relative to first contact, micrometres.",
        sensitivity_sigma_um="Saved model sensitivity scale in micrometres; not calibrated physical uncertainty.",
        reference_s_px="Saved estimated initial-surface coordinate, in reference pixel arc-coordinate units.",
        baseline_s_px="Saved fixed initial-contact reference coordinate in the same units.",
        um_per_px="Saved approximate contact-to-reversal stroke-based scale, not new calibration.",
        prediction_only="1 if neither saved position nor transport-rate observation was used at this sample.",
        absolute_position_valid="Saved model position qualification flag, not proof of visible contact.",
        absolute_position_used="Saved model position observation used in temporal fusion; not directly observed contact.",
        transport_rate_used="Saved model rate observation used in temporal fusion; not directly measured interface slip.",
        side_disagreement="Saved dispersion between side transport rates in reference px/s; not a contact probability.",
        left_surface_gap_px="Modelled left surface-to-tip image gap, not independently measured zero-gap contact.",
        right_surface_gap_px="Modelled right surface-to-tip image gap, not independently measured zero-gap contact.",
        point_id="Always empty: no unique directly observed contact material identity was established.",
        contact_confirmed="Always 0: this file exports model inference rather than confirmed contact candidates.")
    for side in ("left", "right"):
        for suffix, meaning in {
            "rate_supported": "Saved side_flux_valid model qualification flag; not confirmed interface slip.",
            "rate_used": "Saved transport_side_used flag in rate fusion.",
            "support_count": "Number of saved material-track supports for this side model.",
            "model": "Saved observation-model label selected for this side.",
            "reason": "Saved model acceptance/rejection explanation for this side.",
        }.items():
            result["columns"]["model_contact_estimates"][f"{side}_{suffix}"] = meaning


def _contours(repo: Path, case_id: str, out: Path, frame_ids: set[int], result: dict) -> None:
    source = next((repo / "results" / name / case_id / "geometry.npz" for name in
                   ("20261008_named_dual_contact", "20261007_two_layer_surface")
                   if (repo / "results" / name / case_id / "geometry.npz").exists()), None)
    if source is None:
        result["missing_items"].append("contours: no saved classified contour geometry for this case")
        return
    destination = out / "contours.csv"
    count = components = skipped = 0
    frame_count = 0
    states_count = {"1": 0, "2": 0, "3": 0}
    with np.load(source, allow_pickle=False) as arrays, destination.open("w", encoding="utf-8", newline="") as stream:
        prefix = "closed" if "closed_xy" in arrays else "path"
        required = ["frames", f"{prefix}_offsets", f"{prefix}_xy", f"{prefix}_state", f"{prefix}_edge_valid"]
        if any(key not in arrays for key in required):
            raise ValueError(f"Incomplete contour validity schema in {source}")
        xy, states, edges = (arrays[f"{prefix}_{key}"] for key in ("xy", "state", "edge_valid"))
        offsets = arrays[f"{prefix}_offsets"]
        if len(np.unique(arrays["frames"])) != len(arrays["frames"]):
            raise ValueError(f"Duplicate contour frame ID in {source}")
        if (len(offsets) != len(arrays["frames"]) + 1 or int(offsets[0]) != 0 or
                np.any(np.diff(offsets) < 0) or int(offsets[-1]) != len(xy) or
                len(states) != len(xy) or len(edges) != len(xy)):
            raise ValueError(f"Inconsistent packed contour geometry in {source}")
        writer = csv.DictWriter(stream, fieldnames=CONTOUR_COLUMNS)
        writer.writeheader()
        for i, frame in enumerate(arrays["frames"]):
            if int(frame) not in frame_ids:
                continue
            frame_count += 1
            lo, hi = map(int, offsets[i:i + 2])
            component = -1
            previous_index = None
            previous_state = None
            vertex_order = 0
            for index in range(lo, hi):
                state = int(states[index])
                if state == 0 or not np.isfinite(xy[index]).all() or state not in (1, 2, 3):
                    previous_index = None
                    previous_state = None
                    skipped += 1
                    continue
                # A component never bridges a skipped vertex, a state change or
                # an invalid outgoing edge from its predecessor. No wrap edge.
                if previous_index is None or previous_index != index - 1 or previous_state != state or not bool(edges[index - 1]):
                    component += 1
                    components += 1
                    vertex_order = 0
                reason = {1: "segmentation_only_or_conditional", 2: "gradient_and_boundary_contrast_supported",
                          3: "blade_adjacent_visibility_not_confirmed"}[state]
                _write_row(writer, dict(case_id=case_id, frame_id=int(frame), contour_id="material_outline",
                    component_id=f"segment_{component:04d}", vertex_order=vertex_order,
                    contour_kind="visible_free_surface", x_px=xy[index, 0], y_px=xy[index, 1],
                    visibility="visible" if state == 2 else "uncertain", evidence="segmented", point_id="",
                    reference_s_rel_mm="", source_vertex_index=index - lo, source_state=state, reason=reason))
                vertex_order += 1
                count += 1
                states_count[str(state)] += 1
                previous_index, previous_state = index, state
    _evidence(repo, source, out, result, "contours")
    result["files"]["contours"] = destination.name
    result["counts"]["contour_vertices"] = count
    result["contours"] = dict(source_representation=prefix, exported_frames=frame_count,
        exported_vertices=count, component_count=components, skipped_crop_or_invalid_vertices=skipped,
        source_state_counts=states_count, materials_have_point_ids=False, components_are_open=True,
        notes=["Contours are geometry estimates; independent contour vertices are not persistent material IDs.",
               "State 2 has image gradient/contrast support. States 1 and 3 are uncertain; state 0 (image-crop boundary) is omitted.",
               "Components split at every state change, skipped vertex and blade-crossing edge. The final-to-first edge is never added.",
               "A blade-clear edge alone does not prove original-surface identity or contact.",
               "Visible_free_surface denotes current boundary geometry, not verified original top surface or crack-face topology.",
               "Coordinates remain in the saved analysis-video space; no scaling or camera-motion transform is applied here."])
    result["columns"]["contours"] = dict(
        component_id="Open contiguous segment, scoped by frame_id and contour_id; only connect within one component.",
        source_vertex_index="Original within-frame packed contour vertex index.",
        source_state="1 segmentation-only/conditional; 2 image-supported; 3 blade-adjacent/visibility uncertain. State 0 omitted.",
        visibility="visible only for state 2; uncertain for states 1 and 3.",
        evidence="segmented: saved segmentation plus local image refinement, not a material track.",
        point_id="Empty: contour vertices do not carry proven persistent material identity.",
        reference_s_rel_mm="Empty: current contour arc length is not a reference material coordinate.")


def export_optional(repo: Path, case_id: str, out: Path, frame_ids: np.ndarray) -> dict:
    """Write optional saved data into one case exchange directory.

    Force keeps its entire independent sensor sampling. Models and contours
    are restricted by exact membership in ``frame_ids`` without interpolation.
    Missing optional sources are reported, never synthesized.
    """
    repo, out = Path(repo).resolve(), Path(out).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError("case_id must be a plain identifier")
    frames = np.asarray(frame_ids)
    if frames.ndim != 1 or frames.dtype.kind not in "iu" or len(np.unique(frames)) != len(frames):
        raise ValueError("frame_ids must be a one-dimensional array of unique integers")
    out.mkdir(parents=True, exist_ok=True)
    result = dict(schema="optical_flow_exchange_optional_v1", case_id=case_id,
        files={}, evidence_files=[], counts={}, sources_sha256={}, missing_items=[], columns={},
        original_flow_recomputed=False,
        coordinate_space="saved analysis source video pixel centres; x right, y down; no transformation in optional exporter")
    selected = set(map(int, frames))
    _force(repo, case_id, out, result)
    _models(repo, case_id, out, selected, result)
    _contours(repo, case_id, out, selected, result)
    return result
