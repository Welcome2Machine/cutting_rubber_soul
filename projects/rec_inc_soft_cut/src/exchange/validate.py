"""Independently validate an existing local optical-flow exchange, in chunks.

All CSV rows are checked. Coordinate comparison against saved source arrays is
sampled: first/last points in each frame plus eight fixed-seed identities (and
chunk-edge points). This is interface/provenance validation, not physical truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


from .observations import TRACK_COLUMNS
LAB_COLUMNS = ('x_ref_lab_mm', 'y_ref_lab_mm', 'x_lab_mm', 'y_lab_mm', 'ux_mm', 'uy_mm')
TRACK_NUMERIC = tuple(c for c in TRACK_COLUMNS if c not in (
    'case_id', 'point_id', 'reference_kind', 'point_kind', 'track_status', 'invalid_reason',
    'original_seed_point_id', 'parent_point_id', 'quality_source', 'roi_ref', 'roi_current'))
COVERAGE_NUMERIC = ('total_points', 'raw_supported_points', 'valid_points', 'forecast_points',
                    'occluded_points', 'rejected_points', 'out_of_frame_points',
                    'new_observation_segments', 'reidentified_points')
SENTINELS = (-999., -9999., -99999.)


def _require(condition, message):
    if not bool(np.all(condition)):
        raise ValueError(message)


def _read_json(path):
    def forbidden(value):
        raise ValueError(f'Nonfinite JSON constant {value} in {path}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=forbidden)


def _chunks(path, chunk_size):
    yield from pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunk_size,
                           encoding='utf-8-sig', on_bad_lines='error')


def _table_path(folder, name):
    """Accept the original CSV or its lossless gzip form, within the case only."""
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Table filename must stay within the exchange case')
    path = folder / name
    return path if path.is_file() or name.endswith('.gz') else folder / (name + '.gz')


def _numeric(table, columns, context):
    result = {}
    for name in columns:
        if name not in table:
            continue
        text = table[name]
        values = pd.to_numeric(text, errors='coerce').to_numpy(dtype=float)
        present = text.ne('').to_numpy()
        _require(~present | np.isfinite(values), f'{context}: {name} has a nonfinite or nonnumeric value')
        _require(~np.isin(values, SENTINELS), f'{context}: {name} contains a forbidden sentinel')
        result[name] = values
    return result


def _integer(values, context, *, allow_empty=False):
    ok = np.isfinite(values) & (values == np.floor(values))
    _require(ok | (np.isnan(values) if allow_empty else False), f'{context}: integer required')


def _boolean(values, context):
    _require(np.isin(values, [0, 1]), f'{context}: boolean must be 0 or 1')


def _displacement_roundoff_bound(xy, reference, supplied_mm, scale, source_dtype):
    """Forward-error bound, not an allowance for tracking error.

    Saved positions are subtracted in their source dtype. On NumPy 2.x the
    scalar scale may also round to that dtype and the product rounds once.
    Three roundings give gamma(3)=3u/(1-3u), u=eps/2. CSV numbers have nine
    significant decimal digits, whose relative rounding bound is 5e-9. The
    double-precision subtraction used for checking adds only its own eps.
    """
    u = np.finfo(source_dtype).eps / 2
    gamma3 = 3*u / (1-3*u)
    decimal_roundoff = 5e-9 / (1-5e-9)
    coordinate_error = decimal_roundoff * (np.abs(xy) + np.abs(reference))
    delta = np.abs(xy-reference)
    arithmetic_error = gamma3 * (delta+coordinate_error)
    verification_error = np.finfo(np.float64).eps * 2 * (delta+coordinate_error)
    derived = (coordinate_error+arithmetic_error+verification_error)*scale + decimal_roundoff*np.abs(supplied_mm)
    # Keep every previously accepted row accepted, so completed checks may be
    # retained when only a dtype-roundoff failure needs a full case recheck.
    previous = max(1e-8, scale*1.2e-5) + 1e-7*delta*scale
    return np.maximum(previous, derived)


def _source_paths(case_dir, metadata):
    paths = (metadata.get('source_paths') or metadata.get('provenance', {}).get('source_paths')
             or metadata.get('provenance', {}).get('paths'))
    if not paths:
        for filename in ('source_evidence.json', 'provenance.json'):
            file = case_dir / filename
            if file.is_file():
                evidence = _read_json(file)
                paths = evidence.get('paths') or evidence.get('source_paths')
                if paths:
                    break
    if paths:
        output = {}
        for key, value in paths.items():
            path = Path(value)
            output[key] = path if path.is_absolute() else case_dir / path
        return output
    for repo in case_dir.parents:
        manifest_path = repo / 'results/20261005_0930_all/manifest.json'
        if manifest_path.is_file():
            case = next((v for v in _read_json(manifest_path)['cases'] if v['name'] == metadata['case_id']), None)
            if case:
                def rooted(value):
                    p = Path(value)
                    return p if p.is_absolute() else repo / p
                return dict(tracks=rooted(case['tracks_root']) / case['name'] / 'tracking/tracks.npz',
                            measurements_dir=rooted(case['measurements_root']) / case['name'])
    raise ValueError('Original track paths are unavailable for independent source sampling')


class _SourceSampler:
    def __init__(self, paths):
        with np.load(paths['tracks'], allow_pickle=False) as z:
            self.p = z['positions']
            self.seeds = z['seed_points']
            self.raw = z['supported']
            self.ids = z['point_ids']
            self.frames = z['source_frame_indices']
        self.row = {int(f): i for i, f in enumerate(self.frames)}
        self.column = {str(int(i)): j for j, i in enumerate(self.ids)}
        measurement = Path(paths['measurements_dir']) / 'measurement_support.npz'
        with np.load(measurement, allow_pickle=False) as z:
            _require(np.array_equal(z['point_ids'], self.ids), 'Source measurement IDs differ from source tracking IDs')
            self.support = z['supported']
            self.measure_row = {int(f): i for i, f in enumerate(z['frame_indices'])}
        rng = np.random.default_rng(20261012)
        selected = rng.choice(len(self.ids), min(8, len(self.ids)), replace=False)
        self.selected_ids = {str(int(self.ids[i])) for i in selected} | {str(int(self.ids[0])), str(int(self.ids[-1]))}
        self.sample_rows = 0

    def check(self, table, numbers, context):
        chosen = table['original_seed_point_id'].isin(self.selected_ids).to_numpy(copy=True)
        # Chunk edges add checks; they never replace each frame's first/last point.
        frame_values = numbers['frame_id']
        chosen[0] = chosen[-1] = True
        edges = np.flatnonzero(np.diff(frame_values) != 0)
        chosen[edges] = chosen[edges + 1] = True
        indices = np.flatnonzero(chosen)
        local = table.iloc[indices]
        try:
            fr = np.array([self.row[int(f)] for f in numbers['frame_id'][indices]])
            mr = np.array([self.measure_row[int(f)] for f in numbers['frame_id'][indices]])
            point = np.array([self.column[i] for i in local['original_seed_point_id']])
            ref = np.array([self.row[int(f)] for f in numbers['reference_frame_id'][indices]])
        except KeyError as error:
            raise ValueError(f'{context}: sampled source frame/identity missing: {error}') from error
        valid = numbers['valid'][indices] == 1
        raw = numbers['raw_tracking_supported'][indices] == 1
        _require(np.array_equal(raw, self.raw[fr, point]), f'{context}: raw support differs from saved source')
        _require(np.array_equal(valid, self.support[mr, point]), f'{context}: validity differs from saved measurement source')
        xy = np.column_stack((numbers['x_px'][indices], numbers['y_px'][indices]))
        pred = np.column_stack((numbers['x_pred_px'][indices], numbers['y_pred_px'][indices]))
        rejected = np.column_stack((numbers['x_raw_match_px'][indices], numbers['y_raw_match_px'][indices]))
        current = np.where(valid[:, None], xy, np.where(raw[:, None], rejected, pred))
        target = self.p[fr, point]
        _require(np.isclose(current, target, atol=1e-6, rtol=1e-8).all(1), f'{context}: sampled current coordinates differ from saved source')
        reference = np.column_stack((numbers['x_ref_px'][indices], numbers['y_ref_px'][indices]))
        _require(np.isclose(reference, self.p[ref, point], atol=1e-6, rtol=1e-8).all(1), f'{context}: sampled reference coordinates differ from saved source')
        seeds = np.column_stack((numbers['original_seed_x_px'][indices], numbers['original_seed_y_px'][indices]))
        _require(np.isclose(seeds, self.seeds[point], atol=1e-6, rtol=1e-8).all(1), f'{context}: sampled seed coordinates differ from saved source')
        self.sample_rows += len(indices)


def _validate_frames(folder, case_id):
    table = pd.read_csv(folder / 'frames.csv', dtype=str, keep_default_na=False)
    for key in ('case_id', 'frame_id', 'time_s', 'depth_mm', 'depth_video_estimate_mm',
                'geometry_supported', 'geometry_interpolated'):
        _require(key in table, f'frames.csv lacks {key}')
    _require(table['case_id'].eq(case_id), 'frames.csv case_id differs from metadata')
    numeric_cols = [c for c in table if c not in ('case_id', 'stage', 'depth_source', 'blade_tip_status',
                                                'invalid_reason', 'depth_estimate_status', 'blade_marker_id')
                    and (c.endswith(('_px', '_mm', '_s', '_id', '_deg')) or c in (
                        'valid_frame', 'geometry_supported', 'geometry_interpolated'))]
    numbers = _numeric(table, numeric_cols, 'frames.csv')
    frame = numbers['frame_id']
    _integer(frame, 'frames.csv frame_id')
    _require(len(frame) > 0 and frame[0] == 0, 'frames.csv must retain reference frame zero')
    _require(np.diff(frame) > 0, 'frames.csv frame_id must be unique and strictly increasing')
    _require(np.isfinite(numbers['time_s']), 'frames.csv time_s must be finite')
    _require(np.diff(numbers['time_s']) > 0, 'frames.csv time_s must increase strictly')
    _require(table['depth_mm'].eq(''), 'Core depth_mm must be empty without fixed-base laboratory registration')
    if 'capture_time_s' in table:
        _require(table['capture_time_s'].eq(''), 'Unverified raw capture_time_s must remain empty')
    for c in LAB_COLUMNS:
        if c in table:
            _require(table[c].eq(''), f'frames.csv {c} must remain unregistered/empty')
    for c in ('valid_frame', 'geometry_supported', 'geometry_interpolated'):
        if c in numbers:
            _boolean(numbers[c], f'frames.csv {c}')
    _require(~((numbers['geometry_supported'] == 1) & (numbers['geometry_interpolated'] == 1)),
             'frames.csv geometry cannot be supported and interpolated simultaneously')
    if 'blade_tip_status' in table:
        _require(~((numbers['geometry_interpolated'] == 1) & table['blade_tip_status'].eq('observed')),
                 'frames.csv interpolation is mislabeled observed')
    return table, frame.astype(np.int64)


def _validate_frame_source(table, frames, metadata, paths):
    """Check every frame's clock/geometry where source evidence paths exist."""
    checks = {}
    if 'capture_time_estimate_s' in table:
        _require('original_avi_frame_id' in table, 'Capture-time estimate lacks upstream AVI frame IDs')
        avi = pd.to_numeric(table['original_avi_frame_id'], errors='coerce').to_numpy(float)
        _integer(avi, 'Capture-time estimate original_avi_frame_id')
        estimate = pd.to_numeric(table['capture_time_estimate_s'], errors='coerce').to_numpy(float)
        _require(np.isfinite(estimate), 'Capture-time estimate is missing or nonfinite')
        _require(np.isclose(estimate, avi / 200., rtol=1e-10, atol=1e-9),
                 'Capture-time estimate differs from original AVI frame ID / 200')
        _require(metadata.get('source', {}).get('capture_rate_status') == 'camera_overlay_supported_CFR_estimate',
                 'Capture-time estimate lacks the declared camera-overlay evidence status')
        _require(metadata.get('source', {}).get('camera_reported_fps') == 200,
                 'Capture-time estimate lacks the camera-reported 200 fps rate')
        checks['all_capture_time_estimates_match_avi_200fps'] = True
        checks['exact_capture_clock_validated'] = False
    fps = metadata.get('source', {}).get('fps_nominal')
    if fps is not None:
        _require(np.isfinite(fps) and fps > 0, 'Invalid source container fps')
        actual = pd.to_numeric(table['time_s']).to_numpy(float)
        _require(np.isclose(actual, frames / fps, rtol=1e-8, atol=1e-6), 'frames.csv container time differs from selected MP4 clock')
        checks['all_container_times_match'] = True
    if 'mapping' in paths and paths['mapping'].is_file():
        with np.load(paths['mapping'], allow_pickle=False) as z:
            row = {int(f): i for i, f in enumerate(z['frame_indices'])}
            try:
                selected = np.array([row[int(f)] for f in frames])
            except KeyError as error:
                raise ValueError('Frame missing from saved import mapping') from error
            if 'original_avi_frame_id' in table:
                actual = pd.to_numeric(table['original_avi_frame_id'], errors='coerce').to_numpy(float)
                _require(np.array_equal(actual, z['source_indices'][selected]), 'Exported original AVI frame IDs differ from saved mapping')
                checks['all_avi_frame_ids_match'] = True
    if 'geometry' in paths and paths['geometry'].is_file():
        with np.load(paths['geometry'], allow_pickle=False) as z:
            for name in ('geometry_supported', 'geometry_interpolated'):
                source_name = 'supported' if name == 'geometry_supported' else name
                if source_name in z:
                    actual = pd.to_numeric(table[name]).to_numpy(int)
                    _require(np.array_equal(actual, z[source_name][frames]), f'{name} differs from saved geometry')
            for column, values in (('blade_tip_x_px', z['tips_px'][frames, 0]),
                                   ('blade_tip_y_px', z['tips_px'][frames, 1]),
                                   ('depth_video_estimate_mm', z['depth_mm'][frames])):
                if column in table:
                    actual = pd.to_numeric(table[column], errors='coerce').to_numpy(float)
                    present = table[column].ne('').to_numpy()
                    _require(np.isclose(actual[present], values[present], atol=1e-6, rtol=1e-8),
                             f'{column} differs from saved geometry')
            checks['all_supplied_frame_geometry_matches'] = True
    return checks


def validate_case(folder: Path, *, chunk_size=100000) -> dict:
    folder = Path(folder)
    metadata = _read_json(folder / 'metadata.json')
    case_id = metadata['case_id']
    frame_table, frame_ids = _validate_frames(folder, case_id)
    frame_set = set(map(int, frame_ids))
    source_paths = _source_paths(folder, metadata)
    source = _SourceSampler(source_paths)
    frame_source_checks = _validate_frame_source(frame_table, frame_ids, metadata, source_paths)
    scale = metadata.get('coordinates', {}).get('conditional_image_scale_mm_px')
    if scale is not None:
        _require(np.isfinite(scale) and scale > 0, 'Conditional image scale must be positive')
    references, original_references = {}, {}
    continuity = {}
    totals, status_totals, coverage_totals, frame_counts = Counter(), Counter(), {}, Counter()
    displacement_max_error_mm, displacement_max_bound_mm = 0., 0.
    last_frame, carry_ids, carry_seeds = -1, set(), set()
    for table in _chunks(_table_path(folder, metadata.get('files', {}).get('tracks', 'tracks.csv')), chunk_size):
        if not len(table):
            continue
        context = f'tracks.csv data rows {totals["rows"] + 1}..{totals["rows"] + len(table)}'
        _require(tuple(table.columns) == TRACK_COLUMNS, 'tracks.csv columns differ from 40-column exchange schema')
        _require(table['case_id'].eq(case_id), f'{context}: case_id differs')
        text_columns = [c for c in table if c not in TRACK_NUMERIC]
        _require(~table[text_columns].isin(['NaN', 'nan', 'Inf', 'inf', 'Infinity', '-Inf']).any().any(),
                 f'{context}: forbidden nonfinite placeholder')
        values = _numeric(table, TRACK_NUMERIC, context)
        f, r = values['frame_id'], values['reference_frame_id']
        _integer(f, f'{context}: frame_id'); _integer(r, f'{context}: reference_frame_id')
        _require(np.isin(f, frame_ids) & np.isin(r, frame_ids), f'{context}: referenced frame absent from frames.csv')
        _require(r <= f, f'{context}: reference lies after current frame')
        _require(np.diff(f) >= 0, f'{context}: tracks must be exported in frame order for streaming verification')
        _require(f[0] >= last_frame, f'{context}: frame order changed across chunks')
        _require(~table.duplicated(['frame_id', 'point_id']), f'{context}: duplicate frame/point primary key')
        _require(~table.duplicated(['frame_id', 'original_seed_point_id']), f'{context}: original seed duplicated within a frame')
        if f[0] == last_frame:
            first = table.loc[f == last_frame]
            _require(not set(first['point_id']) & carry_ids, f'{context}: duplicate frame/point across chunks')
            _require(not set(first['original_seed_point_id']) & carry_seeds, f'{context}: duplicated original seed across chunks')
        last = table.loc[f == f[-1]]
        if f[-1] != last_frame:
            carry_ids, carry_seeds = set(), set()
        carry_ids.update(last['point_id']); carry_seeds.update(last['original_seed_point_id'])
        last_frame = int(f[-1])
        for c in ('valid', 'raw_tracking_supported', 'measurement_supported', 'motion_step_valid'):
            _boolean(values[c], f'{context}: {c}')
        good, raw = values['valid'] == 1, values['raw_tracking_supported'] == 1
        _require(np.array_equal(good, values['measurement_supported'] == 1), f'{context}: valid differs from measurement_supported')
        _require(~good | raw, f'{context}: raw prediction promoted to valid observation')
        status = table['track_status']
        _require(status.isin(['tracked', 'predicted', 'rejected', 'occluded', 'out_of_frame']), f'{context}: unknown track_status')
        _require(np.array_equal(good, status.eq('tracked').to_numpy()), f'{context}: forecast/occlusion/rejection status contradicts validity')
        _require(~status.eq('predicted').to_numpy() | ~raw, f'{context}: predicted status has image support')
        _require(~status.eq('rejected').to_numpy() | raw, f'{context}: rejected match lacks image support')
        _require(table.loc[good, 'invalid_reason'].eq('') & True, f'{context}: valid observation has invalid reason')
        _require(table.loc[~good, 'invalid_reason'].ne(''), f'{context}: invalid observation lacks reason')
        xy = np.column_stack((values['x_px'], values['y_px']))
        _require(~good | np.isfinite(xy).all(1), f'{context}: valid point lacks finite coordinates')
        _require(table.loc[~good, ['x_px', 'y_px']].eq('').all().all(), f'{context}: invalid main coordinates must be empty')
        for c in LAB_COLUMNS:
            _require(table[c].eq(''), f'{context}: laboratory coordinate/displacement {c} must remain empty')
        _require(np.isfinite(np.column_stack((values['x_ref_px'], values['y_ref_px']))).all(1), f'{context}: reference coordinates missing')
        _require(table['point_id'].ne('') & table['original_seed_point_id'].ne(''), f'{context}: missing point identity')
        _require(table['reference_kind'].isin(['precontact', 'deformed', 'unknown']), f'{context}: bad reference_kind')
        for columns, active in ((('x_pred_px', 'y_pred_px'), ~raw), (('x_raw_match_px', 'y_raw_match_px'), raw & ~good)):
            available = np.column_stack([values[c] for c in columns])
            _require(~active | np.isfinite(available).all(1), f'{context}: {columns} coordinates missing')
            _require(table.loc[~active, list(columns)].eq('').all().all(), f'{context}: {columns} populated for wrong observation class')
        zero = f == 0
        quality_columns = ['confidence', 'fb_error_px', 'position_uncertainty_px', 'selected_reference', 'reference_age_steps']
        _require(table.loc[zero, quality_columns].eq('').all().all(), f'{context}: frame-zero initialization has artificial match quality')
        _require(table.loc[zero, 'quality_source'].eq('seed_initialization'), f'{context}: bad frame-zero quality source')
        _require(table.loc[~raw, 'confidence'].eq(''), f'{context}: forecast has observed confidence')
        new_segment = (r == f) & (f > 0)
        _require(~new_segment | good, f'{context}: new segment lacks observed reference position')
        _require(table.loc[r > 0, 'reference_kind'].eq('deformed'), f'{context}: resumed identity has undeformed reference')
        _require(table.loc[r > 0, 'parent_point_id'].ne(''), f'{context}: resumed segment lacks parent bookkeeping ID')
        _require(~((zero | new_segment | ~good) & (values['motion_step_valid'] == 1)), f'{context}: invalid continuous motion edge')
        group = table.groupby('original_seed_point_id', sort=False)
        prior_good = pd.Series(good, index=table.index).groupby(table['original_seed_point_id'], sort=False).shift()
        prior_frame = pd.Series(f, index=table.index).groupby(table['original_seed_point_id'], sort=False).shift()
        prior_id = group['point_id'].shift()
        first_indices = table.index[~table['original_seed_point_id'].duplicated()]
        for index in first_indices:
            seed_id = table.at[index, 'original_seed_point_id']
            if seed_id in continuity:
                previous_frame, previous_good, previous_id = continuity[seed_id]
                prior_frame.at[index], prior_good.at[index], prior_id.at[index] = previous_frame, previous_good, previous_id
        known = prior_frame.notna().to_numpy()
        was_good = prior_good.fillna(False).astype(bool).to_numpy()
        same_identity = table['point_id'].eq(prior_id).to_numpy()
        resumed = known & good & ~was_good
        _require(~resumed | (~same_identity & new_segment), f'{context}: unsupported gap resumes an unverified old identity')
        adjacent = known & (f == prior_frame.to_numpy(dtype=float) + 1)
        _require(~(adjacent & good & was_good) | same_identity, f'{context}: continuously observed identity silently reset')
        for index in table.index[~table['original_seed_point_id'].duplicated(keep='last')]:
            continuity[table.at[index, 'original_seed_point_id']] = (
                int(table.at[index, 'frame_id']), table.at[index, 'valid'] == '1', table.at[index, 'point_id'])
        ref_xy = np.column_stack((values['x_ref_px'], values['y_ref_px']))
        _require(np.isclose(xy[good & (r == f)], ref_xy[good & (r == f)], rtol=1e-8, atol=1e-6), f'{context}: reference-frame current point differs from its reference')
        if scale is None:
            _require(table[['ux_image_est_mm', 'uy_image_est_mm']].eq('').all().all(), f'{context}: displacement scale is unavailable')
        else:
            expected = (xy - ref_xy) * [scale, -scale]
            supplied = np.column_stack((values['ux_image_est_mm'], values['uy_image_est_mm']))
            error = np.abs(supplied[good]-expected[good])
            bound = _displacement_roundoff_bound(xy[good], ref_xy[good], supplied[good], scale, source.p.dtype)
            _require(error <= bound, f'{context}: image displacement does not match current minus reference')
            if error.size:
                displacement_max_error_mm = max(displacement_max_error_mm, float(error.max()))
                displacement_max_bound_mm = max(displacement_max_bound_mm, float(bound.max()))
            _require(table.loc[~good, ['ux_image_est_mm', 'uy_image_est_mm']].eq('').all().all(), f'{context}: invalid point has observation displacement')
        ref_columns = ['point_id', 'reference_frame_id', 'reference_kind', 'x_ref_px', 'y_ref_px',
                       'original_seed_point_id', 'parent_point_id']
        unique = table[ref_columns].drop_duplicates()
        _require(~unique['point_id'].duplicated(), f'{context}: same identity changes reference')
        for row in unique.itertuples(index=False, name=None):
            previous = references.setdefault(row[0], row[1:])
            _require(previous == row[1:], f'{context}: reference changed across chunks for {row[0]}')
        seeds = table[['original_seed_point_id', 'original_seed_x_px', 'original_seed_y_px']].drop_duplicates()
        _require(~seeds['original_seed_point_id'].duplicated(), f'{context}: original seed coordinates changed')
        for row in seeds.itertuples(index=False, name=None):
            _require(original_references.setdefault(row[0], row[1:]) == row[1:], f'{context}: original seed changed across chunks')
        source.check(table, values, context)
        calculated = pd.DataFrame(dict(frame_id=f.astype(np.int64), roi_ref=table['roi_ref'].to_numpy(),
            total_points=np.ones(len(table), int), raw_supported_points=raw.astype(int), valid_points=good.astype(int),
            forecast_points=(~raw).astype(int), occluded_points=status.eq('occluded').astype(int).to_numpy(),
            rejected_points=status.eq('rejected').astype(int).to_numpy(), out_of_frame_points=status.eq('out_of_frame').astype(int).to_numpy(),
            new_observation_segments=new_segment.astype(int), reidentified_points=np.zeros(len(table), int)))
        for key, row in calculated.groupby(['frame_id', 'roi_ref'], sort=False)[list(COVERAGE_NUMERIC)].sum().iterrows():
            coverage_totals.setdefault((int(key[0]), key[1]), np.zeros(len(COVERAGE_NUMERIC), np.int64))[:] += row.to_numpy(np.int64)
        frame_counts.update({int(k): int(v) for k, v in pd.Series(f).value_counts().items()})
        totals.update(rows=len(table), valid=int(good.sum()), raw_supported=int(raw.sum()), new_segments=int(new_segment.sum()))
        status_totals.update(status.value_counts().to_dict())
    _require(totals['rows'] > 0, 'tracks.csv is empty')
    _require(set(frame_counts) == frame_set, 'Some frames have no material-point rows')
    _require(all(v == len(source.ids) for v in frame_counts.values()), 'A frame does not retain all original spatial samples')
    for pid, reference in references.items():
        parent = reference[-1]
        if parent:
            _require(parent in references, f'{pid}: parent ID absent')
            _require(int(references[parent][0]) < int(reference[0]), f'{pid}: parent segment is not earlier')
            _require(references[parent][-2] == reference[-2], f'{pid}: parent belongs to a different original seed')
    coverage_rows = _validate_coverage(folder, case_id, coverage_totals, frame_ids)
    optional = _validate_optional(folder, case_id, frame_ids, chunk_size)
    return dict(case_id=case_id, passed=True, frame_count=len(frame_ids), track_rows=totals['rows'],
                valid_observations=totals['valid'], raw_supported_observations=totals['raw_supported'],
                observation_segments=len(references), resumed_segments=totals['new_segments'],
                status_counts=dict(status_totals), coverage_rows=coverage_rows,
                source_comparison=dict(sample_rows=source.sample_rows, seed=20261012,
                    fixed_original_ids=sorted(source.selected_ids),
                    method='First/last points of every frame plus up to eight fixed-seed IDs and chunk-edge points; current/reference/seed coordinates and raw/measurement validity checked',
                    numeric_tolerance='Source pixels: atol 1e-6, rtol 1e-8 (CSV 9 significant digits)',
                    is_full_source_row_comparison=False), optional_files=optional,
                frame_source_checks=frame_source_checks,
                displacement_roundoff_check=dict(source_dtype=str(source.p.dtype),
                    method='max(previous atol=max(1e-8,scale*1.2e-5)+rtol1e-7, three source-dtype roundings gamma(3) plus nine-significant-digit CSV rounding); numerical roundoff only',
                    max_absolute_residual_mm=displacement_max_error_mm,
                    max_allowed_roundoff_bound_mm=displacement_max_bound_mm),
                checks_scope='Every exported track row: finite values, identity/reference, keys, frame links, statuses and displacement. All coverage rows cross-checked; source track coordinates sampled.',
                physical_truth_validated=False)


def _validate_coverage(folder, case_id, calculated, frame_ids):
    table = pd.read_csv(folder / 'coverage.csv', dtype=str, keep_default_na=False)
    _require(table['case_id'].eq(case_id), 'coverage.csv case_id differs')
    _require(~table.duplicated(['frame_id', 'roi_ref']), 'coverage.csv duplicate primary key')
    values = _numeric(table, ('frame_id', *COVERAGE_NUMERIC, 'valid_fraction'), 'coverage.csv')
    _integer(values['frame_id'], 'coverage frame_id')
    _require(np.isin(values['frame_id'], frame_ids), 'coverage.csv references an absent frame')
    _require(table['roi_ref'].isin(['near_left', 'near_right', 'far_left', 'far_right']), 'coverage.csv unknown fixed region')
    seen = set()
    for i, row in enumerate(table.itertuples(index=False)):
        key = (int(row.frame_id), row.roi_ref)
        actual = np.array([values[c][i] for c in COVERAGE_NUMERIC])
        expected = calculated.get(key, np.zeros(len(COVERAGE_NUMERIC), int))
        _require(np.array_equal(actual, expected), f'coverage.csv {key}: counts disagree with tracks.csv')
        fraction = values['valid_fraction'][i]
        _require(np.isclose(fraction, actual[2]/actual[0], atol=1e-8) if actual[0] else np.isnan(fraction),
                 f'coverage.csv {key}: invalid fraction')
        seen.add(key)
    _require(set(calculated) <= seen, 'coverage.csv is missing nonempty frame/region rows')
    return len(table)


def _validate_optional(folder, case_id, frame_ids, chunk_size):
    reports = {}
    for filename in ('events.csv', 'force_channels.csv', 'model_contact_estimates.csv', 'contours.csv'):
        path = _table_path(folder, filename)
        if not path.is_file():
            continue
        rows, previous_time, sample_ids = 0, None, set()
        for table in _chunks(path, chunk_size):
            _require(table['case_id'].eq(case_id), f'{filename}: case_id differs')
            _require(~table.isin(['NaN', 'nan', 'Inf', 'inf', 'Infinity', '-Inf', '-999', '-9999', '-99999']).any().any(),
                     f'{filename}: forbidden missing-value token')
            numeric_columns = [c for c in table if c.endswith(('_px', '_mm', '_um', '_N', '_V', '_s'))
                               or c in ('frame_id', 'frame_start', 'frame_end', 'sample_id', 'vertex_order',
                                        'valid', 'prediction_only', 'contact_confirmed')]
            values = _numeric(table, numeric_columns, filename)
            for key in ('frame_id', 'frame_start', 'frame_end'):
                if key in values:
                    present = np.isfinite(values[key])
                    _integer(values[key], f'{filename} {key}', allow_empty=True)
                    _require(np.isin(values[key][present], frame_ids), f'{filename}: reference frame missing')
            if filename == 'events.csv' and 'frame_start' in values and 'frame_end' in values:
                paired = np.isfinite(values['frame_start']) & np.isfinite(values['frame_end'])
                _require(values['frame_start'][paired] <= values['frame_end'][paired], 'events.csv reversed interval')
            if filename == 'force_channels.csv':
                _integer(values['sample_id'], 'force sample_id')
                ids = set(map(int, values['sample_id']))
                _require(len(ids) == len(table) and not ids & sample_ids, 'force sample_id not unique')
                sample_ids.update(ids)
                times = values['time_force_s']
                _require(np.isfinite(times), 'force own-clock times are missing')
                time_good = np.r_[True, np.diff(times) > 0]
                if previous_time is not None and len(times):
                    time_good[0] = times[0] > previous_time
                valid = values['valid'] == 1
                _require(~valid | time_good, 'force non-increasing own-clock sample incorrectly marked valid')
                if len(times):
                    previous_time = times[-1]
            if filename == 'model_contact_estimates.csv' and 'contact_confirmed' in values:
                _require(values['contact_confirmed'] == 0, 'Model contact inference promoted to confirmed contact')
            rows += len(table)
        reports[filename] = dict(rows=rows, finite_values_checked=True, foreign_frames_checked=filename != 'force_channels.csv')
    return reports


def _validate_case_report(folder, chunk_size):
    """Case worker returns data only; the parent process owns validation.json."""
    folder = Path(folder)
    try:
        return validate_case(folder, chunk_size=chunk_size)
    except Exception as error:
        return dict(case_id=folder.name, passed=False, error_type=type(error).__name__, error=str(error))


def validate_root(root: Path, cases=None, *, chunk_size=100000, workers=1) -> dict:
    root = Path(root).resolve()
    selected = list(cases) if cases else sorted(p.name for p in root.iterdir() if (p / 'metadata.json').is_file())
    _require(bool(selected), 'No exchange case directories found')
    _require(len(set(selected)) == len(selected), 'Duplicate requested cases')
    _require(workers in (1, 2, 3), 'workers must be 1, 2 or 3')
    for case in selected:
        _require(Path(case).name == case, 'Case must be a directory name')
    started = time.perf_counter()
    by_case = {}
    def record(case, result):
        by_case[case] = result
        print(f'{case}: {"PASS" if result["passed"] else "FAIL"}', flush=True)
    if workers == 1:
        for case in selected:
            record(case, _validate_case_report(root / case, chunk_size))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks = {pool.submit(_validate_case_report, root / case, chunk_size): case for case in selected}
            for task in as_completed(tasks):
                case = tasks[task]
                try:
                    result = task.result()
                except Exception as error:
                    result = dict(case_id=case, passed=False, error_type=type(error).__name__, error=str(error))
                record(case, result)
    results = [by_case[case] for case in selected]
    report = dict(schema='optical_flow_exchange_validation_1', root=str(root),
                  passed=all(c['passed'] for c in results), cases=results, seconds=time.perf_counter()-started,
                  rows_streamed=True, chunk_size=chunk_size, workers=workers, physical_accuracy_certified=False,
                  interpreter=sys.executable)
    (root / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args(argv)
    return 0 if validate_root(args.root, args.cases, workers=args.workers)['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
