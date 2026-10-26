"""Loading-only motion trends using current material, local inference and images.

Saved long material tracks are read-only. Fresh short image matches estimate
current texture motion; they never acquire an original material ID.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from src.workflows.measurement import ROOT, SOURCE, NAMES, HIDDEN, digest, load_npz, save_json
from src.workflows.tracking import label_reader, encode_review_video
from src.analysis.blade_contact_geometry import blade_contact_geometry
from src.analysis.contact_motion_evidence import supported_local_slopes
from src.analysis.loading_blade_reference import prepare_loading_blade_reference
from src.analysis.loading_tip_trends import (LoadingTrendConfig, GROUP_NAMES, STATE_NAMES,
    current_groups, estimate_frame_trends, finalize_series, local_motion_fit, summarize_observations)
from src.analysis.local_image_motion import LocalImageMotionConfig, estimate_local_image_motion
from src.analysis.local_blade_mask_repair import LocalBladeMaskRepair

TIP_OUT = ROOT/'results/current_loading_tip_trends'
OUT = ROOT/'results/current_loading_interface_sliding'
MEASUREMENTS = ROOT/'results/current_contact_measurements'


def data(name):
    source = SOURCE/name
    plan = json.loads((source/'plan.json').read_text(encoding='utf-8'))
    meta = json.loads((source/'geometry.json').read_text(encoding='utf-8'))
    mapping = np.arange(plan['contact_frame'], plan['reversal_frame']+1)
    with np.load(source/'tracking/tracks.npz', allow_pickle=False) as stored:
        positions = stored['positions'][:plan['reversal_frame']+1].copy()
        original_support = stored['supported'][:len(positions)].copy()
        ids = stored['point_ids'].copy()
    geometry = load_npz(source/'geometry.npz')
    times = load_npz(source/'frame_mapping.npz')['physical_time_s']
    with np.load(MEASUREMENTS/name/'measurement_support.npz', allow_pickle=False) as stored:
        support = stored['supported'][:len(positions)].copy()
        edges = stored['motion_step_valid'][:len(positions)].copy()
    with np.load(MEASUREMENTS/name/'blade_contact_geometry.npz', allow_pickle=False) as stored:
        gap, material = stored['signed_blade_gap_px'][mapping].copy(), stored['material_pixel'][mapping].copy()
        blade = stored['on_blade'][mapping].copy()
    with np.load(MEASUREMENTS/name/'motion_measurements.npz', allow_pickle=False) as stored:
        legacy = stored['group_y_median'][mapping].copy()
    scale = 1000*meta['contact_to_reversal_mm']/meta['contact_to_reversal_px']
    reference = prepare_loading_blade_reference(geometry['tips_px'][mapping], geometry['supported'][mapping], times[mapping])
    return dict(source=source, plan=plan, positions=positions, original_support=original_support,
        ids=ids, support=support, edges=edges, frames=mapping, times=times, gap=gap,
        material=material & ~blade, legacy=legacy, reference=reference, um_per_px=scale,
        original_depth=geometry['depth_mm'][mapping])


def loading_mask_reader(source, name, frames, tips_xy):
    # No witness outside the loading interval, and no estimated mask is reused
    # as an observation for another repair.
    tips = np.full((int(frames[-1])+1, 2), np.nan)
    tips[frames] = tips_xy
    return LocalBladeMaskRepair(label_reader(source/'segmentation'/name/'semantic_masks'), tips)


def geometry_flags(repair, frames):
    estimated = np.array([repair.diagnostics[int(f)]['mask_estimated'] for f in frames], bool)
    uncertain = np.array([repair.diagnostics[int(f)]['reason'] != 'original_tip_mask_available'
                          for f in frames], bool)
    return estimated, uncertain


def apply_geometry_flags(trend, estimated, uncertain):
    # A persistent visible/registered tip discrepancy is not a successful mask
    # repair. Keep its trend value, but qualify the current thin-region geometry.
    before = trend.setdefault('state_before_geometry_check', trend['state'].copy())
    trend['state'] = before.copy()
    trend['state'][uncertain[:, None] & (trend['state'] > 1)] = 2
    trend['blade_mask_estimated'] = estimated
    trend['blade_geometry_uncertain'] = uncertain


def prepare_current_masks(case, name):
    repair = loading_mask_reader(case['source'], name, case['frames'], case['reference']['tips_xy'])
    estimated = np.zeros(len(case['frames']), bool)
    for row, frame in enumerate(case['frames']):
        labels, estimated[row] = repair.read(frame)
        if not estimated[row]:
            continue
        geometry = blade_contact_geometry(labels, case['positions'][frame], case['reference']['tips_xy'][row])
        material = geometry['material_pixel'] & ~geometry['on_blade']
        case['gap'][row], case['material'][row] = geometry['signed_blade_gap_px'], material
        # Exclude recovered blade pixels before fitting temporal point slopes.
        # This only changes analysis support; source tracks remain untouched.
        case['support'][frame] &= material
        case['edges'][frame] &= case['support'][frame]
        if frame+1 < len(case['edges']):
            case['edges'][frame+1] &= case['support'][frame]
    return repair, estimated


def probe_images(case, reader, args, region_builder=None):
    source, frames, reference = case['source'], case['frames'], case['reference']
    times = case['times'][frames]
    cfg = LocalImageMotionConfig(radius_px=90., max_points=480 if region_builder else 240,
        vertical_bin_px=48. if region_builder else 0., affine_rescue=True)
    rows, report, kept = {}, [], []
    cap = cv2.VideoCapture(str(source/'normalized_lossless.mkv'))
    if not cap.isOpened():
        raise OSError('Cannot open shadow-normalized image source')
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    last_image, last_row, last_labels = None, None, None
    try:
        for row, frame in enumerate(frames):
            ok, image = cap.read()
            if not ok:
                raise OSError(f'Missing source image {frame}')
            if row % args.image_step and row != len(frames)-1:
                continue
            labels = reader(frame)
            if last_row is not None and np.isfinite(reference['tips_xy'][[last_row, row]]).all():
                dt = times[row]-times[last_row]
                started = time.perf_counter()
                obs = estimate_local_image_motion(last_image, image, last_labels, labels,
                    reference['tips_xy'][last_row], reference['tips_xy'][row], dt_s=dt, config=cfg,
                    seed_region_mask=None if region_builder is None else region_builder(last_labels, reference['tips_xy'][last_row]))
                good = obs['valid'] & np.isfinite(obs['velocity_xy']).all(1)
                xy = obs['current_xy'][good]
                geom = blade_contact_geometry(labels, xy, reference['tips_xy'][row])
                good2 = geom['material_pixel'] & ~geom['on_blade']
                xy = xy[good2]
                velocity = obs['velocity_xy'][good][good2]-reference['velocity_xy_px_s'][row]
                quality = obs['confidence'][good][good2]
                rows[row] = dict(xy=xy, velocity=velocity, quality=quality,
                                 gap_px=geom['signed_blade_gap_px'][good2])
                kept.append(dict(row=np.full(len(xy), row, np.int32), xy=xy, velocity=velocity,
                    quality=quality, fb_error_px=obs['fb_error_px'][good][good2],
                    correlation=obs['masked_correlation'][good][good2]))
                report.append(dict(row=row, source_frames=[int(frames[last_row]), int(frame)],
                    physical_dt_s=float(dt), candidates=len(good), accepted=len(xy),
                    median_fb_px=float(np.median(obs['fb_error_px'][good])) if good.any() else None,
                    seconds=time.perf_counter()-started))
            last_row, last_image, last_labels = row, image, labels
    finally:
        cap.release()
    arrays = {key: np.concatenate([item[key] for item in kept]) for key in kept[0]} if kept else {}
    return rows, arrays, dict(config=asdict(cfg), pairs=report,
        interpretation='Freshly selected current image texture, not recovered original material IDs; no integration into historical trajectories.')


def reconstruction_checks(points, velocity, valid, gap, material, tip, labels, cfg):
    """Withhold all thin-band observations and predict their average location.

    This is a same-image consistency test, not independent motion ground truth.
    """
    group = current_groups(points, tip, gap, material, cfg)
    checks = []
    for g in (0, 2):
        test = valid & (group == g)
        if test.sum() < 4:
            continue
        center = np.mean(points[test], axis=0)
        target, _, _, cells = summarize_observations(points[test], velocity[test], np.ones(test.sum()), cfg)
        if cells < 2:
            continue
        same = points[:, 0] < tip[0] if g == 0 else points[:, 0] >= tip[0]
        use = (valid & material & same & ~test & (gap >= cfg.thin_px) & (gap < cfg.donor_thin_px)
               & (np.linalg.norm(points-center, axis=1) <= cfg.maximum_donor_radius_px))
        # Apply exactly the production support ray constraint.
        from src.analysis.loading_tip_trends import _visible_to_target
        rows = np.flatnonzero(use)
        rows = rows[_visible_to_target(points[rows], center, labels == 2)]
        fit = local_motion_fit(points[rows], velocity[rows], np.ones(len(rows)), center, cfg)
        if fit['valid']:
            checks.append(dict(group=g, observed_points=int(test.sum()), donor_points=len(rows),
                observed_xy_px_s=target.tolist(), estimated_xy_px_s=fit['velocity'].tolist(),
                y_error_px_s=float(fit['velocity'][1]-target[1]),
                direction_agrees=bool(np.sign(fit['velocity'][1]) == np.sign(target[1])),
                magnitude_above_noise=bool(abs(target[1]) > 2.)))
    return checks


def point_motion(case):
    """Fit loading point velocities, reusing a verified existing result when available."""
    frames, p, reference = case['frames'], case['positions'], case['reference']
    previous = TIP_OUT/case['source'].name
    cache, manifest = previous/'loading_trends.npz', previous/'measurement_summary.json'
    if cache.is_file() and manifest.is_file():
        old = load_npz(cache)
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        same = (np.array_equal(old['frame_indices'], frames) and np.array_equal(old['point_ids'], case['ids'])
                and np.allclose(old['tip_xy'], reference['tips_xy'], equal_nan=True)
                and meta['source_track_sha256'] == digest(case['source']/'tracking/tracks.npz'))
        if same:
            velocity, valid, kind = (old[key] for key in
                ('point_relative_velocity_xy', 'point_velocity_valid', 'point_velocity_kind'))
            return velocity.copy(), valid.copy(), np.where(kind == 1, .4, 1.), kind.copy(), old
    derivative = supported_local_slopes(p, case['times'][:len(p)], case['support'],
        motion_step_valid=case['edges'], half_window_s=.05, minimum_samples=3,
        minimum_span_s=.02, maximum_fit_rms=2.)
    velocity, valid = derivative['slope'][frames].copy(), derivative['valid'][frames].copy()
    quality, kind = np.ones(valid.shape), np.full(valid.shape, 2, np.uint8)
    for row, frame in enumerate(frames):
        if frame == 0:
            continue
        use = ~valid[row] & case['edges'][frame] & case['support'][frame] & case['support'][frame-1]
        velocity[row, use] = (p[frame, use]-p[frame-1, use])/(case['times'][frame]-case['times'][frame-1])
        valid[row, use] = True
        quality[row, use], kind[row, use] = .4, 1
    velocity -= reference['velocity_xy_px_s'][:, None]
    valid &= np.isfinite(velocity).all(2) & (reference['velocity_state'] > 0)[:, None]
    return velocity, valid, quality, kind, None


def measure_tip(name, args):
    cv2.setNumThreads(2)
    started = time.perf_counter()
    folder = args.out/name
    folder.mkdir(parents=True, exist_ok=True)
    case = data(name)
    frames, p = case['frames'], case['positions']
    times, reference, cfg = case['times'][frames], case['reference'], LoadingTrendConfig()
    repair, mask_estimated = prepare_current_masks(case, name)
    reader = lambda frame: repair.read(frame)[0]
    mask_diagnostics = [repair.diagnostics[int(frame)] for frame in frames
                        if repair.diagnostics[int(frame)]['reason'] != 'original_tip_mask_available']
    save_json(folder/'blade_mask_repair.json', dict(estimated_source_frames=frames[mask_estimated].tolist(),
        diagnostics=mask_diagnostics, original_labels_unchanged=True,
        interpretation='Local intersection of two original blade masks translated by registered tips; conditional geometry only.'))
    # Only prefix through reversal is supplied: a centered filter can never
    # borrow withdrawal movement to alter the end of the loading trend.
    derivative = supported_local_slopes(p, case['times'][:len(p)], case['support'],
        motion_step_valid=case['edges'], half_window_s=.05, minimum_samples=3,
        minimum_span_s=.02, maximum_fit_rms=2.)
    velocity, valid = derivative['slope'][frames].copy(), derivative['valid'][frames].copy()
    quality = np.ones(valid.shape)
    kind = np.full(valid.shape, 2, np.uint8)
    for row, frame in enumerate(frames):
        if frame == 0:
            continue
        # A consecutive measured edge remains useful when a full centered
        # window is unavailable. It is given lower weight, never a held forecast.
        use = ~valid[row] & case['edges'][frame] & case['support'][frame] & case['support'][frame-1]
        velocity[row, use] = (p[frame, use]-p[frame-1, use])/(case['times'][frame]-case['times'][frame-1])
        valid[row, use] = True
        quality[row, use], kind[row, use] = .4, 1
    velocity -= reference['velocity_xy_px_s'][:, None]
    valid &= np.isfinite(velocity).all(2) & (reference['velocity_state'] > 0)[:, None]
    print(f'{name}: new short-time image observations', flush=True)
    extra, image_arrays, image_report = probe_images(case, reader, args)
    if image_arrays:
        np.savez_compressed(folder/'short_image_samples.npz', **image_arrays)
    save_json(folder/'short_image_summary.json', image_report)
    records, checks, comparisons = [], [], []
    for row, frame in enumerate(frames):
        labels = reader(frame)
        result = estimate_frame_trends(p[frame], velocity[row], valid[row], case['gap'][row],
            case['material'][row], reference['tips_xy'][row], labels,
            quality=quality[row], extra=extra.get(row), config=cfg)
        # A regional statistic mostly built from single edges is a trend
        # estimate, even when its raw point count exceeds four.
        for g in range(4):
            use = (result['group_index'] == g) & valid[row]
            if result['state'][g] == 4 and np.any(use) and np.mean(kind[row, use] == 1) > .5:
                result['state'][g], result['method'][g] = 2, 'short_supported_edges'
        records.append(result)
        if row % 20 == 0:
            for check in reconstruction_checks(p[frame], velocity[row], valid[row], case['gap'][row],
                    case['material'][row], reference['tips_xy'][row], labels, cfg):
                check['row'] = row
                checks.append(check)
        if row in extra:
            obs = extra[row]
            eg = current_groups(obs['xy'], reference['tips_xy'][row], obs['gap_px'], np.ones(len(obs['xy']), bool), cfg)
            for g in range(4):
                use = eg == g
                direct = result['direct_trend_xy'][g]
                if use.sum() >= 3 and result['observation_count'][g] >= 4 and np.isfinite(direct).all():
                    value, _, _, _ = summarize_observations(obs['xy'][use], obs['velocity'][use], obs['quality'][use], cfg)
                    comparisons.append(dict(row=row, group=g, saved_group_y_px_s=float(direct[1]),
                        fresh_image_y_px_s=float(value[1]), difference_px_s=float(value[1]-direct[1])))
        if row % 300 == 0:
            print(f'{name}: loading trends {row}/{len(frames)}', flush=True)
    trend = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    finalize_series(trend, times, reference['velocity_state'], cfg)
    mask_estimated, geometry_uncertain = geometry_flags(repair, frames)
    apply_geometry_flags(trend, mask_estimated, geometry_uncertain)
    # Show the source of blade coordinates and of their derived velocity separately.
    trend['tip_coordinate_state'] = reference['state']
    depth = (reference['tips_xy'][:, 1]-reference['tips_xy'][0, 1])*case['um_per_px']/1000.
    trend.update(frame_indices=frames, depth_mm=depth, original_depth_mm=case['original_depth'],
        physical_time_s=times, tip_xy=reference['tips_xy'], point_ids=case['ids'],
        legacy_trend_y=case['legacy'], um_per_px=np.array(case['um_per_px']),
        point_relative_velocity_xy=velocity.astype(np.float32), point_velocity_valid=valid,
        point_velocity_kind=kind, blade_velocity_xy=reference['velocity_xy_px_s'],
        group_names=np.asarray(GROUP_NAMES), state_names=np.asarray(STATE_NAMES))
    np.savez_compressed(folder/'loading_trends.npz', **trend)
    np.savez_compressed(folder/'blade_reference.npz', **{k:v for k,v in reference.items() if k != 'metadata'})
    save_json(folder/'blade_reference_summary.json', reference['metadata'])
    errors = np.array([r['y_error_px_s'] for r in checks])
    eligible_direction = [r for r in checks if r['magnitude_above_noise']]
    diffs = np.array([r['difference_px_s'] for r in comparisons])
    validation = dict(local_reconstruction_holdout=checks, fresh_image_comparison=comparisons,
        holdout_samples=len(checks), median_absolute_y_error_px_s=float(np.median(abs(errors))) if len(errors) else None,
        p95_absolute_y_error_px_s=float(np.percentile(abs(errors), 95)) if len(errors) else None,
        direction_agreement_above_2px_s=float(np.mean([r['direction_agrees'] for r in eligible_direction])) if eligible_direction else None,
        independent_image_comparison_samples=len(comparisons),
        median_absolute_saved_fresh_difference_px_s=float(np.median(abs(diffs))) if len(diffs) else None,
        interpretation='Held-out existing point velocities and fresh short image matches are consistency checks, not physical ground truth; coherent lubricant/refraction errors can persist.')
    save_json(folder/'validation.json', validation)
    summaries = []
    for g in range(4):
        state = trend['state'][:, g]
        summaries.append(dict(group=GROUP_NAMES[g], loading_frames=len(frames),
            legacy_available_pct=float(100*np.isfinite(case['legacy'][:, g]).mean()),
            direct_current_available_pct=float(100*np.isfinite(trend['direct_trend_xy'][:, g]).all(1).mean()),
            final_available_pct=float(100*np.mean(state > 0)),
            state_counts={STATE_NAMES[s]:int(np.sum(state == s)) for s in range(5)},
            method_counts={str(k):int(v) for k,v in zip(*np.unique(trend['method'][:, g], return_counts=True))}))
    summary = dict(case=name, frames=len(frames), source_frame_range=[int(frames[0]), int(frames[-1])],
        phase='loading_only', contact_to_reversal_mm=3., depth_range_mm=[float(depth[0]), float(depth[-1])],
        initial_id_tracks_unchanged=True, historical_identity_recovered_by_short_matching=False,
        current_region_is_confirmed_contact=False, um_per_px=case['um_per_px'], config=asdict(cfg),
        groups=summaries, blade_coordinate_interpolated_frames=int(np.sum(reference['state'] == 1)),
        blade_velocity_conditionally_estimated_frames=int(np.sum(reference['velocity_state'] == 1)),
        blade_mask_conditionally_estimated_frames=int(mask_estimated.sum()),
        blade_region_geometry_uncertain_frames=int(geometry_uncertain.sum()),
        new_image_pairs=len(image_report['pairs']), new_image_seconds=sum(r['seconds'] for r in image_report['pairs']),
        validation={k:v for k,v in validation.items() if k not in ('local_reconstruction_holdout', 'fresh_image_comparison')},
        source_track_sha256=digest(case['source']/'tracking/tracks.npz'),
        code_sha256={name:digest(ROOT/name) for name in ('src/workflows/motion.py',
            'src/analysis/loading_tip_trends.py', 'src/analysis/loading_blade_reference.py',
            'src/analysis/local_image_motion.py', 'src/analysis/local_blade_mask_repair.py')},
        interval_definition='Within-region interquartile spread or local-model residual sensitivity, never a statistical confidence interval',
        seconds=time.perf_counter()-started)
    save_json(folder/'measurement_summary.json', summary)
    with (folder/'loading_trends.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_frame', 'physical_time_s', 'depth_mm', 'group', 'state', 'method',
                         'relative_x_um_s', 'relative_y_um_s', 'observed_point_count', 'estimate_donor_count',
                         'blade_reference_state', 'blade_mask_estimated', 'blade_geometry_uncertain'])
        for row, frame in enumerate(frames):
            for g in range(4):
                vector = trend['relative_trend_xy'][row, g]*case['um_per_px']
                writer.writerow([frame, times[row], depth[row], GROUP_NAMES[g], STATE_NAMES[trend['state'][row, g]],
                    trend['method'][row, g], *[float(x) if np.isfinite(x) else '' for x in vector],
                    trend['observation_count'][row, g], trend['estimate_count'][row, g], reference['velocity_state'][row],
                    int(mask_estimated[row]), int(geometry_uncertain[row])])
    print(f'{name}: measurements completed {summary["seconds"]:.1f}s; coverage {[round(s["final_available_pct"],2) for s in summaries]}', flush=True)
    return summary


def render(name, args):
    from src.analysis.loading_tip_review import render_loading_tip_review
    folder, source = args.out/name, SOURCE/name
    trend = load_npz(folder/'loading_trends.npz')
    frames = trend['frame_indices']
    with np.load(source/'tracking/tracks.npz', allow_pickle=False) as stored:
        p, supported, ids = stored['positions'][frames], stored['supported'][frames], stored['point_ids']
    repair = loading_mask_reader(source, name, frames, trend['tip_xy'])
    full_interface = bool(trend.get('full_interface_scope', False))
    output = folder/f'{name}_loading_interface_sliding.mp4' if full_interface else folder/f'{name}_loading_tip_trends.mp4'
    slip = load_npz(folder/'surface_sliding.npz') if full_interface else None
    result = render_loading_tip_review(source/'normalized_lossless.mkv', p, supported, output,
        frame_indices=frames, depth_mm=trend['depth_mm'], tip_xy=trend['tip_xy'], point_ids=ids,
        trend=trend, blade_mask_reader=lambda frame: repair.read(frame)[0] == 2,
        overview_path=folder/f'{name}_loading_overview.png', fps=12,
        **({'slip': slip} if slip is not None else {}))
    result.update(encode_review_video(output))
    save_json(output.with_suffix('.render.json'), result)
    summary = json.loads((folder/'measurement_summary.json').read_text(encoding='utf-8'))
    summary.update(rendering=result, video=str(output))
    save_json(folder/'complete.json', summary)
    print(f'{name}: review complete', flush=True)


def measure_interface(name, args):
    from src.analysis.loading_tip_trends import current_interface_regions, estimate_interface_frame_trends
    from src.analysis.loading_surface_sliding import build_loading_surface_sliding
    cv2.setNumThreads(2)
    started = time.perf_counter()
    folder = args.out/name
    folder.mkdir(parents=True, exist_ok=True)
    case, cfg = data(name), LoadingTrendConfig()
    frames, p, reference = case['frames'], case['positions'], case['reference']
    times = case['times'][frames]
    repair, _ = prepare_current_masks(case, name)
    reader = lambda frame: repair.read(frame)[0]
    mask_estimated, geometry_uncertain = geometry_flags(repair, frames)
    velocity, valid, quality, kind, old_tip = point_motion(case)
    valid &= case['support'][frames]
    print(f'{name}: short-image observations along the whole visible interface', flush=True)
    extra, image_arrays, image_report = probe_images(case, reader, args,
        region_builder=lambda labels, tip: current_interface_regions(labels, tip, cfg)['group_map'] >= 0)
    if image_arrays:
        np.savez_compressed(folder/'short_image_samples.npz', **image_arrays)
    save_json(folder/'short_image_summary.json', image_report)
    query = load_npz(case['source']/'queries.npz')
    sparse = ~query['is_core_point'].astype(bool)
    records = []
    for row, frame in enumerate(frames):
        labels = reader(frame)
        result = estimate_interface_frame_trends(p[frame], velocity[row], valid[row], case['material'][row],
            reference['tips_xy'][row], labels, quality=quality[row], extra=extra.get(row), config=cfg)
        groups = result['group_index']
        result['sparse_observation_count'] = np.array([np.sum(valid[row] & sparse & (groups == g)) for g in range(4)])
        result['upper_observation_count'] = np.array([np.sum(valid[row] & (groups == g) &
            (p[frame, :, 1] < reference['tips_xy'][row, 1]-cfg.radius_px)) for g in range(4)])
        conditional_reference = geometry_uncertain[row] or reference['velocity_state'][row] < 2
        if conditional_reference:
            result['segment_state'][result['segment_state'] > 1] = 2
        for g in range(4):
            for segment in range(result['segment_state'].shape[1]):
                use = valid[row] & (groups == g) & (result['segment_index'] == segment)
                if result['segment_state'][g, segment] == 4 and use.any() and np.mean(kind[row, use] == 1) > .5:
                    result['segment_state'][g, segment] = 2
            if result['state'][g] == 4 and np.any((result['segment_state'][g] > 0) & (result['segment_state'][g] != 4)):
                result['state'][g] = 2
        records.append(result)
        if row % 300 == 0:
            print(f'{name}: full-interface trends {row}/{len(frames)}', flush=True)
    trend = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    finalize_series(trend, times, reference['velocity_state'], cfg)
    apply_geometry_flags(trend, mask_estimated, geometry_uncertain)
    depth = (reference['tips_xy'][:, 1]-reference['tips_xy'][0, 1])*case['um_per_px']/1000.
    trend.update(frame_indices=frames, physical_time_s=times, depth_mm=depth, tip_xy=reference['tips_xy'],
        tip_coordinate_state=reference['state'], point_ids=case['ids'], sparse_point_mask=sparse,
        um_per_px=np.array(case['um_per_px']), full_interface_scope=np.array(True),
        point_relative_velocity_xy=velocity.astype(np.float32), point_velocity_valid=valid,
        point_velocity_kind=kind, blade_velocity_xy=reference['velocity_xy_px_s'], group_names=np.asarray(GROUP_NAMES))
    if old_tip is not None:
        for key in ('relative_trend_xy', 'state', 'group_index', 'observation_count', 'interval_low_y', 'interval_high_y'):
            trend['tip_only_'+key] = old_tip[key]
    else:
        local = [estimate_frame_trends(p[frame], velocity[row], valid[row], case['gap'][row], case['material'][row],
            reference['tips_xy'][row], reader(frame), quality=quality[row], config=cfg) for row, frame in enumerate(frames)]
        for key in ('relative_trend_xy', 'state', 'group_index', 'observation_count'):
            trend['tip_only_'+key] = np.asarray([item[key] for item in local])
    np.savez_compressed(folder/'loading_trends.npz', **trend)
    save_json(folder/'blade_mask_repair.json', dict(estimated_source_frames=frames[mask_estimated].tolist(),
        geometry_uncertain_source_frames=frames[geometry_uncertain].tolist(), original_labels_unchanged=True))
    print(f'{name}: restoring fixed-material surface sliding estimates', flush=True)
    with np.load(MEASUREMENTS/name/'blade_contact_geometry.npz', allow_pickle=False) as stored:
        branches = stored['routing_branch'][frames]
    for row in np.flatnonzero((reference['state'] != 2) | mask_estimated):
        branches[row] = blade_contact_geometry(reader(frames[row]),p[frames[row]],
            reference['tips_xy'][row])['routing_branch']
    slip = build_loading_surface_sliding(load_npz(MEASUREMENTS/name/'surface_address.npz'), frames,
        reference['tips_xy'], surface_reference=load_npz(MEASUREMENTS/name/'initial_surface_extended.npz'),
        um_per_px=case['um_per_px'], seed_points=query['seed_points'], point_ids=case['ids'],
        positions=p[frames], supported=case['support'][frames], branch_labels=branches,
        blade_masks=lambda row: reader(frames[row]) == 2, tip_state=reference['state'],
        blade_geometry_uncertain=geometry_uncertain, contact_frame=int(frames[0]))
    slip.update(depth_mm=depth, physical_time_s=times)
    write_sliding_outputs(folder,slip)
    def number(value):
        return float(value) if np.isfinite(value) else ''
    with (folder/'loading_trends.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_frame', 'depth_mm', 'group', 'state', 'relative_x_um_s', 'relative_y_um_s',
            'observed_points', 'sparse_observed_points', 'above_old_tip_radius_points'])
        for row, frame in enumerate(frames):
            for g in range(4):
                writer.writerow([frame, depth[row], GROUP_NAMES[g], STATE_NAMES[trend['state'][row,g]],
                    *map(number, trend['relative_trend_xy'][row,g]*case['um_per_px']),
                    trend['observation_count'][row,g], trend['sparse_observation_count'][row,g], trend['upper_observation_count'][row,g]])
    with (folder/'interface_segments.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_frame', 'depth_mm', 'group', 'segment_above_tip', 'center_above_tip_px', 'state',
            'relative_x_um_s', 'relative_y_um_s', 'observed_points', 'fresh_image_points', 'surface_to_blade_gap_px'])
        for row, frame in enumerate(frames):
            for g, segment in np.argwhere(trend['segment_pixel_count'][row] > 0):
                writer.writerow([frame, depth[row], GROUP_NAMES[g], segment, trend['segment_center_above_tip_px'][row,segment],
                    STATE_NAMES[trend['segment_state'][row,g,segment]],
                    *map(number, trend['segment_relative_trend_xy'][row,g,segment]*case['um_per_px']),
                    trend['segment_observation_count'][row,g,segment], trend['segment_estimate_count'][row,g,segment],
                    number(trend['segment_surface_to_blade_gap_px'][row,g,segment])])
    groups = [dict(group=GROUP_NAMES[g], final_available_pct=float(100*np.mean(trend['state'][:,g]>0)),
        state_counts={label:int(np.sum(trend['state'][:,g]==s)) for s,label in enumerate(STATE_NAMES)},
        maximum_observed_sparse_points=int(trend['sparse_observation_count'][:,g].max()),
        maximum_observed_upper_points=int(trend['upper_observation_count'][:,g].max())) for g in range(4)]
    summary = dict(case=name, scope='whole_visible_blade_interface', phase='loading_only', frames=len(frames),
        source_frame_range=[int(frames[0]),int(frames[-1])],depth_range_mm=[float(depth[0]),float(depth[-1])],
        groups=groups, surface_sliding=slip['metadata'], config=asdict(cfg), initial_id_tracks_unchanged=True,
        original_sparse_ids_included=True, cached_point_velocity_reused=old_tip is not None,
        source_track_sha256=digest(case['source']/'tracking/tracks.npz'),
        code_sha256={path:digest(ROOT/path) for path in ('src/workflows/motion.py','src/analysis/loading_tip_trends.py',
            'src/analysis/loading_surface_sliding.py','src/analysis/local_image_motion.py')},
        new_image_pairs=len(image_report['pairs']), seconds=time.perf_counter()-started,
        current_region_is_confirmed_contact=False,
        interpretation='Observed material bands along the visible blade flanks, including original sparse points; conditional surface-address projection is kept separate from current velocity.')
    save_json(folder/'measurement_summary.json',summary)
    print(f'{name}: interface and sliding measurements complete in {summary["seconds"]:.1f}s',flush=True)
    return summary


def write_sliding_outputs(folder, slip):
    """Write the same identity estimate to the renderer and reviewable table."""
    np.savez_compressed(folder/'surface_sliding.npz', **{k:v for k,v in slip.items() if k != 'metadata'})
    save_json(folder/'surface_sliding_summary.json', slip['metadata'])
    def number(value):
        return float(value) if np.isfinite(value) else ''
    with (folder/'surface_sliding.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_frame', 'depth_mm', 'migration_um', 'state', 'evidence_kind', 'low_um', 'high_um',
            'initial_surface_x_px', 'initial_surface_y_px', 'normal_offset_px', 'support_count', 'reason'])
        for row, frame in enumerate(slip['frame_indices']):
            writer.writerow([frame, slip['depth_mm'][row], number(slip['migration_um'][row]), slip['state'][row],
                slip['evidence_kind'][row], number(slip['interval_low_um'][row]), number(slip['interval_high_um'][row]),
                *map(number, slip['address_initial_xy'][row]), number(slip['normal_offset_px'][row]),
                slip['support_count'][row], slip['reason'][row]])


def measure(name, args):
    return measure_interface(name,args) if args.scope == 'interface' else measure_tip(name,args)


def index(args):
    if getattr(args, 'scope', 'interface') == 'interface':
        return interface_index(args)
    cards = []
    for name in args.names:
        file = args.out/name/'complete.json'
        if not file.is_file():
            continue
        result = json.loads(file.read_text(encoding='utf-8'))
        rows = ''.join(f'<tr><td>{g["group"]}</td><td>{g["legacy_available_pct"]:.1f}%</td>'
            f'<td>{g["final_available_pct"]:.1f}%</td><td>{g["state_counts"]["supported_group"]}</td></tr>' for g in result['groups'])
        cards.append(f'<section><h2>{html.escape(name)}</h2>'
            f'<video controls preload="metadata" src="{name}/{name}_loading_tip_trends.mp4"></video>'
            f'<p><a href="{name}/loading_trends.csv">逐时刻趋势 CSV</a> · <a href="{name}/loading_trends.npz">NPZ</a> · '
            f'<a href="{name}/validation.json">留出点与短时图像一致性检查</a></p>'
            '<table><tr><th>当前区域</th><th>旧初始分组有曲线</th><th>本版有趋势（含估计）</th><th>直接组支持帧数</th></tr>'
            f'{rows}</table><p>覆盖率的选区和证据来源不同，不是单点精度提升百分比。</p></section>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>下压刀尖附近运动趋势</title><style>body{max-width:1450px;margin:30px auto;padding:0 22px;'
        'background:#121a1d;color:#e1ecec;font:17px/1.7 system-ui}video{width:100%}a{color:#77d7e9}'
        'td,th{border:1px solid #587078;padding:7px 16px}table{border-collapse:collapse}section{margin:30px 0}</style>'
        '<h1>只看下压：刀尖附近尽量完整的运动趋势</h1>'
        '<p>保留去阴影和原有材料点轨迹，新增短时图像补测；不重跑长期光流、不重新训练、不分析退刀。'
        '图例：实线为当前区域有充分支持；彩色虚线为少点、同侧邻域、短时图像或刀具参考/选区几何的条件估计；'
        '灰色点线仅表示 ≤0.065 秒的双端短桥接。此处虚线不再表示退刀。</p>'
        '<ol><li>按当前距刀面 0–12 px / 12–40 px、左右分区，默认刀尖半径60 px；内层必要时扩大到80 px。'
        '它表示当前近刀材料的运动，不等于已确认机械接触，也不用于识别初始表面身份。</li>'
        '<li>有1–3个实际支持点时保留条件趋势；薄层不足时仅从同侧近邻拟合局部速度，最多14 px外推，拒绝跨刀具传播。'
        '密集/压扁点按6 px空间单元减小重复权重，完全没有材料或观测支持时仍保留未知。</li>'
        '<li>每3个导入帧重新从当前纹理选点，作亮度归一化、金字塔LK、前后向和材料像素相关检查，困难点用仿射匹配补救。'
        '短时纹理运动避免长期漂移累积，但它不会恢复丢失的初始材料身份，油膜相干运动仍可能产生误差。'
        '<a href="https://publications.ri.cmu.edu/lucas-kanade-20-years-on-a-unifying-framework-part-3">亮度变化与图像配准原理</a>。</li>'
        '<li>刀具参考仅在下压区间内作双端有界插值。l60长缺口约0.4秒，已用等长遮挡测试核查登记自洽性，'
        '实际缺口仍标为条件估计。所有速度使用采集时间，12 fps仅为播放速度。</li>'
        '<li>刀具轮廓短暂漏掉刀尖时，用前后两个原始刀具轮廓按登记位移对齐，仅取刀尖90 px内的交集。'
        '它不扩充材料区域、不递推使用估计轮廓；相关趋势标为条件估计。n60在约1.65 mm处有3帧使用此补全。</li></ol>'
        '<p>l60和n60部分时段登记刀尖与可见刀具轮廓末端相差约8–16 px，与最薄层厚度同量级。'
        '画面不足以区分暗尖端漏识别与定位偏差，因此不强行延长轮廓；这些时段保留趋势值，并降为条件几何估计。</p>'
        '<p>趋势只作3点中值降噪，不强制单调，不用长时间匀速预测补满，不跨强烈方向反转桥接。'
        '更完整的曲线不意味着材料身份或物理精度已经验证；保留原始估计、来源、点数和敏感性范围供核查。</p>'
        '<p><a href="audit/loading_coverage_summary.json">下压缺失原因审计</a> · '
        '<a href="audit/blade_reference_validation_summary.json">刀具参考留段验证</a> · '
        '<a href="audit/0930n60a1_local_mask_repair_review.png">刀尖轮廓补全复核图</a></p>'
        + ''.join(cards) + '</html>')
    (args.out/'index.html').write_text(page, encoding='utf-8')


def interface_index(args):
    cards = []
    for name in args.names:
        file = args.out/name/'complete.json'
        if not file.is_file():
            continue
        result = json.loads(file.read_text(encoding='utf-8'))
        slip = result['surface_sliding']
        available = 100*(1-slip['unknown_frames']/result['frames'])
        rows = ''.join(f'<tr><td>{html.escape(g["group"])}</td><td>{g["final_available_pct"]:.1f}%</td>'
            f'<td>{g["maximum_observed_sparse_points"]}</td><td>{g["maximum_observed_upper_points"]}</td></tr>' for g in result['groups'])
        cards.append(f'<section><h2>{name}</h2><video controls preload="metadata" '
            f'poster="{name}/{name}_loading_overview_2mm.png" src="{name}/{name}_loading_interface_sliding.mp4"></video>'
            f'<p>表面地址迁移估计覆盖率：{available:.1f}%（包括条件材料块投影）；'
            f'原表面相容推算 {slip["strict_frames"]} 帧，材料块投影推测 {slip["conditional_projection_frames"]} 帧。</p>'
            '<table><tr><th>沿刀材料带</th><th>趋势覆盖率（含估计）</th><th>最多纳入稀疏观测点</th>'
            f'<th>最多纳入旧60px上方观测点</th></tr>{rows}</table>'
            f'<p><a href="{name}/surface_sliding.csv">滑移推测 CSV</a> · '
            f'<a href="{name}/loading_trends.csv">整条材料带趋势 CSV</a> · '
            f'<a href="{name}/interface_segments.csv">沿刀面分段趋势 CSV</a> · '
            f'<a href="{name}/surface_sliding_summary.json">滑移方法及来源</a> · '
            f'<a href="{name}/loading_trends.npz">完整趋势 NPZ（含原刀尖局部结果）</a></p></section>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>完整沿刀面观测与表面滑移估计</title><style>body{max-width:1500px;margin:28px auto;padding:0 22px;'
        'background:#121a1d;color:#e1ecec;font:17px/1.7 system-ui}video{width:100%}a{color:#77d7e9}'
        'td,th{border:1px solid #587078;padding:6px 14px}table{border-collapse:collapse}section{margin:34px 0}</style>'
        '<h1>完整沿刀面观测与表面滑移估计</h1>'
        '<p>仅下压0–3 mm。保留去阴影、原始固定材料ID和已有轨迹；沿左右刀面展开观测，包括上方稀疏区。'
        '左上为材料点叠加原视频，右上为纯点阵及分段相对运动箭头；中间为四组完整材料带的水平/竖直相对趋势；'
        '下方恢复初始斜面地址图和表面方向迁移曲线。</p>'
        '<ol><li>从当前材料靠刀面的边缘向内划分0–12 px表层和12–40 px内层，向上延伸至可见邻接区结束。'
        '材料与刀面的间隙≤8 px为较强几何支持，8–16 px作为条件近刀区域；这仍不等于独立确认机械接触。</li>'
        '<li>每48 px刀面高度单独计算运动，整组按已观测分段的实际长度汇总，密集点不会靠数量压过上方稀疏段。'
        '短时图像补测也扩展到全界面，各高度分段均衡选点。原60 px刀尖局部结果保存在NPZ的tip_only字段。</li>'
        '<li>滑移曲线表示刀尖所对应的初始材料地址沿原始斜面的迁移。原表面相容估计优先；不足时，用具有固定原始ID的'
        '同侧材料块反推刀尖位置，再投影到原始斜面，作为条件推测。投影点与原始表面的法向偏离、拟合残差、支撑点ID均保留。'
        '新取的短时纹理点仅补运动趋势，不参与原始身份推算。该推测不证明摩擦滑移或断裂后新表面的真实身份。</li>'
        '<li>当前图的接触标记始终画在登记刀尖；对应的材料地址只画在初始参考图。所有横坐标都是接触为0的下压深度。'
        '当前区域曲线实线为充分支持，彩色虚线为条件估计；滑移绿色为较强表面支持、粉色为条件推测，未知处断开。</li></ol>'
        '<p>保留真实方向变化。覆盖率包含估计，不代表物理精度；包络表示模型与几何敏感度，不是统计置信区间。</p>'
        + ''.join(cards) + '</html>')
    (args.out/'index.html').write_text(page, encoding='utf-8')


def main(argv=None):
    global SOURCE, MEASUREMENTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE, help='Prepared material tracks')
    parser.add_argument('--measurements', type=Path, default=MEASUREMENTS, help='Blade contact measurements')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--scope', choices=('interface','tip'), default='interface')
    parser.add_argument('--names', nargs='+', default=list(NAMES))
    parser.add_argument('--stage', choices=('measure', 'render', 'all'), default='all')
    parser.add_argument('--image-step', type=int, default=3)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    from src.video_sources import validate_case_names
    try:
        validate_case_names(args.names)
    except ValueError as error:
        parser.error(str(error))
    SOURCE, MEASUREMENTS = args.source.resolve(), args.measurements.resolve()
    if not 1 <= args.image_step <= 6:
        parser.error('--image-step must be 1..6')
    args.out = (args.out or (OUT if args.scope == 'interface' else TIP_OUT)).resolve()
    if args.out in (SOURCE.resolve(), MEASUREMENTS.resolve()):
        parser.error('Outputs must not overwrite previous tracks or analyses')
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    if args.worker:
        for name in args.names:
            if args.stage in ('all', 'measure'):
                measure(name, args)
            if args.stage in ('all', 'render'):
                render(name, args)
        return
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def worker(name):
        folder = args.out/name
        folder.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-u', '-m', 'src.workflows.motion', '--worker', '--names', name,
               '--stage', args.stage, '--scope', args.scope, '--image-step', str(args.image_step), '--out', str(args.out),
               '--source', str(SOURCE), '--measurements', str(MEASUREMENTS)]
        with (folder/f'{args.stage}.log').open('ab') as log:
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=HIDDEN)
        return dict(case=name, exit_code=code)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(worker, args.names))
    save_json(args.out/'batch_status.json', dict(stage=args.stage, results=results, completed=time.time()))
    index(args)
    if any(row['exit_code'] for row in results):
        raise SystemExit('A case failed; see per-case logs')


if __name__ == '__main__':
    main()
