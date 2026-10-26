"""Conditional blade reference repair within an explicitly sliced loading phase.

Callers must pass contact..reversal only. This module neither reads withdrawal
frames nor extrapolates coordinates beyond observed endpoints. A repaired blade
reference does not turn unsupported material tracks into observations.
"""
from __future__ import annotations

import re
import numpy as np


def _runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _robust_line(dt, xy):
    """Joint two-coordinate Huber reweighting with a free intercept."""
    design = np.column_stack((np.ones(len(dt)), dt))
    weight = np.ones(len(dt))
    fit = np.full((2, 2), np.nan)
    for _ in range(4):
        root = np.sqrt(weight)
        fit = np.linalg.lstsq(design * root[:, None], xy * root[:, None], rcond=None)[0]
        residual = np.linalg.norm(xy - design @ fit, axis=1)
        # Median radial residual is deliberately floored: near-exact smooth
        # trajectories must not make floating point noise dominate weights.
        scale = max(.01, 1.4826 * float(np.median(residual)))
        weight = np.minimum(1., 2.5 * scale / np.maximum(residual, 1e-12))
    rms = float(np.sqrt(np.mean(np.sum((xy - design @ fit) ** 2, axis=1))))
    return fit[1], rms


def prepare_loading_blade_reference(tips_xy, supported, times_s, *,
                                    max_gap_s=.5, half_window_s=.075,
                                    minimum_samples=5, minimum_span_s=.025):
    """Repair bounded blade gaps and estimate loading-only blade translation.

    The input arrays must already contain only first contact through reversal.
    Original supported coordinates are copied without alteration. Missing
    coordinates are interpolated in physical time only when both bounding
    observations exist and their end-to-end interval is <= ``max_gap_s``.

    Coordinate ``state``: 2 supplied observation, 1 interpolation, 0 unknown.
    ``velocity_state``: 2 robust fit on original observations with a balanced
    two-sided window; 1 a fit containing an interpolation or using an
    asymmetric/one-sided window; 0 unavailable. Those states express reference
    provenance, not certified physical accuracy. No angular motion is inferred.
    """
    raw = np.asarray(tips_xy, dtype=float)
    observed = np.asarray(supported, dtype=bool)
    times = np.asarray(times_s, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or observed.shape != (len(raw),) or times.shape != (len(raw),):
        raise ValueError('Expected T,2 tips, T support and T physical timestamps')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Physical timestamps must be finite and strictly increasing')
    if (not np.isfinite(max_gap_s) or max_gap_s <= 0 or not np.isfinite(half_window_s)
            or half_window_s <= 0 or minimum_samples < 3 or minimum_span_s <= 0):
        raise ValueError('Invalid interpolation or derivative parameters')
    observed = observed & np.isfinite(raw).all(axis=1)
    repaired = np.full_like(raw, np.nan)
    repaired[observed] = raw[observed]
    state = np.zeros(len(raw), dtype=np.int8)
    state[observed] = 2
    gaps = []
    for lo, hi in _runs(~observed):
        bounded = lo > 0 and hi < len(raw)
        span = float(times[hi] - times[lo - 1]) if bounded else None
        filled = bounded and span <= max_gap_s + 1e-12
        if filled:
            fraction = (times[lo:hi] - times[lo - 1]) / span
            repaired[lo:hi] = (1. - fraction[:, None]) * raw[lo - 1] + fraction[:, None] * raw[hi]
            state[lo:hi] = 1
        gaps.append(dict(start_index=int(lo), stop_index_exclusive=int(hi),
            missing_frames=int(hi - lo),
            missing_sample_span_s=float(times[hi - 1] - times[lo]),
            endpoint_span_s=span, bounded=bool(bounded), interpolated=bool(filled)))

    velocity = np.full_like(repaired, np.nan)
    velocity_state = np.zeros(len(raw), dtype=np.int8)
    velocity_rms = np.full(len(raw), np.nan)
    velocity_samples = np.zeros(len(raw), dtype=np.int16)
    velocity_span = np.full(len(raw), np.nan)
    single_sided = np.zeros(len(raw), bool)
    # Fits cannot jump over an unfilled position gap. At segment boundaries
    # a one-sided fit is permitted but explicitly conditional.
    for begin, end in _runs(state > 0):
        for i in range(begin, end):
            lo = max(begin, int(np.searchsorted(times, times[i] - half_window_s)))
            hi = min(end, int(np.searchsorted(times, times[i] + half_window_s, side='right')))
            if hi - lo < minimum_samples:
                continue
            earlier = times[i] - times[lo]
            later = times[hi - 1] - times[i]
            # Prefer a symmetric fit when a sufficiently sampled balanced
            # window exists. Boundary-only one-sided estimates stay usable.
            radius = min(earlier, later)
            balanced = np.flatnonzero(np.abs(times[lo:hi] - times[i]) <= radius + 1e-12) + lo
            can_balance = (len(balanced) >= minimum_samples and
                times[balanced[-1]] - times[balanced[0]] >= minimum_span_s)
            indices = balanced if can_balance else np.arange(lo, hi)
            span = float(times[indices[-1]] - times[indices[0]])
            if span < minimum_span_s:
                continue
            dt = times[indices] - times[i]
            velocity[i], velocity_rms[i] = _robust_line(dt, repaired[indices])
            velocity_samples[i] = len(indices)
            velocity_span[i] = span
            single_sided[i] = not (np.any(dt < 0) and np.any(dt > 0))
            directly_observed = can_balance and np.all(state[indices] == 2)
            velocity_state[i] = 2 if directly_observed else 1
    bounded_spans = [row['endpoint_span_s'] for row in gaps if row['bounded']]
    metadata = dict(schema='loading_blade_reference_v1',
        input_phase='Caller-sliced first contact through reversal; no withdrawal samples',
        coordinate_states={'0': 'unknown', '1': 'bounded temporal interpolation', '2': 'original observation'},
        velocity_states={'0': 'unknown', '1': 'conditional interpolated or boundary/asymmetric fit',
                         '2': 'balanced original-observation fit'},
        observed_coordinates_modified=False, extrapolated_coordinates=0,
        interpolation='Linear in physical time, never a material-motion assumption',
        maximum_endpoint_gap_s=float(max_gap_s), half_window_s=float(half_window_s),
        minimum_samples=int(minimum_samples), minimum_span_s=float(minimum_span_s),
        original_observed_frames=int(observed.sum()), interpolated_frames=int((state == 1).sum()),
        unknown_frames=int((state == 0).sum()),
        longest_missing_frames=max((row['missing_frames'] for row in gaps), default=0),
        longest_bounded_endpoint_span_s=max(bounded_spans, default=0.), gaps=gaps,
        accuracy_note='Reference registration self-consistency is not ground-truth blade or material accuracy.')
    return dict(tips_xy=repaired, state=state, velocity_xy_px_s=velocity,
        velocity_state=velocity_state, velocity_fit_rms_px=velocity_rms,
        velocity_samples=velocity_samples, velocity_span_s=velocity_span,
        velocity_single_sided=single_sided, metadata=metadata)


def guard_loading_reference(tips, supported, times, *, window_s=.6, tolerance_px=8., max_gap_s=.25):
    xy = np.asarray(tips, float)
    t = np.asarray(times, float)
    good = np.asarray(supported, bool) & np.isfinite(xy).all(1)
    if xy.shape != (len(t), 2) or good.shape != (len(t),):
        raise ValueError('T blade coordinates and support required')
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Increasing physical times required')
    rejected = np.zeros(len(t), bool)
    guide = np.full_like(xy, np.nan)
    residual = np.full(len(t), np.nan)
    # Median local slopes supply a high-breakdown initial line. Fitting all
    # points by ordinary LS first would let the false shaft matches move it.
    for i in np.flatnonzero(good):
        idx = np.flatnonzero(good & (abs(t-t[i]) <= window_s/2))
        if len(idx) < 7 or t[idx[-1]]-t[idx[0]] < .1:
            continue
        stride = max(1, len(idx)//4)
        dt = t[idx[stride:]]-t[idx[:-stride]]
        slope = np.median((xy[idx[stride:]]-xy[idx[:-stride]])/dt[:, None], axis=0)
        shifted = xy[idx]-(t[idx]-t[i])[:, None]*slope
        center = np.median(shifted, axis=0)
        dist = np.linalg.norm(shifted-center, axis=1)
        spread = 1.4826*np.median(dist)
        guide[i] = center
        residual[i] = np.linalg.norm(xy[i]-center)
        rejected[i] = residual[i] > max(tolerance_px, 6*spread)
    # Known contact..reversal loading: a coordinate above first contact or
    # beyond reversal by >8 px is incompatible with this calibrated interval.
    if good[0] and good[-1] and xy[-1, 1] > xy[0, 1]+100:
        rejected |= good & ((xy[:, 1] < xy[0, 1]-tolerance_px)
                           | (xy[:, 1] > xy[-1, 1]+tolerance_px))
    accepted = good & ~rejected
    ref = prepare_loading_blade_reference(xy, accepted, t, max_gap_s=max_gap_s)
    # Geometry windows that contain a rejected tip cannot claim directly
    # observed relative motion, even if a short coordinate gap was bridged.
    derivative_affected = np.zeros(len(t), bool)
    for time in t[rejected]:
        derivative_affected |= abs(t-time) <= .075+1e-12
    ref.update(rejected=rejected, accepted=accepted, diagnostic_guide=guide,
               residual_px=residual, derivative_affected=derivative_affected)
    ref['metadata'].update(guard_window_s=window_s, guard_tolerance_px=tolerance_px,
        rejected_original_observations=int(rejected.sum()),
        repaired_coordinates_are_observations=False,
        limitation='Temporal/image-registration consistency only; original contact and reversal calibration retained.')
    return ref


def restrict_loading_case(case, raw_tips, tip_supported, stop_depth_mm):
    """Copy a saved loading case and discard all samples past its first crossing.

    Calibration (um_per_px) is supplied by the caller. Endpoint selection uses
    supplied observed blade coordinates, not a full-record temporal smoother.
    An unsupported terminal tip is never filled with an after-cutoff witness.
    Source track coordinates and support flags remain read-only inputs.
    """
    target = float(stop_depth_mm)
    if not np.isfinite(target) or target <= 0:
        raise ValueError('stop_depth_mm must be finite and positive')
    frames = np.asarray(case['frames'], int)
    tips = np.asarray(raw_tips, float)[frames]
    good = np.asarray(tip_supported, bool)[frames] & np.isfinite(tips).all(1)
    if len(frames) < 2 or not good[0]:
        raise ValueError('Observed first-contact blade coordinate is required')
    depth = (tips[:, 1]-tips[0, 1])*float(case['um_per_px'])/1000.
    crossed = np.flatnonzero(good & (depth > target+1.e-12))
    boundary = int(crossed[0]) if len(crossed) else len(frames)
    candidates = np.flatnonzero(good[:boundary] & (depth[:boundary] <= target+1.e-12))
    if len(candidates) < 2:
        raise ValueError('Too few observed samples before the requested depth')
    count = int(candidates[-1])+1
    end = int(frames[count-1])
    result = dict(case)
    for key in ('positions', 'original_support', 'support', 'edges', 'times'):
        result[key] = case[key][:end+1].copy()
    for key in ('frames', 'gap', 'material', 'legacy', 'original_depth'):
        result[key] = case[key][:count].copy()
    result['reference'] = prepare_loading_blade_reference(
        tips[:count], good[:count], result['times'][frames[:count]])
    result['cutoff'] = dict(requested_depth_mm=target, actual_depth_mm=float(depth[count-1]),
        contact_source_frame=int(frames[0]), maximum_analysis_source_frame=end,
        analysis_loading_samples=count, endpoint_tip_observed=True,
        selection='Last observed frame before the first observed depth greater than the cutoff',
        post_cutoff_velocity_samples=0, post_cutoff_mask_witnesses=0,
        post_cutoff_fusion_samples=0,
        calibration='Existing contact-to-reversal stroke calibration; full stroke is 3 mm',
        flow_note='Saved optical tracking is reused; its own historical tracking dependencies are unchanged')
    return result


def group_cutoff(name):
    match = re.fullmatch(r'\d{4}[ln](30|33|57|60)(a\d+|c\d+(up|down))', name.lower())
    if match is None:
        raise ValueError('No declared fracture-depth group for '+name)
    angle, kind, direction = match.groups()
    if angle in ('30', '33'):
        return dict(group='30', depth_um=1300, depth_mm=1.3,
                    angle_alias=angle, source='user_declared_group_cutoff')
    group, value = ('60a', 2200) if kind.startswith('a') else (
        ('60up', 550) if direction == 'up' else ('60down', 850))
    return dict(group=group, depth_um=value, depth_mm=value/1000.,
                angle_alias=angle, source='user_declared_group_cutoff')

def guarded_prefix(case, raw_tips, tip_supported, stop_depth_mm):
    """Guard blade coordinates using only the already restricted prefix.

    First contact/reversal set the inherited 3-mm calibration. A raw observed
    first crossing chooses the prefix; the crossing frame is not used for a
    derivative, mask repair, spatial fit, or temporal observation.
    """
    prefix = restrict_loading_case(case, raw_tips, tip_supported, stop_depth_mm)
    for _ in range(8):
        frames = prefix['frames']
        guard = guard_loading_reference(np.asarray(raw_tips)[frames],
            np.asarray(tip_supported)[frames], prefix['times'][frames])
        if not guard['accepted'][0]:
            raise ValueError('Declared-prefix guard rejects first-contact calibration')
        accepted = np.flatnonzero(guard['accepted'])
        if len(accepted) < 2:
            raise ValueError('Too few accepted blade samples in declared prefix')
        count = int(accepted[-1])+1
        if count == len(frames):
            break
        end = int(frames[count-1])
        for key in ('positions', 'original_support', 'support', 'edges', 'times'):
            prefix[key] = prefix[key][:end+1].copy()
        for key in ('frames', 'gap', 'material', 'legacy', 'original_depth'):
            prefix[key] = prefix[key][:count].copy()
    else:
        raise ValueError('Could not establish an observed guarded prefix endpoint')
    original = prefix['reference']['tips_xy'][:len(frames)]
    known = np.isfinite(guard['tips_xy']).all(1)
    changed = known & (np.linalg.norm(guard['tips_xy']-original, axis=1) > .01)
    guard['analysis_qualified'] = known & ~guard['derivative_affected'] & ~changed
    guard['source_frames'] = frames.copy()
    guard['raw_tips'] = np.asarray(raw_tips)[frames].copy()
    guard['metadata']['input_phase'] = 'First contact through user-declared fracture-depth prefix'
    prefix['reference'] = guard
    actual = (guard['tips_xy'][-1, 1]-guard['tips_xy'][0, 1])*prefix['um_per_px']/1000.
    if actual > stop_depth_mm+1.e-10:
        raise AssertionError('Accepted endpoint exceeds the declared depth')
    prefix['cutoff'].update(actual_depth_mm=float(actual), maximum_analysis_source_frame=int(frames[-1]),
        analysis_loading_samples=len(frames), endpoint_tip_observed=True,
        blade_guard_uses_post_cutoff_samples=False,
        requested_depth_kind='User-declared fracture cutoff, not image-detected fracture',
        guarded_original_rejections=int(guard['rejected'].sum()))
    return prefix
