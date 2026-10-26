"""Summarize saved contact curves without substituting an earlier endpoint."""
import numpy as np
import json


KINDS = {0: 'unavailable', 1: 'original_estimate', 2: 'conditional_interpolation'}

TAIL_LABELS = {
    'available_at_cutoff': '规定深度已有估计',
    'local_linear_extrapolation': '末段趋势外推',
    'near_zero_curve_hold': '末段曲线近零：保持外推',
    'insufficient_trend_hold_assumption': '趋势不足：保持假设',
    'initialization_only': '仅有初始化，无法外推',
    'no_available_estimate': '无有效依据',
}


def summarize_curve(depth_mm, offset_um, estimate_kind, cutoff_um):
    depth, offset, kind = map(np.asarray, (depth_mm, offset_um, estimate_kind))
    if depth.ndim != 1 or offset.shape != depth.shape or kind.shape != depth.shape:
        raise ValueError('Curve arrays must be matching one-dimensional arrays')
    valid = np.isfinite(depth) & np.isfinite(offset) & (kind > 0)
    available = np.flatnonzero(valid)
    last = int(available[-1]) if available.size else None
    endpoint = bool(len(depth) and valid[-1])
    return dict(
        table_endpoint_um=float(offset[-1]) if endpoint else None,
        table_endpoint_depth_um=float(depth[-1]*1000) if len(depth) and np.isfinite(depth[-1]) else None,
        table_endpoint_kind=KINDS[int(kind[-1])] if endpoint else 'unavailable',
        last_estimate_um=float(offset[last]) if last is not None else None,
        last_estimate_depth_um=float(depth[last]*1000) if last is not None else None,
        last_estimate_kind=KINDS[int(kind[last])] if last is not None else 'unavailable',
        remaining_depth_to_cutoff_um=max(0., float(cutoff_um-depth[last]*1000)) if last is not None else None,
        completed_coverage=float(valid.mean()) if len(depth) else 0.,
        interpolated_samples=int(((kind == 2) & valid).sum()))


def extrapolate_cutoff(data, cutoff_um):
    """Separate, explicitly conditional table estimate; never writes a track.

    Fit only actual update samples in the final contiguous available segment.
    Three depth windows expose choice sensitivity, not statistical accuracy.
    A prediction-only flat tail cannot establish that material slip stopped.
    """
    x, y = np.asarray(data['depth_mm']), np.asarray(data['offset_um'])
    target = float(cutoff_um)/1000
    available = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= target+1e-9)
    updates = (np.asarray(data['transport_absolute_used'], bool) |
               np.asarray(data['transport_velocity_used'], bool))
    indices = np.flatnonzero(available)
    result = dict(tail_estimate_um=None, tail_estimate_kind='no_available_estimate',
        tail_target_depth_um=float(cutoff_um), tail_extrapolation_depth_um=None,
        tail_fit_span_um=0., tail_supported_samples=0, tail_slope_um_per_mm=None,
        tail_window_min_um=None, tail_window_max_um=None,
        tail_model_choice_min_um=None, tail_model_choice_max_um=None,
        tail_extrapolation_exceeds_fit=False, tail_anchor_sensitivity_um=None,
        tail_fit_details_json='[]', tail_is_conditional=True)
    if not len(indices):
        return result
    end = int(indices[-1])
    horizon = max(0., target-float(x[end]))
    result['tail_extrapolation_depth_um'] = horizon*1000
    if 'sigma_um' in data and np.isfinite(data['sigma_um'][end]):
        result['tail_anchor_sensitivity_um'] = float(data['sigma_um'][end])
    if horizon <= 1e-9:
        result.update(tail_estimate_um=float(y[end]), tail_estimate_kind='available_at_cutoff',
                      tail_is_conditional=False)
        return result
    # A filter initialized to zero then predicting zero is not a zero-slip observation.
    if not (updates & available).any():
        result['tail_estimate_kind'] = 'initialization_only'
        return result
    start = end
    while start > 0 and available[start-1]:
        start -= 1
    segment = np.arange(start, end+1)
    support = segment[updates[segment]]
    result['tail_supported_samples'] = int(((x[support] >= x[end]-.3)).sum())
    resolution = float(np.nanmedian(data['um_per_px']))
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('Positive pixel-to-micron calibration required for near-zero curve test')
    fits, seen = [], set()
    for window in (.1, .2, .3):
        keep = support[x[support] >= x[end]-window]
        signature = tuple(keep.tolist())
        if len(keep) < 5 or signature in seen:
            continue
        span = float(np.ptp(x[keep]))
        if span < .03:
            continue
        seen.add(signature)
        xx, yy = x[keep], y[keep]
        # Median pairwise slope tolerates isolated update jumps; do not clip its sign.
        i, j = np.triu_indices(len(keep), 1)
        dx = xx[j]-xx[i]
        good = np.abs(dx) >= .25*span
        if not good.any():
            continue
        slope = float(np.median((yy[j]-yy[i])[good]/dx[good]))
        fit = dict(window_mm=window, span_mm=span, samples=int(len(keep)),
            slope_um_per_mm=slope, target_um=float(y[end]+slope*horizon),
            near_zero_curve=bool(abs(slope)*span <= resolution and
                                 np.ptp(yy) <= 2*resolution),
            start_depth_mm=float(xx[0]), end_depth_mm=float(xx[-1]))
        fits.append(fit)
    if not fits:
        # Explicit value-preservation scenario, never described as observed sticking.
        result.update(tail_estimate_um=float(y[end]),
                      tail_estimate_kind='insufficient_trend_hold_assumption',
                      tail_extrapolation_exceeds_fit=True)
        return result
    chosen = min(fits, key=lambda item: abs(item['window_mm']-.2))
    stable = (len(fits) >= 2 and max(f['span_mm'] for f in fits) >= .1 and
              all(f['near_zero_curve'] for f in fits))
    value = float(y[end]) if stable else chosen['target_um']
    predictions = [f['target_um'] for f in fits]
    result.update(tail_estimate_um=value,
        tail_estimate_kind='near_zero_curve_hold' if stable else 'local_linear_extrapolation',
        tail_fit_span_um=chosen['span_mm']*1000,
        tail_slope_um_per_mm=0. if stable else chosen['slope_um_per_mm'],
        tail_window_min_um=min(predictions), tail_window_max_um=max(predictions),
        tail_model_choice_min_um=min([float(y[end]), *predictions]),
        tail_model_choice_max_um=max([float(y[end]), *predictions]),
        tail_extrapolation_exceeds_fit=bool(horizon > chosen['span_mm']),
        tail_fit_details_json=json.dumps(fits))
    return result


def tail_estimate_markup(result):
    """Compact, identically labelled table cell in both HTML reports."""
    value, kind = result.get('tail_estimate_um'), result.get('tail_estimate_kind', 'no_available_estimate')
    label = TAIL_LABELS[kind]
    if value is None:
        return f'—<small class="depth-gap">{label}</small>'
    text = f'<strong>{value:.1f}</strong><small class="depth-gap">{label}</small>'
    horizon = result.get('tail_extrapolation_depth_um', 0.)
    text += f'<small class="depth-gap">外推跨度 {horizon:.1f} μm</small>'
    if kind == 'insufficient_trend_hold_assumption':
        return text+'<small class="depth-gap">仅数值保持情景，未证实停止滑移</small>'
    fits = json.loads(result.get('tail_fit_details_json', '[]'))
    if fits:
        text += ('<details class="fit-details"><summary>拟合依据 / 范围</summary>'
                 f'<small class="depth-gap">末段有效更新 {result["tail_supported_samples"]} 点；'
                 f'选用拟合跨度 {result["tail_fit_span_um"]:.1f} μm</small>'
                 f'<small class="depth-gap">{len(fits)} 个不同区间趋势外推：'
                 f'{result["tail_window_min_um"]:.1f}–{result["tail_window_max_um"]:.1f} μm</small>'
                 f'<small class="depth-gap">保持/趋势的情景范围：{result["tail_model_choice_min_um"]:.1f}–'
                 f'{result["tail_model_choice_max_um"]:.1f} μm；非误差界限</small>')
        if result['tail_extrapolation_exceeds_fit']:
            text += '<small class="depth-gap">外推跨度大于拟合跨度，远期依赖假设</small>'
        text += '</details>'
    return text


def interpolate_internal_gaps(depth_mm, offset_um, *, eligible=None):
    """Report-only two-sided interpolation; never an observation or endpoint extrapolation."""
    depth = np.asarray(depth_mm, dtype=float)
    original = np.asarray(offset_um, dtype=float)
    if depth.ndim != 1 or original.shape != depth.shape:
        raise ValueError('Depth and offset must be equal-length one-dimensional arrays')
    allowed = np.ones(depth.shape, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if allowed.shape != depth.shape:
        raise ValueError('Eligibility must match depth')
    completed = original.copy()
    kind = np.isfinite(original).astype(np.uint8)  # 0 unavailable, 1 original, 2 interpolated
    gap_id = np.zeros(depth.shape, dtype=np.int32)
    left_index = np.full(depth.shape, -1, dtype=np.int32)
    right_index = left_index.copy()
    alpha = np.full(depth.shape, np.nan)
    anchors = np.flatnonzero(np.isfinite(original) & np.isfinite(depth) & allowed)
    gaps = []
    for left, right in zip(anchors[:-1], anchors[1:]):
        if right == left + 1:
            continue
        interval = slice(left, right + 1)
        inside = slice(left + 1, right)
        row = dict(left_index=int(left), right_index=int(right), missing_samples=int(right-left-1),
                   left_depth_mm=float(depth[left]), right_depth_mm=float(depth[right]),
                   span_mm=float(depth[right]-depth[left]),
                   left_offset_um=float(original[left]), right_offset_um=float(original[right]))
        reason = None
        if not allowed[interval].all():
            reason = 'outside_eligible_interval'
        elif np.isfinite(original[inside]).any():
            reason = 'interior_value_with_invalid_depth'  # Never replace an existing estimate.
        elif not np.isfinite(depth[interval]).all():
            reason = 'missing_depth_coordinate'
        elif depth[right] <= depth[left]:
            reason = 'nonpositive_depth_span'
        elif ((depth[inside] < depth[left]) | (depth[inside] > depth[right])).any():
            reason = 'depth_outside_anchor_bracket'
        row['filled'] = reason is None
        row['reason'] = reason or 'conditional_linear_interpolation_in_depth'
        row['depth_backsteps'] = int((np.diff(depth[interval]) < 0).sum())
        gaps.append(row)
        if reason:
            continue
        weight = (depth[inside]-depth[left])/(depth[right]-depth[left])
        completed[inside] = (1-weight)*original[left] + weight*original[right]
        kind[inside] = 2
        gap_id[inside] = len(gaps)
        left_index[inside], right_index[inside] = left, right
        alpha[inside] = weight
    return dict(offset_original_um=original.copy(), offset_completed_um=completed,
                estimate_kind=kind, interpolation_gap_id=gap_id,
                left_anchor_index=left_index, right_anchor_index=right_index,
                interpolation_alpha=alpha, gaps=gaps)
