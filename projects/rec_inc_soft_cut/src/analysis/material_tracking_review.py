"""Standalone single-method material-track review and lightweight diagnostics.

Only saved trajectories, support flags and a display video are needed. No neural
tracker, competing method or raw-image remeasurement is imported or executed.
The renderer emits an OpenCV MP4; the batch runner may transcode it to H.264.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.analysis.blade_follow_camera import blade_follow_camera, crop_for_display, point_in_display


@dataclass(frozen=True)
class MaterialReviewConfig:
    view_mode: str = "tip"  # "full" keeps the complete source image visible
    crop_size: tuple = (320, 240)
    panel_size: tuple = (640, 480)
    point_radius: int = 2
    point_alpha: float = .78
    trail_length: int = 3
    color_count: int = 48
    codec: str = "mp4v"


def _inputs(positions, supported, point_ids, frame_indices):
    xy = np.asarray(positions, float)
    if xy.ndim != 3 or xy.shape[-1] != 2 or min(xy.shape[:2]) < 1:
        raise ValueError("positions must have nonempty T,N,2 geometry")
    support = np.asarray(supported, bool)
    if support.shape != xy.shape[:2]:
        raise ValueError("supported must address every fixed time and material ID")
    ids = np.arange(xy.shape[1]) if point_ids is None else np.asarray(point_ids)
    if ids.shape != (xy.shape[1],) or len(np.unique(ids)) != len(ids):
        raise ValueError("point_ids must contain one unique ID per trajectory")
    frames = np.arange(len(xy)) if frame_indices is None else np.asarray(frame_indices)
    if (frames.shape != (len(xy),) or not np.isfinite(frames).all()
            or not np.all(frames == np.floor(frames)) or frames[0] < 0 or np.any(np.diff(frames) <= 0)):
        raise ValueError("frame_indices must be strictly increasing nonnegative source indices")
    return xy, support, ids, frames.astype(int)


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def _stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return dict(count=int(a.size), median=float(np.median(a)) if a.size else None,
                p95=float(np.percentile(a, 95)) if a.size else None,
                p99=float(np.percentile(a, 99)) if a.size else None)


def _phase_partition(frames, depth, *, contact_frame=None, reversal_frame=None, phase_masks=None):
    """Partition a complete cycle without assuming depth remains monotonic.

    Explicit landmarks are SOURCE-frame numbers, not track-array offsets.
    Explicit masks take precedence; omitted mask regions stay unclassified.
    The fallback peak is an estimate only and is reported as such.
    """
    length = len(frames)
    for value in (contact_frame, reversal_frame):
        if value is not None and (not np.isfinite(value) or int(value) != value or value < 0):
            raise ValueError("Contact and reversal landmarks must be nonnegative source-frame integers")
    if contact_frame is not None and reversal_frame is not None and reversal_frame < contact_frame:
        raise ValueError("Reversal must not precede contact")
    known_contact = None
    if contact_frame is not None:
        known_contact = int(contact_frame)
        contact_source = "explicit_source_frame"
    elif depth is not None:
        eligible = np.flatnonzero(np.isfinite(depth) & (depth >= 0))
        known_contact = int(frames[eligible[0]]) if eligible.size else None
        contact_source = "first_nonnegative_depth_estimate" if eligible.size else "unknown"
    else:
        contact_source = "unknown"
    known_reversal = None
    if reversal_frame is not None:
        known_reversal = int(reversal_frame)
        reversal_source = "explicit_source_frame"
    elif known_contact is not None and depth is not None:
        eligible = np.flatnonzero(np.isfinite(depth) & (frames >= known_contact))
        if eligible.size:
            peak = eligible[np.argmax(depth[eligible])]
            known_reversal = int(frames[peak])
        reversal_source = "observed_depth_peak_estimate" if known_reversal is not None else "unknown"
    else:
        reversal_source = "unknown"
    phases = {name: np.zeros(length, bool) for name in ("precontact", "loading", "withdrawal", "unclassified")}
    if phase_masks is not None:
        if not isinstance(phase_masks, dict) or set(phase_masks) - set(phases):
            raise ValueError("phase_masks keys must be precontact, loading, withdrawal or unclassified")
        assigned = np.zeros(length, bool)
        for name, values in phase_masks.items():
            mask = np.asarray(values)
            if mask.shape != (length,) or mask.dtype != bool:
                raise ValueError("Explicit phase masks must be boolean arrays of length T")
            if np.any(assigned & mask):
                raise ValueError("Explicit phase masks must not overlap")
            phases[name] = mask.copy()
            assigned |= mask
        phases["unclassified"] |= ~assigned
        source = "explicit_phase_masks"
    else:
        if known_contact is not None:
            phases["precontact"] = frames < known_contact
            if known_reversal is not None:
                phases["loading"] = (frames >= known_contact) & (frames <= known_reversal)
                phases["withdrawal"] = frames > known_reversal
        elif known_reversal is not None:
            phases["withdrawal"] = frames > known_reversal
        phases["unclassified"] = ~np.logical_or.reduce(list(phases.values()))
        source = "source_landmarks" if contact_source == reversal_source == "explicit_source_frame" else "partly_or_fully_estimated"
    labels = np.full(length, "unclassified", dtype="U12")
    for name, mask in phases.items():
        labels[mask] = name
    contact_index = int(np.searchsorted(frames, known_contact)) if known_contact is not None else None
    if contact_index is not None and contact_index >= length:
        contact_index = None
    peak_index = int(np.searchsorted(frames, known_reversal, side="right")-1) if known_reversal is not None else None
    if peak_index is not None and peak_index < 0:
        peak_index = None
    metadata = dict(source=source, contact_source=contact_source, reversal_source=reversal_source,
                    contact_source_frame=known_contact, reversal_source_frame=known_reversal,
                    contact_track_index=contact_index, reversal_track_index=peak_index,
                    counts={name: int(mask.sum()) for name, mask in phases.items()},
                    interpretation="Source landmarks/masks define phase; missing depth inside a phase does not split it. Peak-based fallback is an estimate, not proof of reversal.")
    return phases, labels, metadata


def _support_gap_summary(measured, member, frames, ids, depth, times, scope):
    bad = ~measured & member[:, None]
    longest = np.zeros(measured.shape[1], int)
    events = []
    for point in range(measured.shape[1]):
        transitions = np.diff(np.r_[False, bad[:, point], False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
            run = int(stop-start)
            longest[point] = max(longest[point], run)
            events.append(dict(point_id=_scalar(ids[point]), samples=run,
                first_source_frame=int(frames[start]), last_source_frame=int(frames[stop-1]),
                start_depth_mm=None if depth is None else _scalar(depth[start]),
                end_depth_mm=None if depth is None else _scalar(depth[stop-1]),
                observed_time_span_s=None if times is None else float(times[stop-1]-times[start])))
    return dict(scope=scope, selected_frames=int(member.sum()), maximum_samples=int(longest.max()),
                points_gt12=int(np.sum(longest > 12)), points_gt30=int(np.sum(longest > 30)),
                event_count=len(events), longest_events=sorted(events, key=lambda r: r['samples'], reverse=True)[:20])


def _coverage(finite, measured, member):
    total = int(member.sum()) * measured.shape[1]
    return dict(frames=int(member.sum()), scheduled_observations=total,
                finite_fraction=float(finite[member].mean()) if total else None,
                supported_fraction=float(measured[member].mean()) if total else None,
                predicted_finite_fraction=float((finite & ~measured)[member].mean()) if total else None,
                unlocated_fraction=float((~finite)[member].mean()) if total else None)


def summarize_material_tracks(positions, supported, *, depth_mm=None, frame_indices=None,
                              point_ids=None, times_s=None, physical_times_s=None,
                              jump_threshold_px=5., contact_frame=None,
                              reversal_frame=None, phase_masks=None):
    """JSON-safe coverage, gaps and motion diagnostics for a complete cycle.

    ``supported`` is an algorithm's internal state, not material-identity truth.
    Landmarks address SOURCE frames; masks address the supplied T samples.
    Loading and withdrawal are separate, including when their depths overlap.
    Missing geometry within a phase does not break a support gap.
    Playback fps is never used to infer physical velocities or gap duration.
    """
    xy, support, ids, frames = _inputs(positions, supported, point_ids, frame_indices)
    length, count = support.shape
    depth = None if depth_mm is None else np.asarray(depth_mm, float)
    if depth is not None and depth.shape != (length,):
        raise ValueError("depth_mm must address all track frames")
    if times_s is not None and physical_times_s is not None:
        raise ValueError("Pass physical timestamps as times_s or physical_times_s, not both")
    supplied_times = times_s if times_s is not None else physical_times_s
    times = None if supplied_times is None else np.asarray(supplied_times, float)
    if times is not None and (times.shape != (length,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("physical_times_s must be finite and strictly increasing")
    if not np.isfinite(jump_threshold_px) or jump_threshold_px <= 0:
        raise ValueError("jump_threshold_px must be positive")
    finite = np.isfinite(xy).all(2)
    measured = support & finite
    phases, phase_labels, phase_metadata = _phase_partition(frames, depth,
        contact_frame=contact_frame, reversal_frame=reversal_frame, phase_masks=phase_masks)
    all_members = np.ones(length, bool)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=2)
    finite_edges = finite[:-1] & finite[1:]
    supported_edges = measured[:-1] & measured[1:]

    def motion(member):
        phase_edges = member[1:] & member[:-1]
        used_edges = supported_edges & phase_edges[:, None]
        all_edges = finite_edges & phase_edges[:, None]
        return dict(jump_threshold_px=float(jump_threshold_px),
            all_finite_edge_count=int(all_edges.sum()), all_finite_step_px=_stats(step[all_edges]),
            all_finite_jump_count=int(np.sum((step > jump_threshold_px) & all_edges)),
            supported_edge_count=int(used_edges.sum()), supported_only_step_px=_stats(step[used_edges]),
            supported_only_jump_count=int(np.sum((step > jump_threshold_px) & used_edges)),
            supported_only_speed_px_s=None if times is None else _stats((step/np.diff(times)[:, None])[used_edges]),
            interpretation="全有限轨迹包含预测；supported-only要求运动两端均通过算法检查，仍不是独立精度证明。阶段内运动要求两端属于同一阶段，阶段交界运动仅计入全视频。")

    phase_bins = {name: [] for name in phases}
    if depth is not None:
        finite_depth = depth[np.isfinite(depth)]
        max_depth = max(3., float(np.max(finite_depth)) if finite_depth.size else 3.)
        min_depth = min(0., float(np.min(finite_depth)) if finite_depth.size else 0.)
        edges = np.arange(np.floor(min_depth/.25)*.25, np.ceil(max_depth/.25)*.25 + .125, .25)
        for name, phase_member in phases.items():
            for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                member = phase_member & np.isfinite(depth) & (depth >= low)
                member &= depth <= high if i == len(edges)-2 else depth < high
                phase_bins[name].append(dict(lower_mm=float(low), upper_mm=float(high),
                    supported_count=int(measured[member].sum()), **_coverage(finite, measured, member)))
    phase_coverage = {name: _coverage(finite, measured, member) for name, member in phases.items()}
    phase_motion = {name: motion(member) for name, member in phases.items()}
    phase_gaps = {name: _support_gap_summary(measured, member, frames, ids, depth, times,
        f"仅统计 {name} 阶段；缺失深度不拆连续缺口，阶段边界会拆分；长度单位为实际采样点。")
        for name, member in phases.items()}
    return dict(schema="single_material_tracking_review_v2", frames=length, points=count,
        point_ids=[_scalar(v) for v in ids], source_frame_indices=frames.tolist(),
        depth_mm=None if depth is None else [_scalar(v) for v in depth],
        first_known_contact_track_index=phase_metadata['contact_track_index'],
        phase=phase_metadata,
        interpretation="算法支持率、平滑程度、保留点数不是材料身份或真实位移精度。未支持但有限的坐标作为预测保留；油膜折射、遮挡和大变形仍可能造成身份错误。",
        all_frames=_coverage(finite, measured, all_members),
        loading=phase_coverage['loading'], withdrawal=phase_coverage['withdrawal'], phases=phase_coverage,
        per_frame=dict(supported=measured.sum(1).tolist(), predicted_finite=(finite & ~measured).sum(1).tolist(),
                       unlocated=(~finite).sum(1).tolist(), phase=phase_labels.tolist()),
        support_gaps=phase_gaps['loading'], phase_support_gaps=phase_gaps,
        full_video_support_gaps=_support_gap_summary(measured, all_members, frames, ids, depth, times,
            "全部输入帧；连续缺口可跨越接触和退刀边界，缺失深度不拆分。长度单位为实际采样点。"),
        loading_motion=phase_motion['loading'], phase_motion=phase_motion, all_video_motion=motion(all_members),
        depth_bins=phase_bins['loading'], phase_depth_bins=phase_bins)


def _clip_segments(segments, width, height):
    """Vectorized line clipping; never cast unbounded/nonfinite coordinates to int."""
    segments = np.asarray(segments, float).reshape(-1, 2, 2)
    finite = np.isfinite(segments).all(axis=(1, 2))
    clean = np.where(np.isfinite(segments), segments, 0.)
    start, delta = clean[:, 0], clean[:, 1]-clean[:, 0]
    low, high = np.zeros(len(clean)), np.ones(len(clean))
    valid = finite.copy()
    for axis, limit in ((0, width-1), (1, height-1)):
        stationary = np.abs(delta[:, axis]) < 1e-12
        valid &= ~stationary | ((start[:, axis] >= 0) & (start[:, axis] <= limit))
        denominator = np.where(stationary, 1., delta[:, axis])
        t0, t1 = -start[:, axis]/denominator, (limit-start[:, axis])/denominator
        low = np.maximum(low, np.where(stationary, 0., np.minimum(t0, t1)))
        high = np.minimum(high, np.where(stationary, 1., np.maximum(t0, t1)))
    valid &= low <= high
    clipped = np.stack((start[valid]+low[valid, None]*delta[valid],
                        start[valid]+high[valid, None]*delta[valid]), axis=1)
    clipped = np.clip(clipped, [0, 0], [width-1, height-1])
    return np.rint(clipped).astype(np.int32), np.flatnonzero(valid)


def _palette(count):
    hsv = np.column_stack((np.arange(count)*179/count, np.full(count, 175), np.full(count, 252))).astype(np.uint8)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]


def _overview_indices(phases, metadata, length):
    """Keep the last frame even if blade/depth geometry is no longer visible."""
    chosen = {}

    def add(index, role):
        if index is not None and 0 <= index < length:
            chosen.setdefault(int(index), []).append(role)

    add(0, 'initial')
    loading = np.flatnonzero(phases['loading'])
    withdrawal = np.flatnonzero(phases['withdrawal'])
    add(int(loading[0]) if loading.size else metadata['contact_track_index'], 'contact')
    if loading.size:
        add(int(loading[len(loading)//2]), 'loading_mid')
        add(int(loading[-1]), 'peak_or_loading_end')
    elif metadata['reversal_track_index'] is not None:
        add(metadata['reversal_track_index'], 'reversal')
    if withdrawal.size:
        add(int(withdrawal[len(withdrawal)//2]), 'withdrawal_mid')
    add(length-1, 'final')
    return chosen


def render_material_review(shadow_video, positions, output_path, *, frame_indices,
                           depth_mm, tip_xy, supported, point_ids, fps=12.,
                           overview_path=None, config=None, contact_frame=None,
                           reversal_frame=None, phase_masks=None, blade_mask_reader=None):
    """Left shadow-normalized texture + points; right the same points on black.

    Both panels share either a full-field camera or one fixed-x, smoothed-tip-y
    display-only camera that follows loading and withdrawal. Small hollow
    markers retain unsupported finite predictions. Trails are batch-clipped and
    drawn once into a shared overlay, never independently in each panel.
    """
    xy, support, ids, frames = _inputs(positions, supported, point_ids, frame_indices)
    cfg = config or MaterialReviewConfig()
    length, count = support.shape
    depth, tips = np.asarray(depth_mm, float), np.asarray(tip_xy, float)
    if depth.shape != (length,) or tips.shape != (length, 2):
        raise ValueError("Depth and tip geometry must address every track frame")
    if (not np.isfinite(fps) or fps <= 0 or cfg.trail_length < 0 or cfg.point_radius < 1
            or not 0 < cfg.point_alpha <= 1 or not 1 <= cfg.color_count <= 180):
        raise ValueError("Invalid playback or display configuration")
    if cfg.view_mode not in ('tip', 'full'):
        raise ValueError("view_mode must be 'tip' or 'full'")
    phases, phase_labels, phase_metadata = _phase_partition(frames, depth,
        contact_frame=contact_frame, reversal_frame=reversal_frame, phase_masks=phase_masks)
    source, output = Path(shadow_video).resolve(), Path(output_path).resolve()
    if source == output or (overview_path is not None and Path(overview_path).resolve() in (source, output)):
        raise ValueError("Review artifacts must not overwrite an input or each other")
    pw, ph = map(int, cfg.panel_size)
    cw, ch = map(int, cfg.crop_size)
    if min(pw, ph, cw, ch) <= 0 or (ph + 152) % 2:
        raise ValueError("Review dimensions must be positive and encode to an even frame size")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise OSError(f"Cannot open shadow-normalized display video: {source}")
    writer = None
    saved, snapshots, snapshot_metadata, points_on_view = 0, [], [], []
    occluded_counts = []
    try:
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if cfg.view_mode == 'full':
            crop = (width, height)
            camera_fallback = False
            camera = dict(origin_xy=np.zeros((length, 2)), mode='fixed_full_source_image')
        else:
            crop = min(cw, width), min(ch, height)
            guide = tips.copy()
            camera_fallback = not np.isfinite(guide[:, 1]).any()
            if camera_fallback:
                guide[:, 1] = height/2
            shaft_x = float(np.median(tips[np.isfinite(tips[:, 0]), 0])) if np.isfinite(tips[:, 0]).any() else width/2
            camera = blade_follow_camera(guide, shaft_x=shaft_x, frame_size=(width, height), crop_size=crop, top=0)
        origins = camera['origin_xy']
        # Fit the source geometry into each panel without stretching its aspect.
        scale = min(pw/crop[0], ph/crop[1])
        vw, vh = max(1, round(crop[0]*scale)), max(1, round(crop[1]*scale))
        offset = np.array([(pw-vw)//2, (ph-vh)//2], int)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*cfg.codec), float(fps), (2*pw, ph+152))
        if not writer.isOpened():
            raise OSError(f"Cannot create review video: {output}")
        palette = _palette(cfg.color_count)
        # A fixed color assignment for the declared fixed-ID ordering; support
        # state and visibility never change it.
        buckets = (np.arange(count)*29) % cfg.color_count
        colors = palette[buckets]
        finite = np.isfinite(xy).all(2)
        measured = support & finite
        maximum_depth = max(3., float(np.max(depth[np.isfinite(depth)])) if np.isfinite(depth).any() else 3.)
        minimum_depth = min(0., float(np.min(depth[np.isfinite(depth)])) if np.isfinite(depth).any() else 0.)
        axis_minimum, axis_maximum = np.floor(minimum_depth*2)/2, np.ceil(maximum_depth*2)/2
        template = np.full((ph+152, 2*pw, 3), (22, 26, 26), np.uint8)
        cv2.putText(template, f"INDEPENDENT MATERIAL TRACKING | {cfg.view_mode.upper()} VIEW", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(template, "Shadow-normalized texture + material IDs", (12, 73), cv2.FONT_HERSHEY_SIMPLEX, .5, (215, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(template, "Same material IDs | point field", (pw+12, 73), cv2.FONT_HERSHEY_SIMPLEX, .5, (215, 220, 220), 1, cv2.LINE_AA)
        legend = ("IDs retained | hollow = unsupported estimate | blade occlusion hidden | cyan = registered tip"
                  if blade_mask_reader is not None else
                  "Color = fixed ID | hollow = unsupported estimate | cyan = blade tip | support != identity proof")
        cv2.putText(template, legend,
                    (12, ph+106), cv2.FONT_HERSHEY_SIMPLEX, .45, (212, 220, 220), 1, cv2.LINE_AA)
        ruler_y = ph+124
        cv2.line(template, (30, ruler_y), (2*pw-30, ruler_y), (110, 120, 125), 1)
        # Bounds are half-mm multiples: a shared grid includes zero once and
        # avoids nearly coincident rounded labels such as 0.0 and 0.1.
        tick_values = np.arange(axis_minimum, axis_maximum+.25, .5)
        for value in tick_values:
            x = int(round(30+(2*pw-60)*(value-axis_minimum)/(axis_maximum-axis_minimum)))
            cv2.line(template, (x, ruler_y-3), (x, ruler_y+3), (160, 170, 175), 1)
            cv2.putText(template, f"{value:.1f}", (max(2, x-12), ruler_y+20), cv2.FONT_HERSHEY_SIMPLEX, .40, (185, 197, 200), 1, cv2.LINE_AA)
        overview_frames = _overview_indices(phases, phase_metadata, length)
        source_index, track_index = -1, 0
        while track_index < length:
            source_index += 1
            if not cap.grab():
                raise ValueError(f"Display video ends before requested source frame {frames[-1]}")
            if source_index != frames[track_index]:
                continue
            ok, image = cap.retrieve()
            if not ok:
                raise OSError(f"Cannot decode source frame {source_index}")
            t = track_index
            current = point_in_display(xy[t], origins[t], crop, (vw, vh))
            ink = np.zeros((ph, pw, 3), np.uint8)
            if cfg.trail_length:
                history = point_in_display(xy[max(0, t-cfg.trail_length):t+1], origins[t], crop, (vw, vh))
                if len(history) > 1:
                    segments = np.stack((history[:-1], history[1:]), axis=2).reshape(-1, 2, 2)
                    clipped, source_segment = _clip_segments(segments, vw, vh)
                    clipped += offset
                    group = np.tile(buckets, len(history)-1)[source_segment]
                    for color_id in np.unique(group):
                        lines = clipped[group == color_id]
                        cv2.polylines(ink, list(lines), False, tuple(int(c) for c in palette[color_id]), 1, cv2.LINE_AA)
            on_view = np.isfinite(current).all(1) & (current[:, 0] >= -.5) & (current[:, 0] < vw-.5)
            on_view &= (current[:, 1] >= -.5) & (current[:, 1] < vh-.5)
            points_on_view.append(int(on_view.sum()))
            current += offset
            for point in np.flatnonzero(on_view):
                cv2.circle(ink, tuple(np.rint(current[point]).astype(int)), cfg.point_radius,
                           tuple(int(v) for v in colors[point]), -1 if measured[t, point] else 1, cv2.LINE_AA)
            occluded = np.zeros(count, bool)
            if blade_mask_reader is not None:
                from .blade_contact_geometry import sample_pixels
                blade_mask = np.asarray(blade_mask_reader(int(frames[t])), bool)
                if blade_mask.shape != (height, width):
                    raise ValueError('Blade occlusion mask must match the source frame')
                occluded, _ = sample_pixels(blade_mask, xy[t])
                display_mask = crop_for_display(blade_mask.astype(np.uint8)*255,
                    origins[t], crop, (vw, vh)) > 0
                viewport = ink[offset[1]:offset[1]+vh, offset[0]:offset[0]+vw]
                viewport[display_mask] = 0
            occluded_counts.append(int(occluded.sum()))
            background = np.zeros((ph, pw, 3), np.uint8)
            background[offset[1]:offset[1]+vh, offset[0]:offset[0]+vw] = crop_for_display(image, origins[t], crop, (vw, vh))
            overlay = cv2.addWeighted(ink, cfg.point_alpha, background, 1-cfg.point_alpha, 0)
            visible_ink = np.any(ink != 0, axis=2)
            left = background.copy()
            left[visible_ink] = overlay[visible_ink]
            right = cv2.convertScaleAbs(ink, alpha=cfg.point_alpha)
            tip = point_in_display(tips[t], origins[t], crop, (vw, vh))
            if np.isfinite(tip).all() and -.5 <= tip[0] < vw-.5 and -.5 <= tip[1] < vh-.5:
                tip += offset
                for panel in (left, right):
                    cv2.drawMarker(panel, tuple(np.rint(tip).astype(int)), (255, 255, 0), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
            canvas = template.copy()
            canvas[84:84+ph, :pw], canvas[84:84+ph, pw:] = left, right
            entry = f"{depth[t]:.3f} mm" if np.isfinite(depth[t]) else "unavailable"
            seen = int((measured[t] & ~occluded).sum())
            predicted, unlocated = int((finite[t] & ~measured[t] & ~occluded).sum()), int((~finite[t]).sum())
            last_status = f"blade-hidden {int(occluded.sum())}" if blade_mask_reader is not None else f"unlocated {unlocated}"
            cv2.putText(canvas, f"Entry (contact = 0): {entry} | {phase_labels[t].upper()} | supported {seen}/{count} | predicted {predicted} | {last_status}",
                        (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .48, (225, 231, 232), 1, cv2.LINE_AA)
            if np.isfinite(depth[t]):
                marker_x = int(round(30+(2*pw-60)*np.clip((depth[t]-axis_minimum)/(axis_maximum-axis_minimum), 0, 1)))
                cv2.line(canvas, (marker_x, ruler_y-7), (marker_x, ruler_y+7), (75, 195, 248), 2)
            writer.write(canvas)
            saved += 1
            if t in overview_frames:
                if overview_path is not None:
                    snapshots.append(canvas.copy())
                snapshot_metadata.append(dict(track_index=t, source_frame=int(frames[t]),
                    depth_mm=_scalar(depth[t]), phase=str(phase_labels[t]), roles=overview_frames[t],
                    camera_origin_xy=origins[t].tolist()))
            track_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if overview_path is not None:
        overview = Path(overview_path)
        overview.parent.mkdir(parents=True, exist_ok=True)
        if snapshots and not cv2.imwrite(str(overview), np.concatenate(snapshots, axis=0)):
            raise OSError(f"Cannot write overview image: {overview}")
    return dict(output_path=str(output), overview_path=None if overview_path is None else str(Path(overview_path)),
        rendered_frames=saved, playback_fps=float(fps), dimensions=[2*pw, ph+152], codec=cfg.codec,
        camera_mode=str(camera['mode']), camera_fallback=camera_fallback, crop_size=list(crop),
        view_mode=cfg.view_mode, image_viewport=dict(size=[vw, vh], offset=offset.tolist()),
        saved_point_count=count, per_frame_points_on_view=points_on_view,
        blade_occlusion_applied=blade_mask_reader is not None,
        per_frame_blade_occluded_ids=occluded_counts,
        phase=phase_metadata, overview_snapshots=snapshot_metadata,
        axis="indentation_mm_from_first_contact", maximum_display_depth_mm=maximum_depth,
        axis_range_mm=[float(axis_minimum), float(axis_maximum)], final_depth_mm=_scalar(depth[-1]),
        display="left shadow-normalized texture + fixed IDs; right identical IDs on black; hollow unsupported predictions",
        interpretation="Display-only camera; no saved trajectory is repaired or removed. Algorithm support is not material-identity proof.")
