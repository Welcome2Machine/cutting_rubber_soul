"""Saved-data review of blade-relative motion and original-surface address.

This module changes display coordinates only. It never relocates a material
point to the blade, fills a missing measurement, or classifies fracture.
"""
from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np

from .blade_follow_camera import blade_follow_camera, crop_for_display, point_in_display


GROUP_COLORS = ((40, 184, 247), (220, 136, 210), (225, 205, 68), (202, 154, 104))
GROUP_LABELS = ("Left surface", "Left subsurface", "Right surface", "Right subsurface")
STATE_COLORS = ((185, 185, 185), (201, 129, 218), (108, 223, 125))


def _phase_runs(frames, depth, values, contact_frame, reversal_frame, branch=None):
    """Finite contiguous line pieces; loading and withdrawal never connect."""
    frames, depth, values = map(np.asarray, (frames, depth, values))
    valid = np.isfinite(depth) & np.isfinite(values)
    phase = np.where(frames < contact_frame, -1, np.where(frames <= reversal_frame, 0, 1))
    start = None
    for i in range(len(frames) + 1):
        active = i < len(frames) and valid[i] and phase[i] >= 0
        split = (start is not None and (not active or phase[i] != phase[i-1]
                 or (branch is not None and branch[i] != branch[i-1])))
        if split:
            yield np.arange(start, i), int(phase[i-1])
            start = None
        if active and start is None:
            start = i


def _sample_mask(mask, xy):
    from .blade_contact_geometry import sample_pixels
    return sample_pixels(np.asarray(mask, bool), xy)[0]


def _arrow_vector(velocity, scale_px_s):
    """Bound display lengths and let vanishing residuals disappear naturally."""
    velocity = np.asarray(velocity, float)
    norm = np.linalg.norm(velocity)
    if not np.isfinite(norm) or norm <= 0:
        return None
    length = 20.*np.tanh(norm/max(float(scale_px_s), 1e-12))
    return None if length < 4. else length*velocity/norm


def _mask_material_ink(ink, blade_mask, origin, crop_size, viewport_size, offset):
    """Erase all material ink on opaque blade pixels, including arrow strokes."""
    display = crop_for_display(np.asarray(blade_mask, np.uint8)*255, origin, crop_size, viewport_size) > 0
    x, y = offset
    w, h = viewport_size
    ink[y:y+h, x:x+w][display] = 0
    return ink


def _marker_geometry(tip, address, row):
    """Keep image-space blade location independent of inverse material map."""
    state = int(np.asarray(address.get('state', []))[row])
    current = np.asarray(tip, float).copy()
    reference = np.asarray(address['reference_marker_xy'][row], float).copy()
    if state <= 0:
        reference[:] = np.nan
    raw = np.asarray(address.get('raw_inverse_material_xy', np.full((row+1, 2), np.nan))[row], float).copy()
    return current, reference, raw, state


def _text(image, value, xy, scale=.52, color=(221, 227, 229), thickness=1):
    cv2.putText(image, str(value), tuple(map(int, xy)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _draw_marker(image, xy, color, *, kind='ring', radius=8):
    if not np.isfinite(xy).all():
        return
    x, y = np.rint(xy).astype(int)
    if not (-radius <= x < image.shape[1]+radius and -radius <= y < image.shape[0]+radius):
        return
    if kind == 'diamond':
        cv2.drawMarker(image, (x, y), color, cv2.MARKER_DIAMOND, 11, 1, cv2.LINE_AA)
    elif kind == 'filled':
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.circle(image, (x, y), radius, color, 2, cv2.LINE_AA)


def _surface_contact(surface):
    boundary, stations = np.asarray(surface['boundary_xy']), np.asarray(surface['surface_s_px'])
    order = np.argsort(stations)
    stations, boundary = stations[order], boundary[order]
    contact_s = float(surface['contact_s_px'])
    if not stations[0] <= contact_s <= stations[-1]:
        return np.full(2, np.nan)
    return np.array([np.interp(contact_s, stations, boundary[:, axis]) for axis in range(2)])


def _plots(frames, depth, motion, address, contact_frame, reversal_frame, um_per_px):
    """Rasterize the two time-invariant plots once; return cursor geometry."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    values = (np.asarray(motion['group_y_median_um_s'], float) if 'group_y_median_um_s' in motion
              else np.asarray(motion['group_y_median'], float)*um_per_px)
    if values.shape != (len(frames), 4):
        raise ValueError('Motion group_y_median[_um_s] must be T,4')
    valid_speed = np.abs(values[np.isfinite(values)])
    scale = max(float(np.percentile(valid_speed, 90)), 1e-6) if valid_speed.size else 1.
    trend = np.tanh(values/scale)
    finite_depth = depth[np.isfinite(depth) & (frames >= contact_frame)]
    xmax = max(3., float(np.max(finite_depth)) if len(finite_depth) else 3.)
    xmin = min(0., float(np.min(finite_depth)) if len(finite_depth) else 0.)
    fig = Figure(figsize=(16, 2.88), dpi=100, facecolor='#141a1c')
    canvas = FigureCanvasAgg(fig)
    axes = [fig.add_axes([.055, .20, .42, .66]), fig.add_axes([.555, .20, .425, .66])]
    for ax in axes:
        ax.set_facecolor('#141a1c')
        ax.tick_params(colors='#d0d7d9', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#607078')
        ax.grid(alpha=.20, color='#aabbcc')
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel('Indentation from first contact (mm)', color='#d0d7d9', fontsize=10)
        ax.axhline(0., lw=.7, color='#77878c')
    ax = axes[0]
    for group, (color, label) in enumerate(zip(GROUP_COLORS, GROUP_LABELS)):
        rgb = np.array(color[::-1])/255.
        labeled = False
        for indices, phase in _phase_runs(frames, depth, trend[:, group], contact_frame, reversal_frame):
            ax.plot(depth[indices], trend[indices, group], color=rgb, lw=1.15,
                    ls='-' if phase == 0 else '--', marker='.' if len(indices) == 1 else None,
                    label=label if not labeled else None)
            labeled = True
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks([-1, 0, 1], ['Up', '0', 'Down'])
    ax.set_title('Relative vertical trend (+ = material down vs blade)', color='#e2e7e8', fontsize=11, pad=10)
    # The four group colors are already labeled immediately above this panel;
    # repeating the legend inside the axes obscures near-zero-depth trends.
    ax = axes[1]
    states = np.asarray(address['state'])
    address_values = np.asarray(address['s_um'], float)
    branch = np.asarray(address.get('branch_label', np.zeros(len(frames))))
    for state, label in ((2, 'Supported surface-address estimate'), (1, 'Conditional surface estimate')):
        use = np.where(states == state, address_values, np.nan)
        color = np.array(STATE_COLORS[state][::-1])/255.
        labeled = False
        for indices, phase in _phase_runs(frames, depth, use, contact_frame, reversal_frame, branch):
            ax.plot(depth[indices], use[indices], color=color, lw=1.3,
                    ls='-' if phase == 0 else '--', marker='.' if len(indices) == 1 else None,
                    label=label if not labeled else None)
            if 'low_um' in address and 'high_um' in address:
                ax.fill_between(depth[indices], np.asarray(address['low_um'])[indices],
                    np.asarray(address['high_um'])[indices], color=color, alpha=.12)
            labeled = True
    ax.set_ylabel('Original-surface address change (um)', color='#d0d7d9', fontsize=9)
    ax.set_title('Which original-surface point is at the tip?', color='#e2e7e8', fontsize=11, pad=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(handles, labels, loc='upper left', fontsize=8, framealpha=.65,
                           facecolor='#141a1c', edgecolor='#607078')
        for txt in legend.get_texts():
            txt.set_color('#e2e7e8')
    else:
        ax.text(.5, .5, 'No supported original-surface address', transform=ax.transAxes,
                ha='center', va='center', color='#d0d7d9', fontsize=11)
    canvas.draw()
    raster = np.asarray(canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    boxes = []
    for ax in axes:
        box = ax.get_window_extent()
        boxes.append([float(box.x0), float(raster.shape[0]-box.y1), float(box.width), float(box.height)])
    return raster, boxes, (xmin, xmax), scale


def render_contact_motion_review(video, positions, supported, output_path, *,
        frame_indices, depth_mm, tip_xy, point_ids, motion, address, surface_reference,
        blade_mask_reader, contact_frame, reversal_frame, um_per_px,
        overview_path=None, fps=12., codec='mp4v'):
    """Render all requested saved frames; parent may encode the output to H.264.

    Source-frame landmarks and frame_indices use the same video coordinate.
    Motion values retain physical acquisition-time units (um/s). Only their
    displayed direction trend uses one shared tanh scale, stated in metadata.
    The current ring is always the registered blade tip, including when no
    original-surface address is available. A gray reference diamond is a raw
    inverse diagnostic and never a projected surface-contact estimate.
    """
    p, support = np.asarray(positions), np.asarray(supported, bool)
    frames, depth, tips, ids = map(np.asarray, (frame_indices, depth_mm, tip_xy, point_ids))
    if (p.ndim != 3 or p.shape[2] != 2 or support.shape != p.shape[:2] or len(p) < 1
            or frames.shape != (len(p),) or depth.shape != (len(p),)
            or tips.shape != (len(p), 2) or ids.shape != (p.shape[1],)):
        raise ValueError('Expected aligned T,N,2 tracks, T,N support, T geometry and N IDs')
    if (not np.isfinite(frames).all() or np.any(frames != np.floor(frames))
            or np.any(np.diff(frames) <= 0) or frames[0] < 0
            or len(np.unique(ids)) != len(ids) or not np.isfinite([fps, um_per_px]).all()
            or fps <= 0 or um_per_px <= 0 or reversal_frame < contact_frame):
        raise ValueError('Invalid frame indices, unique IDs, scale, fps or phase landmarks')
    frames = frames.astype(np.int64)
    for key in ('state', 's_um', 'reference_marker_xy'):
        if key not in address or len(address[key]) != len(p):
            raise ValueError(f'Address field {key} must align with track rows')
    groups = np.asarray(motion['group_index'])
    if groups.shape != support.shape:
        raise ValueError('Motion group_index must align with saved track IDs')
    if 'point_ids' in motion and not np.array_equal(np.asarray(motion['point_ids']), ids):
        raise ValueError('Motion point IDs disagree with saved track ordering')
    source, output = Path(video).resolve(), Path(output_path).resolve()
    if source == output or (overview_path is not None and Path(overview_path).resolve() in (source, output)):
        raise ValueError('Review artifacts must not overwrite source data or each other')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise OSError(f'Cannot open display video: {source}')
    writer = None
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop = (min(384, width), min(288, height))
    guide = tips.astype(float).copy()
    fallback = not np.isfinite(guide[:, 1]).any()
    if fallback:
        guide[:, 1] = height/2
    shaft_x = float(np.median(tips[np.isfinite(tips[:, 0]), 0])) if np.isfinite(tips[:, 0]).any() else width/2
    camera = blade_follow_camera(guide, shaft_x=shaft_x, frame_size=(width, height), crop_size=crop, top=0)
    origins = camera['origin_xy']
    panel_size = (800, 480)
    def fit(size):
        scale = min(panel_size[0]/size[0], panel_size[1]/size[1])
        viewport = (round(size[0]*scale), round(size[1]*scale))
        return viewport, np.array([(800-viewport[0])//2, (480-viewport[1])//2])
    viewport, offset = fit(crop)
    ref_viewport, ref_offset = fit((width, height))
    plots, boxes, axis, trend_scale = _plots(frames, depth, motion, address, contact_frame, reversal_frame, um_per_px)
    snapshots, snapshot_metadata, occluded_counts, shown_counts = [], [], [], []
    selections = {0: ['initial'], len(p)-1: ['final']}
    contact_rows = np.flatnonzero(frames >= contact_frame)
    if contact_rows.size:
        selections.setdefault(int(contact_rows[0]), []).append('contact')
    loading_rows = np.flatnonzero((frames >= contact_frame) & (frames <= reversal_frame) & np.isfinite(depth))
    for target in (1., 2., 3.):
        if loading_rows.size:
            row = int(loading_rows[np.argmin(abs(depth[loading_rows]-target))])
            if abs(float(depth[row])-target) <= .12:
                selections.setdefault(row, []).append(f'{int(target)}mm')
    template = np.full((1000, 1600, 3), (22, 26, 28), np.uint8)
    _text(template, 'MATERIAL MOTION AT THE BLADE TIP', (18, 28), .76)
    _text(template, 'Current material | stable blade-following view', (18, 83), .59)
    _text(template, 'Original material | fixed surface addresses', (818, 83), .59)
    _text(template, 'Cross: blade tip | ring: address state | arrows fade near zero | hollow: prediction', (18, 598), .49)
    _text(template, 'Orange: first contact | green: supported | pink: conditional | gray diamond: raw inverse diagnostic', (818, 598), .45)
    for k, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        x = 18+k*196
        cv2.circle(template, (x+4, 620), 3, color, -1)
        _text(template, label, (x+15, 625), .46, color)
    _text(template, 'Solid: loading | dashed: withdrawal | gaps remain unmeasured', (818, 625), .49)
    template[642:930] = plots
    boundary = np.asarray(surface_reference['boundary_xy'], float)
    station_colors = cv2.applyColorMap(np.linspace(0, 255, len(boundary)).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    reference_contact = _surface_contact(surface_reference)
    ref_initial_points = point_in_display(p[0], np.zeros(2), (width, height), ref_viewport)+ref_offset
    group_velocity = np.asarray(motion.get('relative_velocity_xy_px_s', np.full(p.shape, np.nan)))
    group_valid = np.asarray(motion.get('relative_velocity_valid', support), bool)
    relative_counts = np.asarray(motion.get('group_y_count', np.zeros((len(p), 4))))
    relaxing = np.asarray(motion.get('disjoint_relaxing_patch_count', np.zeros(len(p), int)))
    opening = np.asarray(motion.get('supplied_image_opening_evidence', np.full(len(p), -1)))
    raw_normals = np.asarray(address.get('raw_surface_normal_offset_px', np.full(len(p), np.nan)))
    turns = np.asarray(address.get('surface_turn_deg', np.full(len(p), np.nan)))
    reasons = np.asarray(address.get('reason', np.full(len(p), 'unknown')))
    state_labels = ('ADDRESS UNKNOWN', 'CONDITIONAL SURFACE ADDRESS', 'SUPPORTED SURFACE ADDRESS')
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    ref_background = None
    try:
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), float(fps), (1600, 1000))
        if not writer.isOpened():
            raise OSError(f'Cannot create review video: {output}')
        source_index = -1
        for row, frame_number in enumerate(frames):
            while source_index < frame_number:
                source_index += 1
                if not cap.grab():
                    raise ValueError(f'Display video ends before requested source frame {frame_number}')
            ok, image = cap.retrieve()
            if not ok:
                raise OSError(f'Cannot decode source frame {frame_number}')
            if ref_background is None:
                ref_background = np.zeros((480, 800, 3), np.uint8)
                rx, ry = ref_offset
                rw, rh = ref_viewport
                ref_background[ry:ry+rh, rx:rx+rw] = cv2.resize(image, ref_viewport, interpolation=cv2.INTER_AREA)
                for q in ref_initial_points[np.isfinite(ref_initial_points).all(1)]:
                    cv2.circle(ref_background, tuple(np.rint(q).astype(int)), 1, (96, 105, 109), -1)
                reference_stations = point_in_display(boundary, np.zeros(2), (width, height), ref_viewport)+ref_offset
                for q, color in zip(reference_stations, station_colors):
                    _draw_marker(ref_background, q, tuple(map(int, color)), kind='filled', radius=2)
                q = point_in_display(reference_contact, np.zeros(2), (width, height), ref_viewport)+ref_offset
                _draw_marker(ref_background, q, (0, 184, 255), kind='filled', radius=6)
            left = np.zeros((480, 800, 3), np.uint8)
            vx, vy = offset
            vw, vh = viewport
            left[vy:vy+vh, vx:vx+vw] = crop_for_display(image, origins[row], crop, viewport)
            blade = np.zeros((height, width), bool) if blade_mask_reader is None else np.asarray(blade_mask_reader(int(frame_number)), bool)
            if blade.shape != (height, width):
                raise ValueError('Blade occlusion mask must match source video shape')
            occluded = _sample_mask(blade, p[row])
            occluded_counts.append(int(occluded.sum()))
            finite = np.isfinite(p[row]).all(1)
            on_view_xy = point_in_display(p[row], origins[row], crop, viewport)+offset
            in_view = (finite & (on_view_xy[:, 0] >= vx) & (on_view_xy[:, 0] < vx+vw)
                       & (on_view_xy[:, 1] >= vy) & (on_view_xy[:, 1] < vy+vh))
            shown_counts.append(int((in_view & ~occluded).sum()))
            ink = np.zeros_like(left)
            for idx in np.flatnonzero(in_view):
                group = int(groups[row, idx])
                color = GROUP_COLORS[group] if 0 <= group <= 3 else (153, 165, 172)
                if not support[row, idx]:
                    color = (110, 116, 120)
                cv2.circle(ink, tuple(np.rint(on_view_xy[idx]).astype(int)), 2 if group >= 0 else 1,
                           color, -1 if support[row, idx] else 1, cv2.LINE_AA)
            # A few bounded trend arrows. Near-zero residuals cannot become
            # full-length arrows merely because their noisy direction flips.
            for group in range(4):
                candidates = np.flatnonzero(in_view & ~occluded & support[row] & group_valid[row]
                                            & (groups[row] == group) & np.isfinite(group_velocity[row]).all(1))
                chosen = []
                for idx in candidates[np.argsort(ids[candidates])]:
                    if chosen and np.min(np.linalg.norm(p[row, chosen]-p[row, idx], axis=1)) < 15:
                        continue
                    vector = _arrow_vector(group_velocity[row, idx], trend_scale/um_per_px)
                    if vector is None:
                        continue
                    start = np.rint(on_view_xy[idx]).astype(int)
                    end = np.rint(start+vector).astype(int)
                    cv2.arrowedLine(ink, tuple(start), tuple(end), GROUP_COLORS[group], 1, cv2.LINE_AA, tipLength=.28)
                    chosen.append(idx)
                    if len(chosen) == 2:
                        break
            _mask_material_ink(ink, blade, origins[row], crop, viewport, offset)
            show = np.any(ink, axis=2)
            blend = cv2.addWeighted(left, .36, ink, .64, 0)
            left[show] = blend[show]
            right = ref_background.copy()
            for idx in np.flatnonzero(occluded):
                _draw_marker(right, ref_initial_points[idx], (165, 173, 179), kind='ring', radius=2)
            current, reference, raw, state = _marker_geometry(tips[row], address, row)
            state = max(0, min(2, state))
            current_display = point_in_display(current, origins[row], crop, viewport)+offset
            _draw_marker(left, current_display, STATE_COLORS[state], radius=9)
            if np.isfinite(current_display).all() and 0 <= current_display[0] < 800 and 0 <= current_display[1] < 480:
                cv2.drawMarker(left, tuple(np.rint(current_display).astype(int)), (255, 241, 49),
                               cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
            reference_display = point_in_display(reference, np.zeros(2), (width, height), ref_viewport)+ref_offset
            _draw_marker(right, reference_display, STATE_COLORS[state], radius=8)
            if state == 0 and np.isfinite(raw).all():
                raw_display = point_in_display(raw, np.zeros(2), (width, height), ref_viewport)+ref_offset
                _draw_marker(right, raw_display, (190, 190, 190), kind='diamond')
            canvas = template.copy()
            canvas[94:574, :800], canvas[94:574, 800:] = left, right
            phase = 'PRECONTACT' if frame_number < contact_frame else ('LOADING' if frame_number <= reversal_frame else 'WITHDRAWAL')
            entry = f'{float(depth[row]):.3f} mm' if np.isfinite(depth[row]) else 'unavailable'
            seen = int((support[row] & finite & ~occluded).sum())
            _text(canvas, f'Entry from contact: {entry} | {phase} | supported {seen}/{len(ids)} | blade-hidden {occluded.sum()} | all IDs retained', (18, 54), .54)
            if np.isfinite(depth[row]):
                for x, y, w, h in boxes:
                    cursor = int(round(x+w*np.clip((depth[row]-axis[0])/(axis[1]-axis[0]), 0, 1)))
                    cv2.line(canvas, (cursor, 642+round(y)), (cursor, 642+round(y+h)), (214, 222, 225), 1, cv2.LINE_AA)
            counts = '/'.join(str(int(v)) for v in relative_counts[row])
            image_state = 'opening observed' if opening[row] == 1 else ('no visible opening supplied' if opening[row] == 0 else 'opening unassessed')
            _text(canvas, f'Local relative-motion samples (LS/LD/RS/RD): {counts}', (18, 951), .49)
            _text(canvas, f'Recovering fixed blocks: {int(relaxing[row])} | {image_state} | candidate evidence, not fracture confirmation', (18, 976), .46)
            address_value = float(np.asarray(address['s_um'])[row])
            label = state_labels[state] + (f' | {address_value:+.0f} um' if state > 0 and np.isfinite(address_value) else '')
            _text(canvas, label, (818, 951), .52, STATE_COLORS[state])
            if state > 0:
                note = f'Local surface turn: {turns[row]:+.1f} deg' if np.isfinite(turns[row]) else 'Local surface turn unavailable'
                note += ' | surface identity estimate; not slip proof'
            else:
                normal = f'{raw_normals[row]*um_per_px:+.0f} um inward' if np.isfinite(raw_normals[row]) else 'unavailable'
                note = f'Inverse normal: {normal} | {str(reasons[row]).replace("_", " ")}'
            _text(canvas, note[:103], (818, 976), .43)
            writer.write(canvas)
            saved += 1
            if row in selections:
                record = dict(track_index=row, source_frame=int(frame_number),
                              depth_mm=float(depth[row]) if np.isfinite(depth[row]) else None,
                              phase=phase, roles=selections[row], camera_origin_xy=origins[row].tolist())
                snapshot_metadata.append(record)
                if overview_path is not None:
                    overview = Path(overview_path)
                    overview.parent.mkdir(parents=True, exist_ok=True)
                    snapshot_path = overview.with_name(overview.stem+'_'+selections[row][0]+'.png')
                    if not cv2.imwrite(str(snapshot_path), canvas):
                        raise OSError(f'Cannot write snapshot: {snapshot_path}')
                    record['image_path'] = str(snapshot_path.resolve())
                    snapshots.append(canvas.copy())
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if overview_path is not None and snapshots:
        if not cv2.imwrite(str(overview_path), np.concatenate(snapshots, axis=0)):
            raise OSError(f'Cannot write overview: {overview_path}')
    metadata = dict(schema='contact_motion_review_v1', output_path=str(output), rendered_frames=saved,
        playback_fps=float(fps), dimensions=[1600, 1000], codec=codec,
        point_count=len(ids), all_saved_ids_retained=True, input_tracks_modified=False,
        camera_mode=str(camera['mode']), camera_fallback=fallback, crop_size=list(crop),
        camera_origin_xy=origins.tolist(), reference_panel='fixed whole initial source frame',
        current_marker='registered blade tip; never an inverse material point',
        original_marker='surface-compatible estimated original address; absent when unknown',
        raw_diagnostic='conditional material inverse, may be interior, outside surface or poor fit; not a surface address',
        blade_occlusion_applied=blade_mask_reader is not None, per_frame_blade_occluded_ids=occluded_counts,
        per_frame_material_points_on_view=shown_counts,
        axis='indentation_mm_from_first_contact', axis_range_mm=list(axis),
        loading_line='solid', withdrawal_line='dashed', gap_interpolation=False,
        contact_source_frame=int(contact_frame), reversal_source_frame=int(reversal_frame),
        motion_data_units='um/s from physical acquisition time, not playback fps',
        motion_display_transform='tanh(group vertical relative velocity / shared 90th-percentile absolute speed)',
        motion_display_scale_um_s=trend_scale, positive_relative_y='material image-down relative to blade',
        arrows='at most two supported arrows per group; length=20*tanh(speed/shared scale), hidden below4 display px; not a calibrated displacement',
        uncertainty='address envelope is a sensitivity interval; not statistical confidence',
        fracture_classified=False, overview_snapshots=snapshot_metadata)
    metadata_path = output.with_suffix('.render.json')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    metadata['metadata_path'] = str(metadata_path)
    return metadata
