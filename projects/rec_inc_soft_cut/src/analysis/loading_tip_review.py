"""Loading-only display of blade-neighbourhood motion trends.

The four coloured groups are current geometric neighbourhoods, not persistent
surface identities.  Estimated group trends never create measured material
points.  Every material stroke is occluded by the supplied blade mask.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .blade_follow_camera import blade_follow_camera, crop_for_display, point_in_display
from .blade_contact_geometry import sample_pixels


GROUP_COLORS = ((40, 184, 247), (220, 136, 210), (225, 205, 68), (202, 154, 104))
GROUP_LABELS = ('Left interface band', 'Left inner band', 'Right interface band', 'Right inner band')
STATE_NAMES = ('unknown', 'short bridge', 'conditional estimate', 'image assisted', 'direct supported')


def _text(image, message, xy, scale=.50, color=(221, 227, 229)):
    cv2.putText(image, str(message), tuple(map(int, xy)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def _state_runs(depth, values, states):
    """Contiguous finite segments; do not bridge unknowns or source classes."""
    depth, values, states = map(np.asarray, (depth, values, states))
    source_class = np.where(states == 4, 4, np.where(states >= 2, 2, states))
    valid = np.isfinite(depth) & np.isfinite(values) & (states > 0)
    start = None
    for i in range(len(depth)+1):
        active = i < len(depth) and valid[i]
        split = start is not None and (not active or source_class[i] != source_class[i-1])
        if split:
            yield np.arange(start, i), int(source_class[i-1])
            start = None
        if active and start is None:
            start = i


def _shared_scale(values, states):
    values, states = np.asarray(values, float), np.asarray(states)
    usable = values[(states >= 2) & np.isfinite(values).all(axis=-1)]
    if not usable.size:
        usable = values[(states >= 1) & np.isfinite(values).all(axis=-1)]
    return max(float(np.percentile(np.abs(usable), 90)), 1e-9) if usable.size else 1.


def _arrow_vector(velocity, scale):
    velocity = np.asarray(velocity, float)
    norm = np.linalg.norm(velocity)
    if not np.isfinite(norm) or norm <= 0:
        return None
    length = 24.*np.tanh(norm/max(float(scale), 1e-12))
    return None if length < 4. else velocity*length/norm


def _plots(depth, trend, shared_scale, width=1600, height=300):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    relative = np.asarray(trend['relative_trend_xy'], float)
    state = np.asarray(trend['state'])
    values = np.tanh(relative/shared_scale)
    finite_depth = depth[np.isfinite(depth)]
    xmax = max(3., float(finite_depth.max())) if finite_depth.size else 3.
    xmin = min(0., float(finite_depth.min())) if finite_depth.size else 0.
    fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor='#141a1c')
    canvas = FigureCanvasAgg(fig)
    axes = [fig.add_axes([.055, .20, .42, .66]), fig.add_axes([.555, .20, .425, .66])]
    for axis, ax in enumerate(axes):
        ax.set_facecolor('#141a1c')
        ax.tick_params(colors='#d0d7d9', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#607078')
        ax.grid(alpha=.20, color='#aabbcc')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xlabel('Indentation from first contact (mm)', color='#d0d7d9', fontsize=10)
        ax.axhline(0., lw=.7, color='#77878c')
        ax.set_yticks([-1, 0, 1], ['Left', '0', 'Right'] if axis == 0 else ['Up', '0', 'Down'])
        ax.set_title('Relative horizontal trend (+ = material right vs blade)' if axis == 0 else
                     'Relative vertical trend (+ = material down vs blade)',
                     color='#e2e7e8', fontsize=11, pad=10)
        for group, color in enumerate(GROUP_COLORS):
            rgb = np.asarray(color[::-1])/255.
            for rows, source in _state_runs(depth, values[:, group, axis], state[:, group]):
                line_color = '#adb5b9' if source == 1 else rgb
                style = ':' if source == 1 else ('--' if source == 2 else '-')
                ax.plot(depth[rows], values[rows, group, axis], color=line_color, lw=1.35,
                        ls=style, marker='.' if len(rows) == 1 else None, markersize=2.5)
                if axis == 1 and source >= 2 and 'interval_low_y' in trend and 'interval_high_y' in trend:
                    lo = np.tanh(np.asarray(trend['interval_low_y'])[rows, group]/shared_scale)
                    hi = np.tanh(np.asarray(trend['interval_high_y'])[rows, group]/shared_scale)
                    ax.fill_between(depth[rows], lo, hi, color=rgb, alpha=.055)
    canvas.draw()
    raster = np.asarray(canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    boxes = []
    for ax in axes:
        box = ax.get_window_extent()
        boxes.append([float(box.x0), float(raster.shape[0]-box.y1), float(box.width), float(box.height)])
    return raster, boxes, (xmin, xmax)


def _validated_arrays(positions, supported, frame_indices, depth_mm, tip_xy, point_ids, trend, fps):
    p, support = np.asarray(positions), np.asarray(supported, bool)
    frames, depth, tips, ids = map(np.asarray, (frame_indices, depth_mm, tip_xy, point_ids))
    if (p.ndim != 3 or p.shape[-1] != 2 or len(p) < 1 or support.shape != p.shape[:2]
            or frames.shape != (len(p),) or depth.shape != (len(p),)
            or tips.shape != (len(p), 2) or ids.shape != (p.shape[1],)):
        raise ValueError('Expected aligned T,N,2 tracks, T,N support, T geometry and N IDs')
    if (not np.isfinite(frames).all() or np.any(frames != np.floor(frames))
            or np.any(np.diff(frames) <= 0) or frames[0] < 0
            or len(np.unique(ids)) != len(ids) or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Invalid frame indices, unique point IDs or playback fps')
    expected = {'group_index': support.shape, 'relative_trend_xy': (len(p), 4, 2),
                'state': (len(p), 4), 'observation_count': (len(p), 4),
                'estimate_count': (len(p), 4), 'tip_reference_state': (len(p),)}
    for key, shape in expected.items():
        if key not in trend or np.asarray(trend[key]).shape != shape:
            raise ValueError(f'Trend field {key} must have shape {shape}')
    # These are optional comparison diagnostics from the former tip-only
    # estimator, not measurements of the expanded interface. Never synthesize
    # them from the full-interface trend just to satisfy a display schema.
    for key, shape in {'direct_trend_xy': (len(p), 4, 2), 'legacy_trend_y': (len(p), 4)}.items():
        if key in trend and np.asarray(trend[key]).shape != shape:
            raise ValueError(f'Optional trend field {key} must have shape {shape}')
    if 'point_ids' in trend and not np.array_equal(np.asarray(trend['point_ids']), ids):
        raise ValueError('Trend point IDs disagree with saved track ordering')
    if 'tip_coordinate_state' in trend and (np.asarray(trend['tip_coordinate_state']).shape != (len(p),)
            or not np.isin(np.asarray(trend['tip_coordinate_state']), [0, 1, 2]).all()):
        raise ValueError('Tip coordinate states must align with every frame')
    if 'blade_geometry_uncertain' in trend and np.asarray(trend['blade_geometry_uncertain']).shape != (len(p),):
        raise ValueError('Blade geometry uncertainty must align with every frame')
    state = np.asarray(trend['state'])
    groups = np.asarray(trend['group_index'])
    if (not np.isin(state, np.arange(5)).all() or not np.isin(groups, [-1, 0, 1, 2, 3]).all()
            or not np.isin(np.asarray(trend['tip_reference_state']), [0, 1, 2]).all()):
        raise ValueError('Unknown trend, group or tip reference state')
    if np.any((state > 0) & ~np.isfinite(np.asarray(trend['relative_trend_xy'])).all(axis=-1)):
        raise ValueError('Non-unknown trend states require finite relative x/y values')
    return p, support, frames.astype(np.int64), depth.astype(float), tips.astype(float), ids


def _interface_camera(p, support, groups, tips, frames, width, height, extra_anchors=None):
    """Fixed zoom from the whole loading interface, never point-count tracking."""
    delta = np.asarray(p, float)-tips[:, None, :]
    valid = support & (groups >= 0) & np.isfinite(delta).all(-1)
    selected = delta[valid]
    if extra_anchors is not None:
        anchors = np.asarray(extra_anchors, float).reshape(len(p), -1, 2)-tips[:, None, :]
        selected = np.vstack([selected, anchors[np.isfinite(anchors).all(-1)]])
    if len(selected):
        low, high = selected.min(0)-28., selected.max(0)+28.
    else:
        low, high = np.array([-192., -260.]), np.array([192., 80.])
    above, below = max(120., -float(low[1])), max(64., float(high[1]))
    ch = min(height, max(288, int(np.ceil(above+below))))
    cw = min(width, max(384, int(np.ceil(ch*1.55)), int(np.ceil(2*max(-low[0], high[0])))))
    fraction = np.clip(above/max(above+below, 1), .55, .90)
    guide = tips.copy()
    good = np.isfinite(guide[:, 1])
    if not good.any():
        guide[:, 1] = height/2
        good[:] = True
    shaft_x = float(np.nanmedian(tips[:, 0])) if np.isfinite(tips[:, 0]).any() else width/2
    dense = np.arange(frames[0], frames[-1]+1)
    dense_guide = np.column_stack([np.full(len(dense), shaft_x),
        np.interp(dense, frames[good], guide[good, 1])-(fraction-.58)*ch])
    camera = blade_follow_camera(dense_guide, shaft_x=shaft_x, frame_size=(width, height),
                                crop_size=(cw, ch), top=0)
    return camera, camera['origin_xy'][frames-frames[0]], (cw, ch), dict(
        selected_relative_bounds_xy=[low.tolist(), high.tolist()], requested_tip_fraction=float(fraction),
        framing='fixed zoom from the complete observed interface over loading; smooth blade-only camera')


def _validate_slip(slip, length):
    if slip is None:
        return None
    result = dict(slip)
    for key in ('migration_um', 'state', 'interval_low_um', 'interval_high_um'):
        value = np.asarray(result[key])
        if value.shape != (length,):
            raise ValueError(f'Sliding field {key} must align with loading frames')
        result[key] = value
    if not np.isin(result['state'], [0, 1, 2, 3, 4]).all():
        raise ValueError('Unknown sliding state')
    for key, shape in [('address_initial_xy', (length, 2)), ('initial_contact_xy', (2,))]:
        result[key] = np.asarray(result[key], float)
        if result[key].shape != shape:
            raise ValueError(f'Sliding field {key} must have shape {shape}')
    result['initial_surface_xy'] = np.asarray(result['initial_surface_xy'], float)
    if result['initial_surface_xy'].ndim != 2 or result['initial_surface_xy'].shape[1] != 2:
        raise ValueError('Initial surface must be M by 2')
    active = result['state'] > 0
    if np.any(active & (~np.isfinite(result['migration_um']) | ~np.isfinite(result['address_initial_xy']).all(-1))):
        raise ValueError('Visible sliding states need finite migration and original address')
    return result


def _sliding_plot(depth, slip, width=1200, height=250):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor='#141a1c')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([.075, .21, .90, .60])
    ax.set_facecolor('#141a1c')
    ax.tick_params(colors='#d0d7d9', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#607078')
    ax.grid(alpha=.20, color='#aabbcc')
    finite = depth[np.isfinite(depth)]
    xmin, xmax = (min(0., float(finite.min())), max(3., float(finite.max()))) if len(finite) else (0., 3.)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('Indentation from first contact (mm)', color='#d0d7d9', fontsize=10)
    ax.set_ylabel('Original surface address change (um)', color='#d0d7d9', fontsize=9)
    ax.axhline(0., lw=.7, color='#ad8742')
    ax.set_title('Blade-tip contact address on the INITIAL material surface', color='#e2e7e8', fontsize=11, pad=12)
    if slip is not None:
        value, states = slip['migration_um'], slip['state']
        for rows, source in _state_runs(depth, value, states):
            color = '#89d76f' if source == 4 else ('#afb7ba' if source == 1 else '#d995d4')
            ax.plot(depth[rows], value[rows], color=color,
                    ls='-' if source == 4 else (':' if source == 1 else '--'), lw=1.55,
                    marker='.' if len(rows) == 1 else None, markersize=2.5)
            lo, hi = slip['interval_low_um'][rows], slip['interval_high_um'][rows]
            if np.isfinite(lo).any() and np.isfinite(hi).any():
                ax.fill_between(depth[rows], lo, hi, color=color, alpha=.10)
    else:
        ax.text(.5, .5, 'Surface-address measurements not supplied', transform=ax.transAxes,
                ha='center', color='#b4bec2')
    canvas.draw()
    raster = np.asarray(canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    box = ax.get_window_extent()
    return raster, [box.x0, raster.shape[0]-box.y1, box.width, box.height], (xmin, xmax)


def _reference_view(image, slip, size=(600, 225)):
    """One fixed original configuration; only this inset shows original IDs."""
    width, height = size
    result = np.full((height, width, 3), (13, 18, 20), np.uint8)
    if slip is None or image is None:
        _text(result, 'Original surface reference unavailable', (15, 90), .5)
        return result, None
    addresses = slip['address_initial_xy'][(slip['state'] > 0) & np.isfinite(slip['address_initial_xy']).all(-1)]
    anchors = np.vstack([addresses, slip['initial_contact_xy'][None]])
    anchors = anchors[np.isfinite(anchors).all(-1)]
    ih, iw = image.shape[:2]
    if len(anchors):
        lower, upper = anchors.min(0)-65., anchors.max(0)+65.
        center = (lower+upper)/2
        extent = np.maximum(upper-lower, [160., 160.])
        # Fixed panel ratio, with no anisotropic stretch or moving reference crop.
        extent[0] = max(extent[0], extent[1]*width/height)
        extent[1] = max(extent[1], extent[0]*height/width)
        ratio = min(iw/extent[0], ih/extent[1], 1.)
        extent = np.maximum(1, np.rint(extent*ratio)).astype(int)
        origin = np.clip(center-extent/2, 0, np.array([iw, ih])-extent)
    else:
        extent, origin = np.array([iw, ih]), np.zeros(2)
    display_scale = min(width/extent[0], height/extent[1])
    viewport = np.rint(extent*display_scale).astype(int)
    offset = (np.array(size)-viewport)//2
    result[offset[1]:offset[1]+viewport[1], offset[0]:offset[0]+viewport[0]] = crop_for_display(
        image, origin, tuple(extent), tuple(viewport))
    def project(xy):
        return point_in_display(xy, origin, tuple(extent), tuple(viewport))+offset
    boundary = project(slip['initial_surface_xy'])
    for q in boundary[np.isfinite(boundary).all(-1)][::max(1, len(boundary)//150)]:
        if 0 <= q[0] < width and 0 <= q[1] < height:
            cv2.circle(result, tuple(np.rint(q).astype(int)), 1, (194, 208, 185), -1, cv2.LINE_AA)
    first = project(slip['initial_contact_xy'])
    if np.isfinite(first).all():
        cv2.circle(result, tuple(np.rint(first).astype(int)), 5, (15, 179, 248), -1, cv2.LINE_AA)
    return result, project


def _render_extended_contact_review(video, p, support, output_path, *, frames, depth, tips, ids,
        trend, blade_mask_reader, overview_path, fps, codec, slip):
    source, output = Path(video).resolve(), Path(output_path).resolve()
    if source == output or (overview_path is not None and Path(overview_path).resolve() in (source, output)):
        raise ValueError('Review artifacts must not overwrite source data or each other')
    slip = _validate_slip(slip, len(p))
    groups, state = np.asarray(trend['group_index']), np.asarray(trend['state'])
    velocities = np.asarray(trend['relative_trend_xy'], float)
    has_segments = 'segment_relative_trend_xy' in trend
    if has_segments:
        segment_v = np.asarray(trend['segment_relative_trend_xy'], float)
        segment_state = np.asarray(trend['segment_state'])
        segment_target = np.asarray(trend['segment_target_xy'], float)
        if (segment_v.ndim != 4 or segment_v.shape[0:2] != (len(p), 4) or segment_v.shape[-1] != 2
                or segment_target.shape != segment_v.shape or segment_state.shape != segment_v.shape[:-1]):
            raise ValueError('Segment trends and targets must be T,4,S,2 with T,4,S states')
        if not np.isin(segment_state, [0, 1, 2, 3, 4]).all():
            raise ValueError('Unknown segment source state')
        if np.any((segment_state > 0) & (~np.isfinite(segment_v).all(-1) | ~np.isfinite(segment_target).all(-1))):
            raise ValueError('Visible segments need finite local trends and anchors')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise OSError(f'Cannot open display video: {source}')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reference_image = None
    if slip is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(slip.get('reference_frame_index', 0)))
        success, reference_image = cap.read()
        if not success:
            cap.release()
            raise ValueError('Cannot read the original surface reference frame')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    camera_anchors = None
    if has_segments:
        camera_anchors = np.where((segment_state > 0)[..., None], segment_target, np.nan)
    camera, origins, crop, camera_extra = _interface_camera(p, support, groups, tips, frames, width, height, camera_anchors)
    viewport = np.rint(np.array(crop)*min(900/crop[0], 540/crop[1])).astype(int)
    offset = (np.array([900, 540])-viewport)//2
    vx, vy = offset
    vw, vh = viewport
    scale = _shared_scale(velocities, state)
    plots, boxes, axis = _plots(depth, trend, scale, width=1800, height=225)
    sliding_plot, sliding_box, sliding_axis = _sliding_plot(depth, slip)
    reference_base, reference_project = _reference_view(reference_image, slip)
    template = np.full((1200, 1800, 3), (22, 26, 28), np.uint8)
    _text(template, 'LOADING ONLY | FULL BLADE-ADJACENT MATERIAL + SURFACE SLIDING', (18, 28), .72)
    _text(template, 'Shadow-normalized image | dense AND sparse interface material', (18, 76), .52)
    _text(template, 'Same view | LOCAL segment arrows (not one motion for the full side)', (918, 76), .50)
    _text(template, 'Cross + ring: current blade tip and contact-address state | original material address is shown ONLY in the reference below', (18, 646), .49)
    for group, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        x = 18+group*450
        cv2.circle(template, (x+4, 665), 3, color, -1)
        _text(template, label, (x+16, 670), .48, color)
    template[680:905] = plots
    _text(template, 'Motion: solid = supported | colored dashed = conditional | gray dotted = short bridge. One shared x/y trend scale; not absolute speed.', (18, 921), .48)
    _text(template, 'INITIAL surface | orange = first contact; ring = inferred current address', (18, 944), .44)
    template[950:1200, 600:1800] = sliding_plot
    selections = {}
    for target in (0., 1., 2., 3.):
        rows = np.flatnonzero(np.isfinite(depth))
        if len(rows):
            row = int(rows[np.argmin(abs(depth[rows]-target))])
            if abs(depth[row]-target) <= .15:
                selections.setdefault(row, []).append(f'{int(target)}mm')
    coordinates = np.asarray(trend.get('tip_coordinate_state', trend['tip_reference_state']))
    references = np.asarray(trend['tip_reference_state'])
    uncertain = np.asarray(trend.get('blade_geometry_uncertain', np.zeros(len(p), bool)))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer, written, snapshots, snapshot_metadata = None, 0, [], []
    occluded_counts, shown_counts, interface_counts, arrow_counts = [], [], [], []
    try:
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), float(fps), (1800, 1200))
        if not writer.isOpened():
            raise OSError(f'Cannot create review video: {output}')
        source_index = -1
        for row, frame_number in enumerate(frames):
            while source_index < frame_number:
                source_index += 1
                if not cap.grab():
                    raise ValueError(f'Display video ends before source frame {frame_number}')
            success, image = cap.retrieve()
            if not success:
                raise OSError(f'Cannot decode source frame {frame_number}')
            left = np.zeros((540, 900, 3), np.uint8)
            left[vy:vy+vh, vx:vx+vw] = crop_for_display(image, origins[row], crop, tuple(viewport))
            right = np.full_like(left, (8, 12, 13))
            blade = np.zeros((height, width), bool) if blade_mask_reader is None else np.asarray(blade_mask_reader(int(frame_number)), bool)
            if blade.shape != (height, width):
                raise ValueError('Blade occlusion mask must match source video shape')
            blade_display = crop_for_display(blade.astype(np.uint8)*255, origins[row], crop, tuple(viewport)) > 0
            right[vy:vy+vh, vx:vx+vw][blade_display] = (31, 36, 38)
            occluded = sample_pixels(blade, p[row])[0].astype(bool)
            q = point_in_display(p[row], origins[row], crop, tuple(viewport))+offset
            in_view = np.isfinite(q).all(-1) & (q[:, 0] >= vx) & (q[:, 0] < vx+vw) & (q[:, 1] >= vy) & (q[:, 1] < vy+vh)
            occluded_counts.append(int(occluded.sum()))
            shown_counts.append(int((in_view & ~occluded).sum()))
            interface_counts.append(int((in_view & ~occluded & (groups[row] >= 0)).sum()))
            ink = np.zeros_like(left)
            for idx in np.flatnonzero(in_view & ~occluded):
                group = int(groups[row, idx])
                color = GROUP_COLORS[group] if group >= 0 else (127, 142, 149)
                if not support[row, idx]:
                    color = (90, 101, 111)
                cv2.circle(ink, tuple(np.rint(q[idx]).astype(int)), 2 if group >= 0 else 1,
                           color, -1 if support[row, idx] else 1, cv2.LINE_AA)
            ink[vy:vy+vh, vx:vx+vw][blade_display] = 0
            ink_pixels = np.any(ink, axis=2)
            blended = cv2.addWeighted(left, .30, ink, .70, 0)
            left[ink_pixels], right[ink_pixels] = blended[ink_pixels], ink[ink_pixels]
            arrows, count = np.zeros_like(right), 0
            if has_segments:
                for group, segment in np.argwhere(segment_state[row] > 0):
                    vector = _arrow_vector(segment_v[row, group, segment], scale)
                    start = point_in_display(segment_target[row, group, segment], origins[row], crop, tuple(viewport))+offset
                    if vector is None or not (vx <= start[0] < vx+vw and vy <= start[1] < vy+vh):
                        continue
                    color = GROUP_COLORS[group] if segment_state[row, group, segment] >= 2 else (173, 181, 185)
                    start = np.rint(start).astype(int)
                    cv2.arrowedLine(arrows, tuple(start), tuple(np.rint(start+vector).astype(int)), color, 2, cv2.LINE_AA, tipLength=.3)
                    if segment_state[row, group, segment] < 4:
                        cv2.circle(arrows, tuple(start), 4, color, 1, cv2.LINE_AA)
                    count += 1
            arrows[vy:vy+vh, vx:vx+vw][blade_display] = 0
            arrow_pixels = np.any(arrows, axis=2)
            right[arrow_pixels] = arrows[arrow_pixels]
            arrow_counts.append(count)
            tip_display = point_in_display(tips[row], origins[row], crop, tuple(viewport))+offset
            if np.isfinite(tip_display).all() and 0 <= tip_display[0] < 900 and 0 <= tip_display[1] < 540:
                current_tip = tuple(np.rint(tip_display).astype(int))
                tip_color = (255, 241, 49) if coordinates[row] == 2 else (175, 183, 187)
                for panel in (left, right):
                    cv2.drawMarker(panel, current_tip, tip_color, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
                    if slip is not None and slip['state'][row] > 0:
                        ring_color = (111, 214, 137) if slip['state'][row] == 4 else (212, 149, 217)
                        cv2.circle(panel, current_tip, 10, ring_color, 1, cv2.LINE_AA)
            canvas = template.copy()
            canvas[84:624, :900], canvas[84:624, 900:] = left, right
            entry = f'{depth[row]:.3f} mm' if np.isfinite(depth[row]) else 'unavailable'
            status = (f'Entry: {entry} | tip: {("unknown", "interpolated", "observed")[int(coordinates[row])]} | '
                      f'blade motion: {("unknown", "conditional", "observed")[int(references[row])]} | '
                      f'interface geometry: {"conditional" if uncertain[row] else "observed"} | '
                      f'visible interface IDs: {interface_counts[-1]} | total saved IDs: {len(ids)}')
            _text(canvas, status, (18, 52), .48)
            if np.isfinite(depth[row]):
                for x, y, w, h in boxes:
                    cursor = int(round(x+w*np.clip((depth[row]-axis[0])/(axis[1]-axis[0]), 0, 1)))
                    cv2.line(canvas, (cursor, 680+round(y)), (cursor, 680+round(y+h)), (214, 222, 225), 1, cv2.LINE_AA)
                x, y, w, h = sliding_box
                cursor = int(round(x+w*np.clip((depth[row]-sliding_axis[0])/(sliding_axis[1]-sliding_axis[0]), 0, 1)))
                cv2.line(canvas, (600+cursor, 950+round(y)), (600+cursor, 950+round(y+h)), (214, 222, 225), 1, cv2.LINE_AA)
            reference = reference_base.copy()
            if slip is not None and slip['state'][row] > 0 and reference_project is not None:
                address = reference_project(slip['address_initial_xy'][row])
                color = (111, 214, 137) if slip['state'][row] == 4 else (212, 149, 217)
                cv2.circle(reference, tuple(np.rint(address).astype(int)), 7, color, 2, cv2.LINE_AA)
                evidence = str(np.asarray(slip.get('evidence_kind', np.full(len(p), '')))[row])
                address_type = ('supported address' if slip['state'][row] == 4 else
                    ('conditional block projection' if 'block' in evidence else 'conditional surface map'))
                _text(canvas, f'Address migration: {slip["migration_um"][row]:+.0f} um | {address_type}', (620, 953), .48, color)
            else:
                _text(canvas, 'Address migration: unavailable', (620, 953), .48, (174, 184, 188))
            canvas[950:1175, :600] = reference
            _text(canvas, 'Initial-address inference uses saved material IDs, not newly seeded texture.', (14, 1193), .40)
            writer.write(canvas)
            written += 1
            if row in selections:
                record = dict(track_index=row, source_frame=int(frame_number), depth_mm=float(depth[row]),
                              roles=selections[row], camera_origin_xy=origins[row].tolist())
                if overview_path is not None:
                    overview = Path(overview_path)
                    overview.parent.mkdir(parents=True, exist_ok=True)
                    target = overview.with_name(overview.stem+'_'+selections[row][0]+'.png')
                    if not cv2.imwrite(str(target), canvas):
                        raise OSError(f'Cannot write snapshot: {target}')
                    record['image_path'] = str(target.resolve())
                    snapshots.append(canvas.copy())
                snapshot_metadata.append(record)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if overview_path is not None and snapshots:
        if not cv2.imwrite(str(overview_path), np.concatenate(snapshots, axis=0)):
            raise OSError(f'Cannot write overview: {overview_path}')
    metadata = dict(schema='loading_full_interface_sliding_review_v2', output_path=str(output),
        rendered_frames=written, playback_fps=float(fps), dimensions=[1800, 1200], codec=codec,
        phase='loading_only', point_count=len(ids), all_saved_ids_retained=True, input_tracks_modified=False,
        camera_mode='whole_interface_bounds_fixed_zoom_smooth_tip', camera_origin_xy=origins.tolist(),
        camera_framing=camera_extra, crop_size=list(crop), blade_occlusion_applied=blade_mask_reader is not None,
        per_frame_blade_occluded_ids=occluded_counts, per_frame_material_points_on_view=shown_counts,
        per_frame_interface_points_on_view=interface_counts, per_frame_local_segment_arrows=arrow_counts,
        axis='indentation_mm_from_first_contact', axis_range_mm=list(axis), state_names=list(STATE_NAMES),
        group_names=list(GROUP_LABELS), region_geometry_uncertain_frames=int(uncertain.sum()),
        group_selection='entire current blade-adjacent material interface, including sparse saved IDs',
        solid_line='supported group trend or supported initial-surface contact address',
        dashed_line='conditional estimate', gray_dotted_line='short temporal bridge', unknown_states_hidden=True,
        display_transform='tanh(relative group velocity / shared 90th-percentile absolute component)',
        display_scale_px_s=scale, physical_time_independent_of_playback_fps=True,
        arrows='local along-blade segment trends only; no repeated whole-side arrows',
        original_material_identity_estimated=slip is not None,
        current_contact_marker='registered blade tip only; no line or marker at a remote current material point',
        initial_address_marker='original configuration inset only', fracture_classified=False,
        source_state_counts_by_group=[[int(np.sum(state[:, group] == k)) for k in range(5)] for group in range(4)],
        surface_sliding_state_counts=None if slip is None else [int(np.sum(slip['state'] == k)) for k in range(5)],
        direct_unfilled_coverage_by_group=(np.isfinite(np.asarray(trend['direct_trend_xy'])).all(-1).mean(0).tolist()
                                          if 'direct_trend_xy' in trend else None),
        legacy_initial_layer_coverage_by_group=(np.isfinite(np.asarray(trend['legacy_trend_y'])).mean(0).tolist()
                                               if 'legacy_trend_y' in trend else None),
        overview_snapshots=snapshot_metadata)
    metadata_path = output.with_suffix('.render.json')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    metadata['metadata_path'] = str(metadata_path)
    return metadata


def render_loading_tip_review(video, positions, supported, output_path, *, frame_indices,
        depth_mm, tip_xy, point_ids, trend, blade_mask_reader, overview_path=None, fps=12., codec='mp4v', slip=None):
    """Render a caller-selected loading interval without changing measurements.

    ``state``: 0 unknown, 1 temporal bridge, 2 conditional estimate (sparse or
    neighbour support, or conditional blade velocity), 3 image assisted,
    4 directly supported.  Dashed lines always denote estimates in
    this renderer; withdrawal data must not be supplied.  Group arrows are
    deliberately aggregate illustrations, never per-point velocity estimates.
    """
    p, support, frames, depth, tips, ids = _validated_arrays(
        positions, supported, frame_indices, depth_mm, tip_xy, point_ids, trend, fps)
    if slip is not None or 'segment_relative_trend_xy' in trend:
        return _render_extended_contact_review(video, p, support, output_path,
            frames=frames, depth=depth, tips=tips, ids=ids, trend=trend,
            blade_mask_reader=blade_mask_reader, overview_path=overview_path,
            fps=fps, codec=codec, slip=slip)
    source, output = Path(video).resolve(), Path(output_path).resolve()
    if source == output or (overview_path is not None and Path(overview_path).resolve() in (source, output)):
        raise ValueError('Review artifacts must not overwrite source data or each other')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise OSError(f'Cannot open display video: {source}')
    writer = None
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop = (min(384, width), min(288, height))
    guide = tips.copy()
    fallback = not np.isfinite(guide[:, 1]).any()
    if fallback:
        guide[:, 1] = height/2
    shaft_x = float(np.nanmedian(tips[:, 0])) if np.isfinite(tips[:, 0]).any() else width/2
    # Keep camera filtering at source-frame cadence even in sparse QA renders.
    # Otherwise four widely separated requested frames would be smoothed as
    # though they were four consecutive images and the camera would lag badly.
    dense_frames = np.arange(frames[0], frames[-1]+1)
    dense_guide = np.column_stack([np.full(len(dense_frames), shaft_x),
        np.interp(dense_frames, frames[np.isfinite(guide[:, 1])], guide[np.isfinite(guide[:, 1]), 1])])
    camera = blade_follow_camera(dense_guide, shaft_x=shaft_x, frame_size=(width, height), crop_size=crop, top=0)
    origins = camera['origin_xy'][frames-frames[0]]
    display_scale = min(800/crop[0], 480/crop[1])
    viewport = (round(crop[0]*display_scale), round(crop[1]*display_scale))
    offset = np.array([(800-viewport[0])//2, (480-viewport[1])//2])
    vx, vy = offset
    vw, vh = viewport
    groups, state = np.asarray(trend['group_index']), np.asarray(trend['state'])
    velocities = np.asarray(trend['relative_trend_xy'], float)
    scale = _shared_scale(velocities, state)
    plots, boxes, axis = _plots(depth, trend, scale)
    template = np.full((1000, 1600, 3), (22, 26, 28), np.uint8)
    _text(template, 'LOADING ONLY | MOTION TRENDS NEAR THE BLADE TIP', (18, 28), .75)
    _text(template, 'Shadow-normalized image | current blade-neighbourhood groups', (18, 82), .54)
    _text(template, 'Same view | points and GROUP trend arrows', (818, 82), .56)
    _text(template, 'Cross: blade tip | filled: supported point | hollow gray: predicted point | blade occludes material ink', (18, 594), .48)
    for group, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        x = 18+group*392
        cv2.circle(template, (x+4, 619), 3, color, -1)
        _text(template, label, (x+16, 624), .48, color)
    _text(template, 'Solid: direct support | colored dashed: conditional estimate | gray dotted: short gap bridge | no withdrawal', (18, 647), .49)
    template[655:955] = plots
    selections = {}
    for target in (0., 1., 2., 3.):
        rows = np.flatnonzero(np.isfinite(depth))
        if rows.size:
            row = int(rows[np.argmin(abs(depth[rows]-target))])
            if abs(depth[row]-target) <= .15:
                selections.setdefault(row, []).append(f'{int(target)}mm')
    snapshots, snapshot_metadata, occluded_counts, shown_counts = [], [], [], []
    observations = np.asarray(trend['observation_count'])
    estimates = np.asarray(trend['estimate_count'])
    reference_state = np.asarray(trend['tip_reference_state'])
    coordinate_state = np.asarray(trend.get('tip_coordinate_state', reference_state))
    geometry_uncertain = np.asarray(trend.get('blade_geometry_uncertain', np.zeros(len(p), bool)), bool)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
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
            left = np.zeros((480, 800, 3), np.uint8)
            left[vy:vy+vh, vx:vx+vw] = crop_for_display(image, origins[row], crop, viewport)
            right = np.full_like(left, (8, 12, 13))
            blade = np.zeros((height, width), bool) if blade_mask_reader is None else np.asarray(blade_mask_reader(int(frame_number)), bool)
            if blade.shape != (height, width):
                raise ValueError('Blade occlusion mask must match source video shape')
            blade_display = crop_for_display(blade.astype(np.uint8)*255, origins[row], crop, viewport) > 0
            right[vy:vy+vh, vx:vx+vw][blade_display] = (31, 36, 38)
            occluded = sample_pixels(blade, p[row])[0].astype(bool)
            occluded_counts.append(int(occluded.sum()))
            q = point_in_display(p[row], origins[row], crop, viewport)+offset
            in_view = (np.isfinite(q).all(1) & (q[:, 0] >= vx) & (q[:, 0] < vx+vw)
                       & (q[:, 1] >= vy) & (q[:, 1] < vy+vh))
            shown_counts.append(int((in_view & ~occluded).sum()))
            ink = np.zeros_like(left)
            for idx in np.flatnonzero(in_view):
                group = int(groups[row, idx])
                color = GROUP_COLORS[group] if group >= 0 else (146, 158, 165)
                if not support[row, idx]:
                    color = (97, 108, 116)
                cv2.circle(ink, tuple(np.rint(q[idx]).astype(int)), 2 if group >= 0 else 1,
                           color, -1 if support[row, idx] else 1, cv2.LINE_AA)
            ink[vy:vy+vh, vx:vx+vw][blade_display] = 0
            pixels = np.any(ink, axis=2)
            blended = cv2.addWeighted(left, .34, ink, .66, 0)
            left[pixels] = blended[pixels]
            right[pixels] = ink[pixels]
            arrows = np.zeros_like(right)
            for group in range(4):
                if state[row, group] <= 0:
                    continue
                vector = _arrow_vector(velocities[row, group], scale)
                if vector is None:
                    continue
                candidates = np.flatnonzero(in_view & ~occluded & support[row] & (groups[row] == group))
                if not candidates.size:
                    continue
                centre = np.median(q[candidates], axis=0)
                idx = candidates[np.argmin(np.linalg.norm(q[candidates]-centre, axis=1))]
                start = np.rint(q[idx]).astype(int)
                color = (173, 181, 185) if state[row, group] == 1 else GROUP_COLORS[group]
                cv2.arrowedLine(arrows, tuple(start), tuple(np.rint(start+vector).astype(int)), color, 2, cv2.LINE_AA, tipLength=.3)
                if state[row, group] < 4:
                    cv2.circle(arrows, tuple(start), 4, color, 1, cv2.LINE_AA)
            arrows[vy:vy+vh, vx:vx+vw][blade_display] = 0
            arrow_pixels = np.any(arrows, axis=2)
            right[arrow_pixels] = arrows[arrow_pixels]
            tip_display = point_in_display(tips[row], origins[row], crop, viewport)+offset
            if np.isfinite(tip_display).all() and 0 <= tip_display[0] < 800 and 0 <= tip_display[1] < 480:
                tip_color = (255, 241, 49) if coordinate_state[row] == 2 else (175, 183, 187)
                for panel in (left, right):
                    cv2.drawMarker(panel, tuple(np.rint(tip_display).astype(int)), tip_color,
                                   cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
            canvas = template.copy()
            canvas[94:574, :800], canvas[94:574, 800:] = left, right
            entry = f'{depth[row]:.3f} mm' if np.isfinite(depth[row]) else 'unavailable'
            tip_label = ('unknown', 'interpolated', 'observed')[int(coordinate_state[row])]
            motion_label = ('unknown', 'conditional', 'observed window')[int(reference_state[row])]
            geometry_label = 'conditional' if geometry_uncertain[row] else 'observed'
            status = (f'Entry: {entry} | LOADING | tip: {tip_label} | blade motion: {motion_label}'
                      f' | region geometry: {geometry_label} | blade-hidden: {int(occluded.sum())} | IDs: {len(ids)}')
            status_width = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, .50, 1)[0][0]
            status_scale = min(.50, .50*1564/max(status_width, 1))
            _text(canvas, status, (18, 54), status_scale)
            if np.isfinite(depth[row]):
                for x, y, w, h in boxes:
                    cursor = int(round(x+w*np.clip((depth[row]-axis[0])/(axis[1]-axis[0]), 0, 1)))
                    cv2.line(canvas, (cursor, 655+round(y)), (cursor, 655+round(y+h)), (214, 222, 225), 1, cv2.LINE_AA)
            for group, color in enumerate(GROUP_COLORS):
                x = 18+group*392
                label = f'{STATE_NAMES[int(state[row, group])]} | obs {int(observations[row, group])}, est {int(estimates[row, group])}'
                _text(canvas, label, (x, 975), .41, color)
            _text(canvas, 'One shared trend scale for x/y and all groups; not an absolute speed scale. Faint bands: empirical spread, not confidence limits.', (18, 994), .43)
            writer.write(canvas)
            written += 1
            if row in selections:
                record = dict(track_index=row, source_frame=int(frame_number), depth_mm=float(depth[row]),
                              roles=selections[row], camera_origin_xy=origins[row].tolist())
                if overview_path is not None:
                    overview = Path(overview_path)
                    overview.parent.mkdir(parents=True, exist_ok=True)
                    target = overview.with_name(overview.stem+'_'+selections[row][0]+'.png')
                    if not cv2.imwrite(str(target), canvas):
                        raise OSError(f'Cannot write snapshot: {target}')
                    record['image_path'] = str(target.resolve())
                    snapshots.append(canvas.copy())
                snapshot_metadata.append(record)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if overview_path is not None and snapshots:
        if not cv2.imwrite(str(overview_path), np.concatenate(snapshots, axis=0)):
            raise OSError(f'Cannot write overview: {overview_path}')
    metadata = dict(schema='loading_tip_review_v1', output_path=str(output), rendered_frames=written,
        playback_fps=float(fps), dimensions=[1600, 1000], codec=codec, phase='loading_only',
        point_count=len(ids), all_saved_ids_retained=True, input_tracks_modified=False,
        camera_mode=str(camera['mode']), camera_fallback=fallback, camera_origin_xy=origins.tolist(),
        crop_size=list(crop), blade_occlusion_applied=blade_mask_reader is not None,
        per_frame_blade_occluded_ids=occluded_counts, per_frame_material_points_on_view=shown_counts,
        axis='indentation_mm_from_first_contact', axis_range_mm=list(axis),
        state_names=list(STATE_NAMES), group_names=list(GROUP_LABELS),
        tip_reference_state='blade velocity source: 2 observed window, 1 conditional/interpolated or boundary window, 0 unknown',
        tip_coordinate_state='separate coordinate source when supplied; otherwise falls back to blade reference state',
        region_geometry_source='provided blade_geometry_uncertain flag, separate from blade coordinate and velocity support; absent flag defaults to observed',
        region_geometry_uncertain_frames=int(geometry_uncertain.sum()),
        group_selection='current geometric neighbourhood, not original-surface identity',
        solid_line='direct supported group trend', dashed_line='conditional estimate: sparse support, neighbour/image assistance or conditional blade velocity',
        gray_dotted_line='explicit short temporal bridge', unknown_states_hidden=True,
        display_transform='tanh(relative group velocity / one shared 90th-percentile absolute component)',
        display_scale_px_s=scale, physical_time_independent_of_playback_fps=True,
        arrows='one aggregate group arrow; illustrative direction and bounded length; not per-point measured velocity',
        original_material_identity_estimated=False, fracture_classified=False,
        source_state_counts_by_group=[[int(np.sum(state[:, group] == k)) for k in range(5)] for group in range(4)],
        direct_unfilled_coverage_by_group=(np.isfinite(np.asarray(trend['direct_trend_xy'])).all(-1).mean(0).tolist()
                                          if 'direct_trend_xy' in trend else None),
        legacy_initial_layer_coverage_by_group=(np.isfinite(np.asarray(trend['legacy_trend_y'])).mean(0).tolist()
                                               if 'legacy_trend_y' in trend else None),
        overview_snapshots=snapshot_metadata)
    metadata_path = output.with_suffix('.render.json')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    metadata['metadata_path'] = str(metadata_path)
    return metadata
