"""Current blade-neighborhood trends, separate from initial material identity.

No unsupported saved forecast is an observation. Sparse groups remain visible
as conditional estimates, and one-sided local fits never mix blade sides.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .blade_contact_geometry import sample_pixels


GROUP_NAMES = ('left_near_interface', 'left_inner', 'right_near_interface', 'right_inner')
STATE_NAMES = ('unavailable', 'short_time_bridge', 'local_estimate', 'short_image_assisted', 'supported_group')


@dataclass(frozen=True)
class LoadingTrendConfig:
    radius_px: float = 60.
    expanded_radius_px: float = 80.
    thin_px: float = 12.
    inner_px: float = 40.
    donor_thin_px: float = 32.
    donor_inner_px: float = 55.
    cell_px: float = 6.
    minimum_direct_points: int = 4
    minimum_direct_cells: int = 2
    maximum_extrapolation_px: float = 14.
    maximum_donor_radius_px: float = 42.
    minimum_model_points: int = 5
    maximum_bridge_s: float = .065
    smoothing_half_window_s: float = .03
    interface_attachment_gap_px: float = 8.
    interface_conditional_gap_px: float = 16.
    interface_row_hole_px: int = 2
    interface_segment_px: float = 48.
    interface_balance_cell_px: float = 24.


def weighted_quantile(values, weights, q=.5):
    x, w = np.asarray(values, float), np.asarray(weights, float)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not good.any():
        return np.nan
    x, w = x[good], w[good]
    order = np.argsort(x)
    x, w = x[order], w[order]
    return float(np.interp(q, (np.cumsum(w)-.5*w)/w.sum(), x))


def balanced_weights(points, quality, cell_px=6.):
    """Dense/compressed clusters must not gain confidence just by point count."""
    xy, quality = np.asarray(points, float), np.asarray(quality, float)
    if not len(xy):
        return quality.copy(), 0
    cells = np.floor(xy/cell_px).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    return np.clip(quality, 0., 1.)/counts[inverse], len(counts)


def current_groups(points, tip, gap_px, eligible, config=None, *, radius=None):
    cfg = config or LoadingTrendConfig()
    points, tip = np.asarray(points), np.asarray(tip)
    radius = cfg.radius_px if radius is None else radius
    group = np.full(len(points), -1, np.int8)
    if not np.isfinite(tip).all():
        return group
    offset = points-tip
    valid = (np.asarray(eligible, bool) & np.isfinite(points).all(1)
             & np.isfinite(gap_px) & (np.asarray(gap_px) >= 0)
             & (np.linalg.norm(offset, axis=1) <= radius))
    side = (offset[:, 0] >= 0).astype(int)
    thin = np.asarray(gap_px) < cfg.thin_px
    inner = (np.asarray(gap_px) >= cfg.thin_px) & (np.asarray(gap_px) < cfg.inner_px)
    group[valid & thin] = 2*side[valid & thin]
    group[valid & inner] = 2*side[valid & inner]+1
    return group


def current_material_regions(labels, tip, config=None):
    """Find actual material-pixel targets; absent material is never zero motion."""
    cfg = config or LoadingTrendConfig()
    labels = np.asarray(labels)
    target = np.full((4, 2), np.nan)
    counts = np.zeros(4, int)
    if not np.isfinite(tip).all() or not np.any(labels == 2):
        return dict(target_xy=target, pixel_count=counts)
    distance = cv2.distanceTransform((labels != 2).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = labels.shape
    x0, y0 = np.maximum(np.floor(np.asarray(tip)-cfg.radius_px-1), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(np.asarray(tip)+cfg.radius_px+2), [w, h]).astype(int)
    ys, xs = np.nonzero(labels[y0:y1, x0:x1] == 1)
    xy = np.column_stack((xs+x0, ys+y0)).astype(float)
    if not len(xy):
        return dict(target_xy=target, pixel_count=counts)
    gap = distance[xy[:, 1].astype(int), xy[:, 0].astype(int)]
    groups = current_groups(xy, tip, gap, np.ones(len(xy), bool), cfg)
    for g in range(4):
        members = np.flatnonzero(groups == g)
        counts[g] = len(members)
        if len(members) < 6:
            continue
        desired_gap = 6. if g % 2 == 0 else 24.
        # A medoid is an actual material pixel. No target is projected onto
        # a blade boundary or placed in an empty contact side.
        score = ((gap[members]-desired_gap)/10.)**2 + ((xy[members, 1]-tip[1]+12.)/25.)**2
        target[g] = xy[members[np.argmin(score)]]
    return dict(target_xy=target, pixel_count=counts)


def current_interface_regions(labels, tip, config=None):
    """Actual material bands along both blade flanks, without a tip-radius gate.

    A row is admitted only if material approaches that blade flank within the
    attachment tolerance. Tracing stops at a background run, so a nearby but
    detached free surface cannot supply the subsurface band. This is image
    proximity, not proof of mechanical contact. Segment zero is nearest the
    tip; subsequent segments extend upward along the visible blade.
    """
    cfg = config or LoadingTrendConfig()
    labels, tip = np.asarray(labels), np.asarray(tip, float)
    h, w = labels.shape
    segments = int(np.ceil(h/cfg.interface_segment_px))+1
    group_map = np.full((h, w), -1, np.int8)
    segment_map = np.full((h, w), -1, np.int16)
    geometry_state_map = np.zeros((h, w), np.uint8)
    counts = np.zeros((4, segments), int)
    targets = np.full((4, segments, 2), np.nan)
    geometry_states = np.zeros((4, segments), np.uint8)
    segment_gaps = np.full((4, segments), np.nan)
    segment_lengths = np.zeros((4, segments), float)
    out = dict(group_map=group_map, segment_map=segment_map, geometry_state_map=geometry_state_map,
        segment_pixel_count=counts, segment_target_xy=targets,
        segment_geometry_state=geometry_states, segment_surface_to_blade_gap_px=segment_gaps,
        segment_contact_length_px=segment_lengths,
        segment_center_above_tip_px=(np.arange(segments)+.5)*cfg.interface_segment_px,
        attached_row=np.zeros((h, 2), bool),
        surface_gap_px=np.full((h, 2), np.nan))
    if not np.isfinite(tip).all() or not np.any(labels == 2):
        return out
    # Only the main blade component supplies its two physical flanks. Small
    # isolated class-2 segmentation speckles must not create material bands.
    number, components, stats, _ = cv2.connectedComponentsWithStats(
        (labels == 2).astype(np.uint8), connectivity=8)
    blade = components == (1+np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    distance = cv2.distanceTransform((~blade).astype(np.uint8), cv2.DIST_L2, 5)
    rows = np.flatnonzero(blade.any(1))
    rows = rows[rows <= tip[1]+4.]
    attachment = int(np.floor(cfg.interface_conditional_gap_px))
    # Flanks are nearly vertical in these registered videos. Horizontal rays
    # establish attachment and side; Euclidean blade distance defines layers.
    for y in rows:
        blade_x = np.flatnonzero(blade[y])
        for side, (edge, direction) in enumerate(((blade_x[0], -1), (blade_x[-1], 1))):
            x = edge+direction
            while 0 <= x < w and abs(x-edge) <= attachment and labels[y, x] != 1:
                x += direction
            if not (0 <= x < w) or abs(x-edge) > attachment or labels[y, x] != 1:
                continue
            out['attached_row'][y, side] = True
            out['surface_gap_px'][y, side] = abs(x-edge)
            geometry_state = 2 if abs(x-edge) <= cfg.interface_attachment_gap_px else 1
            surface_distance = distance[y, x]
            holes = 0
            segment = min(segments-1, max(0, int((tip[1]-y)//cfg.interface_segment_px)))
            while 0 <= x < w and distance[y, x]-surface_distance < cfg.inner_px:
                if labels[y, x] == 1:
                    holes = 0
                    group = 2*side+int(distance[y, x]-surface_distance >= cfg.thin_px)
                    group_map[y, x], segment_map[y, x] = group, segment
                    geometry_state_map[y, x] = geometry_state
                else:
                    holes += 1
                    if holes > cfg.interface_row_hole_px or labels[y, x] == 2:
                        break
                x += direction
    yy, xx = np.nonzero(group_map >= 0)
    for group in range(4):
        for segment in np.unique(segment_map[group_map == group]):
            selected = (group_map[yy, xx] == group) & (segment_map[yy, xx] == segment)
            x, y = xx[selected], yy[selected]
            counts[group, segment] = len(x)
            segment_lengths[group, segment] = len(np.unique(y))
            geometry_states[group, segment] = geometry_state_map[y, x].min()
            segment_gaps[group, segment] = np.max(out['surface_gap_px'][np.unique(y), group//2])
            if len(x) < 6:
                continue
            desired_gap = .5*cfg.thin_px if group % 2 == 0 else .5*(cfg.thin_px+cfg.inner_px)
            desired_y = np.median(y)
            score = ((distance[y, x]-desired_gap)/10.)**2 + ((y-desired_y)/20.)**2
            best = np.argmin(score)
            targets[group, segment] = [x[best], y[best]]
    return out


def interface_point_groups(points, regions, eligible=None):
    """Sample current full-interface geometry without assigning material IDs."""
    group, inside = sample_pixels(regions['group_map'], points, fill=-1)
    segment, _ = sample_pixels(regions['segment_map'], points, fill=-1)
    keep = inside if eligible is None else inside & np.asarray(eligible, bool)
    group[~keep], segment[~keep] = -1, -1
    return group, segment


def estimate_interface_frame_trends(points, velocity, valid, material, tip, labels, *,
                                    quality=None, extra=None, config=None, regions=None):
    """Full visible blade-interface trends plus separate fixed-height segments.

    The four-group whole-interface statistic combines segment medians by
    their observed interface length, never their point count. Within each
    segment the original sparse-pitch cells prevent dense clusters dominating.
    Segment curves retain different upstream and near-tip behavior. No
    velocities are inferred across missing segments.
    One or two real sparse observations remain conditional observations.
    """
    from dataclasses import replace
    cfg = config or LoadingTrendConfig()
    aggregate_cfg = replace(cfg, cell_px=cfg.interface_balance_cell_px)
    p, v = np.asarray(points, float), np.asarray(velocity, float)
    q = np.ones(len(p)) if quality is None else np.asarray(quality, float)
    regions = current_interface_regions(labels, tip, cfg) if regions is None else regions
    groups, segments = interface_point_groups(p, regions, material)
    observed = np.asarray(valid, bool) & np.isfinite(v).all(1) & (q > 0)
    ns = regions['segment_pixel_count'].shape[1]
    out = dict(group_index=groups, segment_index=segments,
        relative_trend_xy=np.full((4, 2), np.nan), state=np.zeros(4, np.uint8),
        observation_count=np.zeros(4, int), estimate_count=np.zeros(4, int),
        independent_cells=np.zeros(4, int),
        interval_low_y=np.full(4, np.nan), interval_high_y=np.full(4, np.nan),
        material_pixel_count=regions['segment_pixel_count'].sum(1),
        method=np.full(4, 'unavailable', dtype='<U40'),
        segment_relative_trend_xy=np.full((4, ns, 2), np.nan),
        segment_state=np.zeros((4, ns), np.uint8),
        segment_interval_low_y=np.full((4, ns), np.nan),
        segment_interval_high_y=np.full((4, ns), np.nan),
        segment_geometry_state=regions['segment_geometry_state'],
        segment_surface_to_blade_gap_px=regions['segment_surface_to_blade_gap_px'],
        segment_contact_length_px=regions['segment_contact_length_px'],
        observed_contact_length_fraction=np.zeros(4),
        segment_observation_count=np.zeros((4, ns), int),
        segment_estimate_count=np.zeros((4, ns), int),
        segment_independent_cells=np.zeros((4, ns), int),
        segment_method=np.full((4, ns), 'unavailable', dtype='<U40'),
        segment_pixel_count=regions['segment_pixel_count'],
        segment_target_xy=regions['segment_target_xy'].copy(),
        segment_center_above_tip_px=regions['segment_center_above_tip_px'])
    point_geometry_state, _ = sample_pixels(regions['geometry_state_map'], p)
    out['geometry_state'] = np.zeros(4, np.uint8)
    out['point_geometry_state'] = point_geometry_state
    if extra is None:
        ep, ev, eq = np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    else:
        ep, ev, eq = (np.asarray(extra[k]) for k in ('xy', 'velocity', 'quality'))
    egroups, esegments = interface_point_groups(ep, regions)
    evalid = np.isfinite(ev).all(1) & (eq > 0)

    def statistic(selected, esel, stats_cfg):
        count = int(selected.sum())
        if count:
            value, low, high, cells = summarize_observations(p[selected], v[selected], q[selected], stats_cfg)
            if count >= cfg.minimum_direct_points and cells >= cfg.minimum_direct_cells:
                return value, low, high, cells, 4, 0, 'interface_supported'
        if esel.any():
            pp, vv, qq = np.vstack([p[selected], ep[esel]]), np.vstack([v[selected], ev[esel]]), np.r_[q[selected], eq[esel]]
            value, low, high, cells = summarize_observations(pp, vv, qq, stats_cfg)
            return value, low, high, cells, 3, int(esel.sum()), 'interface_short_image_assisted'
        if count:
            return value, low-2., high+2., cells, 2, 0, 'few_interface_observations'
        return None

    for group in range(4):
        selected = observed & (groups == group)
        esel = evalid & (egroups == group)
        out['observation_count'][group] = int(selected.sum())
        region_states = regions['segment_geometry_state'][group]
        region_states = region_states[region_states > 0]
        out['geometry_state'][group] = int(region_states.min()) if len(region_states) else 0
        stat = statistic(selected, esel, aggregate_cfg)
        if stat is not None:
            value, low, high, cells, state, extras, method = stat
            out['relative_trend_xy'][group] = value
            out['interval_low_y'][group], out['interval_high_y'][group] = low, high
            out['independent_cells'][group], out['state'][group] = cells, state
            out['estimate_count'][group], out['method'][group] = extras, method
            if out['geometry_state'][group] == 1:
                out['state'][group] = min(2, out['state'][group])
        for segment in range(ns):
            selected_segment = selected & (segments == segment)
            extras_segment = esel & (esegments == segment)
            out['segment_observation_count'][group, segment] = int(selected_segment.sum())
            stat = statistic(selected_segment, extras_segment, aggregate_cfg)
            if stat is None:
                continue
            value, low, high, cells, state, extras, method = stat
            out['segment_relative_trend_xy'][group, segment] = value
            out['segment_independent_cells'][group, segment] = cells
            out['segment_state'][group, segment] = state
            if out['segment_geometry_state'][group, segment] == 1:
                out['segment_state'][group, segment] = min(2, state)
            if not np.isfinite(out['segment_target_xy'][group, segment]).all():
                # A newly appearing interface sliver may contain fewer than
                # six semantic pixels yet support a real tracked observation.
                # Keep that conditional trend and anchor it to an actual
                # contributing point, never to a synthetic boundary location.
                candidates = p[selected_segment] if selected_segment.any() else ep[extras_segment]
                center = np.median(candidates, axis=0)
                medoid = candidates[np.argmin(np.linalg.norm(candidates-center, axis=1))]
                out['segment_target_xy'][group, segment] = medoid
                out['segment_state'][group, segment] = min(2, out['segment_state'][group, segment])
            out['segment_estimate_count'][group, segment] = extras
            out['segment_method'][group, segment] = method
            out['segment_interval_low_y'][group, segment] = low
            out['segment_interval_high_y'][group, segment] = high
        # A sparse upstream 48 px segment contributes the same physical length
        # as a dense 48 px segment. More texture points increase its support,
        # but never its share of the whole-interface trend.
        measured = out['segment_state'][group] > 0
        present = regions['segment_pixel_count'][group] >= 6
        length = regions['segment_contact_length_px'][group]
        out['observed_contact_length_fraction'][group] = (
            float(length[measured].sum()/length[present].sum()) if length[present].sum() else 0.)
        if measured.any():
            values, weight = out['segment_relative_trend_xy'][group, measured], length[measured]
            out['relative_trend_xy'][group] = [weighted_quantile(values[:, k], weight) for k in range(2)]
            out['interval_low_y'][group] = weighted_quantile(out['segment_interval_low_y'][group, measured], weight, .25)
            out['interval_high_y'][group] = weighted_quantile(out['segment_interval_high_y'][group, measured], weight, .75)
            out['state'][group] = out['segment_state'][group, measured].min()
            if np.any(present & ~measured):
                out['state'][group] = min(2, out['state'][group])
            out['method'][group] = 'length_balanced_interface_segments'
    return out


def summarize_observations(points, velocity, quality, cfg):
    weight, cells = balanced_weights(points, quality, cfg.cell_px)
    value = np.array([weighted_quantile(velocity[:, k], weight) for k in range(2)])
    q25 = weighted_quantile(velocity[:, 1], weight, .25)
    q75 = weighted_quantile(velocity[:, 1], weight, .75)
    return value, q25, q75, cells


def _visible_to_target(points, target, blade):
    if not np.isfinite(target).all():
        return np.zeros(len(points), bool)
    # Current-image local support must not connect through opaque blade pixels.
    fractions = np.linspace(0., 1., 15)
    samples = target[None, None] + fractions[None, :, None]*(points[:, None]-target)
    hit, inside = sample_pixels(np.asarray(blade, bool), samples)
    return inside.all(1) & ~hit.any(1)


def local_motion_fit(points, velocity, quality, target, config=None):
    """Robust affine velocity near a target; callers supply one material side."""
    from .contact_material_mapping import _hull_distance
    cfg = config or LoadingTrendConfig()
    points, velocity, target = map(lambda x: np.asarray(x, float), (points, velocity, target))
    good = (np.isfinite(points).all(1) & np.isfinite(velocity).all(1)
            & np.isfinite(quality) & (np.asarray(quality) > 0))
    p, v, quality = points[good], velocity[good], np.asarray(quality)[good]
    unavailable = dict(valid=False, velocity=np.full(2, np.nan), low_y=np.nan, high_y=np.nan,
                       method='unavailable', points=len(p), cells=0, residual_px_s=np.nan)
    if len(p) < cfg.minimum_model_points or not np.isfinite(target).all():
        return unavailable
    weight, cells = balanced_weights(p, quality, cfg.cell_px)
    if cells < 3:
        return unavailable
    distance = np.linalg.norm(p-target, axis=1)
    extrapolation = float(_hull_distance(p, target))
    if extrapolation > cfg.maximum_extrapolation_px or distance.min() > 20.:
        return unavailable
    base = weight*np.exp(-.5*(distance/24.)**2)
    design = np.column_stack([np.ones(len(p)), (p-target)/24.])
    rank = np.linalg.matrix_rank(design*np.sqrt(base[:, None]))
    if rank < 3:
        return unavailable
    w = base.copy()
    for _ in range(4):
        normal = design.T@(w[:, None]*design) + np.diag([1e-9, .002, .002])*max(w.sum(), 1.)
        coefficients = np.linalg.solve(normal, design.T@(w[:, None]*v))
        residual = np.linalg.norm(v-design@coefficients, axis=1)
        sigma = max(.5, 1.4826*weighted_quantile(residual, base))
        w = base*np.minimum(1., 2.*sigma/np.maximum(residual, 1e-9))
    prediction = coefficients[0]
    rms = float(np.sqrt(np.average(np.sum((v-design@coefficients)**2, axis=1), weights=base)))
    median = np.array([weighted_quantile(v[:, k], base) for k in range(2)])
    spread = np.array([1.4826*weighted_quantile(abs(v[:, k]-median[k]), base) for k in range(2)])
    # Reject explosive extrapolations; never clip them to manufacture a result.
    if np.any(abs(prediction-median) > 3.*np.maximum(spread, 2.)):
        return unavailable
    width = max(1., rms, spread[1])*(1.+extrapolation/12.)
    return dict(valid=True, velocity=prediction, low_y=float(prediction[1]-width),
        high_y=float(prediction[1]+width), method='same_side_local_affine', points=len(p), cells=cells,
        residual_px_s=rms, extrapolation_px=extrapolation)


def estimate_frame_trends(points, velocity, valid, gap_px, material, tip, labels, *,
                          quality=None, extra=None, config=None):
    """Use current geometry first, local-image observations/neighbor fit second.

    Velocity is already relative to the blade. extra is an independent short
    image observation set with xy, velocity, quality and gap_px; no original IDs.
    """
    cfg = config or LoadingTrendConfig()
    p, v = np.asarray(points, float), np.asarray(velocity, float)
    quality = np.ones(len(p)) if quality is None else np.asarray(quality, float)
    eligible = np.asarray(material, bool) & np.isfinite(gap_px) & (np.asarray(gap_px) >= 0)
    observed = np.asarray(valid, bool) & eligible & np.isfinite(v).all(1) & (quality > 0)
    groups = current_groups(p, tip, gap_px, eligible, cfg)
    regions = current_material_regions(labels, tip, cfg)
    out = dict(group_index=groups, relative_trend_xy=np.full((4, 2), np.nan),
        direct_trend_xy=np.full((4, 2), np.nan), state=np.zeros(4, np.uint8),
        observation_count=np.zeros(4, int), estimate_count=np.zeros(4, int),
        independent_cells=np.zeros(4, int), radius_px=np.full(4, cfg.radius_px),
        interval_low_y=np.full(4, np.nan), interval_high_y=np.full(4, np.nan),
        target_xy=regions['target_xy'], material_pixel_count=regions['pixel_count'],
        method=np.full(4, 'unavailable', dtype='<U40'), model_residual_px_s=np.full(4, np.nan))
    if not np.isfinite(tip).all():
        return out
    if extra is None:
        extra = dict(xy=np.empty((0, 2)), velocity=np.empty((0, 2)), quality=np.empty(0), gap_px=np.empty(0))
    ep, ev, eq, eg = (np.asarray(extra[k]) for k in ('xy', 'velocity', 'quality', 'gap_px'))
    egroup = current_groups(ep, tip, eg, np.ones(len(ep), bool), cfg)
    for g in range(4):
        if regions['pixel_count'][g] < 6:
            out['method'][g] = 'no_material_in_band'
            continue
        selected = observed & (groups == g)
        out['observation_count'][g] = int(selected.sum())
        direct = None
        if selected.any():
            direct = summarize_observations(p[selected], v[selected], quality[selected], cfg)
            out['direct_trend_xy'][g] = direct[0]
            out['independent_cells'][g] = direct[3]
        if direct is not None and selected.sum() >= cfg.minimum_direct_points and direct[3] >= cfg.minimum_direct_cells:
            value, low, high, _ = direct
            out['relative_trend_xy'][g], out['interval_low_y'][g], out['interval_high_y'][g] = value, low, high
            out['state'][g], out['method'][g] = 4, 'current_band_supported'
            continue
        # Short-image observations are corroboration, not duplicate material IDs.
        esel = (egroup == g) & np.isfinite(ev).all(1) & (eq > 0)
        if esel.any():
            cp, cv, cq = np.vstack([p[selected], ep[esel]]), np.vstack([v[selected], ev[esel]]), np.r_[quality[selected], eq[esel]]
            value, low, high, cells = summarize_observations(cp, cv, cq, cfg)
            if cells >= 2 and len(cp) >= 3:
                out['relative_trend_xy'][g], out['interval_low_y'][g], out['interval_high_y'][g] = value, low, high
                out['state'][g], out['method'][g] = 3, 'short_image_assisted'
                out['estimate_count'][g] = int(esel.sum())
                continue
        # Only the inner layer expands its spatial footprint as a fallback.
        if g % 2:
            expanded = current_groups(p, tip, gap_px, eligible, cfg, radius=cfg.expanded_radius_px)
            wide = observed & (expanded == g)
            if wide.sum() >= 4:
                value, low, high, cells = summarize_observations(p[wide], v[wide], quality[wide], cfg)
                if cells >= 2:
                    out['relative_trend_xy'][g], out['interval_low_y'][g], out['interval_high_y'][g] = value, low, high
                    out['state'][g], out['method'][g] = 2, 'expanded_inner_neighborhood'
                    out['estimate_count'][g], out['radius_px'][g] = int(wide.sum()), cfg.expanded_radius_px
                    continue
        target = regions['target_xy'][g]
        same_side = (p[:, 0] < tip[0]) if g < 2 else (p[:, 0] >= tip[0])
        maxgap = cfg.donor_thin_px if g % 2 == 0 else cfg.donor_inner_px
        donors = (observed & same_side & (np.asarray(gap_px) < maxgap)
                  & (np.linalg.norm(p-target, axis=1) <= cfg.maximum_donor_radius_px))
        donor_ids = np.flatnonzero(donors)
        clear = _visible_to_target(p[donor_ids], target, np.asarray(labels) == 2)
        donor_ids = donor_ids[clear]
        fit = local_motion_fit(p[donor_ids], v[donor_ids], quality[donor_ids], target, cfg)
        if fit['valid']:
            out['relative_trend_xy'][g] = fit['velocity']
            out['interval_low_y'][g], out['interval_high_y'][g] = fit['low_y'], fit['high_y']
            out['state'][g], out['method'][g] = 2, fit['method']
            out['estimate_count'][g] = fit['points']
            out['model_residual_px_s'][g] = fit['residual_px_s']
            continue
        # One to three supported local points are useful for a trend, although
        # they are not a fully supported regional statistic.
        if direct is not None:
            value, low, high, _ = direct
            out['relative_trend_xy'][g] = value
            out['interval_low_y'][g], out['interval_high_y'][g] = low-2., high+2.
            out['state'][g], out['method'][g] = 2, 'few_supported_local_points'
        elif esel.any():
            value, low, high, _ = summarize_observations(ep[esel], ev[esel], eq[esel], cfg)
            out['relative_trend_xy'][g] = value
            out['interval_low_y'][g], out['interval_high_y'][g] = low-3., high+3.
            out['state'][g], out['method'][g] = 3, 'few_short_image_points'
            out['estimate_count'][g] = int(esel.sum())
    return out


def finalize_series(trend, times_s, blade_reference_state, config=None):
    """Short, bounded gaps only; no long prediction chain or monotonic prior."""
    cfg = config or LoadingTrendConfig()
    time = np.asarray(times_s, float)
    values = trend['relative_trend_xy'].copy()
    trend['unsmoothed_trend_xy'] = values.copy()
    state = trend['state'].copy()
    blade_state = np.asarray(blade_reference_state)
    state[(blade_state < 2)[:, None] & (state > 0)] = 2
    state[blade_state == 0] = 0
    values[blade_state == 0] = np.nan
    for g in range(4):
        valid = (state[:, g] > 0) & np.isfinite(values[:, g]).all(1)
        rows = np.flatnonzero(valid)
        for a, b in zip(rows[:-1], rows[1:]):
            if b-a <= 1 or time[b]-time[a] > cfg.maximum_bridge_s:
                continue
            if np.any(trend['material_pixel_count'][a:b+1, g] < 6) or np.any(blade_state[a:b+1] == 0):
                continue
            # A large sign change must remain a measured transition, not an
            # invented bridge through an unobserved fracture/slip event.
            va, vb = values[a, g, 1], values[b, g, 1]
            if va*vb < 0 and min(abs(va), abs(vb)) > 2.:
                continue
            alpha = (time[a+1:b]-time[a])/(time[b]-time[a])
            values[a+1:b, g] = values[a, g]+alpha[:, None]*(values[b, g]-values[a, g])
            state[a+1:b, g] = 1
            trend['method'][a+1:b, g] = 'bounded_time_bridge'
            trend['interval_low_y'][a+1:b, g] = min(trend['interval_low_y'][a, g], trend['interval_low_y'][b, g])
            trend['interval_high_y'][a+1:b, g] = max(trend['interval_high_y'][a, g], trend['interval_high_y'][b, g])
        # A three-sample median reduces pixel noise without imposing a direction
        # or letting evidence travel across missing rows. Raw values are saved.
        smoothed = values[:, g].copy()
        for t in range(1, len(time)-1):
            if (state[t-1:t+2, g] > 0).all() and max(time[t]-time[t-1], time[t+1]-time[t]) <= cfg.smoothing_half_window_s:
                window = values[t-1:t+2, g]
                strong_switch = ((window[:-1]*window[1:] < 0)
                    & (np.minimum(abs(window[:-1]), abs(window[1:])) > 2.)).any(axis=0)
                smoothed[t] = np.where(strong_switch, values[t, g], np.median(window, axis=0))
        values[:, g] = smoothed
    trend['relative_trend_xy'], trend['state'] = values, state
    trend['interval_low_y'] = np.minimum(trend['interval_low_y'], values[..., 1])
    trend['interval_high_y'] = np.maximum(trend['interval_high_y'], values[..., 1])
    trend['tip_reference_state'] = blade_state
    return trend
