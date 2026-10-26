"""Supported material/blade kinematics and local geometric recovery evidence.

This postprocessor never tracks images, changes identities, establishes contact
from proximity, or infers elastic energy from displacement. Physical acquisition
timestamps are required; playback frame rate is deliberately not an input.
"""
from dataclasses import asdict, dataclass
import warnings

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ContactMotionConfig:
    near_tip_radius_px: float = 60.
    surface_depth_px: float = 12.
    subsurface_depth_px: float = 48.
    minimum_reference_depth_px: float = -.75
    side_clearance_px: float = 1.
    half_window_s: float = .075
    maximum_gap_s: float = .04
    minimum_samples: int = 5
    minimum_span_s: float = .025
    maximum_position_fit_rms_px: float = 2.
    minimum_group_points: int = 4
    patch_radius_px: float = 18.
    patch_spacing_px: float = 18.
    minimum_patch_points: int = 6
    maximum_patch_points: int = 10
    minimum_patch_pair_length_px: float = 4.
    minimum_shape_distance: float = .01
    minimum_relaxation_rate_s: float = .02
    minimum_blade_advance_px_s: float = .5
    minimum_candidate_duration_s: float = .05
    minimum_baseline_samples: int = 10
    contact_evidence_provenance: str = "not supplied"
    contact_evidence_kind: str = "unverified"  # e.g. image_review or geometric_proximity
    opening_evidence_provenance: str = "not supplied"


GROUP_NAMES = ("left_surface", "left_subsurface", "right_surface", "right_subsurface")
REFERENCES = [
    dict(title="Cutting soft materials: how material differences shape the response",
         year=2026, url="https://www.nature.com/articles/s41524-025-01869-y",
         relevance="Experiments show both abrupt and smooth indentation-to-cutting transitions. Their force data and constitutive/contact model, beyond image kinematics, support interpretation of energy pathways."),
    dict(title="Cutting resistance of soft materials: Effects of blade inclination and friction",
         year=2019, url="https://doi.org/10.1016/j.tafmec.2019.02.017",
         relevance="Elastomer indentation-cutting experiments separate frictional work, strain-energy and fracture-work contributions in an energy balance; relative motion alone does not identify those contributions."),
    dict(title="From pictures to extended finite elements: extended digital image correlation (X-DIC)",
         year=2007, url="https://doi.org/10.1016/j.crme.2007.02.003",
         relevance="The image-correlation formulation explicitly represents displacement discontinuities and demonstrates interface slip. A discontinuity is a kinematic observation, not a unique diagnosis of new fracture."),
]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def _median(values, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _runs(mask):
    diff = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1))


def supported_local_slopes(values, times_s, supported, *, half_window_s=.075,
                           maximum_gap_s=.04, minimum_samples=5, minimum_span_s=.025,
                           maximum_fit_rms=np.inf, motion_step_valid=None):
    """Robust centered slopes within uninterrupted, supported fixed-ID runs.

    values is T,N,D (D can be 1). A missing sample, an explicit invalid edge,
    or a timestamp gap breaks the run; observations on opposite sides of that
    break never enter one fit. First/last samples of each run have no centered
    derivative. Fit RMS is a diagnostic, not a displacement accuracy estimate.
    """
    y = np.asarray(values, float)
    times = np.asarray(times_s, float)
    observed = np.asarray(supported, bool)
    if y.ndim != 3 or observed.shape != y.shape[:2] or times.shape != (len(y),):
        raise ValueError("Expected T,N,D values, T,N support and T physical timestamps")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Physical acquisition timestamps must be finite and increasing")
    if (half_window_s <= 0 or maximum_gap_s <= 0 or minimum_samples < 3
            or minimum_span_s <= 0 or maximum_fit_rms <= 0):
        raise ValueError("Invalid temporal slope settings")
    good = observed & np.isfinite(y).all(2)
    edge = np.zeros(good.shape, bool)
    edge[1:] = good[1:] & good[:-1] & (np.diff(times) <= maximum_gap_s)[:, None]
    if motion_step_valid is not None:
        supplied = np.asarray(motion_step_valid, bool)
        if supplied.shape != good.shape:
            raise ValueError("motion_step_valid must be T,N; row t is edge t-1 to t")
        edge &= supplied
    segments = np.cumsum(~edge, axis=0)
    slope, rms = np.full_like(y, np.nan), np.full(good.shape, np.nan)
    valid, sample_count = np.zeros(good.shape, bool), np.zeros(good.shape, np.int16)
    fit_span = np.full(good.shape, np.nan)
    for t in range(len(times)):
        lo = np.searchsorted(times, times[t]-half_window_s)
        hi = np.searchsorted(times, times[t]+half_window_s, side='right')
        if hi-lo < minimum_samples:
            continue
        dt = times[lo:hi]-times[t]
        keep = good[lo:hi] & (segments[lo:hi] == segments[t]) & good[t]
        begin = np.min(np.where(keep, times[lo:hi, None], np.inf), axis=0)
        end = np.max(np.where(keep, times[lo:hi, None], -np.inf), axis=0)
        # Both sides alone are insufficient: an almost one-sided window makes
        # a curved trajectory's fitted slope refer to a different effective
        # time. Trim to the shorter available side of each fixed-ID run.
        radius = np.minimum(times[t]-begin, end-times[t])
        keep &= np.abs(dt[:, None]) <= radius[None]+1e-12
        count = keep.sum(0)
        earlier, later = keep[dt < 0].any(0), keep[dt > 0].any(0)
        begin = np.min(np.where(keep, times[lo:hi, None], np.inf), axis=0)
        end = np.max(np.where(keep, times[lo:hi, None], -np.inf), axis=0)
        span = end-begin
        eligible = good[t] & earlier & later & (count >= minimum_samples) & (span >= minimum_span_s)
        if not eligible.any():
            continue
        local = np.where(keep[..., None], y[lo:hi], 0.)
        weight = keep.astype(float)
        for _ in range(3):
            sw = weight.sum(0)
            sx = np.sum(weight*dt[:, None], axis=0)
            sxx = np.sum(weight*dt[:, None]**2, axis=0)
            sy = np.sum(weight[..., None]*local, axis=0)
            sxy = np.sum(weight[..., None]*dt[:, None, None]*local, axis=0)
            denominator = sw*sxx-sx*sx
            denominator = np.where(denominator > 1e-14, denominator, np.nan)
            fit = (sw[:, None]*sxy-sx[:, None]*sy)/denominator[:, None]
            intercept = (sy-fit*sx[:, None])/np.maximum(sw[:, None], 1.)
            error = np.linalg.norm(local-intercept[None]-dt[:, None, None]*fit[None], axis=2)
            scale = np.maximum(1e-8, 1.4826*_median(np.where(keep, error, np.nan), axis=0))
            weight = np.nan_to_num(keep*np.minimum(1., 2.5*scale[None]/np.maximum(error, 1e-12)))
        fit_error = np.sqrt(np.sum(np.where(keep, error**2, 0.), axis=0)/np.maximum(count, 1))
        ok = eligible & np.isfinite(fit).all(1) & (fit_error <= maximum_fit_rms)
        slope[t, ok] = fit[ok]
        rms[t, eligible] = fit_error[eligible]
        sample_count[t] = count
        fit_span[t, eligible] = span[eligible]
        valid[t] = ok
    return dict(slope=slope, valid=valid, fit_rms=rms, sample_count=sample_count,
                fit_span_s=fit_span, motion_step_valid=edge)


def _aggregate(values, membership, valid, minimum_points):
    """Aggregate pointwise derivatives, never differentiate a changing centroid."""
    values = np.asarray(values, float)
    result = {name: np.full((len(values), 4), np.nan) for name in
              ('median', 'q25', 'q75', 'positive_fraction', 'negative_fraction')}
    result['count'] = np.zeros((len(values), 4), int)
    result['eligible_count'] = np.zeros((len(values), 4), int)
    for group in range(4):
        member = membership == group
        use = member & valid & np.isfinite(values)
        count = use.sum(1)
        result['count'][:, group], result['eligible_count'][:, group] = count, member.sum(1)
        masked = np.where(use, values, np.nan)
        result['median'][:, group] = _median(masked, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q = np.nanpercentile(masked, [25, 75], axis=1)
        result['q25'][:, group], result['q75'][:, group] = q
        for name, sign in [('positive_fraction', values > 0), ('negative_fraction', values < 0)]:
            result[name][:, group] = np.sum(sign & use, axis=1)/np.maximum(count, 1)
        for name in ('median', 'q25', 'q75', 'positive_fraction', 'negative_fraction'):
            result[name][count < minimum_points, group] = np.nan
    return result


def _fixed_patches(reference, layer, initial_support, cfg):
    """Reference-only compact cohorts; no selection by later motion or survival."""
    patches = []
    for band in range(2):
        eligible = np.flatnonzero((layer == band) & initial_support & np.isfinite(reference).all(1))
        if len(eligible) < cfg.minimum_patch_points:
            continue
        tree = cKDTree(reference[eligible])
        anchors = []
        for point in eligible:
            if anchors and np.min(np.linalg.norm(reference[anchors]-reference[point], axis=1)) < cfg.patch_spacing_px:
                continue
            found = eligible[tree.query_ball_point(reference[point], cfg.patch_radius_px)]
            distance = np.linalg.norm(reference[found]-reference[point], axis=1)
            found = found[np.lexsort((found, distance))][:cfg.maximum_patch_points]
            if len(found) < cfg.minimum_patch_points:
                continue
            pair = np.column_stack(np.triu_indices(len(found), 1))
            lengths = np.linalg.norm(reference[found[pair[:, 1]]]-reference[found[pair[:, 0]]], axis=1)
            pair = pair[lengths >= cfg.minimum_patch_pair_length_px]
            if len(pair) < cfg.minimum_patch_points:
                continue
            anchors.append(point)
            patches.append(dict(member_indices=np.sort(found), layer=band, pair_local_indices=pair,
                # Keep ordering used by pair_local_indices explicitly.
                pair_member_indices=found[pair], reference_pair_length_px=lengths[lengths >= cfg.minimum_patch_pair_length_px]))
    return patches


def analyze_contact_motion(positions, supported, *, times_s, frame_indices, tip_xy,
                           tip_supported, depth_mm, contact_frame, reversal_frame,
                           reference_surface_center, reference_surface_tangent,
                           reference_inward_normal, point_ids=None, reference_xy=None,
                           reference_depth_px=None, um_per_px=None, blade_angle_rad=None,
                           blade_angle_supported=None, observed_contact=None,
                           contact_tangent_xy=None, image_opening_observed=None,
                           motion_step_valid=None, blade_boundary_distance_px=None,
                           config=None):
    """Return a JSON-safe summary and NPZ-ready arrays without changing tracks.

    Image axes: x right, y down. Positive blade angle rotates the image-down
    axis toward image-left. Reference tangent/normal are fixed ORIGINAL-surface
    directions. Local interface-tangent projections need separate supplied
    tangent vectors AND contact evidence; geometric proximity remains a contact
    candidate. Contact/reversal are source frame numbers, not array offsets.
    """
    cfg = config or ContactMotionConfig()
    p, support = np.asarray(positions, float), np.asarray(supported, bool)
    if p.ndim != 3 or p.shape[-1] != 2 or min(p.shape[:2]) < 1 or support.shape != p.shape[:2]:
        raise ValueError("Expected nonempty T,N,2 fixed-ID positions and T,N support")
    nt, count = support.shape
    times, frames = np.asarray(times_s, float), np.asarray(frame_indices)
    tip, tip_good, depth = np.asarray(tip_xy, float), np.array(tip_supported, bool, copy=True), np.asarray(depth_mm, float)
    if (times.shape != (nt,) or frames.shape != (nt,) or tip.shape != (nt, 2)
            or tip_good.shape != (nt,) or depth.shape != (nt,)):
        raise ValueError("All times and blade geometry must align with track rows")
    if (not np.isfinite(frames).all() or not np.all(frames == np.floor(frames))
            or np.any(np.diff(frames) <= 0) or frames[0] < 0):
        raise ValueError("Source frame indices must be increasing nonnegative integers")
    if not np.isfinite([contact_frame, reversal_frame]).all() or reversal_frame < contact_frame:
        raise ValueError("Contact/reversal source-frame landmarks must be ordered and finite")
    if (cfg.surface_depth_px <= cfg.minimum_reference_depth_px or cfg.subsurface_depth_px <= cfg.surface_depth_px
            or cfg.near_tip_radius_px <= 0 or cfg.minimum_group_points < 1
            or cfg.minimum_candidate_duration_s <= 0 or cfg.minimum_baseline_samples < 3
            or cfg.minimum_patch_points < 3 or cfg.maximum_patch_points < cfg.minimum_patch_points):
        raise ValueError("Invalid spatial or evidence settings")
    reference = p[0].copy() if reference_xy is None else np.asarray(reference_xy, float)
    ids = np.arange(count) if point_ids is None else np.asarray(point_ids)
    center, tangent, normal = [np.asarray(v, float) for v in
        (reference_surface_center, reference_surface_tangent, reference_inward_normal)]
    if (reference.shape != (count, 2) or ids.shape != (count,) or len(np.unique(ids)) != count
            or any(v.shape != (2,) or not np.isfinite(v).all() for v in (center, tangent, normal))
            or np.linalg.norm(tangent) < 1e-8 or np.linalg.norm(normal) < 1e-8):
        raise ValueError("Invalid fixed IDs or reference surface geometry")
    tangent, normal = tangent/np.linalg.norm(tangent), normal/np.linalg.norm(normal)
    if abs(tangent@normal) > .02:
        raise ValueError("Reference tangent and inward normal must be perpendicular")
    if um_per_px is not None and (not np.isfinite(um_per_px) or um_per_px <= 0):
        raise ValueError("um_per_px must be a finite positive independently declared scale")
    reference_depth = (reference-center)@normal if reference_depth_px is None else np.asarray(reference_depth_px, float)
    if reference_depth.shape != (count,):
        raise ValueError("reference_depth_px must contain one fixed original-surface depth per ID")
    layer = np.full(count, -1, np.int8)
    layer[(reference_depth >= cfg.minimum_reference_depth_px) & (reference_depth < cfg.surface_depth_px)] = 0
    layer[(reference_depth >= cfg.surface_depth_px) & (reference_depth <= cfg.subsurface_depth_px)] = 1
    use = support & np.isfinite(p).all(2)
    tip_good &= np.isfinite(tip).all(1)
    derivative_kwargs = dict(half_window_s=cfg.half_window_s, maximum_gap_s=cfg.maximum_gap_s,
        minimum_samples=cfg.minimum_samples, minimum_span_s=cfg.minimum_span_s)
    motion = supported_local_slopes(p, times, use, maximum_fit_rms=cfg.maximum_position_fit_rms_px,
        motion_step_valid=motion_step_valid, **derivative_kwargs)
    blade = supported_local_slopes(tip[:, None], times, tip_good[:, None],
        maximum_fit_rms=cfg.maximum_position_fit_rms_px, **derivative_kwargs)
    blade_velocity, blade_valid = blade['slope'][:, 0], blade['valid'][:, 0]
    angle = np.zeros(nt)
    angular_velocity = np.zeros(nt)
    orientation_valid = np.ones(nt, bool)
    if blade_angle_rad is not None:
        angle = np.asarray(blade_angle_rad, float)
        orientation_valid = np.isfinite(angle)
        if angle.shape != (nt,):
            raise ValueError("blade_angle_rad must align with track rows")
        if blade_angle_supported is None:
            raise ValueError("Measured blade angles require explicit blade_angle_supported")
        orientation_valid &= np.asarray(blade_angle_supported, bool)
        if np.asarray(blade_angle_supported).shape != (nt,):
            raise ValueError("blade_angle_supported must align with track rows")
        unwrapped = angle.copy()
        for lo, hi in _runs(orientation_valid):
            unwrapped[lo:hi] = np.unwrap(angle[lo:hi])
        rotation = supported_local_slopes(unwrapped[:, None, None], times,
            orientation_valid[:, None], **derivative_kwargs)
        angular_velocity = rotation['slope'][:, 0, 0]
        blade_valid &= rotation['valid'][:, 0]
    elif blade_angle_supported is not None:
        raise ValueError("blade_angle_supported has no meaning without measured angles")
    blade_axis = np.column_stack((-np.sin(angle), np.cos(angle)))
    blade_right = np.column_stack((np.cos(angle), np.sin(angle)))
    offset = p-tip[:, None]
    spin_velocity = angular_velocity[:, None, None]*np.stack((-offset[..., 1], offset[..., 0]), axis=2)
    relative = motion['slope']-blade_velocity[:, None]-spin_velocity
    relative_valid = motion['valid'] & blade_valid[:, None] & tip_good[:, None]
    relative[~relative_valid] = np.nan
    signed_side = np.einsum('tni,ti->tn', offset, blade_right)
    distance = np.linalg.norm(offset, axis=2)
    near = np.isfinite(p).all(2) & tip_good[:, None] & orientation_valid[:, None] & (distance <= cfg.near_tip_radius_px)
    membership = np.full((nt, count), -1, np.int8)
    for side_index, side in enumerate((-1, 1)):
        for band in range(2):
            membership[near & (side*signed_side >= cfg.side_clearance_px) & (layer[None] == band)] = 2*side_index+band
    arrays = dict(frame_indices=frames.astype(np.int64), physical_time_s=times, depth_mm=depth,
        point_ids=ids, reference_depth_px=reference_depth, reference_layer=layer,
        point_velocity_xy_px_s=motion['slope'], point_velocity_valid=motion['valid'],
        point_velocity_fit_rms_px=motion['fit_rms'], point_velocity_fit_samples=motion['sample_count'],
        point_velocity_fit_span_s=motion['fit_span_s'], blade_velocity_xy_px_s=blade_velocity,
        blade_velocity_valid=blade_valid, blade_angular_velocity_rad_s=angular_velocity,
        relative_velocity_xy_px_s=relative, relative_velocity_valid=relative_valid,
        tip_distance_px=distance, signed_blade_axis_side_px=signed_side,
        group_index=membership, group_names=np.asarray(GROUP_NAMES))
    if blade_boundary_distance_px is not None:
        boundary_distance = np.asarray(blade_boundary_distance_px, float)
        if boundary_distance.shape != (nt, count) or np.any(boundary_distance[np.isfinite(boundary_distance)] < 0):
            raise ValueError("blade_boundary_distance_px must be nonnegative T,N distances or NaN")
        arrays['blade_boundary_distance_px'] = boundary_distance
    components = dict(x=relative[..., 0], y=relative[..., 1],
        original_surface_tangent=np.einsum('tni,i->tn', relative, tangent),
        original_surface_inward_normal=np.einsum('tni,i->tn', relative, normal),
        blade_axis=np.einsum('tni,ti->tn', relative, blade_axis),
        blade_outward=np.einsum('tni,ti->tn', relative, blade_right)*np.sign(signed_side))
    for name, values in components.items():
        arrays[f'relative_{name}_px_s'] = values
        for metric, aggregated in _aggregate(values, membership, relative_valid, cfg.minimum_group_points).items():
            arrays[f'group_{name}_{metric}'] = aggregated
        if um_per_px is not None:
            arrays[f'group_{name}_median_um_s'] = arrays[f'group_{name}_median']*um_per_px
    # Membership overlap is reported because these are Eulerian local trends,
    # even though every derivative above belongs to one persistent material ID.
    overlap = np.full((nt, 4), np.nan)
    for group in range(4):
        a, b = membership[:-1] == group, membership[1:] == group
        overlap[1:, group] = (a & b).sum(1)/np.maximum((a | b).sum(1), 1)
    arrays['group_membership_jaccard_previous'] = overlap
    contact = np.full((nt, count), -1, np.int8)
    if observed_contact is not None:
        supplied = np.asarray(observed_contact)
        if supplied.shape != (nt, count) or not np.isin(supplied, [-1, 0, 1]).all():
            raise ValueError("observed_contact must be T,N tri-state -1 unknown, 0 absent, 1 candidate")
        if np.any(supplied == 1) and cfg.contact_evidence_provenance in ('', 'not supplied'):
            raise ValueError("Contact candidates require declared contact_evidence_provenance")
        contact = supplied.astype(np.int8)
    arrays['supplied_contact_evidence'] = contact
    if contact_tangent_xy is not None:
        interface_tangent = np.asarray(contact_tangent_xy, float)
        if interface_tangent.shape != p.shape:
            raise ValueError("contact_tangent_xy must be T,N,2")
        tangent_length = np.linalg.norm(interface_tangent, axis=2)
        contact_valid = relative_valid & (contact == 1) & np.isfinite(interface_tangent).all(2) & (tangent_length > 1e-8)
        interface_tangent = interface_tangent/np.maximum(tangent_length[..., None], 1e-8)
        interface_value = np.einsum('tni,tni->tn', relative, interface_tangent)
        interface_value[~contact_valid] = np.nan
        arrays['contact_candidate_local_tangent_relative_px_s'] = interface_value
        for metric, aggregated in _aggregate(interface_value, membership, contact_valid, cfg.minimum_group_points).items():
            arrays[f'group_contact_candidate_local_tangent_{metric}'] = aggregated
    opening = np.full(nt, -1, np.int8)
    if image_opening_observed is not None:
        supplied = np.asarray(image_opening_observed)
        if supplied.shape != (nt,) or not np.isin(supplied, [-1, 0, 1]).all():
            raise ValueError("image_opening_observed must be T tri-state -1 unknown, 0 absent, 1 image evidence")
        if np.any(supplied == 1) and cfg.opening_evidence_provenance in ('', 'not supplied'):
            raise ValueError("Image-opening evidence requires declared opening_evidence_provenance")
        opening = supplied.astype(np.int8)
    arrays['supplied_image_opening_evidence'] = opening
    precontact = frames < contact_frame
    loading = (frames >= contact_frame) & (frames <= reversal_frame)
    continuing = loading & blade_valid & (blade_velocity[:, 1] > cfg.minimum_blade_advance_px_s)
    arrays['blade_continued_image_down_advance'] = continuing
    patches = _fixed_patches(reference, layer, use[0], cfg)
    shape = np.full((nt, len(patches)), np.nan)
    patch_good = np.zeros(shape.shape, bool)
    patch_group = np.full(shape.shape, -1, np.int8)
    patch_center = np.full((nt, len(patches), 2), np.nan)
    patch_relative = np.full((nt, len(patches), 2), np.nan)
    patch_edges = np.zeros(shape.shape, bool)
    for k, patch in enumerate(patches):
        idx, pair = patch['member_indices'], patch['pair_member_indices']
        lengths = np.linalg.norm(p[:, pair[:, 1]]-p[:, pair[:, 0]], axis=2)
        valid = use[:, idx].all(1) & (lengths > 1e-6).all(1)
        logs = np.log(np.maximum(lengths, 1e-12)/patch['reference_pair_length_px'])
        shape[valid, k] = np.sqrt(np.mean(logs[valid]**2, axis=1))
        patch_good[:, k] = valid
        patch_center[:, k] = p[:, idx].mean(1)
        patch_edges[:, k] = motion['motion_step_valid'][:, idx].all(1)
        # Every member must be nearby and on the same current side. No patch
        # spanning the blade center line contributes a recovery candidate.
        for group in range(4):
            patch_group[(membership[:, idx] == group).all(1) & valid, k] = group
        moving = relative_valid[:, idx].all(1) & (patch_group[:, k] >= 0)
        patch_relative[moving, k] = relative[moving][:, idx].mean(1)
    patch_rate = supported_local_slopes(shape[..., None], times, patch_good,
        motion_step_valid=patch_edges, **derivative_kwargs) if patches else None
    rates = patch_rate['slope'][..., 0] if patches else np.empty((nt, 0))
    candidate, thresholds = np.zeros(shape.shape, bool), []
    events, patch_description = [], []
    for k, patch in enumerate(patches):
        baseline = rates[precontact, k]
        baseline = baseline[np.isfinite(baseline)]
        adequate = len(baseline) >= cfg.minimum_baseline_samples
        sigma = float(1.4826*np.median(np.abs(baseline-np.median(baseline)))) if adequate else np.nan
        threshold = max(cfg.minimum_relaxation_rate_s, 5*sigma) if adequate else np.nan
        thresholds.append(threshold)
        raw = (continuing & (patch_group[:, k] >= 0) & (shape[:, k] >= cfg.minimum_shape_distance)
               & (rates[:, k] < -threshold))
        for lo, hi in _runs(raw):
            if times[hi-1]-times[lo] < cfg.minimum_candidate_duration_s:
                continue
            candidate[lo:hi, k] = True
            confirmation = min(hi-1, int(np.searchsorted(times, times[lo]+cfg.minimum_candidate_duration_s)))
            events.append(dict(kind='fixed_material_patch_geometric_relaxation_candidate', patch_index=k,
                source_frame_range=[int(frames[lo]), int(frames[hi-1])],
                depth_mm_range=[depth[lo], depth[hi-1]], time_s_range=[times[lo], times[hi-1]],
                first_evidence_source_frame=int(frames[lo]),
                confirmation_source_frame=int(frames[confirmation]),
                most_negative_rate_s=float(np.nanmin(rates[lo:hi, k])),
                reference_shape_distance_at_first=shape[lo, k],
                member_point_ids=ids[patch['member_indices']],
                groups=sorted(set(GROUP_NAMES[j] for j in patch_group[lo:hi, k] if j >= 0)),
                supplied_opening_evidence_overlaps=bool(np.any(opening[lo:hi] == 1)),
                interpretation='Local pair-length geometry moves toward its original configuration while the observed blade still advances. Cause unresolved; not elastic energy or a confirmed fracture onset.'))
        patch_description.append(dict(patch_index=k, member_point_ids=ids[patch['member_indices']],
            member_indices=patch['member_indices'], reference_center_px=reference[patch['member_indices']].mean(0),
            reference_layer='surface' if patch['layer'] == 0 else 'subsurface',
            pair_count=len(patch['pair_member_indices']), baseline_samples=len(baseline),
            baseline_sufficient=adequate, baseline_rate_mad_s=sigma, negative_rate_threshold_s=threshold))
    arrays.update(patch_shape_distance=shape, patch_shape_rate_s=rates, patch_supported=patch_good,
        patch_group_index=patch_group, patch_center_xy=patch_center,
        patch_relative_velocity_xy_px_s=patch_relative,
        patch_relative_original_surface_tangent_px_s=np.einsum('tpi,i->tp', patch_relative, tangent),
        patch_relative_original_surface_inward_normal_px_s=np.einsum('tpi,i->tp', patch_relative, normal),
        patch_relaxation_candidate=candidate, patch_negative_rate_threshold_s=np.asarray(thresholds))
    # Multiple patches sharing IDs are not multiple independent material
    # cohorts. Count a deterministic disjoint subset for the combined flag.
    disjoint_count = np.zeros(nt, int)
    for t in np.flatnonzero(candidate.any(1)):
        used = set()
        for k in np.flatnonzero(candidate[t]):
            members = set(patches[k]['member_indices'].tolist())
            if not members.intersection(used):
                used.update(members)
                disjoint_count[t] += 1
    multiple_intervals = []
    for lo, hi in _runs(disjoint_count >= 2):
        if times[hi-1]-times[lo] < cfg.minimum_candidate_duration_s:
            continue
        multiple_intervals.append(dict(source_frame_range=[int(frames[lo]), int(frames[hi-1])],
            depth_mm_range=[depth[lo], depth[hi-1]], minimum_disjoint_patch_count=int(disjoint_count[lo:hi].min()),
            peak_disjoint_patch_count=int(disjoint_count[lo:hi].max()),
            interpretation='Concurrent geometric relaxation of at least two disjoint fixed-ID patches during continued observed blade advance; common optical-flow errors are still possible. No force/image confirmation is inferred.'))
    arrays['disjoint_relaxing_patch_count'] = disjoint_count
    group_stats = {}
    for group, name in enumerate(GROUP_NAMES):
        selected = membership == group
        observed = selected & relative_valid
        group_stats[name] = dict(initial_layer_ids=int(np.sum(layer == group % 2)),
            nearby_id_observations=int(selected.sum()), supported_velocity_observations=int(observed.sum()),
            supported_velocity_fraction=float(observed.sum()/selected.sum()) if selected.any() else None,
            loading_frames_with_trend=int(np.sum(loading & (arrays['group_x_count'][:, group] >= cfg.minimum_group_points))),
            candidate_patches=len(set(e['patch_index'] for e in events if name in e['groups'])))
    first = min(events, key=lambda event: event['first_evidence_source_frame']) if events else None
    opening_rows = np.flatnonzero((opening == 1) & loading)
    summary = dict(schema='contact_motion_evidence_v1', frames=nt, points=count, config=asdict(cfg),
        physical_time_required=True, playback_fps_used=False, um_per_px=um_per_px,
        contact_frame=int(contact_frame), reversal_frame=int(reversal_frame),
        coordinate_conventions=dict(image='x right, y down', original_surface_tangent=tangent,
            original_surface_inward_normal=normal, blade_rotation='positive angle from image down towards image left',
            relative_velocity='material minus translating blade, and minus omega*J(material_position-tip) when measured'),
        blade_rotation_correction='measured' if blade_angle_rad is not None else 'not measured; translation-only',
        group_names=GROUP_NAMES, groups=group_stats, patches=patch_description,
        candidate_events=sorted(events, key=lambda event: event['first_evidence_source_frame']),
        multi_patch_relaxation_intervals=multiple_intervals,
        first_geometric_relaxation_candidate=first,
        first_supplied_image_opening_source_frame=int(frames[opening_rows[0]]) if opening_rows.size else None,
        physical_fracture_onset=None, energy_release=None,
        accepted_contact_identity_method_changed=False,
        contact_evidence=dict(supplied=observed_contact is not None, kind=cfg.contact_evidence_kind,
            provenance=cfg.contact_evidence_provenance, interpretation='Proximity or a mask interface is a contact candidate, not independent confirmation of mechanical contact; no frictional slip coefficient or force is inferred.'),
        image_opening_evidence=dict(supplied=image_opening_observed is not None,
            provenance=cfg.opening_evidence_provenance,
            interpretation='Image opening is externally supplied evidence, never synthesized from two-sided motion or shape recovery. Its first visible time need not equal physical nucleation.'),
        temporal_interpretation=dict(centered_half_window_s=cfg.half_window_s,
            endpoint_policy='No derivative at run endpoints; no fit bridges an unsupported point, missing blade sample or invalid saved edge.',
            onset_resolution='Centered fits use future samples up to the half-window. Candidate response times are smoothed detector times, not sharp physical onset bounds.'),
        limitations=[
            'Algorithm support, small fitting residuals and smooth curves do not prove persistent physical material identity.',
            'Original-surface/subsurface membership is fixed before tracking; blade proximity cannot turn an interior point into an original-surface point.',
            'Local left/right aggregates have changing spatial membership; pointwise derivatives are measured first, and cohort overlap is reported. Fixed recovery patches never change member IDs.',
            'Geometric recovery is not stored elastic energy. Stress, constitutive behavior, thickness/out-of-plane deformation and dissipation are not determined by these video coordinates.',
            'Slip, changing contact, viscoelastic relaxation, out-of-plane motion, oil/refraction and tracking errors can resemble cutting-related relaxation.',
            'A smooth or hidden crack can occur without a detectable recovery burst. No candidate does not establish absence of fracture.',
            'Continued loading uses image-down blade motion and declared loading phase; uncorrected camera/specimen drift can affect its physical interpretation.',
            'The projected initial-surface components are not the local curved-interface sliding speed. Local tangent/contact evidence is separately required.'
        ], references=REFERENCES)
    return _json_safe(summary), arrays
