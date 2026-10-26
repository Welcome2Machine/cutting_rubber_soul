"""Forward transport of an ORIGINAL material surface, with observability checks.

The reference polyline Gamma0(s) is labelled by its initial arc length in pixels.
Supported, immutable material trajectories locally identify Phi_t(X); this module
evaluates Phi_t(Gamma0(s)) and finds its possible proximity to a blade tip. It does
NOT project a deep inverse material position onto the initial surface. A new
fracture face cannot acquire an old-surface label through this calculation.

``observed-supported`` describes a material-supported interpolation, not an
independent observation of surface contact. ``model-extrapolated`` includes
outside-hull, missing near-surface evidence and unverified side/visibility cases.
Reported address intervals are geometric/model sensitivity sets, never calibrated
confidence intervals. Only the supplied current frame is used: there is no future
data, temporal smoothing, imposed plateau, monotonicity or microscopy target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class SurfaceObservabilityConfig:
    """Pixel-scale acceptance controls; these are not physical slip limits."""

    model: str = 'affine'
    near_surface_band_px: float = 18.0
    neighbor_counts: tuple[int, ...] = (12, 20, 32, 48)
    minimum_points: int = 6
    maximum_radius_px: float = 80.0
    maximum_condition: float = 100.0
    noise_floor_px: float = 0.25
    maximum_fit_rms_px: float = 2.0
    maximum_heldout_rms_px: float = 3.0
    surface_band_px: float = 3.0
    minimum_surface_points: int = 3
    maximum_extrapolation_px: float = 16.0
    contact_tolerance_px: float = 3.0
    ambiguity_tolerance_px: float = 0.5
    minimum_surface_stretch: float = 0.03
    maximum_surface_gap_px: float = 12.0

    def __post_init__(self):
        if self.model not in ('affine', 'surface-quadratic', 'surface-curve'):
            raise ValueError("model must be affine, surface-quadratic or surface-curve")
        if self.minimum_points < 4 or not self.neighbor_counts:
            raise ValueError("At least four points and one neighborhood are required")
        if min(self.neighbor_counts) < self.minimum_points:
            raise ValueError("Neighborhoods must contain minimum_points")
        if tuple(sorted(set(self.neighbor_counts))) != self.neighbor_counts:
            raise ValueError("Neighborhood sizes must be unique and increasing")
        for name, value in vars(self).items():
            if name not in ("neighbor_counts", "model") and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.near_surface_band_px <= self.surface_band_px:
            raise ValueError("The fitting band must include independent inner-layer support")


def _xy(values, name):
    value = np.asarray(values, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    return value


def _distance_to_polyline(points, line):
    """Euclidean distance; this is support selection, not material-address inversion."""
    segments = np.diff(line, axis=0)
    length2 = np.sum(segments**2, axis=1)
    distance = np.full(len(points), np.inf)
    for start, delta, denom in zip(line[:-1], segments, length2):
        u = np.clip((points-start) @ delta/denom, 0., 1.)
        distance = np.minimum(distance, np.linalg.norm(points-start-u[:, None]*delta, axis=1))
    return distance


def _surface_addresses(points, line):
    """Initial near-surface association only; never an inverse current-tip map."""
    segments = np.diff(line, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    starts = np.r_[0., np.cumsum(lengths)]
    best = np.full(len(points), np.inf)
    address, offset = np.zeros(len(points)), np.zeros(len(points))
    for start, delta, length, arc in zip(line[:-1], segments, lengths, starts):
        t = np.clip((points-start) @ delta/length**2, 0., 1.)
        residual = points-start-t[:, None]*delta
        distance = np.linalg.norm(residual, axis=1)
        update = distance < best
        best[update] = distance[update]
        address[update] = arc+t[update]*length
        offset[update] = residual[update] @ np.array([-delta[1], delta[0]])/length
    return address, best, offset


def _clear_edge(first, second, blade_mask, observation_mask):
    samples = max(2, int(np.ceil(np.linalg.norm(second-first)*2))+1)
    edge = np.linspace(first, second, samples)
    return (not np.any(_inside_mask(edge, blade_mask))
            and (observation_mask is None or not np.any(_inside_mask(edge, ~np.asarray(observation_mask, bool)))))


def _curve_fit(address, displacement, query_s, config):
    """Robust quadratic displacement along initial arc length; rank is 1D.

    A near-surface strip is transported with the same local displacement as its
    boundary. This thin-strip extension is explicit, conditional and cannot
    recover an unknown through-depth displacement gradient.
    """
    if len(address) < config.minimum_points:
        return None
    scale = max(float(np.max(abs(address-query_s))), config.noise_floor_px)
    local = (address-query_s)/scale
    design = np.column_stack((np.ones(len(local)), local, local**2))
    radial = np.exp(-2.*local**2)
    weight = radial.copy()
    for _ in range(6):
        weighted = design*np.sqrt(weight[:, None])
        singular = np.linalg.svd(weighted, compute_uv=False)
        condition = singular[0]/max(singular[-1], 1.e-15)
        if np.linalg.matrix_rank(weighted) < 3 or condition > config.maximum_condition:
            return None
        beta = np.linalg.lstsq(weighted, displacement*np.sqrt(weight[:, None]), rcond=None)[0]
        error = np.linalg.norm(design @ beta-displacement, axis=1)
        sigma = max(config.noise_floor_px, 1.4826*np.median(abs(error-np.median(error))))
        weight = radial*np.minimum(1., 2.5*sigma/np.maximum(error, 1.e-12))
    inlier = error <= max(3.*config.noise_floor_px, 2.5*sigma)
    if inlier.sum() < config.minimum_points:
        return None
    return dict(beta=beta, scale=scale, condition=float(condition), inlier=inlier,
                rms=float(np.sqrt(np.mean(error[inlier]**2))))


def _map_surface_curve(ref, cur, surface, observed, ids, sides, query_side,
                       blade_mask, observation_mask, cfg, result):
    """Fit only connected original near-surface material IDs, never deep points."""
    address, distance, offset = _surface_addresses(ref, surface)
    near = distance <= cfg.surface_band_px
    displacement = cur-ref
    count = len(surface)
    result.update(model_basis='Gamma0(s) + [1,s,s^2] displacement; initial arc-length coordinate',
                  jacobian_definition='Not applicable: a 1D boundary does not identify a 2D deformation gradient',
                  thin_strip_assumption='Near-surface displacement extends to Gamma0 at the same initial arc length; depth gradient is unmeasured',
                  heldout_definition='Leave-one-material-ID-out prediction on the original near-surface strip; no inherited fit weights',
                  curve_component=np.full(count, -1, dtype=int),
                  curve_heldout_rms_px=np.full(count, np.nan),
                  curve_rejected_support_ids=[[] for _ in range(count)],
                  reference_surface_offset_px=[[] for _ in range(count)],
                  maximum_reference_surface_distance_px=np.full(count, np.nan),
                  minimum_reference_surface_distance_px=np.full(count, np.nan))
    # Missing observations interrupt the INITIAL material chain even if its
    # current endpoints are close after compression. Known side labels also split
    # it; current x position is never used to invent sides.
    components = []
    labels = [None] if sides is None else list(np.unique(query_side))
    for label in labels:
        order = np.flatnonzero(near & (True if sides is None else sides == label))
        order = order[np.argsort(address[order], kind='stable')]
        component = []
        for row in order:
            split = not observed[row]
            if component and not split:
                previous = component[-1]
                split = (address[row]-address[previous] > cfg.maximum_surface_gap_px
                         or not _clear_edge(cur[previous], cur[row], blade_mask, observation_mask))
            if split and component:
                components.append((label, np.asarray(component, int)))
                component = []
            if observed[row]:
                component.append(row)
        if component:
            components.append((label, np.asarray(component, int)))
    for i, (query, query_s) in enumerate(zip(surface, result['s_px'])):
        matched = [(number, rows) for number, (label, rows) in enumerate(components)
                   if (query_side is None or label == query_side[i])
                   and address[rows[0]]-1.e-8 <= query_s <= address[rows[-1]]+1.e-8]
        if len(matched) != 1:
            result['reason'][i] = 'surface_support_gap' if not matched else 'overlapping_surface_support_chains'
            result['failure_category'][i] = 'disconnected_surface_support'
            continue
        component, rows = matched[0]
        radius = abs(address[rows]-query_s)
        order = rows[np.argsort(radius, kind='stable')]
        order = order[abs(address[order]-query_s) <= cfg.maximum_radius_px]
        for size in sorted(set(min(n, len(order)) for n in cfg.neighbor_counts)):
            if size <= cfg.minimum_points:
                continue
            use = order[:size]
            use = use[np.argsort(address[use], kind='stable')]
            attempt = dict(neighbors=size, status='rank_condition_or_inlier_failure')
            result['fit_attempts'][i].append(attempt)
            fit = _curve_fit(address[use], displacement[use], query_s, cfg)
            if fit is None:
                result['reason'][i] = 'rank_or_condition_failure'
                result['failure_category'][i] = 'model_identification_failure'
                continue
            errors = []
            # Distinct immutable identities are held out, including robustly
            # rejected observations. Training fit weights cannot hide their error.
            for j in np.unique(np.linspace(0, len(use)-1, min(12, len(use))).astype(int)):
                train = np.arange(len(use)) != j
                held = _curve_fit(address[use[train]], displacement[use[train]], query_s, cfg)
                if held is None:
                    errors.append(np.inf)
                    continue
                z = (address[use[j]]-query_s)/held['scale']
                errors.append(np.linalg.norm(np.array([1., z, z*z]) @ held['beta']-displacement[use[j]]))
            heldout = float(np.sqrt(np.mean(np.square(errors))))
            attempt.update(condition=fit['condition'], fit_rms_px=fit['rms'], heldout_rms_px=heldout,
                           curve_heldout_rms_px=heldout, predicted_xy_px=(query+fit['beta'][0]).tolist())
            if fit['rms'] > cfg.maximum_fit_rms_px or heldout > cfg.maximum_heldout_rms_px:
                result['reason'][i] = attempt['status'] = 'heldout_or_fit_failure'
                result['failure_category'][i] = 'independent_model_prediction_failure'
                continue
            used = use[fit['inlier']]
            # No extrapolation along the surface, nor gaps opened by rejected IDs.
            if query_s < address[used].min()-1.e-8 or query_s > address[used].max()+1.e-8:
                result['reason'][i] = attempt['status'] = 'surface_arc_extrapolation'
                result['failure_category'][i] = 'geometric_extrapolation_exclusion'
                continue
            if not np.all(fit['inlier']):
                rejected_s = address[use[~fit['inlier']]]
                left = address[used][address[used] <= query_s+1.e-8]
                right = address[used][address[used] >= query_s-1.e-8]
                if len(left) and len(right) and np.any((rejected_s >= left.max()) & (rejected_s <= right.min())):
                    result['reason'][i] = attempt['status'] = 'rejected_surface_identity_gap'
                    result['failure_category'][i] = 'disconnected_surface_support'
                    continue
            mapped = query+fit['beta'][0]
            if _inside_mask(mapped[None], blade_mask)[0]:
                result['reason'][i] = attempt['status'] = 'mapped_surface_inside_blade'
                result['failure_category'][i] = 'blade_geometry_exclusion'
                continue
            if observation_mask is not None and _inside_mask(mapped[None], ~np.asarray(observation_mask, bool))[0]:
                result['reason'][i] = attempt['status'] = 'mapped_surface_not_visible'
                result['failure_category'][i] = 'insufficient_image_support'
                continue
            result['mapped_xy_px'][i] = mapped
            result['support_count'][i] = result['surface_support_count'][i] = len(used)
            result['support_ids'][i] = ids[used].tolist()
            result['curve_rejected_support_ids'][i] = ids[use[~fit['inlier']]].tolist()
            result['reference_surface_offset_px'][i] = offset[used].tolist()
            result['maximum_reference_surface_distance_px'][i] = distance[used].max()
            result['minimum_reference_surface_distance_px'][i] = distance[used].min()
            result['curve_component'][i] = component
            for key, value in dict(condition=fit['condition'], fit_rms_px=fit['rms'], heldout_rms_px=heldout,
                                   curve_heldout_rms_px=heldout, hull_extrapolation_px=0.,
                                   radius_px=float(abs(address[use]-query_s).max())).items():
                result[key][i] = value
            result['evidence'][i] = 'model-extrapolated'
            result['reason'][i] = attempt['status'] = 'conditional_thin_strip_transport'
            result['failure_category'][i] = 'conditional_evidence'
            break
    result['segment_valid'] = (np.isfinite(result['mapped_xy_px'][:-1]).all(1)
                               & np.isfinite(result['mapped_xy_px'][1:]).all(1)
                               & (result['curve_component'][:-1] == result['curve_component'][1:]))
    result['segment_reason'] = np.where(result['segment_valid'], 'conditional_curve_segment',
                                        'missing_or_disconnected_support').astype('<U48')
    for i in np.flatnonzero(result['segment_valid']):
        first_s, second_s = result['s_px'][i:i+2]
        rows = components[result['curve_component'][i]][1]
        interior = rows[(address[rows] > first_s+1.e-8) & (address[rows] < second_s-1.e-8)]
        rejected = result['curve_rejected_support_ids'][i]+result['curve_rejected_support_ids'][i+1]
        if np.any(np.isin(ids[interior], rejected)):
            result['segment_valid'][i] = False
            result['segment_reason'][i] = 'rejected_surface_identity_gap'
            continue
        # The output is a polyline. A coarse surface sampling must not turn a
        # validated curved fit into an unvalidated straight bridge between nodes.
        if len(interior):
            fraction = (address[interior]-first_s)/(second_s-first_s)
            end_displacement = result['mapped_xy_px'][i:i+2]-surface[i:i+2]
            linear = end_displacement[0]+fraction[:, None]*(end_displacement[1]-end_displacement[0])
            if np.any(np.linalg.norm(displacement[interior]-linear, axis=1) > cfg.maximum_heldout_rms_px):
                result['segment_valid'][i] = False
                result['segment_reason'][i] = 'surface_polyline_resolution_or_support_error'
                continue
        result['segment_valid'][i] = _clear_edge(*result['mapped_xy_px'][i:i+2], blade_mask, observation_mask)
        if not result['segment_valid'][i]:
            result['segment_reason'][i] = 'blade_or_visibility_gap'
    return result


def _inside_mask(points, mask):
    """Out-of-image coordinates are unverifiable and therefore excluded."""
    if mask is None:
        return np.zeros(len(points), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Masks must have shape (height, width)")
    finite = np.isfinite(points).all(axis=1)
    xy = np.rint(np.where(finite[:, None], points, -1)).astype(int)
    valid = finite & (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    result = np.ones(len(points), dtype=bool)
    result[valid] = mask[xy[valid, 1], xy[valid, 0]]
    return result


def _design(reference, query, scale, frame, model):
    local = (reference-query) @ frame.T/scale
    columns = [np.ones(len(reference)), local[:, 0], local[:, 1]]
    if model == 'surface-quadratic':
        columns.append(local[:, 0]**2)
    return np.column_stack(columns)


def _fit(reference, current, query, config, frame=None):
    """Robust fit with measured design rank; no ridge term hides absent support.

    The optional quadratic model is [1,t,n,t^2] in the INITIAL surface frame.
    Its extra curvature coefficient must survive independent held-out prediction.
    There is no fallback from an unidentifiable quadratic fit to an affine fit.
    """
    if config.model == 'surface-quadratic' and frame is None:
        raise ValueError("surface-quadratic requires an initial tangent/normal frame")
    frame = np.eye(2) if frame is None else np.asarray(frame, float)
    if frame.shape != (2, 2) or not np.allclose(frame @ frame.T, np.eye(2), atol=1.e-6):
        raise ValueError("Initial surface frame must be orthonormal")
    scale = max(np.max(np.linalg.norm(reference-query, axis=1)), config.noise_floor_px)
    design = _design(reference, query, scale, frame, config.model)
    radial = np.exp(-2.*np.sum(design[:, 1:3]**2, axis=1))
    weight = radial.copy()
    beta = None
    for _ in range(4):
        weighted = design*np.sqrt(weight[:, None])
        singular = np.linalg.svd(weighted, compute_uv=False)
        condition = singular[0]/max(singular[-1], 1.e-15)
        if np.linalg.matrix_rank(weighted) < design.shape[1] or condition > config.maximum_condition:
            return None
        beta = np.linalg.lstsq(weighted, current*np.sqrt(weight[:, None]), rcond=None)[0]
        error = np.linalg.norm(design @ beta-current, axis=1)
        sigma = max(config.noise_floor_px, 1.4826*np.median(abs(error-np.median(error))))
        weight = radial*np.minimum(1., 2.5*sigma/np.maximum(error, 1.e-12))
    inlier = error <= max(3.*config.noise_floor_px, 2.5*sigma)
    if inlier.sum() < max(config.minimum_points, design.shape[1]+2):
        return None
    return dict(beta=beta, scale=scale, condition=condition, inlier=inlier,
                frame=frame, model=config.model, rms=float(np.sqrt(np.mean(error[inlier]**2))))


def _predict(fit, reference, query):
    return _design(reference, query, fit['scale'], fit['frame'], fit['model']) @ fit['beta']


def _heldout(reference, current, query, near_surface, config, frame=None):
    """Fit from TRAINING observations only; no inherited full-fit robust weights."""
    errors = []
    # At most twelve spatially distributed leave-one-out samples keep cost bounded.
    selected = np.unique(np.linspace(0, len(reference)-1, min(12, len(reference))).astype(int))
    for j in selected:
        train = np.arange(len(reference)) != j
        fit = _fit(reference[train], current[train], query, config, frame)
        if fit is None:
            return np.inf, np.nan
        errors.append(np.linalg.norm(_predict(fit, reference[j:j+1], query)[0]-current[j]))
    rms = float(np.sqrt(np.mean(np.square(errors))))
    # Hold out the entire near-surface band: a deep-only fit must predict these
    # independent observations. This explicitly exposes boundary/depth mismatch.
    surface_rms = np.nan
    if near_surface.sum() >= config.minimum_surface_points and (~near_surface).sum() >= config.minimum_points:
        fit = _fit(reference[~near_surface], current[~near_surface], query, config, frame)
        if fit is not None:
            err = _predict(fit, reference[near_surface], query)-current[near_surface]
            surface_rms = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    return rms, surface_rms


def _hull_distance(points, query):
    try:
        hull = ConvexHull(points)
    except QhullError:
        return np.inf
    if np.max(hull.equations[:, :2] @ query+hull.equations[:, 2]) <= 1.e-7:
        return 0.
    boundary = points[np.r_[hull.vertices, hull.vertices[0]]]
    return float(_distance_to_polyline(query[None], boundary)[0])


def map_material_surface(reference_points, current_points, supported, initial_surface, *,
                         point_ids=None, point_side=None, surface_side=None,
                         blade_mask=None, observation_mask=None, config=None):
    """Map a labelled initial surface without relabelling interior or fracture points.

    ``point_side`` and ``surface_side`` are independently established contact-side
    labels, not labels inferred from their current x coordinate. When supplied,
    both are mandatory and opposite-side neighbors are excluded. A scalar surface
    side applies to every surface node. ``observation_mask=True`` identifies usable
    current image pixels; unsupported/shadow/occluded observations must be false.
    Without a visibility mask or side labels, results remain conditional.

    NaN current positions are missing observations. IDs must be unique and stable;
    their correspondence across frames is the caller's responsibility. All units
    are image pixels. Reference surface endpoints and intermediate nodes must be
    distinct consecutively. Surface order and arc-length labels never change.

    The default affine model retains its original full-neighborhood behavior.
    ``surface-quadratic`` is explicitly opt-in: neighbors are restricted to the
    initial ``near_surface_band_px`` and [1,t,n,t^2] describes tangential bending
    without mixing the entire depth. ``surface_band_px`` remains the separate
    near-surface observation layer withheld for independent depth validation.
    ``surface-curve`` instead uses only ``surface_band_px`` original near-surface
    identities. A 1D quadratic displacement model follows their initial arc
    length, with no 2D-area rank requirement. This conditional thin-strip model
    reports actual initial support offsets and does not infer a 2D Jacobian.
    """
    cfg = config or SurfaceObservabilityConfig()
    ref = _xy(reference_points, 'reference_points')
    cur = _xy(current_points, 'current_points')
    surface = _xy(initial_surface, 'initial_surface')
    support = np.asarray(supported, dtype=bool)
    if cur.shape != ref.shape or support.shape != (len(ref),):
        raise ValueError("Trajectories and support must share immutable point rows")
    if len(surface) < 2 or not np.isfinite(surface).all() or not np.isfinite(ref).all():
        raise ValueError("Reference geometry must be finite with at least two surface nodes")
    ds = np.linalg.norm(np.diff(surface, axis=0), axis=1)
    if np.any(ds <= 1.e-9):
        raise ValueError("Consecutive surface nodes must differ")
    ids = np.arange(len(ref)) if point_ids is None else np.asarray(point_ids)
    if ids.shape != (len(ref),) or len(np.unique(ids)) != len(ids):
        raise ValueError("point_ids must be unique immutable material identities")
    if (point_side is None) != (surface_side is None):
        raise ValueError("Provide point_side and surface_side together")
    sides = None if point_side is None else np.asarray(point_side)
    query_side = None if surface_side is None else np.broadcast_to(np.asarray(surface_side), (len(surface),))
    if sides is not None and sides.shape != (len(ref),):
        raise ValueError("point_side must have one label per material identity")
    observed = support & np.isfinite(cur).all(axis=1) & ~_inside_mask(cur, blade_mask)
    if observation_mask is not None:
        observed &= ~_inside_mask(cur, ~np.asarray(observation_mask, dtype=bool))
    distance_surface = _distance_to_polyline(ref, surface)
    count = len(surface)
    result = dict(initial_xy_px=surface.copy(), s_px=np.r_[0., np.cumsum(ds)],
                  model=cfg.model, configuration=asdict(cfg),
                  model_basis='[1,x,y]' if cfg.model == 'affine' else '[1,t,n,t^2] in initial surface frame',
                  mapped_xy_px=np.full_like(surface, np.nan),
                  evidence=np.full(count, 'unavailable', dtype='<U24'),
                  reason=np.full(count, 'insufficient_observations', dtype='<U48'),
                  failure_category=np.full(count, 'insufficient_image_support', dtype='<U48'),
                  fit_attempts=[[] for _ in range(count)],
                  support_count=np.zeros(count, dtype=int), surface_support_count=np.zeros(count, dtype=int),
                  support_ids=[[] for _ in range(count)], side_separation_verified=sides is not None,
                  visibility_verified=observation_mask is not None,
                  evidence_definition='observed-supported means material-supported interpolation; it does not prove blade contact',
                  new_fracture_surface_identity_assigned=False)
    for name in ('condition', 'fit_rms_px', 'heldout_rms_px', 'surface_heldout_rms_px',
                 'hull_extrapolation_px', 'radius_px', 'jacobian_determinant', 'minimum_support_jacobian_determinant'):
        result[name] = np.full(count, np.nan)
    if cfg.model == 'surface-curve':
        return _map_surface_curve(ref, cur, surface, observed, ids, sides, query_side,
                                  blade_mask, observation_mask, cfg, result)
    for i, query in enumerate(surface):
        eligible = observed.copy()
        if sides is not None:
            eligible &= sides == query_side[i]
        frame = None
        if cfg.model == 'surface-quadratic':
            eligible &= distance_surface <= cfg.near_surface_band_px
            lo, hi = max(0, i-2), min(count-1, i+2)
            if query_side is not None:
                while lo < i and query_side[lo] != query_side[i]:
                    lo += 1
                while hi > i and query_side[hi] != query_side[i]:
                    hi -= 1
            tangent = surface[hi]-surface[lo]
            if np.linalg.norm(tangent) < 1.e-9:
                result['reason'][i] = 'initial_surface_tangent_unidentifiable'
                result['failure_category'][i] = 'reference_geometry_unidentifiable'
                continue
            tangent /= np.linalg.norm(tangent)
            frame = np.array([tangent, [-tangent[1], tangent[0]]])
        radius = np.linalg.norm(ref-query, axis=1)
        order = np.flatnonzero(eligible & (radius <= cfg.maximum_radius_px))
        order = order[np.argsort(radius[order], kind='stable')]
        sizes = sorted(set(min(n, len(order)) for n in cfg.neighbor_counts))
        for size in sizes:
            if size <= cfg.minimum_points:
                continue  # an independent held-out observation is also required
            use = order[:size]
            attempt = dict(neighbors=size, status='rank_condition_or_inlier_failure')
            result['fit_attempts'][i].append(attempt)
            fit = _fit(ref[use], cur[use], query, cfg, frame)
            if fit is None:
                result['reason'][i] = 'rank_or_condition_failure'
                result['failure_category'][i] = 'model_identification_failure'
                continue
            # Validation includes all accepted input observations, including any
            # robustly rejected ones; a clean training fit cannot hide mismatch.
            near = distance_surface[use] <= cfg.surface_band_px
            heldout, surface_error = _heldout(ref[use], cur[use], query, near, cfg, frame)
            attempt.update(condition=fit['condition'], fit_rms_px=fit['rms'], heldout_rms_px=heldout,
                           surface_heldout_rms_px=surface_error, predicted_xy_px=fit['beta'][0].tolist(),
                           status='heldout_or_fit_failure')
            if fit['rms'] > cfg.maximum_fit_rms_px or heldout > cfg.maximum_heldout_rms_px:
                result['reason'][i] = 'heldout_or_fit_failure'
                result['failure_category'][i] = 'independent_model_prediction_failure'
                continue
            attempt['status'] = 'surface_depth_prediction_mismatch'
            if np.isfinite(surface_error) and surface_error > cfg.maximum_heldout_rms_px:
                result['reason'][i] = 'surface_depth_prediction_mismatch'
                result['failure_category'][i] = 'independent_layer_prediction_failure'
                continue
            used = use[fit['inlier']]
            extrap = _hull_distance(ref[used], query)
            if extrap > cfg.maximum_extrapolation_px:
                result['reason'][i] = 'extrapolation_too_far'
                result['failure_category'][i] = 'geometric_extrapolation_exclusion'
                attempt['status'] = result['reason'][i]
                continue
            mapped = fit['beta'][0]
            if blade_mask is not None and _inside_mask(mapped[None], blade_mask)[0]:
                result['reason'][i] = 'mapped_surface_inside_blade'
                result['failure_category'][i] = 'blade_geometry_exclusion'
                attempt['status'] = result['reason'][i]
                continue
            jac = fit['beta'][1:3].T @ fit['frame']/fit['scale']
            determinant = float(np.linalg.det(jac))
            minimum_determinant = determinant
            if cfg.model == 'surface-quadratic':
                tangential = (ref[used]-query) @ fit['frame'][0]/fit['scale']
                # The determinant is affine in t for this basis: testing its
                # support extrema also excludes folds between the observations.
                for t in (float(tangential.min()), float(tangential.max())):
                    local_jac = fit['beta'][1:3].T.copy()
                    local_jac[:, 0] += 2.*t*fit['beta'][3]
                    minimum_determinant = min(minimum_determinant, float(np.linalg.det(local_jac @ fit['frame']/fit['scale'])))
            if minimum_determinant <= 0:
                result['reason'][i] = 'folded_or_singular_material_map'
                result['failure_category'][i] = 'nonphysical_material_map'
                attempt['status'] = result['reason'][i]
                continue
            result['mapped_xy_px'][i] = mapped
            result['support_count'][i] = len(used)
            result['surface_support_count'][i] = np.sum(distance_surface[used] <= cfg.surface_band_px)
            result['support_ids'][i] = ids[used].tolist()
            for key, value in dict(condition=fit['condition'], fit_rms_px=fit['rms'],
                                   heldout_rms_px=heldout, surface_heldout_rms_px=surface_error,
                                   hull_extrapolation_px=extrap, radius_px=float(radius[use].max()),
                                   jacobian_determinant=determinant,
                                   minimum_support_jacobian_determinant=minimum_determinant).items():
                result[key][i] = value
            visible = observation_mask is not None and not _inside_mask(mapped[None], ~np.asarray(observation_mask, bool))[0]
            fully_supported = (extrap <= 1.e-7 and np.isfinite(surface_error)
                               and result['surface_support_count'][i] >= cfg.minimum_surface_points
                               and sides is not None and visible)
            result['evidence'][i] = 'observed-supported' if fully_supported else 'model-extrapolated'
            result['reason'][i] = 'supported_interpolation' if fully_supported else 'conditional_surface_transport'
            result['failure_category'][i] = 'none' if fully_supported else 'conditional_evidence'
            attempt['status'] = result['reason'][i]
            break
    # A sparse polyline edge must not connect across a missing node, side change,
    # or the blade interior even when both endpoints happen to be outside it.
    result['segment_valid'] = np.isfinite(result['mapped_xy_px'][:-1]).all(1) & np.isfinite(result['mapped_xy_px'][1:]).all(1)
    if query_side is not None:
        result['segment_valid'] &= query_side[:-1] == query_side[1:]
    if blade_mask is not None:
        for i in np.flatnonzero(result['segment_valid']):
            ends = result['mapped_xy_px'][i:i+2]
            samples = max(2, int(np.ceil(np.linalg.norm(ends[1]-ends[0])*2))+1)
            edge = np.linspace(ends[0], ends[1], samples)
            result['segment_valid'][i] = not np.any(_inside_mask(edge, blade_mask))
    return result


def _contact_candidates(mapped, tip, cfg):
    xy, s = mapped['mapped_xy_px'], mapped['s_px']
    pieces = []
    for i in range(len(xy)-1):
        if not mapped['segment_valid'][i]:
            continue  # never bridge an unavailable material-address interval
        vector = xy[i+1]-xy[i]
        length2 = float(vector @ vector)
        u = np.clip(float((tip-xy[i]) @ vector)/max(length2, 1.e-20), 0., 1.)
        distance = float(np.linalg.norm(xy[i]+u*vector-tip))
        pieces.append((i, u, distance, length2))
    if not pieces:
        return []
    best = min(p[2] for p in pieces)
    if best > cfg.contact_tolerance_px:
        return []
    threshold = min(cfg.contact_tolerance_px, best+cfg.ambiguity_tolerance_px)
    candidates = []
    for i, u, distance, length2 in pieces:
        if distance > threshold+1.e-9:
            continue
        # Exact preimage of the pixel-distance tube on each linear surface edge.
        vector = xy[i+1]-xy[i]
        if length2 <= 1.e-20:
            low, high = 0., 1.
        else:
            center = float((tip-xy[i]) @ vector)/length2
            perpendicular2 = max(0., float(np.sum((tip-xy[i])**2))-center**2*length2)
            width = np.sqrt(max(0., threshold**2-perpendicular2)/length2)
            low, high = max(0., center-width), min(1., center+width)
        start, end = s[i]+low*(s[i+1]-s[i]), s[i]+high*(s[i+1]-s[i])
        stretch = np.sqrt(length2)/(s[i+1]-s[i])
        direct = np.all(mapped['evidence'][i:i+2] == 'observed-supported')
        value = dict(s_interval_px=[float(start), float(end)],
                     s_estimate_px=float(s[i]+u*(s[i+1]-s[i])),
                     mapped_contact_xy_px=(xy[i]+u*vector).tolist(), gap_px=distance,
                     evidence='observed-supported' if direct else 'model-extrapolated',
                     minimum_surface_stretch=float(stretch),
                     compression_unidentifiable=bool(stretch < cfg.minimum_surface_stretch),
                     surface_segments=[i])
        if candidates and start <= candidates[-1]['s_interval_px'][1]+1.e-7 and i == candidates[-1]['surface_segments'][-1]+1:
            previous = candidates[-1]
            previous['s_interval_px'][1] = float(end)
            previous['surface_segments'].append(i)
            previous['minimum_surface_stretch'] = min(previous['minimum_surface_stretch'], float(stretch))
            previous['compression_unidentifiable'] |= value['compression_unidentifiable']
            if not direct:
                previous['evidence'] = 'model-extrapolated'
            if distance < previous['gap_px']:
                for key in ('s_estimate_px', 'mapped_contact_xy_px', 'gap_px'):
                    previous[key] = value[key]
        else:
            candidates.append(value)
    return candidates


def contact_from_material_surface(reference_points, current_points, supported,
                                  initial_surface, tip_xy, *, point_ids=None,
                                  point_side=None, surface_side=None, blade_mask=None,
                                  observation_mask=None, config=None):
    """Return possible ORIGINAL-surface addresses at a tip, with genuine gaps.

    A unique candidate is not automatically a measured contact. Compressed surface
    segments may make many addresses indistinguishable; disconnected intervals
    remain separate. No arbitrary nearest deep point is substituted when the
    original surface is absent. Topological evidence is needed to distinguish a
    new fracture face from the original material surface; this API has no such
    evidence and therefore never sets ``contact_observed`` true.
    """
    cfg = config or SurfaceObservabilityConfig()
    tip = np.asarray(tip_xy, dtype=float)
    if tip.shape != (2,) or not np.isfinite(tip).all():
        raise ValueError("tip_xy must be one finite image coordinate")
    mapped = map_material_surface(reference_points, current_points, supported, initial_surface,
                                  point_ids=point_ids, point_side=point_side, surface_side=surface_side,
                                  blade_mask=blade_mask, observation_mask=observation_mask, config=cfg)
    candidates = _contact_candidates(mapped, tip, cfg)
    compressed = any(c['compression_unidentifiable'] for c in candidates)
    state = 'unavailable' if not candidates else 'ambiguous' if len(candidates) > 1 or compressed else 'unique_candidate'
    evidence = 'unavailable' if not candidates else 'observed-supported' if all(c['evidence'] == 'observed-supported' for c in candidates) else 'model-extrapolated'
    return dict(surface=mapped, candidates=candidates, state=state, evidence=evidence,
                contact_observed=False, topology_verified=False,
                new_fracture_surface_identity_assigned=False,
                address_definition='Initial-polyline arc length of forward-mapped ORIGINAL material surface near the tip',
                interval_definition='Geometric pixel-tolerance set, not a confidence interval',
                limitation='Material interpolation and image proximity alone do not establish physical contact or intact topology')
