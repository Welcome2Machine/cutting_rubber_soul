"""Independent transport of one immutable original-surface material address.

No current contact address or blade-contact equality enters this estimator.
The image contour can correct the *normal* location of a free material map;
it cannot replace its tangential material identity. Hidden positions remain
labelled predictions, including when a complete contour is available.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FixedSurfaceOriginConfig:
    reference_half_windows_px: tuple = (24., 42., 70.)
    reference_depth_limits_px: tuple = (18., 40., 80.)
    minimum_points: int = 8
    minimum_address_span_px: float = 10.
    minimum_depth_span_px: float = 5.
    coordinate_scale_px: float = 30.
    shallow_depth_px: float = 5.
    maximum_rms_px: float = 3.
    maximum_holdout_px: float = 5.
    maximum_condition: float = 1.e6
    maximum_stretch: float = 12.
    minimum_jacobian: float = .002
    position_noise_floor_px: float = .35
    maximum_normal_correction_px: float = 24.
    directly_supported_normal_correction_px: float = 4.
    maximum_tangential_correction_px: float = 3.
    boundary_ambiguity_arc_separation_px: float = 24.
    boundary_ambiguity_distance_px: float = 2.
    prediction_acceleration_px_s2: float = 4.
    increment_maximum_gap_s: float = .3
    increment_current_radii_px: tuple = (18., 30., 48., 64.)
    increment_reference_half_windows_px: tuple = (24., 42., 70.)
    increment_maximum_nearest_support_px: float = 18.
    increment_maximum_hull_extrapolation_px: float = 12.
    increment_maximum_rms_px: float = .75
    increment_maximum_holdout_px: float = 1.
    blade_side_margin_px: float = 2.


def _sample(labels, points):
    if labels is None:
        return np.ones(len(points), np.uint8)
    xy = np.rint(np.nan_to_num(points, nan=-1.e8)).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < labels.shape[1])
    inside &= (xy[:, 1] >= 0) & (xy[:, 1] < labels.shape[0])
    out = np.zeros(len(points), labels.dtype)
    out[inside] = labels[xy[inside, 1], xy[inside, 0]]
    return out


def _blocked(a, b, labels):
    if labels is None or not np.isfinite([a, b]).all():
        return False
    length = np.linalg.norm(b-a)
    points = a+(b-a)*np.linspace(0., 1., max(2, int(np.ceil(length*2.))+1))[:, None]
    return bool(np.any(_sample(labels, points) == 2))


def _motion_crosses_blade(a, b, labels):
    """Only a visible-to-visible segment can establish a current-mask crossing.

    A moving blade may cover yesterday's image coordinate. Treating that old
    coordinate as a material point *in today's image* creates permanent locks.
    Hidden endpoints instead retain a labelled latent material trajectory.
    """
    if labels is None or np.any(_sample(labels, np.asarray([a, b])) == 2):
        return False
    return _blocked(a, b, labels)


def _robust_fit(X, y, weights, cfg):
    if len(X) < X.shape[1]+2 or np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    base = np.asarray(weights, float)
    w = base.copy()
    for _ in range(4):
        A = X.T@(w[:, None]*X)
        if np.linalg.cond(A) > cfg.maximum_condition:
            return None
        beta = np.linalg.solve(A, X.T@(w[:, None]*y))
        residual = np.linalg.norm(X@beta-y, axis=1)
        scale = max(cfg.position_noise_floor_px, 1.4826*np.median(abs(residual-np.median(residual))))
        w = base*np.minimum(1., 2.5*scale/np.maximum(residual, 1.e-12))
    A = X.T@(w[:, None]*X)
    beta = np.linalg.solve(A, X.T@(w[:, None]*y))
    rms = float(np.sqrt(np.average(np.sum((X@beta-y)**2, axis=1), weights=base)))
    return dict(beta=beta, rms=rms, inverse=np.linalg.inv(A))


class FixedSurfaceOriginTracker:
    """Preserve P0 independently of current contact-address estimation.

    States: 1 shallow-ID support, 2 deeper/large-normal extrapolation,
    3 same-ID incremental propagation, 4 prediction without a position fit.
    ``xy`` is on the supplied contour only when ``on_boundary`` is true.
    A missing/ambiguous contour never causes reassignment to a different ID.
    """
    def __init__(self, seed_points, point_ids, initial_reference, initial_s_px, config=None):
        self.config = config or FixedSurfaceOriginConfig()
        self.seed = np.asarray(seed_points, float)
        self.ids = np.asarray(point_ids, np.int64)
        if self.seed.ndim != 2 or self.seed.shape[1] != 2 or self.ids.shape != (len(self.seed),):
            raise ValueError('Seed positions and immutable IDs must agree')
        if len(np.unique(self.ids)) != len(self.ids):
            raise ValueError('Material IDs must be unique')
        ref = initial_reference
        boundary = np.asarray(ref['boundary_xy'], float)
        address = np.asarray(ref['surface_s_px'], float)
        if np.any(np.diff(address) <= 0) or not address[0] <= initial_s_px <= address[-1]:
            raise ValueError('P0 must lie on the ordered original material surface')
        self.s0 = float(initial_s_px)
        self.initial_xy = np.array([np.interp(self.s0, address, boundary[:, j]) for j in (0, 1)])
        self.tangent = np.array(ref['tangent'], dtype=float, copy=True)
        self.tangent /= np.linalg.norm(self.tangent)
        self.normal = np.array(ref['inward_normal'], dtype=float, copy=True)
        self.normal -= np.dot(self.normal, self.tangent)*self.tangent
        self.normal /= np.linalg.norm(self.normal)
        self.axes = np.column_stack((self.tangent, self.normal))
        along = boundary@self.tangent
        if np.any(np.diff(along) <= 0):
            raise ValueError('Initial surface must be ordered along its reference tangent')
        coordinate = self.seed@self.tangent
        self.s = np.interp(coordinate, along, address)
        self.depth = self.seed@self.normal-np.interp(coordinate, along, boundary@self.normal)
        cfg = self.config
        member = (coordinate >= along[0]) & (coordinate <= along[-1]) & (self.depth >= -.5)
        self.fixed = member & (abs(self.s-self.s0) <= max(cfg.reference_half_windows_px))
        self.fixed &= self.depth <= max(cfg.reference_depth_limits_px)
        self.fixed_indices = np.flatnonzero(self.fixed)
        self.fixed_ids = self.ids[self.fixed].copy()
        self.time = None
        self.raw = self.initial_xy.copy()
        self.xy = self.initial_xy.copy()
        self.velocity = np.zeros(2)
        self.sigma = cfg.position_noise_floor_px
        self.last_observation_time = None
        self.previous_points = None
        self.previous_support = None
        self.side = 0

    def _absolute(self, points, support):
        cfg = self.config
        candidates = []
        for half in cfg.reference_half_windows_px:
            for limit in cfg.reference_depth_limits_px:
                q = np.flatnonzero(self.fixed & support & (abs(self.s-self.s0) <= half) & (self.depth <= limit))
                if (len(q) < cfg.minimum_points or np.ptp(self.s[q]) < cfg.minimum_address_span_px
                        or np.ptp(self.depth[q]) < cfg.minimum_depth_span_px):
                    continue
                # P0 must be bracketed in original material address; a nearby
                # one-sided cloud is not a replacement material identity.
                if not self.s[q].min() <= self.s0 <= self.s[q].max():
                    continue
                X = np.column_stack((np.ones(len(q)), (self.s[q]-self.s0)/cfg.coordinate_scale_px,
                                     self.depth[q]/cfg.coordinate_scale_px))
                cells = np.floor(np.column_stack((self.s[q], self.depth[q]))/6.).astype(int)
                _, inverse, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
                weights = np.exp(-.5*((self.s[q]-self.s0)/half)**2-.5*(self.depth[q]/limit)**2)/count[inverse]
                fit = _robust_fit(X, points[q], weights, cfg)
                if fit is None:
                    continue
                losses = []
                # Whole shallow-band holdout exposes surface extrapolation
                # error that a random neighbour split can conceal.
                splits = [self.depth[q] <= np.quantile(self.depth[q], .25),
                          np.floor((self.s[q]-self.s0)/12.).astype(int) % 3 == 0]
                for test in splits:
                    if test.sum() < 2:
                        continue
                    held = _robust_fit(X[~test], points[q][~test], weights[~test], cfg)
                    if held is not None:
                        losses.append(float(np.sqrt(np.average(
                            np.sum((X[test]@held['beta']-points[q][test])**2, axis=1), weights=weights[test]))))
                if not losses:
                    continue
                holdout = max(losses)
                J = fit['beta'][1:].T/cfg.coordinate_scale_px
                det = float(np.linalg.det(J)*np.linalg.det(self.axes))
                if (fit['rms'] > cfg.maximum_rms_px or holdout > cfg.maximum_holdout_px
                        or det < cfg.minimum_jacobian or np.linalg.svd(J, compute_uv=False)[0] > cfg.maximum_stretch):
                    continue
                noise = max(cfg.position_noise_floor_px, fit['rms'], holdout)
                sigma = noise*np.sqrt(1.+max(0., fit['inverse'][0, 0]))
                # This explicit extrapolation term is a sensitivity allowance,
                # not a calibrated statistical confidence interval.
                sigma = float(np.hypot(sigma, .08*max(0., self.depth[q].min())))
                tangent = J[:, 0]/max(np.linalg.norm(J[:, 0]), 1.e-12)
                state = 1 if np.count_nonzero(self.depth[q] <= cfg.shallow_depth_px) >= 3 else 2
                candidates.append(dict(xy=fit['beta'][0], tangent=tangent, sigma=sigma, state=state,
                    indices=q, rms=fit['rms'], holdout=holdout, minimum_depth=float(self.depth[q].min()),
                    score=fit['rms']+.5*holdout+.04*self.depth[q].min()+.001*half,
                    jacobian=det, half_window=half, depth_limit=limit))
        return min(candidates, key=lambda x:x['score']) if candidates else None

    def _increment_legacy(self, points, support, dt):
        """Frozen broad-neighbourhood baseline for paired validation only."""
        cfg = self.config
        if self.previous_points is None or dt > cfg.increment_maximum_gap_s:
            return None
        q = np.flatnonzero(self.fixed & support & self.previous_support)
        if len(q) < cfg.minimum_points:
            return None
        relative = (self.previous_points[q]-self.raw)/cfg.coordinate_scale_px
        X = np.column_stack((np.ones(len(q)), relative))
        weights = np.exp(-.5*np.sum(relative**2, axis=1)/4.)
        fit = _robust_fit(X, points[q]-self.previous_points[q], weights, cfg)
        if fit is None or fit['rms'] > cfg.maximum_rms_px:
            return None
        F = np.eye(2)+fit['beta'][1:].T/cfg.coordinate_scale_px
        if np.linalg.det(F) <= cfg.minimum_jacobian or np.linalg.svd(F, compute_uv=False)[0] > cfg.maximum_stretch:
            return None
        error = max(cfg.position_noise_floor_px, fit['rms'])*np.sqrt(1.+max(0., fit['inverse'][0, 0]))
        return dict(xy=self.raw+fit['beta'][0], sigma=float(np.hypot(self.sigma, error)),
                    indices=q, rms=fit['rms'])

    def _increment(self, points, support, dt):
        """Local same-ID motion, with measured support for boundary extrapolation.

        The reference neighbourhood never expands or changes material identity.
        Current-distance windows prevent a deeply deformed remote part of that
        neighbourhood from dominating the velocity at P0. Whole closest-point
        and reference-address bands are held out; a rejected fit is no evidence.
        """
        from scipy.spatial import ConvexHull, QhullError

        cfg = self.config
        if self.previous_points is None or dt > cfg.increment_maximum_gap_s:
            return None
        base = self.fixed & support & self.previous_support
        distance = np.linalg.norm(self.previous_points-self.raw, axis=1)
        candidates = []
        for radius in cfg.increment_current_radii_px:
            seen_neighbourhoods = set()
            for half in cfg.increment_reference_half_windows_px:
                q = np.flatnonzero(base & (distance <= radius) & (abs(self.s-self.s0) <= half))
                if len(q) < cfg.minimum_points:
                    continue
                key = q.tobytes()
                if key in seen_neighbourhoods:
                    continue
                seen_neighbourhoods.add(key)
                previous = self.previous_points[q]
                nearest = float(distance[q].min())
                if nearest > cfg.increment_maximum_nearest_support_px:
                    continue
                try:
                    hull = ConvexHull(previous)
                except QhullError:
                    continue
                # Signed convex-hull location is an actual geometric distance,
                # not a count-based confidence boost from a dense point cloud.
                inside = np.all(hull.equations[:, :2]@self.raw+hull.equations[:, 2] <= 1.e-7)
                vertices = previous[hull.vertices]
                a, delta = vertices, np.roll(vertices, -1, axis=0)-vertices
                alpha = np.clip(np.sum((self.raw-a)*delta, axis=1)/np.maximum(np.sum(delta*delta, axis=1), 1.e-12), 0., 1.)
                hull_distance = 0. if inside else float(np.linalg.norm(a+alpha[:, None]*delta-self.raw, axis=1).min())
                if hull_distance > cfg.increment_maximum_hull_extrapolation_px:
                    continue
                X = np.column_stack((np.ones(len(q)), (previous-self.raw)/cfg.coordinate_scale_px))
                cells = np.floor(previous/6.).astype(int)
                _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
                weights = np.exp(-.5*(distance[q]/(.65*radius))**2)/counts[inverse]
                motion = points[q]-previous
                fit = _robust_fit(X, motion, weights, cfg)
                if fit is None or fit['rms'] > cfg.increment_maximum_rms_px:
                    continue
                losses = []
                tests = [distance[q] <= np.quantile(distance[q], .25),
                         np.floor((self.s[q]-self.s0)/8.).astype(int) % 3 == 0,
                         self.depth[q] <= np.quantile(self.depth[q], .25)]
                for test in tests:
                    if test.sum() < 2:
                        continue
                    held = _robust_fit(X[~test], motion[~test], weights[~test], cfg)
                    if held is not None:
                        losses.append(float(np.sqrt(np.average(
                            np.sum((X[test]@held['beta']-motion[test])**2, axis=1), weights=weights[test]))))
                if len(losses) < 2 or max(losses) > cfg.increment_maximum_holdout_px:
                    continue
                F = np.eye(2)+fit['beta'][1:].T/cfg.coordinate_scale_px
                jacobian = float(np.linalg.det(F))
                if jacobian <= cfg.minimum_jacobian or np.linalg.svd(F, compute_uv=False)[0] > cfg.maximum_stretch:
                    continue
                holdout = max(losses)
                error = max(cfg.position_noise_floor_px, fit['rms'], holdout)*np.sqrt(1.+max(0., fit['inverse'][0, 0]))
                # Extrapolation into a gap remains less certain even if nearby
                # observed motions happen to be nearly affine.
                error = float(np.hypot(error, hull_distance*np.linalg.norm(F-np.eye(2), ord=2)))
                candidates.append(dict(xy=self.raw+fit['beta'][0], sigma=float(np.hypot(self.sigma, error)),
                    indices=q, rms=fit['rms'], holdout=holdout, hull_extrapolation=hull_distance,
                    nearest_support=nearest, current_radius=radius, reference_half_window=half,
                    jacobian=jacobian, score=holdout+.1*hull_distance+.02*radius+.005*half))
        return min(candidates, key=lambda x:x['score']) if candidates else None

    def _illegal_side(self, point, tip, labels):
        if labels is None or not np.any(labels == 2) or self.side == 0 or not np.isfinite(tip).all():
            return False
        margin = self.config.blade_side_margin_px
        if point[1] >= tip[1]-margin:
            return False  # A path genuinely below the apex may turn around it.
        return bool(point[0] > tip[0]+margin if self.side == 1 else point[0] < tip[0]-margin)

    def _boundary(self, raw, tangent, path, arc, tip, labels):
        if path is None or arc is None:
            return None
        path, arc = np.asarray(path, float), np.asarray(arc, float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or arc.shape != (len(path),):
            return None
        a, delta = path[:-1], np.diff(path, axis=0)
        length2 = np.sum(delta*delta, axis=1)
        alpha = np.clip(np.sum((raw-a)*delta, axis=1)/np.maximum(length2, 1.e-12), 0., 1.)
        feet = a+alpha[:, None]*delta
        correction = feet-raw
        distance = np.linalg.norm(correction, axis=1)
        tangential = abs(correction@tangent)
        cfg = self.config
        good = np.isfinite(feet).all(1) & (length2 > 1.e-9)
        good &= (distance <= cfg.maximum_normal_correction_px) & (tangential <= cfg.maximum_tangential_correction_px)
        good &= _sample(labels, feet) != 2
        candidates = np.flatnonzero(good)
        stations = arc[:-1]+alpha*np.diff(arc)
        raw_hidden = bool(_sample(labels, raw[None])[0] == 2)
        for i in sorted(candidates, key=lambda j:distance[j]+3.*tangential[j]):
            if (not raw_hidden and _blocked(raw, feet[i], labels)) or _illegal_or_crossing(self, feet[i], tip, labels):
                continue
            # A folded contour may have two different material branches at
            # almost the same normal distance. Adjacent polygon segments are
            # not alternatives: require substantial separation in arc length.
            alternatives = candidates[
                (abs(stations[candidates]-stations[i]) >= cfg.boundary_ambiguity_arc_separation_px)
                & (abs(distance[candidates]-distance[i]) <= cfg.boundary_ambiguity_distance_px)]
            for j in alternatives:
                if (not raw_hidden and _blocked(raw, feet[j], labels)) or _illegal_or_crossing(self, feet[j], tip, labels):
                    continue
                return dict(ambiguous=True,
                    alternative_arc_separation_px=float(abs(stations[j]-stations[i])),
                    alternative_distance_difference_px=float(abs(distance[j]-distance[i])))
            return dict(xy=feet[i], correction=correction[i], distance=float(distance[i]),
                        tangential=float(tangential[i]), arc=float(stations[i]), ambiguous=False)
        return None

    def step(self, current_points, supported, tip_xy, labels, path_xy, path_arc, time_s, branch_labels=None):
        points, support = np.asarray(current_points, float), np.asarray(supported, bool)
        if points.shape != self.seed.shape or support.shape != (len(self.seed),):
            raise ValueError('Every frame must preserve the complete original ID ordering')
        time_s = float(time_s)
        if not np.isfinite(time_s) or self.time is not None and time_s <= self.time:
            raise ValueError('Observation times must be finite and strictly increasing')
        tip = np.asarray(tip_xy, float)
        if tip.shape != (2,):
            raise ValueError('tip_xy must have two coordinates')
        if branch_labels is not None:
            branches = np.asarray(branch_labels)
            if branches.shape != support.shape:
                raise ValueError('Branch labels must preserve original material IDs')
        else:
            branches = None
        support = support & np.isfinite(points).all(1) & (_sample(labels, points) == 1)
        if self.side and branches is not None:
            support &= np.isin(branches, (self.side, 3, 0))
        cfg = self.config
        dt = 0. if self.time is None else time_s-self.time
        # Once both independent position and qualified same-ID motion fail,
        # retain the last position; do not extend a stale velocity indefinitely.
        # The last resolved speed widens uncertainty instead of manufacturing
        # an unobserved material trajectory.
        prediction = self.raw.copy()
        elapsed_unobserved = dt if self.last_observation_time is None else time_s-self.last_observation_time
        prediction_sigma = float(self.sigma+np.linalg.norm(self.velocity)*dt
                                 +cfg.prediction_acceleration_px_s2*dt*max(dt, elapsed_unobserved))
        fit = self._absolute(points, support)
        increment = self._increment(points, support, dt) if self.time is not None else None
        source, candidate, state, sigma = None, prediction, 4, prediction_sigma
        if increment is not None:
            source, candidate, state, sigma = increment, increment['xy'], 3, increment['sigma']
        if fit is not None:
            source, candidate, state, sigma = fit, fit['xy'], fit['state'], fit['sigma']
        free_candidate = candidate.copy()
        increment_fused = False
        absolute_innovation = np.full(2, np.nan)
        if fit is not None and increment is not None:
            absolute_innovation = fit['xy']-increment['xy']
            # Successive fits can switch depth bands when a few IDs disappear.
            # Same-ID incremental motion supplies temporal continuity without
            # changing the original address. These estimates are correlated:
            # do not claim the independent-observation variance reduction.
            gain = increment['sigma']**2/(increment['sigma']**2+fit['sigma']**2)
            fused = increment['xy']+gain*absolute_innovation
            candidate = fused
            sigma = float(np.hypot(max(fit['sigma'], increment['sigma']),
                                   np.linalg.norm(absolute_innovation)*min(gain, 1.-gain)))
            increment_fused = True
        tangent = fit['tangent'] if fit is not None else self.tangent
        geometry = self._boundary(candidate, tangent, path_xy, path_arc, tip, labels)
        geometry_ambiguous = bool(geometry is not None and geometry.get('ambiguous', False))
        ambiguity_arc_separation = np.nan if not geometry_ambiguous else geometry['alternative_arc_separation_px']
        ambiguity_distance_difference = np.nan if not geometry_ambiguous else geometry['alternative_distance_difference_px']
        if geometry_ambiguous:
            geometry = None
        xy = geometry['xy'].copy() if geometry is not None else candidate.copy()
        # Assess the final, normally corrected candidate. A small free-map
        # error inside the knife must not prevent a supported same-side normal
        # correction back to the visible contour.
        hidden = bool(_sample(labels, xy[None])[0] == 2)
        previous_covered = bool(self.time is not None and _sample(labels, self.xy[None])[0] == 2)
        crossing = bool(self.time is not None and _motion_crosses_blade(self.xy, xy, labels))
        side_change = bool(not hidden and self._illegal_side(xy, tip, labels))
        rejected = crossing or side_change
        if rejected:
            candidate = prediction
            xy = candidate.copy()
            geometry = None
            state, source = 4, None
            sigma = max(prediction_sigma, self.sigma+np.linalg.norm(free_candidate-prediction)*.25)
            hidden = bool(_sample(labels, xy[None])[0] == 2)
        correction = np.zeros(2) if geometry is None else geometry['correction']
        if geometry is not None:
            sigma = float(np.hypot(sigma, .25*geometry['distance']))
            if state == 1 and geometry['distance'] > cfg.directly_supported_normal_correction_px:
                state = 2
        if hidden and state == 1:
            state = 2  # A free map hidden by a knife is extrapolation, not an image observation.
        accepted_absolute = fit is not None and not rejected and not hidden
        if accepted_absolute:
            self.last_observation_time = time_s
        if dt > 0 and state != 4:
            self.velocity = (candidate-self.raw)/dt
        elif state == 4:
            # The last resolved velocity sets an uncertainty growth allowance;
            # the displayed position stays fixed when motion evidence is lost.
            sigma = max(sigma, self.sigma+cfg.position_noise_floor_px*max(dt, .01))
        if self.side == 0 and labels is not None and np.any(labels == 2) and np.isfinite(tip).all():
            margin = cfg.blade_side_margin_px
            if state < 4 and xy[1] < tip[1]-margin and abs(xy[0]-tip[0]) > margin:
                self.side = 1 if xy[0] < tip[0] else 2
        age = np.inf if self.last_observation_time is None else time_s-self.last_observation_time
        q = np.empty(0, int) if source is None else source['indices']
        result = dict(raw_xy=candidate.copy(), xy=xy, state=int(state), sigma_px=float(sigma),
            age_since_observation_s=float(age), support_ids=self.ids[q].copy(),
            fixed_neighbourhood_ids=self.fixed_ids.copy(), initial_s_px=self.s0,
            initial_xy=self.initial_xy.copy(), absolute_raw_xy=np.full(2, np.nan) if fit is None else fit['xy'].copy(),
            free_candidate_xy=free_candidate, fit_rms_px=np.nan if fit is None else float(fit['rms']),
            holdout_px=np.nan if fit is None else float(fit['holdout']),
            minimum_initial_depth_px=np.nan if fit is None else float(fit['minimum_depth']),
            material_jacobian=np.nan if fit is None else float(fit['jacobian']),
            same_id_increment_xy=np.full(2, np.nan) if increment is None else increment['xy'].copy(),
            same_id_increment_support_ids=np.empty(0, np.int64) if increment is None else self.ids[increment['indices']].copy(),
            same_id_increment_fused=increment_fused, absolute_increment_innovation_xy=absolute_innovation,
            same_id_increment_holdout_px=np.nan if increment is None else increment.get('holdout', np.nan),
            same_id_increment_hull_extrapolation_px=np.nan if increment is None else increment.get('hull_extrapolation', np.nan),
            same_id_increment_nearest_support_px=np.nan if increment is None else increment.get('nearest_support', np.nan),
            same_id_increment_current_radius_px=np.nan if increment is None else increment.get('current_radius', np.nan),
            same_id_increment_reference_half_window_px=np.nan if increment is None else increment.get('reference_half_window', np.nan),
            same_id_increment_jacobian=np.nan if increment is None else increment.get('jacobian', np.nan),
            geometry_correction_xy=correction.copy(), geometry_correction_px=float(np.linalg.norm(correction)),
            geometry_tangential_correction_px=np.nan if geometry is None else geometry['tangential'],
            contour_arc_px=np.nan if geometry is None else geometry['arc'], on_boundary=geometry is not None,
            geometry_ambiguous=geometry_ambiguous,
            geometry_alternative_arc_separation_px=ambiguity_arc_separation,
            geometry_alternative_distance_difference_px=ambiguity_distance_difference,
            blade_crossing_rejected=bool(crossing), side_change_rejected=bool(side_change),
            hidden_by_blade=hidden, previous_position_covered_by_current_blade=previous_covered,
            raw_hidden_by_blade=bool(_sample(labels, candidate[None])[0] == 2),
            committed_blade_side=int(self.side), independent_position_observation=bool(accepted_absolute),
            image_supported=bool(accepted_absolute and state == 1),
            contact_address_used=False, blade_contact_equality_imposed=False,
            uncertainty_kind='model_sensitivity_not_calibrated_confidence_interval')
        self.raw, self.xy, self.sigma = candidate.copy(), xy.copy(), float(sigma)
        self.previous_points, self.previous_support = points.copy(), support.copy()
        self.time = time_s
        return result


def _illegal_or_crossing(tracker, point, tip, labels):
    return tracker._illegal_side(point, tip, labels) or (tracker.time is not None and _motion_crosses_blade(tracker.xy, point, labels))
