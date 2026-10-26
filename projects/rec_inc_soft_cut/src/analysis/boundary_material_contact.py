"""Conditional surface identity from current-boundary arc and reference depth.

The observed contour supplies geometry, not original material identity. The
model assumes its same-side path to the tip continues the original surface.
Depth and depth/shear terms are estimated explicitly before setting d=0.
This does not assign an interior inverse coordinate to the original surface.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class BoundaryMaterialContactConfig:
    maximum_depths_px: tuple[float, ...] = (36., 60., 100., 160.)
    current_radius_px: float = 230.
    path_half_length_px: float = 210.
    maximum_boundary_distance_px: float = 110.
    reference_half_window_px: float = 120.
    minimum_points: int = 18
    maximum_reference_extrapolation_px: float = 90.
    ambiguity_arc_separation_px: float = 24.
    ambiguity_distance_px: float = 3.
    compatible_rms_px: float = 4.
    compatible_holdout_px: float = 6.
    compatible_geometry_gap_px: float = 6.
    ensemble_loss_tolerance_px: float = 5.


def project_to_path(path, arc, points, *, ambiguity_arc_separation_px=24.):
    """Exact closest segment and nonlocal-alternative ambiguity diagnostics."""
    points = np.asarray(points, float)
    a = path[:-1]
    delta = np.diff(path, axis=0)
    length2 = np.sum(delta*delta, axis=1)
    records = []
    # Chunking bounds memory on complete saved point fields.
    for start in range(0, len(points), 128):
        p = points[start:start+128]
        alpha_raw = np.einsum('nij,ij->ni', p[:, None]-a[None], delta)/np.maximum(length2, 1.e-8)
        alpha = np.clip(alpha_raw, 0., 1.)
        projected = a[None]+alpha[:, :, None]*delta[None]
        distances = np.linalg.norm(p[:, None]-projected, axis=2)
        indices = distances.argmin(axis=1)
        rows = np.arange(len(p))
        r_all = arc[:-1][None]+alpha*np.diff(arc)[None]
        r = r_all[rows, indices]
        distance = distances[rows, indices]
        alternative = np.min(np.where(abs(r_all-r[:, None]) > ambiguity_arc_separation_px,
                                      distances, np.inf), axis=1)
        endpoint_clamp = ((indices == 0) & (alpha_raw[rows, indices] < 0.)) | (
            (indices == len(delta)-1) & (alpha_raw[rows, indices] > 1.))
        records.append((r, projected[rows, indices], distance, alternative, endpoint_clamp))
    return tuple(np.concatenate([r[k] for r in records], axis=0) for k in range(5))


def extract_boundary_path(labels, tip, tangent):
    """Open material outline through the blade neighborhood, excluding frame edges.

    Identity is NOT inferred here. The estimator separately reports whether
    shallow original-surface IDs anchor each side of this geometric path.
    """
    h, w = labels.shape
    contours, _ = cv2.findContours((labels == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        xy = contour[:, 0].astype(float)
        if len(xy) < 50:
            continue
        inside = (xy[:, 0] > 2) & (xy[:, 0] < w-3) & (xy[:, 1] > 2) & (xy[:, 1] < h-3)
        if (~inside).any():
            cut = int(np.flatnonzero(~inside)[0])
            xy, inside = np.roll(xy, -cut, axis=0), np.roll(inside, -cut)
        changes = np.diff(np.r_[False, inside, False].astype(int))
        for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            run = xy[a:b]
            if len(run) >= 50:
                candidates.append((float(np.linalg.norm(run-tip, axis=1).min()), -len(run), run))
    if not candidates:
        return None
    path = min(candidates, key=lambda item:item[:2])[2].copy()
    if np.dot(path[-1]-path[0], tangent) < 0:
        path = path[::-1]
    apex = int(np.linalg.norm(path-tip, axis=1).argmin())
    # Smooth each side separately: never round the apex across its two sides.
    path[:apex+1] = gaussian_filter1d(path[:apex+1], 1.2, axis=0, mode='nearest')
    path[apex:] = gaussian_filter1d(path[apex:], 1.2, axis=0, mode='nearest')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1.e-6]
    arc, path = arc[keep], path[keep]
    stations = np.arange(0., arc[-1], 1.5)
    path = np.column_stack([np.interp(stations, arc, path[:, d]) for d in range(2)])
    r, p, distance, _, _ = project_to_path(path, stations, np.asarray(tip)[None])
    return dict(xy=path, arc=stations-r[0], tip_projection=p[0], tip_gap=float(distance[0]))


def _basis(s, d, origin, quadratic):
    u, v = (s-origin)/60., d/60.
    columns = [np.ones(len(u)), u, v, u*v]
    if quadratic:
        columns.append(v*v)
    return np.column_stack(columns)


def _fit(X, y, base):
    weights = base.copy()
    for _ in range(4):
        root = np.sqrt(weights)
        beta = np.linalg.lstsq(X*root[:, None], y*root, rcond=None)[0]
        residual = y-X@beta
        sigma = max(1., 1.4826*np.median(abs(residual-np.median(residual))))
        weights = base*np.minimum(1., 2.5*sigma/np.maximum(abs(residual), 1.e-8))
    beta = np.linalg.lstsq(X*np.sqrt(weights)[:, None], y*np.sqrt(weights), rcond=None)[0]
    rms = float(np.sqrt(np.average((y-X@beta)**2, weights=base)))
    return beta, rms, X.T@(weights[:, None]*X), weights.sum()


class BoundaryMaterialContactEstimator:
    def __init__(self, seed_points, point_ids, surface_reference, um_per_px, config=None):
        self.seed = np.asarray(seed_points, float)
        self.ids = np.asarray(point_ids, np.int64)
        self.surface = np.asarray(surface_reference['boundary_xy'], float)
        self.addresses = np.asarray(surface_reference['surface_s_px'], float)
        self.tangent = np.asarray(surface_reference['tangent'], float)
        self.normal = np.asarray(surface_reference['inward_normal'], float)
        self.tangent = self.tangent/np.linalg.norm(self.tangent)
        self.normal = self.normal/np.linalg.norm(self.normal)
        self.scale = float(um_per_px)
        self.config = config or BoundaryMaterialContactConfig()
        along = self.seed@self.tangent
        self.s = np.interp(along, self.surface@self.tangent, self.addresses)
        self.d = self.seed@self.normal-np.interp(along, self.surface@self.tangent, self.surface@self.normal)
        self.member = (along >= (self.surface@self.tangent)[0]) & (along <= (self.surface@self.tangent)[-1]) & (self.d >= -.5)

    def estimate_frame(self, current, supported, tip_xy, *, labels=None, path=None,
                       tip_uncertainty_px=2., return_profiles=False):
        cfg = self.config
        current, tip = np.asarray(current, float), np.asarray(tip_xy, float)
        if path is None:
            path = extract_boundary_path(labels, tip, self.tangent)
        empty = dict(state=0, reference_s_px=np.nan, low_s_px=np.nan, high_s_px=np.nan,
            reference_marker_xy=np.full(2, np.nan), contact_compatible=False,
            reason='no_conditioned_boundary_coordinate_model', support_ids=np.empty(0, np.int64),
            path_xy=np.empty((0, 2)), path_arc=np.empty(0), boundary_tip_gap_px=np.nan,
            profiles_count=0, estimated_surface_stretch=np.nan, rms_px=np.nan,
            heldout_px=np.nan, reference_extrapolation_px=np.nan, selected_side=0,
            anchor_ids=np.empty(0, np.int64), address_intervals=np.empty((0, 2)))
        if path is None:
            return empty
        path_xy, arc = path['xy'], path['arc']
        empty.update(path_xy=path_xy, path_arc=arc, boundary_tip_gap_px=path['tip_gap'])
        # A remote arc can fold back close in image space. Keep the complete
        # path for nearest-point and ambiguity queries; restrict the selected
        # coordinate below, after those alternatives have been considered.
        if len(path_xy) < 10:
            return empty
        safe_current = np.nan_to_num(current, nan=-1.e6)
        r, feet, distance, alternate, clamp = project_to_path(
            path_xy, arc, safe_current,
            ambiguity_arc_separation_px=cfg.ambiguity_arc_separation_px)
        usable = np.asarray(supported, bool) & self.member & np.isfinite(current).all(1)
        usable &= ~clamp & (alternate > distance+cfg.ambiguity_distance_px)
        usable &= (distance < cfg.maximum_boundary_distance_px) & (abs(r) > 6.)
        usable &= (abs(r) < cfg.path_half_length_px) & (np.linalg.norm(current-tip, axis=1) < cfg.current_radius_px)
        if labels is not None:
            q = np.rint(safe_current).astype(int)
            inside = (q[:, 0] >= 0) & (q[:, 0] < labels.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < labels.shape[0])
            label = np.zeros(len(q), np.uint8)
            label[inside] = labels[q[inside, 1], q[inside, 0]]
            usable &= label == 1
        profiles = []
        for side in (-1, 1):
            side_ok = usable & (r*side > 0)
            for limit in cfg.maximum_depths_px:
                chosen = side_ok & (self.d <= limit)
                q = np.flatnonzero(chosen)
                if len(q) < cfg.minimum_points or np.ptp(self.d[q]) < 10. or np.ptp(self.s[q]) < 24.:
                    continue
                center = float(np.average(self.s[q], weights=np.exp(-abs(r[q])/60.)))
                q = np.flatnonzero(chosen & (abs(self.s-center) < cfg.reference_half_window_px))
                if len(q) < cfg.minimum_points:
                    continue
                s, d, values = self.s[q], self.d[q], r[q]
                origin = float(np.mean(s))
                weights = np.exp(-.5*(values/120.)**2-.5*(d/limit)**2)
                cells = np.floor(np.column_stack((s, d))/8.).astype(int)
                _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
                weights /= counts[inverse]
                anchors = q[(d <= 4.5) & (distance[q] <= 6.) & (abs(values) >= 15.)]
                for quadratic in (False, True):
                    X = _basis(s, d, origin, quadratic)
                    if np.linalg.matrix_rank(X) < X.shape[1] or np.linalg.cond(X) > 250:
                        continue
                    beta, rms, H, weight_sum = _fit(X, values, weights)
                    stretch = float(beta[1]/60.)
                    if not .1 < stretch < 10.:
                        continue
                    # The scalar boundary coordinate must preserve along-layer
                    # order throughout the fitted depths, not only at d=0.
                    # Its derivative is affine in depth, so endpoints suffice.
                    depth_stretch = (beta[1]+beta[3]*np.array([d.min(), d.max()])/60.)/60.
                    if np.any(depth_stretch <= .1) or np.any(depth_stretch >= 10.):
                        continue
                    address = float(origin-60.*beta[0]/beta[1])
                    if not self.addresses[0] <= address <= self.addresses[-1]:
                        continue
                    extra = float(max(s.min()-address, address-s.max(), 0.))
                    if extra > cfg.maximum_reference_extrapolation_px:
                        continue
                    validations = []
                    # Test depth extrapolation AND the closest-to-contact s band.
                    tests = [d <= np.quantile(d, .3), values*side <= np.quantile(values*side, .25)]
                    for test in tests:
                        train = ~test
                        if test.sum() < 4 or train.sum() < X.shape[1]+3 or np.linalg.matrix_rank(X[train]) < X.shape[1]:
                            continue
                        b, _, _, _ = _fit(X[train], values[train], weights[train])
                        validations.append(float(np.sqrt(np.average((values[test]-X[test]@b)**2, weights=weights[test]))))
                    heldout = max(validations) if len(validations) == 2 else np.inf
                    z = _basis(np.array([address]), np.zeros(1), origin, quadratic)[0]
                    leverage = float(z@np.linalg.pinv(H)@z)
                    # No cap turns infinite validation error into finite precision.
                    location_error = float(tip_uncertainty_px)+float(path['tip_gap'])
                    radius = (location_error+heldout+rms*np.sqrt(1.+leverage*weight_sum/len(q)))/stretch
                    score = rms+.5*heldout+.08*extra
                    compatible = bool(rms <= cfg.compatible_rms_px and heldout <= cfg.compatible_holdout_px
                        and path['tip_gap'] <= cfg.compatible_geometry_gap_px and extra <= 6. and len(anchors) >= 2)
                    profiles.append(dict(reference_s_px=address, side=side, model='quadratic_depth' if quadratic else 'linear_depth',
                        support_ids=self.ids[q], support_indices=q, anchor_ids=self.ids[anchors],
                        low_s_px=address-radius, high_s_px=address+radius, score=score,
                        rms_px=rms, heldout_px=heldout, validation_parts=validations,
                        reference_extrapolation_px=extra, estimated_surface_stretch=stretch,
                        root_leverage=leverage, support_depth_min_px=float(d.min()),
                        support_depth_max_px=float(d.max()), depth_limit_px=limit,
                        contact_compatible=compatible, beta=beta, origin_s_px=origin))
        if not profiles:
            return empty
        best = min(profiles, key=lambda p:p['score'])
        threshold = best['score']+cfg.ensemble_loss_tolerance_px
        comparable = [p for p in profiles if p['score'] <= threshold]
        intervals = np.array([[p['low_s_px'], p['high_s_px']] for p in comparable])
        address = best['reference_s_px']
        result = dict(empty)
        result.update({key:best[key] for key in ('reference_s_px', 'support_ids', 'anchor_ids',
            'contact_compatible', 'rms_px', 'heldout_px', 'reference_extrapolation_px', 'estimated_surface_stretch')})
        result.update(state=1, reason='conditional_original_surface_boundary_continuity',
            low_s_px=float(intervals[:, 0].min()), high_s_px=float(intervals[:, 1].max()),
            reference_marker_xy=np.array([np.interp(address, self.addresses, self.surface[:, d]) for d in range(2)]),
            selected_side=best['side'], profiles_count=len(profiles), address_intervals=intervals,
            selected_model=best['model'], selected_beta=best['beta'], selected_origin_s_px=best['origin_s_px'],
            support_depth_min_px=best['support_depth_min_px'], support_depth_max_px=best['support_depth_max_px'])
        if return_profiles:
            result['profiles'] = profiles
        return result
