"""Conditional original-surface contact from forward material coordinates.

Each profile compares the SAME observed IDs and reference region while scanning
the original address s. The boundary condition chi(s, 0) = blade_tip is an
explicit hypothesis, not an observation. Its data and held-out losses are
reported beside an unconstrained model. No inverse material point is projected.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaterialCoordinateContactConfig:
    reference_half_windows_px: tuple[float, ...] = (42., 72.)
    maximum_depths_px: tuple[float, ...] = (60., 120., 240.)
    models: tuple[str, ...] = ('affine', 'quadratic')
    coordinate_scale_px: float = 24.
    address_sample_pitch_px: float = 1.5
    near_tip_selection_radius_px: float = 100.
    maximum_roi_centers: int = 3
    minimum_points: int = 16
    minimum_address_span_px: float = 18.
    minimum_depth_span_px: float = 12.
    quadratic_regularization: float = .1
    profile_tolerance_px: float = 2.
    ensemble_tolerance_px: float = 4.
    compatible_rms_px: float = 4.
    compatible_heldout_rms_px: float = 6.
    compatible_closure_penalty_px: float = 3.
    blade_boundary_uncertainty_px: float = 2.
    maximum_stretch: float = 12.
    minimum_jacobian: float = .01


def _basis(u, v, model):
    u, v = np.broadcast_arrays(np.asarray(u), np.asarray(v))
    terms = [np.ones_like(u), u, v]
    if model == 'quadratic':
        terms += [u * u, u * v, v * v]
    return np.stack(terms, axis=-1)


def _regularizer(model, config):
    return np.diag([0., 0., 0.] + ([config.quadratic_regularization] * 3 if model == 'quadratic' else []))


class MaterialCoordinateContactEstimator:
    def __init__(self, seed_points, point_ids, surface_reference, um_per_px, config=None):
        self.reference = np.asarray(seed_points, float).copy()
        self.ids = np.asarray(point_ids, np.int64).copy()
        self.boundary = np.asarray(surface_reference['boundary_xy'], float).copy()
        self.addresses = np.asarray(surface_reference['surface_s_px'], float).copy()
        self.tangent = np.asarray(surface_reference['tangent'], float).copy()
        self.normal = np.asarray(surface_reference['inward_normal'], float).copy()
        self.tangent /= np.linalg.norm(self.tangent)
        self.normal /= np.linalg.norm(self.normal)
        self.axes = np.column_stack((self.tangent, self.normal))
        self.contact_s = float(surface_reference['contact_s_px'])
        self.scale = float(um_per_px)
        self.config = config or MaterialCoordinateContactConfig()
        if (self.reference.shape != (len(self.ids), 2)
                or len(np.unique(self.ids)) != len(self.ids)
                or not np.isfinite(self.reference).all()
                or self.boundary.shape != (len(self.addresses), 2)
                or not np.isfinite(self.boundary).all()
                or np.any(np.diff(self.addresses) <= 0)
                or abs(np.linalg.det(self.axes)) < .95
                or not np.isfinite(self.scale) or self.scale <= 0):
            raise ValueError('Invalid original material IDs, surface, or calibration')
        along = self.boundary @ self.tangent
        if np.any(np.diff(along) <= 0):
            raise ValueError('Initial surface must be ordered along its tangent')
        u = self.reference @ self.tangent
        self.point_s = np.interp(u, along, self.addresses)
        self.depth = self.reference @ self.normal - np.interp(u, along, self.boundary @ self.normal)
        self.member = (u >= along[0]) & (u <= along[-1]) & (self.depth >= -.25)
        self.depth = np.maximum(0., self.depth)
        self.pitch = float(np.median(np.diff(self.addresses)))

    @staticmethod
    def _sample(labels, xy, outside=-1.):
        q = np.rint(np.nan_to_num(xy, nan=-1.e8, posinf=-1.e8, neginf=-1.e8)).astype(np.int64)
        inside = ((q[..., 0] >= 0) & (q[..., 0] < labels.shape[1])
                  & (q[..., 1] >= 0) & (q[..., 1] < labels.shape[0]))
        result = np.full(q.shape[:-1], outside, float)
        result[inside] = labels[q[..., 1][inside], q[..., 0][inside]]
        return result

    def surface_point(self, s):
        return np.array([np.interp(s, self.addresses, self.boundary[:, d]) for d in range(2)])

    @staticmethod
    def _groups(branches, valid):
        values = set(np.unique(branches[valid]).tolist()) - {0}
        groups = [(b, (b, 3) if b in (1, 2) and 3 in values else (b,)) for b in sorted(values - {3})]
        if 3 in values and not ({1, 2} & values):
            groups.append((3, (3,)))
        return groups

    def _solve_profile(self, X, y, w, C, tip, model):
        """Equality-constrained WLS by rank-one update of one fixed-data fit."""
        A = X.T @ (w[:, None] * X) + _regularizer(model, self.config)
        if np.linalg.matrix_rank(X * np.sqrt(w[:, None])) < X.shape[1]:
            return None
        try:
            beta = np.linalg.solve(A, X.T @ (w[:, None] * y))
            response = np.linalg.solve(A, C.T).T
        except np.linalg.LinAlgError:
            return None
        leverage = np.einsum('si,si->s', C, response)
        if np.any(leverage <= 1.e-12):
            return None
        gap = tip - C @ beta
        constrained = beta[None] + response[:, :, None] * (gap / leverage[:, None])[:, None, :]
        residual = np.einsum('nk,skd->snd', X, constrained) - y[None]
        rms = np.sqrt(np.sum(w[None] * np.sum(residual ** 2, axis=2), axis=1) / w.sum())
        residual0 = X @ beta - y
        rms0 = float(np.sqrt(np.average(np.sum(residual0 ** 2, axis=1), weights=w)))
        penalty = np.sqrt(np.maximum(0., rms ** 2 - rms0 ** 2))
        return dict(beta=beta, constrained=constrained, rms=rms, rms0=rms0,
                    penalty=penalty, gap=np.linalg.norm(gap, axis=1), leverage=leverage,
                    objective_increase=np.sum(gap ** 2, axis=1) / leverage)

    def _heldout(self, X, y, w, C, tip, model, depth):
        """Whole depth stripes plus the shallowest band are held out."""
        predictions, predictions0, weights = [], [], []
        stripe = np.floor(depth / 12.).astype(int)
        splits = [stripe % 3 == k for k in range(3)]
        shallow = depth <= np.quantile(depth, .25)
        splits.append(shallow)
        valid_splits = 0
        for test in splits:
            train = ~test
            if test.sum() < 3 or train.sum() < X.shape[1] + 3:
                continue
            fit = self._solve_profile(X[train], y[train], w[train], C, tip, model)
            if fit is None:
                continue
            difference = np.einsum('nk,skd->snd', X[test], fit['constrained']) - y[None, test]
            predictions.append(np.sum(difference ** 2, axis=2))
            predictions0.append(np.sum((X[test] @ fit['beta'] - y[test]) ** 2, axis=1))
            weights.append(w[test])
            valid_splits += 1
        if not valid_splits:
            return np.full(len(C), np.nan), np.nan, 0
        weight = np.concatenate(weights)
        return (np.sqrt(np.average(np.concatenate(predictions, axis=1), axis=1, weights=weight)),
                float(np.sqrt(np.average(np.concatenate(predictions0), weights=weight))), valid_splits)

    def _fiber_geometry(self, beta, s_grid, center, depth_max, model, tip, blade_distance):
        cfg = self.config
        # The fitted curved material fiber is checked, never the chord from
        # surface to an interior ID. Native-pixel depth sampling is bounded.
        normal_depth = np.linspace(0., depth_max, max(8, int(np.ceil(depth_max / 2.)) + 1))
        u = (s_grid - center)[:, None] / cfg.coordinate_scale_px
        v = normal_depth[None] / cfg.coordinate_scale_px
        basis = _basis(u, v, model)
        fiber = np.einsum('snk,skd->snd', basis, beta)
        ds = np.broadcast_to(beta[:, None, 1, :] / cfg.coordinate_scale_px, fiber.shape).copy()
        dn = np.broadcast_to(beta[:, None, 2, :] / cfg.coordinate_scale_px, fiber.shape).copy()
        if model == 'quadratic':
            ds += (2 * u[:, :, None] * beta[:, None, 3, :] + v[:, :, None] * beta[:, None, 4, :]) / cfg.coordinate_scale_px
            dn += (u[:, :, None] * beta[:, None, 4, :] + 2 * v[:, :, None] * beta[:, None, 5, :]) / cfg.coordinate_scale_px
        J = np.stack((ds, dn), axis=-1)
        determinant = np.linalg.det(J) * np.linalg.det(self.axes)
        stretch = np.linalg.svd(J, compute_uv=False)[..., 0]
        penetration = np.zeros(len(s_grid))
        offscreen_fraction = np.zeros(len(s_grid))
        if blade_distance is not None:
            sampled = self._sample(blade_distance, fiber, outside=-1.)
            away = np.linalg.norm(fiber - tip, axis=2) > cfg.blade_boundary_uncertainty_px
            penetration = np.max(np.where(away, np.maximum(sampled, 0.), 0.), axis=1)
            offscreen_fraction = np.mean(sampled < 0, axis=1)
        return dict(fiber=fiber, normal_depth=normal_depth,
                    min_jacobian=determinant.min(axis=1), max_stretch=stretch.max(axis=1),
                    penetration=penetration, offscreen_fraction=offscreen_fraction)

    def profile_frame(self, current, supported, tip_xy, *, branch_labels=None,
                      blade_mask=None, tip_uncertainty_px=2.):
        current = np.asarray(current, float)
        tip = np.asarray(tip_xy, float)
        supported = np.asarray(supported, bool)
        if current.shape != self.reference.shape or supported.shape != self.ids.shape or tip.shape != (2,):
            raise ValueError('Frame data must preserve original IDs')
        if not np.isfinite(tip_uncertainty_px) or tip_uncertainty_px < 0:
            raise ValueError('Tip uncertainty must be finite and nonnegative')
        if not np.isfinite(tip).all():
            return []
        cfg = self.config
        valid = supported & self.member & np.isfinite(current).all(axis=1)
        branches = np.ones(len(valid), int) if branch_labels is None else np.asarray(branch_labels, int)
        if branches.shape != self.ids.shape:
            raise ValueError('Branch labels must preserve original IDs')
        valid &= branches > 0
        labels = None if blade_mask is None else np.asarray(blade_mask)
        blade_distance = None
        if labels is not None:
            blade = labels if labels.dtype == bool else labels == 2
            center_label = self._sample(labels, current)
            valid &= (center_label == 0) if labels.dtype == bool else (center_label == 1)
            blade_distance = cv2.distanceTransform(blade.astype(np.uint8), cv2.DIST_L2, 5)
        distance = np.linalg.norm(current - tip, axis=1)
        profiles = []
        for branch, members in self._groups(branches, valid):
            branch_valid = valid & np.isin(branches, members)
            nearby = np.flatnonzero(branch_valid & (distance <= cfg.near_tip_selection_radius_px))
            if len(nearby) < cfg.minimum_points:
                continue
            # Current proximity chooses reference regions only; every currently
            # supported ID in that initial region is retained in the fit,
            # including shallow IDs far from the present tip.
            order = nearby[np.argsort(distance[nearby])]
            centers = []
            for idx in order:
                center = float(self.point_s[idx])
                if not centers or min(abs(np.asarray(centers) - center)) >= 24.:
                    centers.append(center)
                if len(centers) >= cfg.maximum_roi_centers:
                    break
            for center in centers:
                for half_window in cfg.reference_half_windows_px:
                    for depth_limit in cfg.maximum_depths_px:
                        idx = np.flatnonzero(branch_valid & (abs(self.point_s - center) <= half_window)
                                             & (self.depth <= depth_limit))
                        if (len(idx) < cfg.minimum_points
                                or np.ptp(self.point_s[idx]) < cfg.minimum_address_span_px
                                or np.ptp(self.depth[idx]) < cfg.minimum_depth_span_px):
                            continue
                        pitch = min(cfg.address_sample_pitch_px, self.pitch)
                        if pitch <= 0:
                            raise ValueError('Address sampling pitch must be positive')
                        lo, hi = self.point_s[idx].min(), self.point_s[idx].max()
                        first = int(np.ceil((lo - self.addresses[0]) / pitch))
                        last = int(np.floor((hi - self.addresses[0]) / pitch))
                        s_grid = self.addresses[0] + np.arange(first, last + 1) * pitch
                        if len(s_grid) < 3:
                            continue
                        depth = self.depth[idx]
                        uv = np.column_stack((self.point_s[idx] - center, depth)) / cfg.coordinate_scale_px
                        cells = np.floor(np.column_stack((self.point_s[idx], depth)) / 12.).astype(int)
                        _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
                        w = 1. / counts[inverse]
                        for model in cfg.models:
                            if model == 'quadratic' and len(np.unique(np.floor(depth / 6.))) < 4:
                                continue
                            X = _basis(uv[:, 0], uv[:, 1], model)
                            C = _basis((s_grid - center) / cfg.coordinate_scale_px, np.zeros(len(s_grid)), model)
                            fit = self._solve_profile(X, current[idx], w, C, tip, model)
                            if fit is None:
                                continue
                            heldout, heldout0, folds = self._heldout(X, current[idx], w, C, tip, model, depth)
                            geometry = self._fiber_geometry(fit['constrained'], s_grid, center,
                                                            float(depth.max()), model, tip, blade_distance)
                            # Loss components remain exported separately. Geometry
                            # penalties rank conditional hypotheses; they cannot
                            # turn a poor contact closure into observed contact.
                            geometry_cost = (2. * np.maximum(0., geometry['penetration'] - cfg.blade_boundary_uncertainty_px)
                                             + 4. * (geometry['min_jacobian'] <= 0)
                                             + np.maximum(0., geometry['max_stretch'] - cfg.maximum_stretch)
                                             + 20. * geometry['offscreen_fraction'])
                            cv_term = np.where(np.isfinite(heldout), heldout, fit['rms'] + 10.)
                            score = fit['rms'] + .5 * fit['penalty'] + .35 * cv_term + geometry_cost
                            profiles.append(dict(branch_label=int(branch), center_s_px=center,
                                half_window_px=float(half_window), depth_limit_px=float(depth_limit),
                                model=model, support_indices=idx, support_ids=self.ids[idx],
                                support_count=len(idx), effective_cells=int(len(counts)),
                                support_depth_min_px=float(depth.min()), support_depth_max_px=float(depth.max()),
                                s_grid=s_grid.copy(), score_px=score, closure_rms_px=fit['rms'],
                                unconstrained_rms_px=fit['rms0'], closure_penalty_px=fit['penalty'],
                                layer_holdout_rms_px=heldout, unconstrained_layer_holdout_rms_px=heldout0,
                                heldout_folds=folds, unconstrained_surface_gap_px=fit['gap'],
                                extrapolation_leverage=fit['leverage'],
                                constraint_objective_increase=fit['objective_increase'], beta=fit['constrained'],
                                unconstrained_beta=fit['beta'], fiber_xy=geometry['fiber'],
                                fiber_depth_px=geometry['normal_depth'], min_jacobian=geometry['min_jacobian'],
                                max_stretch=geometry['max_stretch'], fiber_blade_penetration_px=geometry['penetration'],
                                offscreen_fraction=geometry['offscreen_fraction']))
        return profiles

    def estimate_frame(self, current, supported, tip_xy, *, branch_labels=None,
                       blade_mask=None, tip_uncertainty_px=2., return_profiles=False):
        cfg = self.config
        profiles = self.profile_frame(current, supported, tip_xy, branch_labels=branch_labels,
                                      blade_mask=blade_mask, tip_uncertainty_px=tip_uncertainty_px)
        result = dict(state=0, reason='insufficient_supported_material_coordinates',
                      reference_s_px=np.nan, reference_marker_xy=np.full(2, np.nan),
                      low_s_px=np.nan, high_s_px=np.nan, contact_compatible=False,
                      support_ids=np.empty(0, np.int64), branch_label=0, profile_count=len(profiles),
                      profile_best_s_px=np.empty(0), profile_best_score_px=np.empty(0),
                      closure_rms_px=np.nan, unconstrained_rms_px=np.nan,
                      closure_penalty_px=np.nan, layer_holdout_rms_px=np.nan,
                      unconstrained_layer_holdout_rms_px=np.nan,
                      unconstrained_surface_gap_px=np.nan, fiber_blade_penetration_px=np.nan,
                      min_jacobian=np.nan, max_stretch=np.nan,
                      support_depth_min_px=np.nan, support_depth_max_px=np.nan,
                      normal_fiber_xy=np.empty((0, 2)), normal_fiber_depth_px=np.empty(0),
                      estimated_surface_xy=np.full_like(self.boundary, np.nan),
                      estimated_surface_s_px=self.addresses.copy(), selected_model='',
                      selected_center_s_px=np.nan, selected_half_window_px=np.nan,
                      selected_depth_limit_px=np.nan, profile_minimum_at_extent=False,
                      extrapolation_leverage=np.nan, constraint_objective_increase=np.nan,
                      tip_uncertainty_px=float(tip_uncertainty_px), tip_address_sensitivity_px=np.nan,
                      address_intervals=np.empty((0, 2)), data_support_label='unavailable')
        if not profiles:
            if return_profiles:
                result['profiles'] = []
            return result
        minima = np.array([np.min(p['score_px']) for p in profiles])
        best_indices = [int(np.argmin(p['score_px'])) for p in profiles]
        best_profile = int(np.argmin(minima))
        chosen, k = profiles[best_profile], best_indices[best_profile]
        # Conditional model sensitivity: retain each comparably supported
        # fixed-ROI profile, with its own address range. This is not a CI.
        intervals = []
        for j, profile in enumerate(profiles):
            if minima[j] > minima[best_profile] + cfg.ensemble_tolerance_px:
                continue
            keep = profile['score_px'] <= minima[j] + cfg.profile_tolerance_px
            selected = profile['s_grid'][keep]
            if len(selected):
                pk = best_indices[j]
                u = (profile['s_grid'][pk] - profile['center_s_px']) / cfg.coordinate_scale_px
                derivative = profile['beta'][pk, 1].copy()
                if profile['model'] == 'quadratic':
                    derivative += 2. * u * profile['beta'][pk, 3]
                stretch = float(np.linalg.norm(derivative / cfg.coordinate_scale_px))
                # Tip-location uncertainty changes the sensitivity envelope,
                # never the best-fit address or its objective. The local
                # forward tangential stretch converts current px to initial s.
                margin = tip_uncertainty_px / max(.05, stretch) + .5 * cfg.address_sample_pitch_px
                intervals.append((max(float(profile['s_grid'][0]), float(selected.min()) - margin),
                                  min(float(profile['s_grid'][-1]), float(selected.max()) + margin)))
        address = float(chosen['s_grid'][k])
        compatible = bool(chosen['closure_rms_px'][k] <= cfg.compatible_rms_px
                          and chosen['layer_holdout_rms_px'][k] <= cfg.compatible_heldout_rms_px
                          and chosen['closure_penalty_px'][k] <= cfg.compatible_closure_penalty_px
                          and chosen['fiber_blade_penetration_px'][k] <= cfg.blade_boundary_uncertainty_px
                          and chosen['min_jacobian'][k] >= cfg.minimum_jacobian
                          and chosen['max_stretch'][k] <= cfg.maximum_stretch
                          and k not in (0, len(chosen['s_grid']) - 1))
        beta = chosen['beta'][k]
        in_roi = ((self.addresses >= chosen['s_grid'][0]) & (self.addresses <= chosen['s_grid'][-1]))
        surface_X = _basis((self.addresses[in_roi] - chosen['center_s_px']) / cfg.coordinate_scale_px,
                           np.zeros(in_roi.sum()), chosen['model'])
        reconstructed = np.full_like(self.boundary, np.nan)
        reconstructed[in_roi] = surface_X @ beta
        derivative = beta[1].copy()
        if chosen['model'] == 'quadratic':
            derivative += 2. * (address - chosen['center_s_px']) / cfg.coordinate_scale_px * beta[3]
        tip_address_sensitivity = tip_uncertainty_px / max(.05, float(np.linalg.norm(derivative / cfg.coordinate_scale_px)))
        reason = 'conditional_original_surface_contact_model' if compatible else 'best_contact_closure_model_has_data_or_geometry_mismatch'
        result.update(state=1, reason=reason, reference_s_px=address,
                      reference_marker_xy=self.surface_point(address),
                      low_s_px=min(low for low, _ in intervals), high_s_px=max(high for _, high in intervals),
                      contact_compatible=compatible, support_ids=chosen['support_ids'],
                      branch_label=chosen['branch_label'],
                      profile_best_s_px=np.array([p['s_grid'][i] for p, i in zip(profiles, best_indices)]),
                      profile_best_score_px=minima, unconstrained_rms_px=chosen['unconstrained_rms_px'],
                      unconstrained_layer_holdout_rms_px=chosen['unconstrained_layer_holdout_rms_px'],
                      support_depth_min_px=chosen['support_depth_min_px'],
                      support_depth_max_px=chosen['support_depth_max_px'],
                      normal_fiber_xy=chosen['fiber_xy'][k], normal_fiber_depth_px=chosen['fiber_depth_px'],
                      estimated_surface_xy=reconstructed, selected_model=chosen['model'],
                      selected_center_s_px=chosen['center_s_px'],
                      selected_half_window_px=chosen['half_window_px'],
                      selected_depth_limit_px=chosen['depth_limit_px'],
                      profile_minimum_at_extent=k in (0, len(chosen['s_grid']) - 1),
                      extrapolation_leverage=float(chosen['extrapolation_leverage'][k]),
                      constraint_objective_increase=float(chosen['constraint_objective_increase'][k]),
                      tip_address_sensitivity_px=tip_address_sensitivity,
                      address_intervals=np.asarray(intervals),
                      data_support_label='conditional_model_compatible' if compatible else 'conditional_model_inconsistent')
        for key in ('closure_rms_px', 'closure_penalty_px', 'layer_holdout_rms_px',
                    'unconstrained_surface_gap_px', 'fiber_blade_penetration_px',
                    'min_jacobian', 'max_stretch'):
            result[key] = float(chosen[key][k])
        if return_profiles:
            result['profiles'] = profiles
        return result
