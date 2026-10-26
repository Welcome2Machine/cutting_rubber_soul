"""Original-surface address transport from fixed material identities.

For a contact on the original surface, chi(s_c(t), 0, t) = b(t), hence
    ds_c/dt = (v_blade - v_surface) dot tau / lambda.
The equation also requires compatible normal velocity and current position.
Neither compression nor missing optical observations implies zero transport.
This module supplies observations and diagnostics, not a mechanical stick law
or a temporally smoothed trajectory. All velocities use acquisition time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .material_coordinate_contact import MaterialCoordinateContactEstimator


@dataclass(frozen=True)
class ContactTransportConfig:
    reference_half_windows_px: tuple[float, ...] = (60., 90.)
    maximum_depths_px: tuple[float, ...] = (60., 120.)
    models: tuple[str, ...] = ('affine', 'bilinear')
    coordinate_scale_px: float = 40.
    donor_radius_px: float = 100.
    shallow_anchor_depth_px: float = 12.
    minimum_points: int = 12
    minimum_address_span_px: float = 18.
    minimum_depth_span_px: float = 12.
    balance_cell_px: float = 6.
    position_noise_floor_px: float = .25
    velocity_noise_floor_px_s: float = .25
    maximum_position_rms_px: float = 4.
    maximum_position_holdout_px: float = 6.
    maximum_leverage: float = 4.
    minimum_surface_stretch: float = .05
    maximum_stretch: float = 12.
    maximum_condition: float = 1.e7
    compatibility_sigma: float = 3.
    minimum_position_tolerance_px: float = 2.
    maximum_contact_gap_px: float = 4.
    root_interior_margin_px: float = 2.
    high_compression_ratio: float = .25
    resolved_velocity_px_s: float = .5


def _basis(s, d, center, scale, model):
    u, v = np.broadcast_arrays((np.asarray(s) - center) / scale,
                               np.asarray(d) / scale)
    values = [np.ones_like(u), u, v]
    if model == 'bilinear':
        values.append(u * v)
    elif model != 'affine':
        raise ValueError('Model must be affine or bilinear')
    return np.stack(values, axis=-1)


def _fit(X, values, initial_weights, floor, max_condition):
    """Robust WLS with a noise floor and current-space occupancy balancing."""
    w0 = np.asarray(initial_weights, float)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    w = w0.copy()
    for _ in range(3):
        w /= max(float(w.mean()), 1.e-12)
        matrix = X.T @ (w[:, None] * X)
        if np.linalg.cond(matrix) > max_condition:
            return None
        inverse = np.linalg.inv(matrix)
        beta = inverse @ (X.T @ (w[:, None] * values))
        residual = np.linalg.norm(X @ beta - values, axis=1)
        scale = max(floor, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        w = w0 * np.minimum(1., 2.5 * scale / np.maximum(residual, 1.e-12))
    # Statistics refer to the weights that produced the returned coefficients.
    w /= max(float(w.mean()), 1.e-12)
    matrix = X.T @ (w[:, None] * X)
    inverse = np.linalg.inv(matrix)
    beta = inverse @ (X.T @ (w[:, None] * values))
    residual = X @ beta - values
    rms = float(np.sqrt(np.average(np.sum(residual ** 2, axis=1), weights=w)))
    return dict(beta=beta, inverse=inverse, rms=rms, weights=w)


def _holdout(X, values, weights, depth, floor, cfg):
    """Leave the whole shallowest depth band out; random holdout is too easy."""
    test = depth <= np.quantile(depth, .25)
    train = ~test
    if test.sum() < 3 or train.sum() < X.shape[1] + 3:
        return np.inf
    fit = _fit(X[train], values[train], weights[train], floor, cfg.maximum_condition)
    if fit is None:
        return np.inf
    residual = np.linalg.norm(X[test] @ fit['beta'] - values[test], axis=1)
    return float(np.sqrt(np.average(residual ** 2, weights=weights[test])))


def _robust_center(values, sigma):
    values, sigma = np.asarray(values, float), np.asarray(sigma, float)
    finite = np.isfinite(values) & np.isfinite(sigma) & (sigma >= 0.)
    if not finite.any():
        available = values[np.isfinite(values)]
        return (float(np.median(available)), np.inf, float(np.std(available))) if len(available) else (np.nan, np.inf, np.nan)
    values, sigma = values[finite], sigma[finite]
    weights = 1. / np.maximum(sigma, 1.e-9) ** 2
    # Correlated alternative models are NOT independent replicate experiments.
    weights = np.minimum(weights, np.median(weights) * 4.)
    center = float(np.average(values, weights=weights))
    spread = float(np.sqrt(np.average((values - center) ** 2, weights=weights)))
    uncertainty = float(np.sqrt(np.average(sigma ** 2, weights=weights) + spread ** 2))
    return center, uncertainty, spread


def _empty_side(side):
    return dict(side=side, flux_px_s=np.nan, flux_sigma_px_s=np.nan,
                flux_valid=False, flux_model_spread_px_s=np.nan,
                absolute_s_px=np.nan, absolute_sigma_px=np.nan,
                absolute_valid=False, absolute_shallow_supported=False,
                surface_xy=np.full(2, np.nan), surface_velocity_xy=np.full(2, np.nan),
                tangent_xy=np.full(2, np.nan), surface_stretch=np.nan,
                normal_compression=np.nan, normal_shear=np.nan,
                normal_relative_velocity_px_s=np.nan, normal_sigma_px_s=np.nan,
                surface_tip_gap_px=np.nan, position_rms_px=np.nan,
                position_holdout_px=np.nan, velocity_holdout_px_s=np.nan,
                leverage=np.nan, minimum_initial_depth_px=np.nan,
                high_compression=False, normal_compatible=False,
                near_stick_observed=False, compressed_near_stick_observed=False,
                geometry_compatible=False, model_count=0, valid_model_count=0,
                flux_eval_s_px=np.nan, support_ids=np.empty(0, np.int64),
                reasons=['no_supported_fixed_id_model'])


class ContactTransportObserver:
    """Separate left/right, fixed-ID observations of reference-address rate.

    A finite invalid flux is diagnostic only. In particular a static material
    with a normally penetrating knife must not be integrated as observed slip.
    The provisional contact address controls one common reference ROI. Models
    differ in ROI size/depth, not in which address best lowers a forced gap.
    """

    def __init__(self, seed_points, point_ids, surface_reference, um_per_px, config=None):
        mapping = MaterialCoordinateContactEstimator(
            seed_points, point_ids, surface_reference, um_per_px)
        self.reference = mapping.reference
        self.ids = mapping.ids
        self.point_s = mapping.point_s
        self.depth = mapping.depth
        self.member = mapping.member
        self.addresses = mapping.addresses
        self.contact_s = mapping.contact_s
        self.axes = mapping.axes
        self.um_per_px = float(um_per_px)
        self.config = config or ContactTransportConfig()

    def _profile(self, indices, current, velocity, velocity_sigma, tip, blade_velocity,
                 blade_sigma, center, model, tip_sigma, side, window, maximum_depth):
        cfg = self.config
        s, d = self.point_s[indices], self.depth[indices]
        X = _basis(s, d, center, cfg.coordinate_scale_px, model)
        xy, vel = current[indices], velocity[indices]
        cells = np.floor(xy / cfg.balance_cell_px).astype(np.int64)
        _, inv, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        weights = 1. / counts[inv]
        position = _fit(X, xy, weights, cfg.position_noise_floor_px, cfg.maximum_condition)
        motion = _fit(X, vel, weights, cfg.velocity_noise_floor_px_s, cfg.maximum_condition)
        if position is None or motion is None:
            return None
        p_hold = _holdout(X, xy, weights, d, cfg.position_noise_floor_px, cfg)
        v_hold = _holdout(X, vel, weights, d, cfg.velocity_noise_floor_px_s, cfg)
        C = _basis(center, 0., center, cfg.coordinate_scale_px, model)
        surface = C @ position['beta']
        surface_velocity = C @ motion['beta']
        ds = position['beta'][1] / cfg.coordinate_scale_px
        dn = position['beta'][2] / cfg.coordinate_scale_px
        stretch = float(np.linalg.norm(ds))
        if stretch < 1.e-9:
            return None
        tau = ds / stretch
        inward = np.array([-tau[1], tau[0]]) * np.sign(np.linalg.det(self.axes))
        normal_compression = float(inward @ dn)
        normal_shear = float(tau @ dn)
        jacobian = float(np.linalg.det(np.column_stack((ds, dn))) * np.linalg.det(self.axes))
        # Check orientation across the entire donor region, not just contact.
        donor_ds = np.broadcast_to(ds, xy.shape).copy()
        donor_dn = np.broadcast_to(dn, xy.shape).copy()
        if model == 'bilinear':
            cross = position['beta'][3] / cfg.coordinate_scale_px ** 2
            donor_ds += d[:, None] * cross
            donor_dn += (s - center)[:, None] * cross
        donor_J = np.stack((donor_ds, donor_dn), axis=-1)
        min_jacobian = float(np.min(np.linalg.det(donor_J) * np.linalg.det(self.axes)))
        max_stretch = float(np.linalg.svd(donor_J, compute_uv=False)[..., 0].max())
        leverage = float(C @ position['inverse'] @ C)
        velocity_leverage = float(C @ motion['inverse'] @ C)
        point_sigma = float(np.sqrt(np.average(velocity_sigma[indices] ** 2, weights=weights)))
        geometry_sigma = max(cfg.position_noise_floor_px, position['rms'], p_hold)
        geometry_sigma *= np.sqrt(1. + max(leverage, 0.))
        velocity_error = max(cfg.velocity_noise_floor_px_s, motion['rms'], v_hold, point_sigma)
        velocity_error *= np.sqrt(1. + max(velocity_leverage, 0.))
        relative = blade_velocity - surface_velocity
        normal_relative = float(relative @ inward)
        tangent_relative = float(relative @ tau)
        flux = tangent_relative / stretch
        normal_sigma = float(np.hypot(blade_sigma, velocity_error))
        # Derivative uncertainty and normal mismatch are model uncertainty,
        # not new displacement evidence. No increase in count reduces it away.
        ds_sigma = geometry_sigma / max(np.ptp(s), 1.)
        flux_sigma = np.sqrt(normal_sigma ** 2 + normal_relative ** 2
                             + (flux * ds_sigma) ** 2) / stretch
        gap = float(np.linalg.norm(surface - tip))
        gap_sigma = float(np.hypot(tip_sigma, geometry_sigma))
        gap_tolerance = max(cfg.minimum_position_tolerance_px,
                            cfg.compatibility_sigma * gap_sigma)
        # Bad geometric fits must never excuse a remote surface: cap the
        # tolerated closure by tip registration precision, not residuals.
        gap_tolerance = min(gap_tolerance, max(cfg.maximum_contact_gap_px, 2. * tip_sigma))
        normal_compatible = abs(normal_relative) <= cfg.compatibility_sigma * normal_sigma
        geometry_compatible = gap <= gap_tolerance
        reasons = []
        if not normal_compatible:
            reasons.append('normal_velocity_incompatible')
        if not geometry_compatible:
            reasons.append('surface_tip_position_incompatible')
        if position['rms'] > cfg.maximum_position_rms_px or p_hold > cfg.maximum_position_holdout_px:
            reasons.append('spatial_holdout_or_fit_failed')
        if not np.isfinite(v_hold):
            reasons.append('velocity_holdout_unavailable')
        if leverage > cfg.maximum_leverage:
            reasons.append('deep_extrapolation_high_leverage')
        if min_jacobian <= 0. or jacobian <= 0. or max_stretch > cfg.maximum_stretch:
            reasons.append('invalid_material_orientation_or_stretch')
        if stretch < cfg.minimum_surface_stretch:
            reasons.append('surface_address_not_resolved_under_compression')
        flux_valid = not reasons
        # Unconstrained forward-surface intersection: current d=0 curve is
        # linear in s for affine/bilinear models. No tip equality is imposed.
        root = center + float((tip - surface) @ ds) / stretch ** 2
        root_xy = surface + (root - center) * ds
        root_gap = float(np.linalg.norm(root_xy - tip))
        inside = (s.min() + cfg.root_interior_margin_px < root
                  < s.max() - cfg.root_interior_margin_px
                  and self.addresses[0] < root < self.addresses[-1])
        root_C = _basis(root, 0., center, cfg.coordinate_scale_px, model)
        root_leverage = float(root_C @ position['inverse'] @ root_C)
        root_velocity = root_C @ motion['beta']
        root_normal_relative = float((blade_velocity - root_velocity) @ inward)
        root_motion_leverage = float(root_C @ motion['inverse'] @ root_C)
        root_velocity_sigma = max(cfg.velocity_noise_floor_px_s, motion['rms'], v_hold, point_sigma)
        root_velocity_sigma *= np.sqrt(1. + max(root_motion_leverage, 0.))
        root_normal_sigma = float(np.hypot(blade_sigma, root_velocity_sigma))
        root_normal_compatible = abs(root_normal_relative) <= cfg.compatibility_sigma * root_normal_sigma
        absolute_valid = bool(inside and root_gap <= gap_tolerance
                              and position['rms'] <= cfg.maximum_position_rms_px
                              and p_hold <= cfg.maximum_position_holdout_px
                              and root_leverage <= cfg.maximum_leverage
                              and np.isfinite(v_hold) and root_normal_compatible
                              and min_jacobian > 0. and jacobian > 0.
                              and cfg.minimum_surface_stretch <= stretch <= cfg.maximum_stretch)
        absolute_sigma = np.hypot(tip_sigma, geometry_sigma * np.sqrt(1. + max(root_leverage, 0.))) / stretch
        shallow = d <= cfg.shallow_anchor_depth_px
        compression_sigma = geometry_sigma / max(np.ptp(d), 1.)
        high_compression = bool(0 < normal_compression
                                and normal_compression + 2. * compression_sigma < cfg.high_compression_ratio
                                and p_hold <= cfg.maximum_position_holdout_px and min_jacobian > 0.)
        near_stick = bool(flux_valid and abs(tangent_relative) <= cfg.resolved_velocity_px_s
                          and normal_sigma <= cfg.resolved_velocity_px_s)
        return dict(side=side, model=model, reference_half_window_px=window,
                    maximum_depth_px=maximum_depth, flux_px_s=float(flux),
                    flux_sigma_px_s=float(flux_sigma), flux_valid=bool(flux_valid),
                    absolute_s_px=float(root), absolute_sigma_px=float(absolute_sigma),
                    absolute_valid=absolute_valid,
                    absolute_shallow_supported=bool(shallow.sum() >= 3),
                    absolute_root_gap_px=root_gap, absolute_root_interior=bool(inside),
                    absolute_normal_relative_velocity_px_s=root_normal_relative,
                    absolute_normal_sigma_px_s=root_normal_sigma,
                    absolute_normal_compatible=bool(root_normal_compatible),
                    surface_xy=surface, surface_velocity_xy=surface_velocity,
                    tangent_xy=tau, surface_stretch=stretch,
                    normal_compression=normal_compression, normal_shear=normal_shear,
                    normal_compression_sigma=compression_sigma,
                    jacobian=jacobian, min_jacobian=min_jacobian,
                    normal_relative_velocity_px_s=normal_relative,
                    tangential_relative_velocity_px_s=tangent_relative,
                    normal_sigma_px_s=normal_sigma, normal_compatible=bool(normal_compatible),
                    geometry_compatible=bool(geometry_compatible),
                    surface_tip_gap_px=gap, position_rms_px=position['rms'],
                    position_holdout_px=p_hold, velocity_holdout_px_s=v_hold,
                    leverage=leverage, root_leverage=root_leverage,
                    minimum_initial_depth_px=float(d.min()),
                    high_compression=high_compression, near_stick_observed=near_stick,
                    compressed_near_stick_observed=bool(high_compression and near_stick),
                    flux_eval_s_px=center, support_ids=self.ids[indices].copy(),
                    support_indices=indices.copy(), coordinate_scale_px=cfg.coordinate_scale_px,
                    reference_center_s_px=center,
                    reasons=reasons, coefficients_xy=position['beta'],
                    coefficients_velocity_xy=motion['beta'])

    @staticmethod
    def _combine_side(side, profiles):
        if not profiles:
            return _empty_side(side)
        valid = [p for p in profiles if p['flux_valid']]
        chosen = valid or profiles
        # Select representative diagnostics; averaging incompatible geometry
        # would create a surface that no model or fixed ID actually observed.
        best = min(chosen, key=lambda p: p['position_holdout_px'] + p['position_rms_px']
                   + p['surface_tip_gap_px'] + 2. * p['leverage'])
        out = dict(best)
        out['flux_px_s'], out['flux_sigma_px_s'], out['flux_model_spread_px_s'] = _robust_center(
            [p['flux_px_s'] for p in chosen], [p['flux_sigma_px_s'] for p in chosen])
        out['flux_valid'] = bool(valid)
        out['model_count'], out['valid_model_count'] = len(profiles), len(valid)
        absolute = [p for p in profiles if p['absolute_valid']]
        if absolute:
            out['absolute_s_px'], out['absolute_sigma_px'], out['absolute_model_spread_px'] = _robust_center(
                [p['absolute_s_px'] for p in absolute], [p['absolute_sigma_px'] for p in absolute])
            out['absolute_valid'] = True
            out['absolute_shallow_supported'] = any(p['absolute_shallow_supported'] for p in absolute)
        else:
            out['absolute_valid'] = False
            out['absolute_model_spread_px'] = np.nan
        return out

    def estimate_frame(self, current, supported, tip_xy, blade_velocity_xy,
                       material_velocity_xy, velocity_valid, *, velocity_sigma_px_s=None,
                       blade_velocity_sigma_px_s=0., branch_labels=None,
                       contact_s_px=None, tip_uncertainty_px=2., return_models=False):
        cfg = self.config
        current, velocity = np.asarray(current, float), np.asarray(material_velocity_xy, float)
        tip, blade_velocity = np.asarray(tip_xy, float), np.asarray(blade_velocity_xy, float)
        support, vvalid = np.asarray(supported, bool), np.asarray(velocity_valid, bool)
        n = len(self.ids)
        if (current.shape != (n, 2) or velocity.shape != (n, 2) or support.shape != (n,)
                or vvalid.shape != (n,) or tip.shape != (2,) or blade_velocity.shape != (2,)):
            raise ValueError('Positions and velocities must preserve all original material IDs')
        center = self.contact_s if contact_s_px is None else float(contact_s_px)
        if not np.isfinite(center) or not np.isfinite(tip_uncertainty_px) or tip_uncertainty_px < 0:
            raise ValueError('Contact address and tip uncertainty must be finite')
        blade_sigma = float(np.linalg.norm(np.asarray(blade_velocity_sigma_px_s, float)))
        if not np.isfinite(blade_sigma):
            raise ValueError('Blade velocity uncertainty must be finite')
        if velocity_sigma_px_s is None:
            velocity_sigma = np.full(n, cfg.velocity_noise_floor_px_s)
        else:
            velocity_sigma = np.asarray(velocity_sigma_px_s, float)
            if velocity_sigma.shape == (n, 2):
                velocity_sigma = np.linalg.norm(velocity_sigma, axis=1)
            if velocity_sigma.shape != (n,):
                raise ValueError('Velocity uncertainty must preserve original IDs')
        branches = np.where(current[:, 0] < tip[0], 1, 2) if branch_labels is None else np.asarray(branch_labels)
        if branches.shape != (n,):
            raise ValueError('Branch labels must preserve original IDs')
        good = (support & vvalid & self.member & np.isfinite(current).all(axis=1)
                & np.isfinite(velocity).all(axis=1) & np.isfinite(velocity_sigma)
                & (velocity_sigma >= 0))
        finite_frame = np.isfinite(tip).all() and np.isfinite(blade_velocity).all()
        distance = np.linalg.norm(current - tip, axis=1)
        profile_sets = []
        for side in (1, 2):
            profiles = []
            # The below-tip region has no crack identity; split it by current
            # blade side instead of using each below-tip point twice.
            below = (branches == 3) & ((current[:, 0] < tip[0]) if side == 1 else (current[:, 0] >= tip[0]))
            eligible = good & ((branches == side) | below)
            if finite_frame:
                for window in cfg.reference_half_windows_px:
                    for maximum_depth in cfg.maximum_depths_px:
                        selected = (eligible & (np.abs(self.point_s - center) <= window)
                                    & (self.depth <= maximum_depth)
                                    & ((distance <= cfg.donor_radius_px)
                                       | (self.depth <= cfg.shallow_anchor_depth_px)))
                        indices = np.flatnonzero(selected)
                        if (len(indices) < cfg.minimum_points
                                or np.ptp(self.point_s[indices]) < cfg.minimum_address_span_px
                                or np.ptp(self.depth[indices]) < cfg.minimum_depth_span_px):
                            continue
                        for model in cfg.models:
                            result = self._profile(indices, current, velocity, velocity_sigma,
                                tip, blade_velocity, blade_sigma, center, model,
                                float(tip_uncertainty_px), side, window, maximum_depth)
                            if result is not None:
                                profiles.append(result)
            profile_sets.append(profiles)
        sides = [self._combine_side(i + 1, profiles) for i, profiles in enumerate(profile_sets)]
        usable = [side for side in sides if side['flux_valid']]
        conditional = [side for side in sides if np.isfinite(side['flux_px_s'])]
        flux, sigma, spread = (np.nan, np.nan, np.nan)
        if usable or conditional:
            ensemble = usable or conditional
            flux, sigma, spread = _robust_center([s['flux_px_s'] for s in ensemble],
                                                [s['flux_sigma_px_s'] for s in ensemble])
        absolute = [side for side in sides if side['absolute_valid']]
        address, address_sigma, address_spread = (np.nan, np.nan, np.nan)
        if absolute:
            address, address_sigma, address_spread = _robust_center(
                [s['absolute_s_px'] for s in absolute], [s['absolute_sigma_px'] for s in absolute])
        agreement = np.nan
        if len(usable) == 2:
            agreement = abs(usable[0]['flux_px_s'] - usable[1]['flux_px_s']) / max(
                np.hypot(usable[0]['flux_sigma_px_s'], usable[1]['flux_sigma_px_s']), 1.e-9)
        result = dict(sides=sides, combined_flux_px_s=flux, combined_flux_sigma_px_s=sigma,
                      combined_flux_valid=bool(usable), bilateral_flux_spread_px_s=spread,
                      bilateral_flux_disagreement_sigma=agreement,
                      absolute_s_px=address, absolute_sigma_px=address_sigma,
                      absolute_valid=bool(absolute), absolute_model_spread_px=address_spread,
                      contact_s_input_px=center, original_ids_preserved=True,
                      compression_imposes_zero_slip=False)
        if return_models:
            result['models'] = profile_sets
        return result
