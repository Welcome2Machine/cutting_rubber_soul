"""Conditional original-surface transport using a curved current boundary.

The current contour supplies geometry, not proof of original-surface identity.
Each side extrapolates fixed-ID velocities in its local tangent/normal frame,
using the SAME reference-coordinate basis as its boundary-address fit. Flux is
evaluated at that fit's surface root, never at the nearest interior material ID.
All uncertainty values are model sensitivity scales, not statistical confidence
intervals. Alternative depth models and the two sides are correlated evidence.
"""
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from .boundary_material_contact import (
    BoundaryMaterialContactConfig, BoundaryMaterialContactEstimator,
    _basis, extract_boundary_path, project_to_path,
)
from .contact_transport_observations import _empty_side, _fit, _robust_center
from .objective_boundary_velocity import fit_objective_velocity, objective_motion_holdout


@dataclass(frozen=True)
class BoundaryContactTransportConfig:
    velocity_model: str = 'objective_rotation'
    minimum_points: int = 12
    velocity_noise_floor_px_s: float = .25
    position_noise_floor_px: float = .25
    maximum_condition: float = 1.e7
    maximum_position_rms_px: float = 6.
    maximum_position_holdout_px: float = 10.
    maximum_flux_gap_px: float = 8.
    maximum_root_extrapolation_px: float = 60.
    maximum_leverage: float = 10.
    maximum_velocity_holdout_noise_ratio: float = 6.
    compatibility_sigma: float = 3.
    maximum_absolute_extrapolation_px: float = 6.
    maximum_absolute_rms_px: float = 4.
    maximum_absolute_holdout_px: float = 6.
    resolved_velocity_px_s: float = .5


def _tangent_at(path, arc, stations):
    """Piecewise-linear tangent, preserving separate sides at an apex."""
    indices = np.clip(np.searchsorted(arc, stations, side='right') - 1, 0, len(arc)-2)
    tangent = path[indices+1] - path[indices]
    return tangent / np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1.e-12)


def _motion_holdout(X, values, weights, depth, arc_distance, cfg):
    errors = []
    for test in (depth <= np.quantile(depth, .3),
                 arc_distance <= np.quantile(arc_distance, .25)):
        train = ~test
        if test.sum() < 3 or train.sum() < X.shape[1]+3:
            return np.inf
        fit = _fit(X[train], values[train], weights[train],
                   cfg.velocity_noise_floor_px_s, cfg.maximum_condition)
        if fit is None:
            return np.inf
        errors.append(float(np.sqrt(np.average(
            np.sum((X[test] @ fit['beta'] - values[test])**2, axis=1), weights=weights[test]))))
    return max(errors)


class BoundaryContactTransportObserver:
    """Curved-boundary alternative to ContactTransportObserver.

    ``labels`` is an image segmentation (material label 1), or ``path`` supplies
    ordered ``xy``, increasing arclength ``arc``, and ``tip_gap``. Side 1/2 means
    negative/positive original-oriented boundary arc. ``branch_labels`` is
    accepted for API compatibility; routing uses the complete boundary path,
    including nonlocal ambiguity and endpoint-clamp rejection, rather than x.
    Surface continuity to the tip remains an explicit unproven assumption.
    """

    def __init__(self, seed_points, point_ids, surface_reference, um_per_px,
                 config=None, boundary_config=None):
        self.config = config or BoundaryContactTransportConfig()
        if self.config.velocity_model not in ('objective_rotation', 'legacy_projected'):
            raise ValueError('Unknown boundary velocity model')
        self.boundary = BoundaryMaterialContactEstimator(seed_points, point_ids,
            surface_reference, um_per_px, boundary_config or BoundaryMaterialContactConfig())
        self.ids = self.boundary.ids
        self.contact_s = float(surface_reference.get('contact_s_px', 0.))
        self.um_per_px = float(um_per_px)

    def _profile(self, profile, current, velocity, good, velocity_sigma, blade_velocity,
                 blade_sigma, path, r, tangent, normal, tip_sigma):
        cfg, mapping = self.config, self.boundary
        side = 1 if profile['side'] == -1 else 2
        out = _empty_side(side)
        root = float(profile['reference_s_px'])
        stretch = float(profile['estimated_surface_stretch'])
        gap = float(path['tip_gap'])
        # Approach zero on the selected side, without smoothing across a cusp.
        step = min(1.e-5, .1 * float(np.min(np.diff(path['arc']))))
        tau = _tangent_at(path['xy'], path['arc'], profile['side']*step)
        orientation = np.sign(np.linalg.det(np.column_stack((mapping.tangent, mapping.normal))))
        inward = np.array([-tau[1], tau[0]]) * orientation
        geometry_ok = (not path.get('tip_at_path_endpoint', False)
            and gap <= cfg.maximum_flux_gap_px
            and profile['rms_px'] <= cfg.maximum_position_rms_px
            and profile['heldout_px'] <= cfg.maximum_position_holdout_px
            and profile['reference_extrapolation_px'] <= cfg.maximum_root_extrapolation_px)
        absolute_ok = (geometry_ok and len(profile['anchor_ids']) >= 2
            and gap <= max(4., 2.*tip_sigma)
            and profile['reference_extrapolation_px'] <= cfg.maximum_absolute_extrapolation_px
            and profile['rms_px'] <= cfg.maximum_absolute_rms_px
            and profile['heldout_px'] <= cfg.maximum_absolute_holdout_px)
        absolute_sigma = max(cfg.position_noise_floor_px/stretch,
                             .5*(profile['high_s_px']-profile['low_s_px']))
        out.update(model='boundary_'+profile['model'], boundary_side=profile['side'],
            observation_model='boundary',
            velocity_model=cfg.velocity_model,
            absolute_s_px=root, absolute_sigma_px=absolute_sigma,
            # A position fit alone never authorizes an absolute contact update.
            # In particular, the insufficient-velocity and singular-fit exits
            # below must leave the joint observation invalid.
            absolute_valid=False, joint_absolute_valid=False,
            position_absolute_compatible=bool(absolute_ok), velocity_fit_valid=False,
            normal_compatible=False, absolute_shallow_supported=len(profile['anchor_ids']) >= 2,
            absolute_conditional=True, flux_eval_s_px=root,
            surface_xy=np.asarray(path['tip_projection']), tangent_xy=tau,
            surface_stretch=stretch, surface_tip_gap_px=gap,
            position_rms_px=profile['rms_px'], position_holdout_px=profile['heldout_px'],
            reference_extrapolation_px=profile['reference_extrapolation_px'],
            maximum_depth_px=profile['depth_limit_px'],
            geometry_compatible=bool(geometry_ok), minimum_initial_depth_px=profile['support_depth_min_px'],
            original_surface_identity_proven=False, original_surface_continuity_assumed=True,
            conditional_flux=True, anchor_ids=profile['anchor_ids'],
            boundary_coefficients=profile['beta'], reference_center_s_px=profile['origin_s_px'],
            coordinate_scale_px=60., model_count=1, reasons=[])
        q = profile['support_indices']
        q = q[good[q]]
        out['support_ids'], out['support_indices'] = self.ids[q], q
        if len(q) < cfg.minimum_points:
            out['reasons'] = ['insufficient_continuous_fixed_id_velocity_support']
            return out
        s, d = mapping.s[q], mapping.d[q]
        quadratic = profile['model'] == 'quadratic_depth'
        X = _basis(s, d, profile['origin_s_px'], quadratic)
        C = _basis(np.array([root]), np.zeros(1), profile['origin_s_px'], quadratic)[0]
        # Project RELATIVE velocity before fitting. This exactly preserves zero
        # flux under common rigid translation even on a highly curved contour.
        relative = velocity[q] - blade_velocity
        values = np.column_stack((np.sum(relative*tangent[q], axis=1),
                                  np.sum(relative*normal[q], axis=1)))
        cells = np.floor(np.column_stack((s, d))/8.).astype(np.int64)
        _, inv, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        weights = np.exp(-.5*(r[q]/120.)**2-.5*(d/profile['depth_limit_px'])**2)/counts[inv]
        objective = cfg.velocity_model == 'objective_rotation'
        reference_xy = path['velocity_reference_xy']
        relative_xy = current[q]-reference_xy
        contact_relative_xy = np.asarray(path['tip_projection'])-reference_xy
        if objective:
            fit = fit_objective_velocity(X, relative_xy, tangent[q], normal[q], relative,
                weights, C, contact_relative_xy, tau, inward,
                floor=cfg.velocity_noise_floor_px_s, maximum_condition=cfg.maximum_condition)
        else:
            fit = _fit(X, values, weights, cfg.velocity_noise_floor_px_s, cfg.maximum_condition)
        if fit is None:
            out['reasons'] = ['ill_conditioned_velocity_model']
            return out
        if objective:
            heldout = objective_motion_holdout(X, relative_xy, tangent[q], normal[q], relative,
                weights, d, abs(r[q]), C, contact_relative_xy, tau, inward, cfg)
            leverage = fit['leverage']
            rel_t, rel_n = fit['evaluation']
            out.update(rigid_rotation_rate_rad_s=fit['angular_rate'], velocity_design_rank=fit['rank'],
                velocity_design_columns=fit['columns'], velocity_objective_condition=fit['condition'],
                velocity_contact_estimability_error=fit['contact_estimability_error'],
                rotation_coefficient_is_nuisance=True)
        else:
            heldout = _motion_holdout(X, values, weights, d, abs(r[q]), cfg)
            leverage = float(C @ fit['inverse'] @ C)
            rel_t, rel_n = C @ fit['beta']
        out['velocity_model'] = cfg.velocity_model
        flux, normal_relative = -float(rel_t)/stretch, -float(rel_n)
        point_sigma = float(np.sqrt(np.average(velocity_sigma[q]**2, weights=weights)))
        noise = max(cfg.velocity_noise_floor_px_s, point_sigma)
        velocity_error = max(noise, fit['rms'], heldout)*np.sqrt(1.+max(leverage, 0.))
        geometry_error = max(cfg.position_noise_floor_px, profile['rms_px'], profile['heldout_px'])
        geometry_error += gap + tip_sigma
        # Geometric location error also perturbs the extrapolated velocity.
        ds_velocity = (fit['beta'][1]/60.)
        dn_velocity = (fit['beta'][2]+fit['beta'][3]*(root-profile['origin_s_px'])/60.)/60.
        velocity_gradient = np.linalg.norm(np.column_stack((ds_velocity, dn_velocity)), ord=2)
        if objective:
            velocity_gradient += abs(fit['angular_rate'])
        geometric_velocity_error = geometry_error*velocity_gradient
        normal_sigma = float(np.sqrt(blade_sigma**2+velocity_error**2+geometric_velocity_error**2))
        stretch_sigma = geometry_error/max(float(np.ptp(s)), 1.)*np.sqrt(1.+max(leverage, 0.))
        flux_sigma = float(np.sqrt(normal_sigma**2+normal_relative**2+(flux*stretch_sigma)**2)/stretch)
        normal_ok = abs(normal_relative) <= cfg.compatibility_sigma*normal_sigma
        fit_ok = (np.isfinite(heldout) and leverage <= cfg.maximum_leverage
            and heldout <= cfg.maximum_velocity_holdout_noise_ratio*noise)
        valid = bool(geometry_ok and normal_ok and fit_ok)
        reasons = []
        if not geometry_ok:
            reasons.append('boundary_geometry_or_reference_extrapolation_incompatible')
        if not normal_ok:
            reasons.append('normal_velocity_incompatible')
        if not fit_ok:
            reasons.append('velocity_holdout_or_leverage_incompatible')
        out.update(flux_px_s=flux, flux_sigma_px_s=flux_sigma, flux_valid=valid,
            normal_relative_velocity_px_s=normal_relative, normal_sigma_px_s=normal_sigma,
            normal_compatible=bool(normal_ok), velocity_fit_valid=bool(fit_ok), velocity_holdout_px_s=heldout,
            velocity_rms_px_s=fit['rms'], leverage=leverage,
            surface_velocity_xy=blade_velocity+rel_t*tau+rel_n*inward,
            tangential_relative_velocity_px_s=-float(rel_t),
            surface_stretch_sigma=stretch_sigma, geometric_velocity_sigma_px_s=geometric_velocity_error,
            coefficients_relative_velocity_tn=fit['beta'], reasons=reasons,
            valid_model_count=int(valid), near_stick_observed=bool(valid
                and abs(flux)+2.*flux_sigma <= cfg.resolved_velocity_px_s))
        # A dynamically rejected contact cannot supply a direct absolute update.
        out['absolute_valid'] = out['joint_absolute_valid'] = bool(absolute_ok and normal_ok and fit_ok)
        return out

    @staticmethod
    def _combine(side, profiles):
        if not profiles:
            empty = _empty_side(side)
            empty.update(observation_model='boundary', joint_absolute_valid=False,
                         position_absolute_compatible=False, velocity_fit_valid=False,
                         normal_compatible=False)
            return empty
        valid = [p for p in profiles if p['flux_valid']]
        chosen = valid or profiles
        best = min(chosen, key=lambda p: p['position_rms_px']+p['position_holdout_px'])
        out = dict(best)
        out['flux_px_s'], out['flux_sigma_px_s'], out['flux_model_spread_px_s'] = _robust_center(
            [p['flux_px_s'] for p in chosen], [p['flux_sigma_px_s'] for p in chosen])
        out['flux_valid'] = bool(valid)
        out['model_count'], out['valid_model_count'] = len(profiles), len(valid)
        absolute = [p for p in profiles if p['absolute_valid']]
        candidates = absolute or profiles
        out['absolute_s_px'], out['absolute_sigma_px'], out['absolute_model_spread_px'] = _robust_center(
            [p['absolute_s_px'] for p in candidates], [p['absolute_sigma_px'] for p in candidates])
        out['absolute_valid'] = out['joint_absolute_valid'] = bool(absolute)
        out['observation_model'] = 'boundary'
        return out

    def estimate_frame(self, current, supported, tip_xy, blade_velocity_xy,
                       material_velocity_xy, velocity_valid, *, velocity_sigma_px_s=None,
                       blade_velocity_sigma_px_s=0., branch_labels=None, contact_s_px=None,
                       tip_uncertainty_px=2., return_models=False, labels=None, path=None):
        n, cfg = len(self.ids), self.config
        current, velocity = np.asarray(current, float), np.asarray(material_velocity_xy, float)
        support, vvalid = np.asarray(supported, bool), np.asarray(velocity_valid, bool)
        tip, blade = np.asarray(tip_xy, float), np.asarray(blade_velocity_xy, float)
        if (current.shape != (n, 2) or velocity.shape != (n, 2) or support.shape != (n,)
                or vvalid.shape != (n,) or tip.shape != (2,) or blade.shape != (2,)):
            raise ValueError('Positions and velocities must preserve all original material IDs')
        if branch_labels is not None and np.asarray(branch_labels).shape != (n,):
            raise ValueError('Branch labels must preserve original material IDs')
        if not np.isfinite(tip_uncertainty_px) or tip_uncertainty_px < 0:
            raise ValueError('Tip uncertainty must be finite and nonnegative')
        blade_sigma = float(np.linalg.norm(np.asarray(blade_velocity_sigma_px_s, float)))
        if not np.isfinite(blade_sigma):
            raise ValueError('Blade velocity uncertainty must be finite')
        sigma = np.full(n, cfg.velocity_noise_floor_px_s) if velocity_sigma_px_s is None else np.asarray(velocity_sigma_px_s, float)
        if sigma.shape == (n, 2):
            sigma = np.linalg.norm(sigma, axis=1)
        if sigma.shape != (n,):
            raise ValueError('Velocity uncertainty must preserve original material IDs')
        if path is None and labels is None:
            raise ValueError('A current boundary path or material segmentation labels is required')
        if path is None:
            path = extract_boundary_path(np.asarray(labels), tip, self.boundary.tangent)
        profile_sets = [[], []]
        if path is not None:
            path = dict(path)
            xy, arc = np.asarray(path['xy'], float), np.asarray(path['arc'], float)
            if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or arc.shape != (len(xy),)
                    or not np.isfinite(xy).all() or not np.isfinite(arc).all() or np.any(np.diff(arc) <= 0)):
                raise ValueError('Boundary path must be finite with strictly increasing arc')
            tr, foot, gap, _, clamped = project_to_path(xy, arc, tip[None])
            path.update(xy=xy, arc=arc-tr[0], tip_projection=foot[0],
                velocity_reference_xy=tip,
                tip_at_path_endpoint=bool(clamped[0] or tr[0] <= arc[0]+1.e-8 or tr[0] >= arc[-1]-1.e-8),
                tip_gap=max(float(gap[0]), float(path.get('tip_gap', 0.))))
            geometry = self.boundary.estimate_frame(current, support, tip, labels=labels,
                path=path, tip_uncertainty_px=tip_uncertainty_px, return_profiles=True)
            safe = np.nan_to_num(current, nan=-1.e6)
            r, _, _, _, _ = project_to_path(xy, path['arc'], safe,
                ambiguity_arc_separation_px=self.boundary.config.ambiguity_arc_separation_px)
            tangent = _tangent_at(xy, path['arc'], r)
            orientation = np.sign(np.linalg.det(np.column_stack((self.boundary.tangent, self.boundary.normal))))
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))*orientation
            good = (support & vvalid & np.isfinite(velocity).all(1) & np.isfinite(sigma) & (sigma >= 0))
            if not np.isfinite(blade).all() or clamped[0]:
                good[:] = False
            for profile_index, p in enumerate(geometry.get('profiles', [])):
                result = self._profile(p, current, velocity, good, sigma, blade, blade_sigma,
                    path, r, tangent, normal, float(tip_uncertainty_px))
                # Frame-local identity ties this exact position fit to its own
                # velocity/normal checks. Model names repeat across depths and
                # are not identifiers. Early returns retain the same evidence.
                result['position_profile_index'] = int(profile_index)
                result['position_profile_id'] = f'boundary:{profile_index}'
                if return_models:
                    result['position_profile'] = deepcopy(p)
                profile_sets[result['side']-1].append(result)
        sides = [self._combine(i+1, ps) for i, ps in enumerate(profile_sets)]
        valid = [s for s in sides if s['flux_valid']]
        finite = [s for s in sides if np.isfinite(s['flux_px_s'])]
        candidates = valid or finite
        flux, flux_sigma, spread = _robust_center([s['flux_px_s'] for s in candidates],
                                                  [s['flux_sigma_px_s'] for s in candidates])
        absolute = [s for s in sides if s['absolute_valid']]
        absolute_candidates = absolute or [s for s in sides if np.isfinite(s['absolute_s_px'])]
        address, address_sigma, address_spread = _robust_center(
            [s['absolute_s_px'] for s in absolute_candidates], [s['absolute_sigma_px'] for s in absolute_candidates])
        disagreement = (abs(valid[0]['flux_px_s']-valid[1]['flux_px_s'])/
            max(np.hypot(valid[0]['flux_sigma_px_s'], valid[1]['flux_sigma_px_s']), 1.e-9)) if len(valid) == 2 else np.nan
        out = dict(sides=sides, combined_flux_px_s=flux, combined_flux_sigma_px_s=flux_sigma,
            combined_flux_valid=bool(valid), bilateral_flux_spread_px_s=spread,
            bilateral_flux_disagreement_sigma=disagreement, absolute_s_px=address,
            absolute_sigma_px=address_sigma, absolute_valid=bool(absolute), absolute_model_spread_px=address_spread,
            contact_s_input_px=self.contact_s if contact_s_px is None else float(contact_s_px),
            original_ids_preserved=True, compression_imposes_zero_slip=False,
            original_surface_identity_proven=False, original_surface_continuity_assumed=True,
            observation_model='conditional_curved_boundary_transport', side_routing='signed_boundary_arc',
            path_xy=np.empty((0, 2)) if path is None else path['xy'],
            path_arc=np.empty(0) if path is None else path['arc'],
            boundary_tip_gap_px=np.nan if path is None else path['tip_gap'])
        if return_models:
            out['models'] = profile_sets
        return out
