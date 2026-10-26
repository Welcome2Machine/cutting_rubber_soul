"""Softly couple a measured contour, fixed material origin, and surface stretch.

Geometry and P0 are upstream inputs; this module never changes either. The
reference address is inferred with explicit depth/shear nuisance terms. A P0
check can calibrate a supported material map, but cannot force a contact point
onto the blade or manufacture an observation during an identity gap.
"""
from __future__ import annotations

import numpy as np

from .boundary_material_contact import _basis


def couple_origin_profile(profile, mapping, current_arc, initial_s, origin_arc,
                          origin_sigma, origin_state, geometry_correction,
                          maximum_holdout_px=6.):
    """One scalar map r(s,d), softly checked against the independent fixed P0.

    r is current-boundary arc, zero at the independently located geometric C.
    At d=0, dr/ds = lambda. Therefore a shared identity predicts
    r(s0) + lambda * (s_C-s0) = 0 for the local affine-in-s surface model.
    This is not the assertion that current arc length itself equals slip.
    """
    p=profile
    raw=float(p['reference_s_px']);stretch=float(p['estimated_surface_stretch'])
    radius=float(max(raw-p['low_s_px'],p['high_s_px']-raw))
    out=dict(raw_s_px=raw,reference_s_px=raw,sigma_px=radius,origin_used=False,
             origin_residual_px=np.nan,origin_residual_after_px=np.nan,
             intercept_correction_px=0.,origin_weight=0.,rms_after_px=float(p['rms_px']),
             reason='independent_origin_unavailable')
    if (origin_state not in (1,2) or not np.isfinite([origin_arc,origin_sigma,geometry_correction]).all()
            or origin_sigma<=0. or origin_sigma>8. or geometry_correction>8.):
        return out
    if (not np.isfinite([raw,radius,stretch,p['rms_px'],p['heldout_px'],initial_s]).all()
            or radius<=0. or not np.isfinite(p['beta']).all()):
        out['reason']='invalid_material_profile';return out
    if origin_arc*float(p.get('side',np.sign(origin_arc) or 1)) < -max(4.,2.*origin_sigma):
        out['reason']='origin_on_other_current_boundary_branch';return out
    if not .1<stretch<10.:
        out['reason']='surface_stretch_unresolved';return out
    indices=np.asarray(p['support_indices'],int)
    s,d=mapping.s[indices],mapping.d[indices]
    values=np.asarray(current_arc)[indices]
    if not len(indices) or not np.isfinite(np.r_[s,d,values]).all():
        out['reason']='nonfinite_material_correspondence';return out
    if not s.min()-6.<=initial_s<=s.max()+6.:
        out['reason']='origin_outside_same_reference_patch';return out
    quadratic=p['model']=='quadratic_depth'
    at_origin=_basis(np.array([initial_s]),np.zeros(1),p['origin_s_px'],quadratic)[0]
    predicted=float(at_origin@p['beta'])
    residual=float(origin_arc-predicted)
    model_sigma=max(2.,float(p['rms_px']),float(p['heldout_px']))
    # Same saved IDs may influence both measurements. Use inflated variances;
    # no sqrt(N) gain and no claim of statistically independent evidence.
    sigma0=np.hypot(origin_sigma,geometry_correction)
    total=np.hypot(model_sigma,sigma0)*np.sqrt(2.)
    out['origin_residual_px']=residual
    if abs(residual)>max(12.,3.*total):
        out['reason']='origin_identity_and_local_map_disagree';return out
    weight=model_sigma**2/(model_sigma**2+2.*sigma0**2)
    correction=float(weight*residual)
    corrected=raw-correction/stretch
    if not mapping.addresses[0]<=corrected<=mapping.addresses[-1]:
        out['reason']='corrected_address_outside_original_surface';return out
    X=_basis(s,d,p['origin_s_px'],quadratic)
    before=X@p['beta']-values
    after=before+correction
    rms_before=float(np.sqrt(np.mean(before**2)))
    rms_after=float(np.sqrt(np.mean(after**2)))
    # The origin is not allowed to override a large amount of contradictory
    # tracked texture. This gate protects geometry from an overconfident P0.
    if (not np.isfinite([rms_after,rms_before]).all() or rms_after>max(4.,rms_before+1.5)
            or p['heldout_px']+abs(correction)>maximum_holdout_px):
        out['reason']='origin_correction_conflicts_with_material_observations';return out
    out.update(reference_s_px=float(corrected),
        sigma_px=float(max(radius,np.hypot(sigma0,model_sigma)/stretch,abs(corrected-raw))),
        origin_used=True,origin_residual_after_px=float(residual-correction),
        intercept_correction_px=correction,origin_weight=float(weight),
        rms_after_px=rms_after,reason='soft_fixed_origin_and_surface_stretch_coupling')
    return out


def conservative_address(candidates):
    """Choose an actual candidate and cover the union of all sensitivity ranges.

    Duplicating a correlated model cannot change the result. Incompatible modes
    are not averaged into a material address supported by neither candidate;
    the caller must separately gate observation use on interval compatibility.
    """
    if not candidates:return np.nan,np.nan
    values=np.array([v[0] for v in candidates],float)
    sigma=np.array([v[1] for v in candidates],float)
    finite=np.isfinite(values)&np.isfinite(sigma)&(sigma>0.)
    if not finite.any():return np.nan,np.nan
    values,sigma=values[finite],sigma[finite]
    selected=int(np.lexsort((values,sigma))[0])
    value=float(values[selected])
    return value,float(np.max(abs(values-value)+sigma))


def joint_profile_qualified(motion_profile):
    """Eligibility belongs to this exact position/velocity model, never a side.

    The explicit flags are deliberately all required. A legacy side summary or
    a profile from another depth cannot stand in for the attached position fit.
    """
    p = motion_profile.get('position_profile')
    if p is None:
        return False
    checks = ('position_absolute_compatible', 'velocity_fit_valid', 'normal_compatible',
              'flux_valid', 'absolute_valid', 'joint_absolute_valid')
    return bool(all(motion_profile.get(key, False) for key in checks)
                and p.get('contact_compatible', False))


def couple_joint_transport_profiles(transport_record, mapping, current_arc, initial_s,
                                    origin_arc, origin_sigma, origin_state, geometry_correction,
                                    *, geometry_qualified=True, maximum_holdout_px=6.):
    """Couple only individually qualified position/velocity profile pairs.

    ``transport_record`` must come from ``return_models=True``. ``profiles``
    contains per-profile diagnostics; ``selected`` contains at most one jointly
    qualified candidate per physical boundary side. With no selected candidate,
    absolute_valid is False: no side-level or frame-level fallback is allowed.
    Position-profile indices/IDs are unique only within this observer frame.
    """
    if 'models' not in transport_record:
        raise ValueError('Per-profile models are required; call the boundary observer with return_models=True')
    diagnostics, candidates = [], []
    seen = set()
    for side_models in transport_record['models']:
        for motion in side_models:
            if 'position_profile' not in motion or 'position_profile_index' not in motion:
                raise ValueError('Each motion model must carry its exact position_profile and unique frame index')
            index = int(motion['position_profile_index'])
            if index in seen:
                raise ValueError('Duplicate frame-local position profile index')
            seen.add(index)
            p = motion['position_profile']
            eligible = bool(geometry_qualified and joint_profile_qualified(motion))
            coupled = couple_origin_profile(p, mapping, current_arc, initial_s, origin_arc,
                origin_sigma, origin_state if eligible else 4, geometry_correction, maximum_holdout_px)
            if not eligible:
                coupled['reason'] = ('geometry_not_qualified' if not geometry_qualified
                                     else 'position_velocity_normal_not_jointly_qualified')
            raw = float(p['reference_s_px'])
            coupled.update(position_profile_index=index,
                position_profile_id=motion.get('position_profile_id', f'boundary:{index}'),
                side=p['side'], model=p['model'], depth_limit_px=float(p['depth_limit_px']),
                support_ids=np.asarray(p['support_ids']).copy(),
                support_indices=np.asarray(p['support_indices']).copy(),
                velocity_support_ids=np.asarray(motion.get('support_ids', [])).copy(),
                score=float(p['score']), contact_compatible=bool(p.get('contact_compatible', False)),
                raw_sigma_px=float(max(raw-p['low_s_px'], p['high_s_px']-raw)),
                position_rms_px=float(p['rms_px']), position_holdout_px=float(p['heldout_px']),
                position_absolute_compatible=bool(motion.get('position_absolute_compatible', False)),
                velocity_fit_valid=bool(motion.get('velocity_fit_valid', False)),
                normal_compatible=bool(motion.get('normal_compatible', False)),
                flux_valid=bool(motion.get('flux_valid', False)),
                absolute_valid=bool(motion.get('absolute_valid', False)),
                profile_absolute_valid=bool(motion.get('absolute_valid', False)),
                joint_absolute_valid=bool(motion.get('joint_absolute_valid', False)),
                velocity_rms_px_s=float(motion.get('velocity_rms_px_s', np.nan)),
                velocity_holdout_px_s=float(motion.get('velocity_holdout_px_s', np.nan)),
                normal_relative_velocity_px_s=float(motion.get('normal_relative_velocity_px_s', np.nan)),
                normal_sigma_px_s=float(motion.get('normal_sigma_px_s', np.nan)),
                velocity_leverage=float(motion.get('leverage', np.nan)),
                qualified_for_selection=eligible,
                motion_reasons=list(motion.get('reasons', [])), observation_model='boundary')
            diagnostics.append(coupled)
            if eligible:
                candidates.append(coupled)
    selected = []
    for side in (-1, 1):
        group = [p for p in candidates if p['side'] == side]
        if group:
            selected.append(min(group, key=lambda p: p['score']))
    compatible, value, sigma, chosen_id = False, np.nan, np.nan, None
    if selected:
        pairs = [(p['reference_s_px'], p['sigma_px']) for p in selected]
        compatible = max(v-s for v, s in pairs) <= min(v+s for v, s in pairs)
        if compatible:
            value, sigma = conservative_address(pairs)
            choice = int(np.lexsort(([p['reference_s_px'] for p in selected],
                                     [p['sigma_px'] for p in selected]))[0])
            chosen_id = selected[choice]['position_profile_id']
    return dict(profiles=diagnostics, selected=selected, absolute_s_px=float(value),
        absolute_sigma_px=float(sigma), absolute_valid=bool(selected and compatible),
        compatible=bool(compatible), origin_used=bool(compatible and any(p['origin_used'] for p in selected)),
        chosen_position_profile_id=chosen_id,
        selected_position_profile_ids=[p['position_profile_id'] for p in selected],
        observation_model='boundary', geometry_qualified=bool(geometry_qualified),
        qualification_scope='same_position_profile_and_its_own_velocity_normal_checks')
