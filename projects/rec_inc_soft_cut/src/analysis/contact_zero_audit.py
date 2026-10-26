"""Independent first-contact audit from preloading fixed-material motion.

This is a calibration candidate, not a fracture detector or a slip target fit.
The reference-plane gap only bounds the search. Its zero is not assumed to
equal contact because the initial segmentation and tip have finite offsets.
"""
import numpy as np


def sustained_first(mask, count=5):
    mask = np.asarray(mask, bool)
    if len(mask) < count:
        return None
    ids = np.flatnonzero(np.convolve(mask.astype(int), np.ones(count, int), 'valid') == count)
    return int(ids[0]) if len(ids) else None


def hinge_onset(times, displacement, plane_gap, qualified):
    """Fit the first persistent local motion above a precontact linear drift.

    Baseline comes from the approach with >=15 px normal clearance. The
    two-component displacement fit allows different directions of deformation.
    No fitted displacement, depth, or angle-specific target is supplied.
    """
    t, y, gap = map(np.asarray, (times, displacement, plane_gap))
    valid = np.asarray(qualified, bool) & np.isfinite(y).all(1) & np.isfinite(gap)
    baseline = np.flatnonzero(valid & (gap < -15.))
    # Use the last approach segment, avoiding very early blade-entry artifacts.
    if len(baseline) < 16:
        return dict(accepted=False, reason='insufficient_observed_precontact_baseline')
    baseline = baseline[-min(len(baseline), 100):]
    if np.ptp(t[baseline]) < .15:
        return dict(accepted=False, reason='short_precontact_baseline')
    t0 = t[baseline[0]]
    design = np.column_stack((np.ones(len(t)), t-t0))
    coef = np.linalg.lstsq(design[baseline], y[baseline], rcond=None)[0]
    residual = y-design@coef
    noise = 1.4826*np.median(abs(residual[baseline]-np.median(residual[baseline], axis=0)), axis=0)
    threshold = max(.6, 5*float(np.linalg.norm(noise)))
    amplitude = np.linalg.norm(residual, axis=1)
    onset = sustained_first(valid & (gap >= -15.) & (amplitude > threshold))
    if onset is None:
        return dict(accepted=False, reason='no_persistent_local_motion', threshold_px=threshold)
    # Only the first few pixels of response fit the onset; large deformation
    # must not retrospectively move this datum to a later mechanical event.
    reached = np.flatnonzero(valid & (np.arange(len(t)) >= onset) & (amplitude >= 3.))
    stop = int(reached[0]) if len(reached) else min(len(t)-1, onset+40)
    stop = min(stop+1, onset+50, len(t)-1)
    if t[stop]-t[onset] < .1:
        stop = min(len(t)-1, int(np.searchsorted(t, t[onset]+.1)))
    begin = max(int(baseline[0]), onset-100)
    ids = np.flatnonzero(valid & (np.arange(len(t)) >= begin) & (np.arange(len(t)) <= stop))
    if len(ids) < 25 or amplitude[stop] < max(1.2, 2*threshold):
        return dict(accepted=False, reason='insufficient_initial_response', threshold_px=threshold)
    fits = []
    for k in range(begin+8, onset+1):
        if not valid[k] or gap[k] < -15. or gap[k] > 10.:
            continue
        if t[k]-t[ids[0]] < .12 or t[ids[-1]]-t[k] < .10:
            continue
        X = np.column_stack((np.ones(len(ids)), t[ids]-t[k], np.maximum(t[ids]-t[k], 0)))
        beta = np.linalg.lstsq(X, y[ids], rcond=None)[0]
        rms = float(np.sqrt(np.mean(np.sum((X@beta-y[ids])**2, axis=1))))
        fits.append((rms, k, beta))
    if not fits:
        return dict(accepted=False, reason='onset_not_bracketed')
    rms, k, beta = min(fits, key=lambda item: item[0])
    after = float(np.linalg.norm(beta[2])*(t[stop]-t[k]))
    accepted = rms <= .5 and after >= 1.2 and k > begin+8
    return dict(accepted=bool(accepted), reason='motion_hinge' if accepted else 'poor_hinge_fit',
        frame=int(k), threshold_frame=int(onset), rms_px=rms, initial_response_px=after,
        baseline_frames=[int(baseline[0]), int(baseline[-1])],
        fit_frames=[int(ids[0]), int(ids[-1])], threshold_px=threshold,
        signed_reference_plane_gap_px=float(gap[k]))


def audit_contact(times, positions, supported, seeds, tips, tip_supported,
                  mapping, old_contact, reversal):
    """Use fixed IDs at three radii; return an auditable candidate and signals."""
    count = min(len(positions), reversal+1)
    p, ok, q = np.asarray(positions[:count]), np.asarray(supported[:count], bool), np.asarray(seeds)
    times, tips = np.asarray(times[:count]), np.asarray(tips[:count])
    valid_tip = np.asarray(tip_supported[:count], bool) & np.isfinite(tips).all(1)
    center = mapping.contact_s
    far = mapping.member & (abs(mapping.point_s-center) >= 180.)
    if far.sum() < 25:
        return dict(accepted=False, reason='insufficient_far_field_ids'), {}
    delta = p-q
    far_valid = ok & far
    common = np.nanmedian(np.where(far_valid[..., None], delta, np.nan), axis=1)
    common[far_valid.sum(axis=1) < 20] = np.nan
    u = (tips-common)@mapping.tangent
    gap = (tips-common)@mapping.normal-np.interp(u, mapping.boundary@mapping.tangent,
                                                     mapping.boundary@mapping.normal)
    # Restrict to the approach and first 12 pixels of normal entry; later
    # deformation cannot be used as a new initial-contact event.
    crossed = sustained_first(valid_tip & (gap >= 12.))
    stop = min(count, crossed+1) if crossed is not None else min(count, old_contact+150)
    residual = delta-common[:, None, :]
    variants, signals, counts = [], [], []
    for radius in (30., 45., 60.):
        near = mapping.member & (abs(mapping.point_s-center) <= radius) & (mapping.depth <= 40.)
        support = ok & near
        signal = np.nanmedian(np.where(support[..., None], residual, np.nan), axis=1)
        qualified = valid_tip & (support.sum(axis=1) >= max(12, int(.6*near.sum())))
        result = hinge_onset(times[:stop], signal[:stop], gap[:stop], qualified[:stop])
        result.update(radius_px=radius, fixed_ids=int(near.sum()))
        variants.append(result); signals.append(signal); counts.append(support.sum(axis=1))
    accepted = [v for v in variants if v['accepted']]
    result = dict(accepted=False, old_contact_frame=int(old_contact), variants=variants,
                  reason='insufficient_cross_radius_agreement', no_microscopy_target_used=True)
    if len(accepted) >= 2:
        frames = np.array([v['frame'] for v in accepted], int)
        ys = tips[frames, 1]
        k = int(np.rint(np.median(frames)))
        if not valid_tip[k]:
            choices = np.flatnonzero(valid_tip & (np.arange(count) >= frames.min()) & (np.arange(count) <= frames.max()))
            k = int(choices[np.argmin(abs(choices-k))]) if len(choices) else int(frames[0])
        result.update(candidate_frame=k, candidate_frame_range=[int(frames.min()), int(frames.max())],
            cross_radius_vertical_spread_px=float(np.ptp(ys)),
            old_minus_candidate_vertical_px=float(tips[old_contact, 1]-tips[k, 1]),
            signed_reference_plane_gap_px=float(gap[k]),
            accepted=bool(np.ptp(ys) <= 6. and valid_tip[k]),
            reason='cross_radius_motion_onset' if np.ptp(ys) <= 6. else 'radius_dependent_onset')
    return result, dict(times_s=times, reference_plane_gap_px=gap,
        far_field_translation_px=common, near_displacement_px=np.asarray(signals),
        near_supported_count=np.asarray(counts), tip_supported=valid_tip, tips_px=tips)
