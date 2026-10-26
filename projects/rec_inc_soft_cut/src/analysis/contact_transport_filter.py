"""Shared causal contact filter with explicitly distinct evidence recipes.

Both recipes preserve missing-position states and bounded forecasts. They do
not enforce positive slip, monotonic growth, a terminal value, or fracture.
The two public entry points retain the historical bounded and recovered-window
numerics; sharing the state update does not make their qualification identical.
"""
from dataclasses import asdict, dataclass
import numpy as np
from .contact_transport_smoother import _psd


@dataclass(frozen=True)
class TransportPolicy:
    max_prediction_s: float = .25
    single_side_anchor_s: float = .30
    zero_window_s: float = .45
    zero_min_span_s: float = .35
    zero_max_sample_gap_s: float = .075
    zero_resolution_px: float = 1.
    disagreement_z: float = 3.
    release_z: float = 3.

    @property
    def zero_velocity_px_s(self):
        return self.zero_resolution_px/self.zero_window_s


WINDOWED_STATES = {0: 'unavailable', 1: 'short prediction', 2: 'motion/address supported',
          3: 'observed near-zero hold', 4: 'held after near-zero evidence; unobserved assumption',
          5: 'imposed first-contact zero; not a measured velocity'}


BOUNDED_STATES = {key: value for key, value in WINDOWED_STATES.items() if key != 5}

def window_evidence(times, flux, sensitivity, used, incompatible, policy):
    """Past-window signed increments, evaluated separately for each side.

    The saved sensitivities mix measurement noise and model error. They cannot
    be divided by sqrt(N). Until components are separately validated, their
    integrated bound assumes full temporal correlation. This intentionally
    does not turn a noisy, zero-centred signal into proof of sticking.
    """
    n = len(times)
    delta = np.full((n, 2), np.nan)
    bound = np.full((n, 2), np.nan)
    excursion = np.full((n, 2), np.nan)
    span = np.full((n, 2), np.nan)
    complete = np.zeros((n, 2), bool)
    zero = np.zeros(n, bool)
    moving = np.zeros(n, bool)
    for i in range(n):
        start = np.searchsorted(times, times[i]-policy.zero_window_s)
        active = used[start:i+1].any(axis=0)
        for side in np.flatnonzero(active):
            ids = np.flatnonzero(used[start:i+1, side])+start
            if (not used[i, side] or len(ids) < 5
                    or times[ids[-1]]-times[ids[0]] < policy.zero_min_span_s
                    or np.max(np.diff(times[ids])) > policy.zero_max_sample_gap_s):
                continue
            dt = np.diff(times[ids])
            v, e = flux[ids, side], sensitivity[ids, side]
            increments = .5*(v[1:]+v[:-1])*dt
            path = np.r_[0., np.cumsum(increments)]
            delta[i, side] = path[-1]
            bound[i, side] = np.sum(.5*(e[1:]+e[:-1])*dt)
            excursion[i, side] = np.ptp(path)
            span[i, side] = times[ids[-1]]-times[ids[0]]
            complete[i, side] = True
        # A side that has appeared in this window cannot be silently dropped
        # to make two disagreeing sides cancel, or make the mean look zero.
        if not active.any() or not complete[i, active].all() or incompatible[start:i+1].any():
            continue
        zero[i] = bool(np.all(abs(delta[i, active]) <= 2*bound[i, active])
                       and np.all(abs(delta[i, active])+2*bound[i, active] <= policy.zero_resolution_px)
                       and np.all(excursion[i, active] <= policy.zero_resolution_px))
        moving[i] = bool(np.any(abs(delta[i, active])-2*bound[i, active] > policy.zero_resolution_px))
    return dict(delta=delta, sensitivity=bound, excursion=excursion, span=span,
                available=complete, zero=zero, moving=moving)


def local_segments(times, observed, error, valid, maximum_gap):
    """Keep recoverable local increments without inventing displacement in gaps.

    Each segment starts at its own zero. Segment values are never added to the
    first-contact offset after its datum has been lost.
    """
    n = len(times)
    segment = np.full(n, -1, np.int32)
    displacement = np.full(n, np.nan)
    sensitivity = np.full(n, np.nan)
    previous = None
    label = -1
    for i in np.flatnonzero(valid):
        if previous is None or times[i]-times[previous] > maximum_gap:
            label += 1
            displacement[i], sensitivity[i] = 0., 0.
        else:
            dt = times[i]-times[previous]
            displacement[i] = displacement[previous]+.5*(observed[i]+observed[previous])*dt
            sensitivity[i] = sensitivity[previous]+.5*(error[i]+error[previous])*dt
        segment[i] = label
        previous = i
    return segment, displacement, sensitivity


def _instantaneous_zero_evidence(t, flux, raw_sigma, used, incompatible, policy):
    motion = used.any(axis=1)
    zero_candidate = np.zeros(len(t), bool)
    for i in range(len(t)):
        start = np.searchsorted(t, t[i]-policy.zero_window_s)
        inds = np.flatnonzero(motion[start:i+1])+start
        if not motion[i] or len(inds) < 5 or t[inds[-1]]-t[inds[0]] < policy.zero_min_span_s:
            continue
        if np.max(np.diff(t[inds])) > policy.zero_max_sample_gap_s:
            continue
        # Check every contributing side: cancellation is not sticking evidence.
        samples = flux[inds][used[inds]]
        uncertainties = raw_sigma[inds][used[inds]]
        limit = policy.zero_velocity_px_s
        zero_candidate[i] = (np.quantile(abs(samples), .9) <= limit
            and np.sqrt(np.mean(samples**2)) <= .65*limit
            and abs(np.median(samples))+2*np.median(uncertainties) <= limit
            and not incompatible[start:i+1].any())
    return zero_candidate


def _fuse(data, obs, acceleration_sigma, policy, *, windowed):
    """Return relative address, provenance and an unreported search guide.

    Position loss cannot be healed by a later velocity alone. An absolute
    address is needed to restore the displacement datum after an unbounded gap.
    Near-zero evidence uses finite-window increments of each side separately.
    Weak signed motion is retained with uncertainty, not direction-selected.
    """
    t = np.asarray(data['times'], float)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Strictly increasing finite physical times required')
    if not np.isfinite(acceleration_sigma) or acceleration_sigma <= 0:
        raise ValueError('Positive finite process scale required')
    for value in asdict(policy).values():
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Positive finite policy settings required')
    initial = float(data['initial']['contact_s_px'])
    flux = np.asarray(obs['side_flux_px_s'], float)
    raw_sigma = np.asarray(obs['side_flux_sigma_px_s'], float)
    valid = np.asarray(obs['side_flux_valid'], bool).copy()
    n = len(t)
    if flux.shape != (n, 2) or raw_sigma.shape != flux.shape or valid.shape != flux.shape:
        raise ValueError('Expected T x 2 side observations')
    valid &= np.isfinite(flux) & np.isfinite(raw_sigma) & (raw_sigma > 0)
    absolute = np.asarray(obs['absolute_s_px'], float)-initial
    abs_sigma = np.asarray(obs['absolute_sigma_px'], float)
    abs_valid = np.asarray(obs['absolute_valid'], bool).copy()
    abs_valid &= np.isfinite(absolute) & np.isfinite(abs_sigma) & (abs_sigma > 0)
    if absolute.shape != (n,) or abs_sigma.shape != (n,) or abs_valid.shape != (n,):
        raise ValueError('Expected T absolute observations')
    correlation = np.sqrt(max(1., .15/((t[1]-t[0]) if windowed else np.median(np.diff(t)))))
    used = np.zeros_like(valid)
    incompatible = np.zeros(n, bool)
    weak_single = np.zeros(n, bool)
    stale_single = np.zeros(n, bool)
    observed = np.full(n, np.nan)
    error = np.full(n, np.nan)
    anchor_time = t[0]
    for i in range(n):
        row = valid[i].copy()
        if row.all():
            incompatible[i] = abs(flux[i, 0]-flux[i, 1]) > policy.disagreement_z*np.hypot(*raw_sigma[i])
            if incompatible[i]:
                row[:] = False
            else:
                anchor_time = t[i]
        if abs_valid[i]:
            anchor_time = t[i]
        if row.sum() == 1:
            j = np.flatnonzero(row)[0]
            near_zero_precise = abs(flux[i, j])+2*raw_sigma[i, j] <= policy.zero_velocity_px_s
            weak_single[i] = abs(flux[i, j]) < 2*raw_sigma[i, j] and not near_zero_precise
            stale_single[i] = t[i]-anchor_time > policy.single_side_anchor_s
            # Frozen bounded fusion rejects unresolved unilateral direction;
            # windowed recovery retains the signed value and its sensitivity.
            if not windowed and weak_single[i]:
                row[:] = False
        used[i] = row
        if row.any():
            observed[i] = np.median(flux[i, row])
            spread = np.ptp(flux[i, row])/2
            error[i] = max(float(np.max(raw_sigma[i, row])), spread)*correlation
            if row.sum() == 1:
                error[i] *= 2.  # model dependence of unilateral transport
    motion = used.any(axis=1)
    evidence = window_evidence(t, flux, raw_sigma, used, incompatible, policy) if windowed else None
    zero_candidate = evidence['zero'] if windowed else _instantaneous_zero_evidence(
        t, flux, raw_sigma, used, incompatible, policy)
    # [relative offset, address velocity, common absolute-observation bias]
    x = np.array([0., 0., absolute[0] if abs_valid[0] else 0.])
    prior_v = 10000. if windowed else max(10., np.nanmedian(error[motion]) if motion.any() else 10.)*1000.
    P = np.diag([0., prior_v**2, (abs_sigma[0]*correlation)**2 if abs_valid[0] else 4.])
    states = np.zeros((n, 3))
    cov = np.zeros((n, 3, 3))
    status = np.zeros(n, np.uint8)
    available = np.zeros(n, bool)
    holding = False
    lost = False
    last_update = t[0]
    last_zero = np.nan
    held_offset = np.nan
    released = np.zeros(n, bool)
    held = np.zeros(n, bool)
    age = np.zeros(n)
    innovation = np.full((n, 2), np.nan)
    Hs = (np.array([1., 0., 1.]), np.array([0., 1., 0.]))
    for i in range(n):
        dt = t[i]-t[i-1] if i else 0.
        age[i] = t[i]-last_update
        if holding:
            resolved_v = motion[i] and abs(observed[i]) > max(2*policy.zero_velocity_px_s,
                                                             policy.release_z*error[i])
            resolved_s = abs_valid[i] and abs(absolute[i]-(held_offset+x[2])) > max(
                2*policy.zero_resolution_px, policy.release_z*np.sqrt(P[0, 0]+P[2, 2]+(abs_sigma[i]*correlation)**2))
            # A gradual restart may never exceed the instantaneous release
            # threshold. Expire the hold when observed nonzero evidence
            # persists, instead of silently locking small genuine motion.
            observed_hold_expired = bool(motion[i] and not zero_candidate[i]
                                         and t[i]-last_zero > policy.max_prediction_s)
            if resolved_v or resolved_s or (windowed and (evidence['moving'][i] or observed_hold_expired)):
                holding = False
                released[i] = True
                # If the hold was unobserved for too long, motion may have
                # resumed at an unknown instant; velocity cannot restore offset.
                if t[i]-(last_update if windowed else last_zero) > policy.max_prediction_s and not abs_valid[i]:
                    lost = True
                P[1, 1] = max(P[1, 1], acceleration_sigma**2*max(dt, .025))
        if not holding:
            allowed_dt = min(dt, max(0., policy.max_prediction_s-(t[i-1]-last_update))) if i else 0.
            A = np.array([[1., allowed_dt, 0.], [0., 1., 0.], [0., 0., 1.]])
            Q = np.zeros((3, 3))
            Q[:2, :2] = acceleration_sigma**2*np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
            x, P = A@x, _psd(A@P@A.T+Q)
            if age[i] > policy.max_prediction_s+1e-12:
                lost = True
                x[1] = 0.  # no indefinite internal guide drift either
            paired = abs_valid[i] and motion[i]
            for k, (value, sigma, good) in enumerate(((absolute[i], abs_sigma[i]*correlation, abs_valid[i]),
                                                      (observed[i], error[i], motion[i]))):
                if not good or (i == 0 and k == 0):
                    continue
                H = Hs[k]
                R = sigma**2*(2 if paired else 1)
                residual = value-H@x
                gain = P@H/(H@P@H+R)
                x += gain*residual
                I = np.eye(3)-np.outer(gain, H)
                P = _psd(I@P@I.T+R*np.outer(gain, gain))
                innovation[i, k] = residual
            if abs_valid[i]:
                lost = False
            if zero_candidate[i] and not lost:
                holding = True
                held_offset = float(x[0])
        if holding:
            x[0], x[1] = held_offset, 0.
            # Exact reported mean does not imply exact physical certainty.
            P[0, 0] += (policy.zero_velocity_px_s*dt)**2
            P[1, :] = P[:, 1] = 0.
            P[1, 1] = policy.zero_velocity_px_s**2
            if zero_candidate[i]:
                last_zero = t[i]
            status[i] = 3 if zero_candidate[i] else 4
        elif not lost:
            status[i] = 2 if motion[i] or abs_valid[i] else 1
        if motion[i] or abs_valid[i]:
            last_update = t[i]
            age[i] = 0.
        if i == 0:
            x[0] = 0.
            P[0, :] = P[:, 0] = 0.
            status[i] = 5 if windowed else 2
        states[i], cov[i] = x, P
        held[i] = holding
        available[i] = status[i] > 0
    sigma = np.sqrt(np.maximum(cov[:, 0, 0], 0))
    offset = np.where(available, states[:, 0], np.nan)
    result = dict(offset=offset, offset_sigma=np.where(available, sigma, np.nan),
        reference_s=initial+offset, search_reference_s=initial+states[:, 0],
        velocity=np.where(available | motion, states[:, 1], np.nan),
        filtered_state=states, filtered_covariance=cov,
        side_used=used, fused_flux=observed, fused_sigma=error,
        side_incompatible=incompatible, weak_single_side_rejected=np.zeros(n, bool) if windowed else weak_single,
        unanchored_single_side=stale_single,
        absolute_used=abs_valid, velocity_used=motion,
        prediction_only=~(motion | abs_valid) | (status == 4),
        state=status, offset_available=available, holding=held,
        zero_evidence=zero_candidate, hold_released=released,
        observation_age_s=age, innovation=innovation,
        policy=asdict(policy), causal_temporal_fusion=True,
        temporal_monotonicity_imposed=False, velocity_deadband_px_s=policy.zero_velocity_px_s,
        uncertainty_kind='conditional_model_sensitivity_not_statistical_confidence_interval')
    if windowed:
        local_id, local_offset, local_sensitivity = local_segments(
            t, observed, error, motion, policy.zero_max_sample_gap_s)
        result.update(weak_single_side_retained=weak_single,
            side_window_delta_px=evidence['delta'], side_window_sensitivity_px=evidence['sensitivity'],
            side_window_excursion_px=evidence['excursion'], side_window_span_s=evidence['span'],
            side_window_available=evidence['available'], local_velocity_available=motion,
            local_segment_id=local_id, local_segment_offset_px=local_offset,
            local_segment_sensitivity_px=local_sensitivity,
            fracture_cutoff_required=False, fracture_detected=False,
            zero_evidence_rule='signed window increment plus unreduced model sensitivity; each side separately')
    return result


def fuse_bounded(data, obs, acceleration_sigma, policy=TransportPolicy()):
    """Frozen bounded recipe: instantaneous near-zero evidence and direction gate."""
    return _fuse(data, obs, acceleration_sigma, policy, windowed=False)


def fuse_windowed(data, obs, acceleration_sigma, policy=TransportPolicy()):
    """Recovery recipe: signed finite-window evidence, retaining weak motion."""
    return _fuse(data, obs, acceleration_sigma, policy, windowed=True)
