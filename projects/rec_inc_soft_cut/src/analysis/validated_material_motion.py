"""Fixed-ID derivatives with explicitly tested one-sided endpoint support.

Interior centered estimates are preserved. A missing centered derivative may
be recovered by two local quadratic fits on the SAME uninterrupted ID run.
No positions are filled, no invalid tracking edge is crossed, and no future
sample outside the supplied prefix is accessed. The uncertainty is a model
sensitivity, not an independently calibrated confidence interval.
"""
from __future__ import annotations

import numpy as np

from .contact_motion_evidence import supported_local_slopes


def _quadratic(dt, values, keep, noise_floor):
    """Batched local fits; derivative is evaluated at dt=0, including endpoints."""
    scale = max(float(np.max(abs(dt))), 1.e-9)
    z = dt / scale
    X = np.column_stack((np.ones(len(z)), z, z*z))
    base = keep.astype(float)
    weight = base.copy()
    y = np.where(keep[..., None], values, 0.)
    for _ in range(3):
        H = np.einsum('ti,tn,tj->nij', X, weight, X)
        inverse = np.linalg.pinv(H, rcond=1.e-10)
        beta = inverse @ np.einsum('ti,tn,tnd->nid', X, weight, y)
        residual = y - np.einsum('ti,nid->tnd', X, beta)
        error = np.linalg.norm(residual, axis=2)
        rms = np.sqrt(np.sum(base*error**2, axis=0)/np.maximum(base.sum(0), 1.))
        weight = base*np.minimum(1., 2.5*np.maximum(noise_floor, rms)[None]/np.maximum(error, 1.e-12))
    sigma = np.maximum(noise_floor, rms)*np.sqrt(np.maximum(inverse[:, 1, 1], 0.))/scale
    return dict(slope=beta[:, 1]/scale, rms=rms, sigma=sigma, beta=beta, scale=scale,
                condition=np.linalg.cond(H), count=keep.sum(0))


def _heldout_error(dt, values, keep, noise_floor):
    """Refit three disjoint temporal folds, including robust weights, from training only."""
    error_sum=np.zeros(keep.shape[1]);count=np.zeros(keep.shape[1]);bad=np.zeros(keep.shape[1],bool)
    for fold in range(3):
        test=keep & ((np.arange(len(dt)) % 3 == fold)[:,None])
        train=keep & ~test
        fit=_quadratic(dt,values,train,noise_floor)
        z=dt/fit['scale'];X=np.column_stack((np.ones(len(z)),z,z*z))
        error=np.sum((values-np.einsum('ti,nid->tnd',X,fit['beta']))**2,axis=2)
        error_sum+=np.sum(np.where(test,error,0.),axis=0)
        count+=test.sum(0)
        bad|=(train.sum(0)<3)|(fit['condition']>1.e6)
    result=np.sqrt(error_sum/np.maximum(count,1.))
    result[bad]=np.inf
    return result


def validated_local_slopes(values, times_s, supported, *, motion_step_valid=None,
                           half_window_s=.075, maximum_gap_s=.04,
                           minimum_samples=5, minimum_span_s=.025,
                           maximum_fit_rms=2., position_noise_floor_px=.25,
                           fallback_minimum_samples=7, agreement_sigma=3.):
    """Return centered derivatives plus separately labelled endpoint recovery.

Quadratic recovery uses the full available window and its inner 75% window.
Both fits must pass conditioning and held-out-position checks, and their
derivatives must agree within their conservative uncertainty. Recovery is
reported separately from observed positions; it does not upgrade ID evidence.
"""
    if position_noise_floor_px <= 0 or fallback_minimum_samples < 7 or agreement_sigma <= 0:
        raise ValueError('Recovery requires positive noise and at least seven samples')
    result = supported_local_slopes(values, times_s, supported,
        motion_step_valid=motion_step_valid, half_window_s=half_window_s,
        maximum_gap_s=maximum_gap_s, minimum_samples=minimum_samples,
        minimum_span_s=minimum_span_s, maximum_fit_rms=maximum_fit_rms)
    y, times = np.asarray(values, float), np.asarray(times_s, float)
    good = np.asarray(supported, bool) & np.isfinite(y).all(2)
    edge = result['motion_step_valid']
    segments = np.cumsum(~edge, axis=0)
    shape = good.shape
    result['source_kind'] = result['valid'].astype(np.int8)  # 1=centered; 2=quadratic recovery
    result['one_sided'] = np.zeros(shape, bool)
    result['cross_scale_z'] = np.full(shape, np.nan)
    result['heldout_position_rms'] = np.full(shape, np.nan)
    result['velocity_sigma'] = np.maximum(position_noise_floor_px, result['fit_rms']) / np.maximum(result['fit_span_s'], minimum_span_s)*np.sqrt(12.)
    result['recovery_attempted'] = np.zeros(shape, bool)
    for t, now in enumerate(times):
        ids = np.flatnonzero(good[t] & ~result['valid'][t])
        if not len(ids):
            continue
        lo = np.searchsorted(times, now-half_window_s-1.e-12)
        hi = np.searchsorted(times, now+half_window_s+1.e-12, side='right')
        dt = times[lo:hi]-now
        keep = good[lo:hi, ids] & (segments[lo:hi, ids] == segments[t, ids])
        count = keep.sum(0)
        keep_ids = count >= fallback_minimum_samples
        ids, keep = ids[keep_ids], keep[:, keep_ids]
        if not len(ids):
            continue
        begin = np.min(np.where(keep, dt[:, None], np.inf), axis=0)
        end = np.max(np.where(keep, dt[:, None], -np.inf), axis=0)
        span = end-begin
        radius = np.maximum(abs(begin), abs(end))
        inner = keep & (abs(dt[:, None]) <= .75*radius[None]+1.e-12)
        eligible = (span >= max(minimum_span_s, .05)) & (inner.sum(0) >= 5)
        ids, keep, inner, span = ids[eligible], keep[:, eligible], inner[:, eligible], span[eligible]
        if not len(ids):
            continue
        result['recovery_attempted'][t, ids] = True
        outer_fit = _quadratic(dt, y[lo:hi, ids], keep, position_noise_floor_px)
        inner_fit = _quadratic(dt, y[lo:hi, ids], inner, position_noise_floor_px)
        outer_holdout = _heldout_error(dt, y[lo:hi, ids], keep, position_noise_floor_px)
        inner_holdout = _heldout_error(dt, y[lo:hi, ids], inner, position_noise_floor_px)
        disagreement = np.linalg.norm(outer_fit['slope']-inner_fit['slope'], axis=1)
        noise = np.maximum(outer_fit['sigma'], position_noise_floor_px/span*np.sqrt(12.))
        inner_noise = np.maximum(inner_fit['sigma'], position_noise_floor_px/span*np.sqrt(12.))
        z = disagreement/np.maximum(np.hypot(noise, inner_noise), 1.e-12)
        accepted = ((outer_fit['condition'] < 1.e6) & (inner_fit['condition'] < 1.e6)
            & (outer_fit['rms'] <= maximum_fit_rms) & (inner_fit['rms'] <= maximum_fit_rms)
            & (outer_holdout <= maximum_fit_rms) & (inner_holdout <= maximum_fit_rms)
            & (z <= agreement_sigma) & np.isfinite(outer_fit['slope']).all(1))
        result['cross_scale_z'][t, ids] = z
        result['heldout_position_rms'][t, ids] = outer_holdout
        use = ids[accepted]
        result['slope'][t, use] = outer_fit['slope'][accepted]
        result['fit_rms'][t, use] = outer_fit['rms'][accepted]
        result['fit_span_s'][t, use] = span[accepted]
        result['sample_count'][t, use] = outer_fit['count'][accepted]
        result['valid'][t, use] = True
        result['source_kind'][t, use] = 2
        result['one_sided'][t, use] = ~(keep[dt < 0][:, accepted].any(0) & keep[dt > 0][:, accepted].any(0))
        result['velocity_sigma'][t, use] = np.maximum(noise[accepted], disagreement[accepted])
    result['velocity_sigma'][~result['valid']] = np.nan
    return result
