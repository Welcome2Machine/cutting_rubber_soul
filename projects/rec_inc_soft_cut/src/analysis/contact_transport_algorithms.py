"""Shared packing, conservative model mixing and contact temporal fusion.

This module contains numerical operations only; it has no case/output paths
and importing it never starts processing.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from .contact_transport_reference import smooth_contact_transport
from .contact_transport_observations import _robust_center


def legacy_qualification(record):
    """Frozen MMT-v1 compatibility, including its historical early return.

    This reproduces a known legacy qualification defect for comparison only.
    OBT-v1 must never call this function. Keeping it explicit avoids silently
    changing the method the user requested to retain.
    """
    from .boundary_contact_transport import BoundaryContactTransportObserver
    result = deepcopy(record)
    for models in result['models']:
        for model in models:
            if any(reason in model.get('reasons', []) for reason in (
                    'insufficient_continuous_fixed_id_velocity_support',
                    'ill_conditioned_velocity_model')):
                model['absolute_valid'] = bool(model['position_absolute_compatible'])
    result['sides'] = [BoundaryContactTransportObserver._combine(i+1, models)
                       for i, models in enumerate(result['models'])]
    absolute = [side for side in result['sides'] if side['absolute_valid']]
    result['absolute_valid'] = bool(absolute)
    if absolute:
        result['absolute_s_px'], result['absolute_sigma_px'], _ = _robust_center(
            [side['absolute_s_px'] for side in absolute],
            [side['absolute_sigma_px'] for side in absolute])
    return result


def fixed_parameter_fusion(data, observations, acceleration, initial_velocity_sigma=None):
    """Shared-bias fusion with explicitly fixed process parameters."""
    times = data['times']-data['times'][0]
    correlation = np.sqrt(max(1., .15/np.median(np.diff(times))))
    return smooth_contact_transport(
        times=times, initial_s=float(data['initial']['contact_s_px']), initial_sigma=2.,
        side_flux=observations['side_flux_px_s'],
        side_sigma=observations['side_flux_sigma_px_s']*correlation,
        side_valid=observations['side_flux_valid'],
        absolute_s=observations['absolute_s_px'],
        absolute_sigma=observations['absolute_sigma_px']*correlation,
        absolute_valid=observations['absolute_valid'], acceleration_sigma=acceleration,
        initial_velocity_sigma=initial_velocity_sigma)

def pack_records(records):
    out = {}
    aliases = dict(lambda_t='surface_stretch',lambda_perp='normal_compression',shear='normal_shear',
        normal_residual_px_s='normal_relative_velocity_px_s',surface_gap_px='surface_tip_gap_px',
        heldout_rms_px='position_holdout_px',minimum_depth_px='minimum_initial_depth_px',
        compression='high_compression')
    # Side dictionaries have an explicit common schema defined by the observer.
    for name in ('flux_px_s', 'flux_sigma_px_s', 'flux_valid', 'lambda_t', 'lambda_perp',
                 'shear', 'normal_residual_px_s', 'surface_gap_px', 'position_rms_px',
                 'heldout_rms_px', 'leverage', 'support_count', 'minimum_depth_px',
                 'compression', 'reason', 'surface_xy', 'tangent_xy'):
        default = False if name in ('flux_valid', 'compression') else '' if name == 'reason' else [np.nan,np.nan] if name in ('surface_xy','tangent_xy') else np.nan
        if name=='support_count':
            out['side_'+name] = np.asarray([[len(side.get('support_ids',[])) for side in r['sides']] for r in records])
        elif name=='reason':
            out['side_'+name] = np.asarray([[';'.join(side.get('reasons',[])) for side in r['sides']] for r in records])
        else:
            out['side_'+name] = np.asarray([[side.get(aliases.get(name,name), default) for side in record['sides']] for record in records])
    for name in ('absolute_s_px', 'absolute_sigma_px', 'absolute_valid'):
        out[name] = np.asarray([r.get(name, False if name.endswith('valid') else np.nan) for r in records])
    ids = [np.unique(np.concatenate([np.asarray(s.get('support_ids',[]),int) for s in r['sides']])) for r in records]
    out['support_id_offsets'] = np.r_[0,np.cumsum([len(v) for v in ids])]
    out['support_ids_flat'] = np.concatenate(ids)
    out['side_observation_model']=np.asarray([[s.get('observation_model','cartesian') for s in r['sides']] for r in records])
    return out


def conservative(values,sigmas):
    values,sigmas=np.asarray(values,float),np.asarray(sigmas,float)
    w=1./np.maximum(sigmas,1e-6)**2
    w=np.minimum(w,np.median(w)*4.)
    mean=float(np.average(values,weights=w))
    spread=float(np.sqrt(np.average((values-mean)**2,weights=w)))
    # These use the same tracks. Repeated models do not create new evidence.
    return mean,float(np.hypot(np.max(sigmas),spread))


def combine_models(cartesian,curved):
    sides=[]
    for a,b in zip(cartesian['sides'],curved['sides']):
        good=[v for v in (a,b) if v['flux_valid']]
        chosen=min(good or [a,b],key=lambda v:v.get('flux_sigma_px_s',np.inf) if np.isfinite(v.get('flux_sigma_px_s',np.nan)) else np.inf)
        s=dict(chosen)
        s['observation_model']='cartesian+curved' if len(good)==2 else 'cartesian' if chosen is a else 'curved'
        s['flux_valid']=bool(good)
        if good:
            s['flux_px_s'],s['flux_sigma_px_s']=conservative([v['flux_px_s'] for v in good],[v['flux_sigma_px_s'] for v in good])
        # Cartesian layer geometry remains a separate diagnostic; a curved
        # scalar-coordinate model cannot measure normal compression itself.
        s['cartesian_compression_valid']=bool(a['flux_valid'])
        for key in ('normal_compression','normal_shear','high_compression'):
            s['cartesian_diagnostic_'+key]=a.get(key,False if key=='high_compression' else np.nan)
            s[key]=a.get(key,False if key=='high_compression' else np.nan) if a['flux_valid'] else (False if key=='high_compression' else np.nan)
        s['support_ids']=np.unique(np.concatenate([np.asarray(v.get('support_ids',[]),int) for v in (good or [chosen])]))
        absolute=[v for v in (a,b) if v.get('absolute_valid',False)]
        s['absolute_valid']=bool(absolute)
        if absolute:
            s['absolute_s_px'],s['absolute_sigma_px']=conservative([v['absolute_s_px'] for v in absolute],[v['absolute_sigma_px'] for v in absolute])
            # Visible-surrogate validation shows bias can survive fit checks.
            # Use a depth-dependent model discrepancy floor, not a slip cap.
            shallow=min(v.get('minimum_initial_depth_px',0.) for v in absolute)
            discrepancy=2.+.15*max(0.,float(shallow))
            s['absolute_sigma_px']=float(np.hypot(s['absolute_sigma_px'],discrepancy))
        sides.append(s)
    result=dict(sides=sides)
    absolute=[s for s in sides if s.get('absolute_valid',False)]
    result['absolute_valid']=bool(absolute)
    result['absolute_s_px'],result['absolute_sigma_px']=(conservative([s['absolute_s_px'] for s in absolute],[s['absolute_sigma_px'] for s in absolute]) if absolute else (np.nan,np.nan))
    return result


def acceleration_scale(times, flux, valid, sigma):
    q = np.full(len(times), np.nan)
    for i in range(len(q)):
        use = valid[i] & np.isfinite(flux[i])
        if use.any(): q[i] = np.median(flux[i,use])
    ok = np.isfinite(q[1:]) & np.isfinite(q[:-1])
    dt = np.diff(times)
    changes = np.abs(np.diff(q)[ok])/np.sqrt(dt[ok])
    noise = sigma[valid & np.isfinite(sigma)]
    # Spectrum scale is estimated from the actual observed changes, never a
    # desired slip curve. The second term prevents overconfidence in gaps.
    empiric = float(np.median(changes)*1.4826) if len(changes) else 0.
    floor = float(np.median(noise)) / np.sqrt(.15) if len(noise) else 20.
    return max(empiric, floor, 1.)


def solve(data, obs, noise_multiplier=1.):
    times = data['times']-data['times'][0]
    # Windows overlap. Inflate observation noise for their correlation instead
    # of treating each rendered frame as independent evidence.
    correlation = np.sqrt(max(1., .15/float(np.median(np.diff(times)))))
    flux, sigma, valid = obs['side_flux_px_s'], obs['side_flux_sigma_px_s'], obs['side_flux_valid'].astype(bool)
    accel = acceleration_scale(times, flux, valid, sigma)*noise_multiplier
    kwargs = dict(times=times, initial_s=float(data['initial']['contact_s_px']), initial_sigma=2.,
        side_flux=flux, side_sigma=sigma*correlation, side_valid=valid,
        absolute_s=obs['absolute_s_px'], absolute_sigma=obs['absolute_sigma_px']*correlation,
        absolute_valid=obs['absolute_valid'].astype(bool), acceleration_sigma=accel)
    result = smooth_contact_transport(**kwargs)
    return result, dict(acceleration_sigma=accel, observation_correlation_factor=correlation,
                       velocity_window_s=.15, process_model='constant velocity with white acceleration; no monotonicity or sticking constraint')
