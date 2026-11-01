"""Prospective time-resolution budgets, never changes physical forces/history.

Energy is an additive absolute ledger. Velocity/impulse indicators use successive
accepted endpoint accelerations/forces and are empirical local BE estimates, not
error bounds. Independent common-time comparisons remain mandatory.
"""
import numpy as np
from release_dynamics import DEFECT_KEYS
from prototype.zero_gap_cut import EnergyResolutionRequired


def validate(q):
    required={'version','energy_scale_n_mm','velocity_scale_sqrt_nmm','blade_impulse_scale_n_s','window_s','energy_fraction','velocity_fraction','impulse_fraction','reference_sha256'}
    if set(q)!=required or q['version']!='endpoint_budget_v2':raise ValueError('Unregistered event resolution policy')
    for k in required-{'version','reference_sha256'}:
        if not np.isfinite(q[k]) or q[k]<=0:raise ValueError('Invalid event budget')
    if len(q['reference_sha256'])!=64:raise ValueError('Unbound event reference')
    return q.copy()


def measure(mass,velocity,old_velocity,old_acceleration,h,force,old_force,energy,q):
    q=validate(q);mass=np.asarray(mass);h=float(h)
    if not np.isfinite(h) or h<=0:raise ValueError('Positive physical dt required')
    acceleration=(velocity-old_velocity)/h
    # BE endpoint accelerations, no Newton iteration or intermediate states.
    dv=.5*h*(acceleration-old_acceleration)
    velocity_estimate=float(np.sqrt(mass@np.sum(dv*dv,axis=1)))
    impulse_estimate=.5*h*float(np.linalg.norm(np.asarray(force)-np.asarray(old_force)))
    numerical=sum(abs(energy[k]) for k in (*DEFECT_KEYS,'friction_time_defect_n_mm','algebraic_balance_residual_n_mm'))
    values=dict(absolute_numerical_n_mm=numerical,velocity_estimate_sqrt_nmm=velocity_estimate,blade_impulse_estimate_n_s=impulse_estimate)
    allowances=dict(absolute_numerical_n_mm=q['energy_fraction']*q['energy_scale_n_mm']*h/q['window_s'],velocity_estimate_sqrt_nmm=q['velocity_fraction']*q['velocity_scale_sqrt_nmm']*np.sqrt(h/q['window_s']),blade_impulse_estimate_n_s=q['impulse_fraction']*q['blade_impulse_scale_n_s']*np.sqrt(h/q['window_s']))
    ratios={k:values[k]/allowances[k] for k in values}
    return dict(values=values,allowances=allowances,ratios=ratios,maximum_ratio=max(ratios.values()),passed=all(v<=1 for v in ratios.values()),physical_dt_s=h,local_estimates_are_strict_bounds=False)


def bind_reference(limits,dt):
    from pathlib import Path
    import json
    from .frozen_scene import digest
    q=validate(limits['event_resolution'][str(dt)]);ref=limits['event_control_reference']
    if digest(Path(ref['file']))!=q['reference_sha256'] or ref['sha256']!=q['reference_sha256']:raise ValueError('Event control reference changed')
    data=json.loads(Path(ref['file']).read_text())
    expected=dict(energy_scale_n_mm=data['event_energy_reference']['value_n_mm'],velocity_scale_sqrt_nmm=data['velocity_scale_sqrt_nmm'],blade_impulse_scale_n_s=data['blade_impulse_scale_n_s'])
    if any(q[k]!=v for k,v in expected.items()):raise ValueError('Event control scales do not match reference')
    return q


def enforce(p,x,rec,previous,q):
    old_acceleration=(previous['velocity'][:p.n]-previous['previous_velocity'][:p.n])/float(previous['material_dt_s'])
    check=measure(p.mass,p.endpoint_velocity(x)[:p.n],p.previous_velocity[:p.n],old_acceleration,p.material_dt,
        [rec['reaction_normal_fz_n'],rec['reaction_friction_fz_n']],previous['blade_force_components_n'],rec['energy'],q)
    rec['event_resolution']=check
    if not check['passed']:
        key=max(check['ratios'],key=check['ratios'].get)
        error=EnergyResolutionRequired('Registered event time budget exceeded: '+key)
        error.failure_measurement=dict(quantity='event_'+key,value=check['values'][key],limit=check['allowances'][key],units={'absolute_numerical_n_mm':'N mm','velocity_estimate_sqrt_nmm':'sqrt(N mm)','blade_impulse_estimate_n_s':'N s'}[key],comparison='upper')
        raise error
    return check
