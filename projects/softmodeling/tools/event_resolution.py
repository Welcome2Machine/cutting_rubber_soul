"""Prospective time-resolution budgets, never changes physical forces/history.

Energy is an additive absolute ledger. Velocity/impulse indicators use successive
accepted endpoint accelerations/forces and are empirical local BE estimates, not
error bounds. Independent common-time comparisons remain mandatory.
"""
import numpy as np
from release_dynamics import DEFECT_KEYS
from prototype.zero_gap_cut import EnergyResolutionRequired


def absolute_work(energy):
    return sum(abs(energy[k]) for k in (*DEFECT_KEYS,'friction_time_defect_n_mm','algebraic_balance_residual_n_mm'))


def initial_spent(limits,q,time_s):
    if q is None or q['version']=='endpoint_budget_v2':return 0.
    origin=limits.get('event_budget_origin_s',float('inf'))
    if abs(origin-time_s)<=1e-12:
        if 'inherited_event_budget' in limits:raise ValueError('Unexpected prefix at budget origin')
        return 0.
    from pathlib import Path
    import json
    from .frozen_scene import digest
    ref=limits.get('inherited_event_budget')
    if ref is None:raise ValueError('Cumulative hotspot needs its bound accepted prefix')
    directory=Path(ref['directory']);audit=Path(ref['audit_file'])
    if digest(directory/'result.json')!=ref['result_sha256'] or digest(audit)!=ref['audit_sha256']:
        raise ValueError('Changed budget prefix evidence')
    raw=json.loads((directory/'result.json').read_text());proof=json.loads(audit.read_text());i=limits['parent_state_index']
    if (not proof['passed'] or proof['result_sha256']!=ref['result_sha256'] or proof['source_hashes']!=raw['source_hashes']
        or len(proof['states'])!=len(raw['steps']) or abs(raw['initial_time_s']-origin)>1e-12):
        raise ValueError('Unverified full budget prefix')
    row=raw['steps'][i]
    if abs(row['time_s']-time_s)>1e-12 or row['state_sha256']!=ref['state_sha256'] or digest(directory/row['state_file'])!=ref['state_sha256']:
        raise ValueError('Budget prefix endpoint mismatch')
    return sum(s['absolute_numerical_including_equation_work_n_mm'] for s in proof['states'][:i+1])


def validate(q):
    required={'version','energy_scale_n_mm','velocity_scale_sqrt_nmm','blade_impulse_scale_n_s','window_s','energy_fraction','velocity_fraction','impulse_fraction','reference_sha256'}
    extra={'step_budget_fraction','remaining_budget_fraction'} if q.get('version')=='cumulative_budget_v1' else set()
    if q.get('version') not in ('endpoint_budget_v2','cumulative_budget_v1') or set(q)-{'event_bracket_s'}!=required|extra:raise ValueError('Unregistered event resolution policy')
    for k in required-{'version','reference_sha256'}:
        if not np.isfinite(q[k]) or q[k]<=0:raise ValueError('Invalid event budget')
    if len(q['reference_sha256'])!=64:raise ValueError('Unbound event reference')
    for k in extra:
        if not np.isfinite(q[k]) or not 0<q[k]<1:raise ValueError('Invalid cumulative allocation fraction')
    if 'event_bracket_s' in q and (not np.isfinite(q['event_bracket_s']) or q['event_bracket_s']<=0):raise ValueError('Invalid event bracket')
    return q.copy()


def measure(mass,velocity,old_velocity,old_acceleration,h,force,old_force,energy,q,*,spent=None):
    q=validate(q);mass=np.asarray(mass);h=float(h)
    if not np.isfinite(h) or h<=0:raise ValueError('Positive physical dt required')
    acceleration=(velocity-old_velocity)/h
    # BE endpoint accelerations, no Newton iteration or intermediate states.
    dv=.5*h*(acceleration-old_acceleration)
    velocity_estimate=float(np.sqrt(mass@np.sum(dv*dv,axis=1)))
    impulse_estimate=.5*h*float(np.linalg.norm(np.asarray(force)-np.asarray(old_force)))
    numerical=absolute_work(energy)
    values=dict(absolute_numerical_n_mm=numerical,velocity_estimate_sqrt_nmm=velocity_estimate,blade_impulse_estimate_n_s=impulse_estimate)
    allowances=dict(absolute_numerical_n_mm=q['energy_fraction']*q['energy_scale_n_mm']*h/q['window_s'],velocity_estimate_sqrt_nmm=q['velocity_fraction']*q['velocity_scale_sqrt_nmm']*np.sqrt(h/q['window_s']),blade_impulse_estimate_n_s=q['impulse_fraction']*q['blade_impulse_scale_n_s']*np.sqrt(h/q['window_s']))
    ledger=None
    if q['version']=='cumulative_budget_v1':
        total=q['energy_fraction']*q['energy_scale_n_mm']
        if spent is None or not np.isfinite(spent) or not 0<=spent<total:raise ValueError('Bound accepted-prefix numerical expenditure required')
        remaining=total-spent
        allowances['absolute_numerical_n_mm']=min(total*q['step_budget_fraction'],remaining*q['remaining_budget_fraction'])
        ledger=dict(total_n_mm=total,spent_before_n_mm=spent,remaining_before_n_mm=remaining,
                    candidate_spent_after_n_mm=spent+numerical,step_cap_n_mm=total*q['step_budget_fraction'])
    ratios={k:values[k]/allowances[k] for k in values}
    return dict(values=values,allowances=allowances,ratios=ratios,maximum_ratio=max(ratios.values()),passed=all(v<=1 for v in ratios.values()),physical_dt_s=h,local_estimates_are_strict_bounds=False,ledger=ledger)


def event_signature(p,x,history,active_faces,active_contact):
    from .audit_precrack_entry import strict_contacts
    from .report_coupled_growth import face_contacts
    failed=active_faces & np.all(history>=2*p.toughness/p.strength,axis=1)
    return dict(top_locked=bool(active_contact[p.seam[-1]]),failed_faces=np.flatnonzero(failed).tolist(),
        A=bool(strict_contacts(p,x,p.connected_length(failed))),
        B=any(c['strict_face_interior'] for c in face_contacts(p,x,failed)))


def locate_event(p,x,previous,tolerance):
    # Old geometry must use its old exposed topology. Separate objects are
    # deliberately unnecessary: restore the live graph in a finally clause.
    new=event_signature(p,x,p.cohesive_response(x)['history'],p.active_faces,p.active_contact)
    saved=(p.active_faces.copy(),p.released.copy())
    try:
        p.active_faces=previous['active_faces'].copy();p.released=previous['released'].copy();p.rebuild_contact()
        old=event_signature(p,previous['positions'],previous['history'],p.active_faces,previous['active_contact'])
    finally:
        p.active_faces,p.released=saved;p.rebuild_contact();p.set_free()
    changed=[k for k in old if old[k]!=new[k]]
    return dict(before=old,after=new,changed=changed,bracket_width_s=float(p.material_dt),
                tolerance_s=tolerance,passed=not changed or p.material_dt<=tolerance,
                event_time_error_bound=False)


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
        [rec['reaction_normal_fz_n'],rec['reaction_friction_fz_n']],previous['blade_force_components_n'],rec['energy'],q,
        spent=previous.get('absolute_numerical_spent_n_mm'))
    if 'event_bracket_s' in q:
        event=locate_event(p,x,previous,q['event_bracket_s']);check['event_location']=event
        check['values']['event_bracket_s']=p.material_dt if event['changed'] else 0.
        check['allowances']['event_bracket_s']=q['event_bracket_s']
        check['ratios']['event_bracket_s']=check['values']['event_bracket_s']/q['event_bracket_s']
        check['passed'] &= event['passed'];check['maximum_ratio']=max(check['ratios'].values())
    rec['event_resolution']=check
    if not check['passed']:
        key=max(check['ratios'],key=check['ratios'].get)
        error=EnergyResolutionRequired('Registered event time budget exceeded: '+key)
        error.failure_measurement=dict(quantity='event_'+key,value=check['values'][key],limit=check['allowances'][key],units={'absolute_numerical_n_mm':'N mm','velocity_estimate_sqrt_nmm':'sqrt(N mm)','blade_impulse_estimate_n_s':'N s','event_bracket_s':'s'}[key],comparison='upper')
        raise error
    return check
