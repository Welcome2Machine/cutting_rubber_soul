import numpy as np
import pytest
from tools.event_resolution import measure,initial_spent
from test_event_resolution import policy,energy
import time

def test_cumulative_budget_does_not_reset_or_cancel_and_keeps_velocity_indicator():
    old=policy();q=dict(old,version='cumulative_budget_v1',step_budget_fraction=.01,remaining_budget_fraction=.1)
    args=(np.ones(1),np.zeros((1,2)),np.zeros((1,2)),np.zeros((1,2)),.01,[0.,0.],[0.,0.],energy())
    with pytest.raises(ValueError):measure(*args,q)
    a=measure(*args,q,spent=0.);b=measure(*args,q,spent=.0099)
    assert a['allowances']['absolute_numerical_n_mm']==pytest.approx(.0001)
    assert b['allowances']['absolute_numerical_n_mm']==pytest.approx(.00001)
    assert a['allowances']['velocity_estimate_sqrt_nmm']==measure(*args,old)['allowances']['velocity_estimate_sqrt_nmm']
    with pytest.raises(ValueError):initial_spent(dict(event_budget_origin_s=0),q,.5)
    assert initial_spent(dict(event_budget_origin_s=.5),q,.5)==0


def test_inherited_budget_binds_audited_parent_and_uses_full_absolute_prefix(tmp_path):
    import json
    from tools.frozen_scene import digest
    q=dict(policy(),version='cumulative_budget_v1',step_budget_fraction=.01,remaining_budget_fraction=.1)
    state=tmp_path/'state.npz';state.write_bytes(b'bound parent bytes')
    raw=dict(initial_time_s=0.,source_hashes={'kernel':'unchanged'},steps=[dict(time_s=.5,state_file=state.name,state_sha256=digest(state))])
    result=tmp_path/'result.json';result.write_text(json.dumps(raw))
    proof=dict(passed=True,result_sha256=digest(result),source_hashes=raw['source_hashes'],states=[dict(absolute_numerical_including_equation_work_n_mm=.003)])
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(proof))
    limits=dict(parent_state_index=0,event_budget_origin_s=0.,inherited_event_budget=dict(directory=str(tmp_path),audit_file=str(audit),result_sha256=digest(result),audit_sha256=digest(audit),state_sha256=digest(state)))
    assert initial_spent(limits,q,.5)==.003
    with pytest.raises(ValueError):initial_spent(limits,q,.4)
    state.write_bytes(b'wrong parent')
    with pytest.raises(ValueError):initial_spent(limits,q,.5)


def test_actual_event_localization_rejection_rolls_back(tmp_path):
    from test_coupled_cohesive import config
    from tools.run_coupled_growth import problem
    from tools.run_fracture_route import snapshot
    from tools.face_witness import accepted_fields,array_hash
    from tools import event_resolution as guard
    from prototype.zero_gap_cut import EnergyResolutionRequired
    from unittest.mock import patch
    c=dict(config(),coupled_control=dict(material_layer='maxwell',solve_tolerance_n=1e-7));p=problem(c)
    old=p.released.copy();x,row=p.step(p.rest,p.start,p.start+1e-4,time.monotonic()+20)
    s=snapshot(p,x,old,p.start+1e-4);s['blade_force_components_n']=[row[k] for k in ('reaction_normal_fz_n','reaction_friction_fz_n')]
    before=array_hash(accepted_fields(p,x));original=p.acceptance
    q=dict(policy(),event_bracket_s=1e-12)
    def wrapped(y,*args):
        rec=original(y,*args);guard.enforce(p,y,rec,s,q);return rec
    p.acceptance=wrapped
    with patch.object(guard,'locate_event',return_value=dict(changed=['top_locked'],passed=False)):
        with pytest.raises(EnergyResolutionRequired):p.step(x,float(s['depth_mm']),float(s['depth_mm'])+1e-4,time.monotonic()+20)
    assert array_hash(accepted_fields(p,x))==before
