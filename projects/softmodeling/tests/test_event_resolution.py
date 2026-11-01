import numpy as np
import pytest
from tools.event_resolution import measure,enforce
from tools.event_partition import Partition,verify
from release_dynamics import DEFECT_KEYS
from prototype.zero_gap_cut import EnergyResolutionRequired


def policy():return dict(version='endpoint_budget_v2',energy_scale_n_mm=1.,velocity_scale_sqrt_nmm=1.,blade_impulse_scale_n_s=1.,window_s=1.,energy_fraction=.01,velocity_fraction=.02,impulse_fraction=.01,reference_sha256='a'*64)
def energy(n=0.):return dict.fromkeys((*DEFECT_KEYS,'friction_time_defect_n_mm','algebraic_balance_residual_n_mm'),n)

def test_empirical_indicators_and_additive_absolute_energy_budget():
    e=energy();e['bulk_endpoint_defect_n_mm']=.0005;e['barrier_endpoint_defect_n_mm']=-.0005
    q=measure(np.array([1.]),np.array([[.1,0.]]),np.zeros((1,2)),np.array([[1.,0.]]),.1,[1.,0.],[1.,0.],e,policy())
    assert q['passed'] and q['values']['absolute_numerical_n_mm']==.001
    assert q['values']['velocity_estimate_sqrt_nmm']==0.
    assert not q['local_estimates_are_strict_bounds']
    e['barrier_endpoint_defect_n_mm']=-.001
    assert not measure(np.array([1.]),np.array([[.1,0.]]),np.zeros((1,2)),np.array([[1.,0.]]),.1,[1.,0.],[1.,0.],e,policy())['passed']


def test_velocity_and_normal_friction_impulse_not_hidden_by_energy_cancellation():
    q=measure(np.array([1.]),np.array([[1.,0.]]),np.zeros((1,2)),np.zeros((1,2)),.1,[1.,-1.],[0.,0.],energy(),policy())
    assert not q['passed'];assert q['values']['blade_impulse_estimate_n_s']>0.
    assert q['values']['velocity_estimate_sqrt_nmm']==.5


def test_recovery_requires_actual_error_evidence_and_replays():
    c=dict(maximum_bisections=12,recovery='original_physical_gates_only',maximum_step_growth='accepted_error_ratio_at_most_quarter')
    p=Partition([1.,2.],c);p.decision('rejected','physical_energy_resolution')
    p.decision('accepted',resolution_ratio=.8);p.decision('accepted',resolution_ratio=.8)
    assert p.pending[0][1]-p.pending[0][0]==.5
    p.decision('accepted',resolution_ratio=.1);p.decision('accepted',resolution_ratio=.1)
    offsets,complete=verify(p.saved(),dict(dt_s=[1.],window_s=2.,adaptive_dyadic=c),1.)
    assert complete and offsets.tolist()==[.5,1.,1.5,2.]


def test_actual_fe_resolution_rejection_rolls_back_all_committed_fields():
    from test_coupled_cohesive import config
    from tools.run_coupled_growth import problem
    from tools.run_fracture_route import snapshot
    from tools.face_witness import accepted_fields,array_hash
    import time
    c=dict(config(),coupled_control=dict(material_layer='maxwell',solve_tolerance_n=1e-7))
    p=problem(c);released=p.released.copy();x,rec=p.step(p.rest,p.start,p.start+1e-4,time.monotonic()+20)
    s=snapshot(p,x,released,p.start+1e-4);old=float(s['depth_mm']);s['blade_force_components_n']=[rec['reaction_normal_fz_n'],rec['reaction_friction_fz_n']]
    before=array_hash(accepted_fields(p,x));original=p.acceptance;control=policy();control['energy_fraction']=1e-30
    def guarded(y,*args):
        row=original(y,*args);enforce(p,y,row,s,control);return row
    p.acceptance=guarded
    with pytest.raises(EnergyResolutionRequired):p.step(x,old,old+1e-4,time.monotonic()+20)
    assert array_hash(accepted_fields(p,x))==before


def test_registered_shared_quiet_prefix_cannot_hide_later_large_steps():
    from tools.contact_time_schedule import requested_schedule
    q=dict(dt_s=[.1],window_s=1.,common_prefix=dict(through_s=.5,dt_s=.25),schedule_provenance='Bound shared pre-event prefix',schedules={'0.1':[.25,.5,.6,.7,.8,.9,1.]})
    a,meta=requested_schedule(q,.1);assert meta['nominal_step_scope']=='post-prefix segment'
    q['schedules']['0.1']=[.25,.5,.7,.8,.9,1.]
    with pytest.raises(ValueError):requested_schedule(q,.1)


def test_acceleration_jump_proxy_can_refine_without_a_false_fixed_error_floor():
    ratios=[]
    for h in (.01,.001,.0001):
        q=measure(np.array([1.]),np.array([[h,0.]]),np.zeros((1,2)),np.zeros((1,2)),h,[0.,0.],[0.,0.],energy(),policy())
        ratios.append(q['ratios']['velocity_estimate_sqrt_nmm'])
    assert ratios[0]>ratios[1]>ratios[2] and ratios[2]<1
