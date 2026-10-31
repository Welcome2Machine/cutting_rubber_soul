import time
import numpy as np
import pytest
from test_face_cut import config
from tools.run_endpoint_face_route import problem
from tools.face_witness import WitnessStore,qualification


def test_qualifications_do_not_promote_precrack_or_model_activation():
    rows=[dict(surface_connected_failed_length_mm=.5,released_nodes=5,blade_entered_surface_channel=False)]
    q=qualification(rows,.5)
    assert q['preexisting_connected_length_mm']==.5 and q['new_connected_failed_length_mm']==0
    assert not q['model_activation_occurred'] and not q['physical_initiation_prediction_validated']
    rows[0]['face_events']=[{}];q=qualification(rows,.5)
    assert q['model_activation_occurred'] and not q['activation_witness_verified']
    assert not q['blade_contacted_strict_new_face']


def test_pure_precrack_preserves_existing_failure_and_exact_intact_faces():
    c=config(.5,'endpoint_friction');c['pure_precrack']=True;p=problem(c)
    history=p.history.copy();x=p.rest.copy();x[p.seam[-1],0]+=.2
    np.testing.assert_array_equal(p.cohesive_response(x)['history'],history)
    assert not p.propose_insertion(x,None,0,time.monotonic()+1).any()
    assert not p.released[np.flatnonzero(p.rest[p.seam,1]==4.5)[0]]
    p.previous=x.copy()
    q=p.cohesive_algorithmic(x)
    np.testing.assert_allclose(q['traction'][p.precracked_faces],0.,atol=1e-14)
    np.testing.assert_array_equal(p.previous_history,history)


def test_failure_store_captures_actual_attempt_and_exact_rollback(tmp_path,monkeypatch):
    c=config(.5,'endpoint_friction');p=problem(c);store=WitnessStore(tmp_path,dict(config=c),max_signatures=1)
    x=p.rest.copy();store.begin(p,x,dict(parent={'file':'initial.npz','sha256':'test'},physical_dt_s=.0005))
    def fail(z,*args):
        z=z.copy();z[p.free[0]//2,p.free[0]%2]+=1e-7;p.set_primary_increment(z,z-p.previous)
        raise TimeoutError('controlled delay')
    monkeypatch.setattr(p,'solve_active',fail)
    with pytest.raises(TimeoutError):p.step(x,p.start,p.start+.0001,time.monotonic()+1)
    result=store.rejected(p,x,TimeoutError('controlled delay'),'nonlinear_budget')
    assert result['rollback_verified']
    path=next((tmp_path/'witnesses/rejected').glob('*.npz'))
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['positions'],p.failed_positions)
        assert float(z['material_dt_s'])==pytest.approx(.0005)
        assert not np.array_equal(z['positions'],x)


def test_ffm_cannot_mix_secant_virtual_potential_with_constant_traction():
    from test_energy_cut import candidate
    from tools.run_zero_gap_cut import problem as ffm
    c=candidate();c.update(fracture_criterion='finite_increment',time_integrator='backward_euler',cohesive_unloading='constant_traction')
    for key in ('kinematic_theta','barrier_quadrature','barrier_quadrature_rule','insertion_rule'):c.pop(key,None)
    with pytest.raises(ValueError,match='secant'):ffm(c)


def test_post_activation_trial_restores_released_graph_and_dofs():
    from face_cut import eligible_nodes
    from tools.face_witness import trial_fields
    from tools.audit_face_witness import restore
    c=config(.5,'endpoint_friction');p=problem(c)
    p.material_dt=.001;p.previous_barrier_energy=0.
    p.friction=p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
    for faces in (slice(44,46),slice(43,45)):
        p.previous_active_faces=p.active_faces.copy()
        p.active_faces[faces]=True;p.released=eligible_nodes(p.active_faces)
        p.rebuild_contact();p.active_contact[:]=False;p.active_contact[p.seam[~p.released]]=True;p.set_free()
        x=p.rest.copy();p.set_primary_increment(x,x-p.previous)
        s=trial_fields(p,x);q=restore(c,s)
        np.testing.assert_array_equal(q.released,p.released)
        np.testing.assert_array_equal(q.free,p.free)
        np.testing.assert_array_equal(q.surface,p.surface)
        s['released']=s['released'].copy();s['released'][44]=~s['released'][44]
        with pytest.raises(AssertionError):restore(c,s)


def test_first_worst_latest_retains_energy_free_middle_maximum(tmp_path):
    import json
    from tools.face_witness import trial_fields
    from prototype.zero_gap_cut import EnergyResolutionRequired
    c=config(.5,'endpoint_friction');p=problem(c);x=p.rest.copy()
    p.material_dt=.001;p.previous_barrier_energy=0.
    store=WitnessStore(tmp_path,dict(config=c),max_signatures=1)
    for value in (1.,100.,2.):
        store.begin(p,x,dict(sequence=value));store.trial=(trial_fields(p,x),dict(stage='acceptance'))
        e=EnergyResolutionRequired(f'Traction excess {value} N')
        e.failure_measurement=dict(quantity='endpoint_traction_excess_half_n',value=value,limit=p.force_tolerance_n,units='N',comparison='upper')
        store.rejected(p,x,e,'physical_resolution')
    item=next(iter(store.signatures.values()))
    for key,value in [('first',1),('worst',100),('latest',2)]:
        with np.load(tmp_path/item[key]['file'],allow_pickle=False) as z:m=json.loads(str(z['metadata_json']))
        assert m['rejection_measurement']['value']==value
    assert len(list((tmp_path/'witnesses/rejected').glob('*.npz')))==3


def test_unmeasured_timeout_is_not_claimed_worst(tmp_path):
    from tools.face_witness import trial_fields
    c=config(.5,'endpoint_friction');p=problem(c);x=p.rest.copy()
    p.material_dt=.001;p.previous_barrier_energy=0.
    store=WitnessStore(tmp_path,dict(config=c));store.begin(p,x,{})
    store.trial=(trial_fields(p,x),dict(stage='force_solve'))
    store.rejected(p,x,TimeoutError('deadline'),'compute_budget')
    item=next(iter(store.signatures.values()))
    assert item['worst'] is None and item['worst_status']=='not_comparable'
