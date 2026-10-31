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
