import numpy as np
import pytest
from mechanics.impulse_reference import advance,midpoint_stop_counterexample


def test_midpoint_nonnegative_work_misses_endpoint_reversal():
    r=midpoint_stop_counterexample(1.,1.,1.,3.)
    assert r['velocity']==-1. and r['friction_work']==0 and r['displacement']==0


@pytest.mark.parametrize('h',[.08,.04,.02])
def test_exact_stop_time_endpoint_impulse_work_and_position(h):
    m=.001;N=.02;mu=.35;x=np.zeros(2);v=np.array([1.,0.]);t=0.;rows=[]
    while t<.32-1e-14:
        r=advance(x,v,t,min(h,.32-t),mass=m,force=[0,-N],mu=mu);rows.append(r);x,v,t=r['position'],r['velocity'],r['time_s']
    stopping=[e['time_s'] for r in rows for e in r['events'] if e['kind']=='stick']
    assert stopping[0]==pytest.approx(m/(mu*N),abs=1e-12)
    np.testing.assert_allclose(v,0,atol=1e-14)
    assert x[0]==pytest.approx(m/(2*mu*N),abs=1e-12)
    assert sum(r['friction_dissipation_n_mm'] for r in rows)==pytest.approx(.5*m,abs=1e-14)
    np.testing.assert_allclose(sum(r['contact_impulse_n_s'] for r in rows),[-m,N*.32],atol=1e-13)
    assert max(np.linalg.norm(r['momentum_residual_n_s']) for r in rows)<1e-14
    assert max(abs(r['numerical_work_n_mm']) for r in rows)<1e-14


def test_moving_wall_impact_power_is_not_material_energy():
    r=advance([0.,0.],[1.,-.3],0.,.1,mass=.001,force=[0.,0.],wall_velocity=[.2,.05])
    pn=.001*.35;pt=-.35*pn
    np.testing.assert_allclose(r['contact_impulse_n_s'],[pt,pn],atol=1e-15)
    assert r['wall_work_n_mm']==pytest.approx(.2*pt+.05*pn)
    assert r['normal_impact_loss_n_mm']==pytest.approx(.5*.001*.35**2)
    assert abs(r['numerical_work_n_mm'])<1e-15
    assert r['minimum_gap_mm']>=-1e-12
    assert r['velocity'][1]==pytest.approx(.05)


def test_penetration_is_rejected_without_projection():
    with pytest.raises(ValueError,match='penetration'):advance([0,-.01],[0,0],0,.1,mass=1,force=[0,0])


def test_saved_c1_tiny_return_event_is_not_discarded_by_absolute_time_floor():
    # Exact node inputs from round7/new_negative_minimal, not a reduced mass.
    m=6.666666666666646e-11;f=np.array([.0012053823221488402,-.039566369797932543])
    v=np.array([-19.911961501423303,6.7762260825201945e-9]);h=2.0599365234373612e-5
    q=advance([4.763011904588151,0.],v,0.,h,mass=m,force=f,mu=0.)
    hit=-2*m*v[1]/f[1]
    assert 0<hit<1e-15
    assert next(e['time_s'] for e in q['events'] if e['kind']=='contact')==pytest.approx(hit,rel=1e-12,abs=1e-30)
    assert q['velocity'][1]==pytest.approx(0.,abs=1e-14)
    assert q['minimum_gap_mm']>=-1e-12
    assert np.linalg.norm(q['momentum_residual_n_s'])/h<1e-4
    assert abs(q['numerical_work_n_mm'])<1e-12


def test_positive_short_interval_is_actually_integrated():
    q=advance([0,1],[1,0],0,1e-16,mass=1,force=[0,0])
    assert q['position'][0]==1e-16
