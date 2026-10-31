import time
import numpy as np
import pytest
from test_precrack_entry import cfg
from coupled_contact import problem
from tools.face_witness import accepted_fields,array_hash


def test_endpoint_bulk_tangent_and_force_are_updated_in_newton():
    p=problem(cfg(.2));p.material_dt=.001
    x=p.rest.copy();x[:p.n,1]*=.9999
    u=(x[:p.n]-p.mesh.reference_xy).ravel();r,k,dofs,_,_=p.material_bulk(u)
    rng=np.random.default_rng(36);d=rng.normal(size=u.shape);eps=1e-6
    rp=p.material_bulk(u+eps*d,with_tangent=False)[0];rm=p.material_bulk(u-eps*d,with_tangent=False)[0]
    fd=(rp-rm)/(2*eps);analytic=np.einsum('eij,ej->ei',k,d[dofs])
    assert np.linalg.norm(fd-analytic)<=1e-7*np.linalg.norm(analytic)
    assert np.linalg.norm(r)>1e-3


@pytest.mark.parametrize('layer',['frozen_metric','maxwell'])
def test_coupled_free_step_kinematics_history_and_energy(layer):
    p=problem(cfg(.2),layer);old=p.rest.copy();metric=p.material_metric.copy()
    x,r=p.step(old,p.start,p.start+1e-4,time.monotonic()+20)
    np.testing.assert_allclose(p.step_increment(x)[:p.n],p.material_dt*p.velocity[:p.n],rtol=0,atol=1e-15)
    assert r['residual_n']<=1e-4 and r['accepted_chord_ccd_safe']
    assert r['energy']['friction_time_defect_n_mm']==0
    assert abs(r['energy']['algebraic_balance_residual_n_mm'])<1e-10
    if layer=='frozen_metric':np.testing.assert_array_equal(p.material_metric,metric)


def test_failed_coupled_step_keeps_all_committed_history(monkeypatch):
    p=problem(cfg(.2));x=p.rest.copy();before=array_hash(accepted_fields(p,x))
    def fail(*args):raise TimeoutError('controlled computation delay')
    monkeypatch.setattr(p,'solve_active',fail)
    with pytest.raises(TimeoutError):p.step(x,p.start,p.start+1e-4,time.monotonic()+1)
    assert array_hash(accepted_fields(p,x))==before


@pytest.mark.parametrize('layer',['frozen_metric','maxwell'])
def test_full_coupled_contact_jacobian_includes_geometry_and_material(layer):
    p=problem(cfg(.2),layer);p.material_dt=.0005
    p.previous[p.n:]=p.prescribed_blade_positions(.3581)
    x=p.previous.copy();x[p.n:]=p.prescribed_blade_positions(.3582)
    p.friction=p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
    p.set_primary_increment(x,x-p.previous)
    lag=p.lag(x);_,g,H,_=p.evaluate(x,lag,projected=False)
    assert np.linalg.norm(p.normal_response(x)[0])>0
    J=p.coupled_friction_tangent(x,lag,H)
    rng=np.random.default_rng(19);d=np.zeros(p.total);d[p.free]=rng.normal(size=len(p.free));d=d.reshape(x.shape)
    eps=1e-8
    gp=p.evaluate(x+eps*d,p.lag(x+eps*d),tangent=False)[1]
    gm=p.evaluate(x-eps*d,p.lag(x-eps*d),tangent=False)[1]
    fd=(gp-gm)/(2*eps);analytic=J@d.ravel()
    assert np.linalg.norm(fd-analytic)<=2e-5*np.linalg.norm(analytic)


def test_solver_target_can_be_tightened_but_acceptance_cannot_be_relaxed():
    p=problem(cfg(.2),'frozen_metric',1e-7)
    assert p.force_tolerance_n==pytest.approx(1e-7/np.sqrt(2))
    with pytest.raises(ValueError,match='tightened'):problem(cfg(.2),'frozen_metric',1.001e-4)
