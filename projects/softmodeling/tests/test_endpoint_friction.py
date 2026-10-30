import time
import numpy as np
import pytest
from test_energy_cut import candidate
from test_stiff_contact import contact_fixture
from tools.run_zero_gap_cut import problem, snapshot, audit
from endpoint_friction import velocity_increment
from stiff_contact import install as install_method, momentum_ledger as impulse_ledger

def install(p):
    return install_method(p, "endpoint_friction")


def prepared():
    p,x=contact_fixture();install(p)
    p.material_dt=1e-4;p.previous=x.copy();p.previous[p.n:,1]+=1e-5
    p.previous_velocity=np.zeros_like(x)
    p.friction=p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
    delta=np.zeros(p.total);delta[p.free]=1e-6
    x+=delta.reshape(-1,2);p.set_primary_increment(x,x-p.previous)
    p.previous_barrier_energy=float(p.barrier(p.collisions(p.previous),p.cmesh,p.previous[p.surface]))
    return p,x


def test_full_endpoint_velocity_friction_jacobian_and_separate_work_identity():
    p,x=prepared();lag=p.lag(x)
    _,g,H,_=p.evaluate(x,lag,tangent=True,projected=False)
    H=p.coupled_friction_tangent(x,lag,H)
    d=np.zeros(p.total);d[p.free]=np.random.default_rng(506).normal(size=len(p.free));d/=np.linalg.norm(d)
    eps=1e-7;xp=x+eps*d.reshape(-1,2);xm=x-eps*d.reshape(-1,2)
    fd=(p.evaluate(xp,p.lag(xp),tangent=False)[1]-p.evaluate(xm,p.lag(xm),tangent=False)[1])/(2*eps)
    assert np.linalg.norm(fd-H@d)/np.linalg.norm(fd)<2e-4
    e=p.energy_ledger(x);du=p.step_increment(x)
    expected=-2*float(g[:2*p.n]@du[:p.n].ravel())
    assert e['algebraic_balance_residual_n_mm']==pytest.approx(expected,abs=3e-10)
    assert e['backward_euler_inertia_loss_n_mm']==0
    assert e['friction_work_n_mm']>=0
    assert e['friction_algorithmic_work_n_mm']-e['friction_work_n_mm']==pytest.approx(e['friction_time_defect_n_mm'])


def test_prescribed_blade_velocity_not_doubled_by_midpoint_formula():
    p,x=prepared();du=p.step_increment(x);dv=velocity_increment(p,du)
    np.testing.assert_array_equal(dv[p.n:],du[p.n:])
    np.testing.assert_allclose(dv[:p.n],p.material_dt*p.endpoint_velocity(x)[:p.n])


def test_free_complete_step_keeps_original_inertia_and_audits():
    cfg=dict(candidate(),energy_tolerance=.00025,energy_normalization='mechanical_reservoir',energy_absolute_tolerance_n_mm=1e-7)
    p=install(problem(cfg));old=p.released.copy()
    x,rec=p.step(p.rest.copy(),p.start,-.015,time.monotonic()+20)
    assert audit(install(problem(cfg)),snapshot(p,x,old,-.015),rec)['passed']
    _,impulse=impulse_ledger(p,x)
    assert impulse['equivalent_force_residual_n']<=1e-4
    assert impulse['numerical_kinetic_loss_n_mm']==0


def test_additional_friction_time_defect_cannot_escape_original_energy_gate(monkeypatch):
    from endpoint_friction import acceptance
    from prototype.energy_cut import EnergyCutProblem
    from prototype.zero_gap_cut import EnergyResolutionRequired
    from release_dynamics import DEFECT_KEYS
    p=install(problem(dict(candidate(),energy_tolerance=.001,energy_normalization='mechanical_reservoir')))
    e={k:0. for k in DEFECT_KEYS}
    e.update(mechanical_reservoir_n_mm=1.,friction_work_n_mm=.002,
        friction_algorithmic_work_n_mm=.00300001,friction_time_defect_n_mm=.00100001)
    monkeypatch.setattr(EnergyCutProblem,'acceptance',lambda *a:dict(energy=e))
    with pytest.raises(EnergyResolutionRequired):acceptance(p,p.rest,0.,0,0)


def test_native_stiff_friction_limit_uses_endpoint_velocity_without_reversal():
    from scipy.optimize import brentq
    from prototype.ipc_reference import IPCProblem
    p,x=contact_fixture();install(p)
    direction=np.zeros_like(x);direction[:p.n,0]=1.;direction[p.mesh.bottom_nodes]=0.
    lag=IPCProblem.lag(p,x);potential=p.ipc.FrictionPotential(p.friction_velocity_epsilon)
    mass=float(p.mass_dof@(direction.ravel()**2));v0=1.
    drag=lambda v:float(potential.gradient(lag,p.cmesh,v*direction[p.surface])@direction[p.surface].ravel())
    p.material_dt=10*mass/drag(.5);p.previous_velocity=v0*direction
    p.friction=p.ipc.FrictionPotential(p.friction_velocity_epsilon*p.material_dt)
    def force(v1):
        du=p.material_dt*.5*(v0+v1)*direction
        inc=velocity_increment(p,du)[p.surface]
        return float(p.friction.gradient(lag,p.cmesh,inc)@direction[p.surface].ravel())
    v1=brentq(lambda v:mass*(v-v0)/p.material_dt+force(v),0.,v0,xtol=1e-15)
    assert 0<=v1<.1*v0
    physical=p.material_dt*force(v1)*v1
    numerical=p.material_dt*force(v1)*(.5*(v0+v1)-v1)
    assert abs(.5*mass*(v1*v1-v0*v0)+physical+numerical)<1e-12
    assert numerical>0 and physical>=0


def test_saved_energy_audit_includes_endpoint_friction_time_error():
    from release_dynamics import check_energy_gate, DEFECT_KEYS
    from prototype.zero_gap_cut import EnergyResolutionRequired
    p=install(problem(dict(candidate(),energy_tolerance=.001,energy_normalization='mechanical_reservoir')))
    e={k:0. for k in DEFECT_KEYS}
    e.update(mechanical_reservoir_n_mm=1.,friction_time_defect_n_mm=-.00100001)
    with pytest.raises(EnergyResolutionRequired):check_energy_gate(p,e)
