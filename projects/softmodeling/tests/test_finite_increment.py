import time
import numpy as np
import pytest
from mechanics.discrete_gradient import frozen_response, frozen_energy
from prototype.finite_increment_cut import finite_patches, coupled_margin, FrozenVirtualCrack
from tools.run_zero_gap_cut import problem
from test_energy_cut import candidate
from step_control import VirtualRootIncomplete


def config():
    return dict(candidate(), fracture_criterion='finite_increment', ffm_virtual_seconds=10.,
                ffm_max_candidates=256, ffm_friction_epsilon_mm=1e-6)


def test_frozen_elastic_gradient_and_full_tangent():
    rng=np.random.default_rng(831)
    f=np.eye(2)+.1*rng.normal(size=(4,2,2));a=np.tile(np.diag([1.2,1/1.2,1.]),(4,1,1))
    q=frozen_response(f,a,.14,1.24,23.343)
    eps=1e-6
    for j,d in enumerate(np.eye(4).reshape(4,2,2)):
        plus=frozen_response(f+eps*d,a,.14,1.24,23.343)
        minus=frozen_response(f-eps*d,a,.14,1.24,23.343)
        np.testing.assert_allclose(q['P'].reshape(-1,4)[:,j],(plus['energy']-minus['energy'])/(2*eps),atol=3e-9)
        np.testing.assert_allclose(q['tangent'][:,:,j],((plus['P']-minus['P'])/(2*eps)).reshape(-1,4),rtol=1e-7,atol=2e-8)


def test_same_finite_patch_must_satisfy_both_conditions_and_pay_friction():
    assert not coupled_margin(np.array([.3,.1]),.2,1.,0.,.1)[0]
    assert not coupled_margin(np.array([.3,.3]),.2,.05,0.,.1)[0]
    assert not coupled_margin(np.array([.3,.3]),.2,.15,.06,.1)[0]
    assert not coupled_margin(np.array([.3,.3]),.2,.15,.04,.1,.02)[0]
    assert coupled_margin(np.array([.3,.3]),.2,.15,.04,.1)[0]
    with pytest.raises(ValueError):coupled_margin(np.array([.3]),.2,1.,-.01,.1)


def test_candidates_are_finite_connected_and_cannot_cross_compression_or_old_cracks():
    y=np.arange(6.)*.1;area=np.ones(6)
    patches=finite_patches(np.array([1.,.3,.3,-1.,.3,.3]),np.array([False]*5+[True]),y,area,.2)
    assert {tuple(p) for p in patches}=={(1,),(2,),(1,2),(4,)}
    assert all(0 not in p and 3 not in p and 5 not in p for p in patches)


def test_plot_comparison_preserves_a_narrow_force_pulse():
    from tools.report_finite_increment import curve_difference,connected_length
    # A triangular unit pulse of width 2e-9 is missed by a regular depth grid.
    d=np.array([0.,.5-1e-9,.5,.5+1e-9,1.]);f=np.array([1.,1.,2.,1.,1.])
    q=curve_difference(np.array([0.,1.]),np.ones(2),d,f,1.)
    assert q['maximum_absolute_difference_n']==1.
    assert q['relative_l2']**2==pytest.approx(2e-9/3,rel=1e-7)
    assert connected_length(np.array([False,True,True,False]),np.arange(4.))==0.
    assert connected_length(np.array([False,False,True,True]),np.arange(4.))==1.


def test_virtual_fields_do_not_advance_or_mutate_physical_history():
    p=problem(config());p.material_dt=.025
    fields={k:getattr(p,k).copy() for k in ('previous','previous_material_metric','history','released','velocity')}
    v=FrozenVirtualCrack(p,p.rest,p.material_metric,p.history,p.released,np.zeros_like(p.released),1e-6)
    x,g,res=v.equilibrium(p.rest,p.start,time.monotonic()+10)
    assert res<1e-10 and abs(v.stored(x))<1e-10
    v.previous_material_metric[0,0,0]=2
    v.released[-1]=True
    for key,value in fields.items():np.testing.assert_array_equal(getattr(p,key),value)


def test_virtual_does_not_include_mass_in_stored_energy_or_force():
    p=problem(config());p.material_dt=.001
    mass=p.mass_dof.copy()
    v=FrozenVirtualCrack(p,p.rest,p.material_metric,p.history,p.released,np.zeros_like(p.released),1e-6)
    x=p.rest.copy();x[:p.n,1]*=.999
    e0,g0,h0,j0=v.evaluate(x,None,projected=False)
    v.mass_dof*=1000;v.previous_velocity[:]=123.;v.material_dt=12.
    e1,g1,h1,j1=v.evaluate(x,None,projected=False)
    assert e1==e0 and j1==j0
    np.testing.assert_array_equal(g1,g0);np.testing.assert_array_equal(h1.toarray(),h0.toarray())
    np.testing.assert_array_equal(p.mass_dof,mass)


def test_failed_virtual_probe_cannot_release_physical_ties(monkeypatch):
    p=problem(config());p.material_dt=.025
    previous=p.released.copy()
    def fail(*args):raise TimeoutError('virtual')
    monkeypatch.setattr(FrozenVirtualCrack,'equilibrium',fail)
    with pytest.raises(VirtualRootIncomplete):p.propose_insertion(p.rest,np.ones(len(p.seam)),0,time.monotonic()+10)
    np.testing.assert_array_equal(p.released,previous)


def test_virtual_continuation_cannot_leak_artificial_load_into_energy(monkeypatch):
    import prototype.finite_increment_cut as module
    p=problem(config());p.material_dt=.025
    v=FrozenVirtualCrack(p,p.rest,p.material_metric,p.history,p.released,np.zeros_like(p.released),1e-6)
    v.virtual_solver_load[v.base_free[0]]=1.
    with pytest.raises(RuntimeError,match='Artificial'):
        v.stored(v.rest)
    v.virtual_solver_load[:]=0
    def fail(*args):raise module.LoadStepFailure('deliberate no root')
    monkeypatch.setattr(module,'solve',fail)
    maximum=v.max_iterations
    with pytest.raises(module.LoadStepFailure,match='continuation stalled'):
        v.continued_equilibrium(v.rest,0,time.monotonic()+10)
    assert not np.any(v.virtual_solver_load)
    assert v.max_iterations==maximum


def test_frozen_virtual_energy_force_and_tangent_with_partly_damaged_seam():
    p=problem(config());p.material_dt=.025
    p.released[-2:]=True
    history=p.history.copy();history[-2:]=.02
    v=FrozenVirtualCrack(p,p.rest,p.material_metric,history,p.released,np.zeros_like(p.released),1e-6)
    x=p.rest.copy();x[p.seam[-2:],0]+=.004
    d=np.zeros_like(x);d[p.seam[-2:],0]=[.3,.4]
    e,g,h,j=v.evaluate(x,None,projected=False)
    eps=1e-7
    ep,gp,_,_=v.evaluate(x+eps*d,None,tangent=False)
    em,gm,_,_=v.evaluate(x-eps*d,None,tangent=False)
    assert (ep-em)/(2*eps)==pytest.approx(g@d.ravel(),rel=1e-6,abs=1e-7)
    np.testing.assert_allclose((gp-gm)/(2*eps),h@d.ravel(),rtol=2e-6,atol=1e-5)


def test_primary_ptc_resolves_dynamic_updates_below_coordinate_precision():
    from prototype.force_globalization import solve_increment_ptc
    p=problem(dict(config(),nonlinear_solver='increment_ptc'));p.material_dt=1e-12
    x=p.rest.copy();increment=np.zeros_like(x)
    dof=p.free[-1];increment.ravel()[dof]=1e-16
    p.set_primary_increment(x,increment)
    assert np.array_equal(x,p.previous+increment)
    initial=np.linalg.norm(p.evaluate(x,p.lag(x),tangent=False)[1][p.free])
    assert initial>1e-4
    # Unbounded CCD fractions can overflow for this deliberately sub-ulp ray;
    # they are conservative infinity bounds, not a physical force warning.
    with np.errstate(over='ignore',divide='ignore'):
        value,it=solve_increment_ptc(p,x,p.start,time.monotonic()+10)
    residual=np.sqrt(2)*np.linalg.norm(p.evaluate(value,p.lag(value),tangent=False)[1][p.free])
    assert residual<1e-4 and it>0
    assert abs(p.step_increment(value).ravel()[dof])<abs(increment.ravel()[dof])


def test_restored_checkpoint_preserves_next_step_and_energy():
    from tools.run_zero_gap_cut import snapshot
    from tools.continue_zero_gap_cut import restore
    p=problem(dict(config(),nonlinear_solver='increment_ptc'))
    x,r=p.step(p.rest.copy(),p.start,.005,time.monotonic()+20)
    z=snapshot(p,x,np.zeros_like(p.released),.005)
    q=problem(dict(config(),nonlinear_solver='increment_ptc'))
    value,depth=restore(q,z,[(.005,x)])
    # Use identical single-state predictors in both paths, excluding a solver
    # warm-start difference from this restart consistency check.
    p.seed_predictor_history([(.005,x)])
    a,ra=p.step(x,.005,.01,time.monotonic()+20)
    b,rb=q.step(value,depth,.01,time.monotonic()+20)
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    np.testing.assert_allclose(p.material_metric,q.material_metric,rtol=0,atol=1e-12)
    np.testing.assert_allclose(p.velocity,q.velocity,rtol=0,atol=1e-12)
    assert ra['energy']==rb['energy']


def test_friction_continuation_restores_original_friction_after_failure(monkeypatch):
    import prototype.force_globalization as fg
    p=problem(dict(config(),nonlinear_solver='friction_continuation'));p.material_dt=1e-8
    original=p.friction;maximum=p.max_iterations
    old=(p.material_metric.copy(),p.history.copy(),p.velocity.copy(),p.released.copy())
    def fail(*args):raise TimeoutError('stage budget')
    monkeypatch.setattr(fg,'solve',fail);monkeypatch.setattr(fg,'solve_increment_ptc',fail)
    with pytest.raises(TimeoutError):fg.solve_friction_continuation(p,p.rest,p.start,time.monotonic()+1)
    assert p.friction is original and p.max_iterations==maximum
    for a,b in zip(old,(p.material_metric,p.history,p.velocity,p.released)):np.testing.assert_array_equal(a,b)


def test_solver_only_continuation_accepts_original_force_and_energy():
    cfg=dict(config(),nonlinear_solver='friction_continuation')
    p=problem(cfg)
    x,r=p.step(p.rest.copy(),p.start,.005,time.monotonic()+20)
    assert r['force_continuation']['original_friction_restored']
    assert r['residual_n']<1e-4
    from tools.run_zero_gap_cut import snapshot,audit
    z=snapshot(p,x,np.zeros_like(p.released),.005)
    assert audit(problem(cfg),z,r)['passed']


@pytest.mark.parametrize('spacing',[.1,.075])
def test_reference_constraint_traction_matches_uniform_first_piola_patch(spacing):
    p=problem(dict(config(),seam_spacing_mm=spacing))
    v=FrozenVirtualCrack(p,p.rest,p.material_metric,p.history,p.released,np.zeros_like(p.released),1e-6)
    f=np.diag([1.04,1/1.04])
    u=(p.mesh.reference_xy-np.array([15.,0.]))@(f-np.eye(2)).T
    local,_,dofs,_,_=v.material_bulk(u.ravel(),with_tangent=False)
    gradient=np.zeros(p.total);np.add.at(gradient,dofs.ravel(),local.ravel())
    measured=-gradient[2*p.seam]/p.seam_area
    expected=frozen_response(f[None],np.eye(3)[None],.14,1.24,p.config.bulk_modulus_n_mm2)['P'][0,0,0]
    np.testing.assert_allclose(measured,expected,rtol=1e-10,atol=1e-11)
    assert p.seam_area.sum()==pytest.approx(100.)
