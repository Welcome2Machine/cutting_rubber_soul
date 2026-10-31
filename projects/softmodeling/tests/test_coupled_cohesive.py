"""Coupled softening controls use exact area/history, not a pure-precrack toggle."""
import time
import numpy as np
import pytest
from test_precrack_entry import cfg
from coupled_cohesive import problem
from coupled_contact import problem as pure
from face_cut import eligible_nodes
from tools.face_witness import accepted_fields,array_hash


def config():return dict(cfg(.2),pure_precrack=False)


def patch():
    p=problem(config(),solve_tolerance_n=1e-7)
    # Explicit conditional two-face test fixture, not a predicted activation.
    p.active_faces[43:45]=True;p.previous_active_faces=p.active_faces.copy()
    p.released=eligible_nodes(p.active_faces);p.rebuild_contact();p.active_contact[p.seam]=~p.released;p.set_free()
    p.previous_history=p.history.copy();return p


def test_growth_factory_preserves_real_mass_material_and_fully_failed_initial_gc():
    p=problem(config());q=pure(cfg(.2),'maxwell',1e-9)
    np.testing.assert_array_equal(p.rest,q.rest);np.testing.assert_array_equal(p.mass,q.mass)
    assert p.material==q.material and p.strength==q.strength and p.toughness==q.toughness
    assert p.kinematic_theta==1 and not p.pure_precrack
    assert p.activation_tolerance_n==pytest.approx(1e-4/np.sqrt(2))
    assert p.force_tolerance_n==pytest.approx(1e-9/np.sqrt(2))
    assert p.initial_fracture_energy_n_mm==q.initial_fracture_energy_n_mm
    np.testing.assert_array_equal(p.cohesive_response(p.rest)['history'],p.history)


def test_softening_endpoint_tangent_and_numerical_work_are_explicit():
    p=patch();p.material_dt=.001;x=p.rest.copy();x[p.seam[44],0]+=.003
    q=p.cohesive_algorithmic(x);f,H=p.face_force(q)
    assert np.min(q['tangent'][43:45])<0
    d=np.zeros_like(x);d[p.seam[44],0]=.3;eps=1e-7
    fp=p.face_force(p.cohesive_algorithmic(x+eps*d))[0];fm=p.face_force(p.cohesive_algorithmic(x-eps*d))[0]
    np.testing.assert_allclose((fp-fm)/(2*eps),(H@d.ravel())[2*p.seam],atol=1e-8)
    old=p.cohesive_response(p.previous);w=p.rule['weights_mm2']*p.active_faces[:,None]
    physical=float(np.sum(w*(q['stored']+q['dissipated']-old['stored']-old['dissipated'])))
    work=2*float(f@(x-p.previous)[p.seam,0])
    assert work-physical<0, 'Softening endpoint quadrature has a signed numerical term, not hidden physical damping'


def test_softening_full_area_pays_gc_once_after_unload_reload():
    p=patch();history=p.history.copy();newfaces=p.active_faces&~p.precracked_faces
    weights=p.rule['weights_mm2']*newfaces[:,None];total=0.
    for gap in (.01,.3,.02,.4):
        x=p.rest.copy();x[p.seam[43:46],0]+=.5*gap
        old=p.cohesive_response(p.rest,history);q=p.cohesive_response(x,history)
        total+=np.sum(weights*(q['dissipated']-old['dissipated']));history=q['history']
    assert total==pytest.approx(p.toughness*p.rule['face_area_mm2'][newfaces].sum())


def test_failed_coupled_growth_rolls_back_face_points_and_active_graph(monkeypatch):
    p=problem(config());x=p.rest.copy();before=array_hash(accepted_fields(p,x))
    def fail(*args):
        p.active_faces[43:45]=True;p.released=eligible_nodes(p.active_faces);p.rebuild_contact()
        raise TimeoutError('Controlled growth solve interruption')
    monkeypatch.setattr(p,'solve_active',fail)
    with pytest.raises(TimeoutError):p.step(x,p.start,p.start+1e-4,time.monotonic()+10)
    assert before==array_hash(accepted_fields(p,x))
    assert p.failed_active_faces[43:45].all()


def test_real_kinetic_pulse_softening_step_closes_coupled_energy():
    from tools.run_fracture_route import snapshot
    from tools.audit_coupled_contact import independent_balance
    p=patch();x=p.rest.copy();x[p.seam[44],0]+=.001
    # Specified conditional initial kinetic pulse; true mass, no force/damping
    # modifications. This is a unit control, never a production continuation.
    p.history=p.cohesive_response(x,p.history)['history'].copy();p.previous_history=p.history.copy()
    p.velocity[p.seam[44],0]=100.;old=p.history.copy();released=p.released.copy()
    from prototype.zero_gap_cut import EnergyResolutionRequired
    before=array_hash(accepted_fields(p,x))
    with pytest.raises(EnergyResolutionRequired):p.step(x,p.start,p.start+p.loading_rate*1e-8,time.monotonic()+30)
    assert array_hash(accepted_fields(p,x))==before
    h=5e-9;y,rec=p.step(x,p.start,p.start+p.loading_rate*h,time.monotonic()+30)
    s=snapshot(p,y,released,p.start+p.loading_rate*h)
    assert np.any(p.history[43:45]>old[43:45])
    assert rec['energy']['fracture_dissipation_n_mm']>0 and rec['residual_n']<1e-4
    audit=independent_balance(p,s)
    assert audit['independent_fracture_n_mm']==pytest.approx(rec['energy']['fracture_dissipation_n_mm'],abs=1e-14)


def test_own_initial_growth_driver_and_independent_clock(tmp_path,monkeypatch):
    from tools import run_coupled_growth as growth
    from tools import run_fracture_route as route
    c=dict(config(),coupled_control=dict(material_layer='maxwell',solve_tolerance_n=1e-7))
    monkeypatch.setattr(route,'problem',growth.problem);monkeypatch.setattr(route,'sources',growth.sources);monkeypatch.setattr(route,'finish',growth.finish)
    r=route.run(c,tmp_path/'run',30,stop_depth=.001,ipc_threads=2,blas_threads=1)
    assert r['status']=='completed_pilot' and r['own_initial_state']
    a=growth.audit(tmp_path/'run',tmp_path/'audit.json',30,ipc_threads=2,blas_threads=1)
    assert a['passed'],a.get('error')
    assert a['complete_requested_path'] and a['new_fracture_dissipation_n_mm']==0.


def test_own_path_depth_caps_do_not_jump_over_declared_refinement_interval():
    from tools.run_coupled_growth import problem as configured
    c=dict(config(),coupled_control=dict(material_layer='maxwell',solve_tolerance_n=1e-9,
       depth_step_caps=[dict(through_mm=.355,max_dt_s=.025),dict(through_mm=.365,max_dt_s=.000125),dict(through_mm=2.,max_dt_s=.0005)]))
    p=configured(c)
    assert p.next_loading_depth(.354,.005,2.)==.355
    assert p.next_loading_depth(.355,.005,2.)==pytest.approx(.355025)
    assert p.next_loading_depth(.366,.005,2.)==pytest.approx(.3661)
