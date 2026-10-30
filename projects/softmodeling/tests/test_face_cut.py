"""Mechanical invariants of the actual-geometry P1 face route, not curve fitting."""
import time
import numpy as np
import pytest
from test_energy_cut import candidate
from tools.run_fracture_route import problem,snapshot,restore_audit
from face_cut import eligible_nodes


def config(precrack=0.,friction='midpoint'):
    return dict(schema='softcut-fracture-route/1',base=dict(candidate(),fracture_criterion='local_or_neighbourhood',energy_tolerance=.00025,energy_normalization='mechanical_reservoir',energy_absolute_tolerance_n_mm=1e-7,attempt_seconds=30),interface='face',precrack_mm=precrack,quadrature_order=4,friction_time=friction)


def test_intact_faces_have_exact_zero_gap_mass_and_no_added_compliance():
    from tools.run_zero_gap_cut import problem as original
    c=config();p=problem(c);q=original(c['base'])
    np.testing.assert_array_equal(p.rest,q.rest);np.testing.assert_array_equal(p.mass,q.mass)
    np.testing.assert_array_equal(p.free,q.free);np.testing.assert_array_equal(p.surface,q.surface)
    p.material_dt=q.material_dt=.01
    p.friction=p.ipc.FrictionPotential(p.friction_velocity_epsilon*.01);q.friction=q.ipc.FrictionPotential(q.friction_velocity_epsilon*.01)
    x=p.rest.copy();x[:p.n,1]*=.9999
    for obj in (p,q):obj.set_primary_increment(x,x-obj.previous)
    a=p.evaluate(x,None,projected=False);b=q.evaluate(x,None,projected=False)
    np.testing.assert_allclose(a[1],b[1],atol=1e-13)
    np.testing.assert_allclose(a[2].toarray(),b[2].toarray(),atol=1e-12)


def test_adjacent_intact_face_pins_endpoint_and_surface_precrack_is_not_recharged():
    np.testing.assert_array_equal(eligible_nodes([False,True,False]),[False,False,False,False])
    np.testing.assert_array_equal(eligible_nodes([False,True,True]),[False,False,True,True])
    p=problem(config(.5));tip=np.flatnonzero(p.rest[p.seam,1]==4.5)[0]
    assert not p.released[tip] and p.released[tip+1:].all()
    assert p.initial_fracture_energy_n_mm==pytest.approx(.006*.5*20)
    assert p.connected_length(p.precracked_faces)==pytest.approx(.5)
    assert not p.history[~p.precracked_faces].any()
    q=p.cohesive_response(p.rest,p.history)
    assert np.max(q['traction'][p.precracked_faces])==0
    assert np.sum(q['stored'])==0
    with pytest.raises(ValueError,match='coincide'):problem(config(.53))


def test_face_discrete_work_tangent_and_gc_full_support_once():
    p=problem(config());p.active_faces[47:49]=True;p.released=eligible_nodes(p.active_faces);p.rebuild_contact()
    x=p.rest.copy();x[p.seam[48],0]+=.02
    q=p.cohesive_algorithmic(x);f,H=p.face_force(q)
    du=x-p.previous;new=p.cohesive_response(x);old=p.cohesive_response(p.previous)
    w=p.rule['weights_mm2']*p.active_faces[:,None]
    assert 2*f@du[p.seam,0]==pytest.approx(np.sum(w*(new['stored']+new['dissipated']-old['stored']-old['dissipated'])),abs=1e-12)
    d=np.zeros_like(x);d[p.seam[48],0]=.3;eps=1e-7
    fp=p.face_force(p.cohesive_algorithmic(x+eps*d))[0];fm=p.face_force(p.cohesive_algorithmic(x-eps*d))[0]
    np.testing.assert_allclose((fp-fm)/(2*eps),(H@d.ravel())[2*p.seam],atol=1e-8)
    # Open both faces uniformly, then unload/reload: their full reference area,
    # including the tied-tip quadrature support, is charged once, not tributaries.
    total=0.;history=p.history.copy()
    for opening in (.01,.08,.02,.1):
        z=p.rest.copy();z[p.seam[47:50],0]+=.5*opening
        before=p.cohesive_response(p.rest,history)['dissipated']
        q=p.cohesive_response(z,history);total+=np.sum(w*(q['dissipated']-before));history=q['history']
    assert total==pytest.approx(4*.006)


def test_traction_overshoot_is_not_hidden_as_history_or_changed_strength(monkeypatch):
    p=problem(config());x=p.rest.copy();p.face_events=[]
    g=np.zeros(p.total);g[2*p.seam[48]]=-p.seam_area[48]*(p.strength+.01)
    monkeypatch.setattr(p,'evaluate',lambda *a,**k:(0.,g,None,1.))
    from prototype.zero_gap_cut import EnergyResolutionRequired
    with pytest.raises(EnergyResolutionRequired,match='traction event'):p.propose_insertion(x,None,0,time.monotonic()+1)
    assert not p.active_faces.any() and not p.history.any() and p.strength==.2
    g[2*p.seam[48]]=-p.seam_area[48]*p.strength-.5*p.force_tolerance_n
    mask=p.propose_insertion(x,None,0,time.monotonic()+1)
    assert np.flatnonzero(mask).tolist()==[48]
    p.released|=mask;p.rebuild_contact()
    assert p.active_faces[47:49].all() and p.released[48] and not p.released[47] and not p.released[49]
    assert p.face_events[-1]['new_area_mm2']==pytest.approx(4.)
    assert p.face_events[-1]['contact_topology_energy_jump_n_mm']==0


def test_failed_face_activation_rolls_back_both_topology_and_point_history(monkeypatch):
    p=problem(config());before=p.active_faces.copy()
    def fail(*args):
        p.active_faces[47:49]=True;p.released=eligible_nodes(p.active_faces);p.rebuild_contact()
        raise TimeoutError('controlled computation delay')
    monkeypatch.setattr(p,'solve_active',fail)
    with pytest.raises(TimeoutError):p.step(p.rest,p.start,p.start+1e-5,time.monotonic()+1)
    np.testing.assert_array_equal(p.active_faces,before);assert not p.history.any();assert not p.released.any()
    assert p.failed_active_faces[47:49].all()
    assert p.material_dt==pytest.approx(1e-5/.2)


def test_closed_precrack_has_no_gap_offset_and_full_temporal_ccd_blocks_crossing():
    p=problem(config(.5));x=p.rest.copy();direction=np.zeros_like(x)
    direction[p.seam[-1],0]=-.001
    assert p.safe_step(x,direction)==0.
    assert not p.accepted_chord_is_safe(x+direction)
    assert np.array_equal(p.rest[p.seam,0],np.full(len(p.seam),15.))


def test_contact_topology_creation_cannot_supply_unrecorded_energy(monkeypatch):
    p=problem(config());p.face_events=[];p.activation_positions=p.rest.copy()
    p.pending_faces=p.active_faces.copy();p.pending_faces[47:49]=True;p.pending_handoff={}
    original=p.barrier
    class Barrier:
        def __call__(self,*args):return original(*args)+float(p.active_faces[47])
    p.barrier=Barrier()
    with pytest.raises(RuntimeError,match='Unfunded'):p.rebuild_contact()
    assert p.face_events[-1]['contact_topology_energy_jump_n_mm']==2.


@pytest.mark.parametrize('friction',['midpoint','endpoint_friction'])
def test_actual_geometry_precrack_step_full_audit_and_history_tamper(friction):
    p=problem(config(.5,friction));old=p.released.copy();x,rec=p.step(p.rest,p.start,p.start+1e-4,time.monotonic()+20)
    state=snapshot(p,x,old,p.start+1e-4)
    assert restore_audit(problem(config(.5,friction)),state,rec)['accepted_chord_ccd_safe']
    assert rec['energy']['fracture_dissipation_n_mm']==pytest.approx(0.,abs=1e-15)
    state['active_faces']=state['active_faces'].copy();state['active_faces'][48]=False
    with pytest.raises((AssertionError,ValueError)):restore_audit(problem(config(.5,friction)),state,rec)
