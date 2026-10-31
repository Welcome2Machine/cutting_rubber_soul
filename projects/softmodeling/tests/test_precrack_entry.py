import time
import numpy as np
import pytest
from test_face_cut import config
from tools.run_precrack_entry import problem
from tools.run_endpoint_face_route import problem as original
from tools.run_fracture_route import snapshot,restore_audit


def cfg(gap):return dict(config(.5,'endpoint_friction'),pure_precrack=True,initial_slit_halfwidth_mm=gap)


def test_zero_gap_control_is_identical_geometry_and_mass():
    p=problem(cfg(0.));q=original(dict(config(.5,'endpoint_friction'),pure_precrack=True))
    np.testing.assert_array_equal(p.rest,q.rest);np.testing.assert_array_equal(p.mass,q.mass)
    np.testing.assert_array_equal(p.history,q.history);np.testing.assert_array_equal(p.free,q.free)


@pytest.mark.parametrize('gap',[.04,.2])
def test_notch_has_real_reference_volume_mass_and_no_initial_strain_or_gc_charge(gap):
    p=problem(cfg(gap));q=problem(cfg(0.))
    np.testing.assert_allclose(p.deformation(p.rest),np.broadcast_to(np.eye(2),(len(p.mesh.triangles),2,2)),atol=1e-12)
    assert 2*p.mass.sum()<2*q.mass.sum()
    assert 2*p.mass.sum()==pytest.approx(2*p.mesh.bulk_data[0].sum()*p.config.width_mm*p.density)
    assert p.rest[p.seam[-1],0]==pytest.approx(15+gap)
    assert p.rest[p.seam[-6],0]==15.
    assert p.connected_length(p.precracked_faces)==pytest.approx(.5)
    assert p.rule['face_area_mm2'][p.precracked_faces].sum()>.5*p.config.width_mm
    np.testing.assert_allclose(p.cohesive_algorithmic(p.rest)['traction'][p.precracked_faces],0.,atol=1e-14)
    old=p.released.copy();history=p.history.copy();x,rec=p.step(p.rest,p.start,p.start+1e-4,time.monotonic()+20)
    assert rec['energy']['fracture_dissipation_n_mm']==pytest.approx(0.,abs=1e-14)
    np.testing.assert_array_equal(p.history,history)
    assert restore_audit(problem(cfg(gap)),snapshot(p,x,old,p.start+1e-4),rec)['passed']


def test_saved_pure_notch_trial_restores_its_own_reference_geometry():
    from tools.face_witness import trial_fields
    from tools.audit_face_witness import restore
    c=cfg(.2);p=problem(c)
    x,_=p.step(p.rest,p.start,p.start+1e-4,time.monotonic()+20)
    s=trial_fields(p,x);q=restore(c,s)
    np.testing.assert_array_equal(q.rest,p.rest)
    np.testing.assert_array_equal(q.mass,p.mass)
    np.testing.assert_array_equal(q.active_faces,p.precracked_faces)
