from types import SimpleNamespace
import numpy as np
from tools import report_coupled_growth as report


def test_full_failure_connectivity_and_strict_new_edge_entry(monkeypatch):
    np.testing.assert_array_equal(report.connected_faces(np.array([True,False,True])),[False,False,True])
    p=SimpleNamespace(seam=np.array([0,1,2]),active_faces=np.array([True,True]),precracked_faces=np.array([False,True]))
    x=np.array([[0.,0.],[0.,1.],[0.,2.],[.01,.5]])
    c=dict(body_nodes=[0,1],blade_nodes=[3],kind='ev_collisions')
    monkeypatch.setattr(report,'blade_contacts',lambda p,x:[c])
    r=report.face_contacts(p,x,np.array([True,True]))[0]
    assert r['strict_face_interior'] and r['connected'] and r['classification']=='new_fully_failed'
    assert r['edge_coordinate']==.5 and r['gap_mm']==.01
    x[3,1]=0.
    assert not report.face_contacts(p,x,np.array([True,True]))[0]['strict_face_interior']
    x[3,1]=.5
    r=report.face_contacts(p,x,np.array([False,True]))[0]
    assert r['classification']=='new_softening' and not r['fully_failed']
    c['body_nodes']=[1]
    assert not report.face_contacts(p,x,np.array([True,True]))[0]['strict_face_interior']
