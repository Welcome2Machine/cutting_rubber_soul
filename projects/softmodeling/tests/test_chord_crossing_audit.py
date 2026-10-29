import numpy as np
from tools.audit_virtual_chord import verify_witness


def test_exact_rational_crossing_and_outside_edge_rejection():
    start=np.array([[0.,0.],[1.,0.],[.5,1.]])
    end=start.copy();end[2,1]=-1
    assert verify_witness(start,end,[0,1,2],[.4,.6])['passed']
    start[2,0]=2;end[2,0]=2
    assert not verify_witness(start,end,[0,1,2],[.4,.6])['passed']


def test_near_miss_does_not_become_a_crossing():
    start=np.array([[0.,0.],[1.,0.],[.5,1e-12]])
    end=start.copy();end[2,0]=.6
    assert not verify_witness(start,end,[0,1,2],[.4,.6])['passed']
