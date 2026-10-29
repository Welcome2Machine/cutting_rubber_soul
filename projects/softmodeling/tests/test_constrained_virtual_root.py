import numpy as np
import pytest
from scipy.sparse import csc_matrix
from virtual_constrained_root import signed_chord_gap,constrained_direction
from tools.run_zero_gap_cut import problem
from test_finite_increment import config


def test_signed_chord_gap_envelope_derivative():
    anchor=np.array([[0.,0.],[1.,0.],[.4,.2]])
    end=np.array([[.1,.1],[1.1,.2],[.6,.03]])
    gap,gradient,_=signed_chord_gap(anchor,end,(0,1),2)
    delta=np.array([[.2,-.1],[.3,.4],[-.2,.5]])
    h=1e-6
    fd=(signed_chord_gap(anchor,end+h*delta,(0,1),2)[0]-signed_chord_gap(anchor,end-h*delta,(0,1),2)[0])/(2*h)
    assert np.sum(gradient*delta)==pytest.approx(fd,abs=1e-9)


def test_constrained_linear_solve_is_minimum_residual_on_the_boundary():
    a=csc_matrix([[4.,2.],[-1.,3.]])
    g=np.array([1.,2.]);n=np.array([1.,-.2]);bound=.4
    direction,newton,active=constrained_direction(a,g,n,bound)
    assert active and n@direction==pytest.approx(bound,abs=1e-14)
    tangent=np.array([.2,1.]);residual=g+a@direction
    assert residual@(a@tangent)==pytest.approx(0.,abs=1e-13)
    assert np.linalg.norm(a@newton+g)<1e-14
    for step in [-1.,-.1,.1,1.]:
        assert np.linalg.norm(g+a@(direction+step*tangent))>np.linalg.norm(residual)


def test_constrained_recovery_requires_explicit_original_ccd_controls():
    c=config();c['ffm_virtual_solver']='chord_constrained'
    with pytest.raises(ValueError,match='explicit CCD'):problem(c)
    c.update(ffm_ccd_tolerance=1e-9,ffm_chord_partition='geometry',ffm_chord_max_queries=256)
    assert problem(c).ffm_virtual_solver=='chord_constrained'
