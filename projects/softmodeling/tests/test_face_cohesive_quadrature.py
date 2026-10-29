"""Independent face kernel; does not certify the nodal production trajectory."""
import numpy as np
import pytest
from face_cohesive_quadrature import face_rule,response


@pytest.mark.parametrize('spacing',[.1,.05,.025])
def test_face_and_tributary_total_area_and_gc_are_identical(spacing):
    y=np.linspace(0,1,round(1/spacing)+1);rule=face_rule(y,20)
    assert rule['face_area_mm2'].sum()==pytest.approx(20)
    assert rule['nodal_tributary_area_mm2'].sum()==pytest.approx(20)
    old=np.zeros(rule['history_shape']);q=response(np.full(len(y),.1),old,rule,.2,.006)
    assert q['dissipated_n_mm']==pytest.approx(.006*20)
    assert q['required_fracture_work_n_mm']==pytest.approx(.006*20)
    assert q['fully_failed_faces'].all() and not old.any() and not q['physical_history_committed']


def test_nonuniform_opening_exact_potential_force_and_tangent():
    rule=face_rule([0,.1,.2],20);opening=np.array([.005,.04,.01]);old=np.full(rule['history_shape'],.005)
    q=response(opening,old,rule,.2,.006);d=np.array([.2,-.1,.3]);eps=1e-7
    plus=response(opening+eps*d,old,rule,.2,.006);minus=response(opening-eps*d,old,rule,.2,.006)
    assert (plus['incremental_potential_n_mm']-minus['incremental_potential_n_mm'])/(2*eps)==pytest.approx(q['force_wrt_full_opening_n']@d,abs=1e-10)
    np.testing.assert_allclose((plus['force_wrt_full_opening_n']-minus['force_wrt_full_opening_n'])/(2*eps),q['tangent_wrt_full_opening_n_mm']@d,atol=1e-8)
    assert np.array_equal(old,np.full(rule['history_shape'],.005))


def test_face_support_is_paid_once_and_nodal_history_cannot_be_transplanted():
    rule=face_rule([0,.1,.2],20);old=np.zeros(rule['history_shape'])
    q=response([0,.06,0],old,rule,.2,.006)
    assert q['finite_crack_area_mm2']==pytest.approx(4)
    assert q['incremental_potential_n_mm']==pytest.approx(.016)
    assert q['required_fracture_work_n_mm']==pytest.approx(.024)
    assert not q['fully_failed_faces'].all()
    with pytest.raises(ValueError,match='history identity'):response([0,.06,0],np.zeros(3),rule,.2,.006)


def test_unloading_preserves_face_histories_and_nonnegative_stored_energy():
    rule=face_rule([0,.1,.2],20,4);old=np.full(rule['history_shape'],.04)
    q=response([.01,.02,.01],old,rule,.2,.006)
    np.testing.assert_array_equal(q['candidate_history'],old)
    assert q['stored_n_mm']>=0 and q['dissipated_n_mm']>0
    with pytest.raises(ValueError,match='Nonnegative'):response([-.1,0,0],old,rule,.2,.006)
