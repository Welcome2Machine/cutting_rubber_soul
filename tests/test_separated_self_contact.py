"""Candidate domain guards; no claim of general self-contact replacement."""
import unittest
import numpy as np
from scipy.sparse import csc_matrix
from prototype.separated_self_contact import certify_separated,enable_separated_self_contact,POLICY
from prototype.ipc_reference import toolkit,LoadStepFailure


class SeparatedSelfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ipc=toolkit();cls.ipc.set_num_threads(1)

    def geometry(self):
        rest=np.array([[0.,0.],[.005,0.],[.01,0.],[.015,0.]])
        edges=np.array([[0,1],[1,2],[2,3]])
        return rest,self.ipc.CollisionMesh(rest,edges),{tuple(e) for e in edges}

    def test_connected_compression_is_not_opposing_surface_contact(self):
        x,m,a=self.geometry();x[2:,0]-=.003
        self.assertGreater(certify_separated(self.ipc,m,x,.0025,a),0)
        self.assertFalse(self.ipc.has_intersections(m,x))

    def test_fold_is_rejected_and_original_ccd_sees_crossing(self):
        _,m,a=self.geometry();x=np.array([[0.,0.],[.005,0.],[.001,.001],[.006,.001]])
        with self.assertRaisesRegex(LoadStepFailure,'nonadjacent'):
            certify_separated(self.ipc,m,x,.0025,a)
        crossed=x.copy();crossed[2:,1]=-.001
        bound=self.ipc.compute_collision_free_stepsize(m,x,crossed)
        self.assertGreater(bound,0);self.assertLess(bound,1)

    def test_policy_is_explicit_and_history_audit_fails_closed(self):
        class Base:
            contact_formulation='test'
            def checkpoint_data(self):return {'history':42}
            def audit_state(self,state,record):return state['history']
        p=Base();x,m,a=self.geometry();p.self_mesh=m
        p.ipc=self.ipc;p.body_surface=np.arange(4);p.self_dofs=np.arange(8);p.total=8;p.dhat=.0025
        enable_separated_self_contact(p);self.assertIn(POLICY,p.contact_formulation)
        state=p.checkpoint_data();self.assertEqual(p.audit_state(state,{}),42)
        with self.assertRaisesRegex(ValueError,'policy'):p.audit_state({'history':42},{})
        x[2:,0]-=.003;e,g,h,w=p._self_terms(x,True)
        self.assertEqual(e,0);self.assertEqual(w,0);self.assertEqual(h.nnz,0)
        np.testing.assert_array_equal(g,np.zeros(8))

    def test_experimental_speed_is_vertical_at_both_angles(self):
        from types import SimpleNamespace
        from tools.run_separated_contact import make
        for angle in (33.,57.):
            a=SimpleNamespace(angle=angle,coarse=True,spacing=.000625,order=5,
                contact_basis='P1',surface_spacing=.0025,lab_speed=.2)
            p=make(a,False)
            np.testing.assert_allclose(p.loading_direction*p.loading_rate,[0.,-.2],atol=2e-16)
            self.assertAlmostEqual(p.motion_signature()['normal_speed_mm_s'],.2*np.cos(np.deg2rad(angle)))
            q=.03;delta=p.prescribed_blade_positions(q)[0]-p.blade_at_zero[0]
            self.assertAlmostEqual(-delta[1],q/np.cos(np.deg2rad(angle)),places=13)


if __name__=='__main__':unittest.main()
