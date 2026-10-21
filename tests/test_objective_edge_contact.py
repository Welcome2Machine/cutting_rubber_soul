"""Force/moment/derivative checks for a bounded local contact repair candidate."""
import unittest
import numpy as np
from prototype.contact_controls import objective_edge_contact


class ObjectiveEdgeContactTests(unittest.TestCase):
    def data(self, x, old, **kwargs):
        return objective_edge_contact(x,old,weight=.01,dhat=.0025,stiffness=20.,
            mu=.35,epsilon=1e-5,**kwargs)

    def test_common_finite_rotation_no_friction_and_moment_balance(self):
        old=np.array([[.003,.0015],[0.,0.],[.01,0.]])
        for angle in (.02,.3,1.7):
            Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            x=old@Q.T+[.07,-.09];r=self.data(x,old)
            self.assertAlmostEqual(r['slip_mm'],0.,delta=5e-17)
            self.assertLess(np.linalg.norm(r['friction_gradient']),1e-10)
            g=r['gradient'].reshape(3,2)
            np.testing.assert_allclose(g.sum(axis=0),0.,atol=1e-12)
            self.assertAlmostEqual(float(np.sum(g*np.column_stack((-x[:,1],x[:,0])))),0.,delta=1e-12)

    def test_nonzero_slip_requires_finite_gap_moment_correction(self):
        old=np.array([[.003,.0015],[0.,0.],[.01,0.]])
        x=old.copy();x[0,0]+=.0001
        r=self.data(x,old);f=r['friction_gradient'].reshape(3,2)
        self.assertAlmostEqual(r['slip_mm'],.0001,places=15)
        self.assertAlmostEqual(f[0,0],.35*r['normal_capacity_n'],places=12)
        self.assertAlmostEqual(f[0,1],0.,places=12)
        # Conventional barycentric tangential tractions leave a finite couple.
        # The objective virtual-work derivative balances it on the edge nodes.
        self.assertGreater(abs(f[1,1]),0.)
        np.testing.assert_allclose(f.sum(axis=0),0.,atol=1e-12)
        self.assertAlmostEqual(float(np.sum(f*np.column_stack((-x[:,1],x[:,0])))),0.,delta=1e-12)
        self.assertGreater(r['friction_dissipation_nmm'],0.)

    def test_fresh_jacobian_includes_capacity_geometry_and_slip_history(self):
        old=np.array([[.003,.0015],[0.,0.],[.01,0.]])
        for slip in (-2e-5,-3e-6,0.,3e-6,2e-5):
            x=old.copy();x[0]+=[slip,-.0002];x[2]+=[.00005,.00002]
            r=self.data(x,old);h=1e-9
            fd=np.column_stack([(self.data(x+h*d.reshape(3,2),old)['gradient']-
                self.data(x-h*d.reshape(3,2),old)['gradient'])/(2*h) for d in np.eye(6)])
            np.testing.assert_allclose(r['jacobian'],fd,rtol=3e-6,atol=2e-5)
            hs=np.column_stack([(self.data(x+h*d.reshape(3,2),old)['slip_gradient']-
                self.data(x-h*d.reshape(3,2),old)['slip_gradient'])/(2*h) for d in np.eye(6)])
            np.testing.assert_allclose(r['slip_hessian'],hs,rtol=1e-6,atol=1e-6)

    def test_covariance_with_real_slip_and_independent_finite_rotation(self):
        old=np.array([[.003,.0015],[0.,0.],[.01,0.]])
        x=old.copy();x[0,0]+=3e-6
        Q=np.array([[.6,-.8],[.8,.6]])
        a=self.data(x,old);b=self.data(x@Q.T+[.1,.2],old)
        self.assertAlmostEqual(a['slip_mm'],b['slip_mm'],delta=1e-16)
        np.testing.assert_allclose(a['gradient'].reshape(3,2)@Q.T,b['gradient'].reshape(3,2),rtol=1e-10,atol=1e-10)

    def test_no_slip_from_normal_motion_or_master_stretch(self):
        old=np.array([[.003,.0015],[0.,0.],[.01,0.]])
        x=old.copy();x[0,1]+=.0003
        self.assertAlmostEqual(self.data(x,old)['slip_mm'],0.,delta=1e-17)
        x=old.copy();x[:,0]*=1.5
        self.assertAlmostEqual(self.data(x,old)['slip_mm'],0.,delta=1e-17)

    def test_crossing_and_negative_correction_weights_fail_closed(self):
        old=np.array([[.003,.0015],[0.,0.],[.01,0.]])
        for point in (-.001,0.,.01,.011):
            x=old.copy();x[0,0]=point
            with self.assertRaisesRegex(ValueError,'surface chart'):self.data(x,old)
        with self.assertRaisesRegex(ValueError,'weight'):
            objective_edge_contact(old,old,weight=-.01,dhat=.0025,stiffness=20.,mu=.35,epsilon=1e-5)


if __name__=='__main__':unittest.main()
