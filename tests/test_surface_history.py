"""Qualification limits of smooth material-coordinate contact, not an FE test."""
import unittest
import numpy as np
from prototype.surface_history import SurfaceChart, surface_contact


class SurfaceHistoryTests(unittest.TestCase):
    def setUp(self):
        self.u=np.linspace(-.01,.01,5);self.chart=SurfaceChart(self.u)
        self.nodes=np.column_stack((self.u,.02-np.sqrt(.02**2-self.u**2)))
        self.kw=dict(weight=.01,dhat=.0025,stiffness=132.,mu=.35,epsilon=1e-5)

    def configuration(self, u, gap=.0015, nodes=None):
        nodes=self.nodes if nodes is None else nodes
        c=self.chart.geometry(nodes);v=c(u,1);t=v/np.linalg.norm(v)
        return np.vstack((c(u)+gap*np.array([-t[1],t[0]]),nodes))

    def data(self,x,old,**extra):
        return surface_contact(self.chart,x,old,**self.kw,**extra)

    def test_flat_cross_knot_matches_persistent_edge(self):
        nodes=np.column_stack((self.u,np.zeros(len(self.u))))
        old=self.configuration(-.002,nodes=nodes);x=self.configuration(.003,nodes=nodes)
        q=self.data(x,old)
        self.assertAlmostEqual(q['slip_mm'],.005,places=13)
        self.assertAlmostEqual(q['gradient'][0,0],.35*q['normal_capacity_n'],places=10)

    def test_common_rotation_and_real_slip_covariance(self):
        old=self.configuration(-.002);x=self.configuration(.001)
        angle=.7;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rigid=self.data(old@Q.T+[.2,-.3],old)
        self.assertLess(abs(rigid['slip_mm']),1e-14)
        self.assertLess(np.linalg.norm(rigid['friction_gradient']),2e-8)
        a=self.data(x,old);b=self.data(x@Q.T+[.2,-.3],old)
        self.assertAlmostEqual(a['slip_mm'],b['slip_mm'],delta=1e-14)
        np.testing.assert_allclose(a['gradient']@Q.T,b['gradient'],atol=1e-8,rtol=1e-8)

    def test_force_and_moment_balance_with_nonzero_slip(self):
        x=self.configuration(.002);old=self.configuration(-.003)
        q=self.data(x,old)
        for name in ('normal_gradient','friction_gradient','gradient'):
            g=q[name];np.testing.assert_allclose(g.sum(axis=0),0.,atol=1e-10)
            self.assertLess(abs(np.sum(g*np.column_stack((-x[:,1],x[:,0])))),1e-11)
        self.assertGreater(q['friction_dissipation_nmm'],0.)

    def test_analytic_slip_and_normal_force_derivatives(self):
        old=self.configuration(-.001);x=self.configuration(.0013)
        q=self.data(x,old);h=2e-9
        for j,d in enumerate(np.eye(x.size).reshape((-1,)+x.shape)):
            a=self.data(x+h*d,old);b=self.data(x-h*d,old)
            self.assertAlmostEqual((a['slip_mm']-b['slip_mm'])/(2*h),
                                   q['slip_gradient'].ravel()[j],delta=2e-6)
            self.assertAlmostEqual((a['normal_energy_nmm']-b['normal_energy_nmm'])/(2*h),
                                   q['normal_gradient'].ravel()[j],delta=3e-5)

    def test_cross_knot_complete_force_is_continuous(self):
        old=self.configuration(-.002);errors=[]
        for h in (1e-5,1e-6,1e-7,1e-8):
            a=self.data(self.configuration(-h),old)
            b=self.data(self.configuration(h),old)
            errors.append(np.linalg.norm(a['gradient']-b['gradient']))
        self.assertLess(errors[-1],.002*errors[0])
        self.assertTrue(all(b<a for a,b in zip(errors,errors[1:])))

    def test_normal_motion_and_material_stretch_do_not_reset_identity(self):
        old=self.configuration(.002)
        for x in (self.configuration(.002,gap=.002),
                  self.configuration(.002,nodes=self.nodes*np.array([1.3,.9]))):
            self.assertLess(abs(self.data(x,old)['slip_mm']),1e-14)

    def test_curved_finite_gap_is_not_claimed_equivalent_to_point_coulomb(self):
        old=self.configuration(-.001);x=self.configuration(.001)
        q=self.data(x,old)
        self.assertGreater(abs(q['slave_tangential_metric']-1),.01)
        # The work-conjugate surface force differs from a point traction at
        # a finite gap. This is an explicit unqualified modeling approximation.
        self.assertNotIn('jacobian',q)
        fine=self.data(self.configuration(.001,gap=1e-6),self.configuration(-.001,gap=1e-6))
        self.assertLess(abs(fine['slave_tangential_metric']-1),1e-3)

    def test_ambiguous_endpoint_new_contact_and_signed_weight_fail_closed(self):
        flat=np.column_stack((self.u,np.zeros(len(self.u))))
        with self.assertRaisesRegex(ValueError,'Endpoint'):
            self.chart.project(np.array([.03,.001]),flat)
        symmetric=np.column_stack((self.u,150*self.u**2))
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            self.chart.project(np.array([0.,.01]),symmetric)
        with self.assertRaisesRegex(ValueError,'event history'):
            self.data(self.configuration(.001),self.configuration(-.001,gap=.003))
        with self.assertRaisesRegex(ValueError,'weight'):
            surface_contact(self.chart,self.configuration(.001),self.configuration(-.001),
                            **dict(self.kw,weight=-.01))


if __name__=='__main__':unittest.main()
