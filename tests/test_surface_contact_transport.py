"""Transported Coulomb traction, fresh tangent and trial-only event history."""
import unittest
import numpy as np
from prototype.surface_history import SurfaceChart,surface_contact
from prototype.surface_contact_transport import (initial_history,evaluate_contact,
    commit_contact,corotated_path,ContactPathError)


class TransportContactTests(unittest.TestCase):
    def setUp(self):
        self.u=np.linspace(-.01,.01,5);self.chart=SurfaceChart(self.u)
        self.nodes=np.column_stack((self.u,.02-np.sqrt(.02**2-self.u**2)))
        self.kw=dict(weight=.01,dhat=.0025,stiffness=132.,mu=.35,epsilon=1e-5)

    def config(self,u,gap=.0015,nodes=None):
        nodes=self.nodes if nodes is None else nodes;c=self.chart.geometry(nodes)
        t=c(u,1);t=t/np.linalg.norm(t)
        return np.vstack((c(u)+gap*np.array([-t[1],t[0]]),nodes))

    def history(self,x):return initial_history(self.chart,x,dhat=self.kw['dhat'])

    def evaluate(self,x,old,**extra):
        return evaluate_contact(self.chart,x,old,**dict(self.kw,**extra))

    def test_projection_second_derivatives_and_metric_are_exact(self):
        x=self.config(.0013);q=self.chart.differential(x[0],x[1:]);h=1e-8
        for j,d in enumerate(np.eye(x.size).reshape((-1,)+x.shape)):
            a=self.chart.differential((x+h*d)[0],(x+h*d)[1:])
            b=self.chart.differential((x-h*d)[0],(x-h*d)[1:])
            np.testing.assert_allclose((a['parameter_gradient']-b['parameter_gradient']).ravel()/(2*h),
                q['parameter_hessian'][:,j],atol=4e-6,rtol=2e-6)
            np.testing.assert_allclose((a['gap_gradient']-b['gap_gradient']).ravel()/(2*h),
                q['gap_hessian'][:,j],atol=4e-6,rtol=2e-6)
            self.assertAlmostEqual((a['transport_metric']-b['transport_metric'])/(2*h),q['metric_gradient'][j],delta=4e-6)

    def test_curved_finite_gap_preserves_point_coulomb_limit_and_tangency(self):
        old=self.config(-.002);x=self.config(.001)
        arc=surface_contact(self.chart,x,old,**self.kw)
        q=self.evaluate(x,self.history(old));t=q['slave_tangent']
        self.assertGreater(arc['slave_tangential_metric'],1.07)
        self.assertAlmostEqual(q['slave_tangential_metric'],1.,places=13)
        np.testing.assert_allclose(q['friction_gradient'][0],.35*q['normal_capacity_n']*t,atol=1e-12)
        self.assertGreater(q['friction_work_nmm'],0)

    def test_rigid_rotation_real_sliding_force_and_moment(self):
        old=self.config(-.002);x=self.config(.001);history=self.history(old)
        theta=.7;Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        rigid=self.evaluate(old@Q.T+[.1,-.2],history)
        self.assertLess(abs(rigid['slip_mm']),1e-14)
        self.assertLess(np.linalg.norm(rigid['friction_gradient']),1e-8)
        a=self.evaluate(x,history);b=self.evaluate(x@Q.T+[.1,-.2],history)
        np.testing.assert_allclose(b['gradient'],a['gradient']@Q.T,atol=1e-9,rtol=1e-8)
        for name in ('normal_gradient','friction_gradient','gradient'):
            g=a[name];np.testing.assert_allclose(g.sum(axis=0),0.,atol=1e-12)
            self.assertLess(abs(np.sum(g*np.column_stack((-x[:,1],x[:,0])))),1e-12)

    def check_jacobian(self,x,history,*,epsilon=1e-5):
        q=self.evaluate(x,history,epsilon=epsilon);h=2e-9
        fd=np.column_stack([((self.evaluate(x+h*d,history,epsilon=epsilon)['gradient']-
            self.evaluate(x-h*d,history,epsilon=epsilon)['gradient'])/(2*h)).ravel()
            for d in np.eye(x.size).reshape((-1,)+x.shape)])
        np.testing.assert_allclose(fd,q['jacobian'],rtol=3e-5,atol=5e-5)
        return q

    def test_full_nonsymmetric_jacobian_in_sliding_and_smoothing(self):
        old=self.config(.001);history=self.history(old)
        self.check_jacobian(self.config(.0015),history)
        self.check_jacobian(self.config(.001003),history)

    def test_cross_knot_complete_force_continuity(self):
        history=self.history(self.config(-.002));errors=[]
        for h in (1e-5,1e-6,1e-7,1e-8):
            errors.append(np.linalg.norm(self.evaluate(self.config(h),history)['gradient']-
                                         self.evaluate(self.config(-h),history)['gradient']))
        self.assertLess(errors[-1],.002*errors[0])

    def test_corotated_path_endpoint_and_time_derivatives(self):
        old=self.config(-.002);x=self.config(.001);x[2,1]+=.0005
        theta=.4;Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        x=x@Q.T+[.1,-.1];s=.37;y,v,J=corotated_path(old,x,s)
        h=1e-7
        np.testing.assert_allclose((corotated_path(old,x,s+h)[0]-corotated_path(old,x,s-h)[0])/(2*h),v,rtol=2e-6,atol=2e-9)
        for j,d in enumerate(np.eye(x.size).reshape((-1,)+x.shape)):
            fd=(corotated_path(old,x+h*d,s)[0]-corotated_path(old,x-h*d,s)[0]).ravel()/(2*h)
            np.testing.assert_allclose(fd,J[:,j],rtol=2e-6,atol=2e-9)

    def test_first_activation_uses_only_post_event_motion_and_full_event_derivative(self):
        flat=np.column_stack((self.u,np.zeros(len(self.u))))
        old=self.config(-.004,gap=.004,nodes=flat);x=self.config(.002,gap=.001,nodes=flat)
        history=self.history(old);q=self.check_jacobian(x,history,epsilon=.02)
        self.assertEqual(q['status'],'activated')
        self.assertAlmostEqual(q['activation']['fraction'],.5,places=12)
        self.assertAlmostEqual(q['slip_mm'],.003,places=12)
        self.assertAlmostEqual(q['activation']['parameter'],-.001,places=12)
        # Contact trials are pure; activation derivatives cannot be frozen away.
        self.assertFalse(history.active);self.assertEqual(history.generation,0)
        self.assertGreater(np.linalg.norm(q['activation']['parameter_gradient']),.01)

    def test_activation_path_is_objective_under_endpoint_rigid_motion(self):
        flat=np.column_stack((self.u,np.zeros(len(self.u))))
        old=self.config(-.004,gap=.004,nodes=flat);x=self.config(.002,gap=.001,nodes=flat)
        theta=.7;Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        history=self.history(old);a=self.evaluate(x,history,epsilon=.02)
        b=self.evaluate(x@Q.T+[.1,-.2],history,epsilon=.02)
        self.assertAlmostEqual(a['activation']['fraction'],b['activation']['fraction'],places=11)
        self.assertAlmostEqual(a['slip_mm'],b['slip_mm'],places=12)
        np.testing.assert_allclose(a['gradient']@Q.T,b['gradient'],atol=2e-9,rtol=2e-8)
        self.check_jacobian(x@Q.T+[.1,-.2],history,epsilon=.02)

    def test_commit_detach_recontact_rejection_and_stale_history(self):
        flat=np.column_stack((self.u,np.zeros(len(self.u))))
        old=self.config(-.002,nodes=flat);h=self.history(old)
        trial=self.evaluate(self.config(0.,nodes=flat),h)
        with self.assertRaisesRegex(ValueError,'equilibrium'):commit_contact(h,trial,equilibrium_accepted=False)
        h1=commit_contact(h,trial,equilibrium_accepted=True)
        with self.assertRaisesRegex(ValueError,'Stale'):commit_contact(h1,trial,equilibrium_accepted=True)
        detached=self.evaluate(self.config(-.004,gap=.004,nodes=flat),h1)
        np.testing.assert_array_equal(detached['gradient'],0)
        h2=commit_contact(h1,detached,equilibrium_accepted=True)
        self.assertIsNone(h2.parameter)
        trial2=self.evaluate(self.config(.002,gap=.001,nodes=flat),h2)
        self.assertAlmostEqual(trial2['slip_mm'],.003,places=12)
        self.assertEqual(trial2['status'],'activated')
        with self.assertRaises(ValueError):h2.positions[0,0]=1

    def test_hidden_entry_and_exit_is_rejected_instead_of_dropped(self):
        nodes=self.nodes*np.array([1.,-1.]);old=np.vstack(([-.007,.002],nodes));x=np.vstack(([.007,.002],nodes))
        history=self.history(old);self.assertFalse(history.active)
        with self.assertRaisesRegex(ContactPathError,'Multiple'):self.evaluate(x,history)

    def test_conservative_path_does_not_use_uniform_sample_absence_as_proof(self):
        nodes=np.column_stack((self.u,-20*self.u**2))
        old=np.vstack(([-.0075,.002499],nodes));x=np.vstack(([.0065,.002499],nodes))
        history=self.history(old)
        for fraction in np.linspace(0,1,17):
            y=corotated_path(old,x,fraction)[0]
            self.assertGreater(self.chart.project(y[0],y[1:])['gap'],self.kw['dhat'])
        # A narrow active interval lies strictly between those sample times.
        y=corotated_path(old,x,.0075/.014)[0]
        self.assertLess(self.chart.project(y[0],y[1:])['gap'],self.kw['dhat'])
        with self.assertRaises(ContactPathError):self.evaluate(x,history)

    def test_normal_motion_fixed_identity_stretch_and_detachment_limit(self):
        old=self.config(.001);h=self.history(old)
        a=self.evaluate(self.config(.001,gap=.002),h)
        self.assertLess(abs(a['slip_mm']),1e-14)
        b=self.evaluate(self.config(.001,nodes=self.nodes*np.array([1.1,.9])),h)
        self.assertLess(abs(b['slip_mm']),1e-14)
        forces=[np.linalg.norm(self.evaluate(self.config(.002,gap=.0025-eps),h)['gradient'])
                for eps in (1e-5,1e-6,1e-7)]
        self.assertLess(forces[-1],.001*forces[0])


if __name__=='__main__':unittest.main()
