"""Knife path history and complete FE assembly; no experimental fit claims."""
import json
import unittest
import numpy as np
from prototype.transported_blade import TransportedBladeProblem,transported_contact
from prototype.ipc_reference import LoadStepFailure


class TransportedBladeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=TransportedBladeProblem(maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2,
            bulk_modulus=23.34333333333331,shear_modulus=.14,local_spacing=.2,surface_spacing=.02,
            mu=.35,tip_radius=.02,dhat=.0025,inclination=0,contact_spacing=.0025)
        cls.p.material_dt=.05
        cls.geometry=cls.p.path_geometry;cls.center=cls.geometry.centers[-1]

    def point(self,x,old,epsilon=1e-5):
        return transported_contact(self.geometry,np.atleast_2d(x),np.atleast_2d(old),np.array([.01]),
            .0025,132.,.35,epsilon)

    def on_arc(self,theta,gap=.001):
        return self.center+(.02+gap)*np.array([np.cos(theta),np.sin(theta)])

    def flat(self):
        a,b=self.geometry.starts[2],self.geometry.ends[2];t=(b-a)/np.linalg.norm(b-a)
        n=np.array([t[1],-t[0]])
        return .5*(a+b),t,n

    def jacobian(self,x,old,epsilon=1e-5):
        # f1 is C1 but has a |s| term: zero-slip FD error is O(h/epsilon).
        # Column norms avoid dividing by individual near-zero cross entries.
        q=self.point(x,old,epsilon);errors=[];J=q['jacobian'][0]
        for h in (1e-8,1e-9,1e-10):
            fd=np.column_stack([(self.point(x+h*d,old,epsilon)['gradient'][0]-
                self.point(x-h*d,old,epsilon)['gradient'][0])/(2*h) for d in np.eye(2)])
            errors.append(float(np.max(np.linalg.norm(J-fd,axis=0)/np.maximum(np.linalg.norm(J,axis=0),1e-12))))
        self.assertLess(min(errors),1e-5)
        self.assertTrue(any(a<1e-4 and b<1e-4 for a,b in zip(errors,errors[1:])))
        return q

    def test_flat_and_first_activation_omit_precontact_motion(self):
        c,t,n=self.flat();old=c-.004*t+.004*n;x=c+.002*t+.001*n
        q=self.jacobian(x,old,epsilon=.02)
        self.assertAlmostEqual(q['slip_mm'][0],.003,places=10)
        self.assertAlmostEqual(q['events'][0]['start'],.5,places=10)
        self.assertEqual(q['events'][0]['status'],'activated')
        q=self.jacobian(c+.002*t+.001*n,c-.001*t+.001*n)
        self.assertAlmostEqual(q['slip_mm'][0],.003,places=10)

    def test_circle_stick_slip_and_zero_motion_full_derivative(self):
        for theta in (-2.4,-1.57,-.7):
            x=self.on_arc(theta)
            for angle in (0.,1e-4,-1e-4,.01):
                old=self.on_arc(theta-angle)
                q=self.jacobian(x,old)
                self.assertGreaterEqual(q['friction_work'],-1e-16)
                self.assertLessEqual(np.linalg.norm(q['friction_gradient'][0]),.35*q['normal_capacity_n'][0]*(1+1e-12))

    def test_circle_flank_smoothing_force_is_continuous(self):
        a=np.deg2rad(self.p.config.blade_tip_angle_deg/2);n=np.array([np.cos(-a),np.sin(-a)])
        t=np.array([-n[1],n[0]]);old=self.on_arc(-a-1e-4)
        errors=[]
        for h in (1e-6,1e-7,1e-8,1e-9):
            left=self.on_arc(-a-h/.021);right=self.center+.021*n+h*t
            errors.append(np.linalg.norm(self.point(left,old)['gradient']-self.point(right,old)['gradient']))
        self.assertLess(errors[-1],.002*errors[0])
        self.jacobian(self.center+.021*n+1e-7*t,old)
        self.jacobian(self.on_arc(-a-1e-7/.021),old)

    def test_intermediate_contact_and_penetration_are_not_skipped(self):
        old=self.center+[-.04,-.021];x=self.center+[.04,-.021]
        with self.assertRaisesRegex(LoadStepFailure,'excursion'):self.geometry.path(old,x,.0025)
        with self.assertRaisesRegex(LoadStepFailure,'crosses'):self.geometry.path(old+[0,.01],x+[0,.01],.0025)

    def test_relative_frame_zero_motion_translation_and_detach(self):
        x=self.on_arc(-1.5);q=self.point(x,x)
        self.assertLess(np.linalg.norm(q['friction_gradient']),1e-12)
        # Superposed rigid endpoint poses give the same material coordinates in
        # the rigid blade frame. The FE host supports translation only.
        angle=.7;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        global_point=x@Q.T+[.2,-.4];back=(global_point-[.2,-.4])@Q
        q=self.point(back,x);self.assertLess(np.linalg.norm(q['friction_gradient']),1e-9)
        far=self.on_arc(-1.5,gap=.004)
        self.assertEqual(self.geometry.path(x,far,.0025)['status'],'detached')
        np.testing.assert_array_equal(self.point(far,x)['gradient'],0.)

    def test_fixed_reference_area_and_no_duplicate_material_integral(self):
        p=self.p;perimeter=np.linalg.norm(np.diff(p.rest[p.body_edges],axis=1)[:,0],axis=1).sum()
        self.assertAlmostEqual(p.weights.sum(),perimeter,places=10)
        np.testing.assert_allclose(np.asarray(p.Q.sum(axis=1)).ravel(),1.,atol=1e-15)
        self.assertEqual(len({tuple(zip(p.Q[i].indices,p.Q[i].data)) for i in range(p.Q.shape[0])}),p.Q.shape[0])
        np.testing.assert_allclose(p.D@np.tile([.7,-.4],len(p.rest)),0.,atol=1e-15)

    def test_complete_assembled_fresh_jacobian_with_actual_activation(self):
        p=self.p;p.previous=p.rest.copy();x=p.rest.copy()
        x[:p.n,1]-=.007*p.mesh.local_sn[:,1]/p.config.height_mm;x[p.n:,1]-=.01
        self.assertEqual(p.safe_step(p.previous,x-p.previous),1.)
        _,g,A,_=p.evaluate(x,projected=False)
        data,_,_=p._blade_data(x);self.assertGreater(len(data['events']),0)
        rng=np.random.default_rng(77);d=np.zeros_like(x);d[:p.n]=rng.normal(size=(p.n,2));d[p.mesh.bottom_nodes]=0
        # Translation of the whole rigid blade is the only allowed blade DOF.
        d[p.n:]=[.1,-.2];d/=np.max(abs(d));action=A@d.ravel();errors=[]
        for h in (1e-7,1e-8,1e-9):
            gp=p.evaluate(x+h*d,tangent=False)[1];gm=p.evaluate(x-h*d,tangent=False)[1]
            errors.append(np.linalg.norm((gp-gm)/(2*h)-action)/np.linalg.norm(action))
        self.assertLess(min(errors),1e-5)
        self.assertGreaterEqual(sum(e<1e-4 for e in errors),2)
        self.assertLess(np.linalg.norm(g.reshape(-1,2).sum(axis=0)),1e-10)

    def test_native_self_contact_retained_and_history_policy_guard(self):
        p=self.p;x=p.rest.copy();edge=p.body_edges[0]
        point=next(v for v in p.body_surface if v not in edge and
            all(v not in e for e in p.body_edges if any(k in e for k in edge)))
        a,b=x[edge];t=b-a;n=np.array([-t[1],t[0]])/np.linalg.norm(t);x[point]=(a+b)/2+.001*n
        self.assertGreater(p.barrier(p._self_collisions(x),p.self_mesh,x[p.body_surface]),0.)
        state=p.checkpoint_data();state['blade_contact_policy_json']=np.array(json.dumps({'wrong':True}))
        with self.assertRaisesRegex(ValueError,'policy'):p.audit_state(state,{})


if __name__=='__main__':unittest.main()
