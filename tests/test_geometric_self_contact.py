"""Physical/derivative checks for the explicit 2D GCP + self friction adapter."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from prototype.ipc_reference import toolkit
from prototype.geometric_self_contact import GeometricSelfContact,orient_boundary,slip_mollifier


def shape(gap):
    return np.array([[-.01,0],[.01,0],[.01,.02],[gap/2,.02],[gap/2,.006],[-gap/2,.006],[-gap/2,.02],[-.01,.02]])


class GeometricContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ipc=toolkit();cls.ipc.set_num_threads(1)

    def make(self,mu=.35,rotation=None,scale=1.,reference=None):
        R=shape(.006) if reference is None else reference.copy()
        if rotation is not None:R=R@rotation.T
        E=np.array([[i,(i+1)%8] for i in range(8)])
        return GeometricSelfContact(self.ipc,R,E,dhat=.0025,stiffness_mpa=6.6,
                                    width_mm=20.*scale,mu=mu,eps=1e-5)

    def test_reference_support_and_closed_orientation_fail_closed(self):
        p=self.make();e,g,h,w=p.terms(p.reference,p.reference,tangent=True)
        self.assertEqual(e,0);self.assertEqual(np.linalg.norm(g),0);self.assertEqual(h.nnz,0)
        with self.assertRaisesRegex(ValueError,'Nonuniform'):self.make(reference=shape(.004))
        E=p.edges.copy();E[0]=E[0,::-1]
        with self.assertRaisesRegex(ValueError,'oriented'):
            GeometricSelfContact(self.ipc,p.reference,E,dhat=.0025,stiffness_mpa=6.6,width_mm=20,mu=.35,eps=1e-5)

    def test_reference_orientation_not_deformed_vertex_chord(self):
        R=np.array([[0.,0.],[1.,0.],[0.,1.]])
        E=orient_boundary(R,np.array([[0,1,2]]),np.array([[1,0],[1,2],[0,2]]))
        np.testing.assert_array_equal(E,[[0,1],[1,2],[2,0]])

    def test_native_stencil_scatter_matches_whole_potential(self):
        p=self.make(0);x=shape(.001)+np.random.default_rng(41).normal(size=(8,2))*2e-5
        c=self.ipc.SmoothCollisions();c.build(p.mesh,x,p.params,False)
        self.assertGreater(len(c),0);e=0.;g=np.zeros(x.size);h=np.zeros((x.size,x.size))
        for i in range(len(c)):
            pair=c[i];ids,ev=p.stencil(pair);local=x[ids].ravel();dofs=(2*ids[:,None]+np.arange(2)).ravel()
            e+=p.potential(pair,local);np.add.at(g,dofs,p.potential.gradient(pair,local))
            H=p.potential.hessian(pair,local);np.add.at(h,(dofs[:,None],dofs[None,:]),H)
        self.assertAlmostEqual(e,p.potential(c,p.mesh,x),places=14)
        np.testing.assert_allclose(g,p.potential.gradient(c,p.mesh,x),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(h,p.potential.hessian(c,p.mesh,x).toarray(),rtol=1e-12,atol=1e-10)

    def test_compression_dissipation_cone_and_ccd(self):
        p=self.make();energies=[]
        for gap in (.004,.002,.001,.0005):
            x=shape(gap);e,g,h,w=p.terms(x,x,tangent=True);energies.append(e)
            self.assertAlmostEqual(w,0);self.assertFalse(self.ipc.has_intersections(p.mesh,x))
        self.assertEqual(energies[0],0);self.assertTrue(np.all(np.diff(energies)>0))
        bound=self.ipc.compute_collision_free_stepsize(p.mesh,shape(.001),shape(-.001))
        self.assertGreater(bound,0);self.assertLess(bound,.5)
        x=shape(.001);old=x.copy();old[2:5,1]-=1e-4
        e,g,h,w=p.terms(x,old,tangent=True)
        self.assertGreater(w,0);self.assertLessEqual(p.last['cone_violation_n'],1e-12)
        np.testing.assert_allclose(g.reshape(-1,2).sum(axis=0),0,atol=1e-10)

    def test_full_refreshed_friction_tangent_on_smooth_branches(self):
        p=self.make();d=np.random.default_rng(18).normal(size=(8,2));d/=np.linalg.norm(d)
        for seed in (41,42,43):
            x=shape(.001)+np.random.default_rng(seed).normal(size=(8,2))*2e-5
            for slip in (2e-6,5e-5):
                old=x.copy();old[2:5,1]-=slip
                _,g,H,w=p.terms(x,old,tangent=True);eps=1e-9
                gp=p.terms(x+eps*d,old,tangent=False)[1];gm=p.terms(x-eps*d,old,tangent=False)[1]
                want=H@d.ravel();err=np.linalg.norm((gp-gm)/(2*eps)-want)/np.linalg.norm(want)
                self.assertLess(err,3e-5,(seed,slip,err));self.assertGreaterEqual(w,0)

    def test_covariance_width_scaling_and_no_history_mutation(self):
        p=self.make();x=shape(.001)+np.random.default_rng(41).normal(size=(8,2))*2e-5
        old=x.copy();old[2:5,1]-=3e-6;before=old.copy();e,g,H,w=p.terms(x,old,tangent=True)
        phi=.63;rot=np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        q=self.make(rotation=rot);er,gr,Hr,wr=q.terms(x@rot.T,old@rot.T,tangent=True)
        np.testing.assert_allclose(gr.reshape(-1,2),g.reshape(-1,2)@rot.T,rtol=2e-9,atol=1e-10)
        self.assertAlmostEqual(er,e,places=10);self.assertAlmostEqual(wr,w,places=12)
        q=self.make(scale=2);es,gs,Hs,ws=q.terms(x,old,tangent=True)
        np.testing.assert_allclose(gs,2*g);self.assertAlmostEqual(es,2*e);self.assertAlmostEqual(ws,2*w)
        # Arbitrary rejected evaluations cannot mutate committed material points.
        p.terms(x+1e-6,old,tangent=False);np.testing.assert_array_equal(old,before)
        np.testing.assert_allclose(p.terms(x,old,tangent=False)[1],g,rtol=1e-13,atol=1e-12)

    def test_switch_is_continuous_first_order_not_assumed_C2(self):
        p=self.make(0);x=shape(.001);d=np.random.default_rng(18).normal(size=x.shape);d/=np.linalg.norm(d)
        e,g,_,_=p.terms(x,x,tangent=True);errors=[]
        for eps in (1e-7,1e-9):
            ep=p.terms(x+eps*d,x,tangent=False)[0];em=p.terms(x-eps*d,x,tangent=False)[0]
            errors.append(abs((ep-em)/(2*eps)-g@d.ravel())/abs(g@d.ravel()))
        self.assertLess(errors[-1],1e-5);self.assertLess(errors[-1],errors[0]/50)

    def test_slip_regularization_is_bounded_dissipative(self):
        for s in np.linspace(-2e-5,2e-5,51):
            phi,d,dd=slip_mollifier(s,1e-5)
            self.assertGreaterEqual(phi,0);self.assertGreaterEqual(d*s,0);self.assertGreaterEqual(dd,0);self.assertLessEqual(abs(d),1)

    def test_nonzero_friction_force_is_continuous_at_feature_tie(self):
        p=self.make();x=shape(.001);old=x.copy();old[2:5,1]-=3e-6
        d=np.random.default_rng(18).normal(size=x.shape);d/=np.linalg.norm(d)
        reference=p.terms(x,old,tangent=False)[1];errors=[]
        for eps in (1e-7,1e-9):
            errors.append(max(np.linalg.norm(p.terms(x+sign*eps*d,old,tangent=False)[1]-reference)
                              for sign in (-1,1))/np.linalg.norm(reference))
        self.assertLess(errors[-1],2e-4);self.assertLess(errors[-1],errors[0]/40)

    def test_fixed_loading_does_not_retry_or_subdivide(self):
        from tools.run_augmented_blade import fixed_load_step
        p=SimpleNamespace(step=Mock(side_effect=RuntimeError('stop')));reject=Mock();x=np.zeros((2,2))
        with self.assertRaisesRegex(RuntimeError,'stop'):list(fixed_load_step(p,x,0.,.005,100.,reject))
        self.assertEqual(p.step.call_count,1);self.assertEqual(reject.call_count,1)
        self.assertEqual(reject.call_args.args[0]['policy'],'fixed-no-cutback')

    def test_nested_contact_model_resolves_predicted_activation(self):
        from scipy.sparse import eye,diags
        from scipy.sparse.linalg import spsolve
        from prototype.al_globalization import GlobalizedAugmentedBladeProblem
        p=SimpleNamespace(free=np.array([0,1]),multipliers=np.zeros((1,2)),penalty=250.,
            config=SimpleNamespace(friction_coefficient=.35),linear_direction=lambda a,b:spsolve(a,b))
        contact=dict(kinematic=eye(2,format='csc'),force_lambda=diags([-1.,1.],format='csc'),
                     gap_average=np.array([.1]),slip_average=np.array([.3]))
        old=p.multipliers.copy()
        solution=GlobalizedAugmentedBladeProblem.contact_model_direction(p,eye(2,format='csc'),
            contact,np.array([1.,.1]),np.ones(2),np.zeros(4))
        self.assertIsNotNone(solution);self.assertGreaterEqual(solution[1],2)
        np.testing.assert_allclose(solution[0],[-.1,-.3,.9,.2],atol=1e-12)
        np.testing.assert_array_equal(p.multipliers,old)


if __name__=='__main__':unittest.main()
