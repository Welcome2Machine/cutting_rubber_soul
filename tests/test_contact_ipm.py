"""Independent physical and algebraic checks for the replacement contact core."""
import unittest
import numpy as np
from scipy.sparse import csc_matrix, diags
from prototype.contact_ipm import solve_affine, cone_slack, cone_derivative, positive_metric, InteriorFailure


class InteriorContactTests(unittest.TestCase):
    def test_exact_frictionless_limit_ignores_tangential_motion(self):
        for gap in (-.1,.1):
            trace=[]
            x,lam,_=solve_affine(diags([2.,3.]),np.zeros(2),diags([1.,1.]),
                np.array([[gap,100.]]),np.ones(1),0.,initial_traction=np.array([[.3,.2]]),observer=trace.append)
            np.testing.assert_allclose(lam,[[max(0.,2*gap),0.]],atol=1e-7)
            self.assertEqual(lam[0,1],0.)
            self.assertAlmostEqual(x[1],0.,places=12)
            self.assertTrue(all(e['minimum_primal']>0 and e['minimum_dual']>0 for e in trace))

    def test_open_stick_and_both_sliding_directions_without_dilatancy(self):
        k=diags([2.,3.]);j=diags([1.,1.]);mu=.35
        for gap,slip,expected in [(-.1,.1,[0.,0.]),(.1,.02,[.2,.06]),
                                  (.1,.1,[.2,.07]),(.1,-.1,[.2,-.07])]:
            with self.subTest(gap=gap,slip=slip):
                trace=[]
                x,lam,info=solve_affine(k,np.zeros(2),j,np.array([[gap,slip]]),np.ones(1),mu,observer=trace.append)
                np.testing.assert_allclose(lam[0],expected,atol=2e-7)
                np.testing.assert_allclose(k@x+lam.ravel(),0.,atol=1e-6)
                if gap>0:self.assertAlmostEqual(gap+x[0],0.,places=8)
                self.assertTrue(all(e['minimum_primal']>0 and e['minimum_dual']>0 for e in trace))
                self.assertGreaterEqual(float(lam[0,1]*(slip+x[1])),-1e-12)

    def test_de_saxce_derivative_and_exact_cone_complementarity(self):
        j=csc_matrix([[1.,.2],[-.3,.8],[.7,.1],[.6,-.4]])
        offset=np.array([[.03,.02],[-.1,-.05]]);v=np.array([.3,-.2]);eps=1e-7
        actual=(cone_slack(offset+(eps*(j@v)).reshape(2,2),.35)-cone_slack(offset-(eps*(j@v)).reshape(2,2),.35))/(2*eps)
        np.testing.assert_allclose(actual,cone_derivative(offset,j,.35)@v,rtol=1e-8,atol=1e-9)
        for slip,shear in ((.2,.35),(-.2,-.35)):
            r=np.array([1-shear/.35,1+shear/.35])
            np.testing.assert_allclose(r*cone_slack(np.array([[0.,slip]]),.35),0.,atol=1e-15)

    def test_redundant_contact_constraints_share_load(self):
        j=csc_matrix(np.tile(np.eye(2),(2,1)))
        x,lam,_=solve_affine(diags([2.,3.]),np.zeros(2),j,np.tile([.1,.1],(2,1)),np.array([.5,.5]),.35)
        # Redundant multipliers need not be unique; only their resultant is.
        np.testing.assert_allclose(lam.mean(axis=0),[.2,.07],atol=2e-7)
        self.assertTrue(np.all(abs(lam[:,1])<=.35*lam[:,0]+1e-12))
        np.testing.assert_allclose([.1+x[0],.1+x[1]],[0.,.1-.07/3],atol=1e-8)

    def test_coupled_manufactured_solution_and_area_units(self):
        rng=np.random.default_rng(234)
        a=rng.normal(size=(6,6));k=csc_matrix(a.T@a+3*np.eye(6))
        j=csc_matrix(np.eye(6)+.05*rng.normal(size=(6,6)));w=np.array([.2,2.,.7])
        expected_x=rng.normal(size=6)*.01
        expected_lam=np.array([[.2,.02],[.3,.105],[0.,0.]])
        kin=np.array([[0.,0.],[0.,.08],[-.1,.1]])
        offset=kin-(j@expected_x).reshape(3,2)
        force=-k@expected_x-j.T@(np.repeat(w,2)*expected_lam.ravel())
        saved=[v.copy() for v in (force,offset,w,expected_lam)]
        x,lam,_=solve_affine(k,force,j,offset,w,.35,initial_traction=expected_lam)
        np.testing.assert_allclose(x,expected_x,atol=1e-7)
        np.testing.assert_allclose(lam,expected_lam,atol=2e-6)
        for actual,old in zip((force,offset,w,expected_lam),saved):np.testing.assert_array_equal(actual,old)

    def test_positive_metric_is_certified_and_does_not_modify_hessian(self):
        k=csc_matrix([[1.,3.],[3.,-2.]])
        old=k.toarray().copy();metric,info=positive_metric(k)
        self.assertGreater(np.linalg.eigvalsh(metric.toarray()).min(),0.)
        self.assertGreater(info['metric_shift'],0.)
        np.testing.assert_array_equal(k.toarray(),old)
        with self.assertRaisesRegex(ValueError,'symmetric'):
            positive_metric(csc_matrix([[1.,2.],[1.,2.]]))
        # Positive definite need not be diagonally dominant: keep this Hessian.
        k=csc_matrix([[2.,3.],[3.,5.]])
        metric,info=positive_metric(k)
        self.assertEqual(info['metric_shift'],0.)
        np.testing.assert_allclose(metric.toarray(),k.toarray(),atol=1e-15)

    def test_budget_exhaustion_is_not_success(self):
        with self.assertRaises(InteriorFailure):
            solve_affine(diags([2.,3.]),np.zeros(2),diags([1.,1.]),np.array([[.1,.1]]),np.ones(1),.35,max_iterations=1)

    def test_finite_strain_outer_compression_and_original_audit(self):
        self.check_finite_strain_compression(.35)

    def test_frictionless_finite_strain_outer_compression(self):
        self.check_finite_strain_compression(0.)

    def test_geometry_gate_retries_equilibrium_without_bypassing_audit(self):
        self.check_finite_strain_compression(0.,geometry_gate=True)

    def check_finite_strain_compression(self,mu,geometry_gate=False):
        import contextlib,io
        from unittest.mock import patch
        from prototype.quadratic import QuadraticProblem
        from prototype.model_config import FEMConfig
        from prototype.initiation import InitiationOptions,ContactResolutionError
        from prototype.material import response
        from prototype.contact_ipm import interior_step
        c=FEMConfig(length_mm=1,height_mm=1,width_mm=1,drop_s_mm=.5,inclination_deg=0,
            mesh_size_mm=.25,max_indentation_mm=.3,friction_coefficient=mu,
            newton_tolerance_n=1e-6,contact_multiplier_tolerance_n_mm2=1e-6)
        p=QuadraticProblem(c,InitiationOptions(refinement_mm=.25,surface_spacing_mm=.125,
            mesh_grading='graded',contact_discretization='quadratic_integral',volumetric_discretization='quadratic_full'))
        select=np.all(np.isclose(p.mesh.reference_xy[p.edges,1],1),axis=1)
        p.mesh.boundary_midpoints=p.mesh.boundary_midpoints[select]
        p.mesh.macro_boundary_edges=p.edges[select];p._rebuild_boundary(p.edges[select])
        p.fixed=np.unique(np.r_[2*np.arange(p.mesh.node_count),2*p.mesh.bottom_nodes+1])
        p.free=np.setdiff1d(np.arange(p.mesh.dof_count),p.fixed)
        def plane(x,c,o,depth,with_tangent=False):
            value=(x[:,1]-1+depth,np.tile([0.,-1.],(len(x),1)))
            return value+(np.zeros((len(x),2,2)),) if with_tangent else value
        previous=np.zeros(p.mesh.dof_count);history=p.initial_contact_state();old=0.
        original_audit=p._mixed_step
        checks=[]
        def gate(*args,**kwargs):
            # Exercise an independent acceptance veto after force convergence;
            # all returned states must still pass the real mechanical audit.
            r,_,_,_=p.assemble(kwargs['initial'],args[0],args[2],args[1],args[3],
                kwargs['initial_multipliers'][:,0],kwargs['initial_multipliers'][:,1],
                with_tangent=False,roundoff_guard=False)
            checks.append(float(np.linalg.norm(r[p.free])))
            if len(checks)==1:
                raise ContactResolutionError('independent geometry veto',[0],u=kwargs['initial'],
                    multipliers=kwargs['initial_multipliers'],depth=args[2])
            return original_audit(*args,**kwargs)
        if geometry_gate:p._mixed_step=gate
        with patch('prototype.initiation.rounded_blade_distance',plane),contextlib.redirect_stdout(io.StringIO()):
            for depth in (.01,.1):
                saved=previous.copy();saved_history=history.copy()
                seed={}
                if geometry_gate and old==0.:
                    exact=np.zeros_like(previous)
                    exact[1::2]=-depth*p.mesh.reference_xy[:,1]
                    traction=history.copy()
                    traction[:,0]=-response(np.array([[[1.,0.],[0.,1-depth]]]),c,tangent=False)['P'][0,1,1]
                    interior=p.free[(p.mesh.reference_xy[p.free//2,1]>.1)&(p.mesh.reference_xy[p.free//2,1]<.9)][0]
                    perturbed=exact.copy();perturbed[interior]+=1e-8
                    residual=p.assemble(perturbed,previous,depth,old,.1,traction[:,0],traction[:,1],with_tangent=False)[0]
                    perturbed=exact+(perturbed-exact)*(.5*c.newton_tolerance_n/np.linalg.norm(residual[p.free]))
                    seed=dict(initial=perturbed,initial_multipliers=traction)
                u,lam,info=interior_step(p,previous,old,depth,.1,history,trust_radius=.1,seconds=15.,**seed)
                r,_,q,d=p.assemble(u,previous,depth,old,.1,lam[:,0],lam[:,1],with_tangent=False,roundoff_guard=False)
                self.assertLessEqual(np.linalg.norm(r[p.free]),c.newton_tolerance_n)
                self.assertLessEqual(np.max(abs(np.column_stack((q['pressure'],q['shear']))-lam)),c.contact_multiplier_tolerance_n_mm2)
                self.assertLessEqual(p.edge_penetration(u,depth),c.contact_penetration_tolerance_mm)
                expected=-response(np.array([[[1.,0.],[0.,1-depth]]]),c,tangent=False)['P'][0,1,1]
                self.assertAlmostEqual(info['reaction_fz_n'],expected,places=5)
                np.testing.assert_allclose(u.reshape(-1,2)[:,1],-depth*p.mesh.reference_xy[:,1],atol=1e-7)
                np.testing.assert_array_equal(previous,saved);np.testing.assert_array_equal(history,saved_history)
                previous,history,old=u,lam,depth
        if geometry_gate:
            self.assertGreaterEqual(len(checks),3)
            self.assertLess(checks[1],checks[0])
            # A persistent geometry veto at an already equilibrated state must
            # remain an explicit failure, never be converted into success.
            saved_u=previous.copy();saved_history=history.copy()
            with patch('prototype.initiation.rounded_blade_distance',plane), \
                    patch.object(p,'_mixed_step',side_effect=ContactResolutionError('persistent veto',[0])), \
                    contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ContactResolutionError):
                interior_step(p,previous,old,old,.1,history,seconds=15.)
            np.testing.assert_array_equal(previous,saved_u)
            np.testing.assert_array_equal(history,saved_history)


if __name__=='__main__':unittest.main()
