"""Independent finite-strain, mixed algebra, geometry and IPC adapter checks."""
import unittest
import numpy as np
from scipy.sparse import eye
from threadpoolctl import threadpool_limits
from prototype.ipc_mini import MiniElement
from prototype.pressure_pairing import ContinuousP1Volume
from prototype.model_config import FEMConfig
from tests.test_ipc_p2 import rectangle, assembled


class MiniTests(unittest.TestCase):
    def setUp(self):
        self.pool = threadpool_limits(2)
        self.el = MiniElement(rectangle(.5), FEMConfig(width_mm=1.))
        self.nv = ContinuousP1Volume(self.el)

    def tearDown(self):
        self.pool.restore_original_limits()

    def evaluate(self, u, tangent=True):
        r,k,d,e,j = self.el.bulk(u, with_tangent=tangent)
        r,k,B,e = self.nv.correct(u,r,k,e)
        g,A,_,_ = assembled((r,k,d,e,j),len(u))
        H = None if B is None else A.toarray()+B.T@self.nv._mass_factor.solve(B.toarray())
        return e,g,H,A,B

    def test_finite_strain_derivatives_and_rigid_motion(self):
        el=self.el; xy=el.mesh.reference_xy
        rng=np.random.default_rng(44)
        u=(xy@np.array([[.12,.2],[-.08,-.14]])).ravel()+rng.normal(size=xy.size)*.002
        e,g,H,A,B=self.evaluate(u)
        v=rng.normal(size=len(u));v/=np.linalg.norm(v);h=2e-7
        ep,gp,*_=self.evaluate(u+h*v,False);em,gm,*_=self.evaluate(u-h*v,False)
        self.assertAlmostEqual((ep-em)/(2*h),g@v,delta=2e-8)
        np.testing.assert_allclose((gp-gm)/(2*h),H@v,atol=2e-7,rtol=2e-6)
        np.testing.assert_allclose(H,H.T,atol=2e-12)
        a=.8;Q=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        ur=(xy+u.reshape(-1,2))@Q.T-xy
        ur[:el.base_nodes]+=[.35,-.41]  # translation applies only to actual vertices
        er,gr,*_=self.evaluate(ur.ravel(),False)
        self.assertAlmostEqual(er,e,delta=1e-11)
        np.testing.assert_allclose(gr.reshape(-1,2),g.reshape(-1,2)@Q.T,atol=1e-11)

    def test_condensation_matches_full_nonsymmetric_system(self):
        from prototype.mixed_system import solve_mixed
        el=self.el;xy=el.mesh.reference_xy
        u=(xy@np.array([[.1,.2],[0.,-.1]])).ravel()
        _,g,H,A,B=self.evaluate(u)
        # Nonsymmetric boundary contribution, with no fictitious bubble contact.
        A=A.tolil();A[0,3]+=.21;A=A.tocsc()
        H[0,3]+=.21
        rhs=np.random.default_rng(83).normal(size=len(u));metric=50*eye(len(u));shift=.4
        du,info=solve_mixed(A,metric,B,self.nv.pressure_mass,self.nv._mass_factor.solve,
                           shift,rhs,el.bubble_dofs)
        exact=np.linalg.solve(A.toarray()+shift*metric.toarray()+(1+shift)*(H-A.toarray()),rhs)
        np.testing.assert_allclose(du,exact,atol=2e-13)
        self.assertLess(info['factored_unknowns'],info['full_unknowns'])
        self.assertIsNone(info['fallback'])

    def test_homogeneous_material_and_pressure(self):
        xy=self.el.mesh.reference_xy;F=np.array([[1.2,.3],[.1,.8]])
        u=(xy@(F-np.eye(2)).T).ravel()
        r,k,d,e,j=self.el.bulk(u)
        rm,km,B,em=self.nv.correct(u,r,k,e)
        self.assertAlmostEqual(e,em,places=11)
        np.testing.assert_allclose(rm,r,atol=1e-12)
        np.testing.assert_allclose(self.nv.last['pressure_nodes'],self.nv.K*(np.linalg.det(F)-1),atol=2e-13)
        self.assertLess(np.max(np.abs(r[:,-2:])),1e-12)

    def test_reused_factor_solves_current_condensed_system(self):
        from prototype.mixed_system import solve_mixed
        from prototype.linear_solve import ReusedLUSolver
        el=self.el; u=(el.mesh.reference_xy@np.array([[.1,.2],[0.,-.1]])).ravel()
        _,g,H,A,B=self.evaluate(u)
        metric=50*eye(len(u)); solver=ReusedLUSolver(); methods=[]
        for shift in (.4,.4001,.8):
            rhs=np.random.default_rng(19).normal(size=len(u))
            direct,_=solve_mixed(A,metric,B,self.nv.pressure_mass,
                                self.nv._mass_factor.solve,shift,rhs,el.bubble_dofs)
            reused,info=solve_mixed(A,metric,B,self.nv.pressure_mass,
                                   self.nv._mass_factor.solve,shift,rhs,el.bubble_dofs,solver)
            np.testing.assert_allclose(reused,direct,atol=2e-10,rtol=1e-9)
            methods.append(info['linear']['method'])
        self.assertIn('gmres_reused_lu',methods)

    def test_mixed_residual_correction_checks_physical_system(self):
        from prototype.mixed_system import solve_mixed
        from scipy.sparse.linalg import splu
        class PerturbedSolve:
            def solve(self,A,b):
                x=splu(A).solve(b);x[0]+=1e-7
                return x,dict(method='injected_backward_error_for_test')
        el=self.el;u=np.zeros(el.mesh.dof_count)
        _,g,H,A,B=self.evaluate(u);metric=50*eye(len(u));shift=.4
        rhs=np.random.default_rng(83).normal(size=len(u))
        exact,_=solve_mixed(A,metric,B,self.nv.pressure_mass,self.nv._mass_factor.solve,
                           shift,rhs,el.bubble_dofs)
        fixed,info=solve_mixed(A,metric,B,self.nv.pressure_mass,self.nv._mass_factor.solve,
                              shift,rhs,el.bubble_dofs,PerturbedSolve())
        self.assertTrue(info['physical_residual_refinements'])
        self.assertLessEqual(info['physical_linear_residual_n'],info['residual_limit_n'])
        np.testing.assert_allclose(fixed,exact,atol=2e-12)

    def test_fused_mixed_assembly_matches_independent_correction(self):
        el=self.el;xy=el.mesh.reference_xy
        u=(xy@np.array([[.15,.35],[-.06,-.3]])).ravel()
        u[el.bubble_dofs.ravel()]=np.random.default_rng(29).normal(size=el.bubble_dofs.size)*.003
        for tangent in (False,True):
            r,k,d,e,j=el.bulk(u,with_tangent=tangent)
            r,k,B,e=self.nv.correct(u,r,k,e)
            rf,kf,df,ef,jf=self.nv.bulk(u,with_tangent=tangent)
            np.testing.assert_allclose(rf,r,atol=3e-13)
            np.testing.assert_array_equal(d,df)
            self.assertAlmostEqual(e,ef,delta=2e-13)
            self.assertEqual(j,jf)
            if tangent:
                np.testing.assert_allclose(kf,k,atol=4e-12)
                np.testing.assert_allclose(self.nv.pressure_coupling.toarray(),B.toarray(),atol=2e-14)

    def test_real_j_inside_element_and_time_path(self):
        el=self.el;u=np.zeros(el.mesh.dof_count)
        u[el.bubble_dofs[0,0]]=.4
        F=el.deformation(u,el.vertex_grad)
        self.assertTrue(np.all(np.linalg.det(F[:,:3])>0))
        self.assertLess(el.minimum_j(u),0)
        with self.assertRaises(FloatingPointError):el.bulk(u)
        # Rotation through a singular affine straight-line path: endpoints det=1.
        d=(-2*el.mesh.reference_xy).ravel()
        a=el.safe_step(np.zeros_like(d),d)
        self.assertLess(a,.5)
        for t in np.linspace(0,a,9):self.assertGreater(el.minimum_j(t*d),0)

    def test_pressure_spectrum_two_meshes(self):
        from tools.check_pressure_pairing import spectrum
        for n in (3,6):
            for F in (np.eye(2),np.array([[1.25,.5],[.08,.7]])):
                row=spectrum(n,'mini',F)
                self.assertEqual(row['nullity'],1)
                self.assertGreater(row['beta'],.05)

    def test_original_contact_trace_and_internal_predictor(self):
        from prototype.ipc_reference import IPCProblem
        from prototype.tip_contact import TipContactTracker
        kw=dict(mu=.35,local_spacing=.4,surface_spacing=.05,fine_halfwidth=.3,
                inclination=60,tip_radius=.02,nose_facets=64,dhat=.0025)
        try:
            p1=IPCProblem(**kw);p=IPCProblem(**kw,element='mini')
        except ModuleNotFoundError as exc:
            self.skipTest(f'Native IPC unavailable: {exc}')
        np.testing.assert_array_equal(p.rest[p.surface],p1.rest[p1.surface])
        np.testing.assert_array_equal(p.cmesh.edges,p1.cmesh.edges)
        self.assertFalse(np.any((p.surface>=p.enrichment.base_nodes)&(p.surface<p.n)))
        pred=p._loading_predictor(p.rest,p.start,p.start+.001)
        np.testing.assert_array_equal(pred[p.enrichment.base_nodes:p.n],0.)
        self.assertTrue(np.all(np.isin(p.enrichment.bubble_dofs,p.free)))
        self.assertEqual(p.reference_contact_check,p1.reference_contact_check)
        # Check that the unchanged observer can be constructed on packed state.
        self.assertIsNotNone(TipContactTracker(p.rest,p.mesh.triangles,p.n,p.config,p.dhat,p.nose_facets))

    def test_interior_refinement_preserves_domain_and_contact_trace(self):
        from prototype.ipc_mini import refine_interior
        base=rectangle(.25)
        def boundary(mesh):
            t=mesh.triangles
            e=np.sort(np.vstack((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]])),axis=1)
            edges,counts=np.unique(e,axis=0,return_counts=True)
            self.assertTrue(np.all(counts<=2))
            return edges[counts==1]
        for level in (1,2):
            mesh=refine_interior(base,np.array([.5,1.]),.35,level)
            np.testing.assert_array_equal(boundary(mesh),boundary(base))
            np.testing.assert_array_equal(mesh.reference_xy[:base.node_count],base.reference_xy)
            np.testing.assert_array_equal(mesh.bottom_nodes,base.bottom_nodes)
            self.assertTrue(np.all(mesh.bulk_data[0]>0))
            self.assertAlmostEqual(mesh.bulk_data[0].sum(),base.bulk_data[0].sum(),places=13)
            self.assertGreater(mesh.node_count,base.node_count)

    def test_shape_window_does_not_extrapolate_short_prefix(self):
        from tools.check_mini_contact import force_shape
        def report(d):
            return {'steps':[dict(depth_mm=x,reaction_fz_n=.03+2*x+3*x*x) for x in d]}
        self.assertEqual(force_shape(report([0,.01,.02]))['status'],'insufficient_depth')
        shape=force_shape(report([0,.25,.5,.75,1]))
        self.assertAlmostEqual(shape['half_depth_ratio'],1.75/5)
        self.assertAlmostEqual(shape['stiffness_growth_ratio'],3.25/1.75)

    def test_coarsened_interior_preserves_boundary_and_fine_tip_patch(self):
        from prototype.ipc_mini import coarsen_interior
        base, coarse = rectangle(.0625), rectangle(.25)
        center = np.array([.5,1.])
        mesh = coarsen_interior(base,coarse,center,.2)
        self.assertLess(mesh.node_count,base.node_count)
        np.testing.assert_array_equal(mesh.reference_xy[mesh.bottom_nodes],base.reference_xy[base.bottom_nodes])
        near = base.local_sn[np.linalg.norm(base.local_sn-center,axis=1)<=.2]
        for value in near:
            self.assertTrue(np.any(np.all(mesh.local_sn==value,axis=1)))
        ids={tuple(v):i for i,v in enumerate(mesh.local_sn)}
        cells={tuple(sorted(cell)) for cell in mesh.triangles}
        for cell in base.triangles:
            if np.all(np.linalg.norm(base.local_sn[cell]-center,axis=1)<=.2):
                self.assertIn(tuple(sorted(ids[tuple(base.local_sn[v])] for v in cell)),cells)
        self.assertTrue(np.all(mesh.bulk_data[0]>0))
        self.assertAlmostEqual(mesh.bulk_data[0].sum(),base.bulk_data[0].sum(),places=13)

    def test_fused_kernel_across_compression_rotation_and_ogden_fallback(self):
        from dataclasses import replace
        for alpha in (1.,2.,3.):
            el=MiniElement(rectangle(.5),replace(self.el.config,ogden_alpha=alpha))
            nv=ContinuousP1Volume(el)
            for F in (np.array([[.55,.2],[-.1,.65]]),np.array([[1.6,.4],[.1,.72]])):
                u=(el.mesh.reference_xy@(F-np.eye(2)).T).ravel()
                u[el.bubble_dofs.ravel()]=np.random.default_rng(2).normal(size=el.bubble_dofs.size)*.0005
                for tangent in (False,True):
                    r,k,d,e,j=el.bulk(u,with_tangent=tangent)
                    r,k,B,e=nv.correct(u,r,k,e)
                    rf,kf,df,ef,jf=nv.bulk(u,with_tangent=tangent)
                    np.testing.assert_allclose(r,rf,atol=2e-12,rtol=2e-11)
                    self.assertAlmostEqual(e,ef,delta=2e-12)
                    if tangent:
                        np.testing.assert_allclose(k,kf,atol=2e-10,rtol=2e-10)
                        np.testing.assert_allclose(B.toarray(),nv.pressure_coupling.toarray(),atol=2e-13)


if __name__=='__main__':unittest.main()
