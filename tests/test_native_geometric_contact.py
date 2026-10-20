"""Independent contact partitions, tangent, CCD and fresh-history checks."""
import contextlib
import io
import time
import unittest
import numpy as np
from prototype.ipc_viscoelastic import ViscoelasticProblem
from prototype.native_geometric_contact import NativeGeometricProblem
from prototype.geometric_self_contact import GeometricSelfContact
from tests.test_geometric_self_contact import shape

KW=dict(maxwell_shear=1.24,relaxation_s=1.7,loading_rate=.2,bulk_modulus=23.34333333333331,
    case='patch',mu=.35,shear_modulus=.14,local_spacing=.1,surface_spacing=.025,
    dhat=.0025,max_iterations=500,predictor='secant',element='p1')


class NativeGeometricTests(unittest.TestCase):
    def make(self):
        p=NativeGeometricProblem(**KW);p.ipc.set_num_threads(2);p.material_dt=.025
        return p

    def test_force_filter_is_cross_only_but_full_ccd_retains_self(self):
        p=self.make();n=len(p.body_surface)
        self.assertFalse(p.cmesh.can_collide(0,2))
        self.assertTrue(p.cmesh.can_collide(0,n))
        self.assertFalse(p.cmesh.can_collide(n,n+1))
        self.assertTrue(p.full_ccd_mesh.can_collide(0,2))
        self.assertTrue(p.full_ccd_mesh.can_collide(0,n))
        np.testing.assert_array_equal(p.cmesh.edges,p.full_ccd_mesh.edges)

    def test_zero_self_case_matches_native_force_tangent_and_metric(self):
        p=self.make();old=ViscoelasticProblem(**KW);old.material_dt=.025
        x=p.affine_loading_candidate(p.rest,p.start,.003);p.previous=p.rest.copy()
        a=p.evaluate(x,p.lag(x),projected=False,with_metric=True)
        b=old.evaluate(x,old.lag(x),projected=False,with_metric=True)
        self.assertEqual(p.geometric_self.last['active_pairs'],0)
        np.testing.assert_allclose(a[1],b[1],rtol=1e-12,atol=1e-10)
        for i in (2,4):np.testing.assert_allclose(a[i].toarray(),b[i].toarray(),rtol=1e-12,atol=1e-9)
        np.testing.assert_allclose(p.safe_step(p.rest,x-p.rest),old.safe_step(p.rest,x-p.rest))

    def test_coupled_blade_tangent_and_history_immutable(self):
        p=self.make();x=p.affine_loading_candidate(p.rest,p.start,.003)
        old=p.previous.copy();metric=p.previous_material_metric.copy()
        lag=p.lag(x);_,g,h,_=p.evaluate(x,lag,projected=False)
        H=p.coupled_friction_tangent(x,lag,h)
        d=np.random.default_rng(3).normal(size=x.shape)*.001;d[p.n:]=0
        eps=1e-7
        gp=p.evaluate(x+eps*d,p.lag(x+eps*d),tangent=False)[1]
        gm=p.evaluate(x-eps*d,p.lag(x-eps*d),tangent=False)[1]
        np.testing.assert_allclose((gp-gm)/(2*eps),H@d.ravel(),rtol=4e-4,atol=2e-6)
        np.testing.assert_array_equal(p.previous,old);np.testing.assert_array_equal(p.previous_material_metric,metric)

    def test_body_active_metric_psd_and_scatter_no_blade_force(self):
        p=self.make();ref=shape(.006);E=np.array([[i,(i+1)%8] for i in range(8)])
        q=GeometricSelfContact(p.ipc,ref,E,dhat=.0025,stiffness_mpa=6.6,width_mm=20,mu=.35,eps=1e-5)
        x=shape(.001)+np.random.default_rng(41).normal(size=(8,2))*2e-5
        old=x.copy();old[2:5,1]-=3e-6
        a=q.terms(x,old,tangent=True,with_metric=True)
        self.assertGreater(q.last['active_pairs'],0)
        np.testing.assert_allclose(a[4].toarray(),a[4].T.toarray(),atol=1e-8)
        self.assertGreater(np.linalg.eigvalsh(a[4].toarray()).min(),-1e-7)
        self.assertGreater(np.linalg.norm((a[2]-a[2].T).data),1e-5)
        # Exercise full-size scatter with nonzero active self-contact blocks.
        p.self_dofs=(2*p.body_surface[:8,None]+np.arange(2)).ravel()
        H=p._scatter(a[2]);self.assertEqual(H[2*p.n:].nnz,0)
        d=np.zeros(p.total);d[p.self_dofs]=np.arange(16)/16
        np.testing.assert_allclose((H@d)[p.self_dofs],a[2]@d[p.self_dofs])

    def test_actual_loading_fresh_audit_and_policy_rejection(self):
        p=self.make();x=p.rest.copy();old=p.start
        with contextlib.redirect_stdout(io.StringIO()):
            for depth in (0.,.005):x,r=p.step(x,old,depth,time.monotonic()+45);old=depth
        state=dict(reference_xy=p.rest,triangles=p.mesh.triangles,positions=x,
            previous_positions=p.previous.copy(),depth_mm=old,**p.checkpoint_data())
        audit=self.make();self.assertTrue(audit.audit_state(state,r)['passed'])
        state['body_self_contact_policy']=np.array('old-native')
        with self.assertRaisesRegex(ValueError,'policy'):audit.audit_state(state,r)

    def test_interval_gate_detects_interior_holes_not_only_endpoints(self):
        from tools.run_native_gcp import interval_distance
        self.assertAlmostEqual(interval_distance([[0.,1.]],[[0.,.2],[.8,1.]]),.3)
        self.assertAlmostEqual(interval_distance([[0.,.2],[.8,1.]],[[0.,1.]]),.3)
        self.assertEqual(interval_distance([[0.,1.]],[[0.,1.]]),0.)
        self.assertIsNone(interval_distance([],[]))
        self.assertTrue(np.isinf(interval_distance([],[[0.,1.]])))

    def test_frozen_worker_runtime_does_not_resolve_from_source_snapshot(self):
        from tools.run_native_gcp import runtime_environment
        e=runtime_environment(dict(runtime_path='D:/runtime/ipc',pardiso_runtime='D:/runtime/mkl'))
        self.assertEqual(e['SOFTMODELING_PARDISO_RUNTIME'],'D:/runtime/mkl')
        self.assertEqual(e['PYTHONPATH'],'D:/runtime/ipc')
        self.assertEqual(e['OPENBLAS_NUM_THREADS'],'1')


if __name__=='__main__':unittest.main()
