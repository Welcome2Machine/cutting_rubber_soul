"""Physical equivalence and failure protection of the explicit compute backend."""
import contextlib
import io
import argparse
import tempfile
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import bmat,csr_matrix,diags
from threadpoolctl import threadpool_limits
from mechanics.finite_strain import response
from prototype.al_performance import FastAugmentedBladeProblem,ParallelMaterial,PardisoDirection
from tools.run_augmented_blade import make_problem


class ALPerformanceTests(unittest.TestCase):
    def setUp(self):
        c=threadpool_limits(1);c.__enter__();self.addCleanup(c.__exit__,None,None,None)

    def fast(self,**kwargs):
        p=make_problem(57,coarse=True,spacing=.000625)
        p.__class__=FastAugmentedBladeProblem;p.configure_compute(**kwargs)
        self.addCleanup(p.close_compute);p.ipc.set_num_threads(2)
        return p

    def test_partitioned_response_preserves_cross_modes_and_history(self):
        p=self.fast(linear_backend='superlu');rng=np.random.default_rng(901)
        F=np.tile(np.eye(2),(61,1,1))+rng.normal(0,.12,(61,2,2))
        A=np.tile(np.eye(3),(61,1,1));saved=A.copy()
        kernel=ParallelMaterial(3,11);self.addCleanup(kernel.close)
        for mode in ('plane_strain','plane_stress'):
            for tangent in (False,True):
                a=response(F,A,.05,p.material,tangent=tangent,kinematics=mode)
                b=kernel.evaluate(F,A,.05,p.material,tangent=tangent,kinematics=mode)
                for key in a:
                    if mode=='plane_stress' and key in ('plane_stress_residual','plane_stress_scaled_residual'):
                        # Independent batches stop once EACH point meets the
                        # existing 1e-9 stress tolerance; a global batch may do
                        # extra iterations on already converged points.
                        self.assertLess(np.max(abs(a[key])),1.1e-9)
                        self.assertLess(np.max(abs(b[key])),1.1e-9)
                    else:np.testing.assert_allclose(a[key],b[key],rtol=1e-9,atol=1e-10,err_msg=key)
        np.testing.assert_array_equal(A,saved)

    def test_active_geometry_matches_full_space_and_path_checks(self):
        old=make_problem(57,coarse=True,spacing=.000625);p=self.fast(linear_backend='superlu')
        rng=np.random.default_rng(18);u=np.zeros(2*p.n)
        for scale in (1e-4,.002,.01):
            for _ in range(6):
                d=rng.normal(0,scale,(p.n,2))
                np.testing.assert_allclose(p.enrichment.minimum_j(d),old.enrichment.minimum_j(d),rtol=1e-12,atol=1e-12)
                self.assertEqual(p.enrichment.safe_step(u,d),old.enrichment.safe_step(u,d))
        # A globally inverted affine field must still be rejected.
        bad=np.zeros_like(p.mesh.reference_xy);bad[:,1]=-1.1*p.mesh.reference_xy[:,1]
        self.assertLess(p.enrichment.minimum_j(bad),0)
        with self.assertRaises(FloatingPointError):p.enrichment.safe_step(bad,bad)

    def test_multicore_solve_and_inaccurate_direction_fallback(self):
        solver=PardisoDirection(4);self.addCleanup(solver.close)
        rng=np.random.default_rng(4);B=csr_matrix(rng.normal(size=(12,80)))
        K=diags([np.ones(79)*-.2,np.ones(80)*3,np.ones(79)*.1],[-1,0,1])
        A=bmat([[K,B.T],[B,None]],format='csc');truth=rng.normal(size=92);b=A@truth
        np.testing.assert_allclose(solver.solve(A,b),truth,rtol=1e-11,atol=1e-11)
        with patch.object(solver.solver,'solve',return_value=np.full(92,np.nan)):
            np.testing.assert_allclose(solver.solve(A,b),truth,rtol=1e-11,atol=1e-11)
        self.assertIsNotNone(solver.rows[-1]['fallback_reason'])

    def test_actual_loading_matches_original_with_original_fresh_audit(self):
        a=make_problem(57,coarse=True,spacing=.000625);a.ipc.set_num_threads(2)
        b=self.fast(material_workers=2,batch_size=2048,linear_threads=4)
        xa=a.rest.copy();xb=b.rest.copy();old=a.start
        with contextlib.redirect_stdout(io.StringIO()):
            for depth in (0.,.005,.01):
                xa,ra=a.step(xa,old,depth,time.monotonic()+90)
                xb,rb=b.step(xb,old,depth,time.monotonic()+90);old=depth
                np.testing.assert_allclose(xa,xb,rtol=0,atol=1e-7)
                np.testing.assert_allclose(a.material_metric,b.material_metric,rtol=1e-7,atol=1e-8)
                self.assertLess(abs(ra['reaction_fz_n']-rb['reaction_fz_n']),1e-5)
                state=dict(reference_xy=b.rest,triangles=b.mesh.triangles,positions=xb,
                    previous_positions=b.previous,depth_mm=depth,**b.checkpoint_data())
                self.assertTrue(a.audit_state(state,rb)['passed'])
        self.assertFalse(any(x['fallback_reason'] for x in b.parallel_linear.rows))

    def test_saved_fast_history_resume_matches_continuous_path(self):
        from tools import run_augmented_blade as runner
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            def factory(angle,**kwargs):return self.fast(material_workers=2,batch_size=2048,linear_threads=4)
            def run(name,depth,resume=None):
                return runner.run(argparse.Namespace(output=root/name,angle=57.,coarse=True,
                    spacing=.000625,depth=depth,increment=.005,seconds=90.,resume=resume,
                    phase='run',blas_threads=1),problem_factory=factory)
            direct=run('direct',.01);prefix=run('prefix',.005);resumed=run('resumed',.01,root/'prefix')
            self.assertTrue(direct['completed']);self.assertTrue(prefix['completed']);self.assertTrue(resumed['completed'])
            with np.load(root/'direct/states/0002.npz') as a,np.load(root/'resumed/states/0002.npz') as b:
                for key in ('positions','previous_positions','material_metric','previous_material_metric',
                            'al_multipliers','material_dt_s','cumulative_viscous_dissipation_n_mm'):
                    np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-10,err_msg=key)
            original=make_problem(57,coarse=True,spacing=.000625)
            self.assertTrue(runner.audit(original,root/'resumed',resumed)['passed'])


if __name__=='__main__':unittest.main()
