"""Fresh nonsymmetric force globalization, safeguards and bounded fallback."""
import contextlib
import io
import time
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from prototype.ipc_reference import IPCProblem, BASELINE
from prototype.residual_newton import ResidualNewtonMixin, residual_armijo
from prototype.compact_p2_assembly import compact_matrices


class Fallback:
    _ptc_trial=IPCProblem._ptc_trial

    def solve_friction(self,x,depth,deadline):
        self.fallback_budget=self.max_iterations
        return x,0


class Candidate(ResidualNewtonMixin,Fallback):
    pass


def problem(residual,jacobian):
    p=Candidate();p.nodal_volume=None;p.enrichment=None;p.max_iterations=500
    p.total=2;p.free=np.arange(2);p.options=SimpleNamespace(tip_radius_mm=1000.)
    p.lag=lambda x:x.copy();p.safe_step=lambda x,d:1.
    p.previous=np.array([[9.,8.]])
    p.evaluate=lambda x,lag,tangent=True,**kwargs:(0.,residual(x.ravel()),csc_matrix(jacobian(x.ravel())),1.)
    p.coupled_friction_tangent=lambda x,lag,H:H
    return p


class NewtonTests(unittest.TestCase):
    def solve(self,p,x):
        log=io.StringIO()
        with contextlib.redirect_stdout(log):value,count=p.solve_friction(x,0.,time.monotonic()+5.)
        return value,count,log.getvalue()

    def test_nonsymmetric_exact_jacobian_reaches_original_force_tolerance(self):
        A=np.array([[1.,100.],[0.,1.]]);r=np.array([-1.,1.])
        p=problem(lambda x:r+A@x,lambda x:A)
        x,it,log=self.solve(p,np.zeros((1,2)))
        self.assertLess(np.linalg.norm(r+A@x.ravel()),BASELINE['force_tolerance_n'])
        self.assertEqual(it,1);self.assertNotIn('NEWTON_FALLBACK',log)
        np.testing.assert_array_equal(p.previous,[[9.,8.]])

    def test_large_nonlinear_direction_backtracks_and_converges(self):
        p=problem(lambda x:np.array([x[0]**3-1.,x[1]]),lambda x:np.diag([3*x[0]**2,1.]))
        x,it,log=self.solve(p,np.array([[.1,0.]]))
        self.assertLess(abs(x[0,0]**3-1),BASELINE['force_tolerance_n'])
        self.assertIn('accepted=False',log);self.assertNotIn('NEWTON_FALLBACK',log)

    def test_ccd_cap_is_obeyed_before_residual_evaluation(self):
        p=problem(lambda x:x-np.array([1.,0.]),lambda x:np.eye(2))
        p.safe_step=lambda x,d:.25;p.newton_phase_limit=1
        x,_,log=self.solve(p,np.zeros((1,2)))
        np.testing.assert_allclose(x,[[.25,0.]])
        self.assertEqual(p.fallback_budget,499);self.assertEqual(p.max_iterations,500)

    def test_no_progress_returns_to_original_ptc_without_extra_iterations(self):
        p=problem(lambda x:np.ones(2),lambda x:np.eye(2))
        x,_,log=self.solve(p,np.zeros((1,2)))
        np.testing.assert_array_equal(x,[[0.,0.]])
        self.assertEqual(p.fallback_budget,499);self.assertEqual(p.max_iterations,500)
        self.assertIn('NEWTON_FALLBACK',log)

    def test_unrelated_assembly_errors_are_not_swallowed(self):
        def residual(x):raise RuntimeError('material bug')
        p=problem(residual,lambda x:np.eye(2))
        with self.assertRaisesRegex(RuntimeError,'material bug'):self.solve(p,np.zeros((1,2)))

    def test_merit_rejects_non_descent_nonfinite_or_invalid_step(self):
        self.assertTrue(residual_armijo(1.,.5,-1.,.5))
        for values in ((1.,.5,1.,1.),(1.,1.01,-1.,1.),(1.,float('nan'),-1.,1.),
                       (1.,.5,-1.,0.),(1.,.5,-1.,2.),(1.,-.5,-1.,1.)):
            self.assertFalse(residual_armijo(*values))

    def test_default_ipc_method_is_not_replaced(self):
        self.assertNotEqual(IPCProblem.solve_friction,ResidualNewtonMixin.solve_friction)
        self.assertEqual(BASELINE['force_tolerance_n'],1e-4)


class CompactAssemblyTests(unittest.TestCase):
    def test_factory_opt_in_keeps_physical_signature_and_original_default(self):
        from tools.run_angle_path_study import make_problem,signature
        from prototype.compact_p2_assembly import CompactP2AssemblyMixin
        p=make_problem(57,'oblique',coarse=True)
        q=make_problem(57,'oblique',coarse=True,compact_assembly=True)
        self.assertNotIsInstance(p,CompactP2AssemblyMixin)
        self.assertIsInstance(q,CompactP2AssemblyMixin)
        self.assertEqual(signature(p),signature(q))
        np.testing.assert_array_equal(p.material_metric,q.material_metric)
        self.assertEqual(p.max_iterations,q.max_iterations)

    def test_indefinite_padded_blocks_match_dense_reference_and_psd(self):
        rng=np.random.default_rng(125)
        k=np.zeros((8,12,12));dofs=rng.integers(0,30,(8,12))
        linear=np.array([0,2,3,5,6]);active=np.array([1,4,7])
        for ids,n in ((linear,6),(active,12)):
            a=rng.normal(size=(len(ids),n,n));k[ids,:n,:n]=a+a.transpose(0,2,1)
        rows=np.broadcast_to(dofs[:,:,None],k.shape).ravel()
        cols=np.broadcast_to(dofs[:,None,:],k.shape).ravel()
        from scipy.sparse import coo_matrix
        eig,v=np.linalg.eigh(k);kp=(v*np.maximum(eig,0)[:,None,:])@v.transpose(0,2,1)
        raw=coo_matrix((k.ravel(),(rows,cols)),shape=(30,30)).toarray()
        psd=coo_matrix((kp.ravel(),(rows,cols)),shape=(30,30)).toarray()
        for projected,metric in ((False,False),(False,True),(True,False),(True,True)):
            h,m=compact_matrices(k,dofs,30,linear,active,projected=projected,with_metric=metric)
            np.testing.assert_allclose(h.toarray(),psd if projected else raw,rtol=0,atol=1e-12)
            if projected or metric:np.testing.assert_allclose(m.toarray(),psd,rtol=0,atol=1e-12)
            else:self.assertIsNone(m)


if __name__=='__main__':unittest.main()
