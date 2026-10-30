"""A reused factor never substitutes for solving the current linear system."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

from prototype.linear_solve import ReusedLUSolver


class ReusedLUSolverTests(unittest.TestCase):
    def setUp(self):
        # Deliberately nonsymmetric, including a changed sparsity pattern below.
        self.A = csc_matrix([[4., 1., 0.], [-2., 5., 1.], [0., -.5, 3.]])
        self.b = np.array([1., -2., .3])
        self.solver = ReusedLUSolver()

    def assert_current_solution(self, A, b, x, stats):
        np.testing.assert_allclose(x, np.linalg.solve(A.toarray(), b),
                                   rtol=1e-9, atol=1e-12)
        actual = float(np.linalg.norm(b-A@x))
        self.assertAlmostEqual(stats['true_residual'], actual, delta=1e-15)
        self.assertLessEqual(actual, max(1e-10*np.linalg.norm(b), 1e-12))

    @patch('prototype.linear_solve.time.perf_counter', return_value=0.)
    def test_changed_matrix_and_shift_use_current_system(self, _clock):
        # This is a current-matrix accuracy/reuse test, not a scheduler timing
        # test. CPU preemption may legitimately trigger the 50 ms fallback;
        # the separate controlled-clock test covers that budget behavior.
        x, initial = self.solver.solve(self.A, self.b)
        self.assertEqual(initial['method'], 'direct_lu')
        self.assertEqual(initial['iterations'], 0)
        self.assertIsNone(initial['fallback_reason'])
        changed = self.A+csc_matrix([[.3, 0., .8], [0., -.2, 0.], [-.1, 0., .5]])
        for shift in (.2, 1., 3.2):
            A = changed+shift*diags([1., 2., 3.])
            b = self.b+shift
            with patch('prototype.linear_solve.splu', wraps=splu) as factor:
                x, stats = self.solver.solve(A, b)
                factor.assert_not_called()
            self.assertEqual(stats['method'], 'gmres_reused_lu')
            self.assertGreater(stats['iterations'], 0)
            self.assertLessEqual(stats['iterations'], 32)
            self.assertEqual(stats['factor_s'], 0.)
            self.assert_current_solution(A, b, x, stats)

    def test_false_gmres_success_is_rejected_by_true_residual(self):
        self.solver.solve(self.A, self.b)
        A = self.A+diags([1., 1., 1.])
        with patch('prototype.linear_solve.gmres', return_value=(np.zeros(3), 0)):
            x, stats = self.solver.solve(A, self.b)
        self.assertEqual(stats['method'], 'direct_lu')
        self.assertEqual(stats['fallback_reason'], 'gmres_true_residual')
        self.assert_current_solution(A, self.b, x, stats)

    def test_nonconverged_gmres_falls_back_even_if_candidate_is_good(self):
        self.solver.solve(self.A, self.b)
        exact = np.linalg.solve(self.A.toarray(), self.b)
        with patch('prototype.linear_solve.gmres', return_value=(exact, 1)):
            x, stats = self.solver.solve(self.A, self.b)
        self.assertEqual(stats['method'], 'direct_lu')
        self.assertEqual(stats['fallback_reason'], 'gmres_info_1')
        self.assert_current_solution(self.A, self.b, x, stats)

    def test_nonfinite_gmres_and_exception_fall_back(self):
        self.solver.solve(self.A, self.b)
        for result in ((np.full(3, np.nan), 0), (np.full(3, np.inf), 0)):
            with patch('prototype.linear_solve.gmres', return_value=result):
                x, stats = self.solver.solve(self.A, self.b)
            self.assertEqual(stats['fallback_reason'], 'gmres_true_residual')
            self.assert_current_solution(self.A, self.b, x, stats)
        with patch('prototype.linear_solve.gmres', side_effect=RuntimeError('failed')):
            x, stats = self.solver.solve(self.A, self.b)
        self.assertEqual(stats['fallback_reason'], 'gmres_exception_RuntimeError')
        self.assert_current_solution(self.A, self.b, x, stats)

    def test_time_budget_stops_reuse_and_refactors(self):
        self.solver.solve(self.A, self.b)
        self.solver._factor_seconds = .02
        clock = [0.]

        def fake_gmres(A, b, **kwargs):
            self.assertEqual(kwargs['restart'], 16)
            self.assertEqual(kwargs['maxiter'], 2)
            self.assertEqual(kwargs['callback_type'], 'pr_norm')
            self.assertEqual(kwargs['rtol'], 1e-11)
            self.assertEqual(kwargs['atol'], 1e-13)
            clock[0] = .06
            kwargs['callback'](.1)
            self.fail('Budget callback should abandon the reuse attempt')

        with patch('prototype.linear_solve.time.perf_counter', side_effect=lambda: clock[0]), \
                patch('prototype.linear_solve.gmres', side_effect=fake_gmres):
            x, stats = self.solver.solve(self.A, self.b)
        self.assertEqual(stats['fallback_reason'], 'gmres_time_budget')
        self.assertEqual(stats['iterations'], 1)
        self.assertEqual(stats['reuse_budget_s'], .05)
        self.assert_current_solution(self.A, self.b, x, stats)

    def test_zero_rhs_needs_no_factor_and_does_not_erase_cache(self):
        for n in (0, 3):
            with patch('prototype.linear_solve.splu') as factor:
                x, stats = self.solver.solve(csc_matrix((n, n)), np.zeros(n))
                factor.assert_not_called()
            np.testing.assert_array_equal(x, np.zeros(n))
            self.assertEqual(stats['method'], 'zero_rhs')
        self.solver.solve(self.A, self.b)
        old = self.solver._lu
        self.solver.solve(self.A, np.zeros(3))
        self.assertIs(self.solver._lu, old)

    def test_shape_change_refactors_and_invalid_inputs_reject(self):
        self.solver.solve(self.A, self.b)
        A = diags([2., 3.]).tocsc(); b = np.array([1., 4.])
        with patch('prototype.linear_solve.gmres') as iterative:
            x, stats = self.solver.solve(A, b)
            iterative.assert_not_called()
        self.assertEqual(stats['fallback_reason'], 'matrix_shape_changed')
        self.assert_current_solution(A, b, x, stats)
        for A, b in ((np.ones((2, 3)), np.ones(2)), (np.eye(2), np.ones(3)),
                     (np.eye(2), np.ones((2, 1))), (np.eye(2), [np.nan, 0.]),
                     ([[np.inf]], [1.])):
            with self.assertRaises(ValueError): self.solver.solve(A, b)

    def test_direct_refinement_uses_current_residual(self):
        actual = splu(self.A)

        class InexactFirstSolve:
            def __init__(self): self.count = 0
            def solve(self, b):
                self.count += 1
                x = actual.solve(b)
                return x+1e-4 if self.count == 1 else x

        with patch('prototype.linear_solve.splu', return_value=InexactFirstSolve()) as factory:
            x, stats = self.solver.solve(self.A, self.b)
        self.assertEqual(factory.call_args.kwargs['permc_spec'], 'COLAMD')
        self.assertEqual(stats['refinement_steps'], 1)
        self.assert_current_solution(self.A, self.b, x, stats)

    def test_bad_direct_solve_is_not_returned_as_success(self):
        class BadFactor:
            def solve(self, b): return np.zeros_like(b)
        with patch('prototype.linear_solve.splu', return_value=BadFactor()):
            with self.assertRaisesRegex(RuntimeError, 'linear residual failed'):
                self.solver.solve(self.A, self.b)
        self.assertIsNone(self.solver._lu)
        with self.assertRaisesRegex(RuntimeError, 'factorization failed'):
            self.solver.solve(csc_matrix((3, 3)), self.b)


if __name__ == '__main__':
    unittest.main()
