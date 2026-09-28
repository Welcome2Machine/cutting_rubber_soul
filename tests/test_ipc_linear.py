"""Bounded linear reuse must preserve the unshifted mechanics acceptance."""

import contextlib
import io
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, diags, eye
from scipy.sparse.linalg import spsolve, MatrixRankWarning

from prototype.ipc_reference import IPCProblem, FixedElementAssembly, ShiftRetryLinearSolver, new_work_counters, BASELINE


class FixedElementAssemblyTests(unittest.TestCase):
    def test_duplicate_nonsymmetric_entries_match_coo_and_matrices_do_not_alias(self):
        dofs = np.array([[0, 2, 4], [4, 2, 1], [2, 4, 5]])
        values = np.random.default_rng(43).normal(size=(3, 3, 3))
        graph = FixedElementAssembly(dofs, 8)
        rows = np.broadcast_to(dofs[:, :, None], values.shape).ravel()
        cols = np.broadcast_to(dofs[:, None, :], values.shape).ravel()
        expected = coo_matrix((values.ravel(), (rows, cols)), shape=(8, 8)).tocsc()
        first = graph.matrix(values)
        second = graph.matrix(2*values)
        np.testing.assert_allclose(first.toarray(), expected.toarray(), atol=1e-15)
        np.testing.assert_allclose(second.toarray(), 2*expected.toarray(), atol=2e-15)
        self.assertTrue(first.has_canonical_format)
        self.assertFalse(np.shares_memory(first.data, second.data))
        empty = graph.matrix(np.zeros_like(values))
        empty.eliminate_zeros()
        np.testing.assert_allclose(graph.matrix(values).toarray(), expected.toarray(), atol=1e-15)
        values[:] = 0
        dofs[:] = 0
        np.testing.assert_allclose(first.toarray(), expected.toarray(), atol=1e-15)
        self.assertFalse(np.array_equal(graph.dofs, dofs))

    def test_wrong_element_shape_is_rejected(self):
        graph = FixedElementAssembly(np.array([[0, 1]]), 3)
        with self.assertRaisesRegex(ValueError, "shape"):
            graph.matrix(np.zeros((1, 3, 3)))


class ShiftRetryLinearTests(unittest.TestCase):
    def seeded_solver(self, size=8):
        counters = new_work_counters()
        solver = ShiftRetryLinearSolver(counters, "retry-gmres")
        matrix = diags((np.full(size-1, -.4), np.linspace(2., 4., size),
                        np.full(size-1, .2)), (-1, 0, 1), format="csc")
        rhs = np.linspace(-1., 2., size)
        np.testing.assert_allclose(solver.solve(matrix, rhs), spsolve(matrix, rhs), atol=1e-14)
        # Tiny test factors cost microseconds. Give GMRES a deterministic generous
        # budget here; the independent expired-budget test covers the time gate.
        solver.factor_seconds = 10.
        return solver, matrix, rhs

    def test_nonsymmetric_shift_retry_reuses_factor_with_true_residual(self):
        solver, matrix, rhs = self.seeded_solver()
        shifted = matrix + .003 * eye(matrix.shape[0], format="csc")
        result = solver.solve(shifted, rhs)
        self.assertLessEqual(np.linalg.norm(shifted @ result-rhs), 1e-8*np.linalg.norm(rhs))
        self.assertEqual(solver.counters["factorizations"], 1)
        self.assertEqual(solver.counters["krylov_accepted"], 1)
        self.assertGreater(solver.counters["krylov_iterations"], 0)
        self.assertLessEqual(solver.counters["krylov_iterations"], 8)

    def test_reported_success_with_wrong_true_residual_falls_back(self):
        solver, matrix, rhs = self.seeded_solver()
        for returned in (np.zeros_like(rhs), np.full_like(rhs, np.nan)):
            with self.subTest(nonfinite=bool(np.isnan(returned).any())):
                before = solver.counters["linear_fallbacks"]
                with patch("prototype.ipc_reference.gmres", return_value=(returned, 0)):
                    result = solver.solve(matrix, rhs)
                self.assertEqual(solver.counters["linear_fallbacks"], before+1)
                np.testing.assert_allclose(matrix @ result, rhs, atol=1e-14)
                solver.factor_seconds = 10.

    def test_broken_or_nonfinite_preconditioner_falls_back_to_direct(self):
        for failure in (RuntimeError("failed triangular solve"), np.full(8, np.nan)):
            solver, matrix, rhs = self.seeded_solver()
            fake = Mock()
            if isinstance(failure, Exception):
                fake.solve.side_effect = failure
            else:
                fake.solve.return_value = failure
            solver.factor = fake
            result = solver.solve(matrix, rhs)
            np.testing.assert_allclose(matrix @ result, rhs, atol=1e-14)
            self.assertEqual(solver.counters["linear_fallbacks"], 1)
            self.assertEqual(solver.counters["factorizations"], 2)

    def test_actual_krylov_iteration_limit_and_direct_fallback(self):
        counters = new_work_counters()
        solver = ShiftRetryLinearSolver(counters, "retry-gmres")
        rhs = np.ones(32)
        solver.solve(eye(32, format="csc"), rhs)
        solver.factor_seconds = 100.
        matrix = diags(np.geomspace(1., 1e5, 32), format="csc")
        result = solver.solve(matrix, rhs)
        self.assertEqual(counters["krylov_iterations"], 8)
        self.assertEqual(counters["linear_fallbacks"], 1)
        np.testing.assert_allclose(matrix @ result, rhs, atol=1e-14)

    def test_expired_budget_returns_direct_solution_without_iterations(self):
        solver, matrix, rhs = self.seeded_solver()
        solver.factor_seconds = 0.
        result = solver.solve(matrix, rhs)
        self.assertEqual(solver.counters["krylov_iterations"], 0)
        self.assertEqual(solver.counters["linear_fallbacks"], 1)
        np.testing.assert_allclose(matrix @ result, rhs, atol=1e-14)

    def test_direct_baseline_never_enters_reuse(self):
        counters = new_work_counters()
        solver = ShiftRetryLinearSolver(counters)
        matrix = csc_matrix([[3., .5], [-1., 2.]])
        rhs = np.array([1., 2.])
        with patch("prototype.ipc_reference.gmres", side_effect=AssertionError("baseline changed")):
            for scale in (1., 2.):
                result = solver.solve(scale*matrix, rhs)
                np.testing.assert_allclose(scale*matrix @ result, rhs, atol=1e-14)
        self.assertEqual(counters["factorizations"], 2)
        self.assertEqual(counters["krylov_attempts"], 0)

    def test_singular_factor_preserves_nonfinite_shift_recovery(self):
        counters = new_work_counters()
        solver = ShiftRetryLinearSolver(counters, "retry-gmres")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MatrixRankWarning)
            result = solver.solve(csc_matrix((2, 2)), np.ones(2))
        self.assertFalse(np.isfinite(result).all())
        self.assertTrue(any(issubclass(item.category, MatrixRankWarning) for item in caught))
        self.assertIsNone(solver.factor)
        # At the next shift, PTC can factor again instead of aborting this load.
        np.testing.assert_allclose(solver.solve(eye(2, format="csc"), np.ones(2)), 1.)
        self.assertEqual(counters["factorizations"], 3)

    def test_global_deadline_does_not_fall_back_after_krylov(self):
        solver, matrix, rhs = self.seeded_solver()
        with patch("prototype.ipc_reference.time.monotonic", return_value=0.) as clock:
            def exhausted(*args, **kwargs):
                clock.return_value = 2.
                return np.zeros_like(rhs), 0
            with patch("prototype.ipc_reference.gmres", side_effect=exhausted):
                with self.assertRaisesRegex(TimeoutError, "wall budget"):
                    solver.solve(matrix, rhs, deadline=1.)
        self.assertEqual(solver.counters["factorizations"], 1)
        self.assertEqual(solver.counters["linear_fallbacks"], 0)

    def test_expired_global_deadline_skips_first_factor(self):
        counters = new_work_counters()
        solver = ShiftRetryLinearSolver(counters, "retry-gmres")
        with self.assertRaisesRegex(TimeoutError, "wall budget"):
            solver.solve(eye(2, format="csc"), np.ones(2), deadline=-1.)
        self.assertEqual(counters["factorizations"], 0)

    def toy_problem(self, max_iterations):
        p = IPCProblem.__new__(IPCProblem)
        p.work_counters = new_work_counters()
        p.friction_linear_solver = "retry-gmres"
        p.max_iterations = max_iterations
        p.total = 2
        p.free = np.array([0])
        p.options = SimpleNamespace(tip_radius_mm=10.)
        p.lag = lambda x: None
        p.safe_step = lambda x, d: 1.
        p.coupled_friction_tangent = lambda x, lag, h: h

        def evaluate(x, lag, tangent=True, projected=False, with_metric=False):
            result = (0., np.array([x[0, 0]-1., 0.]), eye(2, format="csc"), 1.)
            return result+(eye(2, format="csc"),) if with_metric else result

        p.evaluate = evaluate
        return p

    def test_every_new_nonlinear_state_gets_its_own_factor(self):
        p = self.toy_problem(8)
        with contextlib.redirect_stdout(io.StringIO()):
            result, iterations = p.solve_friction(np.zeros((1, 2)), 0., np.inf)
        self.assertGreater(iterations, 1)
        self.assertLess(abs(result[0, 0]-1.), 1e-4)
        self.assertEqual(p.work_counters["factorizations"], iterations)
        self.assertEqual(p.work_counters["krylov_attempts"], 0)
        self.assertEqual(p.work_counters["mechanical_iterations"], iterations)

    def test_balanced_trial_skips_only_unused_final_tangent(self):
        p = self.toy_problem(8)
        original = p.evaluate
        p.evaluate = Mock(side_effect=original)
        with contextlib.redirect_stdout(io.StringIO()):
            result, iterations = p.solve_friction(np.zeros((1, 2)), 0., np.inf)
        self.assertLess(abs(result[0, 0]-1.), BASELINE["force_tolerance_n"])
        self.assertEqual(p.work_counters["converged_tangent_skips"], 1)
        paired = sum(call.kwargs.get("with_metric", False) for call in p.evaluate.call_args_list)
        self.assertEqual(paired, iterations)
        # Final successful force evaluation was fresh, not a lagged linear model.
        self.assertFalse(p.evaluate.call_args_list[-1].kwargs["tangent"])

    def test_work_survives_nonlinear_failure_and_next_solve(self):
        p = self.toy_problem(1)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "iteration budget"):
                    p.solve_friction(np.zeros((1, 2)), 0., np.inf)
        self.assertEqual(p.work_counters["mechanical_solves"], 2)
        self.assertEqual(p.work_counters["factorizations"], 2)
        self.assertEqual(p.work_counters["mechanical_iterations"], 2)
        self.assertGreater(p.work_counters["mechanical_seconds"], 0.)
        self.assertGreater(p.work_counters["linear_solve_seconds"], 0.)

    def test_singular_first_ptc_shift_still_reaches_force_balance(self):
        p = self.toy_problem(40)
        c = .1*(1.+BASELINE["hessian_regularization_n_mm"])

        def quadratic(x, lag, tangent=True, projected=False, with_metric=False):
            value = x[0, 0]
            result = (0., np.array([value*value-c*value-.1, 0.]),
                      diags([2*value-c, 1.], format="csc"), 1.)
            return result+(eye(2, format="csc"),) if with_metric else result

        p.evaluate = quadratic
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", MatrixRankWarning)
            result, iterations = p.solve_friction(np.zeros((1, 2)), 0., np.inf)
        self.assertGreater(iterations, 0)
        self.assertGreater(p.work_counters["model_rejections"], 0)
        self.assertGreater(p.work_counters["factorizations"], iterations)
        self.assertLess(np.linalg.norm(quadratic(result, None)[1]), BASELINE["force_tolerance_n"])


if __name__ == "__main__":
    unittest.main()
