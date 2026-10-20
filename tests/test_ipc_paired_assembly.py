"""Paired exact/projected assembly agrees with independent native IPC calls."""
import unittest
import contextlib
import io
import json
import time
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from prototype.ipc_reference import IPCProblem


class PairedAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=1)
        self.threads.__enter__()
        self.addCleanup(self.threads.__exit__, None, None, None)
        self.p = IPCProblem('patch', mu=.35, local_spacing=.2,
                            surface_spacing=.2, element='p2-interior')
        p = self.p
        x = p.rest.copy()
        x[p.n:, 1] -= 2*p.dhat-.006
        # Nonaffine interior enrichment, while retaining an unambiguous flat
        # contact geometry for independent finite-difference checks.
        rng = np.random.default_rng(491)
        x[p.enrichment.base_nodes:p.n] += rng.normal(
            size=(p.n-p.enrichment.base_nodes, 2))*1e-4
        p.previous = x.copy()
        x[p.n:, 0] += .002
        self.x = x

    def assert_paired_equivalent(self, x, lag):
        p = self.p
        paired = p.evaluate(x, lag, projected=False, with_metric=True)
        exact = p.evaluate(x, lag, projected=False)
        projected = p.evaluate(x, lag, projected=True)
        for reference in (exact, projected):
            self.assertAlmostEqual(paired[0], reference[0], delta=1e-12)
            self.assertEqual(paired[3], reference[3])
            # Native contact reductions can differ by a last rounding bit.
            np.testing.assert_allclose(paired[1], reference[1], rtol=1e-14, atol=1e-12)
        np.testing.assert_allclose(paired[2].toarray(), exact[2].toarray(), rtol=1e-14, atol=1e-12)
        np.testing.assert_allclose(paired[4].toarray(), projected[2].toarray(), rtol=1e-14, atol=1e-12)
        return paired

    def test_nonaffine_p2_matches_two_independent_assemblies(self):
        self.assert_paired_equivalent(self.x, self.p.lag(self.x))
        # The no-friction branch must also work when a metric is requested.
        self.assert_paired_equivalent(self.x, None)

    def test_p1_friction_uses_one_bulk_evaluation_per_coupled_tangent(self):
        from prototype import ipc_reference
        p = IPCProblem('patch', mu=.35, local_spacing=.2, surface_spacing=.2)
        self.p = p
        x = p.rest.copy()
        x[p.n:, 1] -= 2*p.dhat-.006
        p.previous = x.copy()
        x[p.n:, 0] += .002
        self.assert_paired_equivalent(x, p.lag(x))
        self.assert_paired_equivalent(x, None)
        # Check the actual solver, not just the optional evaluate interface.
        p.max_iterations = 1
        with patch.object(ipc_reference, 'bulk', wraps=ipc_reference.bulk) as call:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'iteration budget'):
                p.solve_friction(x, 0., time.monotonic()+10)
            tangent_calls = [c for c in call.call_args_list if c.kwargs['with_tangent']]
            self.assertEqual(len(tangent_calls), 2)  # iterations 0 and 1, not two each

    def test_paired_call_evaluates_bulk_only_once(self):
        p = self.p
        lag = p.lag(self.x)
        with patch.object(p.enrichment, 'bulk', wraps=p.enrichment.bulk) as call:
            result = p.evaluate(self.x, lag, projected=False, with_metric=True)
            self.assertEqual(call.call_count, 1)
            self.assertTrue(call.call_args.kwargs['with_tangent'])
            self.assertEqual(len(result), 5)
        with patch.object(p.enrichment, 'bulk', wraps=p.enrichment.bulk) as call:
            projected = p.evaluate(self.x, lag, projected=True, with_metric=True)
            self.assertEqual(call.call_count, 1)
            self.assertIs(projected[2], projected[4])
        with self.assertRaisesRegex(ValueError, 'requires the tangent'):
            p.evaluate(self.x, lag, tangent=False, with_metric=True)

    def test_exact_force_and_coupled_jacobian_match_finite_differences(self):
        p, x = self.p, self.x
        lag = p.lag(x)
        E, g, H, _, _ = p.evaluate(x, lag, projected=False, with_metric=True)
        A = p.coupled_friction_tangent(x, lag, H)
        v = np.random.default_rng(35).normal(size=x.shape)
        v /= np.linalg.norm(v)
        eps = 1e-7
        plus = p.evaluate(x+eps*v, lag, tangent=False)
        minus = p.evaluate(x-eps*v, lag, tangent=False)
        self.assertAlmostEqual((plus[0]-minus[0])/(2*eps), float(g@v.ravel()), delta=2e-7)
        fd = (plus[1]-minus[1])/(2*eps)
        self.assertLess(np.linalg.norm(H@v.ravel()-fd)/np.linalg.norm(fd), 2e-7)
        gp = p.evaluate(x+eps*v, p.lag(x+eps*v), tangent=False)[1]
        gm = p.evaluate(x-eps*v, p.lag(x-eps*v), tangent=False)[1]
        fresh = (gp-gm)/(2*eps)
        self.assertLess(np.linalg.norm(A@v.ravel()-fresh)/np.linalg.norm(fresh), 2e-7)
        self.assertGreater(np.linalg.norm(H@v.ravel()-fresh)/np.linalg.norm(fresh), .01)

    def test_changed_positions_previous_and_lag_cannot_reuse_stale_data(self):
        p, x = self.p, self.x
        lag0 = p.lag(x)
        first = self.assert_paired_equivalent(x, lag0)
        moved = x.copy()
        moved[p.n:, 1] += .0003
        moved[p.enrichment.base_nodes:p.n, 0] += .0001
        second = self.assert_paired_equivalent(moved, p.lag(moved))
        self.assertGreater(np.linalg.norm(first[1]-second[1]), 1e-3)
        # Put the relative tangential movement inside friction smoothing, so
        # changing only the previous accepted state necessarily changes force.
        p.previous[p.n:, 0] += .001997
        history = self.assert_paired_equivalent(moved, p.lag(moved))
        self.assertGreater(np.linalg.norm(history[1]-second[1]), 1e-3)
        different_lag = self.assert_paired_equivalent(moved, lag0)
        self.assertGreater(np.linalg.norm(different_lag[1]-history[1]), 1e-3)


    def test_reused_lu_preserves_accepted_frictional_p2_history(self):
        results = []
        for method in ('direct', 'reused-lu'):
            p = IPCProblem('patch', mu=.35, local_spacing=.2,
                           surface_spacing=.2, element='p2-interior',
                           linear_solver=method)
            x, old = p.rest.copy(), p.start
            log = io.StringIO()
            states, forces = [], []
            with contextlib.redirect_stdout(log):
                for depth in (0., .01, .02):
                    before = x.copy()
                    x, info = p.step(x, old, depth, time.monotonic()+30)
                    np.testing.assert_array_equal(p.previous, before)
                    fresh = p.evaluate(x, p.lag(x), tangent=False)
                    self.assertLessEqual(np.linalg.norm(fresh[1][p.free]), 1e-4)
                    self.assertGreater(fresh[3], 0.)
                    self.assertGreaterEqual(info['friction_work_n_mm'], -1e-12)
                    self.assertFalse(p.ipc.has_intersections(p.cmesh, x[p.surface]))
                    states.append(x.copy()); forces.append(info['reaction_fz_n'])
                    old = depth
            if method == 'reused-lu':
                stats = [json.loads(line[7:]) for line in log.getvalue().splitlines()
                         if line.startswith('LINEAR ')]
                self.assertTrue(stats)
                self.assertTrue(any(s['method'] == 'gmres_reused_lu' for s in stats))
                self.assertTrue(all(s['true_residual'] <= s['residual_limit'] for s in stats))
            results.append((states, forces))
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-8, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
