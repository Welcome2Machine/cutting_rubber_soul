"""Same-ray PTC globalization; original residual and geometry gates are retained."""
import contextlib
import io
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import diags, eye

from prototype.ipc_reference import (IPCProblem, LoadStepFailure,
                                     residual_armijo_decrease, residual_merit_decrease)


class PTCModelBacktrackingTests(unittest.TestCase):
    def problem(self, residual, jacobian, *, backtracks=None):
        p = object.__new__(IPCProblem)
        p.max_iterations = 1
        p.linear_solver = 'direct'
        p.enrichment = None
        p.nodal_volume = None
        p.free = np.array([0])
        p.total = 4
        p.options = SimpleNamespace(tip_radius_mm=1000.)
        p.stagnation_window = 0
        p.friction_globalization = 'model-error'
        p.lag = lambda x: None
        p.safe_step = lambda x, direction: 1.
        p.coupled_friction_tangent = lambda x, lag, H: H
        p.evaluated = []
        p.previous = np.array([[0., 2.], [3., 4.]])
        p.material_metric = np.eye(3)[None]
        p.cohesive_maximum = np.zeros((1, 2))
        if backtracks is not None:
            p.ptc_model_backtracks = backtracks

        def evaluate(x, lag=None, tangent=True, **kwargs):
            p.evaluated.append((x.copy(), tangent))
            result = (0., np.array([residual(x[0, 0]), 0., 0., 0.]),
                      diags([jacobian(x[0, 0]), 1., 1., 1.], format='csc'), 1.)
            return result+(eye(4, format='csc'),) if kwargs.get('with_metric') else result

        p.evaluate = evaluate
        return p

    def one_direction(self, p, failure='iteration budget'):
        log = io.StringIO()
        with contextlib.redirect_stdout(log), self.assertRaisesRegex(LoadStepFailure, failure):
            p.solve_friction(p.previous.copy(), .01, time.monotonic()+3)
        traces = [json.loads(line.split(' ', 1)[1]) for line in log.getvalue().splitlines()
                  if line.startswith('PTC_MODEL_BACKTRACK ')]
        return log.getvalue(), traces

    def test_zero_option_preserves_original_iteration_and_evaluation_sequence(self):
        force = lambda u: u-1.+100.*max(u-.1, 0.)**2
        tangent = lambda u: 1.+200.*max(u-.1, 0.)
        original = self.problem(force, tangent)
        explicit_zero = self.problem(force, tangent, backtracks=0)
        a, ta = self.one_direction(original)
        b, tb = self.one_direction(explicit_zero)
        self.assertEqual(ta, [])
        self.assertEqual(tb, [])
        self.assertEqual(a.count('FRICTION_PTC_REJECT'), b.count('FRICTION_PTC_REJECT'))
        self.assertEqual(len(original.evaluated), len(explicit_zero.evaluated))
        for (x, tx), (y, ty) in zip(original.evaluated, explicit_zero.evaluated):
            self.assertEqual(tx, ty)
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(original.trial, explicit_zero.trial)

    def test_nonlinear_activation_uses_scaled_true_model_and_safe_ray_prefix(self):
        force = lambda u: u-1.+100.*max(u-.1, 0.)**2
        tangent = lambda u: 1.+200.*max(u-.1, 0.)
        p = self.problem(force, tangent, backtracks=12)
        p.safe_step = lambda x, direction: .5
        before = p.previous.copy()
        material = p.material_metric.copy()
        cohesive = p.cohesive_maximum.copy()
        log, traces = self.one_direction(p)
        self.assertEqual(log.count('FRICTION_PTC_REJECT'), 0)
        self.assertEqual([q['backtrack'] for q in traces], [0, 1, 2])
        self.assertEqual([q['status'] for q in traces], ['rejected', 'rejected', 'accepted'])
        du = 1./(1.+.1*(1.+1e-9))
        for q in traces:
            self.assertLessEqual(q['alpha'], q['ccd_j_safe_alpha'])
            predicted_error = abs(force(q['alpha']*du)+1.-q['alpha']*du)
            self.assertAlmostEqual(q['model_error_n'], predicted_error, places=13)
        self.assertLessEqual(traces[-1]['model_error_n'], traces[-1]['model_limit_n'])
        self.assertAlmostEqual(p.trial[0, 0], traces[-1]['alpha']*du)
        self.assertGreater(abs(force(p.trial[0, 0])), 1e-4)  # not falsely accepted as equilibrium
        for x, _ in p.evaluated:
            np.testing.assert_array_equal(x.ravel()[[1, 2, 3]], before.ravel()[[1, 2, 3]])
        np.testing.assert_array_equal(p.previous, before)
        np.testing.assert_array_equal(p.material_metric, material)
        np.testing.assert_array_equal(p.cohesive_maximum, cohesive)
        self.assertEqual(p.ptc_model_backtrack_summary['shortened_accepted_directions'], 1)

    def test_already_accurate_trial_does_not_add_force_evaluations(self):
        p = self.problem(lambda u: u-1., lambda u: 1., backtracks=20)
        _, traces = self.one_direction(p)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]['backtrack'], 0)
        self.assertEqual(traces[0]['status'], 'accepted')
        self.assertEqual(len(p.evaluated), 3)
        self.assertEqual(p.ptc_model_backtrack_summary['trial_evaluations'], 0)

    def test_nonfinite_trials_exhaust_both_original_and_backtrack_budgets(self):
        p = self.problem(lambda u: -1. if u == 0. else np.nan, lambda u: 1., backtracks=2)
        log, traces = self.one_direction(p, failure='model mismatch')
        self.assertEqual(log.count('FRICTION_PTC_REJECT'), 18)
        self.assertEqual(len(traces), 18*3)
        self.assertTrue(all(q['status'] == 'rejected' for q in traces))
        self.assertNotIn('FRICTION_PTC_UPDATE', log)
        np.testing.assert_array_equal(p.trial, p.previous)

    def test_backtracking_checks_wall_budget_before_each_extra_evaluation(self):
        p = self.problem(lambda u: u-1.+100.*max(u-.1, 0.)**2,
                         lambda u: 1.+200.*max(u-.1, 0.), backtracks=20)
        with patch('prototype.ipc_reference.time.monotonic', side_effect=[0., 0., 2.]), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(TimeoutError):
            p.solve_friction(p.previous.copy(), .01, 1.)
        self.assertEqual(len(p.evaluated), 2)
        np.testing.assert_array_equal(p.trial, p.previous)

    def test_explicit_false_guard_preserves_existing_backtrack_sequence(self):
        force = lambda u: u-1.+100.*max(u-.1, 0.)**2
        tangent = lambda u: 1.+200.*max(u-.1, 0.)
        original = self.problem(force, tangent, backtracks=12)
        explicit = self.problem(force, tangent, backtracks=12)
        explicit.ptc_residual_guard = False
        _, a = self.one_direction(original)
        _, b = self.one_direction(explicit)
        self.assertEqual(a, b)
        self.assertEqual(len(original.evaluated), len(explicit.evaluated))
        for (x, tx), (y, ty) in zip(original.evaluated, explicit.evaluated):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(tx, ty)

    def test_guard_vetoes_model_accurate_growth_and_allows_small_true_decrease(self):
        # This smooth nonlinear residual has an exact analytic tangent. At the
        # imposed CCD-safe alpha the old absolute model-error test accepts a
        # residual increase. A half-step decreases by less than 10 percent.
        force = lambda u: 1.-u+25.*u*u-u**3
        tangent = lambda u: -1.+50.*u-3.*u*u
        old = self.problem(force, tangent, backtracks=12)
        guarded = self.problem(force, tangent, backtracks=12)
        guarded.ptc_residual_guard = True
        old.safe_step = guarded.safe_step = lambda x, direction: 1./16.
        _, before = self.one_direction(old)
        _, after = self.one_direction(guarded)
        self.assertEqual(before[0]['status'], 'accepted')
        self.assertGreater(before[0]['trial_residual_n'], before[0]['residual_n'])
        self.assertEqual(after[0]['status'], 'rejected')
        self.assertFalse(after[0]['residual_guard_passed'])
        self.assertLessEqual(after[0]['model_error_n'], after[0]['model_limit_n'])
        self.assertEqual(after[-1]['status'], 'accepted')
        self.assertTrue(after[-1]['residual_guard_passed'])
        self.assertGreater(after[-1]['trial_residual_n'], .9*after[-1]['residual_n'])
        self.assertLess(after[-1]['trial_residual_n'], after[-1]['residual_n'])
        self.assertEqual(after[-1]['alpha'], after[0]['alpha']/2.)
        self.assertLessEqual(after[-1]['trial_residual_n']**2, after[-1]['residual_merit_limit_n2'])
        for x, _ in guarded.evaluated:
            np.testing.assert_array_equal(x.ravel()[[1, 2, 3]], guarded.previous.ravel()[[1, 2, 3]])
        np.testing.assert_array_equal(guarded.material_metric, np.eye(3)[None])
        np.testing.assert_array_equal(guarded.cohesive_maximum, np.zeros((1, 2)))

    def test_guard_never_accepts_non_descent_direction_as_equilibrium(self):
        p = self.problem(lambda u: 1.-.05*u, lambda u: -.05, backtracks=2)
        p.ptc_residual_guard = True
        p.safe_step = lambda x, direction: .01
        log, traces = self.one_direction(p, failure='model mismatch')
        self.assertEqual(len(traces), 18*3)
        self.assertTrue(all(q['residual_merit_slope'] > 0. for q in traces))
        self.assertTrue(all(q['status'] == 'rejected' for q in traces))
        self.assertNotIn('FRICTION_PTC_UPDATE', log)
        np.testing.assert_array_equal(p.trial, p.previous)

    def test_guard_keeps_original_final_force_tolerance(self):
        p = self.problem(lambda u: u-1e-3, lambda u: 1., backtracks=12)
        p.ptc_residual_guard = True
        with contextlib.redirect_stdout(io.StringIO()):
            x, iteration = p.solve_friction(p.previous.copy(), .01, time.monotonic()+3)
        self.assertEqual(iteration, 1)
        self.assertLessEqual(abs(x[0, 0]-1e-3), 1e-4)
        np.testing.assert_array_equal(p.previous, [[0., 2.], [3., 4.]])

    def test_standard_armijo_has_no_ten_percent_requirement_or_roundoff_noop(self):
        self.assertTrue(residual_armijo_decrease(1., .999, -1., .001))
        self.assertFalse(residual_merit_decrease(1., .999, -1., .001))
        for values in ((1., 1., -1., 1e-30), (1., 1.01, -1., .01),
                       (1., .9, 0., .1), (1., .9, 1., .1), (1., np.nan, -1., .1)):
            with self.subTest(values=values):
                self.assertFalse(residual_armijo_decrease(*values))

    def test_default_shift_policy_preserves_original_trial_sequence(self):
        force = lambda u: u-1.+100.*max(u-.1, 0.)**2
        tangent = lambda u: 1.+200.*max(u-.1, 0.)
        original = self.problem(force, tangent, backtracks=12)
        explicit = self.problem(force, tangent, backtracks=12)
        explicit.ptc_shift_policy = 'residual_ratio'
        _, a = self.one_direction(original)
        _, b = self.one_direction(explicit)
        self.assertEqual(a, b)
        self.assertEqual(original.ptc_shift_summary, explicit.ptc_shift_summary)
        self.assertEqual(len(original.evaluated), len(explicit.evaluated))
        for (x, tx), (y, ty) in zip(original.evaluated, explicit.evaluated):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(tx, ty)

    def test_backtrack_aware_shift_uses_only_extra_model_halvings(self):
        p = self.problem(lambda u: u-1.+100.*max(u-.1, 0.)**2,
                         lambda u: 1.+200.*max(u-.1, 0.), backtracks=12)
        p.ptc_shift_policy = 'backtrack_aware'
        p.ptc_residual_guard = False
        p.safe_step = lambda x, direction: .5
        log, traces = self.one_direction(p)
        last = traces[-1]
        self.assertEqual(last['backtrack'], 2)
        self.assertEqual(last['alpha'], .125)
        shifts = [json.loads(line.split(' ', 1)[1]) for line in log.splitlines()
                  if line.startswith('PTC_SHIFT_POLICY ')]
        self.assertEqual(len(shifts), 1)
        self.assertEqual(shifts[0]['model_halvings'], 2)
        self.assertAlmostEqual(shifts[0]['shift_next'], .4)
        self.assertEqual(p.ptc_shift_summary['last_shift_next'], shifts[0]['shift_next'])
        self.assertEqual(p.ptc_shift_summary['backtrack_increases'], 1)
        self.assertNotEqual(shifts[0]['shift_next'], .1/last['alpha'])
        np.testing.assert_array_equal(p.previous, [[0., 2.], [3., 4.]])
        np.testing.assert_array_equal(p.cohesive_maximum, np.zeros((1, 2)))

    def test_geometric_clipping_without_model_halving_does_not_raise_shift(self):
        p = self.problem(lambda u: u-1., lambda u: 1., backtracks=20)
        p.ptc_shift_policy = 'backtrack_aware'
        p.safe_step = lambda x, direction: .01
        _, traces = self.one_direction(p)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]['backtrack'], 0)
        self.assertEqual(traces[0]['alpha'], .01)
        self.assertEqual(p.ptc_shift_summary['last_shift_next'],
                         p.ptc_shift_summary['last_residual_ratio_next_shift'])
        self.assertEqual(p.ptc_shift_summary['backtrack_increases'], 0)


if __name__ == '__main__':
    unittest.main()
