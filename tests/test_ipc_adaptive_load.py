"""Accepted-history rollback, bounded cutback, and native substep restart."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from prototype.ipc_reference import (IPCProblem, LoadStepFailure, adaptive_load_steps,
                                     parse_args, run, toolkit)


class FakeProblem:
    def __init__(self, behavior):
        self.predictor = 'secant'
        self.stagnation_window = 40
        self.previous = np.array([[-1.]])
        self._accepted_predictor_history = [(0., np.array([[0.]]))]
        self.behavior = behavior
        self.calls = []
        self.windows = []

    def step(self, x, old, target, deadline):
        np.testing.assert_array_equal(x, [[old]])
        self.calls.append((old, target, self.predictor))
        self.windows.append(self.stagnation_window)
        self.previous = x.copy()
        if self.behavior(old, target, self.predictor):
            # Deliberately corrupt scratch/history to exercise real rollback.
            self._accepted_predictor_history.append((999., np.array([[999.]])))
            self.previous[:] = 999.
            raise LoadStepFailure('synthetic nonlinear failure')
        value = np.array([[target]])
        self._accepted_predictor_history = (self._accepted_predictor_history+[(target, value.copy())])[-2:]
        return value, dict(depth_mm=target)


class AdaptiveLoadTests(unittest.TestCase):
    def collect(self, problem, minimum=.0025, target=.01, events=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return list(adaptive_load_steps(problem, np.array([[0.]]), 0., target,
                                           time.monotonic()+10, minimum,
                                           None if events is None else events.append))

    def test_failed_trials_rollback_before_affine_and_every_substep_keeps_history(self):
        p = FakeProblem(lambda a, b, mode: b-a > .005+1e-12)
        events = []
        values = self.collect(p, events=events)
        self.assertEqual([r['depth_mm'] for _, r in values], [.005, .01])
        self.assertEqual(len(events), 2)
        self.assertEqual(p.calls[:2], [(0., .01, 'secant'), (0., .01, 'affine')])
        np.testing.assert_array_equal(p.previous, [[.005]])
        self.assertEqual([d for d, _ in p._accepted_predictor_history], [.005, .01])
        self.assertEqual(p.predictor, 'secant')

    def test_affine_retries_same_load_without_creating_substeps(self):
        p = FakeProblem(lambda a, b, mode: mode == 'secant')
        values = self.collect(p)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][1]['load_step']['predictor'], 'affine')
        np.testing.assert_array_equal(p.previous, [[0.]])
        self.assertEqual(p.windows, [40, 0])
        self.assertEqual(p.stagnation_window, 40)

    def test_minimum_increment_is_terminal_not_accepted(self):
        p = FakeProblem(lambda *args: True)
        with self.assertRaisesRegex(LoadStepFailure, 'adaptive load exhausted'):
            self.collect(p)
        self.assertEqual([d for d, _ in p._accepted_predictor_history], [0.])
        np.testing.assert_array_equal(p.failed_previous, [[0.]])
        self.assertTrue(all(b-a >= .0025-1e-12 for a, b, _ in p.calls))

    def test_unrepresentable_bisection_is_terminal_even_with_tiny_minimum(self):
        p = FakeProblem(lambda *args: True)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(p, 'step', side_effect=LoadStepFailure('rejected')), \
                self.assertRaisesRegex(LoadStepFailure, 'adaptive load exhausted'):
            list(adaptive_load_steps(p, np.array([[1.]]), 1., np.nextafter(1., 2.),
                                     time.monotonic()+1, 1e-30))

    def test_timeout_and_unexpected_errors_are_never_retried(self):
        for error in (TimeoutError('deadline'), RuntimeError('implementation fault'), OSError('disk'),
                      FloatingPointError('unrelated constitutive error')):
            p = FakeProblem(lambda *args: False)
            with patch.object(p, 'step', side_effect=error) as call, self.assertRaises(type(error)):
                self.collect(p)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(p.predictor, 'secant')

    def test_enriched_geometric_stall_cuts_back_and_preserves_history(self):
        from types import SimpleNamespace
        p = FakeProblem(lambda *args: False)
        original = p.step
        probe = object.__new__(IPCProblem)
        probe.enrichment = object()
        probe.options = SimpleNamespace(tip_radius_mm=.02)
        probe.safe_step = lambda x, d: 1e-12
        def step(x, old, target, deadline):
            if target-old > .005+1e-12:
                probe._ptc_trial(np.zeros((1,2)), np.ones((1,2)), 1.)
            return original(x, old, target, deadline)
        p.step = step
        events = []
        values = self.collect(p, events=events)
        self.assertEqual([r['depth_mm'] for _,r in values], [.005,.01])
        self.assertTrue(all('geometric stagnation' in e['error'] for e in events))
        np.testing.assert_array_equal(p.previous, [[.005]])

    def test_positive_path_bound_does_not_override_rounded_endpoint_geometry(self):
        from types import SimpleNamespace
        probe = object.__new__(IPCProblem)
        probe.enrichment = SimpleNamespace(minimum_j=lambda u: -2e-15)
        probe.n = 1
        probe.mesh = SimpleNamespace(reference_xy=np.zeros((1,2)))
        probe.options = SimpleNamespace(tip_radius_mm=.02)
        probe.safe_step = lambda x, d: 1.
        with self.assertRaisesRegex(LoadStepFailure, 'rounded trial'):
            probe._ptc_trial(np.zeros((1,2)),np.ones((1,2)),1.)
        probe.enrichment.minimum_j = lambda u: .8
        trial,alpha,_ = probe._ptc_trial(np.zeros((1,2)),np.ones((1,2)),1.)
        np.testing.assert_array_equal(trial, np.full((1,2),.01))
        self.assertEqual(alpha,.01)

    def test_partial_substeps_are_yielded_before_later_failure(self):
        p = FakeProblem(lambda a, b, mode: b > .005)
        accepted = []
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(LoadStepFailure):
            for _, record in adaptive_load_steps(p, np.array([[0.]]), 0., .01,
                                                 time.monotonic()+10, .0025):
                accepted.append(record['depth_mm'])
        self.assertEqual(accepted, [.005])
        self.assertEqual(p._accepted_predictor_history[-1][0], .005)
        np.testing.assert_array_equal(p.failed_previous, [[.005]])

    def test_cli_protects_baseline_and_validates_minimum(self):
        with tempfile.TemporaryDirectory() as work, contextlib.redirect_stderr(io.StringIO()):
            for extra in (['--baseline', '--adaptive-load'],
                          ['--adaptive-load', '--minimum-increment', '0'],
                          ['--adaptive-load', '--minimum-increment', '1']):
                with self.assertRaises(SystemExit):
                    parse_args(['--output', str(Path(work)/'run'), *extra])

    def test_native_saved_substeps_resume_with_identical_friction_history(self):
        try:
            toolkit().set_num_threads(1)
        except ModuleNotFoundError:
            self.skipTest('Optional ipctk==1.6.0 not installed')
        common = ['--case', 'patch', '--friction', '.35', '--local-spacing', '.2',
                  '--surface-spacing', '.2', '--increment', '.01', '--seconds', '30',
                  '--threads', '1', '--hessian-mode', 'regularized', '--no-visualize',
                  '--predictor', 'secant', '--adaptive-load']
        original_step = IPCProblem.step

        def force_cutback(problem, x, old, depth, deadline):
            if abs(old) < 1e-12 and abs(depth-.01) < 1e-12:
                raise LoadStepFailure('test forces one cutback before real native solves')
            return original_step(problem, x, old, depth, deadline)

        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(IPCProblem, 'step', force_cutback):
                for name, depth, extra in (('prefix', '.01', []), ('continuous', '.02', []),
                                          ('resumed', '.02', ['--resume', str(root/'prefix')])):
                    self.assertTrue(run(parse_args(common+['--output', str(root/name), '--depth', depth]+extra)))
            prefix = json.loads((root/'prefix/result.json').read_text())
            self.assertEqual([s['depth_mm'] for s in prefix['steps']], [0., .005, .01])
            self.assertEqual(len(prefix['rejected_load_attempts']), 2)
            for i in range(4):
                with np.load(root/'continuous/states'/f'{i:04d}.npz') as a, \
                        np.load(root/'resumed/states'/f'{i:04d}.npz') as b:
                    for key in ('positions', 'previous_positions', 'depth_mm'):
                        np.testing.assert_array_equal(a[key], b[key])


if __name__ == '__main__':
    unittest.main()
