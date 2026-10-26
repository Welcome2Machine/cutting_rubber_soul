"""Residual-merit globalization safeguards; small native contact only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix, eye
from threadpoolctl import threadpool_limits

from prototype.ipc_reference import (BASELINE, IPCProblem, LoadStepFailure,
    adaptive_load_steps, parse_args, residual_merit_decrease, run, toolkit)


class ResidualMeritTests(unittest.TestCase):
    def test_true_derivative_armijo_and_meaningful_drop_are_separate_gates(self):
        self.assertTrue(residual_merit_decrease(1., .8, -.5, 1.))
        for args in ((1., .95, -.5, 1.), (1., .8, 0., 1.),
                     (1., .8, .1, 1.), (1., .8, -2000., 1.),
                     (1., .8, -.5, 0.), (1., .8, -.5, -1.),
                     (0., 0., -.5, 1.), (1., -.1, -.5, 1.)):
            with self.subTest(args=args):
                self.assertFalse(residual_merit_decrease(*args))
        for i in range(4):
            for bad in (np.nan, np.inf, -np.inf):
                args = [1., .8, -.5, 1.]
                args[i] = bad
                with self.subTest(index=i, bad=bad):
                    self.assertFalse(residual_merit_decrease(*args))

    def one_iteration(self, residual, jacobian, *, mode='residual-merit',
                      failure='iteration budget'):
        p = object.__new__(IPCProblem)
        p.max_iterations = 1
        p.linear_solver = 'direct'
        p.enrichment = None
        p.nodal_volume = None
        p.free = np.arange(2)
        p.total = 2
        p.options = SimpleNamespace(tip_radius_mm=1000.)
        p.stagnation_window = 0
        p.friction_globalization = mode
        p.lag = lambda x: None
        p.safe_step = lambda x, direction: 1.
        p.coupled_friction_tangent = lambda x, lag, H: csc_matrix(jacobian(x.ravel()))

        def evaluate(x, lag=None, tangent=True, **kwargs):
            result = (0., residual(x.ravel()), eye(2, format='csc'), 1.)
            return result+(eye(2, format='csc'),) if kwargs.get('with_metric') else result

        p.evaluate = evaluate
        log = io.StringIO()
        with contextlib.redirect_stdout(log), self.assertRaisesRegex(LoadStepFailure, failure):
            p.solve_friction(np.zeros((1, 2)), 0., time.monotonic()+2)
        return p.trial, log.getvalue()

    def test_nonsymmetric_positive_spectrum_does_not_imply_merit_descent(self):
        # The actual initial shift is .1. A has positive eigenvalues and M=I,
        # yet its PTC direction increases Euclidean residual norm.
        A = np.array([[1., 100.], [0., 1.]])
        r = np.array([-1., 1.])
        du = np.linalg.solve(A+.1*np.eye(2), -r)
        slope = float(r @ A @ du)
        self.assertGreater(slope, 0.)
        self.assertFalse(residual_merit_decrease(np.linalg.norm(r), .5, slope, 1.))
        trial, log = self.one_iteration(lambda x: r+A@x, lambda x: A)
        np.testing.assert_allclose(trial.ravel(), du, atol=2e-9, rtol=1e-9)
        self.assertGreater(np.linalg.norm(r+A@trial.ravel()), np.linalg.norm(r))
        self.assertIn('criterion=model_error', log)
        self.assertNotIn('criterion=fresh_residual_merit', log)

    def test_smooth_residual_can_improve_despite_rejected_linear_model(self):
        residual = lambda x: np.array([1.+x[0]-.6*x[0]**2, x[1]])
        jacobian = lambda x: np.diag([1.-1.2*x[0], 1.])
        trial, log = self.one_iteration(residual, jacobian)
        self.assertAlmostEqual(trial[0, 0], -1/1.1, places=8)
        self.assertIn('criterion=fresh_residual_merit', log)
        self.assertIn('iteration=1 fresh_residual_N=', log)
        self.assertIn('metric_shift=0.1', log.split('iteration=1')[1])

    def test_nonfinite_trial_is_rejected_by_actual_iteration(self):
        for bad in (np.nan, np.inf):
            residual = lambda x: np.array([bad if np.any(x) else 1., 0.])
            trial, log = self.one_iteration(residual, lambda x: np.eye(2), failure='model mismatch')
            np.testing.assert_array_equal(trial, np.zeros((1, 2)))
            self.assertNotIn('FRICTION_PTC_UPDATE', log)
            self.assertEqual(log.count('FRICTION_PTC_REJECT'), 18)

    def test_cli_preserves_frozen_baseline_and_rejects_discontinuous_ogc(self):
        with tempfile.TemporaryDirectory() as root:
            base = ['--output', root]
            self.assertEqual(parse_args(base).friction_globalization, 'model-error')
            self.assertEqual(parse_args(base+['--current', '--friction-globalization',
                                            'residual-merit']).friction_globalization, 'residual-merit')
            for extra in (['--baseline', '--friction-globalization', 'residual-merit'],
                          ['--contact-formulation', 'offset-geometric',
                           '--friction-globalization', 'residual-merit']):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    parse_args(base+extra)


class AdaptiveMeritTests(unittest.TestCase):
    def problem(self, predictor='secant', terminal=None):
        p = SimpleNamespace(predictor=predictor, friction_globalization='residual-merit',
            stagnation_window=40, previous=np.array([[-1.]]),
            _accepted_predictor_history=[(0., np.array([[0.]]))], calls=[])

        def step(x, old, target, deadline):
            p.calls.append((p.predictor, p.friction_globalization, p.stagnation_window))
            if terminal is not None:
                raise terminal
            if p.friction_globalization == 'residual-merit':
                p.previous[:] = 999.
                p._accepted_predictor_history.append((999., np.array([[999.]])))
                raise LoadStepFailure('candidate rejected')
            np.testing.assert_array_equal(p.previous, [[-1.]])
            self.assertEqual(len(p._accepted_predictor_history), 1)
            p.previous = x.copy()
            value = np.array([[target]])
            p._accepted_predictor_history.append((target, value.copy()))
            return value, dict(depth_mm=target)

        p.step = step
        return p

    def test_failure_restores_original_model_path_and_records_actual_strategy(self):
        for predictor in ('secant', 'affine'):
            p = self.problem(predictor)
            events = []
            with contextlib.redirect_stdout(io.StringIO()):
                result = list(adaptive_load_steps(p, np.array([[0.]]), 0., .01,
                    time.monotonic()+2, .0025, events.append))
            self.assertEqual(p.calls, [(predictor, 'residual-merit', 40 if predictor=='secant' else 0),
                                       ('affine', 'model-error', 0)])
            self.assertEqual(result[0][1]['load_step']['friction_globalization'], 'model-error')
            self.assertEqual(events[0]['friction_globalization'], 'residual-merit')
            self.assertEqual(p.predictor, predictor)
            self.assertEqual(p.friction_globalization, 'residual-merit')
            self.assertEqual(p.stagnation_window, 40)
            np.testing.assert_array_equal(p.previous, [[0.]])
            self.assertEqual([d for d, _ in p._accepted_predictor_history], [0., .01])

    def test_timeout_and_implementation_errors_restore_modes_without_retry(self):
        for error in (TimeoutError('wall budget'), RuntimeError('implementation fault')):
            p = self.problem(terminal=error)
            with self.assertRaises(type(error)):
                list(adaptive_load_steps(p, np.array([[0.]]), 0., .01, time.monotonic()+2, .0025))
            self.assertEqual(len(p.calls), 1)
            self.assertEqual(p.friction_globalization, 'residual-merit')
            self.assertEqual(p.predictor, 'secant')
            self.assertEqual(p.stagnation_window, 40)


class NativeMeritTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            toolkit().set_num_threads(1)
        except ModuleNotFoundError:
            raise unittest.SkipTest('Optional ipctk==1.6.0 not installed')

    def test_three_patch_states_preserve_physics_history_and_report(self):
        common = ['--case', 'patch', '--friction', '.35', '--local-spacing', '.2',
                  '--surface-spacing', '.2', '--increment', '.01', '--depth', '.02',
                  '--seconds', '10', '--threads', '1', '--hessian-mode', 'regularized',
                  '--no-visualize', '--predictor', 'secant', '--adaptive-load']
        with tempfile.TemporaryDirectory() as work, threadpool_limits(limits=1):
            reports, states = {}, {}
            for mode in ('model-error', 'residual-merit'):
                out = Path(work)/mode
                args = parse_args(common+['--output', str(out), '--friction-globalization', mode])
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(run(args))
                report = reports[mode] = json.loads((out/'result.json').read_text())
                self.assertEqual(report['friction_globalization'], mode)
                self.assertEqual([s['depth_mm'] for s in report['steps']], [0., .01, .02])
                self.assertEqual(report['force_tolerance_n'], BASELINE['force_tolerance_n'])
                if mode == 'residual-merit':
                    self.assertEqual(report['friction_merit_settings']['armijo_coefficient'], 1e-4)
                    self.assertEqual(report['friction_merit_settings']['residual_ratio_limit'], .9)
                    self.assertEqual(report['friction_merit_settings']['adaptive_fallback'], 'affine/model-error')
                else:
                    self.assertIsNone(report['friction_merit_settings'])
                p = IPCProblem('patch', mu=.35, local_spacing=.2, surface_spacing=.2,
                               hessian_mode='regularized', friction_globalization=mode)
                states[mode] = []
                for i, record in enumerate(report['steps']):
                    with np.load(out/'states'/f'{i:04d}.npz') as saved:
                        x, previous = saved['positions'].copy(), saved['previous_positions'].copy()
                    p.previous = previous
                    checked = p.acceptance(x, record['depth_mm'], 0, 0)
                    self.assertLessEqual(checked['residual_n'], BASELINE['force_tolerance_n'])
                    self.assertGreater(checked['min_j'], 0.)
                    self.assertGreater(checked['minimum_distance_mm'], 0.)
                    self.assertGreaterEqual(checked['friction_work_n_mm'], -1e-12)
                    self.assertAlmostEqual(checked['reaction_fz_n'], record['reaction_fz_n'], places=10)
                    if i:
                        np.testing.assert_array_equal(previous, states[mode][-1])
                    states[mode].append(x)
            for old, new in zip(reports['model-error']['steps'], reports['residual-merit']['steps']):
                self.assertAlmostEqual(old['reaction_fz_n'], new['reaction_fz_n'], delta=2e-4)
            for old, new in zip(states['model-error'], states['residual-merit']):
                np.testing.assert_allclose(old, new, rtol=0., atol=1e-5)

            # Missing new metadata is explicitly the historical model-error
            # strategy. Its actual accepted prefix survives an opt-in resume.
            source = Path(work)/'model-error'
            legacy = json.loads((source/'result.json').read_text())
            legacy.pop('friction_globalization')
            legacy.pop('friction_merit_settings')
            (source/'result.json').write_text(json.dumps(legacy))
            resumed = Path(work)/'resumed'
            args = parse_args(common+['--output', str(resumed), '--depth', '.03',
                '--resume', str(source), '--friction-globalization', 'residual-merit'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(run(args))
            report = json.loads((resumed/'result.json').read_text())
            self.assertEqual(report['resume']['previous_friction_globalization'], 'model-error')
            self.assertIsNone(report['resume']['previous_friction_merit_settings'])
            self.assertEqual(report['steps'][:3], legacy['steps'])
            self.assertEqual(report['steps'][-1]['friction_globalization'], 'residual-merit')
            self.assertEqual(json.loads((resumed/'resume_source_result.json').read_text()), legacy)
            for i in range(3):
                self.assertEqual((resumed/'states'/f'{i:04d}.npz').read_bytes(),
                                 (source/'states'/f'{i:04d}.npz').read_bytes())
            with np.load(resumed/'states/0003.npz') as last:
                np.testing.assert_array_equal(last['previous_positions'], states['model-error'][-1])


if __name__ == '__main__':
    unittest.main()
