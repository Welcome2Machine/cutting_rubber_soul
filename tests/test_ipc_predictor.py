"""Predictor selection safeguards plus a real small-patch continuation check."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.ipc_reference import IPCProblem, parse_args, restore_accepted_prefix, run, toolkit


class PredictorTests(unittest.TestCase):
    def setUp(self):
        try:
            toolkit().set_num_threads(1)
        except ModuleNotFoundError:
            self.skipTest('Optional ipctk==1.6.0 not installed')
        threads = threadpool_limits(limits=1)
        threads.__enter__()
        self.addCleanup(threads.__exit__, None, None, None)
        self.p = IPCProblem('patch', mu=.35, local_spacing=.2, surface_spacing=.2,
                            hessian_mode='regularized', predictor='secant')
        p = self.p
        # Synthetic boundary-valid histories exercise initial-guess selection;
        # only the separate native continuation test claims force acceptance.
        self.older = p.rest.copy()
        self.older[p.n:, 1] -= 0.-p.start
        self.old = self.older.copy()
        self.old[:p.n, 1] -= .01*p.mesh.local_sn[:, 1]/p.config.height_mm
        self.old[p.n:, 1] -= .01
        self.old.ravel()[p.free[0]] += .001
        p.seed_predictor_history([(0., self.older), (.01, self.old)])
        p.previous = self.old.copy()
        self.affine = self.old.copy()
        self.affine[:p.n, 1] -= .01*p.mesh.local_sn[:, 1]/p.config.height_mm
        self.affine[p.n:, 1] -= .01
        self.secant = 2*self.old-self.older
        self.secant[p.n:] = self.affine[p.n:]

    def select(self, evaluate, safe=None):
        p = self.p
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(p, 'evaluate', side_effect=evaluate), \
                patch.object(p, 'lag', return_value=None), \
                patch.object(p, 'safe_step', side_effect=safe or (lambda x, d: 1.)), \
                patch.object(p.ipc, 'has_intersections', return_value=False):
            return p._loading_predictor(self.old, .01, .02)

    def force_near(self, desired):
        def evaluate(x, lag, tangent):
            np.testing.assert_array_equal(self.p.previous, self.old)
            g = np.zeros(self.p.total)
            g[self.p.free[0]] = np.linalg.norm(x-desired)
            return 0., g, None, .9
        return evaluate

    def test_residual_selects_half_secant_without_changing_friction_or_boundaries(self):
        desired = .5*(self.affine+self.secant)
        desired[self.p.n:] = self.affine[self.p.n:]
        chosen = self.select(self.force_near(desired))
        np.testing.assert_allclose(chosen, desired, atol=1e-15)
        fixed = np.setdiff1d(np.arange(2*self.p.n), self.p.free)
        np.testing.assert_array_equal(chosen.ravel()[fixed], self.p.rest.ravel()[fixed])
        np.testing.assert_array_equal(chosen[self.p.n:], self.affine[self.p.n:])
        np.testing.assert_array_equal(self.p.previous, self.old)

    def test_worse_secant_falls_back_to_affine(self):
        np.testing.assert_array_equal(self.select(self.force_near(self.affine)), self.affine)

    def test_unsafe_candidates_not_evaluated_even_if_their_force_would_be_better(self):
        calls = []
        def evaluate(x, lag, tangent):
            calls.append(x.copy())
            return self.force_near(self.secant)(x, lag, tangent)
        def safe(x, direction):
            np.testing.assert_array_equal(x, self.old)
            return 1. if np.array_equal(x+direction, self.affine) else .5
        chosen = self.select(evaluate, safe)
        np.testing.assert_array_equal(chosen, self.affine)
        self.assertEqual(len(calls), 1)

    def test_nonfinite_force_candidates_fall_back(self):
        def evaluate(x, lag, tangent):
            g = np.zeros(self.p.total)
            g[self.p.free[0]] = 1. if np.array_equal(x, self.affine) else np.nan
            return 0., g, None, .9
        np.testing.assert_array_equal(self.select(evaluate), self.affine)

    def test_no_safe_candidate_preserves_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'Prescribed-motion predictor is not collision free'):
            self.select(self.force_near(self.secant), lambda x, d: .5)

    def test_missing_or_mismatched_history_uses_only_affine(self):
        self.p.seed_predictor_history([(.01, self.old)])
        evaluate = self.force_near(self.secant)
        with patch.object(self.p, 'evaluate', side_effect=evaluate) as observed:
            chosen = self.select(observed)
        np.testing.assert_array_equal(chosen, self.affine)
        self.assertEqual(observed.call_count, 1)
        wrong = self.old.copy()
        wrong.ravel()[self.p.free[0]] += .001
        self.p.seed_predictor_history([(0., self.older), (.01, wrong)])
        np.testing.assert_array_equal(self.select(evaluate), self.affine)

    def test_failure_before_or_at_acceptance_does_not_update_history(self):
        p = self.p
        before = [(d, x.copy()) for d, x in p._accepted_predictor_history]
        for failure in ('solve', 'acceptance'):
            with patch.object(p, '_loading_predictor', return_value=self.affine.copy()), \
                    patch.object(p, 'solve_friction', side_effect=(RuntimeError('failed solve')
                        if failure == 'solve' else None), return_value=(self.affine.copy(), 3)), \
                    patch.object(p, 'acceptance', side_effect=RuntimeError('failed acceptance')):
                with self.assertRaises(RuntimeError):
                    p.step(self.old, .01, .02, time.monotonic()+10)
            for (d, x), (expected_d, expected_x) in zip(p._accepted_predictor_history, before):
                self.assertEqual(d, expected_d)
                np.testing.assert_array_equal(x, expected_x)
            np.testing.assert_array_equal(p.previous, self.old)

    def test_seed_copies_history_rejects_invalid_input_and_keeps_previous(self):
        p = self.p
        history = self.old.copy()
        p.seed_predictor_history([(.01, history)])
        history[:] = 0.
        np.testing.assert_array_equal(p._accepted_predictor_history[0][1], self.old)
        for states in ([], [(0., self.older)]*2, [(-1., self.older)], [(.01, history)]):
            with self.assertRaises(ValueError):
                p.seed_predictor_history(states)
        np.testing.assert_array_equal(p.previous, self.old)

    def test_blade_roundoff_tolerance_is_scaled_capped_and_rejects_real_displacement(self):
        p = self.p
        # Use a large reference coordinate to exercise the scaled comparison,
        # independent of local experiment/checkpoint fixtures.
        p.rest[p.n:, 1] += 60.
        positions = p.rest.copy()
        positions[p.n:, 1] -= .02-p.start
        tolerance = p.blade_boundary_roundoff_tolerance(.02)
        self.assertGreater(tolerance, 1e-12)
        self.assertLessEqual(tolerance, 1e-10)
        rounded = positions.copy()
        rounded[p.n:, 1] -= 1.15e-12
        p.seed_predictor_history([(.02, rounded)])
        for increment in (1e-9, 1e-6):
            moved = positions.copy()
            moved[p.n:, 1] += increment
            with self.assertRaisesRegex(ValueError, 'prescribed boundaries'):
                p.seed_predictor_history([(.02, moved)])
        fixed = np.setdiff1d(np.arange(2*p.n), p.free)
        moved = positions.copy()
        moved.ravel()[fixed[0]] += 2e-12
        with self.assertRaisesRegex(ValueError, 'prescribed boundaries'):
            p.seed_predictor_history([(.02, moved)])
        self.assertEqual(p.blade_boundary_roundoff_tolerance(1e10), 1e-10)

    def test_defaults_and_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'run')
            self.assertEqual(parse_args(['--output', path]).predictor, 'affine')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--output', path, '--baseline', '--predictor', 'secant'])

    def test_native_continuous_and_restored_history_and_force_acceptance_agree(self):
        common = ['--case', 'patch', '--no-visualize', '--friction', '.35',
                  '--local-spacing', '.2', '--surface-spacing', '.2', '--increment', '.01',
                  '--seconds', '30', '--threads', '1', '--hessian-mode', 'regularized',
                  '--predictor', 'secant']
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root/'prefix'
            self.assertTrue(run(parse_args(common+['--output', str(source), '--depth', '.01'])))
            report = json.loads((source/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(report['predictor'], 'secant')
            restored_report = dict(report, steps=[], status='running', completed=False, target_mm=.02)
            out = root/'restored'
            (out/'states').mkdir(parents=True)
            p = self.p
            x, depth = restore_accepted_prefix(source, out, restored_report, p, np.array([0., .01, .02]))
            with np.load(source/'states/0001.npz') as state:
                np.testing.assert_array_equal(p.previous, state['previous_positions'])
                np.testing.assert_array_equal(p._accepted_predictor_history[0][1], state['previous_positions'])
                np.testing.assert_array_equal(p._accepted_predictor_history[1][1], state['positions'])
            self.assertEqual([d for d, _ in p._accepted_predictor_history], [0., .01])
            self.assertEqual(restored_report['resume']['previous_predictor'], 'secant')
            resumed_x, resumed = p.step(x, depth, .02, time.monotonic()+30)
            q = IPCProblem('patch', mu=.35, local_spacing=.2, surface_spacing=.2,
                           hessian_mode='regularized', predictor='secant')
            continuous_x, old = q.rest.copy(), q.start
            for target in (0., .01, .02):
                previous = continuous_x.copy()
                continuous_x, accepted = q.step(continuous_x, old, target, time.monotonic()+30)
                np.testing.assert_array_equal(q.previous, previous)
                self.assertLessEqual(accepted['residual_n'], 1e-4)
                self.assertGreater(accepted['min_j'], 0)
                self.assertGreater(accepted['minimum_distance_mm'], 0)
                self.assertGreaterEqual(accepted['friction_work_n_mm'], -1e-12)
                old = target
            np.testing.assert_allclose(resumed_x, continuous_x, rtol=0, atol=1e-11)
            self.assertAlmostEqual(resumed['reaction_fz_n'], accepted['reaction_fz_n'], delta=1e-9)
            self.assertEqual([d for d, _ in q._accepted_predictor_history], [.01, .02])


if __name__ == '__main__':
    unittest.main()
