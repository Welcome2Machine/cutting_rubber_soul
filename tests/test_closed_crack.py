"""Bounded kernel checks for the prescribed, already debonded seam benchmark.

These checks do not establish free crack nucleation or a converged cutting case.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.closed_crack import (ClosedCrackProblem, Loading, _visualize_result,
                                   run, trapezoidal_work, wedge_edge_gap, wedge_gap)


class ClosedCrackKernelTests(unittest.TestCase):
    def setUp(self):
        self.problem = ClosedCrackProblem(nx=2, ny=3)

    def test_coincident_seam_has_independent_dofs_and_original_volume(self):
        p = self.problem
        left, right = p.seams
        self.assertFalse(set(left) & set(right))
        np.testing.assert_array_equal(p.reference[left], p.reference[right])
        self.assertTrue(np.all(np.ptp(p.sides[p.tri], axis=1) == 0))
        self.assertAlmostEqual(p.volumes.sum(), 2.0 * p.config.width_mm)
        gaps, gradient, groups, _ = p.contacts(p.reference, Loading())
        np.testing.assert_array_equal(gaps[groups == 'seam'], 0.0)
        self.assertTrue(np.all(np.asarray(abs(gradient[groups == 'seam']).sum(axis=1)) > 0.0))

    def test_bulk_is_objective_for_separate_body_rigid_motions(self):
        p = self.problem
        x = p.reference.copy()
        for side, angle in ((-1, 0.37), (1, -0.21)):
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            mask = p.sides == side
            x[mask] = x[mask] @ rotation.T + [side * 0.18, side * 0.27]
        energy, force, _, q = p.bulk(x, False)
        self.assertAlmostEqual(energy, 0.0, delta=2e-12)
        np.testing.assert_allclose(force, 0.0, atol=2e-12)
        np.testing.assert_allclose(q['J'], 1.0, atol=2e-15)

    def test_nonaffine_bulk_force_matches_energy_derivative(self):
        p = self.problem
        rng = np.random.default_rng(716)
        x = p.reference @ np.array([[1.07, 0.04], [-0.02, 0.96]]).T
        x += 0.002 * rng.normal(size=x.shape)
        _, force, _, _ = p.bulk(x, False)
        for _ in range(4):
            direction = rng.normal(size=x.shape)
            step = 1e-6
            difference = (p.bulk(x + step * direction, False)[0]
                          - p.bulk(x - step * direction, False)[0]) / (2.0 * step)
            self.assertAlmostEqual(force @ direction.ravel(), difference, delta=2e-6)

    def test_open_faces_release_old_compressive_multipliers_without_tension(self):
        p = self.problem
        x = p.reference.copy()
        x[:, 0] += p.sides * 0.03
        gaps, gradient, groups, _ = p.contacts(x, Loading())
        np.testing.assert_allclose(gaps[groups == 'seam'], 0.06, atol=1e-15)
        old_multiplier = np.where(groups == 'seam', 2.0, 0.0)
        active = np.maximum(0.0, old_multiplier - p.penalty * gaps)
        np.testing.assert_array_equal(active, 0.0)
        np.testing.assert_array_equal(gradient.T @ active, 0.0)
        _, force, _, _ = p.objective(x, Loading(), old_multiplier, [], False)
        np.testing.assert_allclose(force, 0.0, atol=2e-12)

    def test_sliding_completely_beyond_seam_has_no_infinite_line_constraint(self):
        p = self.problem
        x = p.reference.copy()
        x[p.sides == 1, 1] += 1.2
        gaps, gradient, groups, _ = p.contacts(x, Loading(tip_y=4.0))
        self.assertTrue(np.all(gaps[groups == 'seam'] > 0.0))
        np.testing.assert_array_equal(gradient[groups == 'seam'].toarray(), 0.0)

    def test_active_normal_contact_force_matches_augmented_energy_derivative(self):
        p = self.problem
        x = p.reference.copy()
        # Deliberately infeasible AL state, with projections strictly inside
        # target segments rather than at a nondifferentiable facet switch.
        x[:, 0] -= p.sides * 0.004
        x[:, 1] += p.sides * 0.035
        load = Loading(tip_y=3.0)
        gaps, _, groups, _ = p.contacts(x, load)
        multiplier = np.where(groups == 'seam', 0.7, 0.0)
        self.assertLess(gaps[groups == 'seam'].min(), 0.0)
        _, force, _, _ = p.objective(x, load, multiplier, [], False)
        direction = np.random.default_rng(200).normal(size=x.shape)
        h = 1e-7
        difference = (p.objective(x + h * direction, load, multiplier, [], False)[0]
                      - p.objective(x - h * direction, load, multiplier, [], False)[0]) / (2.0 * h)
        self.assertAlmostEqual(force @ direction.ravel(), difference, delta=2e-6)

    def test_friction_uses_previous_physical_state_and_conserves_pair_force(self):
        p = self.problem
        previous = p.previous.copy()
        nodes = np.array([0, 1])
        weights = np.array([0.3, 0.7])
        tangent = np.array([0.6, 0.8])
        B = (weights[:, None] * tangent).ravel()
        lag = [(2.1, nodes, B, 0.011)]
        x = previous.copy()
        x[nodes] += [0.025, -0.017]
        energy, force, _, dissipation, knife_force = p.friction_terms(x, lag, False)
        self.assertGreater(energy, 0.0)
        self.assertGreater(dissipation, 0.0)
        np.testing.assert_allclose(force.reshape(-1, 2).sum(axis=0) + knife_force, 0.0,
                                   atol=1e-15)
        tool_increment = np.array([0.0, lag[0][3] / tangent[1]])
        self.assertAlmostEqual(dissipation, force @ (x - previous).ravel()
                               + knife_force @ tool_increment, delta=1e-14)
        direction = np.random.default_rng(11).normal(size=x.shape)
        h = 1e-6
        difference = (p.friction_terms(x + h * direction, lag, False)[0]
                      - p.friction_terms(x - h * direction, lag, False)[0]) / (2.0 * h)
        self.assertAlmostEqual(force @ direction.ravel(), difference, delta=1e-8)
        p.friction_terms(previous, lag, False)
        np.testing.assert_array_equal(p.previous, previous)

    def test_pure_opening_equilibrium_is_detached_and_keeps_accepted_history(self):
        p = self.problem
        old = p.reference.copy()
        loading = Loading(compression=-0.02)
        with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
            x, result = p.solve(old, loading, time.monotonic() + 15.0)
        expected = old.copy()
        expected[:, 0] += p.sides * 0.02
        np.testing.assert_allclose(x, expected, atol=1e-6)
        self.assertLess(result['force_residual_n'], 1e-4)
        self.assertEqual(result['contacts']['seam']['active'], 0)
        self.assertLess(abs(result['elastic_energy_n_mm']), 1e-9)
        np.testing.assert_array_equal(p.previous, old)
        np.testing.assert_array_equal(old, p.reference)
        self.assertEqual(p.previous_load, loading)


class RoundedWedgeEdgeTests(unittest.TestCase):
    @staticmethod
    def edge_value(x):
        return wedge_edge_gap(*np.asarray(x).reshape(2, 2), 0.0, 0.02, 1)

    def check_envelope_gradient(self, coordinates):
        x = np.asarray(coordinates, dtype=float).ravel()
        _, normal, parameter = self.edge_value(x)
        analytic = (np.array([1.0 - parameter, parameter])[:, None] * normal).ravel()
        h = 1e-7
        finite_difference = np.array([
            (self.edge_value(x + h * direction)[0]
             - self.edge_value(x - h * direction)[0]) / (2.0 * h)
            for direction in np.eye(4)
        ])
        np.testing.assert_allclose(analytic, finite_difference, rtol=2e-7, atol=2e-9)

    def test_edge_detects_tip_penetration_with_both_nodes_outside(self):
        a, b = np.array([[-0.1, 0.01], [0.1, 0.01]])
        self.assertGreater(wedge_gap(a, 0.0)[0], 0.0)
        self.assertGreater(wedge_gap(b, 0.0)[0], 0.0)
        gap, normal, parameter = self.edge_value([a, b])
        self.assertAlmostEqual(gap, -0.01)
        self.assertAlmostEqual(parameter, 0.5)
        np.testing.assert_allclose(normal, [0.0, -1.0], atol=1e-15)

    def test_smooth_cap_minimizer_has_correct_envelope_gradient(self):
        self.check_envelope_gradient([[-0.04, -0.005], [0.07, 0.005]])

    def test_endpoint_minimizer_has_correct_envelope_gradient(self):
        self.check_envelope_gradient([[0.04, 0.05], [0.1, 0.1]])

    def test_infeasible_medial_axis_minimizer_has_correct_envelope_gradient(self):
        self.check_envelope_gradient([[-0.2, 0.2], [0.3, 0.4]])


class ClosedCrackReportTests(unittest.TestCase):
    @staticmethod
    def args(output, **changes):
        values=dict(output=output,nx=2,ny=3,penalty=20000.,seconds=10.,threads=1,max_steps=1)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_trapezoidal_external_work_is_exact_for_linear_elastic_loading_and_unloading(self):
        stiffness=np.array([2.,7.])
        for old,new in (([0.,0.],[.2,-.1]),([.2,-.1],[.07,.05])):
            old,new=np.asarray(old),np.asarray(new)
            work=trapezoidal_work(stiffness*old,stiffness*new,new-old)
            energy_change=.5*stiffness@(new**2-old**2)
            self.assertAlmostEqual(work,energy_change,places=15)

    def test_prefix_report_uses_accepted_endpoint_work_and_snapshots_baseline(self):
        with TemporaryDirectory() as temporary:
            out=Path(temporary)/'prefix'
            with patch('prototype.closed_crack._visualize_result'),contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(run(self.args(out)))
            report=json.loads((out/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'prefix_completed')
            self.assertTrue(report['completed'])
            self.assertFalse(report['full_schedule_completed'])
            self.assertEqual(report['accepted_step_count'],1)
            self.assertEqual(report['requested_step_count'],1)
            self.assertEqual(report['full_schedule_step_count'],50)
            self.assertEqual(report['completed_stages'],[])
            self.assertEqual(report['stage_counts']['compress'],dict(full=4,requested=1,accepted=1))
            step=report['steps'][0]
            self.assertAlmostEqual(step['external_trapezoidal_work_n_mm'],
                                   .5*step['external_endpoint_work_n_mm'],places=14)
            self.assertLess(abs(step['trapezoidal_energy_surplus_n_mm']),
                            abs(step['external_endpoint_work_n_mm']-step['elastic_energy_change_n_mm']))
            self.assertIn('knife_external_force_n',step)
            self.assertNotIn('knife_reaction_n',step)
            self.assertIn('opposite sign',report['force_convention'])
            baseline=(out/'source'/'prototype'/'ipc_baseline.json').read_bytes()
            self.assertEqual(hashlib.sha256(baseline).hexdigest(),report['baseline_config_sha256'])
            self.assertEqual(report['source_sha256']['prototype/ipc_baseline.json'],report['baseline_config_sha256'])
            self.assertEqual([report['geometry'][key] for key in ('length_mm','height_mm','width_mm')],[2.,1.,20.])
            self.assertAlmostEqual(report['geometry']['reference_volume_mm3'],40.)
            with np.load(out/'states'/'0000.npz') as state:
                np.testing.assert_array_equal(state['clamp_external_force_n'],step['clamp_external_force_n'])

    def test_failure_keeps_trial_and_accepted_multipliers_separate(self):
        def fail(problem,old,load,deadline):
            trial=problem.prescribe(old.copy(),load)
            problem._trial_state(trial,np.full_like(problem.multipliers,3.),load,4,2,'line_search',7)
            raise RuntimeError('deliberate bounded failure')
        with TemporaryDirectory() as temporary:
            out=Path(temporary)/'failed'
            with patch.object(ClosedCrackProblem,'solve',fail),contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(run(self.args(out)))
            report=json.loads((out/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['accepted_step_count'],0)
            self.assertFalse(report['full_schedule_completed'])
            self.assertEqual(report['failed_trial_iteration']['outer'],4)
            with np.load(out/'rejected.npz') as rejected:
                np.testing.assert_array_equal(rejected['trial_multipliers'],3.)
                np.testing.assert_array_equal(rejected['accepted_multipliers'],0.)
                self.assertEqual(json.loads(str(rejected['trial_loading_json']))['compression'],.005)
                self.assertEqual(json.loads(str(rejected['accepted_loading_json']))['compression'],0.)
                self.assertEqual(json.loads(str(rejected['trial_iteration_json']))['line_search'],7)
            post=json.loads((out/'visualization_status.json').read_text(encoding='utf-8'))
            self.assertEqual(post['visualization_status'],'skipped')

    def test_nonfinite_inputs_fail_before_creating_run_directory(self):
        self.assertEqual(ClosedCrackProblem(nx=1,ny=1).penalty,20000.)
        for value in (np.nan,np.inf,-np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):ClosedCrackProblem(penalty=value)
                with self.assertRaises(ValueError):ClosedCrackProblem(knife_mu=value)
                with self.assertRaises(ValueError):Loading(tip_y=value)
                with TemporaryDirectory() as temporary:
                    out=Path(temporary)/'invalid'
                    with self.assertRaises(ValueError):run(self.args(out,seconds=value))
                    self.assertFalse(out.exists())
        with self.assertRaises(ValueError):ClosedCrackProblem(nx=2.5)

    def test_render_failure_does_not_change_final_solver_report(self):
        with TemporaryDirectory() as temporary:
            out=Path(temporary);(out/'source').mkdir()
            content=b'{"status":"prefix_completed","completed":true}'
            (out/'result.json').write_bytes(content)
            with patch('tools.visualize_closed_crack.render',side_effect=RuntimeError('plot failure')), \
                    contextlib.redirect_stderr(io.StringIO()):
                _visualize_result(out,True)
            self.assertEqual((out/'result.json').read_bytes(),content)
            post=json.loads((out/'visualization_status.json').read_text(encoding='utf-8'))
            self.assertEqual(post['visualization_status'],'failed')
            self.assertIn('plot failure',post['error'])
            self.assertTrue((out/'source'/'tools'/'visualize_closed_crack.py').is_file())


if __name__ == '__main__':
    unittest.main()
