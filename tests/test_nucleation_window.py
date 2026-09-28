"""Source-window history and material-coordinate diagnostics, without local fixtures."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from prototype.directional_fracture import RationalAT1
from prototype.ipc_cutting import (DirectionalFractureProblem, damage_diagnostics,
                                   import_intact_window, run_case)
from prototype.ipc_reference import BASELINE_PATH, IPCProblem, ROOT, toolkit
from run_support import save_npz, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DamageMaterialMeasureTests(unittest.TestCase):
    def problem(self, width=20.):
        # Two unequal reference triangles: areas 1 and 2 mm^2.
        reference = np.array([[0., 0.], [2., 0.], [0., 1.], [0., -2.]])
        triangles = np.array([[0, 1, 2], [0, 3, 1]])
        phase = RationalAT1(reference, triangles, width, 2.15, .04,
                            critical_release_mpa=16.9923272986)
        return SimpleNamespace(rest=reference, mesh=SimpleNamespace(triangles=triangles),
                               phase=phase, config=SimpleNamespace(width_mm=width),
                               damage=np.array([.3, .3, .3, 0.]))

    def test_reference_area_integral_and_threshold_crossing_are_not_current_area(self):
        p = self.problem()
        old = np.array([.1, .1, .1, 0.])
        result = damage_diagnostics(p, old)
        # Integral of a linear field is reference area times mean nodal d.
        self.assertAlmostEqual(result['damage_weighted_reference_area_mm2'], .7)
        self.assertAlmostEqual(result['damage_increment_reference_area_mm2'], .7-7/30)
        by_threshold = {row['mean_nodal_damage_threshold']: row
                        for row in result['threshold_regions']}
        self.assertAlmostEqual(by_threshold[.1]['reference_area_mm2'], 3.)
        self.assertAlmostEqual(by_threshold[.1]['newly_crossed_reference_area_mm2'], 2.)
        self.assertEqual(by_threshold[.1]['reference_bbox_xz_mm'], [[0., -2.], [2., 1.]])
        self.assertEqual(by_threshold[.5]['reference_area_mm2'], 0.)
        self.assertIsNone(by_threshold[.5]['reference_bbox_xz_mm'])
        self.assertTrue(result['material_coordinates'])
        self.assertFalse(result['crack_opening_available'])
        # Out-of-plane integration width must not scale a 2-D area measure.
        q = self.problem(width=7.)
        scaled = damage_diagnostics(q, old)
        for key in ('damage_weighted_reference_area_mm2', 'damage_increment_reference_area_mm2'):
            self.assertAlmostEqual(result[key], scaled[key], places=14)
        self.assertEqual(result['threshold_regions'], scaled['threshold_regions'])

    def test_damage_healing_is_rejected_and_unloading_has_no_new_damage_area(self):
        p = self.problem()
        with self.assertRaisesRegex(ValueError, 'irreversible'):
            damage_diagnostics(p, p.damage+.01)
        result = damage_diagnostics(p, p.damage.copy())
        self.assertEqual(result['damage_increment_reference_area_mm2'], 0.)
        self.assertTrue(all(row['newly_crossed_reference_area_mm2'] == 0.
                            for row in result['threshold_regions']))


class NativeNucleationWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            toolkit()
        except ModuleNotFoundError:
            raise unittest.SkipTest('Local IPC toolkit unavailable')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'intact'
        self.source.mkdir()
        (self.source/'states').mkdir()
        with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
            p = IPCProblem(case='patch', mu=.35, dhat=.0025, stiffness=6.6,
                           local_spacing=.2, surface_spacing=.1, fine_halfwidth=3.,
                           nose_facets=64, tip_radius=.02, inclination=0.,
                           hessian_mode='regularized', max_iterations=100)
            p.ipc.set_num_threads(1)
            report = dict(status='completed', completed=True, case='patch', target_mm=.02,
                          config=p.config.to_dict(), volumetric_discretization='element',
                          simplifications=dict(fracture=False, tip_radius_mm=.02, nose_facets=64,
                                               friction_smoothing_mm=1e-5),
                          mesh_settings=dict(local_spacing_mm=.2, surface_spacing_mm=.1,
                                             fine_halfwidth_mm=3., bulk_spacing_mm=1.),
                          dhat_mm=.0025, barrier_stiffness_mpa=6.6,
                          baseline_config_sha256=digest(BASELINE_PATH),
                          ipc_binary_sha256=digest(p.ipc.__file__), hessian_mode='regularized',
                          source_sha256={name: digest(ROOT/name) for name in
                                         ('prototype/material.py', 'mechanics/finite_strain.py')},
                          loading_increment_mm=.01, force_tolerance_n=1e-4, steps=[])
            x, old = p.rest.copy(), p.start
            for i, target in enumerate((0., .01, .02)):
                previous = x.copy()
                x, balance = p.step(x, old, target, time.monotonic()+30.)
                report['steps'].append(balance)
                save_npz(self.source/'states'/f'{i:04d}.npz', reference_xy=p.rest,
                         triangles=p.mesh.triangles, positions=x, previous_positions=previous,
                         body_nodes=p.n, depth_mm=target)
                old = target
            report.update(last_depth_mm=.02, last_checkpoint='states/0002.npz')
            write_json(self.source/'result.json', report)
        self.args = SimpleNamespace(source=self.source, start_depth=.01, max_iterations=100,
            ell=.04, threads=1, friction_linear_solver='direct', orientation_samples=16,
            refine_orientations=False, orientation_lock_damage=1e-8)

    def imported(self, report=None):
        with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
            return import_intact_window(self.args, {} if report is None else report,
                                        time.monotonic()+30.)

    def rewrite_state(self, index, change):
        path = self.source/'states'/f'{index:04d}.npz'
        with np.load(path, allow_pickle=False) as state:
            values = {name: state[name].copy() for name in state.files}
        change(values)
        save_npz(path, **values)

    def rewrite_report(self, change):
        path = self.source/'result.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        change(value)
        write_json(path, value)

    def test_anchor_keeps_original_increment_and_next_step_advances_it_once(self):
        calls = []
        acceptance = DirectionalFractureProblem.acceptance
        def observe(p, x, *args, **kwargs):
            calls.append(p.previous.copy())
            return acceptance(p, x, *args, **kwargs)
        report = {}
        with patch.object(DirectionalFractureProblem, 'acceptance', observe):
            p, x, old, balance, history = self.imported(report)
        with np.load(self.source/'states/0001.npz') as state:
            np.testing.assert_array_equal(p.previous, state['previous_positions'])
            np.testing.assert_array_equal(calls[0], state['previous_positions'])
            np.testing.assert_array_equal(x, state['positions'])
            self.assertFalse(np.array_equal(p.previous, x))
        np.testing.assert_array_equal(history[1], p.previous)
        self.assertEqual(history[0], 0.)
        self.assertTrue(balance['source_anchor'])
        self.assertTrue(report['intact_source']['prefix_screen_completed'])
        self.assertEqual(len(report['intact_source']['prefix_screen']), 2)
        np.testing.assert_array_equal(p.damage, 0.)
        with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
            _, advanced = p.advance(x, old, .015, time.monotonic()+30.)
        np.testing.assert_array_equal(p.previous, x)
        self.assertLessEqual(advanced['residual_n'], 1e-4)

    def test_earlier_instability_cannot_be_skipped_by_a_later_stable_anchor(self):
        real_stability = RationalAT1.intact_stability
        calls = []
        def earlier_unstable(phase, release):
            value = real_stability(phase, release)
            calls.append(True)
            if len(calls) == 1:
                value['kkt_residual_mpa'] = .01
                value['ratio'] = np.full_like(value['ratio'], 1.01)
            return value
        report = {}
        with patch.object(RationalAT1, 'intact_stability', earlier_unstable):
            with self.assertRaisesRegex(ValueError, 'earlier local instability at 0 mm'):
                self.imported(report)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report['intact_source']['earlier_local_instability']['depth_mm'], 0.)
        self.assertFalse(report['intact_source']['prefix_screen_completed'])

    def test_changed_reference_geometry_is_rejected(self):
        self.rewrite_state(1, lambda values: values['reference_xy'].__setitem__((0, 0), .001))
        with self.assertRaises(AssertionError):
            self.imported()

    def test_broken_previous_position_chain_is_rejected(self):
        self.rewrite_state(1, lambda values: values['previous_positions'].__setitem__((0, 1), .001))
        with self.assertRaises(AssertionError):
            self.imported()

    def test_self_consistent_history_cannot_skip_an_intermediate_source_step(self):
        with np.load(self.source/'states/0000.npz') as first, \
                np.load(self.source/'states/0002.npz') as last:
            values = {name: last[name].copy() for name in last.files}
            values['previous_positions'] = first['positions'].copy()
        save_npz(self.source/'states/0001.npz', **values)
        (self.source/'states/0002.npz').unlink()
        self.rewrite_report(lambda value: value['steps'].pop(1))
        self.rewrite_report(lambda value: value.update(last_checkpoint='states/0001.npz'))
        self.args.start_depth = .02
        # A contiguous state-file chain alone is not a full .01 mm source prefix.
        with self.assertRaises((ValueError, AssertionError)):
            self.imported()

    def test_rigidly_shifted_states_cannot_move_the_fixed_boundary_or_blade_path(self):
        previous = None
        for index in range(3):
            path = self.source/'states'/f'{index:04d}.npz'
            with np.load(path) as state:
                values = {name: state[name].copy() for name in state.files}
            # Rigid translation preserves strains, relative gaps and reactions.
            # It nevertheless violates the prescribed bottom and blade motion.
            values['positions'][:, 0] += .1
            if previous is not None:
                values['previous_positions'] = previous
            previous = values['positions'].copy()
            save_npz(path, **values)
        with self.assertRaises((ValueError, AssertionError)):
            self.imported()

    def test_material_source_and_material_parameter_mismatch_are_rejected(self):
        path = self.source/'result.json'
        original = path.read_bytes()
        self.rewrite_report(lambda value: value['source_sha256'].__setitem__(
            'prototype/material.py', '0'*64))
        with self.assertRaisesRegex(ValueError, 'Material source mismatch'):
            self.imported()
        path.write_bytes(original)
        self.rewrite_report(lambda value: value['config'].__setitem__('shear_modulus_n_mm2', .3))
        with self.assertRaisesRegex(ValueError, 'Source configuration mismatch'):
            self.imported()

    def test_source_rejection_writes_failed_terminal_json_before_any_increment(self):
        args = SimpleNamespace(**vars(self.args), output=self.root/'rejected', depth=.02,
            increment=.005, min_increment=.00125, seconds=30., no_visualize=True,
            predictor='affine', case='patch', inclination=0., friction=.35, local_spacing=.2)
        with patch('prototype.ipc_cutting.import_intact_window',
                   side_effect=ValueError('Earlier source instability')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run_case(args))
        result = json.loads((args.output/'result.json').read_text(encoding='utf-8'))
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['completed'])
        self.assertFalse(result['completed_full_case'])
        self.assertIsNone(result['attempted_depth_mm'])
        self.assertEqual(result['steps'], [])
        self.assertIn('Earlier source instability', result['error'])
        self.assertFalse((args.output/'anchor.npz').exists())
        self.assertEqual(list((args.output/'states').iterdir()), [])
        self.assertIn('ValueError: Earlier source instability',
                      (args.output/'run.log').read_text(encoding='utf-8'))

    def test_window_completion_preserves_anchor_and_is_not_full_case_completion(self):
        args = SimpleNamespace(**vars(self.args), output=self.root/'window', depth=.02,
            increment=.005, min_increment=.00125, seconds=30., no_visualize=True,
            predictor='affine', case='patch', inclination=0., friction=.35, local_spacing=.2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(run_case(args))
        result = json.loads((args.output/'result.json').read_text(encoding='utf-8'))
        self.assertEqual(result['status'], 'completed_window')
        self.assertTrue(result['completed'])
        self.assertFalse(result['completed_full_case'])
        self.assertEqual(result['run_kind'], 'nucleation_window')
        self.assertEqual(result['physical_target_mm'], .3)
        self.assertEqual(result['work_accumulation'], 'window increments only')
        self.assertEqual(result['window_start_mm'], .01)
        self.assertEqual(len(result['steps']), 2)
        with np.load(args.output/'anchor.npz') as anchor, \
                np.load(args.output/'states/0000.npz') as first:
            np.testing.assert_array_equal(first['previous_positions'], anchor['positions'])
            np.testing.assert_array_equal(anchor['damage'], 0.)
            self.assertFalse(np.array_equal(anchor['positions'], anchor['previous_positions']))


if __name__ == '__main__':
    unittest.main()
