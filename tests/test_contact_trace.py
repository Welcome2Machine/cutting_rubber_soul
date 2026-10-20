"""Trace IO and lookup checks with synthetic accepted states, without IPC solves."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from prototype.contact_trace import export_trace, query_depth, validate_loading_schedule, write_trace_csv
from prototype.ipc_reference import record_tip_contact
from tests.test_tip_contact import fixture


def write_history(source, depths=(0., .05, .1), increment=.05, load_control=None):
    """Synthetic positive-J history is only an IO fixture, not equilibrium data."""
    _, reference, triangles, count, config = fixture()
    source.mkdir()
    (source / 'states').mkdir()
    report = dict(status='completed', completed=True, case='oblique', target_mm=depths[-1],
                  config=vars(config), dhat_mm=.01, force_tolerance_n=1e-4,
                  loading_increment_mm=increment, simplifications=dict(fracture=False, nose_facets=64),
                  source_sha256={'synthetic_fixture': 'not a solver run'}, steps=[])
    if load_control is not None:
        report['load_control'] = load_control
    previous = reference.copy()
    zero = reference.copy()
    start = -2 * report['dhat_mm'] / max(np.cos(np.deg2rad(config.inclination_deg)), .1)
    zero[count:, 1] += start
    for index, depth in enumerate(depths):
        positions = zero.copy()
        positions[:count, 1] *= .995-depth
        positions[count:, 1] -= depth
        np.savez_compressed(source / 'states' / f'{index:04d}.npz', reference_xy=reference,
                            triangles=triangles, body_nodes=count, positions=positions,
                            previous_positions=previous, depth_mm=depth)
        report['steps'].append(dict(depth_mm=depth, residual_n=0., min_j=.995-depth,
                                   reaction_fz_n=depth))
        previous = positions.copy()
    (source / 'result.json').write_text(json.dumps(report), encoding='utf-8')
    return report


def adaptive_control():
    return dict(mode='adaptive_bisection', minimum_increment_mm=.0003125,
                maximum_increment_mm=.01, output_grid_increment_mm=.01,
                retry_predictor='affine', stagnation_window=40, stagnation_improvement_fraction=.1)


class LoadingScheduleTests(unittest.TestCase):
    def report(self, depths, *, completed=False, adaptive=True, target=3.):
        report = dict(completed=completed, target_mm=target, loading_increment_mm=.01,
                      steps=[dict(depth_mm=float(depth)) for depth in depths])
        if adaptive:
            report['load_control'] = adaptive_control()
        return report

    def test_adaptive_partial_can_end_between_grid_points_without_interpolation(self):
        depths = np.r_[np.arange(0., .531, .01), .535]
        report = self.report(depths)
        np.testing.assert_array_equal(validate_loading_schedule(report), depths)
        with self.assertRaisesRegex(ValueError, 'zero to target'):
            validate_loading_schedule(report, require_complete=True)
        report['completed'] = True
        with self.assertRaisesRegex(ValueError, 'zero to target'):
            validate_loading_schedule(report)

    def test_complete_adaptive_three_mm_and_short_anchor_remainder(self):
        depths = np.r_[0., .0099, np.arange(.01, 3., .01), 3.]
        np.testing.assert_array_equal(validate_loading_schedule(self.report(depths, completed=True)), depths)
        # 0.0001 mm before the first output anchor is smaller than the retry floor.
        self.assertLess(depths[2] - depths[1], adaptive_control()['minimum_increment_mm'])

    def test_invalid_adaptive_schedules_rejected(self):
        histories = ([.001], [0., .005, .015], [0., .011], [0., .005, .005],
                     [0., .005, .004], [0., float('nan')], [], [0., 3.001])
        for depths in histories:
            with self.subTest(depths=depths), self.assertRaises(ValueError):
                validate_loading_schedule(self.report(depths))

    def test_invalid_adaptive_metadata_rejected(self):
        for key, value in (('mode', 'unknown'), ('minimum_increment_mm', 0.),
                           ('minimum_increment_mm', .02), ('maximum_increment_mm', .02),
                           ('output_grid_increment_mm', .005), ('minimum_increment_mm', float('nan'))):
            report = self.report([0., .005])
            report['load_control'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_loading_schedule(report)

    def test_fixed_history_requires_original_grid_and_target(self):
        validate_loading_schedule(self.report([0., .01], adaptive=False))
        validate_loading_schedule(self.report([0., .01, .02], adaptive=False, completed=True, target=.02))
        for depths in ([0., .005], [0., .02]):
            with self.subTest(depths=depths), self.assertRaises(ValueError):
                validate_loading_schedule(self.report(depths, adaptive=False))


class ContactTraceLookupTests(unittest.TestCase):
    def setUp(self):
        self.steps = [
            dict(depth_mm=0., tip_contact=dict(status='unique', reference_offset_mm=0.)),
            dict(depth_mm=.1, tip_contact=dict(status='no_tip_contact', reference_offset_mm=None)),
            dict(depth_mm=.2, tip_contact=dict(status='ambiguous', reference_offset_mm=None,
                                              candidates=[{'reference_offset_mm': -.02}, {'reference_offset_mm': .03}])),
            dict(depth_mm=.3),
        ]

    def test_csv_preserves_zero_undefined_and_multiple_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'contact_trace.csv'
            write_trace_csv(path, self.steps)
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]['reference_offset_mm']), 0.)
            self.assertEqual([row['reference_offset_mm'] for row in rows[1:]], ['', '', ''])
            self.assertEqual([row['status'] for row in rows], ['unique', 'no_tip_contact', 'ambiguous', ''])
            self.assertFalse(path.with_suffix('.csv.tmp').exists())
            write_trace_csv(path, self.steps[:1])
            with path.open(newline='', encoding='utf-8') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_exact_query_keeps_true_zero(self):
        result = query_depth(self.steps, 0.)
        self.assertEqual(result['kind'], 'exact_accepted_depth')
        self.assertFalse(result['interpolated'])
        self.assertEqual(result['samples'][0]['tip_contact']['reference_offset_mm'], 0.)

    def test_brackets_do_not_invent_a_contact_for_missing_or_ambiguous_values(self):
        for depth, expected in ((.05, [0., .1]), (.15, [.1, .2]), (.25, [.2, .3])):
            with self.subTest(depth=depth):
                result = query_depth(self.steps, depth)
                self.assertEqual(result['kind'], 'bracketing_accepted_depths')
                self.assertFalse(result['interpolated'])
                self.assertEqual([sample['depth_mm'] for sample in result['samples']], expected)
                self.assertNotIn('reference_offset_mm', result)
        self.assertEqual(query_depth(self.steps, .2)['samples'][0]['tip_contact']['status'], 'ambiguous')
        self.assertIsNone(query_depth(self.steps, .3)['samples'][0]['tip_contact'])

    def test_no_extrapolation_and_invalid_history_rejected(self):
        for depth in (-.001, .301, float('inf'), float('nan')):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                query_depth(self.steps, depth)
        for steps in ([], self.steps[::-1], [self.steps[0], self.steps[0]]):
            with self.assertRaises(ValueError):
                query_depth(steps, 0.)


class ContactTraceIntegrationTests(unittest.TestCase):
    def test_adaptive_export_and_stress_frames_keep_every_accepted_substep(self):
        from tools.visualize_ipc import ROOT, load_frames
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source', Path(folder) / 'export'
            depths = (0., .005, .01, .015, .02)
            report = write_history(source, depths, .01, adaptive_control())
            for name in ('prototype/material.py', 'mechanics/finite_strain.py'):
                report['source_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            (source / 'result.json').write_text(json.dumps(report), encoding='utf-8')
            with patch('tools.visualize_ipc.plot_tip_contact'):
                result = export_trace(source, output)
            np.testing.assert_array_equal([step['depth_mm'] for step in result['steps']], depths)
            frames = load_frames(source)[5]
            np.testing.assert_array_equal([frame['depth'] for frame in frames], depths)
            report['steps'].pop(2)  # Reaching 0.015 does not substitute for the omitted 0.01 anchor.
            (source / 'result.json').write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'output-grid'):
                load_frames(source)
            with self.assertRaisesRegex(ValueError, 'output-grid'):
                export_trace(source, Path(folder) / 'invalid_export')

    def test_observation_failure_preserves_accepted_step_and_positions(self):
        positions = np.array([[0., 1.], [2., 3.]])
        original = positions.copy()
        step = dict(depth_mm=.1, reaction_fz_n=12., residual_n=1e-6, min_j=.9)
        report = dict(status='completed', completed=True, steps=[step], tip_contact_tracking={'errors': []})
        tracker = Mock()
        tracker.observe.side_effect = ValueError('synthetic observation failure')
        with contextlib.redirect_stdout(io.StringIO()):
            record_tip_contact(tracker, positions, step, report)
        np.testing.assert_array_equal(positions, original)
        self.assertIs(report['steps'][0], step)
        self.assertEqual(len(report['steps']), 1)
        self.assertTrue(report['completed'])
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(step['reaction_fz_n'], 12.)
        self.assertEqual(step['residual_n'], 1e-6)
        self.assertEqual(step['tip_contact']['status'], 'error')
        self.assertEqual(len(report['tip_contact_tracking']['errors']), 1)

    def test_synthetic_full_export_preserves_source_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source', Path(folder) / 'export'
            write_history(source)
            before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob('*') if path.is_file()}
            with patch('tools.visualize_ipc.plot_tip_contact') as plot:
                result = export_trace(source, output)
            plot.assert_called_once()
            self.assertEqual(result['status'], 'completed_postprocessing')
            self.assertFalse(result['mechanical_solve_performed'])
            self.assertEqual(len(result['steps']), 3)
            self.assertEqual(len(result['source_checkpoints_sha256']), 3)
            self.assertEqual(result['source_result_sha256'], hashlib.sha256(before['result.json']).hexdigest())
            self.assertTrue((output / 'contact_trace.json').is_file())
            self.assertTrue((output / 'contact_trace.csv').is_file())
            for name, data in before.items():
                self.assertEqual((source / name).read_bytes(), data)
            self.assertTrue(all(step['tip_contact']['status'] == 'unique' for step in result['steps']))
            with self.assertRaisesRegex(ValueError, 'empty'):
                export_trace(source, output)

    def test_export_rejects_incomplete_report_and_source_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            report = write_history(source)
            with self.assertRaisesRegex(ValueError, 'separate'):
                export_trace(source, source / 'export')
            report['completed'] = False
            (source / 'result.json').write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'completed intact'):
                export_trace(source, Path(folder) / 'export')

    def test_missing_step_or_checkpoint_is_not_completed_history(self):
        for defect in ('skipped_depth', 'missing_checkpoint'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / 'source'
                report = write_history(source)
                if defect == 'skipped_depth':
                    report['steps'].pop(1)
                    (source / 'result.json').write_text(json.dumps(report), encoding='utf-8')
                else:
                    (source / 'states' / '0001.npz').unlink()
                with self.assertRaises((ValueError, AssertionError, FileNotFoundError)):
                    export_trace(source, Path(folder) / 'export')
                self.assertFalse((Path(folder) / 'export').exists())

    def test_plot_failure_preserves_trace_and_has_independent_failure_status(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source', Path(folder) / 'export'
            write_history(source)
            with patch('tools.visualize_ipc.plot_tip_contact', side_effect=RuntimeError('synthetic plot failure')):
                result = export_trace(source, output)
            self.assertEqual(result['status'], 'completed_postprocessing')
            self.assertFalse(result['mechanical_solve_performed'])
            self.assertEqual(result['visualization']['status'], 'failed')
            self.assertIn('synthetic plot failure', result['visualization']['error'])
            self.assertTrue((output / 'contact_trace.csv').is_file())
            saved = json.loads((output / 'contact_trace.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['visualization']['status'], 'failed')
            self.assertEqual(len(saved['steps']), 3)
            sample = query_depth(saved['steps'], .05)['samples'][0]
            self.assertEqual(sample['tip_contact']['status'], 'unique')
            self.assertFalse((output / 'tip_contact_offset.png').exists())

    def test_export_rejects_broken_saved_history(self):
        for defect in ('slip_history', 'topology', 'knife_motion', 'minimum_j', 'state_depth'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / 'source'
                write_history(source)
                path = source / 'states' / '0001.npz'
                with np.load(path, allow_pickle=False) as saved:
                    state = {key: saved[key].copy() for key in saved.files}
                if defect == 'slip_history':
                    state['previous_positions'][0, 0] += .01
                elif defect == 'topology':
                    state['reference_xy'][0, 0] += .01
                elif defect == 'knife_motion':
                    state['positions'][int(state['body_nodes']), 0] += .01
                elif defect == 'minimum_j':
                    state['positions'][0, 1] += .01
                else:
                    state['depth_mm'] += .001
                np.savez_compressed(path, **state)
                with patch('tools.visualize_ipc.plot_tip_contact'), self.assertRaises((ValueError, AssertionError)):
                    export_trace(source, Path(folder) / 'export')
                self.assertFalse((Path(folder) / 'export').exists())


if __name__ == '__main__':
    unittest.main()
