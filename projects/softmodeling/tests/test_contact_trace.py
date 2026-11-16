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

from prototype.contact_trace import (export_trace, query_depth, validate_loading_schedule,
                                     write_trace_csv, geometry_metadata, accumulate_motion)
from prototype.ipc_reference import record_tip_contact
from tests.test_tip_contact import fixture


def write_history(source, depths=(0., .05, .1), increment=.05, load_control=None, facets=64):
    """Synthetic positive-J history is only an IO fixture, not equilibrium data."""
    _, reference, triangles, count, config = fixture(facets=facets)
    source.mkdir()
    (source / 'states').mkdir()
    report = dict(status='completed', completed=True, case='oblique', target_mm=depths[-1],
                  config=vars(config), dhat_mm=.01, force_tolerance_n=1e-4,
                  loading_increment_mm=increment, simplifications=dict(fracture=False, nose_facets=facets),
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

class QualifiedPrefixTests(unittest.TestCase):
    def make_prefix(self, source):
        report = write_history(source, facets=32)
        report.update(status='budget_exhausted', completed=False)
        rows = [dict(step, passed=i < 2, checkpoint_sha256=hashlib.sha256(
            (source/'states'/f'{i:04d}.npz').read_bytes()).hexdigest()) for i, step in enumerate(report['steps'])]
        (source/'result.json').write_text(json.dumps(report))
        (source/'audit.json').write_text(json.dumps(dict(passed=False, rows=rows, qualified_prefix_states=2)))
        manifest = source.parent/'manifest.json'
        manifest.write_text(json.dumps(dict(files=[dict(path='source/'+name,
            sha256=hashlib.sha256((source/name).read_bytes()).hexdigest()) for name in ('result.json', 'audit.json')])))
        return report, manifest

    def test_32_facet_authenticated_prefix_retains_failure_and_excludes_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source'
            report, manifest = self.make_prefix(source)
            with patch('tools.visualize_ipc.plot_tip_contact'):
                result = export_trace(source, Path(folder)/'export', qualified_prefix=True, manifest_path=manifest)
            self.assertEqual(len(result['steps']), 2)
            self.assertEqual(result['tip_contact_tracking']['nose_facets'], 32)
            self.assertEqual(result['source_status'], 'budget_exhausted')
            self.assertFalse(result['source_completed'])
            self.assertEqual(json.loads((source/'result.json').read_text()), report)

    def test_unbound_or_tampered_legacy_prefix_fails_closed(self):
        for defect in ('unbound', 'source', 'checkpoint'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                source = Path(folder)/'source'
                report, manifest = self.make_prefix(source)
                if defect == 'source':
                    report['source_sha256']['synthetic_fixture'] = 'changed'
                    (source/'result.json').write_text(json.dumps(report))
                elif defect == 'checkpoint':
                    with (source/'states'/'0001.npz').open('ab') as stream:
                        stream.write(b'changed')
                with self.assertRaises(ValueError):
                    export_trace(source, Path(folder)/'export', qualified_prefix=True,
                                 manifest_path=None if defect == 'unbound' else manifest)

    def test_conflicting_metadata_and_missing_facet_evidence_rejected(self):
        report = dict(config={}, signature=dict(config={'inclination_deg': 0}, dhat=.01),
                      settings=dict(nose_facets=32), resolved_model=dict(geometry=dict(nose_facets=64)))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            geometry_metadata(report)
        report.pop('settings');report['resolved_model'] = {}
        with self.assertRaisesRegex(ValueError, 'nose_facets'):
            geometry_metadata(report)

    def test_adjacent_edge_crossing_and_segment_gaps(self):
        motion = dict(valid=True, status='valid_proximity', transported_mm_s=2.)
        cumulative = {}
        c = lambda edge, s: dict(status='unique', edge_nodes=edge, reference_s_mm=s)
        a, b, far = c([1, 3], 1.), c([3, 5], 1.1), c([9, 11], 3.)
        first = accumulate_motion(None, motion, a, None, None, cumulative, .1)
        second = accumulate_motion(None, motion, b, a, first, cumulative, .1)
        self.assertIsNone(second['break_reason'])
        self.assertAlmostEqual(second['absolute_segment_mm'], .2)
        jump = accumulate_motion(None, motion, far, b, second, cumulative, .1)
        self.assertEqual(jump['break_reason'], 'nonadjacent_projection_jump')
        self.assertEqual(jump['absolute_segment_mm'], 0.)
        for status in ('no_tip_contact', 'ambiguous', 'nonunique_tangent'):
            invalid = accumulate_motion(None, dict(valid=False, status=status), {}, b, second, cumulative, .1)
            self.assertIsNone(invalid['absolute_segment_mm'])
            born = accumulate_motion(None, motion, b, {}, invalid, cumulative, .1)
            self.assertEqual(born['break_reason'], 'first_valid_endpoint_or_contact_birth')

class AuditBindingTests(unittest.TestCase):
    def test_modern_record_source_and_settings_binding_reject_tampering(self):
        from prototype.contact_trace import record_digest, audit_coverage
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source'
            report=write_history(source)
            rows=[dict(s,passed=True,checkpoint_sha256='fixture',accepted_record_sha256=record_digest(s)) for s in report['steps']]
            binding=dict(signature_sha256=record_digest(report.get('signature')),
                producer_sources_sha256=record_digest(report.get('source_sha256')),
                settings_sha256=record_digest(report.get('settings')))
            (source/'audit.json').write_text(json.dumps(dict(passed=True,rows=rows,source_binding=binding, source_result_sha256=hashlib.sha256((source/'result.json').read_bytes()).hexdigest())))
            self.assertEqual(audit_coverage(source,report)[0],3)
            for field in ('settings','source_sha256','steps'):
                changed=json.loads(json.dumps(report))
                if field=='steps':changed['steps'][0]['tip_contact']={'status':'tampered'}
                else:changed[field]={'tampered':True}
                with self.assertRaisesRegex(ValueError,'binding|record'):
                    audit_coverage(source,changed)

    def test_online_metadata_and_footprint_share_the_offline_observer(self):
        from tests.test_tip_contact import place_pole
        tracker,reference,_,n,_=fixture(facets=32)
        x=place_pole(tracker,reference,n,[1.3,1.005])
        report={};step=dict(depth_mm=0.)
        with contextlib.redirect_stdout(io.StringIO()):
            record_tip_contact(tracker,x,step,report)
        self.assertEqual(step['tip_contact'],tracker.observe(x,0.))
        self.assertEqual(step['blade_contact_footprint'],tracker.observe_footprint(x,0.))
        self.assertEqual({k:v for k,v in report['tip_contact_tracking'].items() if k!='errors'},tracker.metadata)


class FinalResultBindingTests(unittest.TestCase):
    def test_final_binding_catches_status_and_request_edits(self):
        from prototype.contact_trace import record_digest, audit_coverage
        from prototype.material_case import bind_final_audit_result
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source';report=write_history(source)
            rows=[dict(s,passed=True,checkpoint_sha256='fixture',accepted_record_sha256=record_digest(s)) for s in report['steps']]
            binding=dict(signature_sha256=record_digest(report.get('signature')),producer_sources_sha256=record_digest(report.get('source_sha256')),settings_sha256=record_digest(report.get('settings')))
            (source/'audit.json').write_text(json.dumps(dict(passed=True,rows=rows,source_binding=binding)))
            with self.assertRaisesRegex(ValueError,'final result bytes'):
                audit_coverage(source,report)
            bind_final_audit_result(source,report)
            self.assertEqual(audit_coverage(source,report)[0],3)
            report['completed']=False;report['target_mm']=1.
            (source/'result.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'final result bytes'):
                audit_coverage(source,report)
