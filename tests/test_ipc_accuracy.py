"""Synthetic terminal/prefix acceptance checks for the IPC accuracy report."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_ipc_accuracy import compare, write_comparison


def report():
    config = dict(length_mm=30., height_mm=5., width_mm=20., inclination_deg=60., drop_s_mm=7.,
                  blade_thickness_mm=.245, blade_tip_angle_deg=23., shear_modulus_n_mm2=.66,
                  poisson_ratio=.49, friction_coefficient=.35)
    return dict(case='oblique', config=config, completed=True, status='completed', target_mm=3.,
        last_depth_mm=3., force_tolerance_n=1e-4, dhat_mm=.0025, barrier_stiffness_mpa=6.6,
        ipc_version='1.6.0', ipc_binary_sha256='a'*64, source_sha256={'material.py': 'b'*64},
        elapsed_s=100., mesh_settings={'local_spacing_mm': .05}, loading_increment_mm=1.,
        simplifications=dict(fracture=False, tip_radius_mm=.02, nose_facets=64,
                             material='finite strain elastic', friction_smoothing_mm=1e-5),
        steps=[dict(depth_mm=d, reaction_fz_n=f, residual_n=1e-5, min_j=.8,
                    minimum_distance_mm=.001, friction_work_n_mm=0.,
                    tip_contact=dict(status='unique', depth_mm=d, gap_mm=.001,
                                     reference_offset_mm=.1*d)) for d, f in ((0., 0.), (1., 10.), (2., 20.), (3., 30.))])


def adaptive_report():
    result = report()
    result['load_control'] = dict(mode='adaptive_bisection', minimum_increment_mm=.125,
        maximum_increment_mm=1., output_grid_increment_mm=1., retry_predictor='affine',
        stagnation_window=40, stagnation_improvement_fraction=.1)
    extra = copy.deepcopy(result['steps'][1])
    extra.update(depth_mm=.5, reaction_fz_n=5.)
    extra['tip_contact'].update(depth_mm=.5, reference_offset_mm=.05)
    result['steps'].insert(1, extra)
    return result


class AccuracyTests(unittest.TestCase):
    def test_complete_agreement_allows_numerical_changes(self):
        a = report(); b = copy.deepcopy(a)
        b.update(mesh_settings={'local_spacing_mm': .1}, linear_solver='new implementation', elapsed_s=50.)
        b['source_sha256']['material.py'] = 'c'*64
        result = compare(a, b)
        self.assertTrue(result['accuracy_qualified'])
        self.assertEqual(result['matching']['matched_states'], 4)
        self.assertEqual(result['elapsed']['reference_over_candidate_elapsed_ratio'], 2.)
        self.assertEqual(result['source_provenance']['changed_sources'], ['material.py'])

    def test_sparse_candidate_exact_depths_and_failure_prefix(self):
        a = report(); b = copy.deepcopy(a)
        b['steps'] = [b['steps'][0], b['steps'][3]]; b['loading_increment_mm'] = 3.
        self.assertTrue(compare(a, b)['accuracy_qualified'])
        b.update(completed=False, status='failed', last_depth_mm=0., error='TimeoutError: budget')
        b['steps'] = b['steps'][:1]
        result = compare(a, b)
        self.assertFalse(result['accuracy_qualified'])
        self.assertEqual(len(result['points']), 1)
        self.assertIsNone(result['elapsed']['reference_over_candidate_elapsed_ratio'])
        self.assertEqual(result['candidate_error'], 'TimeoutError: budget')

    def test_unmatched_depth_is_not_interpolated(self):
        a = report(); b = copy.deepcopy(a)
        b['steps'][1]['depth_mm'] = 1.01
        b['steps'][1]['tip_contact']['depth_mm'] = 1.01
        result = compare(a, b)
        self.assertFalse(result['accuracy_qualified'])
        self.assertEqual(result['matching']['unmatched_candidate_depths_mm'], [1.01])
        self.assertEqual(len(result['points']), 3)

    def test_adaptive_extra_substeps_are_audited_but_not_accuracy_assessed(self):
        a = report(); b = adaptive_report()
        b['steps'][1]['reaction_fz_n'] = 999.  # No reference at .5; no fabricated comparison.
        result = compare(a, b)
        self.assertTrue(result['accuracy_qualified'])
        self.assertTrue(result['candidate_completed_3mm'])
        self.assertTrue(result['matching']['qualification_grid_coverage_to_3mm'])
        self.assertFalse(result['matching']['full_candidate_coverage_to_3mm'])
        self.assertFalse(result['matching']['all_candidate_depths_matched'])
        self.assertEqual(result['matching']['unassessed_adaptive_substep_depths_mm'], [.5])
        self.assertEqual(result['matching']['missing_reference_output_depths_mm'], [])
        self.assertEqual(result['matching']['required_reference_output_depths_mm'], [0., 1., 2., 3.])
        self.assertEqual([row['depth_mm'] for row in result['points']], [0., 1., 2., 3.])
        self.assertEqual(result['force']['maximum_absolute_difference_n'], 0.)
        self.assertIn('not accuracy-assessed', result['accuracy_scope'])
        self.assertEqual(result['candidate_audit']['accepted_states'], 5)
        b['steps'][1]['residual_n'] = .001
        result = compare(a, b)
        self.assertFalse(result['accuracy_qualified'])
        self.assertEqual(result['candidate_audit']['accepted_states'], 1)

    def test_adaptive_missing_output_or_bad_schedule_is_not_qualified(self):
        a = report(); b = adaptive_report()
        b['steps'][2]['depth_mm'] = 1.25
        b['steps'][2]['tip_contact']['depth_mm'] = 1.25
        result = compare(a, b)
        self.assertFalse(result['accuracy_qualified'])
        self.assertFalse(result['candidate_completed_3mm'])
        self.assertEqual(result['matching']['missing_reference_output_depths_mm'], [1.])
        self.assertIn('skipped', result['candidate_audit']['error'])
        b = adaptive_report(); b['steps'] = [b['steps'][0], b['steps'][-1]]
        b['loading_increment_mm'] = 3.
        b['load_control'].update(maximum_increment_mm=3., output_grid_increment_mm=3.)
        result = compare(a, b)
        self.assertTrue(result['candidate_completed_3mm'])
        self.assertFalse(result['accuracy_qualified'])
        self.assertEqual(result['matching']['missing_reference_output_depths_mm'], [1., 2.])
        for key, value in (('minimum_increment_mm', float('nan')), ('minimum_increment_mm', 2.),
                           ('maximum_increment_mm', 2.), ('output_grid_increment_mm', 2.),
                           ('mode', 'unknown')):
            with self.subTest(key=key):
                b = adaptive_report(); b['load_control'][key] = value
                result = compare(a, b)
                self.assertFalse(result['candidate_completed_3mm'])
                self.assertFalse(result['accuracy_qualified'])
                self.assertIsNotNone(result['candidate_audit']['error'])

    def test_adaptive_incomplete_substep_prefix_remains_unqualified(self):
        a = report(); b = adaptive_report()
        b.update(completed=False, status='failed', last_depth_mm=.5, error='iteration budget')
        b['steps'] = b['steps'][:2]
        result = compare(a, b)
        self.assertFalse(result['accuracy_qualified'])
        self.assertIsNone(result['candidate_audit']['error'])
        self.assertEqual(result['matching']['unassessed_adaptive_substep_depths_mm'], [.5])
        self.assertEqual(result['candidate_audit']['loading_grid']['missing_output_depths_mm'], [1., 2., 3.])
        self.assertIsNone(result['elapsed']['reference_over_candidate_elapsed_ratio'])

    def test_force_absolute_relative_and_tip_thresholds(self):
        a = report(); b = copy.deepcopy(a)
        b['steps'][0]['reaction_fz_n'] = .2
        b['steps'][1]['reaction_fz_n'] = 10.5
        b['steps'][0]['tip_contact']['reference_offset_mm'] = .02
        self.assertTrue(compare(a, b)['accuracy_qualified'])
        b['steps'][0]['reaction_fz_n'] += 1e-6
        self.assertEqual(compare(a, b)['force']['failures'], 1)
        b['steps'][0]['tip_contact']['reference_offset_mm'] += 1e-6
        self.assertEqual(compare(a, b)['tip']['unique_failures'], 1)

    def test_missing_ambiguous_and_invalid_observations_are_not_zero(self):
        a = report(); b = copy.deepcopy(a)
        a['steps'][0].pop('tip_contact'); b['steps'][0].pop('tip_contact')
        a['steps'][1]['tip_contact']['status'] = b['steps'][1]['tip_contact']['status'] = 'ambiguous'
        result = compare(a, b)
        self.assertTrue(result['accuracy_qualified'])
        self.assertIsNone(result['points'][0]['tip_difference_mm'])
        self.assertIsNone(result['points'][1]['candidate_tip_offset_mm'])
        b['steps'][1]['tip_contact']['status'] = 'no_tip_contact'
        self.assertEqual(compare(a, b)['tip']['status_mismatch_count'], 1)
        b['steps'][2]['tip_contact']['reference_offset_mm'] = float('nan')
        result = compare(a, b)
        self.assertFalse(result['accuracy_qualified'])
        self.assertEqual(result['tip']['invalid_or_error_pairs'], 1)
        for r in (a, b):
            for step in r['steps']: step.pop('tip_contact', None)
        self.assertFalse(compare(a, b)['accuracy_qualified'])
        a = report(); b = copy.deepcopy(a)
        a['tip_contact_tracking'] = {'method': 'fixed_tip'}
        b['tip_contact_tracking'] = None
        for step in b['steps']: step.pop('tip_contact')
        result = compare(a, b)
        self.assertFalse(result['accuracy_qualified'])
        self.assertEqual(result['tip']['status_mismatch_count'], 4)

    def test_physics_reference_and_invalid_reported_suffix_fail_closed(self):
        a = report(); b = copy.deepcopy(a); b['config']['friction_coefficient'] = .3
        with self.assertRaisesRegex(ValueError, 'Physical settings differ'): compare(a, b)
        b = report(); a['completed'] = False
        with self.assertRaisesRegex(ValueError, 'Reference must'): compare(a, b)
        for key, value in (('residual_n', .001), ('min_j', 0.), ('minimum_distance_mm', 0.),
                           ('friction_work_n_mm', -1e-5), ('reaction_fz_n', float('nan'))):
            with self.subTest(key=key):
                a = report(); b = copy.deepcopy(a); b['steps'][2][key] = value
                result = compare(a, b)
                self.assertFalse(result['accuracy_qualified'])
                self.assertEqual(len(result['points']), 2)
                self.assertIsNotNone(result['candidate_audit']['error'])
        a = report(); b = copy.deepcopy(a); a['force_tolerance_n'] = b['force_tolerance_n'] = .01
        with self.assertRaisesRegex(ValueError, 'no looser'): compare(a, b)

    def test_legacy_law_defaults_match_but_changed_alpha_or_contact_are_rejected(self):
        a = report(); b = copy.deepcopy(a)
        b['config']['ogden_alpha'] = 2.
        b['contact_formulation'] = 'improved-max'
        self.assertTrue(compare(a, b)['accuracy_qualified'])
        self.assertNotIn('ogden_alpha', a['config'])
        for alpha, contact in ((1., 'improved-max'), (2., 'offset-geometric')):
            with self.subTest(alpha=alpha, contact=contact):
                b['config']['ogden_alpha'] = alpha
                b['contact_formulation'] = contact
                with self.assertRaisesRegex(ValueError, 'Physical settings differ'):
                    compare(a, b)

    def test_resumed_elapsed_is_not_used_as_whole_run_speedup(self):
        a = report(); b = copy.deepcopy(a); b['resume'] = {'source': 'prior run'}
        result = compare(a, b)
        self.assertTrue(result['accuracy_qualified'])
        self.assertIsNone(result['elapsed']['reference_over_candidate_elapsed_ratio'])

    def test_outputs_refresh_existing_comparison_and_keep_empty_csv_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root/'reference'; b = root/'candidate'; out = root/'comparison'
            a.mkdir(); b.mkdir(); out.mkdir(); ra = report(); rb = copy.deepcopy(ra)
            for r in (ra, rb): r['steps'][0].pop('tip_contact')
            (a/'result.json').write_text(json.dumps(ra), encoding='utf-8')
            (b/'result.json').write_text(json.dumps(rb), encoding='utf-8')
            result = write_comparison(a, b, out)
            write_comparison(a, b, out)
            saved = json.loads((out/'accuracy.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['provenance'], result['provenance'])
            self.assertEqual(len(saved['provenance']['reference_sha256']), 64)
            with (out/'accuracy.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['reference_tip_offset_mm'], '')
            self.assertEqual(rows[0]['tip_difference_mm'], '')
            self.assertNotIn('NaN', (out/'accuracy.json').read_text(encoding='utf-8'))

    def test_pending_is_explicit_and_does_not_hide_invalid_existing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root/'reference'; b = root/'candidate'; out = root/'comparison'
            a.mkdir(); ra = report()
            (a/'result.json').write_text(json.dumps(ra), encoding='utf-8')
            with self.assertRaises(FileNotFoundError):
                write_comparison(a, b, out)
            self.assertFalse(out.exists())
            pending = write_comparison(a, b, out, allow_pending=True)
            self.assertEqual(pending['candidate_status'], 'pending')
            self.assertFalse(pending['accuracy_qualified'])
            self.assertFalse(pending['candidate_completed_3mm'])
            self.assertIsNone(pending['provenance']['candidate_sha256'])
            self.assertEqual(pending['physical_comparison'], 'not_performed_candidate_missing')
            with (out/'accuracy.csv').open(newline='', encoding='utf-8') as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            b.mkdir(); (b/'result.json').write_text('{invalid', encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError):
                write_comparison(a, b, out, allow_pending=True)
            rb = copy.deepcopy(ra); rb['config']['friction_coefficient'] = .2
            (b/'result.json').write_text(json.dumps(rb), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Physical settings differ'):
                write_comparison(a, b, out, allow_pending=True)
            (b/'result.json').write_text(json.dumps(ra), encoding='utf-8')
            self.assertTrue(write_comparison(a, b, out, allow_pending=True)['accuracy_qualified'])


if __name__ == '__main__':
    unittest.main()
