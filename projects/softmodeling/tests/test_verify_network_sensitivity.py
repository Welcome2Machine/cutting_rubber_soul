"""Synthetic report tests: no FE loading, subprocess solves or experimental data."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.verify_network_sensitivity import (
    PROJECT, BOUNDED_COMPRESSION, canonical_contact_stencils, compare_manifest,
    compression_regularization_metadata, evidence_equal,
    curve_comparison, digest, main, record_digest, runstudy,
    sampled_path_distance, template, validate_manifest, write_json,
)


def config():
    c = json.loads((PROJECT/'configs/fracture_network.json').read_text(encoding='utf-8-sig'))
    c.update(preparation_clearance_mm=.02, target_mm=.1, increment_mm=.01)
    c['cohesive']['compression_stiffness_mpa_per_mm'] = 0.
    return c


def bounded_config():
    c = config()
    c.update(schema='softcut-cohesive/3', closure_regularization=BOUNDED_COMPRESSION,
             minimum_increment_mm=.0001, ccd_tolerance_mm=1e-7)
    c['cohesive'].update(compression_stiffness_mpa_per_mm=5000.,
                         compression_regularization_mm=c['reference_gap_mm'])
    return c


class NetworkSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'prototype').mkdir()
        (self.root/'prototype/synthetic.py').write_text('# synthetic result fixture\n')
        self.manifest = template(config(), 'normal_stiffness', [50., 100.], 30)

    def result(self, case, *, force_scale=1., end=.1, completed=True):
        """Build unmistakably synthetic saved evidence for parser/gate checks."""
        directory = self.root/case['result_directory']
        (directory/'states').mkdir(parents=True)
        xy = np.array([[0., 0.], [1., 0.], [0., 1.], [1.01, 0.], [1.01, 1.], [.01, 1.]])
        tri = np.array([[0, 1, 2], [3, 4, 5]])
        seg = np.array([[1, 3, 2, 5]])
        steps, rows = [], []
        for i, depth in enumerate([0., end]):
            path = directory/'states'/f'{i:04d}.npz'
            np.savez(path, reference_xy=xy, triangles=tri, interface_segments=seg,
                     reference_interface_areas_mm2=np.array([np.sqrt(2.)*20]), depth_mm=depth,
                     reference_interface_lengths_mm=np.array([np.sqrt(2.)]),
                     original_interface_edges=np.array([[1, 2]]), interface_exterior_seed=np.array([True]),
                     cohesive_maximum=np.zeros((1, 2)), previous_cohesive_maximum=np.zeros((1, 2)))
            force = force_scale*depth
            steps.append(dict(depth_mm=depth, reaction_fz_n=force,
                              material=dict(dt_s=.1 if i == 0 else end/.2, stored_energy_n_mm=.7),
                              fracture=dict(cohesive_stored_energy_n_mm=.2, compression_stored_energy_n_mm=.1,
                                            ipc_barrier_stored_energy_n_mm=.1, surface_connected_failed_segments=[],
                                            fracture_dissipation_n_mm=0., max_damage=0., separated_segments=0,
                                            surface_connected_crack_length_mm=0.)))
            rows.append(dict(depth_mm=depth, reaction_fz_n=force, residual_n=1e-7, min_j=1.,
                             passed=True, cohesive_history_passed=True, checkpoint_sha256=digest(path),
                             max_damage=0., accepted_record_sha256=record_digest(steps[-1])))
        write_json(directory/'audit.json', dict(passed=True, cohesive_history_chain_passed=True, rows=rows))
        write_json(directory/'result.json', dict(
            fixture='synthetic_parser_test_not_FE_evidence', configuration=case['config'],
            completed=completed, status='completed' if completed else 'failed', steps=steps,
            source_sha256={'prototype/synthetic.py': digest(self.root/'prototype/synthetic.py')},
            signature={'config': {'width_mm': 20.}}, fracture_topology_metadata={'bulk_area_ratio': .99,
                'preparation': {'mode': 'explicit_vertical_clearance', 'initial_depth_mm': -.02, 'vertical_clearance_mm': .02}},
            events={'damage_onset': {'status': 'not_observed_in_accepted_prefix'}},
            event_qualification='independently_audited_prefix', interpreter=sys.executable,
            energy_accounting={'energy_balance_qualified': False, 'discretization_converged': False,
                               'unaccounted_energy_after_barrier_n_mm': .5}))
        return directory

    def compare(self):
        return compare_manifest(self.manifest, self.root, self.root)

    def test_family_rejects_physical_and_preparation_confounds(self):
        mutations = [('angle_deg', 57), ('friction_coefficient', .1),
                     ('preparation_clearance_mm', .01), ('target_mm', .2), ('increment_mm', .005),
                     ('reference_gap_mm', .008), ('minimum_increment_mm', .0001)]
        for key, value in mutations:
            with self.subTest(key=key):
                m = copy.deepcopy(self.manifest)
                m['cases'][1]['config'][key] = value
                with self.assertRaises(ValueError):
                    validate_manifest(m)
        for group, key in [('material', 'maxwell_shear_mpa'), ('cohesive', 'toughness_n_per_mm'),
                           ('cohesive', 'normal_strength_mpa'), ('cohesive', 'shear_strength_mpa')]:
            with self.subTest(group=group, key=key):
                m = copy.deepcopy(self.manifest)
                m['cases'][1]['config'][group][key] *= 2
                with self.assertRaises(ValueError):
                    validate_manifest(m)

    def test_time_family_and_version3_are_explicit(self):
        c = config()
        c.update(schema='softcut-cohesive/3', closure_regularization='reversible_compression_penalty',
                 minimum_increment_mm=.0001, ccd_tolerance_mm=1e-7)
        c['cohesive']['compression_stiffness_mpa_per_mm'] = 5000.
        m = template(c, 'time', [.01, .005, .0025], 30)
        self.assertEqual(validate_manifest(m), m)
        m['cases'][1]['config']['preparation_clearance_mm'] = .01
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_gap_scan_cannot_silently_scale_contact_distance(self):
        m = template(config(), 'gap', [.006, .008], 30)
        m['cases'][1]['config']['dhat_mm'] *= .5
        with self.assertRaisesRegex(ValueError, 'Confounded'):
            validate_manifest(m)

    def test_contact_canonicalization_ignores_only_unordered_geometry_identity(self):
        raw = [
            {'kind': 'VV', 'global_nodes': [9, 3], 'segment_ids': [4],
             'touches_surface_connected_crack': False, 'attribution_ambiguous': True},
            {'kind': 'EV', 'global_edge_nodes': [8, 2], 'global_vertex': 7,
             'segment_ids': [5], 'touches_surface_connected_crack': True,
             'gauss_damage': [[1., 1.]]}]
        before = copy.deepcopy(raw)
        canonical = canonical_contact_stencils(raw)
        self.assertTrue(evidence_equal({'contact_stencils': raw}, {'contact_stencils': canonical}))
        self.assertEqual(raw, before)
        self.assertEqual(canonical[0]['global_edge_nodes'], [2, 8])
        self.assertEqual(canonical[1]['global_nodes'], [3, 9])
        self.assertEqual(canonical[0]['global_vertex'], 7)
        self.assertNotEqual(record_digest({'contact_stencils': raw}), record_digest({'contact_stencils': canonical}))
        mutations = [lambda s: s[0].update(global_vertex=6),
                     lambda s: s[0].update(segment_ids=[6]),
                     lambda s: s[0].update(touches_surface_connected_crack=False),
                     lambda s: s[1].update(attribution_ambiguous=False),
                     lambda s: s.append(copy.deepcopy(s[0]))]
        for mutation in mutations:
            altered = copy.deepcopy(canonical)
            mutation(altered)
            self.assertFalse(evidence_equal({'contact_stencils': raw}, {'contact_stencils': altered}))

    def test_numerical_guard_must_be_the_same_within_a_control_family(self):
        m = copy.deepcopy(self.manifest)
        for case in m['cases']:
            case['config']['ptc_residual_guard'] = True
        self.assertEqual(validate_manifest(m), m)
        m['cases'][1]['config']['ptc_residual_guard'] = False
        with self.assertRaisesRegex(ValueError, 'Confounded'):
            validate_manifest(m)

    def test_bounded_gap_template_changes_only_gap_and_its_derived_length(self):
        m = template(bounded_config(), 'gap', [.006, .008, .012], 30)
        self.assertEqual(validate_manifest(m), m)
        for case, gap in zip(m['cases'], [.006, .008, .012]):
            c = case['config']
            self.assertEqual(c['reference_gap_mm'], gap)
            self.assertEqual(c['cohesive']['compression_regularization_mm'], gap)
            self.assertEqual(c['cohesive']['compression_stiffness_mpa_per_mm'], 5000.)
            self.assertEqual(c['dhat_mm'], .0025)
            self.assertEqual(c['ccd_tolerance_mm'], 1e-7)

    def test_bounded_compression_rejects_missing_or_non_derived_length(self):
        valid = template(bounded_config(), 'gap', [.006, .008], 30)
        for ell in (None, 0., -.006, .007, .008*(1+1e-12)):
            with self.subTest(ell=ell):
                m = copy.deepcopy(valid)
                m['cases'][1]['config']['cohesive']['compression_regularization_mm'] = ell
                with self.assertRaises(ValueError):
                    validate_manifest(m)
        c = bounded_config()
        del c['cohesive']['compression_regularization_mm']
        with self.assertRaises(ValueError):
            template(c, 'gap', [.006, .008], 30)

    def test_bounded_gap_family_cannot_smuggle_other_law_or_contact_changes(self):
        valid = template(bounded_config(), 'gap', [.006, .008], 30)
        for group, key in [('cohesive', 'normal_stiffness_mpa_per_mm'),
                           ('cohesive', 'compression_stiffness_mpa_per_mm'),
                           (None, 'dhat_mm'), (None, 'ccd_tolerance_mm')]:
            with self.subTest(key=key):
                m = copy.deepcopy(valid)
                c = m['cases'][1]['config']
                container = c[group] if group else c
                container[key] *= .5
                with self.assertRaisesRegex(ValueError, 'Confounded'):
                    validate_manifest(m)
        m = copy.deepcopy(valid)
        m['cases'][1]['config']['cohesive']['new_undeclared_law_parameter'] = 1.
        with self.assertRaisesRegex(ValueError, 'Confounded'):
            validate_manifest(m)

    def test_linear_and_legacy_modes_require_none_and_bound_is_branch_stress(self):
        legacy = config()
        legacy['cohesive']['compression_regularization_mm'] = None
        validate_manifest(template(legacy, 'gap', [.006, .008], 30))
        linear = bounded_config()
        linear['closure_regularization'] = 'reversible_compression_penalty'
        linear['cohesive']['compression_regularization_mm'] = None
        validate_manifest(template(linear, 'gap', [.006, .008], 30))
        for c in (legacy, linear):
            c['cohesive']['compression_regularization_mm'] = .006
            with self.assertRaises(ValueError):
                template(c, 'gap', [.006, .008], 30)
        metadata = compression_regularization_metadata(bounded_config())
        self.assertAlmostEqual(metadata['compression_branch_traction_magnitude_bound_mpa'], 30.)
        self.assertIn('not a bound on IPC', metadata['bound_scope'])
        self.assertIsNone(compression_regularization_metadata(config())['compression_branch_traction_magnitude_bound_mpa'])

    def test_full_comparison_preserves_energy_and_computes_geometry(self):
        self.result(self.manifest['cases'][0], force_scale=1.)
        self.result(self.manifest['cases'][1], force_scale=2.)
        report = self.compare()
        self.assertEqual(report['status'], 'comparison_available')
        self.assertFalse(report['convergence_qualified'])
        pair = report['families'][0]['pairs'][0]
        self.assertAlmostEqual(pair['relative_L2'], .5)
        self.assertAlmostEqual(pair['endpoint_difference_n'], -.1)
        self.assertAlmostEqual(pair['external_work_difference_n_mm'], -.005)
        c = report['cases'][0]
        self.assertFalse(c['energy_qualification_preserved'])
        self.assertEqual(c['geometry']['bulk_area_ratio'], .99)
        self.assertGreater(c['geometry']['g_over_h']['min'], 0)
        self.assertGreater(c['geometry']['Kn_h_over_E0']['min'], 0)
        self.assertAlmostEqual(c['maximum_virtual_compression_stored_fraction'], .1)
        self.assertEqual(c['audit']['status'], 'authenticated_existing_fresh_audit')
        self.assertEqual(report['experimental_inputs_used'], [])

    def test_short_prefix_stays_inconclusive_and_is_never_extrapolated(self):
        self.result(self.manifest['cases'][0], end=.05, completed=False)
        self.result(self.manifest['cases'][1], end=.1)
        r = self.compare()
        pair = r['families'][0]['pairs'][0]
        self.assertEqual(r['status'], 'inconclusive')
        self.assertEqual(pair['common_qualified_interval_mm'], [0., .05])
        self.assertIsNone(pair['endpoint_difference_n'])
        self.assertIsNone(pair['external_work_difference_n_mm'])
        self.assertIsNone(pair['sampled_reference_path_distance_mm'])
        self.assertTrue(pair['no_extrapolation'])

    def test_missing_result_missing_audit_and_tampered_checkpoint_do_not_pass(self):
        self.assertEqual(self.compare()['status'], 'inconclusive')
        dirs = [self.result(c) for c in self.manifest['cases']]
        (dirs[0]/'states/0001.npz').write_bytes(b'changed after audit')
        r = self.compare()
        self.assertEqual(r['cases'][0]['qualified_states'], 1)
        self.assertEqual(r['status'], 'inconclusive')
        (dirs[1]/'audit.json').unlink()
        r = self.compare()
        self.assertEqual(r['cases'][1]['qualified_states'], 0)
        self.assertFalse(r['cases'][1]['completed_target_and_audit'])

    def test_changed_report_force_is_not_authenticated_by_checkpoint_alone(self):
        dirs = [self.result(c) for c in self.manifest['cases']]
        p = dirs[0]/'result.json'
        j = json.loads(p.read_text())
        j['steps'][1]['reaction_fz_n'] = 100.
        write_json(p, j)
        r = self.compare()
        self.assertEqual(r['cases'][0]['qualified_states'], 1)
        self.assertEqual(r['status'], 'inconclusive')

    def test_corrupt_first_checkpoint_and_report_are_inconclusive(self):
        dirs = [self.result(c) for c in self.manifest['cases']]
        (dirs[0]/'states/0000.npz').write_bytes(b'corrupt file')
        (dirs[1]/'result.json').write_text('{incomplete')
        report = self.compare()
        self.assertEqual(report['status'], 'inconclusive')
        self.assertTrue(all(c['status'] == 'invalid_result' for c in report['cases']))

    def test_source_change_cannot_create_comparison_qualification(self):
        for c in self.manifest['cases']:
            self.result(c)
        (self.root/'prototype/synthetic.py').write_text('# changed numerical source')
        r = self.compare()
        self.assertEqual(r['status'], 'inconclusive')
        self.assertTrue(r['families'][0]['source_identity_same'])
        self.assertFalse(r['families'][0]['current_sources_match'])

    def test_actual_history_time_must_match_common_preparation(self):
        dirs = [self.result(c) for c in self.manifest['cases']]
        p = dirs[0]/'result.json'
        j = json.loads(p.read_text())
        j['steps'][0]['material']['dt_s'] = .01
        write_json(p, j)
        r = self.compare()
        self.assertFalse(r['cases'][0]['physical_schedule_matches_depth_and_speed'])
        self.assertEqual(r['status'], 'inconclusive')

    def test_top_level_event_tamper_cannot_bypass_a_valid_checkpoint_audit(self):
        dirs = [self.result(c) for c in self.manifest['cases']]
        p = dirs[0]/'result.json'
        j = json.loads(p.read_text())
        j['events']['damage_onset'] = {'status': 'observed', 'state_index': 1, 'depth_interval_mm': [0., .1], 'fz_n': .1}
        write_json(p, j)
        r = self.compare()
        self.assertTrue(r['cases'][0]['audit']['passed'])
        self.assertEqual(r['cases'][0]['events']['damage_onset']['status'], 'not_observed_in_accepted_prefix')
        self.assertEqual(r['status'], 'inconclusive')
        self.assertIn('damage_onset', r['cases'][0]['event_evidence']['mismatches'])

    def test_old_audit_without_record_hash_does_not_authenticate_fracture_annotations(self):
        dirs = [self.result(c) for c in self.manifest['cases']]
        p, a = dirs[0]/'result.json', dirs[0]/'audit.json'
        j, audit = json.loads(p.read_text()), json.loads(a.read_text())
        for row in audit['rows']:
            row.pop('accepted_record_sha256')
        j['steps'][1]['fracture']['surface_connected_failed_segments'] = [0]
        j['steps'][1]['fracture']['surface_connected_crack_length_mm'] = np.sqrt(2.)
        write_json(p, j); write_json(a, audit)
        r = self.compare()
        self.assertEqual(r['cases'][0]['fracture_checkpoint_evidence']['status'], 'inconsistent_result_fracture')
        self.assertIsNone(r['cases'][0]['maximum_virtual_compression_stored_fraction'])
        self.assertEqual(r['status'], 'inconclusive')

    def test_contact_event_without_fresh_contact_evidence_remains_unknown(self):
        dirs = [self.result(c) for c in self.manifest['cases']]
        p = dirs[0]/'result.json'
        j = json.loads(p.read_text())
        j['events']['blade_in_surface_connected_crack'] = {'status': 'observed', 'state_index': 1}
        write_json(p, j)
        r = self.compare()
        event = r['cases'][0]['events']['blade_in_surface_connected_crack']
        self.assertEqual(event['status'], 'unknown_no_independent_contact_evidence')
        self.assertEqual(r['cases'][0]['event_evidence']['contact_status'], 'unknown')

    def test_fresh_contact_audit_blocks_a_false_top_level_contact_event(self):
        dirs = [self.result(c) for c in self.manifest['cases']]
        p, a = dirs[0]/'result.json', dirs[0]/'audit.json'
        j, audit = json.loads(p.read_text()), json.loads(a.read_text())
        for step, row in zip(j['steps'], audit['rows']):
            step['fracture']['blade_surface_connected_crack_contacts'] = 0
            row.update(cohesive_events_passed=True,
                       fracture={'max_damage': 0., 'blade_surface_connected_crack_contacts': 0},
                       material=copy.deepcopy(step['material']), accepted_record_sha256=record_digest(step))
        j['events']['blade_in_surface_connected_crack'] = {'status': 'observed', 'state_index': 1}
        write_json(p, j); write_json(a, audit)
        r = self.compare()
        c = r['cases'][0]
        self.assertTrue(c['audit']['passed'])
        self.assertEqual(c['event_evidence']['contact_status'], 'independent_audit_evidence')
        self.assertEqual(c['events']['blade_in_surface_connected_crack']['status'], 'not_observed_in_accepted_prefix')
        self.assertEqual(r['status'], 'inconclusive')

    def test_fresh_material_audit_is_checked_even_without_a_record_hash(self):
        dirs = [self.result(c) for c in self.manifest['cases']]
        p, a = dirs[0]/'result.json', dirs[0]/'audit.json'
        j, audit = json.loads(p.read_text()), json.loads(a.read_text())
        for step, row in zip(j['steps'], audit['rows']):
            row.pop('accepted_record_sha256')
            row['material'] = copy.deepcopy(step['material'])
        j['steps'][1]['material']['stored_energy_n_mm'] = 9.
        write_json(p, j); write_json(a, audit)
        r = self.compare()
        self.assertEqual(r['cases'][0]['qualified_states'], 1)
        self.assertEqual(r['status'], 'inconclusive')

    def test_one_failed_gauss_point_does_not_become_a_separated_crack(self):
        dirs = [self.result(c) for c in self.manifest['cases']]
        directory = dirs[0]
        checkpoint = directory/'states/0001.npz'
        with np.load(checkpoint) as s:
            state = {key: s[key] for key in s.files}
        state['cohesive_maximum'] = np.array([[.06, 0.]])
        np.savez(checkpoint, **state)
        p, a = directory/'result.json', directory/'audit.json'
        j, audit = json.loads(p.read_text()), json.loads(a.read_text())
        j['steps'][1]['fracture'].update(max_damage=1., fracture_dissipation_n_mm=.5*np.sqrt(2.)*20*.006)
        j['events']['damage_onset'] = {'status': 'observed', 'state_index': 1,
                                      'depth_interval_mm': [0., .1], 'fz_n': .1}
        audit['rows'][1].update(max_damage=1., checkpoint_sha256=digest(checkpoint),
                                accepted_record_sha256=record_digest(j['steps'][1]))
        write_json(p, j); write_json(a, audit)
        r = self.compare()
        evidence = r['cases'][0]['fracture_checkpoint_evidence']
        self.assertEqual(evidence['status'], 'verified_checkpoint_history_and_topology')
        self.assertEqual(evidence['rows'][1]['separated_gauss_points'], 1)
        self.assertEqual(evidence['rows'][1]['separated_segments'], 0)
        self.assertEqual(evidence['rows'][1]['surface_connected_failed_segments'], [])
        self.assertEqual(r['cases'][0]['events']['first_fully_separated_segment']['status'], 'not_observed_in_accepted_prefix')

    def test_zero_force_reference_is_unresolved(self):
        for c in self.manifest['cases']:
            self.result(c, force_scale=0.)
        r = self.compare()
        self.assertEqual(r['status'], 'inconclusive')
        self.assertIsNone(r['families'][0]['pairs'][0]['relative_L2'])

    def test_sampled_path_distance_reports_unknown_without_a_crack(self):
        self.assertIsNone(sampled_path_distance([], []))
        a = [[[0., 0.], [1., 0.]]]
        b = [[[0., .2], [1., .2]]]
        self.assertAlmostEqual(sampled_path_distance(a, b), .2)

    def test_budgets_identifiers_and_existing_output_are_guarded(self):
        m = copy.deepcopy(self.manifest)
        m['cases'][0]['id'] = '../escape'
        with self.assertRaises(ValueError):
            validate_manifest(m)
        m = copy.deepcopy(self.manifest)
        m['total_wall_budget_s'] = 1
        with self.assertRaises(ValueError):
            validate_manifest(m)
        existing = self.root/'existing'
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            runstudy(self.manifest, existing)

    def test_runstudy_uses_one_bounded_argument_list_and_stops_without_retry(self):
        with patch('tools.verify_network_sensitivity.subprocess.run', return_value=subprocess.CompletedProcess([], 2)) as run:
            report = runstudy(self.manifest, self.root/'run')
        self.assertEqual(run.call_count, 1)
        args = run.call_args.args[0]
        self.assertIsInstance(args, list)
        self.assertIn('tools.run_cohesive_cut', args)
        self.assertEqual(run.call_args.kwargs['timeout'], 90)
        self.assertNotIn('shell', run.call_args.kwargs)
        self.assertEqual(report['new_fe_runs_started'], 1)
        self.assertEqual(report['status'], 'inconclusive')
        self.assertEqual(report['execution'][0]['retries'], 0)

    def test_launcher_timeout_is_recorded_and_no_extension_occurs(self):
        with patch('tools.verify_network_sensitivity.subprocess.run', side_effect=subprocess.TimeoutExpired('synthetic', 90)) as run:
            report = runstudy(self.manifest, self.root/'timeout')
        self.assertEqual(run.call_count, 1)
        self.assertEqual(report['execution'][0]['status'], 'launcher_timeout')

    def test_compare_cli_writes_html_without_any_subprocess(self):
        for c in self.manifest['cases']:
            self.result(c)
        manifest = self.root/'manifest.json'
        write_json(manifest, self.manifest)
        with patch('tools.verify_network_sensitivity.subprocess.run') as run:
            code = main(['compare', '--manifest', str(manifest), '--output', str(self.root/'report')])
        # Synthetic sources deliberately are not the actual project sources.
        self.assertEqual(code, 2)
        run.assert_not_called()
        self.assertTrue((self.root/'report/index.html').is_file())
        self.assertTrue((self.root/'report/sensitivity.json').is_file())


if __name__ == '__main__':
    unittest.main()
