"""Saved-field parsing and local geometry tests; no FE construction or loading."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from prototype.cohesive_mixed import MixedModeLaw, interface_response
from tools.audit_network_pairing import (PROJECT, audit_result, digest, main,
    pairing_metrics, record_digest, segment_distance)


def fixture_geometry(slip=1., height=.1, gap=6e-7):
    x = np.array([[0., 0.], [gap*.5, slip], [0., height], [gap*.5, height+slip]])
    return x, np.array([[0, 1, 2, 3]]), np.full((1, 2), gap), np.array([height]), np.zeros((1, 2))


def fixture_law():
    return MixedModeLaw(normal_stiffness_mpa_per_mm=5000., compression_stiffness_mpa_per_mm=5000.,
                        compression_regularization_mm=6e-7)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


class PairingGeometryTests(unittest.TestCase):
    def test_finite_segment_distances_include_endpoint_crossing_and_collinear_cases(self):
        cases = [([[0, 0], [1, 0], [.5, -1], [.5, 1]], 0.),
                 ([[0, 0], [1, 0], [2, 0], [3, 0]], 1.),
                 ([[0, 0], [2, 0], [1, 0], [3, 0]], 0.),
                 ([[0, 0], [1, 0], [1, 1], [2, 1]], 1.),
                 ([[0, 0], [1, 0], [2, 1], [3, 1]], np.sqrt(2.))]
        for points, expected in cases:
            with self.subTest(points=points):
                self.assertAlmostEqual(segment_distance(*np.asarray(points, float)), expected)

    def test_far_slip_pressure_bound_does_not_hide_geometric_force_couple(self):
        law = fixture_law()
        values = []
        for slip in (1., 10.):
            x, seg, gaps, areas, offsets = fixture_geometry(slip)
            history = np.full((1, 2), slip+1.)
            q = pairing_metrics(x, seg, gaps, areas, offsets, history, law, 2.5e-7)
            self.assertTrue(q['history_consistent_with_saved_geometry'])
            self.assertEqual(q['outside_ipc_activation_compression_active_pairs'], 1)
            pair = q['interfaces'][0]
            self.assertAlmostEqual(pair['finite_face_distance_mm'], np.hypot(slip-.1, 3e-7))
            self.assertAlmostEqual(pair['tangential_offset_over_mean_length'], slip/.1)
            self.assertLess(max(map(abs, pair['compression_traction_mpa'])), 5000.*6e-7)
            self.assertGreater(pair['face_centroid_couple_norm_n_mm'], 0.)
            self.assertLess(pair['total_internal_force_balance_norm_n'], 1e-12)
            self.assertLess(abs(pair['total_internal_moment_about_pair_centroid_n_mm']), 1e-12)
            values.append(pair)
        np.testing.assert_allclose(values[0]['compression_traction_mpa'], values[1]['compression_traction_mpa'])
        self.assertGreater(values[1]['compression_gradient_norm_n'], 9*values[0]['compression_gradient_norm_n'])

    def test_rigid_rotation_keeps_distance_energy_and_force_norm(self):
        law = fixture_law()
        x, seg, gaps, areas, offsets = fixture_geometry()
        angle = .93
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        before = pairing_metrics(x, seg, gaps, areas, offsets, np.ones((1, 2))*2, law, 2.5e-7)
        after = pairing_metrics(x@rotation.T+[.2, -.3], seg, gaps, areas, offsets, np.ones((1, 2))*2, law, 2.5e-7)
        for key in ('compression_stored_energy_n_mm', 'compression_assembled_gradient_norm_n',
                    'maximum_remote_face_couple_norm_n_mm'):
            self.assertAlmostEqual(after[key], before[key], delta=max(1e-14, abs(before[key])*1e-8))
        np.testing.assert_allclose(after['interfaces'][0]['compression_gradient_n'],
            np.asarray(before['interfaces'][0]['compression_gradient_n'])@rotation.T, rtol=1e-8, atol=1e-12)

    def test_one_failed_gauss_point_is_not_a_complete_crack_face(self):
        law = fixture_law()
        x, seg, gaps, areas, offsets = fixture_geometry(slip=0.)
        q = pairing_metrics(x, seg, gaps, areas, offsets,
            np.array([[law.final_opening_mm, 0.]]), law, 2.5e-7)
        self.assertEqual(q['fully_failed_interfaces'], 0)
        self.assertEqual(q['partially_failed_interfaces'], 1)
        self.assertEqual(q['interfaces'], [])
        self.assertGreater(q['compression_stored_energy_n_mm'], 0.)

    def test_nearby_pair_within_ipc_distance_is_not_classified_remote(self):
        law = fixture_law()
        x, seg, gaps, areas, offsets = fixture_geometry(slip=0.)
        q = pairing_metrics(x, seg, gaps, areas, offsets, np.ones((1, 2)), law, 4e-7)
        self.assertEqual(q['geometrically_disjoint_compression_active_pairs'], 1)
        self.assertEqual(q['outside_ipc_activation_compression_active_pairs'], 0)


class PairingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory/'states').mkdir()
        law = fixture_law()
        x, seg, gaps, areas, offsets = fixture_geometry()
        history = np.ones((1, 2))*2.
        q = interface_response(x, seg, gaps, areas, history, law, tangent=False, reference_slips=offsets)
        self.config = dict(cohesive_kind='mixed_mode_energy', fracture_topology='edge_network',
                           cohesive=asdict(law), dhat_mm=2.5e-7, target_mm=.01)
        self.state = dict(positions=x, depth_mm=0., body_nodes=4, cohesive_maximum=history,
            previous_cohesive_maximum=np.zeros_like(history), interface_segments=seg,
            reference_interface_gaps_mm=gaps, reference_interface_areas_mm2=areas,
            reference_interface_slips_mm=offsets, fracture_topology='edge_network',
            cohesive_law_parameters_json=json.dumps(law.to_dict()),
            compression_stored_energy_n_mm=q['compression_stored_energy_n_mm'])
        np.savez(self.directory/'states/0000.npz', **self.state)
        step = dict(depth_mm=0., reaction_fz_n=0., fracture=dict(compression_stored_energy_n_mm=q['compression_stored_energy_n_mm']))
        sources = ('prototype/cohesive_mixed.py', 'prototype/cohesive_cut.py',
                   'prototype/ipc_reference.py', 'prototype/ipc_viscoelastic.py', 'tools/run_cohesive_cut.py')
        self.report = dict(fixture='synthetic_parser_test_not_FE_evidence', configuration=self.config,
            cohesive_law=law.to_dict(), signature=dict(fracture=dict(law=law.to_dict())),
            source_sha256={name: digest(PROJECT/name) for name in sources},
            completed=False, status='failed', steps=[step])
        self.audit = dict(passed=True, cohesive_history_chain_passed=True, rows=[dict(
            passed=True, cohesive_history_passed=True, depth_mm=0., residual_n=0., min_j=1.,
            checkpoint_sha256=digest(self.directory/'states/0000.npz'), accepted_record_sha256=record_digest(step))])
        self.save()

    def save(self):
        write_json(self.directory/'config.json', self.config)
        write_json(self.directory/'result.json', self.report)
        write_json(self.directory/'audit.json', self.audit)

    def test_authenticated_prefix_remains_not_complete_or_physically_qualified(self):
        result = audit_result(self.directory)
        self.assertEqual(result['qualification'], 'authenticated_saved_prefix')
        self.assertFalse(result['completed_target'])
        self.assertFalse(result['physical_pairing_adequacy_qualified'])
        self.assertFalse(result['equilibrium_solve_performed'])
        self.assertEqual(result['maximum_outside_ipc_activation_compression_active_pairs'], 1)

    def test_missing_or_partial_audit_is_diagnostic_even_when_state_can_be_read(self):
        for audit in ({}, dict(self.audit, passed=False), dict(self.audit, cohesive_history_chain_passed=False)):
            with self.subTest(audit=list(audit)):
                write_json(self.directory/'audit.json', audit)
                result = audit_result(self.directory)
                self.assertEqual(result['qualification'], 'diagnostic_only')
                self.assertEqual(result['audited_state_count'], 0)
                self.assertEqual(result['maximum_outside_ipc_activation_compression_active_pairs'], 1)

    def test_changed_record_or_npz_hash_cannot_pass(self):
        self.report['steps'][0]['reaction_fz_n'] = 1.
        self.save()
        self.assertEqual(audit_result(self.directory)['qualification'], 'diagnostic_only')
        self.report['steps'][0]['reaction_fz_n'] = 0.
        self.save()
        self.state['extra_array'] = np.array([42])
        np.savez(self.directory/'states/0000.npz', **self.state)
        self.assertEqual(audit_result(self.directory)['qualification'], 'diagnostic_only')

    def test_config_law_and_source_identity_are_required(self):
        original = copy.deepcopy(self.report)
        for mutate in (lambda r: r['source_sha256'].update({'prototype/ipc_reference.py': '0'*64}),
                       lambda r: r['signature']['fracture']['law'].update(compression_regularization_mm=1e-6),
                       lambda r: r['configuration'].update(dhat_mm=1e-7)):
            self.report = copy.deepcopy(original)
            mutate(self.report)
            write_json(self.directory/'result.json', self.report)
            with self.subTest(source=self.report['source_sha256']):
                self.assertEqual(audit_result(self.directory)['qualification'], 'diagnostic_only')

    def test_rehashed_history_chain_tamper_is_independently_rejected(self):
        self.state['previous_cohesive_maximum'] = np.ones((1, 2))
        np.savez(self.directory/'states/0000.npz', **self.state)
        self.audit['rows'][0]['checkpoint_sha256'] = digest(self.directory/'states/0000.npz')
        self.save()
        result = audit_result(self.directory)
        self.assertEqual(result['qualification'], 'diagnostic_only')
        self.assertIn('cohesive_history_chain_mismatch', result['rows'][0]['reasons'])

    def test_cli_cannot_overwrite_source_or_existing_artifact(self):
        before = digest(self.directory/'result.json')
        with self.assertRaises(FileExistsError):
            main([str(self.directory), '--output', str(self.directory/'result.json')])
        self.assertEqual(digest(self.directory/'result.json'), before)
        out = self.directory/'pairing.json'
        self.assertEqual(main([str(self.directory), '--output', str(out)]), 0)
        self.assertEqual(json.loads(out.read_text())['qualification'], 'authenticated_saved_prefix')

    def test_files_changed_during_read_cannot_be_authenticated(self):
        def modifying_reader(*args, **kwargs):
            path = self.directory/'result.json'
            path.write_text(path.read_text()+' ', encoding='utf-8')
            return pairing_metrics(*args, **kwargs)
        with patch('tools.audit_network_pairing.pairing_metrics', side_effect=modifying_reader):
            result = audit_result(self.directory)
        self.assertEqual(result['qualification'], 'diagnostic_only')
        self.assertFalse(result['input_files_stable_during_read'])


if __name__ == '__main__':
    unittest.main()
