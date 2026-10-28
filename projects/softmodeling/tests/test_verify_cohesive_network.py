"""Small real equilibrium, path selection, energy and rejected-history checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.verify_cohesive_network import NetworkPatch, run, run_case


class NetworkVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plus, cls.plus_fields = run_case(1, steps=80, seconds=30)
        cls.minus, cls.minus_fields = run_case(-1, steps=80, seconds=30)
        cls.fine, cls.fine_fields = run_case(1, steps=160, seconds=30)

    def test_all_edges_compete_and_center_dofs_are_actually_free(self):
        problem = NetworkPatch()
        self.assertEqual(len(problem.topology.segments), 4)
        self.assertEqual(len(problem.free), 8)
        self.assertTrue(np.all(problem.topology.original_vertex_ids[problem.free//2] == 4))
        self.assertEqual(problem.history.shape, (4, 2))
        np.testing.assert_array_equal(problem.history, np.zeros((4, 2)))
        self.assertLess(problem.topology.bulk_area_ratio, 1.)
        self.assertGreater(problem.topology.bulk_area_ratio, .98)
        # The free centers do not follow the imposed affine boundary field.
        affine = self.plus_fields['reference']+.08*problem.loading
        self.assertGreater(np.linalg.norm((self.plus_fields['positions'][-1]-affine).ravel()[problem.free]), .01)

    def test_mirror_loading_selects_different_paths_with_same_response(self):
        self.assertEqual(self.plus['final_failed_interface_ids'], [1, 3])
        self.assertEqual(self.minus['final_failed_interface_ids'], [0, 2])
        self.assertTrue(self.plus['no_seeded_or_weakened_interfaces'])
        self.assertTrue(self.minus['no_seeded_or_weakened_interfaces'])
        np.testing.assert_allclose([r['generalized_reaction_n_mm'] for r in self.plus['rows']],
                                   [r['generalized_reaction_n_mm'] for r in self.minus['rows']], atol=1e-10)

    def test_every_state_passes_force_history_geometry_and_energy_audits(self):
        for result in (self.plus, self.minus, self.fine):
            with self.subTest(steps=result['steps_requested'], sign=result['direction_sign']):
                self.assertTrue(result['completed'])
                self.assertTrue(result['audit']['passed'])
                self.assertLess(result['audit']['maximum_residual_n'], 1e-8)
                self.assertGreater(result['audit']['min_j'], 0.)
                self.assertTrue(result['energy']['qualified'])
                self.assertTrue(result['stable_accepted_branch'])
                self.assertTrue(result['contact']['full_ccd'])
                self.assertEqual(result['contact']['kind'], 'native_full_EV_VV_improved_max')
                self.assertFalse(result['failures'])

    def test_external_work_and_selected_path_converge_under_step_halving(self):
        self.assertEqual(self.plus['final_failed_interface_ids'], self.fine['final_failed_interface_ids'])
        self.assertLess(self.fine['energy']['maximum_absolute_residual_n_mm'],
                        .6*self.plus['energy']['maximum_absolute_residual_n_mm'])
        self.assertLess(self.fine['energy']['maximum_relative_residual'], 2e-5)
        self.assertTrue(np.all(np.diff(self.plus_fields['history'], axis=0) >= -1e-14))

    def test_rejected_trial_preserves_accepted_history_and_saves_actual_failure(self):
        def reject(problem, accepted, previous_load, load, deadline):
            problem.last_trial_positions = accepted.copy()+.001
            # A noncommitted trial is deliberately damaging.
            trial = problem.law.response(.1, .1, problem.history)
            self.assertTrue(np.all(trial['damage'] == 1.))
            raise RuntimeError('Injected rejected equilibrium trial')

        with patch.object(NetworkPatch, 'solve', reject):
            result, fields = run_case(steps=4, seconds=30)
        self.assertFalse(result['completed'])
        self.assertFalse(result['energy']['qualified'])
        self.assertEqual(result['accepted_states'], 1)
        self.assertTrue(result['failures'][0]['accepted_history_unchanged'])
        np.testing.assert_array_equal(fields['history'], np.zeros((1, 4, 2)))
        np.testing.assert_array_equal(fields['failed_previous'][0], fields['positions'][0])
        np.testing.assert_allclose(fields['failed_positions'][0], fields['positions'][0]+.001)
        np.testing.assert_array_equal(fields['failed_previous_history'][0], fields['history'][0])

    def test_report_gates_provenance_and_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'new'
            self.assertEqual(run(output, seconds=30), 0)
            report = json.loads((output/'verification.json').read_text(encoding='utf-8'))
            self.assertTrue(report['passed'])
            self.assertTrue(all(g['passed'] for g in report['gates']))
            self.assertTrue(report['no_inverse_fitting'])
            self.assertEqual(report['experimental_inputs_used'], [])
            self.assertIsNone(report['physical_onset_depth_prediction'])
            self.assertEqual(report['source_sha256'], report['source_sha256_at_end'])
            self.assertTrue((output/'network.svg').is_file())
            self.assertEqual(len(list(output.glob('case_*.npz'))), 4)
            with self.assertRaises(FileExistsError):
                run(output, seconds=30)

    def test_invalid_cases_fail_before_loading(self):
        for options in (dict(seconds=0), dict(seconds=121), dict(steps=3), dict(target=.3),
                        dict(target=float('nan')), dict(direction_sign=0), dict(gap_mm=-1.)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_case(**options)


if __name__ == '__main__':
    unittest.main()
